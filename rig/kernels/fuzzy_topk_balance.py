"""Differentiable balance objectives for grouped fuzzy Top-K.

The ordinary fuzzy kernel retains one ReLU-positive winner from every fixed
feature group.  This module leaves that forward operation unchanged and also
returns three training-only objectives:

``switch``
    The Switch-style product of stop-gradient hard winner load and mean soft
    choice probability.  It directly penalizes persistent winner monopoly.
``importance``
    The squared concentration of the mean soft choice probability.  This is a
    smooth population-balance objective without a hard-routing factor.
``alive``
    Either a squared per-token margin on negative group maxima or a global-
    batch activation-frequency floor.  Both address the separate failure mode
    where an entire group falls below the ReLU boundary.  The frequency floor
    uses a straight-through bias surrogate so it does not recompute scores.

The choicewise custom VJP shares the ordinary feature-scoring forward pass and
recomputes scores once in its reverse rule.  Auxiliary preactivation
cotangents are merged into the existing dX/dW_up/db_up contractions; decoder
semantics and decoder gradients remain exactly those of fuzzy Top-K.
"""

from __future__ import annotations

from dataclasses import dataclass
import functools
import math
from typing import Callable, Literal

import jax
from jax import lax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P

from .fuzzy_topk import fuzzy_topk_relu


FUZZY_BALANCE_STAT_NAMES = (
    "fuzzy.load_balance_loss",
    "fuzzy.importance_balance_loss",
    "fuzzy.alive_margin_loss",
)

FuzzyTopKBalanceCallable = Callable[
    [jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    tuple[jax.Array, jax.Array],
]


@dataclass(frozen=True, slots=True)
class FuzzyTopKBalanceConfig:
    """Static training-only objective and grouped-selection contract."""

    top_k: int
    mode: Literal["none", "switch", "importance", "bias"] = "switch"
    temperature: float = 1.0
    alive_mode: Literal["token_margin", "frequency_floor"] = "token_margin"
    alive_margin: float = 0.0
    alive_target: float = 0.1

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.mode not in ("none", "switch", "importance", "bias"):
            raise ValueError(f"unknown fuzzy balance mode: {self.mode!r}")
        if not math.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError("temperature must be finite and positive")
        if not math.isfinite(self.alive_margin):
            raise ValueError("alive_margin must be finite")
        if self.alive_mode not in ("token_margin", "frequency_floor"):
            raise ValueError(f"unknown fuzzy alive mode: {self.alive_mode!r}")
        if not math.isfinite(self.alive_target) or not 0.0 < self.alive_target <= 1.0:
            raise ValueError("alive_target must be finite and in (0, 1]")

    @property
    def compute_soft_statistics(self) -> bool:
        return self.mode in ("switch", "importance")

    @property
    def compute_load_statistics(self) -> bool:
        return self.mode in ("switch", "bias")

    @property
    def compute_activation_statistics(self) -> bool:
        return self.alive_mode == "frequency_floor"


def _validate_inputs(
    x: jax.Array,
    up_weight: jax.Array,
    up_bias: jax.Array,
    down_weight: jax.Array,
    down_bias: jax.Array,
    config: FuzzyTopKBalanceConfig,
) -> tuple[int, int, int]:
    model_width = x.shape[-1]
    if x.ndim < 2:
        raise ValueError(f"x must have at least two dimensions, got {x.shape}")
    if up_weight.ndim != 2 or up_weight.shape[0] != model_width:
        raise ValueError(
            f"up_weight must have shape ({model_width}, hidden_width), "
            f"got {up_weight.shape}"
        )
    hidden_width = up_weight.shape[1]
    if up_bias.shape != (hidden_width,):
        raise ValueError(
            f"up_bias must have shape {(hidden_width,)}, got {up_bias.shape}"
        )
    if down_weight.shape != (hidden_width, model_width):
        raise ValueError(
            f"down_weight must have shape {(hidden_width, model_width)}, "
            f"got {down_weight.shape}"
        )
    if down_bias.shape != (model_width,):
        raise ValueError(
            f"down_bias must have shape {(model_width,)}, got {down_bias.shape}"
        )
    if config.top_k > hidden_width or hidden_width % config.top_k:
        raise ValueError(
            f"top_k {config.top_k} must divide hidden width {hidden_width}"
        )
    return model_width, hidden_width, hidden_width // config.top_k


def _preactivations(
    x: jax.Array, up_weight: jax.Array, up_bias: jax.Array
) -> jax.Array:
    hidden = jnp.einsum(
        "...d,dh->...h",
        x,
        up_weight.astype(x.dtype),
        preferred_element_type=jnp.float32,
    )
    return hidden + up_bias.astype(x.dtype)


def _selection_from_hidden(
    hidden: jax.Array, *, top_k: int
) -> tuple[jax.Array, jax.Array, jax.Array]:
    choices = hidden.shape[-1] // top_k
    grouped = hidden.reshape((*hidden.shape[:-1], top_k, choices))
    maxima = jnp.max(grouped, axis=-1)
    winners = jnp.argmax(grouped, axis=-1)
    return jax.nn.relu(maxima).astype(hidden.dtype), winners, maxima


def _choicewise_decode(
    values: jax.Array,
    winners: jax.Array,
    down_weight: jax.Array,
    down_bias: jax.Array,
    *,
    model_width: int,
) -> jax.Array:
    top_k = values.shape[-1]
    choices = down_weight.shape[0] // top_k
    grouped_down = down_weight.reshape((top_k, choices, model_width))
    accumulator = jnp.zeros((*values.shape[:-1], model_width), jnp.float32)

    def visit_choice(choice, output):
        active = jnp.where(winners == choice, values, 0.0)
        return output + jnp.einsum(
            "...k,kd->...d",
            active,
            grouped_down[:, choice, :].astype(values.dtype),
            preferred_element_type=jnp.float32,
        )

    output = lax.fori_loop(0, choices, visit_choice, accumulator)
    return (output + down_bias.astype(jnp.float32)).astype(values.dtype)


def _raw_balance_sums(
    hidden: jax.Array,
    winners: jax.Array,
    maxima: jax.Array,
    *,
    top_k: int,
    temperature: float,
    alive_margin: float,
    compute_soft_statistics: bool,
    compute_load_statistics: bool,
    compute_activation_statistics: bool,
    token_margin_alive: bool,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    choices = hidden.shape[-1] // top_k
    reduction_axes = tuple(range(winners.ndim - 1))
    if compute_load_statistics or compute_activation_statistics:
        load_sums = jnp.zeros((top_k, choices), jnp.float32)
        active_sums = jnp.zeros((top_k, choices), jnp.float32)

        def visit_choice(choice, statistics):
            loads, activations = statistics
            winner_mask = winners == choice
            if compute_load_statistics:
                loads = loads.at[:, choice].set(
                    jnp.sum(winner_mask.astype(jnp.float32), axis=reduction_axes)
                )
            if compute_activation_statistics:
                active = winner_mask & (maxima > 0.0)
                activations = activations.at[:, choice].set(
                    jnp.sum(active.astype(jnp.float32), axis=reduction_axes)
                )
            return loads, activations

        load_sums, active_sums = lax.fori_loop(
            0, choices, visit_choice, (load_sums, active_sums)
        )
    else:
        load_sums = jnp.zeros((top_k, choices), jnp.float32)
        active_sums = jnp.zeros((top_k, choices), jnp.float32)
    if compute_soft_statistics:
        grouped = hidden.reshape((*hidden.shape[:-1], top_k, choices))
        probabilities = jax.nn.softmax(
            grouped.astype(jnp.float32) / jnp.float32(temperature), axis=-1
        )
        soft_sums = jnp.sum(probabilities, axis=reduction_axes)
    else:
        soft_sums = jnp.zeros((top_k, choices), jnp.float32)
    if token_margin_alive:
        deficit = jax.nn.relu(jnp.float32(alive_margin) - maxima.astype(jnp.float32))
        alive_sum = jnp.sum(jnp.square(deficit), dtype=jnp.float32)
    else:
        deficit = jnp.zeros_like(maxima, dtype=jnp.float32)
        alive_sum = jnp.asarray(0.0, jnp.float32)
    return (
        soft_sums,
        lax.stop_gradient(load_sums),
        lax.stop_gradient(active_sums),
        alive_sum,
        deficit,
    )


def _choicewise_forward(
    x: jax.Array,
    up_weight: jax.Array,
    up_bias: jax.Array,
    down_weight: jax.Array,
    down_bias: jax.Array,
    *,
    config: FuzzyTopKBalanceConfig,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    hidden = _preactivations(x, up_weight, up_bias)
    values, winners, maxima = _selection_from_hidden(hidden, top_k=config.top_k)
    values = values.astype(x.dtype)
    output = _choicewise_decode(
        values,
        winners,
        down_weight,
        down_bias,
        model_width=x.shape[-1],
    )
    soft_sums, load_sums, active_sums, alive_sum, deficit = _raw_balance_sums(
        hidden,
        winners,
        maxima,
        top_k=config.top_k,
        temperature=config.temperature,
        alive_margin=config.alive_margin,
        compute_soft_statistics=config.compute_soft_statistics,
        compute_load_statistics=config.compute_load_statistics,
        compute_activation_statistics=config.compute_activation_statistics,
        token_margin_alive=config.alive_mode == "token_margin",
    )
    return (
        output,
        values,
        winners,
        soft_sums,
        load_sums,
        active_sums,
        alive_sum,
        deficit,
    )


def _choicewise_backward(
    residuals: tuple[jax.Array, ...],
    cotangents: tuple[jax.Array, ...],
    *,
    config: FuzzyTopKBalanceConfig,
) -> tuple[jax.Array, ...]:
    (
        x,
        up_weight,
        up_bias,
        down_weight,
        down_bias,
        values,
        winners,
        alive_deficit,
    ) = residuals
    (
        output_cotangent,
        soft_cotangent,
        _load_cotangent,
        _active_cotangent,
        alive_cotangent,
    ) = cotangents
    model_width, hidden_width = up_weight.shape
    top_k = config.top_k
    choices = hidden_width // top_k
    tokens = math.prod(x.shape[:-1])

    flat_x = x.reshape((tokens, model_width)).astype(jnp.float32)
    flat_output_cotangent = output_cotangent.reshape((tokens, model_width)).astype(
        jnp.float32
    )
    flat_values = values.reshape((tokens, top_k)).astype(jnp.float32)
    flat_winners = winners.reshape((tokens, top_k))
    grouped_up = up_weight.reshape((model_width, top_k, choices)).astype(jnp.float32)
    grouped_down = down_weight.reshape((top_k, choices, model_width)).astype(
        jnp.float32
    )

    auxiliary_cotangent = jnp.zeros((tokens, top_k, choices), jnp.float32)
    if config.compute_soft_statistics:
        # A soft population objective needs every four-choice probability.
        # Recompute scores once instead of retaining [tokens, H] per block.
        hidden = _preactivations(x, up_weight, up_bias).reshape(
            (tokens, top_k, choices)
        )
        probabilities = jax.nn.softmax(
            hidden.astype(jnp.float32) / jnp.float32(config.temperature), axis=-1
        )
        soft_cotangent = soft_cotangent.astype(jnp.float32)[None, :, :]
        centered = soft_cotangent - jnp.sum(
            probabilities * soft_cotangent, axis=-1, keepdims=True
        )
        auxiliary_cotangent = auxiliary_cotangent + (
            probabilities * centered / jnp.float32(config.temperature)
        )

    maximum_cotangent = (
        -jnp.float32(2.0)
        * alive_deficit.reshape((tokens, top_k)).astype(jnp.float32)
        * alive_cotangent.astype(jnp.float32)
    )
    initial = (
        jnp.zeros((tokens, model_width), jnp.float32),
        jnp.zeros(grouped_up.shape, jnp.float32),
        jnp.zeros((top_k, choices), jnp.float32),
        jnp.zeros(grouped_down.shape, jnp.float32),
    )

    def visit_choice(choice, carry):
        dx, up_gradient, up_bias_gradient, down_gradient = carry
        winner_mask = flat_winners == choice
        choice_values = jnp.where(winner_mask, flat_values, 0.0)
        choice_down = grouped_down[:, choice, :]
        value_cotangent = jnp.einsum(
            "td,kd->tk",
            flat_output_cotangent,
            choice_down,
            preferred_element_type=jnp.float32,
        )
        main_cotangent = jnp.where(
            winner_mask & (flat_values > 0.0), value_cotangent, 0.0
        )
        preactivation_cotangent = (
            main_cotangent
            + auxiliary_cotangent[:, :, choice]
            + jnp.where(winner_mask, maximum_cotangent, 0.0)
        )
        choice_up = grouped_up[:, :, choice]
        dx = dx + jnp.einsum(
            "tk,dk->td",
            preactivation_cotangent,
            choice_up,
            preferred_element_type=jnp.float32,
        )
        up_gradient = up_gradient.at[:, :, choice].set(
            jnp.einsum(
                "td,tk->dk",
                flat_x,
                preactivation_cotangent,
                preferred_element_type=jnp.float32,
            )
        )
        up_bias_gradient = up_bias_gradient.at[:, choice].set(
            jnp.sum(preactivation_cotangent, axis=0)
        )
        down_gradient = down_gradient.at[:, choice, :].set(
            jnp.einsum(
                "tk,td->kd",
                choice_values,
                flat_output_cotangent,
                preferred_element_type=jnp.float32,
            )
        )
        return dx, up_gradient, up_bias_gradient, down_gradient

    dx, up_gradient, up_bias_gradient, down_gradient = lax.fori_loop(
        0, choices, visit_choice, initial
    )
    down_bias_gradient = jnp.sum(flat_output_cotangent, axis=0)
    return (
        dx.reshape(x.shape).astype(x.dtype),
        up_gradient.reshape(up_weight.shape).astype(up_weight.dtype),
        up_bias_gradient.reshape(up_bias.shape).astype(up_bias.dtype),
        down_gradient.reshape(down_weight.shape).astype(down_weight.dtype),
        down_bias_gradient.astype(down_bias.dtype),
    )


@functools.cache
def _make_choicewise_operation(config: FuzzyTopKBalanceConfig):
    @jax.custom_vjp
    def operation(x, up_weight, up_bias, down_weight, down_bias):
        (
            output,
            _values,
            _winners,
            soft_sums,
            load_sums,
            active_sums,
            alive_sum,
            _deficit,
        ) = _choicewise_forward(
            x,
            up_weight,
            up_bias,
            down_weight,
            down_bias,
            config=config,
        )
        return output, soft_sums, load_sums, active_sums, alive_sum

    def forward_rule(x, up_weight, up_bias, down_weight, down_bias):
        (
            output,
            values,
            winners,
            soft_sums,
            load_sums,
            active_sums,
            alive_sum,
            alive_deficit,
        ) = _choicewise_forward(
            x, up_weight, up_bias, down_weight, down_bias, config=config
        )
        residuals = (
            x,
            up_weight,
            up_bias,
            down_weight,
            down_bias,
            values,
            winners,
            alive_deficit,
        )
        return (output, soft_sums, load_sums, active_sums, alive_sum), residuals

    def backward_rule(residuals, cotangents):
        return _choicewise_backward(residuals, cotangents, config=config)

    operation.defvjp(forward_rule, backward_rule)
    return operation


@functools.partial(jax.jit, static_argnames=("config",))
def _balanced_choicewise_fuzzy_topk_mlp(
    x: jax.Array,
    up_weight: jax.Array,
    up_bias: jax.Array,
    down_weight: jax.Array,
    down_bias: jax.Array,
    *,
    config: FuzzyTopKBalanceConfig,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Named boundary for profiling and physical matrix-FLOP accounting."""

    return _make_choicewise_operation(config)(
        x, up_weight, up_bias, down_weight, down_bias
    )


@functools.partial(jax.jit, static_argnames=("config",))
def _low_overhead_choicewise_fuzzy_topk_mlp(
    x: jax.Array,
    up_weight: jax.Array,
    up_bias: jax.Array,
    down_weight: jax.Array,
    down_bias: jax.Array,
    *,
    config: FuzzyTopKBalanceConfig,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Named no-softmax boundary for survival and hard-load bias balance."""

    return _make_choicewise_operation(config)(
        x, up_weight, up_bias, down_weight, down_bias
    )


def _normalize_sums(
    soft_sums: jax.Array,
    load_sums: jax.Array,
    active_sums: jax.Array,
    alive_sum: jax.Array,
    tokens: jax.Array,
    up_bias: jax.Array,
    *,
    config: FuzzyTopKBalanceConfig,
) -> jax.Array:
    choices = jnp.float32(soft_sums.shape[-1])
    if config.mode == "switch":
        importance = soft_sums / tokens.astype(jnp.float32)
        load = lax.stop_gradient(load_sums / tokens.astype(jnp.float32))
        switch_loss = choices * jnp.mean(jnp.sum(load * importance, axis=-1))
        importance_loss = jnp.mean(
            choices * jnp.sum(jnp.square(importance), axis=-1) - 1.0
        )
    elif config.mode == "importance":
        importance = soft_sums / tokens.astype(jnp.float32)
        switch_loss = jnp.asarray(0.0, jnp.float32)
        importance_loss = jnp.mean(
            choices * jnp.sum(jnp.square(importance), axis=-1) - 1.0
        )
    elif config.mode == "bias":
        load = lax.stop_gradient(load_sums / tokens.astype(jnp.float32))
        centered_load = load - jnp.float32(1.0) / choices
        grouped_bias = up_bias.astype(jnp.float32).reshape(load.shape)
        report = choices * jnp.mean(jnp.sum(jnp.square(centered_load), axis=-1))
        surrogate = choices * jnp.mean(jnp.sum(centered_load * grouped_bias, axis=-1))
        switch_loss = lax.stop_gradient(report - surrogate) + surrogate
        importance_loss = jnp.asarray(0.0, jnp.float32)
    else:
        switch_loss = jnp.asarray(0.0, jnp.float32)
        importance_loss = jnp.asarray(0.0, jnp.float32)
    if config.alive_mode == "frequency_floor":
        activation_frequency = lax.stop_gradient(
            active_sums / tokens.astype(jnp.float32)
        )
        target = jnp.float32(config.alive_target) / choices
        deficit = jax.nn.relu(target - activation_frequency)
        grouped_bias = up_bias.astype(jnp.float32).reshape(active_sums.shape)
        report = choices * jnp.mean(jnp.sum(jnp.square(deficit), axis=-1))
        surrogate = choices * jnp.mean(
            jnp.sum(-jnp.float32(2.0) * deficit * grouped_bias, axis=-1)
        )
        alive_loss = lax.stop_gradient(report - surrogate) + surrogate
    else:
        alive_loss = alive_sum / (tokens.astype(jnp.float32) * config.top_k)
    return jnp.stack((switch_loss, importance_loss, alive_loss)).astype(jnp.float32)


def fuzzy_topk_mlp_with_balance(
    x: jax.Array,
    up_weight: jax.Array,
    up_bias: jax.Array,
    down_weight: jax.Array,
    down_bias: jax.Array,
    *,
    config: FuzzyTopKBalanceConfig,
) -> tuple[jax.Array, jax.Array]:
    """Apply fuzzy Top-K and return local-batch differentiable objectives."""

    _validate_inputs(x, up_weight, up_bias, down_weight, down_bias, config)
    operation = (
        _balanced_choicewise_fuzzy_topk_mlp
        if config.compute_soft_statistics
        else _low_overhead_choicewise_fuzzy_topk_mlp
    )
    output, soft_sums, load_sums, active_sums, alive_sum = operation(
        x,
        up_weight,
        up_bias,
        down_weight,
        down_bias,
        config=config,
    )
    tokens = jnp.asarray(math.prod(x.shape[:-1]), jnp.float32)
    return output, _normalize_sums(
        soft_sums,
        load_sums,
        active_sums,
        alive_sum,
        tokens,
        up_bias,
        config=config,
    )


def make_mesh_fuzzy_topk_mlp_with_balance(
    *, config: FuzzyTopKBalanceConfig, mesh: Mesh
) -> FuzzyTopKBalanceCallable:
    """Build a data-sharded fuzzy MLP with global-batch balance objectives."""

    def local_operation(x, up_weight, up_bias, down_weight, down_bias):
        _validate_inputs(x, up_weight, up_bias, down_weight, down_bias, config)
        operation = (
            _balanced_choicewise_fuzzy_topk_mlp
            if config.compute_soft_statistics
            else _low_overhead_choicewise_fuzzy_topk_mlp
        )
        output, soft_sums, load_sums, active_sums, alive_sum = operation(
            x,
            up_weight,
            up_bias,
            down_weight,
            down_bias,
            config=config,
        )
        global_soft_sums = lax.psum(soft_sums, "data")
        global_load_sums = lax.psum(load_sums, "data")
        global_active_sums = lax.psum(active_sums, "data")
        global_alive_sum = lax.psum(alive_sum, "data")
        global_tokens = lax.psum(
            jnp.asarray(math.prod(x.shape[:-1]), jnp.float32), "data"
        )
        statistics = _normalize_sums(
            global_soft_sums,
            global_load_sums,
            global_active_sums,
            global_alive_sum,
            global_tokens,
            up_bias,
            config=config,
        )
        return output, statistics

    batch_partition = P("data", None, None)
    return jax.shard_map(
        local_operation,
        mesh=mesh,
        in_specs=(batch_partition, P(), P(), P(), P()),
        out_specs=(batch_partition, P()),
        check_vma=False,
    )


def naive_fuzzy_topk_mlp_with_balance(
    x: jax.Array,
    up_weight: jax.Array,
    up_bias: jax.Array,
    down_weight: jax.Array,
    down_bias: jax.Array,
    *,
    config: FuzzyTopKBalanceConfig,
) -> tuple[jax.Array, jax.Array]:
    """Literal dense-hidden oracle used to validate values and gradients."""

    _validate_inputs(x, up_weight, up_bias, down_weight, down_bias, config)
    hidden = _preactivations(x, up_weight, up_bias)
    values, indices = fuzzy_topk_relu(hidden, top_k=config.top_k)
    sparse_hidden = jnp.zeros_like(hidden)
    sparse_hidden = jnp.put_along_axis(
        sparse_hidden, indices, values, axis=-1, inplace=False
    )
    output = jnp.einsum(
        "...h,hd->...d",
        sparse_hidden,
        down_weight.astype(x.dtype),
        preferred_element_type=jnp.float32,
    )
    output = (output + down_bias.astype(x.dtype)).astype(x.dtype)
    choices = hidden.shape[-1] // config.top_k
    grouped = hidden.reshape((*hidden.shape[:-1], config.top_k, choices))
    winners = jnp.argmax(grouped, axis=-1)
    maxima = jnp.max(grouped, axis=-1)
    soft_sums, load_sums, active_sums, alive_sum, _deficit = _raw_balance_sums(
        hidden,
        winners,
        maxima,
        top_k=config.top_k,
        temperature=config.temperature,
        alive_margin=config.alive_margin,
        compute_soft_statistics=config.compute_soft_statistics,
        compute_load_statistics=config.compute_load_statistics,
        compute_activation_statistics=config.compute_activation_statistics,
        token_margin_alive=config.alive_mode == "token_margin",
    )
    tokens = jnp.asarray(math.prod(x.shape[:-1]), jnp.float32)
    return output, _normalize_sums(
        soft_sums,
        load_sums,
        active_sums,
        alive_sum,
        tokens,
        up_bias,
        config=config,
    )


__all__ = (
    "FUZZY_BALANCE_STAT_NAMES",
    "FuzzyTopKBalanceCallable",
    "FuzzyTopKBalanceConfig",
    "fuzzy_topk_mlp_with_balance",
    "make_mesh_fuzzy_topk_mlp_with_balance",
    "naive_fuzzy_topk_mlp_with_balance",
)
