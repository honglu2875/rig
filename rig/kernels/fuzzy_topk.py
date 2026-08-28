"""Grouped approximate Top-K with feature-specific decoder rows.

The hidden dictionary is partitioned into ``top_k`` fixed groups and one
winner is retained from every group.  Feature order is exchangeable at random
initialization, so a contiguous layout has the distribution of a one-time
random permutation without paying for a runtime shuffle.

The reference path is the literal gather implementation.  The choicewise path
has identical values and gradients, but visits the small alternatives axis and
uses regular dense contractions instead of a global TopK, a loop over
``top_k``, or indirect ``[tokens, top_k, d_model]`` decoder reads.
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


Backend = Literal["choicewise", "reference"]
FuzzyTopKCallable = Callable[
    [jax.Array, jax.Array, jax.Array, jax.Array, jax.Array], jax.Array
]
FuzzyTopKDiagnosticCallable = Callable[
    [jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    tuple[jax.Array, jax.Array],
]
FUZZY_FEATURE_STAT_NAMES = (
    "fuzzy.winner_frequency",
    "fuzzy.activation_frequency",
    "fuzzy.activation_mean",
    "fuzzy.activation_rms",
)


@dataclass(frozen=True, slots=True)
class FuzzyTopKConfig:
    """Static approximate-selection and execution contract."""

    top_k: int
    backend: Backend = "choicewise"

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.backend not in ("choicewise", "reference"):
            raise ValueError(f"unknown fuzzy TopK backend: {self.backend!r}")


def _validate_inputs(
    x: jax.Array,
    up_weight: jax.Array,
    up_bias: jax.Array,
    down_weight: jax.Array,
    down_bias: jax.Array,
    config: FuzzyTopKConfig,
) -> tuple[int, int, int]:
    if x.ndim < 2:
        raise ValueError(f"x must have at least two dimensions, got {x.shape}")
    model_width = x.shape[-1]
    if up_weight.ndim != 2 or up_weight.shape[0] != model_width:
        raise ValueError(
            "up_weight must have shape "
            f"({model_width}, hidden_width), got {up_weight.shape}"
        )
    hidden_width = up_weight.shape[1]
    if up_bias.shape != (hidden_width,):
        raise ValueError(
            f"up_bias must have shape {(hidden_width,)}, got {up_bias.shape}"
        )
    if down_weight.shape != (hidden_width, model_width):
        raise ValueError(
            "down_weight must have shape "
            f"{(hidden_width, model_width)}, got {down_weight.shape}"
        )
    if down_bias.shape != (model_width,):
        raise ValueError(
            f"down_bias must have shape {(model_width,)}, got {down_bias.shape}"
        )
    if config.top_k > hidden_width:
        raise ValueError(f"top_k {config.top_k} exceeds hidden width {hidden_width}")
    if hidden_width % config.top_k:
        raise ValueError(
            f"hidden width {hidden_width} must be divisible by top_k {config.top_k}"
        )
    return model_width, hidden_width, hidden_width // config.top_k


def fuzzy_topk_relu(
    preactivations: jax.Array, *, top_k: int
) -> tuple[jax.Array, jax.Array]:
    """Keep one positive winner from each fixed random-equivalent group.

    Returns ``top_k`` values and their original feature indices.  Reducing
    before ReLU is value-equivalent to reducing ``ReLU(preactivations)`` and
    avoids a full-width ReLU mask.  An all-negative group can report any local
    winner because its value and every gradient are zero.
    """

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    hidden_width = preactivations.shape[-1]
    if hidden_width % top_k:
        raise ValueError(
            f"hidden width {hidden_width} must be divisible by top_k {top_k}"
        )
    choices = hidden_width // top_k
    grouped = preactivations.reshape((*preactivations.shape[:-1], top_k, choices))
    maximum = jnp.max(grouped, axis=-1)
    winners = jnp.argmax(grouped, axis=-1)
    offsets = jnp.arange(top_k, dtype=winners.dtype) * choices
    return jax.nn.relu(maximum), winners + offsets


def naive_fuzzy_topk_mlp(
    x: jax.Array,
    up_weight: jax.Array,
    up_bias: jax.Array,
    down_weight: jax.Array,
    down_bias: jax.Array,
    *,
    top_k: int,
) -> jax.Array:
    """Literal score -> grouped TopK-ReLU -> feature-specific dense oracle."""

    config = FuzzyTopKConfig(top_k=top_k, backend="reference")
    _validate_inputs(x, up_weight, up_bias, down_weight, down_bias, config)
    hidden = jnp.einsum(
        "...d,dh->...h",
        x,
        up_weight.astype(x.dtype),
        preferred_element_type=jnp.float32,
    )
    hidden = hidden + up_bias.astype(x.dtype)
    values, indices = fuzzy_topk_relu(hidden, top_k=top_k)
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
    return (output + down_bias.astype(x.dtype)).astype(x.dtype)


def _selection(
    x: jax.Array,
    up_weight: jax.Array,
    up_bias: jax.Array,
    *,
    top_k: int,
) -> tuple[jax.Array, jax.Array]:
    hidden = jnp.einsum(
        "...d,dh->...h",
        x,
        up_weight.astype(x.dtype),
        preferred_element_type=jnp.float32,
    )
    hidden = hidden + up_bias.astype(x.dtype)
    values, indices = fuzzy_topk_relu(hidden, top_k=top_k)
    choices = hidden.shape[-1] // top_k
    return values.astype(x.dtype), indices % choices


def _reference_forward(
    x: jax.Array,
    up_weight: jax.Array,
    up_bias: jax.Array,
    down_weight: jax.Array,
    down_bias: jax.Array,
    *,
    top_k: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    values, local_winners = _selection(x, up_weight, up_bias, top_k=top_k)
    choices = up_weight.shape[1] // top_k
    offsets = jnp.arange(top_k, dtype=local_winners.dtype) * choices
    indices = local_winners + offsets
    selected_down = down_weight[indices]
    output = jnp.einsum(
        "...k,...kd->...d",
        values,
        selected_down.astype(x.dtype),
        preferred_element_type=jnp.float32,
    )
    return (
        (output + down_bias.astype(x.dtype)).astype(x.dtype),
        values,
        local_winners,
    )


def _choicewise_forward(
    x: jax.Array,
    up_weight: jax.Array,
    up_bias: jax.Array,
    down_weight: jax.Array,
    down_bias: jax.Array,
    *,
    top_k: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    values, winners = _selection(x, up_weight, up_bias, top_k=top_k)
    choices = up_weight.shape[1] // top_k
    grouped_down = down_weight.reshape((top_k, choices, x.shape[-1]))
    accumulator = jnp.zeros((*x.shape[:-1], x.shape[-1]), jnp.float32)

    def visit_choice(choice, output):
        active = jnp.where(winners == choice, values, 0.0)
        return output + jnp.einsum(
            "...k,kd->...d",
            active,
            grouped_down[:, choice, :].astype(x.dtype),
            preferred_element_type=jnp.float32,
        )

    output = lax.fori_loop(0, choices, visit_choice, accumulator)
    return (
        (output + down_bias.astype(jnp.float32)).astype(x.dtype),
        values,
        winners,
    )


def _feature_stat_sums(
    values: jax.Array, winners: jax.Array, *, choices: int
) -> jax.Array:
    """Return exact per-feature batch sums as ``[stat, hidden_width]``.

    The four raw statistics are winner count, positive-winner count, positive
    activation sum, and positive activation squared sum. Normalization waits
    until after a mesh-wide psum so multi-device frequencies describe the
    global batch rather than one shard.
    """

    top_k = values.shape[-1]
    reduction_axes = tuple(range(values.ndim - 1))
    grouped = jnp.zeros((4, top_k, choices), jnp.float32)

    def visit_choice(choice, statistics):
        winner_mask = winners == choice
        active = jnp.where(winner_mask, values, 0.0).astype(jnp.float32)
        choice_statistics = jnp.stack(
            (
                jnp.sum(winner_mask.astype(jnp.float32), axis=reduction_axes),
                jnp.sum((active > 0.0).astype(jnp.float32), axis=reduction_axes),
                jnp.sum(active, axis=reduction_axes),
                jnp.sum(jnp.square(active), axis=reduction_axes),
            )
        )
        return statistics.at[:, :, choice].set(choice_statistics)

    return lax.fori_loop(0, choices, visit_choice, grouped).reshape(
        (4, top_k * choices)
    )


def _normalize_feature_stat_sums(sums: jax.Array, tokens: jax.Array) -> jax.Array:
    denominator = tokens.astype(jnp.float32)
    return jnp.stack(
        (
            sums[0] / denominator,
            sums[1] / denominator,
            sums[2] / denominator,
            jnp.sqrt(jnp.maximum(sums[3] / denominator, 0.0)),
        )
    ).astype(jnp.float32)


def _choicewise_forward_with_statistics(
    x: jax.Array,
    up_weight: jax.Array,
    up_bias: jax.Array,
    down_weight: jax.Array,
    down_bias: jax.Array,
    *,
    top_k: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Choicewise forward with feature reductions fused into its four passes."""

    values, winners = _selection(x, up_weight, up_bias, top_k=top_k)
    choices = up_weight.shape[1] // top_k
    grouped_down = down_weight.reshape((top_k, choices, x.shape[-1]))
    reduction_axes = tuple(range(values.ndim - 1))
    initial = (
        jnp.zeros((*x.shape[:-1], x.shape[-1]), jnp.float32),
        jnp.zeros((4, top_k, choices), jnp.float32),
    )

    def visit_choice(choice, carry):
        output, statistics = carry
        winner_mask = winners == choice
        active = jnp.where(winner_mask, values, 0.0)
        output = output + jnp.einsum(
            "...k,kd->...d",
            active,
            grouped_down[:, choice, :].astype(x.dtype),
            preferred_element_type=jnp.float32,
        )
        active32 = active.astype(jnp.float32)
        choice_statistics = jnp.stack(
            (
                jnp.sum(winner_mask.astype(jnp.float32), axis=reduction_axes),
                jnp.sum((active32 > 0.0).astype(jnp.float32), axis=reduction_axes),
                jnp.sum(active32, axis=reduction_axes),
                jnp.sum(jnp.square(active32), axis=reduction_axes),
            )
        )
        statistics = statistics.at[:, :, choice].set(choice_statistics)
        return output, statistics

    output, statistics = lax.fori_loop(0, choices, visit_choice, initial)
    return (
        (output + down_bias.astype(jnp.float32)).astype(x.dtype),
        values,
        winners,
        statistics.reshape((4, top_k * choices)),
    )


def _choicewise_backward(
    residuals: tuple[jax.Array, ...],
    output_cotangent: jax.Array,
    *,
    top_k: int,
) -> tuple[jax.Array, ...]:
    """Shared reverse rule for ordinary and feature-diagnostic executions."""

    x, up_weight, up_bias, down_weight, down_bias, values, winners = residuals
    model_width, hidden_width = up_weight.shape
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
        preactivation_cotangent = jnp.where(
            winner_mask & (flat_values > 0.0), value_cotangent, 0.0
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
def _make_choicewise_fuzzy_topk(top_k: int):
    """Build a custom VJP which loops over choices, never active features."""

    @jax.custom_vjp
    def operation(x, up_weight, up_bias, down_weight, down_bias):
        output, _, _ = _choicewise_forward(
            x,
            up_weight,
            up_bias,
            down_weight,
            down_bias,
            top_k=top_k,
        )
        return output

    def forward_rule(x, up_weight, up_bias, down_weight, down_bias):
        output, values, winners = _choicewise_forward(
            x,
            up_weight,
            up_bias,
            down_weight,
            down_bias,
            top_k=top_k,
        )
        return output, (
            x,
            up_weight,
            up_bias,
            down_weight,
            down_bias,
            values,
            winners,
        )

    def backward_rule(residuals, output_cotangent):
        return _choicewise_backward(residuals, output_cotangent, top_k=top_k)

    operation.defvjp(forward_rule, backward_rule)
    return operation


@functools.cache
def _make_choicewise_fuzzy_topk_with_statistics(top_k: int):
    """Build the same custom VJP while exposing stop-gradient feature sums."""

    @jax.custom_vjp
    def operation(x, up_weight, up_bias, down_weight, down_bias):
        output, _, _, statistics = _choicewise_forward_with_statistics(
            x,
            up_weight,
            up_bias,
            down_weight,
            down_bias,
            top_k=top_k,
        )
        return output, lax.stop_gradient(statistics)

    def forward_rule(x, up_weight, up_bias, down_weight, down_bias):
        output, values, winners, statistics = _choicewise_forward_with_statistics(
            x,
            up_weight,
            up_bias,
            down_weight,
            down_bias,
            top_k=top_k,
        )
        residuals = (
            x,
            up_weight,
            up_bias,
            down_weight,
            down_bias,
            values,
            winners,
        )
        return (output, lax.stop_gradient(statistics)), residuals

    def backward_rule(residuals, cotangents):
        output_cotangent, _statistics_cotangent = cotangents
        return _choicewise_backward(residuals, output_cotangent, top_k=top_k)

    operation.defvjp(forward_rule, backward_rule)
    return operation


@functools.partial(jax.jit, static_argnames=("top_k",))
def _choicewise_fuzzy_topk_mlp(
    x: jax.Array,
    up_weight: jax.Array,
    up_bias: jax.Array,
    down_weight: jax.Array,
    down_bias: jax.Array,
    *,
    top_k: int,
) -> jax.Array:
    """Named regular-contraction boundary for profiling and FLOP accounting."""

    return _make_choicewise_fuzzy_topk(top_k)(
        x, up_weight, up_bias, down_weight, down_bias
    )


@functools.partial(jax.jit, static_argnames=("top_k",))
def _choicewise_fuzzy_topk_mlp_with_statistics(
    x: jax.Array,
    up_weight: jax.Array,
    up_bias: jax.Array,
    down_weight: jax.Array,
    down_bias: jax.Array,
    *,
    top_k: int,
) -> tuple[jax.Array, jax.Array]:
    """Diagnostic boundary; reductions share the four decoder-choice passes."""

    return _make_choicewise_fuzzy_topk_with_statistics(top_k)(
        x, up_weight, up_bias, down_weight, down_bias
    )


@functools.partial(jax.jit, static_argnames=("top_k",))
def _reference_fuzzy_topk_mlp(
    x: jax.Array,
    up_weight: jax.Array,
    up_bias: jax.Array,
    down_weight: jax.Array,
    down_bias: jax.Array,
    *,
    top_k: int,
) -> jax.Array:
    output, _, _ = _reference_forward(
        x, up_weight, up_bias, down_weight, down_bias, top_k=top_k
    )
    return output


@functools.partial(jax.jit, static_argnames=("top_k",))
def _reference_fuzzy_topk_mlp_with_statistics(
    x: jax.Array,
    up_weight: jax.Array,
    up_bias: jax.Array,
    down_weight: jax.Array,
    down_bias: jax.Array,
    *,
    top_k: int,
) -> tuple[jax.Array, jax.Array]:
    output, values, winners = _reference_forward(
        x, up_weight, up_bias, down_weight, down_bias, top_k=top_k
    )
    choices = up_weight.shape[1] // top_k
    statistics = _feature_stat_sums(values, winners, choices=choices)
    return output, lax.stop_gradient(statistics)


def fuzzy_topk_mlp(
    x: jax.Array,
    up_weight: jax.Array,
    up_bias: jax.Array,
    down_weight: jax.Array,
    down_bias: jax.Array,
    *,
    config: FuzzyTopKConfig,
) -> jax.Array:
    """Apply grouped approximate Top-K with feature-specific decoder rows."""

    _validate_inputs(x, up_weight, up_bias, down_weight, down_bias, config)
    operation = {
        "choicewise": _choicewise_fuzzy_topk_mlp,
        "reference": _reference_fuzzy_topk_mlp,
    }[config.backend]
    return operation(
        x,
        up_weight,
        up_bias,
        down_weight,
        down_bias,
        top_k=config.top_k,
    )


def _fuzzy_topk_mlp_with_stat_sums(
    x: jax.Array,
    up_weight: jax.Array,
    up_bias: jax.Array,
    down_weight: jax.Array,
    down_bias: jax.Array,
    *,
    config: FuzzyTopKConfig,
) -> tuple[jax.Array, jax.Array]:
    _validate_inputs(x, up_weight, up_bias, down_weight, down_bias, config)
    operation = {
        "choicewise": _choicewise_fuzzy_topk_mlp_with_statistics,
        "reference": _reference_fuzzy_topk_mlp_with_statistics,
    }[config.backend]
    return operation(
        x,
        up_weight,
        up_bias,
        down_weight,
        down_bias,
        top_k=config.top_k,
    )


def fuzzy_topk_mlp_with_diagnostics(
    x: jax.Array,
    up_weight: jax.Array,
    up_bias: jax.Array,
    down_weight: jax.Array,
    down_bias: jax.Array,
    *,
    config: FuzzyTopKConfig,
) -> tuple[jax.Array, jax.Array]:
    """Apply fuzzy TopK and return exact local-batch feature statistics."""

    output, sums = _fuzzy_topk_mlp_with_stat_sums(
        x, up_weight, up_bias, down_weight, down_bias, config=config
    )
    tokens = jnp.asarray(math.prod(x.shape[:-1]), jnp.float32)
    return output, _normalize_feature_stat_sums(sums, tokens)


def make_mesh_fuzzy_topk_mlp(
    *, config: FuzzyTopKConfig, mesh: Mesh
) -> FuzzyTopKCallable:
    """Build a data-sharded boundary with replicated MLP parameters."""

    def local_operation(x, up_weight, up_bias, down_weight, down_bias):
        return fuzzy_topk_mlp(
            x,
            up_weight,
            up_bias,
            down_weight,
            down_bias,
            config=config,
        )

    batch_partition = P("data", None, None)
    return jax.shard_map(
        local_operation,
        mesh=mesh,
        in_specs=(batch_partition, P(), P(), P(), P()),
        out_specs=batch_partition,
        check_vma=False,
    )


def make_mesh_fuzzy_topk_mlp_with_diagnostics(
    *, config: FuzzyTopKConfig, mesh: Mesh
) -> FuzzyTopKDiagnosticCallable:
    """Build a data-sharded MLP returning global-batch feature statistics."""

    def local_operation(x, up_weight, up_bias, down_weight, down_bias):
        output, local_sums = _fuzzy_topk_mlp_with_stat_sums(
            x,
            up_weight,
            up_bias,
            down_weight,
            down_bias,
            config=config,
        )
        global_sums = lax.psum(local_sums, "data")
        global_tokens = lax.psum(
            jnp.asarray(math.prod(x.shape[:-1]), jnp.float32), "data"
        )
        return output, _normalize_feature_stat_sums(global_sums, global_tokens)

    batch_partition = P("data", None, None)
    return jax.shard_map(
        local_operation,
        mesh=mesh,
        in_specs=(batch_partition, P(), P(), P(), P()),
        out_specs=(batch_partition, P()),
        check_vma=False,
    )


__all__ = (
    "Backend",
    "FUZZY_FEATURE_STAT_NAMES",
    "FuzzyTopKCallable",
    "FuzzyTopKConfig",
    "FuzzyTopKDiagnosticCallable",
    "fuzzy_topk_mlp",
    "fuzzy_topk_mlp_with_diagnostics",
    "fuzzy_topk_relu",
    "make_mesh_fuzzy_topk_mlp",
    "make_mesh_fuzzy_topk_mlp_with_diagnostics",
    "naive_fuzzy_topk_mlp",
)
