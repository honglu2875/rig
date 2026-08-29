"""Paper-inspired ghost gradients for dead fuzzy Top-K features.

The ordinary fuzzy MLP keeps one ReLU-positive winner from each fixed feature
group.  This module leaves that numerical forward pass unchanged and adds a
training-only ghost path for features that have not activated recently.

The paper's AuxK loss reconstructs an autoencoder residual.  A transformer MLP
does not reconstruct its input, so there is no corresponding residual target.
Instead, the ghost path uses the downstream language-model cotangent as its
training signal.  It is exactly zero in the primal computation, while its
custom reverse rule updates dead feature rows of ``W_up``, ``b_up``, and
``W_down``.  The auxiliary encoder input is stop-gradient, so this training
signal cannot perturb earlier residual-stream representations.

For efficiency, the auxiliary selection cycles through cohorts of the ordinary
random groups.  With ``aux_k = top_k / 8``, every group is eligible once per
eight optimizer steps and the extra decoder-side reverse work is one eighth of
the ordinary fuzzy path.  Feature scoring is shared with the main forward.
"""

from __future__ import annotations

from dataclasses import dataclass
import functools
import math
from typing import Callable

import jax
from jax import lax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P


FuzzyTopKAuxKCallable = Callable[
    [
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
    ],
    tuple[jax.Array, jax.Array],
]


@dataclass(frozen=True, slots=True)
class FuzzyTopKAuxKConfig:
    """Static grouped-selection and ghost-gradient contract."""

    top_k: int
    aux_k: int
    coefficient: float = 1.0 / 32.0

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.aux_k <= 0:
            raise ValueError("aux_k must be positive")
        if self.aux_k > self.top_k or self.top_k % self.aux_k:
            raise ValueError("aux_k must divide top_k")
        if not math.isfinite(self.coefficient) or self.coefficient < 0.0:
            raise ValueError("coefficient must be finite and nonnegative")

    @property
    def cohort_count(self) -> int:
        return self.top_k // self.aux_k


def _validate_inputs(
    x: jax.Array,
    up_weight: jax.Array,
    up_bias: jax.Array,
    down_weight: jax.Array,
    down_bias: jax.Array,
    dead_mask: jax.Array,
    cohort: jax.Array,
    config: FuzzyTopKAuxKConfig,
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
    if dead_mask.shape != (hidden_width,) or dead_mask.dtype != jnp.bool_:
        raise ValueError(
            f"dead_mask must be boolean with shape {(hidden_width,)}, "
            f"got {dead_mask.dtype} {dead_mask.shape}"
        )
    if cohort.shape != () or not jnp.issubdtype(cohort.dtype, jnp.integer):
        raise ValueError("cohort must be a scalar integer array")
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


def _active_sums(
    winners: jax.Array, maxima: jax.Array, *, choices: int
) -> jax.Array:
    top_k = winners.shape[-1]
    reduction_axes = tuple(range(winners.ndim - 1))
    counts = jnp.zeros((top_k, choices), jnp.float32)

    def visit_choice(choice, values):
        active = (winners == choice) & (maxima > 0.0)
        return values.at[:, choice].set(
            jnp.sum(active.astype(jnp.float32), axis=reduction_axes)
        )

    return lax.fori_loop(0, choices, visit_choice, counts).reshape(-1)


def _auxiliary_selection(
    hidden: jax.Array,
    dead_mask: jax.Array,
    cohort: jax.Array,
    *,
    config: FuzzyTopKAuxKConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    choices = hidden.shape[-1] // config.top_k
    grouped_hidden = hidden.reshape(
        (*hidden.shape[:-1], config.top_k, choices)
    )
    grouped_dead = dead_mask.reshape((config.top_k, choices))
    cohort = jnp.mod(cohort, jnp.asarray(config.cohort_count, cohort.dtype))
    group_ids = cohort + config.cohort_count * jnp.arange(config.aux_k)
    candidate_hidden = jnp.take(grouped_hidden, group_ids, axis=-2)
    candidate_dead = jnp.take(grouped_dead, group_ids, axis=0)
    masked = jnp.where(
        candidate_dead,
        candidate_hidden,
        jnp.asarray(jnp.finfo(hidden.dtype).min, hidden.dtype),
    )
    maxima = jnp.max(masked, axis=-1)
    winners = jnp.argmax(masked, axis=-1)
    has_dead = jnp.any(candidate_dead, axis=-1)
    values = jnp.where(has_dead, jax.nn.relu(maxima), 0.0).astype(hidden.dtype)
    return values, winners, group_ids


def _forward(
    x: jax.Array,
    up_weight: jax.Array,
    up_bias: jax.Array,
    down_weight: jax.Array,
    down_bias: jax.Array,
    dead_mask: jax.Array,
    cohort: jax.Array,
    *,
    config: FuzzyTopKAuxKConfig,
):
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
    counts = lax.stop_gradient(
        _active_sums(
            winners,
            maxima,
            choices=hidden.shape[-1] // config.top_k,
        )
    )
    aux_values, aux_winners, group_ids = _auxiliary_selection(
        hidden, dead_mask, cohort, config=config
    )
    return output, counts, values, winners, aux_values, aux_winners, group_ids


def _backward(
    residuals: tuple[jax.Array, ...],
    cotangents: tuple[jax.Array, ...],
    *,
    config: FuzzyTopKAuxKConfig,
):
    (
        x,
        up_weight,
        up_bias,
        down_weight,
        down_bias,
        main_values,
        main_winners,
        aux_values,
        aux_winners,
        group_ids,
    ) = residuals
    output_cotangent, _counts_cotangent = cotangents
    model_width, hidden_width = up_weight.shape
    top_k = config.top_k
    choices = hidden_width // top_k
    tokens = math.prod(x.shape[:-1])

    flat_x = x.reshape((tokens, model_width)).astype(jnp.float32)
    flat_output_cotangent = output_cotangent.reshape(
        (tokens, model_width)
    ).astype(jnp.float32)
    flat_values = main_values.reshape((tokens, top_k)).astype(jnp.float32)
    flat_winners = main_winners.reshape((tokens, top_k))
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

    def visit_main_choice(choice, carry):
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
        0, choices, visit_main_choice, initial
    )

    # The auxiliary cohort touches only aux_k ordinary groups.  It uses the
    # downstream cotangent for parameter gradients, but intentionally adds
    # nothing to dx: its encoder input is stop-gradient by contract.
    flat_aux_values = aux_values.reshape((tokens, config.aux_k)).astype(jnp.float32)
    flat_aux_winners = aux_winners.reshape((tokens, config.aux_k))
    selected_down = jnp.take(grouped_down, group_ids, axis=0)
    aux_up_gradient = jnp.zeros(
        (model_width, config.aux_k, choices), jnp.float32
    )
    aux_bias_gradient = jnp.zeros((config.aux_k, choices), jnp.float32)
    aux_down_gradient = jnp.zeros(
        (config.aux_k, choices, model_width), jnp.float32
    )
    scaled_output_cotangent = (
        jnp.float32(config.coefficient) * flat_output_cotangent
    )

    def visit_aux_choice(choice, carry):
        selected_up_gradient, selected_bias_gradient, selected_down_gradient = carry
        winner_mask = flat_aux_winners == choice
        choice_values = jnp.where(winner_mask, flat_aux_values, 0.0)
        value_cotangent = jnp.einsum(
            "td,kd->tk",
            scaled_output_cotangent,
            selected_down[:, choice, :],
            preferred_element_type=jnp.float32,
        )
        preactivation_cotangent = jnp.where(
            winner_mask & (flat_aux_values > 0.0), value_cotangent, 0.0
        )
        selected_up_gradient = selected_up_gradient.at[:, :, choice].set(
            jnp.einsum(
                "td,tk->dk",
                flat_x,
                preactivation_cotangent,
                preferred_element_type=jnp.float32,
            )
        )
        selected_bias_gradient = selected_bias_gradient.at[:, choice].set(
            jnp.sum(preactivation_cotangent, axis=0)
        )
        selected_down_gradient = selected_down_gradient.at[:, choice, :].set(
            jnp.einsum(
                "tk,td->kd",
                choice_values,
                scaled_output_cotangent,
                preferred_element_type=jnp.float32,
            )
        )
        return selected_up_gradient, selected_bias_gradient, selected_down_gradient

    aux_up_gradient, aux_bias_gradient, aux_down_gradient = lax.fori_loop(
        0,
        choices,
        visit_aux_choice,
        (aux_up_gradient, aux_bias_gradient, aux_down_gradient),
    )
    up_gradient = up_gradient.at[:, group_ids, :].add(aux_up_gradient)
    up_bias_gradient = up_bias_gradient.at[group_ids, :].add(aux_bias_gradient)
    down_gradient = down_gradient.at[group_ids, :, :].add(aux_down_gradient)
    down_bias_gradient = jnp.sum(flat_output_cotangent, axis=0)

    return (
        dx.reshape(x.shape).astype(x.dtype),
        up_gradient.reshape(up_weight.shape).astype(up_weight.dtype),
        up_bias_gradient.reshape(up_bias.shape).astype(up_bias.dtype),
        down_gradient.reshape(down_weight.shape).astype(down_weight.dtype),
        down_bias_gradient.astype(down_bias.dtype),
        None,
        None,
    )


@functools.cache
def _make_operation(config: FuzzyTopKAuxKConfig):
    @jax.custom_vjp
    def operation(
        x, up_weight, up_bias, down_weight, down_bias, dead_mask, cohort
    ):
        output, counts, *_ = _forward(
            x,
            up_weight,
            up_bias,
            down_weight,
            down_bias,
            dead_mask,
            cohort,
            config=config,
        )
        return output, counts

    def forward_rule(
        x, up_weight, up_bias, down_weight, down_bias, dead_mask, cohort
    ):
        output, counts, values, winners, aux_values, aux_winners, group_ids = _forward(
            x,
            up_weight,
            up_bias,
            down_weight,
            down_bias,
            dead_mask,
            cohort,
            config=config,
        )
        residuals = (
            x,
            up_weight,
            up_bias,
            down_weight,
            down_bias,
            values,
            winners,
            aux_values,
            aux_winners,
            group_ids,
        )
        return (output, counts), residuals

    def backward_rule(residuals, cotangents):
        return _backward(residuals, cotangents, config=config)

    operation.defvjp(forward_rule, backward_rule)
    return operation


@functools.partial(jax.jit, static_argnames=("config",))
def _choicewise_fuzzy_topk_auxk_mlp(
    x: jax.Array,
    up_weight: jax.Array,
    up_bias: jax.Array,
    down_weight: jax.Array,
    down_bias: jax.Array,
    dead_mask: jax.Array,
    cohort: jax.Array,
    *,
    config: FuzzyTopKAuxKConfig,
) -> tuple[jax.Array, jax.Array]:
    """Named boundary for profiling and physical matrix-FLOP accounting."""

    return _make_operation(config)(
        x, up_weight, up_bias, down_weight, down_bias, dead_mask, cohort
    )


def fuzzy_topk_mlp_with_auxk(
    x: jax.Array,
    up_weight: jax.Array,
    up_bias: jax.Array,
    down_weight: jax.Array,
    down_bias: jax.Array,
    dead_mask: jax.Array,
    cohort: jax.Array,
    *,
    config: FuzzyTopKAuxKConfig,
) -> tuple[jax.Array, jax.Array]:
    """Apply ordinary fuzzy Top-K plus zero-forward dead-feature gradients."""

    _validate_inputs(
        x,
        up_weight,
        up_bias,
        down_weight,
        down_bias,
        dead_mask,
        cohort,
        config,
    )
    return _choicewise_fuzzy_topk_auxk_mlp(
        x,
        up_weight,
        up_bias,
        down_weight,
        down_bias,
        dead_mask,
        cohort,
        config=config,
    )


def naive_fuzzy_topk_mlp_with_auxk(
    x: jax.Array,
    up_weight: jax.Array,
    up_bias: jax.Array,
    down_weight: jax.Array,
    down_bias: jax.Array,
    dead_mask: jax.Array,
    cohort: jax.Array,
    *,
    config: FuzzyTopKAuxKConfig,
) -> tuple[jax.Array, jax.Array]:
    """Literal autodiff oracle for the zero-forward ghost construction."""

    _validate_inputs(
        x,
        up_weight,
        up_bias,
        down_weight,
        down_bias,
        dead_mask,
        cohort,
        config,
    )
    hidden = _preactivations(x, up_weight, up_bias)
    values, winners, maxima = _selection_from_hidden(hidden, top_k=config.top_k)
    output = _choicewise_decode(
        values.astype(x.dtype),
        winners,
        down_weight,
        down_bias,
        model_width=x.shape[-1],
    )
    counts = lax.stop_gradient(
        _active_sums(
            winners,
            maxima,
            choices=hidden.shape[-1] // config.top_k,
        )
    )

    # Recompute only in the oracle so normal autodiff can express the intended
    # stop-gradient input semantics.  The custom implementation above shares
    # the ordinary preactivations and manually omits this path from dX.
    aux_hidden = _preactivations(lax.stop_gradient(x), up_weight, up_bias)
    aux_values, aux_winners, group_ids = _auxiliary_selection(
        aux_hidden, dead_mask, cohort, config=config
    )
    choices = hidden.shape[-1] // config.top_k
    grouped_down = down_weight.reshape(
        (config.top_k, choices, x.shape[-1])
    )
    selected_down = jnp.take(grouped_down, group_ids, axis=0)
    ghost = jnp.zeros((*aux_values.shape[:-1], x.shape[-1]), jnp.float32)

    def visit_choice(choice, accumulator):
        active = jnp.where(aux_winners == choice, aux_values, 0.0)
        return accumulator + jnp.einsum(
            "...k,kd->...d",
            active,
            selected_down[:, choice, :].astype(x.dtype),
            preferred_element_type=jnp.float32,
        )

    ghost = lax.fori_loop(0, choices, visit_choice, ghost).astype(x.dtype)
    zero_forward_ghost = jnp.float32(config.coefficient) * (
        ghost - lax.stop_gradient(ghost)
    )
    return output + zero_forward_ghost.astype(output.dtype), counts


def make_mesh_fuzzy_topk_mlp_with_auxk(
    *, config: FuzzyTopKAuxKConfig, mesh: Mesh
) -> FuzzyTopKAuxKCallable:
    """Build a data-sharded ghost path with global main-activation counts."""

    def local_operation(
        x, up_weight, up_bias, down_weight, down_bias, dead_mask, cohort
    ):
        output, local_counts = fuzzy_topk_mlp_with_auxk(
            x,
            up_weight,
            up_bias,
            down_weight,
            down_bias,
            dead_mask,
            cohort,
            config=config,
        )
        return output, lax.psum(local_counts, "data")

    batch_partition = P("data", None, None)
    return jax.shard_map(
        local_operation,
        mesh=mesh,
        in_specs=(batch_partition, P(), P(), P(), P(), P(), P()),
        out_specs=(batch_partition, P()),
        check_vma=False,
    )


__all__ = (
    "FuzzyTopKAuxKCallable",
    "FuzzyTopKAuxKConfig",
    "fuzzy_topk_mlp_with_auxk",
    "make_mesh_fuzzy_topk_mlp_with_auxk",
    "naive_fuzzy_topk_mlp_with_auxk",
)
