"""Training-only reconstruction decoders for grouped fuzzy Top-K MLPs.

The deployed transformer MLP remains the ordinary fuzzy Top-K operation.  A
second decoder sees the same selected activations during training and learns to
reconstruct the normalized MLP input.  Its loss updates the shared encoder and
the reconstruction decoder, but deliberately sends no gradient to the input
or to the transformer's ordinary DOWN parameters.

The optional AuxK path follows the reconstruction objective from Gao et al.,
"Scaling and evaluating sparse autoencoders": dead features decode a
stop-gradient copy of the main reconstruction residual.  Selection remains the
project's fixed-group approximation.  With ``K=4D`` and ``k_aux=D/2``, one of
eight rotating group cohorts supplies one dead candidate per group.
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


RECONSTRUCTION_STAT_NAMES = ("fuzzy_reconstruction.nmse",)
RECONSTRUCTION_AUXK_STAT_NAMES = (
    "fuzzy_reconstruction.nmse",
    "fuzzy_reconstruction.auxk_nmse",
    "fuzzy_reconstruction.auxk_positive_fraction",
)

FuzzyTopKReconstructionCallable = Callable[
    [
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
    ],
    tuple[jax.Array, jax.Array],
]
FuzzyTopKReconstructionAuxKCallable = Callable[
    [
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
    ],
    tuple[jax.Array, jax.Array, jax.Array],
]


@dataclass(frozen=True, slots=True)
class FuzzyTopKReconstructionConfig:
    """Static selection contract for the train-only reconstruction head."""

    top_k: int
    aux_k: int | None = None

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.aux_k is not None:
            if self.aux_k <= 0:
                raise ValueError("aux_k must be positive")
            if self.aux_k > self.top_k or self.top_k % self.aux_k:
                raise ValueError("aux_k must divide top_k")

    @property
    def auxk_enabled(self) -> bool:
        return self.aux_k is not None

    @property
    def cohort_count(self) -> int:
        if self.aux_k is None:
            raise ValueError("reconstruction-only configuration has no cohorts")
        return self.top_k // self.aux_k


def _validate_inputs(
    x: jax.Array,
    up_weight: jax.Array,
    up_bias: jax.Array,
    down_weight: jax.Array,
    down_bias: jax.Array,
    reconstruction_weight: jax.Array,
    config: FuzzyTopKReconstructionConfig,
    *,
    dead_mask: jax.Array | None = None,
    cohort: jax.Array | None = None,
) -> tuple[int, int, int]:
    if x.ndim < 2:
        raise ValueError(f"x must have at least two dimensions, got {x.shape}")
    model_width = x.shape[-1]
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
    expected_decoder = (hidden_width, model_width)
    if down_weight.shape != expected_decoder:
        raise ValueError(
            f"down_weight must have shape {expected_decoder}, got {down_weight.shape}"
        )
    if reconstruction_weight.shape != expected_decoder:
        raise ValueError(
            "reconstruction_weight must have shape "
            f"{expected_decoder}, got {reconstruction_weight.shape}"
        )
    if down_bias.shape != (model_width,):
        raise ValueError(
            f"down_bias must have shape {(model_width,)}, got {down_bias.shape}"
        )
    if config.top_k > hidden_width or hidden_width % config.top_k:
        raise ValueError(
            f"top_k {config.top_k} must divide hidden width {hidden_width}"
        )
    choices = hidden_width // config.top_k
    if config.auxk_enabled:
        if dead_mask is None or cohort is None:
            raise ValueError("AuxK requires dead_mask and cohort")
        if dead_mask.shape != (hidden_width,) or dead_mask.dtype != jnp.bool_:
            raise ValueError(
                f"dead_mask must be boolean with shape {(hidden_width,)}, "
                f"got {dead_mask.dtype} {dead_mask.shape}"
            )
        if cohort.shape != () or not jnp.issubdtype(cohort.dtype, jnp.integer):
            raise ValueError("cohort must be a scalar integer array")
    elif dead_mask is not None or cohort is not None:
        raise ValueError("reconstruction-only operation cannot receive AuxK state")
    return model_width, hidden_width, choices


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
    weight: jax.Array,
    *,
    model_width: int,
    bias: jax.Array | None = None,
) -> jax.Array:
    top_k = values.shape[-1]
    choices = weight.shape[0] // top_k
    grouped_weight = weight.reshape((top_k, choices, model_width))
    accumulator = jnp.zeros((*values.shape[:-1], model_width), jnp.float32)

    def visit_choice(choice, output):
        active = jnp.where(winners == choice, values, 0.0)
        return output + jnp.einsum(
            "...k,kd->...d",
            active,
            grouped_weight[:, choice, :].astype(values.dtype),
            preferred_element_type=jnp.float32,
        )

    output = lax.fori_loop(0, choices, visit_choice, accumulator)
    if bias is not None:
        output = output + bias.astype(jnp.float32)
    return output.astype(values.dtype)


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
    config: FuzzyTopKReconstructionConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    if config.aux_k is None:
        raise ValueError("AuxK selection requires aux_k")
    choices = hidden.shape[-1] // config.top_k
    grouped_hidden = hidden.reshape((*hidden.shape[:-1], config.top_k, choices))
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


def _centered_square_sum(values: jax.Array) -> jax.Array:
    flat = values.reshape((-1, values.shape[-1])).astype(jnp.float32)
    centered = flat - jnp.mean(flat, axis=0, keepdims=True)
    return jnp.sum(jnp.square(centered), dtype=jnp.float32)


def _forward(
    x: jax.Array,
    up_weight: jax.Array,
    up_bias: jax.Array,
    down_weight: jax.Array,
    down_bias: jax.Array,
    reconstruction_weight: jax.Array,
    dead_mask: jax.Array | None,
    cohort: jax.Array | None,
    *,
    config: FuzzyTopKReconstructionConfig,
):
    hidden = _preactivations(x, up_weight, up_bias)
    values, winners, maxima = _selection_from_hidden(hidden, top_k=config.top_k)
    output = _choicewise_decode(
        values,
        winners,
        down_weight,
        model_width=x.shape[-1],
        bias=down_bias,
    )
    reconstruction = _choicewise_decode(
        values,
        winners,
        reconstruction_weight,
        model_width=x.shape[-1],
    )
    reconstruction_error = (
        reconstruction.astype(jnp.float32) - x.astype(jnp.float32)
    )
    reconstruction_numerator = jnp.sum(
        jnp.square(reconstruction_error), dtype=jnp.float32
    )
    reconstruction_denominator = _centered_square_sum(x)

    if not config.auxk_enabled:
        raw_statistics = jnp.stack(
            (reconstruction_numerator, reconstruction_denominator)
        )
        return (
            output,
            raw_statistics,
            values,
            winners,
            reconstruction_error.astype(x.dtype),
        )

    if dead_mask is None or cohort is None or config.aux_k is None:
        raise AssertionError("AuxK forward lacks declared state")
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
    choices = hidden.shape[-1] // config.top_k
    grouped_reconstruction = reconstruction_weight.reshape(
        (config.top_k, choices, x.shape[-1])
    )
    selected_reconstruction = jnp.take(
        grouped_reconstruction, group_ids, axis=0
    )
    aux_reconstruction = jnp.zeros(
        (*aux_values.shape[:-1], x.shape[-1]), jnp.float32
    )

    def visit_aux_choice(choice, accumulator):
        active = jnp.where(aux_winners == choice, aux_values, 0.0)
        return accumulator + jnp.einsum(
            "...k,kd->...d",
            active,
            selected_reconstruction[:, choice, :].astype(x.dtype),
            preferred_element_type=jnp.float32,
        )

    aux_reconstruction = lax.fori_loop(
        0, choices, visit_aux_choice, aux_reconstruction
    )
    # Exactly as in SAE AuxK, the main reconstruction residual is a detached
    # target.  The reconstruction decoder has no output bias, so no bias
    # correction is required in either the prediction or target.
    residual_target = lax.stop_gradient(
        x.astype(jnp.float32) - reconstruction.astype(jnp.float32)
    )
    aux_error = aux_reconstruction - residual_target
    aux_numerator = jnp.sum(jnp.square(aux_error), dtype=jnp.float32)
    aux_denominator = _centered_square_sum(residual_target)
    aux_positive = jnp.sum((aux_values > 0.0).astype(jnp.float32))
    raw_statistics = jnp.stack(
        (
            reconstruction_numerator,
            reconstruction_denominator,
            aux_numerator,
            aux_denominator,
            aux_positive,
        )
    )
    return (
        output,
        raw_statistics,
        counts,
        values,
        winners,
        reconstruction_error.astype(x.dtype),
        aux_values,
        aux_winners,
        group_ids,
        aux_error.astype(x.dtype),
    )


def _backward(
    residuals: tuple[jax.Array, ...],
    output_cotangent: jax.Array,
    raw_statistics_cotangent: jax.Array,
    *,
    config: FuzzyTopKReconstructionConfig,
):
    if config.auxk_enabled:
        (
            x,
            up_weight,
            up_bias,
            down_weight,
            down_bias,
            reconstruction_weight,
            main_values,
            main_winners,
            reconstruction_error,
            aux_values,
            aux_winners,
            group_ids,
            aux_error,
        ) = residuals
    else:
        (
            x,
            up_weight,
            up_bias,
            down_weight,
            down_bias,
            reconstruction_weight,
            main_values,
            main_winners,
            reconstruction_error,
        ) = residuals

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
    flat_reconstruction_error = reconstruction_error.reshape(
        (tokens, model_width)
    ).astype(jnp.float32)
    reconstruction_cotangent = (
        2.0
        * raw_statistics_cotangent[0].astype(jnp.float32)
        * flat_reconstruction_error
    )
    grouped_up = up_weight.reshape((model_width, top_k, choices)).astype(jnp.float32)
    grouped_down = down_weight.reshape((top_k, choices, model_width)).astype(
        jnp.float32
    )
    grouped_reconstruction = reconstruction_weight.reshape(
        (top_k, choices, model_width)
    ).astype(jnp.float32)

    initial = (
        jnp.zeros((tokens, model_width), jnp.float32),
        jnp.zeros(grouped_up.shape, jnp.float32),
        jnp.zeros((top_k, choices), jnp.float32),
        jnp.zeros(grouped_down.shape, jnp.float32),
        jnp.zeros(grouped_reconstruction.shape, jnp.float32),
    )

    def visit_main_choice(choice, carry):
        dx, up_gradient, up_bias_gradient, down_gradient, reconstruction_gradient = (
            carry
        )
        winner_mask = flat_winners == choice
        choice_values = jnp.where(winner_mask, flat_values, 0.0)
        language_value_cotangent = jnp.einsum(
            "td,kd->tk",
            flat_output_cotangent,
            grouped_down[:, choice, :],
            preferred_element_type=jnp.float32,
        )
        reconstruction_value_cotangent = jnp.einsum(
            "td,kd->tk",
            reconstruction_cotangent,
            grouped_reconstruction[:, choice, :],
            preferred_element_type=jnp.float32,
        )
        active = winner_mask & (flat_values > 0.0)
        language_preactivation = jnp.where(
            active, language_value_cotangent, 0.0
        )
        reconstruction_preactivation = jnp.where(
            active, reconstruction_value_cotangent, 0.0
        )
        combined_preactivation = (
            language_preactivation + reconstruction_preactivation
        )
        # Only the ordinary language-model branch is allowed to alter the
        # incoming residual stream.  Both reconstruction objectives still
        # update W_up and b_up through the combined preactivation cotangent.
        dx = dx + jnp.einsum(
            "tk,dk->td",
            language_preactivation,
            grouped_up[:, :, choice],
            preferred_element_type=jnp.float32,
        )
        up_gradient = up_gradient.at[:, :, choice].set(
            jnp.einsum(
                "td,tk->dk",
                flat_x,
                combined_preactivation,
                preferred_element_type=jnp.float32,
            )
        )
        up_bias_gradient = up_bias_gradient.at[:, choice].set(
            jnp.sum(combined_preactivation, axis=0)
        )
        down_gradient = down_gradient.at[:, choice, :].set(
            jnp.einsum(
                "tk,td->kd",
                choice_values,
                flat_output_cotangent,
                preferred_element_type=jnp.float32,
            )
        )
        reconstruction_gradient = reconstruction_gradient.at[:, choice, :].set(
            jnp.einsum(
                "tk,td->kd",
                choice_values,
                reconstruction_cotangent,
                preferred_element_type=jnp.float32,
            )
        )
        return (
            dx,
            up_gradient,
            up_bias_gradient,
            down_gradient,
            reconstruction_gradient,
        )

    (
        dx,
        up_gradient,
        up_bias_gradient,
        down_gradient,
        reconstruction_gradient,
    ) = lax.fori_loop(0, choices, visit_main_choice, initial)

    if config.auxk_enabled:
        if config.aux_k is None:
            raise AssertionError("AuxK backward lacks aux_k")
        flat_aux_values = aux_values.reshape((tokens, config.aux_k)).astype(
            jnp.float32
        )
        flat_aux_winners = aux_winners.reshape((tokens, config.aux_k))
        flat_aux_error = aux_error.reshape((tokens, model_width)).astype(jnp.float32)
        aux_output_cotangent = (
            2.0
            * raw_statistics_cotangent[2].astype(jnp.float32)
            * flat_aux_error
        )
        selected_reconstruction = jnp.take(
            grouped_reconstruction, group_ids, axis=0
        )
        selected_up_gradient = jnp.zeros(
            (model_width, config.aux_k, choices), jnp.float32
        )
        selected_bias_gradient = jnp.zeros(
            (config.aux_k, choices), jnp.float32
        )
        selected_reconstruction_gradient = jnp.zeros(
            (config.aux_k, choices, model_width), jnp.float32
        )

        def visit_aux_choice(choice, carry):
            aux_up, aux_bias, aux_reconstruction_gradient = carry
            winner_mask = flat_aux_winners == choice
            choice_values = jnp.where(winner_mask, flat_aux_values, 0.0)
            value_cotangent = jnp.einsum(
                "td,kd->tk",
                aux_output_cotangent,
                selected_reconstruction[:, choice, :],
                preferred_element_type=jnp.float32,
            )
            preactivation_cotangent = jnp.where(
                winner_mask & (flat_aux_values > 0.0), value_cotangent, 0.0
            )
            aux_up = aux_up.at[:, :, choice].set(
                jnp.einsum(
                    "td,tk->dk",
                    flat_x,
                    preactivation_cotangent,
                    preferred_element_type=jnp.float32,
                )
            )
            aux_bias = aux_bias.at[:, choice].set(
                jnp.sum(preactivation_cotangent, axis=0)
            )
            aux_reconstruction_gradient = aux_reconstruction_gradient.at[
                :, choice, :
            ].set(
                jnp.einsum(
                    "tk,td->kd",
                    choice_values,
                    aux_output_cotangent,
                    preferred_element_type=jnp.float32,
                )
            )
            return aux_up, aux_bias, aux_reconstruction_gradient

        (
            selected_up_gradient,
            selected_bias_gradient,
            selected_reconstruction_gradient,
        ) = lax.fori_loop(
            0,
            choices,
            visit_aux_choice,
            (
                selected_up_gradient,
                selected_bias_gradient,
                selected_reconstruction_gradient,
            ),
        )
        up_gradient = up_gradient.at[:, group_ids, :].add(selected_up_gradient)
        up_bias_gradient = up_bias_gradient.at[group_ids, :].add(
            selected_bias_gradient
        )
        reconstruction_gradient = reconstruction_gradient.at[group_ids, :, :].add(
            selected_reconstruction_gradient
        )

    down_bias_gradient = jnp.sum(flat_output_cotangent, axis=0)
    result = (
        dx.reshape(x.shape).astype(x.dtype),
        up_gradient.reshape(up_weight.shape).astype(up_weight.dtype),
        up_bias_gradient.reshape(up_bias.shape).astype(up_bias.dtype),
        down_gradient.reshape(down_weight.shape).astype(down_weight.dtype),
        down_bias_gradient.astype(down_bias.dtype),
        reconstruction_gradient.reshape(reconstruction_weight.shape).astype(
            reconstruction_weight.dtype
        ),
    )
    return result


@functools.cache
def _make_reconstruction_operation(config: FuzzyTopKReconstructionConfig):
    if config.auxk_enabled:
        raise ValueError("reconstruction-only operation received AuxK config")

    @jax.custom_vjp
    def operation(x, up_weight, up_bias, down_weight, down_bias, reconstruction_weight):
        output, raw_statistics, *_ = _forward(
            x,
            up_weight,
            up_bias,
            down_weight,
            down_bias,
            reconstruction_weight,
            None,
            None,
            config=config,
        )
        return output, raw_statistics

    def forward_rule(
        x, up_weight, up_bias, down_weight, down_bias, reconstruction_weight
    ):
        output, raw_statistics, values, winners, reconstruction_error = _forward(
            x,
            up_weight,
            up_bias,
            down_weight,
            down_bias,
            reconstruction_weight,
            None,
            None,
            config=config,
        )
        residuals = (
            x,
            up_weight,
            up_bias,
            down_weight,
            down_bias,
            reconstruction_weight,
            values,
            winners,
            reconstruction_error,
        )
        return (output, raw_statistics), residuals

    def backward_rule(residuals, cotangents):
        output_cotangent, raw_statistics_cotangent = cotangents
        return _backward(
            residuals,
            output_cotangent,
            raw_statistics_cotangent,
            config=config,
        )

    operation.defvjp(forward_rule, backward_rule)
    return operation


@functools.cache
def _make_reconstruction_auxk_operation(config: FuzzyTopKReconstructionConfig):
    if not config.auxk_enabled:
        raise ValueError("AuxK operation requires aux_k")

    @jax.custom_vjp
    def operation(
        x,
        up_weight,
        up_bias,
        down_weight,
        down_bias,
        reconstruction_weight,
        dead_mask,
        cohort,
    ):
        output, raw_statistics, counts, *_ = _forward(
            x,
            up_weight,
            up_bias,
            down_weight,
            down_bias,
            reconstruction_weight,
            dead_mask,
            cohort,
            config=config,
        )
        return output, raw_statistics, counts

    def forward_rule(
        x,
        up_weight,
        up_bias,
        down_weight,
        down_bias,
        reconstruction_weight,
        dead_mask,
        cohort,
    ):
        (
            output,
            raw_statistics,
            counts,
            values,
            winners,
            reconstruction_error,
            aux_values,
            aux_winners,
            group_ids,
            aux_error,
        ) = _forward(
            x,
            up_weight,
            up_bias,
            down_weight,
            down_bias,
            reconstruction_weight,
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
            reconstruction_weight,
            values,
            winners,
            reconstruction_error,
            aux_values,
            aux_winners,
            group_ids,
            aux_error,
        )
        return (output, raw_statistics, counts), residuals

    def backward_rule(residuals, cotangents):
        output_cotangent, raw_statistics_cotangent, _counts_cotangent = cotangents
        gradients = _backward(
            residuals,
            output_cotangent,
            raw_statistics_cotangent,
            config=config,
        )
        return (*gradients, None, None)

    operation.defvjp(forward_rule, backward_rule)
    return operation


@functools.partial(jax.jit, static_argnames=("config",))
def _choicewise_fuzzy_topk_reconstruction_mlp(
    x: jax.Array,
    up_weight: jax.Array,
    up_bias: jax.Array,
    down_weight: jax.Array,
    down_bias: jax.Array,
    reconstruction_weight: jax.Array,
    *,
    config: FuzzyTopKReconstructionConfig,
) -> tuple[jax.Array, jax.Array]:
    """Named boundary for profiling and physical matrix-FLOP accounting."""

    return _make_reconstruction_operation(config)(
        x,
        up_weight,
        up_bias,
        down_weight,
        down_bias,
        reconstruction_weight,
    )


@functools.partial(jax.jit, static_argnames=("config",))
def _choicewise_fuzzy_topk_reconstruction_auxk_mlp(
    x: jax.Array,
    up_weight: jax.Array,
    up_bias: jax.Array,
    down_weight: jax.Array,
    down_bias: jax.Array,
    reconstruction_weight: jax.Array,
    dead_mask: jax.Array,
    cohort: jax.Array,
    *,
    config: FuzzyTopKReconstructionConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Named AuxK boundary for profiling and physical FLOP accounting."""

    return _make_reconstruction_auxk_operation(config)(
        x,
        up_weight,
        up_bias,
        down_weight,
        down_bias,
        reconstruction_weight,
        dead_mask,
        cohort,
    )


def _normalize_statistics(
    raw_statistics: jax.Array,
    tokens: jax.Array,
    *,
    config: FuzzyTopKReconstructionConfig,
) -> jax.Array:
    epsilon = jnp.asarray(1.0e-12, jnp.float32)
    reconstruction_nmse = raw_statistics[0] / jnp.maximum(
        raw_statistics[1], epsilon
    )
    if not config.auxk_enabled:
        return jnp.stack((reconstruction_nmse,))
    if config.aux_k is None:
        raise AssertionError("AuxK normalization lacks aux_k")
    auxk_nmse = raw_statistics[2] / jnp.maximum(raw_statistics[3], epsilon)
    auxk_positive_fraction = raw_statistics[4] / (
        tokens * jnp.asarray(config.aux_k, jnp.float32)
    )
    return jnp.stack(
        (reconstruction_nmse, auxk_nmse, auxk_positive_fraction)
    )


def fuzzy_topk_mlp_with_reconstruction(
    x: jax.Array,
    up_weight: jax.Array,
    up_bias: jax.Array,
    down_weight: jax.Array,
    down_bias: jax.Array,
    reconstruction_weight: jax.Array,
    *,
    config: FuzzyTopKReconstructionConfig,
) -> tuple[jax.Array, jax.Array]:
    """Apply the parent MLP and return its local-batch reconstruction NMSE."""

    _validate_inputs(
        x,
        up_weight,
        up_bias,
        down_weight,
        down_bias,
        reconstruction_weight,
        config,
    )
    output, raw_statistics = _choicewise_fuzzy_topk_reconstruction_mlp(
        x,
        up_weight,
        up_bias,
        down_weight,
        down_bias,
        reconstruction_weight,
        config=config,
    )
    tokens = jnp.asarray(math.prod(x.shape[:-1]), jnp.float32)
    return output, _normalize_statistics(raw_statistics, tokens, config=config)


def fuzzy_topk_mlp_with_reconstruction_auxk(
    x: jax.Array,
    up_weight: jax.Array,
    up_bias: jax.Array,
    down_weight: jax.Array,
    down_bias: jax.Array,
    reconstruction_weight: jax.Array,
    dead_mask: jax.Array,
    cohort: jax.Array,
    *,
    config: FuzzyTopKReconstructionConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Apply parent + reconstruction + literal residual AuxK objectives."""

    _validate_inputs(
        x,
        up_weight,
        up_bias,
        down_weight,
        down_bias,
        reconstruction_weight,
        config,
        dead_mask=dead_mask,
        cohort=cohort,
    )
    output, raw_statistics, counts = (
        _choicewise_fuzzy_topk_reconstruction_auxk_mlp(
            x,
            up_weight,
            up_bias,
            down_weight,
            down_bias,
            reconstruction_weight,
            dead_mask,
            cohort,
            config=config,
        )
    )
    tokens = jnp.asarray(math.prod(x.shape[:-1]), jnp.float32)
    return (
        output,
        _normalize_statistics(raw_statistics, tokens, config=config),
        counts,
    )


def make_mesh_fuzzy_topk_mlp_with_reconstruction(
    *, config: FuzzyTopKReconstructionConfig, mesh: Mesh
) -> FuzzyTopKReconstructionCallable:
    """Build a data-sharded reconstruction head with global-batch NMSE."""

    if config.auxk_enabled:
        raise ValueError("reconstruction-only mesh callable received aux_k")

    def local_operation(
        x, up_weight, up_bias, down_weight, down_bias, reconstruction_weight
    ):
        _validate_inputs(
            x,
            up_weight,
            up_bias,
            down_weight,
            down_bias,
            reconstruction_weight,
            config,
        )
        output, raw_statistics = _choicewise_fuzzy_topk_reconstruction_mlp(
            x,
            up_weight,
            up_bias,
            down_weight,
            down_bias,
            reconstruction_weight,
            config=config,
        )
        global_statistics = lax.psum(raw_statistics, "data")
        global_tokens = lax.psum(
            jnp.asarray(math.prod(x.shape[:-1]), jnp.float32), "data"
        )
        return output, _normalize_statistics(
            global_statistics, global_tokens, config=config
        )

    batch_partition = P("data", None, None)
    return jax.shard_map(
        local_operation,
        mesh=mesh,
        in_specs=(batch_partition, P(), P(), P(), P(), P()),
        out_specs=(batch_partition, P()),
        check_vma=False,
    )


def make_mesh_fuzzy_topk_mlp_with_reconstruction_auxk(
    *, config: FuzzyTopKReconstructionConfig, mesh: Mesh
) -> FuzzyTopKReconstructionAuxKCallable:
    """Build global-batch reconstruction/AuxK objectives and activity counts."""

    if not config.auxk_enabled:
        raise ValueError("AuxK mesh callable requires aux_k")

    def local_operation(
        x,
        up_weight,
        up_bias,
        down_weight,
        down_bias,
        reconstruction_weight,
        dead_mask,
        cohort,
    ):
        _validate_inputs(
            x,
            up_weight,
            up_bias,
            down_weight,
            down_bias,
            reconstruction_weight,
            config,
            dead_mask=dead_mask,
            cohort=cohort,
        )
        output, raw_statistics, local_counts = (
            _choicewise_fuzzy_topk_reconstruction_auxk_mlp(
                x,
                up_weight,
                up_bias,
                down_weight,
                down_bias,
                reconstruction_weight,
                dead_mask,
                cohort,
                config=config,
            )
        )
        global_statistics = lax.psum(raw_statistics, "data")
        global_counts = lax.psum(local_counts, "data")
        global_tokens = lax.psum(
            jnp.asarray(math.prod(x.shape[:-1]), jnp.float32), "data"
        )
        return (
            output,
            _normalize_statistics(global_statistics, global_tokens, config=config),
            global_counts,
        )

    batch_partition = P("data", None, None)
    return jax.shard_map(
        local_operation,
        mesh=mesh,
        in_specs=(batch_partition, P(), P(), P(), P(), P(), P(), P()),
        out_specs=(batch_partition, P(), P()),
        check_vma=False,
    )


def _naive_reconstruction_hidden(
    x: jax.Array, up_weight: jax.Array, up_bias: jax.Array
) -> jax.Array:
    """Recompute scores only in the oracle to express stop-gradient input."""

    return _preactivations(lax.stop_gradient(x), up_weight, up_bias)


def naive_fuzzy_topk_mlp_with_reconstruction(
    x: jax.Array,
    up_weight: jax.Array,
    up_bias: jax.Array,
    down_weight: jax.Array,
    down_bias: jax.Array,
    reconstruction_weight: jax.Array,
    *,
    config: FuzzyTopKReconstructionConfig,
) -> tuple[jax.Array, jax.Array]:
    """Literal autodiff oracle for the reconstruction-only objective."""

    _validate_inputs(
        x,
        up_weight,
        up_bias,
        down_weight,
        down_bias,
        reconstruction_weight,
        config,
    )
    hidden = _preactivations(x, up_weight, up_bias)
    values, winners, _ = _selection_from_hidden(hidden, top_k=config.top_k)
    output = _choicewise_decode(
        values,
        winners,
        down_weight,
        model_width=x.shape[-1],
        bias=down_bias,
    )
    reconstruction_hidden = _naive_reconstruction_hidden(x, up_weight, up_bias)
    reconstruction_values, reconstruction_winners, _ = _selection_from_hidden(
        reconstruction_hidden, top_k=config.top_k
    )
    reconstruction = _choicewise_decode(
        reconstruction_values,
        reconstruction_winners,
        reconstruction_weight,
        model_width=x.shape[-1],
    )
    error = reconstruction.astype(jnp.float32) - lax.stop_gradient(
        x.astype(jnp.float32)
    )
    numerator = jnp.sum(jnp.square(error), dtype=jnp.float32)
    denominator = lax.stop_gradient(_centered_square_sum(x))
    statistics = _normalize_statistics(
        jnp.stack((numerator, denominator)),
        jnp.asarray(math.prod(x.shape[:-1]), jnp.float32),
        config=config,
    )
    return output, statistics


def naive_fuzzy_topk_mlp_with_reconstruction_auxk(
    x: jax.Array,
    up_weight: jax.Array,
    up_bias: jax.Array,
    down_weight: jax.Array,
    down_bias: jax.Array,
    reconstruction_weight: jax.Array,
    dead_mask: jax.Array,
    cohort: jax.Array,
    *,
    config: FuzzyTopKReconstructionConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Literal autodiff oracle for reconstruction plus fuzzy AuxK."""

    _validate_inputs(
        x,
        up_weight,
        up_bias,
        down_weight,
        down_bias,
        reconstruction_weight,
        config,
        dead_mask=dead_mask,
        cohort=cohort,
    )
    hidden = _preactivations(x, up_weight, up_bias)
    values, winners, maxima = _selection_from_hidden(hidden, top_k=config.top_k)
    output = _choicewise_decode(
        values,
        winners,
        down_weight,
        model_width=x.shape[-1],
        bias=down_bias,
    )
    counts = lax.stop_gradient(
        _active_sums(
            winners,
            maxima,
            choices=hidden.shape[-1] // config.top_k,
        )
    )
    reconstruction_hidden = _naive_reconstruction_hidden(x, up_weight, up_bias)
    reconstruction_values, reconstruction_winners, _ = _selection_from_hidden(
        reconstruction_hidden, top_k=config.top_k
    )
    reconstruction = _choicewise_decode(
        reconstruction_values,
        reconstruction_winners,
        reconstruction_weight,
        model_width=x.shape[-1],
    )
    reconstruction_error = reconstruction.astype(jnp.float32) - lax.stop_gradient(
        x.astype(jnp.float32)
    )
    reconstruction_numerator = jnp.sum(
        jnp.square(reconstruction_error), dtype=jnp.float32
    )
    reconstruction_denominator = lax.stop_gradient(_centered_square_sum(x))

    if config.aux_k is None:
        raise AssertionError("AuxK oracle lacks aux_k")
    aux_values, aux_winners, group_ids = _auxiliary_selection(
        reconstruction_hidden, dead_mask, cohort, config=config
    )
    choices = reconstruction_hidden.shape[-1] // config.top_k
    grouped_reconstruction = reconstruction_weight.reshape(
        (config.top_k, choices, x.shape[-1])
    )
    selected_reconstruction = jnp.take(
        grouped_reconstruction, group_ids, axis=0
    )
    aux_reconstruction = jnp.zeros(
        (*aux_values.shape[:-1], x.shape[-1]), jnp.float32
    )

    def visit_choice(choice, accumulator):
        active = jnp.where(aux_winners == choice, aux_values, 0.0)
        return accumulator + jnp.einsum(
            "...k,kd->...d",
            active,
            selected_reconstruction[:, choice, :].astype(x.dtype),
            preferred_element_type=jnp.float32,
        )

    aux_reconstruction = lax.fori_loop(
        0, choices, visit_choice, aux_reconstruction
    )
    residual_target = lax.stop_gradient(
        x.astype(jnp.float32) - reconstruction.astype(jnp.float32)
    )
    aux_error = aux_reconstruction - residual_target
    aux_numerator = jnp.sum(jnp.square(aux_error), dtype=jnp.float32)
    aux_denominator = lax.stop_gradient(_centered_square_sum(residual_target))
    aux_positive = lax.stop_gradient(
        jnp.sum((aux_values > 0.0).astype(jnp.float32))
    )
    raw_statistics = jnp.stack(
        (
            reconstruction_numerator,
            reconstruction_denominator,
            aux_numerator,
            aux_denominator,
            aux_positive,
        )
    )
    statistics = _normalize_statistics(
        raw_statistics,
        jnp.asarray(math.prod(x.shape[:-1]), jnp.float32),
        config=config,
    )
    return output, statistics, counts


__all__ = (
    "FuzzyTopKReconstructionAuxKCallable",
    "FuzzyTopKReconstructionCallable",
    "FuzzyTopKReconstructionConfig",
    "RECONSTRUCTION_AUXK_STAT_NAMES",
    "RECONSTRUCTION_STAT_NAMES",
    "fuzzy_topk_mlp_with_reconstruction",
    "fuzzy_topk_mlp_with_reconstruction_auxk",
    "make_mesh_fuzzy_topk_mlp_with_reconstruction",
    "make_mesh_fuzzy_topk_mlp_with_reconstruction_auxk",
    "naive_fuzzy_topk_mlp_with_reconstruction",
    "naive_fuzzy_topk_mlp_with_reconstruction_auxk",
)
