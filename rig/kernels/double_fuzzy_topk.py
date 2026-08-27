"""Two-stage grouped approximate Top-K with a selected-row UP projection.

The input width is partitioned into fixed groups and one signed maximum is
retained from every group.  Those selected coordinates feed the dictionary
projection.  The hidden dictionary is then partitioned independently and one
positive winner is retained per hidden group before feature-specific decoding.

``reference`` is the literal selected-row oracle. ``choicewise`` performs the
same mathematics with regular masked contractions. ``pallas_up`` uses the
selected-row Pallas kernel only for the sparse UP forward pass and retains the
bounded choicewise reverse path. ``pallas_up_dx`` also uses selected rows for
the input cotangent; it has fewer issued multiplies but a long ``K`` assignment
grid, so it remains an explicit timing candidate rather than the default.
"""

from __future__ import annotations

from dataclasses import dataclass
import functools
import math
from typing import Callable, Literal

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P

from .fuzzy_topk import FuzzyTopKConfig, fuzzy_topk_mlp, fuzzy_topk_relu
from .sparse_mlp import pallas_sparse_decode, reference_sparse_decode


Backend = Literal["pallas_up", "pallas_up_dx", "choicewise", "reference"]
TPU_VECTOR_LANES = 128
TPU_BF16_SUBLANES = 8
INPUT_GROUP_SIZE = 4
DoubleFuzzyTopKCallable = Callable[
    [jax.Array, jax.Array, jax.Array, jax.Array, jax.Array], jax.Array
]


@dataclass(frozen=True, slots=True)
class DoubleFuzzyTopKConfig:
    """Static two-stage selection and implementation contract."""

    top_k: int
    input_group_size: int = 4
    backend: Backend = "pallas_up"
    token_block: int = 32
    output_block: int = 128
    interpret: bool = False
    debug: bool = False

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.input_group_size <= 0:
            raise ValueError("input_group_size must be positive")
        if self.backend not in (
            "pallas_up",
            "pallas_up_dx",
            "choicewise",
            "reference",
        ):
            raise ValueError(f"unknown double fuzzy TopK backend: {self.backend!r}")
        if self.token_block <= 0:
            raise ValueError("token_block must be positive")
        if self.output_block <= 0 or self.output_block % 128:
            raise ValueError("output_block must be a positive multiple of 128")
        if self.interpret and self.backend not in ("pallas_up", "pallas_up_dx"):
            raise ValueError("interpret mode applies only to a Pallas backend")


def _validate_inputs(
    x: jax.Array,
    up_weight: jax.Array,
    up_bias: jax.Array,
    down_weight: jax.Array,
    down_bias: jax.Array,
    config: DoubleFuzzyTopKConfig,
) -> tuple[int, int, int, int, int]:
    if x.ndim < 2:
        raise ValueError(f"x must have at least two dimensions, got {x.shape}")
    model_width = x.shape[-1]
    if model_width % config.input_group_size:
        raise ValueError(
            f"model width {model_width} must be divisible by input_group_size "
            f"{config.input_group_size}"
        )
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
    if (
        config.backend in ("pallas_up", "pallas_up_dx")
        and hidden_width % config.output_block
    ):
        raise ValueError(
            f"pallas_up output_block {config.output_block} must divide hidden "
            f"width {hidden_width}"
        )
    if config.backend in ("pallas_up", "pallas_up_dx") and (
        config.input_group_size != INPUT_GROUP_SIZE
        or model_width % TPU_BF16_SUBLANES
    ):
        raise ValueError(
            "Pallas grouped sparse UP requires input_group_size 4 and model "
            "width divisible by 8"
        )
    input_top_k = model_width // config.input_group_size
    hidden_choices = hidden_width // config.top_k
    return (
        model_width,
        hidden_width,
        input_top_k,
        config.input_group_size,
        hidden_choices,
    )


def grouped_signed_max(
    inputs: jax.Array, *, group_size: int
) -> tuple[jax.Array, jax.Array]:
    """Keep one signed maximum from every fixed contiguous input group."""

    if group_size <= 0:
        raise ValueError("group_size must be positive")
    width = inputs.shape[-1]
    if width % group_size:
        raise ValueError(f"input width {width} must be divisible by {group_size}")
    groups = width // group_size
    grouped = inputs.reshape((*inputs.shape[:-1], groups, group_size))
    values = jnp.max(grouped, axis=-1)
    winners = jnp.argmax(grouped, axis=-1)
    offsets = jnp.arange(groups, dtype=winners.dtype) * group_size
    return values, winners + offsets


def _grouped_sparse_up_kernel(
    winners_ref,
    values_ref,
    up_tile_ref,
    bias_ref,
    output_ref,
    accumulator_ref,
) -> None:
    """Accumulate two four-coordinate groups from one physical TPU row tile."""

    token_count, input_top_k = values_ref.shape
    group_pair = pl.program_id(1)

    # PrefetchScalarGridSpec places the bounded winner/value blocks in SMEM.
    # TPU Pallas permits only scalar loads from that memory space: a convenient
    # NumPy column slice such as ``ref[:, group]`` lowers to an illegal vector
    # SMEM load.  Spell out the static token block so each read remains scalar,
    # then assemble the values in registers/VMEM for the vector computation.
    def load_smem_column(ref, column, dtype):
        return jnp.stack(
            tuple(ref[token, column] for token in range(token_count))
        ).astype(dtype)

    @pl.when(group_pair == 0)
    def initialize_accumulator() -> None:
        accumulator_ref[...] = jnp.zeros(accumulator_ref.shape, jnp.float32)

    first_group = 2 * group_pair
    second_group = first_group + 1
    selected_tile = up_tile_ref[...].astype(jnp.float32)
    row_ids = lax.broadcasted_iota(
        jnp.int32,
        (token_count, TPU_BF16_SUBLANES, selected_tile.shape[1]),
        dimension=1,
    )
    first_winner = load_smem_column(winners_ref, first_group, jnp.int32)
    second_winner = (
        load_smem_column(winners_ref, second_group, jnp.int32)
        + INPUT_GROUP_SIZE
    )
    first_winner = first_winner[:, None, None]
    second_winner = second_winner[:, None, None]
    broadcast_tile = selected_tile[None, :, :]
    first_rows = jnp.sum(
        jnp.where(row_ids == first_winner, broadcast_tile, 0.0),
        axis=1,
    )
    second_rows = jnp.sum(
        jnp.where(row_ids == second_winner, broadcast_tile, 0.0),
        axis=1,
    )
    first_values = load_smem_column(values_ref, first_group, jnp.float32)
    second_values = load_smem_column(values_ref, second_group, jnp.float32)
    contribution = first_values[:, None] * first_rows
    contribution += second_values[:, None] * second_rows
    accumulator_ref[...] = accumulator_ref[...] + contribution

    @pl.when(group_pair == input_top_k // 2 - 1)
    def store_output() -> None:
        output_ref[...] = (accumulator_ref[...] + bias_ref[0, :]).astype(
            output_ref.dtype
        )


def _pallas_grouped_sparse_up_block(
    values: jax.Array,
    winners: jax.Array,
    up_weight: jax.Array,
    up_bias: jax.Array,
    *,
    output_block: int,
    interpret: bool,
    debug: bool,
) -> jax.Array:
    """Project one fixed token block while reusing each eight-row weight tile."""

    token_count, input_top_k = values.shape
    _, hidden_width = up_weight.shape
    group_pairs = input_top_k // 2
    grid = (hidden_width // output_block, group_pairs)

    def weight_index(output_index, group_pair, _winners_ref, _values_ref):
        return group_pair, output_index

    def bias_index(output_index, _group_pair, _winners_ref, _values_ref):
        return 0, output_index

    def output_index(output_index, _group_pair, _winners_ref, _values_ref):
        return 0, output_index

    with jax.named_scope("double_fuzzy_grouped_sparse_up"):
        return pl.pallas_call(
            _grouped_sparse_up_kernel,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=2,
                grid=grid,
                in_specs=(
                    pl.BlockSpec(
                        (TPU_BF16_SUBLANES, output_block), weight_index
                    ),
                    pl.BlockSpec((1, output_block), bias_index),
                ),
                out_specs=pl.BlockSpec(
                    (token_count, output_block), output_index
                ),
                scratch_shapes=(
                    pltpu.VMEM((token_count, output_block), jnp.float32),
                ),
            ),
            out_shape=jax.ShapeDtypeStruct(
                (token_count, hidden_width), values.dtype
            ),
            interpret=interpret,
            debug=debug,
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel", "arbitrary")
            ),
            name="double_fuzzy_grouped_sparse_up",
        )(
            # v4 Mosaic does not support masked 8-bit SMEM addressing. Keep
            # the bounded prefetched indices in int32 so this kernel has the
            # same transport contract on v4 and newer TPU generations.
            winners.astype(jnp.int32),
            values,
            up_weight,
            up_bias[None, :],
        )


def pallas_grouped_sparse_up(
    values: jax.Array,
    winners: jax.Array,
    up_weight: jax.Array,
    up_bias: jax.Array,
    *,
    token_block: int = 32,
    output_block: int = TPU_VECTOR_LANES,
    interpret: bool = False,
    debug: bool = False,
) -> jax.Array:
    """Project one winner per four inputs, sharing tiles across tokens/groups."""

    if winners.shape != values.shape:
        raise ValueError(
            f"winners must match values shape {values.shape}, got {winners.shape}"
        )
    if values.ndim < 2:
        raise ValueError(f"values must have at least two dimensions, got {values.shape}")
    if token_block <= 0:
        raise ValueError("token_block must be positive")
    if output_block <= 0 or output_block % TPU_VECTOR_LANES:
        raise ValueError("output_block must be a positive multiple of 128")
    prefix = values.shape[:-1]
    input_top_k = values.shape[-1]
    if input_top_k <= 0 or input_top_k % 2:
        raise ValueError(
            "pallas grouped sparse UP requires a positive even input_top_k"
        )
    model_width = input_top_k * INPUT_GROUP_SIZE
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
    if hidden_width % output_block:
        raise ValueError(
            f"output_block {output_block} must divide hidden width {hidden_width}"
        )

    tokens = math.prod(prefix)
    flat_values = values.reshape((tokens, input_top_k))
    flat_winners = winners.reshape((tokens, input_top_k))
    padded_tokens = math.ceil(tokens / token_block) * token_block
    padding = padded_tokens - tokens
    if padding:
        flat_values = jnp.pad(flat_values, ((0, padding), (0, 0)))
        flat_winners = jnp.pad(flat_winners, ((0, padding), (0, 0)))
    value_blocks = flat_values.reshape((-1, token_block, input_top_k))
    winner_blocks = flat_winners.reshape((-1, token_block, input_top_k))

    def project_one(blocks):
        value_block, winner_block = blocks
        return _pallas_grouped_sparse_up_block(
            value_block,
            winner_block,
            up_weight,
            up_bias,
            output_block=output_block,
            interpret=interpret,
            debug=debug,
        )

    projected = jax.vmap(project_one)((value_blocks, winner_blocks))
    return projected.reshape((padded_tokens, hidden_width))[:tokens].reshape(
        (*prefix, hidden_width)
    )


def naive_double_fuzzy_topk_mlp(
    x: jax.Array,
    up_weight: jax.Array,
    up_bias: jax.Array,
    down_weight: jax.Array,
    down_bias: jax.Array,
    *,
    top_k: int,
    input_group_size: int = 4,
) -> jax.Array:
    """Literal sparse-input -> dense dictionary -> sparse-hidden oracle."""

    config = DoubleFuzzyTopKConfig(
        top_k=top_k,
        input_group_size=input_group_size,
        backend="reference",
    )
    _validate_inputs(x, up_weight, up_bias, down_weight, down_bias, config)
    input_values, input_indices = grouped_signed_max(
        x, group_size=input_group_size
    )
    sparse_input = jnp.zeros_like(x)
    sparse_input = jnp.put_along_axis(
        sparse_input, input_indices, input_values, axis=-1, inplace=False
    )
    hidden = jnp.einsum(
        "...d,dh->...h",
        sparse_input,
        up_weight.astype(x.dtype),
        preferred_element_type=jnp.float32,
    )
    hidden = hidden + up_bias.astype(x.dtype)
    hidden_values, hidden_indices = fuzzy_topk_relu(hidden, top_k=top_k)
    sparse_hidden = jnp.zeros_like(hidden)
    sparse_hidden = jnp.put_along_axis(
        sparse_hidden, hidden_indices, hidden_values, axis=-1, inplace=False
    )
    output = jnp.einsum(
        "...h,hd->...d",
        sparse_hidden,
        down_weight.astype(x.dtype),
        preferred_element_type=jnp.float32,
    )
    return (output + down_bias.astype(x.dtype)).astype(x.dtype)


def _choicewise_sparse_up(
    input_values: jax.Array,
    input_winners: jax.Array,
    up_weight: jax.Array,
    up_bias: jax.Array,
    *,
    input_group_size: int,
) -> jax.Array:
    groups = input_values.shape[-1]
    hidden_width = up_weight.shape[1]
    grouped_up = up_weight.reshape((groups, input_group_size, hidden_width))
    accumulator = jnp.zeros((*input_values.shape[:-1], hidden_width), jnp.float32)

    def visit_choice(choice, output):
        active = jnp.where(input_winners == choice, input_values, 0.0)
        return output + jnp.einsum(
            "...q,qh->...h",
            active,
            grouped_up[:, choice, :].astype(input_values.dtype),
            preferred_element_type=jnp.float32,
        )

    hidden = lax.fori_loop(0, input_group_size, visit_choice, accumulator)
    return (hidden + up_bias.astype(jnp.float32)).astype(input_values.dtype)


def _choicewise_decode(
    hidden_values: jax.Array,
    hidden_winners: jax.Array,
    down_weight: jax.Array,
    down_bias: jax.Array,
    *,
    hidden_choices: int,
) -> jax.Array:
    top_k = hidden_values.shape[-1]
    model_width = down_weight.shape[1]
    grouped_down = down_weight.reshape((top_k, hidden_choices, model_width))
    accumulator = jnp.zeros(
        (*hidden_values.shape[:-1], model_width), jnp.float32
    )

    def visit_choice(choice, output):
        active = jnp.where(hidden_winners == choice, hidden_values, 0.0)
        return output + jnp.einsum(
            "...k,kd->...d",
            active,
            grouped_down[:, choice, :].astype(hidden_values.dtype),
            preferred_element_type=jnp.float32,
        )

    output = lax.fori_loop(0, hidden_choices, visit_choice, accumulator)
    return (output + down_bias.astype(jnp.float32)).astype(hidden_values.dtype)


def _forward(
    x: jax.Array,
    up_weight: jax.Array,
    up_bias: jax.Array,
    down_weight: jax.Array,
    down_bias: jax.Array,
    *,
    config: DoubleFuzzyTopKConfig,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    (
        _model_width,
        _hidden_width,
        _input_top_k,
        input_group_size,
        hidden_choices,
    ) = _validate_inputs(x, up_weight, up_bias, down_weight, down_bias, config)
    input_values, input_indices = grouped_signed_max(
        x, group_size=input_group_size
    )
    input_winners = input_indices % input_group_size
    if config.backend in ("pallas_up", "pallas_up_dx"):
        hidden = pallas_grouped_sparse_up(
            input_values.astype(x.dtype),
            input_winners,
            up_weight.astype(x.dtype),
            up_bias.astype(x.dtype),
            token_block=config.token_block,
            output_block=config.output_block,
            interpret=config.interpret,
            debug=config.debug,
        )
    elif config.backend == "choicewise":
        hidden = _choicewise_sparse_up(
            input_values.astype(x.dtype),
            input_winners,
            up_weight,
            up_bias,
            input_group_size=input_group_size,
        )
    else:
        hidden = reference_sparse_decode(
            input_values.astype(x.dtype),
            input_indices,
            up_weight.astype(x.dtype),
            up_bias.astype(x.dtype),
        )
    hidden_values, hidden_indices = fuzzy_topk_relu(hidden, top_k=config.top_k)
    hidden_winners = hidden_indices % hidden_choices
    if config.backend == "reference":
        output = reference_sparse_decode(
            hidden_values.astype(x.dtype),
            hidden_indices,
            down_weight.astype(x.dtype),
            down_bias.astype(x.dtype),
        )
    else:
        output = _choicewise_decode(
            hidden_values.astype(x.dtype),
            hidden_winners,
            down_weight,
            down_bias,
            hidden_choices=hidden_choices,
        )
    return (
        output,
        input_values.astype(x.dtype),
        input_winners,
        hidden_values.astype(x.dtype),
        hidden_winners,
    )


def _choicewise_input_cotangent(
    preactivation_cotangent: jax.Array,
    hidden_winners: jax.Array,
    input_winners: jax.Array,
    grouped_up: jax.Array,
) -> jax.Array:
    input_group_size = grouped_up.shape[1]
    hidden_choices = grouped_up.shape[3]
    tokens, input_top_k = input_winners.shape
    selected = jnp.zeros((tokens, input_top_k), jnp.float32)

    def visit_input_choice(input_choice, carry):
        def visit_hidden_choice(hidden_choice, inner):
            active_hidden = jnp.where(
                hidden_winners == hidden_choice,
                preactivation_cotangent,
                0.0,
            )
            contribution = jnp.einsum(
                "tk,qk->tq",
                active_hidden,
                grouped_up[:, input_choice, :, hidden_choice],
                preferred_element_type=jnp.float32,
            )
            return inner + jnp.where(
                input_winners == input_choice, contribution, 0.0
            )

        return lax.fori_loop(0, hidden_choices, visit_hidden_choice, carry)

    return lax.fori_loop(0, input_group_size, visit_input_choice, selected)


@functools.cache
def _make_double_fuzzy_topk(config: DoubleFuzzyTopKConfig):
    """Build the explicit VJP for both nondifferentiable group selectors."""

    @jax.custom_vjp
    def operation(x, up_weight, up_bias, down_weight, down_bias):
        output, _, _, _, _ = _forward(
            x,
            up_weight,
            up_bias,
            down_weight,
            down_bias,
            config=config,
        )
        return output

    def forward_rule(x, up_weight, up_bias, down_weight, down_bias):
        output, input_values, input_winners, hidden_values, hidden_winners = _forward(
            x,
            up_weight,
            up_bias,
            down_weight,
            down_bias,
            config=config,
        )
        return output, (
            x,
            up_weight,
            up_bias,
            down_weight,
            down_bias,
            input_values,
            input_winners,
            hidden_values,
            hidden_winners,
        )

    def backward_rule(residuals, output_cotangent):
        (
            x,
            up_weight,
            up_bias,
            down_weight,
            down_bias,
            input_values,
            input_winners,
            hidden_values,
            hidden_winners,
        ) = residuals
        model_width, hidden_width = up_weight.shape
        input_group_size = config.input_group_size
        input_top_k = model_width // input_group_size
        hidden_choices = hidden_width // config.top_k
        tokens = math.prod(x.shape[:-1])
        flat_output_cotangent = output_cotangent.reshape(
            (tokens, model_width)
        ).astype(jnp.float32)
        flat_input_values = input_values.reshape((tokens, input_top_k)).astype(
            jnp.float32
        )
        flat_input_winners = input_winners.reshape((tokens, input_top_k))
        flat_hidden_values = hidden_values.reshape((tokens, config.top_k)).astype(
            jnp.float32
        )
        flat_hidden_winners = hidden_winners.reshape((tokens, config.top_k))
        grouped_up = up_weight.reshape(
            (input_top_k, input_group_size, config.top_k, hidden_choices)
        ).astype(jnp.float32)
        grouped_down = down_weight.reshape(
            (config.top_k, hidden_choices, model_width)
        ).astype(jnp.float32)

        hidden_initial = (
            jnp.zeros((tokens, config.top_k), jnp.float32),
            jnp.zeros((config.top_k, hidden_choices), jnp.float32),
            jnp.zeros(grouped_down.shape, jnp.float32),
        )

        def visit_hidden_choice(hidden_choice, carry):
            dz_values, up_bias_gradient, down_gradient = carry
            winner_mask = flat_hidden_winners == hidden_choice
            choice_values = jnp.where(winner_mask, flat_hidden_values, 0.0)
            choice_down = grouped_down[:, hidden_choice, :]
            value_cotangent = jnp.einsum(
                "td,kd->tk",
                flat_output_cotangent,
                choice_down,
                preferred_element_type=jnp.float32,
            )
            choice_dz = jnp.where(
                winner_mask & (flat_hidden_values > 0.0),
                value_cotangent,
                0.0,
            )
            dz_values = dz_values + choice_dz
            up_bias_gradient = up_bias_gradient.at[:, hidden_choice].set(
                jnp.sum(choice_dz, axis=0)
            )
            down_gradient = down_gradient.at[:, hidden_choice, :].set(
                jnp.einsum(
                    "tk,td->kd",
                    choice_values,
                    flat_output_cotangent,
                    preferred_element_type=jnp.float32,
                )
            )
            return dz_values, up_bias_gradient, down_gradient

        dz_values, up_bias_gradient, down_gradient = lax.fori_loop(
            0, hidden_choices, visit_hidden_choice, hidden_initial
        )

        up_gradient = jnp.zeros(grouped_up.shape, jnp.float32)

        def visit_input_choice(input_choice, carry):
            active_input = jnp.where(
                flat_input_winners == input_choice, flat_input_values, 0.0
            )

            def visit_hidden_choice(hidden_choice, inner):
                active_hidden = jnp.where(
                    flat_hidden_winners == hidden_choice, dz_values, 0.0
                )
                block = jnp.einsum(
                    "tq,tk->qk",
                    active_input,
                    active_hidden,
                    preferred_element_type=jnp.float32,
                )
                return inner.at[:, input_choice, :, hidden_choice].set(block)

            return lax.fori_loop(
                0, hidden_choices, visit_hidden_choice, carry
            )

        up_gradient = lax.fori_loop(
            0, input_group_size, visit_input_choice, up_gradient
        )

        input_offsets = (
            jnp.arange(input_top_k, dtype=flat_input_winners.dtype)
            * input_group_size
        )
        input_indices = flat_input_winners + input_offsets
        if config.backend in ("pallas_up_dx", "reference"):
            hidden_offsets = (
                jnp.arange(config.top_k, dtype=flat_hidden_winners.dtype)
                * hidden_choices
            )
            hidden_indices = flat_hidden_winners + hidden_offsets
            zero_bias = jnp.zeros((model_width,), x.dtype)
            if config.backend == "pallas_up_dx":
                padded_model_width = (
                    math.ceil(model_width / config.output_block)
                    * config.output_block
                )
                padded_up_transpose = jnp.pad(
                    up_weight.T.astype(x.dtype),
                    ((0, 0), (0, padded_model_width - model_width)),
                )
                padded_zero_bias = jnp.zeros((padded_model_width,), x.dtype)
                dense_input_cotangent = pallas_sparse_decode(
                    dz_values.astype(x.dtype),
                    hidden_indices,
                    padded_up_transpose,
                    padded_zero_bias,
                    token_block=config.token_block,
                    output_block=config.output_block,
                    interpret=config.interpret,
                    debug=config.debug,
                )[..., :model_width]
            else:
                dense_input_cotangent = reference_sparse_decode(
                    dz_values.astype(x.dtype),
                    hidden_indices,
                    up_weight.T.astype(x.dtype),
                    zero_bias,
                )
            selected_input_cotangent = jnp.take_along_axis(
                dense_input_cotangent.astype(jnp.float32),
                input_indices,
                axis=-1,
            )
        else:
            selected_input_cotangent = _choicewise_input_cotangent(
                dz_values,
                flat_hidden_winners,
                flat_input_winners,
                grouped_up,
            )

        flat_dx = jnp.zeros((tokens, model_width), jnp.float32)
        flat_dx = jnp.put_along_axis(
            flat_dx,
            input_indices,
            selected_input_cotangent,
            axis=-1,
            inplace=False,
        )
        down_bias_gradient = jnp.sum(flat_output_cotangent, axis=0)
        return (
            flat_dx.reshape(x.shape).astype(x.dtype),
            up_gradient.reshape(up_weight.shape).astype(up_weight.dtype),
            up_bias_gradient.reshape((hidden_width,)).astype(up_bias.dtype),
            down_gradient.reshape(down_weight.shape).astype(down_weight.dtype),
            down_bias_gradient.astype(down_bias.dtype),
        )

    operation.defvjp(forward_rule, backward_rule)
    return operation


@functools.partial(jax.jit, static_argnames=("config",))
def _pallas_up_double_fuzzy_topk_mlp(
    x: jax.Array,
    up_weight: jax.Array,
    up_bias: jax.Array,
    down_weight: jax.Array,
    down_bias: jax.Array,
    *,
    config: DoubleFuzzyTopKConfig,
) -> jax.Array:
    """Named hybrid boundary with a selected-row Pallas UP forward."""

    return _make_double_fuzzy_topk(config)(
        x, up_weight, up_bias, down_weight, down_bias
    )


@functools.partial(jax.jit, static_argnames=("config",))
def _choicewise_double_fuzzy_topk_mlp(
    x: jax.Array,
    up_weight: jax.Array,
    up_bias: jax.Array,
    down_weight: jax.Array,
    down_bias: jax.Array,
    *,
    config: DoubleFuzzyTopKConfig,
) -> jax.Array:
    """Reuse the proven fuzzy kernel after zero-filling inner winners.

    This composition has exactly the two-stage selector semantics but lets the
    parent fuzzy custom VJP issue large regular UP contractions.  The earlier
    factorized implementation decomposed the reverse pass across the Cartesian
    product of four input and four hidden choices; those 16 small contractions
    had the same nominal FLOPs but substantially worse TPU utilization.
    """

    input_values, input_indices = grouped_signed_max(
        x, group_size=config.input_group_size
    )
    sparse_input = jnp.zeros_like(x)
    sparse_input = jnp.put_along_axis(
        sparse_input,
        input_indices,
        input_values,
        axis=-1,
        inplace=False,
    )
    return fuzzy_topk_mlp(
        sparse_input,
        up_weight,
        up_bias,
        down_weight,
        down_bias,
        config=FuzzyTopKConfig(top_k=config.top_k, backend="choicewise"),
    )


@functools.partial(jax.jit, static_argnames=("config",))
def _pallas_up_dx_double_fuzzy_topk_mlp(
    x: jax.Array,
    up_weight: jax.Array,
    up_bias: jax.Array,
    down_weight: jax.Array,
    down_bias: jax.Array,
    *,
    config: DoubleFuzzyTopKConfig,
) -> jax.Array:
    """Named Pallas boundary with selected UP forward and input cotangent."""

    return _make_double_fuzzy_topk(config)(
        x, up_weight, up_bias, down_weight, down_bias
    )


@functools.partial(jax.jit, static_argnames=("config",))
def _reference_double_fuzzy_topk_mlp(
    x: jax.Array,
    up_weight: jax.Array,
    up_bias: jax.Array,
    down_weight: jax.Array,
    down_bias: jax.Array,
    *,
    config: DoubleFuzzyTopKConfig,
) -> jax.Array:
    """Named literal selected-row boundary for numerical tests."""

    return _make_double_fuzzy_topk(config)(
        x, up_weight, up_bias, down_weight, down_bias
    )


def double_fuzzy_topk_mlp(
    x: jax.Array,
    up_weight: jax.Array,
    up_bias: jax.Array,
    down_weight: jax.Array,
    down_bias: jax.Array,
    *,
    config: DoubleFuzzyTopKConfig,
) -> jax.Array:
    """Apply signed input grouping and grouped hidden TopK-ReLU."""

    _validate_inputs(x, up_weight, up_bias, down_weight, down_bias, config)
    operation = {
        "pallas_up": _pallas_up_double_fuzzy_topk_mlp,
        "pallas_up_dx": _pallas_up_dx_double_fuzzy_topk_mlp,
        "choicewise": _choicewise_double_fuzzy_topk_mlp,
        "reference": _reference_double_fuzzy_topk_mlp,
    }[config.backend]
    return operation(
        x,
        up_weight,
        up_bias,
        down_weight,
        down_bias,
        config=config,
    )


def make_mesh_double_fuzzy_topk_mlp(
    *, config: DoubleFuzzyTopKConfig, mesh: Mesh
) -> DoubleFuzzyTopKCallable:
    """Build a data-sharded boundary with replicated MLP parameters."""

    def local_operation(x, up_weight, up_bias, down_weight, down_bias):
        return double_fuzzy_topk_mlp(
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


__all__ = (
    "Backend",
    "DoubleFuzzyTopKCallable",
    "DoubleFuzzyTopKConfig",
    "double_fuzzy_topk_mlp",
    "grouped_signed_max",
    "make_mesh_double_fuzzy_topk_mlp",
    "naive_double_fuzzy_topk_mlp",
    "pallas_grouped_sparse_up",
)
