"""Top-k sparse MLP used by the sparse-autoencoder recipe.

The encoder scores every dictionary element to discover the exact top-k set.
The reference and older Pallas backends then decode selected rows directly.
On TPU v4 that token-dependent gather is much slower than an MXU matmul, so the
production ``pallas_masked`` backend emits the exact dense zero mask and uses a
dense hardware decoder. See ``docs/KERNELS.md`` for the semantic and physical
FLOP distinction.
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


Backend = Literal["auto", "pallas", "pallas_masked", "reference"]
SparseMlpCallable = Callable[
    [jax.Array, jax.Array, jax.Array, jax.Array, jax.Array], jax.Array
]
TPU_VECTOR_LANES = 128
TPU_BF16_SUBLANES = 8
PALLAS_MASKED_MAX_BLOCK_ELEMENTS = 786_432


@dataclass(frozen=True, slots=True)
class SparseMlpConfig:
    """Static kernel choices captured outside the compiled training step."""

    top_k: int
    backend: Backend = "auto"
    token_block: int = 128
    output_block: int = TPU_VECTOR_LANES
    interpret: bool = False
    debug: bool = False

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.backend not in ("auto", "pallas", "pallas_masked", "reference"):
            raise ValueError(f"unknown sparse MLP backend: {self.backend!r}")
        if self.token_block <= 0:
            raise ValueError("token_block must be positive")
        if self.output_block <= 0 or self.output_block % TPU_VECTOR_LANES:
            raise ValueError("output_block must be a positive multiple of 128")
        if self.interpret and self.backend == "reference":
            raise ValueError("interpret mode applies only to the Pallas backend")


def _validate_inputs(
    x: jax.Array,
    up_weight: jax.Array,
    up_bias: jax.Array,
    down_weight: jax.Array,
    down_bias: jax.Array,
    config: SparseMlpConfig,
) -> None:
    if x.ndim < 2:
        raise ValueError(f"x must have at least two dimensions, got {x.shape}")
    model_width = x.shape[-1]
    if up_weight.ndim != 2:
        raise ValueError(f"up_weight must be rank 2, got {up_weight.shape}")
    if up_weight.shape[0] != model_width:
        raise ValueError(
            f"up_weight input width {up_weight.shape[0]} does not match x {model_width}"
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
    if config.backend == "pallas" and model_width % config.output_block:
        raise ValueError(
            f"Pallas output_block {config.output_block} must divide model width "
            f"{model_width}"
        )


def topk_relu(preactivations: jax.Array, *, top_k: int) -> tuple[jax.Array, jax.Array]:
    """Return the nonzero values and indices of a per-row TopK-ReLU.

    ReLU is applied first, then exactly ``top_k`` coordinates are retained.
    Rows with fewer than ``top_k`` positive coordinates therefore contain
    selected zeros; those entries have zero output and zero derivative.
    """

    return lax.top_k(jax.nn.relu(preactivations), top_k)


def pallas_masked_token_block(hidden_width: int, preferred: int) -> int:
    """Resolve the largest safe eight-row token block up to ``preferred``.

    The bound keeps the double-buffered activation/result windows and packed
    int32 keys within v4's 16 MiB VMEM. Larger dictionaries therefore shrink
    the block automatically while small tiers retain more parallel work.
    """

    if hidden_width <= 0 or hidden_width > 65_536:
        raise ValueError("pallas_masked hidden width must be in [1, 65,536]")
    if preferred <= 0 or preferred % TPU_BF16_SUBLANES:
        raise ValueError("preferred token block must be a positive multiple of 8")
    maximum = (
        PALLAS_MASKED_MAX_BLOCK_ELEMENTS // hidden_width // TPU_BF16_SUBLANES
    ) * TPU_BF16_SUBLANES
    if maximum < TPU_BF16_SUBLANES:
        raise ValueError("hidden width leaves no viable pallas_masked token block")
    return min(preferred, maximum)


def pallas_masked_topk_relu(
    activated: jax.Array,
    *,
    top_k: int,
    token_block: int,
    interpret: bool,
    debug: bool,
) -> jax.Array:
    """Select exact BF16 TopK support and return a dense zero-masked array.

    TPU v4 has no hardware path for a different element-sparse decoder matrix
    on every token. Gathering decoder rows therefore moves orders of magnitude
    more data than a dense MXU matmul. This kernel instead makes the expensive
    part cheap: it finds the exact support in VMEM, emits the selected BF16
    activations, and lets the following dense matmul reuse decoder weights.

    Positive BF16 bit patterns preserve numerical order. Packing each value
    with the reverse column index also reproduces ``lax.top_k``'s stable
    lower-index tie break. Every packed key is unique, so a fixed-width binary
    search finds the kth key without a sort, a large index result, or a prefix
    scan. The decoder still performs dense hardware FLOPs; callers and reports
    must not mistake this systems optimization for an element-sparse MXU.
    """

    if activated.dtype != jnp.bfloat16:
        raise ValueError("pallas_masked requires bfloat16 activations")
    if activated.ndim < 2:
        raise ValueError("activated must have at least two dimensions")
    hidden_width = activated.shape[-1]
    if top_k <= 0 or top_k > hidden_width:
        raise ValueError("top_k must be positive and no larger than hidden width")
    if hidden_width > 65_536:
        raise ValueError("pallas_masked supports hidden widths up to 65,536")
    if hidden_width % TPU_VECTOR_LANES:
        raise ValueError("pallas_masked hidden width must be a multiple of 128")
    token_block = pallas_masked_token_block(hidden_width, token_block)

    prefix = activated.shape[:-1]
    tokens = math.prod(prefix)
    padded_tokens = math.ceil(tokens / token_block) * token_block
    flat = activated.reshape((tokens, hidden_width))
    if padded_tokens != tokens:
        flat = jnp.pad(flat, ((0, padded_tokens - tokens), (0, 0)))
    index_bits = max(1, (hidden_width - 1).bit_length())
    search_iterations = 15 + index_bits
    maximum_key = (0x7FFF << index_bits) | (hidden_width - 1)

    def kernel(activated_ref, selected_ref):
        values = activated_ref[...]
        bits = lax.bitcast_convert_type(values, jnp.uint16).astype(jnp.int32)
        columns = lax.broadcasted_iota(jnp.int32, bits.shape, 1)
        keys = (bits << index_bits) | (hidden_width - 1 - columns)
        lower = jnp.zeros((token_block, 1), jnp.int32)
        upper = jnp.full((token_block, 1), maximum_key, jnp.int32)

        def bisect(_iteration, bounds):
            low, high = bounds
            distance = high - low
            middle = low + distance // 2 + distance % 2
            count = jnp.sum(
                keys >= middle, axis=1, keepdims=True, dtype=jnp.int32
            )
            keep_upper_half = count >= top_k
            return jnp.where(keep_upper_half, middle, low), jnp.where(
                keep_upper_half, high, middle - 1
            )

        threshold, _ = lax.fori_loop(
            0, search_iterations, bisect, (lower, upper)
        )
        # ReLU-selected zeros have zero value and zero derivative, so omitting
        # their nominal support preserves the complete differentiable result.
        selected_ref[...] = jnp.where((keys >= threshold) & (bits > 0), values, 0)

    with jax.named_scope("exact_bf16_topk_select"):
        selected = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(flat.shape, flat.dtype),
            grid=(padded_tokens // token_block,),
            in_specs=(
                pl.BlockSpec(
                    (token_block, hidden_width), lambda program: (program, 0)
                ),
            ),
            out_specs=pl.BlockSpec(
                (token_block, hidden_width), lambda program: (program, 0)
            ),
            interpret=interpret,
            debug=debug,
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel",)
            ),
            name="exact_bf16_topk_select",
        )(flat)
    return selected[:tokens].reshape((*prefix, hidden_width))


@functools.cache
def _make_pallas_masked_topk_relu(config: SparseMlpConfig):
    @jax.custom_vjp
    def operation(activated: jax.Array) -> jax.Array:
        return pallas_masked_topk_relu(
            activated,
            top_k=config.top_k,
            token_block=config.token_block,
            interpret=config.interpret,
            debug=config.debug,
        )

    def forward(activated):
        selected = pallas_masked_topk_relu(
            activated,
            top_k=config.top_k,
            token_block=config.token_block,
            interpret=config.interpret,
            debug=config.debug,
        )
        return selected, selected > 0

    def backward(mask, cotangent):
        return (jnp.where(mask, cotangent, 0),)

    operation.defvjp(forward, backward)
    return operation


def reference_sparse_decode(
    values: jax.Array,
    indices: jax.Array,
    down_weight: jax.Array,
    down_bias: jax.Array,
) -> jax.Array:
    """Small differentiable oracle which gathers only selected decoder rows."""

    selected_weight = down_weight[indices]
    output = jnp.einsum(
        "...k,...kd->...d",
        values,
        selected_weight,
        preferred_element_type=jnp.float32,
    )
    return (output + down_bias).astype(values.dtype)


def naive_dense_topk_mlp(
    x: jax.Array,
    up_weight: jax.Array,
    up_bias: jax.Array,
    down_weight: jax.Array,
    down_bias: jax.Array,
    *,
    top_k: int,
) -> jax.Array:
    """Literal dense -> TopK-ReLU -> dense oracle for fuzzy kernel tests."""

    hidden = jnp.einsum("...d,dh->...h", x, up_weight.astype(x.dtype))
    hidden = hidden + up_bias.astype(x.dtype)
    values, indices = topk_relu(hidden, top_k=top_k)
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


def _sparse_decode_kernel(
    indices_ref,
    values_ref,
    down_tile_ref,
    bias_ref,
    output_ref,
    accumulator_ref,
) -> None:
    """Accumulate one selected row in an automatically pipelined grid."""

    token_count, top_k = values_ref.shape
    assignment = pl.program_id(1)

    @pl.when(assignment == 0)
    def initialize_accumulator() -> None:
        accumulator_ref[...] = jnp.zeros(accumulator_ref.shape, jnp.float32)

    token = assignment // top_k
    slot = assignment % top_k
    feature = indices_ref[token, slot]
    row_in_tile = feature % TPU_BF16_SUBLANES
    selected_tile = down_tile_ref[...].astype(jnp.float32)
    row_ids = lax.broadcasted_iota(
        feature.dtype,
        selected_tile.shape,
        dimension=0,
    )
    selected_row = jnp.sum(
        jnp.where(row_ids == row_in_tile, selected_tile, 0.0),
        axis=0,
    )
    value = values_ref[token, slot].astype(jnp.float32)
    accumulator_ref[token, :] = accumulator_ref[token, :] + value * selected_row

    @pl.when(assignment == pl.num_programs(1) - 1)
    def store_output() -> None:
        output_ref[...] = (accumulator_ref[...] + bias_ref[0, :]).astype(
            output_ref.dtype
        )


def _pallas_sparse_decode_block(
    values: jax.Array,
    indices: jax.Array,
    down_weight: jax.Array,
    down_bias: jax.Array,
    *,
    output_block: int,
    interpret: bool,
    debug: bool,
) -> jax.Array:
    """Decode a fixed token block with a pipelined sparse gather grid."""

    token_count, top_k = values.shape
    _, output_width = down_weight.shape
    assignments = token_count * top_k
    grid = (output_width // output_block, assignments)

    def weight_index(output_index, assignment, indices_ref, _values_ref):
        token = assignment // top_k
        slot = assignment % top_k
        feature_tile = indices_ref[token, slot] // TPU_BF16_SUBLANES
        return feature_tile, output_index

    def bias_index(output_index, _assignment, _indices_ref, _values_ref):
        return 0, output_index

    def output_index(output_index, _assignment, _indices_ref, _values_ref):
        return 0, output_index

    with jax.named_scope("sparse_topk_decoder"):
        return pl.pallas_call(
            _sparse_decode_kernel,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                # Both arrays are deliberately bounded by token_block. SMEM
                # gives the irregular inner grid cheap scalar addressing.
                num_scalar_prefetch=2,
                grid=grid,
                in_specs=(
                    # Exact unit sparsity maps each selected feature to its
                    # physical eight-row BF16 tile. Assignments are the inner
                    # axis so Pallas pipelines transfers while retaining the
                    # output accumulator in VMEM.
                    pl.BlockSpec((TPU_BF16_SUBLANES, output_block), weight_index),
                    pl.BlockSpec((1, output_block), bias_index),
                ),
                out_specs=pl.BlockSpec((token_count, output_block), output_index),
                scratch_shapes=(pltpu.VMEM((token_count, output_block), jnp.float32),),
            ),
            out_shape=jax.ShapeDtypeStruct((token_count, output_width), values.dtype),
            interpret=interpret,
            debug=debug,
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel", "arbitrary")
            ),
            name="sparse_topk_decode",
        )(
            indices.astype(jnp.int32),
            values.astype(jnp.float32),
            down_weight,
            down_bias[None, :],
        )


def pallas_sparse_decode(
    values: jax.Array,
    indices: jax.Array,
    down_weight: jax.Array,
    down_bias: jax.Array,
    *,
    token_block: int = 128,
    output_block: int = TPU_VECTOR_LANES,
    interpret: bool = False,
    debug: bool = False,
) -> jax.Array:
    """Chunked exact sparse decode without materializing ``[..., k, d]``.

    ``indices`` and values for one token block are scalar-prefetched into TPU
    SMEM. The assignment grid pipelines the aligned eight-row BF16 tile that
    contains each selected decoder row, selects that row in VMEM, and retains
    the FP32 output accumulator across all assignments. Chunking bounds SMEM
    and VMEM use independently of sequence length.
    """

    prefix = values.shape[:-1]
    top_k = values.shape[-1]
    if indices.shape != values.shape:
        raise ValueError(
            f"indices must match values shape {values.shape}, got {indices.shape}"
        )
    if down_weight.ndim != 2 or down_weight.shape[0] <= 0:
        raise ValueError("down_weight must be a nonempty rank-2 array")
    output_width = down_weight.shape[1]
    if down_bias.shape != (output_width,):
        raise ValueError(
            f"down_bias must have shape {(output_width,)}, got {down_bias.shape}"
        )
    if output_width % output_block:
        raise ValueError(
            f"output_block {output_block} must divide output width {output_width}"
        )

    tokens = math.prod(prefix)
    flat_values = values.reshape((tokens, top_k))
    flat_indices = indices.reshape((tokens, top_k))
    padded_tokens = math.ceil(tokens / token_block) * token_block
    padding = padded_tokens - tokens
    if padding:
        flat_values = jnp.pad(flat_values, ((0, padding), (0, 0)))
        flat_indices = jnp.pad(flat_indices, ((0, padding), (0, 0)))
    value_blocks = flat_values.reshape((-1, token_block, top_k))
    index_blocks = flat_indices.reshape((-1, token_block, top_k))

    def decode_one(blocks):
        value_block, index_block = blocks
        return _pallas_sparse_decode_block(
            value_block,
            index_block,
            down_weight,
            down_bias,
            output_block=output_block,
            interpret=interpret,
            debug=debug,
        )

    # Batch the Pallas call itself so token blocks become a parallel kernel-grid
    # dimension. ``lax.map`` would serialize those blocks and leave most of a
    # training TPU idle.
    decoded = jax.vmap(decode_one)((value_blocks, index_blocks))
    return decoded.reshape((padded_tokens, output_width))[:tokens].reshape(
        (*prefix, output_width)
    )


def _select_backend(config: SparseMlpConfig) -> Literal["pallas", "reference"]:
    if config.backend == "auto":
        return "pallas" if jax.default_backend() == "tpu" else "reference"
    return config.backend


def _forward_with_selection(
    x: jax.Array,
    up_weight: jax.Array,
    up_bias: jax.Array,
    down_weight: jax.Array,
    down_bias: jax.Array,
    *,
    config: SparseMlpConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    _validate_inputs(x, up_weight, up_bias, down_weight, down_bias, config)
    hidden = jnp.einsum("...d,dh->...h", x, up_weight.astype(x.dtype))
    hidden = hidden + up_bias.astype(x.dtype)
    values, indices = topk_relu(hidden, top_k=config.top_k)
    if _select_backend(config) == "pallas":
        output = pallas_sparse_decode(
            values.astype(x.dtype),
            indices,
            down_weight.astype(x.dtype),
            down_bias.astype(x.dtype),
            token_block=config.token_block,
            output_block=config.output_block,
            interpret=config.interpret,
            debug=config.debug,
        )
    else:
        output = reference_sparse_decode(
            values.astype(x.dtype),
            indices,
            down_weight.astype(x.dtype),
            down_bias.astype(x.dtype),
        )
    return output, values.astype(x.dtype), indices


@functools.cache
def _make_sparse_topk_mlp(config: SparseMlpConfig):
    """Build one custom-VJP operation for a static sparse-kernel contract."""

    @jax.custom_vjp
    def operation(
        x: jax.Array,
        up_weight: jax.Array,
        up_bias: jax.Array,
        down_weight: jax.Array,
        down_bias: jax.Array,
    ) -> jax.Array:
        output, _, _ = _forward_with_selection(
            x,
            up_weight,
            up_bias,
            down_weight,
            down_bias,
            config=config,
        )
        return output

    def forward_rule(x, up_weight, up_bias, down_weight, down_bias):
        output, values, indices = _forward_with_selection(
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
            values,
            indices,
        )

    def backward_rule(residuals, output_cotangent):
        """Sparse VJP without constructing a dense hidden cotangent.

        The loop holds only one selected decoder/encoder row per token at a
        time.  Dense-shaped parameter gradients are unavoidable because AdamW
        owns dense moment arrays, but no ``[tokens, top_k, width]`` temporary is
        materialized.
        """

        (
            x,
            up_weight,
            up_bias,
            down_weight,
            down_bias,
            values,
            indices,
        ) = residuals
        model_width, hidden_width = up_weight.shape
        tokens = math.prod(x.shape[:-1])
        flat_x = x.reshape((tokens, model_width)).astype(jnp.float32)
        flat_output_cotangent = output_cotangent.reshape((tokens, model_width)).astype(
            jnp.float32
        )
        flat_values = values.reshape((tokens, config.top_k)).astype(jnp.float32)
        flat_indices = indices.reshape((tokens, config.top_k))
        up_rows = up_weight.T.astype(jnp.float32)
        down_rows = down_weight.astype(jnp.float32)

        initial = (
            jnp.zeros((tokens, model_width), jnp.float32),
            jnp.zeros((hidden_width, model_width), jnp.float32),
            jnp.zeros((hidden_width, model_width), jnp.float32),
            jnp.zeros((hidden_width,), jnp.float32),
        )

        def visit_slot(slot, carry):
            dx, up_gradient_rows, down_gradient, up_bias_gradient = carry
            feature = flat_indices[:, slot]
            value = flat_values[:, slot]
            selected_down = down_rows[feature]
            value_cotangent = jnp.sum(flat_output_cotangent * selected_down, axis=-1)
            # ReLU defines the zero entries selected from an all-negative row
            # to be inactive. Top-k's index choice itself is nondifferentiable.
            preactivation_cotangent = value_cotangent * (value > 0.0)
            selected_up = up_rows[feature]
            dx = dx + preactivation_cotangent[:, None] * selected_up
            down_gradient = down_gradient.at[feature].add(
                value[:, None] * flat_output_cotangent
            )
            up_gradient_rows = up_gradient_rows.at[feature].add(
                preactivation_cotangent[:, None] * flat_x
            )
            up_bias_gradient = up_bias_gradient.at[feature].add(preactivation_cotangent)
            return dx, up_gradient_rows, down_gradient, up_bias_gradient

        dx, up_gradient_rows, down_gradient, up_bias_gradient = lax.fori_loop(
            0, config.top_k, visit_slot, initial
        )
        down_bias_gradient = jnp.sum(flat_output_cotangent, axis=0)
        return (
            dx.reshape(x.shape).astype(x.dtype),
            up_gradient_rows.T.astype(up_weight.dtype),
            up_bias_gradient.astype(up_bias.dtype),
            down_gradient.astype(down_weight.dtype),
            down_bias_gradient.astype(down_bias.dtype),
        )

    operation.defvjp(forward_rule, backward_rule)
    return operation


@functools.partial(jax.jit, static_argnames=("config",))
def _selected_sparse_topk_mlp(
    x: jax.Array,
    up_weight: jax.Array,
    up_bias: jax.Array,
    down_weight: jax.Array,
    down_bias: jax.Array,
    *,
    config: SparseMlpConfig,
) -> jax.Array:
    """Exact TopK-ReLU MLP with a sparse decoder and sparse custom VJP."""

    return _make_sparse_topk_mlp(config)(x, up_weight, up_bias, down_weight, down_bias)


@jax.jit
def _dense_relu_mlp(
    x: jax.Array,
    up_weight: jax.Array,
    up_bias: jax.Array,
    down_weight: jax.Array,
    down_bias: jax.Array,
) -> jax.Array:
    """Dense fast path for the exact, full-support TopK endpoint."""

    hidden = jnp.einsum("...d,dh->...h", x, up_weight.astype(x.dtype))
    hidden = jax.nn.relu(hidden + up_bias.astype(x.dtype))
    output = jnp.einsum("...h,hd->...d", hidden, down_weight.astype(x.dtype))
    return (output + down_bias.astype(x.dtype)).astype(x.dtype)


@functools.partial(jax.jit, static_argnames=("config",))
def _pallas_masked_dense_mlp(
    x: jax.Array,
    up_weight: jax.Array,
    up_bias: jax.Array,
    down_weight: jax.Array,
    down_bias: jax.Array,
    *,
    config: SparseMlpConfig,
) -> jax.Array:
    """Exact TopK-ReLU with Pallas selection and a dense TPU decoder.

    The output and VJP are those of dense -> TopK-ReLU -> dense. The zeroed
    decoder matmul deliberately uses MXU hardware rather than attempting an
    element-sparse gather that v4 cannot execute efficiently.
    """

    hidden = jnp.einsum("...d,dh->...h", x, up_weight.astype(x.dtype))
    activated = jax.nn.relu(hidden + up_bias.astype(x.dtype))
    selected = _make_pallas_masked_topk_relu(config)(activated)
    output = jnp.einsum(
        "...h,hd->...d",
        selected,
        down_weight.astype(x.dtype),
        preferred_element_type=jnp.float32,
    )
    return (output + down_bias.astype(x.dtype)).astype(x.dtype)


def sparse_topk_mlp(
    x: jax.Array,
    up_weight: jax.Array,
    up_bias: jax.Array,
    down_weight: jax.Array,
    down_bias: jax.Array,
    *,
    config: SparseMlpConfig,
) -> jax.Array:
    """Exact TopK-ReLU MLP with dense and selected-row execution paths.

    Selecting the complete dictionary is mathematically just a dense ReLU MLP.
    Avoiding a pointless full sort, gather, and sparse VJP makes that endpoint
    a faithful activation control rather than a systems penalty.
    """

    _validate_inputs(x, up_weight, up_bias, down_weight, down_bias, config)
    if config.top_k == up_weight.shape[1]:
        return _dense_relu_mlp(
            x,
            up_weight,
            up_bias,
            down_weight,
            down_bias,
        )
    if config.backend == "pallas_masked":
        return _pallas_masked_dense_mlp(
            x,
            up_weight,
            up_bias,
            down_weight,
            down_bias,
            config=config,
        )
    return _selected_sparse_topk_mlp(
        x,
        up_weight,
        up_bias,
        down_weight,
        down_bias,
        config=config,
    )


def make_mesh_sparse_topk_mlp(
    *, config: SparseMlpConfig, mesh: Mesh
) -> SparseMlpCallable:
    """Build the explicit data-sharded boundary required by Mosaic kernels.

    Parameters remain replicated. Only the leading token-batch axis is split,
    so each Pallas invocation sees its chip-local sequences and performs no
    collectives.
    """

    def local_operation(x, up_weight, up_bias, down_weight, down_bias):
        return sparse_topk_mlp(
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
    "SparseMlpCallable",
    "SparseMlpConfig",
    "make_mesh_sparse_topk_mlp",
    "naive_dense_topk_mlp",
    "pallas_masked_topk_relu",
    "pallas_masked_token_block",
    "pallas_sparse_decode",
    "reference_sparse_decode",
    "sparse_topk_mlp",
    "topk_relu",
)
