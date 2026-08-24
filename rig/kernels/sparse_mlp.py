"""Top-k sparse MLP used by the sparse-autoencoder recipe.

The encoder must score every dictionary element in order to discover the exact
top-k set.  The sparse part begins after that selection: the decoder gathers
only the selected rows of ``down_weight`` and never constructs a dense hidden
array full of zeros.  On TPU that gather/reduction is a Pallas kernel; the
small pure-JAX implementation is both the CPU fallback and the numerical
oracle for tests.
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


Backend = Literal["auto", "pallas", "reference"]
SparseMlpCallable = Callable[
    [jax.Array, jax.Array, jax.Array, jax.Array, jax.Array], jax.Array
]
TPU_VECTOR_LANES = 128


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
        if self.backend not in ("auto", "pallas", "reference"):
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

    hidden = jnp.einsum(
        "...d,dh->...h", x, up_weight, preferred_element_type=jnp.float32
    )
    hidden = hidden + up_bias
    values, indices = topk_relu(hidden, top_k=top_k)
    sparse_hidden = jnp.zeros_like(hidden)
    sparse_hidden = jnp.put_along_axis(
        sparse_hidden, indices, values, axis=-1, inplace=False
    )
    output = jnp.einsum(
        "...h,hd->...d",
        sparse_hidden,
        down_weight,
        preferred_element_type=jnp.float32,
    )
    return (output + down_bias).astype(x.dtype)


def _sparse_decode_kernel(
    indices_ref,
    values_ref,
    down_weight_ref,
    bias_ref,
    output_ref,
    selected_row_ref,
    accumulator_ref,
) -> None:
    """Decode one token block and output tile by DMAing selected rows only."""

    token_count, top_k = values_ref.shape
    output_width = output_ref.shape[-1]
    output_start = pl.program_id(0) * output_width
    accumulator_ref[...] = jnp.zeros(accumulator_ref.shape, jnp.float32)

    @pl.loop(0, token_count * top_k, unroll=False)
    def visit_assignment(assignment) -> None:
        token = assignment // top_k
        slot = assignment % top_k
        feature = indices_ref[token, slot]
        pltpu.sync_copy(
            down_weight_ref.at[feature, pl.ds(output_start, output_width)],
            selected_row_ref,
        )
        value = values_ref[token, slot].astype(jnp.float32)
        accumulator_ref[token, :] = accumulator_ref[
            token, :
        ] + value * selected_row_ref[...].astype(jnp.float32)

    output_ref[...] = (accumulator_ref[...] + bias_ref[0, :]).astype(output_ref.dtype)


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
    """Decode a fixed token block using one indirect DMA per output tile."""

    token_count, top_k = values.shape
    _, output_width = down_weight.shape
    grid = (output_width // output_block,)

    def bias_index(output_index, _indices_ref, _values_ref):
        return 0, output_index

    def output_index(output_index, _indices_ref, _values_ref):
        return 0, output_index

    with jax.named_scope("sparse_topk_decoder"):
        return pl.pallas_call(
            _sparse_decode_kernel,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                # Both arrays are deliberately bounded by token_block.  SMEM
                # gives the irregular inner loop cheap scalar addressing.
                num_scalar_prefetch=2,
                grid=grid,
                in_specs=(
                    # Keep the full decoder matrix in HBM.  The kernel issues
                    # one dynamic row DMA for each selected activation.
                    pl.BlockSpec(memory_space=pltpu.HBM),
                    pl.BlockSpec((1, output_block), bias_index),
                ),
                out_specs=pl.BlockSpec((token_count, output_block), output_index),
                scratch_shapes=(
                    pltpu.VMEM((output_block,), down_weight.dtype),
                    pltpu.VMEM((token_count, output_block), jnp.float32),
                ),
            ),
            out_shape=jax.ShapeDtypeStruct((token_count, output_width), values.dtype),
            interpret=interpret,
            debug=debug,
            compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
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

    ``indices`` and ``values`` for one token block are scalar-prefetched into
    TPU SMEM.  The kernel then DMA-loads exactly the selected rows of
    ``down_weight`` for each output tile.  Chunking bounds both SMEM and VMEM
    use independently of sequence length.
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
def sparse_topk_mlp(
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
    "pallas_sparse_decode",
    "reference_sparse_decode",
    "sparse_topk_mlp",
    "topk_relu",
)
