"""Pure-JAX tiled tied-output projection and cross entropy.

A dense language-model head materializes ``[batch, sequence, storage_vocab]``
logits.  GPT-2-small's official shape makes that several GiB even in BF16.
This implementation instead walks the vocabulary in static tiles and keeps at
most ``[batch * sequence, vocab_tile_size]`` logits live.

This is not a Pallas kernel and does not claim to fuse the final transformer
block. It is a memory-efficient JAX primitive with a custom VJP. The VJP
deliberately recomputes one vocabulary tile at a time during the backward pass,
trading one additional output-projection pass for bounded activation memory;
neither direction constructs the full logits tensor. Dot products use the
requested compute dtype while logits, online log-sum-exp, probabilities,
gradient accumulation, and the final loss use FP32.

``DEFAULT_VOCAB_TILE_SIZE`` is a TPU-tuned starting point, not a promise that
one size is optimal for every shape. It is a multiple of the TPU MXU's 128-wide
dimension and bounds per-chip temporary logits to roughly 64 MiB for 8,192
token positions. On TPU v4, 2,048 was fastest in a 1,024/2,048/4,096/8,192
sweep for GPT-2-small's exact per-chip ``[8, 1024, 768]`` shape. Callers should
retune substantially different shapes. The storage vocabulary is padded
internally to the next complete tile, while ``semantic_vocab_size`` masks
non-token rows (50,257 versus GPT-2's padded 50,304-row embedding table).
"""

from __future__ import annotations

from functools import partial
import math
from typing import Any

import jax
import jax.numpy as jnp


DEFAULT_VOCAB_TILE_SIZE = 2_048


def _normalize_compute_dtype(compute_dtype: Any) -> jnp.dtype:
    dtype = jnp.dtype(compute_dtype)
    if not jnp.issubdtype(dtype, jnp.floating):
        raise TypeError(f"compute_dtype must be floating point, got {dtype}")
    return dtype


def _validate_inputs(
    hidden: jax.Array,
    embedding: jax.Array,
    targets: jax.Array,
    semantic_vocab_size: int,
    vocab_tile_size: int,
) -> None:
    if hidden.ndim < 1:
        raise ValueError("hidden must have at least one dimension")
    if embedding.ndim != 2:
        raise ValueError(
            f"embedding must have shape [storage_vocab, width], got {embedding.shape}"
        )
    if hidden.shape[-1] != embedding.shape[1]:
        raise ValueError(
            "hidden width must match the embedding width; "
            f"got {hidden.shape[-1]} and {embedding.shape[1]}"
        )
    if targets.shape != hidden.shape[:-1]:
        raise ValueError(
            "targets must match hidden's leading dimensions; "
            f"got {targets.shape} and {hidden.shape[:-1]}"
        )
    if not jnp.issubdtype(targets.dtype, jnp.integer):
        raise TypeError(f"targets must have an integer dtype, got {targets.dtype}")
    if not 0 < semantic_vocab_size <= embedding.shape[0]:
        raise ValueError(
            "semantic_vocab_size must be in [1, storage_vocab]; "
            f"got {semantic_vocab_size} for {embedding.shape[0]} rows"
        )
    if vocab_tile_size <= 0:
        raise ValueError(f"vocab_tile_size must be positive, got {vocab_tile_size}")


def _validate_multi_targets(hidden: jax.Array, targets: jax.Array) -> None:
    if targets.ndim != hidden.ndim or targets.shape[:-1] != hidden.shape[:-1]:
        raise ValueError(
            "multi-target ids must have shape hidden.shape[:-1] + [K]; "
            f"got {targets.shape} for hidden {hidden.shape}"
        )
    if targets.shape[-1] <= 0:
        raise ValueError("multi-target sample count K must be positive")
    if not jnp.issubdtype(targets.dtype, jnp.integer):
        raise TypeError(f"targets must have an integer dtype, got {targets.dtype}")


def _padded_embedding(
    embedding: jax.Array, semantic_vocab_size: int, vocab_tile_size: int
) -> tuple[jax.Array, int]:
    del semantic_vocab_size  # The semantic boundary is applied to every tile.
    storage_vocab = embedding.shape[0]
    tile_count = (storage_vocab + vocab_tile_size - 1) // vocab_tile_size
    padded_vocab = tile_count * vocab_tile_size
    if padded_vocab == storage_vocab:
        return embedding, tile_count
    padding = ((0, padded_vocab - storage_vocab), (0, 0))
    return jnp.pad(embedding, padding), tile_count


def _tile_logits(
    flat_hidden: jax.Array,
    padded_embedding: jax.Array,
    tile_index: jax.Array,
    vocab_tile_size: int,
    compute_dtype: jnp.dtype,
) -> jax.Array:
    start = tile_index * vocab_tile_size
    weights = jax.lax.dynamic_slice(
        padded_embedding,
        (start, jnp.asarray(0, start.dtype)),
        (vocab_tile_size, padded_embedding.shape[1]),
    )
    # preferred_element_type preserves the MXU's FP32 accumulator rather than
    # rounding the dot output to BF16 before log-sum-exp.
    return jnp.einsum(
        "nd,vd->nv",
        flat_hidden.astype(compute_dtype),
        weights.astype(compute_dtype),
        preferred_element_type=jnp.float32,
    )


def _losses_and_residual(
    hidden: jax.Array,
    embedding: jax.Array,
    targets: jax.Array,
    semantic_vocab_size: int,
    vocab_tile_size: int,
    compute_dtype: jnp.dtype,
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    flat_hidden = hidden.reshape((-1, hidden.shape[-1]))
    flat_targets = targets.reshape((-1,))
    padded_embedding, tile_count = _padded_embedding(
        embedding, semantic_vocab_size, vocab_tile_size
    )
    positions = jnp.arange(vocab_tile_size, dtype=jnp.int32)
    initial_max = jnp.full((flat_hidden.shape[0],), -jnp.inf, dtype=jnp.float32)
    initial_sum = jnp.zeros((flat_hidden.shape[0],), dtype=jnp.float32)

    def online_logsumexp(
        tile_index: jax.Array, state: tuple[jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array]:
        running_max, running_sum = state
        start = tile_index * vocab_tile_size
        logits = _tile_logits(
            flat_hidden,
            padded_embedding,
            tile_index,
            vocab_tile_size,
            compute_dtype,
        )
        valid = start + positions < semantic_vocab_size
        logits = jnp.where(valid[None, :], logits, -jnp.inf)
        tile_max = jnp.max(logits, axis=1)
        next_max = jnp.maximum(running_max, tile_max)
        next_sum = running_sum * jnp.exp(running_max - next_max)
        next_sum += jnp.sum(jnp.exp(logits - next_max[:, None]), axis=1)
        return next_max, next_sum

    maximum, exponential_sum = jax.lax.fori_loop(
        0, tile_count, online_logsumexp, (initial_max, initial_sum)
    )
    log_normalizer = maximum + jnp.log(exponential_sum)

    # Gathering the target rows costs only one hidden-sized temporary and avoids
    # constructing a one-hot [positions, vocab] tensor in the forward pass.
    valid_target = (flat_targets >= 0) & (flat_targets < semantic_vocab_size)
    # Clamp before gathering so the documented invalid-target path is defined
    # on every backend. Use the same FP32-accumulating dot contract as the
    # denominator tiles; a BF16 elementwise multiply followed by a reduction is
    # not numerically equivalent to an MXU dot at realistic model widths.
    safe_targets = jnp.clip(flat_targets, 0, semantic_vocab_size - 1)
    target_weights = embedding[safe_targets]
    target_logits = jnp.einsum(
        "nd,nd->n",
        flat_hidden.astype(compute_dtype),
        target_weights.astype(compute_dtype),
        preferred_element_type=jnp.float32,
    )
    losses = jnp.where(valid_target, log_normalizer - target_logits, jnp.inf)
    residual = (
        flat_hidden,
        embedding,
        flat_targets,
        log_normalizer,
    )
    return losses.reshape(targets.shape), residual


@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5))
def _tiled_losses(
    hidden: jax.Array,
    embedding: jax.Array,
    targets: jax.Array,
    semantic_vocab_size: int,
    vocab_tile_size: int,
    compute_dtype: jnp.dtype,
) -> jax.Array:
    losses, _ = _losses_and_residual(
        hidden,
        embedding,
        targets,
        semantic_vocab_size,
        vocab_tile_size,
        compute_dtype,
    )
    return losses


def _tiled_losses_fwd(
    hidden: jax.Array,
    embedding: jax.Array,
    targets: jax.Array,
    semantic_vocab_size: int,
    vocab_tile_size: int,
    compute_dtype: jnp.dtype,
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    return _losses_and_residual(
        hidden,
        embedding,
        targets,
        semantic_vocab_size,
        vocab_tile_size,
        compute_dtype,
    )


def _tiled_losses_bwd(
    semantic_vocab_size: int,
    vocab_tile_size: int,
    compute_dtype: jnp.dtype,
    residual: tuple[jax.Array, ...],
    loss_cotangent: jax.Array,
) -> tuple[jax.Array | None, ...]:
    flat_hidden, embedding, flat_targets, log_normalizer = residual
    padded_embedding, tile_count = _padded_embedding(
        embedding, semantic_vocab_size, vocab_tile_size
    )
    padded_vocab = padded_embedding.shape[0]
    positions = jnp.arange(vocab_tile_size, dtype=jnp.int32)
    valid_target = (flat_targets >= 0) & (flat_targets < semantic_vocab_size)
    # Invalid targets deliberately have infinite loss but no defined class
    # contribution. Their cotangents must not leak the denominator gradient.
    flat_cotangent = jnp.where(
        valid_target,
        loss_cotangent.reshape((-1,)).astype(jnp.float32),
        0.0,
    )
    grad_hidden = jnp.zeros(flat_hidden.shape, dtype=jnp.float32)
    grad_embedding = jnp.zeros((padded_vocab, embedding.shape[1]), dtype=jnp.float32)

    def tile_backward(
        tile_index: jax.Array, state: tuple[jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array]:
        hidden_gradient, embedding_gradient = state
        start = tile_index * vocab_tile_size
        weights = jax.lax.dynamic_slice(
            padded_embedding,
            (start, jnp.asarray(0, start.dtype)),
            (vocab_tile_size, padded_embedding.shape[1]),
        )
        logits = _tile_logits(
            flat_hidden,
            padded_embedding,
            tile_index,
            vocab_tile_size,
            compute_dtype,
        )
        vocabulary_ids = start + positions
        valid = vocabulary_ids < semantic_vocab_size
        probabilities = jnp.where(
            valid[None, :],
            jnp.exp(logits - log_normalizer[:, None]),
            0.0,
        )
        target_mass = flat_targets[:, None] == vocabulary_ids[None, :]
        logits_gradient = (
            probabilities - target_mass.astype(jnp.float32)
        ) * flat_cotangent[:, None]

        hidden_update = jnp.einsum(
            "nv,vd->nd",
            logits_gradient.astype(compute_dtype),
            weights.astype(compute_dtype),
            preferred_element_type=jnp.float32,
        )
        weight_update = jnp.einsum(
            "nv,nd->vd",
            logits_gradient.astype(compute_dtype),
            flat_hidden.astype(compute_dtype),
            preferred_element_type=jnp.float32,
        )
        hidden_gradient += hidden_update
        embedding_gradient = jax.lax.dynamic_update_slice(
            embedding_gradient,
            weight_update,
            (start, jnp.asarray(0, start.dtype)),
        )
        return hidden_gradient, embedding_gradient

    grad_hidden, grad_embedding = jax.lax.fori_loop(
        0, tile_count, tile_backward, (grad_hidden, grad_embedding)
    )
    return (
        grad_hidden.reshape(loss_cotangent.shape + (flat_hidden.shape[-1],)).astype(
            flat_hidden.dtype
        ),
        grad_embedding[: embedding.shape[0]].astype(embedding.dtype),
        None,
    )


_tiled_losses.defvjp(_tiled_losses_fwd, _tiled_losses_bwd)


def _weighted_multi_losses_and_residual(
    hidden: jax.Array,
    embedding: jax.Array,
    primary_targets: jax.Array,
    distill_targets: jax.Array,
    primary_weight: float,
    distill_weight: float,
    semantic_vocab_size: int,
    vocab_tile_size: int,
    compute_dtype: jnp.dtype,
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    """Forward pass for one primary and K sampled distillation targets."""

    flat_hidden = hidden.reshape((-1, hidden.shape[-1]))
    flat_primary = primary_targets.reshape((-1,))
    samples_per_position = distill_targets.shape[-1]
    flat_distill = distill_targets.reshape((-1, samples_per_position))
    padded_embedding, tile_count = _padded_embedding(
        embedding, semantic_vocab_size, vocab_tile_size
    )
    positions = jnp.arange(vocab_tile_size, dtype=jnp.int32)
    initial_max = jnp.full((flat_hidden.shape[0],), -jnp.inf, dtype=jnp.float32)
    initial_sum = jnp.zeros((flat_hidden.shape[0],), dtype=jnp.float32)

    def online_logsumexp(
        tile_index: jax.Array, state: tuple[jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array]:
        running_max, running_sum = state
        start = tile_index * vocab_tile_size
        logits = _tile_logits(
            flat_hidden,
            padded_embedding,
            tile_index,
            vocab_tile_size,
            compute_dtype,
        )
        valid = start + positions < semantic_vocab_size
        logits = jnp.where(valid[None, :], logits, -jnp.inf)
        tile_max = jnp.max(logits, axis=1)
        next_max = jnp.maximum(running_max, tile_max)
        next_sum = running_sum * jnp.exp(running_max - next_max)
        next_sum += jnp.sum(jnp.exp(logits - next_max[:, None]), axis=1)
        return next_max, next_sum

    maximum, exponential_sum = jax.lax.fori_loop(
        0, tile_count, online_logsumexp, (initial_max, initial_sum)
    )
    log_normalizer = maximum + jnp.log(exponential_sum)

    valid_primary = (flat_primary >= 0) & (flat_primary < semantic_vocab_size)
    valid_distill = jnp.all(
        (flat_distill >= 0) & (flat_distill < semantic_vocab_size), axis=1
    )
    safe_primary = jnp.clip(flat_primary, 0, semantic_vocab_size - 1)
    safe_distill = jnp.clip(flat_distill, 0, semantic_vocab_size - 1)
    primary_logits = jnp.einsum(
        "nd,nd->n",
        flat_hidden.astype(compute_dtype),
        embedding[safe_primary].astype(compute_dtype),
        preferred_element_type=jnp.float32,
    )

    # Walk the static sample dimension so only one [positions, width] gather
    # is live at a time. Materializing [positions, K, width] would turn even
    # K=16 into a multi-GiB temporary at the 8k training shape.
    def add_sample_logit(sample_index: jax.Array, total: jax.Array) -> jax.Array:
        sample_targets = jax.lax.dynamic_index_in_dim(
            safe_distill, sample_index, axis=1, keepdims=False
        )
        sample_logits = jnp.einsum(
            "nd,nd->n",
            flat_hidden.astype(compute_dtype),
            embedding[sample_targets].astype(compute_dtype),
            preferred_element_type=jnp.float32,
        )
        return total + sample_logits

    distill_logits = jax.lax.fori_loop(
        0,
        samples_per_position,
        add_sample_logit,
        jnp.zeros((flat_hidden.shape[0],), dtype=jnp.float32),
    ) / float(samples_per_position)
    valid = jnp.logical_and(
        valid_primary if primary_weight > 0.0 else True,
        valid_distill if distill_weight > 0.0 else True,
    )
    losses = (
        (primary_weight + distill_weight) * log_normalizer
        - primary_weight * primary_logits
        - distill_weight * distill_logits
    )
    losses = jnp.where(valid, losses, jnp.inf)
    residual = (
        flat_hidden,
        embedding,
        flat_primary,
        flat_distill,
        log_normalizer,
    )
    return losses.reshape(primary_targets.shape), residual


@partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6, 7, 8))
def _tiled_weighted_multi_losses(
    hidden: jax.Array,
    embedding: jax.Array,
    primary_targets: jax.Array,
    distill_targets: jax.Array,
    primary_weight: float,
    distill_weight: float,
    semantic_vocab_size: int,
    vocab_tile_size: int,
    compute_dtype: jnp.dtype,
) -> jax.Array:
    losses, _ = _weighted_multi_losses_and_residual(
        hidden,
        embedding,
        primary_targets,
        distill_targets,
        primary_weight,
        distill_weight,
        semantic_vocab_size,
        vocab_tile_size,
        compute_dtype,
    )
    return losses


def _tiled_weighted_multi_losses_fwd(
    hidden: jax.Array,
    embedding: jax.Array,
    primary_targets: jax.Array,
    distill_targets: jax.Array,
    primary_weight: float,
    distill_weight: float,
    semantic_vocab_size: int,
    vocab_tile_size: int,
    compute_dtype: jnp.dtype,
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    return _weighted_multi_losses_and_residual(
        hidden,
        embedding,
        primary_targets,
        distill_targets,
        primary_weight,
        distill_weight,
        semantic_vocab_size,
        vocab_tile_size,
        compute_dtype,
    )


def _tiled_weighted_multi_losses_bwd(
    primary_weight: float,
    distill_weight: float,
    semantic_vocab_size: int,
    vocab_tile_size: int,
    compute_dtype: jnp.dtype,
    residual: tuple[jax.Array, ...],
    loss_cotangent: jax.Array,
) -> tuple[jax.Array | None, ...]:
    flat_hidden, embedding, flat_primary, flat_distill, log_normalizer = residual
    samples_per_position = flat_distill.shape[1]
    padded_embedding, tile_count = _padded_embedding(
        embedding, semantic_vocab_size, vocab_tile_size
    )
    padded_vocab = padded_embedding.shape[0]
    positions = jnp.arange(vocab_tile_size, dtype=jnp.int32)
    valid_primary = (flat_primary >= 0) & (flat_primary < semantic_vocab_size)
    valid_distill = jnp.all(
        (flat_distill >= 0) & (flat_distill < semantic_vocab_size), axis=1
    )
    valid = jnp.logical_and(
        valid_primary if primary_weight > 0.0 else True,
        valid_distill if distill_weight > 0.0 else True,
    )
    flat_cotangent = jnp.where(
        valid,
        loss_cotangent.reshape((-1,)).astype(jnp.float32),
        0.0,
    )
    grad_hidden = jnp.zeros(flat_hidden.shape, dtype=jnp.float32)
    grad_embedding = jnp.zeros((padded_vocab, embedding.shape[1]), dtype=jnp.float32)
    total_weight = primary_weight + distill_weight

    def tile_backward(
        tile_index: jax.Array, state: tuple[jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array]:
        hidden_gradient, embedding_gradient = state
        start = tile_index * vocab_tile_size
        weights = jax.lax.dynamic_slice(
            padded_embedding,
            (start, jnp.asarray(0, start.dtype)),
            (vocab_tile_size, padded_embedding.shape[1]),
        )
        logits = _tile_logits(
            flat_hidden,
            padded_embedding,
            tile_index,
            vocab_tile_size,
            compute_dtype,
        )
        vocabulary_ids = start + positions
        semantic = vocabulary_ids < semantic_vocab_size
        probabilities = jnp.where(
            semantic[None, :],
            jnp.exp(logits - log_normalizer[:, None]),
            0.0,
        )
        primary_mass = flat_primary[:, None] == vocabulary_ids[None, :]

        # Build only this vocabulary tile of the empirical teacher
        # distribution. The O(K) comparisons sit beside an O(width) MXU
        # projection, while the live target mass stays [positions, tile].
        def add_sample_mass(sample_index: jax.Array, mass: jax.Array) -> jax.Array:
            sample_targets = jax.lax.dynamic_index_in_dim(
                flat_distill, sample_index, axis=1, keepdims=False
            )
            return mass + (
                sample_targets[:, None] == vocabulary_ids[None, :]
            ).astype(jnp.float32)

        distill_mass = jax.lax.fori_loop(
            0,
            samples_per_position,
            add_sample_mass,
            jnp.zeros_like(probabilities, dtype=jnp.float32),
        ) / float(samples_per_position)
        logits_gradient = (
            total_weight * probabilities
            - primary_weight * primary_mass.astype(jnp.float32)
            - distill_weight * distill_mass
        ) * flat_cotangent[:, None]

        hidden_gradient += jnp.einsum(
            "nv,vd->nd",
            logits_gradient.astype(compute_dtype),
            weights.astype(compute_dtype),
            preferred_element_type=jnp.float32,
        )
        weight_update = jnp.einsum(
            "nv,nd->vd",
            logits_gradient.astype(compute_dtype),
            flat_hidden.astype(compute_dtype),
            preferred_element_type=jnp.float32,
        )
        embedding_gradient = jax.lax.dynamic_update_slice(
            embedding_gradient,
            weight_update,
            (start, jnp.asarray(0, start.dtype)),
        )
        return hidden_gradient, embedding_gradient

    grad_hidden, grad_embedding = jax.lax.fori_loop(
        0, tile_count, tile_backward, (grad_hidden, grad_embedding)
    )
    return (
        grad_hidden.reshape(loss_cotangent.shape + (flat_hidden.shape[-1],)).astype(
            flat_hidden.dtype
        ),
        grad_embedding[: embedding.shape[0]].astype(embedding.dtype),
        None,
        None,
    )


_tiled_weighted_multi_losses.defvjp(
    _tiled_weighted_multi_losses_fwd, _tiled_weighted_multi_losses_bwd
)


def tiled_tied_cross_entropy_losses(
    hidden: jax.Array,
    embedding: jax.Array,
    targets: jax.Array,
    *,
    semantic_vocab_size: int,
    vocab_tile_size: int = DEFAULT_VOCAB_TILE_SIZE,
    compute_dtype: Any = jnp.bfloat16,
) -> jax.Array:
    """Return one cross-entropy value per target without full vocabulary logits.

    Args:
        hidden: Final normalized activations shaped ``[..., width]``.
        embedding: Tied embedding table shaped ``[storage_vocab, width]``.
        targets: Integer token ids shaped like ``hidden.shape[:-1]``.
        semantic_vocab_size: Number of real output classes. Rows at or above
            this boundary remain available for aligned storage but receive no
            probability mass or output-head gradient.
        vocab_tile_size: Static vocabulary tile. Multiples of 128 are natural
            TPU tuning candidates; non-divisible storage sizes are supported.
        compute_dtype: Operand dtype for projection matmuls. Accumulation and
            every numerically sensitive reduction remain FP32.

    Invalid target ids produce an infinite loss. Dataset validation should
    reject them before compilation.
    """

    semantic_vocab_size = int(semantic_vocab_size)
    vocab_tile_size = int(vocab_tile_size)
    compute_dtype = _normalize_compute_dtype(compute_dtype)
    _validate_inputs(hidden, embedding, targets, semantic_vocab_size, vocab_tile_size)
    return _tiled_losses(
        hidden,
        embedding,
        targets,
        semantic_vocab_size,
        vocab_tile_size,
        compute_dtype,
    )


def tiled_tied_weighted_multi_cross_entropy_losses(
    hidden: jax.Array,
    embedding: jax.Array,
    primary_targets: jax.Array,
    distill_targets: jax.Array,
    *,
    primary_weight: float,
    distill_weight: float,
    semantic_vocab_size: int,
    vocab_tile_size: int = DEFAULT_VOCAB_TILE_SIZE,
    compute_dtype: Any = jnp.bfloat16,
) -> jax.Array:
    """Return primary CE plus the mean CE of K sampled teacher targets.

    ``distill_targets`` has shape ``hidden.shape[:-1] + [K]``.  When its K
    targets are iid samples from a teacher distribution, their mean is an
    unbiased lower-variance estimator of the corresponding soft-target cross
    entropy.  The implementation shares one vocabulary-wide normalizer and
    never materializes a dense teacher distribution.
    """

    primary_weight = float(primary_weight)
    distill_weight = float(distill_weight)
    if (
        not math.isfinite(primary_weight)
        or not math.isfinite(distill_weight)
        or primary_weight < 0.0
        or distill_weight < 0.0
        or primary_weight + distill_weight <= 0.0
    ):
        raise ValueError("target weights must be finite, nonnegative, and not both zero")
    semantic_vocab_size = int(semantic_vocab_size)
    vocab_tile_size = int(vocab_tile_size)
    compute_dtype = _normalize_compute_dtype(compute_dtype)
    _validate_inputs(
        hidden,
        embedding,
        primary_targets,
        semantic_vocab_size,
        vocab_tile_size,
    )
    _validate_multi_targets(hidden, distill_targets)
    return _tiled_weighted_multi_losses(
        hidden,
        embedding,
        primary_targets,
        distill_targets,
        primary_weight,
        distill_weight,
        semantic_vocab_size,
        vocab_tile_size,
        compute_dtype,
    )


def tiled_tied_weighted_dual_cross_entropy_losses(
    hidden: jax.Array,
    embedding: jax.Array,
    primary_targets: jax.Array,
    distill_targets: jax.Array,
    *,
    primary_weight: float,
    distill_weight: float,
    semantic_vocab_size: int,
    vocab_tile_size: int = DEFAULT_VOCAB_TILE_SIZE,
    compute_dtype: Any = jnp.bfloat16,
) -> jax.Array:
    """Backward-compatible K=1 weighted teacher loss."""

    return tiled_tied_weighted_multi_cross_entropy_losses(
        hidden,
        embedding,
        primary_targets,
        distill_targets[..., None],
        primary_weight=primary_weight,
        distill_weight=distill_weight,
        semantic_vocab_size=semantic_vocab_size,
        vocab_tile_size=vocab_tile_size,
        compute_dtype=compute_dtype,
    )


def tiled_tied_cross_entropy(
    hidden: jax.Array,
    embedding: jax.Array,
    targets: jax.Array,
    *,
    semantic_vocab_size: int,
    vocab_tile_size: int = DEFAULT_VOCAB_TILE_SIZE,
    compute_dtype: Any = jnp.bfloat16,
) -> jax.Array:
    """Return mean tiled tied-embedding cross entropy in FP32."""

    losses = tiled_tied_cross_entropy_losses(
        hidden,
        embedding,
        targets,
        semantic_vocab_size=semantic_vocab_size,
        vocab_tile_size=vocab_tile_size,
        compute_dtype=compute_dtype,
    )
    return jnp.mean(losses, dtype=jnp.float32)


def tiled_tied_weighted_multi_cross_entropy(
    hidden: jax.Array,
    embedding: jax.Array,
    primary_targets: jax.Array,
    distill_targets: jax.Array,
    *,
    primary_weight: float,
    distill_weight: float,
    semantic_vocab_size: int,
    vocab_tile_size: int = DEFAULT_VOCAB_TILE_SIZE,
    compute_dtype: Any = jnp.bfloat16,
) -> jax.Array:
    """Return the mean weighted K-sample teacher cross entropy in FP32."""

    losses = tiled_tied_weighted_multi_cross_entropy_losses(
        hidden,
        embedding,
        primary_targets,
        distill_targets,
        primary_weight=primary_weight,
        distill_weight=distill_weight,
        semantic_vocab_size=semantic_vocab_size,
        vocab_tile_size=vocab_tile_size,
        compute_dtype=compute_dtype,
    )
    return jnp.mean(losses, dtype=jnp.float32)


def tiled_tied_weighted_dual_cross_entropy(
    hidden: jax.Array,
    embedding: jax.Array,
    primary_targets: jax.Array,
    distill_targets: jax.Array,
    *,
    primary_weight: float,
    distill_weight: float,
    semantic_vocab_size: int,
    vocab_tile_size: int = DEFAULT_VOCAB_TILE_SIZE,
    compute_dtype: Any = jnp.bfloat16,
) -> jax.Array:
    """Backward-compatible mean K=1 weighted teacher loss."""

    return tiled_tied_weighted_multi_cross_entropy(
        hidden,
        embedding,
        primary_targets,
        distill_targets[..., None],
        primary_weight=primary_weight,
        distill_weight=distill_weight,
        semantic_vocab_size=semantic_vocab_size,
        vocab_tile_size=vocab_tile_size,
        compute_dtype=compute_dtype,
    )
