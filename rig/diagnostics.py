"""Shared sparse statistics for GPT-shaped parameter trees.

Recipes own when diagnostics run and which update they observe. This module
only defines the stable scope partition and the numerical reductions used by
the run-log protocol.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from rig.metrics import (
    DIAGNOSTIC_CORE_STATS,
    DIAGNOSTIC_EXTENDED_STATS,
    DIAGNOSTIC_FAMILIES,
    DIAGNOSTIC_PERCENTILE_STATS,
)


DIAGNOSTIC_PERCENTILE_SAMPLE_SIZE = 2_048


@dataclass(frozen=True, slots=True)
class DiagnosticScope:
    """One logical parameter region and the arrays reduced within it."""

    scope: str
    layer: int | None
    index: int | None
    leaves: tuple[Any, ...]

    @property
    def element_count(self) -> int:
        return sum(int(value.size) for value in self.leaves)

    @property
    def metadata(self) -> "DiagnosticScopeMetadata":
        return DiagnosticScopeMetadata(
            self.scope,
            self.layer,
            self.index,
            self.element_count,
        )


@dataclass(frozen=True, slots=True)
class DiagnosticScopeMetadata:
    """Stable log address and element count for one diagnostic scope."""

    scope: str
    layer: int | None
    index: int | None
    element_count: int


def diagnostic_scopes(
    tree: Mapping[str, Any],
    *,
    include_experts: bool = False,
) -> tuple[DiagnosticScope, ...]:
    """Group a GPT parameter-shaped tree into stable logical report scopes."""

    embeddings = tuple(jax.tree_util.tree_leaves(tree["token_embedding"]))
    blocks: list[DiagnosticScope] = []
    for layer, block in enumerate(tree["blocks"]):
        blocks.append(
            DiagnosticScope(
                "block",
                layer,
                None,
                tuple(jax.tree_util.tree_leaves(block)),
            )
        )

        if not include_experts:
            continue
        expert_names = tuple(sorted(name for name in block if name.startswith("expert_")))
        if not expert_names:
            # Mixed dense/routed stacks are valid; only routed blocks gain the
            # nested scopes when this optional view is requested.
            continue
        expert_counts = {
            int(block[name].shape[0])
            for name in expert_names
            if block[name].ndim >= 1
        }
        if len(expert_counts) != 1 or any(block[name].ndim < 1 for name in expert_names):
            raise ValueError(
                f"block {layer} expert parameters must share a leading expert axis"
            )
        experts = expert_counts.pop()
        if experts <= 0:  # pragma: no cover - arrays cannot have a negative axis
            raise ValueError(f"block {layer} has no experts")
        for expert in range(experts):
            blocks.append(
                DiagnosticScope(
                    "expert",
                    layer,
                    expert,
                    tuple(block[name][expert] for name in expert_names),
                )
            )

    final_norm = tuple(jax.tree_util.tree_leaves(tree["final_ln_scale"]))
    output: tuple[DiagnosticScope, ...] = (
        DiagnosticScope(
            "overall", None, None, tuple(jax.tree_util.tree_leaves(tree))
        ),
        DiagnosticScope("embeddings", None, None, embeddings),
        *blocks,
        DiagnosticScope("final_norm", None, None, final_norm),
    )
    if "output_embedding" in tree:
        output = (
            output[0],
            output[1],
            DiagnosticScope(
                "unembedding",
                None,
                None,
                tuple(jax.tree_util.tree_leaves(tree["output_embedding"])),
            ),
            *output[2:],
        )
    return output


def diagnostic_scope_metadata(
    params: Mapping[str, Any],
    *,
    include_experts: bool = False,
) -> tuple[DiagnosticScopeMetadata, ...]:
    """Return scope labels and exact element counts without device work."""

    return tuple(
        scope.metadata
        for scope in diagnostic_scopes(params, include_experts=include_experts)
    )


def _sampled_percentiles(values: Sequence[jax.Array], count: int) -> jax.Array:
    """Approximate five percentiles from a deterministic uniform scope sample."""

    sample_count = min(count, DIAGNOSTIC_PERCENTILE_SAMPLE_SIZE)
    # Midpoints of equal-width intervals cover the complete logical flattened
    # scope without constructing that potentially multi-gigabyte concatenation.
    positions = tuple(
        ((2 * index + 1) * count) // (2 * sample_count)
        for index in range(sample_count)
    )
    samples: list[jax.Array] = []
    cursor = 0
    offset = 0
    for value in values:
        stop = offset + int(value.size)
        start_cursor = cursor
        while cursor < sample_count and positions[cursor] < stop:
            cursor += 1
        if cursor > start_cursor:
            local_positions = jnp.asarray(
                [position - offset for position in positions[start_cursor:cursor]],
                jnp.int32,
            )
            samples.append(value.reshape(-1)[local_positions])
        offset = stop
    if cursor != sample_count:  # pragma: no cover - static accounting invariant
        raise AssertionError("diagnostic percentile sample did not cover its scope")
    sample = samples[0] if len(samples) == 1 else jnp.concatenate(samples)
    return jnp.percentile(
        sample,
        jnp.asarray((1.0, 10.0, 50.0, 90.0, 99.0), jnp.float32),
        method="linear",
    ).astype(jnp.float32)


def _diagnostic_stat_vector(
    values: Sequence[jax.Array],
    statistics: Sequence[str],
) -> jax.Array:
    """Return requested norms, moments, and sampled percentiles for arrays."""

    if not statistics or len(statistics) != len(set(statistics)):
        raise ValueError("diagnostic statistics must be nonempty and unique")
    unknown = set(statistics) - set(DIAGNOSTIC_EXTENDED_STATS)
    if unknown:
        raise ValueError(f"unknown diagnostic statistics: {sorted(unknown)}")

    values32 = tuple(value.astype(jnp.float32) for value in values)
    count = sum(int(value.size) for value in values32)
    if count <= 0:  # pragma: no cover - model scopes are statically nonempty
        raise ValueError("diagnostic scope cannot be empty")
    zero = jnp.asarray(0.0, dtype=jnp.float32)
    total = sum((jnp.sum(value) for value in values32), zero)
    mean = total / float(count)

    # Complete the mean before the centered reduction instead of deriving
    # higher moments from cancellation-prone raw power sums.
    l1_sum = sum((jnp.sum(jnp.abs(value)) for value in values32), zero)
    square_sum = sum((jnp.sum(jnp.square(value)) for value in values32), zero)
    variance_sum = sum((jnp.sum(jnp.square(value - mean)) for value in values32), zero)
    third_sum = sum((jnp.sum(jnp.power(value - mean, 3)) for value in values32), zero)
    fourth_sum = sum((jnp.sum(jnp.power(value - mean, 4)) for value in values32), zero)
    resolved = {
        "l1_norm": l1_sum,
        "l2_norm": jnp.sqrt(jnp.maximum(square_sum, zero)),
        "mean": mean,
        "std": jnp.sqrt(jnp.maximum(variance_sum / float(count), zero)),
        "third_moment": third_sum / float(count),
        "fourth_moment": fourth_sum / float(count),
    }
    if any(stat in DIAGNOSTIC_PERCENTILE_STATS for stat in statistics):
        resolved.update(
            zip(
                DIAGNOSTIC_PERCENTILE_STATS,
                _sampled_percentiles(values32, count),
                strict=True,
            )
        )
    return jnp.stack(tuple(resolved[stat] for stat in statistics)).astype(jnp.float32)


def diagnostic_values(
    params_before: Mapping[str, Any],
    raw_gradients: Mapping[str, Any],
    params_after: Mapping[str, Any],
    *,
    include_experts: bool = False,
    statistics: Sequence[str] = DIAGNOSTIC_CORE_STATS,
) -> jax.Array:
    """Return the protocol's ``[scope, family, statistic]`` diagnostic grid.

    ``param`` observes the parameter after this step, so the final point exactly
    matches the checkpoint. ``grad`` is the raw gradient before global clipping.
    ``update`` is the signed actual delta ``params_after - params_before``,
    including clipping, optimizer behavior, and decay.
    """

    updates = jax.tree_util.tree_map(
        lambda after, before: after - before, params_after, params_before
    )
    family_scopes = tuple(
        diagnostic_scopes(tree, include_experts=include_experts)
        for tree in (params_after, raw_gradients, updates)
    )
    scope_count = len(family_scopes[0])
    return jnp.stack(
        tuple(
            jnp.stack(
                tuple(
                    _diagnostic_stat_vector(
                        family_scopes[family][scope].leaves,
                        statistics,
                    )
                    for family in range(len(DIAGNOSTIC_FAMILIES))
                )
            )
            for scope in range(scope_count)
        )
    )


__all__ = (
    "DIAGNOSTIC_PERCENTILE_SAMPLE_SIZE",
    "DiagnosticScope",
    "DiagnosticScopeMetadata",
    "diagnostic_scope_metadata",
    "diagnostic_scopes",
    "diagnostic_values",
)
