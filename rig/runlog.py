"""Everything a run writes, and the profiler window it writes it around.

A recipe decides what to measure. It does not need to decide how a curve
reaches disk, how a checkpoint is made atomic, or how the harness is told a
run finished -- those are the same for every recipe and changing them per
recipe would break the collation the harness does across runs.

Nothing here takes a recipe's config. The writers take the two numbers the
axes are derived from (``tokens_per_step``, ``flops_per_token``) plus the step
the run ended on, and :func:`save_checkpoint` takes an opaque metadata mapping
that the recipe builds, because the model contract inside it is exactly the
part that is not shared.

On ``validation.csv`` staying CSV: see :func:`write_validation_csv`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import argparse
import csv
import json
import os

import jax
import numpy as np

from rig import logpack
from rig.diagnostics import DiagnosticScopeMetadata
from rig.metrics import DIAGNOSTIC_CORE_STATS, DIAGNOSTIC_FAMILIES
from rig.mesh import local_device_get
from rig.nn import flatten_arrays


RESULT_PREFIX = "RIG_RESULT="
CHECKPOINT_NAME = "checkpoint.npz"
TRAINING_LOG_NAME = f"training{logpack.SUFFIX}"
VALIDATION_CSV_NAME = "validation.csv"
DIAGNOSTICS_LOG_NAME = f"diagnostics{logpack.SUFFIX}"


@dataclass(frozen=True)
class ValidationRow:
    step: int
    tokens_processed: int
    kind: str
    domain: str
    validation_tokens: int
    validation_loss: float
    perplexity: float
    validation_seconds: float
    canonical: bool


@dataclass(frozen=True)
class DiagnosticPoint:
    """Host-side statistics captured at one optimizer step."""

    step: int
    values: np.ndarray


def xprof_step_window(
    args: argparse.Namespace, total_steps: int
) -> tuple[int, int] | None:
    """Return the inclusive 1-based capture window, validating its bounds."""

    if args.xprof_dir is None:
        return None
    # ``validate_args`` establishes that these are both positive integers.
    start = int(args.xprof_start_step)
    end = start + int(args.xprof_steps) - 1
    if start > total_steps or end > total_steps:
        raise ValueError(
            "XProf capture window must fit inside the training run; "
            f"requested steps {start}..{end} of a {total_steps}-step run"
        )
    return start, end


def profiler_options(platform: str, device_count: int) -> Any:
    """Build an XProf configuration with useful TPU compute and sync events."""

    options = jax.profiler.ProfileOptions()
    options.python_tracer_level = 0
    options.host_tracer_level = 2
    if platform == "tpu":
        options.advanced_configuration = {
            "tpu_trace_mode": "TRACE_COMPUTE_AND_SYNC",
            "tpu_num_chips_to_profile_per_task": device_count,
        }
    return options


def save_checkpoint(output_dir: Path, params: Any, metadata: Mapping[str, Any]) -> None:
    """Write params plus an opaque metadata blob as one atomic ``.npz``.

    ``metadata`` is whatever the recipe wants to travel with the weights --
    typically its model contract. This function does not read it, because what
    identifies a model is exactly the part that is not shared between recipes.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = flatten_arrays(local_device_get(params))
    arrays["metadata.json"] = np.frombuffer(
        json.dumps(metadata, sort_keys=True).encode("utf-8"), dtype=np.uint8
    )
    destination = output_dir / CHECKPOINT_NAME
    temporary = output_dir / f".{CHECKPOINT_NAME}.tmp"
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def write_result(output_dir: Path, result: Mapping[str, Any]) -> None:
    destination = output_dir / "metrics.json"
    temporary = output_dir / ".metrics.json.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


# The per-layer routing statistics, in the order the recipe stacks them.
ROUTER_SUMMARY_METRICS = ("router.entropy", "router.top1_gate", "router.logit_rms")


def training_log_columns(
    routing_layers: int = 0, experts: int = 0
) -> tuple[logpack.Column, ...]:
    """The per-step scalars, in the order ``history`` stores them.

    A routed run appends its routing statistics after the three dense ones:
    first model-wide, then per layer, and per layer the full per-expert load
    vector. Per layer matters because routing does not collapse uniformly --
    one layer can send everything to a single expert while the model-wide
    average still looks even, and the loss curve alone never shows it.

    Per-expert load is stored rather than summarized because with a handful of
    experts the vector *is* the distribution: max, min, and any histogram of it
    are derivable, so storing those too would only be storing them twice.

    Dense runs pass no routing arguments and get exactly the three columns they
    always had, so nothing about their artifacts changes.
    """

    columns = [
        logpack.column("train_loss"),
        logpack.column("learning_rate"),
        logpack.column("grad_norm"),
    ]
    if not routing_layers:
        return tuple(columns)

    columns += [
        logpack.column("router.balance_loss"),
        logpack.column("router.max_load"),
        logpack.column("router.min_load"),
        *(logpack.column(name) for name in ROUTER_SUMMARY_METRICS),
    ]
    for layer in range(routing_layers):
        columns += [
            logpack.column(name, "block", layer) for name in ROUTER_SUMMARY_METRICS
        ]
        columns += [
            logpack.column("router.load", "expert", layer, index=expert)
            for expert in range(experts)
        ]
    return tuple(columns)


def diagnostic_log_columns(
    scope_metadata: Sequence[
        DiagnosticScopeMetadata | tuple[str, int | None, int]
    ],
    *,
    statistics: Sequence[str] = DIAGNOSTIC_CORE_STATS,
) -> tuple[logpack.Column, ...]:
    """Flatten the ``[scope, family, stat]`` grid into registry-addressed columns.

    Order matches ``diagnostic_values``' array layout exactly, so a captured
    point flattens straight into a record with no per-value bookkeeping.
    """

    if not statistics or len(statistics) != len(set(statistics)):
        raise ValueError("diagnostic statistics must be nonempty and unique")
    columns: list[logpack.Column] = []
    for entry in scope_metadata:
        metadata = (
            entry
            if isinstance(entry, DiagnosticScopeMetadata)
            else DiagnosticScopeMetadata(entry[0], entry[1], None, entry[2])
        )
        columns.extend(
            logpack.column(
                f"{family}.{stat}",
                metadata.scope,
                metadata.layer,
                element_count=metadata.element_count,
                index=metadata.index,
            )
            for family in DIAGNOSTIC_FAMILIES
            for stat in statistics
        )
    return tuple(columns)


DIAGNOSTIC_FLUSH_POINTS = 64


def open_log(
    destination: Path,
    columns: Sequence[logpack.Column],
    *,
    tokens_per_step: int,
    flops_per_token: int | None,
) -> logpack.LogWriter | None:
    """Open a log for incremental appends, or None when it cannot be recorded.

    Returning None rather than raising keeps logging a convenience: a run must
    not die because its curve could not be opened.
    """

    if flops_per_token is None or flops_per_token <= 0:
        return None
    try:
        return logpack.LogWriter(
            destination,
            columns,
            tokens_per_step=tokens_per_step,
            flops_per_token=float(flops_per_token),
        )
    except (OSError, ValueError, KeyError):
        return None


def append_log_row(
    writer: logpack.LogWriter | None, step: int, values: Iterable[float]
) -> None:
    """Append one already-materialized sample, best effort.

    Salvage for runs that never reach their final write: on preemptible
    hardware the job can vanish at any step, and the complete artifacts are
    written only after the loop exits. Because the log is append-only, whatever
    reached disk stays readable with no repair, and a run that finishes replaces
    the file at full resolution.

    Never raises. A partial artifact is a convenience; failing the run because a
    convenience could not be written would be worse than losing it.
    """

    if writer is None:
        return
    try:
        writer.append(step, values)
    except (OSError, ValueError):
        return


def close_log(writer: logpack.LogWriter | None) -> None:
    """Close a log writer, tolerating a handle that never opened."""

    if writer is None:
        return
    try:
        writer.close()
    except OSError:
        return


def write_training_log(
    output_dir: Path,
    history: np.ndarray,
    *,
    tokens_per_step: int,
    final_step: int,
    flops_per_token: int | None = None,
    columns: Sequence[logpack.Column] | None = None,
) -> None:
    """Atomically persist every optimizer step without timing host transfers.

    Supersedes the coarser rows appended during the run: this writes the full
    per-step history pulled from the device buffer once, at the end.

    Because it supersedes rather than merges, anything a run appends live but
    does not also keep in the device history is silently dropped when this
    runs. Whatever ``columns`` describes must therefore be exactly what the
    history rows carry -- which is why the width is checked against them.
    """

    if history.shape[0] > final_step:
        # The device buffer is sized for the full horizon; an early-stopped run
        # fills only its prefix and the remainder is untouched zeros.
        history = history[:final_step]
    resolved = tuple(training_log_columns() if columns is None else columns)
    if history.shape != (final_step, len(resolved)):
        raise ValueError(
            f"training history has shape {history.shape}; "
            f"expected {(final_step, len(resolved))}"
        )
    if flops_per_token is not None and flops_per_token <= 0:
        raise ValueError("flops_per_token must be positive when provided")
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / TRAINING_LOG_NAME
    temporary = output_dir / f".{TRAINING_LOG_NAME}.tmp"
    with logpack.LogWriter(
        temporary,
        resolved,
        tokens_per_step=tokens_per_step,
        # Only the FLOP axis needs this; a run without a traced count still
        # records its curve, and the axis is simply not meaningful.
        flops_per_token=float(flops_per_token or 1.0),
    ) as writer:
        for index, row in enumerate(history, 1):
            writer.append(index, row)
    os.replace(temporary, destination)


def write_diagnostics_log(
    output_dir: Path,
    points: Sequence[DiagnosticPoint],
    scope_metadata: Sequence[
        DiagnosticScopeMetadata | tuple[str, int | None, int]
    ],
    *,
    tokens_per_step: int,
    final_step: int,
    flops_per_token: int,
    statistics: Sequence[str] = DIAGNOSTIC_CORE_STATS,
) -> None:
    """Atomically persist the sparse optimizer diagnostics."""

    if not points:
        raise ValueError("diagnostic history cannot be empty")
    if flops_per_token <= 0:
        raise ValueError("flops_per_token must be positive")
    expected_shape = (
        len(scope_metadata),
        len(DIAGNOSTIC_FAMILIES),
        len(statistics),
    )
    if points[-1].step != final_step:
        raise ValueError("diagnostic history must include the final optimizer step")
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / DIAGNOSTICS_LOG_NAME
    temporary = output_dir / f".{DIAGNOSTICS_LOG_NAME}.tmp"
    previous_step = 0
    with logpack.LogWriter(
        temporary,
        diagnostic_log_columns(scope_metadata, statistics=statistics),
        tokens_per_step=tokens_per_step,
        flops_per_token=float(flops_per_token),
    ) as writer:
        for point in points:
            if point.step <= previous_step or point.step > final_step:
                raise ValueError("diagnostic steps must be unique and increasing")
            previous_step = point.step
            values = np.asarray(point.values, dtype=np.float32)
            if values.shape != expected_shape:
                raise ValueError(
                    f"diagnostic values have shape {values.shape}; "
                    f"expected {expected_shape}"
                )
            if not np.all(np.isfinite(values)):
                raise FloatingPointError(
                    f"diagnostic values at step {point.step} must be finite"
                )
            writer.append(point.step, values.reshape(-1))
    os.replace(temporary, destination)


def write_validation_csv(output_dir: Path, rows: Sequence[ValidationRow]) -> None:
    """Persist FineWeb probes/final and optional downstream domain scores.

    Deliberately still CSV while the curves are packed logs, for two reasons.

    Size is not one of them either way: an official run writes on the order of
    165 rows -- one canonical score, a probe every 500 steps, ten domains and a
    macro -- which is about 15 KB. The packing that took diagnostics from
    144 MB to 6.4 MB has nothing to work on here.

    The blocking reason is that two of these columns are categorical. A packed
    log is a float32 matrix addressed by permanent metric ids, and ``domain``
    holds names that come from a *dataset manifest*. Encoding them would mean
    minting a permanent global id for every Fresh10 domain, so the add-only
    registry would grow with each dataset and a retired dataset's names could
    never be reused. That is the wrong thing for a permanent id space to
    absorb, and it would buy 15 KB.
    """

    canonical_rows = [row for row in rows if row.canonical]
    if len(canonical_rows) != 1 or canonical_rows[0].kind != "fineweb":
        raise ValueError(
            "validation history must contain one canonical in-distribution row"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / VALIDATION_CSV_NAME
    temporary = output_dir / f".{VALIDATION_CSV_NAME}.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "step",
                "tokens_processed",
                "kind",
                "domain",
                "validation_tokens",
                "validation_loss",
                "perplexity",
                "validation_seconds",
                "canonical",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    int(row.step),
                    int(row.tokens_processed),
                    row.kind,
                    row.domain,
                    int(row.validation_tokens),
                    float(row.validation_loss),
                    float(row.perplexity),
                    float(row.validation_seconds),
                    "true" if row.canonical else "false",
                )
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
