"""Build a self-contained, dependency-free HTML report from benchmark run logs.

The report reader is intentionally conservative.  A successful result event and a
sound training CSV are required, ledger hashes are checked when available, and
malformed runs are listed rather than silently mixed into plots.  This is not a
replacement for ``rig verify``; it is a fast, read-only integrity gate for
visualization.
"""

from __future__ import annotations

import base64
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile
import shutil
from typing import Any, Mapping, Sequence

import numpy as np

from rig import logpack

# ``metrics`` is a local name in several readers, so the registry is
# imported under a distinct one rather than being shadowed.
from rig import metrics as metrics_registry


_MAX_JSON_BYTES = 4 * 1024 * 1024
_MAX_ARTIFACT_BYTES = 128 * 1024 * 1024
TRAINING_LOG_NAME = f"training{logpack.SUFFIX}"
DIAGNOSTICS_LOG_NAME = f"diagnostics{logpack.SUFFIX}"
# One budget for every series, so nothing in a file sits at a different
# fidelity than anything beside it. These single-file reports are a portable
# overview; the lossless originals live in the dataset repository, and the
# study page fetches them on request. 0 embeds every recorded sample.
_MAX_CHART_POINTS = 1_400
# 0 keeps every recorded diagnostic step, so the step dragger moves at the
# granularity the run actually recorded. A positive value thins the step axis,
# trading dragger resolution for file size; first and last are always kept.
_MAX_LAYER_SNAPSHOTS = 1_400
_LAYER_KEY = re.compile(r"(?:^|/)(?:blocks?|layers?|h)(?:/|_)(\d+)(?:/|$)")
_COLORS = (
    "#7dd3fc",
    "#f9a8d4",
    "#86efac",
    "#fde047",
    "#c4b5fd",
    "#fb923c",
    "#67e8f9",
    "#fca5a5",
    "#a3e635",
    "#d8b4fe",
    "#fcd34d",
    "#5eead4",
)
_FRESH10_ORDER = (
    "science",
    "medicine",
    "software",
    "history",
    "fiction",
    "government",
    "legal",
    "economics",
    "climate",
    "education",
)
_TRAIN_REQUIRED = ("step", "tokens_processed", "train_loss")
_VALIDATION_REQUIRED = (
    "step",
    "tokens_processed",
    "kind",
    "domain",
    "validation_loss",
)
_DIAGNOSTIC_REQUIRED = (
    "step",
    "tokens_processed",
    "cumulative_estimated_flops",
    "scope",
    "layer",
    "family",
    "stat",
    "value",
    "element_count",
)
_DIAGNOSTIC_SCOPES = frozenset(
    {"overall", "embeddings", "unembedding", "block", "expert", "final_norm"}
)
_DIAGNOSTIC_FAMILIES = ("grad", "update", "param")
_DIAGNOSTIC_STATS = (
    "l1_norm",
    "l2_norm",
    "mean",
    "std",
    "third_moment",
    "fourth_moment",
    "p01",
    "p10",
    "p50",
    "p90",
    "p99",
)
_PRIMARY_TRAIN_METRICS = ("train_loss", "learning_rate")
_LAYER_METRICS = (
    "l2_norm",
    "l1_norm",
    "mean",
    "std",
    "third_moment",
    "fourth_moment",
    "min",
    "max",
)


class ReportError(ValueError):
    """A report input or destination is invalid."""


@dataclass(frozen=True)
class ReportSummary:
    """Result of one report build."""

    output_path: Path
    included: tuple[str, ...]
    skipped: Mapping[str, str]
    notices: tuple[str, ...] = ()


@dataclass
class _Run:
    """One recorded run, holding its logs as packed arrays rather than rows.

    ``training`` and ``diagnostics`` are read with one memcpy each and sliced by
    column, so a 76k-step run costs a few milliseconds and a few megabytes
    instead of the million-plus row dictionaries the long-form CSV required.
    """

    run_id: str
    result: dict[str, Any]
    record: dict[str, Any] | None
    training: logpack.Log
    validation: list[dict[str, Any]]
    flop_source: str
    diagnostics: logpack.Log | None = None
    layer_stats: dict[str, list[dict[str, float]]] = field(default_factory=dict)
    notices: list[str] = field(default_factory=list)

    @property
    def final_step(self) -> int:
        return int(self.training.steps[-1])

    def training_points(self, metric: str) -> list[list[float]]:
        """Return ``[step, cumulative_flops, value]`` triples for one scalar."""

        values = self.training.series(metric)
        if values is None:
            return []
        return _points(self.training, values)

    def diagnostic_points(
        self,
        family: str,
        statistic: str,
        scope: str = "overall",
        layer: int | None = None,
    ) -> list[list[float]]:
        if self.diagnostics is None:
            return []
        values = self.diagnostics.series(f"{family}.{statistic}", scope, layer)
        if values is None:
            return []
        return _points(self.diagnostics, values)

    def has_layer_diagnostics(self) -> bool:
        if self.diagnostics is None:
            return False
        overall = metrics_registry.scope("overall").id
        return any(entry.scope_id != overall for entry in self.diagnostics.columns)


def _points(log: logpack.Log, values: "np.ndarray") -> list[list[float]]:
    """Zip a column against the step and FLOP axes, dropping non-finite samples."""

    finite = np.isfinite(values)
    steps = log.steps[finite]
    flops = log.axis("cumulative_flops")[finite]
    return [
        [float(step), float(flop), float(value)]
        for step, flop, value in zip(steps, flops, values[finite])
    ]


def build_report(
    runs_dir: Path,
    output_path: Path,
    *,
    max_chart_points: int = _MAX_CHART_POINTS,
    layer_snapshots: int = _MAX_LAYER_SNAPSHOTS,
    select: str | None = None,
) -> ReportSummary:
    """Scan ``runs_dir`` and write one portable report HTML file.

    Every successful run is plotted, whatever its profile, token budget, or
    loss. ``max_chart_points`` bounds the embedded data and canvas work per
    series; the first and last points are always retained.

    ``select`` is an optional regular expression matched against the run id.
    One report per line of research is the working pattern here -- a short note
    plus a single self-contained page, after which the raw logs can be cleared
    -- and that needs a way to crystallize one family without disturbing runs
    that are still in flight. Non-matching directories are omitted silently
    rather than reported as skipped: they were never candidates.
    """

    runs_dir = Path(runs_dir).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if isinstance(max_chart_points, bool) or max_chart_points < 0:
        raise ReportError("max_chart_points must be 0 (every sample) or positive")
    if 0 < max_chart_points < 32:
        raise ReportError("max_chart_points must be 0 (every sample) or at least 32")
    if isinstance(layer_snapshots, bool) or layer_snapshots < 0:
        raise ReportError("layer_snapshots must be 0 (every step) or a positive count")
    if 0 < layer_snapshots < 2:
        raise ReportError("layer_snapshots must be 0 (every step) or at least 2")
    if not runs_dir.is_dir():
        raise ReportError(f"runs directory does not exist: {runs_dir}")
    if output_path.exists() and (output_path.is_dir() or output_path.is_symlink()):
        raise ReportError(
            "report output must be a regular file, not a directory or symlink"
        )

    try:
        selector = re.compile(select) if select is not None else None
    except re.error as error:
        raise ReportError(f"invalid --select regular expression: {error}") from error

    records, duplicate_record_ids, ledger_notices = _read_ledger(
        runs_dir / "records.jsonl"
    )
    included: list[_Run] = []
    skipped: dict[str, str] = {}
    for candidate in sorted(runs_dir.iterdir()):
        if (
            not candidate.is_dir()
            or candidate.is_symlink()
            or candidate.name.startswith(".")
        ):
            continue
        if selector is not None and not selector.search(candidate.name):
            continue
        if not (candidate / "result.json").exists():
            # Failed/partial harness directories are expected and are not plot-able.
            continue
        if candidate.name in duplicate_record_ids:
            skipped[candidate.name] = (
                "records.jsonl contains duplicate entries for this run ID; "
                "the ledger identity is ambiguous"
            )
            continue
        try:
            admission_reason = _report_admission_reason(candidate)
        except (OSError, ReportError, json.JSONDecodeError) as exc:
            skipped[candidate.name] = str(exc)
            continue
        if admission_reason is not None:
            skipped[candidate.name] = admission_reason
            continue
        try:
            run = _read_run(candidate, records.get(candidate.name), max_chart_points)
        except (OSError, ReportError, csv.Error, json.JSONDecodeError) as exc:
            skipped[candidate.name] = str(exc)
            continue
        included.append(run)

    notices = list(ledger_notices)
    notices.append(
        "Every successful run is plotted. Profile and token budget are shown "
        "per run; qualification is decided by `rig leaderboard`, not here."
    )
    notices.extend(notice for run in included for notice in run.notices)
    payload = _report_payload(
        included,
        skipped,
        notices,
        max_chart_points=max_chart_points,
        layer_snapshots=layer_snapshots,
    )
    html = _render_html(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return ReportSummary(
        output_path=output_path,
        included=tuple(run.run_id for run in included),
        skipped=skipped,
        notices=tuple(notices),
    )


def _report_admission_reason(run_dir: Path) -> str | None:
    """Admit any successful run; the dashboard is for looking at all of them.

    There is deliberately no token-budget, loss, or profile gate. Those once
    restricted the report to the historical 624,984,064-token v4-8 baseline,
    which silently excluded every run of the current tiered family. Whether a
    run *qualifies* is a leaderboard question, answered by ``rig leaderboard``
    against the target in docs/RULES.md; it is not a reason to hide a curve.

    Structural integrity is still enforced: a malformed, unsuccessful, or
    wrong-schema result is an error rather than a silent omission.
    """

    result_path = _regular_file(run_dir, "result.json")
    if result_path.stat().st_size > _MAX_JSON_BYTES:
        raise ReportError("result.json is implausibly large")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ReportError("result.json is not an object")
    if result.get("schema_version") != 1 or result.get("status") != "ok":
        raise ReportError("result.json is not a successful schema-v1 result")
    if result.get("profile") not in {"smoke", "dev", "official"}:
        raise ReportError("result profile is invalid")
    if result.get("track") not in {"open", "sample_efficiency"}:
        raise ReportError("result track is invalid")
    metrics = _object(result.get("metrics"), "result metrics")
    _positive_int(metrics.get("tokens_processed"), "tokens_processed")
    _finite(metrics.get("validation_loss"), "validation_loss")
    return None


def _read_ledger(
    path: Path,
) -> tuple[dict[str, dict[str, Any]], set[str], list[str]]:
    if not path.exists():
        return {}, set(), ["records.jsonl is absent; runs are marked as unledgered."]
    if not path.is_file() or path.is_symlink():
        return (
            {},
            set(),
            ["records.jsonl is not a regular file; ledger checks were skipped."],
        )
    records: dict[str, dict[str, Any]] = {}
    duplicate_ids: set[str] = set()
    notices: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                notices.append(f"Ignored malformed records.jsonl line {number}.")
                continue
            if not isinstance(value, dict) or not isinstance(value.get("run_id"), str):
                notices.append(f"Ignored non-record records.jsonl line {number}.")
                continue
            run_id = value["run_id"]
            if run_id in records or run_id in duplicate_ids:
                records.pop(run_id, None)
                if run_id not in duplicate_ids:
                    notices.append(
                        f"Duplicate records.jsonl run ID {run_id!r}; that run was excluded."
                    )
                duplicate_ids.add(run_id)
                continue
            records[run_id] = value
    return records, duplicate_ids, notices


def _read_run(path: Path, record: dict[str, Any] | None, limit: int) -> _Run:
    run_id = path.name
    result_path = _regular_file(path, "result.json")
    if result_path.stat().st_size > _MAX_JSON_BYTES:
        raise ReportError("result.json is implausibly large")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ReportError("result.json is not an object")
    if result.get("schema_version") != 1 or result.get("status") != "ok":
        raise ReportError("result.json is not a successful schema-v1 result")
    _check_identity(result, record, run_id)

    metrics = _object(result.get("metrics"), "result metrics")
    final_tokens = _positive_int(metrics.get("tokens_processed"), "tokens_processed")
    _positive_finite(metrics.get("train_seconds"), "train_seconds")
    final_train_loss = _finite(metrics.get("train_loss"), "train_loss")
    _finite(metrics.get("validation_loss"), "validation_loss")

    training_path = _artifact_path(
        path, result, "training_curve", TRAINING_LOG_NAME, required=True
    )
    _check_record_artifact(training_path, path, record, "training_curve")
    training = _read_training(training_path)
    if int(training.axis("tokens_processed")[-1]) != final_tokens:
        raise ReportError(
            f"{TRAINING_LOG_NAME} final token count disagrees with result.json"
        )
    loss = training.series("train_loss")
    if loss is None:
        raise ReportError(f"{TRAINING_LOG_NAME} does not record train_loss")
    if not _close(float(loss[-1]), final_train_loss):
        raise ReportError(f"{TRAINING_LOG_NAME} final loss disagrees with result.json")

    flop_source = "traced" if training.flops_per_token > 1.0 else "unavailable"
    validation_path = _artifact_path(
        path, result, "validation_curve", "validation.csv", required=False
    )
    if validation_path is not None:
        _check_record_artifact(validation_path, path, record, "validation_curve")
        validation = _read_validation(validation_path, training)
    else:
        validation = []
    _validate_and_fill_evaluations(validation, result, training)

    run = _Run(
        run_id=run_id,
        result=result,
        record=record,
        training=training,
        validation=validation,
        flop_source=flop_source,
    )
    diagnostics_path = _artifact_path(
        path, result, "diagnostics", DIAGNOSTICS_LOG_NAME, required=False
    )
    if diagnostics_path is not None:
        _check_record_artifact(diagnostics_path, path, record, "diagnostics")
        run.diagnostics = _read_diagnostics(diagnostics_path, training)
    else:
        run.notices.append(
            f"{run_id}: {DIAGNOSTICS_LOG_NAME} was not recorded; "
            "diagnostic plots are unavailable."
        )
    if record is None:
        run.notices.append(
            f"{run_id}: not present in records.jsonl (shown as unledgered)."
        )
    elif "qualified" in record and not isinstance(record["qualified"], bool):
        run.notices.append(f"{run_id}: invalid ledger qualified flag was ignored.")
    # Long-form diagnostics already contain final param/grad/update scope values.
    # Avoid hashing and scanning a large checkpoint when that artifact is present.
    if run.diagnostics is None:
        _read_checkpoint_layers(path, run)
    # Store the limit on the transient object without widening the public payload.
    run.result["_report_point_limit"] = limit
    return run


def _read_diagnostics(path: Path, training: logpack.Log) -> logpack.Log:
    """Read the optional diagnostics log and check its structure, not its rows.

    Under the long-form CSV every one of a million-plus rows had to be parsed
    and cross-checked against the training curve. The packed log carries its
    column identities in a table and its axes in the header, so the same
    guarantees now cost a handful of comparisons: the grid is complete because
    the columns are fixed, and the axes agree because both files declare them.
    """

    try:
        log = logpack.read_log(path)
    except logpack.LogError as error:
        raise ReportError(str(error)) from error

    if log.tokens_per_step != training.tokens_per_step:
        raise ReportError(
            f"{DIAGNOSTICS_LOG_NAME} token accounting disagrees with "
            f"{TRAINING_LOG_NAME}"
        )
    if not _close(log.flops_per_token, training.flops_per_token):
        raise ReportError(
            f"{DIAGNOSTICS_LOG_NAME} FLOP accounting disagrees with {TRAINING_LOG_NAME}"
        )
    if len(log) == 0:
        raise ReportError(f"{DIAGNOSTICS_LOG_NAME} has no samples")
    if np.any(np.diff(log.steps) <= 0):
        raise ReportError(f"{DIAGNOSTICS_LOG_NAME} steps must increase")
    if int(log.steps[0]) != 1:
        raise ReportError(f"{DIAGNOSTICS_LOG_NAME} must include optimizer step 1")
    final_step = int(training.steps[-1])
    if int(log.steps[-1]) != final_step:
        raise ReportError(
            f"{DIAGNOSTICS_LOG_NAME} does not contain the final training step"
        )
    if int(log.steps[-1]) > final_step:
        raise ReportError(f"{DIAGNOSTICS_LOG_NAME} runs past the training curve")

    _check_diagnostic_scopes(log)
    return log


def _check_diagnostic_scopes(log: logpack.Log) -> None:
    """Require the scope layout the layer charts assume."""

    layers_by_scope: dict[int, set[int]] = {}
    for entry in log.columns:
        layers_by_scope.setdefault(entry.scope_id, set()).add(entry.layer)
    names = {
        scope.name
        for scope_id in layers_by_scope
        if (scope := metrics_registry.scope_by_id(scope_id)) is not None
    }
    if not {"overall", "embeddings", "final_norm"} <= names:
        raise ReportError(
            f"{DIAGNOSTICS_LOG_NAME} is missing an overall, embeddings, or "
            "final_norm scope"
        )
    block = metrics_registry.scope("block").id
    if block in layers_by_scope:
        layers = sorted(layers_by_scope[block])
        if layers != list(range(len(layers))):
            raise ReportError(
                f"{DIAGNOSTICS_LOG_NAME} block scopes must be contiguous and "
                "start at layer 0"
            )

    expert = metrics_registry.scope("expert").id
    if expert in layers_by_scope:
        indexes_by_layer: dict[int, set[int]] = {}
        for entry in log.columns:
            if entry.scope_id == expert:
                indexes_by_layer.setdefault(entry.layer, set()).add(entry.index)
        block_layers = layers_by_scope.get(block, set())
        if not set(indexes_by_layer) <= block_layers:
            raise ReportError(
                f"{DIAGNOSTICS_LOG_NAME} expert scopes must belong to a block"
            )
        for layer, indexes in indexes_by_layer.items():
            ordered = sorted(indexes)
            if ordered != list(range(len(ordered))):
                raise ReportError(
                    f"{DIAGNOSTICS_LOG_NAME} block {layer} expert scopes must be "
                    "contiguous and start at expert 0"
                )

    # The overall scope covers the whole model, so its element count has to be
    # the sum of the disjoint scopes that partition it. Expert scopes overlap
    # their parent blocks and are intentionally excluded from this sum.
    overall = metrics_registry.scope("overall").id
    partition_scopes = {
        metrics_registry.scope(name).id
        for name in ("embeddings", "unembedding", "block", "final_norm")
    }
    counts: dict[tuple[int, int, int], int] = {}
    for entry in log.columns:
        counts.setdefault(
            (entry.scope_id, entry.layer, entry.index), entry.element_count
        )
    total = counts.get((overall, -1, -1))
    parts = sum(
        value
        for (scope_id, _, _), value in counts.items()
        if scope_id in partition_scopes
    )
    if total is not None and parts and total != parts:
        raise ReportError(
            f"{DIAGNOSTICS_LOG_NAME} overall element_count disagrees with its "
            "model scopes"
        )


def _check_identity(
    result: Mapping[str, Any], record: Mapping[str, Any] | None, run_id: str
) -> None:
    for name in ("track", "profile"):
        if not isinstance(result.get(name), str) or not result[name]:
            raise ReportError(f"result {name} is missing")
    if isinstance(result.get("seed"), bool) or not isinstance(result.get("seed"), int):
        raise ReportError("result seed is invalid")
    if record is None:
        return
    if record.get("run_id") != run_id or record.get("status") != "ok":
        raise ReportError("ledger identity/status does not match the run directory")
    for name in ("track", "profile", "seed"):
        if record.get(name) != result.get(name):
            raise ReportError(f"ledger {name} disagrees with result.json")
    result_metrics = _object(result.get("metrics"), "result metrics")
    record_metrics = _object(record.get("metrics"), "ledger metrics")
    for name in ("tokens_processed", "train_seconds", "validation_loss"):
        left = result_metrics.get(name)
        right = record_metrics.get(name)
        if name == "tokens_processed":
            if left != right:
                raise ReportError(f"ledger {name} disagrees with result.json")
        elif not _close(_finite(left, name), _finite(right, name)):
            raise ReportError(f"ledger {name} disagrees with result.json")


def _artifact_path(
    run_dir: Path,
    result: Mapping[str, Any],
    artifact_name: str,
    fallback: str,
    *,
    required: bool,
) -> Path | None:
    artifacts = result.get("artifacts", {})
    if not isinstance(artifacts, dict):
        raise ReportError("result artifacts is not an object")
    declared = artifacts.get(artifact_name)
    name = declared if declared is not None else fallback
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ReportError(f"artifact {artifact_name} has an unsafe path")
    unresolved = run_dir / name
    try:
        resolved = unresolved.resolve(strict=False)
        resolved.relative_to(run_dir.resolve())
    except (OSError, ValueError) as exc:
        raise ReportError(
            f"artifact {artifact_name} escapes its run directory"
        ) from exc
    if not resolved.exists():
        if declared is not None or required:
            raise ReportError(f"artifact {artifact_name} is missing")
        return None
    if not resolved.is_file() or unresolved.is_symlink():
        raise ReportError(f"artifact {artifact_name} is not a regular file")
    if resolved.stat().st_size > _MAX_ARTIFACT_BYTES:
        raise ReportError(f"artifact {artifact_name} is implausibly large")
    return resolved


def _regular_file(run_dir: Path, name: str) -> Path:
    path = run_dir / name
    if not path.is_file() or path.is_symlink():
        raise ReportError(f"{name} is not a regular file")
    return path


def _check_record_artifact(
    path: Path, run_dir: Path, record: Mapping[str, Any] | None, name: str
) -> None:
    if record is None:
        return
    artifacts = _object(record.get("artifacts", {}), "ledger artifacts")
    value = artifacts.get(name)
    if not isinstance(value, dict):
        raise ReportError(f"ledger is missing the {name} artifact")
    try:
        relative = path.relative_to(run_dir.resolve()).as_posix()
    except ValueError as exc:
        raise ReportError(f"{name} is outside its run directory") from exc
    if value.get("path") != relative:
        raise ReportError(f"ledger path for {name} disagrees with result.json")
    if value.get("bytes") != path.stat().st_size:
        raise ReportError(f"ledger byte count for {name} no longer matches")
    expected = value.get("sha256")
    if not isinstance(expected, str) or _sha256(path) != expected:
        raise ReportError(f"ledger SHA-256 for {name} no longer matches")


def _read_training(path: Path) -> logpack.Log:
    """Read the training log and check that it covers every optimizer step."""

    try:
        log = logpack.read_log(path)
    except logpack.LogError as error:
        raise ReportError(str(error)) from error
    if len(log) == 0:
        raise ReportError(f"{TRAINING_LOG_NAME} has no samples")
    expected = np.arange(1, len(log) + 1, dtype=log.steps.dtype)
    if not np.array_equal(log.steps, expected):
        raise ReportError(
            f"{TRAINING_LOG_NAME} must contain exactly one sample for every "
            "optimizer step starting at 1"
        )
    loss = log.series("train_loss")
    if loss is None or not np.all(np.isfinite(loss)):
        raise ReportError(f"{TRAINING_LOG_NAME} train_loss must be finite")
    return log


def _read_validation(path: Path, training: logpack.Log) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        if not fields or len(fields) != len(set(fields)):
            raise ReportError("validation.csv has a missing or duplicate header")
        missing = [name for name in _VALIDATION_REQUIRED if name not in fields]
        if missing:
            raise ReportError("validation.csv is missing: " + ", ".join(missing))
        raw_rows = list(reader)
    rows: list[dict[str, Any]] = []
    prior_step = 0
    final_step = int(training.steps[-1])
    for number, raw in enumerate(raw_rows, 2):
        step = _positive_int_text(raw.get("step"), f"validation.csv:{number} step")
        tokens = _positive_int_text(
            raw.get("tokens_processed"), f"validation.csv:{number} tokens"
        )
        if step < prior_step or step > final_step:
            raise ReportError(
                "validation.csv steps must be ordered and within training"
            )
        if step * training.tokens_per_step != tokens:
            raise ReportError(
                f"validation.csv:{number} token count disagrees with "
                f"{TRAINING_LOG_NAME} step {step}"
            )
        kind, domain = raw.get("kind", "").strip(), raw.get("domain", "").strip()
        if not kind or not domain:
            raise ReportError(f"validation.csv:{number} kind/domain is empty")
        row = {
            "step": step,
            "tokens_processed": tokens,
            "kind": kind,
            "domain": domain,
            "validation_loss": _finite_text(
                raw.get("validation_loss"), f"validation.csv:{number} validation_loss"
            ),
            "estimated_flops": _flops_at_step(training, step),
        }
        rows.append(row)
        prior_step = step
    return rows


def _flops_at_step(training: logpack.Log, step: int) -> float:
    """Cumulative FLOPs at any step, including one between recorded samples.

    The axis is exactly linear in the step -- every step costs the same batch --
    so what used to be a walk with interpolation is now a multiply.
    """

    return float(step) * training.tokens_per_step * training.flops_per_token


def _validate_and_fill_evaluations(
    rows: list[dict[str, Any]],
    result: Mapping[str, Any],
    training: logpack.Log,
) -> None:
    metrics = _object(result.get("metrics"), "result metrics")
    validation_loss = _finite(metrics.get("validation_loss"), "validation_loss")
    final_step = int(training.steps[-1])
    final_tokens = int(training.axis("tokens_processed")[-1])
    final_flops = _flops_at_step(training, final_step)
    fineweb_final = [
        row
        for row in rows
        if row["kind"] == "fineweb"
        or (row["domain"] == "fineweb" and str(row["kind"]).endswith("final"))
    ]
    if fineweb_final and not _close(
        fineweb_final[-1]["validation_loss"], validation_loss
    ):
        raise ReportError(
            "validation.csv final FineWeb loss disagrees with result.json"
        )
    if not fineweb_final:
        rows.append(
            {
                "step": final_step,
                "tokens_processed": final_tokens,
                "kind": "fineweb",
                "domain": "fineweb",
                "validation_loss": validation_loss,
                "estimated_flops": final_flops,
                "synthetic": True,
            }
        )

    evaluations = result.get("evaluations")
    if evaluations is None:
        return
    evaluations = _object(evaluations, "evaluations")
    evaluated_fineweb = evaluations.get("fineweb")
    if evaluated_fineweb is not None:
        evaluated_loss = _finite(
            _object(evaluated_fineweb, "evaluations.fineweb").get("loss"),
            "evaluations.fineweb.loss",
        )
        if not _close(evaluated_loss, validation_loss):
            raise ReportError("evaluations.fineweb loss disagrees with result metrics")
    fresh = evaluations.get("fresh10")
    if fresh is None:
        return
    fresh = _object(fresh, "evaluations.fresh10")
    macro = _finite(fresh.get("macro_loss"), "Fresh10 macro_loss")
    macro_rows = [row for row in rows if row["kind"] == "downstream_macro"]
    if macro_rows and not _close(macro_rows[-1]["validation_loss"], macro):
        raise ReportError(
            "validation.csv Fresh10 macro loss disagrees with result.json"
        )
    if not macro_rows:
        rows.append(
            {
                "step": final_step,
                "tokens_processed": final_tokens,
                "kind": "downstream_macro",
                "domain": "fresh10_macro",
                "validation_loss": macro,
                "estimated_flops": final_flops,
                "synthetic": True,
            }
        )
    domains = _object(fresh.get("domains"), "evaluations.fresh10.domains")
    if len(domains) != 10:
        raise ReportError("evaluations.fresh10 must contain exactly ten domains")
    domain_losses = [
        _finite(_object(value, f"Fresh10 {domain}").get("loss"), f"Fresh10 {domain}")
        for domain, value in domains.items()
    ]
    if not _close(macro, math.fsum(domain_losses) / len(domain_losses)):
        raise ReportError("Fresh10 macro loss disagrees with its ten domain losses")
    for domain, value in domains.items():
        expected = _finite(
            _object(value, f"Fresh10 {domain}").get("loss"), f"Fresh10 {domain}"
        )
        matches = [
            row
            for row in rows
            if row["kind"] == "downstream" and row["domain"] == domain
        ]
        if matches and not _close(matches[-1]["validation_loss"], expected):
            raise ReportError(
                f"validation.csv Fresh10 {domain} loss disagrees with result.json"
            )
        if not matches:
            rows.append(
                {
                    "step": final_step,
                    "tokens_processed": final_tokens,
                    "kind": "downstream",
                    "domain": domain,
                    "validation_loss": expected,
                    "estimated_flops": final_flops,
                    "synthetic": True,
                }
            )


def _read_checkpoint_layers(run_dir: Path, run: _Run) -> None:
    checkpoint_name = run.result.get("checkpoint")
    if not isinstance(checkpoint_name, str) or not checkpoint_name:
        run.notices.append(f"{run.run_id}: result has no usable checkpoint path.")
        return
    try:
        checkpoint = _contained_optional(run_dir, checkpoint_name)
    except ReportError as exc:
        run.notices.append(f"{run.run_id}: checkpoint ignored ({exc}).")
        return
    if checkpoint is None:
        retained = (
            run.record.get("checkpoint", {}).get("retained")
            if isinstance(run.record, dict)
            else None
        )
        policy = (
            "retention policy removed it" if retained is False else "file is absent"
        )
        run.notices.append(f"{run.run_id}: per-layer plots unavailable ({policy}).")
        return

    if run.record is not None:
        ledger = run.record.get("checkpoint")
        if not isinstance(ledger, dict):
            run.notices.append(
                f"{run.run_id}: checkpoint lacks a ledger record; layer scan skipped."
            )
            return
        if ledger.get("bytes") != checkpoint.stat().st_size:
            run.notices.append(
                f"{run.run_id}: checkpoint size differs from ledger; layer scan skipped."
            )
            return
        expected = ledger.get("sha256")
        if not isinstance(expected, str) or _sha256(checkpoint) != expected:
            run.notices.append(
                f"{run.run_id}: checkpoint SHA-256 differs; layer scan skipped."
            )
            return
    try:
        run.layer_stats = _checkpoint_layer_stats(checkpoint)
    except (OSError, ValueError, TypeError, ImportError) as exc:
        run.notices.append(f"{run.run_id}: checkpoint layer scan failed ({exc}).")
        return
    if not run.layer_stats:
        run.notices.append(
            f"{run.run_id}: checkpoint contains no recognized logical layer arrays."
        )


def _contained_optional(run_dir: Path, relative: str) -> Path | None:
    candidate = Path(relative)
    if candidate.is_absolute():
        raise ReportError("absolute checkpoint path")
    unresolved = run_dir / candidate
    resolved = unresolved.resolve(strict=False)
    try:
        resolved.relative_to(run_dir.resolve())
    except ValueError as exc:
        raise ReportError("checkpoint path escapes run directory") from exc
    if not resolved.exists():
        return None
    if not resolved.is_file() or unresolved.is_symlink():
        raise ReportError("checkpoint is not a regular file")
    return resolved


def _checkpoint_layer_stats(path: Path) -> dict[str, list[dict[str, float]]]:
    # NumPy is deliberately imported lazily: `rig report` must not import JAX
    # or initialize a TPU, and ordinary CLI commands should remain instant.
    import numpy as np

    # Streaming raw sums avoid materializing multiple full-size power arrays.
    # With current reference checkpoints this keeps peak host memory near one
    # float64 copy of the largest tensor, rather than loading the whole NPZ.
    accumulators: dict[tuple[str, int], list[float]] = {}
    chunk_elements = 1_000_000
    with np.load(path, allow_pickle=False) as archive:
        for key in archive.files:
            identity = _layer_identity(key)
            if identity is None:
                continue
            array = archive[key]
            numeric = (
                np.issubdtype(array.dtype, np.number) or array.dtype.name == "bfloat16"
            )
            if not numeric or array.size == 0:
                continue
            # count, sum(|x|), sum(x^2), sum(x), sum(x^3), sum(x^4), min, max
            acc = accumulators.setdefault(
                identity,
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, math.inf, -math.inf],
            )
            flattened = array.reshape(-1)
            for start in range(0, flattened.size, chunk_elements):
                values = np.asarray(
                    flattened[start : start + chunk_elements], dtype=np.float64
                )
                if not bool(np.all(np.isfinite(values))):
                    raise ValueError(f"non-finite layer array {key!r}")
                acc[0] += values.size
                acc[1] += float(np.sum(np.abs(values), dtype=np.float64))
                squared = values * values
                acc[2] += float(np.sum(squared, dtype=np.float64))
                acc[3] += float(np.sum(values, dtype=np.float64))
                acc[4] += float(np.sum(squared * values, dtype=np.float64))
                acc[5] += float(np.sum(squared * squared, dtype=np.float64))
                acc[6] = min(acc[6], float(np.min(values)))
                acc[7] = max(acc[7], float(np.max(values)))

    result: dict[str, list[dict[str, float]]] = {}
    for (family, layer), acc in sorted(accumulators.items()):
        count, l1, sum2, total, sum3, sum4, minimum, maximum = acc
        mean = total / count
        raw2, raw3, raw4 = sum2 / count, sum3 / count, sum4 / count
        variance = max(0.0, raw2 - mean * mean)
        third = raw3 - 3.0 * mean * raw2 + 2.0 * mean**3
        fourth = raw4 - 4.0 * mean * raw3 + 6.0 * mean * mean * raw2 - 3.0 * mean**4
        result.setdefault(family, []).append(
            {
                "layer": float(layer),
                "l1_norm": l1,
                "l2_norm": math.sqrt(max(0.0, sum2)),
                "mean": mean,
                "std": math.sqrt(variance),
                "third_moment": third,
                "fourth_moment": max(0.0, fourth),
                "min": minimum,
                "max": maximum,
            }
        )
    return result


def _layer_identity(key: str) -> tuple[str, int] | None:
    normalized = key.lower().replace(".", "/")
    first = normalized.split("/", 1)[0]
    family = {
        "params": "param",
        "parameters": "param",
        "param": "param",
        "grads": "grad",
        "gradients": "grad",
        "grad": "grad",
        "updates": "update",
        "update": "update",
    }.get(first)
    match = _LAYER_KEY.search(normalized)
    return (family, int(match.group(1))) if family is not None and match else None


def _report_payload(
    runs: Sequence[_Run],
    skipped: Mapping[str, str],
    notices: Sequence[str],
    *,
    max_chart_points: int,
    layer_snapshots: int = _MAX_LAYER_SNAPSHOTS,
) -> dict[str, Any]:
    run_rows: list[dict[str, Any]] = []
    for index, run in enumerate(runs):
        result = run.result
        metrics = _object(result["metrics"], "result metrics")
        evaluation = result.get("evaluations", {})
        fresh = evaluation.get("fresh10") if isinstance(evaluation, dict) else None
        fresh_loss = fresh.get("macro_loss") if isinstance(fresh, dict) else None
        recipe = (
            run.record.get("recipe")
            if isinstance(run.record, dict)
            and isinstance(run.record.get("recipe"), str)
            else _recipe_from_run_id(run.run_id)
        )
        # Records began carrying a bare "recipe" field, which then won over the
        # run-id fallback that had been supplying the folded-in run name. The
        # name is the whole point of naming a run, so take it from the record
        # where it actually lives rather than re-parsing the directory.
        run_name = run.record.get("name") if isinstance(run.record, dict) else None
        if isinstance(run_name, str) and run_name and run_name not in recipe:
            recipe = f"{recipe}-{run_name}"
        _, classification = _default_run_selection(str(result["profile"]))
        qualified = run.record.get("qualified") if run.record else None
        if not isinstance(qualified, bool):
            qualified = None
        run_rows.append(
            {
                "id": run.run_id,
                "recipe": recipe,
                "label": f"{recipe} · {result['profile']} · {run.run_id[:15]}",
                "track": result["track"],
                "profile": result["profile"],
                "classification": classification,
                "selected": True,
                "seed": result["seed"],
                "color": _COLORS[index % len(_COLORS)],
                "ledger": run.record is not None,
                "flopSource": run.flop_source,
                "finalStep": run.final_step,
                # Topology, because it changes the number. The data stream is
                # invariant under it -- the same seed draws the same global
                # batches on any process count -- but gradients reduce across a
                # different number of devices and each chip holds a different
                # share of the batch, and the same run has landed 0.004-0.023
                # nats apart across two slices for that reason alone.
                "chip": _hardware(result).get("chip"),
                "processes": _hardware(result).get("processes"),
                "devices": _hardware(result).get("devices"),
                "tokens": int(metrics["tokens_processed"]),
                "trainSeconds": metrics.get("train_seconds"),
                "trainLoss": metrics.get("train_loss"),
                "validationLoss": metrics.get("validation_loss"),
                "fresh10Loss": fresh_loss,
                "qualified": qualified,
                "hasLayerStats": bool(run.layer_stats) or run.has_layer_diagnostics(),
                "riglogs": sorted(
                    {
                        artifact
                        for artifact in (result.get("artifacts") or {}).values()
                        if isinstance(artifact, str)
                        and artifact.endswith(logpack.SUFFIX)
                        and "/" not in artifact
                        and "\\" not in artifact
                    }
                ),
            }
        )

    time_charts: list[dict[str, Any]] = []
    time_charts.append(_training_chart(runs, "train_loss", "Training loss", "loss"))
    time_charts.append(_validation_chart(runs, "fineweb", "FineWeb validation loss"))
    time_charts.append(_validation_chart(runs, "fresh10_macro", "Fresh10 macro loss"))
    domains = list(_FRESH10_ORDER)
    discovered = sorted(
        {
            str(row["domain"])
            for run in runs
            for row in run.validation
            if row["kind"] == "downstream"
        }
        - set(domains)
    )
    domains.extend(discovered)
    for domain in domains:
        chart = _validation_chart(runs, domain, f"Fresh10 · {_humanize(domain)}")
        if chart["series"]:
            time_charts.append(chart)
    time_charts.append(_training_chart(runs, "learning_rate", "Learning rate", "rate"))
    # Routing, for mixture-of-experts runs. Every one of these is empty for a
    # dense run and dropped by the filter below, so adding them costs those
    # runs nothing. Without them a routed report shows a loss curve and no
    # sign of whether the router was doing anything, which is the one question
    # a routed run exists to answer.
    for spec in _ROUTER_CHARTS:
        if spec.kind != "line":
            # No other kind is rendered yet; drawing it as a timeline would be
            # worse than not drawing it, because it would look correct.
            continue
        time_charts.append(_training_chart(runs, spec.metric, spec.title, spec.unit))

    time_charts = [chart for chart in time_charts if chart["series"]]

    overall_scope = metrics_registry.scope("overall").id
    recorded_overall = {
        (entry.metric.family, entry.metric.stat)
        for run in runs
        if run.diagnostics is not None
        for entry in run.diagnostics.columns
        if entry.scope_id == overall_scope
        and entry.metric is not None
        and entry.metric.family is not None
    }
    requested_overall = [
        (family, statistic)
        for family in ("grad", "update", "param")
        for statistic in (
            "l1_norm",
            "l2_norm",
            "mean",
            "std",
            "third_moment",
            "fourth_moment",
        )
    ]
    missing_overall = [
        _overall_metric_label(identity)
        for identity in requested_overall
        if identity not in recorded_overall
    ]
    recorded_labels = [
        _overall_metric_label(identity)
        for identity in requested_overall
        if identity in recorded_overall
    ]
    notices = list(notices) + [
        "Overall training diagnostic coverage: recorded "
        + (", ".join(recorded_labels) if recorded_labels else "none")
        + "; not recorded: "
        + (", ".join(missing_overall) if missing_overall else "none")
        + "."
    ]

    diagnostic_charts = _overall_diagnostic_charts(runs)
    layer_charts = _final_diagnostic_charts(runs, layer_snapshots)

    missing_final_families = []
    for family in _DIAGNOSTIC_FAMILIES:
        if not any(chart["family"] == family for chart in layer_charts):
            missing_final_families.append(family)
    if missing_final_families:
        notices = list(notices) + [
            "Final-checkpoint "
            + "/".join(missing_final_families)
            + " scope metrics are unavailable; they appear when diagnostics.csv is recorded "
            "or a retained checkpoint contains those arrays."
        ]

    return {
        "meta": {
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "included": len(runs),
            "skipped": len(skipped),
            "defaultXAxis": "flops",
            "defaultXScale": "log",
            "flopsLabel": "Estimated cumulative FLOPs",
            "maxChartPoints": max_chart_points,
            # The chart budget uses 0 as the sentinel for an unthinned payload,
            # but a browser control always needs a positive upper bound. Keep
            # those meanings separate so full-resolution reports do not clamp
            # their smoothing span to one sample.
            "smoothingMaxSamples": max_chart_points or _MAX_CHART_POINTS,
            "profile": "official",
        },
        "runs": run_rows,
        "timeCharts": time_charts,
        "diagnosticCharts": diagnostic_charts,
        "layerCharts": layer_charts,
        "notices": list(dict.fromkeys(notices)),
        "skipped": [{"run": key, "reason": value} for key, value in skipped.items()],
    }


@dataclass(frozen=True)
class SeriesChart:
    """One model-wide series to plot, and how to plot it.

    The registry below is meant to grow. A metric added to
    ``rig/metrics.py`` records itself into every artifact automatically, but
    nothing plots it until it appears here -- which is deliberate, because how
    a quantity should be drawn is a judgement the registry has no way to make.

    ``kind`` carries that judgement. Everything so far is a line against the
    time axis; a quantity that is a distribution rather than a scalar -- a
    routing histogram, say -- wants bars against expert index, with counts on
    the other axis, and would arrive here as a new kind rather than being bent
    into a timeline. A chart whose metric no run recorded is dropped before the
    payload is written, so declaring one costs nothing to runs without it.
    """

    metric: str
    title: str
    unit: str
    kind: str = "line"


# Model-wide routing series, in the order they are worth reading: is the load
# even, is the router deciding, and is anything running away.
_ROUTER_CHARTS = (
    SeriesChart("router.balance_loss", "Router balance loss", "1.0 even, E collapsed"),
    SeriesChart("router.max_load", "Busiest expert's share", "fraction of assignments"),
    SeriesChart("router.min_load", "Idlest expert's share", "fraction of assignments"),
    SeriesChart("router.entropy", "Routing entropy", "nats; ln(E) is no preference"),
    SeriesChart("router.top1_gate", "Mean top-1 gate weight", "weight"),
    SeriesChart("router.logit_rms", "Router logit RMS", "RMS"),
)


def _hardware(result: Mapping[str, Any]) -> dict[str, Any]:
    """Chip kind, process count, and device count, when the run recorded them."""

    system = result.get("system")
    if not isinstance(system, dict):
        return {}
    kinds = system.get("device_kinds")
    chip = kinds[0] if isinstance(kinds, list) and kinds else None
    return {
        "chip": chip if isinstance(chip, str) else None,
        "processes": system.get("process_count"),
        "devices": system.get("device_count"),
    }


def _training_chart(
    runs: Sequence[_Run], metric: str, title: str, y_label: str
) -> dict[str, Any]:
    series: list[dict[str, Any]] = []
    for run in runs:
        points = run.training_points(metric)
        if points:
            limit = int(run.result.get("_report_point_limit", _MAX_CHART_POINTS))
            # The report opens in equi-FLOP mode, so point selection must preserve
            # shape in that coordinate rather than silently optimizing for steps.
            points = _lttb(points, limit, x_index=1)
            series.append({"run": run.run_id, "points": points})
    return {
        "key": metric,
        "title": title,
        "yLabel": y_label,
        "series": series,
    }


def _overall_diagnostic_charts(runs: Sequence[_Run]) -> list[dict[str, Any]]:
    """Build one timeline chart per family/statistic pair.

    Long-form diagnostics are authoritative.  The legacy scalar training columns
    remain a read-only fallback so older official artifacts do not lose the one
    gradient norm they already recorded.
    """

    charts: list[dict[str, Any]] = []
    for family in _DIAGNOSTIC_FAMILIES:
        for statistic in _DIAGNOSTIC_STATS:
            series: list[dict[str, Any]] = []
            for run in runs:
                points = run.diagnostic_points(family, statistic)
                if points:
                    limit = int(
                        run.result.get("_report_point_limit", _MAX_CHART_POINTS)
                    )
                    series.append(
                        {"run": run.run_id, "points": _lttb(points, limit, x_index=1)}
                    )
            if series:
                charts.append(
                    {
                        "key": f"overall_{family}_{statistic}",
                        "family": family,
                        "stat": statistic,
                        "title": _humanize(statistic),
                        "yLabel": _metric_unit(statistic),
                        "series": series,
                    }
                )
    return charts


def _compact(value: Any) -> Any:
    """Trim a diagnostic value to float32's worth of digits.

    These come off the accelerator as float32, so a full double repr spends
    roughly seventeen characters to encode about seven meaningful ones. Across a
    step x scope grid that is most of the embedded payload.
    """

    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    if not math.isfinite(value):
        return None
    return float(f"{value:.7g}")


def _final_diagnostic_charts(
    runs: Sequence[_Run], snapshots: int = _MAX_LAYER_SNAPSHOTS
) -> list[dict[str, Any]]:
    """Build per-scope diagnostic snapshots the viewer can scrub through by step.

    Each series carries a fixed scope layout plus one value row per retained
    step, so the client can render any recorded step without re-deriving the
    layout. The final step is always present, which keeps the historical
    final-snapshot behaviour available as the default.
    """

    charts: list[dict[str, Any]] = []
    for family in _DIAGNOSTIC_FAMILIES:
        for statistic in _DIAGNOSTIC_STATS:
            series: list[dict[str, Any]] = []
            for run in runs:
                layout = _diagnostic_scope_layout(run.diagnostics, family, statistic)
                if layout:
                    # The column table fixes the scope layout for the whole run,
                    # so every frame aligns without per-step reconciliation.
                    assert run.diagnostics is not None
                    keep = _subsample_indices(len(run.diagnostics), snapshots)
                    order = [position for position, _, _ in layout]
                    frame = run.diagnostics.values[
                        np.ix_(keep, [index for _, _, index in layout])
                    ]
                    series.append(
                        {
                            "run": run.run_id,
                            "scopes": [
                                [position, label] for position, label, _ in layout
                            ],
                            "steps": [
                                int(run.diagnostics.steps[index]) for index in keep
                            ],
                            "values": [
                                [_compact(float(value)) for value in row]
                                for row in frame
                            ],
                        }
                    )
                    continue
                # Checkpoint-derived arrays are a useful fallback for runs
                # without diagnostics; the log wins whenever both exist.
                legacy = [
                    [int(row["layer"]), row[statistic], f"block {int(row['layer'])}"]
                    for row in run.layer_stats.get(family, [])
                    if statistic in row
                ]
                if legacy:
                    legacy.sort(key=lambda point: point[0])
                    series.append(
                        {
                            "run": run.run_id,
                            "scopes": [[point[0], point[2]] for point in legacy],
                            "steps": [run.final_step],
                            "values": [[_compact(point[1]) for point in legacy]],
                        }
                    )
            if series:
                charts.append(
                    {
                        "key": f"layer_{family}_{statistic}",
                        "family": family,
                        "stat": statistic,
                        "title": _humanize(statistic),
                        "yLabel": _metric_unit(statistic),
                        "series": series,
                    }
                )
    return charts


def _diagnostic_scope_layout(
    log: logpack.Log | None, family: str, statistic: str
) -> list[tuple[int, str, int]]:
    """Order one statistic's per-scope columns along the model axis.

    Returns ``(position, label, column index)`` sorted by position, with the
    embeddings pinned before layer 0 and the trailing scopes after the last
    block, so the layer chart reads left to right through the network.
    """

    if log is None:
        return []
    wanted = metrics_registry.metric(f"{family}.{statistic}").id
    block = metrics_registry.scope("block").id
    selected = [
        (index, entry)
        for index, entry in enumerate(log.columns)
        if entry.metric_id == wanted
        and entry.scope_id != metrics_registry.scope("overall").id
    ]
    layers = [entry.layer for _, entry in selected if entry.scope_id == block]
    final_position = (max(layers) + 1) if layers else 1
    trailing = {
        "embeddings": (-1, "embeddings"),
        "final_norm": (final_position, "final norm"),
        "unembedding": (final_position + 1, "unembedding"),
    }
    layout: list[tuple[int, str, int]] = []
    for index, entry in selected:
        if entry.scope_id == block:
            layout.append((entry.layer, f"block {entry.layer}", index))
            continue
        scope = metrics_registry.scope_by_id(entry.scope_id)
        if scope is not None and scope.name in trailing:
            position, label = trailing[scope.name]
            layout.append((position, label, index))
    layout.sort(key=lambda item: item[0])
    return layout


def _subsample_indices(total: int, limit: int) -> list[int]:
    """Evenly thin sample positions, always keeping the first and last."""

    if limit <= 0 or total <= limit:
        return list(range(total))
    picked = {0, total - 1}
    for index in range(limit):
        picked.add(round(index * (total - 1) / (limit - 1)))
    return sorted(picked)


def _validation_chart(
    runs: Sequence[_Run], identity: str, title: str
) -> dict[str, Any]:
    series = []
    for run in runs:
        if identity == "fineweb":
            selected = [
                row
                for row in run.validation
                if row["domain"] == "fineweb"
                and row["kind"] in {"fineweb_probe", "fineweb", "fineweb_final"}
            ]
        elif identity == "fresh10_macro":
            selected = [
                row for row in run.validation if row["kind"] == "downstream_macro"
            ]
        else:
            selected = [
                row
                for row in run.validation
                if row["kind"] == "downstream" and row["domain"] == identity
            ]
        if selected:
            series.append(
                {
                    "run": run.run_id,
                    "points": [
                        [row["step"], row["estimated_flops"], row["validation_loss"]]
                        for row in selected
                    ],
                }
            )
    return {
        "key": f"validation_{identity}",
        "title": title,
        "yLabel": "loss",
        "series": series,
    }


def _lttb(
    points: Sequence[Sequence[float]], threshold: int, *, x_index: int = 1
) -> list[list[float]]:
    """Largest-Triangle-Three-Buckets downsampling, preserving endpoints.

    Time-series rows are ``[step, cumulative FLOPs, value]`` and therefore use
    the report's default FLOP coordinate unless a caller explicitly chooses a
    different x column.
    """

    if threshold >= len(points) or threshold == 0:
        return [list(point) for point in points]
    sampled: list[list[float]] = [list(points[0])]
    every = (len(points) - 2) / (threshold - 2)
    anchor = 0
    for bucket in range(threshold - 2):
        average_start = int(math.floor((bucket + 1) * every)) + 1
        average_end = min(int(math.floor((bucket + 2) * every)) + 1, len(points))
        average_range = points[average_start:average_end]
        if average_range:
            average_x = math.fsum(point[x_index] for point in average_range) / len(
                average_range
            )
            average_y = math.fsum(point[2] for point in average_range) / len(
                average_range
            )
        else:
            average_x, average_y = points[-1][x_index], points[-1][2]
        range_start = int(math.floor(bucket * every)) + 1
        range_end = min(int(math.floor((bucket + 1) * every)) + 1, len(points) - 1)
        point_a = points[anchor]
        max_area = -1.0
        selected = range_start
        for index in range(range_start, max(range_start + 1, range_end)):
            point = points[index]
            area = abs(
                (point_a[x_index] - average_x) * (point[2] - point_a[2])
                - (point_a[x_index] - point[x_index]) * (average_y - point_a[2])
            )
            if area > max_area:
                max_area = area
                selected = index
        sampled.append(list(points[selected]))
        anchor = selected
    sampled.append(list(points[-1]))
    return sampled


def _diagnostic_metric(name: str) -> bool:
    lowered = name.lower()
    if lowered in {"grad_norm", "update_norm", "parameter_norm", "param_norm"}:
        return True
    family = any(token in lowered for token in ("grad", "update", "param"))
    statistic = any(
        token in lowered
        for token in (
            "norm",
            "mean",
            "std",
            "moment",
            "min",
            "max",
            "percentile",
            "hist",
            "skew",
            "kurt",
        )
    )
    return family and statistic


def _overall_metric_label(identity: tuple[str, str]) -> str:
    family, statistic = identity
    family_label = {"grad": "gradient", "update": "update", "param": "parameter"}[
        family
    ]
    statistic_label = {
        "l1_norm": "L1 norm",
        "l2_norm": "L2 norm",
        "mean": "mean",
        "std": "std",
        "third_moment": "third moment",
        "fourth_moment": "fourth moment",
    }[statistic]
    return f"{family_label} {statistic_label}"


def _default_run_selection(profile: str) -> tuple[bool, str]:
    """Return the initial visibility and an honest display classification.

    Everything starts visible: comparing runs is the point of the dashboard,
    and defaulting them to hidden behind a historical token budget is what made
    it render nothing. The classification still says what each run is, so an
    official result is never mistaken for a smoke test.
    """

    if profile == "official":
        return True, "official"
    if profile == "dev":
        return True, "diagnostic"
    return True, "smoke"


def _metric_unit(name: str) -> str:
    lowered = name.lower()
    if "loss" in lowered:
        return "loss"
    if "learning_rate" in lowered or lowered == "lr":
        return "rate"
    if "norm" in lowered:
        return "norm"
    return "value"


def _humanize(value: str) -> str:
    return value.replace("_", " ").replace("-", " ").strip().title()


def _recipe_from_run_id(run_id: str) -> str:
    # Run IDs are timestamp-recipe-random.  This is display-only; the ledger is
    # authoritative whenever it exists.
    match = re.match(r"^\d{8}T\d{6}\.\d+Z-(.+)-[0-9a-f]{8}$", run_id)
    return match.group(1) if match else run_id


def _study_run_name(result: Mapping[str, Any]) -> str:
    """Name a run by what varies, so a folder listing reads as a grid.

    Falls back to the run id when a field is missing: a directory that cannot
    be named descriptively should still be exported, under a name that is at
    least unique.
    """

    metrics = result.get("metrics") or {}
    steps = metrics.get("training_steps") or 0
    tokens = metrics.get("tokens_processed") or 0
    sequence = (result.get("contract") or {}).get("sequence_length") or 1024
    rate = metrics.get("base_learning_rate") or 0.0
    tier = metrics.get("model_tier")
    if not (steps and tokens and rate and tier):
        return ""
    batch = tokens // steps // sequence
    exponent = round(math.log2(rate))
    tpp = round(metrics.get("tokens_per_parameter") or 0)
    # Routing and the duration treatment are not derivable from the coordinates
    # above. A mixed study can therefore contain distinct runs at the same
    # tier, batch, rate, and seed; retain those distinctions in the archive
    # name so one run cannot be exported on top of another.
    routed = "-moe" if (metrics.get("experts") or 0) else ""
    model = (result.get("contract") or {}).get("model") or {}
    duration = (
        "-duration"
        if model.get("parameterization") == "completedp_duration_v1"
        else ""
    )
    return (
        f"{tier}{routed}{duration}-{tpp}tpp-bs{batch}-lr2e{exponent}"
        f"-s{result.get('seed')}"
    )


def export_study(
    runs_dir: Path, target: Path, study: str, *, select: str | None = None
) -> dict[str, Any]:
    """Lay a study out the way the dataset repository expects it.

    One folder per run under ``<target>/<study>/``, named for what varies
    rather than for a timestamp, plus the ledger and the two report payloads
    consumed by the study browser: a compact curves-only snapshot and a full
    view containing every recorded sample.

    The README is written empty on purpose. Nothing here can infer why a sweep
    was run or what it showed, and a plausible-looking generated description
    would be worse than a blank one: it would be read as though someone had
    checked it.
    """

    pattern = re.compile(select) if select else None
    destination = target / study
    destination.mkdir(parents=True, exist_ok=True)

    ledger: list[str] = []
    records = {}
    ledger_path = runs_dir / "records.jsonl"
    if ledger_path.is_file():
        for line in ledger_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entry = json.loads(line)
                records[entry.get("run_id")] = entry

    exported = 0
    for run in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        result_path = run / "result.json"
        if not result_path.is_file():
            continue
        if pattern and not pattern.search(run.name):
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "ok":
            continue
        name = _study_run_name(result) or run.name
        folder = destination / name
        folder.mkdir(parents=True, exist_ok=True)
        for artifact in (
            TRAINING_LOG_NAME,
            DIAGNOSTICS_LOG_NAME,
            "validation.csv",
            "result.json",
            "metrics.json",
        ):
            source = run / artifact
            if source.is_file():
                shutil.copy2(source, folder / artifact)
        entry = records.get(run.name)
        if entry is not None:
            ledger.append(json.dumps({**entry, "name": name, "folder": name}))
        exported += 1

    (destination / "records.jsonl").write_text(
        "".join(f"{line}\n" for line in ledger), encoding="utf-8"
    )
    (destination / "README.md").write_text("", encoding="utf-8")

    snapshot_bytes = 0
    full_bytes = 0
    if exported:
        with tempfile.TemporaryDirectory() as scratch:
            full_page = Path(scratch) / "full.html"
            build_report(
                destination,
                full_page,
                max_chart_points=0,
                layer_snapshots=0,
            )
            full_blob = _embedded_report_blob(full_page)
            (destination / "full.json.gz").write_bytes(full_blob)
            full_bytes = len(full_blob)

            snapshot_page = Path(scratch) / "snapshot.html"
            build_report(
                destination,
                snapshot_page,
                max_chart_points=200,
                layer_snapshots=4,
            )
            payload = json.loads(gzip.decompress(_embedded_report_blob(snapshot_page)))
            payload["diagnosticCharts"] = []
            payload["layerCharts"] = []
            snapshot_blob = gzip.compress(
                json.dumps(payload, separators=(",", ":")).encode("utf-8"),
                compresslevel=9,
                mtime=0,
            )
            (destination / "snapshot.json.gz").write_bytes(snapshot_blob)
            snapshot_bytes = len(snapshot_blob)

    total = sum(f.stat().st_size for f in destination.rglob("*") if f.is_file())
    return {
        "path": destination,
        "runs": exported,
        "ledgered": len(ledger),
        "bytes": total,
        "snapshot_bytes": snapshot_bytes,
        "full_bytes": full_bytes,
        "readme": destination / "README.md",
    }


def _embedded_report_blob(report: Path) -> bytes:
    """Extract the browser-consumable gzip payload from a rendered report."""

    match = re.search(
        r'<script type="application/gzip-base64" id="report-data">(.*?)</script>',
        report.read_text(encoding="utf-8"),
        re.S,
    )
    if match is None:
        raise ReportError(f"rendered report has no embedded payload: {report}")
    try:
        blob = base64.b64decode(match.group(1).strip(), validate=True)
    except ValueError as exc:
        raise ReportError(f"rendered report payload is invalid: {report}") from exc
    if not blob.startswith(b"\x1f\x8b"):
        raise ReportError(f"rendered report payload is not gzip data: {report}")
    return blob


def build_study_browser(
    output_path: Path, *, repo: str, studies: Sequence[Mapping[str, Any]]
) -> Path:
    """Write a dashboard that carries no data and fetches it on request.

    The committed reports are summaries; this page is the way to the logs
    behind them. It ships the same charting code with an empty payload, lists
    the studies, and fetches nothing until one is chosen -- then only that
    study's overview, a fraction of a megabyte. Every study also links to its
    raw files. When a study publishes a larger full report payload, that is a
    separate, labelled click that says how many megabytes it is about to pull,
    because on a metered connection a page that helpfully preloaded a hundred
    megabytes would be a hostile page.

    Every report payload it fetches is ordinary report JSON, so nothing here
    needs to understand the packed raw-log format. That keeps the browser and
    the writer from having to agree about anything but JSON.
    """

    payload = {
        "meta": {
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "included": 0,
            "skipped": 0,
            "defaultXAxis": "flops",
            "defaultXScale": "log",
            "flopsLabel": "Estimated cumulative FLOPs",
            "maxChartPoints": _MAX_CHART_POINTS or 1_400,
            "smoothingMaxSamples": _MAX_CHART_POINTS,
            "profile": "official",
        },
        "remote": {"repo": repo, "studies": [dict(entry) for entry in studies]},
        "runs": [],
        "timeCharts": [],
        "diagnosticCharts": [],
        "layerCharts": [],
        "notices": [],
        "skipped": [],
    }
    html = _render_html(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def _render_html(payload: Mapping[str, Any]) -> str:
    """Embed the payload as gzip, base64-encoded into a script text node.

    The payload is nearly the whole file -- the CSS/JS shell is under 40 KB --
    and it is numeric JSON, which gzip halves even after base64's 33% tax. The
    base64 alphabet also cannot contain the characters that would terminate a
    script element, so the escaping the plain-JSON path needed is unnecessary.
    """

    encoded = json.dumps(
        payload, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    # mtime=0 keeps an unchanged report byte-identical between builds.
    compressed = gzip.compress(encoded, compresslevel=9, mtime=0)
    return _HTML.replace(
        "__REPORT_DATA__", base64.b64encode(compressed).decode("ascii")
    )


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ReportError(f"{name} is not an object")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ReportError(f"{name} is not a positive integer")
    return value


def _positive_int_text(value: str | None, name: str) -> int:
    if value is None or not re.fullmatch(r"[1-9][0-9]*", value.strip()):
        raise ReportError(f"{name} is not a positive integer")
    return int(value)


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReportError(f"{name} is not numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ReportError(f"{name} is not finite")
    return number


def _positive_finite(value: Any, name: str) -> float:
    number = _finite(value, name)
    if number <= 0:
        raise ReportError(f"{name} must be greater than zero")
    return number


def _finite_text(value: str | None, name: str) -> float:
    if value is None:
        raise ReportError(f"{name} is missing")
    try:
        number = float(value)
    except ValueError as exc:
        raise ReportError(f"{name} is not numeric") from exc
    if not math.isfinite(number):
        raise ReportError(f"{name} is not finite")
    return number


def _close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-6, abs_tol=1e-9)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="dark">
<title>GPT TPU Rig Report</title>
<style>
:root{--bg:#080b12;--panel:#101521;--panel2:#151b29;--line:#273247;--text:#edf4ff;--muted:#91a0b8;--accent:#7dd3fc;--good:#86efac;--warn:#fde047;--bad:#fca5a5;--radius:14px;font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:var(--text);background:var(--bg);font-synthesis:none}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 75% -10%,#17233a 0,transparent 35rem),var(--bg);min-height:100vh}button,input{font:inherit}.shell{display:grid;grid-template-columns:280px minmax(0,1fr);min-height:100vh}.sidebar{position:sticky;top:0;height:100vh;overflow:auto;border-right:1px solid var(--line);background:rgba(10,14,23,.94);backdrop-filter:blur(16px);padding:20px}.brand{font-weight:800;letter-spacing:.02em;font-size:17px}.brand b{color:var(--accent)}.subtle{color:var(--muted);font-size:12px;line-height:1.5}.side-actions{display:flex;gap:7px;margin:18px 0 10px}.ghost{border:1px solid var(--line);color:var(--muted);background:var(--panel);border-radius:8px;padding:6px 9px;cursor:pointer}.ghost:hover{color:var(--text);border-color:#41516d}.search{width:100%;border:1px solid var(--line);background:#090d16;color:var(--text);padding:9px 10px;border-radius:9px;outline:none}.search:focus{border-color:var(--accent)}.run-list{display:grid;gap:7px;margin-top:12px}.run-toggle{display:grid;grid-template-columns:auto 10px 1fr;gap:9px;align-items:start;padding:9px;border:1px solid transparent;border-radius:10px;cursor:pointer}.run-toggle:hover{background:var(--panel);border-color:var(--line)}.run-toggle input{margin-top:3px;accent-color:var(--accent)}.dot{width:9px;height:9px;border-radius:50%;margin-top:4px;box-shadow:0 0 14px currentColor}.run-name{font-size:12px;font-weight:650;overflow-wrap:anywhere}.run-meta{display:block;color:var(--muted);font-size:10px;margin-top:3px}.main{min-width:0;padding:26px 28px 60px}.top{display:flex;justify-content:space-between;gap:20px;align-items:flex-start;margin-bottom:20px}.eyebrow{font-size:11px;letter-spacing:.16em;text-transform:uppercase;color:var(--accent);font-weight:750}.top h1{font-size:clamp(25px,3vw,42px);letter-spacing:-.04em;margin:5px 0 7px}.stats{display:flex;gap:8px;flex-wrap:wrap}.pill{border:1px solid var(--line);background:rgba(16,21,33,.8);border-radius:999px;padding:7px 10px;color:var(--muted);font-size:12px}.pill strong{color:var(--text)}.axis-control{flex:none;border:1px solid var(--line);border-radius:11px;padding:4px;background:#090d16;display:flex}.axis-control label{cursor:pointer}.axis-control input{position:absolute;opacity:0;pointer-events:none}.axis-control span{display:block;padding:8px 11px;border-radius:7px;color:var(--muted);font-size:12px;font-weight:700}.axis-control input:checked+span{background:var(--panel2);color:var(--text);box-shadow:0 1px 5px #0008}.axis-hint{text-align:right;color:var(--muted);font-size:10px;margin-top:6px}.export-row{display:flex;justify-content:flex-end;align-items:center;gap:8px;margin-top:9px}.export-status{font-size:10px;color:var(--muted);text-align:right;max-width:250px;overflow-wrap:anywhere}.export-status.bad{color:var(--bad)}.ghost[disabled]{opacity:.55;cursor:progress}.notice-wrap{display:grid;gap:7px;margin:0 0 16px}.fold>summary{cursor:pointer;list-style:none;display:flex;align-items:center;gap:8px}.fold>summary::-webkit-details-marker{display:none}.fold>summary::before{content:'\25B8';font-size:9px;color:var(--accent)}.fold[open]>summary::before{content:'\25BE'}.fold>summary:hover{color:var(--text)}.fold-count{color:var(--muted);font-weight:400}.notice{padding:10px 12px;border:1px solid #3d3940;background:#18161c;color:#c6b9c6;border-radius:10px;font-size:12px}.section-title{margin:30px 0 12px;font-size:13px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted)}.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,410px),1fr));gap:12px}.chart{background:linear-gradient(145deg,rgba(20,27,42,.92),rgba(12,17,27,.92));border:1px solid var(--line);border-radius:var(--radius);padding:15px;min-width:0}.chart-head{display:flex;align-items:baseline;justify-content:space-between;gap:10px;margin-bottom:7px}.chart h2{font-size:14px;margin:0;letter-spacing:-.01em}.chart-unit{color:var(--muted);font-size:10px}.canvas-wrap{height:270px;position:relative}.chart canvas{display:block;width:100%;height:100%;touch-action:none}.tooltip{position:fixed;z-index:20;pointer-events:none;background:#070a11ec;border:1px solid #36435b;border-radius:8px;padding:7px 9px;box-shadow:0 12px 30px #000a;font-size:11px;line-height:1.45;display:none;max-width:270px}.empty{border:1px dashed var(--line);border-radius:var(--radius);padding:30px;color:var(--muted);text-align:center}.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:var(--radius);background:var(--panel)}table{border-collapse:collapse;width:100%;font-size:12px;white-space:nowrap}th,td{text-align:right;padding:10px 12px;border-bottom:1px solid var(--line)}th{color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.08em}th:first-child,td:first-child{text-align:left}tbody tr:last-child td{border-bottom:0}.status{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:7px}.footer{color:var(--muted);font-size:11px;margin-top:26px}.mobile-runs{display:none}.skip details{color:var(--muted);font-size:12px}.skip summary{cursor:pointer}.skip code{color:var(--bad);white-space:normal;overflow-wrap:anywhere}@media(max-width:850px){.shell{grid-template-columns:1fr}.sidebar{display:none;position:fixed;z-index:30;width:min(88vw,320px);box-shadow:20px 0 70px #000;height:100vh}.sidebar.open{display:block}.main{padding:20px 14px 50px}.mobile-runs{display:inline-flex}.top{align-items:stretch;flex-direction:column}.axis-control{align-self:flex-start}.axis-hint{text-align:left}.export-row{justify-content:flex-start}.export-status{text-align:left}.canvas-wrap{height:240px}}@media(prefers-reduced-motion:reduce){*{scroll-behavior:auto!important}}
.analysis-controls{display:flex;align-items:center;gap:10px 14px;flex-wrap:wrap;border:1px solid var(--line);background:rgba(9,13,22,.82);border-radius:12px;padding:10px 12px;margin:0 0 18px}.control-title{font-size:11px;font-weight:800;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}.smoothing-control{border:1px solid var(--line);border-radius:9px;padding:3px;background:#070b13;display:flex;flex-wrap:wrap}.smoothing-control label{cursor:pointer}.smoothing-control input{position:absolute;opacity:0;pointer-events:none}.smoothing-control span{display:block;padding:6px 9px;border-radius:6px;color:var(--muted);font-size:11px;font-weight:700}.smoothing-control input:checked+span{background:var(--panel2);color:var(--text);box-shadow:0 1px 4px #0008}.smooth-level{display:flex;align-items:center;gap:7px;color:var(--muted);font-size:11px;font-weight:700}.smooth-level input{width:76px;border:1px solid var(--line);border-radius:7px;background:#070b13;color:var(--text);padding:6px 7px;outline:none}.smooth-level input:focus{border-color:var(--accent)}.smooth-level input:disabled{opacity:.45}.smoothing-hint{flex:1 1 100%;color:var(--muted);font-size:10px;line-height:1.45}.chart-head{align-items:center}.chart-tools{display:flex;align-items:center;gap:5px}.icon-button{border:1px solid transparent;background:transparent;color:var(--muted);border-radius:7px;padding:4px 7px;cursor:pointer;line-height:1}.icon-button:hover,.icon-button:focus-visible{border-color:var(--line);color:var(--text);outline:none}.chart canvas{cursor:crosshair}.chart canvas.dragging{cursor:crosshair}.chart.family-hidden{display:none}.family-control{display:flex;gap:5px;margin:12px 0;flex-wrap:wrap}.family-control button{border:1px solid var(--line);background:#090d16;color:var(--muted);border-radius:8px;padding:7px 11px;cursor:pointer;font-weight:700;font-size:12px}.family-control button.active{background:var(--panel2);color:var(--text);border-color:#49617f}.tooltip{z-index:60}.focus-dialog{width:min(96vw,1500px);height:min(94vh,980px);padding:0;border:1px solid #43516a;border-radius:16px;background:var(--panel);color:var(--text);box-shadow:0 30px 100px #000d}.focus-dialog::backdrop{background:#03050ae8;backdrop-filter:blur(4px)}.focus-shell{height:100%;display:flex;flex-direction:column;padding:16px}.focus-head{display:flex;justify-content:space-between;align-items:center;gap:12px}.focus-head h2{font-size:17px;margin:0}.focus-canvas{flex:1;min-height:0;margin-top:8px}.focus-canvas canvas{display:block;width:100%;height:100%;touch-action:none;cursor:crosshair}.interaction-hint{color:var(--muted);font-size:10px;margin-top:7px}.layer-step{display:flex;align-items:center;gap:11px;border:1px solid var(--line);background:rgba(9,13,22,.82);border-radius:11px;padding:9px 12px;margin:0 0 12px}.layer-step label{font-size:11px;font-weight:800;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}.layer-step input[type=range]{flex:1;min-width:140px;accent-color:var(--accent)}.layer-step output{font:11px ui-monospace,monospace;color:var(--text);min-width:9ch;text-align:right}@media(max-width:850px){.focus-dialog{width:100vw;height:100vh;max-width:none;max-height:none;border-radius:0}.analysis-controls{align-items:flex-start}.smoothing-control{width:100%}.smoothing-control label{flex:1;text-align:center}}
.study-picker{position:fixed;inset:0;background:var(--bg);overflow:auto;z-index:50;padding:38px 22px}
.study-inner{max-width:1000px;margin:0 auto}
.study-title{font-size:26px;margin:0 0 6px}
.study-sub{color:#91a0b8;margin:0 0 26px;font-size:14px}
.study-grid{display:grid;gap:12px;grid-template-columns:repeat(auto-fill,minmax(232px,1fr));margin-bottom:26px}
.study-card{display:flex;flex-direction:column;gap:5px;text-align:left;padding:14px 15px;border-radius:10px;
 border:1px solid var(--line);background:var(--panel);color:var(--text);cursor:pointer;font:inherit}
.study-card:hover{border-color:#3d76d6}
.study-card.on{border-color:#4d8dfd;background:var(--panel2)}
.study-name{font-weight:600;font-size:14px}
.study-meta,.study-load{color:#91a0b8;font-size:12px}
.study-detail{border-top:1px solid var(--line);padding-top:18px}
.study-doc{max-width:760px}
.study-actions{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:20px 0 8px}
.study-go,.study-action{padding:9px 15px;border-radius:8px;border:1px solid var(--line);
 background:var(--panel2);color:var(--text);cursor:pointer;font:inherit;font-size:13px}
.study-go{border-color:#4d8dfd}
.study-action{text-decoration:none;display:inline-flex;align-items:center}
.study-action:disabled{opacity:.5;cursor:default}
.study-warn{color:#d7a44a;font-size:12px;flex:1 1 260px}
.study-status{color:#91a0b8;font-size:13px;min-height:18px}
.study-status.bad{color:#e0736b}
#back-to-studies{margin-bottom:8px}
.md-h{font-size:15px;margin:18px 0 6px}
.md-p{margin:6px 0;font-size:13px;line-height:1.55}
.md-list{margin:6px 0 6px 18px;font-size:13px;line-height:1.6}
.md-code{background:var(--panel2);border:1px solid var(--line);border-radius:8px;padding:11px 13px;
 overflow-x:auto;font-size:12px;line-height:1.5;white-space:pre}
</style>
</head>
<body>
<div class="shell">
<aside class="sidebar" id="sidebar">
  <div class="brand"><b>◆</b> GPT TPU RIG</div>
  <p class="subtle">Toggle any number of complete, integrity-checked runs. Colors stay consistent across every plot.</p>
  <input class="search" id="run-search" type="search" placeholder="Filter runs…" aria-label="Filter runs">
  <div class="side-actions"><button class="ghost" id="all-runs">All</button><button class="ghost" id="no-runs">None</button><button class="ghost mobile-runs" id="close-runs">Close</button></div>
  <div class="run-list" id="run-list"></div>
</aside>
<main class="main">
  <div class="top">
    <div><div class="eyebrow">Static performance dossier</div><h1>Training, at a glance.</h1><div class="stats" id="stats"></div></div>
    <div><button class="ghost mobile-runs" id="open-runs">Choose runs</button><div class="axis-control" role="radiogroup" aria-label="Time-series x-axis"><label><input type="radio" name="axis" value="flops" checked><span>equi-FLOP</span></label><label><input type="radio" name="axis" value="step"><span>equi-step</span></label></div><div class="axis-hint" id="axis-hint">Estimated cumulative FLOPs</div><div class="export-row"><button class="ghost" id="export-runs" type="button">Export selection</button><button class="ghost" id="export-riglogs" type="button" hidden>Export .riglogs</button><span class="export-status" id="export-status" role="status" aria-live="polite"></span></div></div>
  </div>
  <div class="analysis-controls" aria-label="Scientific chart controls">
    <span class="control-title">Timeline x scale</span>
    <div class="smoothing-control" id="x-scale-control" role="radiogroup" aria-label="Timeline x-axis scale">
      <label><input type="radio" name="x-scale" value="log" checked><span>Log</span></label>
      <label><input type="radio" name="x-scale" value="linear"><span>Linear</span></label>
    </div>
    <span class="control-title">Timeline smoothing</span>
    <div class="smoothing-control" id="smoothing-control" role="radiogroup" aria-label="Smoothing method">
      <label><input type="radio" name="smoothing" value="raw" checked><span>Raw</span></label>
      <label><input type="radio" name="smoothing" value="ema"><span>EMA</span></label>
      <label><input type="radio" name="smoothing" value="mean"><span>Centered mean</span></label>
      <label><input type="radio" name="smoothing" value="median"><span>Centered median</span></label>
    </div>
    <label class="smooth-level" for="smoothing-level"><span id="smoothing-level-label">Span</span><input id="smoothing-level" type="number" min="1" max="1400" step="1" value="21" inputmode="numeric" disabled><span>samples</span></label>
    <div class="smoothing-hint" id="smoothing-hint" aria-live="polite"></div>
  </div>
  <details class="fold" id="summary-fold">
    <summary class="section-title fold-head">Run summary</summary>
    <div id="summary"></div>
  </details>
  <div class="section-title">Training timeline</div><div class="charts" id="time-charts"></div>
  <div class="section-title">Overall diagnostics timeline</div>
  <div class="family-control" id="family-control" role="group" aria-label="Diagnostic family"></div>
  <div class="charts" id="diagnostic-charts"></div>
  <div class="section-title">Per-scope diagnostic snapshot · categorical model scope / logical layer remains linear</div>
  <div class="layer-step" id="layer-step-control" hidden>
    <label for="layer-step">Step</label>
    <input type="range" id="layer-step" min="0" max="0" step="1" value="0" aria-label="Layer snapshot step">
    <output id="layer-step-value">—</output>
    <button class="ghost" id="layer-step-last" type="button">Last</button>
  </div>
  <div class="charts" id="layer-charts"></div>
  <details class="fold" id="notices-fold" hidden>
    <summary class="section-title fold-head">Notices <span class="fold-count" id="notices-count"></span></summary>
    <div class="notice-wrap" id="notices"></div>
  </details>
  <div class="skip" id="skipped"></div>
  <div class="footer" id="footer"></div>
</main>
</div>
<div class="tooltip" id="tooltip"></div>
<dialog class="focus-dialog" id="focus-dialog" aria-labelledby="focus-title">
  <div class="focus-shell">
    <div class="focus-head"><h2 id="focus-title"></h2><div><button class="ghost" id="focus-reset">Reset view</button> <button class="ghost" id="focus-close">Close</button></div></div>
    <div class="interaction-hint">Drag a rectangle to zoom both axes · click to pin a vertical line, click again to clear · double-click to reset · wheel does not zoom · Esc to close</div>
    <div class="focus-canvas"><canvas id="focus-canvas"></canvas></div>
  </div>
</dialog>
<script type="application/gzip-base64" id="report-data">__REPORT_DATA__</script>
<script>
(()=>{'use strict';
let D, runMap, visible, cameFromPicker=false;
const families=['grad','update','param'];
const HF='https://huggingface.co/datasets/';
function mdToHtml(src){
 // Enough markdown for a study card: headings, bullets, fences, code, links,
 // bold. Deliberately small -- these files are written by us, not by the web.
 const lines=String(src).split(/\r?\n/),out=[];let list=false,fence=false,buf=[];
 const inline=t=>esc(t).replace(/`([^`]+)`/g,'<code>$1</code>')
   .replace(/\*\*([^*]+)\*\*/g,'<strong>$1</strong>')
   .replace(/\[([^\]]+)\]\(([^)]+)\)/g,'<a href="$2" target="_blank" rel="noopener">$1</a>');
 const closeList=()=>{if(list){out.push('</ul>');list=false}};
 for(const line of lines){
  if(/^```/.test(line)){
   if(fence){out.push('<pre class="md-code">'+esc(buf.join('\n'))+'</pre>');buf=[];fence=false}
   else{closeList();fence=true}
   continue}
  if(fence){buf.push(line);continue}
  const h=/^(#{1,4})\s+(.*)$/.exec(line);
  if(h){closeList();const n=h[1].length;out.push(`<h${n+2} class="md-h">${inline(h[2])}</h${n+2}>`);continue}
  const li=/^\s*[-*]\s+(.*)$/.exec(line);
  if(li){if(!list){out.push('<ul class="md-list">');list=true}out.push('<li>'+inline(li[1])+'</li>');continue}
  if(!line.trim()){closeList();continue}
  closeList();out.push('<p class="md-p">'+inline(line)+'</p>')}
 closeList();
 if(fence&&buf.length)out.push('<pre class="md-code">'+esc(buf.join('\n'))+'</pre>');
 return out.join('')}
async function decodeGzipJson(response,onProgress){
 const total=Number(response.headers.get('content-length')||0),reader=response.body.getReader(),chunks=[];
 let got=0;
 for(;;){const {done,value}=await reader.read();if(done)break;chunks.push(value);got+=value.length;if(onProgress)onProgress(got,total)}
 if(typeof DecompressionStream!=='function')
  throw new Error('this browser cannot inflate the payload (needs DecompressionStream)');
 const stream=new Blob(chunks).stream().pipeThrough(new DecompressionStream('gzip'));
 return JSON.parse(await new Response(stream).text())}
async function fetchGzipJson(url,onProgress){
 const response=await fetch(url);
 if(!response.ok)throw new Error(url.split('/').pop()+' → HTTP '+response.status);
 return decodeGzipJson(response,onProgress)}
const FULL_CACHE='rig-study-full-v1';
function fullCacheKey(url,version){return url+'?rig-cache-version='+encodeURIComponent(String(version))}
async function openFullCache(){
 if(!('caches' in window))return null;
 try{return await caches.open(FULL_CACHE)}catch(error){return null}}
async function cachedFullPayload(url,version,onProgress){
 let cache=await openFullCache();const key=fullCacheKey(url,version);
 if(cache){
  let hit=null;try{hit=await cache.match(key)}catch(error){cache=null}
  if(hit)try{return{payload:await decodeGzipJson(hit,(got,total)=>{if(onProgress)onProgress(got,total,true)}),cache:'hit'}}
  catch(error){try{await cache.delete(key)}catch(ignored){}}}
 const response=await fetch(url);
 if(!response.ok)throw new Error(url.split('/').pop()+' → HTTP '+response.status);
 const writing=cache?cache.put(key,response.clone()).then(async()=>{
  const prefix=url+'?rig-cache-version=';
  for(const request of await cache.keys())if(request.url.startsWith(prefix)&&request.url!==new URL(key).href)await cache.delete(request);
  return true}).catch(()=>false):Promise.resolve(false),
  payload=await decodeGzipJson(response,(got,total)=>{if(onProgress)onProgress(got,total,false)}),stored=await writing;
 return{payload,cache:stored?'stored':'unavailable'}}
function mb(n){return (n/1048576).toFixed(n<10485760?1:0)+' MB'}
function withStudySource(payload,remote,study){return{...payload,rawSource:{repo:remote.repo,study:study.name}}}
function chooseStudy(remote){
 // Nothing is fetched until a study is picked, and the only thing a pick
 // fetches is the small overview. Anything larger is a separate, labelled act.
 return new Promise(resolve=>{
  const root=document.createElement('div');root.className='study-picker';
  root.innerHTML='<div class="study-inner"><h1 class="study-title">rig training logs</h1>'
   +'<p class="study-sub">'+esc(remote.studies.length)+' studies · '
   +esc(remote.studies.reduce((a,s)=>a+s.runs,0))+' runs · raw logs at '
   +'<a href="'+HF+esc(remote.repo)+'" target="_blank" rel="noopener">'+esc(remote.repo)+'</a></p>'
   +'<div class="study-grid">'+remote.studies.map((s,i)=>
     '<button class="study-card" data-index="'+i+'">'
     +'<span class="study-name">'+esc(s.title||s.name)+'</span>'
     +'<span class="study-meta">'+esc(s.runs)+' runs · '+esc(s.tier||'')+'</span>'
     +'<span class="study-load">overview '+mb(s.snapshot||0)+'</span></button>').join('')
   +'</div><div class="study-detail" id="study-detail"></div></div>';
  document.body.appendChild(root);
  root.querySelectorAll('.study-card').forEach(button=>{
   button.onclick=async()=>{
    const study=remote.studies[Number(button.dataset.index)],
     detail=root.querySelector('#study-detail');
    root.querySelectorAll('.study-card').forEach(b=>b.classList.toggle('on',b===button));
    detail.innerHTML='<div class="study-status">Loading overview…</div>';
    try{
     const base=HF+remote.repo+'/resolve/main/'+study.name+'/',
      browse=HF+remote.repo+'/tree/main/'+study.name;
     let card='';
     try{const r=await fetch(base+'README.md');if(r.ok)card=mdToHtml(await r.text())}catch(e){}
     const payload=await fetchGzipJson(base+'snapshot.json.gz');
     detail.innerHTML='<div class="study-doc">'+card+'</div>'
      +'<div class="study-actions"><button class="study-go" id="study-go">Open overview</button>'
      +'<a class="study-action" href="'+esc(browse)+'" target="_blank" rel="noopener">Browse raw logs</a>'
      +(study.full?'<button class="study-action" id="study-full">Load full report ('
        +mb(study.full)+')</button><span class="study-warn">Loads a '+mb(study.full)
        +' report payload from Hugging Face. Nothing is fetched until you click.</span>':'')
      +'</div><div class="study-status" id="study-progress"></div>';
     detail.querySelector('#study-go').onclick=()=>{cameFromPicker=true;root.remove();resolve(withStudySource(payload,remote,study))};
     const fullButton=detail.querySelector('#study-full');
     if(fullButton)fullButton.onclick=async()=>{
      fullButton.disabled=true;
      const status=detail.querySelector('#study-progress');
      try{
       const loaded=await cachedFullPayload(base+'full.json.gz',study.full,(got,total,cached)=>{
        status.textContent=(cached?'Reading cached ':'Downloading ')+mb(got)+(total?' of '+mb(total):'')+'…'}),
        full=loaded.payload;
       fullCacheStatus=loaded.cache;status.textContent=loaded.cache==='hit'?'Rendering cached report…':loaded.cache==='stored'?'Rendering… cached for next time.':'Rendering… browser cache unavailable.';
       cameFromPicker=true;root.remove();resolve(withStudySource(full,remote,study))}
      catch(error){status.textContent='Download failed: '+String(error&&error.message||error);
       fullButton.disabled=false}}}
    catch(error){detail.innerHTML='<div class="study-status bad">Could not load: '
     +esc(String(error&&error.message||error))+'</div>'}}})})}
async function loadPayload(){
 // The payload is gzip, base64-encoded: it is ~99% of the file and numeric
 // JSON, which compresses about 2x even after base64's 33% overhead.
 const node=document.getElementById('report-data'), raw=node.textContent.trim();
 if(node.type==='application/json')return JSON.parse(raw);
 const binary=atob(raw), bytes=new Uint8Array(binary.length);
 for(let i=0;i<binary.length;i++)bytes[i]=binary.charCodeAt(i);
 if(typeof DecompressionStream!=='function')
  throw new Error('this browser cannot inflate the report payload (needs DecompressionStream)');
 const stream=new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
 const payload=JSON.parse(await new Response(stream).text());
 return payload.remote?chooseStudy(payload.remote):payload}
const smoothCache=new WeakMap();
let axis='flops', xScale='log', family='grad', smoothing='raw', charts=[], frame=0, focusItem=null, fullCacheStatus=null;
// hoverX follows the pointer, pinnedX survives until clicked again, layerStep
// is the snapshot the layer charts show. All three are in step space, which is
// axis-independent, so they stay put when the x axis switches to FLOPs.
let hoverX=null, hoverPx=null, pinnedX=null, layerStep=null, layerSteps=[];
const $=id=>document.getElementById(id), esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=(n,d=3)=>{if(n==null||!Number.isFinite(+n))return '—';n=+n;const a=Math.abs(n);if(a>=1e18)return (n/1e18).toFixed(2)+' EF';if(a>=1e15)return (n/1e15).toFixed(2)+' PF';if(a>=1e12)return (n/1e12).toFixed(2)+' T';if(a>=1e9)return (n/1e9).toFixed(2)+' B';if(a>=1e6)return (n/1e6).toFixed(2)+' M';if(a>=1e3)return (n/1e3).toFixed(2)+' k';if(a>0&&a<.001)return n.toExponential(2);const f=n.toFixed(d);return d?f.replace(/(\.\d*?[1-9])0+$|\.0+$/,'$1'):f};
function smoothingMaximum(){const explicit=Number(D.meta.smoothingMaxSamples);if(Number.isFinite(explicit)&&explicit>0)return Math.round(explicit);const budget=Number(D.meta.maxChartPoints);return Number.isFinite(budget)&&budget>0?Math.round(budget):1400}
function effectiveSpan(normalize=false){const input=$('smoothing-level'),maximum=smoothingMaximum();let span=Math.round(Number(input.value));if(!Number.isFinite(span))span=21;span=Math.max(1,Math.min(maximum,span));if((smoothing==='mean'||smoothing==='median')&&span%2===0)span=span===maximum?Math.max(1,span-1):span+1;if(normalize)input.value=String(span);return span}
function smoothingName(){const span=effectiveSpan();return smoothing==='ema'?`EMA (span ${span})`:smoothing==='mean'?`centered mean (${span})`:smoothing==='median'?`centered median (${span})`:'raw'}
function updateSmoothingControls(normalize=false){const input=$('smoothing-level'),span=effectiveSpan(normalize);input.disabled=smoothing==='raw';$('smoothing-level-label').textContent=smoothing==='ema'?'Span':smoothing==='raw'?'Span':'Window';let detail;if(smoothing==='raw')detail='Exact recorded samples; no display smoothing.';else if(smoothing==='ema'){const alpha=2/(span+1);detail=`Causal EMA initialized at the first point; span ${span} gives α = 2/(span + 1) = ${alpha.toPrecision(4)}.`}else if(smoothing==='mean')detail=`Centered ${span}-sample arithmetic mean, with truncated endpoint windows; this avoids EMA phase lag but blurs peaks and is sensitive to outliers.`;else detail=`Centered ${span}-sample median, with truncated endpoint windows; this rejects isolated spikes and preserves step edges, but is not an amplitude average.`;$('smoothing-hint').textContent=detail+' Smoothing is display-only: a faint raw trace remains underneath and tooltips retain the raw sample. The learning-rate schedule and layer snapshots stay raw. Timeline x axes default to logarithmic; non-positive samples are hidden until Linear is selected. Categorical layer-snapshot axes remain linear. Span/window counts embedded plot samples. Drag a rectangle to zoom both axes; double-click or ↺ resets; the wheel never changes chart axes. Hovering any timeline draws a synchronized crosshair on all of them; a single click pins a vertical line there and clicking again clears it. The layer-snapshot step dragger marks its position on every timeline with a fainter dashed line.'}
function updateAxisHint(){$('axis-hint').textContent=(axis==='flops'?'Estimated cumulative FLOPs':'Optimizer step')+(xScale==='log'?' · logarithmic':' · linear')}
function init(){
 $('smoothing-level').max=String(smoothingMaximum());
 $('export-riglogs').hidden=!D.rawSource;
 $('stats').innerHTML=`<span class="pill"><strong>${D.meta.included}</strong> included</span><span class="pill"><strong>${D.meta.skipped}</strong> excluded</span><span class="pill">default <strong>equi-FLOP · log x</strong></span>`;
 $('notices').innerHTML=D.notices.map(n=>`<div class="notice">${esc(n)}</div>`).join('');
 $('notices-count').textContent=D.notices.length?'('+D.notices.length+')':'';
 $('notices-fold').hidden=!D.notices.length;
 buildRuns(); buildSummary(); buildCharts(); buildSkipped(); updateSmoothingControls(true); updateAxisHint();
 if(cameFromPicker){
  const back=document.createElement('button');
  back.className='ghost';back.id='back-to-studies';back.textContent='← All studies';
  back.onclick=()=>location.reload();
  const eyebrow=document.querySelector('.top .eyebrow');
  if(eyebrow&&eyebrow.parentNode)eyebrow.parentNode.insertBefore(back,eyebrow);}
 const cacheNote=fullCacheStatus==='hit'?' · full payload from browser cache':fullCacheStatus==='stored'?' · full payload cached by browser':fullCacheStatus==='unavailable'?' · browser cache unavailable':'';
 $('footer').textContent=(cameFromPicker
  ?`Generated ${new Date(D.meta.generatedAt).toLocaleString()} · fetched from HuggingFace`
  :`Generated ${new Date(D.meta.generatedAt).toLocaleString()} · portable HTML · no network or external JavaScript`)+cacheNote;
 document.querySelectorAll('input[name=axis]').forEach(r=>r.addEventListener('change',()=>{if(!r.checked)return;axis=r.value;resetViews();updateAxisHint();schedule()}));
 document.querySelectorAll('input[name=x-scale]').forEach(r=>r.addEventListener('change',()=>{if(!r.checked)return;xScale=r.value;resetViews();updateAxisHint();schedule()}));
 document.querySelectorAll('input[name=smoothing]').forEach(r=>r.addEventListener('change',()=>{if(!r.checked)return;smoothing=r.value;updateSmoothingControls(true);schedule()}));
 $('smoothing-level').oninput=()=>{updateSmoothingControls();schedule()}; $('smoothing-level').onchange=()=>{updateSmoothingControls(true);schedule()};
 $('all-runs').onclick=()=>{D.runs.forEach(r=>visible.add(r.id));syncChecks();resetViews();schedule()}; $('no-runs').onclick=()=>{visible.clear();syncChecks();resetViews();schedule()};
 $('run-search').oninput=e=>{const q=e.target.value.toLowerCase();document.querySelectorAll('.run-toggle').forEach(x=>x.hidden=!x.dataset.search.includes(q))};
 $('open-runs').onclick=()=>{$('sidebar').classList.add('open')}; $('close-runs').onclick=()=>{$('sidebar').classList.remove('open')};
 $('export-runs').onclick=()=>{exportSelection()};
 $('export-riglogs').onclick=()=>{exportRiglogs()};
 $('focus-close').onclick=()=>closeFocus(); $('focus-reset').onclick=()=>{if(focusItem){focusItem.view=null;redraw(focusItem)}};
 $('layer-step').addEventListener('input',e=>{const i=Number(e.target.value);if(layerSteps[i]!==undefined)setLayerStep(layerSteps[i])});
 $('layer-step-last').addEventListener('click',()=>{if(!layerSteps.length)return;$('layer-step').value=String(layerSteps.length-1);setLayerStep(layerSteps[layerSteps.length-1])});
 $('focus-dialog').addEventListener('close',finishFocus);
 new ResizeObserver(()=>{if(focusItem&&$('focus-dialog').open)redraw(focusItem)}).observe($('focus-dialog'));
 new ResizeObserver(schedule).observe(document.querySelector('.main')); schedule();
}
// Topology belongs next to the number it produced: the same seed on a
// different chip count has landed 0.004-0.023 nats away, purely through
// reduction order and per-chip batch, with identical data.
function hw(r){
 if(!r.chip&&!r.devices)return '';
 const bits=[r.chip,r.devices?r.devices+' chips':null,r.processes?r.processes+'p':null].filter(Boolean);
 return ' · '+esc(bits.join(' '))}
function buildRuns(){$('run-list').innerHTML=D.runs.map(r=>`<label class="run-toggle" data-search="${esc((r.label+' '+r.id+' '+r.classification+' '+(r.chip||'')).toLowerCase())}"><input type="checkbox" data-run="${esc(r.id)}" ${r.selected?'checked':''}><span class="dot" style="color:${r.color};background:${r.color}"></span><span class="run-name">${esc(r.label)}<small class="run-meta">${esc(r.classification)} · step ${fmt(r.finalStep,0)} · val ${fmt(r.validationLoss,4)}${hw(r)}${r.ledger?' · ledger ✓':' · unledgered'}</small></span></label>`).join('')||'<p class="subtle">No plot-able runs found.</p>';document.querySelectorAll('[data-run]').forEach(x=>x.onchange=()=>{x.checked?visible.add(x.dataset.run):visible.delete(x.dataset.run);resetViews();buildLayerSteps();schedule()})}
function syncChecks(){document.querySelectorAll('[data-run]').forEach(x=>x.checked=visible.has(x.dataset.run))}
function buildSummary(){if(!D.runs.length){$('summary').innerHTML='<div class="empty">No eligible run passed the report completeness, profile, and qualification checks.</div>';return}$('summary').innerHTML=`<div class="table-wrap"><table><thead><tr><th>Run</th><th>Class</th><th>Steps</th><th>Tokens</th><th>Train s</th><th>Train loss</th><th>Val loss</th><th>Fresh10</th><th>FLOP x</th><th>Final scopes</th></tr></thead><tbody>${D.runs.map(r=>`<tr><td><span class="status" style="background:${r.color}"></span>${esc(r.label)}</td><td>${esc(r.classification)}</td><td>${fmt(r.finalStep,0)}</td><td>${fmt(r.tokens,0)}</td><td>${fmt(r.trainSeconds,2)}</td><td>${fmt(r.trainLoss,4)}</td><td>${fmt(r.validationLoss,4)}</td><td>${fmt(r.fresh10Loss,4)}</td><td title="${esc(r.flopSource)}">${r.flopSource.startsWith('derived:')?'derived':'logged'}</td><td>${r.hasLayerStats?'yes':'—'}</td></tr>`).join('')}</tbody></table></div>`}
function buildCharts(){charts=[];makeGroup($('time-charts'),D.timeCharts,'time');makeGroup($('diagnostic-charts'),D.diagnosticCharts,'time');makeGroup($('layer-charts'),D.layerCharts,'layer');buildLayerSteps();buildFamilies();if(!D.diagnosticCharts.length)$('diagnostic-charts').innerHTML='<div class="empty">No included run recorded overall diagnostics.</div>';if(!D.layerCharts.length)$('layer-charts').innerHTML='<div class="empty">No included run recorded final model-scope diagnostics or retained compatible layer arrays.</div>'}
function makeGroup(root,data,type){root.innerHTML=data.map(c=>`<article class="chart" data-family="${esc(c.family||'')}"><div class="chart-head"><h2>${esc(c.title)}</h2><div class="chart-tools"><span class="chart-unit">${esc(c.yLabel)}</span><button class="icon-button reset-chart" title="Reset view" aria-label="Reset ${esc(c.title)} view">↺</button><button class="icon-button expand-chart" title="Open full panel" aria-label="Enlarge ${esc(c.title)}">⛶</button></div></div><div class="canvas-wrap"><canvas aria-label="${esc(c.title)}"></canvas></div></article>`).join('');[...root.querySelectorAll('.chart')].forEach((article,i)=>{const item={canvas:article.querySelector('canvas'),article,data:data[i],type,view:null,drag:null};article.querySelector('.reset-chart').onclick=()=>{item.view=null;redraw(item)};article.querySelector('.expand-chart').onclick=()=>openFocus(item);attachCanvas(item);charts.push(item)})}
function buildLayerSteps(){
 const all=new Set();for(const c of D.layerCharts)for(const s of c.series)if(visible.has(s.run))for(const step of s.steps)all.add(step);
 layerSteps=[...all].sort((a,b)=>a-b);
 const wrap=$('layer-step-control'),slider=$('layer-step');
 wrap.hidden=layerSteps.length<2;
 if(!layerSteps.length){layerStep=null;$('layer-step-value').textContent='—';return}
 // Default to the last recorded step, which is the historical behaviour.
 if(layerStep===null||!layerSteps.includes(layerStep))layerStep=layerSteps[layerSteps.length-1];
 slider.max=String(layerSteps.length-1);slider.value=String(layerSteps.indexOf(layerStep));
 $('layer-step-value').textContent='step '+fmt(layerStep,0)}
function setLayerStep(step){if(step===layerStep)return;layerStep=step;$('layer-step-value').textContent='step '+fmt(layerStep,0);schedule()}
function buildFamilies(){const available=new Set([...D.diagnosticCharts,...D.layerCharts].map(c=>c.family));family=families.find(x=>available.has(x))||'grad';$('family-control').innerHTML=families.map(x=>`<button data-family-button="${x}" aria-pressed="false" ${available.has(x)?'':'disabled'}>${x==='param'?'Parameter':x[0].toUpperCase()+x.slice(1)}</button>`).join('');document.querySelectorAll('[data-family-button]').forEach(b=>b.onclick=()=>{family=b.dataset.familyButton;applyFamily()});applyFamily()}
function applyFamily(){document.querySelectorAll('[data-family-button]').forEach(b=>{const active=b.dataset.familyButton===family;b.classList.toggle('active',active);b.setAttribute('aria-pressed',String(active))});charts.forEach(item=>{if(item.data.family)item.article.classList.toggle('family-hidden',item.data.family!==family)});setFamilyEmpty($('diagnostic-charts'),D.diagnosticCharts,'No overall '+family+' diagnostics were recorded.');setFamilyEmpty($('layer-charts'),D.layerCharts,'No final model-scope '+family+' diagnostics were recorded.');schedule()}
function setFamilyEmpty(root,data,message){let empty=root.querySelector('.family-empty');const missing=!data.some(c=>c.family===family);if(missing&&!empty){empty=document.createElement('div');empty.className='empty family-empty';root.appendChild(empty)}if(empty){empty.textContent=message;empty.hidden=!missing}}
function buildSkipped(){if(!D.skipped.length)return;$('skipped').innerHTML=`<div class="section-title">Excluded by profile / completeness / official admission qualification / integrity scan</div>${D.skipped.map(x=>`<details><summary>${esc(x.run)}</summary><code>${esc(x.reason)}</code></details>`).join('')}`}
function schedule(){cancelAnimationFrame(frame);frame=requestAnimationFrame(()=>{charts.forEach(item=>{if(!item.article.classList.contains('family-hidden'))draw(item)});if(focusItem&&$('focus-dialog').open)draw(focusItem)})}
function redraw(item){cancelAnimationFrame(item._frame||0);item._frame=requestAnimationFrame(()=>draw(item))}
function resetViews(){charts.forEach(item=>item.view=null);if(focusItem)focusItem.view=null}
function xOf(item,p){return item.type==='layer'?p[0]:p[axis==='flops'?1:0]}function yOf(item,p){return item.type==='layer'?p[1]:p[2]}
function frameIndex(s,step){let best=0,bd=Infinity;for(let i=0;i<s.steps.length;i++){const d=Math.abs(s.steps[i]-step);if(d<bd){bd=d;best=i}}return best}
const frameCache=new WeakMap();
function layerFrame(s,step){
 // Identity must be stable across redraws: draw() decides whether a series
 // is smoothed by comparing array identity, and a fresh array every call
 // made every layer series look smoothed, which then dereferenced the
 // s.points that layer series do not have. It also avoids rebuilding every
 // frame on each repaint while the step dragger is being scrubbed.
 let hit=frameCache.get(s);
 if(hit&&hit.step===step)return hit.pts;
 const i=frameIndex(s,step),row=s.values[i];
 const pts=s.scopes.map((sc,j)=>[sc[0],row[j],sc[1]]);
 frameCache.set(s,{step,pts});return pts}
function seriesPoints(item,s){return item.type==='layer'?layerFrame(s,layerStep):s.points}
function chartXScale(item){return item.type==='layer'?'linear':xScale}
function validX(item,x){return Number.isFinite(x)&&(chartXScale(item)==='linear'||x>0)}
function transformX(item,x){return chartXScale(item)==='log'?Math.log10(x):x}
function untransformX(item,x){return chartXScale(item)==='log'?10**x:x}
function dataBounds(item){let x0=Infinity,x1=-Infinity,y0=Infinity,y1=-Infinity,count=0;for(const s of item.data.series){if(!visible.has(s.run))continue;
  // Bounds must match what draw() actually plots. Layer charts show one
  // step's frame at a time; sizing the axis to every retained step instead
  // let a single early spike (grad_clip is off) set the ceiling forever.
  const pts=seriesPoints(item,s);
  for(const p of pts){const x=xOf(item,p),y=yOf(item,p);if(validX(item,x)&&Number.isFinite(y)){x0=Math.min(x0,x);x1=Math.max(x1,x);y0=Math.min(y0,y);y1=Math.max(y1,y);count++}}}if(!count)return null;if(x0===x1){if(chartXScale(item)==='log'){const q=Math.sqrt(10);x0/=q;x1*=q}else{x0-=.5;x1+=.5}}if(y0===y1){const q=Math.abs(y0)*.05||.5;y0-=q;y1+=q}else{const q=(y1-y0)*.08;y0-=q;y1+=q}return{x0,x1,y0,y1}}
function bounds(item){const base=dataBounds(item);return base?(item.view||base):null}
function smoothingApplies(item){return smoothing!=='raw'&&effectiveSpan()>1&&item.type==='time'&&item.data.key!=='learning_rate'}
function displayPoints(item,series){const source=seriesPoints(item,series);if(!smoothingApplies(item)||source.length<2)return source;const span=effectiveSpan(),key=smoothing+':'+span;let cache=smoothCache.get(series);if(!cache){cache=new Map();smoothCache.set(series,cache)}if(cache.has(key))return cache.get(key);let result;if(smoothing==='ema'){const alpha=2/(span+1);let value=source[0][2];result=source.map((point,index)=>{if(index)value=alpha*point[2]+(1-alpha)*value;return[point[0],point[1],value]})}else{const half=(span-1)>>1;if(smoothing==='mean'){const prefix=[0];for(const point of source)prefix.push(prefix[prefix.length-1]+point[2]);result=source.map((point,index)=>{const lo=Math.max(0,index-half),hi=Math.min(source.length-1,index+half);return[point[0],point[1],(prefix[hi+1]-prefix[lo])/(hi-lo+1)]})}else result=source.map((point,index)=>{const lo=Math.max(0,index-half),hi=Math.min(source.length-1,index+half),values=[];for(let i=lo;i<=hi;i++)values.push(source[i][2]);values.sort((a,b)=>a-b);const middle=values.length>>1,value=values.length%2?values[middle]:(values[middle-1]+values[middle])/2;return[point[0],point[1],value]})}cache.set(key,result);return result}
function drawBudget(width){return Math.max(128,Math.round(width*2))}
// The payload carries every recorded sample, so one series can hold tens of
// thousands of points -- far more than a canvas has columns for. Drawing them
// all is slow and, worse, unreadable: the line becomes a solid band. Each
// series is therefore reduced to the visible span at roughly two points per
// CSS pixel, and the reduction is redone whenever the zoom changes, so detail
// appears as you go in rather than being decided once when the file was built.
function visibleSlice(item,points){
 const v=item.view;if(!v||points.length<3)return points;
 const at=i=>xOf(item,points[i]);
 let lo=0,hi=points.length;
 while(lo<hi){const mid=(lo+hi)>>1;if(at(mid)<v.x0)lo=mid+1;else hi=mid}
 const start=Math.max(0,lo-1);
 lo=0;hi=points.length;
 while(lo<hi){const mid=(lo+hi)>>1;if(at(mid)<=v.x1)lo=mid+1;else hi=mid}
 const end=Math.min(points.length,lo+1);
 return end-start>=2?points.slice(start,end):points}
// Each bucket keeps its lowest and highest sample rather than one
// representative, emitted in the order they occur so x stays increasing. A
// lone spike therefore survives at every zoom level. Picking one point per
// bucket -- which is what downsampling for shape does -- would drop it, and
// the curve would look clean and be wrong, which is the failure worth ruling
// out here: gradient spikes are usually the thing being looked for.
function envelope(item,points,budget){
 if(points.length<=budget)return points;
 const buckets=Math.max(1,budget>>1),out=[];
 for(let i=0;i<buckets;i++){
  const lo=Math.floor(i*points.length/buckets),hi=Math.floor((i+1)*points.length/buckets);
  if(hi<=lo)continue;
  let mn=lo,mx=lo,vmn=yOf(item,points[lo]),vmx=vmn;
  for(let j=lo+1;j<hi;j++){const v=yOf(item,points[j]);if(v<vmn){vmn=v;mn=j}else if(v>vmx){vmx=v;mx=j}}
  if(mn===mx)out.push(points[mn]);
  else if(mn<mx){out.push(points[mn]);out.push(points[mx])}
  else{out.push(points[mx]);out.push(points[mn])}}
 return out}
function reduceForDraw(item,points,budget){return envelope(item,visibleSlice(item,points),budget)}
function draw(item){
 const c=item.canvas,rect=c.getBoundingClientRect(),dpr=Math.min(devicePixelRatio||1,2),w=Math.max(1,rect.width),h=Math.max(1,rect.height);
 if(c.width!==Math.round(w*dpr)||c.height!==Math.round(h*dpr)){c.width=Math.round(w*dpr);c.height=Math.round(h*dpr)}
 const g=c.getContext('2d');g.setTransform(dpr,0,0,dpr,0,0);g.clearRect(0,0,w,h);const m={l:54,r:14,t:12,b:31},b=bounds(item);
 if(!b){
  // Hide the whole panel rather than leaving an empty frame with a caption.
  // Metrics come and go -- a routed run records routing series a dense one
  // never will -- so a report is normally full of charts that do not apply to
  // whatever is selected, and a grid of empty frames buries the ones that do.
  // The panel is hidden rather than removed so it comes straight back when the
  // selection changes.
  if(item.article){item.article.hidden=true;item._plot=null;return}
  g.fillStyle='#91a0b8';g.font='12px system-ui';g.textAlign='center';
  g.fillText('No selected run has this metric',w/2,h/2);item._plot=null;return}
 if(item.article)item.article.hidden=false;
 const tx0=transformX(item,b.x0),tx1=transformX(item,b.x1),X=x=>m.l+(transformX(item,x)-tx0)/(tx1-tx0)*(w-m.l-m.r),Y=y=>m.t+(b.y1-y)/(b.y1-b.y0)*(h-m.t-m.b),IX=x=>untransformX(item,tx0+(x-m.l)/(w-m.l-m.r)*(tx1-tx0)),IY=y=>b.y1-(y-m.t)/(h-m.t-m.b)*(b.y1-b.y0);
 g.strokeStyle='#253047';g.lineWidth=1;g.fillStyle='#7e8da6';g.font='10px ui-monospace,monospace';
 for(let i=0;i<=4;i++){const y=m.t+(h-m.t-m.b)*i/4,v=b.y1-(b.y1-b.y0)*i/4;g.beginPath();g.moveTo(m.l,y);g.lineTo(w-m.r,y);g.stroke();g.textAlign='right';g.fillText(fmt(v,3),m.l-7,y+3)}
 for(let i=0;i<=4;i++){const x=m.l+(w-m.l-m.r)*i/4,v=untransformX(item,tx0+(tx1-tx0)*i/4);g.textAlign=i===0?'left':i===4?'right':'center';g.fillText(fmt(v,item.type==='layer'?0:2),x,h-9)}
 g.save();g.beginPath();g.rect(m.l,m.t,w-m.l-m.r,h-m.t-m.b);g.clip();
 const budget=drawBudget(w-m.l-m.r);
 for(const s of item.data.series){if(!visible.has(s.run))continue;const r=runMap.get(s.run),fullBase=seriesPoints(item,s),fullShown=displayPoints(item,s),smoothed=fullShown!==fullBase,base=smoothed?reduceForDraw(item,fullBase,budget):null,shown=reduceForDraw(item,fullShown,budget);const stroke=(points,alpha,width)=>{g.strokeStyle=r.color;g.lineWidth=width;g.globalAlpha=alpha;g.beginPath();let count=0;for(const point of points){const x=xOf(item,point),y=yOf(item,point);if(!validX(item,x)||!Number.isFinite(y))continue;count?g.lineTo(X(x),Y(y)):g.moveTo(X(x),Y(y));count++}g.stroke();return count};if(smoothed)stroke(base,.22,1);const count=stroke(shown,.92,smoothed?2.2:1.7);if(count<=20){g.fillStyle=r.color;for(const point of shown){const x=xOf(item,point),y=yOf(item,point);if(!validX(item,x)||!Number.isFinite(y))continue;g.beginPath();g.arc(X(x),Y(y),2.3,0,Math.PI*2);g.fill()}}}
 g.restore();g.globalAlpha=1;item._plot={b,m,w,h,X,Y,IX,IY};
 drawMarkers(item,g);
 if(item.drag){const d=item.drag,left=Math.min(d.x0,d.x1),top=Math.min(d.y0,d.y1),width=Math.abs(d.x1-d.x0),height=Math.abs(d.y1-d.y0);g.save();g.fillStyle='rgba(125,211,252,.13)';g.strokeStyle='#7dd3fc';g.lineWidth=1.25;g.setLineDash([5,3]);g.fillRect(left,top,width,height);g.strokeRect(left+.5,top+.5,Math.max(0,width-1),Math.max(0,height-1));g.restore()}
}
function axisToStep(value){
 if(axis!=='flops')return value;
 for(const c of D.timeCharts)for(const s of c.series){if(!visible.has(s.run))continue;
  const pts=s.points;if(!pts.length)continue;
  let lo=0,hi=pts.length-1;if(value<=pts[0][1])return pts[0][0];if(value>=pts[hi][1])return pts[hi][0];
  while(lo<hi-1){const mid=(lo+hi)>>1;if(pts[mid][1]<value)lo=mid;else hi=mid}
  const a=pts[lo],b2=pts[hi],t=(value-a[1])/((b2[1]-a[1])||1);return a[0]+t*(b2[0]-a[0])}
 return value}
function setHover(item,e){if(item.type==='layer'||!item._plot)return;
 // Track the pointer exactly and throttle on the *pixel* instead of the value.
 // Quantizing the value left a visible stride that grew worse as you zoomed in,
 // because a fixed step-space grid does not shrink with the view.
 const rect=item.canvas.getBoundingClientRect(),px=Math.round(e.clientX-rect.left);
 if(px===hoverPx)return;hoverPx=px;hoverX=axisToStep(item._plot.IX(px));schedule()}
function stepToAxis(step){
 // Markers are held in step space; on a FLOPs axis they are mapped through any
 // visible run's own step->FLOPs curve, which is what makes them line up.
 if(axis!=='flops')return step;
 for(const c of D.timeCharts)for(const s of c.series){if(!visible.has(s.run))continue;
  const pts=s.points;if(!pts.length)continue;
  let lo=0,hi=pts.length-1;if(step<=pts[0][0])return pts[0][1];if(step>=pts[hi][0])return pts[hi][1];
  while(lo<hi-1){const mid=(lo+hi)>>1;if(pts[mid][0]<step)lo=mid;else hi=mid}
  const a=pts[lo],b2=pts[hi],t=(step-a[0])/((b2[0]-a[0])||1);return a[1]+t*(b2[1]-a[1])}
 return null}
function markerLine(item,g,step,color,width,dash,alpha){
 if(step==null||item.type==='layer')return;const p=item._plot,x=stepToAxis(step);
 if(x==null||!validX(item,x))return;const px=p.X(x);
 if(px<p.m.l-.5||px>p.w-p.m.r+.5)return;
 g.save();g.globalAlpha=alpha;g.strokeStyle=color;g.lineWidth=width;g.setLineDash(dash);
 g.beginPath();g.moveTo(px,p.m.t);g.lineTo(px,p.h-p.m.b);g.stroke();g.restore()}
function drawMarkers(item,g){
 // Fainter dashed = where the layer dragger is; solid = pinned by a click;
 // thin = the live cursor. Three deliberately distinct weights.
 markerLine(item,g,layerStep,'#f0abfc',1.5,[3,4],.55);
 markerLine(item,g,pinnedX,'#7dd3fc',1.5,[],.85);
 markerLine(item,g,hoverX,'#93a4bf',1,[2,3],.55)}
function eventPoint(e,item,clampToPlot=false){const p=item._plot,rect=item.canvas.getBoundingClientRect();let x=e.clientX-rect.left,y=e.clientY-rect.top;if(clampToPlot){x=Math.max(p.m.l,Math.min(p.w-p.m.r,x));y=Math.max(p.m.t,Math.min(p.h-p.m.b,y))}return{x,y}}
function attachCanvas(item){const c=item.canvas;c.ondblclick=()=>{item.view=null;redraw(item)};c.onpointerdown=e=>{if(e.button!==0||!item._plot)return;const q=eventPoint(e,item);if(q.x<item._plot.m.l||q.x>item._plot.w-item._plot.m.r||q.y<item._plot.m.t||q.y>item._plot.h-item._plot.m.b)return;e.preventDefault();item.drag={x0:q.x,y0:q.y,x1:q.x,y1:q.y};c.setPointerCapture(e.pointerId);c.classList.add('dragging');hideTip()};c.onpointermove=e=>{if(item.drag){const q=eventPoint(e,item,true);item.drag.x1=q.x;item.drag.y1=q.y;redraw(item);return}setHover(item,e);tip(e,item)};c.onpointerup=e=>finishBox(e,item);c.onpointercancel=e=>cancelBox(e,item);c.onpointerleave=()=>{if(!item.drag)hideTip()}}
function finishBox(e,item){const c=item.canvas,d=item.drag;if(!d)return;const q=eventPoint(e,item,true);d.x1=q.x;d.y1=q.y;item.drag=null;c.classList.remove('dragging');if(c.hasPointerCapture(e.pointerId))c.releasePointerCapture(e.pointerId);if(Math.abs(d.x1-d.x0)<4&&Math.abs(d.y1-d.y0)<4&&item._plot&&item.type!=='layer'){
  // A click, not a zoom box: toggle the pinned vertical line.
  const step=axisToStep(item._plot.IX(d.x1));
  // Clicking the pinned line again clears it; judged in pixels so the target
  // stays the same size at every zoom level.
  let onPinned=false;
  if(pinnedX!==null){const at=stepToAxis(pinnedX);
   if(at!=null&&validX(item,at))onPinned=Math.abs(item._plot.X(at)-d.x1)<=6}
  pinnedX=onPinned?null:step;
  redraw(item);schedule();return}
 if(Math.abs(d.x1-d.x0)>=8&&Math.abs(d.y1-d.y0)>=8&&item._plot){const p=item._plot,left=Math.min(d.x0,d.x1),right=Math.max(d.x0,d.x1),top=Math.min(d.y0,d.y1),bottom=Math.max(d.y0,d.y1),xa=p.IX(left),xb=p.IX(right),ya=p.IY(top),yb=p.IY(bottom);item.view={x0:Math.min(xa,xb),x1:Math.max(xa,xb),y0:Math.min(ya,yb),y1:Math.max(ya,yb)}}redraw(item)}
function cancelBox(e,item){const c=item.canvas;item.drag=null;c.classList.remove('dragging');if(c.hasPointerCapture(e.pointerId))c.releasePointerCapture(e.pointerId);redraw(item)}
function nearest(series,item,target){let lo=0,hi=series.length;while(lo<hi){const mid=(lo+hi)>>1;if(xOf(item,series[mid])<target)lo=mid+1;else hi=mid}let best=null;for(const index of [lo-1,lo])if(index>=0&&index<series.length){const point=series[index],x=xOf(item,point);if(!validX(item,x))continue;const dist=Math.abs(transformX(item,x)-transformX(item,target));if(!best||dist<best.dist)best={p:point,dist,index}}return best}
function tip(e,item){if(!item._plot)return;const rect=item.canvas.getBoundingClientRect(),px=e.clientX-rect.left,py=e.clientY-rect.top,target=item._plot.IX(px);let best=null;for(const s of item.data.series){if(!visible.has(s.run))continue;const shown=displayPoints(item,s),found=nearest(shown,item,target);if(!found)continue;const x=xOf(item,found.p),y=yOf(item,found.p),dx=item._plot.X(x)-px,dy=item._plot.Y(y)-py,dist=Math.hypot(dx,dy),raw=seriesPoints(item,s)[found.index];if(!best||dist<best.dist)best={dist,p:found.p,x,y,rawY:raw?yOf(item,raw):null,smoothed:shown!==seriesPoints(item,s),run:runMap.get(s.run)}}if(!best||best.dist>42){hideTip();return}const xline=item.type==='layer'
  ? esc(best.p[2]||'model scope')+(layerStep!=null?' · step '+fmt(layerStep,0):'')
  : 'step '+fmt(best.p[0],0)+' · FLOPs '+fmt(best.p[1],2),valueLine=best.smoothed?`${esc(item.data.title)} · ${esc(smoothingName())}: ${fmt(best.y,5)}<br>Raw sample: ${fmt(best.rawY,5)}`:`${esc(item.data.title)}: ${fmt(best.y,5)}`,t=$('tooltip');t.innerHTML=`<b style="color:${best.run.color}">${esc(best.run.label)}</b><br>${xline}<br>${valueLine}`;t.style.display='block';t.style.left=Math.max(4,Math.min(innerWidth-285,e.clientX+13))+'px';t.style.top=Math.max(4,Math.min(innerHeight-105,e.clientY+13))+'px'}
function hideTip(){$('tooltip').style.display='none'}
function openFocus(source){const dialog=$('focus-dialog'),title=(source.data.family?source.data.family.toUpperCase()+' · ':'')+source.data.title;$('focus-title').textContent=title;$('focus-canvas').setAttribute('aria-label',title);focusItem={canvas:$('focus-canvas'),data:source.data,type:source.type,view:source.view?{...source.view}:null,drag:null,source};attachCanvas(focusItem);dialog.showModal();requestAnimationFrame(()=>draw(focusItem))}
function closeFocus(){const dialog=$('focus-dialog');if(dialog.open)dialog.close()}
function finishFocus(){hideTip();if(focusItem){focusItem.source.view=focusItem.view?{...focusItem.view}:null;redraw(focusItem.source)}focusItem=null}
// Export rebuilds this same page around a payload filtered to the selected
// runs. The shell is captured before init() touches the DOM, with the payload
// swapped for a slot so the megabytes of base64 are never held twice.
const PAYLOAD_SLOT='@@RIG_PAYLOAD@@';
let shell=null;
function captureShell(){
 const node=document.getElementById('report-data'),raw=node.textContent;
 node.textContent=PAYLOAD_SLOT;
 shell='<!doctype html>\n'+document.documentElement.outerHTML+'\n';
 node.textContent=raw}
function exportSlug(runs){
 if(runs.length===1){
  const s=String(runs[0].label||runs[0].id).toLowerCase().replace(/[^a-z0-9]+/g,'-').replace(/^-+|-+$/g,'').slice(0,40);
  if(s)return s}
 return runs.length+'-runs'}
function selectedPayload(){
 const keep=visible,
  pick=list=>list.map(c=>({...c,series:c.series.filter(s=>keep.has(s.run))})).filter(c=>c.series.length),
  runs=D.runs.filter(r=>keep.has(r.id)).map(r=>({...r,selected:true})),
  // A partial export must say so on its own face; a full one stays a copy.
  notices=runs.length<D.runs.length
   ?[...D.notices,'Exported subset: '+runs.length+' of '+D.runs.length+' runs from the report generated '+D.meta.generatedAt+'.']
   :D.notices;
 return{...D,meta:{...D.meta,included:runs.length,skipped:0},runs,notices,skipped:[],
  timeCharts:pick(D.timeCharts),diagnosticCharts:pick(D.diagnosticCharts),layerCharts:pick(D.layerCharts)}}
async function packPayload(payload){
 // Nothing could have loaded this page without DecompressionStream, so its
 // counterpart is present too; refusing beats emitting a differently-shaped file.
 if(typeof CompressionStream!=='function')
  throw new Error('this browser cannot compress the export (needs CompressionStream)');
 const stream=new Blob([JSON.stringify(payload)]).stream().pipeThrough(new CompressionStream('gzip')),
  bytes=new Uint8Array(await new Response(stream).arrayBuffer());
 let binary='';
 // Chunked: String.fromCharCode.apply overflows the call stack on megabytes.
 for(let i=0;i<bytes.length;i+=0x8000)binary+=String.fromCharCode.apply(null,bytes.subarray(i,i+0x8000));
 return btoa(binary)}
async function chooseSaveFile(name,description,mime,extension){
 if(typeof window.showSaveFilePicker!=='function')return null;
 return window.showSaveFilePicker({suggestedName:name,types:[{description,accept:{[mime]:[extension]}}]})}
async function saveBlob(blob,name,handle){
 if(handle){const output=await handle.createWritable();await output.write(blob);await output.close();return handle.name||name}
 const url=URL.createObjectURL(blob),a=document.createElement('a');
 a.href=url;a.download=name;document.body.appendChild(a);a.click();a.remove();
 setTimeout(()=>URL.revokeObjectURL(url),30000);return name}
function cancelled(error){return error&&error.name==='AbortError'}
function setExportBusy(busy){$('export-runs').disabled=busy;$('export-riglogs').disabled=busy}
function exportStatus(message,bad){const el=$('export-status');el.textContent=message||'';el.classList.toggle('bad',!!bad)}
async function exportSelection(){
 if(!visible.size){exportStatus('Select at least one run to export.',true);return}
 const payload=selectedPayload(),name='report-'+exportSlug(payload.runs)+'.html';
 setExportBusy(true);exportStatus('Choose where to save the report…');
 try{
  const handle=await chooseSaveFile(name,'Self-contained rig report','text/html','.html');
  if(!shell)throw new Error('page shell was not captured');
  exportStatus('Preparing export…');
  const encoded=await packPayload(payload),
   html=shell.replace(PAYLOAD_SLOT,()=>encoded);
  if(html.length<=shell.length)throw new Error('could not splice the payload into the page shell');
  const saved=await saveBlob(new Blob([html],{type:'text/html'}),name,handle);
  exportStatus(saved+' · '+payload.runs.length+' run'+(payload.runs.length===1?'':'s')+' · '+(html.length/1048576).toFixed(1)+' MB')}
 catch(error){exportStatus(cancelled(error)?'Export cancelled.':'Export failed: '+String(error&&error.message||error),!cancelled(error))}
 finally{setExportBusy(false)}}
const tarEncoder=new TextEncoder();
function tarText(header,offset,width,value){const bytes=tarEncoder.encode(value);if(bytes.length>width)throw new Error('archive path is too long: '+value);header.set(bytes,offset)}
function tarOctal(header,offset,width,value){const encoded=Math.floor(value).toString(8);if(encoded.length>=width)throw new Error('archive entry is too large');tarText(header,offset,width,encoded.padStart(width-1,'0')+'\0')}
function tarHeader(path,size){
 const header=new Uint8Array(512),slash=path.lastIndexOf('/'),name=slash<0?path:path.slice(slash+1),prefix=slash<0?'':path.slice(0,slash);
 tarText(header,0,100,name);if(prefix)tarText(header,345,155,prefix);
 tarOctal(header,100,8,0o644);tarOctal(header,108,8,0);tarOctal(header,116,8,0);
 tarOctal(header,124,12,size);tarOctal(header,136,12,0);header.fill(32,148,156);header[156]=48;
 tarText(header,257,6,'ustar\0');tarText(header,263,2,'00');
 const checksum=header.reduce((sum,value)=>sum+value,0).toString(8).padStart(6,'0');
 tarText(header,148,8,checksum+'\0 ');return header}
function tarArchive(entries){
 const parts=[];for(const entry of entries){parts.push(tarHeader(entry.path,entry.blob.size),entry.blob);const padding=(512-entry.blob.size%512)%512;if(padding)parts.push(new Uint8Array(padding))}
 parts.push(new Uint8Array(1024));return new Blob(parts,{type:'application/x-tar'})}
function safeArchivePart(value){const part=String(value).replace(/[^A-Za-z0-9._-]+/g,'-').replace(/^\.+/,'');return part||'run'}
function riglogNames(run){
 const declared=Array.isArray(run.riglogs)?run.riglogs.filter(name=>typeof name==='string'&&/^[A-Za-z0-9._-]+\.riglog$/.test(name)):[];
 return declared.length?declared:['training.riglog','diagnostics.riglog','diagnostics-partial.riglog']}
async function fetchRiglogs(runs){
 const source=D.rawSource;if(!source||!source.repo||!source.study)throw new Error('raw .riglogs are available from reports opened through the study browser');
 const repo=String(source.repo).split('/').map(encodeURIComponent).join('/'),base=HF+repo+'/resolve/main/'+encodeURIComponent(source.study)+'/',entries=[];
 let attempted=0,total=runs.reduce((count,run)=>count+riglogNames(run).length,0);
 for(const run of runs)for(const name of riglogNames(run)){
  attempted++;exportStatus('Fetching .riglogs '+attempted+'/'+total+' · '+run.id+'/'+name+'…');
  const response=await fetch(base+encodeURIComponent(run.id)+'/'+encodeURIComponent(name));
  if(response.status===404&&name!=='training.riglog')continue;
  if(!response.ok)throw new Error(run.id+'/'+name+' → HTTP '+response.status);
  const blob=await response.blob(),magic=new Uint8Array(await blob.slice(0,8).arrayBuffer()),expected=[82,73,71,76,79,71,0,1];
  if(magic.length!==expected.length||expected.some((value,index)=>magic[index]!==value))throw new Error(run.id+'/'+name+' is not a valid .riglog');
  entries.push({path:safeArchivePart(run.id)+'/'+name,blob})}
 if(!entries.length)throw new Error('the selected runs did not publish any .riglog files');
 return entries}
async function exportRiglogs(){
 const runs=D.runs.filter(run=>visible.has(run.id));
 if(!runs.length){exportStatus('Select at least one run to export.',true);return}
 const name='riglogs-'+exportSlug(runs)+'.tar.gz';setExportBusy(true);exportStatus('Choose where to save the .riglog archive…');
 try{
  const handle=await chooseSaveFile(name,'Selected raw rig logs','application/gzip','.gz');
  if(typeof CompressionStream!=='function')throw new Error('this browser cannot compress the archive (needs CompressionStream)');
  const entries=await fetchRiglogs(runs);exportStatus('Packing '+entries.length+' original .riglog file'+(entries.length===1?'':'s')+'…');
  const stream=tarArchive(entries).stream().pipeThrough(new CompressionStream('gzip')),
   blob=await new Response(stream).blob(),saved=await saveBlob(blob,name,handle);
  exportStatus(saved+' · '+entries.length+' .riglog file'+(entries.length===1?'':'s')+' · '+mb(blob.size))}
 catch(error){exportStatus(cancelled(error)?'Export cancelled.':'Export failed: '+String(error&&error.message||error),!cancelled(error))}
 finally{setExportBusy(false)}}
loadPayload().then(payload=>{
 D=payload;
 runMap=new Map(D.runs.map(r=>[r.id,r]));
 visible=new Set(D.runs.filter(r=>r.selected).map(r=>r.id));
 captureShell();
 init();
}).catch(error=>{
 // The summary lives in a fold; a load failure must not be hidden behind it.
 const fold=document.getElementById('summary-fold');if(fold)fold.open=true;
 document.getElementById('summary').innerHTML=
  '<div class="empty">Could not load report data: '+esc(String(error&&error.message||error))+'</div>';
});
})();
</script>
</body>
</html>
"""
