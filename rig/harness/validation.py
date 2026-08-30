"""Validation of result events, contracts, and artifact paths."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping

from .errors import ResultValidationError
from .models import ValidationResult


RESULT_PREFIX = "RIG_RESULT="
SCHEMA_VERSION = 1
MAX_RESULT_BYTES = 1_000_000
_ARTIFACT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
FRESH10_DOMAIN_COUNT = 10
FRESH10_TOKENS_PER_DOMAIN = 8_192
FRESH10_TOTAL_TOKENS = FRESH10_DOMAIN_COUNT * FRESH10_TOKENS_PER_DOMAIN
_EVALUATION_REL_TOLERANCE = 1.0e-6
_EVALUATION_ABS_TOLERANCE = 1.0e-9


def parse_result_line(stdout: str) -> dict[str, Any]:
    """Parse the final non-empty stdout line as a v1 result event."""

    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines or not lines[-1].startswith(RESULT_PREFIX):
        raise ResultValidationError(
            f"final non-empty stdout line must begin with {RESULT_PREFIX!r}"
        )
    encoded = lines[-1][len(RESULT_PREFIX) :]
    if not encoded:
        raise ResultValidationError("result event contains no JSON payload")
    if len(encoded.encode("utf-8")) > MAX_RESULT_BYTES:
        raise ResultValidationError("result event is larger than 1 MB")
    try:
        payload = json.loads(encoded)
    except json.JSONDecodeError as exc:
        raise ResultValidationError(f"result event is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ResultValidationError("result payload must be a JSON object")
    return payload


def _finite_number(value: Any, name: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResultValidationError(f"{name} must be a number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ResultValidationError(
            f"{name} must be finite and >= {minimum:g}"
        ) from exc
    if not math.isfinite(number) or number < minimum:
        raise ResultValidationError(f"{name} must be finite and >= {minimum:g}")
    return number


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ResultValidationError(f"{name} must be a positive integer")
    return value


def _plain_object(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ResultValidationError(f"{name} must be a JSON object")
    return value


def _contained_candidate(run_dir: Path, relative: Any) -> tuple[Path, Path, Path]:
    """Resolve a declared artifact path without requiring the leaf to exist."""

    if not isinstance(relative, str) or not relative.strip():
        raise ResultValidationError("checkpoint must be a non-empty relative path")
    candidate = Path(relative)
    if candidate.is_absolute():
        raise ResultValidationError(
            "checkpoint path must be relative to the run directory"
        )
    root = run_dir.resolve()
    unresolved = root / candidate
    resolved = unresolved.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ResultValidationError(
            "checkpoint path escapes the run directory"
        ) from exc
    return root, unresolved, resolved


def _contained_file(
    run_dir: Path, relative: Any, *, allow_missing: bool = False
) -> Path | None:
    """Resolve a contained file, optionally accepting a deliberately absent leaf."""

    _, unresolved, resolved = _contained_candidate(run_dir, relative)
    if not resolved.is_file():
        if allow_missing and not unresolved.exists() and not unresolved.is_symlink():
            return None
        raise ResultValidationError("checkpoint is not a regular file")
    if unresolved.is_symlink():
        raise ResultValidationError("checkpoint may not be a symbolic link")
    # A symlink in a parent is caught by the resolved containment check. Reject a
    # changed target between resolution and hashing as well as practical stdlib can.
    try:
        if not os.path.samefile(resolved, unresolved):
            raise ResultValidationError("checkpoint changed while being validated")
    except OSError as exc:
        raise ResultValidationError("checkpoint could not be inspected") from exc
    return resolved


def contained_file(run_dir: Path, relative: Any) -> Path:
    """Resolve a result artifact without permitting absolute paths or escapes."""

    resolved = _contained_file(run_dir, relative)
    assert resolved is not None
    return resolved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_downstream_contract(
    expected_tokens: Any,
) -> dict[str, int] | None:
    """Normalize an optional caller-owned Fresh10 identity/count contract."""

    token_contract: dict[str, int] | None = None
    if expected_tokens is not None:
        if not isinstance(expected_tokens, Mapping):
            raise ResultValidationError("expected_downstream_tokens must be a mapping")
        token_contract = {}
        for name, count in expected_tokens.items():
            if not isinstance(name, str) or not name or name.strip() != name:
                raise ResultValidationError(
                    "expected_downstream_tokens keys must be non-empty, trimmed strings"
                )
            token_contract[name] = _positive_integer(
                count, f"expected_downstream_tokens[{name!r}]"
            )
        if len(token_contract) != FRESH10_DOMAIN_COUNT:
            raise ResultValidationError(
                f"expected_downstream_tokens must contain exactly {FRESH10_DOMAIN_COUNT} domains"
            )

    return token_contract


def _evaluation_row(
    value: Any,
    name: str,
    *,
    expected_tokens: int | None = None,
    canonical: bool = False,
) -> tuple[dict[str, Any], float, int, float]:
    row = _plain_object(value, name)
    loss = _finite_number(row.get("loss"), f"{name}.loss")
    perplexity = _finite_number(row.get("perplexity"), f"{name}.perplexity")
    if perplexity <= 0.0:
        raise ResultValidationError(f"{name}.perplexity must be greater than zero")
    try:
        expected_perplexity = math.exp(loss)
    except OverflowError as exc:
        raise ResultValidationError(
            f"{name}.loss is too large for finite perplexity"
        ) from exc
    _require_close(perplexity, expected_perplexity, f"{name}.perplexity")
    scored_tokens = _positive_integer(row.get("scored_tokens"), f"{name}.scored_tokens")
    seconds = _finite_number(row.get("seconds"), f"{name}.seconds")
    if canonical and row.get("canonical") is not True:
        raise ResultValidationError(f"{name}.canonical must be true")
    if expected_tokens is not None and scored_tokens != expected_tokens:
        raise ResultValidationError(
            f"{name}.scored_tokens must be exactly {expected_tokens:,}; "
            f"got {scored_tokens:,}"
        )
    return dict(row), loss, scored_tokens, seconds


def _require_close(actual: float, expected: float, name: str) -> None:
    if not math.isclose(
        actual,
        expected,
        rel_tol=_EVALUATION_REL_TOLERANCE,
        abs_tol=_EVALUATION_ABS_TOLERANCE,
    ):
        raise ResultValidationError(
            f"{name} is inconsistent: expected {expected!r}, got {actual!r}"
        )


def _validate_evaluations(
    value: Any,
    *,
    validation_loss: float,
    expected_validation_tokens: int | None,
    expected_downstream_tokens: Any,
) -> dict[str, Any]:
    evaluations = _plain_object(value, "evaluations")
    expected_downstream = _expected_downstream_contract(expected_downstream_tokens)

    fineweb_tokens = expected_validation_tokens
    fineweb, fineweb_loss, _, _ = _evaluation_row(
        evaluations.get("fineweb"),
        "evaluations.fineweb",
        expected_tokens=fineweb_tokens,
        canonical=True,
    )
    del fineweb
    _require_close(fineweb_loss, validation_loss, "evaluations.fineweb.loss")

    if "fresh10" not in evaluations:
        if expected_downstream is not None:
            raise ResultValidationError(
                "evaluations.fresh10 is required by the downstream evaluation contract"
            )
        return json.loads(json.dumps(evaluations, ensure_ascii=False, allow_nan=False))

    fresh10 = _plain_object(evaluations["fresh10"], "evaluations.fresh10")
    domains = _plain_object(fresh10.get("domains"), "evaluations.fresh10.domains")
    if len(domains) != FRESH10_DOMAIN_COUNT:
        raise ResultValidationError(
            "evaluations.fresh10.domains must contain exactly "
            f"{FRESH10_DOMAIN_COUNT} named rows"
        )
    if any(
        not isinstance(name, str) or not name or name.strip() != name
        for name in domains
    ):
        raise ResultValidationError(
            "evaluations.fresh10 domain names must be non-empty, trimmed strings"
        )
    if expected_downstream is not None and set(domains) != set(expected_downstream):
        missing = sorted(set(expected_downstream) - set(domains))
        unexpected = sorted(set(domains) - set(expected_downstream))
        detail = []
        if missing:
            detail.append("missing " + ", ".join(missing))
        if unexpected:
            detail.append("unexpected " + ", ".join(unexpected))
        raise ResultValidationError(
            "evaluations.fresh10 domain names do not match the expected contract"
            + (": " + "; ".join(detail) if detail else "")
        )

    losses: list[float] = []
    domain_seconds: list[float] = []
    scored_total = 0
    for name, row in domains.items():
        expected_tokens = (
            expected_downstream[name]
            if expected_downstream is not None
            else FRESH10_TOKENS_PER_DOMAIN
        )
        _normalized, loss, scored_tokens, seconds = _evaluation_row(
            row,
            f"evaluations.fresh10.domains[{name!r}]",
            expected_tokens=expected_tokens,
        )
        losses.append(loss)
        domain_seconds.append(seconds)
        scored_total += scored_tokens

    macro_loss = _finite_number(
        fresh10.get("macro_loss"), "evaluations.fresh10.macro_loss"
    )
    expected_macro_loss = math.fsum(losses) / FRESH10_DOMAIN_COUNT
    _require_close(macro_loss, expected_macro_loss, "evaluations.fresh10.macro_loss")
    macro_perplexity = _finite_number(
        fresh10.get("macro_perplexity"), "evaluations.fresh10.macro_perplexity"
    )
    if macro_perplexity <= 0.0:
        raise ResultValidationError(
            "evaluations.fresh10.macro_perplexity must be greater than zero"
        )
    try:
        expected_macro_perplexity = math.exp(expected_macro_loss)
    except OverflowError as exc:
        raise ResultValidationError(
            "evaluations.fresh10.macro_loss is too large for finite perplexity"
        ) from exc
    _require_close(
        macro_perplexity,
        expected_macro_perplexity,
        "evaluations.fresh10.macro_perplexity",
    )
    declared_total = _positive_integer(
        fresh10.get("scored_tokens"), "evaluations.fresh10.scored_tokens"
    )
    if declared_total != scored_total:
        raise ResultValidationError(
            "evaluations.fresh10.scored_tokens must equal the sum of domain rows: "
            f"expected {scored_total:,}, got {declared_total:,}"
        )
    declared_seconds = _finite_number(
        fresh10.get("seconds"), "evaluations.fresh10.seconds"
    )
    _require_close(
        declared_seconds,
        math.fsum(domain_seconds),
        "evaluations.fresh10.seconds",
    )

    return json.loads(json.dumps(evaluations, ensure_ascii=False, allow_nan=False))


def validate_result(
    payload: Mapping[str, Any],
    *,
    run_dir: Path,
    expected_training_tokens: int | None = None,
    expected_validation_tokens: int | None = None,
    expected_downstream_tokens: Mapping[str, int] | None = None,
    require_checkpoint: bool = True,
    allow_missing_checkpoint: bool = False,
) -> ValidationResult:
    """Validate a trainer result and, optionally, independently evaluate it."""

    _ensure_json(payload, "result payload")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ResultValidationError(f"schema_version must be {SCHEMA_VERSION}")
    if payload.get("status") != "ok":
        raise ResultValidationError("result status must be 'ok'")
    implementation = payload.get("implementation")
    if implementation is not None and not isinstance(implementation, Mapping):
        raise ResultValidationError("implementation must be a JSON object")
    metrics = _plain_object(payload.get("metrics"), "metrics")
    declared_time = _finite_number(
        metrics.get("train_seconds"), "metrics.train_seconds", minimum=0.0
    )
    if declared_time <= 0:
        raise ResultValidationError("metrics.train_seconds must be greater than zero")
    tokens = _positive_integer(
        metrics.get("tokens_processed"), "metrics.tokens_processed"
    )
    if expected_training_tokens is not None and tokens != expected_training_tokens:
        raise ResultValidationError(
            "metrics.tokens_processed must match the fixed training-token budget: "
            f"expected {expected_training_tokens:,}, got {tokens:,}"
        )
    loss = _finite_number(metrics.get("validation_loss"), "metrics.validation_loss")
    if expected_validation_tokens is not None:
        validation_tokens = _positive_integer(
            metrics.get("validation_tokens"), "metrics.validation_tokens"
        )
        if validation_tokens != expected_validation_tokens:
            raise ResultValidationError(
                "metrics.validation_tokens must match the fixed validation prefix: "
                f"expected {expected_validation_tokens:,}, got {validation_tokens:,}"
            )
    downstream_required = expected_downstream_tokens is not None
    if "evaluations" in payload:
        evaluations = _validate_evaluations(
            payload["evaluations"],
            validation_loss=loss,
            expected_validation_tokens=expected_validation_tokens,
            expected_downstream_tokens=expected_downstream_tokens,
        )
    elif downstream_required:
        # Normalize the caller contract first so malformed expectations fail with
        # the most useful error even when the result omitted evaluations entirely.
        _expected_downstream_contract(expected_downstream_tokens)
        raise ResultValidationError(
            "evaluations are required by the downstream evaluation contract"
        )
    else:
        evaluations = None
    declared_checkpoint = payload.get("checkpoint")
    if declared_checkpoint is None:
        if require_checkpoint:
            raise ResultValidationError("checkpoint is required for this run")
        checkpoint = None
    else:
        checkpoint = _contained_file(
            run_dir,
            declared_checkpoint,
            allow_missing=allow_missing_checkpoint,
        )
    artifact_paths: dict[str, Path] = {}
    artifacts = payload.get("artifacts", {})
    if not isinstance(artifacts, dict):
        raise ResultValidationError("artifacts must be a JSON object")
    for name, relative in artifacts.items():
        if not isinstance(name, str) or not _ARTIFACT_NAME.fullmatch(name):
            raise ResultValidationError(f"invalid artifact name: {name!r}")
        artifact_paths[name] = contained_file(run_dir, relative)

    checkpoint_size = checkpoint.stat().st_size if checkpoint is not None else None
    return ValidationResult(
        payload=dict(payload),
        checkpoint_path=checkpoint,
        checkpoint_sha256=(sha256_file(checkpoint) if checkpoint is not None else None),
        checkpoint_bytes=checkpoint_size,
        declared_train_seconds=declared_time,
        tokens_processed=tokens,
        validation_loss=loss,
        declared_metrics=dict(metrics),
        evaluations=evaluations,
        artifacts=artifact_paths,
    )


def _ensure_json(value: Any, name: str) -> None:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ResultValidationError(
            f"{name} must contain only finite JSON values"
        ) from exc


def verify_run(
    run_dir: Path,
    *,
    expected_training_tokens: int | None = None,
    expected_validation_tokens: int | None = None,
    expected_downstream_tokens: Mapping[str, int] | None = None,
    require_checkpoint: bool = True,
    allow_missing_checkpoint: bool = False,
) -> ValidationResult:
    """Re-validate an existing run from its captured stdout log."""

    stdout_path = run_dir / "stdout.log"
    if not stdout_path.is_file():
        raise ResultValidationError(f"missing captured stdout log: {stdout_path}")
    payload = parse_result_line(
        stdout_path.read_text(encoding="utf-8", errors="replace")
    )
    return validate_result(
        payload,
        run_dir=run_dir,
        expected_training_tokens=expected_training_tokens,
        expected_validation_tokens=expected_validation_tokens,
        expected_downstream_tokens=expected_downstream_tokens,
        require_checkpoint=require_checkpoint,
        allow_missing_checkpoint=allow_missing_checkpoint,
    )
