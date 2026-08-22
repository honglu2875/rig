"""Recipe process lifecycle and immutable record construction."""

from __future__ import annotations

import codecs
import hashlib
import json
import math
import os
import re
import secrets
import selectors
import signal
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Callable, TextIO

from ..cohort import validate_cohort
from ..configfile import profile_config_filename
from ..plan import RecipePlan, validate_recipe_plan
from ..recipe_args import RUNNER_MANAGED_FLAGS
from .cluster import (
    build_distributed_launch_command,
    fetch_run_artifacts,
    pdsh_environment,
    terminate_distributed_workers,
)
from .errors import ConfigurationError, RecipeError, ResultValidationError
from .models import RunConfig, RunOutcome
from .records import append_record
from .validation import (
    parse_result_line,
    sha256_file,
    validate_result,
)


_RECIPE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_PROFILE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
def run_recipe(config: RunConfig) -> RunOutcome:
    """Run, validate, record, and apply checkpoint retention for one recipe.

    This is process isolation for accidental mistakes, not a security sandbox. A
    recipe is trusted local Python code and can access the invoking user's data.
    """

    checked = _validate_config(config)
    (
        repo_root,
        recipe_dir,
        recipe_config,
        runs_dir,
        records_path,
        configured_provenance,
        plan,
        cohort,
    ) = checked
    provenance = _collect_provenance(
        repo_root,
        recipe_dir / "train.py",
        recipe_config,
        configured_provenance,
    )
    run_id = _new_run_id(config.recipe, config.name)
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=False, exist_ok=False)
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    result_path = run_dir / "result.json"

    trainer_command = [
        config.python_executable or sys.executable,
        str(recipe_dir / "train.py"),
        "--output-dir",
        str(run_dir),
        "--seed",
        str(config.seed),
        "--profile",
        config.profile,
        *(["--omit-checkpoint"] if config.checkpoint_policy == "none" else []),
        *[str(argument) for argument in config.trainer_args],
    ]
    configured_environment = {
        str(key): str(value) for key, value in config.environment.items()
    }
    managed_environment = {
        "RIG_RUN_ID": run_id,
        "RIG_OUTPUT_DIR": str(run_dir),
        # Every attempt receives a fresh persistent cache. This keeps cold
        # compilation reproducible and prevents run order from advantaging
        # later recipes.
        "JAX_COMPILATION_CACHE_DIR": str(run_dir / ".jax_cache"),
        "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
        "PYTHONUNBUFFERED": "1",
    }
    environment = os.environ.copy()
    environment.update(configured_environment)
    environment.update(managed_environment)
    # Where the work runs and whether it is a multi-process job are separate
    # questions. A remote single host (v6e-8, say) launches over pdsh but takes
    # no distributed init; an in-slice v4-32 does both. Keying everything on
    # tpu_vm_count would run a remote single-host job on this machine, which
    # has no accelerator.
    launch_remotely = config.tpu_vm_count > 1 or config.remote_controller
    distributed = config.tpu_vm_count > 1
    if launch_remotely:
        remote_environment = {
            **configured_environment,
            **managed_environment,
            # Peer filesystems are independent. Keep their fresh compilation
            # caches ephemeral instead of leaving shadow run directories.
            "JAX_COMPILATION_CACHE_DIR": f"/tmp/rig-jax-cache-{run_id}",
            "RIG_CLUSTER_WORKER": "1",
            # Whoever owns the artifacts announces itself here. In the
            # ordinary setup that is this machine; under a remote
            # controller it is a peer, and this machine is not in the
            # slice at all -- announcing gethostname() there would match
            # no worker and every process would decline to write results.
            "RIG_CONTROLLER_HOSTNAME": (
                config.artifact_hostname or socket.gethostname()
            ),
        }
        if distributed:
            remote_environment["RIG_DISTRIBUTED"] = "1"
            remote_environment["RIG_PROCESS_COUNT"] = str(config.tpu_vm_count)
        command = build_distributed_launch_command(
            host_expression=config.tpu_vm_hosts,
            host_count=config.tpu_vm_count,
            cwd=recipe_dir,
            command=trainer_command,
            environment=remote_environment,
        )
        environment = pdsh_environment(environment)
    else:
        command = trainer_command
    started_at = datetime.now(timezone.utc)
    monotonic_start = time.perf_counter()

    def clean_distributed_workers() -> None:
        if not launch_remotely:
            return
        cleaned = terminate_distributed_workers(
            host_expression=config.tpu_vm_hosts,
            host_count=config.tpu_vm_count,
            executable=Path(trainer_command[0]),
            script=Path(trainer_command[1]),
            output_dir=run_dir,
            environment=environment,
        )
        if not cleaned:
            print(
                "warning: could not verify distributed worker cleanup; check the "
                "configured TPU VM hosts before starting another run",
                file=sys.stderr,
            )

    try:
        with (
            stdout_path.open("wb") as stdout_handle,
            stderr_path.open("wb") as stderr_handle,
        ):
            return_code, timed_out = _run_process(
                command,
                cwd=recipe_dir,
                environment=environment,
                stdout_handle=stdout_handle,
                stderr_handle=stderr_handle,
                timeout_seconds=float(config.timeout_seconds),
                tick=_artifact_puller(config, run_dir, environment),
                tick_seconds=_PULL_INTERVAL_SECONDS,
            )
    except BaseException:
        clean_distributed_workers()
        raise
    if timed_out or return_code != 0:
        clean_distributed_workers()
    observed_seconds = time.perf_counter() - monotonic_start
    finished_at = datetime.now(timezone.utc)
    _discard_compilation_cache(run_dir / ".jax_cache")

    stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace")
    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
    if timed_out:
        raise RecipeError(
            f"recipe timed out after {config.timeout_seconds:g}s; logs: {run_dir}"
        )
    if return_code != 0:
        tail = _tail(stderr_text)
        detail = f" ({tail})" if tail else ""
        raise RecipeError(
            f"recipe exited with status {return_code}{detail}; logs: {run_dir}"
        )

    if config.remote_controller and config.artifact_host:
        # The artifact host wrote everything to its own disk. Nothing below
        # can read the run until it is here, so pull before validating.
        fetch_run_artifacts(
            config.artifact_host,
            run_dir,
            run_dir,
            environment=environment,
        )

    payload = parse_result_line(stdout_text)
    _validate_payload_identity(payload, config)
    validated = validate_result(
        payload,
        run_dir=run_dir,
        expected_training_tokens=plan.expected_tokens,
        expected_validation_tokens=config.expected_validation_tokens,
        expected_downstream_tokens=config.expected_downstream_tokens,
        require_checkpoint=config.checkpoint_policy != "none",
    )
    # Preserve the exact accepted payload independently from potentially noisy logs.
    result_path.write_text(
        json.dumps(
            payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )
    relative_checkpoint = (
        validated.checkpoint_path.relative_to(run_dir.resolve()).as_posix()
        if validated.checkpoint_path is not None
        else None
    )
    contract = (
        payload.get("contract") if isinstance(payload.get("contract"), dict) else None
    )
    qualified = validated.validation_loss <= float(config.target_loss)
    recorded_metrics = dict(validated.declared_metrics)
    # These normalized values, rather than potentially surprising numeric JSON
    # representations in the raw payload, are the canonical scoring fields.
    recorded_metrics.update(
        {
            "train_seconds": validated.declared_train_seconds,
            "tokens_processed": validated.tokens_processed,
            "validation_loss": validated.validation_loss,
        }
    )
    record: dict[str, Any] = {
        "record_version": 1,
        "run_id": run_id,
        "status": "ok",
        "qualified": qualified,
        "recipe": config.recipe,
        "name": config.name,
        # Kept as a constant for readers of historical records. Competition
        # tracks are no longer an experiment axis.
        "track": "open",
        "profile": config.profile,
        "run_kind": plan.run_kind,
        "seed": config.seed,
        "timestamps": {
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
        },
        "target_loss": float(config.target_loss),
        "constraints": {
            "training_tokens": plan.expected_tokens,
            "validation_tokens": config.expected_validation_tokens,
        },
        "timing": {"observed_wall_seconds": observed_seconds},
        "metrics": recorded_metrics,
        "contract": contract,
        "implementation": (
            dict(payload["implementation"])
            if isinstance(payload.get("implementation"), dict)
            else None
        ),
        "system": payload.get("system"),
        "plan": {**plan.as_dict(), "sha256": plan.sha256},
        "cohort": cohort,
        "cohort_id": cohort.get("cohort_id") if cohort is not None else None,
        "checkpoint": (
            {
                "path": relative_checkpoint,
                "sha256": validated.checkpoint_sha256,
                "bytes": validated.checkpoint_bytes,
                "retained": True,
            }
            if validated.checkpoint_path is not None
            else None
        ),
        "artifacts": {
            name: {
                "path": path.relative_to(run_dir.resolve()).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for name, path in validated.artifacts.items()
        },
        "logs": {
            "stdout": "stdout.log",
            "stdout_sha256": _sha256_bytes(stdout_path.read_bytes()),
            "stderr": "stderr.log",
            "stderr_sha256": _sha256_bytes(stderr_path.read_bytes()),
        },
        "command": command,
        "trainer_command": trainer_command,
        "provenance": provenance,
    }
    if validated.evaluations is not None:
        # Validation returns a JSON round-trip copy, so the immutable record keeps
        # the entire accepted evaluation block without retaining caller aliases.
        record["evaluations"] = dict(validated.evaluations)

    keep_checkpoint = config.checkpoint_policy == "always" or (
        config.checkpoint_policy == "qualifying" and qualified
    )
    checkpoint_path: Path | None = validated.checkpoint_path
    if checkpoint_path is not None and not keep_checkpoint:
        validated.checkpoint_path.unlink()
        checkpoint_path = None
        assert isinstance(record["checkpoint"], dict)
        record["checkpoint"]["retained"] = False
    append_record(records_path, record)
    return RunOutcome(
        run_id=run_id,
        run_dir=run_dir,
        record=record,
        record_path=records_path,
        checkpoint_path=checkpoint_path,
    )


class _LiveStderr:
    """Best-effort byte-preserving stderr output, including redirected text streams."""

    def __init__(self, stream: TextIO | None) -> None:
        self._stream = stream
        self._binary = getattr(stream, "buffer", None) if stream is not None else None
        self._decoder = (
            codecs.getincrementaldecoder("utf-8")(errors="replace")
            if stream is not None and self._binary is None
            else None
        )
        self._enabled = stream is not None

    def write(self, value: bytes) -> None:
        if not self._enabled:
            return
        try:
            if self._binary is not None:
                self._binary.write(value)
                self._binary.flush()
            elif self._decoder is not None and self._stream is not None:
                text = self._decoder.decode(value)
                if text:
                    self._stream.write(text)
                    self._stream.flush()
        except (OSError, TypeError, UnicodeError, ValueError):
            # A closed terminal or a test capture stream must not invalidate an
            # otherwise successful benchmark. The file capture remains canonical.
            self._enabled = False

    def finish(self) -> None:
        if not self._enabled or self._decoder is None or self._stream is None:
            return
        try:
            tail = self._decoder.decode(b"", final=True)
            if tail:
                self._stream.write(tail)
            self._stream.flush()
        except (OSError, TypeError, UnicodeError, ValueError):
            self._enabled = False


# Opportunistic salvage cadence. On preemptible hardware the job can vanish at
# any step, so partial artifacts are pulled while it runs rather than only on
# success. Bounded loss, negligible cost: these are small text files.
_PULL_INTERVAL_SECONDS = 60.0


def _artifact_puller(
    config: RunConfig, run_dir: Path, environment: Mapping[str, str]
) -> Callable[[], None] | None:
    """Return a best-effort puller, or None when artifacts are already local."""

    if not (config.remote_controller and config.artifact_host):
        return None

    def pull() -> None:
        try:
            fetch_run_artifacts(
                config.artifact_host, run_dir, run_dir, environment=environment
            )
        except Exception:
            # A failed opportunistic pull is not a failed run. The next tick
            # retries, and the authoritative pull still runs on completion.
            return

    return pull


def _run_process(
    command: list[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    stdout_handle: BinaryIO,
    stderr_handle: BinaryIO,
    timeout_seconds: float,
    tick: Callable[[], None] | None = None,
    tick_seconds: float = 60.0,
) -> tuple[int | None, bool]:
    """Capture stdout and tee stderr without allowing either pipe to block the child.

    ``tick`` runs at most every ``tick_seconds`` from the same loop that drains
    the pipes, which is where opportunistic artifact pulls happen. It must not
    raise and must not block for long: this loop also enforces the timeout.
    """

    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=environment,
        stdout=stdout_handle,
        stderr=subprocess.PIPE,
        start_new_session=True,
        bufsize=0,
    )
    assert process.stderr is not None  # stderr=subprocess.PIPE
    stderr_pipe = process.stderr
    descriptor = stderr_pipe.fileno()
    os.set_blocking(descriptor, False)
    selector = selectors.DefaultSelector()
    selector.register(stderr_pipe, selectors.EVENT_READ)
    live_stderr = _LiveStderr(sys.stderr)
    deadline = time.perf_counter() + timeout_seconds
    timed_out = False
    next_tick = time.perf_counter() + tick_seconds

    try:
        while True:
            return_code = process.poll()
            remaining = deadline - time.perf_counter()
            if return_code is None and remaining <= 0:
                timed_out = True
                _kill_process_group(process)
                return_code = process.wait()

            wait_seconds = (
                0.0 if return_code is not None else min(0.1, max(0.0, remaining))
            )
            for _key, _events in selector.select(wait_seconds):
                _drain_stderr(descriptor, stderr_handle, live_stderr)

            if tick is not None and time.perf_counter() >= next_tick:
                next_tick = time.perf_counter() + tick_seconds
                tick()

            return_code = process.poll()
            if return_code is not None:
                # poll() observing process exit means all writes by the direct child
                # are already available. A bounded final drain avoids hanging if a
                # stray descendant inherited the pipe and remains alive.
                _drain_stderr(descriptor, stderr_handle, live_stderr)
                return (None if timed_out else return_code), timed_out
    finally:
        if process.poll() is None:
            _kill_process_group(process)
            process.wait()
        selector.close()
        stderr_pipe.close()
        live_stderr.finish()


def _drain_stderr(
    descriptor: int,
    stderr_handle: BinaryIO,
    live_stderr: _LiveStderr,
    *,
    max_chunks: int = 64,
) -> None:
    for _ in range(max_chunks):
        try:
            chunk = os.read(descriptor, 64 * 1024)
        except BlockingIOError:
            return
        if not chunk:
            return
        stderr_handle.write(chunk)
        stderr_handle.flush()
        live_stderr.write(chunk)


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass
    if process.poll() is None:
        try:
            process.kill()
        except ProcessLookupError:
            pass


def _validate_payload_identity(payload: Mapping[str, Any], config: RunConfig) -> None:
    expected = {"profile": config.profile, "seed": config.seed}
    for name, expected_value in expected.items():
        actual = payload.get(name)
        if (
            name not in payload
            or type(actual) is not type(expected_value)
            or actual != expected_value
        ):
            raise ResultValidationError(
                f"result {name} must exactly match the run configuration: "
                f"expected {expected_value!r}, got {actual!r}"
            )
    if config.profile == "official":
        _validate_official_system(
            payload.get("system"), expected_process_count=config.tpu_vm_count
        )


def _validate_official_system(value: Any, *, expected_process_count: int = 1) -> None:
    if not isinstance(value, dict):
        raise ResultValidationError("official result system must be a JSON object")
    expected_scalars = {
        "platform": "tpu",
        "device_count": 4 * expected_process_count,
        "local_device_count": 4,
        "process_count": expected_process_count,
    }
    for name, expected in expected_scalars.items():
        actual = value.get(name)
        if type(actual) is not type(expected) or actual != expected:
            raise ResultValidationError(
                f"official result system.{name} must be {expected!r}; got {actual!r}"
            )
    kinds = value.get("device_kinds")
    if kinds != ["TPU v4"]:
        raise ResultValidationError(
            "official result system.device_kinds must be exactly ['TPU v4']"
        )


def _validate_config(
    config: RunConfig,
) -> tuple[
    Path,
    Path,
    Path,
    Path,
    Path,
    dict[str, Any],
    RecipePlan,
    dict[str, Any] | None,
]:
    if not _RECIPE_NAME.fullmatch(config.recipe):
        raise ConfigurationError(
            "recipe must be a simple name containing only letters, digits, '.', '_' or '-'"
        )
    if not _PROFILE_NAME.fullmatch(config.profile):
        raise ConfigurationError("profile must be a non-empty simple name")
    if config.checkpoint_policy not in ("always", "qualifying", "none"):
        raise ConfigurationError("invalid checkpoint policy")
    if config.checkpoint_policy == "none" and config.profile != "dev":
        raise ConfigurationError(
            "checkpoint omission is restricted to development research runs"
        )
    if (
        isinstance(config.tpu_vm_count, bool)
        or not isinstance(config.tpu_vm_count, int)
        or config.tpu_vm_count <= 0
    ):
        raise ConfigurationError("tpu_vm_count must be a positive integer")
    if not isinstance(config.tpu_vm_hosts, str):
        raise ConfigurationError("tpu_vm_hosts must be a string")
    if config.tpu_vm_count > 1 and not config.tpu_vm_hosts.strip():
        raise ConfigurationError(
            "tpu_vm_hosts is required when tpu_vm_count is greater than 1"
        )
    if any(character.isspace() for character in config.tpu_vm_hosts) or any(
        character in config.tpu_vm_hosts for character in "\x00\r\n"
    ):
        raise ConfigurationError("tpu_vm_hosts must be a whitespace-free single line")
    if (
        isinstance(config.seed, bool)
        or not isinstance(config.seed, int)
        or config.seed < 0
    ):
        raise ConfigurationError("seed must be a non-negative integer")
    timeout_seconds = _finite_config_number(config.timeout_seconds)
    if timeout_seconds is None or timeout_seconds <= 0:
        raise ConfigurationError("timeout_seconds must be greater than zero")
    if isinstance(config.trainer_args, (str, bytes)) or not isinstance(
        config.trainer_args, Sequence
    ):
        raise ConfigurationError("trainer_args must be a sequence of arguments")
    target_loss = _finite_config_number(config.target_loss)
    if target_loss is None or target_loss < 0:
        raise ConfigurationError("target_loss must be a finite non-negative number")
    for argument in config.trainer_args:
        rendered = str(argument)
        if "\x00" in rendered:
            raise ConfigurationError("trainer arguments may not contain NUL bytes")
        for flag in RUNNER_MANAGED_FLAGS:
            if rendered == flag or rendered.startswith(flag + "="):
                raise ConfigurationError(
                    f"trainer arguments may not override reserved flag {flag}"
                )
    configured_provenance = _copy_finite_mapping(config.provenance, "provenance")
    try:
        plan = validate_recipe_plan(config.plan)
    except Exception as exc:
        raise ConfigurationError(str(exc)) from exc
    if plan.payload["profile"] != config.profile:
        raise ConfigurationError(
            "recipe plan profile must exactly match the run configuration"
        )
    if config.cohort is None:
        cohort = None
    else:
        try:
            cohort = validate_cohort(config.cohort)
        except Exception as exc:
            raise ConfigurationError(str(exc)) from exc
    _validate_cohort_alignment(
        cohort,
        plan,
        target_loss=float(target_loss),
        tpu_vm_count=config.tpu_vm_count,
    )

    repo_root = config.repo_root.resolve()
    if not repo_root.is_dir():
        raise ConfigurationError(f"repository root does not exist: {repo_root}")
    recipes_root = (repo_root / "recipes").resolve()
    recipe_dir = (recipes_root / config.recipe).resolve()
    try:
        recipe_dir.relative_to(recipes_root)
    except ValueError as exc:  # defensive in addition to name regex
        raise ConfigurationError("recipe path escapes recipes directory") from exc
    trainer = recipe_dir / "train.py"
    if not trainer.is_file() or trainer.is_symlink():
        raise ConfigurationError(f"recipe entry script not found: {trainer}")
    recipe_config = recipe_dir / profile_config_filename(config.profile)
    if not recipe_config.is_file() or recipe_config.is_symlink():
        raise ConfigurationError(
            f"recipe configuration file not found: {recipe_config}"
        )
    actual_config_sha256 = sha256_file(recipe_config)
    if plan.payload["config_sha256"] != actual_config_sha256:
        raise ConfigurationError(
            "recipe plan config_sha256 does not match the current sibling "
            f"{recipe_config.name}: expected {actual_config_sha256}, "
            f"got {plan.payload['config_sha256']}"
        )

    runs_dir = _resolve_managed_path(
        repo_root, config.runs_dir, "runs_dir", directory=True
    )
    records_path = _resolve_managed_path(repo_root, config.records_path, "records_path")
    runs_dir.mkdir(parents=True, exist_ok=True)
    records_path.parent.mkdir(parents=True, exist_ok=True)
    if config.expected_validation_tokens is not None and (
        isinstance(config.expected_validation_tokens, bool)
        or not isinstance(config.expected_validation_tokens, int)
        or config.expected_validation_tokens <= 0
    ):
        raise ConfigurationError(
            "expected_validation_tokens must be a positive integer"
        )
    if config.expected_downstream_tokens is not None:
        if not isinstance(config.expected_downstream_tokens, Mapping):
            raise ConfigurationError("expected_downstream_tokens must be a mapping")
        if len(config.expected_downstream_tokens) != 10:
            raise ConfigurationError(
                "expected_downstream_tokens must contain exactly 10 domains"
            )
        for name, count in config.expected_downstream_tokens.items():
            if not isinstance(name, str) or not name or name.strip() != name:
                raise ConfigurationError(
                    "expected_downstream_tokens keys must be non-empty, trimmed strings"
                )
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise ConfigurationError(
                    f"expected_downstream_tokens[{name!r}] must be a positive integer"
                )
    return (
        repo_root,
        recipe_dir,
        recipe_config,
        runs_dir,
        records_path,
        configured_provenance,
        plan,
        cohort,
    )


def _validate_cohort_alignment(
    cohort: Mapping[str, Any] | None,
    plan: RecipePlan,
    *,
    target_loss: float,
    tpu_vm_count: int,
) -> None:
    if cohort is None:
        return
    if plan.run_kind != "full":
        raise ConfigurationError("only a full recipe plan may carry a cohort")

    payload = plan.payload
    expected_fields = {
        "profile": payload["profile"],
        "tier": payload["tier"],
        "declared_parameters": payload["declared_parameters"],
    }
    for name, expected in expected_fields.items():
        if cohort[name] != expected:
            raise ConfigurationError(
                f"cohort {name} must match the recipe plan: "
                f"expected {expected!r}, got {cohort[name]!r}"
            )

    expected_tpp = format(float(payload["target_tokens_per_parameter"]), ".15g")
    actual_tpp = cohort["horizon"]["target_tokens_per_parameter"]
    if actual_tpp != expected_tpp:
        raise ConfigurationError(
            "cohort target_tokens_per_parameter must match the recipe plan: "
            f"expected {expected_tpp!r}, got {actual_tpp!r}"
        )
    expected_loss = format(target_loss, ".15g")
    actual_loss = cohort["qualification"]["target_loss"]
    if actual_loss != expected_loss:
        raise ConfigurationError(
            "cohort target_loss must match the run configuration: "
            f"expected {expected_loss!r}, got {actual_loss!r}"
        )
    actual_hosts = cohort["hardware"]["tpu_vm_count"]
    if actual_hosts != tpu_vm_count:
        raise ConfigurationError(
            "cohort tpu_vm_count must match the run configuration: "
            f"expected {tpu_vm_count}, got {actual_hosts}"
        )


def _copy_finite_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{label} must be a mapping")
    try:
        encoded = json.dumps(
            dict(value), sort_keys=True, ensure_ascii=False, allow_nan=False
        )
        copied = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(
            f"{label} must contain only finite JSON values: {exc}"
        ) from exc
    if not isinstance(copied, dict):  # defensive: the input was already a mapping
        raise ConfigurationError(f"{label} must encode as a JSON object")
    return copied


def _collect_provenance(
    repo_root: Path,
    trainer: Path,
    recipe_config: Path,
    configured: Mapping[str, Any],
) -> dict[str, Any]:
    owned_keys = {"train_py", "config_yaml", "shared_python", "uv_lock", "git"}
    collisions = owned_keys.intersection(configured)
    if collisions:
        raise ConfigurationError(
            "provenance may not override harness-owned keys: "
            + ", ".join(sorted(collisions))
        )
    provenance: dict[str, Any] = {
        **dict(configured),
        "train_py": _file_provenance(repo_root, trainer),
        "config_yaml": _file_provenance(repo_root, recipe_config),
        "shared_python": _python_tree_provenance(repo_root),
        "uv_lock": None,
        "git": _git_provenance(repo_root),
    }
    lockfile = repo_root / "uv.lock"
    if lockfile.is_file():
        provenance["uv_lock"] = _file_provenance(repo_root, lockfile)
    return provenance


def _python_tree_provenance(repo_root: Path) -> dict[str, Any]:
    """Hash shared Python dependencies, including dirty working-tree bytes."""

    paths: list[Path] = []
    for package in ("rig",):
        package_root = repo_root / package
        if package_root.is_dir():
            paths.extend(path for path in package_root.rglob("*.py") if path.is_file())
    paths.sort(key=lambda item: item.relative_to(repo_root).as_posix())
    digest = hashlib.sha256()
    entries: list[dict[str, Any]] = []
    total_bytes = 0
    for path in paths:
        item = _file_provenance(repo_root, path)
        digest.update(str(item["path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(item["sha256"]).encode("ascii"))
        digest.update(b"\0")
        entries.append(item)
        total_bytes += int(item["bytes"])
    return {
        "sha256": digest.hexdigest(),
        "files": len(entries),
        "bytes": total_bytes,
        "entries": entries,
    }


def _file_provenance(repo_root: Path, path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
    except OSError as exc:
        raise ConfigurationError(
            f"could not hash provenance file {path}: {exc}"
        ) from exc
    try:
        relative_path = path.relative_to(repo_root).as_posix()
    except ValueError:
        relative_path = str(path)
    return {"path": relative_path, "sha256": digest.hexdigest(), "bytes": size}


def _git_provenance(repo_root: Path) -> dict[str, Any] | None:
    head = _git_output(repo_root, "rev-parse", "--verify", "HEAD")
    status = _git_output(repo_root, "status", "--porcelain=v1", "--untracked-files=all")
    if head is None and status is None:
        return None
    result: dict[str, Any] = {}
    if head is not None:
        result["head"] = head.decode("ascii", errors="replace").strip()
    if status is not None:
        result.update(
            {
                "dirty": bool(status),
                "status_porcelain": status.decode("utf-8", errors="replace"),
                "status_porcelain_sha256": _sha256_bytes(status),
            }
        )
    return result


def _git_output(repo_root: Path, *arguments: str) -> bytes | None:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout if completed.returncode == 0 else None


def _finite_config_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None


def _discard_compilation_cache(cache_path: Path) -> None:
    """Remove only the harness-owned per-run compilation cache, best effort."""

    try:
        if cache_path.is_symlink() or cache_path.is_file():
            cache_path.unlink()
        elif cache_path.is_dir():
            shutil.rmtree(cache_path)
    except OSError:
        # Cache cleanup must not replace the real recipe outcome. The run
        # directory remains available for manual cleanup if the filesystem refuses.
        pass


def _resolve_managed_path(
    repo_root: Path, configured: Path, label: str, *, directory: bool = False
) -> Path:
    path = configured if configured.is_absolute() else repo_root / configured
    resolved = path.resolve()
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise ConfigurationError(
            f"{label} must be contained in the repository"
        ) from exc
    if not directory and resolved == repo_root:
        raise ConfigurationError(f"{label} must name a file below the repository root")
    return resolved


def _new_run_id(recipe: str, name: str = "") -> str:
    """Compose a sortable, unique, and -- when named -- readable run directory."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    parts = [timestamp, recipe]
    if name:
        parts.append(name)
    parts.append(secrets.token_hex(4))
    return "-".join(parts)


def _tail(text: str, limit: int = 240) -> str:
    compact = " ".join(text.strip().split())
    return compact[-limit:]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
