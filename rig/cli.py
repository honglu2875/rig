"""Friendly command-line front end for preparation, runs, and leaderboards."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
from typing import Any, Callable, Iterable, Sequence

from .arguments import (
    CHECKPOINT_POLICIES as _CHECKPOINT_POLICIES,
    COLORS as _COLORS,
    PROFILES as _PROFILES,
    build_parser,
)
from .harness import (
    HarnessError,
    normalize_run_name,
    RunConfig,
    load_records,
    rank_records,
    render_leaderboard,
    run_recipe,
    verify_run,
)
from .cohort import CohortError, build_cohort, validate_cohort
from .plan import RecipePlan, resolve_recipe_plan
from .recipe_args import RIG_MANAGED_RECIPE_FLAGS
from .harness.cluster import (
    ClusterError,
    ClusterInventory,
    bootstrap_uv,
    infer_host_expression,
    prepare_ram_cache,
    probe_cluster,
    run_pdsh,
    seal_ram_cache_command,
    RAM_CACHE_ROOT,
    ship_dataset,
    ship_uv_binary,
    ship_uv_cache,
    ship_uv_python,
    sync_workspace,
)

from .config import (
    ConfigError,
    LocalConfig,
    config_path,
    load_config,
    repo_root,
    resolve_path,
    save_config,
    with_overrides,
)
from .configfile import PROFILE_CONFIG_FILENAMES, profile_config_filename
from .data import (
    DataError,
    FRESH10_DOMAINS,
    PreparedFresh10,
    PreparedDataset,
    prepare as prepare_data,
    prepare_fresh10,
    sha256_file,
    verify_dataset,
    verify_fresh10,
)
from .data_routing import (
    SCALED_CACHE_SUBDIRECTORY,
    dataset_names,
    named_preparation_route,
    resolve_preparation_manifest,
    smoke_preparation_route,
)
from .doctor import (
    doctor_ok,
    environment_checks,
    render_doctor,
    run_doctor,
)
from .console import Console
from .report import build_report, export_study
from .rules import OFFICIAL_TARGET_LOSS, OFFICIAL_VALIDATION_PREDICTIONS


_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
# What happens to model weights. Deliberately named for the decision rather
# than the outcome, so future options (a checkpointing frequency, keeping the
# last N) extend this axis instead of adding another flag beside it.


def _checkpoint_policy(args: argparse.Namespace, fallback: str, *, profile: str) -> str:
    """Resolve the policy from the flag, falling back to saved settings.

    A saved default of ``none`` means "do not keep sweep weights", which is
    about research runs. It must not silently refuse an official run, whose
    checkpoint is required, so the fallback yields to ``qualifying`` where
    ``none`` is not legal. An *explicit* ``--checkpoint-policy none`` there is
    still an error: that is the caller asking for something disallowed, rather
    than a default reaching somewhere it was not meant to.
    """

    explicit = getattr(args, "checkpoint_policy", None)
    chosen = explicit or fallback
    if chosen not in _CHECKPOINT_POLICIES:
        raise ConfigError(
            f"unknown checkpoint policy {chosen!r}; expected one of "
            + ", ".join(_CHECKPOINT_POLICIES)
        )
    if explicit is None and chosen == "none" and not _is_research_run(profile):
        return "qualifying"
    return chosen


def _is_research_run(profile: str) -> bool:
    """Whether weights may be skipped entirely for this run."""

    return profile == "dev"


_CLUSTER_WORKER_ENV = "RIG_CLUSTER_WORKER"
_CONTROLLER_HOST_ENV = "RIG_CONTROLLER_HOSTNAME"
_DISTRIBUTED_ENV = "RIG_DISTRIBUTED"
_PROCESS_COUNT_ENV = "RIG_PROCESS_COUNT"
_RECIPE_ARGUMENT_COMMANDS = frozenset({"run", "profile"})
_RECIPE_HELP_ARGUMENTS = frozenset({("-h",), ("--help",)})


class Style:
    CODES = {
        "reset": "\033[0m",
        "bold": "\033[1m",
        "dim": "\033[2m",
        "cyan": "\033[38;5;81m",
        "blue": "\033[38;5;75m",
        "green": "\033[38;5;114m",
        "yellow": "\033[38;5;221m",
        "magenta": "\033[38;5;176m",
        "red": "\033[38;5;203m",
    }

    def __init__(self, mode: str = "auto") -> None:
        self.enabled = mode == "always" or (
            mode == "auto" and sys.stdout.isatty() and "NO_COLOR" not in os.environ
        )

    def text(self, value: object, *styles: str) -> str:
        raw = str(value)
        if not self.enabled or not styles:
            return raw
        return "".join(self.CODES[item] for item in styles) + raw + self.CODES["reset"]

    def banner(self, subtitle: str) -> None:
        print(
            f"\n  {self.text('◆', 'magenta', 'bold')}"
            f"{self.text(' GPT TPU RIG ', 'bold')}"
            f"{self.text(subtitle, 'cyan')}\n",
            flush=True,
        )

    def heading(self, value: str) -> None:
        print(f"\n  {self.text('●', 'magenta')} {self.text(value, 'bold')}", flush=True)

    def ok(self, value: str) -> None:
        print(f"  {self.text('✓', 'green', 'bold')} {value}", flush=True)

    def note(self, value: str) -> None:
        print(f"  {self.text('→', 'cyan')} {value}", flush=True)


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    arguments = list(argv) if argv is not None else sys.argv[1:]
    args = _parse_arguments(parser, arguments)
    try:
        if (
            args.command in _RECIPE_ARGUMENT_COMMANDS
            and args.recipe_args in _RECIPE_HELP_ARGUMENTS
        ):
            return command_recipe_help(args)
        if args.command == "prepare":
            return command_prepare(args)
        if args.command == "doctor":
            return command_doctor(args)
        if args.command == "run":
            return command_run(args)
        if args.command == "profile":
            return command_profile(args)
        if args.command == "verify":
            return command_verify(args)
        if args.command == "dataset":
            return command_dataset(args)
        if args.command == "leaderboard":
            return command_leaderboard(args)
        if args.command == "report":
            return command_report(args)
        if args.command == "clone":
            return command_clone(args)
        if args.command == "settings":
            return command_settings(args)
        parser.error(f"unknown command {args.command!r}")
    except (
        ClusterError,
        ConfigError,
        DataError,
        HarnessError,
        OSError,
        ValueError,
    ) as exc:
        style = Style(getattr(args, "color", None) or "auto")
        print(f"\n  {style.text('error:', 'red', 'bold')} {exc}\n", file=sys.stderr)
        return 1
    return 0


def _parse_arguments(
    parser: argparse.ArgumentParser,
    arguments: Sequence[str],
) -> argparse.Namespace:
    """Parse the public CLI and isolate an explicit recipe-local ``--`` tail."""

    public_arguments = list(arguments)
    recipe_arguments: tuple[str, ...] = ()
    if (
        public_arguments
        and public_arguments[0] in _RECIPE_ARGUMENT_COMMANDS
        and "--" in public_arguments
    ):
        boundary = public_arguments.index("--")
        recipe_arguments = tuple(public_arguments[boundary + 1 :])
        if not recipe_arguments:
            parser.error(
                f"rig {public_arguments[0]} -- must be followed by recipe-local arguments"
            )
        public_arguments = public_arguments[:boundary]
    args = parser.parse_args(public_arguments)
    if args.command in _RECIPE_ARGUMENT_COMMANDS:
        args.recipe_args = recipe_arguments
    return args


def _work_runs_elsewhere(config: LocalConfig) -> bool:
    """Whether the accelerators are on other machines rather than this one.

    Distinct from "is this a multi-process job". A remote single host (a v6e-8,
    say) needs the full fan-out -- source sync, uv, data, remote launch -- but
    takes no distributed initialization. Gating that work on tpu_vm_count alone
    silently skipped every step for such a host and then ran the trainer here,
    where there is no accelerator.
    """

    return config.tpu_vm_count > 1 or config.remote_controller


def command_prepare(args: argparse.Namespace) -> int:
    root = repo_root()
    current = load_config(root, cluster=getattr(args, "cluster", None))
    train_shards_override = args.train_shards
    if (
        args.dataset is not None
        and args.dataset != current.dataset
        and train_shards_override is None
    ):
        # A prefix count belongs to one manifest. Selecting another corpus
        # without an explicit count means that corpus's full published prefix,
        # not the previous corpus's coincidentally numbered prefix.
        train_shards_override = 0
    proposed = with_overrides(
        current,
        {
            "data_path": str(args.path) if args.path is not None else None,
            "artifacts_path": str(args.artifacts)
            if args.artifacts is not None
            else None,
            "active_cluster": getattr(args, "cluster", None),
            "tpu_vm_count": args.tpu_vm_count,
            "tpu_vm_hosts": args.tpu_vm_hosts,
            "default_profile": args.run_profile,
            "checkpoint_policy": args.checkpoint_policy,
            "color": args.color,
            "target_loss": args.target_loss,
            "dataset": args.dataset,
            "train_shards": train_shards_override,
        },
    )
    preparation_type = args.profile or proposed.default_profile
    interactive = not (args.non_interactive or args.yes)
    if interactive and not sys.stdin.isatty():
        raise ConfigError(
            "prepare needs a terminal for its wizard; pass --non-interactive with explicit flags"
        )
    run_diagnostics = not args.no_doctor
    data_work = not args.no_download
    require_tpu = proposed.default_profile == "official"
    save = not args.no_save
    if interactive:
        (
            proposed,
            preparation_type,
            run_diagnostics,
            require_tpu,
            data_work,
            save,
        ) = _prepare_wizard(
            proposed,
            preparation_type=preparation_type,
            run_diagnostics=run_diagnostics,
            require_tpu=require_tpu,
            download=data_work,
            save=save,
        )

    style = Style(proposed.color)
    style.banner("prepare")
    data_path = resolve_path(proposed.data_path, root)
    artifacts_path = resolve_path(proposed.artifacts_path, root)
    _ensure_artifacts_inside_repo(artifacts_path, root)
    route = _route_for_config(proposed, preparation_type)
    route_root = route.data_root(data_path)
    route_manifest = (
        resolve_preparation_manifest(route) if data_work else route.manifest
    )
    style.note(route.summary())
    if route.is_scaled:
        style.note(f"scaled shards use the dedicated cache {route_root}")
    if args.check_only and save:
        style.note("check-only mode does not write .rig.toml")
        save = False
    if save:
        destination = save_config(proposed, root)
        style.ok(f"saved personal defaults to {destination.relative_to(root)}")
    else:
        style.note("settings are temporary (--no-save)")

    cluster_controller = _work_runs_elsewhere(proposed) and not _is_cluster_worker()
    inventory: ClusterInventory | None = None
    if cluster_controller:
        inventory = _prepare_cluster(
            proposed,
            args,
            root=root,
            artifacts_path=artifacts_path,
            style=style,
        )

    if run_diagnostics and not cluster_controller:
        style.heading("Machine diagnostics")
        results = run_doctor(
            environment_checks(
                data_path=data_path,
                profile=preparation_type,
                require_tpu=require_tpu,
                expected_process_count=_expected_process_count(proposed),
                accelerator=proposed.accelerator,
                chips_per_host=proposed.chips_per_host,
                route=route,
                check_data=args.check_only and not route.is_scaled,
                compile_probe=True,
            )
        )
        print(_indent(render_doctor(results, color=style.enabled)))
        if not doctor_ok(results):
            raise ConfigError("machine diagnostics failed; resolve the errors above")

    if not data_work:
        style.note("dataset preparation explicitly skipped (--no-download)")
    elif inventory is not None:
        style.heading("Dataset caches")
        style.note(
            f"preparing datasets and validations concurrently on "
            f"{len(inventory.hosts)} TPU VMs"
        )
        _run_cluster_prepare(
            proposed,
            args,
            inventory,
            preparation_type=preparation_type,
            root=root,
        )
    else:
        style.heading("Dataset cache")
        shards = route.train_shards
        if args.check_only:
            prepared = verify_dataset(route_manifest, route_root, train_shards=shards)
        else:
            progress = _progress_reporter(style)
            prepared = prepare_data(
                route_root,
                route_manifest,
                train_shards=shards,
                offline=args.offline,
                force=args.force,
                progress=progress,
                timeout=args.timeout,
            )
        _print_prepared(prepared, style)
        if preparation_type == "official":
            style.heading("Fresh-domain diagnostic")
            if args.check_only:
                fresh10 = verify_fresh10(data_path)
            else:
                fresh10 = prepare_fresh10(
                    data_path,
                    offline=args.offline,
                    force=args.force,
                    progress=_progress_reporter(style),
                    timeout=args.timeout,
                )
            _print_fresh10(fresh10, style)

    if inventory is not None:
        if run_diagnostics:
            style.heading("Distributed machine diagnostics")
            _run_cluster_doctor(
                proposed,
                inventory,
                profile=preparation_type,
                data_path=proposed.data_path,
                require_tpu=require_tpu,
                # Each remote scaled prepare already verifies its routed
                # manifest and nested cache; avoid hashing all 40 shards twice
                # in the same preparation transaction.
                check_data=(data_work or args.check_only) and not route.is_scaled,
                quick=False,
                color=proposed.color,
                root=root,
            )
    return 0


def command_doctor(args: argparse.Namespace) -> int:
    config = load_config(cluster=getattr(args, "cluster", None))
    profile = args.profile or config.default_profile
    path = resolve_path(args.path or config.data_path)
    color = args.color or config.color
    if _work_runs_elsewhere(config) and not _is_cluster_worker():
        inventory = _probe_configured_cluster(config)
        return _run_cluster_doctor(
            config,
            inventory,
            profile=profile,
            data_path=str(args.path or config.data_path),
            require_tpu=args.require_tpu or profile == "official",
            check_data=not args.skip_data,
            quick=args.quick,
            color=color,
            root=repo_root(),
            cluster=getattr(args, "cluster", None),
        )

    process_index = _initialize_distributed_worker(config)
    is_controller = _is_controller_process(process_index)
    style = Style(color)
    if is_controller:
        style.banner("doctor")
    results = run_doctor(
        environment_checks(
            data_path=path,
            profile=profile,
            require_tpu=args.require_tpu or profile == "official",
            expected_process_count=_expected_process_count(config),
            accelerator=config.accelerator,
            chips_per_host=config.chips_per_host,
            route=_route_for_config(config, _runtime_preparation_type(profile)),
            check_data=not args.skip_data,
            compile_probe=not args.quick,
        )
    )
    healthy = doctor_ok(results)
    if is_controller:
        print(_indent(render_doctor(results, color=style.enabled)))
    elif not healthy:
        print(render_doctor(results, color=False), file=sys.stderr)
    return 0 if healthy else 1


def _recipe_entry(root: Path, recipe: str, profile: str) -> tuple[Path, Path]:
    """Resolve a recipe entry and selected config without path traversal."""

    if not _NAME.fullmatch(recipe):
        raise ConfigError(
            "recipe names may contain only letters, digits, '.', '_' and '-'"
        )
    recipes_root = (root / "recipes").resolve()
    recipe_dir = (recipes_root / recipe).resolve()
    try:
        recipe_dir.relative_to(recipes_root)
    except ValueError as exc:
        raise ConfigError("recipe path escapes recipes directory") from exc
    trainer = recipe_dir / "train.py"
    experiment_config = recipe_dir / profile_config_filename(profile)
    if not trainer.is_file() or trainer.is_symlink():
        raise ConfigError(f"recipe entry script not found: {trainer}")
    if not experiment_config.is_file() or experiment_config.is_symlink():
        raise ConfigError(f"recipe configuration file not found: {experiment_config}")
    return recipe_dir, trainer


def _scientific_trainer_args(args: argparse.Namespace) -> list[str]:
    """Translate the deliberately small public research surface to a recipe."""

    result: list[str] = []
    optional = (
        ("--tier", getattr(args, "tier", None)),
        ("--context", getattr(args, "context", None)),
        ("--tokens-per-parameter", getattr(args, "tokens_per_parameter", None)),
        ("--base-learning-rate", getattr(args, "base_learning_rate", None)),
        ("--batch-size", getattr(args, "batch_size", None)),
        ("--stop-after-step", getattr(args, "stop_after_step", None)),
    )
    for flag, value in optional:
        if value is not None:
            result.extend((flag, str(value)))
    return result


def _recipe_specific_trainer_args(args: argparse.Namespace) -> list[str]:
    """Validate and return arguments after the explicit recipe boundary."""

    result = [str(value) for value in getattr(args, "recipe_args", ())]
    if any(argument in {"-h", "--help"} for argument in result):
        raise ConfigError("recipe --help must be requested alone after --")
    for argument in result:
        flag = argument.split("=", 1)[0]
        if flag in RIG_MANAGED_RECIPE_FLAGS:
            raise ConfigError(
                f"recipe-local arguments may not override harness-managed {flag}"
            )
        if argument == "--":
            raise ConfigError("recipe-local arguments cannot contain another --")
    return result


def command_recipe_help(args: argparse.Namespace) -> int:
    """Show the selected recipe's own argument surface without preparing a run."""

    root = repo_root()
    profile = args.profile or "dev"
    recipe_dir, trainer = _recipe_entry(root, args.recipe, profile)
    completed = subprocess.run(
        [sys.executable, str(trainer), "--help"],
        cwd=recipe_dir,
        check=False,
    )
    if completed.returncode != 0:
        raise ConfigError(f"recipe help exited with status {completed.returncode}")
    return 0


def command_run(args: argparse.Namespace) -> int:
    config = load_config(cluster=getattr(args, "cluster", None))
    root = repo_root()
    profile = args.profile or config.default_profile
    color = args.color or config.color
    style = Style(color)
    # Resolved first: an interactive prompt must not wait behind dataset
    # verification and cluster synchronization.
    run_name = _resolve_run_name(args, style)
    trainer_color = "always" if style.enabled else "never"
    target_loss = _effective_target_loss(
        profile, requested=None, development_default=config.target_loss
    )
    recipe_dir, trainer = _recipe_entry(root, args.recipe, profile)
    scientific_args = _scientific_trainer_args(args)
    recipe_args = _recipe_specific_trainer_args(args)
    python_executable = root / ".venv" / "bin" / "python"
    style.heading("Resolving recipe plan")
    plan = resolve_recipe_plan(
        python_executable=python_executable,
        trainer=trainer,
        arguments=(
            "--profile",
            profile,
            *scientific_args,
            *recipe_args,
        ),
        cwd=recipe_dir,
    )
    style.ok(
        f"{plan.payload['tier']}: {plan.payload['schedule_steps']:,} schedule steps, "
        f"{plan.expected_tokens:,} executed tokens ({plan.run_kind})"
    )
    configured_data_path = str(args.data_path or config.data_path)
    if _work_runs_elsewhere(config):
        configured_data_path = _cluster_data_argument(configured_data_path, root)
    data_path = resolve_path(configured_data_path, root)
    artifacts = resolve_path(config.artifacts_path, root)
    _ensure_artifacts_inside_repo(artifacts, root)
    if profile == "official" and args.skip_data_check:
        raise ConfigError("official runs require full dataset SHA-256 verification")
    # A profile controls runtime/evaluation policy. The saved dataset name and
    # shard prefix control which immutable FineWeb corpus backs non-smoke runs.
    # Keeping those axes separate lets a short dev study consume the same
    # rank-disjoint corpus as the eventual official family run.
    run_data_type = _runtime_preparation_type(profile)
    route = _route_for_config(config, run_data_type)
    _require_prepared_dataset(
        config, route, data_path, cluster=getattr(args, "cluster", None)
    )
    manifest = resolve_preparation_manifest(route)
    route_root = route.data_root(data_path)
    shards = route.train_shards
    style.heading("Verifying cached data")
    prepared = verify_dataset(
        manifest,
        route_root,
        train_shards=shards,
        verify_hash=not args.skip_data_check,
    )
    if not args.skip_data_check:
        style.ok(
            f"{prepared.name}: {prepared.train_tokens:,} train / "
            f"{prepared.validation_tokens:,} validation tokens"
        )
    else:
        style.note(
            "SHA-256 scan skipped; headers and exact shard selection still checked"
        )
    expected_validation_tokens = _validation_contract(plan, prepared, profile)

    fresh10: PreparedFresh10 | None = None
    if profile == "official":
        fresh10 = verify_fresh10(data_path, verify_hash=True)
        style.ok(
            f"{fresh10.name}: {len(fresh10.domains)} domains / "
            f"{fresh10.scored_tokens:,} scored tokens"
        )

    artifact_target = ""
    artifact_hostname = ""
    if _work_runs_elsewhere(config):
        style.heading("Synchronizing TPU VM cluster")
        inventory = _probe_configured_cluster(config)
        sync_workspace(
            root,
            inventory,
            artifacts_path=artifacts,
            data_path=data_path,
        )
        style.ok(
            f"current source synchronized to {len(inventory.remote_hosts)} peer VMs"
        )
        artifact_target = inventory.artifact_host
        artifact_hostname = inventory.reported_hostnames[artifact_target]
        if config.remote_controller:
            style.ok(
                f"remote controller: artifacts land on {artifact_hostname} "
                "and are pulled back after the run"
            )

    dataset_id, tokenizer_id = _data_identity(
        run_data_type, prepared_name=prepared.name
    )
    trainer_args = [
        *scientific_args,
        *recipe_args,
        "--data-format",
        "llmc",
        "--dataset-id",
        dataset_id,
        "--tokenizer-id",
        tokenizer_id,
        "--color",
        trainer_color,
    ]
    for train_file in prepared.train_files:
        trainer_args.extend(("--train-data", str(train_file)))
    for validation_file in prepared.validation_files:
        trainer_args.extend(("--val-data", str(validation_file)))
    if fresh10 is not None:
        trainer_args.extend(("--downstream-manifest", str(fresh10.manifest_path)))
        trainer_args.extend(("--downstream-root", str(fresh10.root)))
    timeout = (
        args.timeout or {"smoke": 300.0, "dev": 3600.0, "official": 21600.0}[profile]
    )
    policy = _checkpoint_policy(args, config.checkpoint_policy, profile=profile)
    if policy == "none" and not _is_research_run(profile):
        raise ConfigError(
            "--checkpoint-policy none is restricted to development research runs"
        )
    study_values = (args.study_id, args.study_point, args.study_suite_sha256)
    if any(value is not None for value in study_values) and not all(
        value is not None for value in study_values
    ):
        raise ConfigError(
            "--study-id, --study-point, and --study-suite-sha256 must be supplied together"
        )
    if args.study_id is not None and (
        not _NAME.fullmatch(args.study_id) or not _NAME.fullmatch(args.study_point)
    ):
        raise ConfigError("study and point IDs must be simple filesystem-safe names")
    if args.study_suite_sha256 is not None and not re.fullmatch(
        r"[0-9a-f]{64}", args.study_suite_sha256
    ):
        raise ConfigError("study suite SHA-256 must be 64 lowercase hexadecimal digits")
    provenance = _data_provenance(
        prepared,
        profile=run_data_type,
        integrity="headers+size" if args.skip_data_check else "sha256",
        repo=root,
        fresh10=fresh10,
    )
    cohort = build_cohort(
        plan=plan,
        dataset_id=dataset_id,
        tokenizer_id=tokenizer_id,
        dataset_provenance=provenance,
        accelerator=config.accelerator,
        tpu_vm_count=config.tpu_vm_count,
        chips_per_host=config.chips_per_host,
        target_loss=target_loss,
    )
    style.banner(f"run / {profile} / {plan.run_kind}")
    outcome = run_recipe(
        RunConfig(
            repo_root=root,
            recipe=args.recipe,
            name=run_name,
            runs_dir=artifacts,
            records_path=artifacts / "records.jsonl",
            plan=plan.as_dict(),
            cohort=cohort,
            profile=profile,
            seed=args.seed,
            timeout_seconds=timeout,
            target_loss=target_loss,
            expected_validation_tokens=expected_validation_tokens,
            expected_downstream_tokens=(
                {domain.name: domain.scored_tokens for domain in fresh10.domains}
                if fresh10 is not None
                else None
            ),
            trainer_args=tuple(trainer_args),
            checkpoint_policy=policy,
            python_executable=str(python_executable),
            remote_controller=config.remote_controller,
            artifact_host=artifact_target,
            artifact_hostname=artifact_hostname,
            environment={},
            provenance={
                **provenance,
                "cluster": {
                    "tpu_vm_count": config.tpu_vm_count,
                    "tpu_vm_hosts": config.tpu_vm_hosts,
                },
                **(
                    {
                        "study": {
                            "study_id": args.study_id,
                            "point_id": args.study_point,
                            "suite_sha256": args.study_suite_sha256,
                        }
                    }
                    if args.study_id is not None
                    else {}
                ),
            },
            tpu_vm_count=config.tpu_vm_count,
            tpu_vm_hosts=config.tpu_vm_hosts,
        )
    )
    metrics = outcome.record["metrics"]
    qualified = bool(outcome.record["qualified"])
    marker = (
        style.text("QUALIFIED", "green", "bold")
        if qualified
        else style.text("NOT QUALIFIED", "yellow", "bold")
    )
    style.heading("Recorded result")
    print(
        f"  {marker}  loss {metrics['validation_loss']:.4f}  target ≤ {target_loss:.4f}"
    )
    print(
        f"  train {metrics['train_seconds']:.3f}s  tokens {metrics['tokens_processed']:,}"
    )
    evaluations = outcome.record.get("evaluations")
    if isinstance(evaluations, dict):
        fresh = evaluations.get("fresh10")
        if isinstance(fresh, dict):
            print(
                f"  fresh10 macro loss {float(fresh['macro_loss']):.4f}  "
                f"ppl {float(fresh['macro_perplexity']):.2f}"
            )
    print(f"  run {outcome.run_id}\n")
    if cohort is not None:
        print(f"  cohort {cohort['cohort_id']}\n")
    return 0


def command_profile(args: argparse.Namespace) -> int:
    """Run one bounded diagnostic on the configured JAX process topology."""

    root = repo_root()
    if not config_path(root).is_file():
        raise ConfigError("no saved default profile found; run `make prepare` first")
    config = load_config(root, cluster=getattr(args, "cluster", None))
    profile = args.profile or config.default_profile
    if profile == "smoke":
        raise ConfigError("XProf capture requires a fixed-TPP dev or official profile")
    color = args.color or config.color
    style = Style(color)
    recipe_dir, trainer = _recipe_entry(root, args.recipe, profile)
    scientific_args = _scientific_trainer_args(args)
    recipe_args = _recipe_specific_trainer_args(args)
    python_executable = root / ".venv" / "bin" / "python"
    output_dir = resolve_path(args.output_dir, root)
    xprof_dir = output_dir / "xprof"
    plan = resolve_recipe_plan(
        python_executable=python_executable,
        trainer=trainer,
        arguments=(
            "--profile",
            profile,
            *scientific_args,
            *recipe_args,
            "--xprof-dir",
            str(xprof_dir),
            "--xprof-start-step",
            str(args.xprof_start_step),
            "--xprof-steps",
            str(args.xprof_steps),
            "--diagnostic-mode",
        ),
        cwd=recipe_dir,
    )
    final_step = int(plan.payload["stop_after_step"] or plan.payload["schedule_steps"])
    if args.xprof_start_step + args.xprof_steps - 1 > final_step:
        raise ConfigError("the XProf capture window must fit inside --stop-after-step")

    configured_data_path = str(args.data_path or config.data_path)
    if _work_runs_elsewhere(config):
        configured_data_path = _cluster_data_argument(configured_data_path, root)
    data_path = resolve_path(configured_data_path, root)
    run_data_type = _runtime_preparation_type(profile)
    route = _route_for_config(config, run_data_type)
    manifest = resolve_preparation_manifest(route)
    route_root = route.data_root(data_path)
    shards = route.train_shards
    style.heading("Verifying cached profile data")
    prepared = verify_dataset(manifest, route_root, train_shards=shards)
    style.ok(
        f"{prepared.name}: {prepared.train_tokens:,} train / "
        f"{prepared.validation_tokens:,} validation tokens"
    )

    dataset_id, tokenizer_id = _data_identity(
        run_data_type, prepared_name=prepared.name
    )
    trainer_color = "always" if style.enabled else "never"
    trainer_command = [
        str(python_executable),
        str(trainer),
        "--output-dir",
        str(output_dir),
        "--seed",
        str(args.seed),
        "--profile",
        profile,
        *scientific_args,
        *recipe_args,
        "--data-format",
        "llmc",
        "--dataset-id",
        dataset_id,
        "--tokenizer-id",
        tokenizer_id,
    ]
    for train_file in prepared.train_files:
        trainer_command.extend(("--train-data", str(train_file)))
    for validation_file in prepared.validation_files:
        trainer_command.extend(("--val-data", str(validation_file)))
    trainer_command.extend(
        (
            "--xprof-dir",
            str(xprof_dir),
            "--xprof-start-step",
            str(args.xprof_start_step),
            "--xprof-steps",
            str(args.xprof_steps),
            "--diagnostic-mode",
            "--color",
            trainer_color,
        )
    )

    style.banner(f"profile / {args.recipe} / {profile}")
    if _work_runs_elsewhere(config):
        inventory = _probe_configured_cluster(config)
        style.note(
            f"synchronizing source and launching all {len(inventory.hosts)} TPU VMs"
        )
        sync_workspace(
            root,
            inventory,
            artifacts_path=output_dir,
            data_path=data_path,
        )
        remote_environment = {
            _CLUSTER_WORKER_ENV: "1",
            _CONTROLLER_HOST_ENV: inventory.reported_hostnames[inventory.artifact_host],
            _DISTRIBUTED_ENV: "1",
            _PROCESS_COUNT_ENV: str(config.tpu_vm_count),
            "JAX_COMPILATION_CACHE_DIR": f"/tmp/rig-profile-cache-{os.getpid()}",
            "PYTHONUNBUFFERED": "1",
        }
        assignments = " ".join(
            f"{key}={shlex.quote(value)}" for key, value in remote_environment.items()
        )
        remote = (
            f"cd {shlex.quote(str(recipe_dir))} && "
            f"env {assignments} {shlex.join(trainer_command)}"
        )
        run_pdsh(
            inventory.hosts,
            remote,
            labels=True,
            timeout=float(args.timeout),
            # A partially launched JAX collective must be torn down, not
            # replayed while surviving workers may still be waiting.
            retry_transport=False,
        )
    else:
        try:
            completed = subprocess.run(
                trainer_command,
                cwd=recipe_dir,
                check=False,
                timeout=float(args.timeout),
            )
        except subprocess.TimeoutExpired as exc:
            raise ConfigError(
                f"profile trainer timed out after {float(args.timeout):g}s"
            ) from exc
        if completed.returncode != 0:
            raise ConfigError(
                f"profile trainer exited with status {completed.returncode}"
            )
    style.ok(f"worker 0 XProf trace saved to {xprof_dir}")
    return 0


def command_verify(args: argparse.Namespace) -> int:
    config = load_config(cluster=getattr(args, "cluster", None))
    root = repo_root()
    artifacts = resolve_path(config.artifacts_path, root)
    candidate = Path(args.run).expanduser()
    run_dir = (
        candidate.resolve() if candidate.exists() else (artifacts / args.run).resolve()
    )
    records = load_records(artifacts / "records.jsonl")
    record = next(
        (item for item in reversed(records) if item.get("run_id") == run_dir.name), None
    )
    profile = args.profile or (
        str(record["profile"]) if record is not None else config.default_profile
    )
    require_checkpoint = _record_requires_checkpoint(record)
    expected_downstream = (
        _recorded_downstream_tokens(record)
        if profile == "official" and record is not None
        else None
    )
    expected_validation = None
    if profile == "official":
        expected_validation = (
            _recorded_validation_tokens(record) if record is not None else None
        ) or OFFICIAL_VALIDATION_PREDICTIONS
    result = verify_run(
        run_dir,
        expected_training_tokens=(
            _recorded_training_tokens(record) if record is not None else None
        ),
        expected_validation_tokens=expected_validation,
        expected_downstream_tokens=expected_downstream,
        require_checkpoint=require_checkpoint,
    )
    if record is not None:
        stdout_sha256 = sha256_file(run_dir / "stdout.log")
        expected_stdout = record.get("logs", {}).get("stdout_sha256")
        if stdout_sha256 != expected_stdout:
            raise HarnessError(
                "captured stdout hash no longer matches its immutable record"
            )
        recorded_checkpoint = record.get("checkpoint")
        expected_checkpoint = (
            recorded_checkpoint.get("sha256")
            if require_checkpoint and isinstance(recorded_checkpoint, dict)
            else None
        )
        if result.checkpoint_sha256 != expected_checkpoint:
            raise HarnessError("checkpoint hash no longer matches its immutable record")
        recorded_artifacts = record.get("artifacts", {})
        if set(result.artifacts) != set(recorded_artifacts):
            raise HarnessError("run artifacts no longer match their immutable record")
        for name, path in result.artifacts.items():
            expected_artifact = recorded_artifacts.get(name, {}).get("sha256")
            if sha256_file(path) != expected_artifact:
                raise HarnessError(
                    f"artifact {name!r} hash no longer matches its immutable record"
                )
    checkpoint_label = (
        result.checkpoint_sha256[:12]
        if result.checkpoint_sha256 is not None
        else "omitted"
    )
    print(
        f"verified {run_dir.name} ({profile}): loss={result.validation_loss:.4f}, "
        f"tokens={result.tokens_processed:,}, checkpoint={checkpoint_label}"
    )
    return 0


def _dataset_cache_root(config: LocalConfig, override: Path | None = None) -> Path:
    return resolve_path(str(override) if override else config.data_path)


def _validation_contract(
    plan: RecipePlan,
    prepared: PreparedDataset,
    profile: str,
) -> int | None:
    """Bind official recipe coverage to the selected corpus before launch."""

    if profile != "official":
        return None
    expected = prepared.validation_prefix_tokens
    if plan.validation_predictions != expected:
        raise ConfigError(
            "official recipe validation coverage does not match the selected "
            f"dataset: plan={plan.validation_predictions:,}, "
            f"manifest={expected:,} predictions"
        )
    return expected


def _dataset_presence(root: Path) -> dict[str, tuple[int, int]]:
    """Map corpus name -> (shard files present, bytes) in a local cache root."""

    found: dict[str, tuple[int, int]] = {}
    for name in dataset_names():
        route = named_preparation_route(name)
        directory = route.data_root(root)
        if not directory.is_dir():
            continue
        files = [entry for entry in directory.iterdir() if entry.suffix == ".bin"]
        found[name] = (len(files), sum(entry.stat().st_size for entry in files))
    return found


def _route_for_config(config: LocalConfig, preparation_type: str):
    """Resolve the explicit corpus, with smoke as the sole inline exception."""

    if preparation_type == "smoke":
        return smoke_preparation_route()
    return named_preparation_route(
        config.dataset,
        train_shards=config.train_shards or None,
    )


def _runtime_preparation_type(profile: str) -> str:
    """Map an execution profile to synthetic or named immutable data."""

    return "smoke" if profile == "smoke" else "official"


def _require_prepared_dataset(
    config: LocalConfig, route, root: Path, *, cluster: str | None
) -> None:
    """Fail with the fixing command when a named corpus is not present."""

    directory = route.data_root(root)
    if directory.is_dir() and any(directory.glob("*.bin")):
        return
    if route.profile == "smoke":
        cluster_flag = f" --cluster {cluster}" if cluster else ""
        raise ConfigError(
            f"smoke data is not prepared at {directory}"
            f"\n  prepare it here:  rig prepare --profile smoke{cluster_flag}"
        )
    ship = (
        f"\n  then ship it:     rig dataset ship {config.dataset} --cluster {cluster}"
        if cluster
        else ""
    )
    raise ConfigError(
        f"dataset {config.dataset!r} is not prepared at {directory}"
        f"\n  prepare it here:  rig dataset prepare {config.dataset}{ship}"
    )


def command_dataset(args: argparse.Namespace) -> int:
    """Prepare and distribute corpora by name, decoupled from saved settings.

    Naming a corpus independently of its training horizon lets callers prepare
    data without changing run settings, and lets `rig run` report a missing
    dataset with the command that fixes it instead of silently choosing another
    one.
    """

    config = load_config(cluster=getattr(args, "cluster", None))
    style = Style(getattr(args, "color", None) or config.color)
    action = args.dataset_command

    if action == "list":
        root = _dataset_cache_root(config)
        present = _dataset_presence(root)
        style.heading("corpora")
        for name in dataset_names():
            route = named_preparation_route(name)
            here = present.get(name)
            state = (
                f"{here[0]} shard files, {here[1] / 2**30:.1f} GiB here"
                if here
                else "not prepared here"
            )
            style.note(
                f"{name:<5} up to {route.train_capacity:>15,} train tokens  ·  {state}"
            )
        return 0

    if action == "status":
        root = _dataset_cache_root(config)
        present = _dataset_presence(root)
        style.heading("dataset status")
        for name in dataset_names():
            here = present.get(name)
            style.note(
                f"{name:<5} local: " + (f"{here[0]} shards" if here else "absent")
            )
        if getattr(args, "cluster", None):
            inventory = _probe_configured_cluster(config)
            listing = (
                f"for d in {shlex.quote(str(RAM_CACHE_ROOT / SCALED_CACHE_SUBDIRECTORY))}/*; "
                'do [ -d "$d" ] && echo "$(basename $d) $(ls "$d" | wc -l)"; done'
            )
            style.heading(f"on {args.cluster}")
            run_pdsh(inventory.hosts, listing, labels=True)
        return 0

    if action == "prepare":
        route = named_preparation_route(args.name, train_shards=args.shards)
        root = _dataset_cache_root(config, args.path)
        style.heading(f"preparing {args.name}")
        style.note(
            f"{route.train_shards} train shards "
            f"({route.train_capacity:,} tokens) into {root}"
        )
        prepared = prepare_data(
            root / route.cache_subdirectory,
            resolve_preparation_manifest(route),
            train_shards=route.train_shards,
            offline=args.offline,
            check_only=args.check_only,
            force=args.force,
            progress=_progress_reporter(style),
            timeout=args.timeout,
        )
        _print_prepared(prepared, style)
        return 0

    if action == "ship":
        route = named_preparation_route(args.name)
        root = _dataset_cache_root(config)
        source = root / route.cache_subdirectory
        if not source.is_dir():
            raise ConfigError(
                f"{args.name} is not prepared here ({source} is missing); run "
                f"`rig dataset prepare {args.name}` first"
            )
        if not _work_runs_elsewhere(config):
            raise ConfigError(
                "shipping needs a cluster whose accelerators are elsewhere; "
                "pass --cluster"
            )
        inventory = _probe_configured_cluster(config)
        style.heading(f"shipping {args.name}")
        count = ship_dataset(inventory.remote_hosts, source)
        style.ok(f"{args.name} present and sealed on {count} host(s)")
        return 0

    raise ConfigError(f"unknown dataset action: {action}")


def command_leaderboard(args: argparse.Namespace) -> int:
    config = load_config(cluster=getattr(args, "cluster", None))
    artifacts = resolve_path(config.artifacts_path)
    records = load_records(artifacts / "records.jsonl")
    style = Style(args.color or config.color)
    cohorts: dict[str, dict[str, Any]] = {}
    for record in records:
        if (
            record.get("status") != "ok"
            or record.get("run_kind") != "full"
            or record.get("profile") != args.profile
            or not isinstance(record.get("cohort"), dict)
        ):
            continue
        try:
            cohort = validate_cohort(record["cohort"])
        except CohortError:
            continue
        cohort_id = cohort["cohort_id"]
        if record.get("cohort_id") != cohort_id:
            continue
        cohorts.setdefault(cohort_id, cohort)
    cohort_ids = sorted(cohorts)
    if args.cohort is not None:
        if args.cohort not in cohort_ids:
            raise ConfigError(
                f"no full {args.profile} records belong to cohort {args.cohort}"
            )
        cohort_ids = [args.cohort]
    if not cohort_ids:
        print(f"No cohort-tagged full {args.profile} runs.")
        return 0
    rendered: list[str] = []
    for cohort_id in cohort_ids:
        ranked = rank_records(
            records,
            cohort_id=cohort_id,
            profile=args.profile,
            best_per_recipe=not args.all_recipes,
        )
        cohort = cohorts[cohort_id]
        target_loss = cohort["qualification"]["target_loss"]
        tpp = cohort["horizon"]["target_tokens_per_parameter"]
        rendered.append(
            f"tier {cohort['tier']} · {tpp} TPP · target loss ≤ {target_loss}\n"
            + render_leaderboard(ranked, cohort_id=cohort_id, color=style.enabled)
        )
    print("\n\n".join(rendered))
    return 0


def command_report(args: argparse.Namespace) -> int:
    root = repo_root()
    runs = args.runs if args.runs.is_absolute() else root / args.runs
    output = args.output if args.output.is_absolute() else root / args.output

    if args.study_export_target is not None:
        if not args.study_name:
            raise ConfigError("--study-export-target needs --study-name")
        target = (
            args.study_export_target
            if args.study_export_target.is_absolute()
            else root / args.study_export_target
        )
        summary = export_study(runs, target, args.study_name, select=args.select)
        print(
            f"study {summary['path']}: {summary['runs']} run(s), "
            f"{summary['ledgered']} ledgered, {summary['bytes'] / 1e6:.1f} MB "
            f"(snapshot {summary['snapshot_bytes'] / 1e6:.2f} MB, "
            f"full view {summary['full_bytes'] / 1e6:.1f} MB)"
        )
        if not summary["runs"]:
            return 1
        # Yellow, and loud: an empty README that nobody notices becomes a study
        # nobody can reproduce, and this is the one moment the person who ran
        # the sweep is still holding the reason it was run.
        console = Console(getattr(args, "color", "auto") or "auto")
        console.warn(f"{summary['readme']} is empty -- fill it in before publishing.")
        console.warn(
            "It is the study's landing page: the setup as bullet points, the "
            "command that reproduces the sweep, and what it showed."
        )
        return 0

    summary = build_report(
        runs,
        output,
        max_chart_points=args.max_points,
        layer_snapshots=args.layer_snapshots,
        select=args.select,
    )
    relative = (
        summary.output_path.relative_to(root)
        if summary.output_path.is_relative_to(root)
        else summary.output_path
    )
    print(
        f"report {relative}: {len(summary.included)} run(s) plotted, "
        f"{len(summary.skipped)} skipped"
    )
    for run_id, reason in summary.skipped.items():
        print(f"  skipped {run_id}: {reason}")
    return 0


def command_clone(args: argparse.Namespace) -> int:
    if not _NAME.fullmatch(args.source) or not _NAME.fullmatch(args.name):
        raise ConfigError(
            "recipe names may contain only letters, digits, '.', '_' and '-'"
        )
    root = repo_root()
    source = root / "recipes" / args.source
    destination = root / "recipes" / args.name
    if not (source / "train.py").is_file():
        raise ConfigError(f"source recipe does not exist: {source}")
    source_configs = tuple(
        source / filename for filename in PROFILE_CONFIG_FILENAMES.values()
    )
    for source_config in source_configs:
        if not source_config.is_file() or source_config.is_symlink():
            raise ConfigError(
                f"source recipe configuration does not exist: {source_config}"
            )
    if destination.exists():
        raise ConfigError(f"destination already exists: {destination}")
    destination.mkdir(parents=True)
    shutil.copy2(source / "train.py", destination / "train.py")
    for source_config in source_configs:
        shutil.copy2(source_config, destination / source_config.name)
    if (source / "README.md").is_file():
        shutil.copy2(source / "README.md", destination / "README.md")
    print(f"cloned {args.source} -> {args.name} ({destination})")
    return 0


def command_settings(args: argparse.Namespace) -> int:
    config = load_config(cluster=getattr(args, "cluster", None))
    payload = asdict(config)
    root = repo_root()
    payload["data_path_resolved"] = str(resolve_path(config.data_path, root))
    payload["artifacts_path_resolved"] = str(resolve_path(config.artifacts_path, root))
    payload["config_path"] = str(config_path())
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        width = max(len(key) for key in payload)
        for key, value in payload.items():
            print(f"{key:<{width}}  {value}")
    return 0


def _ship_prepared_data(
    config: LocalConfig, inventory: ClusterInventory, *, style: Style
) -> None:
    """Send this host's prepared corpus to peers that cannot download it.

    Every directory already present in the local RAM cache is mirrored, so the
    same call serves whichever corpus is installed here. Sealing happens inside
    ship_dataset, in the same session as the copy.
    """

    if not RAM_CACHE_ROOT.is_dir():
        raise ConfigError(
            f"{RAM_CACHE_ROOT} holds no prepared data to ship; run `rig prepare` "
            "here first"
        )
    sources = sorted(
        entry
        for entry in RAM_CACHE_ROOT.iterdir()
        if entry.is_dir() and not entry.name.startswith(".")
    )
    if not sources:
        raise ConfigError(f"{RAM_CACHE_ROOT} contains no dataset directories")
    for source in sources:
        style.note(f"copying {source.name} to peer VMs")
        count = ship_dataset(inventory.remote_hosts, source)
        style.ok(f"{source.name} present and sealed on {count} peer VMs")


def _prepare_cluster(
    config: LocalConfig,
    args: argparse.Namespace,
    *,
    root: Path,
    artifacts_path: Path,
    style: Style,
) -> ClusterInventory:
    style.heading("TPU VM cluster")
    inventory = probe_cluster(
        config.tpu_vm_hosts,
        config.tpu_vm_count,
        remote_controller=config.remote_controller,
        artifact_host=config.artifact_host,
    )
    style.ok(
        f"passwordless SSH ready on {len(inventory.hosts)} hosts "
        f"({len(inventory.remote_hosts)} peer VMs)"
    )
    if _uses_repo_shm_cache(config.data_path, root):
        style.note("checking RAM-backed /dev/shm and configuring the shm cache link")
        prepare_ram_cache(root, inventory, create_link=not args.check_only)
        style.ok("shm points to writable RAM-backed storage on every TPU VM")
    if args.check_only:
        style.note("check-only mode does not synchronize source or environments")
    else:
        style.note(
            "incrementally synchronizing source and personal settings to peer VMs"
        )
        sync_workspace(
            root,
            inventory,
            artifacts_path=artifacts_path,
            data_path=resolve_path(config.data_path, root),
        )
        style.note("synchronizing the frozen uv environment on peer VMs")
        shipped = ship_uv_binary(inventory.remote_hosts)
        if shipped:
            style.ok(f"shipped {shipped} to peers that lacked it")
        interpreters = ship_uv_python(inventory.remote_hosts)
        if interpreters:
            style.ok(f"shipped interpreters: {', '.join(interpreters)}")
        cached = ship_uv_cache(inventory.remote_hosts, Path("/tmp/uv-cache"))
        if cached:
            style.ok(f"shipped the uv package cache to {cached} peer VMs")
        if getattr(args, "ship_data", False):
            _ship_prepared_data(config, inventory, style=style)
        bootstrap_uv(root, inventory.remote_hosts, offline=args.offline)
    return inventory


def _run_cluster_prepare(
    config: LocalConfig,
    args: argparse.Namespace,
    inventory: ClusterInventory,
    *,
    preparation_type: str,
    root: Path,
) -> None:
    if not inventory.hosts:
        return
    data_argument = _cluster_data_argument(config.data_path, root)
    command = [
        str(root / ".venv" / "bin" / "python"),
        "-m",
        "rig",
        "prepare",
        "--non-interactive",
        "--no-save",
        "--no-doctor",
        "--path",
        data_argument,
        "--profile",
        preparation_type,
        "--artifacts",
        config.artifacts_path,
        "--tpu-vm-count",
        str(config.tpu_vm_count),
        "--tpu-vm-hosts",
        config.tpu_vm_hosts,
        "--run-profile",
        config.default_profile,
        "--checkpoint-policy",
        config.checkpoint_policy,
        "--color",
        "never",
        "--dataset",
        config.dataset,
        "--timeout",
        str(args.timeout),
    ]
    # Peers inherit the mirrored .rig.toml, but an explicit selection on the
    # controller must reach them or they resolve a different cluster -- or, if
    # the file has no active cluster, refuse to resolve one at all.
    if getattr(args, "cluster", None):
        command.extend(("--cluster", args.cluster))
    if config.train_shards:
        command.extend(("--train-shards", str(config.train_shards)))
    if args.offline:
        command.append("--offline")
    if args.check_only:
        command.append("--check-only")
    if args.force:
        command.append("--force")
    worker_command = f"env {_CLUSTER_WORKER_ENV}=1 {shlex.join(command)}"
    if _uses_repo_shm_cache(config.data_path, root) and not args.check_only:
        worker_command = f"{worker_command} && {seal_ram_cache_command()}"
    remote = f"cd {shlex.quote(str(root.resolve()))} && {worker_command}"
    run_pdsh(
        inventory.hosts,
        remote,
        labels=True,
        timeout=_remote_prepare_timeout(config, args, preparation_type),
    )


def _remote_prepare_timeout(
    config: LocalConfig,
    args: argparse.Namespace,
    preparation_type: str,
) -> float:
    """Return a whole-peer cap distinct from the per-request HTTP timeout.

    The explicit dataset selection determines the nominal bytes each peer installs.
    Estimate transfer plus verification at 10 MiB/s, add 50% for contention and
    retries, then add 30 minutes for fixed setup.  This gives hero roughly 6.5
    hours while retaining a 15-minute absolute floor.  Offline and check-only
    paths still use the same safe upper bound; it is a deadline, not an expected
    duration.
    """

    route = _route_for_config(config, preparation_type)
    token_bytes = 2 * (
        (route.train_capacity or route.train_shards * 100_000_000)
        + (100_000_000 if preparation_type != "smoke" else 0)
    )
    transfer_and_verify = token_bytes / (10 * 1024**2)
    route_deadline = transfer_and_verify * 1.5 + 30 * 60.0
    return max(900.0, float(args.timeout) * 20.0, route_deadline)


def _uses_repo_shm_cache(value: str, root: Path) -> bool:
    """Recognize the conventional checkout ``shm`` path without following links."""

    configured = Path(value).expanduser()
    if not configured.is_absolute():
        configured = root / configured
    lexical = Path(os.path.abspath(configured))
    return lexical in {root.resolve() / "shm", Path("/dev/shm")}


def _cluster_data_argument(value: str, root: Path) -> str:
    """Route conventional multi-host shm use through the protected cache link."""

    if _uses_repo_shm_cache(value, root):
        return str(root.resolve() / "shm")
    return value


def _probe_configured_cluster(config: LocalConfig) -> ClusterInventory:
    if not _work_runs_elsewhere(config):
        raise ConfigError(
            "this operation needs accelerators on other machines: set "
            "tpu_vm_count above 1, or remote_controller for a single remote host"
        )
    return probe_cluster(
        config.tpu_vm_hosts,
        config.tpu_vm_count,
        remote_controller=config.remote_controller,
        artifact_host=config.artifact_host,
    )


def _run_cluster_doctor(
    config: LocalConfig,
    inventory: ClusterInventory,
    *,
    profile: str,
    data_path: str,
    require_tpu: bool,
    check_data: bool,
    quick: bool,
    color: str,
    root: Path,
    cluster: str | None = None,
) -> int:
    data_path = _cluster_data_argument(data_path, root)
    command = [
        str(root / ".venv" / "bin" / "python"),
        "-m",
        "rig",
        "doctor",
        "--path",
        data_path,
        "--profile",
        profile,
        "--color",
        color,
    ]
    # Peers must evaluate the same cluster contract; otherwise a controller
    # invoked with --cluster would check itself against one profile and its
    # peers against whatever their file happens to make active.
    if cluster:
        command.extend(("--cluster", cluster))
    if require_tpu:
        command.append("--require-tpu")
    if not check_data:
        command.append("--skip-data")
    if quick:
        command.append("--quick")
    remote = (
        f"cd {shlex.quote(str(root.resolve()))} && env "
        f"{_CLUSTER_WORKER_ENV}=1 "
        f"{_CONTROLLER_HOST_ENV}={shlex.quote(inventory.reported_hostnames[inventory.artifact_host])} "
        f"{_DISTRIBUTED_ENV}=1 "
        f"{_PROCESS_COUNT_ENV}={config.tpu_vm_count} {shlex.join(command)}"
    )
    run_pdsh(
        inventory.hosts,
        remote,
        labels=True,
        timeout=900.0,
    )
    return 0


def _is_cluster_worker() -> bool:
    return os.environ.get(_CLUSTER_WORKER_ENV) == "1"


def _expected_process_count(config: LocalConfig) -> int:
    raw = os.environ.get(_PROCESS_COUNT_ENV)
    if raw is None:
        return config.tpu_vm_count
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{_PROCESS_COUNT_ENV} must be a positive integer") from exc
    if value <= 0:
        raise ConfigError(f"{_PROCESS_COUNT_ENV} must be a positive integer")
    return value


def _is_controller_process(process_index: int) -> bool:
    configured = os.environ.get(_CONTROLLER_HOST_ENV)
    if configured is None:
        return process_index == 0
    local = os.uname().nodename.strip().split(".", 1)[0]
    expected = configured.strip().split(".", 1)[0]
    if not expected:
        raise ConfigError(f"{_CONTROLLER_HOST_ENV} may not be empty")
    return local == expected


def _initialize_distributed_worker(config: LocalConfig) -> int:
    expected = _expected_process_count(config)
    if not _is_cluster_worker() or expected <= 1:
        return 0
    import jax

    jax.distributed.initialize()
    actual = int(jax.process_count())
    if actual != expected:
        raise ConfigError(
            f"JAX discovered {actual} processes, but prepare configured {expected} TPU VM hosts"
        )
    return int(jax.process_index())


def _prepare_wizard(
    config: LocalConfig,
    *,
    preparation_type: str,
    run_diagnostics: bool,
    require_tpu: bool,
    download: bool,
    save: bool,
) -> tuple[LocalConfig, str, bool, bool, bool, bool]:
    style = Style(config.color)
    style.banner("interactive preparation")
    print(
        "  Choose personal defaults. Official data/model rules remain versioned in Git.\n"
    )
    data_path = _ask("Data cache root", config.data_path, style)
    preparation_type = _choose(
        "Preparation execution type",
        _PROFILES,
        preparation_type,
        style,
        descriptions={
            "smoke": "tiny generated CI data",
            "dev": "the explicitly selected corpus without Fresh10",
            "official": "the explicitly selected immutable corpus plus Fresh10",
        },
    )
    dataset = _choose(
        "Non-smoke dataset",
        dataset_names(),
        config.dataset,
        style,
    )
    route = named_preparation_route(dataset)
    saved_shards = config.train_shards if dataset == config.dataset else 0
    train_shards = _ask_int(
        f"Train shards (1-{route.train_shards})",
        saved_shards or route.train_shards,
        style,
        maximum=route.train_shards,
    )
    artifacts = _ask("Persistent run/artifact directory", config.artifacts_path, style)
    tpu_vm_count = _ask_int("TPU VM hosts in this JAX job", config.tpu_vm_count, style)
    if tpu_vm_count > 1:
        inferred_hosts = infer_host_expression(tpu_vm_count)
        default_hosts = config.tpu_vm_hosts if config.tpu_vm_hosts else inferred_hosts
        if not default_hosts:
            default_hosts = f"tpu-worker-[0-{tpu_vm_count - 1}]"
        tpu_vm_hosts = _ask("pdsh host expression", default_hosts, style)
    else:
        tpu_vm_hosts = ""
    run_profile = _choose(
        "Default run execution type", _PROFILES, config.default_profile, style
    )
    checkpoint_policy = _choose(
        "Checkpoint policy",
        _CHECKPOINT_POLICIES,
        config.checkpoint_policy,
        style,
        descriptions={
            "always": "keep the final weights",
            "qualifying": "keep them only at or below the target loss",
            "none": "never write weights; keep metrics and curves",
        },
    )
    color = _choose("Terminal colors", _COLORS, config.color, style)
    target = _ask_float(
        "Smoke/development qualification target", config.target_loss, style
    )
    run_diagnostics = _confirm(
        "Run environment diagnostics now", run_diagnostics, style
    )
    if run_diagnostics:
        require_tpu = _confirm(
            "Require a healthy Cloud TPU v4 topology on every configured VM",
            require_tpu,
            style,
        )
    save = _confirm("Save these personal defaults", save, style)
    resolved = with_overrides(
        config,
        {
            "data_path": data_path,
            "artifacts_path": artifacts,
            "tpu_vm_count": tpu_vm_count,
            "tpu_vm_hosts": tpu_vm_hosts,
            "dataset": dataset,
            "train_shards": train_shards,
            "default_profile": run_profile,
            "checkpoint_policy": checkpoint_policy,
            "color": color,
            "target_loss": target,
        },
    )
    return (
        resolved,
        preparation_type,
        run_diagnostics,
        require_tpu,
        download,
        save,
    )


def _resolve_run_name(args: argparse.Namespace, style: Style) -> str:
    """Return the run label, prompting when one was not supplied.

    Naming runs is worth a deliberate keystroke, so an interactive invocation
    always asks. Everything non-interactive -- a study loop, a pdsh worker, a
    piped shell -- silently keeps the unnamed default, because a prompt nobody
    can answer is a hang.
    """

    if args.name is not None:
        name = normalize_run_name(args.name)
        if not name:
            raise ConfigError(
                f"--name {args.name!r} contains no letters or digits to name a run by"
            )
        return name
    if _is_cluster_worker() or not sys.stdin.isatty():
        return ""
    while True:
        answer = input(
            f"  {style.text('Run name', 'bold')} "
            f"{style.text('[enter for unnamed]', 'dim')}: "
        ).strip()
        if not answer:
            return ""
        name = normalize_run_name(answer)
        if name:
            if name != answer:
                style.note(f"using {name}")
            return name
        style.note("a name needs at least one letter or digit; enter to skip")


def _ask(prompt: str, default: str, style: Style) -> str:
    while True:
        rendered = style.text(prompt, "bold")
        answer = input(f"  {rendered} {style.text(f'[{default}]', 'dim')}: ").strip()
        value = answer or default
        if value:
            return value


def _ask_float(prompt: str, default: float, style: Style) -> float:
    while True:
        raw = _ask(prompt, str(default), style)
        try:
            value = float(raw)
        except ValueError:
            print("  Enter a number.")
            continue
        if value >= 0 and value < float("inf"):
            return value
        print("  Enter a finite non-negative number.")


def _ask_int(
    prompt: str, default: int, style: Style, *, maximum: int | None = None
) -> int:
    while True:
        raw = _ask(prompt, str(default), style)
        try:
            value = int(raw)
        except ValueError:
            message = (
                f"  Enter an integer from 1 to {maximum}."
                if maximum is not None
                else "  Enter a positive integer."
            )
            print(message)
            continue
        if value > 0 and (maximum is None or value <= maximum):
            return value
        message = (
            f"  Enter an integer from 1 to {maximum}."
            if maximum is not None
            else "  Enter a positive integer."
        )
        print(message)


def _choose(
    prompt: str,
    choices: Sequence[str],
    default: str,
    style: Style,
    descriptions: dict[str, str] | None = None,
) -> str:
    descriptions = descriptions or {}
    print(f"  {style.text(prompt, 'bold')}")
    for index, choice in enumerate(choices, 1):
        selected = style.text("●", "cyan") if choice == default else "○"
        detail = f" — {descriptions[choice]}" if choice in descriptions else ""
        print(f"    {selected} {index}. {choice}{style.text(detail, 'dim')}")
    while True:
        answer = input(f"    {style.text(f'[{default}]', 'dim')}: ").strip()
        if not answer:
            return default
        if answer.isdigit() and 1 <= int(answer) <= len(choices):
            return choices[int(answer) - 1]
        if answer in choices:
            return answer
        print(f"    Choose 1-{len(choices)} or enter a listed name.")


def _confirm(prompt: str, default: bool, style: Style) -> bool:
    marker = "Y/n" if default else "y/N"
    while True:
        answer = (
            input(
                f"  {style.text(prompt, 'bold')} {style.text(f'[{marker}]', 'dim')}: "
            )
            .strip()
            .lower()
        )
        if not answer:
            return default
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("  Enter y or n.")


def _progress_reporter(style: Style) -> Callable[[str, int, int], None]:
    last: dict[str, int] = {}

    def report(name: str, completed: int, total: int) -> None:
        percent = 100 if total <= 0 else min(100, int(completed * 100 / total))
        bucket = percent // 10
        previous = last.get(name)
        if completed != total and previous == bucket:
            return
        last[name] = bucket
        if sys.stdout.isatty():
            width = 20
            filled = int(width * percent / 100)
            bar = style.text("━" * filled, "green") + style.text(
                "─" * (width - filled), "dim"
            )
            print(
                f"\r  {name:<30} {bar} {percent:3d}% "
                f"{completed / 2**20:7.1f}/{total / 2**20:.1f} MiB",
                end="\n" if completed == total else "",
                flush=True,
            )
        elif completed == total or bucket != previous:
            print(f"  {name}: {percent}%")

    return report


def _print_prepared(prepared: PreparedDataset, style: Style) -> None:
    style.ok(f"cache ready at {prepared.root}")
    print(f"  manifest       sha256:{prepared.manifest_sha256[:12]}")
    print(
        f"  training       {len(prepared.train_files)} shard(s), {prepared.train_tokens:,} tokens"
    )
    print(
        f"  validation     {len(prepared.validation_files)} shard(s), {prepared.validation_tokens:,} tokens"
    )
    print(
        f"  fixed prefix   {prepared.validation_prefix_tokens:,} validation predictions"
    )


def _print_fresh10(prepared: PreparedFresh10, style: Style) -> None:
    style.ok(f"fresh10 ready at {prepared.root}")
    print(f"  manifest       sha256:{prepared.manifest_sha256[:12]}")
    print(f"  domains        {len(prepared.domains)} ({', '.join(FRESH10_DOMAINS)})")
    print(f"  scored tokens  {prepared.scored_tokens:,} total")


def _data_identity(
    profile: str, *, prepared_name: str | None = None
) -> tuple[str, str]:
    if profile == "smoke":
        return "smoke", "synthetic-byte-v1"
    return prepared_name or "fineweb10b-gpt2", "gpt2"


def _effective_target_loss(
    profile: str,
    *,
    requested: float | None,
    development_default: float,
) -> float:
    if profile == "official":
        if requested is not None and requested > OFFICIAL_TARGET_LOSS:
            raise ConfigError(
                f"official target may not be easier than {OFFICIAL_TARGET_LOSS:.2f}"
            )
        return requested if requested is not None else OFFICIAL_TARGET_LOSS
    return requested if requested is not None else development_default


def _data_provenance(
    prepared: PreparedDataset,
    *,
    profile: str,
    integrity: str,
    repo: Path,
    fresh10: PreparedFresh10 | None = None,
) -> dict[str, Any]:
    try:
        manifest_path = (
            prepared.manifest_path.resolve().relative_to(repo.resolve()).as_posix()
        )
    except ValueError:
        manifest_path = str(prepared.manifest_path.resolve())
    manifest_file_sha256 = (
        sha256_file(prepared.manifest_path)
        if prepared.manifest_path.is_file()
        else None
    )
    result: dict[str, Any] = {
        "dataset": {
            "name": prepared.name,
            "profile": profile,
            "manifest": {
                "path": manifest_path,
                "sha256": manifest_file_sha256,
                "canonical_sha256": prepared.manifest_sha256,
            },
            "integrity": integrity,
            "train_files": [path.name for path in prepared.train_files],
            "validation_files": [path.name for path in prepared.validation_files],
            "train_tokens_available": prepared.train_tokens,
            "validation_tokens_available": prepared.validation_tokens,
            "validation_prefix_tokens": prepared.validation_prefix_tokens,
        }
    }
    if fresh10 is not None:
        try:
            fresh_manifest_path = (
                fresh10.manifest_path.resolve().relative_to(repo.resolve()).as_posix()
            )
        except ValueError:
            fresh_manifest_path = str(fresh10.manifest_path.resolve())
        result["fresh10"] = {
            "name": fresh10.name,
            "manifest": {
                "path": fresh_manifest_path,
                "sha256": sha256_file(fresh10.manifest_path),
                "canonical_sha256": fresh10.manifest_sha256,
            },
            "integrity": "sha256",
            "scored_tokens": fresh10.scored_tokens,
            "domains": {
                domain.name: {
                    "file": domain.path.name,
                    "sha256": domain.sha256,
                    "scored_tokens": domain.scored_tokens,
                }
                for domain in fresh10.domains
            },
        }
    return result


def _recorded_downstream_tokens(record: dict[str, Any]) -> dict[str, int] | None:
    """Recover the Fresh10 identity/count contract captured for a prior run."""

    provenance = record.get("provenance")
    fresh10 = provenance.get("fresh10") if isinstance(provenance, dict) else None
    domains = fresh10.get("domains") if isinstance(fresh10, dict) else None
    if domains is None:
        return None
    if not isinstance(domains, dict):
        raise HarnessError("recorded Fresh10 provenance has invalid domains")
    result: dict[str, int] = {}
    for name, row in domains.items():
        count = row.get("scored_tokens") if isinstance(row, dict) else None
        if (
            not isinstance(name, str)
            or not name
            or name.strip() != name
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
        ):
            raise HarnessError("recorded Fresh10 provenance has an invalid domain row")
        result[name] = count
    return result


def _recorded_training_tokens(record: dict[str, Any]) -> int | None:
    """Recover a token constraint without retroactively invalidating old runs."""

    constraints = record.get("constraints")
    if constraints is None:
        return None
    if not isinstance(constraints, dict):
        raise HarnessError("recorded run constraints are invalid")
    value = constraints.get("training_tokens")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise HarnessError("recorded training-token constraint is invalid")
    return value


def _recorded_validation_tokens(record: dict[str, Any]) -> int | None:
    """Recover the exact validation coverage captured for a prior run."""

    constraints = record.get("constraints")
    if constraints is None:
        return None
    if not isinstance(constraints, dict):
        raise HarnessError("recorded run constraints are invalid")
    value = constraints.get("validation_tokens")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise HarnessError("recorded validation-token constraint is invalid")
    return value


def _record_requires_checkpoint(record: dict[str, Any] | None) -> bool:
    """Recover whether a prior run intentionally retained its checkpoint."""

    if record is None:
        return True
    checkpoint = record.get("checkpoint")
    if checkpoint is None:
        return False
    if not isinstance(checkpoint, dict):
        raise HarnessError("recorded checkpoint metadata is invalid")
    retained = checkpoint.get("retained", True)
    if not isinstance(retained, bool):
        raise HarnessError("recorded checkpoint retention state is invalid")
    return retained


def _ensure_artifacts_inside_repo(path: Path, root: Path) -> None:
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ConfigError(
            f"artifact directory must stay on persistent storage inside the repository: {path}"
        ) from exc


def _indent(value: str) -> str:
    return "\n".join("  " + line for line in value.splitlines())


if __name__ == "__main__":
    raise SystemExit(main())
