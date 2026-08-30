"""The public ``rig`` command-line grammar.

Keeping argument declaration separate from command execution makes the public
surface reviewable as one small, side-effect-free module. Unknown options are
rejected here unless ``rig run`` or ``rig profile`` places them after an
explicit ``--`` recipe boundary; command execution validates that tail before
forwarding it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .data_routing import dataset_names


PROFILES = ("smoke", "dev", "official")
CHECKPOINT_POLICIES = ("always", "qualifying", "none")
COLORS = ("auto", "always", "never")
_RECIPE_ARGUMENT_EPILOG = (
    "Recipe-local options may follow an explicit boundary, for example: "
    "rig run my_recipe --profile dev -- --my-option value"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rig",
        description="Prepare, run, and score single-entry JAX trainers on TPU slices.",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser(
        "prepare",
        help="interactive machine, cache, and personal-default setup",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    prepare.add_argument(
        "--path", type=Path, help="exact dataset cache root (for example shm/)"
    )
    prepare.add_argument(
        "--profile", choices=PROFILES, help="execution type to prepare"
    )
    prepare.add_argument(
        "--artifacts", type=Path, help="persistent run artifact directory"
    )
    prepare.add_argument(
        "--tpu-vm-count",
        type=positive_int,
        help="number of TPU VM hosts participating in one JAX job",
    )
    prepare.add_argument(
        "--tpu-vm-hosts", help="pdsh expression containing every TPU VM host"
    )
    prepare.add_argument(
        "--run-profile", choices=PROFILES, help="default run execution type"
    )
    prepare.add_argument(
        "--checkpoint-policy",
        choices=CHECKPOINT_POLICIES,
        help="default checkpoint policy for runs",
    )
    prepare.add_argument("--cluster", help="named cluster profile from .rig.toml")
    prepare.add_argument("--color", choices=COLORS, help="terminal color preference")
    prepare.add_argument(
        "--target-loss",
        type=nonnegative_float,
        help="default qualification target for smoke/development runs only",
    )
    prepare.add_argument(
        "--dataset",
        choices=dataset_names(),
        help="immutable corpus used by every non-smoke run",
    )
    prepare.add_argument(
        "--train-shards", type=positive_int, help="override train shard count"
    )
    prepare.add_argument("--offline", action="store_true", help="forbid network access")
    prepare.add_argument(
        "--check-only", action="store_true", help="verify without mutation"
    )
    prepare.add_argument(
        "--force", action="store_true", help="replace invalid cached shards"
    )
    prepare.add_argument(
        "--timeout",
        type=positive_float,
        default=60.0,
        help="per-request network timeout",
    )
    prepare.add_argument(
        "--non-interactive", action="store_true", help="use flags/current defaults"
    )
    prepare.add_argument(
        "--yes", action="store_true", help="accept defaults and run non-interactively"
    )
    prepare.add_argument(
        "--no-doctor", action="store_true", help="skip environment diagnostics"
    )
    prepare.add_argument(
        "--no-download", action="store_true", help="save settings without data work"
    )
    prepare.add_argument(
        "--ship-data",
        action="store_true",
        help="copy prepared data to peers instead of downloading on every host",
    )
    prepare.add_argument(
        "--no-save", action="store_true", help="do not write .rig.toml"
    )

    doctor = commands.add_parser(
        "doctor",
        help="validate Python, JAX, TPU topology, storage, and cached data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    doctor.add_argument("--path", type=Path, help="dataset cache root")
    doctor.add_argument("--profile", choices=PROFILES, help="execution type")
    doctor.add_argument(
        "--require-tpu",
        action="store_true",
        help="require the configured TPU topology",
    )
    doctor.add_argument(
        "--quick", action="store_true", help="skip compile/collective probe"
    )
    doctor.add_argument(
        "--skip-data", action="store_true", help="skip dataset integrity scan"
    )
    doctor.add_argument("--cluster", help="named cluster profile from .rig.toml")
    doctor.add_argument("--color", choices=COLORS)

    run = commands.add_parser(
        "run",
        help="execute, validate, and record one recipe",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=_RECIPE_ARGUMENT_EPILOG,
    )
    run.add_argument("recipe", help="folder name beneath recipes/")
    run.add_argument("--profile", choices=PROFILES)
    run.add_argument(
        "--tier", help="model-family tier; defaults to family.default_tier"
    )
    run.add_argument(
        "--context",
        help="named context preset; defaults to family.default_context",
    )
    run.add_argument(
        "--tokens-per-parameter",
        type=positive_float,
        help="research budget rounded by the recipe to a complete global step",
    )
    run.add_argument(
        "--base-learning-rate",
        type=positive_float,
        help="research override for the family's transferable base learning rate",
    )
    run.add_argument(
        "--batch-size", type=positive_int, help="research-only global batch override"
    )
    run.add_argument(
        "--stop-after-step",
        type=positive_int,
        help="stop early while retaining the fixed-TPP schedule",
    )
    run.add_argument("--name", help="short label folded into the run directory name")
    run.add_argument("--data-path", type=Path)
    run.add_argument("--seed", type=nonnegative_int, default=1337)
    run.add_argument(
        "--timeout", type=positive_float, help="whole-process timeout in seconds"
    )
    run.add_argument(
        "--checkpoint-policy",
        choices=CHECKPOINT_POLICIES,
        help="always keep weights, keep only when qualifying, or never write them",
    )
    run.add_argument("--cluster", help="named cluster profile from .rig.toml")
    run.add_argument("--color", choices=COLORS)
    run.add_argument("--skip-data-check", action="store_true")
    run.add_argument("--study-id", help=argparse.SUPPRESS)
    run.add_argument("--study-point", help=argparse.SUPPRESS)
    run.add_argument("--study-suite-sha256", help=argparse.SUPPRESS)

    profile = commands.add_parser(
        "profile",
        help="capture a bounded XProf trace from a distributed recipe",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=_RECIPE_ARGUMENT_EPILOG,
    )
    profile.add_argument(
        "recipe", nargs="?", default="reference", help="folder name beneath recipes/"
    )
    profile.add_argument(
        "--profile", choices=PROFILES, help="default execution-type override"
    )
    profile.add_argument(
        "--tier", help="model-family tier; defaults to family.default_tier"
    )
    profile.add_argument(
        "--context",
        help="named context preset; defaults to family.default_context",
    )
    profile.add_argument("--tokens-per-parameter", type=positive_float)
    profile.add_argument("--base-learning-rate", type=positive_float)
    profile.add_argument("--batch-size", type=positive_int)
    profile.add_argument("--data-path", type=Path, help="dataset cache root override")
    profile.add_argument("--output-dir", type=Path, required=True)
    profile.add_argument("--stop-after-step", type=positive_int, default=100)
    profile.add_argument("--xprof-start-step", type=positive_int, default=11)
    profile.add_argument("--xprof-steps", type=positive_int, default=10)
    profile.add_argument("--seed", type=nonnegative_int, default=1337)
    profile.add_argument("--timeout", type=positive_float, default=7200.0)
    profile.add_argument("--cluster", help="named cluster profile from .rig.toml")
    profile.add_argument("--color", choices=COLORS)

    verify = commands.add_parser(
        "verify", help="re-validate a captured run and checkpoint"
    )
    verify.add_argument("run", help="run ID or path")
    verify.add_argument("--profile", choices=PROFILES)

    dataset = commands.add_parser(
        "dataset",
        help="prepare and distribute named corpora, independently of settings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    dataset_actions = dataset.add_subparsers(dest="dataset_command", required=True)
    dataset_actions.add_parser("list", help="show every corpus and whether it is here")
    dataset_prepare = dataset_actions.add_parser(
        "prepare", help="download and verify one corpus by name"
    )
    dataset_prepare.add_argument("name", help="corpus name, for example hero")
    dataset_prepare.add_argument(
        "--shards", type=positive_int, help="train shards to fetch; omit for all"
    )
    dataset_prepare.add_argument("--path", type=Path, help="cache root override")
    dataset_prepare.add_argument("--offline", action="store_true")
    dataset_prepare.add_argument("--force", action="store_true")
    dataset_prepare.add_argument("--check-only", action="store_true")
    dataset_prepare.add_argument("--timeout", type=positive_float, default=60.0)
    dataset_prepare.add_argument("--color", choices=COLORS)
    dataset_ship = dataset_actions.add_parser(
        "ship", help="copy a prepared corpus to a cluster that cannot download it"
    )
    dataset_ship.add_argument("name", help="corpus name, for example hero")
    dataset_ship.add_argument("--cluster", help="target cluster profile")
    dataset_ship.add_argument("--color", choices=COLORS)
    dataset_status = dataset_actions.add_parser(
        "status", help="report which corpora are present here and on a cluster"
    )
    dataset_status.add_argument("--cluster", help="also check this cluster's hosts")
    dataset_status.add_argument("--color", choices=COLORS)

    leaderboard = commands.add_parser(
        "leaderboard", help="render recorded qualifying scores"
    )
    leaderboard.add_argument("--profile", choices=PROFILES, default="official")
    leaderboard.add_argument(
        "--cohort", help="full cohort SHA-256; omit to render each cohort separately"
    )
    leaderboard.add_argument("--all-recipes", action="store_true")
    leaderboard.add_argument("--color", choices=COLORS)

    report = commands.add_parser(
        "report",
        help="build a self-contained HTML comparison of completed run logs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    report.add_argument(
        "--runs", type=Path, default=Path("runs"), help="run log directory"
    )
    report.add_argument(
        "--output",
        type=Path,
        default=Path("report.html"),
        help="standalone HTML destination",
    )
    report.add_argument(
        "--layer-snapshots",
        type=nonnegative_int,
        default=1_400,
        help="recorded steps retained per layer-snapshot chart; 0 keeps all",
    )
    report.add_argument(
        "--max-points",
        type=nonnegative_int,
        default=1_400,
        help="embedded points per run and series; 0 keeps every sample",
    )
    report.add_argument("--color", choices=COLORS, help="terminal color preference")
    report.add_argument(
        "--study-export-target",
        type=Path,
        default=None,
        help=(
            "export selected runs plus snapshot and full browser payloads as a "
            "study folder instead of rendering HTML"
        ),
    )
    report.add_argument(
        "--study-name",
        default=None,
        help="folder name for --study-export-target; required with it",
    )
    report.add_argument(
        "--study-full-max-points",
        type=nonnegative_int,
        default=0,
        help=(
            "points retained per run/series in the expanded study-browser "
            "payload; 0 keeps every sample"
        ),
    )
    report.add_argument(
        "--study-full-layer-snapshots",
        type=nonnegative_int,
        default=0,
        help=(
            "step frames retained per layer chart in the expanded "
            "study-browser payload; 0 keeps every frame"
        ),
    )
    report.add_argument(
        "--study-lossy-fuzzy",
        action="store_true",
        help=(
            "add the exact widening-step fuzzy vector companion used by the "
            "study-browser explorer"
        ),
    )
    report.add_argument("--select", help="regular expression matched against run IDs")

    clone = commands.add_parser(
        "clone", help="clone one recipe into a new algorithm folder"
    )
    clone.add_argument("source", nargs="?", default="reference")
    clone.add_argument("name")

    settings = commands.add_parser("settings", help="show resolved local preferences")
    settings.add_argument("--json", action="store_true")
    return parser


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed < float("inf"):
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed < float("inf"):
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return parsed


__all__ = [
    "CHECKPOINT_POLICIES",
    "COLORS",
    "PROFILES",
    "build_parser",
]
