"""Explicit argument groups shared by executable training recipes.

These helpers describe the stable invocation protocol between ``rig`` and a
recipe.  They deliberately do not add scientific controls such as model tier,
context, token horizon, learning rate, or batch size; those remain visible in
each recipe's ``build_parser`` function.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Literal, Sequence

from rig.arguments import COLORS, positive_int


StandardExecutionType = Literal["smoke", "dev", "official"]


# ``run_recipe`` injects these itself and must reject duplicate values from any
# caller supplying ``RunConfig.trainer_args``. Keep this set beside the helpers
# that declare the corresponding trainer-side protocol rather than restating it
# in the process runner.
RUNNER_MANAGED_FLAGS = frozenset(
    {
        "--output-dir",
        "--seed",
        "--profile",
        "--omit-checkpoint",
    }
)

# ``rig run`` and ``rig profile`` own these protocol and common-research flags
# before their explicit ``--`` boundary. A recipe may define every other flag
# locally, but allowing its opaque tail to repeat one of these would make plan
# resolution and the actual invocation order-dependent.
RIG_MANAGED_RECIPE_FLAGS = RUNNER_MANAGED_FLAGS | frozenset(
    {
        "--print-plan",
        "--tier",
        "--context",
        "--tokens-per-parameter",
        "--base-learning-rate",
        "--batch-size",
        "--stop-after-step",
        "--train-data",
        "--val-data",
        "--data-dtype",
        "--data-format",
        "--dataset-id",
        "--tokenizer-id",
        "--downstream-manifest",
        "--downstream-root",
        "--color",
        "--diagnostic-mode",
        "--xprof-dir",
        "--xprof-start-step",
        "--xprof-steps",
    }
)


def new_recipe_parser(*, description: str) -> argparse.ArgumentParser:
    """Create the common strict parser shell without adding any arguments."""

    return argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )


def add_standard_config_arguments(
    group: argparse._ArgumentGroup,
    *,
    default_output_dir: Path,
    profiles: Sequence[str],
) -> None:
    """Add ``--output-dir``, ``--seed``, ``--profile``, ``--color``, and the
    hidden ``--print-plan`` harness protocol flag.
    """

    group.add_argument("--output-dir", type=Path, default=default_output_dir)
    group.add_argument("--seed", type=int, default=1337)
    group.add_argument("--profile", choices=profiles, default=None)
    group.add_argument("--color", choices=COLORS, default="auto")
    group.add_argument("--print-plan", action="store_true", help=argparse.SUPPRESS)


def add_standard_xprof_arguments(parser: argparse.ArgumentParser) -> None:
    """Add ``--xprof-dir``, ``--xprof-start-step``, ``--xprof-steps``,
    ``--diagnostic-mode``, and ``--omit-checkpoint`` in a ``profiling`` group.
    """

    profiling = parser.add_argument_group("profiling")
    profiling.add_argument(
        "--xprof-dir",
        type=Path,
        default=None,
        help="write an XProf trace for a bounded training-step window",
    )
    profiling.add_argument(
        "--xprof-start-step",
        type=positive_int,
        default=None,
        help="first 1-based step to capture; required with --xprof-dir",
    )
    profiling.add_argument(
        "--xprof-steps",
        type=positive_int,
        default=None,
        help="number of consecutive steps to capture; required with --xprof-dir",
    )
    profiling.add_argument(
        "--diagnostic-mode",
        action="store_true",
        help="XProf-only execution without evaluation, diagnostics, checkpoint, or result",
    )
    profiling.add_argument(
        "--omit-checkpoint",
        action="store_true",
        # Internal harness control. Users choose one public
        # `rig run --checkpoint-policy` value; keeping this accepted but hidden
        # lets the generic harness tell a standalone recipe not to serialize.
        help=argparse.SUPPRESS,
    )


def add_standard_data_arguments(parser: argparse.ArgumentParser) -> None:
    """Add ``--train-data``, ``--val-data``, ``--data-dtype``,
    ``--dataset-id``, ``--tokenizer-id``, ``--data-format``,
    ``--downstream-manifest``, and ``--downstream-root`` in a ``data`` group.
    """

    data = parser.add_argument_group("data")
    data.add_argument(
        "--train-data",
        type=Path,
        action="append",
        default=[],
        help="explicit training shard; repeat for multiple shards",
    )
    data.add_argument(
        "--val-data",
        type=Path,
        action="append",
        default=[],
        help="explicit validation shard; repeat for multiple shards",
    )
    data.add_argument(
        "--data-dtype",
        choices=("uint8", "uint16", "uint32", "int32"),
        default="uint16",
        help="dtype for raw .bin token files",
    )
    data.add_argument(
        "--dataset-id", default=None, help="stable dataset identifier for records"
    )
    data.add_argument(
        "--tokenizer-id", default=None, help="stable tokenizer identifier for records"
    )
    data.add_argument(
        "--data-format",
        choices=("auto", "raw", "llmc"),
        default="auto",
        help="raw binaries or llm.c 256-int-header shards",
    )
    data.add_argument(
        "--downstream-manifest",
        type=Path,
        default=None,
        help="fresh10 manifest containing domain shard paths and document spans",
    )
    data.add_argument(
        "--downstream-root",
        type=Path,
        default=None,
        help="directory containing shards named by --downstream-manifest",
    )


def add_standard_reporting_arguments(group: argparse._ArgumentGroup) -> None:
    """Add the hardware-reporting-only ``--peak-tflops`` argument."""

    group.add_argument(
        "--peak-tflops",
        type=float,
        default=None,
        help="hardware bf16 peak for the whole mesh; enables an MFU estimate",
    )


def validate_standard_data_arguments(args: argparse.Namespace) -> None:
    """Validate relationships among arguments added by the data helper."""

    if bool(args.train_data) != bool(args.val_data):
        raise ValueError("--train-data and --val-data must be supplied together")
    if args.downstream_root is not None and args.downstream_manifest is None:
        raise ValueError("--downstream-root requires --downstream-manifest")


def validate_standard_xprof_arguments(
    args: argparse.Namespace, *, execution_type: StandardExecutionType
) -> None:
    """Validate relationships among arguments added by the XProf helper."""

    xprof_window_args = (args.xprof_start_step, args.xprof_steps)
    if args.xprof_dir is None:
        if any(value is not None for value in xprof_window_args):
            raise ValueError("--xprof-start-step and --xprof-steps require --xprof-dir")
        if args.diagnostic_mode:
            raise ValueError("--diagnostic-mode requires --xprof-dir")
    elif any(value is None for value in xprof_window_args):
        raise ValueError(
            "--xprof-dir requires both --xprof-start-step and --xprof-steps"
        )
    if args.omit_checkpoint and args.diagnostic_mode:
        raise ValueError(
            "--omit-checkpoint and --diagnostic-mode are mutually exclusive"
        )
    if args.omit_checkpoint and execution_type != "dev":
        raise ValueError("--omit-checkpoint is restricted to development research runs")
    if args.diagnostic_mode and args.downstream_manifest is not None:
        raise ValueError(
            "--diagnostic-mode cannot be combined with downstream evaluation data"
        )


def validate_standard_reporting_arguments(args: argparse.Namespace) -> None:
    """Validate arguments added by the reporting helper."""

    if args.peak_tflops is not None and (
        not math.isfinite(args.peak_tflops) or args.peak_tflops <= 0.0
    ):
        raise ValueError("--peak-tflops must be positive")


__all__ = (
    "RIG_MANAGED_RECIPE_FLAGS",
    "RUNNER_MANAGED_FLAGS",
    "StandardExecutionType",
    "add_standard_config_arguments",
    "add_standard_data_arguments",
    "add_standard_reporting_arguments",
    "add_standard_xprof_arguments",
    "new_recipe_parser",
    "validate_standard_data_arguments",
    "validate_standard_reporting_arguments",
    "validate_standard_xprof_arguments",
)
