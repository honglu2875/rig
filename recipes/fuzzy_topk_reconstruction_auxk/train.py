#!/usr/bin/env python3
"""A compact, dependency-light GPT trainer for the GPT TPU rig.

Everything involved in training lives in this file.  The default model is sized
for a TPU v4-8 and uses pure JAX: model state is replicated while the global
batch is sharded over every visible device.  ``--smoke`` selects a tiny CPU-
friendly configuration and the built-in byte corpus means the script never
requires a download.

Prepared data is supplied as explicit train and validation shard paths. The
final stdout line of a competition run is a machine-readable result and is
intentionally never colorized. Diagnostic XProf runs deliberately omit it.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
import hashlib
from importlib import metadata as importlib_metadata
import json
import math
import os
from pathlib import Path
import platform as host_platform
import re
import socket
import sys
import time
from dataclasses import dataclass
from typing import Annotated, Any, Iterable, Literal, Mapping, Sequence

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
from jax.experimental import multihost_utils
import yaml
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from rig import logpack, vectorlog
from rig.arguments import (
    nonnegative_float,
    nonnegative_int,
    positive_float,
    positive_int,
)
from rig.attention import (
    AttentionCallable,
    AttentionRuntime,
    attention_console_rows,
    attention_runtime_metadata,
    attention_softmax_scale,
    document_segments,
    make_mesh_attention,
    resolve_attention_runtime,
)
from rig.recipe_args import (
    StandardExecutionType,
    add_standard_config_arguments,
    add_standard_data_arguments,
    add_standard_reporting_arguments,
    add_standard_xprof_arguments,
    new_recipe_parser,
    validate_standard_data_arguments,
    validate_standard_reporting_arguments,
    validate_standard_xprof_arguments,
)
from rig.configfile import profile_config_filename, read_config_document
from rig.configschema import (
    Bounds,
    ConfigSchema,
    Length,
    Matches,
    NonnegativeFloat,
    NonnegativeInt,
    PositiveFloat,
    PositiveInt,
)
from rig.diagnostics import diagnostic_scope_metadata, diagnostic_values
from rig.runlog import (
    CHECKPOINT_NAME,
    DIAGNOSTICS_LOG_NAME,
    DIAGNOSTIC_FLUSH_POINTS,
    RESULT_PREFIX,
    TRAINING_LOG_NAME,
    VALIDATION_CSV_NAME,
    DiagnosticPoint,
    ValidationRow,
    append_log_row,
    close_log,
    diagnostic_log_columns,
    open_log,
    profiler_options,
    save_checkpoint,
    training_log_columns,
    write_diagnostics_log,
    write_result,
    write_training_log,
    write_validation_csv,
    xprof_step_window,
)
from rig.evaluation import (
    EvaluationReport,
    evaluate_downstream_domains,
    evaluate_validation_prefix,
    should_run_diagnostics,
    should_run_validation_probe,
)
from rig.nn import (
    apply_rotary,
    flatten_arrays,
    linear,
    normal,
    parameter_count,
    rms_norm,
)
from rig.tokens import (
    DownstreamDomain,
    ShardedTokens,
    ShuffledEpochBatchStream,
    TokenDataset,
    downstream_batches,
    file_sha256,
    load_dataset,
    load_downstream_domains,
)
from rig.console import (
    Console,
    format_count,
    format_rate,
    standard_data_rows,
    standard_identity_rows,
    standard_kernel_rows,
    standard_schedule_rows,
    standard_training_rows,
)
from rig.mesh import (
    finite_metric,
    initialize_distributed_runtime,
    inferred_peak_tflops,
    is_controller_process,
    local_batch_size,
    local_device_get,
    put_host_local_array,
    put_replicated_tree,
    rank_local_slice,
    sync_tree,
    system_metadata,
    validate_official_topology,
)
from rig.flops import (
    FlopBreakdown,
    FlopError,
    Site,
    count_training_flops,
    default_rules,
    describe,
)
from rig.kernels import (
    AttentionConfig,
    FUZZY_FEATURE_STAT_NAMES,
    FuzzyTopKCallable,
    FuzzyTopKConfig,
    FuzzyTopKDiagnosticCallable,
    FuzzyTopKReconstructionAuxKCallable,
    FuzzyTopKReconstructionCallable,
    FuzzyTopKReconstructionConfig,
    RECONSTRUCTION_AUXK_STAT_NAMES,
    RECONSTRUCTION_STAT_NAMES,
    fuzzy_topk_mlp,
    fuzzy_topk_mlp_with_reconstruction,
    fuzzy_topk_mlp_with_reconstruction_auxk,
    make_causal_attention,
    make_mesh_fuzzy_topk_mlp,
    make_mesh_fuzzy_topk_mlp_with_diagnostics,
    make_mesh_fuzzy_topk_mlp_with_reconstruction,
    make_mesh_fuzzy_topk_mlp_with_reconstruction_auxk,
    select_attention_tiles,
    tiled_tied_cross_entropy,
    tiled_tied_cross_entropy_losses,
)


SCHEMA_VERSION = 1
RECIPE_DIR = Path(__file__).resolve().parent
RECIPE_NAME = RECIPE_DIR.name
_VALID_PROFILES = ("smoke", "dev", "official")
_DOMAIN_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_TIER_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
_CONTEXT_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
FUZZY_SPARSITY_LOG_NAME = f"fuzzy_sparsity{vectorlog.SUFFIX}"
_FUZZY_SPARSITY_TEMP_NAME = f".{FUZZY_SPARSITY_LOG_NAME}.tmp"
AUXK_AGE_STAT_NAMES = (
    "fuzzy_auxk.batch_active_fraction",
    "fuzzy_auxk.tracked_dead_fraction",
)


# A deliberately small, original corpus for offline and smoke-test use.  The
# repeated motifs make it possible for tiny models to show measurable progress,
# while the shuffled clauses prevent every training window from being identical.


@dataclass(frozen=True, slots=True)
class Config:
    steps: int
    batch_size: int
    seq_len: int
    sampling: str
    layers: int
    heads: int
    d_model: int
    mlp_mult: int
    mlp_top_k: int
    normalization: str
    position_encoding: str
    mlp_activation: str
    tier: str
    # Stable run-protocol name, derived from the typed model definition rather
    # than declared independently in the selected YAML.
    declared_parameters: int | None
    parameterization: str
    base_width: int
    base_depth: int
    depth_alpha: float
    init_std: float
    attention_scale: str
    embeddings: str
    data_multiplier: float
    batch_multiplier: float
    target_tokens_per_parameter: float | None
    learning_rate: float
    min_lr_ratio: float
    warmup_steps: int
    weight_decay: float
    adam_epsilon: float
    beta1: float
    beta2: float
    grad_clip: float
    eval_batches: int
    val_every: int
    val_probe_batches: int
    diagnostics_every: int
    sparsity_diagnostics_every: int
    log_every: int
    vocab_size: int
    semantic_vocab_size: int
    attention_backend: str
    sparse_mlp_backend: str
    reconstruction_coefficient: float
    reconstruction_decoder_unit_norm: bool
    auxk_mode: str
    auxk_coefficient: float
    auxk_width_ratio: float
    dead_tokens_threshold: int
    loss_backend: str
    vocab_tile_size: int
    dtype_name: str
    config_schema_version: int
    config_sha256: str
    config_filename: str
    execution_type: str
    context_preset: str
    # Optimizer step after which to stop. steps, warmup, and m_D still resolve
    # from the full horizon, so the trajectory matches the untruncated run up
    # to this point. None runs to completion.
    stop_after_step: int | None = None
    # Block-diagonal attention over documents. The selected context preset owns
    # this policy together with sequence length and the recipe-local batch anchor.
    document_masking: bool = False
    document_boundary_token: int = 50256

    @property
    def final_step(self) -> int:
        """Last optimizer step this run takes; steps stays the schedule horizon."""

        return self.stop_after_step or self.steps

    @property
    def width_multiplier(self) -> float:
        """Width ratio derived from the selected and base tiers."""

        return self.d_model / float(self.base_width)

    @property
    def depth_multiplier(self) -> float:
        """Depth ratio derived from the selected and base tiers."""

        return self.layers / float(self.base_depth)

    @property
    def tokens_per_parameter(self) -> float | None:
        """Achieved full-schedule TPP after rounding to whole optimizer steps."""

        if self.declared_parameters is None:
            return None
        return (
            self.steps
            * self.batch_size
            * self.seq_len
            / float(self.declared_parameters)
        )

    @property
    def validation_predictions(self) -> int:
        """Number of next-token predictions in the final validation pass."""

        return self.eval_batches * self.batch_size * self.seq_len

    @property
    def auxk_enabled(self) -> bool:
        """Whether the literal residual-reconstruction AuxK path is active."""

        return self.auxk_mode == "auxk"

    @property
    def aux_k(self) -> int:
        """Number of fuzzy auxiliary winners per token and layer."""

        return int(round(self.d_model * self.auxk_width_ratio))

    @property
    def auxk_cohort_count(self) -> int:
        """Number of rotating fixed-group cohorts visited by AuxK."""

        return self.mlp_top_k // self.aux_k

    @property
    def dead_after_steps(self) -> int:
        """Global steps corresponding to the configured token-age threshold."""

        tokens_per_step = self.batch_size * self.seq_len
        return max(1, math.ceil(self.dead_tokens_threshold / tokens_per_step))

    @property
    def reconstruction_parameter_count(self) -> int:
        """Parameters discarded together with the training-only decoder."""

        return self.layers * self.mlp_mult * self.d_model * self.d_model

    @property
    def compute_dtype(self) -> Any:
        """JAX dtype derived from the serializable dtype name."""

        return jnp.bfloat16 if self.dtype_name == "bfloat16" else jnp.float32


TierName = Annotated[str, Matches(_TIER_NAME.pattern)]
ContextName = Annotated[str, Matches(_CONTEXT_NAME.pattern)]
Probability = Annotated[float, Bounds(ge=0.0, le=1.0)]
OpenProbability = Annotated[float, Bounds(ge=0.0, lt=1.0)]
DepthAlpha = Annotated[float, Bounds(ge=0.0, le=1.0)]


@dataclass(frozen=True, slots=True)
class ContextPreset:
    """One coupled sequence-length, batch-anchor, and masking preset."""

    seq_len: PositiveInt
    reference_batch_size: PositiveInt
    document_masking: bool

    @property
    def tokens_per_step(self) -> int:
        """Number of tokens in one recipe-default global optimizer step."""

        return self.reference_batch_size * self.seq_len


@dataclass(frozen=True, slots=True)
class ParameterizationDefinition:
    """Initialization and optimizer scaling shared by every family tier."""

    name: Literal["standard", "completep_fixed_tpp_v1"]
    base_tier: TierName
    depth_alpha: DepthAlpha
    init_std: PositiveFloat
    attention_scale: Literal["inverse_sqrt_head_dim", "inverse_head_dim"]
    embeddings: Literal["tied", "untied"]


@dataclass(frozen=True, slots=True)
class ModelDefinition:
    """Architecture fields represented literally in the selected YAML."""

    layers: PositiveInt
    heads: PositiveInt
    d_model: PositiveInt
    mlp_mult: PositiveInt
    mlp_top_k: PositiveInt
    normalization: Literal["rms_norm"]
    position_encoding: Literal["rope_base_10000"]
    mlp_activation: Literal["fuzzy_topk_relu"]
    vocab_size: PositiveInt
    semantic_vocab_size: PositiveInt

    @property
    def head_dim(self) -> int:
        """Width of one attention head."""

        return self.d_model // self.heads

    def validate(self, label: str) -> None:
        """Enforce architecture relations that no single annotation can express."""

        if self.semantic_vocab_size > self.vocab_size:
            raise ValueError(f"{label}.semantic_vocab_size must not exceed vocab_size")
        if self.d_model % self.heads:
            raise ValueError(f"{label}.d_model must be divisible by heads")
        if self.head_dim % 2:
            raise ValueError(f"{label} head dimension must be even for RoPE")
        if self.mlp_top_k > self.mlp_mult * self.d_model:
            raise ValueError(f"{label}.mlp_top_k must not exceed mlp_mult * d_model")
        hidden_width = self.mlp_mult * self.d_model
        if hidden_width % self.mlp_top_k:
            raise ValueError(f"{label}.mlp_top_k must divide mlp_mult * d_model")


@dataclass(frozen=True, slots=True)
class TierDefinition:
    model: ModelDefinition

    @property
    def tpp_parameters(self) -> int:
        """Parameter denominator used by the fixed-TPP ladder."""

        return self.tpp_parameters_for_mlp_mult(self.model.mlp_mult)

    def tpp_parameters_for_mlp_mult(self, mlp_mult: int) -> int:
        """Stored parameter count for one explicit dictionary expansion."""

        return self.stored_parameters(
            layers=self.model.layers,
            mlp_mult=mlp_mult,
        )

    def stored_parameters(self, *, layers: int, mlp_mult: int) -> int:
        """Stored parameters for one explicit depth/dictionary treatment."""

        model = self.model
        width = model.d_model
        return (
            2 * model.vocab_size * width
            + layers * ((4 + 2 * mlp_mult) * width * width + (mlp_mult + 7) * width)
            + width
        )


@dataclass(frozen=True, slots=True)
class FamilyDefinition:
    default_tier: TierName
    default_context: ContextName
    contexts: Annotated[dict[ContextName, ContextPreset], Length(ge=1)]
    parameterization: ParameterizationDefinition
    tiers: Annotated[dict[TierName, TierDefinition], Length(ge=1)]

    @property
    def base_model(self) -> ModelDefinition:
        """Architecture anchoring the parameterization multipliers."""

        return self.tiers[self.parameterization.base_tier].model

    @property
    def base_width(self) -> int:
        return self.base_model.d_model

    @property
    def base_depth(self) -> int:
        return self.base_model.layers

    @property
    def base_tpp_parameters(self) -> int:
        return self.tiers[self.parameterization.base_tier].tpp_parameters

    def base_tpp_parameters_for_mlp_mult(self, mlp_mult: int) -> int:
        """Base-tier denominator under the same sparse dictionary width."""

        return self.tiers[self.parameterization.base_tier].tpp_parameters_for_mlp_mult(
            mlp_mult
        )


Sampling = Literal["random_windows", "shuffled_epochs"]
ComputeDtype = Literal["bfloat16", "float32"]


@dataclass(frozen=True, slots=True)
class DurationDefinition:
    """One explicit stopping policy; exactly one field must be present."""

    steps: PositiveInt | None = None
    tokens_per_parameter: PositiveFloat | None = None

    def validate(self, label: str) -> None:
        if (self.steps is None) == (self.tokens_per_parameter is None):
            raise ValueError(
                f"{label} must define exactly one of steps and tokens_per_parameter"
            )

    @property
    def is_fixed_tpp(self) -> bool:
        return self.tokens_per_parameter is not None


@dataclass(frozen=True, slots=True)
class TrainingSettings:
    duration: DurationDefinition
    sampling: Sampling
    dtype: ComputeDtype


@dataclass(frozen=True, slots=True)
class KernelSettings:
    attention_backend: Literal["dense", "jax_flash", "tpu_flash"]
    sparse_mlp_backend: Literal["choicewise", "reference"]
    loss_backend: Literal["dense", "tiled"]
    vocab_tile_size: PositiveInt


@dataclass(frozen=True, slots=True)
class OptimizerSettings:
    learning_rate: PositiveFloat
    min_lr_ratio: Probability
    warmup_ratio: OpenProbability
    weight_decay: NonnegativeFloat
    adam_epsilon: PositiveFloat
    beta1: OpenProbability
    beta2: OpenProbability
    grad_clip: NonnegativeFloat


@dataclass(frozen=True, slots=True)
class ValidationProbeSettings:
    every_steps: PositiveInt
    predictions: PositiveInt


@dataclass(frozen=True, slots=True)
class EvaluationSettings:
    final_predictions: PositiveInt
    probe: ValidationProbeSettings | None


@dataclass(frozen=True, slots=True)
class LoggingSettings:
    diagnostics_every: NonnegativeInt
    sparsity_diagnostics_every: NonnegativeInt
    log_every: PositiveInt


@dataclass(frozen=True, slots=True)
class AuxKSettings:
    """Literal dead-feature residual reconstruction settings."""

    mode: Literal["none", "auxk"]
    coefficient: NonnegativeFloat
    width_ratio: PositiveFloat
    dead_tokens_threshold: PositiveInt

    def validate(self, label: str) -> None:
        if self.mode == "none" and self.coefficient != 0.0:
            raise ValueError(f"{label}.coefficient must be zero when mode is none")
        if self.mode == "auxk" and self.coefficient == 0.0:
            raise ValueError(f"{label}.coefficient must be positive in auxk mode")


@dataclass(frozen=True, slots=True)
class ReconstructionSettings:
    """Train-only decoder objective attached to every fuzzy MLP block."""

    coefficient: PositiveFloat
    decoder_unit_norm: bool
    auxk: AuxKSettings

    def validate(self, label: str) -> None:
        if not self.decoder_unit_norm:
            raise ValueError(f"{label}.decoder_unit_norm must remain true")
        self.auxk.validate(f"{label}.auxk")


@dataclass(frozen=True, slots=True)
class RunDefinition:
    training: TrainingSettings
    kernels: KernelSettings
    optimizer: OptimizerSettings
    evaluation: EvaluationSettings
    logging: LoggingSettings
    reconstruction: ReconstructionSettings


@dataclass(frozen=True, slots=True)
class ExperimentConfig(ConfigSchema):
    """Complete typed representation of one selected standalone YAML file."""

    schema_version: Literal[6]
    execution_type: StandardExecutionType
    family: FamilyDefinition
    run: RunDefinition

    def validate(self, label: str) -> None:
        """Enforce the few scientific contracts involving multiple fields."""

        family = self.family
        if family.default_context not in family.contexts:
            raise ValueError(
                f"{label} family.default_context must name a defined context preset"
            )
        if family.default_tier not in family.tiers:
            raise ValueError(f"{label} family.default_tier must name a defined tier")
        base_tier = family.parameterization.base_tier
        if base_tier not in family.tiers:
            raise ValueError(
                f"{label} family.parameterization.base_tier must name a defined tier"
            )

        for tier_name, tier in family.tiers.items():
            model_label = f"{label} family.tiers.{tier_name}.model"
            tier.model.validate(model_label)
            if (
                family.parameterization.attention_scale == "inverse_head_dim"
                and tier.model.head_dim != 64
            ):
                raise ValueError(
                    f"{label} family.tiers.{tier_name} must use 64-wide heads"
                )

        run = self.run
        run.training.duration.validate(f"{label} run.training.duration")
        run.reconstruction.validate(f"{label} run.reconstruction")
        if (
            run.kernels.attention_backend != "dense"
            and run.training.dtype != "bfloat16"
        ):
            raise ValueError(
                f"{label} run.kernels.attention_backend "
                f"{run.kernels.attention_backend} requires training.dtype bfloat16"
            )
        probe = run.evaluation.probe
        if probe is not None and probe.predictions > run.evaluation.final_predictions:
            raise ValueError(
                f"{label} run.evaluation.probe.predictions must not exceed "
                "final_predictions"
            )
        self._validate_evaluation(family.default_context, label)

    def _validate_evaluation(self, context_name: str, label: str) -> None:
        """Check an explicitly requested fixed validation prediction budget."""

        tokens_per_step = self.family.contexts[context_name].tokens_per_step
        evaluation = self.run.evaluation
        budgets = [("final_predictions", evaluation.final_predictions)]
        if evaluation.probe is not None:
            budgets.append(("probe.predictions", evaluation.probe.predictions))
        for field, predictions in budgets:
            if predictions % tokens_per_step:
                raise ValueError(
                    f"{label} batch_size * seq_len must divide "
                    f"run.evaluation.{field} ({predictions:,})"
                )

    def resolve_selection(
        self,
        *,
        tier: str | None = None,
        context: str | None = None,
        label: str = "experiment config",
    ) -> tuple[str, str]:
        """Resolve runtime selectors on an already-validated config."""

        tier_name = tier or self.family.default_tier
        if tier_name not in self.family.tiers:
            raise ValueError(
                f"unknown model tier {tier_name!r}; expected "
                + ", ".join(sorted(self.family.tiers))
            )
        context_name = context or self.family.default_context
        if context_name not in self.family.contexts:
            raise ValueError(
                f"unknown context preset {context_name!r}; expected "
                + ", ".join(sorted(self.family.contexts))
            )

        self._validate_evaluation(context_name, label)

        return tier_name, context_name


_UINT64_MASK = (1 << 64) - 1


def experiment_config_path(profile: str) -> Path:
    """Return the sole configuration document selected by ``profile``."""

    if profile not in _VALID_PROFILES:
        raise ValueError(f"unknown experiment profile: {profile!r}")
    return RECIPE_DIR / profile_config_filename(profile)


def load_experiment_config(profile: str) -> tuple[ExperimentConfig, str]:
    """Load and validate one standalone YAML config with its source digest."""

    path = experiment_config_path(profile)
    mapping, source_sha256 = read_config_document(path)
    experiment_config = ExperimentConfig.from_mapping(mapping, label=path.name)
    if experiment_config.execution_type != profile:
        raise ValueError(
            f"{path.name} execution_type must be {profile!r}; "
            f"got {experiment_config.execution_type!r}"
        )
    experiment_config.validate(path.name)
    return experiment_config, source_sha256


def build_parser() -> argparse.ArgumentParser:
    parser = new_recipe_parser(
        description=(
            "Train a decoder-only GPT with JAX. Static experiment settings come "
            "from the execution-type-selected YAML beside this entry script."
        )
    )
    run = parser.add_argument_group("run")
    add_standard_config_arguments(
        run,
        default_output_dir=Path("runs") / RECIPE_NAME,
        profiles=_VALID_PROFILES,
    )
    run.add_argument(
        "--tier",
        default=None,
        help="model-family size tier; defaults to family.default_tier",
    )
    run.add_argument(
        "--context",
        default=None,
        help="named context preset; defaults to family.default_context",
    )
    run.add_argument(
        "--stop-after-step",
        type=positive_int,
        default=None,
        help=(
            "stop after this optimizer step while keeping the configured "
            "fixed-TPP schedule unchanged"
        ),
    )
    run.add_argument(
        "--tokens-per-parameter",
        type=float,
        default=None,
        help="research budget rounded to the nearest complete global step",
    )
    add_standard_xprof_arguments(parser)

    add_standard_data_arguments(parser)

    optim = parser.add_argument_group("optimization")
    optim.add_argument(
        "--base-learning-rate",
        type=float,
        default=None,
        help="research override for the transferable base learning rate",
    )
    optim.add_argument(
        "--batch-size",
        type=positive_int,
        default=None,
        help="research override for the global sequence batch",
    )
    sparse = parser.add_argument_group("sparse MLP")
    sparse.add_argument(
        "--sparse-layers",
        type=positive_int,
        default=None,
        help="research override for transformer depth",
    )
    sparse.add_argument(
        "--sparse-mlp-mult",
        type=positive_int,
        default=None,
        help="research override for stored dictionary width / d_model",
    )
    sparse.add_argument(
        "--sparse-top-k",
        type=positive_int,
        default=None,
        help="research override for grouped approximate-TopK coordinates per token",
    )
    sparse.add_argument(
        "--sparse-mlp-backend",
        choices=("choicewise", "reference"),
        default=None,
        help="regular choice-wise contractions or literal gathered oracle",
    )
    sparse.add_argument(
        "--sparse-training-steps",
        type=positive_int,
        default=None,
        help=(
            "explicit full cosine-schedule horizon for equi-FLOP research; "
            "mutually exclusive with --tokens-per-parameter"
        ),
    )
    sparse.add_argument(
        "--sparsity-diagnostics-every",
        type=nonnegative_int,
        default=None,
        help=(
            "override the per-feature diagnostic cadence; zero disables it "
            "without changing the training computation"
        ),
    )
    sparse.add_argument(
        "--reconstruction-coefficient",
        type=positive_float,
        default=None,
        help="coefficient on the mean per-layer normalized reconstruction MSE",
    )
    sparse.add_argument(
        "--fuzzy-auxk-mode",
        choices=("none", "auxk"),
        default=None,
        help="disable or add literal dead-feature residual reconstruction",
    )
    sparse.add_argument(
        "--fuzzy-auxk-coefficient",
        type=nonnegative_float,
        default=None,
        help="coefficient on mean per-layer AuxK NMSE (paper default: 1/32)",
    )
    sparse.add_argument(
        "--fuzzy-auxk-width-ratio",
        type=positive_float,
        default=None,
        help="auxiliary winners per token divided by d_model",
    )
    sparse.add_argument(
        "--fuzzy-dead-tokens-threshold",
        type=positive_int,
        default=None,
        help="tokens without a positive main activation before a feature is dead",
    )
    add_standard_reporting_arguments(optim)
    return parser


def validate_args(
    args: argparse.Namespace,
    experiment_config: ExperimentConfig,
) -> None:
    profile = selected_profile(args)
    config_filename = profile_config_filename(profile)
    experiment_config.resolve_selection(
        tier=args.tier, context=args.context, label=config_filename
    )
    if args.tokens_per_parameter is not None and (
        not math.isfinite(args.tokens_per_parameter) or args.tokens_per_parameter <= 0.0
    ):
        raise ValueError("--tokens-per-parameter must be finite and positive")
    if args.base_learning_rate is not None and (
        not math.isfinite(args.base_learning_rate) or args.base_learning_rate <= 0.0
    ):
        raise ValueError("--base-learning-rate must be finite and positive")
    if args.sparse_training_steps is not None and args.tokens_per_parameter is not None:
        raise ValueError(
            "--sparse-training-steps and --tokens-per-parameter are mutually exclusive"
        )
    validate_standard_data_arguments(args)
    validate_standard_reporting_arguments(args)
    validate_standard_xprof_arguments(
        args, execution_type=experiment_config.execution_type
    )


def should_compile_evaluation(
    args: argparse.Namespace,
    config: Config,
    downstream_domains: Sequence[DownstreamDomain],
) -> bool:
    """Return whether this invocation can execute any validation workload."""

    return not args.diagnostic_mode or config.val_every > 0 or bool(downstream_domains)


def should_run_sparsity_diagnostics(
    step: int, *, every: int, final_step: int
) -> bool:
    """Capture the first batch, fixed cadence, and exact final batch."""

    return every > 0 and (step == 1 or step % every == 0 or step == final_step)


def selected_profile(args: argparse.Namespace) -> str:
    return args.profile or "dev"


def resolve_config(
    args: argparse.Namespace,
    platform: str,
    *,
    experiment_config: ExperimentConfig,
    config_sha256: str,
) -> Config:
    profile = selected_profile(args)
    config_filename = profile_config_filename(profile)
    selected_tier_name, selected_context_name = experiment_config.resolve_selection(
        tier=args.tier, context=args.context, label=config_filename
    )
    family = experiment_config.family
    definition = experiment_config.run
    reconstruction = definition.reconstruction
    tier = family.tiers[selected_tier_name]
    model = tier.model
    layers = args.sparse_layers or model.layers
    mlp_mult = args.sparse_mlp_mult or model.mlp_mult
    mlp_top_k = args.sparse_top_k or model.mlp_top_k
    hidden_width = mlp_mult * model.d_model
    if mlp_top_k > hidden_width:
        raise ValueError(
            f"--sparse-top-k {mlp_top_k} exceeds the resolved hidden width "
            f"{hidden_width}"
        )
    if hidden_width % mlp_top_k:
        raise ValueError(
            f"--sparse-top-k {mlp_top_k} must divide the resolved hidden width "
            f"{hidden_width}"
        )
    base_tpp_parameters = family.base_tpp_parameters_for_mlp_mult(mlp_mult)
    duration = definition.training.duration
    tpp_parameters = (
        tier.stored_parameters(layers=layers, mlp_mult=mlp_mult)
        if duration.is_fixed_tpp
        else None
    )
    kernels = definition.kernels
    optimizer = definition.optimizer
    evaluation = definition.evaluation
    logging = definition.logging
    training = definition.training
    context = family.contexts[selected_context_name]
    family_parameterization = family.parameterization
    batch_anchor = context.reference_batch_size
    seq_len = context.seq_len
    document_masking = context.document_masking
    requested_tpp = (
        args.tokens_per_parameter
        if args.tokens_per_parameter is not None
        else duration.tokens_per_parameter
    )
    tier_name = selected_tier_name
    parameterization = family_parameterization.name
    base_width = family.base_width
    base_depth = family.base_depth
    depth_alpha = family_parameterization.depth_alpha
    init_std = family_parameterization.init_std
    attention_scale = family_parameterization.attention_scale
    embeddings = family_parameterization.embeddings
    context_preset = selected_context_name

    batch_size = args.batch_size or batch_anchor
    tokens_per_step = batch_size * seq_len
    early_stop = getattr(args, "stop_after_step", None)
    explicit_steps = args.sparse_training_steps
    if explicit_steps is not None:
        if not duration.is_fixed_tpp:
            raise ValueError(
                "--sparse-training-steps requires a fixed-TPP dev or official profile"
            )
        if tpp_parameters is None:
            raise AssertionError("explicit step horizon has no parameter denominator")
        steps = explicit_steps
        requested_tpp = steps * tokens_per_step / float(tpp_parameters)
    elif not duration.is_fixed_tpp:
        if args.tokens_per_parameter is not None:
            raise ValueError(
                "--tokens-per-parameter cannot override a fixed-step configuration"
            )
        if early_stop is not None:
            raise ValueError("--stop-after-step requires a fixed-TPP configuration")
        if duration.steps is None:
            raise AssertionError("fixed-step duration has no step count")
        steps = duration.steps
    else:
        if requested_tpp is None:
            raise AssertionError("fixed-TPP duration did not resolve a horizon")
        if tpp_parameters is None:
            raise AssertionError("fixed-TPP duration has no parameter denominator")
        ideal_tokens = float(tpp_parameters) * requested_tpp
        steps = max(1, int(math.floor(ideal_tokens / tokens_per_step + 0.5)))
    if early_stop is not None and early_stop > steps:
        raise ValueError(
            f"--stop-after-step {early_stop:,} is past the {steps:,}-step "
            "horizon this configuration resolves to"
        )

    predictions_per_batch = batch_size * seq_len

    def evaluation_batches(predictions: int, field: str) -> int:
        if predictions % predictions_per_batch:
            raise ValueError(
                f"{config_filename} run.evaluation.{field} requires batch_size * "
                f"seq_len ({predictions_per_batch:,}) to divide {predictions:,} exactly"
            )
        return predictions // predictions_per_batch

    eval_batches = evaluation_batches(evaluation.final_predictions, "final_predictions")
    probe = evaluation.probe
    val_every = 0 if args.diagnostic_mode or probe is None else probe.every_steps
    val_probe_batches = (
        evaluation_batches(probe.predictions, "probe.predictions")
        if probe is not None
        else 0
    )
    log_every = steps if args.diagnostic_mode else logging.log_every
    diagnostics_every = 0 if args.diagnostic_mode else logging.diagnostics_every
    configured_sparsity_cadence = (
        logging.sparsity_diagnostics_every
        if args.sparsity_diagnostics_every is None
        else args.sparsity_diagnostics_every
    )
    sparsity_diagnostics_every = (
        0 if args.diagnostic_mode else configured_sparsity_cadence
    )

    dtype_name = training.dtype
    compute_dtype = jnp.bfloat16 if dtype_name == "bfloat16" else jnp.float32
    attention_backend = kernels.attention_backend
    sparse_mlp_backend = args.sparse_mlp_backend or kernels.sparse_mlp_backend
    reconstruction_coefficient = (
        reconstruction.coefficient
        if args.reconstruction_coefficient is None
        else args.reconstruction_coefficient
    )
    auxk = reconstruction.auxk
    auxk_mode = args.fuzzy_auxk_mode or auxk.mode
    auxk_coefficient = (
        auxk.coefficient
        if args.fuzzy_auxk_coefficient is None
        else args.fuzzy_auxk_coefficient
    )
    auxk_width_ratio = (
        auxk.width_ratio
        if args.fuzzy_auxk_width_ratio is None
        else args.fuzzy_auxk_width_ratio
    )
    dead_tokens_threshold = (
        auxk.dead_tokens_threshold
        if args.fuzzy_dead_tokens_threshold is None
        else args.fuzzy_dead_tokens_threshold
    )
    if sparse_mlp_backend != "choicewise":
        raise ValueError("the reconstruction decoder requires the choicewise backend")
    if auxk_mode == "none" and auxk_coefficient != 0.0:
        raise ValueError("--fuzzy-auxk-coefficient must be zero when mode is none")
    if auxk_mode == "auxk" and auxk_coefficient == 0.0:
        raise ValueError("--fuzzy-auxk-coefficient must be positive in auxk mode")
    aux_k_float = model.d_model * auxk_width_ratio
    aux_k = int(round(aux_k_float))
    if not math.isclose(aux_k_float, aux_k, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError("--fuzzy-auxk-width-ratio must resolve to a whole aux_k")
    if aux_k <= 0 or aux_k > mlp_top_k or mlp_top_k % aux_k:
        raise ValueError(
            "resolved aux_k must be positive and divide the grouped fuzzy top_k"
        )
    if attention_backend != "dense" and platform != "tpu":
        raise ValueError(
            f"{config_filename} attention_backend {attention_backend} requires a TPU runtime"
        )
    if attention_backend != "dense" and compute_dtype != jnp.bfloat16:
        raise ValueError(
            f"{config_filename} attention_backend {attention_backend} currently requires "
            "dtype bfloat16"
        )

    batch_multiplier = batch_size / float(batch_anchor)
    # This project reanchors every TPP ladder. The multiplier captures only the
    # model-size-induced data growth within one fixed-TPP ladder; it deliberately
    # omits any cross-horizon TPP / TPP_0 factor.
    # An explicit step horizon defines a new equi-FLOP comparison anchor. Do
    # not reinterpret its intentionally different token count as fixed-TPP
    # model-ladder scaling in the optimizer.
    data_multiplier = (
        1.0
        if explicit_steps is not None
        else (
            tpp_parameters / float(base_tpp_parameters)
            if tpp_parameters is not None
            else 1.0
        )
    )
    base_learning_rate = (
        args.base_learning_rate
        if args.base_learning_rate is not None
        else optimizer.learning_rate
    )
    warmup_steps = int(math.floor(steps * optimizer.warmup_ratio + 0.5))
    if steps > 1:
        warmup_steps = min(warmup_steps, steps - 1)
    else:
        warmup_steps = 0

    return Config(
        steps=steps,
        stop_after_step=early_stop,
        document_masking=document_masking,
        batch_size=batch_size,
        seq_len=seq_len,
        sampling=training.sampling,
        layers=layers,
        heads=model.heads,
        d_model=model.d_model,
        mlp_mult=mlp_mult,
        mlp_top_k=mlp_top_k,
        normalization=model.normalization,
        position_encoding=model.position_encoding,
        mlp_activation=model.mlp_activation,
        tier=tier_name,
        declared_parameters=tpp_parameters,
        parameterization=parameterization,
        base_width=base_width,
        base_depth=base_depth,
        depth_alpha=depth_alpha,
        init_std=init_std,
        attention_scale=attention_scale,
        embeddings=embeddings,
        data_multiplier=data_multiplier,
        batch_multiplier=batch_multiplier,
        target_tokens_per_parameter=requested_tpp,
        learning_rate=base_learning_rate,
        min_lr_ratio=optimizer.min_lr_ratio,
        warmup_steps=warmup_steps,
        weight_decay=optimizer.weight_decay,
        adam_epsilon=optimizer.adam_epsilon,
        beta1=optimizer.beta1,
        beta2=optimizer.beta2,
        grad_clip=optimizer.grad_clip,
        eval_batches=eval_batches,
        val_every=val_every,
        val_probe_batches=val_probe_batches,
        diagnostics_every=diagnostics_every,
        sparsity_diagnostics_every=sparsity_diagnostics_every,
        log_every=log_every,
        vocab_size=model.vocab_size,
        semantic_vocab_size=model.semantic_vocab_size,
        attention_backend=attention_backend,
        sparse_mlp_backend=sparse_mlp_backend,
        reconstruction_coefficient=reconstruction_coefficient,
        reconstruction_decoder_unit_norm=reconstruction.decoder_unit_norm,
        auxk_mode=auxk_mode,
        auxk_coefficient=auxk_coefficient,
        auxk_width_ratio=auxk_width_ratio,
        dead_tokens_threshold=dead_tokens_threshold,
        loss_backend=kernels.loss_backend,
        vocab_tile_size=kernels.vocab_tile_size,
        dtype_name=dtype_name,
        config_schema_version=experiment_config.schema_version,
        config_sha256=config_sha256,
        config_filename=config_filename,
        execution_type=experiment_config.execution_type,
        context_preset=context_preset,
    )


def init_params(config: Config, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    d_model = config.d_model
    hidden = config.mlp_mult * d_model
    hidden_scale = (
        config.init_std / math.sqrt(config.width_multiplier)
        if config.parameterization == "completep_fixed_tpp_v1"
        else 0.02
    )
    blocks: list[dict[str, np.ndarray]] = []
    for _ in range(config.layers):
        # Preserve the parent's RNG order exactly: both deployed matrices are
        # drawn before the train-only decoder is derived without consuming RNG.
        qkv_w = normal(rng, (d_model, 3 * d_model), hidden_scale)
        attn_w = normal(rng, (d_model, d_model), hidden_scale)
        mlp_up_w = normal(rng, (d_model, hidden), hidden_scale)
        mlp_down_w = normal(rng, (hidden, d_model), hidden_scale)
        reconstruction_w = mlp_up_w.T.copy()
        reconstruction_w /= np.maximum(
            np.linalg.norm(reconstruction_w, axis=-1, keepdims=True), 1.0e-12
        )
        blocks.append(
            {
                "ln1_scale": np.ones((d_model,), dtype=np.float32),
                "qkv_w": qkv_w,
                "qkv_b": np.zeros((3 * d_model,), dtype=np.float32),
                "attn_w": attn_w,
                "attn_b": np.zeros((d_model,), dtype=np.float32),
                "ln2_scale": np.ones((d_model,), dtype=np.float32),
                "mlp_up_w": mlp_up_w,
                "mlp_up_b": np.zeros((hidden,), dtype=np.float32),
                "mlp_down_w": mlp_down_w,
                "mlp_down_b": np.zeros((d_model,), dtype=np.float32),
                "mlp_reconstruction_w": reconstruction_w,
            }
        )
    result = {
        "token_embedding": normal(rng, (config.vocab_size, d_model), config.init_std),
        "blocks": blocks,
        "final_ln_scale": np.ones((d_model,), dtype=np.float32),
    }
    if config.embeddings == "untied":
        result["output_embedding"] = normal(
            rng,
            (config.vocab_size, d_model),
            config.init_std / config.width_multiplier,
        )
    return result


def contract_model_metadata(config: Config) -> dict[str, Any]:
    """Return the resolved model architecture metadata."""

    return {
        "layers": config.layers,
        "heads": config.heads,
        "d_model": config.d_model,
        "mlp_mult": config.mlp_mult,
        "mlp_top_k": config.mlp_top_k,
        "normalization": config.normalization,
        "position_encoding": config.position_encoding,
        "mlp_activation": config.mlp_activation,
        "vocab_size": config.vocab_size,
        "semantic_vocab_size": config.semantic_vocab_size,
        "tied_embeddings": config.embeddings == "tied",
        "tier": config.tier,
        "parameterization": config.parameterization,
    }


def reconstruction_metadata(config: Config) -> dict[str, Any]:
    """Return the exact train-only reconstruction/AuxK contract."""

    return {
        "target": "stop_gradient(rms_norm_mlp_input)",
        "decoder": "separate_no_bias_discard_after_training",
        "decoder_initialization": "unit_rows(transpose(mlp_up_w))",
        "decoder_unit_norm": config.reconstruction_decoder_unit_norm,
        "decoder_gradient_projection": "row_tangent",
        "decoder_weight_decay": False,
        "input_gradient": "blocked_for_reconstruction_and_auxk",
        "deployed_down_gradient": "language_model_only",
        "coefficient": config.reconstruction_coefficient,
        "layer_reduction": "mean",
        "normalization": "global_centered_target_sum_squares",
        "auxk": {
            "mode": config.auxk_mode,
            "coefficient": config.auxk_coefficient,
            "width_ratio": config.auxk_width_ratio,
            "aux_k": config.aux_k,
            "selector": "one_positive_dead_winner_per_rotating_fixed_group",
            "cohort_count": config.auxk_cohort_count,
            "target": "stop_gradient(input-main_reconstruction)",
            "dead_tokens_threshold": config.dead_tokens_threshold,
            "dead_after_steps": config.dead_after_steps,
            "age_reset": "any_positive_main_activation_in_global_batch",
            "auxiliary_activation_does_not_reset_age": True,
        },
    }


def model_console_rows(
    config: Config, deployment_parameters: int, total_parameters: int
) -> tuple[tuple[str, object], ...]:
    """Describe the recipe-owned architecture inside the standard run card."""

    return (
        (
            "model",
            f"{config.tier} · L{config.layers} D{config.d_model} H{config.heads} "
            f"RoPE RMSNorm fuzzy-TopK-ReLU MLP×{config.mlp_mult} "
            f"k={config.mlp_top_k}",
        ),
        ("deployment parameters", format_count(deployment_parameters)),
        (
            "train-only reconstruction decoder",
            f"{format_count(config.reconstruction_parameter_count)} parameters · "
            f"NMSE×{config.reconstruction_coefficient:g}",
        ),
        ("optimized parameters", format_count(total_parameters)),
        (
            "literal AuxK",
            (
                f"k_aux={config.aux_k} · {config.auxk_cohort_count} cohorts · "
                f"NMSE×{config.auxk_coefficient:g} · dead after "
                f"{config.dead_tokens_threshold:,} tokens"
                if config.auxk_enabled
                else "disabled"
            ),
        ),
    )


def experiment_config_metadata(config: Config) -> dict[str, Any]:
    """Return stable source identity and the fully resolved experiment values."""

    return {
        "schema_version": config.config_schema_version,
        "path": config.config_filename,
        "sha256": config.config_sha256,
        "profile": config.execution_type,
        "context_preset": config.context_preset,
        "resolved": {
            "training": {
                "steps": config.steps,
                "train_tokens": config.steps * config.batch_size * config.seq_len,
                "batch_size": config.batch_size,
                "seq_len": config.seq_len,
                "sampling": config.sampling,
                "dtype": config.dtype_name,
                "tokens_per_parameter": config.tokens_per_parameter,
                "target_tokens_per_parameter": config.target_tokens_per_parameter,
            },
            "model": contract_model_metadata(config),
            "kernels": {
                "attention_backend": config.attention_backend,
                "sparse_mlp_backend": config.sparse_mlp_backend,
                "loss_backend": config.loss_backend,
                "vocab_tile_size": config.vocab_tile_size,
                "document_masking": config.document_masking,
            },
            "reconstruction": reconstruction_metadata(config),
            "optimizer": {
                "learning_rate": config.learning_rate,
                "effective": effective_optimizer_metadata(config),
                "min_lr_ratio": config.min_lr_ratio,
                "warmup_steps": config.warmup_steps,
                "weight_decay": config.weight_decay,
                "adam_epsilon": config.adam_epsilon,
                "beta1": config.beta1,
                "beta2": config.beta2,
                "grad_clip": config.grad_clip,
            },
            "parameterization": {
                "name": config.parameterization,
                "base_width": config.base_width,
                "base_depth": config.base_depth,
                "width_multiplier": config.width_multiplier,
                "depth_multiplier": config.depth_multiplier,
                "ladder_data_multiplier": config.data_multiplier,
                "batch_ratio": config.batch_multiplier,
                "depth_alpha": config.depth_alpha,
                "init_std": config.init_std,
                "attention_scale": config.attention_scale,
                "embeddings": config.embeddings,
            },
            "evaluation": {
                "final_predictions": config.validation_predictions,
                "eval_batches": config.eval_batches,
                "val_every": config.val_every,
                "probe_predictions": (
                    config.val_probe_batches * config.batch_size * config.seq_len
                ),
                "val_probe_batches": config.val_probe_batches,
            },
            "logging": {
                "diagnostics_every": config.diagnostics_every,
                "sparsity_diagnostics_every": config.sparsity_diagnostics_every,
                "log_every": config.log_every,
            },
        },
    }


def resolved_plan_metadata(config: Config) -> dict[str, Any]:
    """Return the deterministic, data-independent execution contract."""

    tokens_per_step = config.batch_size * config.seq_len
    return {
        "schema_version": 3,
        "config_schema_version": config.config_schema_version,
        "config_sha256": config.config_sha256,
        "profile": config.execution_type,
        "context_preset": config.context_preset,
        "document_masking": config.document_masking,
        "tier": config.tier,
        "run_kind": (
            "smoke"
            if config.target_tokens_per_parameter is None
            else ("diagnostic" if config.stop_after_step is not None else "full")
        ),
        "parameterization": config.parameterization,
        "weight_decay_policy": "weights_and_embeddings_only_v2",
        "declared_parameters": config.declared_parameters,
        "batch_size": config.batch_size,
        "sequence_length": config.seq_len,
        "tokens_per_step": tokens_per_step,
        "target_tokens_per_parameter": config.target_tokens_per_parameter,
        "achieved_tokens_per_parameter": config.tokens_per_parameter,
        "schedule_steps": config.steps,
        "stop_after_step": config.stop_after_step,
        "planned_tokens": config.steps * tokens_per_step,
        "expected_tokens": config.final_step * tokens_per_step,
        "validation_predictions": config.validation_predictions,
        "base_learning_rate": config.learning_rate,
        "batch_ratio": config.batch_multiplier,
        "ladder_data_multiplier": config.data_multiplier,
    }


def checkpoint_metadata(
    config: Config, seed: int, attention_runtime: AttentionRuntime
) -> dict[str, Any]:
    """Describe this model well enough to reconstruct it from the weights.

    Travels inside checkpoint.npz. It stays here rather than in rig/ because
    naming a model's layers, activation, and parameterization is exactly what
    a recipe is for.
    """

    return {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "configuration": experiment_config_metadata(config),
        "model": {
            "vocab_size": config.vocab_size,
            "semantic_vocab_size": config.semantic_vocab_size,
            "seq_len": config.seq_len,
            "layers": config.layers,
            "heads": config.heads,
            "d_model": config.d_model,
            "mlp_mult": config.mlp_mult,
            "mlp_top_k": config.mlp_top_k,
            "normalization": config.normalization,
            "position_encoding": config.position_encoding,
            "mlp_activation": config.mlp_activation,
            "dtype": config.dtype_name,
            "attention_backend": config.attention_backend,
            "sparse_mlp_backend": config.sparse_mlp_backend,
            "attention_tuning": attention_runtime_metadata(attention_runtime),
            "loss_backend": config.loss_backend,
            "vocab_tile_size": config.vocab_tile_size,
            "tied_embeddings": config.embeddings == "tied",
            "tier": config.tier,
            "parameterization": config.parameterization,
            "training_only_reconstruction": reconstruction_metadata(config),
        },
    }


def implementation_metadata(
    config: Config, runtime: AttentionRuntime
) -> dict[str, Any]:
    """Return systems/kernel provenance that may vary in either track."""

    return {
        "attention_backend": config.attention_backend,
        "sparse_mlp_backend": config.sparse_mlp_backend,
        "attention_tuning": attention_runtime_metadata(runtime),
        "loss_backend": config.loss_backend,
        "vocab_tile_size": config.vocab_tile_size,
        "weight_decay_policy": "weights_and_embeddings_only_v2",
        "context_preset": config.context_preset,
        "document_masking": config.document_masking,
        "reconstruction": reconstruction_metadata(config),
        "configuration": experiment_config_metadata(config),
    }


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FuzzyReconstructionStats:
    """Per-layer objectives and optional global main-path activity counts."""

    # [layers, 1] for reconstruction or [layers, 3] with AuxK.
    layers: jax.Array
    # [layers, hidden_width] with AuxK, otherwise the fixed empty [0, 0] array.
    active_counts: jax.Array


def gpt_hidden(
    params: Mapping[str, Any],
    tokens: jax.Array,
    config: Config,
    attention_fn: AttentionCallable | None = None,
    sparse_mlp_fn: FuzzyTopKCallable | None = None,
    sparse_mlp_diagnostic_fn: FuzzyTopKDiagnosticCallable | None = None,
    reconstruction_mlp_fn: FuzzyTopKReconstructionCallable | None = None,
    reconstruction_auxk_mlp_fn: FuzzyTopKReconstructionAuxKCallable | None = None,
    dead_mask: jax.Array | None = None,
    auxk_cohort: jax.Array | None = None,
) -> (
    jax.Array
    | tuple[jax.Array, jax.Array]
    | tuple[jax.Array, FuzzyReconstructionStats]
):
    """Return final normalized token representations before the tied head."""

    supplied = sum(
        operation is not None
        for operation in (
            sparse_mlp_fn,
            sparse_mlp_diagnostic_fn,
            reconstruction_mlp_fn,
            reconstruction_auxk_mlp_fn,
        )
    )
    if supplied > 1:
        raise ValueError(
            "ordinary, diagnostic, reconstruction, and AuxK MLP callables are exclusive"
        )
    if reconstruction_auxk_mlp_fn is not None and (
        dead_mask is None or auxk_cohort is None
    ):
        raise ValueError("reconstruction AuxK requires a dead mask and cohort")
    dtype = config.compute_dtype
    batch, length = tokens.shape
    del batch
    x = params["token_embedding"][tokens].astype(dtype)
    head_dim = config.d_model // config.heads
    if config.attention_backend != "dense":
        # Direct construction keeps this function convenient for single-device
        # tests. Multi-device training supplies an explicit shard_map wrapper;
        # Mosaic kernels cannot be partitioned automatically by an outer jit.
        attention = attention_fn or make_causal_attention(
            AttentionConfig(
                backend=config.attention_backend,
                tiles=select_attention_tiles(
                    sequence=length, head_dim=head_dim, training=True
                ),
                softmax_scale=attention_softmax_scale(config.attention_scale, head_dim),
            )
        )
        causal = None
    else:
        attention = None
        causal = jnp.tril(jnp.ones((length, length), dtype=jnp.bool_))[None, None, :, :]

    segments = (
        document_segments(tokens, config.document_boundary_token)
        if config.document_masking
        else None
    )
    sparse_mlp_config = FuzzyTopKConfig(
        top_k=config.mlp_top_k,
        backend=config.sparse_mlp_backend,
    )
    feature_statistics: list[jax.Array] = []
    reconstruction_statistics: list[jax.Array] = []
    active_counts: list[jax.Array] = []

    for layer_index, block in enumerate(params["blocks"]):
        residual = x
        x_norm = rms_norm(x, block["ln1_scale"], dtype)
        qkv = linear(x_norm, block["qkv_w"], block["qkv_b"], dtype)
        query, key, value = jnp.split(qkv, 3, axis=-1)
        query = query.reshape(tokens.shape[0], length, config.heads, head_dim)
        key = key.reshape(tokens.shape[0], length, config.heads, head_dim)
        value = value.reshape(tokens.shape[0], length, config.heads, head_dim)
        query = apply_rotary(query)
        key = apply_rotary(key)
        if attention is not None:
            attended = attention(
                jnp.transpose(query, (0, 2, 1, 3)),
                jnp.transpose(key, (0, 2, 1, 3)),
                jnp.transpose(value, (0, 2, 1, 3)),
                *((segments,) if segments is not None else ()),
            )
            attended = jnp.transpose(attended, (0, 2, 1, 3))
        else:
            scores = jnp.einsum("bthd,bshd->bhts", query, key)
            scores = scores.astype(jnp.float32) * attention_softmax_scale(
                config.attention_scale, head_dim
            )
            visible = causal
            if segments is not None:
                same = segments[:, None, :, None] == segments[:, None, None, :]
                visible = jnp.logical_and(visible, same)
            scores = jnp.where(visible, scores, jnp.finfo(jnp.float32).min)
            probabilities = jax.nn.softmax(scores, axis=-1).astype(dtype)
            attended = jnp.einsum("bhts,bshd->bthd", probabilities, value)
        attended = attended.reshape(tokens.shape[0], length, config.d_model)
        x = residual + (config.depth_multiplier ** (-config.depth_alpha)) * linear(
            attended, block["attn_w"], block["attn_b"], dtype
        )

        residual = x
        x_norm = rms_norm(x, block["ln2_scale"], dtype)
        mlp_arguments = (
            x_norm,
            block["mlp_up_w"],
            block["mlp_up_b"],
            block["mlp_down_w"],
            block["mlp_down_b"],
        )
        if sparse_mlp_diagnostic_fn is not None:
            mlp_output, layer_statistics = sparse_mlp_diagnostic_fn(*mlp_arguments)
            feature_statistics.append(layer_statistics)
        elif reconstruction_mlp_fn is not None:
            mlp_output, layer_statistics = reconstruction_mlp_fn(
                *mlp_arguments,
                block["mlp_reconstruction_w"],
            )
            reconstruction_statistics.append(layer_statistics)
        elif reconstruction_auxk_mlp_fn is not None:
            assert dead_mask is not None and auxk_cohort is not None
            mlp_output, layer_statistics, layer_counts = (
                reconstruction_auxk_mlp_fn(
                    *mlp_arguments,
                    block["mlp_reconstruction_w"],
                    dead_mask[layer_index],
                    auxk_cohort,
                )
            )
            reconstruction_statistics.append(layer_statistics)
            active_counts.append(layer_counts)
        else:
            mlp_output = (
                sparse_mlp_fn(*mlp_arguments)
                if sparse_mlp_fn is not None
                else fuzzy_topk_mlp(*mlp_arguments, config=sparse_mlp_config)
            )
        x = residual + (config.depth_multiplier ** (-config.depth_alpha)) * (mlp_output)

    hidden = rms_norm(x, params["final_ln_scale"], dtype)
    if sparse_mlp_diagnostic_fn is not None:
        return hidden, jnp.stack(feature_statistics, axis=1)
    if reconstruction_mlp_fn is not None:
        return hidden, FuzzyReconstructionStats(
            layers=jnp.stack(reconstruction_statistics),
            active_counts=jnp.zeros((0, 0), jnp.float32),
        )
    if reconstruction_auxk_mlp_fn is not None:
        return hidden, FuzzyReconstructionStats(
            layers=jnp.stack(reconstruction_statistics),
            active_counts=jnp.stack(active_counts),
        )
    return hidden


def gpt_logits(
    params: Mapping[str, Any],
    tokens: jax.Array,
    config: Config,
    attention_fn: AttentionCallable | None = None,
    sparse_mlp_fn: FuzzyTopKCallable | None = None,
) -> jax.Array:
    x = gpt_hidden(params, tokens, config, attention_fn, sparse_mlp_fn)
    output_embedding = params.get("output_embedding", params["token_embedding"])
    return jnp.einsum(
        "btd,vd->btv",
        x,
        output_embedding.astype(config.compute_dtype),
    ).astype(jnp.float32)


def cross_entropy(
    params: Mapping[str, Any],
    x: jax.Array,
    y: jax.Array,
    config: Config,
    attention_fn: AttentionCallable | None = None,
    sparse_mlp_fn: FuzzyTopKCallable | None = None,
) -> jax.Array:
    if config.loss_backend == "tiled":
        hidden = gpt_hidden(params, x, config, attention_fn, sparse_mlp_fn)
        return tiled_tied_cross_entropy(
            hidden,
            params.get("output_embedding", params["token_embedding"]),
            y,
            semantic_vocab_size=config.semantic_vocab_size,
            vocab_tile_size=config.vocab_tile_size,
            compute_dtype=config.compute_dtype,
        )
    logits = gpt_logits(params, x, config, attention_fn, sparse_mlp_fn)[
        ..., : config.semantic_vocab_size
    ]
    log_probabilities = jax.nn.log_softmax(logits, axis=-1)
    selected = jnp.take_along_axis(log_probabilities, y[..., None], axis=-1)
    return -jnp.mean(selected, dtype=jnp.float32)


def cross_entropy_and_reconstruction(
    params: Mapping[str, Any],
    x: jax.Array,
    y: jax.Array,
    config: Config,
    attention_fn: AttentionCallable | None = None,
    reconstruction_mlp_fn: FuzzyTopKReconstructionCallable | None = None,
    reconstruction_auxk_mlp_fn: FuzzyTopKReconstructionAuxKCallable | None = None,
    dead_mask: jax.Array | None = None,
    auxk_cohort: jax.Array | None = None,
) -> tuple[jax.Array, tuple[jax.Array, FuzzyReconstructionStats]]:
    """Return the training objective, unpolluted CE, and layer statistics."""

    if config.auxk_enabled:
        if reconstruction_auxk_mlp_fn is None:
            raise ValueError("enabled AuxK has no reconstruction/AuxK MLP callable")
        hidden, returned = gpt_hidden(
            params,
            x,
            config,
            attention_fn,
            reconstruction_auxk_mlp_fn=reconstruction_auxk_mlp_fn,
            dead_mask=dead_mask,
            auxk_cohort=auxk_cohort,
        )
    else:
        if reconstruction_mlp_fn is None:
            raise ValueError("reconstruction training has no reconstruction MLP callable")
        hidden, returned = gpt_hidden(
            params,
            x,
            config,
            attention_fn,
            reconstruction_mlp_fn=reconstruction_mlp_fn,
        )
    if not isinstance(returned, FuzzyReconstructionStats):
        raise AssertionError("reconstruction MLP did not return its statistics")

    if config.loss_backend == "tiled":
        cross_entropy_loss = tiled_tied_cross_entropy(
            hidden,
            params.get("output_embedding", params["token_embedding"]),
            y,
            semantic_vocab_size=config.semantic_vocab_size,
            vocab_tile_size=config.vocab_tile_size,
            compute_dtype=config.compute_dtype,
        )
    else:
        output_embedding = params.get("output_embedding", params["token_embedding"])
        logits = jnp.einsum(
            "btd,vd->btv", hidden, output_embedding.astype(config.compute_dtype)
        ).astype(jnp.float32)[..., : config.semantic_vocab_size]
        log_probabilities = jax.nn.log_softmax(logits, axis=-1)
        selected = jnp.take_along_axis(log_probabilities, y[..., None], axis=-1)
        cross_entropy_loss = -jnp.mean(selected, dtype=jnp.float32)

    objective = cross_entropy_loss + config.reconstruction_coefficient * jnp.mean(
        returned.layers[:, 0]
    )
    if config.auxk_enabled:
        objective = objective + config.auxk_coefficient * jnp.mean(
            returned.layers[:, 1]
        )
    return objective, (cross_entropy_loss, returned)


def fuzzy_sparsity_diagnostics(
    params: Mapping[str, Any],
    x: jax.Array,
    config: Config,
    attention_fn: AttentionCallable | None,
    sparse_mlp_diagnostic_fn: FuzzyTopKDiagnosticCallable,
) -> jax.Array:
    """Observe fuzzy feature vectors without participating in the update graph."""

    _hidden, statistics = gpt_hidden(
        params,
        x,
        config,
        attention_fn,
        sparse_mlp_diagnostic_fn=sparse_mlp_diagnostic_fn,
    )
    return statistics


def learning_rate(step: jax.Array, config: Config) -> jax.Array:
    step_float = step.astype(jnp.float32)
    if config.warmup_steps:
        warmup = jnp.minimum(1.0, step_float / float(config.warmup_steps))
    else:
        warmup = jnp.asarray(1.0, dtype=jnp.float32)
    decay_span = max(1, config.steps - config.warmup_steps)
    progress = jnp.clip(
        (step_float - float(config.warmup_steps)) / float(decay_span), 0.0, 1.0
    )
    cosine = 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
    multiplier = config.min_lr_ratio + (1.0 - config.min_lr_ratio) * cosine
    horizon_scale = math.sqrt(config.batch_multiplier / config.data_multiplier)
    return (
        jnp.asarray(config.learning_rate * horizon_scale, jnp.float32)
        * warmup
        * multiplier
    )


def reconstruction_stat_names(config: Config) -> tuple[str, ...]:
    return (
        RECONSTRUCTION_AUXK_STAT_NAMES
        if config.auxk_enabled
        else RECONSTRUCTION_STAT_NAMES
    )


def model_and_layer_row(layers: jax.Array) -> jax.Array:
    """Flatten model-wide means followed by the exact per-layer values."""

    layers = layers.astype(jnp.float32)
    return jnp.concatenate((jnp.mean(layers, axis=0), layers.reshape(-1)))


def fuzzy_training_log_columns(config: Config) -> tuple[logpack.Column, ...]:
    """Describe CE, reconstruction, and dead-feature state for every step."""

    columns = list(training_log_columns())
    names = reconstruction_stat_names(config)
    columns.extend(logpack.column(name) for name in names)
    for layer in range(config.layers):
        columns.extend(logpack.column(name, "block", layer) for name in names)
    if config.auxk_enabled:
        columns.extend(logpack.column(name) for name in AUXK_AGE_STAT_NAMES)
        for layer in range(config.layers):
            columns.extend(
                logpack.column(name, "block", layer) for name in AUXK_AGE_STAT_NAMES
            )
    return tuple(columns)


def init_optimizer(
    params: Any,
    steps: int,
    *,
    config: Config | None = None,
) -> dict[str, Any]:
    zeros = jax.tree_util.tree_map(lambda value: np.zeros_like(value), params)
    # Keeping the small scalar history on-device avoids a host synchronization
    # on every step. It is copied once, after the synchronized timing boundary.
    history_width = 3
    if config is not None:
        history_width += len(reconstruction_stat_names(config)) * (config.layers + 1)
        if config.auxk_enabled:
            history_width += len(AUXK_AGE_STAT_NAMES) * (config.layers + 1)
    history = np.zeros((steps, history_width), dtype=np.float32)
    result = {
        "step": np.asarray(0, dtype=np.int32),
        "m": zeros,
        "v": zeros,
        "history": history,
    }
    if config is not None and config.auxk_enabled:
        result["dead_steps"] = np.zeros(
            (config.layers, config.mlp_mult * config.d_model), dtype=np.int32
        )
    return result


def weight_decay_mask(params: Any) -> Any:
    """Select AdamW decay from parameter roles, never from array rank.

    Expert-stacked biases are rank two, so shape is not a reliable indication
    that a leaf is a weight.  Parameter names are part of this recipe's
    optimizer contract; failing closed also makes a newly introduced role ask
    for an explicit decay decision.
    """

    def decay_for_path(path: tuple[Any, ...], _value: Any) -> bool:
        name = getattr(path[-1], "key", None) if path else None
        if not isinstance(name, str):
            raise ValueError(f"cannot classify unnamed parameter leaf at {path!r}")
        if name == "mlp_reconstruction_w":
            return False
        if name in {"token_embedding", "output_embedding"} or name.endswith("_w"):
            return True
        if name.endswith(("_b", "_bias", "_scale")):
            return False
        raise ValueError(f"weight-decay policy has no rule for parameter {name!r}")

    return jax.tree_util.tree_map_with_path(decay_for_path, params)


def optimizer_hyperparameter_trees(
    params: Mapping[str, Any], config: Config
) -> tuple[Any, Any, Any]:
    """Return Complete(d)P-inspired LR, epsilon, and decay multipliers per tensor.

    Input/output layers and the residual backbone are intentionally distinct.
    Complete(d)P corrects CompleteP's input-embedding epsilon to ``1 / m_N``;
    the unembedding epsilon remains unscaled after its forward multiplier is
    absorbed into initialization and learning rate.
    """

    if config.parameterization != "completep_fixed_tpp_v1":
        ones = jax.tree_util.tree_map(lambda _: 1.0, params)
        return ones, ones, ones

    width = config.width_multiplier
    depth = config.depth_multiplier
    alpha = config.depth_alpha
    hidden_lr = width**-1 * depth ** (alpha - 1.0)
    hidden_vector_lr = depth ** (alpha - 1.0)
    hidden_epsilon = width**-1 * depth**-alpha

    lr_blocks: list[dict[str, float]] = []
    epsilon_blocks: list[dict[str, float]] = []
    decay_blocks: list[dict[str, float]] = []
    for block in params["blocks"]:
        lr_blocks.append(
            {
                name: (hidden_lr if name.endswith("_w") else hidden_vector_lr)
                for name in block
            }
        )
        epsilon_blocks.append({name: hidden_epsilon for name in block})
        decay_blocks.append(
            {name: (width if name.endswith("_w") else 1.0) for name in block}
        )

    lr_tree: dict[str, Any] = {
        "token_embedding": 1.0,
        "blocks": lr_blocks,
        "final_ln_scale": 1.0,
    }
    epsilon_tree: dict[str, Any] = {
        "token_embedding": width**-1,
        "blocks": epsilon_blocks,
        "final_ln_scale": 1.0,
    }
    decay_tree: dict[str, Any] = {
        "token_embedding": 1.0,
        "blocks": decay_blocks,
        "final_ln_scale": 1.0,
    }
    if "output_embedding" in params:
        # Complete(d)P absorbs the old 1/m_N output multiplier into output
        # initialization and learning rate. Its width-scaled decay keeps the
        # actual AdamW shrink invariant.
        lr_tree["output_embedding"] = width**-1
        epsilon_tree["output_embedding"] = 1.0
        decay_tree["output_embedding"] = width
    return lr_tree, epsilon_tree, decay_tree


def effective_adam_betas(config: Config) -> tuple[float, float]:
    ratio = config.batch_multiplier / config.data_multiplier
    beta1 = 1.0 - (1.0 - config.beta1) * ratio
    beta2 = 1.0 - (1.0 - config.beta2) * ratio
    if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
        raise ValueError(
            "fixed-TPP batch/data scaling produced invalid Adam momenta; "
            "use a closer transfer base"
        )
    return beta1, beta2


def effective_optimizer_metadata(config: Config) -> dict[str, float]:
    """Return the fixed-TPP hybrid's global horizon/batch scalars."""

    beta1, beta2 = effective_adam_betas(config)
    ratio = config.batch_multiplier / config.data_multiplier
    return {
        "global_peak_learning_rate": config.learning_rate * math.sqrt(ratio),
        "adam_epsilon_horizon_multiplier": math.sqrt(1.0 / ratio),
        "weight_decay_horizon_multiplier": math.sqrt(ratio),
        "beta1": beta1,
        "beta2": beta2,
    }


def project_reconstruction_decoder_gradients(
    params: Mapping[str, Any], gradients: Mapping[str, Any]
) -> dict[str, Any]:
    """Remove each decoder row's radial gradient before Adam moments update."""

    projected = dict(gradients)
    projected_blocks: list[dict[str, Any]] = []
    for parameter_block, gradient_block in zip(
        params["blocks"], gradients["blocks"], strict=True
    ):
        block = dict(gradient_block)
        direction = parameter_block["mlp_reconstruction_w"].astype(jnp.float32)
        gradient = block["mlp_reconstruction_w"].astype(jnp.float32)
        norm_squared = jnp.sum(jnp.square(direction), axis=-1, keepdims=True)
        parallel = (
            jnp.sum(gradient * direction, axis=-1, keepdims=True)
            / jnp.maximum(norm_squared, jnp.asarray(1.0e-12, jnp.float32))
        ) * direction
        block["mlp_reconstruction_w"] = gradient - parallel
        projected_blocks.append(block)
    projected["blocks"] = projected_blocks
    return projected


def normalize_reconstruction_decoder_rows(params: Mapping[str, Any]) -> dict[str, Any]:
    """Project every train-only decoder row back to the unit sphere."""

    normalized = dict(params)
    normalized_blocks: list[dict[str, Any]] = []
    for parameter_block in params["blocks"]:
        block = dict(parameter_block)
        decoder = block["mlp_reconstruction_w"].astype(jnp.float32)
        norm = jnp.sqrt(jnp.sum(jnp.square(decoder), axis=-1, keepdims=True))
        block["mlp_reconstruction_w"] = decoder / jnp.maximum(
            norm, jnp.asarray(1.0e-12, jnp.float32)
        )
        normalized_blocks.append(block)
    normalized["blocks"] = normalized_blocks
    return normalized


def deployment_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """Drop every train-only decoder from the model returned after training."""

    deployed = dict(params)
    deployed["blocks"] = [
        {
            name: value
            for name, value in block.items()
            if name != "mlp_reconstruction_w"
        }
        for block in params["blocks"]
    ]
    return deployed


def _apply_training_update(
    params: Any,
    optimizer: Mapping[str, Any],
    x: jax.Array,
    y: jax.Array,
    config: Config,
    decay_mask: Any | None = None,
    attention_fn: AttentionCallable | None = None,
    reconstruction_mlp_fn: FuzzyTopKReconstructionCallable | None = None,
    reconstruction_auxk_mlp_fn: FuzzyTopKReconstructionAuxKCallable | None = None,
) -> tuple[Any, dict[str, Any], dict[str, jax.Array], Any]:
    """Apply one ordinary update and also return the raw, pre-clip gradient.

    Both the ordinary and model-diagnostic executables use this exact function.
    Feature observation happens in a separate forward-only executable, so it
    cannot substitute a different optimizer graph or formula.
    """

    if decay_mask is None:
        decay_mask = weight_decay_mask(params)
    lr_multipliers, epsilon_multipliers, decay_multipliers = (
        optimizer_hyperparameter_trees(params, config)
    )
    beta1, beta2 = effective_adam_betas(config)
    if config.auxk_enabled:
        if "dead_steps" not in optimizer:
            raise ValueError("enabled AuxK training has no dead-feature age state")
        dead_mask = optimizer["dead_steps"] >= jnp.asarray(
            config.dead_after_steps, jnp.int32
        )
        auxk_cohort = jnp.mod(
            optimizer["step"], jnp.asarray(config.auxk_cohort_count, jnp.int32)
        )
    else:
        dead_mask = None
        auxk_cohort = None
    objective_and_aux, gradients = jax.value_and_grad(
        lambda candidate: cross_entropy_and_reconstruction(
            candidate,
            x,
            y,
            config,
            attention_fn,
            reconstruction_mlp_fn,
            reconstruction_auxk_mlp_fn,
            dead_mask,
            auxk_cohort,
        ),
        has_aux=True,
    )(params)
    objective, (loss, reconstruction_stats) = objective_and_aux
    gradients = jax.tree_util.tree_map(lambda grad: grad.astype(jnp.float32), gradients)
    gradients = project_reconstruction_decoder_gradients(params, gradients)
    raw_gradients = gradients
    squared_norms = [
        jnp.sum(jnp.square(grad)) for grad in jax.tree_util.tree_leaves(gradients)
    ]
    grad_norm = jnp.sqrt(sum(squared_norms))
    clip_scale = (
        jnp.minimum(1.0, config.grad_clip / (grad_norm + 1.0e-6))
        if config.grad_clip > 0.0
        else jnp.asarray(1.0, dtype=jnp.float32)
    )
    gradients = jax.tree_util.tree_map(lambda grad: grad * clip_scale, gradients)

    step = optimizer["step"] + jnp.asarray(1, dtype=jnp.int32)
    lr = learning_rate(step, config)
    m = jax.tree_util.tree_map(
        lambda old, grad: beta1 * old + (1.0 - beta1) * grad,
        optimizer["m"],
        gradients,
    )
    v = jax.tree_util.tree_map(
        lambda old, grad: beta2 * old + (1.0 - beta2) * jnp.square(grad),
        optimizer["v"],
        gradients,
    )
    bias_correction1 = 1.0 - beta1 ** step.astype(jnp.float32)
    bias_correction2 = 1.0 - beta2 ** step.astype(jnp.float32)
    epsilon_horizon_scale = math.sqrt(config.data_multiplier / config.batch_multiplier)
    decay_horizon_scale = math.sqrt(config.batch_multiplier / config.data_multiplier)

    def update(
        parameter: jax.Array,
        first: jax.Array,
        second: jax.Array,
        should_decay: bool,
        lr_multiplier: float,
        epsilon_multiplier: float,
        decay_multiplier: float,
    ) -> jax.Array:
        epsilon = config.adam_epsilon * epsilon_horizon_scale * epsilon_multiplier
        adam = (first / bias_correction1) / (
            jnp.sqrt(second / bias_correction2) + epsilon
        )
        decay = (
            config.weight_decay * decay_horizon_scale * decay_multiplier * parameter
            if should_decay
            else 0.0
        )
        return parameter - lr * lr_multiplier * (adam + decay)

    params = jax.tree_util.tree_map(
        update,
        params,
        m,
        v,
        decay_mask,
        lr_multipliers,
        epsilon_multipliers,
        decay_multipliers,
    )
    params = normalize_reconstruction_decoder_rows(params)

    reconstruction_values = model_and_layer_row(reconstruction_stats.layers)
    if config.auxk_enabled:
        active = reconstruction_stats.active_counts > 0.0
        dead_steps = jnp.where(
            active,
            jnp.asarray(0, jnp.int32),
            jnp.minimum(
                optimizer["dead_steps"] + jnp.asarray(1, jnp.int32),
                jnp.asarray(config.dead_after_steps, jnp.int32),
            ),
        )
        age_layers = jnp.stack(
            (
                jnp.mean(active.astype(jnp.float32), axis=1),
                jnp.mean(
                    (dead_steps >= config.dead_after_steps).astype(jnp.float32),
                    axis=1,
                ),
            ),
            axis=1,
        )
        age_values = model_and_layer_row(age_layers)
    else:
        dead_steps = None
        age_values = jnp.zeros((0,), jnp.float32)
    history_row = jnp.concatenate(
        (
            jnp.stack((loss, lr, grad_norm)).astype(jnp.float32),
            reconstruction_values,
            age_values,
        )
    )
    history = optimizer["history"].at[step - 1].set(history_row)
    next_optimizer = {"step": step, "m": m, "v": v, "history": history}
    if dead_steps is not None:
        next_optimizer["dead_steps"] = dead_steps
    return (
        params,
        next_optimizer,
        {
            "loss": loss,
            "objective": objective,
            "grad_norm": grad_norm,
            "learning_rate": lr,
            "reconstruction_row": reconstruction_values,
            "auxk_age_row": age_values,
        },
        raw_gradients,
    )


def train_step(
    params: Any,
    optimizer: Mapping[str, Any],
    x: jax.Array,
    y: jax.Array,
    config: Config,
    decay_mask: Any | None = None,
    attention_fn: AttentionCallable | None = None,
    reconstruction_mlp_fn: FuzzyTopKReconstructionCallable | None = None,
    reconstruction_auxk_mlp_fn: FuzzyTopKReconstructionAuxKCallable | None = None,
) -> tuple[Any, dict[str, Any], dict[str, jax.Array]]:
    params, optimizer, metrics, _ = _apply_training_update(
        params,
        optimizer,
        x,
        y,
        config,
        decay_mask,
        attention_fn,
        reconstruction_mlp_fn,
        reconstruction_auxk_mlp_fn,
    )
    return params, optimizer, metrics


def diagnostic_train_step(
    params: Any,
    optimizer: Mapping[str, Any],
    x: jax.Array,
    y: jax.Array,
    config: Config,
    decay_mask: Any | None = None,
    attention_fn: AttentionCallable | None = None,
    reconstruction_mlp_fn: FuzzyTopKReconstructionCallable | None = None,
    reconstruction_auxk_mlp_fn: FuzzyTopKReconstructionAuxKCallable | None = None,
) -> tuple[Any, dict[str, Any], dict[str, jax.Array], jax.Array]:
    """Run the same update as :func:`train_step` and emit sparse statistics."""

    params_before = params
    params, optimizer, metrics, raw_gradients = _apply_training_update(
        params,
        optimizer,
        x,
        y,
        config,
        decay_mask,
        attention_fn,
        reconstruction_mlp_fn,
        reconstruction_auxk_mlp_fn,
    )
    values = diagnostic_values(params_before, raw_gradients, params)
    return params, optimizer, metrics, values


def eval_step(
    params: Any,
    x: jax.Array,
    y: jax.Array,
    mask: jax.Array,
    config: Config,
    attention_fn: AttentionCallable | None = None,
    sparse_mlp_fn: FuzzyTopKCallable | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Return a loss sum and exact target count for fixed-shape masked eval."""

    if config.loss_backend == "tiled":
        hidden = gpt_hidden(params, x, config, attention_fn, sparse_mlp_fn)
        losses = tiled_tied_cross_entropy_losses(
            hidden,
            params.get("output_embedding", params["token_embedding"]),
            y,
            semantic_vocab_size=config.semantic_vocab_size,
            vocab_tile_size=config.vocab_tile_size,
            compute_dtype=config.compute_dtype,
        )
    else:
        logits = gpt_logits(params, x, config, attention_fn, sparse_mlp_fn)[
            ..., : config.semantic_vocab_size
        ]
        log_probabilities = jax.nn.log_softmax(logits, axis=-1)
        selected = jnp.take_along_axis(log_probabilities, y[..., None], axis=-1)[..., 0]
        losses = -selected
    mask = mask.astype(jnp.float32)
    return (
        jnp.sum(losses * mask, dtype=jnp.float32),
        jnp.sum(mask, dtype=jnp.float32),
    )


def fuzzy_topk_mlp_flop_rule(site: Site) -> int:
    """Bill the regular contractions executed by the choicewise backend.

    The forward scores ``H`` features and decodes every one of the ``H/K``
    choices through a masked ``[M,K] @ [K,D]`` contraction, for ``4 M D H``.
    The custom VJP uses four more choicewise contractions (dValues, dX, dW_up,
    and dW_down), for ``8 M D H``. This is deliberately physical accounting:
    the backend spends dense MXU work to avoid irregular gather/scatter loops.
    """

    if len(site.in_shapes) not in (5, 8):
        raise FlopError(f"unexpected fuzzy_topk_mlp boundary shapes: {site.in_shapes}")
    x_shape = site.in_shapes[0]
    up_shape = site.in_shapes[1]
    if len(x_shape) < 2 or len(up_shape) != 2:
        raise FlopError(f"unexpected fuzzy_topk_mlp operands: {site.in_shapes}")
    tokens = math.prod(x_shape[:-1])
    model_width, hidden_width = up_shape
    if model_width != x_shape[-1]:
        raise FlopError("fuzzy_topk_mlp input and encoder widths disagree")
    if len(site.in_shapes) == 5:
        return 4 * tokens * model_width * hidden_width
    return 8 * tokens * model_width * hidden_width


def fuzzy_topk_reconstruction_flop_rule(site: Site) -> int:
    """Bill parent training plus the shared-activation reconstruction decoder."""

    if len(site.in_shapes) not in (6, 11):
        raise FlopError(
            f"unexpected fuzzy reconstruction boundary shapes: {site.in_shapes}"
        )
    x_shape = site.in_shapes[0]
    up_shape = site.in_shapes[1]
    if len(x_shape) < 2 or len(up_shape) != 2:
        raise FlopError(f"unexpected fuzzy reconstruction operands: {site.in_shapes}")
    tokens = math.prod(x_shape[:-1])
    model_width, hidden_width = up_shape
    if model_width != x_shape[-1]:
        raise FlopError("fuzzy reconstruction input and encoder widths disagree")
    # Forward: score + deployed decode + reconstruction decode = 6 MDH.
    # Backward: two value cotangents, LM-only dX, shared dW_up, and both
    # decoder gradients = 12 MDH.
    coefficient = 6 if len(site.in_shapes) == 6 else 12
    return coefficient * tokens * model_width * hidden_width


def fuzzy_topk_reconstruction_auxk_flop_rule(
    site: Site, *, top_k: int, aux_k: int
) -> int:
    """Bill literal fuzzy AuxK in addition to main reconstruction training."""

    if len(site.in_shapes) not in (8, 15):
        raise FlopError(
            f"unexpected fuzzy reconstruction AuxK shapes: {site.in_shapes}"
        )
    x_shape = site.in_shapes[0]
    up_shape = site.in_shapes[1]
    if len(x_shape) < 2 or len(up_shape) != 2:
        raise FlopError(f"unexpected fuzzy reconstruction operands: {site.in_shapes}")
    tokens = math.prod(x_shape[:-1])
    model_width, hidden_width = up_shape
    if model_width != x_shape[-1]:
        raise FlopError("fuzzy reconstruction input and encoder widths disagree")
    ratio = aux_k / float(top_k)
    # Aux decode adds 2r MDH forward. Its value cotangent, selected dW_up,
    # and selected decoder gradient add 6r MDH backward, where r=k_aux/K.
    coefficient = (6.0 + 2.0 * ratio) if len(site.in_shapes) == 8 else (
        12.0 + 6.0 * ratio
    )
    return int(round(coefficient * tokens * model_width * hidden_width))


def traced_flops(config: Config, params: Mapping[str, Any]) -> FlopBreakdown:
    """Count one training step's algorithmic FLOPs by tracing the model.

    Nothing executes and nothing is allocated: ``jax.make_jaxpr`` builds the
    graph from shapes alone. The count therefore follows the architecture
    automatically -- change the depth, width, head count, or the shape of a
    block and this number moves with it, with no formula to maintain.

    A single sequence is traced and the result divided by ``seq_len``. Every
    term is linear in the batch dimension (attention included, which is
    quadratic in sequence but linear in batch), so one sequence determines
    the per-token cost; ``test_flops_are_linear_in_batch`` pins that down.

    ADDING A COMPONENT
    ------------------
    Ordinary blocks built from matmuls need nothing: they are counted from
    their traced shapes. Two cases do need attention, and both announce
    themselves in ``breakdown.warnings`` rather than failing quietly:

    * A new opaque kernel (anything built with ``pallas_call``) is invisible
      to the tracer. Register its cost with
      ``rules.with_kernel("<kernel name>", rule)`` in ``rig.flops``.
    * A component whose real cost differs from its traced cost -- sparsity
      being the usual reason -- must say so. A mixture-of-experts written as
      "compute every expert, then mask to top-k" contains the full dense work
      in its graph and will be billed for all of it, because the tracer sees
      real multiplications and cannot know a mask discards them. Wrap the
      component in a named ``jax.jit`` and register
      ``rules.with_scope("<name>", rule)``; the walker then bills the rule
      and does not descend. This is the one case no warning can catch, since
      nothing about the graph looks unusual.

    See ``docs/FLOPS.md`` for the full checklist.
    """

    tokens = jnp.zeros((1, config.seq_len), jnp.int32)
    targets = jnp.zeros((1, config.seq_len), jnp.int32)
    reconstruction_config = FuzzyTopKReconstructionConfig(
        top_k=config.mlp_top_k,
        aux_k=config.aux_k if config.auxk_enabled else None,
    )
    reconstruction_mlp_fn = (
        None
        if config.auxk_enabled
        else lambda *operands: fuzzy_topk_mlp_with_reconstruction(
            *operands, config=reconstruction_config
        )
    )
    reconstruction_auxk_mlp_fn = (
        (
            lambda *operands: fuzzy_topk_mlp_with_reconstruction_auxk(
                *operands, config=reconstruction_config
            )
        )
        if config.auxk_enabled
        else None
    )
    dead_mask = (
        jnp.zeros(
            (config.layers, config.mlp_mult * config.d_model), dtype=jnp.bool_
        )
        if config.auxk_enabled
        else None
    )
    cohort = jnp.asarray(0, jnp.int32) if config.auxk_enabled else None

    def loss(trainable: Mapping[str, Any]) -> jax.Array:
        objective, _ = cross_entropy_and_reconstruction(
            trainable,
            tokens,
            targets,
            config,
            reconstruction_mlp_fn=reconstruction_mlp_fn,
            reconstruction_auxk_mlp_fn=reconstruction_auxk_mlp_fn,
            dead_mask=dead_mask,
            auxk_cohort=cohort,
        )
        return objective

    aux_k = config.aux_k
    rules = (
        default_rules()
        .with_scope("_choicewise_fuzzy_topk_mlp", fuzzy_topk_mlp_flop_rule)
        .with_scope(
            "_choicewise_fuzzy_topk_reconstruction_mlp",
            fuzzy_topk_reconstruction_flop_rule,
        )
        .with_scope(
            "_choicewise_fuzzy_topk_reconstruction_auxk_mlp",
            lambda site: fuzzy_topk_reconstruction_auxk_flop_rule(
                site, top_k=config.mlp_top_k, aux_k=aux_k
            ),
        )
    )
    return count_training_flops(loss, params, rules=rules)


# How many diagnostic captures may sit on the accelerator before being pulled
# to the host. Bounded on purpose: without a cap this list grows with the run,
# holding one small device allocation per capture until the very end, and a
# preempted job loses every one of them.


def run(args: argparse.Namespace) -> dict[str, Any] | None:
    profile = selected_profile(args)
    experiment_config, config_sha256 = load_experiment_config(profile)
    validate_args(args, experiment_config)
    if (
        experiment_config.execution_type != "smoke"
        and not args.train_data
        and not args.val_data
    ):
        raise ValueError(
            f"{profile_config_filename(profile)} requires explicit --train-data "
            "and --val-data"
        )
    process_index, process_count = initialize_distributed_runtime()
    is_controller = is_controller_process(process_index)
    console = Console(args.color, active=is_controller)
    console.banner()

    devices = jax.devices()
    if not devices:
        raise RuntimeError("JAX reported no devices")
    validate_official_topology(experiment_config.execution_type, devices)
    platform = devices[0].platform

    dataset = load_dataset(
        train_data=args.train_data,
        val_data=args.val_data,
        data_dtype=args.data_dtype,
        data_format=args.data_format,
        seed=args.seed,
    )
    config = resolve_config(
        args,
        platform,
        experiment_config=experiment_config,
        config_sha256=config_sha256,
    )
    capture_window = xprof_step_window(args, config.final_step)
    downstream_domains = load_downstream_domains(
        manifest=args.downstream_manifest,
        root=args.downstream_root,
        vocab_size=config.semantic_vocab_size,
    )
    diagnostic_mode = args.diagnostic_mode
    needs_evaluation = should_compile_evaluation(args, config, downstream_domains)
    if config.batch_size % len(devices):
        raise ValueError(
            f"global batch size {config.batch_size} must be divisible by "
            f"visible device count {len(devices)}"
        )
    local_batch = local_batch_size(config.batch_size, process_count)
    shuffled_train_stream = (
        ShuffledEpochBatchStream(
            dataset.train,
            global_batch_size=config.batch_size,
            seq_len=config.seq_len,
            vocab_size=config.semantic_vocab_size,
            seed=args.seed + 1,
            process_index=process_index,
            process_count=process_count,
        )
        if config.sampling == "shuffled_epochs"
        else None
    )
    if config.attention_backend != "dense":
        console.phase(
            "Attention tile preflight",
            "resolving the shipped lookup or shape heuristic",
        )
    attention_runtime = resolve_attention_runtime(
        backend=config.attention_backend,
        dtype=config.compute_dtype,
        global_batch_size=config.batch_size,
        heads=config.heads,
        sequence=config.seq_len,
        head_dim=config.d_model // config.heads,
        devices=devices,
    )
    if (
        max(map(len, dataset.train.shards)) < config.seq_len + 1
        or max(map(len, dataset.validation.shards)) < config.seq_len + 1
    ):
        raise ValueError(
            "both data splits need a shard with at least seq_len + 1 tokens; "
            f"got train={len(dataset.train):,}, validation={len(dataset.validation):,}, "
            f"seq_len={config.seq_len}"
        )

    host_params = init_params(config, args.seed)
    host_optimizer = init_optimizer(host_params, config.steps, config=config)
    decay_mask = weight_decay_mask(host_params)
    diagnostic_metadata = diagnostic_scope_metadata(host_params)
    params_total = parameter_count(host_params)
    deployment_parameters = params_total - config.reconstruction_parameter_count
    if (
        config.declared_parameters is not None
        and deployment_parameters != config.declared_parameters
    ):
        raise ValueError(
            f"tier {config.tier} declares {config.declared_parameters:,} parameters, "
            f"but initialized {deployment_parameters:,} deployment parameters plus "
            f"{config.reconstruction_parameter_count:,} train-only decoder parameters"
        )
    flop_breakdown = traced_flops(config, host_params)
    flops_per_token = flop_breakdown.per_token(config.seq_len)
    for warning in flop_breakdown.warnings:
        console.warn(f"FLOP accounting: {warning}")
    tokens_processed = config.final_step * config.batch_size * config.seq_len

    console.table(
        "run configuration",
        (
            *standard_identity_rows(
                config_filename=config.config_filename,
                config_profile=config.execution_type,
                config_sha256=config.config_sha256,
                devices=devices,
                process_count=process_count,
                process_index=process_index,
            ),
            *standard_data_rows(
                source=dataset.source,
                train_tokens=len(dataset.train),
                validation_tokens=len(dataset.validation),
                downstream_domains=len(downstream_domains),
                downstream_tokens=sum(
                    domain.scored_tokens for domain in downstream_domains
                ),
            ),
            *model_console_rows(config, deployment_parameters, params_total),
            *standard_training_rows(
                parameterization=config.parameterization,
                width_multiplier=config.width_multiplier,
                depth_multiplier=config.depth_multiplier,
                data_multiplier=config.data_multiplier,
                batch_size=config.batch_size,
                seq_len=config.seq_len,
                sampling=config.sampling,
                usable_tokens_per_epoch=(
                    shuffled_train_stream.usable_tokens_per_epoch
                    if shuffled_train_stream is not None
                    else None
                ),
                dtype_name=config.dtype_name,
            ),
            *standard_kernel_rows(
                attention_backend=config.attention_backend,
                attention_rows=attention_console_rows(attention_runtime),
                loss_backend=config.loss_backend,
                semantic_vocab_size=config.semantic_vocab_size,
                vocab_tile_size=config.vocab_tile_size,
            ),
            *standard_schedule_rows(
                diagnostics_every=config.diagnostics_every,
                final_step=config.final_step,
                schedule_steps=config.steps,
                early_stopped=config.stop_after_step is not None,
                tokens_processed=tokens_processed,
                total_flops=flops_per_token * tokens_processed,
                flop_breakdown=describe(flop_breakdown),
                capture_window=capture_window,
                xprof_destination=(
                    args.xprof_dir.expanduser().resolve()
                    if capture_window is not None
                    else None
                ),
            ),
            (
                "fuzzy sparsity diagnostics",
                (
                    f"step 1, every {config.sparsity_diagnostics_every:,}, and final"
                    if config.sparsity_diagnostics_every
                    else "disabled"
                ),
            ),
        ),
    )

    mesh = Mesh(np.asarray(devices, dtype=object), ("data",))
    replicated = NamedSharding(mesh, P())
    data_sharding = NamedSharding(mesh, P("data", None))
    attention_fn = make_mesh_attention(
        backend=config.attention_backend,
        mesh=mesh,
        tiles=attention_runtime.tiles,
        softmax_scale=attention_softmax_scale(
            config.attention_scale, config.d_model // config.heads
        ),
        document_masking=config.document_masking,
    )
    sparse_mlp_fn = make_mesh_fuzzy_topk_mlp(
        config=FuzzyTopKConfig(
            top_k=config.mlp_top_k,
            backend=config.sparse_mlp_backend,
        ),
        mesh=mesh,
    )
    reconstruction_kernel_config = FuzzyTopKReconstructionConfig(
        top_k=config.mlp_top_k,
        aux_k=config.aux_k if config.auxk_enabled else None,
    )
    reconstruction_mlp_fn = (
        None
        if config.auxk_enabled
        else make_mesh_fuzzy_topk_mlp_with_reconstruction(
            config=reconstruction_kernel_config,
            mesh=mesh,
        )
    )
    reconstruction_auxk_mlp_fn = (
        make_mesh_fuzzy_topk_mlp_with_reconstruction_auxk(
            config=reconstruction_kernel_config,
            mesh=mesh,
        )
        if config.auxk_enabled
        else None
    )
    sparse_mlp_diagnostic_fn = (
        make_mesh_fuzzy_topk_mlp_with_diagnostics(
            config=FuzzyTopKConfig(
                top_k=config.mlp_top_k,
                backend=config.sparse_mlp_backend,
            ),
            mesh=mesh,
        )
        if config.sparsity_diagnostics_every
        else None
    )
    params = put_replicated_tree(host_params, mesh, replicated, process_count)
    optimizer = put_replicated_tree(host_optimizer, mesh, replicated, process_count)
    del host_params, host_optimizer

    train_rng = np.random.default_rng(args.seed + 1 + process_index * 1_000_003)
    # Compilation may not inspect real data. Shapes and dtypes are sufficient.
    sample_x_host = np.zeros((local_batch, config.seq_len), dtype=np.int32)
    sample_y_host = np.zeros((local_batch, config.seq_len), dtype=np.int32)
    sample_x = put_host_local_array(
        sample_x_host, mesh, P("data", None), data_sharding, process_count
    )
    sample_y = put_host_local_array(
        sample_y_host, mesh, P("data", None), data_sharding, process_count
    )

    compiled_step = jax.jit(
        lambda p, o, x, y: train_step(
            p,
            o,
            x,
            y,
            config,
            decay_mask,
            attention_fn,
            reconstruction_mlp_fn,
            reconstruction_auxk_mlp_fn,
        ),
        in_shardings=(replicated, replicated, data_sharding, data_sharding),
        donate_argnums=(0, 1),
    )
    console.phase("Compiling train step", "compilation is outside train_seconds")
    compile_started = time.perf_counter()
    executable = compiled_step.lower(params, optimizer, sample_x, sample_y).compile()
    train_compile_seconds = time.perf_counter() - compile_started

    diagnostic_executable: Any | None = None
    diagnostic_compile_seconds = 0.0
    if config.diagnostics_every:
        console.phase(
            "Compiling sparse diagnostics",
            "separate executable; compilation is outside train_seconds",
        )
        diagnostic_compile_started = time.perf_counter()
        diagnostic_executable = (
            jax.jit(
                lambda p, o, x, y: diagnostic_train_step(
                    p,
                    o,
                    x,
                    y,
                    config,
                    decay_mask,
                    attention_fn,
                    reconstruction_mlp_fn,
                    reconstruction_auxk_mlp_fn,
                ),
                in_shardings=(replicated, replicated, data_sharding, data_sharding),
                donate_argnums=(0, 1),
            )
            .lower(params, optimizer, sample_x, sample_y)
            .compile()
        )
        diagnostic_compile_seconds = time.perf_counter() - diagnostic_compile_started

    sparsity_diagnostic_executable: Any | None = None
    sparsity_diagnostic_compile_seconds = 0.0
    if config.sparsity_diagnostics_every:
        if sparse_mlp_diagnostic_fn is None:  # defensive construction invariant
            raise AssertionError("sparsity diagnostics have no sparse MLP callable")
        console.phase(
            "Compiling fuzzy sparsity diagnostics",
            "a separate observer keeps the optimizer executable byte-for-byte fixed",
        )
        sparsity_compile_started = time.perf_counter()
        sparsity_diagnostic_executable = (
            jax.jit(
                lambda p, x: fuzzy_sparsity_diagnostics(
                    p,
                    x,
                    config,
                    attention_fn,
                    sparse_mlp_diagnostic_fn,
                ),
                in_shardings=(replicated, data_sharding),
            )
            .lower(params, sample_x)
            .compile()
        )
        sparsity_diagnostic_compile_seconds = (
            time.perf_counter() - sparsity_compile_started
        )

    # Compile evaluation exactly once when it is requested. Diagnostic XProf
    # runs can skip this executable entirely, keeping their setup focused on the
    # training step being inspected.
    compiled_eval: Any | None = None
    sample_mask: jax.Array | None = None
    eval_compile_seconds = 0.0
    if needs_evaluation:
        sample_mask_host = np.ones((local_batch, config.seq_len), dtype=np.float32)
        sample_mask = put_host_local_array(
            sample_mask_host,
            mesh,
            P("data", None),
            data_sharding,
            process_count,
        )
        console.phase("Compiling evaluation", "reused by probes and final validation")
        eval_compile_started = time.perf_counter()
        compiled_eval = (
            jax.jit(
                lambda p, x, y, mask: eval_step(
                    p,
                    x,
                    y,
                    mask,
                    config,
                    attention_fn,
                    sparse_mlp_fn,
                ),
                in_shardings=(replicated, data_sharding, data_sharding, data_sharding),
            )
            .lower(params, sample_x, sample_y, sample_mask)
            .compile()
        )
        eval_compile_seconds = time.perf_counter() - eval_compile_started
    total_compile_seconds = (
        train_compile_seconds
        + diagnostic_compile_seconds
        + sparsity_diagnostic_compile_seconds
        + eval_compile_seconds
    )

    sync_tree((params, optimizer, sample_x, sample_y, sample_mask))
    probe_detail = (
        f"; validation {config.val_probe_batches} batches every {config.val_every} steps"
        if config.val_every
        else "; periodic validation disabled"
    )
    console.phase(
        "Training",
        f"train compiled in {train_compile_seconds:.2f}s, "
        + (
            f"eval in {eval_compile_seconds:.2f}s{probe_detail}"
            if needs_evaluation
            else "evaluation skipped; diagnostic mode"
        ),
    )

    last_metrics: Mapping[str, jax.Array] | None = None
    diagnostic_device_points: list[tuple[int, jax.Array]] = []
    validation_rows: list[ValidationRow] = []
    validation_probe_seconds = 0.0
    if process_count > 1:
        multihost_utils.sync_global_devices("rig-training-start")
    train_started = time.perf_counter()
    xprof_dir = (
        args.xprof_dir.expanduser().resolve() if capture_window is not None else None
    )
    trace_active = False
    # Needed inside the loop for best-effort partial artifacts, not just
    # by the writers that run after it.
    # Captures already pulled off the accelerator. The device list stays
    # bounded by DIAGNOSTIC_FLUSH_POINTS; this keeps the full history so
    # the authoritative writer still sees every point.
    diagnostic_points_host: list[DiagnosticPoint] = []
    output_dir = args.output_dir.expanduser().resolve()
    training_columns = fuzzy_training_log_columns(config)
    progress_log: logpack.LogWriter | None = None
    diagnostic_log: logpack.LogWriter | None = None
    sparsity_log: vectorlog.VectorLogWriter | None = None
    sparsity_point_count = 0
    if is_controller:
        output_dir.mkdir(parents=True, exist_ok=True)
        # A stale file from a reused directory would be appended to.
        (output_dir / TRAINING_LOG_NAME).unlink(missing_ok=True)
        (output_dir / DIAGNOSTICS_LOG_NAME).unlink(missing_ok=True)
        (output_dir / FUZZY_SPARSITY_LOG_NAME).unlink(missing_ok=True)
        (output_dir / _FUZZY_SPARSITY_TEMP_NAME).unlink(missing_ok=True)
        progress_log = open_log(
            output_dir / TRAINING_LOG_NAME,
            training_columns,
            tokens_per_step=config.batch_size * config.seq_len,
            flops_per_token=flops_per_token,
        )
        if config.sparsity_diagnostics_every:
            hidden_width = config.mlp_mult * config.d_model
            sparsity_log = vectorlog.VectorLogWriter(
                output_dir / _FUZZY_SPARSITY_TEMP_NAME,
                FUZZY_FEATURE_STAT_NAMES,
                layer_count=config.layers,
                feature_count=hidden_width,
                group_size=hidden_width // config.mlp_top_k,
                tokens_per_step=config.batch_size * config.seq_len,
                flops_per_token=flops_per_token,
            )
        diagnostic_log = open_log(
            output_dir / DIAGNOSTICS_LOG_NAME,
            diagnostic_log_columns(diagnostic_metadata),
            tokens_per_step=config.batch_size * config.seq_len,
            flops_per_token=flops_per_token,
        )
    try:
        # final_step is the horizon unless --stop-after-step truncates it.
        # The schedule below still spans config.steps, so a truncated run walks
        # exactly the prefix of the trajectory it samples.
        for step_index in range(1, config.final_step + 1):
            if capture_window is not None and step_index == capture_window[0]:
                # Drain earlier asynchronous work before opening the trace. The
                # capture therefore begins at the requested steady-state step,
                # rather than including a backlog dispatched by preceding steps.
                sync_tree((params, optimizer, last_metrics))
                assert xprof_dir is not None
                if is_controller:
                    # TPU VM filesystems are independent. Capture the controller's
                    # local chips while every process still runs the distributed
                    # step; this gives worker 0 a self-contained trace to serve.
                    xprof_dir.mkdir(parents=True, exist_ok=True)
                    console.phase(
                        "Starting XProf capture",
                        f"steps {capture_window[0]}..{capture_window[1]} → {xprof_dir}",
                    )
                    jax.profiler.start_trace(
                        xprof_dir,
                        profiler_options=profiler_options(
                            platform, int(jax.local_device_count())
                        ),
                    )
                    trace_active = True
                if process_count > 1:
                    multihost_utils.sync_global_devices("rig-xprof-capture-started")

            annotation = (
                jax.profiler.StepTraceAnnotation("train", step_num=step_index)
                if trace_active
                else nullcontext()
            )
            with annotation:
                # Keep the host sampling, transfer, dispatch, and any logging
                # synchronization inside the step annotation. This exposes input
                # gaps alongside TPU execution in the same XProf timeline.
                if shuffled_train_stream is None:
                    batch_x, batch_y = dataset.batch(
                        "train",
                        train_rng,
                        local_batch,
                        config.seq_len,
                        config.semantic_vocab_size,
                    )
                else:
                    batch_x, batch_y = shuffled_train_stream.next_batch()
                batch_x = put_host_local_array(
                    batch_x, mesh, P("data", None), data_sharding, process_count
                )
                batch_y = put_host_local_array(
                    batch_y, mesh, P("data", None), data_sharding, process_count
                )
                diagnostic_values_at_step: jax.Array | None = None
                feature_statistics_at_step: jax.Array | None = None
                if should_run_sparsity_diagnostics(
                    step_index,
                    every=config.sparsity_diagnostics_every,
                    final_step=config.final_step,
                ):
                    if sparsity_diagnostic_executable is None:
                        raise AssertionError(
                            "sparsity diagnostic executable was not compiled"
                        )
                    feature_statistics_at_step = sparsity_diagnostic_executable(
                        params, batch_x
                    )
                    # Finish every observer read before the ordinary update
                    # donates the same parameter buffers. This observer has no
                    # state and cannot influence the update executable.
                    sync_tree(feature_statistics_at_step)

                if should_run_diagnostics(
                    step_index,
                    every=config.diagnostics_every,
                    final_step=config.final_step,
                ):
                    if diagnostic_executable is None:  # defensive invariant
                        raise AssertionError("diagnostic executable was not compiled")
                    params, optimizer, last_metrics, diagnostic_values_at_step = (
                        diagnostic_executable(params, optimizer, batch_x, batch_y)
                    )
                else:
                    params, optimizer, last_metrics = executable(
                        params, optimizer, batch_x, batch_y
                    )

                if diagnostic_values_at_step is not None:
                    diagnostic_device_points.append(
                        (step_index, diagnostic_values_at_step)
                    )
                    if len(diagnostic_device_points) >= DIAGNOSTIC_FLUSH_POINTS:
                        # Pull to the host and drop the device references. This
                        # is what bounds accelerator residency: without it one
                        # small allocation per capture lives until the run ends.
                        flushed = [
                            DiagnosticPoint(
                                step, np.asarray(local_device_get(v), dtype=np.float32)
                            )
                            for step, v in diagnostic_device_points
                        ]
                        diagnostic_device_points.clear()
                        diagnostic_points_host.extend(flushed)
                        for point in flushed:
                            append_log_row(
                                diagnostic_log,
                                point.step,
                                np.asarray(point.values, dtype=np.float32).reshape(-1),
                            )
                if feature_statistics_at_step is not None and is_controller:
                    if sparsity_log is None:
                        raise AssertionError("sparsity vector log was not opened")
                    sparsity_log.append(
                        step_index,
                        np.asarray(
                            local_device_get(feature_statistics_at_step),
                            dtype=np.float32,
                        ),
                    )
                    sparsity_point_count += 1
                if should_run_validation_probe(
                    step_index,
                    every=config.val_every,
                    final_step=config.final_step,
                ):
                    # Attribute all preceding asynchronous training work to training,
                    # then start the probe's own honest wall clock inside the helper.
                    sync_tree((params, optimizer, last_metrics))
                    if compiled_eval is None:  # defensive configuration invariant
                        raise AssertionError("validation executable was not compiled")
                    probe = evaluate_validation_prefix(
                        params,
                        dataset,
                        compiled_eval,
                        data_sharding,
                        batch_size=config.batch_size,
                        seq_len=config.seq_len,
                        semantic_vocab_size=config.semantic_vocab_size,
                        batches=config.val_probe_batches,
                        mesh=mesh,
                        process_index=process_index,
                        process_count=process_count,
                    )
                    validation_probe_seconds += probe.seconds
                    validation_rows.append(
                        probe.validation_row(
                            step=step_index,
                            tokens_processed=(
                                step_index * config.batch_size * config.seq_len
                            ),
                            kind="fineweb_probe",
                            domain="fineweb",
                            canonical=False,
                        )
                    )
                    console.validation_probe(
                        step_index,
                        probe.loss,
                        config.val_probe_batches,
                        probe.seconds,
                    )
                should_log = (
                    step_index == 1
                    or step_index == config.final_step
                    or step_index % config.log_every == 0
                )
                if should_log:
                    host_metrics = local_device_get(last_metrics)
                    elapsed_so_far = max(time.perf_counter() - train_started, 1.0e-12)
                    seen_tokens = step_index * config.batch_size * config.seq_len
                    # host_metrics is already on the host for the progress
                    # line, so this adds no synchronization.
                    append_log_row(
                        progress_log,
                        step_index,
                        np.concatenate(
                            (
                                np.asarray(
                                    (
                                        float(host_metrics["loss"]),
                                        float(host_metrics["learning_rate"]),
                                        float(host_metrics["grad_norm"]),
                                    ),
                                    dtype=np.float32,
                                ),
                                np.asarray(
                                    host_metrics["reconstruction_row"],
                                    dtype=np.float32,
                                ),
                                np.asarray(
                                    host_metrics["auxk_age_row"], dtype=np.float32
                                ),
                            )
                        ),
                    )
                    console.step(
                        step_index,
                        config.final_step,
                        float(host_metrics["loss"]),
                        float(host_metrics["learning_rate"]),
                        float(host_metrics["grad_norm"]),
                        seen_tokens / elapsed_so_far,
                    )

                if capture_window is not None and step_index == capture_window[1]:
                    # Include the final synchronization in the trace so all
                    # captured TPU work is exported before profiling stops.
                    sync_tree((params, optimizer, last_metrics))

            if capture_window is not None and step_index == capture_window[1]:
                if process_count > 1:
                    multihost_utils.sync_global_devices("rig-xprof-capture-finished")
                if trace_active:
                    jax.profiler.stop_trace()
                    trace_active = False
                    console.phase("XProf capture saved", str(xprof_dir))
                if process_count > 1:
                    multihost_utils.sync_global_devices("rig-xprof-capture-stopped")
    finally:
        if trace_active:
            # Avoid leaving process-global profiler state active when a sampled
            # batch or training step raises midway through the capture window.
            jax.profiler.stop_trace()
        # Release both handles before the final writers replace these paths, so
        # a salvage append can never land after the authoritative artifact.
        close_log(progress_log)
        close_log(diagnostic_log)
        if sparsity_log is not None:
            sparsity_log.close()

    if last_metrics is None:  # defensive: argparse prevents zero steps
        raise AssertionError("training produced no metrics")
    # Sparse diagnostic reductions are part of benchmark time even if their
    # result branch is otherwise independent of the next optimizer state.
    sync_tree((params, optimizer, last_metrics, diagnostic_device_points))
    if process_count > 1:
        multihost_utils.sync_global_devices("rig-training-finished")
    train_seconds = max(time.perf_counter() - train_started, 1.0e-12)
    final_train = local_device_get(last_metrics)
    training_history = np.asarray(
        local_device_get(optimizer["history"]), dtype=np.float32
    )
    # Points already pulled by an intermediate flush, then whatever is still
    # resident. Dropping the first group here would silently truncate the
    # authoritative file to the last partial buffer.
    diagnostic_points = tuple(
        [
            *diagnostic_points_host,
            *(
                DiagnosticPoint(
                    step, np.asarray(local_device_get(values), dtype=np.float32)
                )
                for step, values in diagnostic_device_points
            ),
        ]
    )
    sparsity_recorded = False
    if is_controller and config.sparsity_diagnostics_every:
        # Keep the growing multi-hundred-MB log under the harness's excluded
        # dotted-temp convention. Otherwise its once-a-minute salvage rsync
        # repeatedly scans and transfers a file that is only authoritative for
        # completed runs. One atomic rename exposes it for the final pull.
        (output_dir / _FUZZY_SPARSITY_TEMP_NAME).replace(
            output_dir / FUZZY_SPARSITY_LOG_NAME
        )
        captured = vectorlog.read_vector_log(output_dir / FUZZY_SPARSITY_LOG_NAME)
        if (
            len(captured) != sparsity_point_count
            or len(captured) == 0
            or int(captured.steps[0]) != 1
            or int(captured.steps[-1]) != config.final_step
        ):
            raise ValueError("fuzzy sparsity log does not cover its capture schedule")
        sparsity_recorded = True
    train_loss = finite_metric("train_loss", float(final_train["loss"]))
    training_objective = finite_metric(
        "training_objective", float(final_train["objective"])
    )
    reconstruction_names = reconstruction_stat_names(config)
    final_reconstruction_row = np.asarray(
        final_train["reconstruction_row"], dtype=np.float32
    )
    reconstruction_width = len(reconstruction_names)
    final_reconstruction_model = final_reconstruction_row[:reconstruction_width]
    final_reconstruction_layers = final_reconstruction_row[
        reconstruction_width:
    ].reshape(config.layers, reconstruction_width)
    final_auxk_age_row = np.asarray(final_train["auxk_age_row"], dtype=np.float32)
    if config.auxk_enabled:
        age_width = len(AUXK_AGE_STAT_NAMES)
        final_auxk_age_model = final_auxk_age_row[:age_width]
        final_auxk_age_layers = final_auxk_age_row[age_width:].reshape(
            config.layers, age_width
        )
    else:
        final_auxk_age_model = np.zeros((0,), dtype=np.float32)
        final_auxk_age_layers = np.zeros((config.layers, 0), dtype=np.float32)

    if diagnostic_mode:
        output_dir = args.output_dir.expanduser().resolve()
        if is_controller:
            write_training_log(
                output_dir,
                training_history,
                tokens_per_step=config.batch_size * config.seq_len,
                final_step=config.final_step,
                flops_per_token=flops_per_token,
                columns=training_columns,
            )
            if diagnostic_points:
                write_diagnostics_log(
                    output_dir,
                    diagnostic_points,
                    diagnostic_metadata,
                    tokens_per_step=config.batch_size * config.seq_len,
                    final_step=config.final_step,
                    flops_per_token=flops_per_token,
                )
        diagnostic_rate = finite_metric(
            "tokens_per_second", tokens_processed / train_seconds, positive=True
        )
        assert capture_window is not None and xprof_dir is not None
        console.table(
            "profile complete",
            (
                ("training steps", f"{config.final_step:,}"),
                ("captured steps", f"{capture_window[0]}..{capture_window[1]}"),
                ("train loss", f"{train_loss:.4f}"),
                ("diagnostic rate", f"{format_rate(diagnostic_rate)} tok/s"),
                ("training curve", output_dir / TRAINING_LOG_NAME),
                (
                    "diagnostics",
                    (
                        output_dir / DIAGNOSTICS_LOG_NAME
                        if diagnostic_points
                        else "disabled"
                    ),
                ),
                ("XProf trace", xprof_dir),
            ),
        )
        if process_count > 1:
            multihost_utils.sync_global_devices("rig-profile-artifacts-written")
        return None

    console.phase(
        "Canonical validation",
        f"{config.eval_batches} deterministic batches outside train_seconds",
    )
    if compiled_eval is None:  # defensive configuration invariant
        raise AssertionError("final validation executable was not compiled")
    canonical_evaluation = evaluate_validation_prefix(
        params,
        dataset,
        compiled_eval,
        data_sharding,
        batch_size=config.batch_size,
        seq_len=config.seq_len,
        semantic_vocab_size=config.semantic_vocab_size,
        batches=config.eval_batches,
        mesh=mesh,
        process_index=process_index,
        process_count=process_count,
    )

    downstream_evaluations = ()
    if downstream_domains:
        console.phase(
            "Fresh-domain validation",
            f"{len(downstream_domains)} domains outside train_seconds",
        )
        downstream_evaluations = evaluate_downstream_domains(
            params,
            downstream_domains,
            compiled_eval,
            data_sharding,
            batch_size=config.batch_size,
            seq_len=config.seq_len,
            mesh=mesh,
            process_index=process_index,
            process_count=process_count,
        )
    else:
        console.phase("Fresh-domain validation", "skipped; no downstream data supplied")
    evaluation_report = EvaluationReport(canonical_evaluation, downstream_evaluations)
    console.evaluations(evaluation_report)
    validation_rows.extend(
        evaluation_report.validation_rows(
            step=config.final_step,
            tokens_processed=tokens_processed,
        )
    )
    validation_loss = canonical_evaluation.loss
    final_validation_seconds = canonical_evaluation.seconds

    output_dir = args.output_dir.expanduser().resolve()
    artifact_names = [TRAINING_LOG_NAME, VALIDATION_CSV_NAME]
    if diagnostic_points:
        artifact_names.append(DIAGNOSTICS_LOG_NAME)
    if sparsity_recorded:
        artifact_names.append(FUZZY_SPARSITY_LOG_NAME)
    if not args.omit_checkpoint:
        artifact_names.append(CHECKPOINT_NAME)
    console.phase("Artifacts", " + ".join(artifact_names))
    if is_controller:
        write_training_log(
            output_dir,
            training_history,
            tokens_per_step=config.batch_size * config.seq_len,
            final_step=config.final_step,
            flops_per_token=flops_per_token,
            columns=training_columns,
        )
        if diagnostic_points:
            write_diagnostics_log(
                output_dir,
                diagnostic_points,
                diagnostic_metadata,
                tokens_per_step=config.batch_size * config.seq_len,
                final_step=config.final_step,
                flops_per_token=flops_per_token,
            )
        write_validation_csv(output_dir, validation_rows)
        if not args.omit_checkpoint:
            save_checkpoint(
                output_dir,
                deployment_params(params),
                checkpoint_metadata(config, args.seed, attention_runtime),
            )

    tokens_per_second = finite_metric(
        "tokens_per_second", tokens_processed / train_seconds, positive=True
    )
    total_flops = int(flops_per_token * tokens_processed)
    achieved_tflops = finite_metric(
        "achieved_tflops", total_flops / train_seconds / 1.0e12
    )
    peak_tflops = inferred_peak_tflops(args.peak_tflops, devices)
    mfu = achieved_tflops / peak_tflops if peak_tflops is not None else 0.0
    using_builtin_data = not args.train_data and not args.val_data
    dataset_id = args.dataset_id or (
        "builtin-byte-v1" if using_builtin_data else "fineweb10b-gpt2"
    )
    tokenizer_id = args.tokenizer_id or ("byte" if using_builtin_data else "gpt2")
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "track": "open",
        "profile": profile,
        "seed": int(args.seed),
        "checkpoint": None if args.omit_checkpoint else CHECKPOINT_NAME,
        "artifacts": {
            "training_curve": TRAINING_LOG_NAME,
            "validation_curve": VALIDATION_CSV_NAME,
            **({"diagnostics": DIAGNOSTICS_LOG_NAME} if diagnostic_points else {}),
            **(
                {"fuzzy_sparsity": FUZZY_SPARSITY_LOG_NAME}
                if sparsity_recorded
                else {}
            ),
        },
        "system": {
            **system_metadata(devices),
            "controller_process_index": process_index,
        },
        "contract": {
            "model_id": "fuzzy-topk-reconstruction-gpt-v1-family",
            "dataset_id": dataset_id,
            "tokenizer_id": tokenizer_id,
            "sequence_length": config.seq_len,
            "context_preset": config.context_preset,
            "model": contract_model_metadata(config),
        },
        # Keep kernel choices in implementation provenance so the architecture
        # metadata remains easy to compare across otherwise different recipes.
        "implementation": implementation_metadata(config, attention_runtime),
        "evaluations": evaluation_report.metadata(),
        "metrics": {
            "train_seconds": finite_metric(
                "train_seconds", train_seconds, positive=True
            ),
            "tokens_processed": int(tokens_processed),
            "training_token_budget": int(tokens_processed),
            "training_steps": int(config.final_step),
            "schedule_steps": int(config.steps),
            "stop_after_step": (
                int(config.stop_after_step)
                if config.stop_after_step is not None
                else None
            ),
            "model_tier": config.tier,
            "parameter_count": int(deployment_parameters),
            "deployment_parameter_count": int(deployment_parameters),
            "train_only_parameter_count": int(config.reconstruction_parameter_count),
            "optimized_parameter_count": int(params_total),
            "tokens_per_parameter": (
                float(config.tokens_per_parameter)
                if config.tokens_per_parameter is not None
                else None
            ),
            "target_tokens_per_parameter": (
                float(config.target_tokens_per_parameter)
                if config.target_tokens_per_parameter is not None
                else None
            ),
            "base_learning_rate": float(config.learning_rate),
            "training_sampling": config.sampling,
            "training_data_sharding": (
                "rank_disjoint_shuffled_windows"
                if shuffled_train_stream is not None
                else "rank_local_random_windows"
            ),
            "training_usable_tokens_per_epoch": int(
                shuffled_train_stream.usable_tokens_per_epoch
                if shuffled_train_stream is not None
                else len(dataset.train)
            ),
            "training_data_epochs": finite_metric(
                "training_data_epochs",
                tokens_processed
                / (
                    shuffled_train_stream.usable_tokens_per_epoch
                    if shuffled_train_stream is not None
                    else len(dataset.train)
                ),
                positive=True,
            ),
            "validation_loss": validation_loss,
            "validation_tokens": canonical_evaluation.scored_tokens,
            "validation_probe_count": sum(
                row.kind == "fineweb_probe" for row in validation_rows
            ),
            "diagnostic_point_count": len(diagnostic_points),
            "diagnostics_every": int(config.diagnostics_every),
            "sparsity_diagnostic_point_count": int(sparsity_point_count),
            "sparsity_diagnostics_every": int(config.sparsity_diagnostics_every),
            "validation_probe_seconds": finite_metric(
                "validation_probe_seconds", validation_probe_seconds
            ),
            "final_validation_seconds": finite_metric(
                "final_validation_seconds", final_validation_seconds, positive=True
            ),
            "train_loss": train_loss,
            "training_objective": training_objective,
            "reconstruction": {
                "model": {
                    name: finite_metric(name, float(value))
                    for name, value in zip(
                        reconstruction_names, final_reconstruction_model, strict=True
                    )
                },
                "layers": [
                    {
                        name: finite_metric(name, float(value))
                        for name, value in zip(
                            reconstruction_names, layer_values, strict=True
                        )
                    }
                    for layer_values in final_reconstruction_layers
                ],
                "auxk_age_model": (
                    {
                        name: finite_metric(name, float(value))
                        for name, value in zip(
                            AUXK_AGE_STAT_NAMES,
                            final_auxk_age_model,
                            strict=True,
                        )
                    }
                    if config.auxk_enabled
                    else {}
                ),
                "auxk_age_layers": (
                    [
                        {
                            name: finite_metric(name, float(value))
                            for name, value in zip(
                                AUXK_AGE_STAT_NAMES, layer_values, strict=True
                            )
                        }
                        for layer_values in final_auxk_age_layers
                    ]
                    if config.auxk_enabled
                    else []
                ),
            },
            "parameters": int(deployment_parameters),
            "flops_per_token": int(flops_per_token),
            "estimated_total_flops": total_flops,
            # Traced from the jaxpr, not a maintained formula. The breakdown
            # attributes the count to its sources so a change in architecture
            # is auditable after the fact; warnings record any work the walker
            # could not account for.
            "flop_accounting": {
                "method": "traced-jaxpr",
                "matmul_per_sequence": int(flop_breakdown.matmul),
                "elementwise_per_sequence": int(flop_breakdown.elementwise),
                "by_site": {
                    label: int(value)
                    for label, value in sorted(flop_breakdown.by_site.items())
                },
                "warnings": list(flop_breakdown.warnings),
            },
            "tokens_per_second": tokens_per_second,
            "achieved_tflops": achieved_tflops,
            "mfu_estimate": finite_metric("mfu_estimate", mfu),
            "attention_tune_seconds": finite_metric(
                "attention_tune_seconds", attention_runtime.tune_seconds
            ),
            "train_compile_seconds": finite_metric(
                "train_compile_seconds", train_compile_seconds
            ),
            "eval_compile_seconds": finite_metric(
                "eval_compile_seconds", eval_compile_seconds
            ),
            "diagnostic_compile_seconds": finite_metric(
                "diagnostic_compile_seconds", diagnostic_compile_seconds
            ),
            "sparsity_diagnostic_compile_seconds": finite_metric(
                "sparsity_diagnostic_compile_seconds",
                sparsity_diagnostic_compile_seconds,
            ),
            "total_compile_seconds": finite_metric(
                "total_compile_seconds", total_compile_seconds
            ),
        },
    }
    if is_controller:
        write_result(output_dir, result)
    console.success(validation_loss, train_seconds, final_validation_seconds)
    if process_count > 1:
        multihost_utils.sync_global_devices("rig-final-artifacts-written")
    return result if is_controller else None


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.print_plan:
        try:
            profile = selected_profile(args)
            experiment_config, config_sha256 = load_experiment_config(profile)
            validate_args(args, experiment_config)
            planned = resolve_config(
                args,
                (
                    "cpu"
                    if experiment_config.run.kernels.attention_backend == "dense"
                    else "tpu"
                ),
                experiment_config=experiment_config,
                config_sha256=config_sha256,
            )
        except Exception as error:
            print(f"\nerror: {error}", file=sys.stderr)
            return 1
        print(
            json.dumps(
                resolved_plan_metadata(planned), sort_keys=True, separators=(",", ":")
            )
        )
        return 0
    try:
        result = run(args)
    except Exception as error:
        # A concise colored-ish error is useful interactively; a traceback can be
        # requested naturally via Python's exception chaining during development.
        print(f"\nerror: {error}", file=sys.stderr)
        if os.environ.get("RIG_DEBUG") == "1":
            raise
        return 1
    if result is not None:
        print(
            RESULT_PREFIX
            + json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
