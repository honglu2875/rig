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
import functools
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

from jax.experimental.pallas.ops.tpu import megablox

from rig import logpack
from rig.arguments import positive_int
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
from rig.diagnostics import (
    DIAGNOSTIC_PERCENTILE_SAMPLE_SIZE,
    diagnostic_scope_metadata,
    diagnostic_values,
)
from rig.metrics import DIAGNOSTIC_EXTENDED_STATS, DIAGNOSTIC_FAMILIES
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
    ROUTER_SUMMARY_METRICS,
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
    count_training_flops,
    default_rules,
    describe,
)
from rig.kernels import (
    AttentionConfig,
    make_causal_attention,
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
_EXPERT_PARAMETER_NAMES = frozenset(
    {"expert_up_w", "expert_up_b", "expert_down_w", "expert_down_b"}
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
    expert_load_scaling_mode: str
    expert_load_scaling_strength: float
    eval_batches: int
    val_every: int
    val_probe_batches: int
    diagnostics_every: int
    log_every: int
    vocab_size: int
    semantic_vocab_size: int
    attention_backend: str
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
    # Routing. experts=0 keeps the dense MLP, so this file still describes the
    # baseline exactly when the keys are absent.
    experts: int = 0
    expert_top_k: int = 2
    expert_mult: int = 2
    router_aux_coefficient: float = 0.01

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
    """The fixed-TPP CompleteP contract shared by every routed family tier."""

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
    normalization: Literal["rms_norm"]
    position_encoding: Literal["rope_base_10000"]
    mlp_activation: Literal["gelu"]
    vocab_size: PositiveInt
    semantic_vocab_size: PositiveInt
    experts: NonnegativeInt = 0
    expert_top_k: PositiveInt = 2
    expert_mult: PositiveInt = 2
    router_aux_coefficient: NonnegativeFloat = 0.01

    @property
    def head_dim(self) -> int:
        """Width of one attention head."""

        return self.d_model // self.heads

    def validate(self, label: str) -> None:
        """Enforce architecture relations that no single annotation can express."""

        if self.semantic_vocab_size > self.vocab_size:
            raise ValueError(
                f"{label}.semantic_vocab_size must not exceed vocab_size"
            )
        if self.d_model % self.heads:
            raise ValueError(f"{label}.d_model must be divisible by heads")
        if self.head_dim % 2:
            raise ValueError(
                f"{label} head dimension must be even for RoPE"
            )
        if self.experts:
            if self.expert_top_k > self.experts:
                raise ValueError(
                    f"{label}.expert_top_k must not exceed experts"
                )
            if self.expert_top_k * self.expert_mult != self.mlp_mult:
                raise ValueError(
                    f"{label} must satisfy expert_top_k * expert_mult "
                    "== mlp_mult so active MLP FLOPs match the dense tier"
                )


@dataclass(frozen=True, slots=True)
class TierDefinition:
    model: ModelDefinition

    @property
    def tpp_parameters(self) -> int:
        """Dense-equivalent parameter denominator used by the fixed-TPP ladder."""

        model = self.model
        width = model.d_model
        return (
            2 * model.vocab_size * width
            + model.layers * (12 * width * width + 11 * width)
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
    expert_load_scaling_mode: Literal["gradient", "update"]
    expert_load_scaling_strength: Probability


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
    log_every: PositiveInt


@dataclass(frozen=True, slots=True)
class RunDefinition:
    training: TrainingSettings
    kernels: KernelSettings
    optimizer: OptimizerSettings
    evaluation: EvaluationSettings
    logging: LoggingSettings


@dataclass(frozen=True, slots=True)
class ExperimentConfig(ConfigSchema):
    """Complete typed representation of one selected standalone YAML file."""

    schema_version: Literal[7]
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
    optim.add_argument(
        "--expert-load-scaling-mode",
        choices=("gradient", "update"),
        default=None,
        help=(
            "override whether current expert load scales gradients before Adam "
            "or normalized AdamW updates"
        ),
    )
    optim.add_argument(
        "--expert-load-scaling-strength",
        type=float,
        default=None,
        help=(
            "override interpolation strength from no scaling (0) to the full "
            "sqrt expert-load ratio (1)"
        ),
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
    if args.expert_load_scaling_strength is not None and (
        not math.isfinite(args.expert_load_scaling_strength)
        or not 0.0 <= args.expert_load_scaling_strength <= 1.0
    ):
        raise ValueError("--expert-load-scaling-strength must be between 0 and 1")
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
    base_tpp_parameters = family.base_tpp_parameters
    definition = experiment_config.run
    tier = family.tiers[selected_tier_name]
    model = tier.model
    duration = definition.training.duration
    tpp_parameters = tier.tpp_parameters if duration.is_fixed_tpp else None
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
    if not duration.is_fixed_tpp:
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

    dtype_name = training.dtype
    compute_dtype = jnp.bfloat16 if dtype_name == "bfloat16" else jnp.float32
    attention_backend = kernels.attention_backend
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
    data_multiplier = (
        tpp_parameters / float(base_tpp_parameters)
        if tpp_parameters is not None
        else 1.0
    )
    base_learning_rate = (
        args.base_learning_rate
        if args.base_learning_rate is not None
        else optimizer.learning_rate
    )
    expert_load_scaling_mode = (
        args.expert_load_scaling_mode
        if args.expert_load_scaling_mode is not None
        else optimizer.expert_load_scaling_mode
    )
    expert_load_scaling_strength = (
        args.expert_load_scaling_strength
        if args.expert_load_scaling_strength is not None
        else optimizer.expert_load_scaling_strength
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
        experts=model.experts,
        expert_top_k=model.expert_top_k,
        expert_mult=model.expert_mult,
        router_aux_coefficient=model.router_aux_coefficient,
        batch_size=batch_size,
        seq_len=seq_len,
        sampling=training.sampling,
        layers=model.layers,
        heads=model.heads,
        d_model=model.d_model,
        mlp_mult=model.mlp_mult,
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
        expert_load_scaling_mode=expert_load_scaling_mode,
        expert_load_scaling_strength=expert_load_scaling_strength,
        eval_batches=eval_batches,
        val_every=val_every,
        val_probe_batches=val_probe_batches,
        diagnostics_every=diagnostics_every,
        log_every=log_every,
        vocab_size=model.vocab_size,
        semantic_vocab_size=model.semantic_vocab_size,
        attention_backend=attention_backend,
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
    # Each expert is expert_mult wide; top_k of them fire, so active MLP FLOPs
    # match the dense mlp_mult when expert_mult * top_k == mlp_mult.
    expert_hidden = config.expert_mult * d_model
    hidden_scale = (
        config.init_std / math.sqrt(config.width_multiplier)
        if config.parameterization == "completep_fixed_tpp_v1"
        else 0.02
    )
    blocks: list[dict[str, np.ndarray]] = []
    for _ in range(config.layers):
        blocks.append(
            {
                "ln1_scale": np.ones((d_model,), dtype=np.float32),
                "qkv_w": normal(rng, (d_model, 3 * d_model), hidden_scale),
                "qkv_b": np.zeros((3 * d_model,), dtype=np.float32),
                "attn_w": normal(rng, (d_model, d_model), hidden_scale),
                "attn_b": np.zeros((d_model,), dtype=np.float32),
                "ln2_scale": np.ones((d_model,), dtype=np.float32),
                **(
                    {
                        # Router logits are a readout: E does not scale with
                        # width, so it follows the unembedding's rules. Small
                        # init keeps routing near-uniform at step one.
                        "router_w": normal(
                            rng, (d_model, config.experts), hidden_scale
                        ),
                        "expert_up_w": normal(
                            rng, (config.experts, d_model, expert_hidden), hidden_scale
                        ),
                        "expert_up_b": np.zeros(
                            (config.experts, expert_hidden), dtype=np.float32
                        ),
                        "expert_down_w": normal(
                            rng, (config.experts, expert_hidden, d_model), hidden_scale
                        ),
                        "expert_down_b": np.zeros(
                            (config.experts, d_model), dtype=np.float32
                        ),
                    }
                    if config.experts
                    else {
                        "mlp_up_w": normal(rng, (d_model, hidden), hidden_scale),
                        "mlp_up_b": np.zeros((hidden,), dtype=np.float32),
                        "mlp_down_w": normal(rng, (hidden, d_model), hidden_scale),
                        "mlp_down_b": np.zeros((d_model,), dtype=np.float32),
                    }
                ),
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
        "normalization": config.normalization,
        "position_encoding": config.position_encoding,
        "mlp_activation": config.mlp_activation,
        "vocab_size": config.vocab_size,
        "semantic_vocab_size": config.semantic_vocab_size,
        "tied_embeddings": config.embeddings == "tied",
        "tier": config.tier,
        "parameterization": config.parameterization,
    }


def model_console_rows(
    config: Config,
    total_parameters: int,
    active_parameters: int,
) -> tuple[tuple[str, object], ...]:
    """Describe routed architecture facts the standard run card cannot infer."""

    if not config.experts:
        return (
            (
                "model",
                f"{config.tier} · L{config.layers} D{config.d_model} H{config.heads} "
                f"RoPE RMSNorm GELU MLP×{config.mlp_mult}",
            ),
            ("parameters", format_count(total_parameters)),
        )
    return (
        (
            "model",
            f"{config.tier} · L{config.layers} D{config.d_model} H{config.heads} · "
            f"MoE {config.experts}×top-{config.expert_top_k} "
            f"expert×{config.expert_mult}",
        ),
        (
            "parameters",
            f"{format_count(active_parameters)} active / "
            f"{format_count(total_parameters)} total",
        ),
        (
            "routing",
            f"dropless · balance coefficient {config.router_aux_coefficient:g} · "
            f"load-{config.expert_load_scaling_mode} strength "
            f"{config.expert_load_scaling_strength:g}",
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
                "loss_backend": config.loss_backend,
                "vocab_tile_size": config.vocab_tile_size,
                "document_masking": config.document_masking,
            },
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
                "expert_load_scaling": {
                    "mode": config.expert_load_scaling_mode,
                    "strength": config.expert_load_scaling_strength,
                    "reference_load": "uniform_1_over_experts",
                    "full_factor": "sqrt(experts_times_current_load)",
                    "load_statistic": "current_global_batch_hard_assignments",
                    "gradient_application": "after_global_clip_before_adam_moments",
                    "update_application": "normalized_adam_plus_decoupled_decay",
                },
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
            "normalization": config.normalization,
            "position_encoding": config.position_encoding,
            "mlp_activation": config.mlp_activation,
            "dtype": config.dtype_name,
            "attention_backend": config.attention_backend,
            "attention_tuning": attention_runtime_metadata(attention_runtime),
            "loss_backend": config.loss_backend,
            "vocab_tile_size": config.vocab_tile_size,
            "tied_embeddings": config.embeddings == "tied",
            "tier": config.tier,
            "parameterization": config.parameterization,
        },
    }


def implementation_metadata(
    config: Config, runtime: AttentionRuntime
) -> dict[str, Any]:
    """Return systems/kernel provenance that may vary in either track."""

    return {
        "attention_backend": config.attention_backend,
        "attention_tuning": attention_runtime_metadata(runtime),
        "loss_backend": config.loss_backend,
        "vocab_tile_size": config.vocab_tile_size,
        "weight_decay_policy": "weights_and_embeddings_only_v2",
        "expert_load_scaling": {
            "mode": config.expert_load_scaling_mode,
            "strength": config.expert_load_scaling_strength,
        },
        "diagnostics": {
            "families": list(DIAGNOSTIC_FAMILIES),
            "statistics": list(DIAGNOSTIC_EXTENDED_STATS),
            "expert_scope": "four_expert_ffn_parameter_tensors",
            "percentile_method": "deterministic_midpoint_scope_sample",
            "percentile_max_elements": DIAGNOSTIC_PERCENTILE_SAMPLE_SIZE,
        },
        "context_preset": config.context_preset,
        "document_masking": config.document_masking,
        "configuration": experiment_config_metadata(config),
    }


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class RouterStats:
    """What the routed blocks report back for logging and the auxiliary loss.

    Registered as a pytree because it is returned as the aux output of
    ``jax.value_and_grad(..., has_aux=True)``: an unregistered dataclass is an
    opaque leaf to JAX, so the tracers inside its fields would not be
    recognized as part of the traced computation and would dangle once the
    transformation exits -- an ``UnexpectedTracerError`` at first use, not at
    construction.
    """

    balance_loss: jax.Array
    # [layers, experts]: the fraction of assignments each expert received.
    load: jax.Array
    # [layers, 3]: entropy, top-1 gate, logit RMS, in ROUTER_SUMMARY_METRICS order.
    summary: jax.Array


def active_parameter_count(params: Any, config: Config) -> int:
    """Parameters a single token actually visits.

    A routed model stores ``experts`` copies of the MLP and visits
    ``expert_top_k`` of them, so its total is not what the tier declares. The
    ladder is defined by *active* parameters -- that is what makes a sparse
    tier comparable with the dense tier of the same name, and what makes the
    two equi-FLOP.
    """

    total = parameter_count(params)
    if not config.experts:
        return total
    width, hidden = config.d_model, config.expert_mult * config.d_model
    per_expert = width * hidden + hidden + hidden * width + width
    unvisited = config.experts - config.expert_top_k
    return total - config.layers * unvisited * per_expert


def expected_active_parameters(config: Config) -> int:
    """What the declared tier size becomes once the MLP is routed.

    Routing preserves the dense MLP's active *width* exactly, because
    ``expert_top_k * expert_mult == mlp_mult``. It adds two things and only
    two: the router projection, and one extra set of expert biases per
    additional expert a token visits. Both are named here rather than absorbed
    into a tolerance, so the check stays a check.
    """

    declared = config.declared_parameters
    if declared is None or not config.experts:
        return declared
    router = config.d_model * config.experts
    extra_biases = (config.expert_top_k - 1) * config.d_model
    return declared + config.layers * (router + extra_biases)


def routed_mlp_local(
    x: jax.Array,
    router_w: jax.Array,
    up_w: jax.Array,
    up_b: jax.Array,
    down_w: jax.Array,
    down_b: jax.Array,
    *,
    experts: int,
    top_k: int,
    dtype: Any,
    axis_name: str | None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Top-k routed experts via a grouped matmul, on one device's tokens.

    Returns the block output, the mean router probability per expert, and the
    fraction of assignments each expert received -- the two statistics the
    balance loss is built from. Both are averaged across the data axis when one
    is given, so the loss sees global load rather than one shard's view.

    Dropless by construction: ``group_sizes`` is data while the total
    ``tokens * top_k`` is static, so every assignment is served and no capacity
    factor exists. Nothing couples one token's routing to another's, which is
    what keeps the model causal.
    """

    batch, length, width = x.shape
    flat = x.reshape(batch * length, width)

    logits = jnp.einsum(
        "md,de->me",
        flat,
        router_w.astype(jnp.float32),
        preferred_element_type=jnp.float32,
    )
    probabilities = jax.nn.softmax(logits, axis=-1)
    chosen_logits, chosen = jax.lax.top_k(logits, top_k)
    gate = jax.nn.softmax(chosen_logits, axis=-1)

    assignments = chosen.reshape(-1)
    order = jnp.argsort(assignments, stable=True)
    sorted_assignments = assignments[order]
    counts = jax.nn.one_hot(assignments, experts, dtype=jnp.int32).sum(0)
    rows = jnp.repeat(jnp.arange(batch * length), top_k)[order]

    # Pallas lowers only in interpret mode off TPU, which is how the routed
    # path stays checkable against a dense reference on CPU.
    interpret = jax.default_backend() != "tpu"
    grouped = flat[rows].astype(dtype)
    hidden = megablox.gmm(grouped, up_w.astype(dtype), counts, interpret=interpret)
    hidden = hidden + up_b[sorted_assignments].astype(hidden.dtype)
    hidden = jax.nn.gelu(hidden, approximate=True)
    out = megablox.gmm(
        hidden.astype(dtype), down_w.astype(dtype), counts, interpret=interpret
    )
    out = out + down_b[sorted_assignments].astype(out.dtype)

    weighted = out * gate.reshape(-1)[order][:, None].astype(out.dtype)
    combined = jnp.zeros((batch * length, width), out.dtype).at[rows].add(weighted)

    mean_probability = probabilities.mean(axis=0)
    load = counts.astype(jnp.float32) / jnp.float32(batch * length * top_k)
    # Standard routing diagnostics, all reduced to one number per layer. Every
    # one is a by-product of tensors this function already has, so the cost is
    # a handful of reductions rather than a second pass.
    entropy = -jnp.sum(probabilities * jnp.log(probabilities + 1.0e-9), axis=-1).mean()
    top1_gate = gate.max(axis=-1).mean()
    # The mean square crosses the collective, not the root of it: averaging
    # per-device roots is not the root of the global average, which would make
    # this number depend on how many devices the run happened to use.
    logit_mean_square = jnp.mean(jnp.square(logits))
    summary = jnp.stack((entropy, top1_gate, logit_mean_square))
    if axis_name is not None:
        mean_probability = jax.lax.pmean(mean_probability, axis_name)
        load = jax.lax.pmean(load, axis_name)
        summary = jax.lax.pmean(summary, axis_name)
    summary = summary.at[2].set(jnp.sqrt(summary[2]))
    return combined.reshape(batch, length, width), mean_probability, load, summary


def make_mesh_routed_mlp(config: Config, mesh: Mesh) -> Any:
    """Wrap the routed MLP in an explicit data-sharded boundary.

    The grouped matmul is a Mosaic kernel, so an outer jit cannot partition it
    automatically -- the same constraint that forces make_mesh_attention to
    exist. Experts stay replicated (plan phase 1): each device routes its own
    tokens among all of them, so there are no expert collectives, only the two
    tiny mean-reductions that give the balance loss a global view.
    """

    if not config.experts:
        return None
    batch_partition = P("data", None, None)
    replicated = P()
    local = functools.partial(
        routed_mlp_local,
        experts=config.experts,
        top_k=config.expert_top_k,
        dtype=config.compute_dtype,
        axis_name="data",
    )
    return jax.shard_map(
        local,
        mesh=mesh,
        in_specs=(
            batch_partition,
            replicated,
            replicated,
            replicated,
            replicated,
            replicated,
        ),
        out_specs=(batch_partition, replicated, replicated, replicated),
        check_vma=False,
    )


def load_balance_loss(mean_probability: jax.Array, load: jax.Array) -> jax.Array:
    """Switch-style auxiliary loss: E * sum_i (f_i * P_i).

    Both arguments are per-expert vectors of length E: ``load`` is the realized
    fraction of assignments an expert received and ``mean_probability`` the
    router's mean probability for it, each already averaged over every token on
    every device. Minimized at 1.0 when both are uniform, and E when one expert
    takes everything.

    This *encourages* balance; it never enforces it, because any rule that
    equalized loads would have to couple one token's routing to another's and
    break causality.

    Both must arrive already reduced over tokens. Passing the raw ``[tokens, E]``
    probability matrix here would average it to a scalar, which makes the whole
    term collapse to the constant 1.0 with no gradient to the router at all --
    a failure that trains happily and simply never balances.
    """

    if mean_probability.ndim != 1 or load.ndim != 1:
        raise ValueError(
            "load_balance_loss takes per-expert vectors already reduced over "
            f"tokens, got shapes {mean_probability.shape} and {load.shape}"
        )
    return jnp.float32(load.shape[0]) * jnp.sum(load * mean_probability)


def gpt_hidden(
    params: Mapping[str, Any],
    tokens: jax.Array,
    config: Config,
    attention_fn: AttentionCallable | None = None,
    routed_fn: Any = None,
) -> tuple[jax.Array, "RouterStats | None"]:
    """Return final token representations, and router statistics when routed."""

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
    router_losses: list[jax.Array] = []
    router_loads: list[jax.Array] = []
    router_summaries: list[jax.Array] = []

    for block in params["blocks"]:
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
        if config.experts:
            routed = routed_fn or functools.partial(
                routed_mlp_local,
                experts=config.experts,
                top_k=config.expert_top_k,
                dtype=dtype,
                axis_name=None,
            )
            mlp_out, mean_probability, load, summary = routed(
                x_norm,
                block["router_w"],
                block["expert_up_w"],
                block["expert_up_b"],
                block["expert_down_w"],
                block["expert_down_b"],
            )
            router_losses.append(load_balance_loss(mean_probability, load))
            router_loads.append(load)
            router_summaries.append(summary)
        else:
            hidden = linear(x_norm, block["mlp_up_w"], block["mlp_up_b"], dtype)
            hidden = jax.nn.gelu(hidden, approximate=True)
            mlp_out = linear(hidden, block["mlp_down_w"], block["mlp_down_b"], dtype)
        x = residual + (config.depth_multiplier ** (-config.depth_alpha)) * mlp_out

    hidden_state = rms_norm(x, params["final_ln_scale"], dtype)
    router = (
        RouterStats(
            balance_loss=jnp.mean(jnp.stack(router_losses)),
            load=jnp.stack(router_loads),
            summary=jnp.stack(router_summaries),
        )
        if config.experts
        else None
    )
    return hidden_state, router


def gpt_logits(
    params: Mapping[str, Any],
    tokens: jax.Array,
    config: Config,
    attention_fn: AttentionCallable | None = None,
    routed_fn: Any = None,
) -> jax.Array:
    return gpt_logits_and_router(params, tokens, config, attention_fn, routed_fn)[0]


def gpt_logits_and_router(
    params: Mapping[str, Any],
    tokens: jax.Array,
    config: Config,
    attention_fn: AttentionCallable | None = None,
    routed_fn: Any = None,
) -> tuple[jax.Array, "RouterStats | None"]:
    """Logits, and the router statistics the balance loss needs.

    Exists because discarding the statistics here is invisible: the model still
    trains, the loss still falls, and the auxiliary term is simply never
    applied. The dense loss backend did exactly that.
    """

    x, router = gpt_hidden(params, tokens, config, attention_fn, routed_fn)
    output_embedding = params.get("output_embedding", params["token_embedding"])
    logits = jnp.einsum(
        "btd,vd->btv",
        x,
        output_embedding.astype(config.compute_dtype),
    ).astype(jnp.float32)
    return logits, router


def cross_entropy(
    params: Mapping[str, Any],
    x: jax.Array,
    y: jax.Array,
    config: Config,
    attention_fn: AttentionCallable | None = None,
    routed_fn: Any = None,
) -> jax.Array:
    """Training objective. For routed models this includes the balance loss.

    The auxiliary term is deliberately *not* part of any reported loss: a run
    that balances well and models badly must not look like the reverse, so
    ``router.load_balance_loss`` is logged as its own metric.
    """

    if config.loss_backend == "tiled":
        hidden, router = gpt_hidden(params, x, config, attention_fn, routed_fn)
        loss = tiled_tied_cross_entropy(
            hidden,
            params.get("output_embedding", params["token_embedding"]),
            y,
            semantic_vocab_size=config.semantic_vocab_size,
            vocab_tile_size=config.vocab_tile_size,
            compute_dtype=config.compute_dtype,
        )
    else:
        logits, router = gpt_logits_and_router(
            params, x, config, attention_fn, routed_fn
        )
        logits = logits[..., : config.semantic_vocab_size]
        log_probabilities = jax.nn.log_softmax(logits, axis=-1)
        selected = jnp.take_along_axis(log_probabilities, y[..., None], axis=-1)
        loss = -jnp.mean(selected, dtype=jnp.float32)
    if router is not None and config.router_aux_coefficient:
        loss = loss + config.router_aux_coefficient * router.balance_loss
    return loss


def cross_entropy_and_router(
    params: Mapping[str, Any],
    x: jax.Array,
    y: jax.Array,
    config: Config,
    attention_fn: AttentionCallable | None = None,
    routed_fn: Any = None,
) -> tuple[jax.Array, "RouterStats | None"]:
    """The training objective, plus the balance loss and per-layer expert load.

    Carried out of the update as ``value_and_grad`` aux so logging costs an
    already-computed array rather than a second forward pass. Dense models
    return ``None`` and log nothing.
    """

    if not config.experts:
        return cross_entropy(params, x, y, config, attention_fn, routed_fn), None

    hidden, router = gpt_hidden(params, x, config, attention_fn, routed_fn)
    if config.loss_backend == "tiled":
        loss = tiled_tied_cross_entropy(
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
        loss = -jnp.mean(selected, dtype=jnp.float32)

    assert router is not None
    if config.router_aux_coefficient:
        loss = loss + config.router_aux_coefficient * router.balance_loss
    return loss, router


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


def init_optimizer(params: Any, config: Config) -> dict[str, Any]:
    zeros = jax.tree_util.tree_map(lambda value: np.zeros_like(value), params)
    # Keeping the small scalar history on-device avoids a host synchronization
    # on every step. It is copied once, after the synchronized timing boundary.
    # A routed run widens it by the routing columns so those are recorded every
    # step too -- the end-of-run rewrite supersedes the sampled rows, so
    # anything absent here is discarded no matter how often it was appended.
    #
    # The width is derived from config rather than taken as a parameter so
    # there is exactly one place that can get it wrong: a caller that forgot
    # to pass router_row_width(config) here once shipped a run whose history
    # buffer was too narrow for the row train_step tried to write into it,
    # and the run never reached the first optimizer step.
    history = np.zeros((config.steps, 3 + router_row_width(config)), dtype=np.float32)
    return {
        "step": np.asarray(0, dtype=np.int32),
        "m": zeros,
        "v": zeros,
        "history": history,
    }


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
        if name in {"token_embedding", "output_embedding"} or name.endswith("_w"):
            return True
        if name.endswith(("_b", "_bias", "_scale")):
            return False
        raise ValueError(f"weight-decay policy has no rule for parameter {name!r}")

    return jax.tree_util.tree_map_with_path(decay_for_path, params)


def expert_load_scale_factors(
    load: jax.Array,
    *,
    experts: int,
    strength: float,
) -> jax.Array:
    """Return one multiplier per layer and expert from current routed load.

    ``load`` is the global hard-assignment distribution already emitted by the
    router, so its balanced reference is ``1 / experts``.  The full rule is
    ``sqrt(load / reference)``.  ``strength`` linearly interpolates from one to
    that rule, leaving a floor of ``1 - strength`` for an unused expert.
    """

    if load.ndim != 2 or load.shape[1] != experts:
        raise ValueError(
            "expert load must have shape [layers, experts]; "
            f"got {load.shape} for {experts} experts"
        )
    if not 0.0 <= strength <= 1.0:
        raise ValueError("expert-load scaling strength must be between 0 and 1")
    load = jax.lax.stop_gradient(load.astype(jnp.float32))
    full_scale = jnp.sqrt(jnp.maximum(load * float(experts), 0.0))
    return 1.0 + jnp.asarray(strength, jnp.float32) * (full_scale - 1.0)


def expert_load_scale_tree(params: Mapping[str, Any], factors: jax.Array) -> Any:
    """Broadcast ``[layer, expert]`` factors over expert parameter leaves only."""

    blocks = params.get("blocks")
    if blocks is None:
        raise ValueError("parameter tree has no blocks")
    if factors.ndim != 2 or factors.shape[0] != len(blocks):
        raise ValueError(
            "expert factors must have one row per parameter block; "
            f"got {factors.shape} for {len(blocks)} blocks"
        )

    block_trees: list[dict[str, Any]] = []
    for layer, block in enumerate(blocks):
        block_tree: dict[str, Any] = {}
        for name, value in block.items():
            if name.startswith("expert_") and name not in _EXPERT_PARAMETER_NAMES:
                raise ValueError(
                    f"expert-load scaling has no rule for parameter {name!r}"
                )
            if name in _EXPERT_PARAMETER_NAMES:
                if value.ndim < 1 or value.shape[0] != factors.shape[1]:
                    raise ValueError(
                        f"{name} must have experts on its leading axis; "
                        f"got {value.shape} for {factors.shape[1]} factors"
                    )
                broadcast_shape = (factors.shape[1],) + (1,) * (value.ndim - 1)
                block_tree[name] = factors[layer].reshape(broadcast_shape)
            else:
                block_tree[name] = jnp.asarray(1.0, jnp.float32)
        block_trees.append(block_tree)

    return {
        name: block_trees if name == "blocks" else jnp.asarray(1.0, jnp.float32)
        for name in params
    }


def apply_expert_load_scaling(tree: Any, scale_tree: Any) -> Any:
    """Multiply only expert slices selected by ``expert_load_scale_tree``."""

    return jax.tree_util.tree_map(
        lambda value, scale: value * scale,
        tree,
        scale_tree,
    )


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


def _apply_training_update(
    params: Any,
    optimizer: Mapping[str, Any],
    x: jax.Array,
    y: jax.Array,
    config: Config,
    decay_mask: Any | None = None,
    attention_fn: AttentionCallable | None = None,
    routed_fn: Any = None,
) -> tuple[Any, dict[str, Any], dict[str, jax.Array], Any]:
    """Apply one ordinary update and also return the raw, pre-clip gradient.

    Both the ordinary and sparse-diagnostic executables use this exact function.
    Diagnostics therefore do not substitute a different optimizer formula.
    """

    if decay_mask is None:
        decay_mask = weight_decay_mask(params)
    lr_multipliers, epsilon_multipliers, decay_multipliers = (
        optimizer_hyperparameter_trees(params, config)
    )
    beta1, beta2 = effective_adam_betas(config)
    (loss, router_aux), gradients = jax.value_and_grad(
        lambda candidate: cross_entropy_and_router(
            candidate, x, y, config, attention_fn, routed_fn
        ),
        has_aux=True,
    )(params)
    gradients = jax.tree_util.tree_map(lambda grad: grad.astype(jnp.float32), gradients)
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

    expert_scales = None
    if config.expert_load_scaling_strength > 0.0:
        if router_aux is None:
            raise ValueError("expert-load scaling requires routed model statistics")
        factors = expert_load_scale_factors(
            router_aux.load,
            experts=config.experts,
            strength=config.expert_load_scaling_strength,
        )
        expert_scales = expert_load_scale_tree(params, factors)
        if config.expert_load_scaling_mode == "gradient":
            # Preserve diagnostics and global clipping as properties of the
            # objective gradient. This intervention changes only what Adam sees.
            gradients = apply_expert_load_scaling(gradients, expert_scales)
        elif config.expert_load_scaling_mode != "update":
            raise ValueError(
                "expert-load scaling mode must be 'gradient' or 'update'"
            )

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
        expert_scale: Any | None = None,
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
        if expert_scale is None:
            return parameter - lr * lr_multiplier * (adam + decay)
        return parameter - lr * lr_multiplier * expert_scale * (adam + decay)

    update_trees = (
        (
            params,
            m,
            v,
            decay_mask,
            lr_multipliers,
            epsilon_multipliers,
            decay_multipliers,
            expert_scales,
        )
        if config.expert_load_scaling_mode == "update" and expert_scales is not None
        else (
            params,
            m,
            v,
            decay_mask,
            lr_multipliers,
            epsilon_multipliers,
            decay_multipliers,
        )
    )
    params = jax.tree_util.tree_map(update, *update_trees)
    routing = router_row(router_aux)
    history_row = jnp.concatenate(
        (jnp.stack((loss, lr, grad_norm)).astype(jnp.float32), routing)
    )
    history = optimizer["history"].at[step - 1].set(history_row)
    return (
        params,
        {"step": step, "m": m, "v": v, "history": history},
        {
            "loss": loss,
            "grad_norm": grad_norm,
            "learning_rate": lr,
            # Empty for a dense model. The shape is static either way, so
            # this does not vary the executable.
            "router_row": routing,
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
    routed_fn: Any = None,
) -> tuple[Any, dict[str, Any], dict[str, jax.Array]]:
    params, optimizer, metrics, _ = _apply_training_update(
        params, optimizer, x, y, config, decay_mask, attention_fn, routed_fn
    )
    return params, optimizer, metrics


def router_row(router: "RouterStats | None") -> jax.Array:
    """Flatten the router statistics into the order training_log_columns names.

    Built on device and stored in the optimizer history, so every step is
    recorded rather than only the sampled ones, and so the live rows and the
    authoritative end-of-run rewrite are necessarily the same numbers in the
    same order. Empty for a dense model.
    """

    if router is None:
        return jnp.zeros((0,), jnp.float32)
    load, summary = router.load, router.summary
    per_layer = jnp.concatenate(
        [
            jnp.concatenate((summary[layer], load[layer]))
            for layer in range(load.shape[0])
        ]
    )
    return jnp.concatenate(
        (
            jnp.stack((router.balance_loss, load.max(), load.min())),
            summary.mean(axis=0),
            # Per layer: the three summary statistics, then the whole load
            # vector. Per-expert load is the exact distribution, so max, min,
            # and any histogram of it are derivable and none are stored.
            per_layer,
        )
    ).astype(jnp.float32)


def router_row_width(config: Config) -> int:
    """How many columns router_row emits, for sizing the history buffer."""

    if not config.experts:
        return 0
    return 6 + config.layers * (len(ROUTER_SUMMARY_METRICS) + config.experts)


def diagnostic_train_step(
    params: Any,
    optimizer: Mapping[str, Any],
    x: jax.Array,
    y: jax.Array,
    config: Config,
    decay_mask: Any | None = None,
    attention_fn: AttentionCallable | None = None,
    routed_fn: Any = None,
) -> tuple[Any, dict[str, Any], dict[str, jax.Array], jax.Array]:
    """Run the same update as :func:`train_step` and emit sparse statistics."""

    params_before = params
    params, optimizer, metrics, raw_gradients = _apply_training_update(
        params, optimizer, x, y, config, decay_mask, attention_fn, routed_fn
    )
    values = diagnostic_values(
        params_before,
        raw_gradients,
        params,
        include_experts=True,
        statistics=DIAGNOSTIC_EXTENDED_STATS,
    )
    return params, optimizer, metrics, values


def eval_step(
    params: Any,
    x: jax.Array,
    y: jax.Array,
    mask: jax.Array,
    config: Config,
    attention_fn: AttentionCallable | None = None,
    routed_fn: Any = None,
) -> tuple[jax.Array, jax.Array]:
    """Return a loss sum and exact target count for fixed-shape masked eval."""

    if config.loss_backend == "tiled":
        hidden, _ = gpt_hidden(params, x, config, attention_fn, routed_fn)
        losses = tiled_tied_cross_entropy_losses(
            hidden,
            params.get("output_embedding", params["token_embedding"]),
            y,
            semantic_vocab_size=config.semantic_vocab_size,
            vocab_tile_size=config.vocab_tile_size,
            compute_dtype=config.compute_dtype,
        )
    else:
        logits = gpt_logits(params, x, config, attention_fn, routed_fn)[
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

    def loss(trainable: Mapping[str, Any]) -> jax.Array:
        return cross_entropy(trainable, tokens, targets, config)

    return count_training_flops(loss, params, rules=default_rules())


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
    host_optimizer = init_optimizer(host_params, config)
    decay_mask = weight_decay_mask(host_params)
    diagnostic_metadata = diagnostic_scope_metadata(
        host_params, include_experts=True
    )
    params_total = parameter_count(host_params)
    params_active = active_parameter_count(host_params, config)
    expected_active = expected_active_parameters(config)
    if expected_active is not None and params_active != expected_active:
        raise ValueError(
            f"tier {config.tier} should have {expected_active:,} active "
            f"parameters, but initialized {params_active:,} "
            f"(total {params_total:,})"
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
            *model_console_rows(config, params_total, params_active),
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
    routed_fn = make_mesh_routed_mlp(config, mesh)
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
            p, o, x, y, config, decay_mask, attention_fn, routed_fn
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
                    p, o, x, y, config, decay_mask, attention_fn, routed_fn
                ),
                in_shardings=(replicated, replicated, data_sharding, data_sharding),
                donate_argnums=(0, 1),
            )
            .lower(params, optimizer, sample_x, sample_y)
            .compile()
        )
        diagnostic_compile_seconds = time.perf_counter() - diagnostic_compile_started

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
                    p, x, y, mask, config, attention_fn, routed_fn
                ),
                in_shardings=(replicated, data_sharding, data_sharding, data_sharding),
            )
            .lower(params, sample_x, sample_y, sample_mask)
            .compile()
        )
        eval_compile_seconds = time.perf_counter() - eval_compile_started
    total_compile_seconds = (
        train_compile_seconds + diagnostic_compile_seconds + eval_compile_seconds
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
    training_columns = training_log_columns(
        config.layers if config.experts else 0, config.experts
    )
    progress_log: logpack.LogWriter | None = None
    diagnostic_log: logpack.LogWriter | None = None
    if is_controller:
        output_dir.mkdir(parents=True, exist_ok=True)
        # A stale file from a reused directory would be appended to.
        (output_dir / TRAINING_LOG_NAME).unlink(missing_ok=True)
        (output_dir / DIAGNOSTICS_LOG_NAME).unlink(missing_ok=True)
        progress_log = open_log(
            output_dir / TRAINING_LOG_NAME,
            training_columns,
            tokens_per_step=config.batch_size * config.seq_len,
            flops_per_token=flops_per_token,
        )
        diagnostic_log = open_log(
            output_dir / DIAGNOSTICS_LOG_NAME,
            diagnostic_log_columns(
                diagnostic_metadata,
                statistics=DIAGNOSTIC_EXTENDED_STATS,
            ),
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
                else:
                    params, optimizer, last_metrics = executable(
                        params, optimizer, batch_x, batch_y
                    )
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
                        (
                            float(host_metrics["loss"]),
                            float(host_metrics["learning_rate"]),
                            float(host_metrics["grad_norm"]),
                            *(float(v) for v in host_metrics["router_row"]),
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
    train_loss = finite_metric("train_loss", float(final_train["loss"]))

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
                    statistics=DIAGNOSTIC_EXTENDED_STATS,
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
                statistics=DIAGNOSTIC_EXTENDED_STATS,
            )
        write_validation_csv(output_dir, validation_rows)
        if not args.omit_checkpoint:
            save_checkpoint(
                output_dir,
                params,
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
        },
        "system": {
            **system_metadata(devices),
            "controller_process_index": process_index,
        },
        "contract": {
            "model_id": "reference-gpt-v3-family",
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
            "parameter_count": int(params_active),
            "total_parameter_count": int(params_total),
            "experts": int(config.experts),
            "expert_top_k": int(config.expert_top_k),
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
            "validation_probe_seconds": finite_metric(
                "validation_probe_seconds", validation_probe_seconds
            ),
            "final_validation_seconds": finite_metric(
                "final_validation_seconds", final_validation_seconds, positive=True
            ),
            "train_loss": train_loss,
            "parameters": int(params_total),
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
