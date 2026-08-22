from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, replace
import csv
import hashlib
import importlib.util
import inspect
from io import StringIO
import json
import math
import os
import re
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from rig import attention as rig_attention
from rig import configfile, evaluation, logpack, metrics, runlog
from rig import tokens as rig_tokens
from rig.kernels import AttentionTiles


TRAINER_PATH = Path(__file__).parents[1] / "recipes" / "reference" / "train.py"
SPEC = importlib.util.spec_from_file_location("reference_train", TRAINER_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - importlib invariant
    raise RuntimeError(f"could not import {TRAINER_PATH}")
trainer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = trainer
SPEC.loader.exec_module(trainer)

_LOADED_CONFIGS = {
    profile: trainer.load_experiment_config(profile)
    for profile in ("smoke", "dev", "official")
}


def _resolve_config(
    args,
    platform: str,
    *,
    experiment_config=None,
    config_sha256: str | None = None,
):
    """Resolve through the one production API while keeping tests concise."""

    profile = trainer.selected_profile(args)
    default_config, default_sha256 = _LOADED_CONFIGS[profile]
    return trainer.resolve_config(
        args,
        platform,
        experiment_config=experiment_config or default_config,
        config_sha256=config_sha256 or default_sha256,
    )


def _validate_args(args, experiment_config=None) -> None:
    profile = trainer.selected_profile(args)
    default_config, _ = _LOADED_CONFIGS[profile]
    trainer.validate_args(args, experiment_config or default_config)


# Tracing needs real parameters, so shrink the model to keep these tests
# cheap. Head dim stays 64 so the flash tile plan resolves as it does in
# training; only depth and width move.
_SMALL = {"layers": 2, "d_model": 128, "heads": 2}


def _fake_config(**fields) -> SimpleNamespace:
    """Stand in for Config where a schedule helper only reads a few fields.

    ``final_step`` is derived exactly as the real property derives it, so these
    fixtures cannot drift into declaring a stopping step the horizon contradicts.
    """

    fields.setdefault("stop_after_step", None)
    return SimpleNamespace(
        final_step=fields["stop_after_step"] or fields.get("steps"), **fields
    )


def _traced_per_token(config) -> int:
    params = trainer.init_params(config, 1337)
    return trainer.traced_flops(config, params).per_token(config.seq_len)


def _replace_experiment_config(
    experiment_config,
    *,
    profile: str,
    tier: str | None = None,
    context_name: str | None = None,
    training: dict[str, object] | None = None,
    model: dict[str, object] | None = None,
    context: dict[str, object] | None = None,
    kernels: dict[str, object] | None = None,
    evaluation: dict[str, object] | None = None,
):
    """Apply deliberate test-only overrides to a typed experiment config."""

    family = experiment_config.family
    tier_name = tier or family.default_tier
    selected_context_name = context_name or family.default_context
    if experiment_config.execution_type != profile:
        raise ValueError(f"test config does not describe {profile!r}")
    definition = experiment_config.run
    if training:
        definition = replace(
            definition,
            training=replace(definition.training, **training),
        )
    if kernels:
        definition = replace(
            definition,
            kernels=replace(definition.kernels, **kernels),
        )
    if evaluation:
        definition = replace(
            definition,
            evaluation=replace(definition.evaluation, **evaluation),
        )
    if model:
        selected_tier = family.tiers[tier_name]
        selected_tier = replace(
            selected_tier,
            model=replace(selected_tier.model, **model),
        )
        family = replace(
            family,
            tiers={**family.tiers, tier_name: selected_tier},
        )
    if context:
        selected_context = replace(family.contexts[selected_context_name], **context)
        family = replace(
            family,
            contexts={**family.contexts, selected_context_name: selected_context},
        )
    return replace(experiment_config, family=family, run=definition)


@dataclass(frozen=True)
class FakeDevice:
    platform: str
    device_kind: str


class TrainerStaticTests(unittest.TestCase):
    def test_reference_block_is_rope_rmsnorm_gelu(self) -> None:
        parser = trainer.build_parser()
        config = _resolve_config(parser.parse_args(["--profile", "smoke"]), "cpu")
        self.assertFalse(hasattr(config, "__dict__"))
        params = trainer.init_params(config, 7)
        self.assertNotIn("position_embedding", params)
        self.assertNotIn("final_ln_bias", params)
        self.assertNotIn("ln1_bias", params["blocks"][0])
        self.assertEqual(trainer.parameter_count(params), 116_160)
        self.assertEqual(
            trainer.contract_model_metadata(config)["position_encoding"],
            "rope_base_10000",
        )
        self.assertEqual(
            trainer.contract_model_metadata(config)["normalization"], "rms_norm"
        )
        self.assertEqual(
            trainer.contract_model_metadata(config)["mlp_activation"], "gelu"
        )

        values = np.arange(24, dtype=np.float32).reshape(1, 3, 2, 4)
        rotated = np.asarray(trainer.apply_rotary(trainer.jnp.asarray(values)))
        np.testing.assert_allclose(rotated[:, 0], values[:, 0], rtol=0, atol=1e-6)
        np.testing.assert_allclose(
            np.sum(rotated * rotated, axis=-1),
            np.sum(values * values, axis=-1),
            rtol=1e-6,
            atol=1e-5,
        )

    def test_family_tiers_have_declared_size_fixed_heads_and_20_tpp(self) -> None:
        parser = trainer.build_parser()
        expected = {
            "60m": (59_918_208, 12, 384, 6),
            "125m": (123_456_640, 12, 640, 10),
            "250m": (244_444_032, 16, 896, 14),
            "500m": (502_602_240, 19, 1280, 20),
            "1b": (989_943_808, 21, 1792, 28),
        }
        experiment_config, config_sha256 = trainer.load_experiment_config("official")
        for tier, (parameters, layers, width, heads) in expected.items():
            with self.subTest(tier=tier):
                self.assertEqual(
                    experiment_config.family.tiers[tier].tpp_parameters, parameters
                )
                config = _resolve_config(
                    parser.parse_args(["--profile", "official", "--tier", tier]),
                    "tpu",
                    experiment_config=experiment_config,
                    config_sha256=config_sha256,
                )
                self.assertEqual(config.declared_parameters, parameters)
                self.assertEqual(
                    (config.layers, config.d_model, config.heads),
                    (layers, width, heads),
                )
                self.assertEqual(config.d_model // config.heads, 64)
                self.assertEqual(config.attention_scale, "inverse_head_dim")
                self.assertAlmostEqual(
                    trainer.attention_softmax_scale(
                        config.attention_scale, config.d_model // config.heads
                    ),
                    1.0 / 64.0,
                )
                self.assertAlmostEqual(config.tokens_per_parameter, 20.0, places=3)
                self.assertEqual(
                    config.steps * config.batch_size * config.seq_len,
                    round(parameters * config.tokens_per_parameter),
                )

    def test_fixed_tpp_completep_hybrid_tensor_and_ladder_multipliers(self) -> None:
        parser = trainer.build_parser()
        args = parser.parse_args(
            ["--profile", "dev", "--tier", "250m", "--tokens-per-parameter", "5"]
        )
        config = _resolve_config(args, "tpu")
        for derived_name in (
            "width_multiplier",
            "depth_multiplier",
            "tokens_per_parameter",
            "compute_dtype",
        ):
            self.assertNotIn(derived_name, trainer.Config.__slots__)
        params = {
            "token_embedding": np.zeros((1, 1), dtype=np.float32),
            "blocks": [
                {
                    "qkv_w": np.zeros((1, 1), dtype=np.float32),
                    "qkv_b": np.zeros((1,), dtype=np.float32),
                    "ln1_scale": np.ones((1,), dtype=np.float32),
                }
            ],
            "final_ln_scale": np.ones((1,), dtype=np.float32),
            "output_embedding": np.zeros((1, 1), dtype=np.float32),
        }
        lr, epsilon, decay = trainer.optimizer_hyperparameter_trees(params, config)
        width = 896 / 384
        depth = 16 / 12
        self.assertEqual(config.width_multiplier, width)
        self.assertEqual(config.depth_multiplier, depth)
        self.assertEqual(config.compute_dtype, trainer.jnp.bfloat16)
        self.assertAlmostEqual(lr["token_embedding"], 1.0)
        self.assertAlmostEqual(epsilon["token_embedding"], width**-1)
        self.assertAlmostEqual(lr["blocks"][0]["qkv_w"], width**-1)
        self.assertAlmostEqual(lr["blocks"][0]["qkv_b"], 1.0)
        self.assertAlmostEqual(epsilon["blocks"][0]["qkv_w"], width**-1 * depth**-1)
        self.assertAlmostEqual(decay["blocks"][0]["qkv_w"], width)
        self.assertAlmostEqual(lr["output_embedding"], width**-1)
        self.assertAlmostEqual(epsilon["output_embedding"], 1.0)
        self.assertAlmostEqual(decay["output_embedding"], width)
        self.assertAlmostEqual(config.data_multiplier, 244_444_032 / 59_918_208)
        effective = trainer.effective_optimizer_metadata(config)
        self.assertAlmostEqual(
            effective["global_peak_learning_rate"],
            config.learning_rate / math.sqrt(config.data_multiplier),
        )

        twenty_tpp = _resolve_config(
            parser.parse_args(
                [
                    "--profile",
                    "dev",
                    "--tier",
                    "250m",
                    "--tokens-per-parameter",
                    "20",
                ]
            ),
            "tpu",
        )
        # Each TPP ladder is reanchored. m_D captures model-size growth within
        # the ladder and therefore does not gain a 20/5 horizon factor.
        self.assertEqual(twenty_tpp.data_multiplier, config.data_multiplier)

    def test_yaml_config_is_authoritative_strict_and_versioned(self) -> None:
        config_path = trainer.experiment_config_path("official")
        source = config_path.read_text(encoding="utf-8")
        experiment_config, config_sha256 = trainer.load_experiment_config("official")
        tier_name, context_name = experiment_config.resolve_selection()
        official = experiment_config.run
        context = experiment_config.family.contexts[context_name]
        model = experiment_config.family.tiers[tier_name].model
        self.assertEqual(experiment_config.schema_version, 6)
        self.assertEqual(experiment_config.execution_type, "official")
        self.assertEqual(official.training.duration.tokens_per_parameter, 20.0)
        self.assertEqual(context_name, "1k")
        self.assertEqual(context.reference_batch_size, 128)
        self.assertEqual(context.seq_len, 1024)
        self.assertFalse(context.document_masking)
        self.assertEqual(official.kernels.attention_backend, "tpu_flash")
        self.assertEqual(official.training.sampling, "shuffled_epochs")
        self.assertEqual(official.training.dtype, "bfloat16")
        self.assertFalse(hasattr(experiment_config, "__dict__"))
        self.assertFalse(hasattr(model, "__dict__"))
        self.assertNotIn("\n      parameters:", source)
        self.assertEqual(
            config_sha256,
            hashlib.sha256(config_path.read_bytes()).hexdigest(),
        )

        expected_files = {
            "smoke": "smoke.yaml",
            "dev": "dev.yaml",
            "official": "config.yaml",
        }
        for profile, filename in expected_files.items():
            with self.subTest(profile=profile):
                selected, _ = trainer.load_experiment_config(profile)
                self.assertEqual(trainer.experiment_config_path(profile).name, filename)
                self.assertEqual(selected.execution_type, profile)
        dev, _ = trainer.load_experiment_config("dev")
        self.assertEqual(dev.family, experiment_config.family)
        self.assertEqual(dev.run.optimizer, official.optimizer)
        self.assertEqual(dev.run.kernels, official.kernels)

        invalid = {
            "duplicate": source.replace(
                "schema_version: 6", "schema_version: 6\nschema_version: 6", 1
            ),
            "unknown": source + "\nunknown: true\n",
            "route mismatch": source.replace(
                "execution_type: official", "execution_type: dev", 1
            ),
            "anchor": source.replace("schema_version: 6", "schema_version: &v 6", 1),
            "alias": source.replace(
                "schema_version: 6", "schema_version: &v 6\nextra: *v", 1
            ),
            "tag": source.replace("schema_version: 6", "schema_version: !!int 6", 1),
            "directive": "%YAML 1.2\n---\n" + source,
            "multiple documents": source + "\n---\n{}\n",
            "nonfinite": source.replace(
                "learning_rate: 0.00390625", "learning_rate: .nan", 1
            ),
            "official validation prefix": source.replace(
                "final_predictions: 10485760",
                "final_predictions: 10485759",
                1,
            ),
            "missing sampling": source.replace(
                "    sampling: shuffled_epochs\n", "", 1
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for label, contents in invalid.items():
                with self.subTest(label=label):
                    path = root / "config.yaml"
                    path.write_text(contents, encoding="utf-8")
                    with (
                        patch.object(
                            trainer, "experiment_config_path", return_value=path
                        ),
                        self.assertRaises(ValueError),
                    ):
                        trainer.load_experiment_config("official")
            path = root / "config.yaml"
            path.write_bytes(b"#" * (configfile.MAX_CONFIG_BYTES + 1))
            with (
                patch.object(trainer, "experiment_config_path", return_value=path),
                self.assertRaisesRegex(ValueError, "safety limit"),
            ):
                trainer.load_experiment_config("official")

            target = root / "target.yaml"
            target.write_text(source, encoding="utf-8")
            symlink = root / "config-link.yaml"
            symlink.symlink_to(target)
            with (
                patch.object(trainer, "experiment_config_path", return_value=symlink),
                self.assertRaisesRegex(ValueError, "non-symlink"),
            ):
                trainer.load_experiment_config("official")

            shuffled_source = source.replace(
                "    sampling: shuffled_epochs\n    dtype: bfloat16",
                "    sampling: random_windows\n    dtype: bfloat16",
                1,
            )
            path.write_text(shuffled_source, encoding="utf-8")
            with patch.object(trainer, "experiment_config_path", return_value=path):
                shuffled, _ = trainer.load_experiment_config("official")
                self.assertEqual(shuffled.run.training.sampling, "random_windows")

        _, long_context_name = experiment_config.resolve_selection(context="8k")
        long_context = experiment_config.family.contexts[long_context_name]
        self.assertEqual(long_context_name, "8k")
        self.assertEqual(long_context.reference_batch_size, 16)
        self.assertEqual(long_context.seq_len, 8192)
        self.assertTrue(long_context.document_masking)
        with self.assertRaisesRegex(ValueError, "unknown context preset"):
            experiment_config.resolve_selection(context="not-defined")

    def test_static_cli_values_are_rejected_but_diagnostic_overrides_resolve(
        self,
    ) -> None:
        with patch.dict(os.environ, {"RIG_TIER": "500m"}):
            parser = trainer.build_parser()
        self.assertIsNone(parser.parse_args([]).tier)
        for option in (
            ("--layers", "13"),
            ("--attention-backend", "dense"),
            ("--learning-rate", "0.001"),
            ("--eval-batches", "1"),
            ("--data-path", "data"),
            ("--val-fraction", "0.1"),
            ("--downstream-data", "science=data.bin"),
        ):
            with self.subTest(option=option), redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit):
                    parser.parse_args(["--profile", "official", *option])
        config = _resolve_config(
            parser.parse_args(
                [
                    "--profile",
                    "official",
                    "--stop-after-step",
                    "100",
                    "--diagnostic-mode",
                ]
            ),
            "tpu",
        )
        self.assertEqual(config.steps, 18_838)
        self.assertEqual(config.final_step, 100)
        self.assertEqual(config.val_every, 0)
        self.assertEqual(config.diagnostics_every, 0)
        self.assertEqual(config.log_every, config.steps)
        self.assertFalse(hasattr(config, "config_overrides"))
        context_config = _resolve_config(
            parser.parse_args(["--profile", "dev", "--context", "8k"]),
            "tpu",
        )
        self.assertEqual(context_config.context_preset, "8k")
        self.assertEqual(context_config.context_preset, "8k")

    def test_fixed_tpp_derives_complete_steps_and_rejects_custom_horizons(self) -> None:
        parser = trainer.build_parser()
        config = _resolve_config(
            parser.parse_args(
                [
                    "--profile",
                    "dev",
                    "--tokens-per-parameter",
                    "5",
                ]
            ),
            "tpu",
        )
        expected = round(
            config.declared_parameters * 5 / (config.batch_size * config.seq_len)
        )
        self.assertEqual(config.steps, expected)
        self.assertAlmostEqual(config.target_tokens_per_parameter, 5.0)
        for removed in ("--steps", "--train-tokens"):
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                parser.parse_args(["--profile", "dev", removed, "20"])

    def test_short_diagnostic_run_resolves_fractional_warmup(self) -> None:
        parser = trainer.build_parser()
        full = _resolve_config(parser.parse_args(["--profile", "official"]), "tpu")
        config = _resolve_config(
            parser.parse_args(["--profile", "official", "--stop-after-step", "100"]),
            "tpu",
        )
        self.assertEqual(config.steps, full.steps)
        self.assertEqual(config.warmup_steps, full.warmup_steps)
        self.assertEqual(config.final_step, 100)
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["--profile", "official", "--warmup-steps", "100"])

    def test_xprof_diagnostic_contract_and_capture_window(self) -> None:
        parser = trainer.build_parser()
        defaults = parser.parse_args([])
        valid = parser.parse_args(
            [
                "--xprof-dir",
                "trace",
                "--xprof-start-step",
                "11",
                "--xprof-steps",
                "20",
                "--diagnostic-mode",
            ]
        )
        _validate_args(valid)

        self.assertEqual(trainer.xprof_step_window(valid, 100), (11, 30))
        self.assertFalse(
            trainer.should_compile_evaluation(valid, SimpleNamespace(val_every=0), ())
        )

        normal = parser.parse_args([])
        _validate_args(normal)
        self.assertIsNone(trainer.xprof_step_window(normal, 100))
        self.assertTrue(
            trainer.should_compile_evaluation(normal, SimpleNamespace(val_every=0), ())
        )

        invalid_commands = (
            ["--xprof-start-step", "1"],
            ["--xprof-dir", "trace", "--xprof-start-step", "1"],
            ["--diagnostic-mode"],
            [
                "--xprof-dir",
                "trace",
                "--xprof-start-step",
                "1",
                "--xprof-steps",
                "1",
                "--diagnostic-mode",
                "--omit-checkpoint",
            ],
            [
                "--xprof-dir",
                "trace",
                "--xprof-start-step",
                "1",
                "--xprof-steps",
                "1",
                "--diagnostic-mode",
                "--downstream-manifest",
                "fresh10.json",
            ],
        )
        for command in invalid_commands:
            with self.subTest(command=command), self.assertRaises(ValueError):
                _validate_args(parser.parse_args(command))
        with self.assertRaisesRegex(ValueError, "must fit inside"):
            trainer.xprof_step_window(valid, 25)

        options = trainer.profiler_options("tpu", 4)
        self.assertEqual(options.python_tracer_level, 0)
        self.assertEqual(options.host_tracer_level, 2)
        self.assertEqual(
            options.advanced_configuration,
            {
                "tpu_trace_mode": "TRACE_COMPUTE_AND_SYNC",
                "tpu_num_chips_to_profile_per_task": 4,
            },
        )

    def test_checkpoint_omission_is_restricted_to_development_research(self) -> None:
        parser = trainer.build_parser()
        valid = parser.parse_args(["--profile", "dev", "--omit-checkpoint"])
        _validate_args(valid)
        invalid_commands = (
            ["--profile", "official", "--omit-checkpoint"],
            [
                "--profile",
                "dev",
                "--omit-checkpoint",
                "--diagnostic-mode",
                "--xprof-dir",
                "trace",
                "--xprof-start-step",
                "1",
                "--xprof-steps",
                "1",
            ],
        )
        for command in invalid_commands:
            with self.subTest(command=command), self.assertRaises(ValueError):
                _validate_args(parser.parse_args(command))

    def test_multihost_xprof_capture_is_owned_by_controller(self) -> None:
        source = TRAINER_PATH.read_text(encoding="utf-8")
        start_trace = source.index("jax.profiler.start_trace(")
        controller_guard = source.rfind("if is_controller:", 0, start_trace)
        self.assertGreater(controller_guard, 0)
        self.assertLess(start_trace - controller_guard, 1_000)
        self.assertIn("rig-xprof-capture-started", source)
        self.assertIn("rig-xprof-capture-finished", source)

    def test_diagnostic_main_omits_competition_result(self) -> None:
        stdout = StringIO()
        with (
            patch.object(trainer, "run", return_value=None),
            redirect_stdout(stdout),
        ):
            self.assertEqual(trainer.main([]), 0)
        self.assertEqual(stdout.getvalue(), "")

    def test_periodic_validation_defaults_and_diagnostic_mode(self) -> None:
        parser = trainer.build_parser()

        official = _resolve_config(parser.parse_args(["--profile", "official"]), "tpu")
        self.assertEqual(official.steps, 18_838)
        self.assertEqual(
            official.steps * official.batch_size * official.seq_len,
            2_469_134_336,
        )
        self.assertEqual(official.val_every, 500)
        self.assertEqual(official.val_probe_batches, 8)
        self.assertEqual(official.eval_batches, 80)
        self.assertEqual(official.validation_predictions, 10_485_760)
        self.assertEqual(official.diagnostics_every, 500)

        smaller_batch = _resolve_config(
            parser.parse_args(["--profile", "official", "--batch-size", "64"]),
            "tpu",
        )
        self.assertEqual(smaller_batch.eval_batches, 160)
        self.assertEqual(smaller_batch.val_probe_batches, 16)
        self.assertEqual(smaller_batch.validation_predictions, 10_485_760)

        smoke = _resolve_config(parser.parse_args(["--profile", "smoke"]), "cpu")
        development = _resolve_config(parser.parse_args(["--profile", "dev"]), "tpu")
        self.assertEqual(smoke.val_every, 0)
        self.assertEqual(smoke.diagnostics_every, 0)
        self.assertEqual(development.val_every, 0)
        self.assertEqual(development.val_probe_batches, 0)
        self.assertEqual(development.eval_batches, 8)
        self.assertEqual(development.diagnostics_every, 10)

        diagnostic = _resolve_config(
            parser.parse_args(
                [
                    "--profile",
                    "official",
                    "--stop-after-step",
                    "100",
                    "--diagnostic-mode",
                ]
            ),
            "tpu",
        )
        self.assertEqual(diagnostic.steps, official.steps)
        self.assertEqual(diagnostic.final_step, 100)
        self.assertEqual(diagnostic.val_every, 0)
        self.assertEqual(diagnostic.diagnostics_every, 0)
        self.assertEqual(diagnostic.log_every, diagnostic.steps)
        self.assertEqual(diagnostic.eval_batches, 80)
        self.assertEqual(diagnostic.val_probe_batches, 8)
        for option in (
            ("--seq-len", "16384"),
            ("--val-probe-batches", "9"),
            ("--val-every", "5"),
        ):
            with self.subTest(option=option), redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit):
                    parser.parse_args(["--profile", "official", *option])

    def test_tiled_loss_resolves_semantic_vocab_and_counts_recompute_flops(
        self,
    ) -> None:
        parser = trainer.build_parser()
        experiment_config, config_sha256 = trainer.load_experiment_config("official")
        dense = _resolve_config(
            parser.parse_args(["--profile", "official"]),
            "tpu",
            experiment_config=_replace_experiment_config(
                experiment_config,
                profile="official",
                kernels={"loss_backend": "dense"},
            ),
            config_sha256=config_sha256,
        )
        tiled_experiment_config = _replace_experiment_config(
            experiment_config,
            profile="official",
            kernels={"loss_backend": "tiled"},
        )
        tiled = _resolve_config(
            parser.parse_args(["--profile", "official"]),
            "tpu",
            experiment_config=tiled_experiment_config,
            config_sha256=config_sha256,
        )
        self.assertEqual(dense.semantic_vocab_size, 50_304)
        # Switching kernels alone preserves the calibrated 50,304-class
        # objective. Masking storage-only rows is an explicit algorithm choice.
        self.assertEqual(tiled.semantic_vocab_size, 50_304)
        self.assertEqual(tiled.vocab_tile_size, 2_048)
        # The tiled head's extra work is now traced, not asserted by formula:
        # its custom VJP recomputes logits and its table pads to a whole tile.
        self.assertGreater(
            _traced_per_token(replace(tiled, **_SMALL)),
            _traced_per_token(replace(dense, **_SMALL)),
        )

        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["--profile", "official", "--loss-backend", "tiled"])

    def test_flash_flops_include_right_padding_for_odd_sequences(self) -> None:
        parser = trainer.build_parser()
        common = ["--profile", "dev"]
        experiment_config, config_sha256 = trainer.load_experiment_config("dev")
        dense = _resolve_config(
            parser.parse_args(common),
            "tpu",
            experiment_config=_replace_experiment_config(
                experiment_config,
                profile="dev",
                context={"seq_len": 129},
                kernels={"attention_backend": "dense"},
                evaluation={"final_predictions": 129 * 128},
            ),
            config_sha256=config_sha256,
        )
        flash = _resolve_config(
            parser.parse_args(common),
            "tpu",
            experiment_config=_replace_experiment_config(
                experiment_config,
                profile="dev",
                context={"seq_len": 129},
                kernels={"attention_backend": "tpu_flash"},
                evaluation={"final_predictions": 129 * 128},
            ),
            config_sha256=config_sha256,
        )
        # Flash right-pads q/k/v to 128-wide tiles, so an unaligned sequence
        # genuinely costs more there than in the dense path. The traced count
        # picks this up from the padded shapes with no formula to maintain.
        self.assertGreater(
            _traced_per_token(replace(flash, **_SMALL)),
            _traced_per_token(replace(dense, **_SMALL)),
        )

    def test_batches_reject_tokens_outside_semantic_vocabulary(self) -> None:
        tokens = np.asarray([0, 1, 2, 7, 3, 0, 1, 2], dtype=np.int32)
        dataset = rig_tokens.TokenDataset(
            rig_tokens.ShardedTokens((tokens,)),
            rig_tokens.ShardedTokens((tokens,)),
            "test",
        )
        rng = np.random.default_rng(3)
        with self.assertRaisesRegex(ValueError, "do not fit"):
            dataset.batch("train", rng, 1, 7, 7)
        with self.assertRaisesRegex(ValueError, "do not fit"):
            dataset.validation_batch(0, 1, 7, 7)

    def test_dataset_loader_requires_explicit_train_and_validation_shards(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.npy"
            validation = root / "validation.npy"
            np.save(train, np.arange(32, dtype=np.uint16))
            np.save(validation, np.arange(16, dtype=np.uint16))

            dataset = rig_tokens.load_dataset(
                train_data=[train],
                val_data=[validation],
                data_dtype="uint16",
                data_format="auto",
                seed=7,
            )
            self.assertEqual((len(dataset.train), len(dataset.validation)), (32, 16))
            with self.assertRaisesRegex(ValueError, "supplied together"):
                rig_tokens.load_dataset(
                    train_data=[train],
                    val_data=[],
                    data_dtype="uint16",
                    data_format="auto",
                    seed=7,
                )

    def test_non_smoke_execution_requires_explicit_shards_before_jax_init(self) -> None:
        args = trainer.build_parser().parse_args(["--profile", "dev"])
        with (
            patch.object(trainer, "initialize_distributed_runtime") as initialize,
            self.assertRaisesRegex(ValueError, "explicit --train-data"),
        ):
            trainer.run(args)
        initialize.assert_not_called()

    def test_shuffled_epoch_stream_is_bijective_deterministic_and_rank_disjoint(
        self,
    ) -> None:
        shards = rig_tokens.ShardedTokens(
            (
                np.arange(9, dtype=np.int32),
                np.arange(20, 29, dtype=np.int32),
            )
        )
        expected_starts = {0, 2, 4, 6, 20, 22, 24, 26}

        def epoch(seed: int) -> tuple[list[int], list[int]]:
            streams = tuple(
                rig_tokens.ShuffledEpochBatchStream(
                    shards,
                    global_batch_size=4,
                    seq_len=2,
                    vocab_size=32,
                    seed=seed,
                    process_index=rank,
                    process_count=2,
                )
                for rank in range(2)
            )
            rank_starts = ([], [])
            for _ in range(2):
                for rank, stream in enumerate(streams):
                    x, y = stream.next_batch()
                    self.assertTrue(np.array_equal(x[:, 1:], y[:, :-1]))
                    rank_starts[rank].extend(int(value) for value in x[:, 0])
            return rank_starts

        first = epoch(17)
        self.assertEqual(set(first[0]).intersection(first[1]), set())
        self.assertEqual(set(first[0] + first[1]), expected_starts)
        self.assertEqual(first, epoch(17))
        self.assertNotEqual(first, epoch(18))

        for size in range(1, 80):
            permuted = [rig_tokens._permute_bounded(i, size, 1234) for i in range(size)]
            self.assertEqual(set(permuted), set(range(size)))

    def test_trainable_flash_attention_backends_are_tpu_only(self) -> None:
        parser = trainer.build_parser()
        experiment_config, config_sha256 = trainer.load_experiment_config("dev")
        for backend in ("jax_flash", "tpu_flash"):
            with self.subTest(backend=backend):
                args = parser.parse_args(["--profile", "dev"])
                backend_config = _replace_experiment_config(
                    experiment_config,
                    profile="dev",
                    kernels={"attention_backend": backend},
                )
                with self.assertRaisesRegex(ValueError, "requires a TPU"):
                    _resolve_config(
                        args,
                        "cpu",
                        experiment_config=backend_config,
                        config_sha256=config_sha256,
                    )
                config = _resolve_config(
                    args,
                    "tpu",
                    experiment_config=backend_config,
                    config_sha256=config_sha256,
                )
                self.assertEqual(config.attention_backend, backend)
        for backend in ("jax_flash", "tpu_flash"):
            with self.subTest(float32_backend=backend):
                with self.assertRaisesRegex(ValueError, "requires .*bfloat16"):
                    _resolve_config(
                        parser.parse_args(["--profile", "dev"]),
                        "tpu",
                        experiment_config=_replace_experiment_config(
                            experiment_config,
                            profile="dev",
                            training={"dtype": "float32"},
                            kernels={"attention_backend": backend},
                        ),
                        config_sha256=config_sha256,
                    )

    def test_attention_backend_comes_from_the_profile_with_no_tuning_flags(
        self,
    ) -> None:
        # The tuning cache and the runtime autotuner were removed: a per-host
        # cache was the only way two processes in one SPMD job could select
        # different tiles for the same program.
        parser = trainer.build_parser()
        defaults = parser.parse_args([])
        self.assertFalse(hasattr(defaults, "attention_backend"))
        self.assertFalse(hasattr(defaults, "attention_tuning_cache"))
        self.assertFalse(hasattr(defaults, "autotune_attention"))
        for flag in ("--attention-tuning-cache", "--autotune-attention"):
            with self.subTest(flag=flag), self.assertRaises(SystemExit):
                parser.parse_args(["--profile", "official", flag, "x"])

        official = _resolve_config(parser.parse_args(["--profile", "official"]), "tpu")
        smoke = _resolve_config(parser.parse_args(["--profile", "smoke"]), "cpu")
        development = _resolve_config(parser.parse_args(["--profile", "dev"]), "tpu")
        self.assertEqual(official.attention_backend, "tpu_flash")
        self.assertEqual(smoke.attention_backend, "dense")
        self.assertEqual(development.attention_backend, "tpu_flash")

    def test_attention_runtime_is_shared_and_resolved_before_mesh_creation(
        self,
    ) -> None:
        self.assertIs(
            trainer.resolve_attention_runtime, rig_attention.resolve_attention_runtime
        )
        self.assertIs(trainer.make_mesh_attention, rig_attention.make_mesh_attention)
        source = TRAINER_PATH.read_text(encoding="utf-8")
        self.assertLess(
            source.index("attention_runtime = resolve_attention_runtime("),
            source.index('mesh = Mesh(np.asarray(devices, dtype=object), ("data",))'),
        )
        self.assertLess(
            source.index('mesh = Mesh(np.asarray(devices, dtype=object), ("data",))'),
            source.index("attention_fn = make_mesh_attention("),
        )

    def test_trainer_never_benchmarks_kernels_at_runtime(self) -> None:
        # Runtime measurement would reintroduce per-host tile selection.
        source = TRAINER_PATH.read_text(encoding="utf-8")
        for banned in ("autotune_attention", "attention_tuning_cache", "cache_path"):
            with self.subTest(symbol=banned):
                self.assertNotIn(banned, source)

    def test_attention_tuning_metadata_is_saved_with_exact_plan(self) -> None:
        parser = trainer.build_parser()
        config = _resolve_config(parser.parse_args(["--profile", "smoke"]), "cpu")
        tiles = AttentionTiles(512, 512, 256, 512, 256, 512, 256, 256, 512, 256)
        runtime = trainer.AttentionRuntime("c" * 64, "cache", tiles, 0.0)
        expected = trainer.attention_runtime_metadata(runtime)
        self.assertEqual(expected["key_digest"], "c" * 64)
        self.assertEqual(expected["resolution_source"], "cache")
        self.assertEqual(len(expected["tiles"]), 10)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            runlog.save_checkpoint(
                output,
                {"weight": np.zeros((2, 2), dtype=np.float32)},
                trainer.checkpoint_metadata(config, 7, runtime),
            )
            with np.load(output / runlog.CHECKPOINT_NAME) as checkpoint:
                metadata = json.loads(bytes(checkpoint["metadata.json"]).decode())
        self.assertEqual(metadata["model"]["attention_tuning"], expected)
        self.assertEqual(metadata["configuration"]["path"], "smoke.yaml")
        self.assertEqual(metadata["configuration"]["sha256"], config.config_sha256)
        self.assertEqual(metadata["configuration"]["resolved"]["model"]["layers"], 2)

        implementation = trainer.implementation_metadata(config, runtime)
        self.assertEqual(implementation["attention_tuning"], expected)
        self.assertEqual(
            implementation["configuration"],
            trainer.experiment_config_metadata(config),
        )
        rows = trainer.attention_console_rows(runtime)
        self.assertEqual(
            rows,
            (
                ("attention tuning", "cache · key cccccccccccc"),
                ("attention fwd", "q512 · kv512/256"),
                ("attention dK/dV", "q512/256 · kv512/256"),
                ("attention dQ", "q256 · kv512/256"),
            ),
        )
        self.assertNotIn("c" * 64, repr(rows))
        source = TRAINER_PATH.read_text(encoding="utf-8")
        self.assertIn('"attention_tune_seconds":', source)

    def test_kernel_provenance_does_not_change_fixed_model_contract(self) -> None:
        parser = trainer.build_parser()
        experiment_config, config_sha256 = trainer.load_experiment_config("official")
        experiment_config = _replace_experiment_config(
            experiment_config,
            profile="official",
            kernels={"loss_backend": "tiled"},
            model={"semantic_vocab_size": 50_257},
        )
        config = _resolve_config(
            parser.parse_args(["--profile", "official"]),
            "tpu",
            experiment_config=experiment_config,
            config_sha256=config_sha256,
        )
        tiles = AttentionTiles(512, 512, 256, 512, 256, 512, 256, 256, 512, 256)
        runtime = trainer.AttentionRuntime("d" * 64, "shipped", tiles, 0.0)
        self.assertEqual(
            trainer.contract_model_metadata(config),
            {
                "layers": 12,
                "heads": 10,
                "d_model": 640,
                "mlp_mult": 4,
                "normalization": "rms_norm",
                "position_encoding": "rope_base_10000",
                "mlp_activation": "gelu",
                "vocab_size": 50_304,
                "semantic_vocab_size": 50_257,
                "tied_embeddings": False,
                "tier": "125m",
                "parameterization": "completep_fixed_tpp_v1",
            },
        )
        implementation = trainer.implementation_metadata(config, runtime)
        self.assertEqual(implementation["attention_backend"], "tpu_flash")
        self.assertEqual(implementation["loss_backend"], "tiled")
        self.assertNotIn("semantic_vocab_size", implementation)
        self.assertNotIn("attention_backend", trainer.contract_model_metadata(config))

    def test_probe_schedule_excludes_final_step(self) -> None:
        selected = [
            step
            for step in range(1, 10)
            if evaluation.should_run_validation_probe(step, every=3, final_step=9)
        ]
        self.assertEqual(selected, [3, 6])
        self.assertFalse(
            evaluation.should_run_validation_probe(1, every=0, final_step=10)
        )

    def test_diagnostic_schedule_includes_first_cadence_and_final(self) -> None:
        selected = [
            step
            for step in range(1, 9)
            if evaluation.should_run_diagnostics(step, every=3, final_step=8)
        ]
        self.assertEqual(selected, [1, 3, 6, 8])
        self.assertFalse(evaluation.should_run_diagnostics(1, every=0, final_step=1))

    def test_validation_prefix_always_starts_at_batch_zero(self) -> None:
        class Dataset:
            def __init__(self) -> None:
                self.indices: list[int] = []

            def validation_batch(
                self, index: int, batch_size: int, seq_len: int, vocab_size: int
            ) -> tuple[np.ndarray, np.ndarray]:
                del vocab_size
                self.indices.append(index)
                values = np.full((batch_size, seq_len), index, dtype=np.int32)
                return values, values

        dataset = Dataset()

        def compiled_eval(
            params: object, x: np.ndarray, y: np.ndarray, mask: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
            del params, y
            loss = np.sum((x.astype(np.float32) + 1.0) * mask)
            return np.asarray(loss, dtype=np.float32), np.asarray(
                mask.sum(), dtype=np.float32
            )

        with patch.object(
            evaluation.jax, "device_put", side_effect=lambda value, _sharding: value
        ):
            result = evaluation.evaluate_validation_prefix(
                object(),
                dataset,
                compiled_eval,
                object(),
                batch_size=2,
                seq_len=4,
                semantic_vocab_size=16,
                batches=3,
            )
        self.assertEqual(dataset.indices, [0, 1, 2])
        self.assertAlmostEqual(result.loss, 2.0)
        self.assertEqual(result.scored_tokens, 24)
        self.assertGreater(result.seconds, 0.0)

    def test_training_log_contains_every_step(self) -> None:
        history = np.asarray(
            [[2.0, 1.0e-3, 0.5], [1.5, 5.0e-4, 0.25]], dtype=np.float32
        )
        config = _fake_config(steps=2, batch_size=4, seq_len=8)
        with tempfile.TemporaryDirectory() as directory:
            runlog.write_training_log(
                Path(directory),
                history,
                tokens_per_step=32,
                final_step=2,
                flops_per_token=10,
            )
            log = logpack.read_log(Path(directory) / runlog.TRAINING_LOG_NAME)
        self.assertEqual(
            [entry.describe() for entry in log.columns],
            ["overall/train_loss", "overall/learning_rate", "overall/grad_norm"],
        )
        np.testing.assert_array_equal(log.steps, [1, 2])
        np.testing.assert_array_equal(log.values, history)
        # The axes are derived from the header, not stored per row.
        np.testing.assert_array_equal(log.axis("tokens_processed"), [32, 64])
        np.testing.assert_array_equal(log.axis("cumulative_flops"), [320.0, 640.0])

    def test_stop_after_step_truncates_without_moving_the_schedule(self) -> None:
        parser = trainer.build_parser()
        horizon = ["--profile", "dev", "--tier", "250m", "--tokens-per-parameter", "5"]
        full = _resolve_config(parser.parse_args(horizon), "tpu")
        stopped = _resolve_config(
            parser.parse_args([*horizon, "--stop-after-step", "60"]),
            "tpu",
        )
        # Everything the trajectory depends on has to be bit-identical, or the
        # truncated run samples a different curve than the one it stands in for.
        self.assertEqual(
            (stopped.steps, stopped.warmup_steps, stopped.data_multiplier),
            (full.steps, full.warmup_steps, full.data_multiplier),
        )
        self.assertEqual(
            trainer.effective_optimizer_metadata(stopped),
            trainer.effective_optimizer_metadata(full),
        )
        self.assertEqual(stopped.final_step, 60)
        self.assertEqual(full.final_step, full.steps)
        self.assertLess(60, full.steps)

        # Custom step and raw-token horizons are deliberately no longer part
        # of the public surface.
        for removed in ("--steps", "--train-tokens"):
            with self.subTest(removed=removed), redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit):
                    parser.parse_args([*horizon, removed, "100"])
        with self.assertRaisesRegex(ValueError, "past the"):
            _resolve_config(
                parser.parse_args([*horizon, "--stop-after-step", str(full.steps + 1)]),
                "tpu",
            )

    def test_early_stopped_artifacts_cover_only_the_steps_taken(self) -> None:
        config = _fake_config(steps=5, stop_after_step=2, batch_size=4, seq_len=8)
        # The device history buffer is sized for the full horizon and only its
        # prefix is written, so the writer must drop the trailing zeros.
        history = np.zeros((5, 3), dtype=np.float32)
        history[:2] = [[2.0, 1.0e-3, 0.5], [1.5, 5.0e-4, 0.25]]
        values = np.arange(2 * 3 * 6, dtype=np.float32).reshape((2, 3, 6))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runlog.write_training_log(
                root, history, tokens_per_step=32, final_step=2, flops_per_token=10
            )
            training = logpack.read_log(root / runlog.TRAINING_LOG_NAME)
            runlog.write_diagnostics_log(
                root,
                (
                    runlog.DiagnosticPoint(1, values),
                    runlog.DiagnosticPoint(2, values + 1.0),
                ),
                (("overall", None, 7), ("block", 0, 4)),
                tokens_per_step=32,
                final_step=2,
                flops_per_token=10,
            )
            diagnostics = logpack.read_log(root / runlog.DIAGNOSTICS_LOG_NAME)
        self.assertEqual(len(training), 2)
        np.testing.assert_array_equal(training.steps, [1, 2])
        np.testing.assert_array_equal(training.values, history[:2])
        np.testing.assert_array_equal(diagnostics.steps, [1, 2])
        # An early-stopped run still fires on the step it stops at.
        self.assertTrue(evaluation.should_run_diagnostics(2, every=100, final_step=2))

    def test_diagnostics_log_flattens_the_grid_and_lands_atomically(self) -> None:
        metadata = (("overall", None, 7), ("block", 0, 4))
        shape = (len(metadata), 3, 6)
        values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        config = _fake_config(steps=2, batch_size=4, seq_len=8)
        points = (
            runlog.DiagnosticPoint(1, values),
            runlog.DiagnosticPoint(2, values + 1.0),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runlog.write_diagnostics_log(
                root,
                points,
                metadata,
                tokens_per_step=32,
                final_step=2,
                flops_per_token=10,
            )
            destination = root / runlog.DIAGNOSTICS_LOG_NAME
            self.assertFalse((root / f".{runlog.DIAGNOSTICS_LOG_NAME}.tmp").exists())
            log = logpack.read_log(destination)

        # Column order is the [scope, family, stat] grid flattened in place, so
        # a captured point becomes a record with no per-value bookkeeping.
        self.assertEqual(len(log.columns), 2 * 3 * 6)
        self.assertEqual(log.columns[0].describe(), "overall/param.l1_norm")
        self.assertEqual(log.columns[-1].describe(), "block[0]/update.fourth_moment")
        np.testing.assert_array_equal(log.values[0], values.reshape(-1))
        np.testing.assert_array_equal(log.values[1], (values + 1.0).reshape(-1))
        np.testing.assert_array_equal(log.steps, [1, 2])

        # element_count rides in the column table once, not on every row.
        self.assertEqual(log.columns[0].element_count, 7)
        self.assertEqual(log.columns[-1].element_count, 4)
        self.assertAlmostEqual(
            float(log.series("update.fourth_moment", "block", 0)[0]), 35.0
        )

    def test_diagnostic_statistics_use_postupdate_param_raw_gradient_and_signed_delta(
        self,
    ) -> None:
        params = {
            "token_embedding": np.asarray([[1.0, 2.0]], dtype=np.float32),
            "blocks": [{"weight": np.asarray([-1.0, 1.0], dtype=np.float32)}],
            "final_ln_scale": np.asarray([2.0], dtype=np.float32),
        }
        gradients = trainer.jax.tree_util.tree_map(
            lambda value: np.full_like(value, 2.0), params
        )
        after = trainer.jax.tree_util.tree_map(
            lambda value: value + np.float32(-0.25), params
        )
        values = np.asarray(trainer.diagnostic_values(params, gradients, after))
        metadata = trainer.diagnostic_scope_metadata(params)
        self.assertEqual(
            tuple(
                (scope.scope, scope.layer, scope.index, scope.element_count)
                for scope in metadata
            ),
            (
                ("overall", None, None, 5),
                ("embeddings", None, None, 2),
                ("block", 0, None, 2),
                ("final_norm", None, None, 1),
            ),
        )
        flattened = np.concatenate(
            [np.ravel(value) for value in trainer.jax.tree_util.tree_leaves(after)]
        ).astype(np.float32)
        expected_param = np.asarray(
            [
                np.abs(flattened).sum(),
                np.linalg.norm(flattened),
                flattened.mean(),
                flattened.std(),
                np.mean((flattened - flattened.mean()) ** 3),
                np.mean((flattened - flattened.mean()) ** 4),
            ]
        )
        np.testing.assert_allclose(values[0, 0], expected_param, rtol=1e-6)
        self.assertAlmostEqual(float(values[0, 1, 2]), 2.0)
        self.assertAlmostEqual(float(values[0, 2, 2]), -0.25)

    def test_diagnostic_executable_preserves_ordinary_optimizer_trajectory(
        self,
    ) -> None:
        parser = trainer.build_parser()
        config = replace(
            _resolve_config(parser.parse_args(["--profile", "smoke"]), "cpu"),
            diagnostics_every=1,
        )
        host_params = trainer.init_params(config, 7)
        decay_mask = trainer.weight_decay_mask(host_params)
        x = (
            np.arange(config.batch_size * config.seq_len, dtype=np.int32).reshape(
                config.batch_size, config.seq_len
            )
            % config.vocab_size
        )
        y = (x + 1) % config.vocab_size

        ordinary = trainer.jax.jit(
            lambda p, o, bx, by: trainer.train_step(p, o, bx, by, config, decay_mask)
        )
        diagnostic = trainer.jax.jit(
            lambda p, o, bx, by: trainer.diagnostic_train_step(
                p, o, bx, by, config, decay_mask
            )
        )
        params_a = trainer.jax.tree_util.tree_map(np.copy, host_params)
        params_b = trainer.jax.tree_util.tree_map(np.copy, host_params)
        optimizer_a = trainer.init_optimizer(params_a, config.steps)
        optimizer_b = trainer.init_optimizer(params_b, config.steps)
        params_a, optimizer_a, metrics_a = ordinary(params_a, optimizer_a, x, y)
        params_b, optimizer_b, metrics_b, diagnostics = diagnostic(
            params_b, optimizer_b, x, y
        )
        trainer.sync_tree((params_a, optimizer_a, params_b, optimizer_b, diagnostics))
        for left, right in zip(
            trainer.jax.tree_util.tree_leaves((params_a, optimizer_a, metrics_a)),
            trainer.jax.tree_util.tree_leaves((params_b, optimizer_b, metrics_b)),
            strict=True,
        ):
            np.testing.assert_array_equal(np.asarray(left), np.asarray(right))

    def test_validation_csv_contains_probes_and_canonical_final_row(self) -> None:
        rows: list[runlog.ValidationRow] = [
            runlog.ValidationRow(
                250,
                8_192_000,
                "fineweb_probe",
                "fineweb",
                262_144,
                4.0,
                np.exp(4.0),
                0.25,
                False,
            ),
            runlog.ValidationRow(
                500,
                16_384_000,
                "fineweb",
                "fineweb",
                10_485_760,
                3.5,
                np.exp(3.5),
                8.0,
                True,
            ),
            runlog.ValidationRow(
                500,
                16_384_000,
                "downstream",
                "science",
                8_192,
                3.0,
                np.exp(3.0),
                0.03,
                False,
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runlog.write_validation_csv(root, rows)
            contents = (root / trainer.VALIDATION_CSV_NAME).read_text().splitlines()
            temporary = root / f".{trainer.VALIDATION_CSV_NAME}.tmp"
            self.assertFalse(temporary.exists())
        self.assertEqual(
            contents[0],
            "step,tokens_processed,kind,domain,validation_tokens,validation_loss,"
            "perplexity,validation_seconds,canonical",
        )
        self.assertEqual(
            contents[1],
            f"250,8192000,fineweb_probe,fineweb,262144,4.0,{np.exp(4.0)},0.25,false",
        )
        self.assertEqual(
            contents[2],
            f"500,16384000,fineweb,fineweb,10485760,3.5,{np.exp(3.5)},8.0,true",
        )
        self.assertEqual(len(contents), 4)

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "canonical"):
                runlog.write_validation_csv(Path(directory), (rows[0], rows[2]))

    def test_downstream_batches_mask_document_boundaries_and_exact_targets(
        self,
    ) -> None:
        domain = rig_tokens.DownstreamDomain(
            "science",
            np.asarray([99, 10, 11, 12, 99, 20, 21], dtype=np.uint16),
            (
                rig_tokens.DocumentSpan(0, 4, 1, 3),
                rig_tokens.DocumentSpan(4, 3, 5, 2),
            ),
        )
        batches = rig_tokens.downstream_batches(domain, seq_len=2, batch_size=2)
        pairs = []
        for x, y, mask in batches:
            flat_x, flat_y, flat_mask = x.ravel(), y.ravel(), mask.ravel()
            pairs.extend(
                (int(flat_x[index]), int(flat_y[index]))
                for index in np.flatnonzero(flat_mask)
            )
        self.assertEqual(pairs, [(99, 10), (10, 11), (11, 12), (99, 20), (20, 21)])
        self.assertNotIn((12, 99), pairs)
        self.assertEqual(sum(int(mask.sum()) for _, _, mask in batches), 5)

    def test_gpt2_downstream_manifest_fits_padded_model_vocabulary(self) -> None:
        parser = trainer.build_parser()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard = root / "science.bin"
            header = np.zeros(256, dtype="<i4")
            header[:3] = (20_240_520, 1, 3)
            tokens = np.asarray([50_256, 1, 2], dtype="<u2")
            shard.write_bytes(header.tobytes() + tokens.tobytes())
            manifest = root / "fresh10.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "kind": "fresh10",
                        "tokenizer": {"name": "gpt2", "vocab_size": 50_257},
                        "domains": [
                            {
                                "name": "science",
                                "path": shard.name,
                                "bytes": shard.stat().st_size,
                                "tokens": 3,
                                "scored_tokens": 2,
                                "sha256": hashlib.sha256(
                                    shard.read_bytes()
                                ).hexdigest(),
                                "documents": [
                                    {
                                        "token_offset": 0,
                                        "token_count": 3,
                                        "score_offset": 1,
                                        "scored_tokens": 2,
                                    }
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            args = parser.parse_args(["--downstream-manifest", str(manifest)])
            domains = rig_tokens.load_downstream_domains(
                manifest=args.downstream_manifest,
                root=args.downstream_root,
                vocab_size=50_304,
            )
            self.assertEqual(domains[0].scored_tokens, 2)
            with self.assertRaisesRegex(ValueError, "must fit the model vocabulary"):
                rig_tokens.load_downstream_domains(
                    manifest=args.downstream_manifest,
                    root=args.downstream_root,
                    vocab_size=50_000,
                )

    def test_console_writes_only_to_stderr(self) -> None:
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            console = trainer.Console("never")
            console.banner()
            console.table(
                "test",
                (("field", "value"), ("long field", "x" * 512)),
            )
            console.phase("phase", "detail")
            console.step(1, 1, 1.25, 1.0e-3, 0.5, 1024.0)
            console.success(1.0, 12.5, 0.25)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("GPT TPU RIG", stderr.getvalue())
        self.assertIn("synchronized training 12.500s", stderr.getvalue())
        self.assertIn("compilation excluded", stderr.getvalue())
        self.assertIn("validation loss", stderr.getvalue())
        table_lines = [line for line in stderr.getvalue().splitlines() if "│" in line]
        self.assertTrue(table_lines)
        self.assertLessEqual(max(map(len, table_lines)), 80)
        self.assertIn("…", stderr.getvalue())

    def test_weight_decay_mask_selects_matrices_not_bias_or_norm(self) -> None:
        params = {
            "token_embedding": np.zeros((16, 8), dtype=np.float32),
            "blocks": [
                {
                    "qkv_w": np.zeros((8, 24), dtype=np.float32),
                    "qkv_b": np.zeros((24,), dtype=np.float32),
                    "ln1_scale": np.ones((8,), dtype=np.float32),
                }
            ],
            "final_ln_bias": np.zeros((8,), dtype=np.float32),
        }
        mask = trainer.weight_decay_mask(params)
        self.assertTrue(mask["token_embedding"])
        self.assertTrue(mask["blocks"][0]["qkv_w"])
        self.assertFalse(mask["blocks"][0]["qkv_b"])
        self.assertFalse(mask["blocks"][0]["ln1_scale"])
        self.assertFalse(mask["final_ln_bias"])

    def test_official_topology_accepts_single_or_multi_host_v4(self) -> None:
        v4_devices = [FakeDevice("tpu", "TPU v4") for _ in range(4)]
        with (
            patch.object(trainer.jax, "local_devices", return_value=v4_devices),
            patch.object(trainer.jax, "process_count", return_value=1),
            patch.object(trainer.jax, "device_count", return_value=4),
        ):
            trainer.validate_official_topology("official", v4_devices)

        global_v4_devices = [FakeDevice("tpu", "TPU v4") for _ in range(8)]
        with (
            patch.object(trainer.jax, "local_devices", return_value=v4_devices),
            patch.object(trainer.jax, "process_count", return_value=2),
            patch.object(trainer.jax, "device_count", return_value=8),
        ):
            trainer.validate_official_topology("official", global_v4_devices)

        invalid_cases = (
            (v4_devices, v4_devices, 2, 4),
            (v4_devices[:2], v4_devices[:2], 1, 2),
            (
                [FakeDevice("cpu", "cpu") for _ in range(4)],
                [FakeDevice("cpu", "cpu") for _ in range(4)],
                1,
                4,
            ),
            (
                [FakeDevice("tpu", "TPU v5p") for _ in range(4)],
                [FakeDevice("tpu", "TPU v5p") for _ in range(4)],
                1,
                4,
            ),
        )
        for devices, local_devices, process_count, device_count in invalid_cases:
            with self.subTest(devices=devices, process_count=process_count):
                with (
                    patch.object(
                        trainer.jax, "local_devices", return_value=local_devices
                    ),
                    patch.object(
                        trainer.jax, "process_count", return_value=process_count
                    ),
                    patch.object(
                        trainer.jax, "device_count", return_value=device_count
                    ),
                    self.assertRaisesRegex(RuntimeError, "4 local TPU v4"),
                ):
                    trainer.validate_official_topology("official", devices)

    def test_rank_local_slice_partitions_global_batch_without_overlap(self) -> None:
        values = np.arange(24, dtype=np.int32).reshape(8, 3)
        pieces = [trainer.rank_local_slice(values, rank, 4) for rank in range(4)]
        np.testing.assert_array_equal(np.concatenate(pieces), values)
        self.assertTrue(all(piece.flags.c_contiguous for piece in pieces))
        with self.assertRaisesRegex(ValueError, "divisible"):
            trainer.rank_local_slice(values[:7], 0, 4)

    def test_controller_hostname_is_independent_of_jax_process_index(self) -> None:
        with (
            patch.dict(
                trainer.os.environ,
                {"RIG_CONTROLLER_HOSTNAME": "slice-w-0"},
                clear=False,
            ),
            patch.object(trainer.socket, "gethostname", return_value="slice-w-0"),
        ):
            self.assertTrue(trainer.is_controller_process(3))
        with (
            patch.dict(
                trainer.os.environ,
                {"RIG_CONTROLLER_HOSTNAME": "slice-w-0"},
                clear=False,
            ),
            patch.object(trainer.socket, "gethostname", return_value="slice-w-2"),
        ):
            self.assertFalse(trainer.is_controller_process(0))

    def test_system_metadata_is_versioned_and_topology_aware(self) -> None:
        devices = [FakeDevice("tpu", "TPU v4") for _ in range(4)]
        with (
            patch.object(trainer.jax, "device_count", return_value=4),
            patch.object(trainer.jax, "local_device_count", return_value=4),
            patch.object(trainer.jax, "process_count", return_value=1),
        ):
            metadata = trainer.system_metadata(devices)
        self.assertEqual(metadata["platform"], "tpu")
        self.assertEqual(metadata["device_count"], 4)
        self.assertEqual(metadata["local_device_count"], 4)
        self.assertEqual(metadata["process_count"], 1)
        self.assertEqual(metadata["device_kinds"], ["TPU v4"])
        self.assertEqual(metadata["jax_version"], trainer.jax.__version__)
        self.assertEqual(metadata["jaxlib_version"], trainer.jaxlib.__version__)
        self.assertIn("libtpu_version", metadata)
        self.assertIn("python_version", metadata)
        self.assertEqual(metadata["device_ids"], [None] * 4)

    def test_compile_metric_names_are_unambiguous(self) -> None:
        source = TRAINER_PATH.read_text(encoding="utf-8")
        self.assertIn('"train_compile_seconds":', source)
        self.assertIn('"eval_compile_seconds":', source)
        self.assertIn('"total_compile_seconds":', source)
        self.assertNotIn('"compile_seconds":', source)
        self.assertEqual(source.count("compiled_eval = "), 1)
        self.assertIn('"validation_curve": VALIDATION_CSV_NAME', source)
        # Whitespace-insensitive: a formatter may reflow the call chain across
        # lines, and what matters is that the eval executable is lowered once
        # and compiled, not how it is laid out.
        compact = re.sub(r"\s+", "", source)
        self.assertIn(
            ").lower(params,sample_x,sample_y,sample_mask).compile()", compact
        )
        probe = source.index("if should_run_validation_probe(")
        synchronize = source.index(
            "sync_tree((params, optimizer, last_metrics))", probe
        )
        evaluate = source.index("evaluate_validation_prefix(", synchronize)
        self.assertLess(synchronize, evaluate)


if __name__ == "__main__":
    unittest.main()


class SalvageTests(unittest.TestCase):
    """Partial artifacts for jobs that never reach their final write."""

    def _config(self, steps: int = 200):
        parser = trainer.build_parser()
        resolved = _resolve_config(parser.parse_args(["--profile", "dev"]), "tpu")
        return replace(
            resolved,
            steps=steps,
            stop_after_step=None,
            warmup_steps=min(resolved.warmup_steps, max(0, steps - 1)),
        )

    def _meta(self):
        return [("embeddings", None, 100), ("block", 0, 200), ("final_norm", None, 50)]

    def _point(self, step: int):
        shape = (
            len(self._meta()),
            len(metrics.DIAGNOSTIC_FAMILIES),
            len(metrics.DIAGNOSTIC_STATS),
        )
        return runlog.DiagnosticPoint(
            step, np.full(shape, step / 1000.0, dtype=np.float32)
        )

    def test_device_residency_is_bounded_by_a_constant(self) -> None:
        # Without a cap the capture list grows with the run: one small device
        # allocation per capture, all live until the loop ends.
        self.assertIsInstance(runlog.DIAGNOSTIC_FLUSH_POINTS, int)
        self.assertGreater(runlog.DIAGNOSTIC_FLUSH_POINTS, 0)
        source = inspect.getsource(trainer.run)
        self.assertIn(
            "len(diagnostic_device_points) >= DIAGNOSTIC_FLUSH_POINTS", source
        )
        self.assertIn("diagnostic_device_points.clear()", source)

    def test_flushed_points_still_reach_the_authoritative_file(self) -> None:
        # Assembling the final tuple from the device list alone would truncate
        # the file to whatever had not yet been flushed.
        source = inspect.getsource(trainer.run)
        assembly = source[source.index("diagnostic_points = tuple(") :]
        self.assertIn("diagnostic_points_host", assembly[:400])

    def test_partial_points_match_the_authoritative_layout(self) -> None:
        config, meta = self._config(), self._meta()
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            destination = out / runlog.DIAGNOSTICS_LOG_NAME
            writer = runlog.open_log(
                destination,
                runlog.diagnostic_log_columns(meta),
                tokens_per_step=config.batch_size * config.seq_len,
                flops_per_token=341_312_256,
            )
            for step in (10, 20):
                runlog.append_log_row(
                    writer, step, self._point(step).values.reshape(-1)
                )
            runlog.close_log(writer)
            partial = logpack.read_log(destination)

            runlog.write_diagnostics_log(
                out,
                [self._point(s) for s in range(10, 201, 10)],
                meta,
                tokens_per_step=config.batch_size * config.seq_len,
                final_step=config.final_step,
                flops_per_token=341_312_256,
            )
            complete = logpack.read_log(destination)

        # Both writers build their columns from the same helper, so a partial
        # file is a prefix of the complete one rather than a different shape.
        self.assertEqual(partial.columns, complete.columns)
        self.assertEqual(len(partial), 2)
        # Superseded, not appended to.
        self.assertEqual(len(complete), 20)
        np.testing.assert_array_equal(partial.values, complete.values[:2])

    def test_partial_training_points_match_the_authoritative_layout(self) -> None:
        config = self._config(25)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            destination = out / runlog.TRAINING_LOG_NAME
            writer = runlog.open_log(
                destination,
                runlog.training_log_columns(),
                tokens_per_step=config.batch_size * config.seq_len,
                flops_per_token=341_312_256,
            )
            for step in (1, 10, 20):
                runlog.append_log_row(writer, step, (1.0, 1e-4, 0.5))
            runlog.close_log(writer)
            partial = logpack.read_log(destination)

            history = np.zeros((config.steps, 3), dtype=np.float32)
            runlog.write_training_log(
                out,
                history,
                tokens_per_step=config.batch_size * config.seq_len,
                final_step=config.final_step,
                flops_per_token=341_312_256,
            )
            complete = logpack.read_log(destination)

        self.assertEqual(partial.columns, complete.columns)
        self.assertEqual(partial.tokens_per_step, complete.tokens_per_step)
        np.testing.assert_array_equal(partial.steps, [1, 10, 20])
        self.assertEqual(len(complete), config.steps)

    def test_salvage_writers_never_fail_a_run(self) -> None:
        config, meta = self._config(), self._meta()
        missing = Path("/nonexistent/rig-salvage")
        for columns in (
            runlog.training_log_columns(),
            runlog.diagnostic_log_columns(meta),
        ):
            with self.subTest(columns=len(columns)):
                writer = runlog.open_log(
                    missing / "x.riglog",
                    columns,
                    tokens_per_step=config.batch_size * config.seq_len,
                    flops_per_token=341_312_256,
                )
                runlog.append_log_row(writer, 1, np.zeros(len(columns)))
                runlog.close_log(writer)
        # A run with no traced FLOP count still trains; it just has no log.
        self.assertIsNone(
            runlog.open_log(
                missing / "x.riglog",
                runlog.training_log_columns(),
                tokens_per_step=config.batch_size * config.seq_len,
                flops_per_token=None,
            )
        )
