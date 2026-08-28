"""Configuration and FLOP contracts for the fuzzy-TopK fork."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from rig.kernels import (  # noqa: E402
    FUZZY_FEATURE_STAT_NAMES,
    FuzzyTopKConfig,
    fuzzy_topk_mlp_with_diagnostics,
)
from rig.plan import validate_recipe_plan  # noqa: E402


ROOT = Path(__file__).parents[1]
TRAINER_PATH = ROOT / "recipes" / "fuzzy_topk_autoencoder" / "train.py"
SPEC = importlib.util.spec_from_file_location(
    "fuzzy_topk_autoencoder_train", TRAINER_PATH
)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import invariant
    raise RuntimeError(f"could not import {TRAINER_PATH}")
trainer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = trainer
SPEC.loader.exec_module(trainer)


def resolved(profile: str = "smoke", *overrides: str):
    parser = trainer.build_parser()
    args = parser.parse_args(["--profile", profile, *overrides])
    document, digest = trainer.load_experiment_config(profile)
    trainer.validate_args(args, document)
    return trainer.resolve_config(
        args,
        "cpu" if profile == "smoke" else "tpu",
        experiment_config=document,
        config_sha256=digest,
    )


class FuzzyTopKRecipeTests(unittest.TestCase):
    def test_dev_defaults_match_sparse_parent_and_target_is_explicit(self) -> None:
        config = resolved("dev", "--tier", "60m")
        self.assertEqual(config.context_preset, "8k")
        self.assertEqual(config.seq_len, 8192)
        self.assertTrue(config.document_masking)
        self.assertEqual(config.mlp_mult, 16)
        self.assertEqual(config.mlp_top_k, 128)
        self.assertEqual(config.sparse_mlp_backend, "choicewise")
        self.assertEqual(config.declared_parameters, 102_440_832)
        self.assertEqual(config.steps, 3_908)
        validate_recipe_plan(trainer.resolved_plan_metadata(config))

        target = resolved(
            "dev", "--tier", "60m", "--sparse-top-k", str(4 * config.d_model)
        )
        self.assertEqual(target.mlp_top_k, 4 * target.d_model)
        self.assertEqual(target.mlp_mult * target.d_model // target.mlp_top_k, 4)
        self.assertEqual(target.sparsity_diagnostics_every, 100)

    def test_feature_diagnostic_cadence_can_be_disabled_for_the_speed_control(
        self,
    ) -> None:
        control = resolved(
            "dev",
            "--tier",
            "60m",
            "--sparsity-diagnostics-every",
            "0",
        )
        treatment = resolved(
            "dev",
            "--tier",
            "60m",
            "--sparsity-diagnostics-every",
            "10",
        )
        self.assertEqual(control.sparsity_diagnostics_every, 0)
        self.assertEqual(treatment.sparsity_diagnostics_every, 10)
        self.assertEqual(
            [
                step
                for step in range(1, 206)
                if trainer.should_run_sparsity_diagnostics(
                    step, every=100, final_step=205
                )
            ],
            [1, 100, 200, 205],
        )

    def test_feature_observer_leaves_the_optimizer_update_exactly_unchanged(self) -> None:
        config = resolved("smoke")
        params = trainer.init_params(config, 3)
        optimizer = jax.tree_util.tree_map(
            jnp.asarray, trainer.init_optimizer(params, config.steps)
        )
        decay_mask = trainer.weight_decay_mask(params)
        tokens = jnp.arange(config.batch_size * config.seq_len, dtype=jnp.int32)
        tokens = tokens.reshape((config.batch_size, config.seq_len))
        x = tokens % config.semantic_vocab_size
        y = (tokens + 1) % config.semantic_vocab_size

        kernel_config = FuzzyTopKConfig(
            top_k=config.mlp_top_k,
            backend=config.sparse_mlp_backend,
        )

        def diagnostic_mlp(*operands):
            return fuzzy_topk_mlp_with_diagnostics(
                *operands, config=kernel_config
            )

        feature_statistics = trainer.fuzzy_sparsity_diagnostics(
            params,
            x,
            config,
            None,
            diagnostic_mlp,
        )
        observed_then_updated = trainer.diagnostic_train_step(
            params, optimizer, x, y, config, decay_mask
        )
        ordinary = trainer.diagnostic_train_step(
            params, optimizer, x, y, config, decay_mask
        )

        for actual_tree, expected_tree in zip(
            observed_then_updated, ordinary, strict=True
        ):
            for actual, expected in zip(
                jax.tree_util.tree_leaves(actual_tree),
                jax.tree_util.tree_leaves(expected_tree),
                strict=True,
            ):
                np.testing.assert_array_equal(actual, expected)
        self.assertEqual(
            feature_statistics.shape,
            (
                len(FUZZY_FEATURE_STAT_NAMES),
                config.layers,
                config.mlp_mult * config.d_model,
            ),
        )

    def test_recipe_local_overrides_preserve_integral_groups(self) -> None:
        config = resolved(
            "dev",
            "--tier",
            "60m",
            "--sparse-layers",
            "13",
            "--sparse-mlp-mult",
            "8",
            "--sparse-top-k",
            "768",
            "--sparse-mlp-backend",
            "reference",
            "--sparse-training-steps",
            "2267",
        )
        self.assertEqual(config.layers, 13)
        self.assertEqual(config.mlp_mult, 8)
        self.assertEqual(config.mlp_top_k, 768)
        self.assertEqual(config.sparse_mlp_backend, "reference")
        self.assertEqual(config.declared_parameters, 77_047_296)
        self.assertEqual(config.steps, 2_267)
        self.assertEqual(config.data_multiplier, 1.0)
        validate_recipe_plan(trainer.resolved_plan_metadata(config))

    def test_override_rejects_nonintegral_groups(self) -> None:
        with self.assertRaisesRegex(ValueError, "must divide"):
            resolved(
                "dev",
                "--tier",
                "60m",
                "--sparse-top-k",
                "1000",
            )

    def test_choicewise_flop_rule_counts_physical_dense_contractions(self) -> None:
        config = resolved("smoke")
        params = trainer.init_params(config, 3)
        breakdown = trainer.traced_flops(config, params)
        fuzzy = breakdown.by_site["_choicewise_fuzzy_topk_mlp"]
        tokens = config.seq_len
        d = config.d_model
        h = config.mlp_mult * d
        self.assertEqual(fuzzy, config.layers * 12 * tokens * d * h)
        self.assertFalse(
            [warning for warning in breakdown.warnings if "fuzzy_topk" in warning]
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
