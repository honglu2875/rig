"""Configuration, neutrality, and FLOP contracts for fuzzy homeostasis."""

from __future__ import annotations

import functools
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
    FUZZY_BALANCE_STAT_NAMES,
    FuzzyTopKBalanceConfig,
    fuzzy_topk_mlp_with_balance,
)
from rig.plan import validate_recipe_plan  # noqa: E402


ROOT = Path(__file__).parents[1]


def load_trainer(recipe: str, module_name: str):
    path = ROOT / "recipes" / recipe / "train.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - import invariant
        raise RuntimeError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


trainer = load_trainer("fuzzy_topk_balanced", "fuzzy_topk_balanced_train")
control_trainer = load_trainer(
    "fuzzy_topk_autoencoder", "fuzzy_topk_autoencoder_balance_control_train"
)


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


def resolved_control(profile: str = "smoke", *overrides: str):
    parser = control_trainer.build_parser()
    args = parser.parse_args(["--profile", profile, *overrides])
    document, digest = control_trainer.load_experiment_config(profile)
    control_trainer.validate_args(args, document)
    return control_trainer.resolve_config(
        args,
        "cpu" if profile == "smoke" else "tpu",
        experiment_config=document,
        config_sha256=digest,
    )


class FuzzyTopKBalancedRecipeTests(unittest.TestCase):
    def test_neutral_defaults_preserve_the_fuzzy_protocol(self) -> None:
        config = resolved("dev", "--tier", "60m")
        self.assertEqual(config.balance_mode, "none")
        self.assertEqual(config.balance_coefficient, 0.0)
        self.assertEqual(config.alive_coefficient, 0.0)
        self.assertFalse(config.balance_enabled)
        self.assertEqual(config.mlp_mult, 16)
        self.assertEqual(config.sparse_mlp_backend, "choicewise")
        validate_recipe_plan(trainer.resolved_plan_metadata(config))

    def test_balance_overrides_are_explicit_and_choicewise_only(self) -> None:
        config = resolved(
            "dev",
            "--tier",
            "60m",
            "--fuzzy-balance-mode",
            "switch",
            "--fuzzy-balance-coefficient",
            "0.01",
            "--fuzzy-balance-temperature",
            "0.7",
            "--fuzzy-alive-coefficient",
            "0.02",
            "--fuzzy-alive-margin",
            "0.1",
        )
        self.assertTrue(config.balance_enabled)
        self.assertTrue(config.soft_balance_enabled)
        self.assertEqual(config.balance_mode, "switch")
        self.assertEqual(config.balance_coefficient, 0.01)
        self.assertEqual(config.balance_temperature, 0.7)
        self.assertEqual(config.alive_coefficient, 0.02)
        self.assertEqual(config.alive_margin, 0.1)

        with self.assertRaisesRegex(ValueError, "choicewise"):
            resolved(
                "dev",
                "--tier",
                "60m",
                "--sparse-mlp-backend",
                "reference",
                "--fuzzy-balance-mode",
                "switch",
                "--fuzzy-balance-coefficient",
                "0.01",
            )

    def test_balance_mode_and_coefficient_cannot_silently_disagree(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be positive"):
            resolved("smoke", "--fuzzy-balance-mode", "switch")
        with self.assertRaisesRegex(ValueError, "must be zero"):
            resolved("smoke", "--fuzzy-balance-coefficient", "0.01")

    def test_neutral_optimizer_update_is_exactly_the_parent_update(self) -> None:
        config = resolved("smoke")
        control = resolved_control("smoke")
        params = trainer.init_params(config, 17)
        control_params = control_trainer.init_params(control, 17)
        optimizer = jax.tree_util.tree_map(
            jnp.asarray, trainer.init_optimizer(params, config.steps)
        )
        control_optimizer = jax.tree_util.tree_map(
            jnp.asarray,
            control_trainer.init_optimizer(control_params, control.steps),
        )
        tokens = jnp.arange(config.batch_size * config.seq_len, dtype=jnp.int32)
        x = (tokens.reshape(config.batch_size, config.seq_len) + 11) % 251
        y = (x + 1) % 251

        actual = trainer.train_step(params, optimizer, x, y, config)
        expected = control_trainer.train_step(
            control_params, control_optimizer, x, y, control
        )
        for actual_tree, expected_tree in zip(actual[:2], expected[:2], strict=True):
            for actual_leaf, expected_leaf in zip(
                jax.tree_util.tree_leaves(actual_tree),
                jax.tree_util.tree_leaves(expected_tree),
                strict=True,
            ):
                np.testing.assert_array_equal(actual_leaf, expected_leaf)
        for name in ("loss", "grad_norm", "learning_rate"):
            np.testing.assert_array_equal(actual[2][name], expected[2][name])
        np.testing.assert_array_equal(actual[2]["objective"], actual[2]["loss"])
        self.assertEqual(actual[2]["balance_row"].shape, (0,))

    def test_auxiliary_objective_does_not_pollute_reported_cross_entropy(self) -> None:
        config = resolved(
            "smoke",
            "--fuzzy-balance-mode",
            "switch",
            "--fuzzy-balance-coefficient",
            "0.01",
            "--fuzzy-alive-coefficient",
            "0.02",
            "--fuzzy-alive-margin",
            "0.1",
        )
        params = jax.tree_util.tree_map(jnp.asarray, trainer.init_params(config, 23))
        tokens = jnp.arange(config.batch_size * config.seq_len, dtype=jnp.int32)
        x = tokens.reshape(config.batch_size, config.seq_len) % 251
        y = (x + 1) % 251
        balance_config = FuzzyTopKBalanceConfig(
            top_k=config.mlp_top_k,
            temperature=config.balance_temperature,
            alive_margin=config.alive_margin,
        )
        balanced_mlp = functools.partial(
            fuzzy_topk_mlp_with_balance, config=balance_config
        )

        objective, (reported_ce, statistics) = trainer.cross_entropy_and_balance(
            params, x, y, config, balanced_mlp_fn=balanced_mlp
        )
        ordinary_ce = trainer.cross_entropy(params, x, y, config)
        np.testing.assert_array_equal(reported_ce, ordinary_ce)
        self.assertGreater(float(objective), float(reported_ce))
        self.assertEqual(
            statistics.layers.shape,
            (config.layers, len(FUZZY_BALANCE_STAT_NAMES)),
        )

    def test_balanced_flop_rule_counts_one_shared_forward_recompute(self) -> None:
        control = resolved("smoke")
        balanced = resolved(
            "smoke",
            "--fuzzy-balance-mode",
            "switch",
            "--fuzzy-balance-coefficient",
            "0.01",
        )
        control_breakdown = trainer.traced_flops(
            control, trainer.init_params(control, 3)
        )
        balanced_breakdown = trainer.traced_flops(
            balanced, trainer.init_params(balanced, 3)
        )
        tokens = balanced.seq_len
        d_model = balanced.d_model
        hidden = balanced.mlp_mult * d_model
        self.assertEqual(
            control_breakdown.by_site["_choicewise_fuzzy_topk_mlp"],
            control.layers * 12 * tokens * d_model * hidden,
        )
        self.assertEqual(
            balanced_breakdown.by_site["_balanced_choicewise_fuzzy_topk_mlp"],
            balanced.layers * 14 * tokens * d_model * hidden,
        )

        low_overhead = resolved(
            "smoke",
            "--fuzzy-balance-mode",
            "bias",
            "--fuzzy-balance-coefficient",
            "0.1",
            "--fuzzy-alive-coefficient",
            "1.0",
        )
        low_overhead_breakdown = trainer.traced_flops(
            low_overhead, trainer.init_params(low_overhead, 3)
        )
        self.assertEqual(
            low_overhead_breakdown.by_site["_low_overhead_choicewise_fuzzy_topk_mlp"],
            low_overhead.layers * 12 * tokens * d_model * hidden,
        )

    def test_balance_log_width_tracks_model_and_layer_statistics(self) -> None:
        neutral = resolved("smoke")
        balanced = resolved(
            "smoke",
            "--fuzzy-balance-mode",
            "importance",
            "--fuzzy-balance-coefficient",
            "0.01",
        )
        base_width = len(trainer.training_log_columns())
        self.assertEqual(len(trainer.fuzzy_training_log_columns(neutral)), base_width)
        self.assertEqual(
            len(trainer.fuzzy_training_log_columns(balanced)),
            base_width + len(FUZZY_BALANCE_STAT_NAMES) * (balanced.layers + 1),
        )
        optimizer = trainer.init_optimizer(
            trainer.init_params(balanced, 3),
            balanced.steps,
            balance_layers=balanced.layers,
        )
        self.assertEqual(
            optimizer["history"].shape[1],
            3 + len(FUZZY_BALANCE_STAT_NAMES) * (balanced.layers + 1),
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
