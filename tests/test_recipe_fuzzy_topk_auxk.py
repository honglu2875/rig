"""Configuration, state, initialization, and FLOP contracts for fuzzy AuxK."""

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
    FuzzyTopKAuxKConfig,
    fuzzy_topk_mlp_with_auxk,
)
from rig.plan import validate_recipe_plan  # noqa: E402


ROOT = Path(__file__).parents[1]


def load_trainer():
    path = ROOT / "recipes" / "fuzzy_topk_auxk" / "train.py"
    spec = importlib.util.spec_from_file_location("fuzzy_topk_auxk_train", path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


trainer = load_trainer()


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


class FuzzyTopKAuxKRecipeTests(unittest.TestCase):
    def test_defaults_encode_the_declared_paper_arm(self) -> None:
        config = resolved("dev", "--tier", "60m", "--sparse-layers", "11")
        self.assertEqual(config.layers, 11)
        self.assertEqual(config.base_depth, 12)
        self.assertAlmostEqual(config.depth_multiplier, 11.0 / 12.0)
        self.assertEqual(config.mlp_mult, 16)
        self.assertEqual(config.mlp_top_k, 4 * config.d_model)
        self.assertEqual(config.init_mode, "tied_directions")
        self.assertTrue(config.auxk_enabled)
        self.assertEqual(config.auxk_coefficient, 1.0 / 32.0)
        self.assertEqual(config.aux_k, config.d_model // 2)
        self.assertEqual(config.mlp_top_k // config.aux_k, 8)
        self.assertEqual(config.dead_tokens_threshold, 10_000_000)
        self.assertEqual(config.dead_after_steps, 77)
        self.assertFalse(config.balance_enabled)
        validate_recipe_plan(trainer.resolved_plan_metadata(config))

    def test_tied_initialization_changes_only_mlp_down_directions(self) -> None:
        tied = resolved("smoke")
        independent = resolved("smoke", "--fuzzy-init-mode", "independent")
        tied_params = trainer.init_params(tied, 17)
        independent_params = trainer.init_params(independent, 17)

        for tied_block, independent_block in zip(
            tied_params["blocks"], independent_params["blocks"], strict=True
        ):
            np.testing.assert_array_equal(
                tied_block["mlp_down_w"], tied_block["mlp_up_w"].T
            )
            for name in tied_block:
                if name != "mlp_down_w":
                    np.testing.assert_array_equal(
                        tied_block[name], independent_block[name]
                    )
        np.testing.assert_array_equal(
            tied_params["token_embedding"], independent_params["token_embedding"]
        )
        np.testing.assert_array_equal(
            tied_params["final_ln_scale"], independent_params["final_ln_scale"]
        )

    def test_auxk_mode_and_coefficient_cannot_silently_disagree(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be zero"):
            resolved("smoke", "--fuzzy-auxk-mode", "none")
        with self.assertRaisesRegex(ValueError, "must be positive"):
            resolved("smoke", "--fuzzy-auxk-coefficient", "0")
        with self.assertRaisesRegex(ValueError, "whole aux_k|divide"):
            resolved("smoke", "--fuzzy-auxk-width-ratio", "0.3")

    def test_ghost_forward_reports_the_ordinary_cross_entropy(self) -> None:
        config = resolved("smoke")
        params = jax.tree_util.tree_map(jnp.asarray, trainer.init_params(config, 23))
        tokens = jnp.arange(config.batch_size * config.seq_len, dtype=jnp.int32)
        x = tokens.reshape(config.batch_size, config.seq_len) % 251
        y = (x + 1) % 251
        operation = functools.partial(
            fuzzy_topk_mlp_with_auxk,
            config=FuzzyTopKAuxKConfig(
                top_k=config.mlp_top_k,
                aux_k=config.aux_k,
                coefficient=config.auxk_coefficient,
            ),
        )
        dead_mask = jnp.ones(
            (config.layers, config.mlp_mult * config.d_model), jnp.bool_
        )

        objective, (reported_ce, statistics) = trainer.cross_entropy_and_balance(
            params,
            x,
            y,
            config,
            auxk_mlp_fn=operation,
            dead_mask=dead_mask,
            auxk_cohort=jnp.asarray(0, jnp.int32),
        )
        ordinary_ce = trainer.cross_entropy(params, x, y, config)
        np.testing.assert_array_equal(reported_ce, ordinary_ce)
        np.testing.assert_array_equal(objective, reported_ce)
        self.assertIsNone(statistics.balance)
        self.assertEqual(
            statistics.auxk.active_counts.shape,
            (config.layers, config.mlp_mult * config.d_model),
        )

    def test_first_update_matches_parent_before_any_feature_is_dead(self) -> None:
        aux = resolved("smoke")
        control = resolved(
            "smoke",
            "--fuzzy-auxk-mode",
            "none",
            "--fuzzy-auxk-coefficient",
            "0",
        )
        params = jax.tree_util.tree_map(jnp.asarray, trainer.init_params(aux, 29))
        control_params = jax.tree_util.tree_map(
            jnp.asarray, trainer.init_params(control, 29)
        )
        optimizer = jax.tree_util.tree_map(
            jnp.asarray,
            trainer.init_optimizer(
                params,
                aux.steps,
                auxk_shape=(aux.layers, aux.mlp_mult * aux.d_model),
            ),
        )
        control_optimizer = jax.tree_util.tree_map(
            jnp.asarray, trainer.init_optimizer(control_params, control.steps)
        )
        tokens = jnp.arange(aux.batch_size * aux.seq_len, dtype=jnp.int32)
        x = (tokens.reshape(aux.batch_size, aux.seq_len) + 7) % 251
        y = (x + 1) % 251
        operation = functools.partial(
            fuzzy_topk_mlp_with_auxk,
            config=FuzzyTopKAuxKConfig(
                top_k=aux.mlp_top_k,
                aux_k=aux.aux_k,
                coefficient=aux.auxk_coefficient,
            ),
        )

        actual = trainer.train_step(
            params, optimizer, x, y, aux, auxk_mlp_fn=operation
        )
        expected = trainer.train_step(
            control_params, control_optimizer, x, y, control
        )
        for actual_leaf, expected_leaf in zip(
            jax.tree_util.tree_leaves(actual[0]),
            jax.tree_util.tree_leaves(expected[0]),
            strict=True,
        ):
            np.testing.assert_array_equal(actual_leaf, expected_leaf)
        for state_name in ("m", "v"):
            for actual_leaf, expected_leaf in zip(
                jax.tree_util.tree_leaves(actual[1][state_name]),
                jax.tree_util.tree_leaves(expected[1][state_name]),
                strict=True,
            ):
                np.testing.assert_array_equal(actual_leaf, expected_leaf)
        # The smoke threshold is one step, so every feature is now either
        # observed active (age zero) or eligible as dead (age one).
        auxk_row = np.asarray(actual[2]["auxk_row"])
        self.assertAlmostEqual(float(auxk_row.sum()), 1.0, places=6)
        self.assertTrue(np.all(np.isin(np.asarray(actual[1]["dead_steps"]), (0, 1))))

    def test_flop_rule_bills_only_the_reverse_auxiliary_cohort(self) -> None:
        config = resolved("smoke")
        breakdown = trainer.traced_flops(config, trainer.init_params(config, 3))
        tokens = config.seq_len
        hidden = config.mlp_mult * config.d_model
        parent = config.layers * 12 * tokens * config.d_model * hidden
        ghost = (
            config.layers
            * 6
            * tokens
            * config.d_model
            * hidden
            // (config.mlp_top_k // config.aux_k)
        )
        self.assertEqual(
            breakdown.by_site["_choicewise_fuzzy_topk_auxk_mlp"], parent + ghost
        )

    def test_log_and_optimizer_state_widths_are_explicit(self) -> None:
        config = resolved("smoke")
        base_width = len(trainer.training_log_columns())
        self.assertEqual(
            len(trainer.fuzzy_training_log_columns(config)),
            base_width + len(trainer.AUXK_STAT_NAMES),
        )
        optimizer = trainer.init_optimizer(
            trainer.init_params(config, 3),
            config.steps,
            auxk_shape=(config.layers, config.mlp_mult * config.d_model),
        )
        self.assertEqual(optimizer["history"].shape[1], 3 + len(trainer.AUXK_STAT_NAMES))
        self.assertEqual(
            optimizer["dead_steps"].shape,
            (config.layers, config.mlp_mult * config.d_model),
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
