"""Optimizer-contract tests for the expert-load-scaled MoE fork."""

from __future__ import annotations

from dataclasses import replace
import importlib.util
import os
from pathlib import Path
import sys
import unittest
from unittest import mock


os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402


TRAINER_PATH = (
    Path(__file__).parents[1] / "recipes" / "expert_load_moe" / "train.py"
)
SPEC = importlib.util.spec_from_file_location("expert_load_moe_train", TRAINER_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - importlib invariant
    raise RuntimeError(f"could not import {TRAINER_PATH}")
trainer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = trainer
SPEC.loader.exec_module(trainer)


def _resolved_smoke(*overrides: str):
    parser = trainer.build_parser()
    args = parser.parse_args(["--profile", "smoke", *overrides])
    document, digest = trainer.load_experiment_config("smoke")
    trainer.validate_args(args, document)
    return trainer.resolve_config(
        args,
        "cpu",
        experiment_config=document,
        config_sha256=digest,
    )


class ConfigurationTests(unittest.TestCase):
    def test_fork_defaults_are_explicit_and_overridable(self) -> None:
        default = _resolved_smoke()
        overridden = _resolved_smoke(
            "--expert-load-scaling-mode",
            "gradient",
            "--expert-load-scaling-strength",
            "0",
        )

        self.assertEqual(trainer.RECIPE_NAME, "expert_load_moe")
        self.assertEqual(default.config_schema_version, 7)
        self.assertEqual(default.expert_load_scaling_mode, "update")
        self.assertEqual(default.expert_load_scaling_strength, 0.5)
        self.assertEqual(overridden.expert_load_scaling_mode, "gradient")
        self.assertEqual(overridden.expert_load_scaling_strength, 0.0)
        self.assertEqual(
            trainer.build_parser().parse_args([]).output_dir,
            Path("runs/expert_load_moe"),
        )

        metadata = trainer.experiment_config_metadata(overridden)
        scaling = metadata["resolved"]["optimizer"]["expert_load_scaling"]
        self.assertEqual(scaling["mode"], "gradient")
        self.assertEqual(scaling["strength"], 0.0)
        self.assertIn("current_global_batch", scaling["load_statistic"])

    def test_cli_strength_rejects_values_outside_the_interpolation(self) -> None:
        parser = trainer.build_parser()
        document, _ = trainer.load_experiment_config("smoke")
        for value in ("-0.01", "1.01", "nan"):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "between 0 and 1"
            ):
                trainer.validate_args(
                    parser.parse_args(
                        ["--profile", "smoke", "--expert-load-scaling-strength", value]
                    ),
                    document,
                )


class ScaleRuleTests(unittest.TestCase):
    def test_rule_is_balanced_at_one_and_strength_softens_the_floor(self) -> None:
        balanced = jnp.full((2, 8), 1.0 / 8.0, jnp.float32)
        np.testing.assert_array_equal(
            trainer.expert_load_scale_factors(
                balanced, experts=8, strength=1.0
            ),
            np.ones((2, 8), np.float32),
        )

        collapsed = jnp.zeros((1, 8), jnp.float32).at[0, 0].set(1.0)
        full = trainer.expert_load_scale_factors(
            collapsed, experts=8, strength=1.0
        )
        softened = trainer.expert_load_scale_factors(
            collapsed, experts=8, strength=0.5
        )
        self.assertAlmostEqual(float(full[0, 0]), np.sqrt(8.0), places=6)
        self.assertEqual(float(full[0, 1]), 0.0)
        self.assertAlmostEqual(
            float(softened[0, 0]), 1.0 + 0.5 * (np.sqrt(8.0) - 1.0), places=6
        )
        self.assertEqual(float(softened[0, 1]), 0.5)

    def test_scale_tree_touches_only_each_expert_leading_axis(self) -> None:
        params = {
            "token_embedding": jnp.ones((5, 3), jnp.float32),
            "blocks": [
                {
                    "router_w": jnp.ones((3, 2), jnp.float32),
                    "expert_up_w": jnp.ones((2, 3, 4), jnp.float32),
                    "expert_up_b": jnp.ones((2, 4), jnp.float32),
                    "expert_down_w": jnp.ones((2, 4, 3), jnp.float32),
                    "expert_down_b": jnp.ones((2, 3), jnp.float32),
                    "ln2_scale": jnp.ones((3,), jnp.float32),
                }
            ],
        }
        factors = jnp.asarray([[0.5, 1.5]], jnp.float32)
        scales = trainer.expert_load_scale_tree(params, factors)
        scaled = trainer.apply_expert_load_scaling(params, scales)

        np.testing.assert_array_equal(scaled["token_embedding"], 1.0)
        np.testing.assert_array_equal(scaled["blocks"][0]["router_w"], 1.0)
        np.testing.assert_array_equal(scaled["blocks"][0]["ln2_scale"], 1.0)
        for name in trainer._EXPERT_PARAMETER_NAMES:
            np.testing.assert_array_equal(scaled["blocks"][0][name][0], 0.5)
            np.testing.assert_array_equal(scaled["blocks"][0][name][1], 1.5)

    def test_unclassified_expert_parameter_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "no rule for parameter"):
            trainer.expert_load_scale_tree(
                {"blocks": [{"expert_new_w": jnp.ones((2, 1), jnp.float32)}]},
                jnp.ones((1, 2), jnp.float32),
            )


class ExpertDiagnosticTests(unittest.TestCase):
    @staticmethod
    def _trees():
        params = {
            "token_embedding": jnp.arange(15, dtype=jnp.float32).reshape(5, 3),
            "blocks": [
                {
                    "router_w": jnp.arange(6, dtype=jnp.float32).reshape(3, 2),
                    "expert_up_w": jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4),
                    "expert_up_b": jnp.arange(8, dtype=jnp.float32).reshape(2, 4),
                    "expert_down_w": jnp.arange(24, dtype=jnp.float32).reshape(2, 4, 3),
                    "expert_down_b": jnp.arange(6, dtype=jnp.float32).reshape(2, 3),
                    "ln2_scale": jnp.ones((3,), jnp.float32),
                }
            ],
            "final_ln_scale": jnp.ones((3,), jnp.float32),
        }
        gradients = jax.tree_util.tree_map(lambda value: value * 0.1, params)
        after = jax.tree_util.tree_map(lambda value: value + 0.25, params)
        return params, gradients, after

    def test_each_expert_gets_the_full_extended_diagnostic_grid(self) -> None:
        params, gradients, after = self._trees()
        metadata = trainer.diagnostic_scope_metadata(
            params, include_experts=True
        )
        values = np.asarray(
            trainer.diagnostic_values(
                params,
                gradients,
                after,
                include_experts=True,
                statistics=trainer.DIAGNOSTIC_EXTENDED_STATS,
            )
        )

        experts = [scope for scope in metadata if scope.scope == "expert"]
        self.assertEqual(
            [(scope.layer, scope.index, scope.element_count) for scope in experts],
            [(0, 0, 31), (0, 1, 31)],
        )
        self.assertEqual(
            values.shape,
            (
                len(metadata),
                len(trainer.DIAGNOSTIC_FAMILIES),
                len(trainer.DIAGNOSTIC_EXTENDED_STATS),
            ),
        )

        columns = trainer.diagnostic_log_columns(
            metadata, statistics=trainer.DIAGNOSTIC_EXTENDED_STATS
        )
        descriptions = {column.describe() for column in columns}
        self.assertIn("block[0]/expert[0]/grad.l1_norm", descriptions)
        self.assertIn("block[0]/expert[1]/update.p99", descriptions)

        expert_zero_scope = next(
            index
            for index, scope in enumerate(metadata)
            if scope.scope == "expert" and scope.index == 0
        )
        p50 = trainer.DIAGNOSTIC_EXTENDED_STATS.index("p50")
        expert_zero_after = np.concatenate(
            [
                np.asarray(after["blocks"][0][name][0]).reshape(-1)
                for name in sorted(
                    name
                    for name in after["blocks"][0]
                    if name.startswith("expert_")
                )
            ]
        )
        np.testing.assert_allclose(
            values[expert_zero_scope, 0, p50],
            np.percentile(expert_zero_after, 50),
            atol=2.0e-6,
        )


class OptimizerModeTests(unittest.TestCase):
    @staticmethod
    def _state(mode: str, strength: float):
        config = replace(
            _resolved_smoke(),
            layers=1,
            experts=2,
            steps=4,
            warmup_steps=0,
            min_lr_ratio=1.0,
            learning_rate=0.1,
            weight_decay=0.0,
            adam_epsilon=1.0e-8,
            batch_multiplier=1.0,
            data_multiplier=1.0,
            expert_load_scaling_mode=mode,
            expert_load_scaling_strength=strength,
        )
        params = {
            "blocks": [
                {
                    "router_w": jnp.ones((1, 2), jnp.float32),
                    "expert_up_w": jnp.ones((2, 1), jnp.float32),
                }
            ]
        }
        optimizer = jax.tree_util.tree_map(
            jnp.asarray, trainer.init_optimizer(params, config)
        )
        return config, params, optimizer

    @staticmethod
    def _step(config, params, optimizer, load):
        router = trainer.RouterStats(
            balance_loss=jnp.asarray(1.0, jnp.float32),
            load=jnp.asarray([load], jnp.float32),
            summary=jnp.zeros((1, 3), jnp.float32),
        )

        def objective(candidate, *_args, **_kwargs):
            loss = sum(jnp.sum(leaf) for leaf in jax.tree_util.tree_leaves(candidate))
            return loss, router

        decay_mask = jax.tree_util.tree_map(lambda _value: False, params)
        with mock.patch.object(trainer, "cross_entropy_and_router", objective):
            return trainer._apply_training_update(
                params,
                optimizer,
                jnp.zeros((1, 1), jnp.int32),
                jnp.zeros((1, 1), jnp.int32),
                config,
                decay_mask,
            )

    def test_update_mode_scales_updates_while_gradient_mode_scales_moments(self) -> None:
        baseline_config, params, baseline_optimizer = self._state("update", 0.0)
        update_config, _, update_optimizer = self._state("update", 1.0)
        gradient_config, _, gradient_optimizer = self._state("gradient", 1.0)
        load = (0.8, 0.2)

        baseline_params, baseline_state, _, baseline_raw = self._step(
            baseline_config, params, baseline_optimizer, load
        )
        update_params, update_state, _, update_raw = self._step(
            update_config, params, update_optimizer, load
        )
        gradient_params, gradient_state, _, gradient_raw = self._step(
            gradient_config, params, gradient_optimizer, load
        )

        factors = np.sqrt(2.0 * np.asarray(load, np.float32))[:, None]
        baseline_delta = 1.0 - np.asarray(
            baseline_params["blocks"][0]["expert_up_w"]
        )
        update_delta = 1.0 - np.asarray(
            update_params["blocks"][0]["expert_up_w"]
        )
        gradient_delta = 1.0 - np.asarray(
            gradient_params["blocks"][0]["expert_up_w"]
        )
        np.testing.assert_allclose(update_delta, baseline_delta * factors, rtol=1e-6)
        # A constant scalar cancels between Adam's first and second moments.
        np.testing.assert_allclose(gradient_delta, baseline_delta, rtol=1e-6)

        np.testing.assert_array_equal(
            update_state["m"]["blocks"][0]["expert_up_w"],
            baseline_state["m"]["blocks"][0]["expert_up_w"],
        )
        np.testing.assert_allclose(
            gradient_state["m"]["blocks"][0]["expert_up_w"],
            baseline_state["m"]["blocks"][0]["expert_up_w"] * factors,
        )
        np.testing.assert_allclose(
            gradient_state["v"]["blocks"][0]["expert_up_w"],
            baseline_state["v"]["blocks"][0]["expert_up_w"] * factors**2,
        )
        # Sparse diagnostics remain gradients of the objective, not optimizer
        # inputs modified by this ablation.
        for raw in (baseline_raw, update_raw, gradient_raw):
            np.testing.assert_array_equal(
                raw["blocks"][0]["expert_up_w"], np.ones((2, 1), np.float32)
            )
        np.testing.assert_array_equal(
            update_params["blocks"][0]["router_w"],
            baseline_params["blocks"][0]["router_w"],
        )
        np.testing.assert_array_equal(
            gradient_params["blocks"][0]["router_w"],
            baseline_params["blocks"][0]["router_w"],
        )

    def test_gradient_mode_reacts_when_load_changes(self) -> None:
        config, params, optimizer = self._state("gradient", 1.0)
        first_params, first_state, _, _ = self._step(
            config, params, optimizer, (0.8, 0.2)
        )
        stable_params, _, _, _ = self._step(
            config, first_params, first_state, (0.8, 0.2)
        )
        switched_params, _, _, _ = self._step(
            config, first_params, first_state, (0.2, 0.8)
        )

        stable_delta = np.asarray(
            first_params["blocks"][0]["expert_up_w"]
            - stable_params["blocks"][0]["expert_up_w"]
        )
        switched_delta = np.asarray(
            first_params["blocks"][0]["expert_up_w"]
            - switched_params["blocks"][0]["expert_up_w"]
        )
        self.assertGreater(float(np.abs(stable_delta - switched_delta).max()), 1.0e-3)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
