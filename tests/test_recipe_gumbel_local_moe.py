"""Scientific and neutral-path gates for the Gumbel-local MoE fork."""

from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.util
import os
from pathlib import Path
import sys
import unittest


os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).parents[1]


def _load(name: str, recipe: str):
    path = ROOT / "recipes" / recipe / "train.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - import invariant
        raise RuntimeError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


trainer = _load("gumbel_local_moe_train", "gumbel_local_moe")
reference = _load("gumbel_local_moe_reference_train", "reference_moe")


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
    def test_local_steps_default_to_two_and_zero_is_explicit(self) -> None:
        from rig.plan import validate_recipe_plan

        default = _resolved_smoke()
        self.assertEqual(default.local_moe_steps, 2)
        self.assertEqual(
            _resolved_smoke("--local-moe-steps", "0").local_moe_steps,
            0,
        )
        self.assertEqual(trainer.RECIPE_NAME, "gumbel_local_moe")
        self.assertEqual(
            trainer.build_parser().parse_args([]).output_dir,
            Path("runs/gumbel_local_moe"),
        )
        validate_recipe_plan(trainer.resolved_plan_metadata(default))
        self.assertEqual(
            trainer.experiment_config_metadata(default)["resolved"]["optimizer"][
                "local_moe_steps"
            ],
            2,
        )

    def test_local_steps_reject_negative_values(self) -> None:
        with self.assertRaisesRegex(
            argparse.ArgumentTypeError, "must be non-negative"
        ):
            trainer.nonnegative_int("-1")


class GumbelRoutingTests(unittest.TestCase):
    def test_noise_changes_only_the_assignment_not_gate_definition(self) -> None:
        logits = jnp.asarray([[2.0, 1.0, 0.0, -1.0]], jnp.float32)
        clean_chosen, _ = trainer.select_top_k_routes(logits, 2)
        chosen, gate = trainer.select_top_k_routes(logits, 2, jax.random.PRNGKey(3))

        np.testing.assert_array_equal(clean_chosen, [[0, 1]])
        np.testing.assert_array_equal(chosen, [[1, 3]])
        expected = jax.nn.softmax(jnp.take_along_axis(logits, chosen, axis=-1))
        np.testing.assert_array_equal(gate, expected)

        repeated = trainer.select_top_k_routes(logits, 2, jax.random.PRNGKey(3))
        np.testing.assert_array_equal(repeated[0], chosen)
        np.testing.assert_array_equal(repeated[1], gate)


class LocalObjectiveTests(unittest.TestCase):
    @staticmethod
    def _config():
        config = trainer.Config.__new__(trainer.Config)
        for name, value in (
            ("layers", 1),
            ("experts", 2),
            ("expert_top_k", 1),
            ("dtype_name", "float32"),
            ("base_depth", 1),
            ("depth_alpha", 0.0),
        ):
            object.__setattr__(config, name, value)
        return config

    def test_objective_contains_descent_and_scale_normalization(self) -> None:
        activations = trainer.MoELocalActivations(
            normalized_inputs=jnp.ones((1, 1, 2, 3), jnp.float32),
            residual_inputs=jnp.zeros((1, 1, 2, 3), jnp.float32),
            outputs=jnp.zeros((1, 1, 2, 3), jnp.float32),
        )
        output_gradients = jnp.ones((1, 1, 2, 3), jnp.float32)
        moments = jnp.ones((1, 3), jnp.float32)
        params = [
            {
                "router_w": jnp.zeros((1, 1), jnp.float32),
                "expert_up_w": jnp.zeros((1,), jnp.float32),
                "expert_up_b": jnp.zeros((1,), jnp.float32),
                "expert_down_w": jnp.zeros((1,), jnp.float32),
                "expert_down_b": jnp.zeros((1,), jnp.float32),
            }
        ]

        def routed(x, router_w, *_):
            output = x * router_w[0, 0]
            return output, jnp.zeros(1), jnp.zeros(1), jnp.zeros(3)

        objective = lambda candidate: trainer.moe_local_objective(
            candidate,
            activations,
            output_gradients,
            moments,
            jax.random.PRNGKey(0),
            self._config(),
            routed,
        )[0]
        loss, gradients = jax.value_and_grad(objective)(params)

        self.assertAlmostEqual(float(loss), 0.5, places=6)
        self.assertAlmostEqual(float(gradients[0]["router_w"][0, 0]), 1.0, places=6)

    def test_second_moment_ema_is_bias_corrected_from_the_first_step(self) -> None:
        current = jnp.asarray([[4.0, 9.0, 16.0]], jnp.float32)
        raw, corrected = trainer.update_moe_local_moments(
            jnp.zeros_like(current),
            current,
            beta2=0.95,
            step=jnp.asarray(1, jnp.int32),
        )
        np.testing.assert_allclose(raw, 0.05 * current, rtol=1.0e-6)
        np.testing.assert_allclose(corrected, current, rtol=1.0e-6)


class LoggingTests(unittest.TestCase):
    def test_each_enabled_layer_adds_four_named_columns(self) -> None:
        from rig import metrics

        config = replace(_resolved_smoke(), layers=2, local_moe_steps=2)
        base_width = 3 + trainer.router_row_width(config)
        columns = trainer.recipe_training_log_columns(config)
        self.assertEqual(len(columns), base_width + 8)
        self.assertEqual(
            [metrics.metric_by_id(column.metric_id).name for column in columns[-8:]],
            list(trainer._LOCAL_MOE_METRICS) * 2,
        )

        losses = jnp.asarray([0.5, 0.25], jnp.float32)
        moments = jnp.asarray([[1.0, 4.0, 9.0], [16.0, 25.0, 36.0]])
        np.testing.assert_array_equal(
            trainer.local_moe_row(config, losses, moments),
            [0.5, 1.0, 2.0, 3.0, 0.25, 4.0, 5.0, 6.0],
        )

    def test_disabled_path_adds_no_columns(self) -> None:
        config = replace(_resolved_smoke(), local_moe_steps=0)
        self.assertEqual(
            len(trainer.recipe_training_log_columns(config)),
            3 + trainer.router_row_width(config),
        )
        self.assertEqual(trainer.local_moe_row_width(config), 0)


class NeutralPathTests(unittest.TestCase):
    def test_zero_local_steps_matches_reference_moe_for_one_update(self) -> None:
        config = replace(
            _resolved_smoke("--local-moe-steps", "0"),
            layers=1,
            batch_size=1,
            steps=1,
        )
        params = trainer.init_params(config, 1350)
        local_optimizer = jax.tree_util.tree_map(
            jnp.asarray, trainer.init_optimizer(params, config, seed=1350)
        )
        reference_optimizer = jax.tree_util.tree_map(
            jnp.asarray, reference.init_optimizer(params, config)
        )
        tokens = jnp.asarray(
            np.random.default_rng(9).integers(
                0,
                config.semantic_vocab_size,
                size=(config.batch_size, config.seq_len + 1),
            )
        )

        local_params, local_optimizer, local_metrics = trainer.train_step(
            params,
            local_optimizer,
            tokens[:, :-1],
            tokens[:, 1:],
            config,
        )
        reference_params, reference_optimizer, reference_metrics = reference.train_step(
            params,
            reference_optimizer,
            tokens[:, :-1],
            tokens[:, 1:],
            config,
        )

        for local, expected in zip(
            jax.tree_util.tree_leaves(local_params),
            jax.tree_util.tree_leaves(reference_params),
            strict=True,
        ):
            np.testing.assert_array_equal(local, expected)
        for name in ("step", "m", "v", "history"):
            local_leaves = jax.tree_util.tree_leaves(local_optimizer[name])
            expected_leaves = jax.tree_util.tree_leaves(reference_optimizer[name])
            for local, expected in zip(local_leaves, expected_leaves, strict=True):
                np.testing.assert_array_equal(local, expected)
        for name in ("loss", "grad_norm", "learning_rate", "router_row"):
            np.testing.assert_array_equal(local_metrics[name], reference_metrics[name])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
