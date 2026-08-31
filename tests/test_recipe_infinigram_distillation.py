"""Objective and configuration contracts for infinigram distillation."""

from __future__ import annotations

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

from rig.plan import validate_recipe_plan  # noqa: E402


ROOT = Path(__file__).parents[1]
TRAINER_PATH = ROOT / "recipes" / "infinigram_distillation" / "train.py"
SPEC = importlib.util.spec_from_file_location(
    "infinigram_distillation_train", TRAINER_PATH
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


class InfiniGramDistillationRecipeTests(unittest.TestCase):
    def test_baseline_defaults_are_neutral_and_plan_remains_standard(self) -> None:
        config = resolved()
        self.assertEqual(config.ground_truth_weight, 1.0)
        self.assertEqual(config.infinigram_weight, 0.0)
        self.assertEqual(config.infinigram_max_context, 0)
        validate_recipe_plan(trainer.resolved_plan_metadata(config))

    def test_treatment_defaults_ground_truth_weight_to_one_minus_b(self) -> None:
        config = resolved(
            "smoke",
            "--infinigram-weight",
            "0.25",
            "--infinigram-index",
            "/index/not-opened-during-plan-resolution",
            "--infinigram-max-context",
            "64",
        )
        self.assertEqual(config.ground_truth_weight, 0.75)
        self.assertEqual(config.infinigram_weight, 0.25)
        self.assertEqual(config.infinigram_max_context, 64)
        validate_recipe_plan(trainer.resolved_plan_metadata(config))

    def test_objective_scale_and_index_contracts_are_rejected_early(self) -> None:
        parser = trainer.build_parser()
        document, _ = trainer.load_experiment_config("smoke")
        cases = (
            ["--profile", "smoke", "--infinigram-weight", "0.25"],
            [
                "--profile",
                "smoke",
                "--ground-truth-weight",
                "1",
                "--infinigram-weight",
                "0.25",
                "--infinigram-index",
                "/unused",
            ],
        )
        for argv in cases:
            with self.subTest(argv=argv), self.assertRaises(ValueError):
                trainer.validate_args(parser.parse_args(argv), document)

    def test_dense_treatment_value_and_gradients_equal_two_cross_entropies(self) -> None:
        baseline = resolved()
        treatment = replace(
            baseline, ground_truth_weight=0.75, infinigram_weight=0.25
        )
        params = jax.tree_util.tree_map(jnp.asarray, trainer.init_params(baseline, 7))
        positions = jnp.arange(
            baseline.batch_size * baseline.seq_len, dtype=jnp.int32
        ).reshape((baseline.batch_size, baseline.seq_len))
        x = positions % baseline.semantic_vocab_size
        y = (positions + 1) % baseline.semantic_vocab_size
        teacher_y = (positions * 7 + 3) % baseline.semantic_vocab_size

        def actual(candidate):
            return trainer.cross_entropy(
                candidate,
                x,
                y,
                treatment,
                distill_targets=teacher_y,
            )

        def expected(candidate):
            return 0.75 * trainer.cross_entropy(
                candidate, x, y, baseline
            ) + 0.25 * trainer.cross_entropy(
                candidate, x, teacher_y, baseline
            )

        actual_value, actual_grad = jax.value_and_grad(actual)(params)
        expected_value, expected_grad = jax.value_and_grad(expected)(params)
        np.testing.assert_allclose(actual_value, expected_value, rtol=2e-6, atol=2e-6)
        for actual_leaf, expected_leaf in zip(
            jax.tree_util.tree_leaves(actual_grad),
            jax.tree_util.tree_leaves(expected_grad),
            strict=True,
        ):
            np.testing.assert_allclose(
                actual_leaf, expected_leaf, rtol=3e-6, atol=3e-6
            )

    def test_teacher_seed_is_stable_and_separates_steps_and_ranks(self) -> None:
        seed = trainer.infinigram_step_seed(17, 23, 2)
        self.assertEqual(seed, trainer.infinigram_step_seed(17, 23, 2))
        self.assertNotEqual(seed, trainer.infinigram_step_seed(17, 24, 2))
        self.assertNotEqual(seed, trainer.infinigram_step_seed(17, 23, 3))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
