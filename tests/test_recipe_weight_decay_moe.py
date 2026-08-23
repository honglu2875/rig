"""Configuration-contract tests for the MoE weight-decay sweep fork."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import unittest


os.environ.setdefault("JAX_PLATFORMS", "cpu")


TRAINER_PATH = (
    Path(__file__).parents[1] / "recipes" / "weight_decay_moe" / "train.py"
)
SPEC = importlib.util.spec_from_file_location("weight_decay_moe_train", TRAINER_PATH)
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
    def test_default_is_source_value_and_override_is_recorded(self) -> None:
        default = _resolved_smoke()
        overridden = _resolved_smoke("--weight-decay", "0.03")
        disabled = _resolved_smoke("--weight-decay", "0")

        self.assertEqual(trainer.RECIPE_NAME, "weight_decay_moe")
        self.assertEqual(default.weight_decay, 0.1)
        self.assertEqual(overridden.weight_decay, 0.03)
        self.assertEqual(disabled.weight_decay, 0.0)
        self.assertEqual(
            trainer.build_parser().parse_args([]).output_dir,
            Path("runs/weight_decay_moe"),
        )

        metadata = trainer.experiment_config_metadata(overridden)
        self.assertEqual(
            metadata["resolved"]["optimizer"]["weight_decay"],
            0.03,
        )

    def test_invalid_override_is_rejected(self) -> None:
        parser = trainer.build_parser()
        document, _ = trainer.load_experiment_config("smoke")
        for value in ("-0.01", "nan", "inf"):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "finite and nonnegative"
            ):
                trainer.validate_args(
                    parser.parse_args(
                        ["--profile", "smoke", "--weight-decay", value]
                    ),
                    document,
                )

    def test_default_fork_matches_reference_scientific_config(self) -> None:
        document, digest = trainer.load_experiment_config("smoke")
        config = _resolved_smoke()

        self.assertEqual(config.config_sha256, digest)
        self.assertEqual(config.weight_decay, document.run.optimizer.weight_decay)
        self.assertEqual(config.router_aux_coefficient, 0.01)
        self.assertEqual(config.experts, 8)
        self.assertEqual(config.expert_top_k, 2)

    def test_declared_study_grid_resolves_exactly(self) -> None:
        parser = trainer.build_parser()
        document, digest = trainer.load_experiment_config("dev")

        for tier, expected_steps in (("60m", 2286), ("125m", 4709)):
            for weight_decay in (
                "0",
                "0.03",
                "0.1",
                "0.3",
                "0.4",
                "0.5",
                "0.6",
                "0.8",
            ):
                with self.subTest(tier=tier, weight_decay=weight_decay):
                    args = parser.parse_args(
                        [
                            "--profile",
                            "dev",
                            "--tier",
                            tier,
                            "--context",
                            "8k",
                            "--tokens-per-parameter",
                            "5",
                            "--batch-size",
                            "16",
                            "--base-learning-rate",
                            "0.00390625",
                            "--weight-decay",
                            weight_decay,
                        ]
                    )
                    trainer.validate_args(args, document)
                    config = trainer.resolve_config(
                        args,
                        "tpu",
                        experiment_config=document,
                        config_sha256=digest,
                    )

                    self.assertEqual(config.steps, expected_steps)
                    self.assertEqual(config.seq_len, 8192)
                    self.assertEqual(config.batch_size, 16)
                    self.assertEqual(config.learning_rate, 0.00390625)
                    self.assertEqual(config.weight_decay, float(weight_decay))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
