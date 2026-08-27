"""Configuration and FLOP contracts for the double-fuzzy TopK fork."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from rig.plan import validate_recipe_plan  # noqa: E402


ROOT = Path(__file__).parents[1]
TRAINER_PATH = ROOT / "recipes" / "double_fuzzy_topk_autoencoder" / "train.py"
SPEC = importlib.util.spec_from_file_location(
    "double_fuzzy_topk_autoencoder_train", TRAINER_PATH
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


class DoubleFuzzyTopKRecipeTests(unittest.TestCase):
    def test_dev_defaults_are_the_requested_two_stage_coordinate(self) -> None:
        config = resolved("dev", "--tier", "60m")
        self.assertEqual(config.context_preset, "8k")
        self.assertEqual(config.seq_len, 8192)
        self.assertTrue(config.document_masking)
        self.assertEqual(config.mlp_mult, 16)
        self.assertEqual(config.mlp_top_k, 4 * config.d_model)
        self.assertEqual(config.mlp_input_group_size, 4)
        self.assertEqual(config.sparse_mlp_backend, "choicewise")
        self.assertEqual(config.sparse_mlp_token_block, 32)
        self.assertEqual(config.sparse_mlp_output_block, 128)
        self.assertEqual(config.declared_parameters, 102_440_832)
        self.assertEqual(config.steps, 3_908)
        validate_recipe_plan(trainer.resolved_plan_metadata(config))

    def test_width_override_preserves_head_width_and_recomputes_budget(self) -> None:
        config = resolved(
            "dev",
            "--tier",
            "60m",
            "--sparse-d-model",
            "512",
            "--sparse-top-k",
            "2048",
        )
        self.assertEqual(config.d_model, 512)
        self.assertEqual(config.heads, 8)
        self.assertEqual(config.d_model // config.heads, 64)
        self.assertEqual(config.mlp_top_k, 4 * config.d_model)
        self.assertGreater(config.declared_parameters, 102_440_832)
        validate_recipe_plan(trainer.resolved_plan_metadata(config))

    def test_recipe_local_overrides_preserve_both_integral_groupings(self) -> None:
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
            "--sparse-input-group-size",
            "8",
            "--sparse-mlp-backend",
            "reference",
            "--sparse-training-steps",
            "2267",
        )
        self.assertEqual(config.layers, 13)
        self.assertEqual(config.mlp_mult, 8)
        self.assertEqual(config.mlp_top_k, 768)
        self.assertEqual(config.mlp_input_group_size, 8)
        self.assertEqual(config.sparse_mlp_backend, "reference")
        self.assertEqual(config.declared_parameters, 77_047_296)
        self.assertEqual(config.steps, 2_267)
        self.assertEqual(config.data_multiplier, 1.0)
        validate_recipe_plan(trainer.resolved_plan_metadata(config))

    def test_override_rejects_nonintegral_groups_and_head_width(self) -> None:
        with self.assertRaisesRegex(ValueError, "must divide"):
            resolved("dev", "--tier", "60m", "--sparse-top-k", "1000")
        with self.assertRaisesRegex(ValueError, "must divide"):
            resolved(
                "dev",
                "--tier",
                "60m",
                "--sparse-input-group-size",
                "5",
            )
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            resolved(
                "dev",
                "--tier",
                "60m",
                "--sparse-d-model",
                "400",
            )

    def test_reference_flop_rules_separate_active_and_issued_work(self) -> None:
        config = resolved("smoke")
        params = trainer.init_params(config, 3)
        physical = trainer.traced_flops(config, params)
        active = trainer.active_traced_flops(config, params)
        scope = "_reference_double_fuzzy_topk_mlp"
        tokens = config.seq_len
        d = config.d_model
        h = config.mlp_mult * d
        q = d // config.mlp_input_group_size
        k = config.mlp_top_k
        expected_physical_per_layer = 2 * tokens * (q * h + k * d) + 2 * (
            tokens * d * (3 * h + k)
        )
        expected_active_per_layer = 2 * tokens * (q * h + k * d) + 4 * (
            tokens * k * (q + d)
        )
        self.assertEqual(
            physical.by_site[scope],
            config.layers * expected_physical_per_layer,
        )
        self.assertEqual(
            active.by_site[scope],
            config.layers * expected_active_per_layer,
        )
        self.assertGreater(physical.by_site[scope], active.by_site[scope])
        self.assertFalse(
            [
                warning
                for warning in (*physical.warnings, *active.warnings)
                if "double_fuzzy" in warning
            ]
        )

    def test_pallas_up_flop_rule_keeps_the_large_k_loop_out_of_reverse(
        self,
    ) -> None:
        config = resolved("smoke", "--sparse-mlp-backend", "pallas_up")
        params = trainer.init_params(config, 4)
        physical = trainer.traced_flops(config, params)
        scope = "_pallas_up_double_fuzzy_topk_mlp"
        tokens = config.seq_len
        d = config.d_model
        h = config.mlp_mult * d
        q = d // config.mlp_input_group_size
        expected_per_layer = 2 * tokens * h * (q + d) + 8 * tokens * d * h
        self.assertEqual(
            physical.by_site[scope],
            config.layers * expected_per_layer,
        )
        self.assertFalse(
            [warning for warning in physical.warnings if "double_fuzzy" in warning]
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
