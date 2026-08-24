"""Configuration and FLOP contracts for the sparse-autoencoder fork."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from rig.plan import validate_recipe_plan  # noqa: E402


ROOT = Path(__file__).parents[1]
TRAINER_PATH = ROOT / "recipes" / "sparse_autoencoder" / "train.py"
SPEC = importlib.util.spec_from_file_location("sparse_autoencoder_train", TRAINER_PATH)
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


class SparseAutoencoderRecipeTests(unittest.TestCase):
    def test_dev_defaults_to_8k_16x_and_k128(self) -> None:
        config = resolved("dev", "--tier", "60m")
        self.assertEqual(config.context_preset, "8k")
        self.assertEqual(config.seq_len, 8192)
        self.assertTrue(config.document_masking)
        self.assertEqual(config.mlp_mult, 16)
        self.assertEqual(config.mlp_top_k, 128)
        self.assertEqual(config.sparse_mlp_backend, "reference")
        self.assertEqual(config.sparse_mlp_token_block, 8)
        self.assertEqual(config.declared_parameters, 102_440_832)
        self.assertEqual(config.steps, 3_908)
        validate_recipe_plan(trainer.resolved_plan_metadata(config))

    def test_recipe_local_overrides_change_only_the_intended_contract(self) -> None:
        config = resolved(
            "dev",
            "--tier",
            "60m",
            "--sparse-mlp-mult",
            "8",
            "--sparse-top-k",
            "64",
            "--sparse-mlp-backend",
            "reference",
            "--sparse-mlp-output-block",
            "384",
        )
        self.assertEqual(config.mlp_mult, 8)
        self.assertEqual(config.mlp_top_k, 64)
        self.assertEqual(config.sparse_mlp_backend, "reference")
        self.assertEqual(config.sparse_mlp_output_block, 384)
        self.assertEqual(config.declared_parameters, 74_092_416)
        self.assertEqual(config.data_multiplier, 1.0)

    def test_stored_parameter_formula_tracks_dictionary_width(self) -> None:
        tier = trainer.load_experiment_config("smoke")[0].family.tiers["smoke"]
        model = tier.model
        expected = (
            2 * model.vocab_size * model.d_model
            + model.layers
            * (
                (4 + 2 * model.mlp_mult) * model.d_model**2
                + (model.mlp_mult + 7) * model.d_model
            )
            + model.d_model
        )
        self.assertEqual(tier.tpp_parameters, expected)

    def test_sparse_flop_rule_counts_forward_and_vjp_contract(self) -> None:
        config = resolved("smoke")
        params = trainer.init_params(config, 3)
        breakdown = trainer.traced_flops(config, params)
        sparse = breakdown.by_site["sparse_topk_mlp"]
        tokens = config.seq_len
        d = config.d_model
        h = config.mlp_mult * d
        k = config.mlp_top_k
        expected_per_layer = 2 * tokens * d * h + 10 * tokens * k * d
        self.assertEqual(sparse, config.layers * expected_per_layer)
        self.assertFalse(
            [warning for warning in breakdown.warnings if "sparse_topk" in warning]
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
