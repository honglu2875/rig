"""Recipe-level contracts for reconstruction and literal AuxK study arms."""

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

from rig.nn import parameter_count  # noqa: E402
from rig.plan import validate_recipe_plan  # noqa: E402


ROOT = Path(__file__).parents[1]


def load_trainer(recipe: str):
    path = ROOT / "recipes" / recipe / "train.py"
    name = f"test_{recipe}_train"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - import invariant
        raise RuntimeError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


parent = load_trainer("fuzzy_topk_autoencoder")
reconstruction = load_trainer("fuzzy_topk_reconstruction")
auxk = load_trainer("fuzzy_topk_reconstruction_auxk")


def resolved(module, profile: str = "smoke", *overrides: str):
    args = module.build_parser().parse_args(["--profile", profile, *overrides])
    document, digest = module.load_experiment_config(profile)
    module.validate_args(args, document)
    return module.resolve_config(
        args,
        "cpu" if profile == "smoke" else "tpu",
        experiment_config=document,
        config_sha256=digest,
    )


class FuzzyTopKReconstructionRecipeTests(unittest.TestCase):
    def test_separate_arm_defaults_and_plan_contracts(self) -> None:
        rec = resolved(reconstruction, "dev", "--tier", "60m")
        treatment = resolved(auxk, "dev", "--tier", "60m")

        for config in (rec, treatment):
            self.assertEqual(config.mlp_mult, 16)
            self.assertEqual(config.mlp_top_k, 4 * config.d_model)
            self.assertEqual(config.aux_k, config.d_model // 2)
            self.assertEqual(config.auxk_cohort_count, 8)
            self.assertEqual(config.reconstruction_coefficient, 1.0)
            self.assertEqual(config.dead_after_steps, 77)
        self.assertEqual(rec.auxk_mode, "none")
        self.assertEqual(rec.auxk_coefficient, 0.0)
        self.assertEqual(treatment.auxk_mode, "auxk")
        self.assertEqual(treatment.auxk_coefficient, 1.0 / 32.0)
        validate_recipe_plan(reconstruction.resolved_plan_metadata(rec))
        validate_recipe_plan(auxk.resolved_plan_metadata(treatment))

    def test_parent_rng_and_deployed_forward_are_bit_identical(self) -> None:
        rec_config = resolved(reconstruction)
        parent_config = resolved(
            parent,
            "smoke",
            "--sparse-mlp-mult",
            str(rec_config.mlp_mult),
            "--sparse-top-k",
            str(rec_config.mlp_top_k),
        )
        rec_params = reconstruction.init_params(rec_config, 17)
        parent_params = parent.init_params(parent_config, 17)
        deployed = reconstruction.deployment_params(rec_params)

        self.assertEqual(
            jax.tree_util.tree_structure(deployed),
            jax.tree_util.tree_structure(parent_params),
        )
        for actual, expected in zip(
            jax.tree_util.tree_leaves(deployed),
            jax.tree_util.tree_leaves(parent_params),
            strict=True,
        ):
            np.testing.assert_array_equal(actual, expected)

        tokens = jnp.arange(
            rec_config.batch_size * rec_config.seq_len, dtype=jnp.int32
        ).reshape((rec_config.batch_size, rec_config.seq_len))
        rec_logits = reconstruction.gpt_logits(deployed, tokens, rec_config)
        parent_logits = parent.gpt_logits(parent_params, tokens, parent_config)
        np.testing.assert_array_equal(rec_logits, parent_logits)

    def test_decoder_is_tied_by_direction_unit_norm_and_train_only(self) -> None:
        config = resolved(reconstruction)
        params = reconstruction.init_params(config, 23)
        deployed = reconstruction.deployment_params(params)
        self.assertEqual(
            parameter_count(params) - parameter_count(deployed),
            config.reconstruction_parameter_count,
        )

        decay = reconstruction.weight_decay_mask(params)
        for block, deployed_block, decay_block in zip(
            params["blocks"], deployed["blocks"], decay["blocks"], strict=True
        ):
            expected = block["mlp_up_w"].T.copy()
            expected /= np.linalg.norm(expected, axis=-1, keepdims=True)
            np.testing.assert_allclose(
                block["mlp_reconstruction_w"], expected, rtol=1e-6, atol=1e-7
            )
            np.testing.assert_allclose(
                np.linalg.norm(block["mlp_reconstruction_w"], axis=-1),
                np.ones((block["mlp_reconstruction_w"].shape[0],)),
                rtol=1e-6,
                atol=1e-6,
            )
            self.assertNotIn("mlp_reconstruction_w", deployed_block)
            self.assertFalse(decay_block["mlp_reconstruction_w"])
            self.assertTrue(decay_block["mlp_up_w"])
            self.assertTrue(decay_block["mlp_down_w"])

    def test_tangent_projection_update_and_log_layout(self) -> None:
        config = resolved(reconstruction)
        params = reconstruction.init_params(config, 29)
        ones = jax.tree_util.tree_map(lambda value: jnp.ones_like(value), params)
        projected = reconstruction.project_reconstruction_decoder_gradients(
            params, ones
        )
        for block, gradient_block in zip(
            params["blocks"], projected["blocks"], strict=True
        ):
            radial = jnp.sum(
                block["mlp_reconstruction_w"]
                * gradient_block["mlp_reconstruction_w"],
                axis=-1,
            )
            np.testing.assert_allclose(radial, np.zeros_like(radial), atol=2e-5)

        optimizer = jax.tree_util.tree_map(
            jnp.asarray,
            reconstruction.init_optimizer(params, config.steps, config=config),
        )
        tokens = jnp.zeros((config.batch_size, config.seq_len), jnp.int32)
        targets = jnp.ones_like(tokens)
        kernel_config = reconstruction.FuzzyTopKReconstructionConfig(
            top_k=config.mlp_top_k
        )

        def reconstruction_fn(*operands):
            return reconstruction.fuzzy_topk_mlp_with_reconstruction(
                *operands, config=kernel_config
            )

        updated, optimizer, metrics = reconstruction.train_step(
            params,
            optimizer,
            tokens,
            targets,
            config,
            reconstruction.weight_decay_mask(params),
            None,
            reconstruction_fn,
        )
        for block in updated["blocks"]:
            norms = jnp.linalg.norm(block["mlp_reconstruction_w"], axis=-1)
            np.testing.assert_allclose(norms, np.ones_like(norms), atol=2e-6)
        self.assertGreater(float(metrics["objective"]), float(metrics["loss"]))
        self.assertEqual(
            optimizer["history"].shape[1],
            len(reconstruction.fuzzy_training_log_columns(config)),
        )

    def test_auxk_updates_dead_age_from_main_activity(self) -> None:
        config = resolved(auxk)
        params = auxk.init_params(config, 31)
        optimizer = jax.tree_util.tree_map(
            jnp.asarray, auxk.init_optimizer(params, config.steps, config=config)
        )
        tokens = jnp.zeros((config.batch_size, config.seq_len), jnp.int32)
        targets = jnp.ones_like(tokens)
        kernel_config = auxk.FuzzyTopKReconstructionConfig(
            top_k=config.mlp_top_k, aux_k=config.aux_k
        )

        def auxk_fn(*operands):
            return auxk.fuzzy_topk_mlp_with_reconstruction_auxk(
                *operands, config=kernel_config
            )

        _updated, optimizer, metrics = auxk.train_step(
            params,
            optimizer,
            tokens,
            targets,
            config,
            auxk.weight_decay_mask(params),
            None,
            None,
            auxk_fn,
        )
        dead_steps = np.asarray(optimizer["dead_steps"])
        self.assertEqual(
            dead_steps.shape, (config.layers, config.mlp_mult * config.d_model)
        )
        self.assertTrue(np.all((dead_steps == 0) | (dead_steps == 1)))
        self.assertGreater(np.count_nonzero(dead_steps), 0)
        self.assertEqual(
            metrics["auxk_age_row"].shape,
            (len(auxk.AUXK_AGE_STAT_NAMES) * (config.layers + 1),),
        )
        self.assertEqual(
            optimizer["history"].shape[1], len(auxk.fuzzy_training_log_columns(config))
        )

    def test_physical_flop_rules_bill_18mdh_and_19mdh(self) -> None:
        rec_config = resolved(reconstruction)
        aux_config = resolved(auxk)
        rec_breakdown = reconstruction.traced_flops(
            rec_config, reconstruction.init_params(rec_config, 37)
        )
        aux_breakdown = auxk.traced_flops(
            aux_config, auxk.init_params(aux_config, 37)
        )
        tokens = rec_config.seq_len
        width = rec_config.d_model
        hidden = rec_config.mlp_mult * width
        self.assertEqual(
            rec_breakdown.by_site["_choicewise_fuzzy_topk_reconstruction_mlp"],
            rec_config.layers * 18 * tokens * width * hidden,
        )
        self.assertEqual(
            aux_breakdown.by_site[
                "_choicewise_fuzzy_topk_reconstruction_auxk_mlp"
            ],
            aux_config.layers * 19 * tokens * width * hidden,
        )
        self.assertFalse(
            [
                warning
                for warning in (*rec_breakdown.warnings, *aux_breakdown.warnings)
                if "fuzzy_topk_reconstruction" in warning
            ]
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
