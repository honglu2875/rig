"""Numerical contracts for paper-inspired fuzzy AuxK ghost gradients."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from rig.kernels import (
    FuzzyTopKAuxKConfig,
    FuzzyTopKConfig,
    fuzzy_topk_mlp,
    fuzzy_topk_mlp_with_auxk,
    make_mesh_fuzzy_topk_mlp_with_auxk,
    naive_fuzzy_topk_mlp_with_auxk,
)


class FuzzyTopKAuxKKernelTests(unittest.TestCase):
    @staticmethod
    def inputs():
        x = jax.random.normal(jax.random.key(31), (2, 5, 16)) * 0.2
        up_weight = jax.random.normal(jax.random.key(32), (16, 64)) * 0.05
        up_bias = jax.random.normal(jax.random.key(33), (64,)) * 0.03 - 0.02
        down_weight = jax.random.normal(jax.random.key(34), (64, 16)) * 0.05
        down_bias = jax.random.normal(jax.random.key(35), (16,)) * 0.01
        dead_mask = (jnp.arange(64) % 3) != 0
        cohort = jnp.asarray(1, jnp.int32)
        return x, up_weight, up_bias, down_weight, down_bias, dead_mask, cohort

    def test_forward_is_exactly_the_parent_fuzzy_operation(self) -> None:
        inputs = self.inputs()
        config = FuzzyTopKAuxKConfig(top_k=16, aux_k=4, coefficient=0.3)
        output, counts = fuzzy_topk_mlp_with_auxk(*inputs, config=config)
        expected = fuzzy_topk_mlp(
            *inputs[:5], config=FuzzyTopKConfig(top_k=16, backend="choicewise")
        )
        np.testing.assert_array_equal(output, expected)
        self.assertEqual(counts.shape, (64,))
        self.assertTrue(np.all(np.asarray(counts) >= 0.0))

    def test_all_parameter_gradients_match_literal_ghost_oracle(self) -> None:
        inputs = self.inputs()
        config = FuzzyTopKAuxKConfig(top_k=16, aux_k=4, coefficient=0.17)
        output_cotangent = jax.random.normal(jax.random.key(36), inputs[0].shape)

        def objective(operation, operands):
            output, _counts = operation(*operands, config=config)
            return jnp.sum(output * output_cotangent)

        actual = fuzzy_topk_mlp_with_auxk(*inputs, config=config)
        expected = naive_fuzzy_topk_mlp_with_auxk(*inputs, config=config)
        for actual_value, expected_value in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(actual_value, expected_value)

        actual_gradients = jax.grad(
            lambda *operands: objective(fuzzy_topk_mlp_with_auxk, operands),
            argnums=(0, 1, 2, 3, 4),
        )(*inputs)
        expected_gradients = jax.grad(
            lambda *operands: objective(naive_fuzzy_topk_mlp_with_auxk, operands),
            argnums=(0, 1, 2, 3, 4),
        )(*inputs)
        for name, actual_gradient, expected_gradient in zip(
            ("x", "up_weight", "up_bias", "down_weight", "down_bias"),
            actual_gradients,
            expected_gradients,
            strict=True,
        ):
            with self.subTest(gradient=name):
                np.testing.assert_allclose(
                    actual_gradient, expected_gradient, rtol=3e-5, atol=2e-6
                )

    def test_ghost_adds_no_input_or_down_bias_gradient(self) -> None:
        inputs = self.inputs()
        parent_config = FuzzyTopKAuxKConfig(top_k=16, aux_k=4, coefficient=0.0)
        ghost_config = FuzzyTopKAuxKConfig(top_k=16, aux_k=4, coefficient=0.25)

        def gradients(config):
            return jax.grad(
                lambda *operands: jnp.square(
                    fuzzy_topk_mlp_with_auxk(*operands, config=config)[0]
                ).mean(),
                argnums=(0, 1, 2, 3, 4),
            )(*inputs)

        parent = gradients(parent_config)
        ghost = gradients(ghost_config)
        np.testing.assert_allclose(ghost[0], parent[0], rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(ghost[4], parent[4], rtol=2e-5, atol=2e-6)
        self.assertGreater(float(jnp.linalg.norm(ghost[1] - parent[1])), 0.0)
        self.assertGreater(float(jnp.linalg.norm(ghost[2] - parent[2])), 0.0)
        self.assertGreater(float(jnp.linalg.norm(ghost[3] - parent[3])), 0.0)

    def test_cohorts_cover_each_main_group_once(self) -> None:
        config = FuzzyTopKAuxKConfig(top_k=16, aux_k=4)
        covered = []
        for cohort in range(config.cohort_count):
            covered.extend(cohort + config.cohort_count * np.arange(config.aux_k))
        np.testing.assert_array_equal(np.sort(covered), np.arange(config.top_k))

    def test_explicit_mesh_boundary_is_trainable(self) -> None:
        inputs = list(self.inputs())
        device_count = min(4, jax.device_count())
        if device_count > inputs[0].shape[0]:
            inputs[0] = jnp.tile(inputs[0], (device_count, 1, 1))[:device_count]
        config = FuzzyTopKAuxKConfig(top_k=16, aux_k=4)
        mesh = Mesh(np.asarray(jax.devices()[:device_count], dtype=object), ("data",))
        operation = make_mesh_fuzzy_topk_mlp_with_auxk(config=config, mesh=mesh)
        output, counts = jax.jit(operation)(*inputs)
        expected_output, expected_counts = naive_fuzzy_topk_mlp_with_auxk(
            *inputs, config=config
        )
        np.testing.assert_allclose(output, expected_output, rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(counts, expected_counts, rtol=2e-5, atol=2e-6)
        gradients = jax.grad(lambda value: jnp.square(operation(value, *inputs[1:])[0]).mean())(
            inputs[0]
        )
        self.assertEqual(gradients.shape, inputs[0].shape)

    def test_config_rejects_invalid_auxiliary_width(self) -> None:
        with self.assertRaisesRegex(ValueError, "divide"):
            FuzzyTopKAuxKConfig(top_k=16, aux_k=3)
        with self.assertRaisesRegex(ValueError, "coefficient"):
            FuzzyTopKAuxKConfig(top_k=16, aux_k=4, coefficient=-1.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
