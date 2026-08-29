"""Numerical contracts for the train-only fuzzy reconstruction heads."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from rig.kernels.fuzzy_topk_reconstruction import (
    FuzzyTopKReconstructionConfig,
    RECONSTRUCTION_AUXK_STAT_NAMES,
    RECONSTRUCTION_STAT_NAMES,
    fuzzy_topk_mlp_with_reconstruction,
    fuzzy_topk_mlp_with_reconstruction_auxk,
    make_mesh_fuzzy_topk_mlp_with_reconstruction,
    make_mesh_fuzzy_topk_mlp_with_reconstruction_auxk,
    naive_fuzzy_topk_mlp_with_reconstruction,
    naive_fuzzy_topk_mlp_with_reconstruction_auxk,
)


class FuzzyTopKReconstructionKernelTests(unittest.TestCase):
    @staticmethod
    def inputs(dtype=jnp.float32):
        x = jax.random.normal(jax.random.key(101), (2, 5, 8), dtype=dtype) * 0.3
        up_weight = (
            jax.random.normal(jax.random.key(102), (8, 32), dtype=dtype) * 0.1
        )
        up_bias = (
            jax.random.normal(jax.random.key(103), (32,), dtype=dtype) * 0.02
        )
        down_weight = (
            jax.random.normal(jax.random.key(104), (32, 8), dtype=dtype) * 0.1
        )
        down_bias = (
            jax.random.normal(jax.random.key(105), (8,), dtype=dtype) * 0.02
        )
        reconstruction_weight = (
            jax.random.normal(jax.random.key(106), (32, 8), dtype=dtype) * 0.2
        )
        return (
            x,
            up_weight,
            up_bias,
            down_weight,
            down_bias,
            reconstruction_weight,
        )

    def test_reconstruction_forward_and_every_gradient_match_oracle(self) -> None:
        inputs = self.inputs()
        config = FuzzyTopKReconstructionConfig(top_k=8)
        output_cotangent = (
            jax.random.normal(jax.random.key(107), inputs[0].shape) * 0.1
        )

        def actual_loss(*operands):
            output, statistics = fuzzy_topk_mlp_with_reconstruction(
                *operands, config=config
            )
            return jnp.sum(output * output_cotangent) + 0.7 * statistics[0]

        def expected_loss(*operands):
            output, statistics = naive_fuzzy_topk_mlp_with_reconstruction(
                *operands, config=config
            )
            return jnp.sum(output * output_cotangent) + 0.7 * statistics[0]

        actual_output, actual_statistics = fuzzy_topk_mlp_with_reconstruction(
            *inputs, config=config
        )
        expected_output, expected_statistics = (
            naive_fuzzy_topk_mlp_with_reconstruction(*inputs, config=config)
        )
        np.testing.assert_allclose(
            actual_output, expected_output, rtol=3e-5, atol=3e-6
        )
        np.testing.assert_allclose(
            actual_statistics, expected_statistics, rtol=3e-5, atol=3e-6
        )

        actual_gradients = jax.grad(actual_loss, argnums=tuple(range(6)))(*inputs)
        expected_gradients = jax.grad(expected_loss, argnums=tuple(range(6)))(*inputs)
        names = (
            "x",
            "up_weight",
            "up_bias",
            "down_weight",
            "down_bias",
            "reconstruction_weight",
        )
        for name, actual, expected in zip(
            names, actual_gradients, expected_gradients, strict=True
        ):
            with self.subTest(gradient=name):
                np.testing.assert_allclose(actual, expected, rtol=4e-5, atol=3e-6)

    def test_reconstruction_loss_does_not_change_input_or_main_decoder(self) -> None:
        inputs = self.inputs()
        config = FuzzyTopKReconstructionConfig(top_k=8)

        def loss(*operands):
            _output, statistics = fuzzy_topk_mlp_with_reconstruction(
                *operands, config=config
            )
            return statistics[0]

        gradients = jax.grad(loss, argnums=tuple(range(6)))(*inputs)
        np.testing.assert_array_equal(gradients[0], np.zeros_like(inputs[0]))
        np.testing.assert_array_equal(gradients[3], np.zeros_like(inputs[3]))
        np.testing.assert_array_equal(gradients[4], np.zeros_like(inputs[4]))
        self.assertGreater(float(jnp.linalg.norm(gradients[1])), 0.0)
        self.assertGreater(float(jnp.linalg.norm(gradients[2])), 0.0)
        self.assertGreater(float(jnp.linalg.norm(gradients[5])), 0.0)

    def test_auxk_forward_counts_and_every_gradient_match_oracle(self) -> None:
        inputs = self.inputs()
        config = FuzzyTopKReconstructionConfig(top_k=8, aux_k=2)
        dead_mask = jnp.asarray(
            [
                True,
                False,
                True,
                False,
                False,
                True,
                True,
                False,
            ]
            * 4,
            jnp.bool_,
        )
        cohort = jnp.asarray(1, jnp.int32)
        output_cotangent = (
            jax.random.normal(jax.random.key(108), inputs[0].shape) * 0.1
        )

        def actual_loss(*operands):
            output, statistics, _counts = (
                fuzzy_topk_mlp_with_reconstruction_auxk(
                    *operands, dead_mask, cohort, config=config
                )
            )
            return (
                jnp.sum(output * output_cotangent)
                + 0.6 * statistics[0]
                + (1.0 / 32.0) * statistics[1]
            )

        def expected_loss(*operands):
            output, statistics, _counts = (
                naive_fuzzy_topk_mlp_with_reconstruction_auxk(
                    *operands, dead_mask, cohort, config=config
                )
            )
            return (
                jnp.sum(output * output_cotangent)
                + 0.6 * statistics[0]
                + (1.0 / 32.0) * statistics[1]
            )

        actual = fuzzy_topk_mlp_with_reconstruction_auxk(
            *inputs, dead_mask, cohort, config=config
        )
        expected = naive_fuzzy_topk_mlp_with_reconstruction_auxk(
            *inputs, dead_mask, cohort, config=config
        )
        for actual_value, expected_value in zip(actual, expected, strict=True):
            np.testing.assert_allclose(
                actual_value, expected_value, rtol=4e-5, atol=3e-6
            )

        actual_gradients = jax.grad(actual_loss, argnums=tuple(range(6)))(*inputs)
        expected_gradients = jax.grad(expected_loss, argnums=tuple(range(6)))(*inputs)
        for actual_gradient, expected_gradient in zip(
            actual_gradients, expected_gradients, strict=True
        ):
            np.testing.assert_allclose(
                actual_gradient, expected_gradient, rtol=5e-5, atol=4e-6
            )

    def test_auxk_loss_only_updates_encoder_and_reconstruction_decoder(self) -> None:
        inputs = self.inputs()
        config = FuzzyTopKReconstructionConfig(top_k=8, aux_k=2)
        dead_mask = jnp.ones((32,), jnp.bool_)
        cohort = jnp.asarray(0, jnp.int32)

        def loss(*operands):
            _output, statistics, _counts = (
                fuzzy_topk_mlp_with_reconstruction_auxk(
                    *operands, dead_mask, cohort, config=config
                )
            )
            return statistics[1]

        gradients = jax.grad(loss, argnums=tuple(range(6)))(*inputs)
        np.testing.assert_array_equal(gradients[0], np.zeros_like(inputs[0]))
        np.testing.assert_array_equal(gradients[3], np.zeros_like(inputs[3]))
        np.testing.assert_array_equal(gradients[4], np.zeros_like(inputs[4]))
        self.assertGreater(float(jnp.linalg.norm(gradients[1])), 0.0)
        self.assertGreater(float(jnp.linalg.norm(gradients[2])), 0.0)
        self.assertGreater(float(jnp.linalg.norm(gradients[5])), 0.0)

    def test_mesh_wrappers_return_global_statistics_and_are_trainable(self) -> None:
        inputs = self.inputs()
        mesh = Mesh(np.asarray(jax.devices()[:1], dtype=object), ("data",))

        reconstruction_config = FuzzyTopKReconstructionConfig(top_k=8)
        reconstruction = make_mesh_fuzzy_topk_mlp_with_reconstruction(
            config=reconstruction_config, mesh=mesh
        )
        output, statistics = jax.jit(reconstruction)(*inputs)
        expected_output, expected_statistics = (
            naive_fuzzy_topk_mlp_with_reconstruction(
                *inputs, config=reconstruction_config
            )
        )
        np.testing.assert_allclose(output, expected_output, rtol=3e-5, atol=3e-6)
        np.testing.assert_allclose(
            statistics, expected_statistics, rtol=3e-5, atol=3e-6
        )
        self.assertEqual(statistics.shape, (len(RECONSTRUCTION_STAT_NAMES),))

        auxk_config = FuzzyTopKReconstructionConfig(top_k=8, aux_k=2)
        auxk = make_mesh_fuzzy_topk_mlp_with_reconstruction_auxk(
            config=auxk_config, mesh=mesh
        )
        dead_mask = jnp.ones((32,), jnp.bool_)
        cohort = jnp.asarray(0, jnp.int32)
        aux_output, aux_statistics, counts = jax.jit(auxk)(
            *inputs, dead_mask, cohort
        )
        expected_output, expected_statistics, expected_counts = (
            naive_fuzzy_topk_mlp_with_reconstruction_auxk(
                *inputs, dead_mask, cohort, config=auxk_config
            )
        )
        np.testing.assert_allclose(
            aux_output, expected_output, rtol=3e-5, atol=3e-6
        )
        np.testing.assert_allclose(
            aux_statistics, expected_statistics, rtol=3e-5, atol=3e-6
        )
        np.testing.assert_allclose(counts, expected_counts, rtol=0.0, atol=0.0)
        self.assertEqual(
            aux_statistics.shape, (len(RECONSTRUCTION_AUXK_STAT_NAMES),)
        )

        gradients = jax.grad(
            lambda *operands: (
                jnp.square(auxk(*operands, dead_mask, cohort)[0]).mean()
                + auxk(*operands, dead_mask, cohort)[1][0]
            ),
            argnums=tuple(range(6)),
        )(*inputs)
        self.assertEqual(len(gradients), 6)

    def test_contract_rejects_nondividing_auxk(self) -> None:
        with self.assertRaisesRegex(ValueError, "aux_k must divide top_k"):
            FuzzyTopKReconstructionConfig(top_k=8, aux_k=3)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
