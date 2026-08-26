"""Numerical contracts for grouped approximate Top-K."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from rig.kernels.fuzzy_topk import (
    FuzzyTopKConfig,
    fuzzy_topk_mlp,
    fuzzy_topk_relu,
    make_mesh_fuzzy_topk_mlp,
    naive_fuzzy_topk_mlp,
)


class FuzzyTopKKernelTests(unittest.TestCase):
    @staticmethod
    def inputs(dtype=jnp.float32):
        x = jax.random.normal(jax.random.key(1), (2, 5, 16), dtype=dtype) * 0.2
        up_weight = jax.random.normal(jax.random.key(2), (16, 64), dtype=dtype) * 0.05
        up_bias = jax.random.normal(jax.random.key(3), (64,), dtype=dtype) * 0.01
        down_weight = jax.random.normal(jax.random.key(4), (64, 16), dtype=dtype) * 0.05
        down_bias = jax.random.normal(jax.random.key(5), (16,), dtype=dtype) * 0.01
        return x, up_weight, up_bias, down_weight, down_bias

    def test_selector_keeps_one_positive_winner_per_fixed_group(self) -> None:
        values, indices = fuzzy_topk_relu(
            jnp.asarray(
                [[-2.0, 4.0, 1.0, -1.0, -4.0, -3.0, -2.0, -1.0]],
                jnp.float32,
            ),
            top_k=2,
        )
        np.testing.assert_array_equal(values, [[4.0, 0.0]])
        np.testing.assert_array_equal(indices, [[1, 7]])

    def test_choicewise_forward_and_every_gradient_match_literal_oracle(
        self,
    ) -> None:
        inputs = self.inputs()
        config = FuzzyTopKConfig(top_k=16, backend="choicewise")
        cotangent = jax.random.normal(jax.random.key(6), inputs[0].shape) * 0.1

        def actual_loss(*operands):
            return jnp.sum(fuzzy_topk_mlp(*operands, config=config) * cotangent)

        def expected_loss(*operands):
            return jnp.sum(naive_fuzzy_topk_mlp(*operands, top_k=16) * cotangent)

        np.testing.assert_allclose(
            fuzzy_topk_mlp(*inputs, config=config),
            naive_fuzzy_topk_mlp(*inputs, top_k=16),
            rtol=2e-5,
            atol=2e-6,
        )
        actual_gradients = jax.grad(actual_loss, argnums=(0, 1, 2, 3, 4))(*inputs)
        expected_gradients = jax.grad(expected_loss, argnums=(0, 1, 2, 3, 4))(*inputs)
        for name, actual, expected in zip(
            ("x", "up_weight", "up_bias", "down_weight", "down_bias"),
            actual_gradients,
            expected_gradients,
            strict=True,
        ):
            with self.subTest(gradient=name):
                np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-6)

    def test_reference_and_choicewise_backends_have_identical_semantics(self) -> None:
        inputs = self.inputs()
        reference = FuzzyTopKConfig(top_k=16, backend="reference")
        choicewise = FuzzyTopKConfig(top_k=16, backend="choicewise")
        np.testing.assert_allclose(
            fuzzy_topk_mlp(*inputs, config=choicewise),
            fuzzy_topk_mlp(*inputs, config=reference),
            rtol=2e-5,
            atol=2e-6,
        )

    def test_contract_rejects_nonintegral_groups(self) -> None:
        x, up_weight, up_bias, down_weight, down_bias = self.inputs()
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            fuzzy_topk_mlp(
                x,
                up_weight,
                up_bias,
                down_weight,
                down_bias,
                config=FuzzyTopKConfig(top_k=15),
            )

    def test_explicit_mesh_boundary_is_trainable(self) -> None:
        inputs = self.inputs()
        mesh = Mesh(np.asarray(jax.devices()[:1], dtype=object), ("data",))
        operation = make_mesh_fuzzy_topk_mlp(
            config=FuzzyTopKConfig(top_k=16), mesh=mesh
        )
        output = jax.jit(operation)(*inputs)
        expected = naive_fuzzy_topk_mlp(*inputs, top_k=16)
        np.testing.assert_allclose(output, expected, rtol=2e-5, atol=2e-6)
        gradients = jax.grad(lambda *args: jnp.square(operation(*args)).mean())(*inputs)
        self.assertEqual(gradients.shape, inputs[0].shape)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
