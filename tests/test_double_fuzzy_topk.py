"""Numerical contracts for two-stage grouped approximate Top-K."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from rig.kernels.double_fuzzy_topk import (
    DoubleFuzzyTopKConfig,
    double_fuzzy_topk_mlp,
    grouped_signed_max,
    make_mesh_double_fuzzy_topk_mlp,
    naive_double_fuzzy_topk_mlp,
)


class DoubleFuzzyTopKKernelTests(unittest.TestCase):
    @staticmethod
    def inputs(dtype=jnp.float32):
        x = jax.random.normal(jax.random.key(11), (1, 3, 128), dtype=dtype) * 0.2
        up_weight = (
            jax.random.normal(jax.random.key(12), (128, 256), dtype=dtype) * 0.05
        )
        up_bias = jax.random.normal(jax.random.key(13), (256,), dtype=dtype) * 0.01
        down_weight = (
            jax.random.normal(jax.random.key(14), (256, 128), dtype=dtype) * 0.05
        )
        down_bias = jax.random.normal(jax.random.key(15), (128,), dtype=dtype) * 0.01
        return x, up_weight, up_bias, down_weight, down_bias

    def test_input_selector_keeps_signed_maximum_from_each_group(self) -> None:
        values, indices = grouped_signed_max(
            jnp.asarray(
                [[-2.0, 4.0, 1.0, -1.0, -4.0, -3.0, -2.0, -1.0]],
                jnp.float32,
            ),
            group_size=4,
        )
        np.testing.assert_array_equal(values, [[4.0, -1.0]])
        np.testing.assert_array_equal(indices, [[1, 7]])

    def assert_backend_matches_oracle(self, config: DoubleFuzzyTopKConfig) -> None:
        inputs = self.inputs()
        cotangent = jax.random.normal(jax.random.key(16), inputs[0].shape) * 0.1

        def actual_loss(*operands):
            return jnp.sum(
                double_fuzzy_topk_mlp(*operands, config=config) * cotangent
            )

        def expected_loss(*operands):
            return jnp.sum(
                naive_double_fuzzy_topk_mlp(
                    *operands,
                    top_k=64,
                    input_group_size=4,
                )
                * cotangent
            )

        np.testing.assert_allclose(
            double_fuzzy_topk_mlp(*inputs, config=config),
            naive_double_fuzzy_topk_mlp(
                *inputs,
                top_k=64,
                input_group_size=4,
            ),
            rtol=2e-5,
            atol=2e-6,
        )
        actual_gradients = jax.grad(
            actual_loss, argnums=(0, 1, 2, 3, 4)
        )(*inputs)
        expected_gradients = jax.grad(
            expected_loss, argnums=(0, 1, 2, 3, 4)
        )(*inputs)
        for name, actual, expected in zip(
            ("x", "up_weight", "up_bias", "down_weight", "down_bias"),
            actual_gradients,
            expected_gradients,
            strict=True,
        ):
            with self.subTest(backend=config.backend, gradient=name):
                np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-6)

    def test_choicewise_output_and_every_gradient_match_oracle(self) -> None:
        self.assert_backend_matches_oracle(
            DoubleFuzzyTopKConfig(top_k=64, backend="choicewise")
        )

    def test_reference_output_and_every_gradient_match_oracle(self) -> None:
        self.assert_backend_matches_oracle(
            DoubleFuzzyTopKConfig(top_k=64, backend="reference")
        )

    def test_pallas_up_output_and_every_gradient_match_oracle(self) -> None:
        self.assert_backend_matches_oracle(
            DoubleFuzzyTopKConfig(
                top_k=64,
                backend="pallas_up",
                token_block=4,
                output_block=128,
                interpret=True,
            )
        )

    def test_pallas_up_bfloat16_output_and_gradients_match_oracle(self) -> None:
        x, up_weight, _up_bias, down_weight, down_bias = self.inputs(jnp.bfloat16)
        # Give every outer four-feature group a decisive winner. BF16 changes
        # accumulation order between a Pallas row reduction and dense einsum;
        # without a margin, an expected near-tie may legitimately change the
        # nondifferentiable argmax and make gradient comparison meaningless.
        up_weight = up_weight * jnp.asarray(0.1, jnp.bfloat16)
        up_bias = jnp.tile(
            jnp.asarray([0.4, 0.1, -0.2, -0.5], jnp.bfloat16),
            64,
        )
        inputs = (x, up_weight, up_bias, down_weight, down_bias)
        config = DoubleFuzzyTopKConfig(
            top_k=64,
            backend="pallas_up",
            token_block=4,
            output_block=128,
            interpret=True,
        )
        cotangent = jax.random.normal(
            jax.random.key(17), inputs[0].shape, dtype=jnp.bfloat16
        )

        def actual_loss(*operands):
            return jnp.sum(
                double_fuzzy_topk_mlp(*operands, config=config) * cotangent
            )

        def expected_loss(*operands):
            return jnp.sum(
                naive_double_fuzzy_topk_mlp(
                    *operands,
                    top_k=64,
                    input_group_size=4,
                )
                * cotangent
            )

        np.testing.assert_allclose(
            double_fuzzy_topk_mlp(*inputs, config=config),
            naive_double_fuzzy_topk_mlp(
                *inputs,
                top_k=64,
                input_group_size=4,
            ),
            rtol=2e-2,
            atol=2e-2,
        )
        for actual, expected in zip(
            jax.grad(actual_loss, argnums=(0, 1, 2, 3, 4))(*inputs),
            jax.grad(expected_loss, argnums=(0, 1, 2, 3, 4))(*inputs),
            strict=True,
        ):
            np.testing.assert_allclose(actual, expected, rtol=2e-2, atol=2e-2)

    def test_pallas_up_pads_non_vector_aligned_input_cotangent(self) -> None:
        dtype = jnp.float32
        x = jax.random.normal(jax.random.key(21), (1, 2, 192), dtype=dtype) * 0.2
        up_weight = (
            jax.random.normal(jax.random.key(22), (192, 256), dtype=dtype) * 0.05
        )
        up_bias = jax.random.normal(jax.random.key(23), (256,), dtype=dtype) * 0.01
        down_weight = (
            jax.random.normal(jax.random.key(24), (256, 192), dtype=dtype) * 0.05
        )
        down_bias = jax.random.normal(jax.random.key(25), (192,), dtype=dtype) * 0.01
        inputs = (x, up_weight, up_bias, down_weight, down_bias)
        config = DoubleFuzzyTopKConfig(
            top_k=64,
            backend="pallas_up_dx",
            token_block=2,
            output_block=128,
            interpret=True,
        )
        cotangent = jax.random.normal(jax.random.key(26), x.shape) * 0.1

        def actual_loss(*operands):
            return jnp.sum(
                double_fuzzy_topk_mlp(*operands, config=config) * cotangent
            )

        def expected_loss(*operands):
            return jnp.sum(
                naive_double_fuzzy_topk_mlp(
                    *operands,
                    top_k=64,
                    input_group_size=4,
                )
                * cotangent
            )

        np.testing.assert_allclose(
            double_fuzzy_topk_mlp(*inputs, config=config),
            naive_double_fuzzy_topk_mlp(
                *inputs,
                top_k=64,
                input_group_size=4,
            ),
            rtol=2e-5,
            atol=2e-6,
        )
        for actual, expected in zip(
            jax.grad(actual_loss, argnums=(0, 1, 2, 3, 4))(*inputs),
            jax.grad(expected_loss, argnums=(0, 1, 2, 3, 4))(*inputs),
            strict=True,
        ):
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-6)

    def test_contract_rejects_nonintegral_groups(self) -> None:
        x, up_weight, up_bias, down_weight, down_bias = self.inputs()
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            double_fuzzy_topk_mlp(
                x[..., :-1],
                up_weight[:-1],
                up_bias,
                down_weight[:, :-1],
                down_bias[:-1],
                config=DoubleFuzzyTopKConfig(top_k=64),
            )
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            double_fuzzy_topk_mlp(
                x,
                up_weight,
                up_bias,
                down_weight,
                down_bias,
                config=DoubleFuzzyTopKConfig(top_k=63, backend="reference"),
            )

    def test_explicit_mesh_boundary_is_trainable(self) -> None:
        inputs = self.inputs()
        mesh = Mesh(np.asarray(jax.devices()[:1], dtype=object), ("data",))
        operation = make_mesh_double_fuzzy_topk_mlp(
            config=DoubleFuzzyTopKConfig(top_k=64, backend="reference"),
            mesh=mesh,
        )
        output = jax.jit(operation)(*inputs)
        expected = naive_double_fuzzy_topk_mlp(
            *inputs,
            top_k=64,
            input_group_size=4,
        )
        np.testing.assert_allclose(output, expected, rtol=2e-5, atol=2e-6)
        gradient = jax.grad(
            lambda x: jnp.square(operation(x, *inputs[1:])).mean()
        )(inputs[0])
        self.assertEqual(gradient.shape, inputs[0].shape)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
