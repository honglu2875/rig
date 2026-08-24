"""Numerical contracts for the exact TopK-ReLU sparse MLP."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from rig.kernels.sparse_mlp import (
    SparseMlpConfig,
    make_mesh_sparse_topk_mlp,
    naive_dense_topk_mlp,
    pallas_sparse_decode,
    reference_sparse_decode,
    sparse_topk_mlp,
    topk_relu,
)


class SparseMlpKernelTests(unittest.TestCase):
    @staticmethod
    def inputs(dtype=jnp.float32):
        x = jax.random.normal(jax.random.key(1), (2, 5, 128), dtype=dtype) * 0.2
        up_weight = jax.random.normal(jax.random.key(2), (128, 256), dtype=dtype) * 0.05
        up_bias = jax.random.normal(jax.random.key(3), (256,), dtype=dtype) * 0.01
        down_weight = (
            jax.random.normal(jax.random.key(4), (256, 128), dtype=dtype) * 0.05
        )
        down_bias = jax.random.normal(jax.random.key(5), (128,), dtype=dtype) * 0.01
        return x, up_weight, up_bias, down_weight, down_bias

    def test_topk_relu_keeps_largest_positive_coordinates(self) -> None:
        values, indices = topk_relu(
            jnp.asarray([[-2.0, 4.0, 1.0, -1.0, 3.0]], jnp.float32), top_k=3
        )
        np.testing.assert_array_equal(indices, [[1, 4, 2]])
        np.testing.assert_array_equal(values, [[4.0, 3.0, 1.0]])

        values, _ = topk_relu(
            jnp.asarray([[-2.0, -4.0, 1.0, -1.0]], jnp.float32), top_k=3
        )
        np.testing.assert_array_equal(values, [[1.0, 0.0, 0.0]])

    def test_pallas_interpreter_matches_selected_row_oracle_with_padding(
        self,
    ) -> None:
        x, up_weight, up_bias, down_weight, down_bias = self.inputs()
        hidden = jnp.einsum("...d,dh->...h", x, up_weight) + up_bias
        values, indices = topk_relu(hidden, top_k=8)
        expected = reference_sparse_decode(values, indices, down_weight, down_bias)
        actual = pallas_sparse_decode(
            values,
            indices,
            down_weight,
            down_bias,
            token_block=8,
            interpret=True,
        )
        self.assertEqual(actual.shape, x.shape)
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)

    def test_fused_forward_and_every_gradient_match_dense_topk_dense(
        self,
    ) -> None:
        inputs = self.inputs()
        config = SparseMlpConfig(
            top_k=8, backend="pallas", token_block=8, interpret=True
        )
        cotangent = jax.random.normal(jax.random.key(6), inputs[0].shape) * 0.1

        def actual_loss(*operands):
            return jnp.sum(sparse_topk_mlp(*operands, config=config) * cotangent)

        def expected_loss(*operands):
            return jnp.sum(naive_dense_topk_mlp(*operands, top_k=8) * cotangent)

        actual = sparse_topk_mlp(*inputs, config=config)
        expected = naive_dense_topk_mlp(*inputs, top_k=8)
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)

        actual_gradients = jax.grad(actual_loss, argnums=(0, 1, 2, 3, 4))(*inputs)
        expected_gradients = jax.grad(expected_loss, argnums=(0, 1, 2, 3, 4))(*inputs)
        for name, actual_gradient, expected_gradient in zip(
            ("x", "up_weight", "up_bias", "down_weight", "down_bias"),
            actual_gradients,
            expected_gradients,
            strict=True,
        ):
            with self.subTest(gradient=name):
                np.testing.assert_allclose(
                    actual_gradient,
                    expected_gradient,
                    rtol=2e-5,
                    atol=1e-6,
                )

    def test_configuration_rejects_invalid_static_kernel_contracts(self) -> None:
        with self.assertRaisesRegex(ValueError, "top_k must be positive"):
            SparseMlpConfig(top_k=0)
        with self.assertRaisesRegex(ValueError, "multiple of 128"):
            SparseMlpConfig(top_k=1, output_block=64)

    def test_explicit_mesh_boundary_is_trainable(self) -> None:
        x, *parameters = self.inputs()
        # One device and the reference decoder are enough to exercise the
        # explicit shard_map boundary and its transpose. The two tests above
        # cover the Pallas decoder itself; nesting Pallas interpret mode in a
        # CPU shard_map leaves interpreter worker threads alive after pytest.
        mesh = Mesh(np.asarray(jax.devices()[:1], dtype=object), ("data",))
        inputs = (x, *parameters)
        operation = make_mesh_sparse_topk_mlp(
            config=SparseMlpConfig(top_k=8, backend="reference"),
            mesh=mesh,
        )
        output = jax.jit(operation)(*inputs)
        expected = naive_dense_topk_mlp(*inputs, top_k=8)
        np.testing.assert_allclose(output, expected, rtol=2e-5, atol=2e-6)
        gradients = jax.grad(lambda *args: jnp.square(operation(*args)).mean())(*inputs)
        self.assertEqual(gradients.shape, inputs[0].shape)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
