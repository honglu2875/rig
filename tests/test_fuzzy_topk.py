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
    FUZZY_FEATURE_STAT_NAMES,
    FuzzyTopKConfig,
    fuzzy_topk_mlp,
    fuzzy_topk_mlp_with_diagnostics,
    fuzzy_topk_relu,
    make_mesh_fuzzy_topk_mlp,
    make_mesh_fuzzy_topk_mlp_with_diagnostics,
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

    def test_feature_diagnostics_match_literal_selected_feature_counts(self) -> None:
        inputs = self.inputs()
        top_k = 16
        output, statistics = fuzzy_topk_mlp_with_diagnostics(
            *inputs, config=FuzzyTopKConfig(top_k=top_k, backend="choicewise")
        )
        hidden = jnp.einsum("...d,dh->...h", inputs[0], inputs[1]) + inputs[2]
        values, indices = fuzzy_topk_relu(hidden, top_k=top_k)
        tokens = int(np.prod(values.shape[:-1]))
        expected = np.zeros((len(FUZZY_FEATURE_STAT_NAMES), hidden.shape[-1]))
        for feature in range(hidden.shape[-1]):
            selected = np.asarray(indices == feature)
            active = np.where(selected, np.asarray(values), 0.0)
            expected[0, feature] = np.sum(selected) / tokens
            expected[1, feature] = np.sum(active > 0.0) / tokens
            expected[2, feature] = np.sum(active) / tokens
            expected[3, feature] = np.sqrt(np.sum(np.square(active)) / tokens)

        np.testing.assert_allclose(
            output,
            fuzzy_topk_mlp(
                *inputs, config=FuzzyTopKConfig(top_k=top_k, backend="choicewise")
            ),
            rtol=2e-5,
            atol=2e-6,
        )
        np.testing.assert_allclose(statistics, expected, rtol=2e-5, atol=2e-6)
        reference_output, reference_statistics = fuzzy_topk_mlp_with_diagnostics(
            *inputs, config=FuzzyTopKConfig(top_k=top_k, backend="reference")
        )
        np.testing.assert_allclose(reference_output, output, rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(
            reference_statistics, statistics, rtol=2e-5, atol=2e-6
        )

    def test_feature_diagnostic_auxiliary_does_not_change_gradients(self) -> None:
        inputs = self.inputs()
        config = FuzzyTopKConfig(top_k=16, backend="choicewise")

        def diagnostic_loss(*operands):
            output, _statistics = fuzzy_topk_mlp_with_diagnostics(
                *operands, config=config
            )
            return jnp.square(output).mean()

        def ordinary_loss(*operands):
            return jnp.square(fuzzy_topk_mlp(*operands, config=config)).mean()

        diagnostic_gradients = jax.grad(diagnostic_loss, argnums=(0, 1, 2, 3, 4))(
            *inputs
        )
        ordinary_gradients = jax.grad(ordinary_loss, argnums=(0, 1, 2, 3, 4))(
            *inputs
        )
        for actual, expected in zip(
            diagnostic_gradients, ordinary_gradients, strict=True
        ):
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-6)

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

        diagnostic_operation = make_mesh_fuzzy_topk_mlp_with_diagnostics(
            config=FuzzyTopKConfig(top_k=16), mesh=mesh
        )
        diagnostic_output, statistics = jax.jit(diagnostic_operation)(*inputs)
        np.testing.assert_allclose(diagnostic_output, expected, rtol=2e-5, atol=2e-6)
        self.assertEqual(statistics.shape, (len(FUZZY_FEATURE_STAT_NAMES), 64))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
