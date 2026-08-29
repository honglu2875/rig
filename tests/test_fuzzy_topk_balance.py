"""Numerical contracts for fuzzy Top-K balance objectives."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from rig.kernels import (
    FUZZY_BALANCE_STAT_NAMES,
    FuzzyTopKBalanceConfig,
    FuzzyTopKConfig,
    fuzzy_topk_mlp,
    fuzzy_topk_mlp_with_balance,
    make_mesh_fuzzy_topk_mlp_with_balance,
    naive_fuzzy_topk_mlp_with_balance,
)


class FuzzyTopKBalanceKernelTests(unittest.TestCase):
    @staticmethod
    def inputs():
        x = jax.random.normal(jax.random.key(21), (2, 5, 16)) * 0.2
        up_weight = jax.random.normal(jax.random.key(22), (16, 64)) * 0.05
        up_bias = jax.random.normal(jax.random.key(23), (64,)) * 0.03 - 0.02
        down_weight = jax.random.normal(jax.random.key(24), (64, 16)) * 0.05
        down_bias = jax.random.normal(jax.random.key(25), (16,)) * 0.01
        return x, up_weight, up_bias, down_weight, down_bias

    def test_forward_and_all_gradients_match_literal_oracle(self) -> None:
        inputs = self.inputs()
        config = FuzzyTopKBalanceConfig(
            top_k=16,
            mode="switch",
            temperature=0.7,
            alive_margin=0.05,
        )
        output_cotangent = jax.random.normal(jax.random.key(26), inputs[0].shape)
        statistic_cotangent = jnp.asarray((0.04, 0.07, 0.11), jnp.float32)

        def actual_loss(*operands):
            output, statistics = fuzzy_topk_mlp_with_balance(*operands, config=config)
            return jnp.sum(output * output_cotangent) + jnp.sum(
                statistics * statistic_cotangent
            )

        def expected_loss(*operands):
            output, statistics = naive_fuzzy_topk_mlp_with_balance(
                *operands, config=config
            )
            return jnp.sum(output * output_cotangent) + jnp.sum(
                statistics * statistic_cotangent
            )

        actual_output, actual_statistics = fuzzy_topk_mlp_with_balance(
            *inputs, config=config
        )
        expected_output, expected_statistics = naive_fuzzy_topk_mlp_with_balance(
            *inputs, config=config
        )
        np.testing.assert_allclose(actual_output, expected_output, rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(
            actual_statistics, expected_statistics, rtol=2e-5, atol=2e-6
        )
        self.assertEqual(actual_statistics.shape, (len(FUZZY_BALANCE_STAT_NAMES),))

        actual_gradients = jax.grad(actual_loss, argnums=(0, 1, 2, 3, 4))(*inputs)
        expected_gradients = jax.grad(expected_loss, argnums=(0, 1, 2, 3, 4))(*inputs)
        for name, actual, expected in zip(
            ("x", "up_weight", "up_bias", "down_weight", "down_bias"),
            actual_gradients,
            expected_gradients,
            strict=True,
        ):
            with self.subTest(gradient=name):
                np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=2e-6)

    def test_forward_is_exactly_the_original_fuzzy_operation(self) -> None:
        inputs = self.inputs()
        output, _ = fuzzy_topk_mlp_with_balance(
            *inputs, config=FuzzyTopKBalanceConfig(top_k=16)
        )
        expected = fuzzy_topk_mlp(
            *inputs, config=FuzzyTopKConfig(top_k=16, backend="choicewise")
        )
        np.testing.assert_allclose(output, expected, rtol=2e-5, atol=2e-6)

    def test_alive_only_path_omits_soft_balance_statistics(self) -> None:
        inputs = self.inputs()
        config = FuzzyTopKBalanceConfig(
            top_k=16,
            mode="none",
            alive_margin=0.1,
        )
        _, statistics = fuzzy_topk_mlp_with_balance(*inputs, config=config)
        np.testing.assert_array_equal(np.asarray(statistics[:2]), np.zeros(2))
        self.assertGreater(float(statistics[2]), 0.0)
        gradient = jax.grad(
            lambda bias: fuzzy_topk_mlp_with_balance(
                inputs[0],
                inputs[1],
                bias,
                inputs[3],
                inputs[4],
                config=config,
            )[1][2]
        )(inputs[2])
        self.assertGreater(float(jnp.linalg.norm(gradient)), 0.0)

    def test_hard_load_bias_mode_has_only_the_declared_direct_aux_gradient(
        self,
    ) -> None:
        inputs = self.inputs()
        config = FuzzyTopKBalanceConfig(top_k=16, mode="bias")

        def actual_loss(*operands):
            output, statistics = fuzzy_topk_mlp_with_balance(*operands, config=config)
            return jnp.square(output).mean() + 0.3 * statistics[0]

        def expected_loss(*operands):
            output, statistics = naive_fuzzy_topk_mlp_with_balance(
                *operands, config=config
            )
            return jnp.square(output).mean() + 0.3 * statistics[0]

        actual_statistics = fuzzy_topk_mlp_with_balance(*inputs, config=config)[1]
        expected_statistics = naive_fuzzy_topk_mlp_with_balance(*inputs, config=config)[
            1
        ]
        np.testing.assert_allclose(
            actual_statistics, expected_statistics, rtol=2e-5, atol=2e-6
        )
        self.assertGreaterEqual(float(actual_statistics[0]), 0.0)
        actual_gradients = jax.grad(actual_loss, argnums=(0, 1, 2, 3, 4))(*inputs)
        expected_gradients = jax.grad(expected_loss, argnums=(0, 1, 2, 3, 4))(*inputs)
        for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
            np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=2e-6)

    def test_frequency_floor_pushes_only_underactive_feature_biases(self) -> None:
        x = jnp.zeros((1, 4, 4), jnp.float32)
        up_weight = jnp.zeros((4, 8), jnp.float32)
        up_bias = jnp.asarray((1.0, -1.0, -1.0, -1.0) * 2, jnp.float32)
        down_weight = jnp.zeros((8, 4), jnp.float32)
        down_bias = jnp.zeros((4,), jnp.float32)
        config = FuzzyTopKBalanceConfig(
            top_k=2,
            mode="none",
            alive_mode="frequency_floor",
            alive_target=0.4,
        )

        def survival_loss(*operands):
            return fuzzy_topk_mlp_with_balance(*operands, config=config)[1][2]

        statistics = fuzzy_topk_mlp_with_balance(
            x, up_weight, up_bias, down_weight, down_bias, config=config
        )[1]
        self.assertAlmostEqual(float(statistics[2]), 0.12, places=6)
        gradients = jax.grad(survival_loss, argnums=(0, 1, 2, 3, 4))(
            x, up_weight, up_bias, down_weight, down_bias
        )
        np.testing.assert_array_equal(gradients[0], jnp.zeros_like(x))
        np.testing.assert_array_equal(gradients[1], jnp.zeros_like(up_weight))
        np.testing.assert_array_equal(gradients[3], jnp.zeros_like(down_weight))
        np.testing.assert_array_equal(gradients[4], jnp.zeros_like(down_bias))
        grouped_bias_gradient = gradients[2].reshape((2, 4))
        np.testing.assert_array_equal(grouped_bias_gradient[:, 0], jnp.zeros(2))
        self.assertTrue(np.all(np.asarray(grouped_bias_gradient[:, 1:]) < 0.0))

    def test_frequency_floor_output_statistics_and_gradients_match_oracle(self) -> None:
        inputs = self.inputs()
        config = FuzzyTopKBalanceConfig(
            top_k=16,
            mode="bias",
            alive_mode="frequency_floor",
            alive_target=0.2,
        )
        output_cotangent = jax.random.normal(jax.random.key(27), inputs[0].shape)
        statistic_cotangent = jnp.asarray((0.3, 0.0, 0.7), jnp.float32)

        def objective(operation, operands):
            output, statistics = operation(*operands, config=config)
            return jnp.sum(output * output_cotangent) + jnp.sum(
                statistics * statistic_cotangent
            )

        actual = fuzzy_topk_mlp_with_balance(*inputs, config=config)
        expected = naive_fuzzy_topk_mlp_with_balance(*inputs, config=config)
        for actual_value, expected_value in zip(actual, expected, strict=True):
            np.testing.assert_allclose(
                actual_value, expected_value, rtol=2e-5, atol=2e-6
            )
        actual_gradients = jax.grad(
            lambda *operands: objective(fuzzy_topk_mlp_with_balance, operands),
            argnums=(0, 1, 2, 3, 4),
        )(*inputs)
        expected_gradients = jax.grad(
            lambda *operands: objective(naive_fuzzy_topk_mlp_with_balance, operands),
            argnums=(0, 1, 2, 3, 4),
        )(*inputs)
        for actual_gradient, expected_gradient in zip(
            actual_gradients, expected_gradients, strict=True
        ):
            np.testing.assert_allclose(
                actual_gradient, expected_gradient, rtol=3e-5, atol=2e-6
            )

    def test_frequency_floor_is_zero_when_every_feature_meets_target(self) -> None:
        x = jnp.eye(4, dtype=jnp.float32)[None, :, :]
        up_weight = jnp.concatenate((jnp.eye(4), jnp.eye(4)), axis=1)
        up_bias = jnp.zeros((8,), jnp.float32)
        down_weight = jnp.zeros((8, 4), jnp.float32)
        down_bias = jnp.zeros((4,), jnp.float32)
        config = FuzzyTopKBalanceConfig(
            top_k=2,
            mode="none",
            alive_mode="frequency_floor",
            alive_target=1.0,
        )

        def loss(bias):
            return fuzzy_topk_mlp_with_balance(
                x, up_weight, bias, down_weight, down_bias, config=config
            )[1][2]

        self.assertEqual(float(loss(up_bias)), 0.0)
        np.testing.assert_array_equal(jax.grad(loss)(up_bias), jnp.zeros_like(up_bias))

    def test_explicit_mesh_boundary_is_trainable(self) -> None:
        inputs = list(self.inputs())
        device_count = min(4, jax.device_count())
        if device_count > inputs[0].shape[0]:
            inputs[0] = jnp.tile(inputs[0], (device_count, 1, 1))[:device_count]
        config = FuzzyTopKBalanceConfig(
            top_k=16,
            mode="bias",
            alive_mode="frequency_floor",
            alive_target=0.2,
        )
        mesh = Mesh(np.asarray(jax.devices()[:device_count], dtype=object), ("data",))
        operation = make_mesh_fuzzy_topk_mlp_with_balance(config=config, mesh=mesh)
        output, statistics = jax.jit(operation)(*inputs)
        expected_output, expected_statistics = naive_fuzzy_topk_mlp_with_balance(
            *inputs, config=config
        )
        np.testing.assert_allclose(output, expected_output, rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(
            statistics, expected_statistics, rtol=2e-5, atol=2e-6
        )
        gradients = jax.grad(
            lambda *args: (
                jnp.square(operation(*args)[0]).mean()
                + 0.01 * operation(*args)[1].sum()
            )
        )(*inputs)
        self.assertEqual(gradients.shape, inputs[0].shape)

    def test_config_rejects_invalid_temperature(self) -> None:
        with self.assertRaisesRegex(ValueError, "temperature"):
            FuzzyTopKBalanceConfig(top_k=16, temperature=0.0)
        with self.assertRaisesRegex(ValueError, "alive mode"):
            FuzzyTopKBalanceConfig(top_k=16, alive_mode="unknown")
        with self.assertRaisesRegex(ValueError, "alive_target"):
            FuzzyTopKBalanceConfig(top_k=16, alive_target=1.1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
