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
        actual_gradients = jax.grad(actual_loss, argnums=(0, 1, 2, 3, 4))(*inputs)
        expected_gradients = jax.grad(expected_loss, argnums=(0, 1, 2, 3, 4))(*inputs)
        for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
            np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=2e-6)

    def test_explicit_mesh_boundary_is_trainable(self) -> None:
        inputs = self.inputs()
        config = FuzzyTopKBalanceConfig(top_k=16, alive_margin=0.05)
        mesh = Mesh(np.asarray(jax.devices()[:1], dtype=object), ("data",))
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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
