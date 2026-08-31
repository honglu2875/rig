from __future__ import annotations

import os

# Keep these focused numerical tests independent of a TPU runtime. The kernel's
# TPU compilation and throughput are exercised by the profiling target.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from rig.kernels import (
    tiled_tied_cross_entropy,
    tiled_tied_cross_entropy_losses,
    tiled_tied_weighted_dual_cross_entropy,
    tiled_tied_weighted_dual_cross_entropy_losses,
)


def dense_losses(
    hidden: jax.Array,
    embedding: jax.Array,
    targets: jax.Array,
    semantic_vocab_size: int,
    compute_dtype: jnp.dtype,
) -> jax.Array:
    logits = jnp.einsum(
        "...d,vd->...v",
        hidden.astype(compute_dtype),
        embedding[:semantic_vocab_size].astype(compute_dtype),
        preferred_element_type=jnp.float32,
    )
    selected = jnp.take_along_axis(
        jax.nn.log_softmax(logits, axis=-1), targets[..., None], axis=-1
    )
    return -selected[..., 0]


def dense_weighted_dual_losses(
    hidden: jax.Array,
    embedding: jax.Array,
    primary_targets: jax.Array,
    distill_targets: jax.Array,
    primary_weight: float,
    distill_weight: float,
    semantic_vocab_size: int,
    compute_dtype: jnp.dtype,
) -> jax.Array:
    return primary_weight * dense_losses(
        hidden, embedding, primary_targets, semantic_vocab_size, compute_dtype
    ) + distill_weight * dense_losses(
        hidden, embedding, distill_targets, semantic_vocab_size, compute_dtype
    )


def jaxpr_shapes(value: object) -> set[tuple[int, ...]]:
    """Collect array shapes recursively, including nested custom-VJP jaxprs."""

    shapes: set[tuple[int, ...]] = set()
    visited: set[int] = set()

    def visit(item: object) -> None:
        if id(item) in visited:
            return
        visited.add(id(item))
        aval = getattr(item, "aval", None)
        shape = getattr(aval, "shape", None)
        if shape is not None:
            shapes.add(tuple(shape))
        if isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)
        jaxpr = getattr(item, "jaxpr", None)
        if jaxpr is not None:
            visit(jaxpr)
        for equation in getattr(item, "eqns", ()):
            visit(equation.invars)
            visit(equation.outvars)
            visit(equation.params)

    visit(value)
    return shapes


class TiledTiedCrossEntropyTests(unittest.TestCase):
    def setUp(self) -> None:
        hidden_rng = np.random.default_rng(7)
        embedding_rng = np.random.default_rng(11)
        self.hidden = jnp.asarray(hidden_rng.normal(size=(2, 3, 5)).astype(np.float32))
        # Deliberately use a semantic vocabulary, padded storage, and a tile
        # size that divides neither one.
        self.embedding = jnp.asarray(
            embedding_rng.normal(size=(13, 5)).astype(np.float32)
        )
        self.targets = jnp.asarray([[0, 1, 10], [3, 8, 6]], dtype=jnp.int32)

    def test_values_match_dense_oracle_with_padded_storage(self) -> None:
        expected = dense_losses(
            self.hidden, self.embedding, self.targets, 11, jnp.float32
        )
        actual = jax.jit(
            lambda hidden, embedding, targets: tiled_tied_cross_entropy_losses(
                hidden,
                embedding,
                targets,
                semantic_vocab_size=11,
                vocab_tile_size=4,
                compute_dtype=jnp.float32,
            )
        )(self.hidden, self.embedding, self.targets)
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)

    def test_value_and_both_gradients_match_dense_oracle(self) -> None:
        def tiled(hidden: jax.Array, embedding: jax.Array) -> jax.Array:
            return tiled_tied_cross_entropy(
                hidden,
                embedding,
                self.targets,
                semantic_vocab_size=11,
                vocab_tile_size=4,
                compute_dtype=jnp.float32,
            )

        def dense(hidden: jax.Array, embedding: jax.Array) -> jax.Array:
            return jnp.mean(
                dense_losses(hidden, embedding, self.targets, 11, jnp.float32),
                dtype=jnp.float32,
            )

        expected = jax.value_and_grad(dense, argnums=(0, 1))(
            self.hidden, self.embedding
        )
        actual = jax.jit(jax.value_and_grad(tiled, argnums=(0, 1)))(
            self.hidden, self.embedding
        )
        np.testing.assert_allclose(actual[0], expected[0], rtol=2e-6, atol=2e-6)
        np.testing.assert_allclose(actual[1][0], expected[1][0], rtol=3e-6, atol=3e-6)
        np.testing.assert_allclose(actual[1][1], expected[1][1], rtol=3e-6, atol=3e-6)
        # Storage-only rows must receive exactly zero output-head gradient.
        np.testing.assert_array_equal(actual[1][1][11:], 0.0)

    def test_weighted_dual_values_and_gradients_share_one_normalizer(self) -> None:
        distill_targets = jnp.asarray([[2, 1, 7], [9, 4, 0]], dtype=jnp.int32)

        def tiled(hidden: jax.Array, embedding: jax.Array) -> jax.Array:
            return tiled_tied_weighted_dual_cross_entropy(
                hidden,
                embedding,
                self.targets,
                distill_targets,
                primary_weight=0.7,
                distill_weight=0.3,
                semantic_vocab_size=11,
                vocab_tile_size=4,
                compute_dtype=jnp.float32,
            )

        def dense(hidden: jax.Array, embedding: jax.Array) -> jax.Array:
            return jnp.mean(
                dense_weighted_dual_losses(
                    hidden,
                    embedding,
                    self.targets,
                    distill_targets,
                    0.7,
                    0.3,
                    11,
                    jnp.float32,
                ),
                dtype=jnp.float32,
            )

        expected = jax.value_and_grad(dense, argnums=(0, 1))(
            self.hidden, self.embedding
        )
        actual = jax.jit(jax.value_and_grad(tiled, argnums=(0, 1)))(
            self.hidden, self.embedding
        )
        np.testing.assert_allclose(actual[0], expected[0], rtol=2e-6, atol=2e-6)
        np.testing.assert_allclose(actual[1][0], expected[1][0], rtol=3e-6, atol=3e-6)
        np.testing.assert_allclose(actual[1][1], expected[1][1], rtol=3e-6, atol=3e-6)
        np.testing.assert_array_equal(actual[1][1][11:], 0.0)

    def test_weighted_dual_per_token_losses_allow_non_unit_total_weight(self) -> None:
        distill_targets = jnp.asarray([[2, 1, 7], [9, 4, 0]], dtype=jnp.int32)
        expected = dense_weighted_dual_losses(
            self.hidden,
            self.embedding,
            self.targets,
            distill_targets,
            0.8,
            0.5,
            11,
            jnp.float32,
        )
        actual = tiled_tied_weighted_dual_cross_entropy_losses(
            self.hidden,
            self.embedding,
            self.targets,
            distill_targets,
            primary_weight=0.8,
            distill_weight=0.5,
            semantic_vocab_size=11,
            vocab_tile_size=4,
            compute_dtype=jnp.float32,
        )
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)

    def test_weighted_dual_rejects_invalid_weights(self) -> None:
        for primary_weight, distill_weight in ((0.0, 0.0), (-0.1, 1.1)):
            with self.subTest(
                primary_weight=primary_weight, distill_weight=distill_weight
            ), self.assertRaises(ValueError):
                tiled_tied_weighted_dual_cross_entropy(
                    self.hidden,
                    self.embedding,
                    self.targets,
                    self.targets,
                    primary_weight=primary_weight,
                    distill_weight=distill_weight,
                    semantic_vocab_size=11,
                )

    def test_bfloat16_compute_is_close_and_returns_input_gradient_dtypes(self) -> None:
        hidden = self.hidden.astype(jnp.bfloat16)

        def loss(hidden: jax.Array, embedding: jax.Array) -> jax.Array:
            return tiled_tied_cross_entropy(
                hidden,
                embedding,
                self.targets,
                semantic_vocab_size=11,
                vocab_tile_size=7,
                compute_dtype=jnp.bfloat16,
            )

        value, gradients = jax.jit(jax.value_and_grad(loss, argnums=(0, 1)))(
            hidden, self.embedding
        )
        oracle = jnp.mean(
            dense_losses(hidden, self.embedding, self.targets, 11, jnp.bfloat16),
            dtype=jnp.float32,
        )
        np.testing.assert_allclose(value, oracle, rtol=4e-3, atol=4e-3)
        self.assertEqual(value.dtype, jnp.float32)
        self.assertEqual(gradients[0].dtype, hidden.dtype)
        self.assertEqual(gradients[1].dtype, self.embedding.dtype)

    def test_bfloat16_wide_target_dot_matches_denominator_contract(self) -> None:
        """Catch target reductions that silently differ from the MXU dot."""

        rng = np.random.default_rng(29)
        hidden = jnp.asarray(rng.normal(0.0, 0.5, size=(2, 3, 768)).astype(np.float32))
        embedding = jnp.asarray(
            rng.normal(0.0, 0.5, size=(259, 768)).astype(np.float32)
        )
        targets = jnp.asarray([[0, 127, 256], [7, 128, 42]], jnp.int32)

        def tiled(h: jax.Array, e: jax.Array) -> jax.Array:
            return tiled_tied_cross_entropy(
                h,
                e,
                targets,
                semantic_vocab_size=257,
                vocab_tile_size=128,
                compute_dtype=jnp.bfloat16,
            )

        def dense(h: jax.Array, e: jax.Array) -> jax.Array:
            return jnp.mean(
                dense_losses(h, e, targets, 257, jnp.bfloat16),
                dtype=jnp.float32,
            )

        expected = jax.value_and_grad(dense, argnums=(0, 1))(hidden, embedding)
        actual = jax.jit(jax.value_and_grad(tiled, argnums=(0, 1)))(hidden, embedding)
        # Online log-sum-exp combines static tiles in a different order from
        # the dense oracle; the target dot itself now has the identical MXU
        # accumulation contract, leaving only small softmax-order roundoff.
        np.testing.assert_allclose(actual[0], expected[0], rtol=2e-4, atol=2e-4)
        np.testing.assert_allclose(actual[1][0], expected[1][0], rtol=5e-3, atol=5e-3)
        np.testing.assert_allclose(actual[1][1], expected[1][1], rtol=5e-3, atol=5e-3)

    def test_invalid_targets_have_infinite_loss_and_zero_gradients(self) -> None:
        invalid = jnp.asarray([[-1, 1, 99], [3, 8, 6]], dtype=jnp.int32)

        def loss(hidden: jax.Array, embedding: jax.Array) -> jax.Array:
            return jnp.sum(
                tiled_tied_cross_entropy_losses(
                    hidden,
                    embedding,
                    invalid,
                    semantic_vocab_size=11,
                    vocab_tile_size=4,
                    compute_dtype=jnp.float32,
                )
            )

        value, gradients = jax.value_and_grad(loss, argnums=(0, 1))(
            self.hidden, self.embedding
        )
        self.assertTrue(bool(jnp.isinf(value)))

        valid_weights = jnp.asarray([[0.0, 1.0, 0.0], [1.0, 1.0, 1.0]])

        def valid_only(hidden: jax.Array, embedding: jax.Array) -> jax.Array:
            losses = tiled_tied_cross_entropy_losses(
                hidden,
                embedding,
                jnp.clip(invalid, 0, 10),
                semantic_vocab_size=11,
                vocab_tile_size=4,
                compute_dtype=jnp.float32,
            )
            return jnp.sum(losses * valid_weights)

        expected = jax.grad(valid_only, argnums=(0, 1))(self.hidden, self.embedding)
        np.testing.assert_allclose(gradients[0], expected[0], rtol=3e-6, atol=3e-6)
        np.testing.assert_allclose(gradients[1], expected[1], rtol=3e-6, atol=3e-6)

    def test_per_token_cotangents_support_masked_evaluation(self) -> None:
        weights = jnp.asarray([[1.0, 0.0, 0.25], [0.5, 1.0, 0.0]])

        def tiled(hidden: jax.Array, embedding: jax.Array) -> jax.Array:
            losses = tiled_tied_cross_entropy_losses(
                hidden,
                embedding,
                self.targets,
                semantic_vocab_size=11,
                vocab_tile_size=4,
                compute_dtype=jnp.float32,
            )
            return jnp.sum(losses * weights)

        def dense(hidden: jax.Array, embedding: jax.Array) -> jax.Array:
            return jnp.sum(
                dense_losses(hidden, embedding, self.targets, 11, jnp.float32) * weights
            )

        expected = jax.value_and_grad(dense, argnums=(0, 1))(
            self.hidden, self.embedding
        )
        actual = jax.value_and_grad(tiled, argnums=(0, 1))(self.hidden, self.embedding)
        np.testing.assert_allclose(actual[0], expected[0], rtol=2e-6, atol=2e-6)
        np.testing.assert_allclose(actual[1][0], expected[1][0], rtol=3e-6, atol=3e-6)
        np.testing.assert_allclose(actual[1][1], expected[1][1], rtol=3e-6, atol=3e-6)

    def test_forward_and_backward_jaxprs_never_contain_full_logits(self) -> None:
        def loss(hidden: jax.Array, embedding: jax.Array) -> jax.Array:
            return tiled_tied_cross_entropy(
                hidden,
                embedding,
                self.targets,
                semantic_vocab_size=11,
                vocab_tile_size=4,
                compute_dtype=jnp.float32,
            )

        forward_shapes = jaxpr_shapes(jax.make_jaxpr(loss)(self.hidden, self.embedding))
        backward_shapes = jaxpr_shapes(
            jax.make_jaxpr(jax.grad(loss, argnums=(0, 1)))(self.hidden, self.embedding)
        )
        for shapes in (forward_shapes, backward_shapes):
            self.assertNotIn((6, 11), shapes)
            self.assertNotIn((2, 3, 11), shapes)
            self.assertNotIn((6, 13), shapes)
        self.assertIn((6, 4), forward_shapes)
        self.assertIn((6, 4), backward_shapes)

    def test_invalid_contracts_fail_before_compilation(self) -> None:
        cases = (
            dict(semantic_vocab_size=0, vocab_tile_size=4),
            dict(semantic_vocab_size=14, vocab_tile_size=4),
            dict(semantic_vocab_size=11, vocab_tile_size=0),
        )
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                tiled_tied_cross_entropy(
                    self.hidden, self.embedding, self.targets, **kwargs
                )
        with self.assertRaises(TypeError):
            tiled_tied_cross_entropy(
                self.hidden,
                self.embedding,
                self.targets.astype(jnp.float32),
                semantic_vocab_size=11,
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
