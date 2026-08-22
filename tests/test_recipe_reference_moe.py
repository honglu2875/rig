"""Gates for the routed (mixture-of-experts) recipe.

These ran as throwaway scripts while the recipe was being written, which is
exactly why two regressions reached a TPU: the grouped matmul is a Mosaic
kernel and cannot be auto-partitioned, and the balance loss silently degenerates
if it is handed an unreduced probability matrix. Both are checkable on CPU in
under a second, so both are checked here.
"""

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
from jax.sharding import Mesh, PartitionSpec as P  # noqa: E402


TRAINER_PATH = Path(__file__).parents[1] / "recipes" / "reference_moe" / "train.py"
SPEC = importlib.util.spec_from_file_location("reference_moe_train", TRAINER_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - importlib invariant
    raise RuntimeError(f"could not import {TRAINER_PATH}")
trainer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = trainer
SPEC.loader.exec_module(trainer)

_LOADED_CONFIGS = {
    profile: trainer.load_experiment_config(profile)
    for profile in ("smoke", "dev", "official")
}


def _resolve_config(args, platform: str):
    experiment_config, config_sha256 = _LOADED_CONFIGS[trainer.selected_profile(args)]
    return trainer.resolve_config(
        args,
        platform,
        experiment_config=experiment_config,
        config_sha256=config_sha256,
    )


EXPERTS = 8
TOP_K = 2
WIDTH = 128
HIDDEN = 256


def _weights(seed: int = 0):
    """Router and expert stacks shaped as init_params lays them out."""

    rng = np.random.default_rng(seed)

    def draw(*shape):
        return jnp.asarray(rng.normal(size=shape) * 0.05, jnp.float32)

    return (
        draw(WIDTH, EXPERTS),
        draw(EXPERTS, WIDTH, HIDDEN),
        draw(EXPERTS, HIDDEN),
        draw(EXPERTS, HIDDEN, WIDTH),
        draw(EXPERTS, WIDTH),
    )


def _routed(x, weights, *, axis_name=None):
    return trainer.routed_mlp_local(
        x,
        *weights,
        experts=EXPERTS,
        top_k=TOP_K,
        dtype=jnp.float32,
        axis_name=axis_name,
    )


def _dense_reference(x, weights):
    """The same computation with no grouping: every expert sees every token.

    Deliberately written the slow, obvious way. If the grouped matmul, the
    argsort, the scatter-add, or the gate normalization is wrong, this disagrees.
    """

    router_w, up_w, up_b, down_w, down_b = weights
    batch, length, width = x.shape
    flat = x.reshape(batch * length, width)

    logits = flat @ router_w
    chosen_logits, chosen = jax.lax.top_k(logits, TOP_K)
    gate = jax.nn.softmax(chosen_logits, axis=-1)

    out = np.zeros((batch * length, width), np.float32)
    for token in range(batch * length):
        for slot in range(TOP_K):
            expert = int(chosen[token, slot])
            hidden = flat[token] @ up_w[expert] + up_b[expert]
            hidden = jax.nn.gelu(hidden, approximate=True)
            out[token] += float(gate[token, slot]) * np.asarray(
                hidden @ down_w[expert] + down_b[expert]
            )
    return jnp.asarray(out).reshape(batch, length, width)


class RoutedMlpTests(unittest.TestCase):
    def test_matches_a_dense_per_expert_reference(self) -> None:
        # tokens * top_k must be a multiple of the grouped matmul's 128 m-tile.
        x = jnp.asarray(
            np.random.default_rng(1).normal(size=(2, 32, WIDTH)) * 0.5, jnp.float32
        )
        weights = _weights()
        got, _, _, _ = _routed(x, weights)
        want = _dense_reference(x, weights)
        self.assertLess(float(jnp.abs(got - want).max()), 1e-5)

    def test_routing_is_dropless(self) -> None:
        """Every assignment is served, so no capacity factor exists to tune.

        ``group_sizes`` is data while the total row count is static, which is
        what lets the grouped matmul stay dropless. If a token were ever
        dropped the realized load would sum to less than one.
        """

        x = jnp.asarray(
            np.random.default_rng(2).normal(size=(4, 32, WIDTH)) * 3.0, jnp.float32
        )
        _, _, load, _ = _routed(x, _weights())
        self.assertAlmostEqual(float(load.sum()), 1.0, places=5)
        self.assertTrue(bool((load >= 0).all()))

    def test_sharded_wrapper_agrees_with_the_local_body(self) -> None:
        """The regression that reached a TPU: gmm needs an explicit shard_map.

        An outer jit refuses to partition a Mosaic kernel at all -- "Mosaic
        kernels cannot be automatically partitioned" -- so the routed MLP needs
        its own sharded boundary exactly as attention does. This runs the real
        wrapper on a multi-device CPU mesh and requires it to reproduce the
        unsharded answer.

        The CPU symptom differs from the TPU one: gmm runs in interpret mode
        here, so deleting the wrapper trips the unbound ``data`` axis of the
        statistics pmean rather than the Mosaic message. Either way the guard
        fires, which is what makes it worth keeping on a CPU suite.
        """

        devices = jax.devices()
        self.assertGreaterEqual(len(devices), 8, "conftest forces 8 CPU devices")
        mesh = Mesh(np.asarray(devices[:8]).reshape(8), ("data",))

        # One sequence per device, and 64 * top_k = 128 rows locally -- the
        # smallest shape that satisfies the m-tile on every shard.
        x = jnp.asarray(
            np.random.default_rng(3).normal(size=(8, 64, WIDTH)), jnp.float32
        )
        weights = _weights()
        config = trainer.Config.__new__(trainer.Config)
        object.__setattr__(config, "experts", EXPERTS)
        object.__setattr__(config, "expert_top_k", TOP_K)
        object.__setattr__(config, "dtype_name", "float32")

        sharded = trainer.make_mesh_routed_mlp(config, mesh)
        with jax.set_mesh(mesh):
            got, got_probability, got_load, got_summary = jax.jit(sharded)(x, *weights)
        want, want_probability, want_load, want_summary = _routed(x, weights)

        self.assertLess(float(jnp.abs(got - want).max()), 1e-5)
        # The two statistics are pmean'd across the data axis, so the sharded
        # run must see the same global load the unsharded one does -- not one
        # device's view of it.
        self.assertLess(float(jnp.abs(got_load - want_load).max()), 1e-6)
        self.assertLess(float(jnp.abs(got_probability - want_probability).max()), 1e-6)
        self.assertLess(float(jnp.abs(got_summary - want_summary).max()), 1e-5)

    def test_dense_mlp_is_untouched_when_experts_is_zero(self) -> None:
        config = trainer.Config.__new__(trainer.Config)
        object.__setattr__(config, "experts", 0)
        self.assertIsNone(trainer.make_mesh_routed_mlp(config, None))


class BalanceLossTests(unittest.TestCase):
    def test_uniform_load_is_the_minimum_and_equals_one(self) -> None:
        uniform = jnp.full((EXPERTS,), 1.0 / EXPERTS, jnp.float32)
        self.assertAlmostEqual(
            float(trainer.load_balance_loss(uniform, uniform)), 1.0, places=5
        )

        collapsed = jnp.zeros((EXPERTS,), jnp.float32).at[0].set(1.0)
        self.assertAlmostEqual(
            float(trainer.load_balance_loss(collapsed, collapsed)),
            float(EXPERTS),
            places=5,
        )
        self.assertGreater(
            float(trainer.load_balance_loss(collapsed, collapsed)),
            float(trainer.load_balance_loss(uniform, uniform)),
        )

    def test_rejects_an_unreduced_probability_matrix(self) -> None:
        """Passing [tokens, E] here is a silent no-op, not a loud error.

        Averaging a per-expert vector over axis 0 gives a scalar, which makes
        the term collapse to a constant 1.0 carrying no gradient to the router.
        The model trains normally and simply never balances, so the shape has
        to be rejected rather than broadcast.
        """

        with self.assertRaisesRegex(ValueError, "already reduced over"):
            trainer.load_balance_loss(
                jnp.full((32, EXPERTS), 1.0 / EXPERTS, jnp.float32),
                jnp.full((EXPERTS,), 1.0 / EXPERTS, jnp.float32),
            )

    def test_gradient_pushes_an_overloaded_expert_down(self) -> None:
        load = jnp.asarray([0.5, 0.2, 0.1, 0.1, 0.05, 0.03, 0.01, 0.01], jnp.float32)
        grad = jax.grad(lambda p: trainer.load_balance_loss(p, load))(
            jnp.full((EXPERTS,), 1.0 / EXPERTS, jnp.float32)
        )
        # d/dP_i = E * f_i, so the busiest expert has the steepest slope and
        # gradient descent lowers its router probability the most.
        self.assertEqual(int(jnp.argmax(grad)), int(jnp.argmax(load)))


class ActiveParameterTests(unittest.TestCase):
    def _config(self, tier: str):
        # platform="tpu" only gets past the guard on tpu_flash; nothing here
        # executes, and parameter counts do not depend on the backend.
        parser = trainer.build_parser()
        return _resolve_config(
            parser.parse_args(["--tier", tier, "--profile", "dev"]), "tpu"
        )

    def test_run_card_names_routing_and_active_versus_total_scale(self) -> None:
        rows = dict(
            trainer.model_console_rows(
                self._config("60m"),
                total_parameters=102_510_000,
                active_parameters=60_110_000,
            )
        )

        self.assertIn("MoE 8×top-2", rows["model"])
        self.assertEqual(rows["parameters"], "60.11M active / 102.51M total")
        self.assertIn("dropless", rows["routing"])
        self.assertIn("0.01", rows["routing"])

    def test_active_count_exceeds_the_dense_tier_only_by_the_router(self) -> None:
        """A routed tier is sized by its *active* parameters, not its total.

        The first TPU smoke test failed because the check compared the declared
        count against the total, which a routed model necessarily exceeds. The
        declared number stays the dense tier size so the sparse and dense
        ladders line up; routing then adds exactly two things, the router
        projection and one extra set of expert biases per additional expert a
        token visits. Both are named rather than absorbed into a tolerance, so
        an unaccounted parameter is a failure and not a rounding difference.
        """

        for tier in ("60m", "125m", "250m", "500m"):
            with self.subTest(tier=tier):
                config = self._config(tier)
                declared = config.declared_parameters
                excess = trainer.expected_active_parameters(config) - declared
                self.assertEqual(
                    excess,
                    config.layers
                    * (
                        config.d_model * config.experts
                        + (config.expert_top_k - 1) * config.d_model
                    ),
                )
                # Small enough that the sparse tier is still the tier it claims
                # to be rather than a quietly larger model.
                self.assertLess(excess / declared, 0.001)

    def test_the_counter_agrees_with_the_closed_form(self) -> None:
        # Only the smallest tier is materialized: 500m totals over a billion
        # parameters, which is minutes and gigabytes for no extra coverage.
        config = self._config("60m")
        params = trainer.init_params(config, 1337)
        self.assertEqual(
            trainer.active_parameter_count(params, config),
            trainer.expected_active_parameters(config),
        )
        # Routing is only worth its complexity if total exceeds active.
        self.assertGreater(
            trainer.parameter_count(params),
            trainer.active_parameter_count(params, config),
        )


class WeightDecayPolicyTests(unittest.TestCase):
    def test_stacked_expert_biases_are_not_decayed(self) -> None:
        params = {
            "token_embedding": np.zeros((16, 8), dtype=np.float32),
            "blocks": [
                {
                    "router_w": np.zeros((8, 4), dtype=np.float32),
                    "expert_up_w": np.zeros((4, 8, 16), dtype=np.float32),
                    "expert_up_b": np.zeros((4, 16), dtype=np.float32),
                    "expert_down_w": np.zeros((4, 16, 8), dtype=np.float32),
                    "expert_down_b": np.zeros((4, 8), dtype=np.float32),
                    "ln1_scale": np.ones((8,), dtype=np.float32),
                }
            ],
            "output_embedding": np.zeros((16, 8), dtype=np.float32),
        }

        mask = trainer.weight_decay_mask(params)

        self.assertTrue(mask["token_embedding"])
        self.assertTrue(mask["output_embedding"])
        self.assertTrue(mask["blocks"][0]["router_w"])
        self.assertTrue(mask["blocks"][0]["expert_up_w"])
        self.assertTrue(mask["blocks"][0]["expert_down_w"])
        self.assertFalse(mask["blocks"][0]["expert_up_b"])
        self.assertFalse(mask["blocks"][0]["expert_down_b"])
        self.assertFalse(mask["blocks"][0]["ln1_scale"])

    def test_new_parameter_roles_require_an_explicit_policy(self) -> None:
        with self.assertRaisesRegex(ValueError, "no rule for parameter 'mystery'"):
            trainer.weight_decay_mask({"mystery": np.zeros((8, 8), dtype=np.float32)})


class ContextPresetTests(unittest.TestCase):
    def test_default_output_directory_tracks_recipe_folder(self) -> None:
        self.assertEqual(trainer.RECIPE_NAME, "reference_moe")
        self.assertEqual(
            trainer.build_parser().parse_args([]).output_dir,
            Path("runs/reference_moe"),
        )

    def test_moe_defaults_to_8k_and_can_select_the_aligned_1k_preset(self) -> None:
        experiment_config, _ = trainer.load_experiment_config("dev")
        native_tier, native_context = experiment_config.resolve_selection()
        short_tier, short_context = experiment_config.resolve_selection(context="1k")

        self.assertEqual(native_tier, short_tier)
        self.assertEqual(native_context, "8k")
        native = experiment_config.family.contexts[native_context]
        self.assertEqual(
            (native.seq_len, native.reference_batch_size),
            (8192, 16),
        )
        self.assertTrue(native.document_masking)
        self.assertEqual(short_context, "1k")
        short = experiment_config.family.contexts[short_context]
        self.assertEqual(
            (short.seq_len, short.reference_batch_size),
            (1024, 128),
        )
        self.assertFalse(short.document_masking)


class RoutedModelTests(unittest.TestCase):
    def _config(self):
        parser = trainer.build_parser()
        config = _resolve_config(parser.parse_args(["--profile", "smoke"]), "cpu")
        from dataclasses import replace

        return replace(config, layers=2, d_model=128, heads=2, seq_len=64)

    def test_gradients_reach_the_router_and_every_expert(self) -> None:
        """A router that receives no gradient looks exactly like a working one.

        Both produce finite losses that go down, because the experts keep
        learning either way. Only the gradient tells them apart.
        """

        config = self._config()
        params = trainer.init_params(config, 1337)
        tokens = jnp.asarray(
            np.random.default_rng(4).integers(
                0, config.semantic_vocab_size, size=(2, config.seq_len + 1)
            )
        )
        grads = jax.grad(
            lambda p: trainer.cross_entropy(p, tokens[:, :-1], tokens[:, 1:], config)
        )(params)

        for index, block in enumerate(grads["blocks"]):
            with self.subTest(layer=index):
                for name in (
                    "router_w",
                    "expert_up_w",
                    "expert_down_w",
                    "expert_up_b",
                    "expert_down_b",
                ):
                    magnitude = float(jnp.abs(block[name]).max())
                    self.assertTrue(np.isfinite(magnitude), f"{name} not finite")
                    self.assertGreater(magnitude, 0.0, f"{name} gets no gradient")
                # Per-expert, not just in aggregate: one dead expert would
                # otherwise hide inside a healthy stack-wide maximum.
                per_expert = jnp.abs(block["expert_up_w"]).max(axis=(1, 2))
                self.assertTrue(
                    bool((per_expert > 0).all()), f"dead experts: {per_expert}"
                )
        self.assertGreater(float(jnp.abs(grads["token_embedding"]).max()), 0.0)

    def test_a_real_training_step_does_not_crash_on_the_history_write(self) -> None:
        """Reached a TPU: init_optimizer once took an explicit width argument.

        A caller could construct a routed config and forget to pass
        router_row_width(config) alongside it, which built a 3-wide history
        while train_step tried to write a much wider row into it -- and no
        prior test called init_optimizer the way the recipe actually does, so
        a pure check of router_row_width in isolation had not caught it.
        init_optimizer now takes the config directly and derives the width
        itself, which removes the duplicated call rather than just detecting
        it -- so this test calls it exactly as command_run does, with nothing
        this test could get right that the recipe could get wrong beside it.
        """

        config = self._config()
        params = trainer.init_params(config, 1337)
        optimizer = trainer.init_optimizer(params, config)
        # The real loop moves the optimizer state onto the mesh with
        # put_replicated_tree before the first step; a single-device jnp.asarray
        # is the part of that conversion this test actually needs -- ``history``
        # must be a jax.Array so ``.at[].set(...)`` is defined on it.
        optimizer = jax.tree_util.tree_map(jnp.asarray, optimizer)
        tokens = jnp.asarray(
            np.random.default_rng(5).integers(
                0, config.semantic_vocab_size, size=(2, config.seq_len + 1)
            )
        )
        _, optimizer, step_metrics = trainer.train_step(
            params, optimizer, tokens[:, :-1], tokens[:, 1:], config, None
        )
        self.assertTrue(np.isfinite(float(step_metrics["loss"])))
        self.assertEqual(
            optimizer["history"].shape[1], 3 + trainer.router_row_width(config)
        )


class RouterLoggingTests(unittest.TestCase):
    """The columns and the values are declared in two places; they must agree.

    A mismatch here is silent and total: every routing series in the artifact
    is shifted by one, so entropy is plotted as a gate weight and expert 7's
    load is read as expert 6's. Nothing crashes and no number looks absurd.
    """

    LAYERS = 3

    def _router(self):
        rng = np.random.default_rng(11)
        load = rng.dirichlet(np.ones(EXPERTS), size=self.LAYERS).astype(np.float32)
        summary = rng.uniform(size=(self.LAYERS, 3)).astype(np.float32)
        stats = trainer.RouterStats(
            balance_loss=jnp.float32(1.25),
            load=jnp.asarray(load),
            summary=jnp.asarray(summary),
        )
        return stats, load, summary

    def test_row_width_matches_the_column_count(self) -> None:
        from rig.runlog import training_log_columns

        columns = training_log_columns(self.LAYERS, EXPERTS)
        row = trainer.router_row(self._router()[0])
        # Three dense scalars are written ahead of the routing ones.
        self.assertEqual(len(columns), 3 + int(row.shape[0]))

    def test_declared_width_matches_the_row_it_sizes(self) -> None:
        """The history buffer is allocated from the declared width alone.

        If it disagrees with the row, the run dies at the first step with a
        shape error -- or worse, silently truncates.
        """

        config = trainer.Config.__new__(trainer.Config)
        object.__setattr__(config, "experts", EXPERTS)
        object.__setattr__(config, "layers", self.LAYERS)
        self.assertEqual(
            trainer.router_row_width(config),
            int(trainer.router_row(self._router()[0]).shape[0]),
        )
        object.__setattr__(config, "experts", 0)
        self.assertEqual(trainer.router_row_width(config), 0)

    def test_every_value_lands_in_the_column_that_names_it(self) -> None:
        from rig import metrics as rig_metrics
        from rig.runlog import ROUTER_SUMMARY_METRICS, training_log_columns

        stats, load, summary = self._router()
        columns = training_log_columns(self.LAYERS, EXPERTS)[3:]
        values = np.asarray(trainer.router_row(stats))

        expected = {}
        for index, name in enumerate(ROUTER_SUMMARY_METRICS):
            expected[(name, -1, -1)] = float(summary[:, index].mean())
            for layer in range(self.LAYERS):
                expected[(name, layer, -1)] = float(summary[layer, index])
        expected[("router.balance_loss", -1, -1)] = 1.25
        expected[("router.max_load", -1, -1)] = float(load.max())
        expected[("router.min_load", -1, -1)] = float(load.min())
        for layer in range(self.LAYERS):
            for expert in range(EXPERTS):
                expected[("router.load", layer, expert)] = float(load[layer, expert])

        self.assertEqual(len(columns), len(expected))
        for column, value in zip(columns, values, strict=True):
            key = (
                rig_metrics.metric_by_id(column.metric_id).name,
                column.layer,
                column.index,
            )
            with self.subTest(column=column.describe()):
                self.assertIn(key, expected)
                self.assertAlmostEqual(float(value), expected[key], places=5)

    def test_a_dense_run_logs_no_routing_columns(self) -> None:
        from rig.runlog import training_log_columns

        self.assertEqual(len(training_log_columns()), 3)
        self.assertEqual(int(trainer.router_row(None).shape[0]), 0)

    def test_a_dense_history_stays_three_columns_wide(self) -> None:
        config = trainer.Config.__new__(trainer.Config)
        object.__setattr__(config, "experts", 0)
        object.__setattr__(config, "steps", 10)
        optimizer = trainer.init_optimizer({"w": jnp.zeros((4,))}, config)
        self.assertEqual(optimizer["history"].shape, (10, 3))

    def test_the_end_of_run_rewrite_keeps_the_routing_columns(self) -> None:
        """The bug this whole path exists for: the rewrite superseded them.

        write_training_log replaces the file from the device history buffer at
        the end of a run. When it did not know about the routing columns it
        wrote three, discarding every routing row the run had appended -- and
        the run still finished clean, so only the artifact showed it.
        """

        import tempfile
        from rig import logpack
        from rig.runlog import training_log_columns, write_training_log

        columns = training_log_columns(self.LAYERS, EXPERTS)
        steps = 4
        history = np.arange(steps * len(columns), dtype=np.float32).reshape(
            steps, len(columns)
        )
        with tempfile.TemporaryDirectory() as directory:
            write_training_log(
                Path(directory),
                history,
                tokens_per_step=1024,
                final_step=steps,
                flops_per_token=1,
                columns=columns,
            )
            log = logpack.read_log(Path(directory) / "training.riglog")
            self.assertEqual(len(log.columns), len(columns))
            self.assertEqual(len(log), steps)
            names = {c.describe() for c in log.columns}
            self.assertIn("block[2]/expert[7]/router.load", names)
            self.assertIn("block[0]/router.entropy", names)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
