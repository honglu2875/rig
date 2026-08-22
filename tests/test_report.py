from __future__ import annotations

import base64
import gzip
import hashlib
import json
from pathlib import Path
from typing import Sequence
import re
import tempfile
import unittest

import numpy as np

from rig import logpack
from rig.report import (
    DIAGNOSTICS_LOG_NAME,
    TRAINING_LOG_NAME,
    ReportError,
    export_study,
    _checkpoint_layer_stats,
    _default_run_selection,
    _diagnostic_metric,
    _lttb,
    _compact,
    _check_diagnostic_scopes,
    _subsample_indices,
    build_report,
    build_study_browser,
)


class ReportTests(unittest.TestCase):
    def test_nested_expert_diagnostics_do_not_double_count_parent_blocks(self) -> None:
        columns = (
            logpack.column("grad.l2_norm", "overall", element_count=8),
            logpack.column("grad.l2_norm", "embeddings", element_count=2),
            logpack.column("grad.l2_norm", "unembedding", element_count=2),
            logpack.column("grad.l2_norm", "block", 0, element_count=3),
            logpack.column("grad.l2_norm", "expert", 0, element_count=2, index=0),
            logpack.column("grad.l2_norm", "expert", 0, element_count=2, index=1),
            logpack.column("grad.l2_norm", "final_norm", element_count=1),
        )
        log = logpack.Log(
            columns=columns,
            steps=np.asarray([1], np.int32),
            values=np.zeros((1, len(columns)), np.float32),
            tokens_per_step=1,
            flops_per_token=1.0,
        )

        # Expert counts overlap their parent block. Only the disjoint model
        # scopes participate in the overall element-count identity.
        _check_diagnostic_scopes(log)

    def test_every_successful_run_is_plotted_whatever_its_profile_or_loss(self) -> None:
        # The report once admitted only official runs at the historical
        # 624,984,064-token budget with loss <= 3.76, which excluded the entire
        # current tiered family. Qualification is a leaderboard question now.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "runs"
            cases = {
                "official-good": ("sample_efficiency", "official", 20, 3.0),
                "official-poor": ("sample_efficiency", "official", 20, 9.9),
                "open-partial": ("open", "official", 20, 3.0),
                "development": ("sample_efficiency", "dev", 20, 7.4),
                "smoke-run": ("sample_efficiency", "smoke", 20, 10.5),
            }
            for name, (track, profile, tokens, validation_loss) in cases.items():
                run = runs / name
                run.mkdir(parents=True)
                _write_training(run, [(4.5, 1e-4, 0.5), (4.0, 9e-5, 0.4)])
                _write_result(
                    run,
                    validation_artifact=False,
                    track=track,
                    profile=profile,
                    tokens=tokens,
                    validation_loss=validation_loss,
                )

            summary = build_report(runs, root / "report.html")
            payload = _payload((root / "report.html").read_text(encoding="utf-8"))

        self.assertEqual(summary.included, tuple(sorted(cases)))
        self.assertEqual(summary.skipped, {})
        classifications = {run["id"]: run["classification"] for run in payload["runs"]}
        self.assertEqual(classifications["official-good"], "official")
        self.assertEqual(classifications["official-poor"], "official")
        self.assertEqual(classifications["development"], "diagnostic")
        self.assertEqual(classifications["smoke-run"], "smoke")

    def test_structurally_invalid_results_are_still_rejected(self) -> None:
        # Removing the qualification gates must not weaken integrity checks.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "runs"
            run = runs / "bad-track"
            run.mkdir(parents=True)
            _write_training(run, [(4.5, 1e-4, 0.5), (4.0, 9e-5, 0.4)])
            _write_result(
                run,
                validation_artifact=False,
                track="not_a_track",
                profile="official",
                tokens=20,
                validation_loss=3.0,
            )
            summary = build_report(runs, root / "report.html")

        self.assertEqual(summary.included, ())
        self.assertIn("track is invalid", summary.skipped["bad-track"])

    def test_long_form_diagnostics_build_all_family_and_final_scope_charts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "runs"
            run = runs / "diagnostic-run"
            run.mkdir(parents=True)
            _write_training(run, [(4.5, 1e-4, 0.5), (4.0, 9e-5, 0.4)])
            _write_diagnostics(run / DIAGNOSTICS_LOG_NAME)
            _write_result(
                run,
                validation_artifact=False,
                diagnostics_artifact=True,
            )

            summary = build_report(runs, root / "report.html")
            html = (root / "report.html").read_text(encoding="utf-8")
            payload = _payload(html)

        self.assertEqual(summary.included, (run.name,))
        self.assertEqual(
            {chart["family"] for chart in payload["diagnosticCharts"]},
            {"grad", "update", "param"},
        )
        self.assertEqual(len(payload["diagnosticCharts"]), 18)
        self.assertEqual(len(payload["layerCharts"]), 18)
        parameter_mean = next(
            chart
            for chart in payload["layerCharts"]
            if chart["family"] == "param" and chart["stat"] == "mean"
        )
        # Layer series now carry a fixed scope layout plus one value row per
        # retained step, so the dragger can render any recorded step.
        series = parameter_mean["series"][0]
        self.assertEqual(
            [scope[1] for scope in series["scopes"]],
            ["embeddings", "block 0", "final norm", "unembedding"],
        )
        self.assertEqual(len(series["values"]), len(series["steps"]))
        self.assertTrue(
            all(len(row) == len(series["scopes"]) for row in series["values"])
        )
        self.assertIn('id="family-control"', html)
        self.assertIn('id="focus-dialog"', html)
        self.assertIn('id="smoothing-control"', html)
        self.assertIn('id="x-scale-control"', html)
        self.assertIn('name="x-scale" value="log" checked', html)
        self.assertIn('name="x-scale" value="linear"', html)
        self.assertIn("Math.log10(x)", html)
        self.assertIn('name="smoothing" value="ema"', html)
        self.assertIn('name="smoothing" value="mean"', html)
        self.assertIn('name="smoothing" value="median"', html)
        self.assertIn("function finishBox(e,item)", html)
        self.assertIn("Raw sample:", html)
        self.assertNotIn("onwheel", html)
        self.assertNotIn("function zoom(", html)
        self.assertNotIn("setInterval", html)

    def test_diagnostics_recorded_on_a_different_axis_exclude_the_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "runs"
            run = runs / "truncated-diagnostics"
            run.mkdir(parents=True)
            _write_training(run, [(4.5, 1e-4, 0.5), (4.0, 9e-5, 0.4)])
            # Diagnostics recorded against a different token accounting than the
            # training curve cannot be plotted on the same axis.
            _write_diagnostics(run / DIAGNOSTICS_LOG_NAME, tokens_per_step=77)
            _write_result(
                run,
                validation_artifact=False,
                diagnostics_artifact=True,
            )

            summary = build_report(runs, root / "report.html")

        self.assertFalse(summary.included)
        self.assertIn("token accounting disagrees", summary.skipped[run.name])

    def test_standalone_report_includes_sound_run_and_both_axis_modes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "runs"
            run = runs / "complete-run"
            run.mkdir(parents=True)
            _write_training(run, [(4.5, 0.001, 2.0), (4.0, 0.0001, 1.5)])
            (run / "validation.csv").write_text(
                "step,tokens_processed,kind,domain,validation_loss\n"
                "1,10,fineweb_probe,fineweb,4.4\n"
                "2,20,fineweb,fineweb,3.7\n",
                encoding="utf-8",
            )
            _write_result(run)

            summary = build_report(runs, root / "report.html", max_chart_points=64)
            html = (root / "report.html").read_text(encoding="utf-8")
            payload = _payload(html)

        self.assertEqual(summary.included, ("complete-run",))
        self.assertFalse(summary.skipped)
        self.assertEqual(payload["meta"]["defaultXAxis"], "flops")
        self.assertEqual(payload["meta"]["defaultXScale"], "log")
        self.assertEqual(payload["meta"]["maxChartPoints"], 64)
        self.assertEqual(payload["meta"]["smoothingMaxSamples"], 64)
        self.assertTrue(payload["runs"][0]["selected"])
        self.assertEqual(payload["runs"][0]["riglogs"], [TRAINING_LOG_NAME])
        self.assertEqual(payload["runs"][0]["classification"], "official")
        self.assertEqual(payload["runs"][0]["flopSource"], "traced")
        train = next(
            chart for chart in payload["timeCharts"] if chart["key"] == "train_loss"
        )
        self.assertEqual(train["series"][0]["points"][-1], [2.0, 2000.0, 4.0])
        # Shell strings stay in the markup; chart titles now live in the
        # compressed payload, so they are asserted through the decoder.
        self.assertIn("equi-FLOP", html)
        self.assertIn("equi-step", html)
        self.assertIn(
            "Learning rate",
            [chart["title"] for chart in payload["timeCharts"]],
        )
        coverage = next(
            notice
            for notice in payload["notices"]
            if notice.startswith("Overall training diagnostic coverage:")
        )
        # This run recorded no diagnostics log, so the grid is entirely
        # unrecorded. Coverage now reflects the log's declared columns rather
        # than being inferred from training-column names.
        self.assertIn("not recorded: gradient L1 norm", coverage)
        self.assertIn("gradient L2 norm", coverage)
        self.assertIn("update fourth moment", coverage)
        self.assertIn("parameter fourth moment", coverage)
        self.assertNotRegex(html, r"<script[^>]+src=|<link[^>]+href=")
        self.assertNotIn(".slice(0,10)", html)
        self.assertNotIn("Math.min(...xs)", html)
        self.assertIn("filter(r=>r.selected)", html)

    def test_every_run_starts_visible_but_keeps_an_honest_label(self) -> None:
        # Visibility and classification are separate concerns: nothing is
        # hidden, but a smoke run is never displayed as an official result.
        self.assertEqual(_default_run_selection("official"), (True, "official"))
        self.assertEqual(_default_run_selection("dev"), (True, "diagnostic"))
        self.assertEqual(_default_run_selection("smoke"), (True, "smoke"))

    def test_the_flop_axis_comes_from_the_header_and_is_exactly_linear(self) -> None:
        # There is no per-sample FLOP column to disagree with itself any more:
        # the axis is step x tokens_per_step x flops_per_token, and the writer
        # refuses a non-positive value for either constant.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "runs"
            run = runs / "flop-axis"
            run.mkdir(parents=True)
            _write_training(
                run,
                [(4.5, 1e-4, 0.5), (4.0, 9e-5, 0.4)],
                tokens_per_step=10,
                flops_per_token=100.0,
            )
            _write_result(run, validation_artifact=False)
            build_report(runs, root / "report.html", max_chart_points=64)
            payload = _payload((root / "report.html").read_text(encoding="utf-8"))

        self.assertEqual(payload["runs"][0]["flopSource"], "traced")
        train = next(
            chart for chart in payload["timeCharts"] if chart["key"] == "train_loss"
        )
        points = train["series"][0]["points"]
        self.assertEqual([point[1] for point in points], [1000.0, 2000.0])

    def test_lttb_defaults_to_the_flop_coordinate(self) -> None:
        points = [
            [1, 1, -4],
            [2, 2, 3],
            [3, 3, -10],
            [4, 21, 6],
            [5, 22, -3],
            [6, 35, 4],
        ]
        by_flops = _lttb(points, 4)
        by_steps = _lttb(points, 4, x_index=0)
        self.assertEqual(by_flops, [points[0], points[1], points[4], points[5]])
        self.assertNotEqual(by_flops, by_steps)

    def test_ledger_hash_mismatch_excludes_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "runs"
            run = runs / "tampered-run"
            run.mkdir(parents=True)
            training = _write_training(run, [(4.5, 1e-4, 0.5), (4.0, 9e-5, 0.4)])
            validation = run / "validation.csv"
            validation.write_text(
                "step,tokens_processed,kind,domain,validation_loss\n"
                "2,20,fineweb,fineweb,3.7\n",
                encoding="utf-8",
            )
            _write_result(run)
            record = {
                "run_id": run.name,
                "status": "ok",
                "recipe": "reference",
                "track": "sample_efficiency",
                "profile": "official",
                "seed": 1,
                "metrics": {
                    "tokens_processed": 20,
                    "train_seconds": 1.0,
                    "validation_loss": 3.7,
                },
                "artifacts": {
                    "training_curve": {
                        "path": TRAINING_LOG_NAME,
                        "bytes": training.stat().st_size,
                        "sha256": "0" * 64,
                    },
                    "validation_curve": {
                        "path": "validation.csv",
                        "bytes": validation.stat().st_size,
                        "sha256": _sha(validation),
                    },
                },
            }
            (runs / "records.jsonl").write_text(
                json.dumps(record) + "\n", encoding="utf-8"
            )

            summary = build_report(runs, root / "report.html")

        self.assertFalse(summary.included)
        self.assertIn("SHA-256", summary.skipped["tampered-run"])

    def test_malformed_unledgered_timing_is_skipped_without_aborting_report(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "runs"
            good = runs / "good-run"
            bad = runs / "bad-run"
            for run in (good, bad):
                run.mkdir(parents=True)
                _write_training(run, [(4.5, 1e-4, 0.5), (4.0, 9e-5, 0.4)])
                _write_result(run, validation_artifact=False)
            bad_result = json.loads((bad / "result.json").read_text(encoding="utf-8"))
            bad_result["metrics"]["train_seconds"] = float("nan")
            (bad / "result.json").write_text(json.dumps(bad_result), encoding="utf-8")

            summary = build_report(runs, root / "report.html")

        self.assertEqual(summary.included, ("good-run",))
        self.assertIn("train_seconds is not finite", summary.skipped["bad-run"])

    def test_invalid_qualified_value_is_sanitized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "runs"
            run = runs / "odd-qualified"
            run.mkdir(parents=True)
            _write_training(run, [(4.5, 1e-4, 0.5), (4.0, 9e-5, 0.4)])
            _write_result(run, validation_artifact=False)
            record = _record_for_run(run, validation=False)
            record["qualified"] = float("nan")
            (runs / "records.jsonl").write_text(
                json.dumps(record) + "\n", encoding="utf-8"
            )

            summary = build_report(runs, root / "report.html")
            payload = _payload((root / "report.html").read_text(encoding="utf-8"))

        self.assertEqual(summary.included, (run.name,))
        self.assertIsNone(payload["runs"][0]["qualified"])
        self.assertTrue(any("qualified flag was ignored" in n for n in summary.notices))

    def test_duplicate_ledger_run_id_excludes_ambiguous_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "runs"
            run = runs / "duplicate-run"
            run.mkdir(parents=True)
            _write_training(run, [(4.5, 1e-4, 0.5), (4.0, 9e-5, 0.4)])
            _write_result(run, validation_artifact=False)
            record = _record_for_run(run, validation=False)
            (runs / "records.jsonl").write_text(
                json.dumps(record) + "\n" + json.dumps(record) + "\n",
                encoding="utf-8",
            )

            summary = build_report(runs, root / "report.html")

        self.assertFalse(summary.included)
        self.assertIn("duplicate entries", summary.skipped[run.name])

    def test_checkpoint_stats_group_parameter_arrays_by_logical_layer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.npz"
            np.savez(
                checkpoint,
                **{
                    "params/blocks/0/a": np.array([1.0, -1.0]),
                    "params/blocks/0/b": np.array([2.0]),
                    "params/blocks/1/a": np.array([3.0, 4.0]),
                    "grads/layers/0/a": np.array([0.5, -0.5]),
                    "updates/h/1/a": np.array([0.25, -0.25]),
                    "metadata.json": np.array([1], dtype=np.uint8),
                },
            )
            stats = _checkpoint_layer_stats(checkpoint)

        self.assertEqual(set(stats), {"param", "grad", "update"})
        self.assertEqual([row["layer"] for row in stats["param"]], [0.0, 1.0])
        self.assertAlmostEqual(stats["param"][0]["l1_norm"], 4.0)
        self.assertAlmostEqual(stats["param"][1]["l2_norm"], 5.0)
        self.assertAlmostEqual(stats["param"][1]["mean"], 3.5)
        self.assertAlmostEqual(stats["grad"][0]["l1_norm"], 1.0)
        self.assertAlmostEqual(stats["update"][0]["l2_norm"], 2**-2 * 2**0.5)

    def test_diagnostic_columns_are_named_by_the_registry(self) -> None:
        # Column identity used to be guessed from ad-hoc CSV header text. It is
        # an explicit registry id now, so the guessing has no inputs left.
        for family in ("param", "grad", "update"):
            for statistic in ("l1_norm", "l2_norm", "std", "third_moment"):
                with self.subTest(metric=f"{family}.{statistic}"):
                    entry = logpack.column(f"{family}.{statistic}", "block", 0)
                    self.assertEqual(entry.metric.family, family)
                    self.assertEqual(entry.metric.stat, statistic)
                    self.assertTrue(_diagnostic_metric(f"{family}_{statistic}"))

    def test_embedded_data_escapes_html_and_cannot_close_script(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "runs"
            run = runs / "script-safe"
            run.mkdir(parents=True)
            _write_training(run, [(4.5, 1e-4, 0.5), (4.0, 9e-5, 0.4)])
            _write_result(run, validation_artifact=False)
            record = {
                "run_id": run.name,
                "status": "ok",
                "recipe": "</script><script>alert(1)</script>",
                "track": "sample_efficiency",
                "profile": "official",
                "seed": 1,
                "metrics": {
                    "tokens_processed": 20,
                    "train_seconds": 1.0,
                    "validation_loss": 3.7,
                },
                "artifacts": {
                    "training_curve": {
                        "path": TRAINING_LOG_NAME,
                        "bytes": (run / TRAINING_LOG_NAME).stat().st_size,
                        "sha256": _sha(run / TRAINING_LOG_NAME),
                    }
                },
            }
            (runs / "records.jsonl").write_text(
                json.dumps(record) + "\n", encoding="utf-8"
            )
            build_report(runs, root / "report.html")
            html = (root / "report.html").read_text(encoding="utf-8")
            payload = _payload(html)

        self.assertEqual(payload["runs"][0]["recipe"], record["recipe"])
        self.assertEqual(html.count("<script>"), 1)
        self.assertNotIn(record["recipe"], html)


def _write_result(
    run: Path,
    *,
    validation_artifact: bool = True,
    diagnostics_artifact: bool = False,
    track: str = "sample_efficiency",
    profile: str = "official",
    tokens: int = 20,
    validation_loss: float = 3.7,
) -> None:
    artifacts = {"training_curve": TRAINING_LOG_NAME}
    if validation_artifact:
        artifacts["validation_curve"] = "validation.csv"
    if diagnostics_artifact:
        artifacts["diagnostics"] = DIAGNOSTICS_LOG_NAME
    result = {
        "schema_version": 1,
        "status": "ok",
        "track": track,
        "profile": profile,
        "seed": 1,
        "checkpoint": "checkpoint.npz",
        "artifacts": artifacts,
        "metrics": {
            "tokens_processed": tokens,
            "train_seconds": 1.0,
            "train_loss": 4.0,
            "validation_loss": validation_loss,
            "flops_per_token": 100,
        },
    }
    (run / "result.json").write_text(json.dumps(result), encoding="utf-8")


def _write_training(
    run: Path,
    rows: Sequence[Sequence[float]],
    *,
    tokens_per_step: int = 10,
    flops_per_token: float = 100.0,
) -> Path:
    """Write a packed training log; rows are ``(loss, learning_rate, grad_norm)``."""

    path = run / TRAINING_LOG_NAME
    with logpack.LogWriter(
        path,
        (
            logpack.column("train_loss"),
            logpack.column("learning_rate"),
            logpack.column("grad_norm"),
        ),
        tokens_per_step=tokens_per_step,
        flops_per_token=flops_per_token,
    ) as writer:
        for step, row in enumerate(rows, 1):
            writer.append(step, row)
    return path


def _write_diagnostics(
    path: Path, *, tokens_per_step: int = 10, flops_per_token: float = 100.0
) -> None:
    families = ("param", "grad", "update")
    statistics = (
        "l1_norm",
        "l2_norm",
        "mean",
        "std",
        "third_moment",
        "fourth_moment",
    )
    scopes = (
        ("overall", None, 8),
        ("embeddings", None, 2),
        ("unembedding", None, 2),
        ("block", 0, 3),
        ("final_norm", None, 1),
    )
    columns = [
        logpack.column(f"{family}.{statistic}", scope, layer, element_count=count)
        for scope, layer, count in scopes
        for family in families
        for statistic in statistics
    ]
    with logpack.LogWriter(
        path, columns, tokens_per_step=tokens_per_step, flops_per_token=flops_per_token
    ) as writer:
        for step in (1, 2):
            values = [
                step + scope_index / 10 + family_index / 100 + stat_index / 1000
                for scope_index, _ in enumerate(scopes)
                for family_index, _ in enumerate(families)
                for stat_index, _ in enumerate(statistics)
            ]
            writer.append(step, values)


def _record_for_run(run: Path, *, validation: bool) -> dict[str, object]:
    record: dict[str, object] = {
        "run_id": run.name,
        "status": "ok",
        "recipe": "reference",
        "track": "sample_efficiency",
        "profile": "official",
        "seed": 1,
        "qualified": True,
        "metrics": {
            "tokens_processed": 20,
            "train_seconds": 1.0,
            "validation_loss": 3.7,
        },
        "artifacts": {
            "training_curve": {
                "path": TRAINING_LOG_NAME,
                "bytes": (run / TRAINING_LOG_NAME).stat().st_size,
                "sha256": _sha(run / TRAINING_LOG_NAME),
            }
        },
    }
    if validation:
        artifacts = record["artifacts"]
        assert isinstance(artifacts, dict)
        artifacts["validation_curve"] = {
            "path": "validation.csv",
            "bytes": (run / "validation.csv").stat().st_size,
            "sha256": _sha(run / "validation.csv"),
        }
    return record


class ChartPointBudgetTests(unittest.TestCase):
    """How many samples a single-file report carries, and how to get them all.

    These files are a portable overview and thin every series to one budget,
    so nothing in a file sits at a different fidelity than anything beside it.
    Thinning cannot be undone by zooming and a downsampled curve looks exactly
    like a real one, so the lossless originals live in the dataset repository
    rather than being reconstructed from these.
    """

    def _report(self, steps: int, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "runs"
            (runs / "long").mkdir(parents=True)
            rows = [(10.0 - index * 0.001, 1e-4, 0.5) for index in range(steps)]
            # One lone spike, the kind a shape-preserving downsample drops.
            rows[steps // 2] = (99.0, 1e-4, 0.5)
            # The report cross-checks the final loss against result.json, and
            # the fixture declares 4.0.
            rows[-1] = (4.0, 1e-4, 0.5)
            _write_training(runs / "long", rows)
            # tokens_per_step is 10 in the fixture; the report cross-checks the
            # declared token count against the curve, so it has to follow.
            _write_result(runs / "long", validation_artifact=False, tokens=steps * 10)
            build_report(runs, root / "report.html", **kwargs)
            return _payload((root / "report.html").read_text(encoding="utf-8"))

    def test_the_default_thins_to_one_budget(self) -> None:
        # These single-file reports are a portable overview, so they carry a
        # bounded number of points. The lossless originals are the dataset.
        payload = self._report(4000)
        chart = next(c for c in payload["timeCharts"] if c["key"] == "train_loss")
        self.assertEqual(len(chart["series"][0]["points"]), 1400)

    def test_zero_embeds_every_sample(self) -> None:
        steps = 4000
        payload = self._report(steps, max_chart_points=0)
        chart = next(c for c in payload["timeCharts"] if c["key"] == "train_loss")
        self.assertEqual(len(chart["series"][0]["points"]), steps)
        # Zero means unthinned to the report builder, not a zero-sample upper
        # bound for the browser's smoothing control.
        self.assertEqual(payload["meta"]["maxChartPoints"], 0)
        self.assertEqual(payload["meta"]["smoothingMaxSamples"], 1400)

    def test_a_spike_survives_at_full_resolution(self) -> None:
        payload = self._report(4000, max_chart_points=0)
        chart = next(c for c in payload["timeCharts"] if c["key"] == "train_loss")
        values = [point[2] for point in chart["series"][0]["points"]]
        self.assertEqual(max(values), 99.0)


class StudyExportTests(unittest.TestCase):
    """Laying runs out the way the dataset repository expects them."""

    def _runs(self, root: Path, *, rich: bool = True) -> Path:
        runs = root / "runs"
        for index, seed in enumerate((1337, 1338)):
            run = runs / f"20260818T00000{index}.000000Z-reference-x-{seed:08x}"
            run.mkdir(parents=True)
            _write_training(run, [(4.5, 1e-4, 0.5), (4.0, 9e-5, 0.4)])
            _write_result(run, validation_artifact=False)
            payload = json.loads((run / "result.json").read_text(encoding="utf-8"))
            payload["seed"] = seed
            if rich:
                # _write_training writes 2 steps at 10 tokens each; the report
                # cross-checks the declared token count against the curve, so
                # these have to agree or the runs are skipped and the snapshot
                # comes back empty.
                payload["contract"] = {"sequence_length": 10}
                payload["metrics"].update(
                    {
                        "model_tier": "60m",
                        "base_learning_rate": 0.00390625,
                        "training_steps": 2,
                        "tokens_per_parameter": 5.0,
                    }
                )
            (run / "result.json").write_text(json.dumps(payload), encoding="utf-8")
        return runs

    def test_it_names_folders_by_what_varies(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = export_study(self._runs(root), root / "out", "demo")
            folders = sorted(p.name for p in summary["path"].iterdir() if p.is_dir())
        self.assertEqual(summary["runs"], 2)
        # A timestamp says when a run happened; this says what it was.
        self.assertEqual(
            folders, ["60m-5tpp-bs1-lr2e-8-s1337", "60m-5tpp-bs1-lr2e-8-s1338"]
        )

    def test_a_routed_run_does_not_overwrite_the_dense_run_beside_it(self) -> None:
        """Routing is not one of the coordinates the name is built from.

        A study holding both families at the same tier, batch, rate, and seed
        would otherwise name them identically and export one on top of the
        other, losing half the study with no error.
        """

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = self._runs(root)
            # Make the second run routed and give it the first one's seed, so
            # every other coordinate in the name matches.
            second = sorted(p for p in runs.iterdir() if p.is_dir())[1]
            payload = json.loads((second / "result.json").read_text(encoding="utf-8"))
            payload["seed"] = 1337
            payload["metrics"]["experts"] = 8
            (second / "result.json").write_text(json.dumps(payload), encoding="utf-8")

            summary = export_study(runs, root / "out", "demo")
            folders = sorted(p.name for p in summary["path"].iterdir() if p.is_dir())

        self.assertEqual(summary["runs"], 2)
        self.assertEqual(
            folders, ["60m-5tpp-bs1-lr2e-8-s1337", "60m-moe-5tpp-bs1-lr2e-8-s1337"]
        )

    def test_a_duration_run_does_not_overwrite_the_reference_run_beside_it(
        self,
    ) -> None:
        """Parameterization is part of a mixed study's run identity."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = self._runs(root)
            second = sorted(p for p in runs.iterdir() if p.is_dir())[1]
            payload = json.loads((second / "result.json").read_text(encoding="utf-8"))
            payload["seed"] = 1337
            payload["contract"]["model"] = {
                "parameterization": "completedp_duration_v1"
            }
            (second / "result.json").write_text(json.dumps(payload), encoding="utf-8")

            summary = export_study(runs, root / "out", "demo")
            folders = sorted(p.name for p in summary["path"].iterdir() if p.is_dir())

        self.assertEqual(summary["runs"], 2)
        self.assertEqual(
            folders,
            [
                "60m-5tpp-bs1-lr2e-8-s1337",
                "60m-duration-5tpp-bs1-lr2e-8-s1337",
            ],
        )

    def test_it_falls_back_to_the_run_id_when_it_cannot_name_a_run(self) -> None:
        """A run missing its tier or learning rate is still worth exporting.

        Under a name that is at least unique -- dropping it would lose data
        over a cosmetic problem.
        """

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = export_study(self._runs(root, rich=False), root / "out", "demo")
            folders = sorted(p.name for p in summary["path"].iterdir() if p.is_dir())
        self.assertEqual(summary["runs"], 2)
        for name in folders:
            self.assertTrue(name.startswith("20260818T"))

    def test_the_readme_is_written_empty(self) -> None:
        """Nothing here knows why a sweep was run, so nothing should guess.

        A generated description would read as though someone had checked it,
        which is worse than a blank one that visibly needs filling in.
        """

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = export_study(self._runs(root), root / "out", "demo")
            self.assertEqual(summary["readme"].name, "README.md")
            self.assertEqual(summary["readme"].read_text(encoding="utf-8"), "")

    def test_it_carries_the_ledger_and_both_browser_views(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = export_study(self._runs(root), root / "out", "demo")
            path = summary["path"]
            self.assertTrue((path / "records.jsonl").is_file())
            full = json.loads(gzip.decompress((path / "full.json.gz").read_bytes()))
            snapshot = json.loads(
                gzip.decompress((path / "snapshot.json.gz").read_bytes())
            )
            self.assertEqual(
                summary["full_bytes"], (path / "full.json.gz").stat().st_size
            )
        # The full payload is the unthinned view loaded explicitly by the study
        # browser. Raw .riglog files remain the archive of record.
        self.assertEqual(full["meta"]["maxChartPoints"], 0)
        self.assertEqual(full["meta"]["smoothingMaxSamples"], 1400)
        self.assertTrue(full["timeCharts"])
        # The snapshot is the overview the browser loads first, before anything
        # larger is fetched, so it carries curves and nothing else.
        self.assertEqual(snapshot["meta"]["maxChartPoints"], 200)
        self.assertEqual(snapshot["meta"]["smoothingMaxSamples"], 200)
        self.assertEqual(snapshot["diagnosticCharts"], [])
        self.assertEqual(snapshot["layerCharts"], [])
        self.assertTrue(snapshot["timeCharts"])


class StudyBrowserTests(unittest.TestCase):
    def test_raw_logs_are_always_linked_and_full_reports_are_optional(self) -> None:
        studies = [
            {"name": "with-report", "title": "With report", "runs": 2, "full": 123},
            {"name": "raw-only", "title": "Raw only", "runs": 3},
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "study-browser.html"
            build_study_browser(output, repo="owner/logs", studies=studies)
            html = output.read_text(encoding="utf-8")

        payload = _payload(html)
        self.assertEqual(payload["remote"]["studies"], studies)
        self.assertEqual(payload["meta"]["smoothingMaxSamples"], 1400)
        self.assertIn("function smoothingMaximum()", html)
        # Already-published full payloads predate smoothingMaxSamples. Their
        # maxChartPoints=0 sentinel must still resolve to a positive limit.
        self.assertIn(
            "return Number.isFinite(budget)&&budget>0?Math.round(budget):1400}",
            html,
        )
        self.assertIn("'/tree/main/'", html)
        self.assertIn("Browse raw logs", html)
        self.assertIn("Load full report (", html)
        self.assertIn("const FULL_CACHE='rig-study-full-v1'", html)
        self.assertIn("cache.put(key,response.clone())", html)
        self.assertIn("cachedFullPayload(base+'full.json.gz',study.full", html)
        self.assertIn("full payload from browser cache", html)
        self.assertIn("study.full?", html)
        self.assertNotIn("Download full logs", html)


def _payload(html: str) -> dict[str, object]:
    """Decode the embedded payload exactly as the page does."""

    match = re.search(
        r'<script type="application/gzip-base64" id="report-data">(.*?)</script>',
        html,
        flags=re.DOTALL,
    )
    assert match is not None
    raw = match.group(1).strip()
    # base64 cannot contain the characters that would terminate a script element.
    assert not any(character in raw for character in "<>&")
    return json.loads(gzip.decompress(base64.b64decode(raw)).decode("utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()


class LayerSnapshotTests(unittest.TestCase):
    def test_subsample_keeps_the_ends_and_bounds_the_count(self) -> None:
        steps = list(range(1, 1000))
        picked = [steps[index] for index in _subsample_indices(len(steps), 80)]
        self.assertLessEqual(len(picked), 81)
        self.assertEqual(picked[0], 1)
        self.assertEqual(picked[-1], 999)
        self.assertEqual(picked, sorted(set(picked)))

    def test_short_lists_are_returned_whole(self) -> None:
        steps = [1, 5, 9]
        self.assertEqual(
            [steps[index] for index in _subsample_indices(len(steps), 80)], steps
        )

    def test_layer_series_expose_aligned_step_frames(self) -> None:
        # The dragger needs one value row per retained step, aligned to a fixed
        # scope layout, and the final step must always be present.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "runs"
            run = runs / "layered"
            run.mkdir(parents=True)
            _write_training(run, [(4.5, 1e-4, 0.5), (4.0, 9e-5, 0.4)])
            _write_diagnostics(run / DIAGNOSTICS_LOG_NAME)
            _write_result(
                run, validation_artifact=False, tokens=20, validation_loss=3.0
            )
            build_report(runs, root / "report.html")
            payload = _payload((root / "report.html").read_text(encoding="utf-8"))

        charts = [c for c in payload["layerCharts"] if c["key"] == "layer_grad_l2_norm"]
        self.assertTrue(charts)
        series = charts[0]["series"][0]
        self.assertEqual(sorted(series), ["run", "scopes", "steps", "values"])
        self.assertEqual(series["steps"][-1], 2, "final step must be retained")
        self.assertEqual(len(series["values"]), len(series["steps"]))
        for row in series["values"]:
            self.assertEqual(len(row), len(series["scopes"]))


class PayloadPackingTests(unittest.TestCase):
    def test_payload_is_compressed_and_smaller_than_plain_json(self) -> None:
        # The payload is ~99% of the file, so packing it is what bounds size.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "runs"
            run = runs / "packed"
            run.mkdir(parents=True)
            _write_training(run, [(4.5 - i / 1000, 1e-4, 0.5) for i in range(1, 400)])
            _write_result(
                run, validation_artifact=False, tokens=3990, validation_loss=3.0
            )
            build_report(runs, root / "report.html")
            html = (root / "report.html").read_text(encoding="utf-8")

        payload = _payload(html)
        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        match = re.search(
            r'<script type="application/gzip-base64" id="report-data">(.*?)</script>',
            html,
            flags=re.DOTALL,
        )
        assert match is not None
        # base64-of-gzip must still beat the plain JSON it replaced.
        self.assertLess(len(match.group(1)), len(raw))
        self.assertIn("meta", payload)
        self.assertIn("runs", payload)

    def test_layer_snapshots_zero_keeps_every_recorded_step(self) -> None:
        self.assertEqual(_subsample_indices(5, 0), [0, 1, 2, 3, 4])
        self.assertEqual(len(_subsample_indices(500, 0)), 500)

    def test_layer_snapshots_must_be_zero_or_at_least_two(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "runs"
            runs.mkdir()
            with self.assertRaises(ReportError):
                build_report(runs, root / "r.html", layer_snapshots=1)
            with self.assertRaises(ReportError):
                build_report(runs, root / "r.html", layer_snapshots=-1)

    def test_values_are_trimmed_to_float32_precision(self) -> None:
        # Full double repr spends ~17 chars to encode ~7 meaningful ones.
        self.assertEqual(_compact(169.65899658203125), 169.659)
        self.assertIsNone(_compact(float("nan")))
        self.assertIsNone(_compact(None))


class ClientSourceGuardTests(unittest.TestCase):
    """Source-level guards for client code no test here can execute."""

    def _script(self) -> str:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "runs"
            runs.mkdir()
            build_report(runs, root / "report.html")
            html = (root / "report.html").read_text(encoding="utf-8")
        scripts = re.findall(
            r"<script(?![^>]*id=)[^>]*>(.*?)</script>", html, re.DOTALL
        )
        return max(scripts, key=len)

    def test_draw_never_reaches_for_series_points(self) -> None:
        # Layer series carry scopes/steps/values and have no `points`. draw()
        # reading s.points made every layer series look "smoothed" and then
        # threw on undefined, so the charts rendered blank.
        script = self._script()
        start = script.index("function draw(item){")
        end = script.index("function axisToStep(", start)
        self.assertNotIn("s.points", script[start:end])
        # Reads go through seriesPoints, whatever the local is called. The
        # binding was renamed when draw started reducing points for display,
        # and pinning the old name would have failed for a rename rather than
        # for the bug this guards.
        self.assertIn("seriesPoints(item,s)", script[start:end])

    def test_the_draw_reduction_keeps_both_extremes(self) -> None:
        """Bucketing must keep each bucket's min and max, not one representative.

        The payload now holds every sample, so drawing has to reduce. Which
        reduction it uses decides whether a lone spike is visible: keeping one
        point per bucket preserves the curve's shape and can drop the spike
        entirely, leaving a clean line that is wrong. Keeping both extremes
        cannot.
        """

        script = self._script()
        start = script.index("function envelope(")
        end = script.index("function reduceForDraw(", start)
        body = script[start:end]
        self.assertIn("out.push(points[mn])", body)
        self.assertIn("out.push(points[mx])", body)

    def test_draw_reduces_against_the_visible_span(self) -> None:
        # Reducing the whole series and then zooming would show the same
        # coarse points however far in you went.
        script = self._script()
        start = script.index("function reduceForDraw(")
        self.assertIn("visibleSlice(item,points)", script[start : start + 200])
        draw_start = script.index("function draw(item){")
        draw_end = script.index("function axisToStep(", draw_start)
        self.assertIn("reduceForDraw(item,", script[draw_start:draw_end])

    def test_layer_frames_are_cached_for_stable_identity(self) -> None:
        # draw() decides "smoothed" by array identity, so layerFrame must not
        # return a fresh array per call.
        script = self._script()
        start = script.index("function layerFrame(")
        end = script.index("function seriesPoints(", start)
        self.assertIn("frameCache.get(s)", script[start:end])
        self.assertIn("frameCache.set(s,", script[start:end])

    def _body_payload(self) -> dict:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "runs"
            (runs / "a").mkdir(parents=True)
            _write_training(runs / "a", [(4.5, 1e-4, 0.5), (4.0, 9e-5, 0.4)])
            _write_result(runs / "a", validation_artifact=False)
            build_report(runs, root / "report.html")
            return _payload((root / "report.html").read_text(encoding="utf-8"))

    def _body(self) -> str:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "runs"
            runs.mkdir()
            build_report(runs, root / "report.html")
            html = (root / "report.html").read_text(encoding="utf-8")
        return html[: html.index('<script type="application/gzip-base64"')]

    def test_notices_sit_below_the_charts_and_the_summary_above(self) -> None:
        # Results lead; diagnostics get pushed out of the primary reading path.
        body = self._body()
        self.assertLess(body.index('id="summary-fold"'), body.index('id="time-charts"'))
        self.assertGreater(
            body.index('id="notices-fold"'), body.index('id="layer-charts"')
        )

    def test_summary_starts_collapsed_but_opens_on_a_load_failure(self) -> None:
        """The charts are what the page is for; the table is reference.

        The fold must still spring open when the payload fails to load, since
        the error handler writes into #summary and a collapsed fold would hide
        the only report of why the page is empty. That guarantee comes from the
        handler forcing it open, not from the markup, which is what lets the
        default be collapsed at all.
        """

        body = self._body()
        self.assertIn('id="summary-fold"', body)
        self.assertNotIn('id="summary-fold" open', body)
        script = self._script()
        start = script.index("}).catch(error=>{")
        self.assertIn("fold.open=true", script[start:])

    def test_a_picker_launched_view_offers_a_way_back(self) -> None:
        """Opening a study from the browser must not be a one-way door.

        The picker removes itself once a study is chosen, so without this the
        only way back to the study list is the browser's own back button --
        which does nothing, because choosing a study never changed the URL.
        """

        script = self._script()
        self.assertIn("cameFromPicker=true", script)
        start = script.index("function init(){")
        end = (
            script.index("function buildRuns(", start)
            if "function buildRuns(" in script[start:]
            else len(script)
        )
        body = script[start:end]
        self.assertIn("back-to-studies", body)
        self.assertIn("location.reload()", body)

    def test_a_panel_with_nothing_to_draw_is_hidden(self) -> None:
        """Metrics come and go, so most reports carry charts that do not apply.

        A routed run records routing series a dense one never will. Leaving an
        empty frame with a caption for each of them buries the charts that do
        have data, so the panel is hidden instead -- and hidden rather than
        removed, so it returns when the selection changes.
        """

        script = self._script()
        start = script.index("function draw(item){")
        end = script.index("function axisToStep(", start)
        body = script[start:end]
        self.assertIn("item.article.hidden=true", body)
        self.assertIn("item.article.hidden=false", body)

    def test_only_recorded_series_reach_the_payload(self) -> None:
        """A declared chart nobody recorded must not ship as an empty chart."""

        from rig.report import _ROUTER_CHARTS

        self.assertTrue(_ROUTER_CHARTS)
        payload = self._body_payload()
        keys = {chart["key"] for chart in payload["timeCharts"]}
        # The fixture runs are dense, so no routing series exists for them.
        for spec in _ROUTER_CHARTS:
            self.assertNotIn(spec.metric, keys)

    def test_runs_carry_the_topology_that_produced_them(self) -> None:
        """Chip kind and device count belong beside the loss they produced.

        The data stream is invariant under process count -- the same seed draws
        the same global batches on any -- but gradients reduce across a
        different number of devices and each chip holds a different share of
        the batch. The same configuration and seed has landed 0.004-0.023 nats
        apart across two slices for that reason alone, so a number without its
        topology is not reproducible.
        """

        script = self._script()
        self.assertIn("function hw(r){", script)
        start = script.index("function buildRuns(){")
        self.assertIn("${hw(r)}", script[start : start + 1200])

    def test_hardware_is_absent_rather_than_invented(self) -> None:
        # Older artifacts predate the system block; they must render, not throw.
        from rig.report import _hardware

        self.assertEqual(_hardware({}), {})
        self.assertEqual(_hardware({"system": "not-an-object"}), {})
        self.assertEqual(
            _hardware(
                {
                    "system": {
                        "device_kinds": ["TPU v4"],
                        "process_count": 4,
                        "device_count": 16,
                    }
                }
            ),
            {"chip": "TPU v4", "processes": 4, "devices": 16},
        )
        self.assertEqual(_hardware({"system": {}})["chip"], None)

    def test_notice_fold_stays_hidden_when_there_is_nothing_to_say(self) -> None:
        body = self._body()
        self.assertIn('<details class="fold" id="notices-fold" hidden>', body)
        script = self._script()
        self.assertIn("$('notices-fold').hidden=!D.notices.length", script)

    def test_export_slot_cannot_collide_with_the_build_time_placeholder(self) -> None:
        # _render_html does a global replace of the build placeholder. If the
        # client's export slot used that same literal, the whole base64 payload
        # would be inlined into the script at build time.
        script = self._script()
        self.assertIn("PAYLOAD_SLOT='@@RIG_PAYLOAD@@'", script)
        self.assertNotIn("__REPORT_DATA__", script)

    def test_export_controls_are_wired(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "runs"
            runs.mkdir()
            build_report(runs, root / "report.html")
            html = (root / "report.html").read_text(encoding="utf-8")
        self.assertIn('id="export-runs"', html)
        self.assertIn('id="export-riglogs"', html)
        self.assertIn('id="export-status"', html)
        self.assertIn("window.showSaveFilePicker", html)
        self.assertEqual(html.count("await chooseSaveFile("), 2)
        # Exactly one live slot: the constant. The payload itself is base64,
        # whose alphabet cannot produce it.
        self.assertEqual(html.count("@@RIG_PAYLOAD@@"), 1)

    def test_export_refuses_an_empty_selection(self) -> None:
        script = self._script()
        start = script.index("async function exportSelection(){")
        end = script.index("const tarEncoder", start)
        self.assertIn("if(!visible.size)", script[start:end])

    def test_riglog_export_preserves_the_original_container(self) -> None:
        """Raw export fetches source files; it does not rebuild them from plots.

        A report may be thinned and its charts do not carry the packed column
        descriptors. Reconstructing from that payload would produce a plausible
        but incomplete file, so selected logs are archived byte-for-byte.
        """

        script = self._script()
        start = script.index("async function fetchRiglogs(runs){")
        end = script.index("loadPayload().then(", start)
        body = script[start:end]
        self.assertIn("response.blob()", body)
        self.assertIn("expected=[82,73,71,76,79,71,0,1]", body)
        self.assertIn("tarArchive(entries)", body)
        self.assertIn("new CompressionStream('gzip')", body)
        self.assertNotIn("selectedPayload()", body)

    def test_study_picker_attaches_the_raw_log_location(self) -> None:
        script = self._script()
        self.assertIn(
            "rawSource:{repo:remote.repo,study:study.name}",
            script,
        )
        self.assertIn("$('export-riglogs').hidden=!D.rawSource", script)

    def test_export_filters_every_chart_family_to_the_selection(self) -> None:
        # A subset export must not smuggle unselected runs' series along.
        script = self._script()
        start = script.index("function selectedPayload(){")
        end = script.index("async function packPayload(", start)
        body = script[start:end]
        self.assertIn("c.series.filter(s=>keep.has(s.run))", body)
        for field in ("timeCharts", "diagnosticCharts", "layerCharts"):
            self.assertIn(f"{field}:pick(D.{field})", body)

    def test_shell_is_captured_before_init_mutates_the_dom(self) -> None:
        # init() rewrites the run list and chart containers, so a shell taken
        # afterwards would bake rendered state into every export.
        script = self._script()
        start = script.index("loadPayload().then(")
        tail = script[start:]
        self.assertLess(tail.index("captureShell()"), tail.index("init()"))

    def test_layer_axis_bounds_track_the_selected_frame_not_the_full_history(
        self,
    ) -> None:
        # dataBounds used to flatten every retained step to find y-bounds, so
        # one early spike (grad_clip is off) set the axis ceiling for every
        # frame forever. Bounds must come from the same frame draw() plots.
        script = self._script()
        start = script.index("function dataBounds(item){")
        end = script.index("function bounds(item){", start)
        self.assertNotIn("s.values.map(row=>", script[start:end])
        self.assertIn("const pts=seriesPoints(item,s);", script[start:end])


class SelectFilterTests(unittest.TestCase):
    """One line of research per report, without disturbing runs in flight."""

    def _three_runs(self, root: Path) -> Path:
        runs = root / "runs"
        for name in ("alpha-lr2e-8", "alpha-lr2e-9", "beta-lr2e-8"):
            run = runs / name
            run.mkdir(parents=True)
            _write_training(run, [(4.5, 1e-4, 0.5), (4.0, 9e-5, 0.4)])
            _write_result(run, validation_artifact=False)
        return runs

    def test_a_regex_selects_a_family(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = self._three_runs(root)
            summary = build_report(runs, root / "r.html", select="^alpha-")
        self.assertEqual(summary.included, ("alpha-lr2e-8", "alpha-lr2e-9"))
        # Non-matching runs are omitted, not reported as skipped: they were
        # never candidates, and listing them would bury real skip reasons.
        self.assertEqual(summary.skipped, {})

    def test_no_selector_keeps_every_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = self._three_runs(root)
            summary = build_report(runs, root / "r.html")
        self.assertEqual(len(summary.included), 3)

    def test_an_invalid_expression_is_refused_by_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = self._three_runs(root)
            with self.assertRaisesRegex(ReportError, "invalid --select"):
                build_report(runs, root / "r.html", select="alpha(")

    def test_the_expression_searches_rather_than_anchors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = self._three_runs(root)
            summary = build_report(runs, root / "r.html", select="lr2e-8")
        self.assertEqual(summary.included, ("alpha-lr2e-8", "beta-lr2e-8"))
