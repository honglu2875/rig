from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from rig import vectorlog


METRICS = (
    "fuzzy.winner_frequency",
    "fuzzy.activation_frequency",
    "fuzzy.activation_mean",
    "fuzzy.activation_rms",
)


class VectorLogTests(unittest.TestCase):
    def test_round_trip_preserves_dense_axes_and_metric_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / f"features{vectorlog.SUFFIX}"
            first = np.arange(4 * 2 * 8, dtype=np.float32).reshape(4, 2, 8)
            second = first + 100.0
            with vectorlog.VectorLogWriter(
                path,
                METRICS,
                layer_count=2,
                feature_count=8,
                group_size=4,
                tokens_per_step=128,
                flops_per_token=256.0,
            ) as writer:
                writer.append(1, first)
                writer.append(5, second)

            log = vectorlog.read_vector_log(path)
            self.assertEqual(log.metric_names, METRICS)
            self.assertEqual(log.layer_count, 2)
            self.assertEqual(log.feature_count, 8)
            self.assertEqual(log.group_size, 4)
            np.testing.assert_array_equal(log.steps, [1, 5])
            np.testing.assert_array_equal(log.values[0], first)
            np.testing.assert_array_equal(
                log.metric("fuzzy.activation_frequency"),
                np.stack((first[1], second[1])),
            )
            np.testing.assert_array_equal(log.axis("tokens_processed"), [128, 640])
            np.testing.assert_array_equal(
                log.axis("cumulative_flops"), [32768.0, 163840.0]
            )

    def test_partial_trailing_record_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / f"features{vectorlog.SUFFIX}"
            values = np.ones((4, 1, 8), dtype=np.float32)
            writer = vectorlog.VectorLogWriter(
                path,
                METRICS,
                layer_count=1,
                feature_count=8,
                group_size=4,
                tokens_per_step=32,
                flops_per_token=64.0,
            )
            writer.append(3, values)
            writer.close()
            with path.open("ab") as handle:
                handle.write(b"partial")

            log = vectorlog.read_vector_log(path)
            np.testing.assert_array_equal(log.steps, [3])
            np.testing.assert_array_equal(log.values[0], values)

    def test_writer_rejects_wrong_tensor_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            writer = vectorlog.VectorLogWriter(
                Path(directory) / f"features{vectorlog.SUFFIX}",
                METRICS,
                layer_count=2,
                feature_count=8,
                group_size=4,
                tokens_per_step=32,
                flops_per_token=64.0,
            )
            with self.assertRaisesRegex(ValueError, "must have shape"):
                writer.append(1, np.zeros((4, 8), np.float32))

    def test_widening_schedule_keeps_dense_prefix_targets_and_exact_final(self) -> None:
        steps = np.asarray([1, *range(10, 20_001, 10), 20_007], dtype=np.int32)
        indices = vectorlog.widening_step_indices(steps)

        self.assertEqual(
            steps[indices].tolist(),
            [
                1,
                *range(10, 201, 10),
                300,
                500,
                900,
                1700,
                3300,
                6500,
                12900,
                20_007,
            ],
        )

    def test_widening_schedule_uses_next_real_capture_for_missing_target(self) -> None:
        steps = np.asarray([1, 50, 100, 150, 200, 350, 550, 950], dtype=np.int32)
        indices = vectorlog.widening_step_indices(steps)

        np.testing.assert_array_equal(
            steps[indices], [1, 50, 100, 150, 200, 350, 550, 950]
        )

    def test_subset_log_preserves_full_vectors_at_selected_steps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / f"source{vectorlog.SUFFIX}"
            destination = root / f"lossy{vectorlog.SUFFIX}"
            values = np.arange(4 * 2 * 8, dtype=np.float32).reshape(4, 2, 8)
            with vectorlog.VectorLogWriter(
                source,
                METRICS,
                layer_count=2,
                feature_count=8,
                group_size=4,
                tokens_per_step=128,
                flops_per_token=256.0,
            ) as writer:
                for step in (1, 10, 20, 30):
                    writer.append(step, values + step)

            copied = vectorlog.write_vector_log_subset(source, destination, [0, 2, 3])

            np.testing.assert_array_equal(copied.steps, [1, 20, 30])
            np.testing.assert_array_equal(copied.values[1], values + 20)
            self.assertEqual(copied.metric_names, METRICS)
            with self.assertRaises(FileExistsError):
                vectorlog.write_vector_log_subset(source, destination, [0])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
