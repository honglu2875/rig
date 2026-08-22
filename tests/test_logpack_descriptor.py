"""Hold the layout descriptor to the standard a foreign reader depends on.

The report embeds packed bytes and parses them in JavaScript. That parser and
the Python writer must agree about every offset and width forever, and a
disagreement is silent: nothing raises, the page simply renders one column's
numbers under another column's name.

The defence is that neither side states the layout independently. The writer
derives it from its dtypes, ``layout_descriptor`` reads it back out of those
same dtypes, and the reader is driven by the descriptor. These tests check the
two properties that makes rest on:

* the descriptor still matches the dtypes it claims to describe, and
* the descriptor is *sufficient* -- a parser that knows nothing about
  ``logpack`` beyond the descriptor can read a real file correctly.

Sufficiency is the one that matters. If something the reader needs is missing
from the descriptor, the JavaScript has to hardcode it, and hardcoding is
exactly the drift this is meant to prevent.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from rig import logpack


def _parse_with_descriptor_only(
    raw: bytes, descriptor: dict
) -> tuple[list, np.ndarray, dict]:
    """Read a packed log using nothing but the descriptor.

    Deliberately written the way the JavaScript has to write it: no logpack
    constants, no numpy structured dtypes, only offsets and element types taken
    from the descriptor. If this cannot be written, neither can the reader.
    """

    scalar = {"<i4": "<i4", "<i8": "<i8", "<f4": "<f4", "<f8": "<f8"}

    magic = bytes(descriptor["magic"])
    if not raw.startswith(magic):
        raise ValueError("bad magic")
    offset = len(magic)

    header = {}
    for field in descriptor["header"]["fields"]:
        header[field["name"]] = np.frombuffer(
            raw, dtype=scalar[field["type"]], count=1, offset=offset + field["offset"]
        )[0]
    offset += descriptor["header"]["itemsize"]

    count = int(header["column_count"])
    stride = descriptor["column"]["itemsize"]
    columns = []
    for index in range(count):
        base = offset + index * stride
        entry = {}
        for field in descriptor["column"]["fields"]:
            entry[field["name"]] = np.frombuffer(
                raw, dtype=scalar[field["type"]], count=1, offset=base + field["offset"]
            )[0]
        columns.append(entry)
    offset += count * stride

    step_size = descriptor["step"]["itemsize"]
    value_size = descriptor["value"]["itemsize"]
    record = step_size + count * value_size
    total = (len(raw) - offset) // record
    steps, values = [], []
    for index in range(total):
        base = offset + index * record
        steps.append(
            int(
                np.frombuffer(
                    raw, dtype=scalar[descriptor["step"]["type"]], count=1, offset=base
                )[0]
            )
        )
        values.append(
            np.frombuffer(
                raw,
                dtype=scalar[descriptor["value"]["type"]],
                count=count,
                offset=base + step_size,
            )
        )
    return columns, np.array(values), {"steps": steps, "header": header}


class DescriptorMatchesTheDtypesTests(unittest.TestCase):
    def test_every_field_offset_comes_from_the_dtype(self) -> None:
        descriptor = logpack.layout_descriptor()
        for key, dtype in (
            ("header", logpack.HEADER_DTYPE),
            ("column", logpack.COLUMN_DTYPE),
        ):
            with self.subTest(block=key):
                self.assertEqual(descriptor[key]["itemsize"], dtype.itemsize)
                self.assertEqual(
                    [field["name"] for field in descriptor[key]["fields"]],
                    list(dtype.names),
                )
                for field in descriptor[key]["fields"]:
                    self.assertEqual(field["offset"], dtype.fields[field["name"]][1])
                    self.assertEqual(field["type"], dtype.fields[field["name"]][0].str)
        self.assertEqual(bytes(descriptor["magic"]), logpack.MAGIC)

    def test_element_types_are_explicit_and_little_endian(self) -> None:
        """A reader has no way to guess these, so they must be stated.

        Endianness in particular: the writer is explicit so a file moves
        between hosts unchanged, and a reader that assumed native order would
        be right on every machine anyone has tried and wrong on the next one.
        """

        descriptor = logpack.layout_descriptor()
        types = [descriptor["step"]["type"], descriptor["value"]["type"]]
        types += [f["type"] for f in descriptor["header"]["fields"]]
        types += [f["type"] for f in descriptor["column"]["fields"]]
        for value in types:
            with self.subTest(type=value):
                self.assertTrue(value.startswith("<"), f"{value} is not little-endian")


class DescriptorIsSufficientTests(unittest.TestCase):
    def _write(self, path: Path):
        columns = (
            logpack.column("train_loss"),
            logpack.column("learning_rate"),
            logpack.column("grad.l2_norm", "block", 3),
            logpack.column("router.load", "expert", 2, index=5),
        )
        with logpack.LogWriter(
            path, columns, tokens_per_step=4096, flops_per_token=1.5e9
        ) as writer:
            for step in range(1, 9):
                writer.append(step, (10.0 - step, 1e-3, step * 0.5, 0.125))
        return columns

    def test_a_descriptor_only_reader_agrees_with_the_real_one(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / f"t{logpack.SUFFIX}"
            self._write(path)
            raw = path.read_bytes()

            columns, values, extra = _parse_with_descriptor_only(
                raw, logpack.layout_descriptor()
            )
            reference = logpack.read_log(path)

            self.assertEqual(len(columns), len(reference.columns))
            np.testing.assert_array_equal(values, reference.values)
            np.testing.assert_array_equal(extra["steps"], reference.steps)
            self.assertEqual(int(extra["header"]["tokens_per_step"]), 4096)
            self.assertAlmostEqual(
                float(extra["header"]["flops_per_token"]), 1.5e9, places=0
            )

    def test_the_reader_recovers_every_addressing_field(self) -> None:
        """Layer and index are what place a series; losing them mislabels it."""

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / f"t{logpack.SUFFIX}"
            written = self._write(path)
            columns, _, _ = _parse_with_descriptor_only(
                path.read_bytes(), logpack.layout_descriptor()
            )
        for parsed, original in zip(columns, written, strict=True):
            with self.subTest(column=original.describe()):
                self.assertEqual(int(parsed["metric_id"]), original.metric_id)
                self.assertEqual(int(parsed["scope_id"]), original.scope_id)
                self.assertEqual(int(parsed["layer"]), original.layer)
                self.assertEqual(int(parsed["index"]), original.index)

    def test_indexed_expert_series_use_the_nested_block_address(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / f"t{logpack.SUFFIX}"
            self._write(path)
            recorded = logpack.read_log(path)

        self.assertEqual(
            recorded.columns[3].describe(), "block[2]/expert[5]/router.load"
        )
        np.testing.assert_array_equal(
            recorded.series("router.load", "expert", 2, index=5),
            np.full(8, 0.125, np.float32),
        )
        with self.assertRaisesRegex(ValueError, "requires an index"):
            recorded.series("router.load", "expert", 2)

    def test_it_reads_a_real_recorded_log(self) -> None:
        # A synthetic file can accidentally agree with a wrong reader; a file
        # a real run wrote cannot be tuned to.
        candidates = sorted(Path("logs").glob("*/*/training.riglog"))
        if not candidates:
            self.skipTest("no converted logs present")
        raw = candidates[0].read_bytes()
        columns, values, _ = _parse_with_descriptor_only(
            raw, logpack.layout_descriptor()
        )
        reference = logpack.read_log(candidates[0])
        self.assertEqual(len(columns), len(reference.columns))
        np.testing.assert_array_equal(values, reference.values)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
