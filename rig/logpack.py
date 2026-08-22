"""The packed log container recorded by runs and read by the report.

A run's training curve and its per-scope diagnostics are the same shape of
data: a growing list of samples, each holding one value per column, indexed by
optimizer step. This module stores that directly -- a header naming the columns
by their registry id, then fixed-width records -- instead of re-spelling the
column labels, the step, and the derivable axes on every row.

Layout
------
``MAGIC`` then, little-endian throughout::

    column_count     int32      then 4 bytes of padding
    tokens_per_step  int64      global batch x sequence length
    flops_per_token  float64    from the traced FLOP count
    columns          column_count x COLUMN_DTYPE (24 bytes each)
    records          n x (step int32, values column_count x float32)

Everything after the header is a fixed stride, which buys two things:
``np.frombuffer`` lifts the whole value block in one pass with no per-record
work, and the file is **append-only** -- a run killed mid-training keeps every
record already on disk, so there is nothing to salvage afterwards.

Why the axes are not stored
---------------------------
``tokens_processed`` and ``cumulative_flops`` were a third of the old CSV, and
both are exact functions of the step, so the two constants they need live in the
header and :meth:`Log.axis` materializes them on demand. Deriving also fixes a
precision problem: ``cumulative_flops`` reaches ~3e19, far past float32's seven
digits, so a stored column would round it. (``tokens_processed`` survives
float32 whenever the batch and sequence length are powers of two, as they are
today -- but it is a count, so it is materialized as int64 regardless.) Both
keep their registry ids; readers address them by id and cannot tell the
difference.

Forward compatibility
---------------------
Columns are identified by :mod:`rig.metrics` ids, never by position or text. A
reader looks up the columns it understands and ignores any it does not, so a
file written by a later build -- more layers, more statistics, a metric this
build has never heard of -- still opens. That is the whole reason the registry
is add-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable, Sequence

import numpy as np

from rig import metrics


# Eight bytes, not seven: the column table and the record block both start
# at a multiple of 8 only because this is, and every field inside them is
# then naturally aligned. The trailing byte is the layout version.
MAGIC = b"RIGLOG\x00\x01"
SUFFIX = ".riglog"

# metric id, scope id, layer (-1 when the scope is not layered), a second index
# whose meaning belongs to the scope, and the number of scalars the scope
# covers. Explicit little-endian so a file written on one host reads
# identically on another.
#
# ``index`` occupies what earlier writers filled with zero and called reserved.
# Giving it meaning moves no bytes and rewrites no history: a scope defines
# whether it uses the slot, and every scope that existed while the field was
# reserved does not. So a log written before this still decodes exactly as it
# did, and its zeros are never read as an index.
COLUMN_DTYPE = np.dtype(
    [
        ("metric_id", "<i4"),
        ("scope_id", "<i4"),
        ("layer", "<i4"),
        ("index", "<i4"),
        ("element_count", "<i8"),
    ]
)
# The fixed header, as a dtype rather than a struct format so its field
# offsets are introspectable. A non-Python reader is generated from these
# offsets rather than repeating them, which is the only thing that keeps such
# a reader from drifting out of step with this writer. The four padding bytes
# after ``column_count`` keep the 64-bit fields naturally aligned.
HEADER_DTYPE = np.dtype(
    {
        "names": ["column_count", "tokens_per_step", "flops_per_token"],
        "formats": ["<i4", "<i8", "<f8"],
        "offsets": [0, 8, 16],
        "itemsize": 24,
    }
)
# int32 keeps the record 4-aligned for any column count, where an int64 step
# would leave odd-column records straddling. The bound is 2^31 optimizer
# steps; the largest run this family can express is ~1.5e5.
_STEP_DTYPE = np.dtype("<i4")
_VALUE_DTYPE = np.dtype("<f4")


class LogError(Exception):
    """Raised when a log file is not readable as a log file."""


@dataclass(frozen=True)
class Column:
    """One series in a log, addressed by its permanent registry ids."""

    metric_id: int
    scope_id: int
    layer: int = -1
    element_count: int = 0
    # Second index within the scope, -1 when the scope does not use one. For
    # ``expert`` it is the expert ordinal, so a routed block is addressed as
    # (layer, expert) rather than needing one metric id per expert.
    index: int = -1

    @property
    def metric(self) -> metrics.Metric | None:
        return metrics.metric_by_id(self.metric_id)

    @property
    def scope(self) -> metrics.Scope | None:
        return metrics.scope_by_id(self.scope_id)

    def describe(self) -> str:
        """Render the column for humans, degrading gracefully on unknown ids."""

        metric = self.metric
        scope = self.scope
        name = metric.name if metric is not None else f"metric:{self.metric_id}"
        where = scope.name if scope is not None else f"scope:{self.scope_id}"
        if (
            scope is not None
            and scope.name == "expert"
            and self.layer >= 0
            and self.index >= 0
        ):
            return f"block[{self.layer}]/expert[{self.index}]/{name}"
        if self.layer >= 0:
            where = f"{where}[{self.layer}]"
        if self.index >= 0:
            where = f"{where}#{self.index}"
        return f"{where}/{name}"


def column(
    metric_name: str,
    scope_name: str = "overall",
    layer: int | None = None,
    element_count: int = 0,
    index: int | None = None,
) -> Column:
    """Build a column from registry names, failing fast on an unregistered one."""

    resolved_scope = metrics.scope(scope_name)
    if resolved_scope.layered and layer is None:
        raise ValueError(f"scope {scope_name!r} requires a layer index")
    if not resolved_scope.layered and layer is not None:
        raise ValueError(f"scope {scope_name!r} does not take a layer index")
    if resolved_scope.indexed and index is None:
        raise ValueError(f"scope {scope_name!r} requires a second index")
    if not resolved_scope.indexed and index is not None:
        raise ValueError(f"scope {scope_name!r} does not take a second index")
    return Column(
        metric_id=metrics.metric(metric_name).id,
        scope_id=resolved_scope.id,
        layer=-1 if layer is None else layer,
        element_count=element_count,
        index=-1 if index is None else index,
    )


def layout_descriptor() -> dict:
    """Describe the byte layout so a reader in another language stays in step.

    Every offset, width, and element type is read back out of the same dtype
    objects the writer uses, so there is exactly one place any of them is
    stated. A reader driven by this descriptor cannot disagree with the writer
    about where a field lives, because neither of them knows independently --
    which is the whole point, since a disagreement would not raise anything, it
    would silently render one column's numbers under another column's name.

    ``tests/test_logpack_descriptor.py`` holds a reader honest against it.
    """

    def fields(dtype):
        return [
            {
                "name": name,
                "offset": int(dtype.fields[name][1]),
                "type": dtype.fields[name][0].str,
            }
            for name in dtype.names
        ]

    return {
        "magic": list(MAGIC),
        "header": {"itemsize": HEADER_DTYPE.itemsize, "fields": fields(HEADER_DTYPE)},
        "column": {"itemsize": COLUMN_DTYPE.itemsize, "fields": fields(COLUMN_DTYPE)},
        "step": {"type": _STEP_DTYPE.str, "itemsize": _STEP_DTYPE.itemsize},
        "value": {"type": _VALUE_DTYPE.str, "itemsize": _VALUE_DTYPE.itemsize},
    }


def _record_size(column_count: int) -> int:
    return _STEP_DTYPE.itemsize + column_count * _VALUE_DTYPE.itemsize


def _header_size(column_count: int) -> int:
    return len(MAGIC) + HEADER_DTYPE.itemsize + column_count * COLUMN_DTYPE.itemsize


class LogWriter:
    """Append records to a log, creating and sealing the header on first use.

    The file is valid after every :meth:`append`, so a preempted run leaves a
    readable log rather than a partial one needing repair.
    """

    def __init__(
        self,
        path: Path,
        columns: Sequence[Column],
        *,
        tokens_per_step: int,
        flops_per_token: float,
    ) -> None:
        if not columns:
            raise ValueError("a log needs at least one column")
        if tokens_per_step <= 0:
            raise ValueError("tokens_per_step must be positive")
        if flops_per_token <= 0:
            raise ValueError("flops_per_token must be positive")
        self.path = Path(path)
        self.columns = tuple(columns)
        self._buffer = np.empty(len(self.columns), dtype=_VALUE_DTYPE)
        self._handle: BinaryIO | None = None
        self._tokens_per_step = int(tokens_per_step)
        self._flops_per_token = float(flops_per_token)
        self._last_step = 0

    def __enter__(self) -> "LogWriter":
        self._open()
        return self

    def __exit__(self, *exception: object) -> None:
        self.close()

    def _open(self) -> BinaryIO:
        if self._handle is not None:
            return self._handle
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("wb")
        table = np.zeros(len(self.columns), dtype=COLUMN_DTYPE)
        for index, entry in enumerate(self.columns):
            table[index] = (
                entry.metric_id,
                entry.scope_id,
                entry.layer,
                entry.index,
                entry.element_count,
            )
        handle.write(MAGIC)
        handle.write(
            np.array(
                (len(self.columns), self._tokens_per_step, self._flops_per_token),
                dtype=HEADER_DTYPE,
            ).tobytes()
        )
        handle.write(table.tobytes())
        handle.flush()
        self._handle = handle
        return handle

    def append(self, step: int, values: Iterable[float]) -> None:
        """Append one sample. Steps must increase, so the index stays sorted."""

        if step <= self._last_step:
            raise ValueError(
                f"log steps must increase; got {step} after {self._last_step}"
            )
        self._buffer[:] = np.fromiter(
            values, dtype=_VALUE_DTYPE, count=len(self.columns)
        )
        handle = self._open()
        handle.write(np.asarray(step, dtype=_STEP_DTYPE).tobytes())
        handle.write(self._buffer.tobytes())
        # A reader that opens the file between two appends must see whole
        # records, so the record never straddles a flush.
        handle.flush()
        self._last_step = step

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


@dataclass(frozen=True)
class Log:
    """A log read into memory: the column table plus one float32 matrix."""

    columns: tuple[Column, ...]
    steps: np.ndarray
    values: np.ndarray
    tokens_per_step: int
    flops_per_token: float

    def __len__(self) -> int:
        return int(self.steps.shape[0])

    def axis(self, name: str) -> np.ndarray:
        """Return an index axis by registry name, derived exactly from the step."""

        if name == "step":
            return self.steps
        if name == "tokens_processed":
            return self.steps * np.int64(self.tokens_per_step)
        if name == "cumulative_flops":
            return self.steps.astype(np.float64) * (
                self.tokens_per_step * self.flops_per_token
            )
        raise KeyError(f"{name!r} is not an index axis")

    def index_of(
        self,
        metric_name: str,
        scope_name: str = "overall",
        layer: int | None = None,
        index: int | None = None,
    ) -> int | None:
        """Return the column position for a series, or None when absent.

        Absent is an ordinary answer, not an error: a run with diagnostics
        disabled, or an older build, simply did not record that column.
        """

        metric = metrics.metric(metric_name)
        scope = metrics.scope(scope_name)
        if scope.indexed and index is None:
            raise ValueError(f"scope {scope_name!r} requires an index")
        if not scope.indexed and index is not None:
            raise ValueError(f"scope {scope_name!r} does not take an index")
        wanted = -1 if layer is None else layer
        wanted_index = -1 if index is None else index
        for position, entry in enumerate(self.columns):
            if (
                entry.metric_id == metric.id
                and entry.scope_id == scope.id
                and entry.layer == wanted
                and entry.index == wanted_index
            ):
                return position
        return None

    def series(
        self,
        metric_name: str,
        scope_name: str = "overall",
        layer: int | None = None,
        index: int | None = None,
    ) -> np.ndarray | None:
        """Return one column across every sample, or None when it was not recorded."""

        position = self.index_of(metric_name, scope_name, layer, index)
        return None if position is None else self.values[:, position]


def read_log(path: Path) -> Log:
    """Read a whole log. One memcpy for the values; no per-record work."""

    raw = Path(path).read_bytes()
    if len(raw) < len(MAGIC) or not raw.startswith(MAGIC):
        raise LogError(f"{path} is not a rig log (bad magic)")
    offset = len(MAGIC)
    if len(raw) < offset + HEADER_DTYPE.itemsize:
        raise LogError(f"{path} is truncated inside its header")
    header = np.frombuffer(raw, dtype=HEADER_DTYPE, count=1, offset=offset)[0]
    column_count = int(header["column_count"])
    tokens_per_step = int(header["tokens_per_step"])
    flops_per_token = float(header["flops_per_token"])
    offset += HEADER_DTYPE.itemsize
    if column_count <= 0:
        raise LogError(f"{path} declares {column_count} columns")
    table_bytes = column_count * COLUMN_DTYPE.itemsize
    if len(raw) < offset + table_bytes:
        raise LogError(f"{path} is truncated inside its column table")
    table = np.frombuffer(raw, dtype=COLUMN_DTYPE, count=column_count, offset=offset)
    offset += table_bytes

    stride = _record_size(column_count)
    available = len(raw) - offset
    # A partial trailing record is what a preempted run leaves behind. Drop it
    # and keep every whole record before it, rather than refusing the file.
    count = available // stride
    body = np.frombuffer(raw, dtype=np.uint8, count=count * stride, offset=offset)
    records = body.reshape(count, stride) if count else body.reshape(0, stride)
    steps = np.ascontiguousarray(records[:, : _STEP_DTYPE.itemsize]).view(_STEP_DTYPE)
    values = np.ascontiguousarray(records[:, _STEP_DTYPE.itemsize :]).view(_VALUE_DTYPE)
    return Log(
        columns=tuple(
            Column(
                metric_id=int(row["metric_id"]),
                scope_id=int(row["scope_id"]),
                layer=int(row["layer"]),
                element_count=int(row["element_count"]),
                index=int(row["index"]),
            )
            for row in table
        ),
        steps=steps.reshape(count),
        values=values.reshape(count, column_count),
        tokens_per_step=int(tokens_per_step),
        flops_per_token=float(flops_per_token),
    )
