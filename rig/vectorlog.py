"""Append-only packed logs for dense per-layer feature vectors.

Scalar :mod:`rig.logpack` columns are deliberately self-describing, but a
feature diagnostic can have hundreds of thousands of coordinates per sample.
Repeating a 24-byte column descriptor for every ``(metric, layer, feature)``
would make its header larger than many complete runs.  This companion format
keeps those three axes dense and implicit while retaining permanent metric ids.

Layout, little-endian throughout::

    MAGIC
    metric_count     int32
    layer_count      int32
    feature_count    int32
    group_size       int32
    tokens_per_step  int64
    flops_per_token  float64
    metric_ids       metric_count x int32
    records          n x (step int32, values float32[metric, layer, feature])

The fixed record stride makes the file valid after every complete append.  A
reader ignores a partial trailing record left by a preemption and memory-maps
the complete prefix, so multi-gigabyte studies do not need to be copied into
RAM merely to compute plots.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable, Sequence

import numpy as np

from rig import metrics


MAGIC = b"RIGFVEC\x01"
SUFFIX = ".rigvec"
HEADER_DTYPE = np.dtype(
    [
        ("metric_count", "<i4"),
        ("layer_count", "<i4"),
        ("feature_count", "<i4"),
        ("group_size", "<i4"),
        ("tokens_per_step", "<i8"),
        ("flops_per_token", "<f8"),
    ]
)
_METRIC_DTYPE = np.dtype("<i4")
_STEP_DTYPE = np.dtype("<i4")
_VALUE_DTYPE = np.dtype("<f4")


class VectorLogError(Exception):
    """Raised when a dense vector log is malformed or unreadable."""


def _record_values(metric_count: int, layer_count: int, feature_count: int) -> int:
    return metric_count * layer_count * feature_count


def _header_size(metric_count: int) -> int:
    return len(MAGIC) + HEADER_DTYPE.itemsize + metric_count * _METRIC_DTYPE.itemsize


def _record_size(metric_count: int, layer_count: int, feature_count: int) -> int:
    return _STEP_DTYPE.itemsize + _record_values(
        metric_count, layer_count, feature_count
    ) * _VALUE_DTYPE.itemsize


class VectorLogWriter:
    """Incrementally write one fixed ``[metric, layer, feature]`` tensor."""

    def __init__(
        self,
        path: Path,
        metric_names: Sequence[str],
        *,
        layer_count: int,
        feature_count: int,
        group_size: int,
        tokens_per_step: int,
        flops_per_token: float,
    ) -> None:
        if not metric_names or len(metric_names) != len(set(metric_names)):
            raise ValueError("vector-log metrics must be nonempty and unique")
        if layer_count <= 0 or feature_count <= 0:
            raise ValueError("vector-log layer and feature counts must be positive")
        if group_size <= 0 or feature_count % group_size:
            raise ValueError("vector-log group size must divide the feature count")
        if tokens_per_step <= 0:
            raise ValueError("tokens_per_step must be positive")
        if not np.isfinite(flops_per_token) or flops_per_token <= 0:
            raise ValueError("flops_per_token must be finite and positive")

        self.path = Path(path)
        self.metric_names = tuple(metric_names)
        self.metric_ids = tuple(metrics.metric(name).id for name in metric_names)
        self.layer_count = int(layer_count)
        self.feature_count = int(feature_count)
        self.group_size = int(group_size)
        self.tokens_per_step = int(tokens_per_step)
        self.flops_per_token = float(flops_per_token)
        self._shape = (
            len(self.metric_names),
            self.layer_count,
            self.feature_count,
        )
        self._handle: BinaryIO | None = None
        self._last_step = 0

    def __enter__(self) -> "VectorLogWriter":
        self._open()
        return self

    def __exit__(self, *exception: object) -> None:
        self.close()

    def _open(self) -> BinaryIO:
        if self._handle is not None:
            return self._handle
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("wb")
        handle.write(MAGIC)
        header = np.zeros(1, dtype=HEADER_DTYPE)
        header[0] = (
            len(self.metric_names),
            self.layer_count,
            self.feature_count,
            self.group_size,
            self.tokens_per_step,
            self.flops_per_token,
        )
        handle.write(header.tobytes())
        handle.write(np.asarray(self.metric_ids, dtype=_METRIC_DTYPE).tobytes())
        handle.flush()
        self._handle = handle
        return handle

    def append(self, step: int, values: Iterable[float] | np.ndarray) -> None:
        if step <= self._last_step:
            raise ValueError(
                f"vector-log steps must increase; got {step} after {self._last_step}"
            )
        array = np.asarray(values, dtype=_VALUE_DTYPE)
        if array.shape != self._shape:
            raise ValueError(
                f"vector-log sample must have shape {self._shape}, got {array.shape}"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError("vector-log sample must be finite")
        handle = self._open()
        handle.write(np.asarray(step, dtype=_STEP_DTYPE).tobytes())
        handle.write(np.ascontiguousarray(array).tobytes())
        handle.flush()
        self._last_step = int(step)

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


@dataclass(frozen=True)
class VectorLog:
    """A memory-mapped ``[sample, metric, layer, feature]`` vector log."""

    path: Path
    metric_ids: tuple[int, ...]
    steps: np.ndarray
    values: np.ndarray
    layer_count: int
    feature_count: int
    group_size: int
    tokens_per_step: int
    flops_per_token: float

    def __len__(self) -> int:
        return int(self.steps.shape[0])

    @property
    def metric_names(self) -> tuple[str | None, ...]:
        return tuple(
            entry.name if (entry := metrics.metric_by_id(identifier)) else None
            for identifier in self.metric_ids
        )

    def metric(self, name: str) -> np.ndarray | None:
        identifier = metrics.metric(name).id
        try:
            position = self.metric_ids.index(identifier)
        except ValueError:
            return None
        return self.values[:, position, :, :]

    def axis(self, name: str) -> np.ndarray:
        if name == "step":
            return self.steps
        if name == "tokens_processed":
            return self.steps * np.int64(self.tokens_per_step)
        if name == "cumulative_flops":
            return self.steps.astype(np.float64) * (
                self.tokens_per_step * self.flops_per_token
            )
        raise KeyError(f"{name!r} is not an index axis")


def read_vector_log(path: Path) -> VectorLog:
    """Validate and memory-map every complete record in ``path``."""

    path = Path(path)
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            magic = handle.read(len(MAGIC))
            header_bytes = handle.read(HEADER_DTYPE.itemsize)
            if magic != MAGIC:
                raise VectorLogError(f"{path} is not a rig vector log (bad magic)")
            if len(header_bytes) != HEADER_DTYPE.itemsize:
                raise VectorLogError(f"{path} is truncated inside its header")
            header = np.frombuffer(header_bytes, dtype=HEADER_DTYPE, count=1)[0]
            metric_count = int(header["metric_count"])
            layer_count = int(header["layer_count"])
            feature_count = int(header["feature_count"])
            group_size = int(header["group_size"])
            tokens_per_step = int(header["tokens_per_step"])
            flops_per_token = float(header["flops_per_token"])
            if metric_count <= 0 or layer_count <= 0 or feature_count <= 0:
                raise VectorLogError(f"{path} declares a nonpositive tensor shape")
            if group_size <= 0 or feature_count % group_size:
                raise VectorLogError(f"{path} declares an invalid feature group size")
            if tokens_per_step <= 0 or not np.isfinite(flops_per_token):
                raise VectorLogError(f"{path} declares invalid axis accounting")
            metric_bytes = handle.read(metric_count * _METRIC_DTYPE.itemsize)
            if len(metric_bytes) != metric_count * _METRIC_DTYPE.itemsize:
                raise VectorLogError(f"{path} is truncated inside its metric table")
    except OSError as exc:
        raise VectorLogError(f"could not read {path}: {exc}") from exc

    metric_ids = tuple(int(value) for value in np.frombuffer(metric_bytes, _METRIC_DTYPE))
    if len(set(metric_ids)) != len(metric_ids) or any(value <= 0 for value in metric_ids):
        raise VectorLogError(f"{path} has duplicate or invalid metric ids")
    offset = _header_size(metric_count)
    stride = _record_size(metric_count, layer_count, feature_count)
    available = max(0, size - offset)
    count = available // stride
    if count:
        mapping = np.memmap(path, mode="r", dtype=np.uint8)
        steps = np.ndarray(
            shape=(count,),
            dtype=_STEP_DTYPE,
            buffer=mapping,
            offset=offset,
            strides=(stride,),
        )
        values = np.ndarray(
            shape=(count, metric_count, layer_count, feature_count),
            dtype=_VALUE_DTYPE,
            buffer=mapping,
            offset=offset + _STEP_DTYPE.itemsize,
            strides=(stride, layer_count * feature_count * 4, feature_count * 4, 4),
        )
    else:
        steps = np.empty(0, dtype=_STEP_DTYPE)
        values = np.empty(
            (0, metric_count, layer_count, feature_count), dtype=_VALUE_DTYPE
        )
    if len(steps) > 1 and np.any(np.diff(steps) <= 0):
        raise VectorLogError(f"{path} steps must increase")
    return VectorLog(
        path=path,
        metric_ids=metric_ids,
        steps=steps,
        values=values,
        layer_count=layer_count,
        feature_count=feature_count,
        group_size=group_size,
        tokens_per_step=tokens_per_step,
        flops_per_token=flops_per_token,
    )


__all__ = (
    "MAGIC",
    "SUFFIX",
    "VectorLog",
    "VectorLogError",
    "VectorLogWriter",
    "read_vector_log",
)
