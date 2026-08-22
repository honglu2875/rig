"""The add-only registry of logged quantities and the scopes they measure.

Every number a run records -- an axis like ``step``, a run-level scalar like
``train_loss``, or a per-scope diagnostic like ``grad.l2_norm`` -- is written
under a stable ``int32`` id. Recorded logs name their columns by id, never by
position or by text, so a file stays readable when the set of columns changes.

Why ids rather than column names
--------------------------------
Columns move. A tier adds layers, a config disables diagnostics, a later
version records a statistic this one does not. Under positional or textual
columns every such change is a format change, and old artifacts rot. Under
ids, a reader looks up the columns it understands and ignores the rest, so
adding a metric costs four bytes per sample and breaks nothing.

The add-only rule
-----------------
**An id, once assigned, is permanent.** Never renumber one, never rename one,
and never reuse the id of a metric you removed -- retire it instead. A run
recorded a year ago must still decode correctly today, and the only thing
making that true is that nobody edited the left column.

``registry.txt`` is the enforcement. It is a checked-in snapshot of every
assignment, and ``tests/test_metrics_registry.py`` fails if any line changes
or disappears. Adding a metric is therefore two deliberate edits: the entry
here, and its line appended to the snapshot. That friction is the point.

Numbering is grouped only for reading convenience -- 1-99 axes and run-level
scalars, 100+ per-scope diagnostics by family. Nothing depends on the ranges.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REGISTRY_PATH = Path(__file__).with_name("registry.txt")

# Diagnostic statistics, and whether the value is a sum over the scope's
# elements rather than an average of them. The unnormalized two scale with
# scope size, so charting them across scopes of different sizes is only
# meaningful alongside the scope's element count.
_UNNORMALIZED_STATS = ("l1_norm", "l2_norm")
_NORMALIZED_STATS = ("mean", "std", "third_moment", "fourth_moment")
_PERCENTILE_STATS = ("p01", "p10", "p50", "p90", "p99")


@dataclass(frozen=True)
class Metric:
    """One logged quantity and its permanent identity."""

    id: int
    name: str
    family: str | None = None
    stat: str | None = None

    @property
    def normalized(self) -> bool:
        """Whether the value is already divided by the scope's element count."""

        return (
            self.stat is None
            or self.stat in _NORMALIZED_STATS
            or self.stat in _PERCENTILE_STATS
        )


@dataclass(frozen=True)
class Scope:
    """One region of the model a diagnostic can be measured over."""

    id: int
    name: str
    layered: bool = False
    # Whether the scope addresses a second axis inside each layer, carried in
    # the column table's index slot. Only ``expert`` does.
    indexed: bool = False


# ---------------------------------------------------------------------------
# Registry. Append only; see the module docstring before editing.
# ---------------------------------------------------------------------------

_AXES_AND_SCALARS = (
    Metric(1, "step"),
    Metric(2, "tokens_processed"),
    Metric(3, "cumulative_flops"),
    Metric(4, "train_loss"),
    Metric(5, "learning_rate"),
    Metric(6, "grad_norm"),
    # Routing. Recorded by mixture-of-experts recipes and absent otherwise,
    # which costs a reader nothing: a dense run simply has no column with
    # these ids. Load is the fraction of assignments an expert received, so
    # 1/experts is perfectly even and 1.0 is total collapse onto one expert.
    Metric(7, "router.balance_loss"),
    Metric(8, "router.max_load"),
    Metric(9, "router.min_load"),
    # The exact per-expert distribution rather than a histogram of it: with a
    # handful of experts the vector *is* the distribution, and it sums to 1.
    Metric(10, "router.load"),
    # How spread out the router's choice is, in nats. log(experts) means it
    # has no preference at all; 0 means it always picks the same expert.
    Metric(11, "router.entropy"),
    # Mean weight the gate puts on the expert it ranked first, so 1/top_k is
    # an even split across the chosen experts and 1.0 is full confidence.
    Metric(12, "router.top1_gate"),
    # Root-mean-square router logit. Grows without bound when nothing holds it
    # down, and a collapsing router shows it here before the loss moves.
    Metric(13, "router.logit_rms"),
)

_DIAGNOSTIC_FAMILY_BASES = (("param", 100), ("grad", 200), ("update", 300))


def _diagnostic_metrics() -> tuple[Metric, ...]:
    """Assign one id per (family, stat) pair, ordered so ids stay reproducible.

    The offsets are fixed by position in these tuples, so a statistic may be
    appended but never inserted or reordered.
    """

    stats = (*_UNNORMALIZED_STATS, *_NORMALIZED_STATS, *_PERCENTILE_STATS)
    return tuple(
        Metric(base + offset, f"{family}.{stat}", family=family, stat=stat)
        for family, base in _DIAGNOSTIC_FAMILY_BASES
        for offset, stat in enumerate(stats)
    )


METRICS: tuple[Metric, ...] = (*_AXES_AND_SCALARS, *_diagnostic_metrics())

# Diagnostic grid profiles derived from the registry rather than restated.
# Existing recipes retain the six-statistic core; opt-in experiments can use
# the extended profile without changing historical artifact layouts.
DIAGNOSTIC_FAMILIES: tuple[str, ...] = tuple(
    family for family, _ in _DIAGNOSTIC_FAMILY_BASES
)
DIAGNOSTIC_CORE_STATS: tuple[str, ...] = (*_UNNORMALIZED_STATS, *_NORMALIZED_STATS)
DIAGNOSTIC_PERCENTILE_STATS: tuple[str, ...] = _PERCENTILE_STATS
DIAGNOSTIC_EXTENDED_STATS: tuple[str, ...] = (
    *DIAGNOSTIC_CORE_STATS,
    *DIAGNOSTIC_PERCENTILE_STATS,
)
# Compatibility name used by the original rectangular diagnostic protocol.
DIAGNOSTIC_STATS: tuple[str, ...] = DIAGNOSTIC_CORE_STATS

SCOPES: tuple[Scope, ...] = (
    Scope(1, "overall"),
    Scope(2, "embeddings"),
    Scope(3, "unembedding"),
    Scope(4, "block", layered=True),
    Scope(5, "final_norm"),
    # One routed expert within one block. Layered by transformer layer and
    # indexed by expert ordinal, so per-expert load is addressable without
    # inventing a metric id per expert.
    Scope(6, "expert", layered=True, indexed=True),
)


def _index(entries, label: str) -> tuple[dict[int, object], dict[str, object]]:
    by_id: dict[int, object] = {}
    by_name: dict[str, object] = {}
    for entry in entries:
        if entry.id <= 0 or entry.id > 0x7FFFFFFF:
            raise ValueError(f"{label} id must be a positive int32: {entry}")
        if entry.id in by_id:
            raise ValueError(f"duplicate {label} id {entry.id}: {entry.name}")
        if entry.name in by_name:
            raise ValueError(f"duplicate {label} name {entry.name!r}")
        by_id[entry.id] = entry
        by_name[entry.name] = entry
    return by_id, by_name


_METRIC_BY_ID, _METRIC_BY_NAME = _index(METRICS, "metric")
_SCOPE_BY_ID, _SCOPE_BY_NAME = _index(SCOPES, "scope")


def metric(name: str) -> Metric:
    """Return the metric registered under ``name``."""

    try:
        return _METRIC_BY_NAME[name]  # type: ignore[return-value]
    except KeyError:
        raise KeyError(
            f"unregistered metric {name!r}; add it to rig/metrics.py and "
            f"append its line to {REGISTRY_PATH.name}"
        ) from None


def scope(name: str) -> Scope:
    """Return the scope registered under ``name``."""

    try:
        return _SCOPE_BY_NAME[name]  # type: ignore[return-value]
    except KeyError:
        raise KeyError(
            f"unregistered scope {name!r}; add it to rig/metrics.py and "
            f"append its line to {REGISTRY_PATH.name}"
        ) from None


def metric_by_id(identifier: int) -> Metric | None:
    """Return the metric for ``identifier``, or None if this build predates it.

    Returning None rather than raising is deliberate: a reader must skip
    columns written by a later version instead of refusing the whole file.
    """

    return _METRIC_BY_ID.get(identifier)  # type: ignore[return-value]


def scope_by_id(identifier: int) -> Scope | None:
    """Return the scope for ``identifier``, or None if this build predates it."""

    return _SCOPE_BY_ID.get(identifier)  # type: ignore[return-value]


def registry_lines() -> tuple[str, ...]:
    """Render the snapshot the add-only test compares against."""

    return (
        *(f"metric\t{entry.id}\t{entry.name}" for entry in METRICS),
        *(f"scope\t{entry.id}\t{entry.name}" for entry in SCOPES),
    )
