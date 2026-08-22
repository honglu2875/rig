from __future__ import annotations

import unittest

from rig import metrics


def _assignments(lines) -> dict[tuple[str, int], str]:
    """Map (kind, id) -> name so comparison does not depend on file order."""

    parsed: dict[tuple[str, int], str] = {}
    for line in lines:
        kind, identifier, name = line.split("\t")
        parsed[(kind, int(identifier))] = name
    return parsed


def _snapshot() -> list[str]:
    text = metrics.REGISTRY_PATH.read_text(encoding="utf-8")
    return [
        line for line in text.splitlines() if line.strip() and not line.startswith("#")
    ]


class MetricsRegistryTests(unittest.TestCase):
    def test_assignments_are_add_only(self) -> None:
        """Every recorded id keeps its name forever, or old logs stop decoding.

        A failure here is almost never a reason to edit the snapshot in place.
        Renumbering or renaming silently reinterprets artifacts already on
        disk; the fix is to restore the assignment and add a new id instead.
        """

        recorded = _assignments(_snapshot())
        current = _assignments(metrics.registry_lines())

        dropped = sorted(key for key in recorded if key not in current)
        self.assertEqual(
            dropped,
            [],
            "these ids were removed from the registry, which orphans every log "
            f"that used them; retire ids in place instead: {dropped}",
        )
        renamed = {
            key: (recorded[key], current[key])
            for key in recorded
            if key in current and current[key] != recorded[key]
        }
        self.assertEqual(
            renamed,
            {},
            "these ids were reassigned, which silently reinterprets artifacts "
            f"already on disk; assign a new id instead: {renamed}",
        )
        added = sorted(key for key in current if key not in recorded)
        self.assertEqual(
            added,
            [],
            "new entries are registered but not snapshotted; append to "
            f"{metrics.REGISTRY_PATH.name}:\n"
            + "\n".join(
                f"{kind}\t{value}\t{current[(kind, value)]}" for kind, value in added
            ),
        )

    def test_ids_and_names_are_unique_positive_int32(self) -> None:
        for entries, label in ((metrics.METRICS, "metric"), (metrics.SCOPES, "scope")):
            with self.subTest(kind=label):
                identifiers = [entry.id for entry in entries]
                names = [entry.name for entry in entries]
                self.assertEqual(len(identifiers), len(set(identifiers)))
                self.assertEqual(len(names), len(set(names)))
                self.assertTrue(all(0 < value <= 0x7FFFFFFF for value in identifiers))

    def test_unknown_ids_resolve_to_none_so_readers_can_skip_them(self) -> None:
        # A file written by a later version carries columns this build has
        # never heard of. Skipping them must not fail the whole read.
        self.assertIsNone(metrics.metric_by_id(999_999))
        self.assertIsNone(metrics.scope_by_id(999_999))
        self.assertEqual(metrics.metric_by_id(4).name, "train_loss")
        self.assertEqual(metrics.scope_by_id(4).name, "block")

    def test_lookup_by_name_points_at_the_registry(self) -> None:
        self.assertEqual(metrics.metric("grad.l2_norm").id, 201)
        self.assertEqual(metrics.metric("grad.l2_norm").family, "grad")
        self.assertEqual(metrics.scope("block").layered, True)
        with self.assertRaisesRegex(KeyError, "registry.txt"):
            metrics.metric("grad.nonexistent")

    def test_normalized_flag_matches_how_the_statistic_is_computed(self) -> None:
        # l1/l2 are sums over the scope; the moments are divided by the count.
        # The report needs the distinction to compare scopes of different size.
        self.assertFalse(metrics.metric("param.l1_norm").normalized)
        self.assertFalse(metrics.metric("param.l2_norm").normalized)
        for stat in (
            "mean",
            "std",
            "third_moment",
            "fourth_moment",
            "p01",
            "p10",
            "p50",
            "p90",
            "p99",
        ):
            with self.subTest(stat=stat):
                self.assertTrue(metrics.metric(f"param.{stat}").normalized)
        self.assertTrue(metrics.metric("train_loss").normalized)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
