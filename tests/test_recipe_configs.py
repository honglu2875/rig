"""Contracts for the three explicit, standalone recipe configurations."""

from __future__ import annotations

from pathlib import Path
import unittest

from rig.configfile import read_config_document


ROOT = Path(__file__).parents[1]
RECIPES = {
    "reference": 6,
    "reference_duration": 6,
    "reference_moe": 6,
    "expert_load_moe": 7,
}


class StandaloneRecipeConfigTests(unittest.TestCase):
    def test_each_execution_type_has_one_complete_document(self) -> None:
        filenames = {
            "official": "config.yaml",
            "dev": "dev.yaml",
            "smoke": "smoke.yaml",
        }
        for recipe, schema_version in RECIPES.items():
            for execution_type, filename in filenames.items():
                with self.subTest(recipe=recipe, execution_type=execution_type):
                    document, _ = read_config_document(
                        ROOT / "recipes" / recipe / filename
                    )
                    self.assertEqual(document["schema_version"], schema_version)
                    self.assertEqual(document["execution_type"], execution_type)
                    self.assertEqual(
                        set(document),
                        {
                            "schema_version",
                            "execution_type",
                            "family",
                            "run",
                        },
                    )
                    self.assertNotIn("profiles", document)

    def test_dev_and_official_scientific_families_cannot_drift(self) -> None:
        for recipe in RECIPES:
            with self.subTest(recipe=recipe):
                root = ROOT / "recipes" / recipe
                official, _ = read_config_document(root / "config.yaml")
                dev, _ = read_config_document(root / "dev.yaml")
                self.assertEqual(dev["family"], official["family"])
                self.assertEqual(dev["run"]["kernels"], official["run"]["kernels"])
                self.assertEqual(dev["run"]["optimizer"], official["run"]["optimizer"])
                self.assertEqual(
                    dev["run"]["training"]["sampling"],
                    official["run"]["training"]["sampling"],
                )
                self.assertEqual(
                    dev["run"]["training"]["dtype"],
                    official["run"]["training"]["dtype"],
                )

    def test_duration_policy_is_explicit_in_each_document(self) -> None:
        for recipe in RECIPES:
            root = ROOT / "recipes" / recipe
            for filename in ("config.yaml", "dev.yaml"):
                document, _ = read_config_document(root / filename)
                self.assertEqual(
                    set(document["run"]["training"]["duration"]),
                    {"tokens_per_parameter"},
                )
            smoke, _ = read_config_document(root / "smoke.yaml")
            self.assertEqual(set(smoke["run"]["training"]["duration"]), {"steps"})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
