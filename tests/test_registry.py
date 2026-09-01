import unittest
from pathlib import Path

from src.evaluation.protocol import HOLDOUT_SECOND, NATIVE, PAIRED, resolve_oos_mode
from src.training.registry import (
    BATCH_SPLIT_MODELS,
    COUNT_MODELS,
    MODELS,
    MODEL_OVERRIDES,
    extra_for,
)
from src.utils.io import load_yaml

TRAINING_CONFIG = Path("config/training.yaml")
VELOCITY_BACKEND_PAIRS = (
    ("pbmc_retained", "pbmc_regvelo"),
    ("bmmc_cite_retained", "bmmc_cite_regvelo"),
    ("papalexi_retained", "papalexi_regvelo"),
)
COMPARE_KEYS = {"kot_velocity_compare_h5ad", "kot_velocity_compare_label"}


class RegistryTests(unittest.TestCase):
    def test_every_model_declares_how_to_load_and_its_oos_capability(self):
        allowed = {NATIVE, HOLDOUT_SECOND, PAIRED}
        for name, spec in MODELS.items():
            self.assertIn("module", spec, name)
            self.assertIn("function", spec, name)
            self.assertIn(spec["oos"], allowed, name)
            self.assertEqual(resolve_oos_mode(name), spec["oos"], name)

    def test_overrides_only_exist_for_registered_models(self):
        self.assertTrue(set(MODEL_OVERRIDES).issubset(MODELS))

    def test_count_and_batch_split_sets_match_the_previous_hardcoded_lists(self):
        self.assertEqual(COUNT_MODELS, {"glue", "totalvi"})
        self.assertEqual(BATCH_SPLIT_MODELS, {"scot", "moscot", "linear_ode"})

    def test_optional_extras_match_the_previous_lookup(self):
        self.assertEqual(extra_for("kot"), "kot")
        self.assertEqual(extra_for("kot_nodyn"), "kot")
        self.assertEqual(extra_for("glue"), "baselines")
        self.assertIsNone(extra_for("scot"))
        self.assertIsNone(extra_for("linear_ode"))


class TrainingConfigTests(unittest.TestCase):
    def test_velocity_backend_pairs_differ_only_in_the_compare_diagnostic(self):
        datasets = load_yaml(TRAINING_CONFIG)["datasets"]
        for base, variant in VELOCITY_BACKEND_PAIRS:
            self.assertEqual(set(datasets[base]), set(datasets[variant]), (base, variant))
            differ = {
                key for key in datasets[base]
                if datasets[base][key] != datasets[variant][key]
            }
            self.assertEqual(differ, COMPARE_KEYS, (base, variant, differ))


if __name__ == "__main__":
    unittest.main()
