import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from collect_run_diagnostics import RANK_LABELS, is_true, normalise_fit_flag, rank_label


class RankLabelTests(unittest.TestCase):
    def test_each_foscttm_names_the_cells_it_averages(self):
        self.assertEqual(rank_label("train_foscttm"), "TRAIN")
        self.assertEqual(rank_label("val_foscttm"), "VAL")
        self.assertEqual(rank_label("mean_foscttm"), "TEST")

    def test_train_is_not_labelled_test(self):
        # The regression: ranking on train_foscttm printed "TEST", which reads as a
        # held-out score for a column measured on the cells the run fitted.
        self.assertNotEqual(rank_label("train_foscttm"), "TEST")

    def test_every_rankable_metric_has_a_label(self):
        from collect_run_diagnostics import RANK_METRICS
        self.assertEqual(set(RANK_METRICS), set(RANK_LABELS))


class IsTrueTests(unittest.TestCase):
    def test_accepts_yaml_and_csv_spellings(self):
        for value in (True, "True", "true", " TRUE "):
            self.assertTrue(is_true(value))

    def test_everything_else_is_false(self):
        for value in (False, "False", "false", None, "", 0):
            self.assertFalse(is_true(value))


class FitFlagTests(unittest.TestCase):
    """The 2026-08-26 sweep recorded train_on_val_side; later runs record
    fit_tuning_subset. Both must land in one column or a --group-by silently pools
    tuning-subset runs together with full-side ones."""

    def test_deprecated_name_alone_sets_the_new_one(self):
        row = {"fit_tuning_subset": None, "train_on_val_side": "True"}
        normalise_fit_flag(row)
        self.assertEqual(row["fit_tuning_subset"], "True")

    def test_new_name_alone_is_left_as_is(self):
        row = {"fit_tuning_subset": "True", "train_on_val_side": None}
        normalise_fit_flag(row)
        self.assertEqual(row["fit_tuning_subset"], "True")

    def test_a_run_that_fitted_the_large_side_stays_false(self):
        row = {"fit_tuning_subset": None, "train_on_val_side": "False"}
        normalise_fit_flag(row)
        self.assertEqual(row["fit_tuning_subset"], "False")

    def test_both_present_and_false_stays_false(self):
        row = {"fit_tuning_subset": "False", "train_on_val_side": "False"}
        normalise_fit_flag(row)
        self.assertEqual(row["fit_tuning_subset"], "False")


if __name__ == "__main__":
    unittest.main()
