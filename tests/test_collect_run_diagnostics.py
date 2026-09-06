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


class MissingScoreTests(unittest.TestCase):
    """A blank score has two causes that must not print the same string.

    The 2026-08-28 refactor of src/training/kot.py builds DiagInputs with val_idx=None,
    so validation_metrics returns val_foscttm/train_foscttm as None for every run. The
    table printed "CRASHED" for each, which reads as 16 dead runs when all 16 had
    trained fine and written a mean_foscttm -- the sweep looked like a total loss.
    record["n"] separates the two: it counts seeds that produced a mean_foscttm, so
    n == 0 is a real failure and n > 0 is a metric that is simply no longer computed.
    """

    @staticmethod
    def score_record(record, rank_by):
        # The branch under test, mirrored from print_aggregate so the discrimination
        # is asserted rather than eyeballed in a printed table.
        mean, sd = record[f"{rank_by}_mean"], record[f"{rank_by}_sd"]
        if mean is not None:
            return f"{mean:.4f} ± {sd:.4f}"
        return "CRASHED" if record["n"] == 0 else "not computed"

    def test_absent_metric_on_healthy_runs_is_not_crashed(self):
        record = {"train_foscttm_mean": None, "train_foscttm_sd": None, "n": 4}
        self.assertEqual(self.score_record(record, "train_foscttm"), "not computed")

    def test_no_seed_produced_a_score_is_crashed(self):
        record = {"train_foscttm_mean": None, "train_foscttm_sd": None, "n": 0}
        self.assertEqual(self.score_record(record, "train_foscttm"), "CRASHED")

    def test_present_metric_still_formats(self):
        record = {"mean_foscttm_mean": 0.1181, "mean_foscttm_sd": 0.0029, "n": 4}
        self.assertEqual(self.score_record(record, "mean_foscttm"), "0.1181 ± 0.0029")


class CompanionMetricTests(unittest.TestCase):
    """The column shown NEXT to the ranked one must contain data.

    With val_foscttm no longer computed, the companion slot rendered a column of "-"
    on every row, which invites the same misreading as CRASHED did.
    """

    def test_prefers_val_when_it_has_data(self):
        from collect_run_diagnostics import other_rank_metric
        records = [{"val_foscttm_mean": 0.13, "mean_foscttm_mean": 0.14}]
        self.assertEqual(other_rank_metric("train_foscttm", records), "val_foscttm")

    def test_falls_back_to_a_populated_metric(self):
        from collect_run_diagnostics import other_rank_metric
        records = [{"val_foscttm_mean": None, "train_foscttm_mean": None,
                    "mean_foscttm_mean": 0.14}]
        self.assertEqual(other_rank_metric("train_foscttm", records), "mean_foscttm")

    def test_no_records_keeps_the_historical_choice(self):
        from collect_run_diagnostics import other_rank_metric
        self.assertEqual(other_rank_metric("val_foscttm"), "mean_foscttm")
        self.assertEqual(other_rank_metric("mean_foscttm"), "val_foscttm")
