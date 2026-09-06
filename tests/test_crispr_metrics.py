import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.crispr_metrics import (
    annotate_effect_sets,
    bootstrap_correlation_difference,
    score_effect_subsets,
    confirmed_knockouts,
    response_cosine,
)


class EffectSetTests(unittest.TestCase):
    def table(self) -> pd.DataFrame:
        # Four well-sampled trans pairs (the primary set), one well-sampled self pair,
        # one undersampled self pair. Replicate rows share the pair-level flags.
        rows = []
        trans = [
            ("ATF2", "PDL1", 2.0, 0.01, 0.1),
            ("STAT2", "PDL1", 0.5, 0.40, 0.1),
            ("IRF1", "CD86", 0.2, 0.05, 0.1),
            ("CAV1", "CD86", 0.1, 0.50, 0.1),
        ]
        for perturbation, protein, delta_p, delta_r, se in trans:
            for replicate in ("rep1", "rep2"):
                rows.append({
                    "perturbation": perturbation, "protein": protein,
                    "replicate": replicate, "delta_protein": delta_p,
                    "delta_cognate_rna": delta_r, "protein_effect_se": se,
                    "passes_threshold": True, "is_self_effect": False,
                })
        for replicate in ("rep1", "rep2"):
            rows.append({
                "perturbation": "CD86", "protein": "CD86", "replicate": replicate,
                "delta_protein": -1.0, "delta_cognate_rna": 0.0,
                "protein_effect_se": 0.1, "passes_threshold": True,
                "is_self_effect": True,
            })
        rows.append({
            "perturbation": "CD274", "protein": "PDL1", "replicate": "rep1",
            "delta_protein": -2.0, "delta_cognate_rna": 0.0,
            "protein_effect_se": 0.1, "passes_threshold": False,
            "is_self_effect": True,
        })
        return pd.DataFrame(rows)

    def test_self_pairs_are_not_in_the_primary_set(self):
        annotated = annotate_effect_sets(self.table())
        primary = annotated[annotated["in_primary"]]
        self.assertFalse(primary["is_self_effect"].any())
        self.assertEqual(primary.groupby(["perturbation", "protein"]).ngroups, 4)

    def test_undersampled_self_pairs_are_not_in_the_self_set(self):
        annotated = annotate_effect_sets(self.table())
        self_set = annotated[annotated["in_self"]]
        self.assertTrue((self_set["perturbation"] == "CD86").all())
        self.assertNotIn("CD274", set(self_set["perturbation"]))

    def test_post_transcriptional_uses_quantiles_not_gene_names(self):
        annotated = annotate_effect_sets(self.table())
        post = annotated[annotated["in_post_transcriptional"]]
        pairs = set(zip(post["perturbation"], post["protein"]))
        self.assertIn(("ATF2", "PDL1"), pairs)
        self.assertNotIn(("CD86", "CD86"), pairs)


class ConfirmedKnockoutTests(unittest.TestCase):
    def test_unresolved_targets_are_returned_not_silently_dropped(self):
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as handle:
            handle.write("knockout,status,note\n")
            handle.write("IRF1,confirmed,\n")
            handle.write("CD274,unresolved,arrayed only\n")
            path = Path(handle.name)
        confirmed, skipped = confirmed_knockouts(path)
        self.assertEqual(confirmed, ["IRF1"])
        self.assertEqual(skipped, ["CD274"])


class PairedBootstrapTests(unittest.TestCase):
    def table(self) -> pd.DataFrame:
        rows = []
        for perturbation in ("KO1", "KO2", "KO3", "KO4"):
            for protein, observed in zip(("A", "B", "C"), (1.0, 2.0, 4.0)):
                rows.append({
                    "perturbation": perturbation,
                    "protein": protein,
                    "observed": observed,
                    "full": observed,
                    "ablation": -observed,
                })
        return pd.DataFrame(rows)

    def test_same_draw_gives_full_minus_ablation_for_both_correlations(self):
        for metric in ("spearman", "pearson"):
            mean, lower, upper = bootstrap_correlation_difference(
                self.table(), "perturbation", "full", "ablation", "observed",
                metric, n_boot=100, seed=7,
            )
            self.assertAlmostEqual(mean, 2.0)
            self.assertAlmostEqual(lower, 2.0)
            self.assertAlmostEqual(upper, 2.0)

    def test_complete_cases_are_shared_by_the_two_arms(self):
        table = self.table()
        table.loc[table["protein"] == "C", "ablation"] = np.nan
        mean, lower, upper = bootstrap_correlation_difference(
            table, "perturbation", "full", "ablation", "observed",
            "spearman", n_boot=50, seed=3,
        )
        self.assertAlmostEqual(mean, 2.0)
        self.assertAlmostEqual(lower, 2.0)
        self.assertAlmostEqual(upper, 2.0)


class ResponseCosineTests(unittest.TestCase):
    def test_single_protein_response_is_not_scored(self):
        frame = pd.DataFrame({
            "perturbation": ["one", "two", "two"],
            "predicted": [1.0, 1.0, 2.0],
            "observed": [-1.0, 1.0, 2.0],
        })
        scores = response_cosine(frame, "predicted", "observed").set_index(
            "perturbation"
        )
        self.assertTrue(np.isnan(scores.loc["one", "cosine"]))
        self.assertEqual(scores.loc["one", "n_proteins"], 1)
        self.assertAlmostEqual(scores.loc["two", "cosine"], 1.0)


if __name__ == "__main__":
    unittest.main()


class SharedEffectSubsetScorerTests(unittest.TestCase):
    """One scorer for every Task B ISP.

    The linear runner and the `task_b` adapter each grew their own and diverged: the
    adapter scored all 100 (perturbation, protein) pairs while the linear runner scored
    the four frozen subsets, so RegVelo and the linear baseline could not be tabled
    together.
    """

    def frame(self):
        rows = []
        for i, pert in enumerate(f"KO{n}" for n in range(6)):
            for protein in "ABCD":
                observed = float(i) + ord(protein)
                rows.append({
                    "perturbation": pert, "replicate": "rep1", "protein": protein,
                    "pred": observed * 2.0, "obs": observed,
                    "in_primary": True, "in_significant": i < 3,
                    "in_post_transcriptional": False, "in_self": False,
                })
        return pd.DataFrame(rows)

    def test_only_flagged_rows_enter_each_subset(self):
        scored = score_effect_subsets(self.frame(), "pred", "obs", 50, 0)
        by_subset = scored.set_index("effect_subset")["n"]
        self.assertEqual(int(by_subset["primary"]), 24)
        self.assertEqual(int(by_subset["significant"]), 12)
        # Empty subsets are dropped rather than reported as zero-effect rows.
        self.assertNotIn("post_transcriptional", by_subset.index)
        self.assertNotIn("self", by_subset.index)

    def test_a_perfectly_ranked_prediction_scores_one(self):
        scored = score_effect_subsets(self.frame(), "pred", "obs", 50, 0)
        primary = scored[scored["effect_subset"] == "primary"].iloc[0]
        self.assertAlmostEqual(primary["spearman"], 1.0)
        self.assertAlmostEqual(primary["response_cosine_mean"], 1.0)

    def test_confidence_intervals_are_produced_per_requested_unit(self):
        scored = score_effect_subsets(self.frame(), "pred", "obs", 50, 0,
                                      unit_columns=("perturbation", "replicate"))
        for column in ("spearman_perturbation_lo", "spearman_replicate_hi",
                       "pearson_perturbation_lo", "pearson_replicate_hi"):
            self.assertIn(column, scored.columns)

    def test_a_table_without_replicates_still_scores(self):
        frame = self.frame().drop(columns=["replicate"])
        scored = score_effect_subsets(frame, "pred", "obs", 50, 0)
        self.assertAlmostEqual(
            scored[scored["effect_subset"] == "primary"].iloc[0]["spearman"], 1.0)
