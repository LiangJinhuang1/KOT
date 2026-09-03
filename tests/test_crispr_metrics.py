import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.evaluation.crispr_metrics import annotate_effect_sets, confirmed_knockouts


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


if __name__ == "__main__":
    unittest.main()
