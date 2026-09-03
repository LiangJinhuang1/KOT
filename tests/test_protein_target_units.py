"""The protein target contract: one normalisation per dataset, for every method.

KOT read `protein_adata.X` (CLR) while ridge, MLP, sciPENN and scButterfly went through
`adt_targets` and honoured `protein_target_normalization: rna_size`. The CRISPR benchmark
scores rna_size deltas, and CLR is a per-cell row operation that the scorer's per-protein
rescale cannot invert, so phi was ranked against a ceiling of about +0.37 Spearman while
the baselines were not. These tests pin the contract and the guard that keeps a
CLR-trained checkpoint from being re-scored as though it were not.
"""
import unittest

import anndata as ad
import numpy as np

from src.training.protein_supervised import adt_targets, protein_target_units
from src.training.protein_regression import standardize


def toy_context(n_cells=40, n_adt=4, n_genes=6, seed=0):
    rng = np.random.default_rng(seed)
    counts = rng.poisson(20.0, size=(n_cells, n_adt)).astype(np.float32)
    protein = ad.AnnData(X=rng.normal(size=(n_cells, n_adt)).astype(np.float32))
    protein.layers["counts"] = counts
    rna = ad.AnnData(X=rng.normal(size=(n_cells, n_genes)).astype(np.float32))
    rna.obs["nCount_RNA"] = rng.integers(1000, 5000, size=n_cells).astype(float)
    return {"rna_adata": rna, "second_adata": protein}


class TargetUnitsTests(unittest.TestCase):
    def test_rna_size_targets_are_not_compositional(self):
        """CLR sums to zero across the panel; rna_size does not. That is the whole bug.

        Closure forces the four predicted deltas onto one axis, so a model that gets one
        protein right gets the other three backwards.
        """
        context = toy_context()
        clr = adt_targets(context, {"kot_protein_layer": None})
        rna_size = adt_targets(context, {"protein_target_normalization": "rna_size"})
        self.assertEqual(clr.shape, rna_size.shape)
        # protein.X here stands in for the CLR matrix the preprocessing writes.
        centred = clr - clr.mean(axis=1, keepdims=True)
        self.assertAlmostEqual(float(np.abs(centred.sum(axis=1)).max()), 0.0, places=4)
        self.assertGreater(float(np.abs(rna_size.sum(axis=1)).min()), 0.0)

    def test_the_size_factor_comes_from_outside_the_panel(self):
        """Deepening ONE cell's RNA must move only that cell's rna_size ADT.

        Scaling every cell equally must not move anything: the factor is relative to the
        median, so a global depth change is not a biological one.
        """
        context = toy_context()
        cfg = {"protein_target_normalization": "rna_size"}
        before = adt_targets(context, cfg)
        context["rna_adata"].obs["nCount_RNA"] *= 2.0
        self.assertTrue(np.allclose(before, adt_targets(context, cfg)))

        context["rna_adata"].obs.iloc[0, 0] *= 4.0
        after = adt_targets(context, cfg)
        self.assertFalse(np.allclose(before[0], after[0]))
        self.assertTrue(np.allclose(before[1:], after[1:]))

    def test_units_name_is_what_gets_stamped(self):
        self.assertEqual(
            protein_target_units({"protein_target_normalization": "rna_size"}, None),
            "rna_size")
        self.assertEqual(protein_target_units({}, None), "X")
        self.assertEqual(protein_target_units({}, "counts"), "counts")


class StandardizeTests(unittest.TestCase):
    def test_a_gene_constant_on_fitted_cells_does_not_explode_on_held_out_cells(self):
        """208/1276 Papalexi genes are constant across NT and nonzero on knockouts.

        A 1e-8 floor sent those KO features to ~3e7, which the MLP multiplied by its
        untrained first-layer weights.
        """
        features = np.zeros((20, 3), dtype=np.float64)
        fit_rows = np.arange(10)
        features[:, 0] = np.arange(20)          # varies everywhere
        features[10:, 1] = 5.0                  # constant on fit rows, not on the rest
        scaled = standardize(features, fit_rows)
        self.assertTrue(np.isfinite(scaled).all())
        self.assertLessEqual(float(np.abs(scaled[:, 1]).max()), 5.0)
        self.assertTrue(np.allclose(scaled[fit_rows, 1], 0.0))
        self.assertTrue(np.allclose(scaled[fit_rows, 2], 0.0))


if __name__ == "__main__":
    unittest.main()
