import unittest

import numpy as np
import pandas as pd
import torch

from src.evaluation.perturbation import (
    affine_calibration,
    bootstrap_spearman_gap,
    build_scoring_set,
    calibrate_direct,
    group_deltas,
    column_order_warning,
    finite_pairs,
    knn_indices,
    predict_protein,
    score,
)


class ScoringSetTests(unittest.TestCase):
    def test_self_pairs_are_dropped_and_trans_pairs_kept(self):
        observed = pd.DataFrame([
            {"knockout": "CD86", "adt": "CD86", "delta": -0.4, "cohens_d": -0.5,
             "qval": 1e-10},
            {"knockout": "CMTM6", "adt": "PDL1", "delta": -0.4, "cohens_d": -0.5,
             "qval": 1e-10},
            {"knockout": "ATF2", "adt": "PDL1", "delta": 0.01, "cohens_d": 0.01,
             "qval": 0.9},
        ])
        adt_gene_map = {"CD86": "CD86", "PDL1": "CD274"}
        kept = build_scoring_set(observed, adt_gene_map, q_max=0.05, d_min=0.1)
        self.assertEqual(list(zip(kept["knockout"], kept["adt"])), [("CMTM6", "PDL1")])


class FeatureSpaceTests(unittest.TestCase):
    def test_matching_adt_columns_are_a_direct_map(self):
        rng = np.random.default_rng(0)
        observed = rng.normal(size=(40, 4))
        self.assertIsNone(column_order_warning(observed, observed, ["a", "b", "c", "d"][:observed.shape[1]]))

    def test_latent_embeddings_are_not_a_direct_map(self):
        rng = np.random.default_rng(0)
        observed = rng.normal(size=(40, 4))
        latent = rng.normal(size=(40, 8))
        self.assertIsNone(column_order_warning(latent, observed, ["a", "b", "c", "d"][:observed.shape[1]]))


class ControlNeighbourTests(unittest.TestCase):
    def test_knn_never_retrieves_knockout_cells(self):
        # Two well-separated clusters: NT at origin, KO at 10. A KO query would
        # retrieve other KO cells if they were allowed.
        rna = np.array([[0.0, 0.0], [0.1, 0.0], [10.0, 0.0], [10.1, 0.0]], dtype=np.float32)
        protein = rna.copy()
        allowed = np.array([True, True, False, False])
        idx = knn_indices(rna, protein, k=1,
                          allowed_neighbors=allowed, device=torch.device("cpu"))
        self.assertTrue(np.all(np.isin(idx.ravel(), [0, 1])))

    def test_direct_is_chosen_by_the_caller_not_inferred(self):
        """The dispatch must follow the model's configuration. Inferring it from the data
        let a normalisation quirk in one protein silently swap the estimator."""
        rng = np.random.default_rng(0)
        observed = rng.normal(size=(20, 4)).astype(np.float32)
        control = np.zeros(20, dtype=bool)
        control[:10] = True
        pred, name = predict_protein(observed, observed, observed, control, k=3,
                                     device=torch.device("cpu"), use_direct=True)
        self.assertEqual(name, "direct")
        np.testing.assert_array_equal(pred, observed)
        # Same inputs, opposite instruction: the data cannot override the caller.
        _pred, name = predict_protein(observed, observed, observed, control, k=3,
                                      device=torch.device("cpu"), use_direct=False)
        self.assertEqual(name, "knn_control")

    def test_direct_refuses_an_embedding_that_is_not_in_adt_coordinates(self):
        rng = np.random.default_rng(0)
        observed = rng.normal(size=(20, 4)).astype(np.float32)
        latent = rng.normal(size=(20, 8)).astype(np.float32)
        control = np.zeros(20, dtype=bool)
        control[:10] = True
        with self.assertRaises(ValueError):
            predict_protein(latent, latent, observed, control, k=3,
                            device=torch.device("cpu"), use_direct=True)

    def test_predict_protein_uses_knn_for_latent(self):
        rng = np.random.default_rng(0)
        observed = rng.normal(size=(20, 4)).astype(np.float32)
        latent = rng.normal(size=(20, 8)).astype(np.float32)
        control = np.zeros(20, dtype=bool)
        control[:10] = True
        pred, name = predict_protein(latent, latent, observed, control, k=3,
                                     device=torch.device("cpu"), use_direct=False)
        self.assertEqual(name, "knn_control")
        self.assertEqual(pred.shape, observed.shape)


class GroupDeltaTests(unittest.TestCase):
    def test_control_mean_is_the_subtrahend(self):
        values = np.array([[0.0], [0.0], [2.0], [2.0]], dtype=np.float32)
        groups = pd.Series(["NT", "NT", "KO", "KO"])
        table = group_deltas(values, ["PDL1"], groups, control_label="NT", min_cells=2)
        self.assertEqual(len(table), 1)
        self.assertAlmostEqual(table["delta"].iloc[0], 2.0)

    def test_score_sign_agreement(self):
        result = score(np.array([1.0, -1.0, 2.0]), np.array([0.5, -0.2, 3.0]))
        self.assertEqual(result["sign_acc"], 1.0)
        self.assertEqual(result["n_pairs"], 3)


class CalibrationTests(unittest.TestCase):
    """The direct predictor trains in CLR units; the benchmark scores in rna_size units."""

    def test_a_known_per_protein_scale_is_recovered(self):
        rng = np.random.default_rng(0)
        predicted = rng.normal(size=(200, 3))
        scale, offset = np.array([2.0, 0.5, -1.5]), np.array([1.0, -3.0, 0.25])
        observed = predicted * scale + offset
        a, b = affine_calibration(predicted, observed, np.arange(200))
        np.testing.assert_allclose(a, scale, atol=1e-8)
        np.testing.assert_allclose(b, offset, atol=1e-8)

    def test_a_constant_column_is_left_alone_rather_than_zeroed(self):
        predicted = np.column_stack([np.arange(20.0), np.ones(20)])
        observed = np.column_stack([np.arange(20.0) * 3, np.arange(20.0)])
        a, _ = affine_calibration(predicted, observed, np.arange(20))
        self.assertAlmostEqual(a[0], 3.0)
        self.assertEqual(a[1], 1.0)

    def test_calibration_never_looks_at_knockout_cells(self):
        """Fitted on controls only, so a knockout shift cannot leak into the scale."""
        rng = np.random.default_rng(1)
        predicted = rng.normal(size=(100, 2))
        observed = predicted * np.array([4.0, 0.25])
        control = np.zeros(100, dtype=bool)
        control[:50] = True
        _, scale_clean = calibrate_direct(predicted, observed, control)
        moved = observed.copy()
        moved[50:] += 17.0
        _, scale_moved = calibrate_direct(predicted, moved, control)
        np.testing.assert_allclose(scale_clean, scale_moved, atol=1e-6)

    def test_calibration_cannot_manufacture_a_group_difference(self):
        """group_deltas subtracts the control mean, so only the per-protein scale acts."""
        rng = np.random.default_rng(2)
        predicted = rng.normal(size=(80, 1))
        observed = predicted * 5.0 + 2.0
        control = np.zeros(80, dtype=bool)
        control[:40] = True
        groups = pd.Series(["NT"] * 40 + ["KO"] * 40)
        calibrated, _ = calibrate_direct(predicted, observed, control)
        raw = group_deltas(predicted, ["P"], groups, "NT", 5)["delta"].iloc[0]
        cal = group_deltas(calibrated, ["P"], groups, "NT", 5)["delta"].iloc[0]
        self.assertAlmostEqual(cal, raw * 5.0, places=4)


class NonFiniteTests(unittest.TestCase):
    def test_score_and_bootstrap_drop_the_same_pairs(self):
        """The bootstrap used to keep NaN pairs, so nanmean averaged a biased subset."""
        pred = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
        ref = np.array([1.0, 1.5, 3.0, 3.5, 5.5, 6.5])
        obs = np.array([1.0, 2.5, 3.0, 3.5, 5.0, 7.0])
        self.assertEqual(score(pred, obs)["n_pairs"], 5)
        self.assertEqual(int(finite_pairs(pred, ref, obs).sum()), 5)
        gap, lo, hi = bootstrap_spearman_gap(pred, ref, obs, n_boot=200, seed=0)
        self.assertTrue(np.isfinite([gap, lo, hi]).all())

    def test_knn_never_uses_a_cell_with_no_protein_embedding(self):
        """holdout_second leaves held-out cells NaN; they are not neighbours."""
        rna = np.array([[0.0], [0.1], [5.0], [5.1]], dtype=np.float32)
        protein = np.array([[0.0], [0.1], [np.nan], [np.nan]], dtype=np.float32)
        observed = np.array([[1.0], [2.0], [99.0], [99.0]], dtype=np.float32)
        control = np.array([True, True, False, False])
        pred, name = predict_protein(rna, protein, observed, control, k=1,
                                     device=torch.device("cpu"), use_direct=False)
        self.assertEqual(name, "knn_control")
        self.assertTrue(np.isfinite(pred).all())
        self.assertTrue((pred < 90).all())


if __name__ == "__main__":
    unittest.main()
