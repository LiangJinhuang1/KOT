import unittest

import numpy as np
import pandas as pd
import torch

from src.evaluation.perturbation import (
    positive_scale,
    replicate_effects,
    sufficient_knockouts,
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


class SufficiencyTests(unittest.TestCase):
    """The frozen design threshold, shared by the response table and the scoring set."""

    def design(self, per_knockout: dict) -> tuple[pd.Series, pd.Series]:
        genes, replicates = ["NT"] * 60, ["rep1", "rep2", "rep3"] * 20
        for knockout, counts in per_knockout.items():
            for replicate, n in counts.items():
                genes += [knockout] * n
                replicates += [replicate] * n
        return pd.Series(genes), pd.Series(replicates)

    def test_needs_the_minimum_in_each_replicate_not_in_total(self):
        """A knockout concentrated in one batch has ONE effect estimate, not two.

        The looser reading -- enough cells overall, spread over enough replicates --
        admits 200+2 cells, whose second replicate cannot carry an effect or contribute
        a replicate correlation.
        """
        groups, replicates = self.design({
            "SPREAD": {"rep1": 25, "rep2": 25},
            "LOPSIDED": {"rep1": 200, "rep2": 2},
        })
        usable = sufficient_knockouts(groups, replicates, "NT", 20, 2)
        self.assertIn("SPREAD", usable)
        self.assertNotIn("LOPSIDED", usable)

    def test_a_third_thin_replicate_does_not_disqualify(self):
        groups, replicates = self.design({"OK": {"rep1": 30, "rep2": 30, "rep3": 3}})
        self.assertEqual(sufficient_knockouts(groups, replicates, "NT", 20, 2), ["OK"])

    def test_the_control_is_never_returned_as_a_knockout(self):
        groups, replicates = self.design({"A": {"rep1": 30, "rep2": 30}})
        self.assertNotIn("NT", sufficient_knockouts(groups, replicates, "NT", 20, 2))


class ReplicateEffectTests(unittest.TestCase):
    ADT = ["CD86", "PDL1"]
    MAP = {"CD86": "CD86", "PDL1": "CD274"}

    def frame(self):
        groups = pd.Series(["NT"] * 8 + ["KO"] * 8)
        replicates = pd.Series(["rep1"] * 4 + ["rep2"] * 4 + ["rep1"] * 4 + ["rep2"] * 4)
        # rep2 carries a +10 batch shift in BOTH arms, so a within-replicate contrast
        # must cancel it and a pooled-control contrast would not.
        protein = np.zeros((16, 2), dtype=float)
        protein[4:8] += 10.0
        protein[12:16] += 10.0
        protein[8:16, 0] += 2.0          # KO raises CD86 by 2 in both replicates
        rna = pd.DataFrame({"CD86": np.zeros(16), "CD274": np.zeros(16)},
                           index=range(16))
        return protein, groups, replicates, rna

    def test_the_control_is_taken_within_the_replicate(self):
        protein, groups, replicates, rna = self.frame()
        effects = replicate_effects(protein, self.ADT, rna, self.MAP, groups, replicates, "NT")
        cd86 = effects[effects["protein"] == "CD86"]
        self.assertEqual(len(cd86), 2)
        np.testing.assert_allclose(cd86["delta_protein"].to_numpy(), [2.0, 2.0])

    def test_cell_counts_and_standard_error_are_reported(self):
        protein, groups, replicates, rna = self.frame()
        effects = replicate_effects(protein, self.ADT, rna, self.MAP, groups, replicates, "NT")
        self.assertTrue((effects["n_ko_cells"] == 4).all())
        self.assertTrue((effects["n_nt_cells"] == 4).all())
        # Both arms are constant within a replicate here, so the Welch SE is exactly zero.
        np.testing.assert_allclose(effects["protein_effect_se"].to_numpy(), 0.0, atol=1e-12)

    def test_cognate_transcript_effect_follows_the_protein_to_gene_map(self):
        protein, groups, replicates, rna = self.frame()
        rna["CD274"] = [0.0] * 8 + [5.0] * 8
        effects = replicate_effects(protein, self.ADT, rna, self.MAP, groups, replicates, "NT")
        pdl1 = effects[effects["protein"] == "PDL1"]
        np.testing.assert_allclose(pdl1["delta_cognate_rna"].to_numpy(), [5.0, 5.0])
        cd86 = effects[effects["protein"] == "CD86"]
        np.testing.assert_allclose(cd86["delta_cognate_rna"].to_numpy(), [0.0, 0.0])


class PositiveScaleTests(unittest.TestCase):
    """A units correction must not be able to invert a prediction.

    The signed least-squares slope could, and an ablation exploited it: the
    reversed-velocity arm predicts protein backwards, the slope flipped it back, and its
    Spearman went from -0.12 to +0.42 and beat the unablated model.
    """

    def test_a_backwards_prediction_is_not_flipped_the_right_way_round(self):
        rows = np.arange(200)
        observed = np.linspace(-1.0, 1.0, 200).reshape(-1, 1)
        backwards = -observed
        scale = positive_scale(backwards, observed, rows)
        self.assertGreater(scale[0], 0.0)
        corrected = backwards * scale
        # Still anti-correlated with the truth, as an inverted prediction should be.
        self.assertLess(float(np.corrcoef(corrected.ravel(), observed.ravel())[0, 1]), 0)

    def test_a_pure_scale_difference_is_corrected(self):
        rows = np.arange(200)
        observed = np.random.default_rng(0).normal(size=(200, 3))
        scale = positive_scale(observed * 0.25, observed, rows)
        np.testing.assert_allclose(scale, 4.0, rtol=1e-6)

    def test_a_constant_column_keeps_scale_one_rather_than_exploding(self):
        rows = np.arange(50)
        observed = np.random.default_rng(1).normal(size=(50, 2))
        predicted = np.column_stack([observed[:, 0], np.zeros(50)])
        scale = positive_scale(predicted, observed, rows)
        self.assertAlmostEqual(scale[1], 1.0)
        self.assertTrue(np.isfinite(scale).all())

    def test_a_nearly_constant_column_is_dropped_not_amplified(self):
        """The zeroVel arm: phi is not constant, only collapsed, so nothing catches it.

        sd 0.002x the observed spread means the scale is ~500x, which turns numerical
        dust into a ranked prediction. NaN drops the column from scoring instead.
        """
        rows = np.arange(200)
        observed = np.random.default_rng(2).normal(size=(200, 2))
        predicted = np.column_stack([observed[:, 0], observed[:, 1] * 0.002])
        unguarded = positive_scale(predicted, observed, rows)
        self.assertGreater(unguarded[1], 100.0)
        self.assertTrue(np.isfinite(unguarded).all())

        guarded = positive_scale(predicted, observed, rows, min_sd_ratio=0.01)
        self.assertTrue(np.isnan(guarded[1]))
        self.assertAlmostEqual(guarded[0], 1.0, places=6)

    def test_the_floor_is_off_by_default(self):
        rows = np.arange(200)
        observed = np.random.default_rng(3).normal(size=(200, 1))
        scale = positive_scale(observed * 1e-6, observed, rows)
        self.assertTrue(np.isfinite(scale).all())
