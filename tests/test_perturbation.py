import unittest

import numpy as np
import pandas as pd
import torch

from src.evaluation.perturbation import (
    build_scoring_set,
    group_deltas,
    column_order_warning,
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


if __name__ == "__main__":
    unittest.main()
