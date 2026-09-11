import unittest

import numpy as np
from sklearn.linear_model import Ridge

from tools.chromatin_multivelo import gene_columns, median_pearson, pairing_of, RIDGE_ALPHA


class GeneColumnsTests(unittest.TestCase):
    def test_kinetic_keeps_the_marked_output_genes(self):
        columns = np.array([2, 5, 9])
        kinetic = np.array([True, False, True])
        np.testing.assert_array_equal(gene_columns(columns, kinetic, "kinetic"), [2, 9])
        np.testing.assert_array_equal(gene_columns(columns, kinetic, "output"), columns)

    def test_unknown_panel_is_rejected(self):
        with self.assertRaises(ValueError):
            gene_columns(np.array([0]), np.array([True]), "hvg")


class PairingTests(unittest.TestCase):
    def test_rna_and_wnn_are_paired_atac_is_not(self):
        self.assertEqual(pairing_of("rna"), "paired_multiome")
        self.assertEqual(pairing_of("wnn_vs_Ms"), "paired_multiome")
        self.assertEqual(pairing_of("atac"), "unpaired_atac_graph")
        self.assertEqual(pairing_of("none"), "cell_matched")


class MedianPearsonTests(unittest.TestCase):
    def test_identical_columns_are_one(self):
        values = np.arange(12, dtype=np.float32).reshape(6, 2)
        self.assertAlmostEqual(median_pearson(values, values), 1.0, places=5)

    def test_a_constant_column_does_not_poison_the_median(self):
        predicted = np.column_stack([np.arange(8, dtype=np.float32), np.ones(8)])
        observed = np.column_stack([np.arange(8, dtype=np.float32), np.ones(8)])
        self.assertAlmostEqual(median_pearson(predicted, observed), 1.0)

    def test_ridge_recovers_a_linear_map(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(80, 4)).astype(np.float32)
        y = x @ np.array([0.5, -1.0, 0.2, 0.8], dtype=np.float32)[:, None]
        y = np.tile(y, (1, 3)) + 0.01 * rng.normal(size=(80, 3)).astype(np.float32)
        fitted = Ridge(alpha=RIDGE_ALPHA).fit(x[:60], y[:60]).predict(x[60:])
        self.assertGreater(median_pearson(fitted.astype(np.float32), y[60:]), 0.9)
