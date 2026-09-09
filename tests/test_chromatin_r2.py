import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from scipy import sparse

from src.data.chromatin_r2 import shared_splicing_targets, load_gamma_anchors, gamma_anchor_loss
from tools.build_gamma_anchors import build_anchors, K562_SHEET, LN2


def anchor_frame():
    return pd.DataFrame({
        "gene_symbol": ["A", "B", "C"], "molecule": "RNA", "source": "fixture",
        "half_life_hours": [1.0, 2.0, 4.0], "anchor_weight": [1.0, 0.5, 1.0],
    })


class SharedRNAUnitTests(unittest.TestCase):
    def test_common_factor_preserves_flux_units_and_gene_order(self):
        u = np.array([[1., 3., 2.], [0., 0., 0.]], dtype=np.float32)
        s = np.array([[4., 0., 10.], [0., 0., 0.]], dtype=np.float32)
        adata = SimpleNamespace(layers={"unspliced": sparse.csr_matrix(u),
                                        "spliced": sparse.csr_matrix(s)})
        actual = np.expm1(shared_splicing_targets(adata, [2, 0]))
        np.testing.assert_allclose(actual[0], [1000., 500., 5000., 2000.], rtol=1e-6)
        np.testing.assert_array_equal(actual[1], 0)
        np.testing.assert_array_equal(adata.layers["unspliced"].toarray(), u)
        np.testing.assert_array_equal(adata.layers["spliced"].toarray(), s)


class GammaAnchorTests(unittest.TestCase):
    def test_panel_order_and_relative_scale_do_not_depend_on_intersection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rna.csv"
            anchor_frame().to_csv(path, index=False)
            full = load_gamma_anchors(str(path), ["C", "absent", "A", "B"])
            subset = load_gamma_anchors(str(path), ["A"])
            self.assertEqual(full["indices"], [0, 2, 3])
            self.assertEqual(full["gene_names"], ["C", "A", "B"])
            self.assertAlmostEqual(full["gamma"][1], subset["gamma"][0])
            self.assertEqual(full["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            np.testing.assert_allclose(full["gamma"], [0.25, 1.0, 0.5])

    def test_physical_conversion_and_gamma_only_gradient(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rna.csv"
            anchor_frame().to_csv(path, index=False)
            anchors = load_gamma_anchors(str(path), ["absent", "B"], hours_per_model_time=3)
            self.assertAlmostEqual(anchors["gamma"][0], 3 * LN2 / 2)
            gamma = torch.tensor([4., anchors["gamma"][0] * 2], requires_grad=True)
            loss = gamma_anchor_loss(gamma, anchors)
            self.assertAlmostEqual(float(loss), LN2 ** 2, places=6)
            loss.backward()
            self.assertEqual(float(gamma.grad[0]), 0.)
            self.assertGreater(float(gamma.grad[1]), 0.)

    def test_rejects_protein_ambiguous_symbols_and_empty_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rna.csv"
            for frame in [anchor_frame().assign(molecule="protein"),
                          anchor_frame().assign(gene_symbol=["42799", "B", "C"]),
                          anchor_frame().assign(gene_symbol=["A", "A", "C"])]:
                frame.to_csv(path, index=False)
                with self.assertRaises(ValueError):
                    load_gamma_anchors(str(path), ["A"])
            anchor_frame().to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load_gamma_anchors(str(path), ["missing"])


class AnchorBuilderTests(unittest.TestCase):
    def test_filters_spreadsheet_dates_and_disagreeing_replicates(self):
        frame = pd.DataFrame({
            "transcript": ["A", "42799", "B", "C"],
            "half_life_rep1": [2., 1., 1., 0.],
            "half_life_rep2": [2., 1., 8., 1.],
            "mean_half_life": [2., 1., 4.5, 0.5],
        })
        with np.errstate(divide="raise", invalid="raise"):
            actual = build_anchors(frame, K562_SHEET, LN2)
        self.assertEqual(actual.gene_symbol.tolist(), ["A"])
        self.assertAlmostEqual(actual.gamma_per_hour.iloc[0], LN2 / 2)

    def test_duplicate_half_lives_keep_the_rate_conversion_consistent(self):
        frame = pd.DataFrame({
            "transcript": ["A", "A"], "half_life_rep1": [1., 3.],
            "half_life_rep2": [1., 3.], "mean_half_life": [1., 3.],
        })
        actual = build_anchors(frame, K562_SHEET, LN2)
        self.assertEqual(len(actual), 1)
        self.assertAlmostEqual(actual.gamma_per_hour.iloc[0], LN2 / 2)


if __name__ == "__main__":
    unittest.main()
