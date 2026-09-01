"""The MaxFuse baseline's inputs, which are where a silent mis-alignment would hide.

MaxFuse takes the two `shared` blocks as COLUMN-MATCHED matrices: shared column j of the
RNA block must be the gene behind shared column j of the protein block. Nothing downstream
checks that, and a matching built on mis-paired columns still runs and still returns an
embedding -- it just scores a different method than the one being reported.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.training.maxfuse import (
    clamp_components,
    paired_shared_blocks,
    permuted_rows,
    shared_feature_pairs,
)


def record(adt, gene, use_for_alignment=True):
    return {"adt_name": adt, "gene_symbol": gene, "use_for_alignment": use_for_alignment}


class SharedFeaturePairsTests(unittest.TestCase):

    RNA = ["CD4", "CD8A", "MS4A1", "ACTB"]
    PROTEIN = ["CD4_ADT", "CD8a_ADT", "CD20_ADT"]

    def test_pairs_are_column_matched(self):
        records = [record("CD4_ADT", "CD4"), record("CD20_ADT", "MS4A1")]
        genes, proteins, dropped = shared_feature_pairs(records, self.RNA, self.PROTEIN)
        self.assertEqual([self.RNA[j] for j in genes], ["CD4", "MS4A1"])
        self.assertEqual([self.PROTEIN[j] for j in proteins], ["CD4_ADT", "CD20_ADT"])
        self.assertEqual(dropped, [])

    def test_alignment_flag_is_honoured(self):
        """use_for_alignment=False is a curated decision that the link is not usable;
        MaxFuse must see the same panel the Sinkhorn term does."""
        records = [record("CD4_ADT", "CD4"), record("CD8a_ADT", "CD8A", False)]
        genes, _proteins, _dropped = shared_feature_pairs(records, self.RNA, self.PROTEIN)
        self.assertEqual(len(genes), 1)

    def test_one_protein_may_contribute_several_columns(self):
        """A protein linked to two genes is two shared columns, not one: the blocks pair
        column by column, not protein by protein."""
        records = [record("CD4_ADT", "CD4"), record("CD4_ADT", "ACTB")]
        genes, proteins, _dropped = shared_feature_pairs(records, self.RNA, self.PROTEIN)
        self.assertEqual(len(genes), 2)
        self.assertEqual(list(proteins), [0, 0])

    def test_links_missing_from_either_matrix_are_reported(self):
        records = [record("CD4_ADT", "CD4"), record("CD20_ADT", "ABSENT")]
        genes, _proteins, dropped = shared_feature_pairs(records, self.RNA, self.PROTEIN)
        self.assertEqual(len(genes), 1)
        self.assertEqual(len(dropped), 1)

    def test_no_surviving_link_is_an_error(self):
        with self.assertRaises(ValueError):
            shared_feature_pairs([record("CD20_ADT", "ABSENT")], self.RNA, self.PROTEIN)


class PairedSharedBlocksTests(unittest.TestCase):

    def test_a_static_column_drops_its_partner_too(self):
        """Pruning the two blocks independently would shift one side's columns and pair
        gene i with a different protein for every column after the drop."""
        rna = np.array([[1.0, 5.0, 2.0], [2.0, 5.0, 4.0], [3.0, 5.0, 8.0]])
        protein = np.array([[1.0, 2.0, 3.0], [4.0, 1.0, 5.0], [9.0, 3.0, 7.0]])
        # gene column 1 is constant; its protein partner is not, and must go with it.
        rna_shared, protein_shared, keep = paired_shared_blocks(
            rna, protein, np.array([0, 1, 2]), np.array([0, 1, 2]))
        self.assertEqual(list(keep), [True, False, True])
        self.assertEqual(rna_shared.shape[1], protein_shared.shape[1])
        self.assertEqual(rna_shared.shape[1], 2)

    def test_blocks_come_back_z_scored(self):
        rng = np.random.default_rng(0)
        rna, protein = rng.normal(size=(50, 3)), rng.normal(size=(50, 3))
        rna_shared, protein_shared, _keep = paired_shared_blocks(
            rna, protein, np.array([0, 1, 2]), np.array([0, 1, 2]))
        for block in (rna_shared, protein_shared):
            np.testing.assert_allclose(block.mean(axis=0), 0, atol=1e-5)
            np.testing.assert_allclose(block.std(axis=0), 1, atol=1e-5)


class ComponentAndPermutationTests(unittest.TestCase):

    def test_components_are_clamped_to_the_panel(self):
        """A 32-plex ADT panel cannot supply the 30 components the CITE-seq tutorial
        assumes for a 228-plex one; MaxFuse would fail inside the SVD instead."""
        self.assertEqual(clamp_components(30, 32, "test"), 30)
        self.assertEqual(clamp_components(30, 20, "test"), 19)
        self.assertEqual(clamp_components(30, 1, "test"), 1)

    def test_permutation_round_trip_restores_order(self):
        """The second modality is fitted shuffled and returned unshuffled; if this is not
        an exact inverse, FOSCTTM scores a mis-ordered embedding."""
        values = np.arange(20).reshape(10, 2)
        permutation, inverse = permuted_rows(10, seed=7, enabled=True)
        np.testing.assert_array_equal(values[permutation][inverse], values)
        self.assertEqual(permuted_rows(10, seed=7, enabled=False), (None, None))


if __name__ == "__main__":
    unittest.main()
