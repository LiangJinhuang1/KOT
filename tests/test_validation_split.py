import unittest

import numpy as np

from src.data.splits import cell_hash_fractions, split_digest, validation_mask

SEED = 20260825
BARCODES = [f"AAACCTG{i:07d}-1" for i in range(20000)]


class ValidationMaskTests(unittest.TestCase):
    """The split has to be the SAME cells everywhere, or nothing is comparable."""

    def test_fraction_zero_selects_nothing(self):
        mask = validation_mask(BARCODES, 0.0, SEED)
        self.assertEqual(mask.sum(), 0)
        self.assertEqual(mask.shape, (len(BARCODES),))

    def test_realized_fraction_is_close_to_requested(self):
        for fraction in (0.1, 0.2):
            mask = validation_mask(BARCODES, fraction, SEED)
            realized = mask.mean()
            # Binomial sd at n=20000 is <0.3%; 1% is a loose bound that still fails
            # loudly if the hash is skewed.
            self.assertAlmostEqual(realized, fraction, delta=0.01)

    def test_repeated_calls_agree(self):
        first = validation_mask(BARCODES, 0.2, SEED)
        second = validation_mask(BARCODES, 0.2, SEED)
        np.testing.assert_array_equal(first, second)

    def test_assignment_does_not_depend_on_cell_order(self):
        # A run that loads cells in a different order must still hold out the same
        # barcodes, otherwise two arms of the same sweep score different cells.
        order = np.random.default_rng(0).permutation(len(BARCODES))
        shuffled = [BARCODES[i] for i in order]
        base = validation_mask(BARCODES, 0.2, SEED)
        np.testing.assert_array_equal(validation_mask(shuffled, 0.2, SEED), base[order])

    def test_assignment_survives_upstream_cell_filtering(self):
        # A new preprocessing cache that drops a few cells must not move OTHER cells
        # across the split -- the reason membership is a threshold on a per-cell hash
        # rather than the first k of a shuffle.
        keep = np.ones(len(BARCODES), dtype=bool)
        keep[::7] = False
        subset = [name for name, k in zip(BARCODES, keep) if k]
        base = validation_mask(BARCODES, 0.2, SEED)
        np.testing.assert_array_equal(validation_mask(subset, 0.2, SEED), base[keep])

    def test_nested_fractions(self):
        # 10% must be a subset of 20%: both threshold the same per-cell value, so
        # changing the fraction never swaps a cell from validation back into training.
        small = validation_mask(BARCODES, 0.1, SEED)
        large = validation_mask(BARCODES, 0.2, SEED)
        self.assertTrue(bool((~large[small]).sum() == 0))

    def test_a_different_split_seed_selects_a_different_set(self):
        base = validation_mask(BARCODES, 0.2, SEED)
        other = validation_mask(BARCODES, 0.2, SEED + 1)
        overlap = (base & other).sum() / base.sum()
        self.assertLess(overlap, 0.35)   # ~0.2 if the two draws are independent

    def test_hash_values_are_in_the_unit_interval(self):
        fractions = cell_hash_fractions(BARCODES[:1000], SEED)
        self.assertTrue(bool((fractions >= 0.0).all() and (fractions < 1.0).all()))

    def test_fraction_outside_the_unit_interval_is_rejected(self):
        with self.assertRaises(ValueError):
            validation_mask(BARCODES, 1.0, SEED)
        with self.assertRaises(ValueError):
            validation_mask(BARCODES, -0.1, SEED)


class SplitDigestTests(unittest.TestCase):
    """The digest is how two runs prove they scored the same cells."""

    def test_same_selection_same_digest_regardless_of_order(self):
        order = np.random.default_rng(1).permutation(len(BARCODES))
        shuffled = [BARCODES[i] for i in order]
        mask = validation_mask(BARCODES, 0.2, SEED)
        self.assertEqual(
            split_digest(BARCODES, mask),
            split_digest(shuffled, mask[order]),
        )

    def test_different_selection_different_digest(self):
        mask = validation_mask(BARCODES, 0.2, SEED)
        other = validation_mask(BARCODES, 0.2, SEED + 1)
        self.assertNotEqual(split_digest(BARCODES, mask), split_digest(BARCODES, other))


if __name__ == "__main__":
    unittest.main()
