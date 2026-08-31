import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.kot import anchor_subset_indices, select_anchor_entries


class AnchorSubsetIndicesTests(unittest.TestCase):
    """The anchor-count ablation must change the COUNT and nothing else.

    If a step down the ladder also re-drew which anchors are kept, a difference between
    two rungs could be the identity of the anchors rather than how many there are.
    """

    def test_ladder_is_nested_within_a_seed(self):
        big = set(anchor_subset_indices(53, 27, seed=42))
        for keep_n in (13, 5, 1):
            self.assertTrue(set(anchor_subset_indices(53, keep_n, seed=42)) <= big)

    def test_requested_count_is_what_comes_back(self):
        for keep_n in (0, 1, 5, 10):
            self.assertEqual(len(anchor_subset_indices(10, keep_n, seed=7)), keep_n)

    def test_full_rung_is_every_anchor_in_order(self):
        """subset_n = n_active must be identical to not subsetting at all, so the full
        rung can be labelled with its count instead of a null."""
        self.assertEqual(anchor_subset_indices(10, 10, seed=7), list(range(10)))

    def test_indices_are_unique_and_in_range(self):
        keep = anchor_subset_indices(53, 27, seed=3)
        self.assertEqual(len(set(keep)), len(keep))
        self.assertTrue(all(0 <= k < 53 for k in keep))

    def test_same_seed_reproduces_and_different_seeds_differ(self):
        self.assertEqual(anchor_subset_indices(53, 13, seed=5),
                         anchor_subset_indices(53, 13, seed=5))
        self.assertNotEqual(anchor_subset_indices(53, 13, seed=5),
                            anchor_subset_indices(53, 13, seed=6))

    def test_out_of_range_count_is_rejected(self):
        with self.assertRaises(ValueError):
            anchor_subset_indices(10, 11, seed=42)
        with self.assertRaises(ValueError):
            anchor_subset_indices(10, -1, seed=42)


class SelectAnchorEntriesTests(unittest.TestCase):
    """The four parallel anchor lists have to stay aligned through every filter."""

    def test_parallel_lists_stay_aligned(self):
        idx, betas, weights, sigmas = select_anchor_entries(
            [0, 2], [10, 11, 12], [0.1, 0.2, 0.3], [1.0, 2.0, 3.0], [0.01, 0.02, 0.03])
        self.assertEqual(idx, [10, 12])
        self.assertEqual(betas, [0.1, 0.3])
        self.assertEqual(weights, [1.0, 3.0])
        self.assertEqual(sigmas, [0.01, 0.03])

    def test_lists_not_supplied_per_anchor_pass_through(self):
        """A manual config may give indices without weights/sigmas; anchor_tensors then
        falls back to defaults, so a length mismatch must not be sliced."""
        idx, betas, weights, sigmas = select_anchor_entries(
            [1], [10, 11], [0.1, 0.2], [], [])
        self.assertEqual(idx, [11])
        self.assertEqual(betas, [0.2])
        self.assertEqual(weights, [])
        self.assertEqual(sigmas, [])


if __name__ == "__main__":
    unittest.main()
