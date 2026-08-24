import unittest

import numpy as np

from src.evaluation.foscttm import calc_domainAveraged_FOSCTTM, calc_frac_idx


class FoscttmTests(unittest.TestCase):
    def test_identity_alignment_scores_zero(self):
        embedding = np.array([[0.0, 0.0], [1.0, 1.0], [3.0, 2.0]], dtype=np.float32)

        self.assertEqual(calc_domainAveraged_FOSCTTM(embedding, embedding), [0.0, 0.0, 0.0])

    def test_singleton_alignment_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            calc_frac_idx(np.zeros((1, 2)), np.zeros((1, 2)))

    def test_unpaired_shapes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "same paired shape"):
            calc_frac_idx(np.zeros((2, 2)), np.zeros((3, 2)))


if __name__ == "__main__":
    unittest.main()
