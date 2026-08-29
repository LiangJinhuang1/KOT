import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.kot import velocity_row_permutation


class VelocityRowPermutationTests(unittest.TestCase):
    """The control must destroy the cell-to-velocity correspondence and nothing else.

    If it also changed the vectors or their norms, a gap against the real run could be
    a scale effect rather than the information in the correspondence.
    """

    def setUp(self):
        self.velocity = np.arange(60, dtype=np.float32).reshape(12, 5)
        self.confidence = np.linspace(0.1, 1.0, 12, dtype=np.float32)

    def test_rows_are_preserved_as_whole_vectors(self):
        out = self.velocity[velocity_row_permutation(12, seed=42)]
        before = {tuple(row) for row in self.velocity}
        after = {tuple(row) for row in out}
        self.assertEqual(before, after)

    def test_per_cell_norms_are_preserved_so_the_gauge_is_unchanged(self):
        out = self.velocity[velocity_row_permutation(12, seed=42)]
        self.assertTrue(np.allclose(np.sort(np.linalg.norm(self.velocity, axis=1)),
                                    np.sort(np.linalg.norm(out, axis=1))))
        self.assertAlmostEqual(float(np.median(np.linalg.norm(self.velocity, axis=1))),
                               float(np.median(np.linalg.norm(out, axis=1))), places=5)

    def test_assignment_actually_changes(self):
        out = self.velocity[velocity_row_permutation(12, seed=42)]
        self.assertFalse(np.array_equal(out, self.velocity))

    def test_same_seed_reproduces_and_different_seeds_differ(self):
        a = velocity_row_permutation(12, seed=7)
        b = velocity_row_permutation(12, seed=7)
        c = velocity_row_permutation(12, seed=8)
        self.assertTrue(np.array_equal(a, b))
        self.assertFalse(np.array_equal(a, c))

    def test_shape_and_dtype_survive(self):
        out = self.velocity[velocity_row_permutation(12, seed=42)]
        self.assertEqual(out.shape, self.velocity.shape)
        self.assertEqual(out.dtype, self.velocity.dtype)

    def test_confidence_follows_its_own_velocity(self):
        """A cell's confidence grades its own velocity fit, so the two move together.

        Applying the permutation to the velocity alone would leave every cell weighting
        someone else's residual by its own confidence, so the shuffled arm would differ
        from the real one in two ways instead of the one the control is measuring.
        """
        perm = velocity_row_permutation(12, seed=42)
        shuffled_velocity = self.velocity[perm]
        shuffled_confidence = self.confidence[perm]
        for cell, source in enumerate(perm):
            self.assertTrue(np.array_equal(shuffled_velocity[cell], self.velocity[source]))
            self.assertEqual(shuffled_confidence[cell], self.confidence[source])


if __name__ == "__main__":
    unittest.main()
