import sys, unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.kot import bounded_head_saturation, constant_output_fraction


class BoundedHeadSaturationTests(unittest.TestCase):
    """alpha_max is one extreme of ~1e5 entries; these fractions describe the bulk."""

    def test_midpoint_mass_is_found(self):
        v = np.array([1.5, 1.5, 0.2, 2.9])          # bounds [0, 3] -> midpoint 1.5
        out = bounded_head_saturation(v, 0.0, 3.0, "alpha")
        self.assertAlmostEqual(out["alpha_frac_at_midpoint"], 0.5)

    def test_edges_are_counted_at_the_right_end(self):
        v = np.array([0.05, 0.05, 1.5, 2.95])
        out = bounded_head_saturation(v, 0.0, 3.0, "alpha")
        self.assertAlmostEqual(out["alpha_frac_near_floor"], 0.5)
        self.assertAlmostEqual(out["alpha_frac_near_ceiling"], 0.25)

    def test_unbounded_head_reports_nothing(self):
        self.assertEqual(bounded_head_saturation(np.arange(5.0), 0.0, None, "kappa"), {})

    def test_shape_is_flattened_so_per_cell_per_protein_works(self):
        out = bounded_head_saturation(np.full((7, 3), 1.5), 0.0, 3.0, "alpha")
        self.assertAlmostEqual(out["alpha_frac_at_midpoint"], 1.0)


class ConstantOutputTests(unittest.TestCase):
    def test_dead_columns_are_counted(self):
        v = np.array([[1.0, 5.0], [1.0, 9.0], [1.0, 7.0]])   # col 0 dead, col 1 alive
        self.assertAlmostEqual(constant_output_fraction(v), 0.5)

    def test_all_alive_is_zero(self):
        self.assertAlmostEqual(constant_output_fraction(np.arange(12.0).reshape(4, 3)), 0.0)


if __name__ == "__main__":
    unittest.main()
