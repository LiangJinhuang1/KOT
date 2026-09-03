import unittest

import numpy as np
from scipy import sparse

from src.data.nt_only import lift_square_sparse, stitch_neighbor_graphs


class NeighborStitchTests(unittest.TestCase):
    """NT rows must not keep knockout neighbours from the shared kNN graph."""

    def test_a_control_row_has_no_edge_to_a_knockout(self):
        fit = np.array([True, True, False, False])
        control = sparse.csr_matrix(np.array([[0.0, 3.0], [3.0, 0.0]]))
        query = sparse.csr_matrix(np.array([
            [0.0, 1.0, 9.0, 0.0],
            [1.0, 0.0, 0.0, 8.0],
            [9.0, 0.0, 0.0, 2.0],
            [0.0, 8.0, 2.0, 0.0],
        ]))
        stitched = stitch_neighbor_graphs(control, query, fit).toarray()
        np.testing.assert_allclose(stitched[0], [0.0, 3.0, 0.0, 0.0])
        np.testing.assert_allclose(stitched[1], [3.0, 0.0, 0.0, 0.0])
        self.assertEqual(stitched[0, 2], 0.0)
        self.assertEqual(stitched[1, 3], 0.0)

    def test_a_knockout_row_keeps_knockout_and_control_neighbours(self):
        fit = np.array([True, True, False, False])
        control = sparse.csr_matrix(np.array([[0.0, 3.0], [3.0, 0.0]]))
        query = sparse.csr_matrix(np.array([
            [0.0, 1.0, 9.0, 0.0],
            [1.0, 0.0, 0.0, 8.0],
            [9.0, 0.0, 0.0, 2.0],
            [0.0, 8.0, 2.0, 0.0],
        ]))
        stitched = stitch_neighbor_graphs(control, query, fit).toarray()
        np.testing.assert_allclose(stitched[2], [9.0, 0.0, 0.0, 2.0])
        np.testing.assert_allclose(stitched[3], [0.0, 8.0, 2.0, 0.0])

    def test_lift_places_the_subgraph_on_the_parent_indices(self):
        sub = sparse.csr_matrix(np.array([[0.0, 4.0], [4.0, 0.0]]))
        lifted = lift_square_sparse(sub, np.array([0, 3]), n=4).toarray()
        expected = np.zeros((4, 4))
        expected[0, 3] = 4.0
        expected[3, 0] = 4.0
        np.testing.assert_allclose(lifted, expected)


if __name__ == "__main__":
    unittest.main()
