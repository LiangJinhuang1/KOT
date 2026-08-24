import sys
import unittest

import numpy as np

import kot
from src.utils.arrays import matrix_from_adata


class PackageTests(unittest.TestCase):
    def test_version_is_exposed(self):
        self.assertEqual(kot.__version__, "0.1.0")

    def test_import_does_not_load_training_stack(self):
        self.assertNotIn("src.training.runner", sys.modules)

    def test_matrix_from_adata_reports_missing_layer(self):
        class MinimalAnnData:
            X = np.ones((2, 2))
            layers = {}

        with self.assertRaisesRegex(ValueError, "missing"):
            matrix_from_adata(MinimalAnnData(), "missing", "RNA")


if __name__ == "__main__":
    unittest.main()
