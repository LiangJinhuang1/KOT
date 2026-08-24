import tempfile
import unittest
from pathlib import Path

import anndata as ad
import numpy as np

from src.data.io import is_valid_h5ad, write_h5ad_atomic


class AtomicH5adTests(unittest.TestCase):
    def test_atomic_write_creates_valid_file_and_parent(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "nested" / "example.h5ad"
            write_h5ad_atomic(ad.AnnData(X=np.eye(2, dtype=np.float32)), output_path)

            self.assertTrue(is_valid_h5ad(output_path))
            self.assertEqual(list(output_path.parent.glob("*.tmp-*")), [])


if __name__ == "__main__":
    unittest.main()
