import multiprocessing
import tempfile
import unittest
from pathlib import Path

import anndata as ad
import numpy as np

from src.evaluation import foscttm
from src.evaluation.foscttm import calc_domainAveraged_FOSCTTM, calc_frac_idx, save_prediction_h5ad


def write_prediction_to_dir(directory: str) -> None:
    foscttm.PREDICTIONS_DIR = Path(directory)
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    save_prediction_h5ad([x, x.copy()], [0.0, 0.0], 0.0, "ds_seed_1", "kot", "protein", ["c0", "c1"])


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


class PredictionH5adTests(unittest.TestCase):
    def test_two_writers_to_the_same_path_do_not_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = multiprocessing.get_context("spawn")
            workers = [ctx.Process(target=write_prediction_to_dir, args=(tmp,)) for _ in range(2)]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join(timeout=30)
            for worker in workers:
                self.assertEqual(worker.exitcode, 0)
            path = Path(tmp) / "kot_ds_seed_1.h5ad"
            self.assertTrue(path.is_file())
            adata = ad.read_h5ad(path)
            self.assertEqual(adata.n_obs, 2)


if __name__ == "__main__":
    unittest.main()
