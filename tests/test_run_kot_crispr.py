import tempfile
import unittest
from pathlib import Path

from run_kot_crispr import discover_seed_checkpoints, observational_output_path


class OutputPathTests(unittest.TestCase):
    def test_custom_name_cannot_overwrite_predictions(self):
        out = Path("/tmp/custom.csv")
        self.assertEqual(
            observational_output_path(out), Path("/tmp/custom_observational.csv")
        )

    def test_conventional_name_is_preserved(self):
        out = Path("/tmp/crispr_predictions_best_align.csv")
        self.assertEqual(
            observational_output_path(out),
            Path("/tmp/crispr_observational_best_align.csv"),
        )


class CheckpointDiscoveryTests(unittest.TestCase):
    def make_checkpoint(self, root: Path, model: str, dataset: str, seed: int) -> Path:
        path = root / model / dataset / f"seed_{seed}" / "checkpoint_best_align.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return path

    def test_dataset_and_model_filters_are_applied_before_ambiguity_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wanted = self.make_checkpoint(root, "kot_nodyn", "papalexi_nt_only", 42)
            self.make_checkpoint(root, "kot", "papalexi_retained", 123)
            found, dataset, model = discover_seed_checkpoints(
                root, "best_align", dataset="papalexi_nt_only", model="kot_nodyn"
            )
            self.assertEqual(found, [(wanted, 42)])
            self.assertEqual(dataset, "papalexi_nt_only")
            self.assertEqual(model, "kot_nodyn")

    def test_model_is_inferred_from_checkpoint_directory_not_root_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wanted = self.make_checkpoint(
                root, "kot_fixedkappa", "papalexi_nt_only", 6
            )
            found, dataset, model = discover_seed_checkpoints(root, "best_align")
            self.assertEqual(found, [(wanted, 6)])
            self.assertEqual(dataset, "papalexi_nt_only")
            self.assertEqual(model, "kot_fixedkappa")


if __name__ == "__main__":
    unittest.main()
