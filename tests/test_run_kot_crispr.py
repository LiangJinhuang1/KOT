import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from run_kot_crispr import (
    discover_seed_checkpoints,
    load_task_b_rna,
    observational_output_path,
    perturbation_predictions,
    score_ablation_correlation_differences,
)


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


class PerturbationPredictionTests(unittest.TestCase):
    class IdentityModel:
        @staticmethod
        def phi(values):
            return values

    def test_saves_rna_shift_norm_and_jacobian_linearization_error(self):
        features = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 2.0],
            [3.0, 2.0],
        ])
        predictions = perturbation_predictions(
            self.IdentityModel(),
            features,
            pd.Series(["NT", "NT", "KO", "KO"]),
            pd.Series(["rep1"] * 4),
            ["A", "B"],
            min_cells=2,
            phi_values=features.numpy(),
            phi_slope=np.ones(2),
        )
        self.assertTrue(np.allclose(predictions["delta_r_norm"], np.sqrt(8.0)))
        self.assertTrue(np.allclose(
            predictions["jacobian_linearization_error"], 0.0
        ))


class AblationDifferenceTests(unittest.TestCase):
    def test_scores_both_correlations_from_the_same_effects(self):
        rows = []
        for arm, sign in (("full", 1.0), ("shuffleVel", -1.0)):
            for perturbation in ("KO1", "KO2", "KO3"):
                for protein, observed in zip(("A", "B", "C"), (1.0, 2.0, 4.0)):
                    rows.append({
                        "arm": arm,
                        "effect_set": "primary",
                        "seed": 1,
                        "perturbation": perturbation,
                        "replicate": "rep1",
                        "protein": protein,
                        "delta_protein": observed,
                        "delta_p_phi": sign * observed,
                    })
        result = score_ablation_correlation_differences(
            pd.DataFrame(rows), ["delta_p_phi"], n_boot=50, seed=9
        )
        self.assertEqual(set(result["metric"]), {"spearman", "pearson"})
        self.assertTrue(np.allclose(result["delta_correlation"], 2.0))
        self.assertTrue(np.allclose(result["delta_boot_mean"], 2.0))
        self.assertTrue(np.allclose(result["delta_boot_lo"], 2.0))
        self.assertTrue(np.allclose(result["delta_boot_hi"], 2.0))


class TaskBInputTests(unittest.TestCase):
    def test_csv_profiles_are_averaged_per_perturbation(self):
        path = Path("/tmp/task_b_predicted_rna.csv")
        pd.DataFrame({
            "perturbation": ["KO1", "KO1", "KO2"],
            "feature_0": [1.0, 3.0, 5.0],
            "feature_1": [2.0, 4.0, 6.0],
        }).to_csv(path, index=False)
        loaded = load_task_b_rna(path, "perturbation", "X_predicted")
        self.assertEqual(loaded["perturbation"].tolist(), ["KO1", "KO2"])
        self.assertTrue(np.allclose(
            loaded.loc[0, ["feature_0", "feature_1"]].to_numpy(dtype=float),
            [2.0, 3.0],
        ))


if __name__ == "__main__":
    unittest.main()
