"""Regression coverage for chromatin evaluation units and training-only preprocessing."""
import argparse
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy import sparse

import run_kot_chromatin as runner
from src.data.chromatin import (
    PREPROCESSING_PROTOCOL, assign_preprocessing_split, build_bmmc, chromatin_features,
    finalize_panel, fit_peak_geometry, normalize_log, preprocessing_rows,
    validate_preprocessing_split,
)
from src.data.chromatin_map import widen_projection
from src.data.chromatin_r2 import SHARED_SPLICED
from src.data.chromatin_velocity import gauge_normalize
from src.losses.chromatin_laws import RELAY


def raw_fixture():
    rng = np.random.default_rng(12)
    counts = rng.poisson(np.linspace(1, 8, 24), size=(160, 24)).astype(np.float32)
    data = ad.AnnData(sparse.csr_matrix(counts))
    data.var_names = [f"G{i}" for i in range(data.n_vars)]
    data.obs_names = [f"cell{i}" for i in range(data.n_obs)]
    data.obs["day"] = np.tile([0, 7], 80)
    data.obs["batch"] = np.tile(["a", "b"], 80)
    data.layers["counts"] = data.X.copy()
    data.layers["unspliced"] = sparse.csr_matrix(rng.poisson(2, size=counts.shape).astype(np.float32))
    data.layers["spliced"] = data.X.copy()
    data.obsm["gene_activity_counts"] = sparse.csr_matrix(rng.binomial(3, .3, size=counts.shape).astype(np.float32))
    data.obsm["gene_activity"] = normalize_log(data.obsm["gene_activity_counts"])
    data.uns["chromatin_source"] = dict(dataset="hspc", n_atac_features=24,
                                        n_rna_genes_full=24, n_overlap_activity_rna=24)
    splits = assign_preprocessing_split(data, 3, .1, .2)
    return data, splits


class TrainingPopulationTests(unittest.TestCase):
    def test_features_ignore_heldout_and_rna_side_atac(self):
        data, _ = raw_fixture()
        source = preprocessing_rows(data, "atac")
        changed = data.copy()
        counts = changed.obsm["gene_activity_counts"].toarray()
        other = np.setdiff1d(np.arange(data.n_obs), source)
        counts[other] = np.random.default_rng(9).binomial(1, .9, size=(len(other), data.n_vars))
        changed.obsm["gene_activity_counts"] = sparse.csr_matrix(counts)
        for transform in ("tfidf_lsi", "tfidf_lsi_batch", "tfidf_gene"):
            with self.subTest(transform=transform):
                original = chromatin_features(data, transform)
                updated = chromatin_features(changed, transform)
                np.testing.assert_allclose(original[source], updated[source], atol=2e-6)
                self.assertFalse(np.allclose(original[other], updated[other]))

    def test_peak_selection_and_lsi_ignore_unavailable_cells(self):
        data, _ = raw_fixture()
        source = preprocessing_rows(data, "atac")
        counts = data.obsm["gene_activity_counts"].toarray()
        changed_counts = counts.copy()
        other = np.setdiff1d(np.arange(data.n_obs), source)
        changed_counts[other] = 0
        changed_counts[other, -1] = 100
        changed = data.copy()
        n_original = fit_peak_geometry(data, sparse.csr_matrix(counts), 5, 3, 8)
        n_changed = fit_peak_geometry(changed, sparse.csr_matrix(changed_counts), 5, 3, 8)
        self.assertEqual(n_original, n_changed)
        np.testing.assert_allclose(data.obsm["lsi"][source], changed.obsm["lsi"][source], atol=2e-6)

    def test_panel_and_map_ignore_unavailable_rna_and_atac(self):
        data, splits = raw_fixture()
        changed = data.copy()
        rna_rows, atac_rows = preprocessing_rows(data, "rna"), preprocessing_rows(data, "atac")
        for name in ("counts", "spliced", "unspliced"):
            counts = changed.layers[name].toarray()
            counts[np.setdiff1d(np.arange(data.n_obs), rna_rows)] = 10000
            changed.layers[name] = sparse.csr_matrix(counts)
        changed.X = changed.layers["counts"].copy()
        activity = changed.obsm["gene_activity_counts"].toarray()
        activity[np.setdiff1d(np.arange(data.n_obs), atac_rows)] = 0
        changed.obsm["gene_activity_counts"] = sparse.csr_matrix(activity)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            first = finalize_panel(data, 8, tmp / "first.h5ad", None, "fixture", tmp / "first.csv")
            second = finalize_panel(changed, 8, tmp / "second.h5ad", None, "fixture", tmp / "second.csv")
            self.assertEqual(list(first.var_names), list(second.var_names))
            pd.testing.assert_frame_equal(pd.read_csv(tmp / "first.csv"), pd.read_csv(tmp / "second.csv"))
            np.testing.assert_allclose(first.obsm["gene_activity"][atac_rows], second.obsm["gene_activity"][atac_rows])
            restored = ad.read_h5ad(tmp / "first.h5ad")
            validate_preprocessing_split(restored, splits, 3)

    def test_bmmc_builder_refits_supplied_lsi_and_freezes_split(self):
        data, splits = raw_fixture()
        processed = ad.AnnData(sparse.hstack([
            data.layers["counts"], data.obsm["gene_activity_counts"]], format="csr"))
        processed.obs = data.obs.copy()
        processed.var_names = list(data.var_names) + [f"peak{i}" for i in range(data.n_vars)]
        processed.var["feature_types"] = ["GEX"] * data.n_vars + ["ATAC"] * data.n_vars
        processed.layers["counts"] = processed.X.copy()
        processed.obsm["ATAC_gene_activity"] = data.obsm["gene_activity_counts"].copy()
        processed.uns["ATAC_gene_activity_var_names"] = data.var_names.to_numpy()
        processed.obsm["ATAC_lsi_red"] = np.full((data.n_obs, 4), np.nan)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            velocity_path = tmp / "velocity.h5ad"
            velocity_path.touch()
            with patch("src.data.chromatin.sc.read_h5ad", side_effect=[processed, data]):
                result = build_bmmc(tmp / "processed.h5ad", velocity_path,
                                    tmp / "panel.h5ad", 8, tmp / "mapping.csv",
                                    n_lsi=5, min_peak_cells=3, seed=8, split_seed=3,
                                    splits=splits)
            self.assertEqual(result.obsm["lsi"].shape, (data.n_obs, 4))
            self.assertTrue(np.isfinite(result.obsm["lsi"]).all())
            validate_preprocessing_split(result, splits, 3)
            self.assertEqual(result.uns["chromatin_preprocessing"]["lsi_fit"], "training_atac_peak_counts")

    def test_gauge_ignores_heldout_speeds_and_preserves_relative_training_speed(self):
        velocity = np.arange(1, 6, dtype=np.float32)[:, None]
        changed = velocity.copy(); changed[2:] *= 100
        fit = np.array([True, True, False, False, False])
        normalized, gauge = gauge_normalize(velocity, fit, return_gauge=True)
        self.assertEqual(gauge, 1.5)
        np.testing.assert_allclose(normalized[:2], gauge_normalize(changed, fit)[:2])
        self.assertEqual(normalized[1, 0] / normalized[0, 0], 2)
        with self.assertRaises(ValueError):
            gauge_normalize(velocity, np.zeros(5, dtype=bool))

    def test_mutated_split_and_velocity_identity_are_rejected(self):
        data, splits = raw_fixture()
        altered = splits.copy()
        row = altered.index[altered["train_side"].eq("atac")][0]
        altered.loc[row, "train_side"] = "rna"
        with self.assertRaisesRegex(ValueError, "changed after preprocessing"):
            validate_preprocessing_split(data, altered, 3)
        fields = dict(protocol_version=4, preprocessing_id=data.uns["chromatin_preprocessing"]["id"],
                      cell_names=data.obs_names.to_numpy(), gene_names=data.var_names.to_numpy())
        runner.validate_velocity_inputs(fields, data)
        with self.assertRaisesRegex(ValueError, "Legacy velocity"):
            runner.validate_velocity_inputs(fields | {"protocol_version": 3}, data)
        with self.assertRaisesRegex(ValueError, "gene order"):
            runner.validate_velocity_inputs(fields | {"gene_names": fields["gene_names"][::-1]}, data)
        with self.assertRaisesRegex(ValueError, "different prepared dataset"):
            runner.validate_velocity_inputs(fields | {"preprocessing_id": "other"}, data)


class ProjectionTests(unittest.TestCase):
    def test_filtered_and_reordered_genes_keep_identity(self):
        base = sparse.csr_matrix([[0., 0., 0., 1.], [0., 1., 0., 0.]])
        actual = widen_projection(base, None, "diagonal_full",
                                  output_genes=pd.Index(["D", "B"]), feature_names=pd.Index(["A", "B", "C", "D"]))
        np.testing.assert_array_equal(actual.toarray(), base.toarray())
        with self.assertRaisesRegex(ValueError, "gene names"):
            widen_projection(base, None, "diagonal_full")

    def test_coaccess_empty_row_uses_gene_name(self):
        values = np.array([[0, 1, 0, 3], [1, 0, 1, 2], [0, 1, 0, 1], [1, 0, 1, 0]], dtype=np.float32)
        actual = widen_projection(sparse.csr_matrix((1, 4)), values, "coaccess", neighbours=0,
                                  output_genes=pd.Index(["D"]), feature_names=pd.Index(["A", "B", "C", "D"]))
        np.testing.assert_array_equal(actual.toarray(), [[0, 0, 0, 1]])

    def test_saved_alpha_map_is_independent_of_trainable_phi(self):
        expected = torch.tensor([[0., .5, .5], [1., 0., 0.]])
        model = SimpleNamespace(phi=SimpleNamespace(gene_projection=torch.ones(4, 3)))
        payload = dict(gene_names=["B", "C"], input_gene_names=["A", "B", "C"], alpha_projection=expected)
        config = dict(gene_map_mode="coaccess", trainable_g=True)
        actual = runner.checkpoint_alpha_projection(payload, config, model, sparse.eye(2, 3))
        np.testing.assert_array_equal(actual.toarray(), expected.numpy())
        del payload["alpha_projection"]
        with self.assertRaisesRegex(ValueError, "did not save"):
            runner.checkpoint_alpha_projection(payload, config, model, sparse.eye(2, 3))


class EvaluationRegressionTests(unittest.TestCase):
    def test_evaluate_cli_defaults_to_val(self):
        args = runner.build_parser().parse_args(["evaluate", "--run-dir", "unused"])
        self.assertEqual(args.eval_split, "val")

    def test_evaluator_uses_saved_target_units_and_linear_joint_blocks(self):
        for target_layer in (SHARED_SPLICED, "spliced_lognorm"):
            with self.subTest(target_layer=target_layer), tempfile.TemporaryDirectory() as tmp:
                data = ad.AnnData(sparse.csr_matrix(np.ones((6, 3), dtype=np.float32)))
                data.var_names = ["A", "B", "C"]
                data.obs_names = [f"cell{i}" for i in range(6)]
                data.obs["DonorID"] = "donor"
                data.obsm["gene_activity"] = np.arange(18, dtype=np.float32).reshape(6, 3)
                data.layers["unspliced"] = sparse.csr_matrix(np.tile([90., 0., 0.], (6, 1)))
                data.layers["spliced"] = sparse.csr_matrix(np.tile([0., 5., 5.], (6, 1)))
                for block in ("unspliced", "spliced"):
                    data.layers[f"{block}_lognorm"] = sparse.csr_matrix(normalize_log(data.layers[block]))
                data.layers["velocity_fixture"] = np.ones((6, 3), dtype=np.float32) * 10
                data.layers["velocity_fixture_u"] = np.ones((6, 3), dtype=np.float32) * 10
                config = dict(split_seed=0, seed=0, target_layer=target_layer, condition="full", gene_map_mode="coaccess")
                projection = torch.tensor([[0., .5, .5], [1., 0., 0.]])
                payload = dict(dataset="bmmc", law=RELAY, gene_names=["B", "C"], input_gene_names=["A", "B", "C"], alpha_projection=projection)
                model = SimpleNamespace(phi=torch.nn.Linear(3, 4))
                fields = dict(velocity=np.ones((6, 3), dtype=np.float32), confidence=np.ones(6),
                              norm=np.ones(6), dynamic_mask=np.ones(6, dtype=bool))
                splits = pd.DataFrame(dict(split=["train", "train", "val", "test", "test", "test"],
                                           train_side=["atac", "rna", "", "", "", ""]), index=data.obs_names)
                splits.to_csv(Path(tmp) / "bmmc_split_seed0.csv")
                args = argparse.Namespace(run_dir=tmp, device="cpu", checkpoint="final",
                                          eval_split="test", max_sinkhorn_cells=8,
                                          tasks_c=False, batch_size=4, reference_layers=["velocity_fixture"])
                seen_targets, seen_velocities, seen_production = [], [], []
                def state_score(predicted, observed, genes):
                    seen_targets.append(observed.copy())
                    return {}, pd.DataFrame()
                def velocity_score(predicted, observed):
                    seen_velocities.append((predicted.copy(), observed.copy()))
                    return {}
                def pushforward(model, law, chromatin, velocity, production, batch_size):
                    seen_production.append((chromatin.numpy(), production.numpy()))
                    shape = (len(chromatin), 4)
                    return np.ones(shape), np.ones(shape), np.full(shape, np.log(10.))
                with patch.object(runner, "CACHE_ROOT", Path(tmp)), \
                     patch.object(runner, "load_dataset", return_value=data), \
                     patch.object(runner, "load_checkpoint", return_value=(model, payload, config)), \
                     patch.object(runner, "load_velocity", return_value=fields), \
                     patch.object(runner, "save_audit"), \
                     patch.object(runner, "gene_map", return_value=(sparse.eye(3).tocsr(), None, None)), \
                     patch.object(runner, "task_a_state", side_effect=state_score), \
                     patch.object(runner, "task_b_pairing", return_value={}), \
                     patch.object(runner, "foscttm_within", return_value={}), \
                     patch.object(runner, "task_d_kinetics", side_effect=velocity_score), \
                     patch.object(runner, "pushforward_and_rhs", side_effect=pushforward), \
                     contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(runner.evaluate_main(args), 0)
                expected = np.log1p(500. if target_layer == SHARED_SPLICED else 5000.)
                np.testing.assert_allclose(seen_targets[0], expected, rtol=1e-6)
                joint, reference = seen_velocities[-1]
                self.assertEqual(joint.shape, (3, 4))
                np.testing.assert_allclose(joint, reference, rtol=1e-6)
                for chromatin, production in seen_production:
                    np.testing.assert_allclose(production, chromatin @ projection.numpy().T)
                self.assertTrue((Path(tmp) / "evaluation_final_test.json").exists())
                self.assertTrue((Path(tmp) / "evaluation_final.json").exists())

    def test_evaluator_val_split_does_not_write_legacy_test_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = ad.AnnData(sparse.csr_matrix(np.ones((6, 3), dtype=np.float32)))
            data.var_names = ["A", "B", "C"]
            data.obs_names = [f"cell{i}" for i in range(6)]
            data.obs["DonorID"] = "donor"
            data.obsm["gene_activity"] = np.arange(18, dtype=np.float32).reshape(6, 3)
            data.layers["unspliced"] = sparse.csr_matrix(np.tile([90., 0., 0.], (6, 1)))
            data.layers["spliced"] = sparse.csr_matrix(np.tile([0., 5., 5.], (6, 1)))
            for block in ("unspliced", "spliced"):
                data.layers[f"{block}_lognorm"] = sparse.csr_matrix(normalize_log(data.layers[block]))
            data.layers["velocity_fixture"] = np.ones((6, 3), dtype=np.float32) * 10
            data.layers["velocity_fixture_u"] = np.ones((6, 3), dtype=np.float32) * 10
            config = dict(split_seed=0, seed=0, target_layer=SHARED_SPLICED, condition="full",
                          gene_map_mode="coaccess")
            projection = torch.tensor([[0., .5, .5], [1., 0., 0.]])
            payload = dict(dataset="bmmc", law=RELAY, gene_names=["B", "C"],
                           input_gene_names=["A", "B", "C"], alpha_projection=projection)
            model = SimpleNamespace(phi=torch.nn.Linear(3, 4))
            fields = dict(velocity=np.ones((6, 3), dtype=np.float32), confidence=np.ones(6),
                          norm=np.ones(6), dynamic_mask=np.ones(6, dtype=bool))
            splits = pd.DataFrame(dict(split=["train", "train", "val", "test", "test", "test"],
                                       train_side=["atac", "rna", "", "", "", ""]), index=data.obs_names)
            splits.to_csv(Path(tmp) / "bmmc_split_seed0.csv")
            args = argparse.Namespace(run_dir=tmp, device="cpu", checkpoint="final",
                                      eval_split="val", max_sinkhorn_cells=8,
                                      tasks_c=False, batch_size=4, reference_layers=["velocity_fixture"])
            seen_targets = []
            def state_score(predicted, observed, genes):
                seen_targets.append(observed.copy())
                return {}, pd.DataFrame()
            def velocity_score(predicted, observed):
                return {}
            def pushforward(model, law, chromatin, velocity, production, batch_size):
                shape = (len(chromatin), 4)
                return np.ones(shape), np.ones(shape), np.full(shape, np.log(10.))
            with patch.object(runner, "CACHE_ROOT", Path(tmp)), \
                 patch.object(runner, "load_dataset", return_value=data), \
                 patch.object(runner, "load_checkpoint", return_value=(model, payload, config)), \
                 patch.object(runner, "load_velocity", return_value=fields), \
                 patch.object(runner, "save_audit"), \
                 patch.object(runner, "gene_map", return_value=(sparse.eye(3).tocsr(), None, None)), \
                 patch.object(runner, "task_a_state", side_effect=state_score), \
                 patch.object(runner, "task_b_pairing", return_value={}), \
                 patch.object(runner, "foscttm_within", return_value={}), \
                 patch.object(runner, "task_d_kinetics", side_effect=velocity_score), \
                 patch.object(runner, "pushforward_and_rhs", side_effect=pushforward), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.evaluate_main(args), 0)
            self.assertEqual(seen_targets[0].shape[0], 1)
            self.assertTrue((Path(tmp) / "evaluation_final_val.json").exists())
            self.assertFalse((Path(tmp) / "evaluation_final.json").exists())
            self.assertFalse((Path(tmp) / "task_a_per_gene_final.csv").exists())

    def test_new_cache_paths_preserve_legacy_paths(self):
        self.assertNotEqual(runner.dataset_path("hspc"), runner.dataset_path("hspc", 0))
        self.assertNotEqual(runner.dataset_path("hspc", 0), runner.dataset_path("hspc", 1))
        self.assertNotEqual(runner.velocity_path("hspc", protocol=3), runner.velocity_path("hspc", protocol=4))
        self.assertIn(PREPROCESSING_PROTOCOL, str(runner.dataset_path("hspc", 0)))


if __name__ == "__main__":
    unittest.main()
