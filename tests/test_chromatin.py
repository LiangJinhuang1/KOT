import tempfile
import unittest
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy import sparse

from run_kot_chromatin import (
    apply_velocity_condition,
    gene_affine_calibration,
    output_gene_mask,
    preflight_verdict,
    rna_target_layer,
    run_directory,
    target_cell_mask,
)
from src.data.chromatin import attach_optional_cell_types, select_bmmc_lsi
from src.data.chromatin_map import permute_chromatin_projection
from src.data.chromatin_velocity import (
    direction_diversity,
    finite_lsi_mask,
    hspc_velocity,
    interpolate_pseudotime,
    train_neighbour_graph,
    speed_reliability,
)
from src.evaluation.chromatin_eval import (
    retrieval_metrics,
    task_b_pairing,
    task_d_tracks, sinkhorn_plan_retrieval,
)
from src.losses.chromatin_laws import REDUCED, RELAY, block_scales, conditions_for_law
from src.models.chromatin_kot import ChromatinKOT


def fake_hspc(n_day0: int = 10, n_day7: int = 10, n_genes: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = n_day0 + n_day7
    adata = ad.AnnData(X=np.zeros((n, n_genes), dtype=np.float32))
    adata.obs["day"] = np.array([0] * n_day0 + [7] * n_day7)
    adata.obsm["gene_activity"] = rng.normal(size=(n, n_genes)).astype(np.float32)
    adata.obsm["lsi"] = rng.normal(size=(n, 3)).astype(np.float32)
    return adata


def fold_split(n_day0: int, n_day7: int) -> np.ndarray:
    split = np.empty(n_day0 + n_day7, dtype=object)
    split[:n_day0] = ["train"] * (n_day0 // 2) + ["test"] * (n_day0 - n_day0 // 2)
    split[n_day0:] = ["train"] * (n_day7 // 2) + ["test"] * (n_day7 - n_day7 // 2)
    return split


class ConditionsForLawTests(unittest.TestCase):
    def test_relay_drops_reverse_zero_permG(self):
        self.assertEqual(conditions_for_law(RELAY), ["full", "shuffle", "noDyn"])
        self.assertIn("permG", conditions_for_law("reduced"))
        self.assertNotIn("reverse", conditions_for_law(RELAY))


class RunDirectoryTests(unittest.TestCase):
    def test_directory_exists_before_stale_marker_is_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pilot"
            path.mkdir()
            marker = path / "preflight_passed.json"
            marker.write_text("stale")
            args = type("Args", (), {"run_dir": str(path), "dataset": "hspc", "law": "reduced", "condition": "full", "seed": 42, "subsample": None})()
            actual = run_directory(args)
            self.assertEqual(actual, path)
            self.assertFalse(marker.exists())


class BmmcLsiTests(unittest.TestCase):
    def test_prefers_reduced_and_otherwise_drops_depth(self):
        processed = ad.AnnData(X=np.zeros((3, 1), dtype=np.float32))
        processed.obsm["ATAC_lsi_full"] = np.arange(15, dtype=np.float32).reshape(3, 5)
        dropped = select_bmmc_lsi(processed)
        self.assertEqual(dropped.shape, (3, 4))
        self.assertTrue(np.allclose(dropped, processed.obsm["ATAC_lsi_full"][:, 1:]))
        processed.obsm["ATAC_lsi_red"] = np.ones((3, 4), dtype=np.float32)
        self.assertTrue(np.allclose(select_bmmc_lsi(processed), 1.0))


class PermGTests(unittest.TestCase):
    def test_permG_keeps_shape_nnz_and_which_genes_have_a_row(self):
        matrix = sparse.csr_matrix(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32))
        permuted = permute_chromatin_projection(matrix, seed=7)
        self.assertEqual(permuted.shape, matrix.shape)
        self.assertEqual(permuted.nnz, matrix.nnz)
        original = np.asarray((matrix != 0).sum(axis=1)).ravel() > 0
        after = np.asarray((permuted != 0).sum(axis=1)).ravel() > 0
        self.assertTrue(np.array_equal(original, after))


class TaskDTrackTests(unittest.TestCase):
    def test_internal_and_biological_are_independent_answers(self):
        jvp_train = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        rhs_train = jvp_train.copy()
        jvp_true = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        rhs_true = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        tracks = task_d_tracks(jvp_train, rhs_train, jvp_true, rhs_true)
        self.assertGreater(tracks["internal_vs_law"]["cell_cosine_median"], 0.99)
        self.assertLess(tracks["biological_vs_law"]["cell_cosine_median"], 0.1)


class PairingMetricTests(unittest.TestCase):
    def test_identity_reports_perfect_fosknn_and_sinkhorn_foscttm(self):
        embedding = np.eye(20, dtype=np.float32)
        metrics = retrieval_metrics(embedding, embedding, seed=0)
        self.assertEqual(metrics["foscttm"], 0.0)
        self.assertGreater(metrics["fosknn_frac0.01"], 0.99)
        pairing = task_b_pairing(embedding, embedding, None, max_sinkhorn_cells=20, seed=0)
        self.assertEqual(pairing["sinkhorn_foscttm"], 0.0)
        self.assertIn("sinkhorn_fosknn_frac0.05", pairing)
        self.assertIn("knn_fosknn_frac0.1", pairing)


class NeighbourGraphTests(unittest.TestCase):
    def test_held_out_cells_are_never_neighbours_of_train_cells(self):
        rng = np.random.default_rng(0)
        lsi = rng.normal(size=(12, 4)).astype(np.float32)
        train = np.zeros(12, dtype=bool)
        train[:8] = True
        graph = train_neighbour_graph(lsi, train, n_neighbors=3)
        train_to_held = graph[np.ix_(np.flatnonzero(train), np.flatnonzero(~train))]
        self.assertEqual(train_to_held.nnz, 0)

    def test_interpolated_pseudotime_is_a_train_neighbour_average(self):
        train_lsi = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        train_tau = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        query = np.array([[1.0, 0.0]], dtype=np.float32)
        tau = interpolate_pseudotime(train_lsi, train_tau, query, n_neighbors=1)
        self.assertAlmostEqual(float(tau[0]), 1.0, places=5)

    def test_nan_lsi_rows_are_not_usable(self):
        lsi = np.zeros((4, 3), dtype=np.float32)
        lsi[1] = np.nan
        self.assertTrue(np.array_equal(finite_lsi_mask(lsi), [True, False, True, True]))

    def test_direction_diversity_handles_a_single_dynamic_cell(self):
        velocity = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
        metrics = direction_diversity(velocity, np.array([True, False]))
        self.assertTrue(np.isnan(metrics["pairwise_cosine_median"]))
        self.assertTrue(np.isnan(metrics["energy_in_mean_direction"]))


    def test_speed_reliability_uses_train_cap_and_suppresses_derivative_outliers(self):
        velocity = np.array([[1.0, 0.0], [2.0, 0.0], [100.0, 0.0], [1000.0, 0.0]])
        fit = np.array([True, True, True, False])
        reliability, cap = speed_reliability(velocity, fit, upper_quantile=0.5)
        self.assertAlmostEqual(cap, 2.0)
        self.assertEqual(float(reliability[0]), 1.0)
        self.assertLess(float(reliability[2]), 0.001)
        self.assertLess(float(reliability[3]), float(reliability[2]))


class HspcSplitIsolationTests(unittest.TestCase):
    def test_held_out_day7_cannot_write_train_velocities(self):
        adata = fake_hspc()
        split = fold_split(10, 10)
        device = torch.device("cpu")
        before = hspc_velocity(adata, epsilon=0.05, n_iterations=50,
                               confidence_quantile=0.0, device=device, split=split)
        held = (adata.obs["day"].to_numpy() == 7) & (split == "test")
        adata.obsm["gene_activity"] = adata.obsm["gene_activity"].copy()
        adata.obsm["gene_activity"][held] += 50.0
        after = hspc_velocity(adata, epsilon=0.05, n_iterations=50,
                              confidence_quantile=0.0, device=device, split=split)
        train_day0 = (adata.obs["day"].to_numpy() == 0) & (split == "train")
        self.assertTrue(np.allclose(before["velocity"][train_day0],
                                    after["velocity"][train_day0], atol=1e-5))

    def test_rna_side_day7_atac_cannot_write_train_velocities(self):
        adata = fake_hspc()
        split = fold_split(10, 10)
        source_mask = split != "train"
        source_mask[[0, 1, 2, 10, 11, 12]] = True
        device = torch.device("cpu")
        before = hspc_velocity(
            adata, epsilon=0.05, n_iterations=50, confidence_quantile=0.0,
            device=device, split=split, source_mask=source_mask,
        )
        excluded_day7 = (adata.obs["day"].to_numpy() == 7) & ~source_mask
        adata.obsm["gene_activity"] = adata.obsm["gene_activity"].copy()
        adata.obsm["gene_activity"][excluded_day7] += 50.0
        after = hspc_velocity(
            adata, epsilon=0.05, n_iterations=50, confidence_quantile=0.0,
            device=device, split=split, source_mask=source_mask,
        )
        train_source_day0 = np.array([0, 1, 2])
        self.assertTrue(np.allclose(before["velocity"][train_source_day0],
                                    after["velocity"][train_source_day0], atol=1e-5))


class PreflightVerdictTests(unittest.TestCase):
    def passing_checks(self) -> dict:
        return {
            "state_prediction_finite": True,
            "jvp_finite": True,
            "prediction_spread_ratio": 0.2,
            "state_pearson_median": 0.1,
            "foscttm": 0.1,
            "foscttm_constant_floor": 0.25,
            "jvp_vs_reference_cosine_median": 0.05,
            "jvp_vs_reference_cosine_centred_median": 0.04,
        }

    def test_full_passes_only_when_every_gate_item_holds(self):
        passed, failures = preflight_verdict(self.passing_checks(), "full")
        self.assertTrue(passed)
        self.assertEqual(failures, [])
        collapsed = self.passing_checks()
        collapsed["prediction_spread_ratio"] = 0.009
        passed, failures = preflight_verdict(collapsed, "full")
        self.assertFalse(passed)
        self.assertTrue(any("collapsed" in line for line in failures))

    def test_nodyn_still_has_to_push_forward_with_the_reference_direction(self):
        checks = self.passing_checks()
        checks["jvp_vs_reference_cosine_median"] = -0.01
        passed, failures = preflight_verdict(checks, "noDyn")
        self.assertFalse(passed)
        self.assertTrue(any("not positive" in line for line in failures))

    def test_direct_pairing_must_beat_the_constant_map_floor(self):
        checks = self.passing_checks()
        checks["foscttm"] = checks["foscttm_constant_floor"]
        passed, failures = preflight_verdict(checks, "full")
        self.assertFalse(passed)
        self.assertTrue(any("constant-map floor" in line for line in failures))

    def test_shuffle_is_not_required_to_recover_biology(self):
        checks = self.passing_checks()
        checks["foscttm"] = 0.5
        checks["jvp_vs_reference_cosine_median"] = -0.2
        passed, _ = preflight_verdict(checks, "shuffle")
        self.assertTrue(passed)


class CellTypeJoinTests(unittest.TestCase):
    def test_matches_barcodes_after_stripping_the_run_suffix(self):
        adata = ad.AnnData(X=np.zeros((2, 1), dtype=np.float32))
        adata.obs_names = pd.Index(["AAAC-1-d0", "AAAG-1-d0"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cell_types.csv"
            pd.DataFrame({"barcode": ["AAAC-1", "AAAG-1"],
                          "cell_type": ["HSC", "Ery"]}).to_csv(path, index=False)
            attach_optional_cell_types(adata, path)
        self.assertEqual(list(adata.obs["cell_type"]), ["HSC", "Ery"])


class VelocityAblationTests(unittest.TestCase):
    def test_shuffle_is_confined_to_the_eligible_training_rows(self):
        velocity = np.arange(30, dtype=np.float32).reshape(10, 3)
        confidence = np.linspace(0.1, 1.0, 10, dtype=np.float32)
        dynamic = np.arange(10) % 2 == 0
        eligible = np.zeros(10, dtype=bool)
        eligible[[1, 3, 6, 8]] = True

        shuffled_v, shuffled_c, shuffled_d = apply_velocity_condition(
            velocity, confidence, dynamic, "shuffle", seed=7, eligible=eligible
        )
        outside = ~eligible
        self.assertTrue(np.array_equal(shuffled_v[outside], velocity[outside]))
        self.assertTrue(np.array_equal(shuffled_c[outside], confidence[outside]))
        self.assertTrue(np.array_equal(shuffled_d[outside], dynamic[outside]))
        self.assertCountEqual(map(tuple, shuffled_v[eligible]), map(tuple, velocity[eligible]))


class TargetProtocolTests(unittest.TestCase):
    @staticmethod
    def relay_adata() -> ad.AnnData:
        adata = ad.AnnData(X=np.ones((3, 2), dtype=np.float32))
        adata.layers["rna_lognorm"] = np.full((3, 2), 9.0, dtype=np.float32)
        adata.layers["spliced_lognorm"] = np.full((3, 2), 2.0, dtype=np.float32)
        adata.layers["unspliced_lognorm"] = np.full((3, 2), 1.0, dtype=np.float32)
        adata.layers["spliced"] = sparse.csr_matrix(
            np.array([[10, 0], [10, 0], [10, 0]], dtype=np.float32)
        )
        adata.layers["unspliced"] = sparse.csr_matrix(
            np.array([[10, 0], [10, 0], [10, 0]], dtype=np.float32)
        )
        adata.obs["has_splicing"] = [True, False, True]
        return adata

    def test_reduced_auto_uses_complete_mature_rna(self):
        adata = self.relay_adata()
        name = rna_target_layer(adata, "auto", REDUCED)
        self.assertEqual(name, "rna_lognorm")
        self.assertTrue(np.array_equal(target_cell_mask(adata, name), [True, True, True]))
        self.assertEqual(rna_target_layer(adata, "auto", RELAY), "spliced_lognorm")

    def test_relay_rejects_total_rna_and_drops_unusable_genes(self):
        adata = self.relay_adata()
        with self.assertRaises(ValueError):
            rna_target_layer(adata, "rna", RELAY)
        self.assertTrue(np.array_equal(output_gene_mask(adata, RELAY), [True, False]))


class RelayGeometryTests(unittest.TestCase):
    def test_block_scales_normalize_u_and_s_separately(self):
        rng = np.random.default_rng(4)
        u = rng.normal(size=(200, 3)).astype(np.float32)
        s = (100.0 * rng.normal(size=(200, 3))).astype(np.float32)
        target = torch.from_numpy(np.concatenate([u, s], axis=1))
        scaled = target / block_scales(target, RELAY)
        u_std = scaled[:, :3].std(dim=0, unbiased=False)
        s_std = scaled[:, 3:].std(dim=0, unbiased=False)
        self.assertTrue(torch.allclose(u_std, torch.ones_like(u_std), atol=1e-4))
        self.assertTrue(torch.allclose(s_std, torch.ones_like(s_std), atol=1e-4))

    def test_relay_model_supports_more_input_than_output_genes(self):
        model = ChromatinKOT(
            n_genes=2,
            law=RELAY,
            n_input_features=5,
            phi_dims=(8,),
            kappa_dims=(4,),
            g_dims=(4,),
        )
        chromatin = torch.randn(7, 5)
        self.assertEqual(tuple(model.phi(chromatin).shape), (7, 4))
        self.assertEqual(tuple(model.g(chromatin).shape), (7, 2))


    def test_gene_affine_path_preserves_correspondence_and_its_jvp(self):
        production = torch.tensor([[0.0, 1.0], [2.0, 3.0], [0.0, 0.0], [0.0, 0.0]])
        target = torch.tensor([[0.0, 0.0], [0.0, 0.0], [10.0, 20.0], [14.0, 26.0]])
        source_rows = torch.tensor([0, 1])
        target_rows = torch.tensor([2, 3])
        scale, bias = gene_affine_calibration(
            production, target, source_rows, target_rows, REDUCED)
        self.assertTrue(torch.allclose(scale, torch.tensor([2.0, 3.0])))
        self.assertTrue(torch.allclose(bias, torch.tensor([10.0, 17.0])))
        model = ChromatinKOT(
            n_genes=2,
            law=REDUCED,
            phi_dims=(4,),
            kappa_dims=(2,),
            g_dims=(2,),
            phi_spectral_norm=False,
            phi_projection=torch.eye(2),
            phi_scale=scale,
            phi_bias=bias,
            phi_residual_weight=0.1,
        )
        inputs = production[source_rows]
        prediction, tangent = torch.func.jvp(
            model.phi, (inputs,), (torch.ones_like(inputs),))
        self.assertTrue(torch.allclose(prediction, target[target_rows]))
        self.assertTrue(torch.allclose(tangent, scale.expand_as(tangent)))
        parameter_names = dict(model.phi.named_parameters())
        self.assertIn("gene_scale", parameter_names)
        self.assertIn("gene_bias", parameter_names)


class SinkhornRetrievalTests(unittest.TestCase):
    def test_identity_plan_is_perfect(self):
        metrics = sinkhorn_plan_retrieval(np.eye(12, dtype=np.float32))
        self.assertEqual(metrics["foscttm"], 0.0)
        self.assertEqual(metrics["top1"], 1.0)

    def test_shifted_plan_is_not_reported_as_perfect(self):
        plan = np.roll(np.eye(12, dtype=np.float32), shift=1, axis=1)
        metrics = sinkhorn_plan_retrieval(plan)
        self.assertEqual(metrics["top1"], 0.0)
        self.assertGreater(metrics["foscttm"], 0.0)


if __name__ == "__main__":
    unittest.main()
