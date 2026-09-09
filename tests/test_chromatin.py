import tempfile
import unittest
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy import sparse

from run_kot_chromatin import (
    alignment_columns,
    apply_velocity_condition,
    held_block_columns,
    gene_affine_calibration,
    output_gene_mask,
    preflight_verdict,
    rna_target_layer,
    run_directory,
    target_cell_mask,
)
from src.data.chromatin import (CHROMATIN_TRANSFORMS, attach_optional_cell_types,
                                chromatin_features, select_bmmc_lsi,
                                splicing_norm_ratio)
from src.evaluation.foscttm import calc_domainAveraged_FOSCTTM, permuted_pairing_floor
from src.data.chromatin_map import permute_chromatin_projection
from src.data.chromatin_r2 import SHARED_SPLICED
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
    task_d_kinetics,
    task_d_tracks, sinkhorn_plan_retrieval,
)
from src.losses.chromatin_laws import (
    LOG1P_MAX, REDUCED, RELAY, block_scales, conditions_for_law, law_coordinates,
    relay_rhs, law_rhs,
)
from src.models.chromatin_kot import PHI_GATES, ChromatinKOT, GeneAffineResidualPhi, residual_gate_init


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
        with self.assertRaises(ValueError):
            conditions_for_law("reduced")
        self.assertNotIn("reverse", conditions_for_law(RELAY))


class RunDirectoryTests(unittest.TestCase):
    def test_existing_runs_and_their_pass_markers_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pilot"
            path.mkdir()
            marker = path / "preflight_passed.json"
            marker.write_text("stale")
            args = type("Args", (), {"run_dir": str(path), "dataset": "hspc", "law": "reduced", "condition": "full", "seed": 42, "subsample": None})()
            with self.assertRaises(FileExistsError):
                run_directory(args)
            self.assertEqual(marker.read_text(), "stale")


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


class PhiGateTests(unittest.TestCase):
    def phi(self, gate: str, n_genes: int = 6):
        scale = torch.tensor([1.0, 0.0, 2.0, 0.0, 0.5, 1.5])[:n_genes]
        phi = GeneAffineResidualPhi(
            n_genes, n_genes, [8], projection=torch.eye(n_genes), scale=scale,
            bias=torch.zeros(n_genes), gate=gate)
        return phi, scale

    def test_scalar_zero_hands_the_network_no_gradient_at_all(self):
        # The mode every run so far used: the MLP is multiplied by tanh(0) = 0, so its
        # own weights get exactly zero gradient on the first step.
        phi, _ = self.phi("scalar-zero")
        phi(torch.ones(3, 6)).sum().backward()
        weights = [p for name, p in phi.net.named_parameters() if "weight" in name]
        self.assertTrue(all(float(p.grad.abs().max()) == 0.0 for p in weights))

    def test_removing_the_gate_lets_the_network_train_from_the_first_step(self):
        phi, _ = self.phi("none")
        phi(torch.ones(3, 6)).sum().backward()
        weights = [p for name, p in phi.net.named_parameters() if "weight" in name]
        self.assertTrue(any(float(p.grad.abs().max()) > 0.0 for p in weights))

    def test_per_gene_gate_opens_widest_where_the_affine_path_is_missing(self):
        phi, scale = self.phi("per-gene")
        opened = phi.residual_scale()
        self.assertEqual(opened.shape, scale.shape)
        self.assertTrue(bool((opened[scale == 0] > opened[scale != 0].max()).all()))
        phi(torch.ones(3, 6)).sum().backward()
        weights = [p for name, p in phi.net.named_parameters() if "weight" in name]
        self.assertTrue(any(float(p.grad.abs().max()) > 0.0 for p in weights))

    def test_every_declared_gate_mode_builds(self):
        for gate in PHI_GATES:
            self.assertEqual(self.phi(gate)[0](torch.ones(2, 6)).shape, (2, 6))
        self.assertTrue(bool((residual_gate_init(torch.zeros(3)) == 1.5).all()))


class CentredCosineNullTests(unittest.TestCase):
    def fields(self, seed: int, signal: float):
        """A reference with a strong shared mean direction plus `signal` worth of per-cell
        structure, and a push-forward that sees only the shared part."""
        rng = np.random.default_rng(seed)
        mean_direction = rng.normal(size=(1, 40)).astype(np.float32)
        per_cell = rng.normal(size=(300, 40)).astype(np.float32)
        reference = mean_direction + per_cell
        pushforward = mean_direction + signal * per_cell + rng.normal(size=(300, 40)).astype(np.float32)
        return pushforward, reference

    def test_a_field_with_no_per_cell_agreement_does_not_clear_its_null(self):
        metrics = task_d_kinetics(*self.fields(0, signal=0.0))
        # The raw cosine is comfortably "positive" — the value the old gate asked for —
        # on a field that agrees with the reference about no individual cell at all.
        self.assertGreater(metrics["cell_cosine_median"], 0.3)
        self.assertLessEqual(metrics["cell_cosine_centred_median"],
                             metrics["cell_cosine_centred_null_p95"])

    def test_real_per_cell_agreement_clears_the_null(self):
        metrics = task_d_kinetics(*self.fields(0, signal=1.0))
        self.assertGreater(metrics["cell_cosine_centred_median"],
                           metrics["cell_cosine_centred_null_p95"])


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
            "foscttm_permuted_floor": 0.5,
            "foscttm_constant_floor": 0.25,
            "jvp_vs_reference_cosine_median": 0.05,
            "jvp_vs_reference_cosine_centred_median": 0.04,
            "jvp_vs_reference_cosine_centred_null_p95": 0.01,
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
        checks["jvp_vs_reference_cosine_centred_median"] = -0.01
        passed, failures = preflight_verdict(checks, "noDyn")
        self.assertFalse(passed)
        self.assertTrue(any("permutation null" in line for line in failures))

    def test_a_positive_centred_cosine_inside_its_null_is_not_evidence(self):
        # The gate this replaced asked only for "> 0", which a ridge fitted on shuffled
        # pairs clears. Chance level for this statistic is the permutation null.
        checks = self.passing_checks()
        checks["jvp_vs_reference_cosine_centred_median"] = 0.006
        checks["jvp_vs_reference_cosine_centred_null_p95"] = 0.008
        passed, failures = preflight_verdict(checks, "full")
        self.assertFalse(passed)
        self.assertTrue(any("permutation null" in line for line in failures))

    def test_a_large_raw_cosine_cannot_carry_a_run_past_the_gate(self):
        checks = self.passing_checks()
        checks["jvp_vs_reference_cosine_median"] = 0.39
        checks["jvp_vs_reference_cosine_centred_median"] = 0.006
        checks["jvp_vs_reference_cosine_centred_null_p95"] = 0.008
        passed, _ = preflight_verdict(checks, "full")
        self.assertFalse(passed)

    def test_a_missing_null_is_a_failure_not_a_pass(self):
        checks = self.passing_checks()
        checks["jvp_vs_reference_cosine_centred_null_p95"] = None
        passed, failures = preflight_verdict(checks, "full")
        self.assertFalse(passed)
        self.assertTrue(any("missing" in line for line in failures))

    def test_direct_pairing_must_beat_the_permuted_pairing_floor(self):
        checks = self.passing_checks()
        checks["foscttm"] = checks["foscttm_permuted_floor"]
        passed, failures = preflight_verdict(checks, "full")
        self.assertFalse(passed)
        self.assertTrue(any("permuted-pairing floor" in line for line in failures))

    def test_the_constant_map_floor_is_no_longer_the_threshold(self):
        """A run between the two floors passes: the constant floor is 0.25 by arithmetic.

        `permuted_pairing_floor` documents why. Every run of 2026-09-09 sat in this band,
        so the gate has to be shown NOT to reject it any more.
        """
        checks = self.passing_checks()
        checks["foscttm"] = 0.29
        passed, failures = preflight_verdict(checks, "full")
        self.assertTrue(passed, failures)

    def test_a_missing_permuted_floor_is_a_failure_not_a_pass(self):
        checks = self.passing_checks()
        checks["foscttm_permuted_floor"] = None
        passed, failures = preflight_verdict(checks, "full")
        self.assertFalse(passed)
        self.assertTrue(any("missing" in line for line in failures))

    def test_shuffle_is_not_required_to_recover_biology(self):
        checks = self.passing_checks()
        checks["foscttm"] = 0.5
        checks["jvp_vs_reference_cosine_centred_median"] = -0.2
        passed, _ = preflight_verdict(checks, "shuffle")
        self.assertTrue(passed)

    def test_total_rna_target_skips_spliced_reference_gate(self):
        checks = self.passing_checks()
        checks["jvp_vs_reference_cosine_median"] = None
        checks["jvp_vs_reference_cosine_centred_median"] = None
        checks["jvp_vs_reference_cosine_centred_null_p95"] = None
        checks["jvp_vs_reference_skipped"] = True
        passed, failures = preflight_verdict(checks, "full")
        self.assertTrue(passed)
        self.assertEqual(failures, [])


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

    def test_reduced_auto_prefers_spliced_when_available(self):
        adata = self.relay_adata()
        name = rna_target_layer(adata, "auto", REDUCED)
        self.assertEqual(name, "spliced_lognorm")
        self.assertTrue(np.array_equal(target_cell_mask(adata, name), [True, False, True]))
        self.assertTrue(np.array_equal(
            output_gene_mask(adata, REDUCED, target_layer=name), [True, False]))
        self.assertEqual(rna_target_layer(adata, "auto", RELAY), SHARED_SPLICED)

    def test_reduced_falls_back_to_total_rna_without_splicing(self):
        adata = ad.AnnData(X=np.ones((2, 2), dtype=np.float32))
        adata.layers["rna_lognorm"] = np.ones((2, 2), dtype=np.float32)
        self.assertEqual(rna_target_layer(adata, "auto", REDUCED), "rna_lognorm")
        self.assertTrue(np.array_equal(
            output_gene_mask(adata, REDUCED, target_layer="rna_lognorm"), [True, True]))

    def test_relay_rejects_total_rna_and_drops_unusable_genes(self):
        adata = self.relay_adata()
        with self.assertRaises(ValueError):
            rna_target_layer(adata, "rna", RELAY)
        self.assertTrue(np.array_equal(output_gene_mask(adata, RELAY), [True, False]))


class UnsplicedTargetTests(unittest.TestCase):
    def test_unspliced_target_is_restricted_to_cells_that_have_splicing(self):
        adata = ad.AnnData(X=np.zeros((3, 2), dtype=np.float32))
        adata.obs["has_splicing"] = np.array([True, False, True])
        for layer in ("spliced_lognorm", "unspliced_lognorm"):
            self.assertTrue(np.array_equal(target_cell_mask(adata, layer),
                                           [True, False, True]))
        self.assertTrue(target_cell_mask(adata, "rna_lognorm").all())


class LogCoordinateLawTests(unittest.TestCase):
    def test_relay_rhs_chain_rules_both_blocks(self):
        prediction = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
        kappa = torch.tensor([[1.25]])
        alpha = torch.tensor([[2.0, 1.5]])
        beta = torch.tensor([0.4, 0.5])
        gamma = torch.tensor([0.2, 0.3])
        u_hat, s_hat = prediction[:, :2], prediction[:, 2:]
        u, s = torch.expm1(u_hat).clamp(min=0.0), torch.expm1(s_hat).clamp(min=0.0)
        du = alpha - beta * u
        ds = beta * u - gamma * s
        expected = torch.cat([
            kappa * du * torch.exp(-u_hat),
            kappa * ds * torch.exp(-s_hat),
        ], dim=1)
        actual = relay_rhs(prediction, kappa, alpha, beta, gamma)
        self.assertTrue(torch.allclose(actual, expected))


class AlignBlockTests(unittest.TestCase):
    def test_each_block_selects_its_own_half_and_the_complement(self):
        device = torch.device("cpu")
        aligned = alignment_columns(RELAY, "unspliced", 3, device)
        held = held_block_columns(RELAY, "unspliced", 3, device)
        self.assertTrue(torch.equal(aligned, torch.tensor([0, 1, 2])))
        self.assertTrue(torch.equal(held, torch.tensor([3, 4, 5])))
        self.assertTrue(torch.equal(alignment_columns(RELAY, "spliced", 3, device),
                                    torch.tensor([3, 4, 5])))
        self.assertTrue(torch.equal(held_block_columns(RELAY, "spliced", 3, device),
                                    torch.tensor([0, 1, 2])))

    def test_joint_and_the_reduced_law_have_no_split(self):
        device = torch.device("cpu")
        for law, block in ((RELAY, "joint"), (REDUCED, "spliced"), (REDUCED, "joint")):
            self.assertIsNone(alignment_columns(law, block, 3, device))
            self.assertIsNone(held_block_columns(law, block, 3, device))


class LawRangeTests(unittest.TestCase):
    """phi is unconstrained; the law's view of it must not be."""

    def test_law_view_is_clamped_to_the_range_a_log1p_count_can_take(self):
        raw = torch.tensor([[-40.0, 0.0, 3.0, 45.6]])
        clamped = law_coordinates(raw)
        self.assertTrue(torch.equal(clamped, torch.tensor([[0.0, 0.0, 3.0, LOG1P_MAX]])))
        self.assertTrue(torch.equal(law_coordinates(torch.tensor([[1.5]])),
                                    torch.tensor([[1.5]])))

    def test_extreme_predictions_no_longer_explode_either_law(self):
        # -40 is the case that opened the relay at loss_dyn 6.15e16; +45.6 is the case
        # that annihilated R1's residual. Both must now stay on the same scale as a
        # sane prediction.
        extreme = torch.tensor([[-40.0, 45.6, -40.0, 45.6]])
        sane = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        kappa, alpha = torch.ones(1, 1), torch.ones(1, 2)
        beta, gamma = torch.full((2,), 0.3), torch.full((2,), 0.2)
        wild = relay_rhs(extreme, kappa, alpha, beta, gamma)
        calm = relay_rhs(sane, kappa, alpha, beta, gamma)
        self.assertTrue(bool(torch.isfinite(wild).all()))
        self.assertLess(float(wild.abs().max()), 1e4 * float(calm.abs().max()) + 1e4)


class SplicingNormRatioTests(unittest.TestCase):
    """u and s are normalised to 10,000 counts separately, so beta*u carries a_s/a_u."""

    def test_regulatory_relay_requires_no_source_rna_ratio(self):
        class Stub:
            gamma = torch.full((2,), 0.2)
            beta = torch.full((2,), 0.4)
            kappa = staticmethod(lambda c: torch.ones(len(c), 1))
            g = staticmethod(lambda c: torch.ones(len(c), 2))
        result = law_rhs(Stub(), RELAY, torch.zeros(1, 4), torch.zeros(1, 2), torch.zeros(1, 2))
        self.assertTrue(torch.equal(result, torch.tensor([[1., 1., 0., 0.]])))

    def test_ratio_matches_normalize_log_factors(self):
        adata = ad.AnnData(X=np.zeros((2, 2), dtype=np.float32))
        adata.layers["unspliced"] = sparse.csr_matrix(np.array([[3.0, 1.0], [0.0, 0.0]], np.float32))
        adata.layers["spliced"] = sparse.csr_matrix(np.array([[2.0, 2.0], [5.0, 0.0]], np.float32))
        # row 0: sum_u 4, sum_s 4 -> 1.0;  row 1: sum_u 0 floored to 1, sum_s 5 -> 0.2
        self.assertTrue(np.allclose(splicing_norm_ratio(adata), [1.0, 0.2]))


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
        self.assertEqual(tuple(model.g(torch.randn(7, 2)).shape), (7, 2))


    def test_gene_affine_path_preserves_correspondence_and_its_jvp(self):
        production = torch.tensor([[0.0, 1.0], [2.0, 3.0], [0.0, 0.0], [0.0, 0.0]])
        target = torch.tensor([[0.0, 0.0], [0.0, 0.0], [10.0, 20.0], [14.0, 26.0]])
        source_rows = torch.tensor([0, 1])
        target_rows = torch.tensor([2, 3])
        scale, bias = gene_affine_calibration(
            production, target, source_rows, target_rows, REDUCED)
        self.assertTrue(torch.allclose(scale, torch.tensor([2.0, 3.0])))
        self.assertTrue(torch.allclose(bias, torch.tensor([10.0, 17.0])))
        phi = GeneAffineResidualPhi(2, 2, [4], projection=torch.eye(2), scale=scale,
                                    bias=bias, residual_weight=0.1, use_spectral_norm=False)
        inputs = production[source_rows]
        prediction, tangent = torch.func.jvp(
            phi, (inputs,), (torch.ones_like(inputs),))
        self.assertTrue(torch.allclose(prediction, target[target_rows]))
        self.assertTrue(torch.allclose(tangent, scale.expand_as(tangent)))
        parameter_names = dict(phi.named_parameters())
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


class PairingFloorTests(unittest.TestCase):
    """Why the gate's baseline changed, locked in as a test rather than a comment."""

    def matrices(self, n=180, d=40, seed=0):
        rng = np.random.default_rng(seed)
        observed = rng.normal(size=(n, d)).astype(np.float32)
        return observed + 0.05 * rng.normal(size=(n, d)).astype(np.float32), observed

    def test_the_constant_map_floor_is_025_whatever_the_data(self):
        """(0.5 + 0)/2: a collapsed reference ties every distance in one direction.

        Exact in real arithmetic; float32 distances break a handful of the ties, so this
        lands within a thousandth rather than on the nose. The point is that it does not
        depend on the data -- two unrelated draws give the same number.
        """
        for seed, n in [(0, 120), (1, 200)]:
            _, observed = self.matrices(n=n, seed=seed)
            constant = np.tile(observed.mean(axis=0), (len(observed), 1))
            self.assertAlmostEqual(
                float(np.mean(calc_domainAveraged_FOSCTTM(constant, observed))), 0.25,
                delta=0.005)

    def test_the_permuted_floor_is_05_and_is_not_degenerate(self):
        _, observed = self.matrices()
        self.assertAlmostEqual(permuted_pairing_floor(observed, observed), 0.5, delta=0.05)

    def test_an_informative_map_sits_between_the_two_floors(self):
        """The band every 2026-09-09 run landed in: beaten by neither reading."""
        prediction, observed = self.matrices()
        score = float(np.mean(calc_domainAveraged_FOSCTTM(prediction, observed)))
        self.assertLess(score, permuted_pairing_floor(prediction, observed))

    def test_a_permuted_pairing_needs_matching_lengths(self):
        prediction, observed = self.matrices()
        with self.assertRaises(ValueError):
            permuted_pairing_floor(prediction[:10], observed)


class ChromatinFeatureTests(unittest.TestCase):
    """c under each transform, and what the transforms are for."""

    def data(self, n=150, d=60, seed=0):
        rng = np.random.default_rng(seed)
        depth = rng.gamma(4.0, 30.0, size=(n, 1))
        counts = rng.poisson(depth * rng.random((1, d)) * 0.2).astype(np.float32)
        adata = ad.AnnData(np.zeros((n, d), dtype=np.float32))
        adata.obsm["gene_activity_counts"] = sparse.csr_matrix(counts)
        adata.obsm["gene_activity"] = counts
        return adata

    def pc1_vs_depth(self, values, depth):
        centred = values - values.mean(axis=0, keepdims=True)
        pc1 = centred @ np.linalg.svd(centred, full_matrices=False)[2][0]
        return abs(float(np.corrcoef(pc1, depth)[0, 1]))

    def test_every_transform_keeps_the_gene_columns_G_expects(self):
        adata = self.data()
        for transform in CHROMATIN_TRANSFORMS:
            values = chromatin_features(adata, transform)
            self.assertEqual(values.shape, (adata.n_obs, adata.obsm["gene_activity"].shape[1]))
            self.assertEqual(values.dtype, np.float32)
            self.assertTrue(np.isfinite(values).all(), transform)

    def test_as_is_is_the_stored_matrix_untouched(self):
        adata = self.data()
        np.testing.assert_allclose(chromatin_features(adata, "as_is"),
                                   adata.obsm["gene_activity"])

    def test_tfidf_lsi_removes_the_depth_axis_that_as_is_leaves_in(self):
        adata = self.data()
        depth = np.asarray(adata.obsm["gene_activity_counts"].sum(axis=1)).ravel()
        self.assertLess(self.pc1_vs_depth(chromatin_features(adata, "tfidf_lsi"), depth),
                        self.pc1_vs_depth(chromatin_features(adata, "as_is"), depth))

    def test_an_unknown_transform_is_refused_rather_than_silently_ignored(self):
        with self.assertRaises(ValueError):
            chromatin_features(self.data(), "quantile")
