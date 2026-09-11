import argparse
import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy import sparse

from run_kot_chromatin import (
    KOT_KAPPA_PRIOR_TARGET,
    REGULATORY_TRANSFORMS,
    TRAINING_YAML_KEYS,
    alignment_columns,
    alpha_features,
    apply_velocity_condition,
    build_parser,
    chromatin_network_params,
    chromatin_param_groups,
    evaluation_split_rows,
    held_block_columns,
    gene_affine_calibration,
    output_gene_mask,
    persist_preflight,
    preflight_main,
    preflight_verdict,
    production_from_phi,
    resolve_r2_training_flags,
    regulatory_features,
    rna_target_layer,
    run_directory,
    stabilization_settings,
    state_metrics,
    target_cell_mask,
    write_evaluation_outputs,
    write_preflight_result,
)
from src.data.chromatin import (CHROMATIN_TRANSFORMS, INPUT_SCREEN_TRANSFORMS,
                                TFIDF_LSI_COMPONENTS, attach_optional_cell_types,
                                chromatin_features, select_bmmc_lsi,
                                splicing_norm_ratio)
from src.evaluation.foscttm import calc_domainAveraged_FOSCTTM, permuted_pairing_floor
from src.data.chromatin_map import (
    genomic_projection, peak_genomic_projection,
    permute_chromatin_projection, widen_projection,
)
from src.data.chromatin_r2 import SHARED_SPLICED
from src.data.chromatin_velocity import (
    VELOCITY_ESTIMATORS,
    euler_lag,
    forward_difference_field,
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
    CP10K_TARGET, GLOBAL_LINEAR, LOG1P_MAX, PANEL_LOG1P_COMPOSITIONAL, REDUCED, RELAY,
    SHARED_LOG1P, block_scales, conditions_for_law, law_coordinates,
    kinetics_loss, kinetics_losses, relay_flux, relay_rhs, law_rhs,
    resolve_kinetic_coords, state_rate_to_linear,
)
from src.models.chromatin_kot import PHI_GATES, ChromatinKOT, GeneAffineResidualPhi, residual_gate_init
from src.training.kot import FixedKappa
from src.utils.io import load_yaml
from tools.chromatin_failure_analysis import (
    CANDIDATE_NAMES, alignment_block, permute_rows,
    unpaired_alignment_candidates,
)


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


class PreflightPersistTests(unittest.TestCase):
    def passing_checks(self) -> dict:
        return PreflightVerdictTests().passing_checks()

    def test_a_pass_writes_the_marker_and_a_fail_removes_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            passed, _ = write_preflight_result(run, self.passing_checks(), "full")
            self.assertTrue(passed)
            self.assertTrue((run / "preflight_passed.json").exists())
            failed = self.passing_checks()
            failed["jvp_vs_reference_cosine_centred_median"] = None
            failed["jvp_vs_reference_cosine_centred_null_p95"] = None
            passed, failures = write_preflight_result(run, failed, "full")
            self.assertFalse(passed)
            self.assertTrue(any("reference" in line for line in failures))
            self.assertFalse((run / "preflight_passed.json").exists())
            self.assertTrue((run / "preflight.json").exists())
            self.assertTrue((run / "preflight_best_align.json").exists())

    def test_scoring_final_does_not_clobber_the_launch_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            write_preflight_result(run, self.passing_checks(), "full", checkpoint="best_align")
            failed = self.passing_checks()
            failed["jvp_vs_reference_cosine_centred_median"] = -0.2
            write_preflight_result(run, failed, "full", checkpoint="final")
            self.assertTrue((run / "preflight_passed.json").exists())
            self.assertEqual(
                json.loads((run / "preflight.json").read_text())["foscttm"],
                self.passing_checks()["foscttm"])
            self.assertEqual(
                json.loads((run / "preflight_final.json").read_text())
                ["jvp_vs_reference_cosine_centred_median"], -0.2)

    def test_a_refresh_records_a_failed_gate_without_exiting_nonzero(self):
        checks = self.passing_checks()
        checks["jvp_vs_reference_cosine_centred_median"] = -0.2
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                code = persist_preflight(
                    run, checks, "full", "preflight", refuse_failed_gates=False)
            self.assertEqual(code, 0)
            self.assertFalse((run / "preflight_passed.json").exists())

    def test_a_refresh_still_fails_if_the_rna_reference_is_missing(self):
        checks = self.passing_checks()
        checks["jvp_vs_reference_cosine_centred_median"] = None
        checks["jvp_vs_reference_cosine_centred_null_p95"] = None
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stdout(io.StringIO()):
                code = persist_preflight(
                    Path(tmp), checks, "full", "preflight", refuse_failed_gates=False)
            self.assertEqual(code, 1)

    def test_training_still_refuses_a_failed_full_gate(self):
        checks = self.passing_checks()
        checks["jvp_vs_reference_cosine_centred_median"] = -0.2
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stdout(io.StringIO()):
                code = persist_preflight(Path(tmp), checks, "full", "train")
            self.assertEqual(code, 1)

    def test_parser_exposes_the_refresh_command(self):
        args = build_parser().parse_args(
            ["preflight", "--run-dir", "cache/chromatin/runs/example"])
        self.assertEqual(args.checkpoint, "best_align")
        self.assertEqual(args.reference_layer, "velocity_scvelo")
        self.assertIs(args.func, preflight_main)


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

    def test_compositional_flux_stays_on_the_panel_simplex(self):
        u = torch.tensor([[1000.0, 2000.0]])
        s = torch.tensor([[3000.0, 4000.0]])
        self.assertAlmostEqual(float((u + s).sum()), CP10K_TARGET)
        prediction = torch.log1p(torch.cat([u, s], dim=1))
        alpha = torch.tensor([[2.0, 1.5]])
        beta = torch.tensor([0.4, 0.5])
        gamma = torch.tensor([0.2, 0.3])
        flux = relay_flux(prediction, alpha, beta, gamma,
                          coords=PANEL_LOG1P_COMPOSITIONAL)
        linear = state_rate_to_linear(flux, prediction, PANEL_LOG1P_COMPOSITIONAL)
        self.assertAlmostEqual(float(linear.sum()), 0.0, places=3)
        uncorrected = state_rate_to_linear(
            relay_flux(prediction, alpha, beta, gamma, coords=SHARED_LOG1P),
            prediction, SHARED_LOG1P)
        self.assertGreater(abs(float(uncorrected.sum())), 1.0)

    def test_global_linear_skips_the_log_chain_rule(self):
        prediction = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        alpha = torch.tensor([[2.0, 1.5]])
        beta = torch.tensor([0.4, 0.5])
        gamma = torch.tensor([0.2, 0.3])
        u, s = prediction[:, :2], prediction[:, 2:]
        expected = torch.cat([alpha - beta * u, beta * u - gamma * s], dim=1)
        actual = relay_flux(prediction, alpha, beta, gamma, coords=GLOBAL_LINEAR)
        self.assertTrue(torch.allclose(actual, expected))
        converted = state_rate_to_linear(actual, prediction, GLOBAL_LINEAR)
        self.assertTrue(torch.allclose(converted, actual))

    def test_an_unknown_kinetic_coordinate_is_refused(self):
        with self.assertRaises(ValueError):
            resolve_kinetic_coords("cp10k_but_forgot_the_ode")


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
        adata.obs["batch"] = np.where(np.arange(n) < n // 2, "b1", "b2")
        return adata

    def batched(self, n=160, d=60, seed=0, offset=5.0):
        """Two batches with a deliberate additive offset between them."""
        adata = self.data(n=n, d=d, seed=seed)
        counts = np.asarray(adata.obsm["gene_activity_counts"].todense())
        second = adata.obs["batch"].to_numpy() == "b2"
        counts[second] += offset
        adata.obsm["gene_activity_counts"] = sparse.csr_matrix(counts)
        adata.obsm["gene_activity"] = counts.astype(np.float32)
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

    def test_batch_centring_removes_the_between_batch_offset(self):
        adata = self.batched()
        second = adata.obs["batch"].to_numpy() == "b2"
        def gap(values):
            return abs(float(values[second].mean() - values[~second].mean()))
        self.assertLess(gap(chromatin_features(adata, "tfidf_lsi_batch")),
                        gap(chromatin_features(adata, "tfidf_lsi")))

    def test_batch_centring_needs_a_batch_column(self):
        adata = self.data()
        del adata.obs["batch"]
        with self.assertRaises(ValueError):
            chromatin_features(adata, "tfidf_lsi_batch")

    def test_an_unknown_transform_is_refused_rather_than_silently_ignored(self):
        with self.assertRaises(ValueError):
            chromatin_features(self.data(), "quantile")

    def test_cp10k_linear_is_library_size_without_log(self):
        adata = self.data()
        linear = chromatin_features(adata, "cp10k_linear")
        logged = chromatin_features(adata, "cp10k_log1p")
        np.testing.assert_allclose(np.log1p(linear), logged, rtol=1e-5, atol=1e-5)
        self.assertGreater(float(linear.max()), float(logged.max()))

    def test_tfidf_lsi_is_the_rank50_reconstruction_of_tfidf_gene(self):
        from sklearn.utils.extmath import randomized_svd
        rng = np.random.default_rng(1)
        adata = self.data(n=200, d=80, seed=1)
        counts = np.asarray(adata.obsm["gene_activity_counts"].todense(), dtype=np.float32)
        counts = counts + rng.integers(0, 8, size=counts.shape).astype(np.float32)
        adata.obsm["gene_activity_counts"] = sparse.csr_matrix(counts)
        gene = chromatin_features(adata, "tfidf_gene")
        lsi = chromatin_features(adata, "tfidf_lsi")
        self.assertEqual(gene.shape, lsi.shape)
        centre = gene.mean(axis=0, keepdims=True)
        _, _, right = randomized_svd(gene - centre, n_components=TFIDF_LSI_COMPONENTS,
                                     random_state=0)
        reconstructed = (((gene - centre) @ right.T) @ right + centre).astype(np.float32)
        np.testing.assert_allclose(lsi, reconstructed, rtol=2e-3, atol=2e-3)
        singular = np.linalg.svd(lsi - centre, compute_uv=False)
        rank = int(np.sum(singular > 1e-3 * singular[0]))
        self.assertLessEqual(rank, TFIDF_LSI_COMPONENTS)

    def test_the_input_screen_is_the_six_phi_gauges(self):
        self.assertEqual(
            list(INPUT_SCREEN_TRANSFORMS),
            ["as_is", "cp10k_linear", "cp10k_log1p", "l2_per_cell", "tfidf_gene", "tfidf_lsi"])
        for name in INPUT_SCREEN_TRANSFORMS:
            self.assertIn(name, CHROMATIN_TRANSFORMS)
        self.assertNotIn("raw_counts", CHROMATIN_TRANSFORMS)

    def test_l2_per_cell_is_depth_removal_without_a_library_sum(self):
        adata = self.data()
        raw = np.asarray(adata.obsm["gene_activity_counts"].todense(), dtype=np.float32)
        values = chromatin_features(adata, "l2_per_cell")
        np.testing.assert_allclose(np.linalg.norm(values, axis=1), 1.0, atol=1e-5)
        np.testing.assert_allclose(
            values * np.linalg.norm(raw, axis=1, keepdims=True).clip(min=1e-6),
            raw, rtol=1e-5, atol=1e-5)
        self.assertFalse(np.allclose(values, chromatin_features(adata, "cp10k_linear")))


class RegulatoryTransformTests(unittest.TestCase):
    """Alpha may read G c in a different gene-activity gauge than phi."""

    def test_same_reuses_the_map_array_including_a_lag(self):
        adata = ChromatinFeatureTests().data()
        mapped = chromatin_features(adata, "cp10k_log1p")
        lagged = euler_lag(mapped, np.ones_like(mapped), 1.0)
        regulatory = regulatory_features(adata, "cp10k_log1p", "same", map_features=lagged)
        self.assertIs(regulatory, lagged)

    def test_linear_regulatory_input_is_not_the_logged_map(self):
        adata = ChromatinFeatureTests().data()
        mapped = chromatin_features(adata, "cp10k_log1p")
        regulatory = regulatory_features(
            adata, "cp10k_log1p", "cp10k_linear", map_features=mapped)
        self.assertFalse(np.allclose(regulatory, mapped))
        np.testing.assert_allclose(regulatory, chromatin_features(adata, "cp10k_linear"))

    def test_a_decoupled_regulatory_input_is_not_lagged_with_phi(self):
        adata = ChromatinFeatureTests().data()
        mapped = chromatin_features(adata, "cp10k_log1p")
        lagged = euler_lag(mapped, np.ones_like(mapped), 1.0)
        regulatory = regulatory_features(
            adata, "cp10k_log1p", "cp10k_linear", map_features=lagged)
        np.testing.assert_allclose(regulatory, chromatin_features(adata, "cp10k_linear"))
        self.assertFalse(np.allclose(regulatory, lagged))

    def test_unknown_regulatory_transform_is_refused(self):
        adata = ChromatinFeatureTests().data()
        with self.assertRaises(ValueError):
            regulatory_features(adata, "tfidf_lsi", "tfidf_gene")
        self.assertEqual(REGULATORY_TRANSFORMS, ("same", "cp10k_linear"))

    def test_parser_defaults_to_the_coupled_map(self):
        args = build_parser().parse_args(
            ["train", "--dataset", "bmmc", "--gamma-anchor-csv", "unused.csv"])
        self.assertEqual(args.regulatory_transform, "same")
        self.assertEqual(args.rna_kinetic_coords, SHARED_LOG1P)


class FailureAnalysisDiagnosticTests(unittest.TestCase):
    def test_the_five_candidates_are_the_ones_the_ambiguity_argument_needs(self):
        self.assertEqual(
            CANDIDATE_NAMES,
            ("constant", "affine_init", "trained_phi", "paired_oracle", "permuted_oracle"))

    def test_relay_sinkhorn_reads_only_the_spliced_block(self):
        values = torch.arange(12, dtype=torch.float32).reshape(2, 6)
        spliced = alignment_block(values, RELAY)
        torch.testing.assert_close(spliced, values[:, 3:])
        torch.testing.assert_close(alignment_block(values[:, 3:], REDUCED), values[:, 3:])

    def test_a_row_permutation_keeps_the_measure_and_destroys_the_pairing(self):
        values = np.arange(12, dtype=np.float32).reshape(4, 3)
        permuted = permute_rows(values, np.random.default_rng(0))
        np.testing.assert_array_equal(np.sort(values, axis=0), np.sort(permuted, axis=0))
        self.assertFalse(np.array_equal(values, permuted))
        tensor = torch.as_tensor(values)
        permuted_t = permute_rows(tensor, np.random.default_rng(0))
        self.assertEqual({tuple(row) for row in tensor.tolist()},
                         {tuple(row) for row in permuted_t.tolist()})

    def test_unpaired_candidates_are_spliced_and_permutation_preserves_the_rna_cloud(self):
        rng = np.random.default_rng(0)
        predicted = torch.randn(16, 6)
        observed = torch.randn(16, 6)
        paired = torch.randn(16, 6)
        affine = torch.randn(16, 6)
        clouds = unpaired_alignment_candidates(
            predicted, observed, paired, rng, affine=affine, law=RELAY)
        self.assertEqual(list(clouds), list(CANDIDATE_NAMES))
        for block in clouds.values():
            self.assertEqual(block.shape, (16, 3))
        paired_rows = {tuple(np.round(row, 5)) for row in clouds["paired_oracle"].numpy()}
        perm_rows = {tuple(np.round(row, 5)) for row in clouds["permuted_oracle"].numpy()}
        self.assertEqual(paired_rows, perm_rows)
        self.assertFalse(torch.equal(clouds["paired_oracle"], clouds["permuted_oracle"]))


class DiagnosticCliTests(unittest.TestCase):
    def test_supervised_and_scvelo_default_to_the_input_screen(self):
        from tools.chromatin_r2_diagnostics import build_parser
        parser = build_parser()
        supervised = parser.parse_args(["supervised"])
        scvelo = parser.parse_args(["scvelo"])
        self.assertEqual(supervised.transforms, list(INPUT_SCREEN_TRANSFORMS))
        self.assertEqual(scvelo.transforms, list(INPUT_SCREEN_TRANSFORMS))
        self.assertEqual(supervised.eval_split, "val")
        self.assertEqual(scvelo.eval_split, "val")

    def test_manifest_and_summary_are_on_the_same_cli(self):
        from tools.chromatin_r2_diagnostics import DEV_JOB_FILES, build_parser
        parser = build_parser()
        manifest = parser.parse_args(["manifest"])
        summary = parser.parse_args(["summary"])
        self.assertEqual(manifest.jobs, list(DEV_JOB_FILES))
        self.assertEqual(summary.jobs, list(DEV_JOB_FILES))


class StateMetricsTests(unittest.TestCase):
    def test_partner_diversity_flags_a_hub(self):
        observed = torch.eye(6)
        identity = state_metrics(observed, observed, torch.arange(6))
        collapsed = state_metrics(torch.zeros_like(observed), observed, torch.arange(6))
        self.assertEqual(identity["partner_diversity"], 1.0)
        self.assertLess(collapsed["partner_diversity"], identity["partner_diversity"])


class IdentityPhi(torch.nn.Module):
    def forward(self, chromatin):
        return chromatin


class ClockedLaw:
    """Identity phi so the JVP equals the supplied velocity."""

    def __init__(self, kappa_value, alpha, beta, gamma):
        self.phi = IdentityPhi()
        self.kappa_value = float(kappa_value)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def kappa(self, chromatin):
        return chromatin.new_full((len(chromatin), 1), self.kappa_value)

    def g(self, production):
        return self.alpha.expand(len(production), -1)


def orthogonal_to(vector: torch.Tensor) -> torch.Tensor:
    axis = torch.zeros_like(vector)
    axis[..., 0] = 1.0
    projected = ((axis * vector).sum(dim=-1, keepdim=True)
                 / vector.pow(2).sum(dim=-1, keepdim=True).clamp(min=1e-12))
    return axis - projected * vector


class KineticsDirectionTests(unittest.TestCase):
    alpha = torch.tensor([[1.5, 0.8]])
    beta = torch.tensor([0.4, 0.5])
    gamma = torch.tensor([0.2, 0.3])

    def parts(self, kappa_value, velocity, prediction, alpha=None):
        alpha = self.alpha if alpha is None else alpha
        chromatin = prediction.repeat(len(velocity), 1) if len(prediction) == 1 else prediction
        tangent = velocity.expand_as(chromatin)
        production = torch.zeros(len(chromatin), 2)
        model = ClockedLaw(kappa_value, alpha, self.beta, self.gamma)
        return kinetics_losses(
            model, RELAY, chromatin, tangent, production,
            torch.ones(len(chromatin)), torch.ones(chromatin.shape[1]),
            torch.ones(chromatin.shape[1]), residual_weight=1.0, direction_weight=1.0)

    def test_relay_rhs_is_kappa_times_the_unclocked_flux(self):
        prediction = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
        kappa = torch.tensor([[0.01]])
        flux = relay_flux(prediction, self.alpha, self.beta, self.gamma)
        self.assertTrue(torch.allclose(
            relay_rhs(prediction, kappa, self.alpha, self.beta, self.gamma), kappa * flux))

    def test_shrinking_kappa_lowers_l2_but_not_direction_when_the_jvp_is_orthogonal(self):
        prediction = torch.tensor([[0.2, 0.4, 0.3, 0.5]])
        velocity = orthogonal_to(relay_flux(prediction, self.alpha, self.beta, self.gamma))
        wide = self.parts(1.0, velocity, prediction)
        collapsed = self.parts(0.001, velocity, prediction)
        self.assertGreater(float(wide["residual"]), 10 * float(collapsed["residual"]))
        self.assertAlmostEqual(float(wide["direction"]), float(collapsed["direction"]), places=5)
        self.assertGreater(float(collapsed["direction"]), 0.9)

    def test_a_zero_flux_and_tiny_jvp_zeros_l2_but_leaves_direction_unfit(self):
        u_hat = torch.tensor([[0.5, 0.6]])
        u = torch.expm1(u_hat)
        alpha = self.beta * u
        s_hat = torch.log1p((self.beta * u) / self.gamma)
        prediction = torch.cat([u_hat, s_hat], dim=1)
        parts = self.parts(0.001, torch.full((1, 4), 1e-8), prediction, alpha=alpha)
        self.assertLess(float(parts["residual"]), 1e-10)
        self.assertGreater(float(parts["direction"]), 0.9)

    def test_direction_term_does_not_train_kappa(self):
        kappa_raw = torch.nn.Parameter(torch.tensor([0.0]))
        chromatin = torch.tensor([[0.2, 0.4, 0.3, 0.5], [0.5, 0.1, 0.4, 0.2]])
        production = torch.zeros(2, 2)
        velocity = torch.randn(2, 4)
        scale = torch.ones(4)
        mask = torch.ones(4)
        confidence = torch.ones(2)

        class Head:
            phi = IdentityPhi()
            beta = self.beta
            gamma = self.gamma

            def kappa(self, chromatin):
                return torch.sigmoid(kappa_raw).expand(len(chromatin), 1)

            def g(self, production):
                return KineticsDirectionTests.alpha.expand(len(production), -1)

        direction = kinetics_loss(
            Head(), RELAY, chromatin, velocity, production, confidence, scale, mask,
            residual_weight=0.0, direction_weight=1.0)
        direction.backward()
        self.assertLess(float(kappa_raw.grad.abs()), 1e-8)
        kappa_raw.grad = None
        residual = kinetics_loss(
            Head(), RELAY, chromatin, velocity, production, confidence, scale, mask,
            residual_weight=1.0, direction_weight=0.0)
        residual.backward()
        self.assertGreater(float(kappa_raw.grad.abs()), 1e-6)

    def test_both_weights_zero_is_refused(self):
        prediction = torch.tensor([[0.2, 0.4, 0.3, 0.5]])
        with self.assertRaises(ValueError):
            kinetics_losses(
                ClockedLaw(1.0, self.alpha, self.beta, self.gamma),
                RELAY, prediction, torch.ones_like(prediction), torch.zeros(1, 2),
                torch.ones(1), torch.ones(4), torch.ones(4),
                residual_weight=0.0, direction_weight=0.0)


class ChromatinR2DefaultsTests(unittest.TestCase):
    def test_yaml_uses_the_r2lsi_residual_and_selects_pairing(self):
        config = load_yaml(Path(__file__).resolve().parents[1] / "config" / "training.yaml")
        r2 = config["chromatin_r2"]
        self.assertEqual(r2["dyn_residual_weight"], 1.0)
        self.assertEqual(r2["dyn_direction_weight"], 0.0)
        self.assertEqual(r2["checkpoint_monitor"], "foscttm")
        protein = config["defaults"]
        self.assertEqual(r2["kot_kappa_min"], protein["kot_kappa_min"])
        self.assertEqual(r2["kot_kappa_max"], protein["kot_kappa_max"])
        self.assertNotIn("dyn_direction_weight", protein)

    def test_regularisers_are_not_inherited_from_the_protein_yaml(self):
        for name in ("lambda_reg", "lambda_kappa_prior", "kappa_prior_target", "grad_clip"):
            self.assertNotIn(name, TRAINING_YAML_KEYS)

    def test_unset_flags_keep_the_legacy_l2_loss(self):
        args = argparse.Namespace(
            dyn_residual_weight=None, dyn_direction_weight=None, checkpoint_monitor=None)
        resolve_r2_training_flags(args)
        self.assertEqual(args.dyn_residual_weight, 1.0)
        self.assertEqual(args.dyn_direction_weight, 0.0)
        self.assertEqual(args.checkpoint_monitor, "val_align")

    def test_zero_weights_are_rejected(self):
        args = argparse.Namespace(
            dyn_residual_weight=0.0, dyn_direction_weight=0.0, checkpoint_monitor="foscttm")
        with self.assertRaises(ValueError):
            resolve_r2_training_flags(args)

    def test_lag_is_refused_on_a_corrupted_velocity(self):
        args = argparse.Namespace(
            dyn_residual_weight=0.0, dyn_direction_weight=1.0, checkpoint_monitor="foscttm",
            phi_lag_tau=1.0, condition="shuffle")
        with self.assertRaises(ValueError):
            resolve_r2_training_flags(args)

    def test_unset_regularisers_are_the_legacy_chromatin_trainer(self):
        args = argparse.Namespace(
            dyn_residual_weight=0.0, dyn_direction_weight=1.0, checkpoint_monitor="foscttm")
        resolve_r2_training_flags(args)
        self.assertEqual(args.stabilization, "legacy")
        self.assertEqual(args.lambda_kappa_prior, 0.0)
        self.assertEqual(args.lambda_reg, 0.0)
        self.assertEqual(args.grad_clip, 5.0)
        self.assertIsNone(args.kappa_prior_target)


class ChromatinStabilizationTests(unittest.TestCase):
    """kot_parity is the RNA→protein regularisers; legacy is the chromatin trainer."""

    def parsed(self, extra):
        args = build_parser().parse_args(
            ["train", "--dataset", "bmmc", "--gamma-anchor-csv", "unused.csv", *extra])
        resolve_r2_training_flags(args)
        return args

    def test_kot_parity_copies_the_rna_protein_regularisers(self):
        args = self.parsed(["--stabilization", "kot_parity"])
        protein = load_yaml(
            Path(__file__).resolve().parents[1] / "config" / "training.yaml")["defaults"]
        self.assertEqual(args.stabilization, "kot_parity")
        self.assertEqual(args.lambda_kappa_prior, protein["lambda_kappa_prior"])
        self.assertEqual(args.lambda_reg, protein["lambda_reg"])
        self.assertEqual(args.grad_clip, 1.0)
        self.assertAlmostEqual(args.kappa_prior_target, protein["kot_kappa_prior"])
        self.assertAlmostEqual(args.kappa_prior_target, KOT_KAPPA_PRIOR_TARGET)
        self.assertAlmostEqual(KOT_KAPPA_PRIOR_TARGET, math.log(2))

    def test_explicit_clip_overrides_the_mode(self):
        args = self.parsed(["--stabilization", "kot_parity", "--grad-clip", "5"])
        self.assertEqual(args.grad_clip, 5.0)
        self.assertEqual(args.lambda_reg, 1.0e-4)

    def test_weight_decay_skips_the_rate_parameters(self):
        class Heads(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.phi = torch.nn.Linear(2, 2, bias=False)
                self.kappa = torch.nn.Linear(2, 1, bias=False)
                self.g = torch.nn.Linear(2, 2, bias=False)
                self.gamma_raw = torch.nn.Parameter(torch.ones(2))
                self.beta_raw = torch.nn.Parameter(torch.ones(2))

        model = Heads()
        loss = sum(param.pow(2).sum() for param in chromatin_network_params(model))
        loss.backward()
        self.assertIsNotNone(model.phi.weight.grad)
        self.assertIsNone(model.gamma_raw.grad)
        self.assertIsNone(model.beta_raw.grad)

    def test_unknown_mode_is_refused(self):
        with self.assertRaises(ValueError):
            stabilization_settings("not-a-mode")

    def test_fixed_kappa_zeros_the_prior_and_is_log2(self):
        args = self.parsed([
            "--stabilization", "kot_parity", "--fixed-kappa", str(math.log(2))])
        self.assertAlmostEqual(args.fixed_kappa, math.log(2))
        self.assertEqual(args.lambda_kappa_prior, 0.0)
        self.assertIsNone(args.kappa_prior_target)

    def test_a_nonpositive_fixed_kappa_is_refused(self):
        with self.assertRaises(ValueError):
            self.parsed(["--stabilization", "kot_parity", "--fixed-kappa", "0"])

    def test_fixed_kappa_emits_a_constant_and_has_no_parameters(self):
        model = ChromatinKOT(
            2, RELAY, n_input_features=2, phi_dims=[4], kappa_dims=[4], g_dims=[4],
            phi_spectral_norm=False)
        model.kappa = FixedKappa(math.log(2))
        self.assertEqual(list(model.kappa.parameters()), [])
        out = model.kappa(torch.randn(5, 2))
        self.assertEqual(tuple(out.shape), (5, 1))
        self.assertTrue(torch.allclose(out, out.new_full(out.shape, math.log(2))))
        groups = chromatin_param_groups(model, 1e-3, None, None, None)
        self.assertTrue(all(group["params"] for group in groups))
        torch.optim.Adam(groups)


class InputLambdaScreenTests(unittest.TestCase):
    """The BMMC screen is kot_parity; legacy is only λ∈{1,9} on B/C/E/F."""

    def payloads(self):
        path = Path(__file__).resolve().parents[1] / "jobs" / "jobs_chromatin_input_lambda_screen.txt"
        return [line for line in path.read_text().splitlines()
                if line and not line.startswith("#")]

    def trains(self, mode=None):
        rows = [line for line in self.payloads() if line.startswith("train ")]
        if mode is None:
            return rows
        return [line for line in rows if f"--stabilization {mode} " in line]

    def test_six_velocity_caches_and_no_hspc(self):
        velocities = [line for line in self.payloads() if line.startswith("velocity ")]
        self.assertEqual(len(velocities), 6)
        self.assertTrue(all("--velocity-tag scr" in line for line in velocities))
        self.assertTrue(all("--dataset bmmc" in line for line in self.payloads()))

    def test_primary_screen_is_thirty_four_kot_parity_runs(self):
        primary = [line for line in self.trains("kot_parity")
                   if "--regulatory-transform cp10k_linear" not in line]
        self.assertEqual(len(primary), 34)
        self.assertTrue(all("--dyn-residual-weight 1" in line for line in primary))
        self.assertTrue(all("--dyn-direction-weight 0" in line for line in primary))
        self.assertTrue(all("--kappa-min 0.001" in line and "--kappa-max 1.5" in line
                            for line in primary))

    def test_decoupled_alpha_is_nine_kot_parity_runs(self):
        decoupled = [line for line in self.trains("kot_parity")
                     if "--regulatory-transform cp10k_linear" in line]
        self.assertEqual(len(decoupled), 9)
        transforms = []
        for line in decoupled:
            self.assertRegex(line, r"--lambda-dyn (1|9|30) ")
            self.assertIn("_reglin_seed42", line)
            self.assertIn("--stabilization kot_parity", line)
            transforms.append(line.split("--chromatin-transform ")[1].split()[0])
        self.assertEqual(
            sorted(set(transforms)),
            ["cp10k_log1p", "tfidf_gene", "tfidf_lsi"])
        self.assertEqual(len(transforms), 9)

    def test_legacy_is_eight_runs_at_lambda_one_and_nine(self):
        legacy = self.trains("legacy")
        self.assertEqual(len(legacy), 8)
        transforms = []
        for line in legacy:
            self.assertRegex(line, r"--lambda-dyn (1|9) ")
            self.assertIn("_legacy_seed42", line)
            transforms.append(line.split("--chromatin-transform ")[1].split()[0])
        self.assertEqual(
            sorted(transforms),
            ["cp10k_linear", "cp10k_linear", "cp10k_log1p", "cp10k_log1p",
             "tfidf_gene", "tfidf_gene", "tfidf_lsi", "tfidf_lsi"])
        self.assertFalse(any("as_is" in line or "l2_per_cell" in line for line in legacy))

    def test_legacy_and_kot_parity_do_not_share_run_directories(self):
        dirs = [line.split("--run-dir ")[1] for line in self.trains()]
        self.assertEqual(len(dirs), len(set(dirs)))
        self.assertEqual(len(self.trains()), 51)
        self.assertTrue(all("--dyn-residual-weight 1" in line for line in self.trains()))
        self.assertTrue(all("--dyn-direction-weight 0" in line for line in self.trains()))
        self.assertTrue(all("--kappa-min 0.001" in line and "--kappa-max 1.5" in line
                            for line in self.trains()))


class VelocityEstimatorTests(unittest.TestCase):
    """A near-tied pseudotime gap must not decide a cell's velocity direction."""

    def field(self, estimator):
        # Three forward neighbours. Two agree on the true direction over healthy gaps; the
        # third is nearly tau-tied and points the opposite way, so its difference quotient
        # is ~100x larger than theirs.
        activity = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [-0.02, 0.0]],
                            dtype=np.float32)
        # 2e-5 against a 1e-2 scale: the ratio measured in the real field, where tau gaps
        # run from 1.4e-4 to a 9.2e-3 median and the 1e-6 guard lets the small ones pass.
        pseudotime = np.array([0.0, 0.01, 0.02, 0.00002], dtype=np.float32)
        connectivity = sparse.csr_matrix(np.array([
            [0, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.float32))
        velocity, _, _ = forward_difference_field(
            activity, connectivity, pseudotime, min_forward=1, estimator=estimator)
        return velocity[0]

    def test_the_quotient_estimator_is_captured_by_the_tied_neighbour(self):
        self.assertLess(self.field("quotient")[0], 0.0)

    def test_the_regression_estimator_follows_the_healthy_neighbours(self):
        self.assertGreater(self.field("regression")[0], 0.0)

    def test_who_has_forward_neighbours_depends_on_the_graph_not_the_coordinates(self):
        connectivity = sparse.csr_matrix(np.array([
            [0, 1, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.float32))
        pseudotime = np.array([0.0, 0.01, 0.02, 0.03], dtype=np.float32)
        ones = np.ones((4, 2), dtype=np.float32)
        other = np.arange(8, dtype=np.float32).reshape(4, 2)
        _, _, n_ones = forward_difference_field(ones, connectivity, pseudotime, min_forward=1)
        _, _, n_other = forward_difference_field(other, connectivity, pseudotime, min_forward=1)
        np.testing.assert_array_equal(n_ones, n_other)

    def test_both_estimators_are_offered(self):
        self.assertEqual(VELOCITY_ESTIMATORS, ["quotient", "regression"])


class AlphaInputTests(unittest.TestCase):
    def tiny(self, alpha_input: str = "gc", n_input: int = 6, n_genes: int = 3):
        return ChromatinKOT(
            n_genes, RELAY, n_input_features=n_input, alpha_input=alpha_input,
            phi_dims=[8], kappa_dims=[4], g_dims=[4],
            phi_spectral_norm=False, activation="silu", init_method="xavier")

    def test_full_lets_alpha_read_the_whole_chromatin_vector(self):
        model = self.tiny("full")
        chromatin, production = torch.randn(5, 6), torch.randn(5, 3)
        self.assertEqual(tuple(alpha_features(model, chromatin, production).shape), (5, 6))
        self.assertEqual(tuple(model.g(chromatin).shape), (5, 3))

    def test_gc_keeps_alpha_on_the_selected_genes(self):
        model = self.tiny("gc")
        chromatin, production = torch.randn(5, 6), torch.randn(5, 3)
        self.assertEqual(tuple(alpha_features(model, chromatin, production).shape), (5, 3))
        self.assertEqual(tuple(model.g(production).shape), (5, 3))

    def test_a_full_checkpoint_cannot_load_into_a_gc_head(self):
        trained = self.tiny("full")
        wrong = self.tiny("gc")
        with self.assertRaises(RuntimeError):
            wrong.load_state_dict(trained.state_dict())
        right = self.tiny("full")
        right.load_state_dict(trained.state_dict())


class EulerLagTests(unittest.TestCase):
    def test_zero_tau_is_the_instantaneous_map(self):
        chromatin = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        velocity = np.ones_like(chromatin)
        np.testing.assert_array_equal(euler_lag(chromatin, velocity, 0.0), chromatin)

    def test_positive_tau_steps_backward_along_the_field(self):
        chromatin = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        velocity = np.array([[0.5, 0.0], [0.0, 1.0]], dtype=np.float32)
        np.testing.assert_allclose(euler_lag(chromatin, velocity, 2.0),
                                   np.array([[0.0, 2.0], [3.0, 2.0]], dtype=np.float32))

    def test_a_zero_velocity_row_is_left_unchanged(self):
        chromatin = np.array([[1.0, 1.0], [9.0, 9.0]], dtype=np.float32)
        velocity = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        lagged = euler_lag(chromatin, velocity, 1.0)
        np.testing.assert_allclose(lagged[0], chromatin[0])
        np.testing.assert_allclose(lagged[1], [8.0, 8.0])


class GenomicProjectionTests(unittest.TestCase):
    def loci(self):
        return pd.DataFrame({
            "chrom": ["chr1", "chr1", "chr2"],
            "start": [0.0, 10_000.0, 0.0],
            "end": [100.0, 10_100.0, 100.0],
        }, index=["A", "B", "C"])

    def curated(self):
        return sparse.csr_matrix(np.eye(3, dtype=np.float32))

    def test_nearby_same_chromosome_genes_mix_and_far_or_other_chrom_do_not(self):
        genes = pd.Index(["A", "B", "C"])
        mapping = genomic_projection(
            self.curated(), genes, genes, self.loci(), max_bp=50_000, decay_bp=50_000)
        dense = mapping.toarray()
        self.assertGreater(dense[0, 1], 0.0)
        self.assertGreater(dense[1, 0], 0.0)
        self.assertEqual(dense[0, 2], 0.0)
        self.assertEqual(dense[2, 0], 0.0)
        np.testing.assert_allclose(dense.sum(axis=1), 1.0, atol=1e-6)

    def test_empty_curated_rows_stay_empty(self):
        projection = sparse.csr_matrix(np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]],
                                                dtype=np.float32))
        genes = pd.Index(["A", "B", "C"])
        mapping = genomic_projection(
            projection, genes, genes, self.loci(), max_bp=50_000, decay_bp=50_000)
        self.assertEqual(mapping[1].nnz, 0)

    def test_peak_genomic_adds_a_distal_peak_that_overlaps_another_gene(self):
        genes = pd.Index(["A", "B", "C"])
        annotation = pd.DataFrame({
            "chrom": ["chr1", "chr1"],
            "start": [10_020, 0],
            "end": [10_150, 50],
            "gene": ["A", "A"],
            "distance": [10_000.0, 0.0],
            "peak_type": ["distal", "promoter"],
        })
        mapping = peak_genomic_projection(
            self.curated(), genes, genes, self.loci(), annotation, decay_bp=50_000)
        dense = mapping.toarray()
        # Distal peak at 10020-10150 sits on B and is assigned to A; self-overlap
        # of A's promoter is skipped so identity is not drowned by peak count.
        self.assertGreater(dense[0, 1], 0.0)
        self.assertGreater(dense[0, 0], dense[0, 1])
        self.assertEqual(dense[1, 0], 0.0)
        np.testing.assert_allclose(dense[0].sum(), 1.0, atol=1e-6)

    def test_peak_genomic_empty_curated_rows_stay_empty(self):
        projection = sparse.csr_matrix(np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]],
                                                dtype=np.float32))
        genes = pd.Index(["A", "B", "C"])
        annotation = pd.DataFrame({
            "chrom": ["chr1"], "start": [10_020], "end": [10_150],
            "gene": ["B"], "distance": [0.0], "peak_type": ["distal"],
        })
        mapping = peak_genomic_projection(
            projection, genes, genes, self.loci(), annotation)
        self.assertEqual(mapping[1].nnz, 0)

    def test_widen_dispatches_genomic_without_activity(self):
        genes = pd.Index(["A", "B", "C"])
        mapping = widen_projection(
            self.curated(), None, "genomic", output_genes=genes, feature_names=genes,
            loci=self.loci(), genomic_bp=50_000, genomic_decay_bp=50_000)
        self.assertGreater(mapping.nnz, 3)

    def test_peak_genomic_is_refused_without_annotation(self):
        with self.assertRaises(ValueError):
            widen_projection(self.curated(), None, "peak-genomic",
                             output_genes=pd.Index(["A", "B", "C"]),
                             feature_names=pd.Index(["A", "B", "C"]),
                             loci=self.loci())


class EvaluationSplitTests(unittest.TestCase):
    def test_val_and_test_drop_uncovered_cells(self):
        split = np.array(["train", "val", "val", "test"])
        covered = np.array([True, True, False, True])
        np.testing.assert_array_equal(evaluation_split_rows(split, covered, "val"), [1])
        np.testing.assert_array_equal(evaluation_split_rows(split, covered, "test"), [3])

    def test_train_is_not_an_evaluation_split(self):
        with self.assertRaisesRegex(ValueError, "val or test"):
            evaluation_split_rows(np.array(["train", "val"]), np.array([True, True]), "train")

    def test_empty_split_is_refused(self):
        with self.assertRaisesRegex(ValueError, "No measured val"):
            evaluation_split_rows(np.array(["train", "test"]), np.array([True, False]), "val")

    def test_val_outputs_do_not_clobber_legacy_test_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            tagged = write_evaluation_outputs(
                run_dir, "best_align", "val", {"eval_split": "val"}, pd.DataFrame())
            self.assertEqual(tagged.name, "evaluation_best_align_val.json")
            self.assertTrue((run_dir / "task_a_per_gene_best_align_val.csv").exists())
            self.assertFalse((run_dir / "evaluation_best_align.json").exists())
            self.assertFalse((run_dir / "task_a_per_gene_best_align.csv").exists())

    def test_test_outputs_keep_legacy_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            write_evaluation_outputs(
                run_dir, "best_align", "test", {"eval_split": "test"}, pd.DataFrame())
            self.assertTrue((run_dir / "evaluation_best_align_test.json").exists())
            self.assertTrue((run_dir / "evaluation_best_align.json").exists())
            self.assertTrue((run_dir / "task_a_per_gene_best_align.csv").exists())


class ProductionFromPhiTests(unittest.TestCase):
    def test_uses_the_stored_projection_not_the_csv_fallback(self):
        projection = torch.tensor([[1.0, 0.0], [0.5, 0.5]], dtype=torch.float32)
        scale = torch.ones(4)
        bias = torch.zeros(4)
        model = ChromatinKOT(
            2, RELAY, n_input_features=2, phi_dims=[8], kappa_dims=[4], g_dims=[4],
            phi_projection=torch.cat([projection, projection], 0),
            phi_scale=scale, phi_bias=bias, phi_spectral_norm=False)
        chromatin = np.array([[2.0, 0.0], [0.0, 4.0]], dtype=np.float32)
        fallback = sparse.csr_matrix(np.eye(2, dtype=np.float32))
        production = production_from_phi(
            model, chromatin, np.array([0, 1]), torch.device("cpu"), 2, fallback)
        np.testing.assert_allclose(production.numpy()[1], [0.0, 2.0])


if __name__ == "__main__":
    unittest.main()
