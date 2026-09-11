"""The development manifest/summary reuse the trainer parser and existing gate files."""
from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch

from tools.chromatin_r2_diagnostics import (
    DEV_JOB_FILES, GEOMETRY_COLUMNS, MANIFEST_COLUMNS, SUMMARY_COLUMNS,
    attach_geometry, checkpoint_probe, history_at, intended_row, kappa_mode,
    load_preflight, run_status, summary_rows,
)


TRAIN = (
    "train --dataset bmmc --law relay --condition full --seed 42 --split-seed 0 "
    "--lambda-dyn 30 --align-block spliced --lambda-held-block 1 "
    "--gamma-anchor-csv config/gamma_anchors_k562_timelapse.csv "
    "--lambda-gamma-anchor 1 --chromatin-transform tfidf_lsi "
    "--regulatory-transform cp10k_linear --stabilization kot_parity "
    "--phi-gate per-gene --kappa-min 0.001 --kappa-max 1.5 "
    "--run-dir {run_dir}"
)


def train_line(run_dir: Path, extra: str = "") -> str:
    return TRAIN.format(run_dir=run_dir) + ((" " + extra) if extra else "")


class KappaModeTests(unittest.TestCase):
    def test_fixed_pinned_and_open(self):
        self.assertEqual(kappa_mode({"fixed_kappa": 0.693, "kappa_min": 0.001,
                                     "kappa_max": 1.5}), "fixed")
        self.assertEqual(kappa_mode({"fixed_kappa": None, "kappa_min": 0.5,
                                     "kappa_max": 0.5}), "pinned")
        self.assertEqual(kappa_mode({"fixed_kappa": None, "kappa_min": 0.001,
                                     "kappa_max": 1.5}), "open")


class ManifestRowTests(unittest.TestCase):
    def test_job_line_fills_intended_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "r2scr_example"
            row = intended_row(train_line(run), sinkhorn_blur=0.1)
        self.assertEqual(row["run_id"], "r2scr_example")
        self.assertEqual(row["map_transform"], "tfidf_lsi")
        self.assertEqual(row["regulatory_transform"], "cp10k_linear")
        self.assertEqual(row["rna_kinetic_coords"], "shared_log1p")
        self.assertEqual(row["lambda_dyn"], 30.0)
        self.assertEqual(row["stabilization"], "kot_parity")
        self.assertEqual(row["kappa_mode"], "open")
        self.assertEqual(row["phi_gate"], "per-gene")
        self.assertEqual(row["status"], "missing")
        self.assertEqual(row["sinkhorn_blur"], 0.1)
        self.assertEqual(row["align_dims"], 32)

    def test_nodyn_records_lambda_zero_even_if_the_flag_is_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "nodyn"
            line = train_line(run).replace("--condition full", "--condition noDyn")
            row = intended_row(line, sinkhorn_blur=0.1)
        self.assertEqual(row["lambda_dyn"], 0.0)

    def test_fixed_kappa_on_the_command_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "kfixed"
            row = intended_row(
                train_line(run, "--fixed-kappa 0.69314718056"), sinkhorn_blur=0.1)
        self.assertEqual(row["kappa_mode"], "fixed")

    def test_status_tracks_checkpoint_then_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            self.assertEqual(run_status(run), "missing")
            (run / "run_config.json").write_text("{}")
            self.assertEqual(run_status(run), "incomplete")
            (run / "checkpoint_best_align.pt").write_bytes(b"x")
            self.assertEqual(run_status(run), "trained")
            (run / "preflight.json").write_text("{}")
            self.assertEqual(run_status(run), "preflight_failed")
            (run / "preflight_passed.json").write_text("{}")
            self.assertEqual(run_status(run), "preflight_passed")


class GeometryAttachTests(unittest.TestCase):
    def test_dim_job_keeps_rank_and_joins_usable_blur(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "r2dim_f_lsi_lam30_reglin_d8_seed42"
            line = train_line(run, "--align-dims 8 --sinkhorn-blur 0.1")
            identity = intended_row(line, sinkhorn_blur=0.1)
        self.assertEqual(identity["align_dims"], 8)
        self.assertEqual(identity["sinkhorn_blur"], 0.1)
        geometry = pd.DataFrame([
            {"align_dims": 8, "blur": 0.1, "oracle_beats_constant": True,
             "oracle_advantage": 26.84, "blur_over_pair_distance": 0.42,
             "explained_variance": 0.067, "neighbour_preservation": 0.034},
            {"align_dims": 32, "blur": 0.1, "oracle_beats_constant": True,
             "oracle_advantage": 5.87, "blur_over_pair_distance": 0.32,
             "explained_variance": 0.107, "neighbour_preservation": 0.083},
        ])
        attached = attach_geometry(pd.DataFrame([identity]), geometry).iloc[0]
        self.assertAlmostEqual(attached["oracle_advantage"], 26.84)
        self.assertTrue(attached["blur_usable"])
        missing = attach_geometry(
            pd.DataFrame([{"align_dims": 32, "sinkhorn_blur": 0.3}]), geometry)
        self.assertTrue(math.isnan(missing.loc[0, "oracle_advantage"]))
        self.assertTrue(math.isnan(missing.loc[0, "blur_usable"]))

    def test_default_jobs_include_the_dim_ladder(self):
        names = [path.name for path in DEV_JOB_FILES]
        self.assertEqual(names[-1], "jobs_chromatin_align_dims.txt")
        self.assertTrue(set(GEOMETRY_COLUMNS) <= set(MANIFEST_COLUMNS))
        self.assertTrue(set(GEOMETRY_COLUMNS) <= set(SUMMARY_COLUMNS))
        self.assertIn("align_dims", SUMMARY_COLUMNS)
        self.assertIn("sinkhorn_blur", SUMMARY_COLUMNS)


class HistoryAtTests(unittest.TestCase):
    def test_exact_epoch_binds_and_epoch_zero_does_not_take_the_first_eval(self):
        history = pd.DataFrame({"epoch": [1, 50], "grad_mag_ratio": [9.0, 2.0]})
        self.assertEqual(history_at(history, 50)["grad_mag_ratio"], 2.0)
        self.assertEqual(history_at(history, 0), {})
        self.assertEqual(history_at(history, 49.5), {})


class SummaryRowTests(unittest.TestCase):
    def test_best_align_reads_untagged_preflight_and_final_stays_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "example"
            run.mkdir()
            (run / "preflight.json").write_text(json.dumps({
                "foscttm": 0.22,
                "unspliced_foscttm": 0.18,
                "state_pearson_median": 0.05,
                "prediction_spread_ratio": 0.4,
                "jvp_vs_reference_cosine_centred_median": 0.08,
                "jvp_vs_reference_cosine_centred_null_p95": 0.03,
                "jvp_vs_reference_gene_pearson_median": 0.01,
                "jvp_rhs_cosine_median": 0.07,
                "residual_norm_median": 3.2,
                "kappa_median": 0.06,
                "kappa_at_floor": 0.0,
                "alpha_at_floor": 0.01,
            }))
            # Passed-marker pairing is not the launch-gate file for campaign tables.
            (run / "preflight_passed.json").write_text(json.dumps({"foscttm": 0.91}))
            torch.save({"epoch": 50, "state_dict": {}}, run / "checkpoint_best_align.pt")
            torch.save({"epoch": 500, "state_dict": {}}, run / "checkpoint_final.pt")
            # Last-eval pairing is not the gate. Binding the matching history
            # rows must still leave val_s_foscttm on the preflight numbers.
            pd.DataFrame({"epoch": [50, 500], "grad_mag_ratio": [2.0, 1.1],
                          "grad_cosine": [-0.2, 0.3],
                          "foscttm": [0.40, 0.99]}).to_csv(
                run / "training_loss.csv", index=False)
            rows = {row["checkpoint"]: row for row in summary_rows(run)}
        best, final = rows["best_align"], rows["final"]
        self.assertEqual(best["val_s_foscttm"], 0.22)
        self.assertEqual(best["grad_mag_ratio"], 2.0)
        self.assertAlmostEqual(best["jvp_scvelo_margin"], 0.05)
        self.assertTrue(math.isnan(final["val_s_foscttm"]))
        self.assertEqual(final["grad_mag_ratio"], 1.1)
        self.assertTrue(math.isnan(final["jvp_scvelo_centred"]))
        self.assertTrue(math.isnan(best["partner_diversity"]))

    def test_tagged_final_preflight_is_preferred(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "example"
            run.mkdir()
            (run / "preflight.json").write_text(json.dumps({"foscttm": 0.22}))
            (run / "preflight_final.json").write_text(json.dumps({"foscttm": 0.31}))
            rows = {row["checkpoint"]: row for row in summary_rows(run)}
            self.assertEqual(rows["best_align"]["val_s_foscttm"], 0.22)
            self.assertEqual(rows["final"]["val_s_foscttm"], 0.31)
            self.assertEqual(load_preflight(run, "final")["foscttm"], 0.31)

    def test_checkpoint_probe_reads_gamma_without_the_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            gamma_anchor = math.log(2)
            torch.save({
                "epoch": 120,
                "state_dict": {"gamma_raw": torch.zeros(2)},
                "gamma_anchors": {"indices": [0], "gamma": [gamma_anchor]},
            }, run / "checkpoint_best_align.pt")
            probe = checkpoint_probe(run, "best_align")
        self.assertEqual(probe["epoch"], 120)
        self.assertLess(probe["gamma_anchor_error"], 0.01)
