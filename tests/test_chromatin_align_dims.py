"""Three align_dims trains on one winner; blur alternatives only when 0.1 is unusable."""
from __future__ import annotations

import math
import unittest

import pandas as pd

from tools.chromatin_align_dims import (
    ALIGN_DIMS, CURRENT_BLUR, CURRENT_DIMS, MAX_BLUR_ALTERNATIVES,
    align_dim_jobs, assess_blur, dim_overrides, job_file_text,
)


WINNER_CONFIG = {
    "dataset": "bmmc",
    "law": "relay",
    "condition": "full",
    "seed": 42,
    "split_seed": 0,
    "lambda_dyn": 30.0,
    "align_block": "spliced",
    "lambda_held_block": 1.0,
    "gamma_anchor_csv": "config/gamma_anchors_k562_timelapse.csv",
    "lambda_gamma_anchor": 1.0,
    "chromatin_transform": "tfidf_lsi",
    "regulatory_transform": "cp10k_linear",
    "velocity_tag": "scr",
    "dyn_residual_weight": 1.0,
    "dyn_direction_weight": 0.0,
    "kappa_min": 0.001,
    "kappa_max": 1.5,
    "checkpoint_monitor": "foscttm",
    "stabilization": "kot_parity",
    "phi_gate": "scalar-zero",
    "lambda_kappa_prior": 0.01,
    "kappa_prior_target": math.log(2),
    "lambda_reg": 1.0e-4,
    "grad_clip": 1.0,
    "align_dims": 32,
    "sinkhorn_blur": 0.1,
}


def winner():
    return {
        "run": "r2scr_f_lsi_lam30_reglin_seed42",
        "picked_for": "alignment",
        "foscttm": 0.227,
        "config": dict(WINNER_CONFIG),
    }


def geometry_row(dims, blur, *, advantage, blur_dist, beats=True):
    return {
        "align_dims": dims, "blur": blur, "oracle_beats_constant": beats,
        "oracle_advantage": advantage, "blur_over_pair_distance": blur_dist,
        "neighbour_preservation": 0.1, "explained_variance": 0.1,
    }


class DimLadderTests(unittest.TestCase):
    def test_three_new_ranks_not_four(self):
        self.assertEqual(ALIGN_DIMS, (8, 16, 32, 64))
        self.assertEqual([spec["tag"] for spec in dim_overrides(CURRENT_DIMS)],
                         ["d8", "d16", "d64"])

    def test_usable_blur_writes_only_the_three_dim_trains(self):
        decision = {"keep": True, "alternatives": [],
                    "reason": "dims 32 blur 0.1: oracle advantage 4.6x, blur/dist 0.31"}
        trains = align_dim_jobs(winner(), decision)
        self.assertEqual(len(trains), 3)
        joined = "\n".join(trains)
        self.assertIn("--align-dims 8", joined)
        self.assertIn("--align-dims 16", joined)
        self.assertIn("--align-dims 64", joined)
        self.assertNotIn("--align-dims 32", joined)
        self.assertTrue(all("--sinkhorn-blur 0.1" in line for line in trains))
        self.assertTrue(all("--chromatin-transform tfidf_lsi" in line for line in trains))
        self.assertTrue(all("--regulatory-transform cp10k_linear" in line for line in trains))
        self.assertTrue(all("--lambda-dyn 30 " in line for line in trains))
        self.assertTrue(all("r2dim_f_lsi_lam30_reglin_d" in line for line in trains))
        self.assertNotIn("--eval-split test", joined)

    def test_unusable_blur_adds_at_most_two_alternatives_at_the_winner_rank(self):
        decision = {"keep": False, "alternatives": [0.2, 0.05],
                    "reason": "not usable"}
        trains = align_dim_jobs(winner(), decision)
        self.assertEqual(len(trains), 5)
        self.assertEqual(MAX_BLUR_ALTERNATIVES, 2)
        blur_lines = [line for line in trains if "_b" in line.split("--run-dir")[1]]
        self.assertEqual(len(blur_lines), 2)
        self.assertTrue(all("--align-dims 32" in line for line in blur_lines))
        self.assertTrue(any("--sinkhorn-blur 0.2" in line for line in blur_lines))
        self.assertTrue(any("--sinkhorn-blur 0.05" in line for line in blur_lines))


class BlurAssessmentTests(unittest.TestCase):
    def test_current_blur_in_band_is_left_alone(self):
        frame = pd.DataFrame([
            geometry_row(32, 0.1, advantage=4.6, blur_dist=0.31),
            geometry_row(32, 0.2, advantage=9.3, blur_dist=0.62),
        ])
        decision = assess_blur(frame)
        self.assertTrue(decision["keep"])
        self.assertEqual(decision["alternatives"], [])
        self.assertEqual(CURRENT_BLUR, 0.1)

    def test_picks_at_most_two_usable_replacements(self):
        frame = pd.DataFrame([
            geometry_row(32, 0.1, advantage=1.05, blur_dist=1.6),
            geometry_row(32, 0.5, advantage=8.0, blur_dist=1.6, beats=True),
            geometry_row(32, 0.2, advantage=9.0, blur_dist=0.30),
            geometry_row(32, 0.05, advantage=3.0, blur_dist=0.16),
            geometry_row(32, 0.02, advantage=2.7, blur_dist=0.06),
        ])
        decision = assess_blur(frame)
        self.assertFalse(decision["keep"])
        self.assertEqual(len(decision["alternatives"]), 2)
        self.assertIn(0.2, decision["alternatives"])

    def test_job_file_names_the_winner_and_not_test(self):
        decision = {"keep": True, "alternatives": [], "reason": "usable"}
        text = job_file_text(winner(), decision)
        self.assertIn("r2scr_f_lsi_lam30_reglin_seed42", text)
        self.assertNotIn("--eval-split test", text)
        self.assertEqual(sum(line.startswith("train ") for line in text.splitlines()), 3)


if __name__ == "__main__":
    unittest.main()
