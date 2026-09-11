"""Two RNA-coordinate follow-ups on one winner, not a new screen."""
from __future__ import annotations

import math
import unittest

from src.losses.chromatin_laws import GLOBAL_LINEAR, PANEL_LOG1P_COMPOSITIONAL
from tools.chromatin_rna_coords import coord_overrides, job_file_text, rna_coord_jobs


WINNER_CONFIG = {
    "dataset": "bmmc",
    "law": "relay",
    "condition": "full",
    "seed": 42,
    "split_seed": 0,
    "lambda_dyn": 9.0,
    "align_block": "spliced",
    "lambda_held_block": 1.0,
    "gamma_anchor_csv": "config/gamma_anchors_k562_timelapse.csv",
    "lambda_gamma_anchor": 1.0,
    "chromatin_transform": "tfidf_lsi",
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
}


class RnaCoordinateJobTests(unittest.TestCase):
    def winner(self):
        return {
            "run": "r2scr_f_lsi_lam9_seed42",
            "picked_for": "alignment",
            "foscttm": 0.18,
            "config": {**WINNER_CONFIG, "chromatin_transform": "tfidf_lsi",
                       "lambda_dyn": 9.0},
        }

    def trains(self):
        return rna_coord_jobs(self.winner())

    def test_two_new_runs_not_three(self):
        self.assertEqual(len(coord_overrides()), 2)
        trains = self.trains()
        self.assertEqual(len(trains), 2)
        self.assertTrue(all(line.startswith("train ") for line in trains))
        self.assertTrue(any("--rna-kinetic-coords panel_log1p_compositional" in line
                            for line in trains))
        self.assertTrue(any("--rna-kinetic-coords global_linear" in line
                            for line in trains))
        self.assertFalse(any("--rna-kinetic-coords shared_log1p" in line for line in trains))
        self.assertTrue(all("r2rna_f_lsi_lam9_" in line for line in trains))

    def test_winner_identity_is_copied(self):
        for line in self.trains():
            self.assertIn("--chromatin-transform tfidf_lsi", line)
            self.assertIn("--lambda-dyn 9 ", line)
            self.assertIn("--velocity-tag scr", line)
            self.assertIn("--dataset bmmc", line)
            self.assertNotIn("--eval-split test", line)

    def test_job_file_names_the_winner_and_not_test(self):
        text = job_file_text(self.winner())
        self.assertIn("r2scr_f_lsi_lam9_seed42", text)
        self.assertNotIn("--eval-split test", text)
        self.assertEqual(PANEL_LOG1P_COMPOSITIONAL, "panel_log1p_compositional")
        self.assertEqual(GLOBAL_LINEAR, "global_linear")


if __name__ == "__main__":
    unittest.main()
