"""Validation Pareto selection and the 16 one-factor kinetic control jobs."""
from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from tools.chromatin_kinetic_controls import (
    KAPPA_BOX, KAPPA_FIXED, SPREAD_FLOOR, control_jobs, control_overrides,
    job_file_text, pick_winners, score_run, stem_from_winner,
)


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
    "regulatory_transform": "same",
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


def healthy_gate(**overrides):
    gate = {
        "foscttm": 0.22,
        "foscttm_permuted_floor": 0.5,
        "prediction_spread_ratio": 0.4,
        "state_pearson_median": 0.1,
        "jvp_vs_reference_cosine_centred_median": 0.08,
        "jvp_vs_reference_cosine_centred_null_p95": 0.03,
        "kappa_at_floor": 0.0,
        "alpha_at_floor": 0.0,
    }
    gate.update(overrides)
    return gate


def write_run(root: Path, name: str, config: dict, gate: dict,
              evaluation: dict | None = None) -> Path:
    run = root / name
    run.mkdir()
    (run / "run_config.json").write_text(json.dumps(config))
    (run / "preflight.json").write_text(json.dumps(gate))
    if evaluation is not None:
        (run / "evaluation_best_align_val.json").write_text(json.dumps(evaluation))
    return run


def scored(**kwargs):
    """A feasible Pareto row. Override individual objectives in kwargs."""
    foscttm = kwargs.pop("foscttm", 0.22)
    spread = kwargs.pop("spread", 0.4)
    jvp_margin = kwargs.pop("jvp_margin", 0.05)
    kappa_at_floor = kwargs.pop("kappa_at_floor", 0.0)
    alpha_at_floor = kwargs.pop("alpha_at_floor", 0.0)
    row = {
        "run": kwargs.pop("run", "r2scr_x"),
        "feasible": kwargs.pop("feasible", True),
        "foscttm": foscttm,
        "spread": spread,
        "jvp_margin": jvp_margin,
        "kappa_at_floor": kappa_at_floor,
        "alpha_at_floor": alpha_at_floor,
        "alignment": -foscttm,
        "spread_score": min(spread, 1.0),
        "rate_health": 1.0 - 0.5 * (kappa_at_floor + alpha_at_floor),
        "config": dict(WINNER_CONFIG),
    }
    row.update(kwargs)
    return row


class KineticSelectionTests(unittest.TestCase):
    def test_nodyn_collapsed_and_null_failing_runs_are_infeasible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nodyn = write_run(root, "r2scr_a_asis_lam0_seed42",
                              {**WINNER_CONFIG, "condition": "noDyn", "lambda_dyn": 0},
                              healthy_gate())
            collapsed = write_run(root, "r2scr_collapsed_seed42", WINNER_CONFIG,
                                  healthy_gate(prediction_spread_ratio=SPREAD_FLOOR / 2))
            null = write_run(root, "r2scr_null_seed42", WINNER_CONFIG,
                             healthy_gate(jvp_vs_reference_cosine_centred_median=0.01,
                                          jvp_vs_reference_cosine_centred_null_p95=0.04))
            floor = write_run(root, "r2scr_floor_seed42", WINNER_CONFIG,
                              healthy_gate(foscttm=0.51, foscttm_permuted_floor=0.5))
            missing = write_run(root, "r2scr_norates_seed42", WINNER_CONFIG,
                                {k: v for k, v in healthy_gate().items()
                                 if k not in ("kappa_at_floor", "alpha_at_floor")})
            self.assertIn("no dynamics", score_run(nodyn)["infeasible"])
            self.assertIn("collapsed spread", score_run(collapsed)["infeasible"])
            self.assertIn("JVP-vs-scVelo does not beat the permutation null",
                          score_run(null)["infeasible"])
            self.assertIn("FOSCTTM not below the permuted floor",
                          score_run(floor)["infeasible"])
            self.assertIn("missing rate-floor statistics",
                          score_run(missing)["infeasible"])
            kappa = write_run(root, "r2scr_kfloor_seed42", WINNER_CONFIG,
                              healthy_gate(kappa_at_floor=0.22))
            alpha = write_run(root, "r2scr_afloor_seed42", WINNER_CONFIG,
                              healthy_gate(alpha_at_floor=0.15))
            self.assertIn("kappa at floor", score_run(kappa)["infeasible"])
            self.assertIn("alpha at floor", score_run(alpha)["infeasible"])

    def test_missing_jvp_can_be_kept_when_selecting_across_the_whole_screen(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = write_run(
                Path(tmp), "r2scr_pending_seed42", WINNER_CONFIG,
                healthy_gate(jvp_vs_reference_cosine_centred_median=None,
                             jvp_vs_reference_cosine_centred_null_p95=None))
            self.assertIn("missing JVP-vs-scVelo margin", score_run(run)["infeasible"])
            allowed = score_run(run, allow_missing_jvp=True)
            self.assertTrue(allowed["feasible"])
            self.assertEqual(allowed["jvp_margin"], 0.0)

    def test_val_evaluation_foscttm_outranks_the_preflight_probe(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = write_run(
                Path(tmp), "r2scr_eval_seed42", WINNER_CONFIG, healthy_gate(foscttm=0.40),
                evaluation={"task_b": {"knn_foscttm": 0.18,
                                       "knn_foscttm_permuted_floor": 0.5}})
            row = score_run(run)
            self.assertTrue(row["feasible"])
            self.assertAlmostEqual(row["foscttm"], 0.18)
            self.assertEqual(row["alignment_source"], "evaluation_task_b")

    def test_the_front_is_not_the_two_lowest_foscttm_scores(self):
        align = scored(run="r2scr_align_seed42", foscttm=0.18, jvp_margin=0.02)
        kinetics = scored(run="r2scr_kinetics_seed42", foscttm=0.24, jvp_margin=0.09)
        dominated = scored(run="r2scr_dominated_seed42", foscttm=0.20, jvp_margin=0.01)
        infeasible = scored(run="r2scr_bestfos_seed42", foscttm=0.10, jvp_margin=0.20,
                            feasible=False)
        winners = pick_winners([align, kinetics, dominated, infeasible])
        self.assertEqual([row["run"] for row in winners],
                         ["r2scr_align_seed42", "r2scr_kinetics_seed42"])
        self.assertEqual([row["picked_for"] for row in winners],
                         ["alignment", "jvp_margin"])
        self.assertNotIn("r2scr_bestfos_seed42", [row["run"] for row in winners])
        self.assertNotIn("r2scr_dominated_seed42", [row["run"] for row in winners])

    def test_a_singleton_front_is_not_padded_with_a_dominated_run(self):
        best = scored(run="r2scr_only_seed42", foscttm=0.15, jvp_margin=0.10,
                      spread=0.9, kappa_at_floor=0.0, alpha_at_floor=0.0)
        worse = scored(run="r2scr_worse_seed42", foscttm=0.30, jvp_margin=0.01,
                       spread=0.2, kappa_at_floor=0.4, alpha_at_floor=0.4)
        winners = pick_winners([best, worse])
        self.assertEqual([row["run"] for row in winners], ["r2scr_only_seed42"])


class KineticJobTests(unittest.TestCase):
    def winners(self):
        return [
            {"run": "r2scr_c_log_lam9_seed42",
             "config": {**WINNER_CONFIG, "chromatin_transform": "cp10k_log1p"},
             "picked_for": "alignment"},
            {"run": "r2scr_f_lsi_lam9_reglin_seed42",
             "config": {**WINNER_CONFIG, "chromatin_transform": "tfidf_lsi",
                        "regulatory_transform": "cp10k_linear", "lambda_dyn": 30.0},
             "picked_for": "jvp_margin"},
        ]

    def trains(self):
        return control_jobs(self.winners())

    def test_eight_new_runs_per_winner_not_twenty_four(self):
        self.assertEqual(len(control_overrides()), 8)
        trains = self.trains()
        self.assertEqual(len(trains), 16)
        self.assertTrue(all(line.startswith("train ") for line in trains))
        tags = [spec["tag"] for spec in control_overrides()]
        self.assertEqual(
            tags,
            ["gate_pergene", "gate_none", "kstrong", "kfixed", "u0", "u0p1", "g0p1", "g10"])
        self.assertNotIn("scalar-zero", tags)
        self.assertEqual(sum("r2kin_c_log_lam9_" in line for line in trains), 8)
        self.assertEqual(sum("r2kin_f_lsi_lam9_reglin_" in line for line in trains), 8)

    def test_baselines_are_not_rerun(self):
        trains = self.trains()
        dirs = [line.split("--run-dir ")[1] for line in trains]
        self.assertFalse(any("gate_scalar" in path or "_u1_" in path or "_g1_" in path
                             for path in dirs))
        gate_lines = [line for line in trains if "_gate_" in line.split("--run-dir")[1]]
        self.assertEqual(len(gate_lines), 4)
        self.assertTrue(any("--phi-gate per-gene" in line for line in gate_lines))
        self.assertTrue(any("--phi-gate none" in line for line in gate_lines))
        u0 = [line for line in trains if line.rstrip().endswith("_u0_seed42")]
        u0p1 = [line for line in trains if line.rstrip().endswith("_u0p1_seed42")]
        g0p1 = [line for line in trains if line.rstrip().endswith("_g0p1_seed42")]
        g10 = [line for line in trains if line.rstrip().endswith("_g10_seed42")]
        self.assertEqual(len(u0), 2)
        self.assertTrue(all("--lambda-held-block 0 " in line for line in u0))
        self.assertTrue(all("--lambda-held-block 0.1 " in line for line in u0p1))
        self.assertTrue(all("--lambda-gamma-anchor 0.1 " in line for line in g0p1))
        self.assertTrue(all("--lambda-gamma-anchor 10 " in line for line in g10))

    def test_fixed_kappa_is_log2_and_does_not_keep_the_pin(self):
        kfixed = [line for line in self.trains() if "_kfixed_" in line]
        self.assertEqual(len(kfixed), 2)
        for line in kfixed:
            self.assertIn("--fixed-kappa", line)
            value = float(line.split("--fixed-kappa ")[1].split()[0])
            self.assertAlmostEqual(value, KAPPA_FIXED)
            self.assertAlmostEqual(value, math.log(2))
            self.assertNotIn("--kappa-min", line)
            self.assertNotIn("--kappa-max", line)
            self.assertIn("--lambda-kappa-prior 0 ", line)

    def test_strong_prior_opens_the_rna_protein_box(self):
        kstrong = [line for line in self.trains() if "_kstrong_" in line]
        self.assertEqual(len(kstrong), 2)
        for line in kstrong:
            self.assertIn(f"--kappa-min {KAPPA_BOX[0]}", line)
            self.assertIn(f"--kappa-max {KAPPA_BOX[1]}", line)
            self.assertIn("--lambda-kappa-prior 0.1", line)
            self.assertNotIn("--fixed-kappa", line)

    def test_winner_identity_is_copied_except_the_one_axis(self):
        trains = self.trains()
        first = [line for line in trains if "r2kin_c_log_lam9_" in line]
        second = [line for line in trains if "r2kin_f_lsi_lam9_reglin_" in line]
        self.assertTrue(all("--chromatin-transform cp10k_log1p" in line for line in first))
        self.assertTrue(all("--lambda-dyn 9 " in line for line in first))
        self.assertTrue(all("--velocity-tag scr" in line for line in trains))
        self.assertTrue(all("--regulatory-transform cp10k_linear" in line for line in second))
        self.assertTrue(all("--lambda-dyn 30 " in line for line in second))
        self.assertTrue(all("--eval-split" not in line for line in trains))

    def test_job_file_has_sixteen_trains_and_no_test_evaluate(self):
        text = job_file_text(self.winners())
        trains = [line for line in text.splitlines() if line.startswith("train ")]
        self.assertEqual(len(trains), 16)
        self.assertNotIn("--eval-split test", text)
        self.assertIn("r2scr_c_log_lam9_seed42", text)
        self.assertEqual(stem_from_winner("r2scr_c_log_lam9_seed42"), "c_log_lam9")


if __name__ == "__main__":
    unittest.main()
