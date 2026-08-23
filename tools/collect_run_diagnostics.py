#!/usr/bin/env python3
"""
Collect every training run into one tidy CSV: hyperparameters joined to metrics,
one row per (run, model, dataset, seed, checkpoint).

Referenced from src/training/kot.py; this is the sweep-reading counterpart to
tools/evaluate_checkpoints.py (which re-runs evaluation). This one only reads what
runs already wrote — diagnostics*.json plus the resolved run_config.yaml — so it is
stdlib-only (json/yaml/csv) and runs on the login node without the container while a
sweep is still going.

Usage
-----
  python tools/collect_run_diagnostics.py                      # every run dir
  python tools/collect_run_diagnostics.py --runs 'cache/training/run_20260824_*'
  python tools/collect_run_diagnostics.py --aggregate          # mean±sd over seeds
  python tools/collect_run_diagnostics.py --all-checkpoints    # + best_align/best_dyn/...
  python tools/collect_run_diagnostics.py --hparams lambda_dyn,lr --all-diagnostics

A run that crashed mid-arm still gets a row with status=missing_foscttm/no_diagnostics,
so a silently lost sweep cell shows up instead of just being absent from the table.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from glob import glob
from pathlib import Path

import yaml

# Sweep axes worth carrying next to the metrics. --hparams adds more.
DEFAULT_HPARAMS = [
    "lambda_dyn", "lambda_prior", "lambda_kappa_prior", "lambda_reg",
    "kot_kappa_prior", "kot_kappa_min", "kot_kappa_max",
    "kot_alpha_min", "kot_alpha_max",
    "n_epochs", "early_stopping_patience", "lr", "batch_size",
    "sinkhorn_reg", "sinkhorn_max_points", "sinkhorn_backend",
    "dyn_warmup_epochs", "g_freeze_epochs",
    "phase1_epochs", "phase2_epochs", "phase3_epochs", "phase3_lr_scale",
    "kot_velocity_gauge_normalize", "kot_kinetics_require_velocity_gene",
    "phi_spectral_norm", "use_anchor", "beta_anchor_csv",
]

# Curated metrics. --all-diagnostics emits every scalar in the json instead.
DEFAULT_METRICS = [
    "mean_foscttm", "runtime_seconds",
    "jvp_rhs_cos_median", "jvp_rhs_cos_mean", "rel_residual_median",
    "time_spearman", "time_mae", "traj_dtw_temporal",
    "kappa_median", "alpha_median", "beta_mean", "beta_anchor_mean_abs_err",
    "phi_variance_ratio", "norm_ratio_median", "v_eff_norm_median",
    "velocity_backend_cos_median",
    "best_align", "best_align_epoch", "best_dyn", "best_dyn_epoch",
    "best_total", "best_total_epoch", "stop_epoch",
]

CHECKPOINTS = ["training", "final", "best_align", "best_dyn", "best_total"]

# Any one of these marks a directory as a real training arm rather than a
# run-level folder like results/. run_config.yaml is written before training
# starts, so even an arm that died on its first epoch is still identifiable.
ARM_MARKERS = ("run_config.yaml", "diagnostics.json", "summary.txt")


def diagnostics_path(arm_dir: Path, checkpoint: str) -> Path:
    """`training` is diagnostics.json — the state at the end of training."""
    if checkpoint == "training":
        return arm_dir / "diagnostics.json"
    return arm_dir / f"diagnostics_{checkpoint}.json"


def arm_dirs(run_dir: Path) -> list[tuple[Path, str, str, str]]:
    """(dir, model, dataset, seed) for every arm of a run.

    Stochastic models write run/model/dataset/seed_N; the deterministic baselines
    (scot, moscot, linear_ode) have no seed level and write straight into
    run/model/dataset. Matching only seed_* dropped those arms from the table
    entirely — 9 whole run dirs and 18 diagnostics.json files as of 2026-08-24 —
    which is the opposite of what this tool is for.
    """
    arms = []
    for dataset_dir in sorted(run_dir.glob("*/*")):
        if not dataset_dir.is_dir():
            continue
        model, dataset = dataset_dir.parts[-2], dataset_dir.parts[-1]
        seed_dirs = sorted(dataset_dir.glob("seed_*"))
        if seed_dirs:
            arms.extend((d, model, dataset, d.name.replace("seed_", "")) for d in seed_dirs)
        elif any((dataset_dir / marker).exists() for marker in ARM_MARKERS):
            arms.append((dataset_dir, model, dataset, "-"))
    return arms


def is_scalar(value) -> bool:
    return isinstance(value, (int, float, str, bool)) or value is None


def collect(run_globs: list[str], checkpoints: list[str],
            hparams: list[str], all_diagnostics: bool) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    metric_keys: list[str] = list(DEFAULT_METRICS)
    seen_metrics = set(metric_keys)

    run_dirs = sorted({Path(d) for pattern in run_globs for d in glob(pattern) if Path(d).is_dir()})
    for run_dir in run_dirs:
        for arm_dir, model, dataset, seed in arm_dirs(run_dir):
            cfg = {}
            cfg_path = arm_dir / "run_config.yaml"
            if cfg_path.exists():
                cfg = (yaml.safe_load(cfg_path.read_text()) or {}).get("run_cfg", {}) or {}

            for checkpoint in checkpoints:
                path = diagnostics_path(arm_dir, checkpoint)
                row = {
                    "run": run_dir.name,
                    "run_dir": str(run_dir),
                    "model": model,
                    "dataset": dataset,
                    "seed": seed,
                    "checkpoint": checkpoint,
                }
                if not path.exists():
                    # Only worth flagging for the checkpoint that every run writes;
                    # the others are genuinely optional.
                    if checkpoint == "training":
                        row["status"] = "no_diagnostics"
                        rows.append({**row, **{k: cfg.get(k) for k in hparams}})
                    continue

                diag = json.loads(path.read_text())
                row["status"] = "ok" if diag.get("mean_foscttm") is not None else "missing_foscttm"
                row.update({k: cfg.get(k) for k in hparams})
                if all_diagnostics:
                    for key, value in diag.items():
                        if not is_scalar(value):
                            continue
                        if key not in seen_metrics:
                            seen_metrics.add(key)
                            metric_keys.append(key)
                        row[key] = value
                else:
                    row.update({k: diag.get(k) for k in metric_keys})
                rows.append(row)

    return rows, metric_keys


def mean_sd(values: list[float]) -> tuple[float, float]:
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, var ** 0.5


def mean_over_seeds(group: list[dict], key: str) -> str:
    """Mean of one metric across a group's seeds, or "-" when no seed reported it.

    The per-seed value is itself usually a median over cells (jvp_rhs_cos_median,
    rel_residual_median); this averages those medians across seeds.
    """
    values = [row[key] for row in group if row.get(key) is not None]
    return f"{sum(values) / len(values):.3f}" if values else "-"


def print_aggregate(rows: list[dict], checkpoint: str) -> None:
    """mean±sd over seeds per (run, model, dataset), ranked by FOSCTTM."""
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        if row.get("checkpoint") != checkpoint:
            continue
        groups.setdefault((row["run"], row["model"], row["dataset"]), []).append(row)

    summary = []
    for (run, model, dataset), group in groups.items():
        scores = [r["mean_foscttm"] for r in group if r.get("mean_foscttm") is not None]
        incomplete = len(group) - len(scores)
        if not scores:
            summary.append((float("inf"), run, model, dataset, None, None, 0, incomplete, group))
            continue
        mean, sd = mean_sd(scores)
        summary.append((mean, run, model, dataset, mean, sd, len(scores), incomplete, group))

    summary.sort(key=lambda s: s[0])
    print(f"{'run':44}{'model':14}{'dataset':22}{'FOSCTTM':>18}{'n':>4}{'bad':>5}"
          f"{'cos':>8}{'relres':>8}{'tSpear':>8}")
    for _, run, model, dataset, mean, sd, n, incomplete, group in summary:
        score = f"{mean:.4f} ± {sd:.4f}" if mean is not None else "CRASHED"
        flag = "" if incomplete == 0 else f"  <-- {incomplete} incomplete"
        print(f"{run[:43]:44}{model:14}{dataset[:21]:22}{score:>18}{n:>4}{incomplete:>5}"
              f"{mean_over_seeds(group, 'jvp_rhs_cos_median'):>8}"
              f"{mean_over_seeds(group, 'rel_residual_median'):>8}"
              f"{mean_over_seeds(group, 'time_spearman'):>8}{flag}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", action="append", default=None,
                        help="Run-dir glob (repeatable). Default: every dir in "
                             "cache/training. A `run_*` default would miss the 35 runs "
                             "the 2026-08-23 curation renamed to semantic names "
                             "(syn_branch_ablation, bmmc_scvelo_kot, ...).")
    parser.add_argument("--out", type=Path, default=Path("cache/results/sweep_summary.csv"))
    parser.add_argument("--checkpoint", default="training", choices=CHECKPOINTS,
                        help="Which diagnostics to read (default: training = diagnostics.json).")
    parser.add_argument("--all-checkpoints", action="store_true",
                        help="One row per checkpoint instead of just --checkpoint.")
    parser.add_argument("--all-diagnostics", action="store_true",
                        help="Emit every scalar in diagnostics.json, not the curated set.")
    parser.add_argument("--hparams", default=None,
                        help="Comma-separated extra run_config keys to add as columns.")
    parser.add_argument("--aggregate", action="store_true",
                        help="Also print mean±sd over seeds, ranked by FOSCTTM.")
    args = parser.parse_args()

    run_globs = args.runs or ["cache/training/*"]
    checkpoints = CHECKPOINTS if args.all_checkpoints else [args.checkpoint]
    hparams = list(DEFAULT_HPARAMS)
    if args.hparams:
        hparams += [k.strip() for k in args.hparams.split(",")
                    if k.strip() and k.strip() not in hparams]

    rows, metric_keys = collect(run_globs, checkpoints, hparams, args.all_diagnostics)
    if not rows:
        print(f"No runs matched: {run_globs}", file=sys.stderr)
        return 1

    columns = (["run", "run_dir", "model", "dataset", "seed", "checkpoint", "status"]
               + hparams + [k for k in metric_keys if k not in hparams])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    bad = [r for r in rows if r.get("status") != "ok"]
    print(f"Wrote {args.out}: {len(rows)} rows from "
          f"{len({r['run'] for r in rows})} runs, {len(columns)} columns")
    if bad:
        print(f"WARNING: {len(bad)} incomplete cell(s) — a crashed or still-running arm:",
              file=sys.stderr)
        for row in bad[:20]:
            print(f"  {row['status']:16} {row['run']}/{row['model']}/"
                  f"{row['dataset']}/seed_{row['seed']}", file=sys.stderr)
        if len(bad) > 20:
            print(f"  ... and {len(bad) - 20} more", file=sys.stderr)

    if args.aggregate:
        print()
        print_aggregate(rows, args.checkpoint if not args.all_checkpoints else "training")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
