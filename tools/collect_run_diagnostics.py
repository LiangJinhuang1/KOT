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
  python tools/collect_run_diagnostics.py --group-by lambda_dyn,lambda_kappa_prior

--group-by is the one to reach for on a sweep: it keys the table on the values a run was
CONFIGURED with instead of the string its directory is named, so an axis becomes a
sortable column rather than a substring like `lam1000_lkp0p1_kt0p69`. It also pools a
sweep that was interrupted and resumed under a second stamp, and writes a small
<out>_by_config.csv -- one row per cell -- next to the per-seed CSV.

Cells are flagged (degenerate / collapsed / dyn-dead) because none of those failures is
visible in the FOSCTTM ranking and one of them sorts first. A flagged cell is not a
result no matter where it ranks.

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
    # Pre-clip gradient norm. It sits ~100-6000x above the fixed 1.0 budget, so the
    # clip binds on every step and acts as a renormalizer rather than a spike guard.
    "grad_norm_median",
]

CHECKPOINTS = ["training", "final", "best_align", "best_dyn", "best_total"]

# Metrics carried into the per-config table as mean ± sd over every seed in the cell.
AGGREGATE_METRICS = [
    "mean_foscttm", "jvp_rhs_cos_median", "rel_residual_median", "time_spearman",
    "phi_variance_ratio", "kappa_median", "alpha_median", "beta_mean",
    # beta_anchor_mean_abs_err is the identifiability claim itself -- how far the
    # learned beta lands from its anchor target. It moves independently of FOSCTTM,
    # so it has to be in the table rather than looked up per-run afterwards.
    "beta_anchor_mean_abs_err",
    "grad_norm_median", "runtime_seconds",
]

# Kept in step with PHI_COLLAPSE_THRESHOLD in src/visualization/runs.py BY HAND: that
# module is the single source of truth for the figures, but importing it pulls
# src/visualization/__init__.py and numpy, which would cost this tool the stdlib-only
# property that lets it run on the login node without the container. Change both.
PHI_COLLAPSE_THRESHOLD = 0.5

# Below this JVP·RHS cosine the kinetics term is being carried but not fit. Bimodality
# in the 20260824 tier-A sweep sets the value: healthy cells sat at 0.75-0.96 and dead
# ones at 0.04-0.18, with nothing in between, and BOTH scored the same FOSCTTM.
DYN_COS_MIN = 0.5

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


def numeric_values(group: list[dict], key: str) -> list[float]:
    """A metric across a group's seeds, skipping seeds that did not report it.

    The per-seed value is itself usually a median over cells (jvp_rhs_cos_median,
    rel_residual_median), so the mean of these is a mean of medians.
    """
    return [row[key] for row in group
            if isinstance(row.get(key), (int, float)) and not isinstance(row.get(key), bool)]


def cell_flag(record: dict, dyn_cos_min: float) -> str:
    """Why a cell is not a usable result, or "" when it is.

    Three failure modes, none visible in the FOSCTTM ranking -- one of them is actively
    flattered by it -- which is the whole reason the flag exists:

    degenerate -- FOSCTTM exactly 0: a shared-latent model scored against itself. It
                 sorts FIRST, so a ranking alone presents it as the winner.
    collapsed -- phi shrank instead of the ODE being satisfied. The kinetics residual
                 is scale-degenerate, so this lowers the loss without fitting anything.
    dyn-dead  -- the JVP-RHS cosine never rose: the kinetics term is carried but not
                 fit. In the 20260824 tier-A sweep every lambda_dyn block contained one
                 kappa-prior cell at cos < 0.2 scoring the SAME FOSCTTM as its cos > 0.9
                 neighbour, so ranking on FOSCTTM alone picks these at random.
    """
    flags = []
    score = record.get("mean_foscttm_mean")
    if score is not None and abs(score) < 1e-12:
        flags.append("degenerate")
    phi = record.get("phi_variance_ratio_mean")
    if phi is not None and phi < PHI_COLLAPSE_THRESHOLD:
        flags.append("collapsed")
    cos = record.get("jvp_rhs_cos_median_mean")
    if cos is not None and cos < dyn_cos_min:
        flags.append("dyn-dead")
    return ",".join(flags)


def aggregate_rows(rows: list[dict], checkpoint: str, key_columns: list[str],
                   dyn_cos_min: float) -> list[dict]:
    """One record per key_columns group: mean ± sd over every seed it contains.

    key_columns is (run, model, dataset) by default. Passing hyperparameter names
    instead groups by the values a run was CONFIGURED with rather than by the string
    its directory happens to be named, which is what makes a sweep readable: the axis
    becomes a sortable column instead of a substring like `lam1000_lkp0p1_kt0p69`. It
    also pools an interrupted sweep that was resumed under a second stamp, since the
    two run dirs carry the same config.
    """
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        if row.get("checkpoint") != checkpoint:
            continue
        groups.setdefault(tuple(row.get(col) for col in key_columns), []).append(row)

    records = []
    for key, group in groups.items():
        record = dict(zip(key_columns, key))
        scores = numeric_values(group, "mean_foscttm")
        record["n"] = len(scores)
        record["bad"] = len(group) - len(scores)
        record["n_runs"] = len({row["run"] for row in group})
        for metric in AGGREGATE_METRICS:
            values = numeric_values(group, metric)
            mean, sd = mean_sd(values) if values else (None, None)
            record[f"{metric}_mean"] = mean
            record[f"{metric}_sd"] = sd
        record["flag"] = cell_flag(record, dyn_cos_min)
        records.append(record)

    records.sort(key=lambda r: (float("inf") if r["mean_foscttm_mean"] is None
                                else r["mean_foscttm_mean"]))
    return records


def format_cell(value) -> str:
    """Compact, so 0.6931471805599453 does not set the column width on its own."""
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def format_metric(value) -> str:
    return "-" if value is None else f"{value:.3f}"


def print_aggregate(records: list[dict], key_columns: list[str]) -> None:
    """Ranked table, one row per config cell. Column widths follow the content.

    Widths are computed rather than fixed: a sweep's run dirs share a long
    run_<stamp>_sw<tier>_<dataset>_ prefix and differ only at the END, so the previous
    fixed 44-char column truncated exactly the part being compared -- the 20260824
    tier-A table printed 25 pbmc cells as five identical-looking names.
    """
    widths = {
        col: max(len(col), max(len(format_cell(r.get(col))) for r in records)) + 2
        for col in key_columns
    }
    print("".join(f"{col:{widths[col]}}" for col in key_columns)
          + f"{'FOSCTTM':>18}{'n':>4}{'bad':>5}{'cos':>8}{'relres':>8}{'tSpear':>8}"
            f"{'βerr':>8}  flag")
    for record in records:
        mean, sd = record["mean_foscttm_mean"], record["mean_foscttm_sd"]
        score = f"{mean:.4f} ± {sd:.4f}" if mean is not None else "CRASHED"
        print("".join(f"{format_cell(record.get(col)):{widths[col]}}" for col in key_columns)
              + f"{score:>18}{record['n']:>4}{record['bad']:>5}"
              + f"{format_metric(record['jvp_rhs_cos_median_mean']):>8}"
              + f"{format_metric(record['rel_residual_median_mean']):>8}"
              + f"{format_metric(record['time_spearman_mean']):>8}"
              + f"{format_metric(record['beta_anchor_mean_abs_err_mean']):>8}"
              + (f"  {record['flag']}" if record["flag"] else ""))


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
    parser.add_argument("--group-by", default=None,
                        help="Comma-separated run_config keys to key the aggregate on "
                             "instead of the run-dir name, e.g. "
                             "lambda_dyn,lambda_kappa_prior. model and dataset stay in "
                             "the key. Implies --aggregate and writes <out>_by_config.csv.")
    parser.add_argument("--dyn-cos-min", type=float, default=DYN_COS_MIN,
                        help=f"Flag a cell dyn-dead below this JVP·RHS cosine "
                             f"(default {DYN_COS_MIN}).")
    args = parser.parse_args()

    run_globs = args.runs or ["cache/training/*"]
    checkpoints = CHECKPOINTS if args.all_checkpoints else [args.checkpoint]
    hparams = list(DEFAULT_HPARAMS)
    if args.hparams:
        hparams += [k.strip() for k in args.hparams.split(",")
                    if k.strip() and k.strip() not in hparams]

    group_by = [k.strip() for k in args.group_by.split(",") if k.strip()] if args.group_by else []
    # A --group-by key that is not collected is silently None for every row, which
    # would collapse the whole sweep into one meaningless cell.
    hparams += [k for k in group_by if k not in hparams]
    key_columns = ["model", "dataset"] + group_by if group_by else ["run", "model", "dataset"]

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

    if args.aggregate or group_by:
        checkpoint = args.checkpoint if not args.all_checkpoints else "training"
        records = aggregate_rows(rows, checkpoint, key_columns, args.dyn_cos_min)
        if group_by:
            by_config = args.out.with_name(f"{args.out.stem}_by_config{args.out.suffix}")
            columns = (key_columns + ["n", "bad", "n_runs", "flag"]
                       + [f"{m}_{stat}" for m in AGGREGATE_METRICS for stat in ("mean", "sd")])
            with open(by_config, "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(records)
            print(f"Wrote {by_config}: {len(records)} config cells")
        print()
        print_aggregate(records, key_columns)
        flagged = [r for r in records if r["flag"]]
        if flagged:
            print(f"\n{len(flagged)} of {len(records)} cells flagged — see cell_flag(). "
                  f"A dyn-dead cell can still rank well on FOSCTTM; it is not a result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
