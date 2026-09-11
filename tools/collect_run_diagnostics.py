#!/usr/bin/env python3
"""Read finished runs into one CSV (login node, no GPU). Rank on val_foscttm
when it exists — mean_foscttm is in-sample. Flag dyn-dead/collapsed so they
cannot win a FOSCTTM ranking. A crashed arm still gets a row.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from glob import glob
from pathlib import Path

import yaml

# Sweep axes next to the metrics; --hparams adds more.
DEFAULT_HPARAMS = [
    # Deprecated alias kept so pre-rename runs share a column with the new key.
    "fit_tuning_subset", "train_on_val_side",
    "lambda_dyn", "lambda_prior", "lambda_kappa_prior", "lambda_reg",
    "kot_kappa_prior", "kot_kappa_min", "kot_kappa_max",
    "kot_alpha_min", "kot_alpha_max",
    "n_epochs", "early_stopping_patience", "early_stopping_monitor",
    "lr", "lr_phi", "lr_alpha_kappa", "lr_beta",
    "lr_warmup_epochs", "lr_warmup_start_factor", "lr_min_factor", "batch_size",
    "val_fraction", "val_holdout_from_training", "val_split_seed",
    "val_split_per_seed", "val_stratify_by",
    "sinkhorn_reg", "sinkhorn_max_points", "sinkhorn_backend",
    "dyn_warmup_epochs", "g_freeze_epochs",
    "phase1_epochs", "phase2_epochs", "phase3_epochs", "phase3_lr_scale",
    "kot_velocity_gauge_normalize", "kot_velocity_shuffle",
    "kot_velocity_ablation", "kot_velocity_corrupt", "kot_s_permute",
    "kot_kinetics_require_velocity_gene",
    "phi_spectral_norm", "use_anchor", "beta_anchor_csv", "beta_anchor_subset_n",
]

DEFAULT_METRICS = [
    # val_foscttm vs mean_foscttm: ranking on trained cells is ranking on the test set.
    "mean_foscttm", "val_foscttm", "val_foscttm_in_full_pool", "val_n_cells",
    # train_foscttm is the selection metric under the two-dataset protocol.
    "train_foscttm", "train_n_cells",
    "val_holdout", "val_split_digest", "group_holdout_digest",
    "n_group_holdout_cells", "runtime_seconds",
    "jvp_rhs_cos_median", "jvp_rhs_cos_mean", "rel_residual_median",
    # Branch accuracy is the degeneracy static OT cannot resolve.
    "branch_accuracy", "branch_accuracy_branched", "branch_n_branched",
    "loss_dyn", "loss_align", "loss_total",
    # traj_dtw_recon vs temporal: collapsed pseudotime looks better on temporal.
    "time_spearman", "time_mae", "traj_dtw_temporal", "traj_dtw_recon",
    "kappa_median", "alpha_median", "beta_mean", "beta_anchor_mean_abs_err",
    "alpha_frac_at_midpoint", "alpha_frac_near_ceiling", "alpha_frac_near_floor",
    "alpha_frac_constant_across_cells", "kappa_frac_near_ceiling", "kappa_frac_near_floor",
    "phi_variance_ratio", "norm_ratio_median", "v_eff_norm_median",
    "velocity_backend_cos_median",
    "best_align", "best_align_epoch", "best_val_align", "best_val_align_epoch",
    "best_dyn", "best_dyn_epoch",
    "best_total", "best_total_epoch", "stop_epoch",
    # Pre-clip: the clip is a renormalizer, not a spike guard.
    "grad_norm_median",
    # Mechanism checks, not scores.
    "grad_mag_ratio_late", "grad_cos_late",
]

CHECKPOINTS = ["training", "final", "best_align", "best_val_align", "best_dyn", "best_total"]

AGGREGATE_METRICS = [
    "mean_foscttm", "val_foscttm", "val_foscttm_in_full_pool",
    "jvp_rhs_cos_median",
    "rel_residual_median", "time_spearman",
    # recon only: temporal improves when pseudotime collapses.
    "traj_dtw_recon",
    "train_foscttm", "train_n_cells", "val_n_cells",
    "phi_variance_ratio", "kappa_median", "alpha_median", "beta_mean",
    # Independent of FOSCTTM, so it belongs in the table.
    "beta_anchor_mean_abs_err",
    "grad_mag_ratio_late", "grad_cos_late",
    "grad_norm_median", "runtime_seconds",
]

# Kept in step by hand: importing visualization pulls too much for a login-node tool.
PHI_COLLAPSE_THRESHOLD = 0.5

# Below this JVP·RHS cosine the kinetics term is carried but not fit.
DYN_COS_MIN = 0.5

# run_config.yaml is written before training, so a first-epoch crash is still an arm.
ARM_MARKERS = ("run_config.yaml", "diagnostics.json", "summary.txt")


def diagnostics_path(arm_dir: Path, checkpoint: str) -> Path:
    """`training` is diagnostics.json — the state at the end of training."""
    if checkpoint == "training":
        return arm_dir / "diagnostics.json"
    return arm_dir / f"diagnostics_{checkpoint}.json"


def arm_dirs(run_dir: Path) -> list[tuple[Path, str, str, str]]:
    """Every arm of a run, including baselines that have no seed_* folder."""
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
                    # Only the checkpoint every run writes; the others are optional.
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
    """A metric across a group's seeds, skipping seeds that did not report it."""
    return [row[key] for row in group
            if isinstance(row.get(key), (int, float)) and not isinstance(row.get(key), bool)]


def cell_flag(record: dict, dyn_cos_min: float) -> str:
    """Why a cell is not a usable result: none of these show up in a FOSCTTM ranking."""
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
                   dyn_cos_min: float, rank_by: str) -> list[dict]:
    """One record per key_columns group: mean ± sd over every seed it contains.

    Grouping by configured values, not directory names, is what makes a sweep readable
    and pools a resume that shares the same config.
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

    records.sort(key=lambda r: (float("inf") if r[f"{rank_by}_mean"] is None
                                else r[f"{rank_by}_mean"]))
    return records


# FOSCTTM is the decision metric; the rest say whether a win let dynamics go slack.
PAIRED_METRICS = [("val_foscttm", True), ("mean_foscttm", True),
                  ("jvp_rhs_cos_median", False),
                  ("time_spearman", False), ("phi_variance_ratio", False),
                  ("grad_cos_late", False)]

# Rank a search on val_foscttm; mean_foscttm is what the validation split exists to avoid.
# train_foscttm is the two-dataset selection metric: the deployment side must stay out.
RANK_METRICS = ["val_foscttm", "mean_foscttm", "train_foscttm"]


def is_true(value) -> bool:
    """A config flag read back from YAML/CSV, where True may arrive as the string."""
    return str(value).strip().lower() == "true"


def normalise_fit_flag(row: dict) -> None:
    """Fold the deprecated train_on_val_side into fit_tuning_subset, in place.

    Without this, grouping on the new key would pool tuning-subset runs with the full side.
    """
    renamed = row.get("fit_tuning_subset")
    deprecated = row.get("train_on_val_side")
    if is_true(renamed) or is_true(deprecated):
        row["fit_tuning_subset"] = "True"
    elif renamed in (None, "") and deprecated not in (None, ""):
        row["fit_tuning_subset"] = str(deprecated)


def cell_key(row: dict, columns: list[str]) -> tuple:
    """A config cell's identity, as formatted strings so 0 and "0" are the same cell."""
    return tuple(format_cell(row.get(col)) for col in columns)


def parse_baseline(spec: str) -> dict[str, str]:
    """`lambda_dyn=100,n_epochs=500` -> {key: value} as strings."""
    baseline = {}
    for item in spec.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError(f"--paired-vs expects key=value, got: {item!r}")
        key, value = item.split("=", 1)
        baseline[key.strip()] = value.strip()
    return baseline


def paired_deltas(rows: list[dict], checkpoint: str, group_by: list[str],
                  baseline: dict[str, str], rank_by: str) -> list[dict]:
    """One record per (dataset, config cell): its per-seed change against the baseline.

    Paired within a seed so shared seed variance does not become a config gap.
    Datasets stay separate: pooling can cancel opposite effects into a flat zero.
    """
    # An unpinned key leaves the baseline under-determined.
    unknown = [key for key in baseline if key not in group_by]
    if unknown:
        raise ValueError(f"--paired-vs keys not in --group-by: {unknown}")
    unpinned = [key for key in group_by if key not in baseline]
    if unpinned:
        raise ValueError(
            f"--paired-vs must pin every --group-by key; missing: {unpinned}. "
            f"A baseline is one cell, not a slice."
        )

    values: dict[tuple, dict] = {}
    datasets = set()
    for row in rows:
        if row.get("checkpoint") != checkpoint or row.get("status") != "ok":
            continue
        context = (row.get("model"), row.get("dataset"), row.get("seed"))
        datasets.add((row.get("model"), row.get("dataset")))
        key = (context, cell_key(row, group_by))
        # Two runs at the same cell: say so rather than differencing against whichever was globbed first.
        if key in values:
            raise ValueError(
                f"Two runs share (model, dataset, seed)={context} and config cell "
                f"{cell_key(row, group_by)}: {values[key].get('run')} and "
                f"{row.get('run')}. Narrow --runs, or give each job its own --run-dir."
            )
        values[key] = row

    cells = sorted({key for _, key in values})
    base_key = tuple(baseline.get(col) or "" for col in group_by)
    if base_key not in cells:
        raise ValueError(f"--paired-vs cell {base_key} not found among {cells}")

    records = []
    for model, dataset in sorted(datasets):
        for cell in cells:
            record = {"model": model, "dataset": dataset}
            record.update(dict(zip(group_by, cell)))
            record["is_baseline"] = cell == base_key
            for metric, lower_is_better in PAIRED_METRICS:
                deltas = []
                for seed_row_key, row in values.items():
                    (row_model, row_dataset, _), key = seed_row_key
                    if key != cell or row_model != model or row_dataset != dataset:
                        continue
                    base_row = values.get((seed_row_key[0], base_key))
                    if base_row is None:
                        continue
                    here, there = row.get(metric), base_row.get(metric)
                    if isinstance(here, (int, float)) and isinstance(there, (int, float)):
                        deltas.append(float(here) - float(there))
                mean, sd = mean_sd(deltas) if deltas else (None, None)
                record[f"{metric}_delta_mean"] = mean
                record[f"{metric}_delta_sd"] = sd
                record[f"{metric}_delta_n"] = len(deltas)
                record[f"{metric}_delta_win"] = sum(
                    1 for d in deltas if (d < 0 if lower_is_better else d > 0)
                )
            records.append(record)

    records.sort(key=lambda r: (r["dataset"], 0 if r["is_baseline"] else 1,
                                float("inf") if r[f"{rank_by}_delta_mean"] is None
                                else r[f"{rank_by}_delta_mean"]))
    return records


# Headers name which cells each FOSCTTM averages over; TEST is a misnomer kept for table continuity.
RANK_LABELS = {
    "val_foscttm":   "VAL",     # cells held out of this run's loss
    "train_foscttm": "TRAIN",   # cells this run fitted
    "mean_foscttm":  "TEST",    # every cell, most of them fitted
}


def rank_label(metric: str) -> str:
    """Short header for a FOSCTTM column."""
    return RANK_LABELS[metric]


def other_rank_metric(rank_by: str, records: list[dict] | None = None) -> str:
    """The FOSCTTM the table is not ranked on, shown next to the one it is.

    Under the two-dataset protocol, train vs val is the comparison worth seeing.
    """
    preferred = "mean_foscttm" if rank_by == "val_foscttm" else "val_foscttm"
    if records is None or any(r.get(f"{preferred}_mean") is not None for r in records):
        return preferred
    for candidate in RANK_METRICS:
        if candidate != rank_by and any(
            r.get(f"{candidate}_mean") is not None for r in records
        ):
            return candidate
    return preferred


def rank_metric_label(metric: str) -> str:
    return "d" + rank_label(metric)


def print_paired(records: list[dict], group_by: list[str], rank_by: str) -> None:
    """One block per dataset, ranked by the paired delta on `rank_by` (negative = better)."""
    other = other_rank_metric(rank_by, records)
    columns = ["dataset"] + group_by
    widths = {
        col: max(len(col), max(len(format_cell(r.get(col))) for r in records)) + 2
        for col in columns
    }
    header = ("".join(f"{col:{widths[col]}}" for col in columns)
              + f"{rank_metric_label(rank_by):>20}{'win':>7}"
              + f"{rank_metric_label(other):>9}"
              + f"{'dcos':>9}{'dtSpear':>9}{'dφvar':>9}{'dgcos':>9}")
    current = None
    for position, record in enumerate(records):
        if record["dataset"] != current:
            current = record["dataset"]
            print(("\n" if position else "") + header)
        cells = "".join(f"{format_cell(record.get(col)):{widths[col]}}" for col in columns)
        if record["is_baseline"]:
            print(cells + f"{'(baseline)':>20}")
            continue
        mean, sd = record[f"{rank_by}_delta_mean"], record[f"{rank_by}_delta_sd"]
        delta = f"{mean:+.4f} ± {sd:.4f}" if mean is not None else "-"
        win = f"{record[f'{rank_by}_delta_win']}/{record[f'{rank_by}_delta_n']}"
        print(cells + f"{delta:>20}{win:>7}"
              + f"{format_delta(record[f'{other}_delta_mean']):>9}"
              + f"{format_delta(record['jvp_rhs_cos_median_delta_mean']):>9}"
              + f"{format_delta(record['time_spearman_delta_mean']):>9}"
              + f"{format_delta(record['phi_variance_ratio_delta_mean']):>9}"
              + f"{format_delta(record['grad_cos_late_delta_mean']):>9}")
    print(f"\n{rank_metric_label(rank_by)} is the per-seed change vs the baseline cell "
          f"on {rank_by}, negative = better;\n`win` counts seeds that improved, and "
          f"{rank_metric_label(other)} is the same delta on {other}, shown but not "
          f"ranked on.\nCheck the mechanism before believing a "
          "win: dcos sharply negative = bought alignment by dropping the dynamics; "
          "dφvar\nsharply negative = phi is shrinking rather than the ODE being "
          "satisfied; dgcos negative = the two terms\nare fighting, so the gain came "
          "at alignment's expense. A cell that wins on one dataset and loses on\nthe "
          "other has not won.")


def format_cell(value) -> str:
    """Compact, so 0.6931471805599453 does not set the column width on its own."""
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def format_metric(value) -> str:
    return "-" if value is None else f"{value:.3f}"


def format_delta(value) -> str:
    """Signed, so a paired table never leaves the direction of a change to be guessed."""
    return "-" if value is None else f"{value:+.3f}"


def print_aggregate(records: list[dict], key_columns: list[str], rank_by: str) -> None:
    """Ranked table, one row per config cell.

    Widths follow the content: a fixed column truncates the part of a stamp that differs.
    """
    widths = {
        col: max(len(col), max(len(format_cell(r.get(col))) for r in records)) + 2
        for col in key_columns
    }
    other = other_rank_metric(rank_by, records)
    ranked_label = rank_label(rank_by)
    print("".join(f"{col:{widths[col]}}" for col in key_columns)
          + f"{ranked_label:>18}{'n':>4}{'bad':>5}"
          + f"{rank_label(other):>8}"
          + f"{'cos':>8}{'relres':>8}{'tSpear':>8}"
            f"{'βerr':>8}{'φvar':>8}{'gratio':>8}{'gcos':>8}  flag")
    for record in records:
        mean, sd = record[f"{rank_by}_mean"], record[f"{rank_by}_sd"]
        # n=0 is a dead arm; n>0 with no value means the metric is no longer computed.
        if mean is not None:
            score = f"{mean:.4f} ± {sd:.4f}"
        elif record["n"] == 0:
            score = "CRASHED"
        else:
            score = "not computed"
        print("".join(f"{format_cell(record.get(col)):{widths[col]}}" for col in key_columns)
              + f"{score:>18}{record['n']:>4}{record['bad']:>5}"
              + f"{format_metric(record[f'{other}_mean']):>8}"
              + f"{format_metric(record['jvp_rhs_cos_median_mean']):>8}"
              + f"{format_metric(record['rel_residual_median_mean']):>8}"
              + f"{format_metric(record['time_spearman_mean']):>8}"
              + f"{format_metric(record['beta_anchor_mean_abs_err_mean']):>8}"
              # These decide collapsed / explain a dynamics-weight result, so they stay in the printed table.
              + f"{format_metric(record['phi_variance_ratio_mean']):>8}"
              + f"{format_metric(record['grad_mag_ratio_late_mean']):>8}"
              + f"{format_metric(record['grad_cos_late_mean']):>8}"
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
    parser.add_argument("--paired-vs", default=None,
                        help="Baseline cell for a per-seed paired comparison, as "
                             "key=value pairs over the --group-by keys, e.g. "
                             "'lambda_dyn=100,n_epochs=500'. Prints "
                             "dFOSCTTM vs that cell within each dataset instead of "
                             "ranking cells by mean±sd, and writes <out>_paired.csv. "
                             "Requires --group-by.")
    parser.add_argument("--rank-by", default="auto", choices=["auto"] + RANK_METRICS,
                        help="Which FOSCTTM ranks the tables. auto (default) picks "
                             "val_foscttm when the runs have a validation split and "
                             "mean_foscttm when they do not. Both are always printed; "
                             "this only chooses the sort key and the paired baseline "
                             "column. Rank a SEARCH on val_foscttm: mean_foscttm is "
                             "measured on cells the runs trained on.")
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
    # A missing --group-by key would silently collapse the sweep into one cell.
    hparams += [k for k in group_by if k not in hparams]
    key_columns = ["model", "dataset"] + group_by if group_by else ["run", "model", "dataset"]

    rows, metric_keys = collect(run_globs, checkpoints, hparams, args.all_diagnostics)
    # Fold the pre-rename flag before anything groups or ranks on it.
    for row in rows:
        normalise_fit_flag(row)
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

    rank_by = args.rank_by
    if rank_by == "auto":
        has_val = any(isinstance(r.get("val_foscttm"), (int, float)) for r in rows)
        rank_by = "val_foscttm" if has_val else "mean_foscttm"
    elif not any(isinstance(r.get(rank_by), (int, float)) for r in rows):
        # Warn rather than print a table of blanks.
        note = ("these runs predate the validation split, ran with val_fraction=0, or "
                "postdate the 2026-08-28 kot.py change that stopped computing the "
                "fitted/held-out FOSCTTM split (run_kot passes val_idx=None)")
        print(f"WARNING: --rank-by {rank_by}, but no run reported one — {note}. "
              f"Falling back to mean_foscttm.", file=sys.stderr)
        rank_by = "mean_foscttm"
    in_sample = [r for r in rows if r.get("val_holdout") is False
                 and isinstance(r.get("val_foscttm"), (int, float))]
    if rank_by == "val_foscttm" and in_sample:
        print(f"WARNING: {len(in_sample)} row(s) have val_holdout=false — their "
              f"validation cells were trained on, so val_foscttm is in-sample there "
              f"and not a selection metric.", file=sys.stderr)

    if args.aggregate or group_by:
        checkpoint = args.checkpoint if not args.all_checkpoints else "training"
        records = aggregate_rows(rows, checkpoint, key_columns, args.dyn_cos_min, rank_by)
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
        print(f"Ranked on {rank_by}.")
        print_aggregate(records, key_columns, rank_by)
        flagged = [r for r in records if r["flag"]]
        if flagged:
            print(f"\n{len(flagged)} of {len(records)} cells flagged — see cell_flag(). "
                  f"A dyn-dead cell can still rank well on FOSCTTM; it is not a result.")

    if args.paired_vs:
        if not group_by:
            print("--paired-vs requires --group-by", file=sys.stderr)
            return 1
        checkpoint = args.checkpoint if not args.all_checkpoints else "training"
        paired = paired_deltas(rows, checkpoint, group_by,
                               parse_baseline(args.paired_vs), rank_by)
        paired_out = args.out.with_name(f"{args.out.stem}_paired{args.out.suffix}")
        columns = (["model", "dataset"] + group_by + ["is_baseline"]
                   + [f"{m}_delta_{stat}" for m, _ in PAIRED_METRICS
                      for stat in ("mean", "sd", "n", "win")])
        with open(paired_out, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(paired)
        print(f"\nWrote {paired_out}: {len(paired)} config cells")
        print()
        print(f"Ranked on {rank_by}.")
        print_paired(paired, group_by, rank_by)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
