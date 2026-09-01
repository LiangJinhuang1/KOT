#!/usr/bin/env python3
"""Aggregate a kinetics-ablation per-seed CSV into one row per (dataset, config, arm).

WHY A DEDICATED TOOL. collect_run_diagnostics.py emits the raw per-seed table, but the arm
an ablation run belongs to is spread across four config columns and one of them is a
LEGACY alias: runs from before the enum carry kot_velocity_shuffle=True and have NaN for
kot_velocity_ablation, so grouping on the enum alone silently drops them into their own
"missing" bucket instead of pairing them with the arm they are. `arm_label` folds all of
that into a single name, mirroring resolve_velocity_ablation in src/training/kot.py.

summarize_runs.py keys on hyperparameters and would collapse every arm of one config into
a single row; this keys on the arm.

Usage:
  python tools/summarize_ablations.py <per_seed.csv> <out.csv>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Metric -> whether lower is better, for the delta-vs-real column.
METRICS = ["mean_foscttm", "val_foscttm", "branch_accuracy", "branch_accuracy_branched",
           "jvp_rhs_cos_median", "rel_residual_median", "loss_dyn", "loss_align",
           "v_eff_norm_median"]
# Columns that identify a configuration independently of which arm it is.
CONFIG_KEYS = ["dataset", "model", "lr_beta", "lr_warmup_epochs"]
ARM_ORDER = ["real", "corrupt0.25", "corrupt0.5", "corrupt0.75", "shuffle",
             "reverse", "zero", "permS", "noDyn"]


def truthy(value) -> bool:
    """A config flag as written by either YAML or the CSV round-trip (True/'True'/1)."""
    return str(value).strip().lower() in ("true", "1", "1.0")


def arm_label(row) -> str:
    """One name per ablation arm, folding the legacy kot_velocity_shuffle flag in.

    Mirrors resolve_velocity_ablation: the boolean predates the enum and is the only
    marker the original permutation-control runs carry. lambda_dyn==0 is checked first
    because it removes the kinetics term outright, which subsumes any velocity setting.
    """
    if float(row.get("lambda_dyn", 1) or 0) == 0.0:
        return "noDyn"
    if truthy(row.get("kot_s_permute")):
        return "permS"
    corrupt = float(row.get("kot_velocity_corrupt") or 0.0)
    if corrupt > 0.0:
        return f"corrupt{corrupt:g}"
    ablation = str(row.get("kot_velocity_ablation") or "none").lower()
    if ablation in ("nan", "none", ""):
        ablation = "shuffle" if truthy(row.get("kot_velocity_shuffle")) else "none"
    return "real" if ablation == "none" else ablation


def mean_sd(values: pd.Series) -> tuple[float, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return float("nan"), float("nan")
    return float(clean.mean()), float(clean.std(ddof=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("per_seed_csv", type=Path)
    parser.add_argument("out_csv", type=Path)
    args = parser.parse_args()

    frame = pd.read_csv(args.per_seed_csv)
    frame["arm"] = frame.apply(arm_label, axis=1)
    keys = [k for k in CONFIG_KEYS if k in frame.columns]
    metrics = [m for m in METRICS if m in frame.columns]

    rows = []
    for config, group in frame.groupby(keys, dropna=False):
        config = config if isinstance(config, tuple) else (config,)
        # Every arm is reported against the real arm of its OWN config, so a delta never
        # crosses a dataset or a learning rate.
        real = group[group["arm"] == "real"]
        for arm in sorted(group["arm"].unique(),
                          key=lambda a: (ARM_ORDER.index(a) if a in ARM_ORDER else 99, a)):
            sub = group[group["arm"] == arm]
            row = dict(zip(keys, config))
            row["arm"] = arm
            row["n_seeds"] = len(sub)
            for metric in metrics:
                mean, sd = mean_sd(sub[metric])
                row[f"{metric}_mean"] = mean
                row[f"{metric}_sd"] = sd
                if metric == "mean_foscttm":
                    row["foscttm_delta_vs_real"] = (
                        mean - mean_sd(real[metric])[0] if not real.empty else float("nan")
                    )
            rows.append(row)

    out = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv}: {len(out)} rows ({out['arm'].nunique()} arms)")
    show = keys + ["arm", "n_seeds", "mean_foscttm_mean", "mean_foscttm_sd",
                   "foscttm_delta_vs_real", "jvp_rhs_cos_median_mean"]
    print(out[[c for c in show if c in out.columns]].to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
