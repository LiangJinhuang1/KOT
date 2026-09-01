#!/usr/bin/env python3
"""Aggregate a per-seed sweep CSV into the cache/results/summary.csv layout.

tools/collect_run_diagnostics.py emits one row per (run, seed) with 84 columns. The
summary table that goes in front of a reader is one row per CONFIGURATION, with a
mean and a seed sd for each of the four numbers worth reading: the FOSCTTM on the
cells the run fitted, the FOSCTTM on the held-out complement, the ODE fit, and the
beta-anchor error. This turns the first into the second, so the paper table and the
sweep never disagree about what a config scored.

`--extra` appends key columns after the standard ones. Use it when the axis that
separates two arms is not in the standard set -- a kinetics control and its real arm are
identical in every standard column, so without it they collapse into one row and the file
silently averages the two arms together. For the kinetics ablations that means
`--extra kot_velocity_ablation,kot_s_permute`.

Usage:
  python tools/summarize_runs.py <per_seed.csv> <out.csv> [--extra col[,col...]]
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

# The summary.csv column contract: the keys a configuration is identified by, then
# mean/sd for each reported metric. Order is the file's column order.
KEY_COLUMNS = ["dataset", "fitted_side", "lr_phi", "lr_alpha_kappa", "lr_beta",
               "lambda_dyn", "lr_warmup_epochs", "sinkhorn_reg"]
# reported name -> column in the per-seed CSV
METRICS = {
    "foscttm_fitted": "mean_foscttm",
    "foscttm_test":   "val_foscttm",
    "jvp_cos_med":    "jvp_rhs_cos_median",
    "ode_relresid":   "rel_residual_median",
    "beta_anchor_err": "beta_anchor_mean_abs_err",
}


def mean_sd(values: list[float]) -> tuple[float, float]:
    """Mean and sample sd (ddof=1), matching collect_run_diagnostics.mean_sd."""
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    return mean, (sum((v - mean) ** 2 for v in values) / (n - 1)) ** 0.5


def as_float(raw):
    """A metric cell, or None where the writer left it blank or wrote a null."""
    return float(raw) if raw not in (None, "", "None", "nan") else None


def fitted_side(row: dict) -> str:
    """Which side of the split this run FITS, in summary.csv's vocabulary.

    fit_tuning_subset=true fits the small tuning subset; false fits the large side.
    train_on_val_side is the deprecated alias and is folded in here so a table
    spanning runs from before and after the rename does not split one config in two.
    """
    flag = row.get("fit_tuning_subset") or row.get("train_on_val_side") or ""
    return "tune" if str(flag).lower() == "true" else "test"


def summarize(rows: list[dict], extra: list[str]) -> list[dict]:
    """One record per configuration cell, mean and sd taken across its seeds."""
    groups = defaultdict(lambda: defaultdict(list))
    seeds = defaultdict(set)
    for row in rows:
        if row.get("checkpoint") != "training" or row.get("status") != "ok":
            continue
        key = tuple([row.get("dataset", ""), fitted_side(row)]
                    + [row.get(c, "") for c in KEY_COLUMNS[2:]]
                    + [row.get(c, "") for c in extra])
        seeds[key].add(row.get("seed"))
        for name, column in METRICS.items():
            value = as_float(row.get(column))
            if value is not None:
                groups[key][name].append(value)

    records = []
    for key in sorted(groups, key=lambda k: (k[0], k[1])):
        record = dict(zip(KEY_COLUMNS + extra, key))
        record["seeds"] = len(seeds[key])
        for name in METRICS:
            values = groups[key][name]
            # Blank, not zero, when a metric was never written: summary.csv leaves
            # foscttm_test empty for runs with no held-out complement, and a 0.0
            # there would read as a perfect score.
            mean, sd = mean_sd(values) if values else ("", "")
            record[f"{name}_mean"] = round(mean, 4) if values else ""
            record[f"{name}_sd"] = round(sd, 4) if values else ""
        records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("per_seed_csv", type=Path)
    parser.add_argument("out_csv", type=Path)
    parser.add_argument("--extra", default="",
                        help="Comma-separated key columns to append after the standard "
                             "ones, for axes summary.csv has no column for.")
    args = parser.parse_args()

    extra = [c for c in args.extra.split(",") if c]
    with open(args.per_seed_csv, newline="") as handle:
        rows = list(csv.DictReader(handle))
    records = summarize(rows, extra)

    fieldnames = KEY_COLUMNS + extra + ["seeds"]
    for name in METRICS:
        fieldnames += [f"{name}_mean", f"{name}_sd"]
    # summary.csv orders the metric block as fitted, test, cos, relresid, anchor —
    # already the METRICS insertion order — but ranks rows by the fitted FOSCTTM.
    records.sort(key=lambda r: (r["dataset"], r["fitted_side"],
                                r["foscttm_fitted_mean"] if r["foscttm_fitted_mean"] != "" else 9.9))
    with open(args.out_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {args.out_csv}: {len(records)} config rows, {len(fieldnames)} columns")


if __name__ == "__main__":
    main()
