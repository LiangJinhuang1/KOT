#!/usr/bin/env python3
"""Three readers of the same run dirs, kept together because their helpers are
not interchangeable: `runs_mean_sd` uses ddof=1, `ablations_mean_sd` ddof=0.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import anndata as ad
import argparse
import csv
import numpy as np
import pandas as pd
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))




# ============================================================================
# runs
# ============================================================================

KEY_COLUMNS = ["dataset", "fitted_side", "lr_phi", "lr_alpha_kappa", "lr_beta",
               "lambda_dyn", "lr_warmup_epochs", "sinkhorn_reg"]


RUNS_METRICS = {
    "foscttm_fitted": "mean_foscttm",
    "foscttm_test":   "val_foscttm",
    "jvp_cos_med":    "jvp_rhs_cos_median",
    "ode_relresid":   "rel_residual_median",
    "beta_anchor_err": "beta_anchor_mean_abs_err",
}


def runs_mean_sd(values: list[float]) -> tuple[float, float]:
    """Mean and sample sd (ddof=1), matching collect_run_diagnostics.runs_mean_sd."""
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    return mean, (sum((v - mean) ** 2 for v in values) / (n - 1)) ** 0.5


def runs_as_float(raw):
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
        for name, column in RUNS_METRICS.items():
            value = runs_as_float(row.get(column))
            if value is not None:
                groups[key][name].append(value)

    records = []
    for key in sorted(groups, key=lambda k: (k[0], k[1])):
        record = dict(zip(KEY_COLUMNS + extra, key))
        record["seeds"] = len(seeds[key])
        for name in RUNS_METRICS:
            values = groups[key][name]
            # Blank, not zero, when a metric was never written: summary.csv leaves
            # foscttm_test empty for runs with no held-out complement, and a 0.0
            # there would read as a perfect score.
            mean, sd = runs_mean_sd(values) if values else ("", "")
            record[f"{name}_mean"] = round(mean, 4) if values else ""
            record[f"{name}_sd"] = round(sd, 4) if values else ""
        records.append(record)
    return records


def runs_main() -> None:
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
    for name in RUNS_METRICS:
        fieldnames += [f"{name}_mean", f"{name}_sd"]
    # summary.csv orders the metric block as fitted, test, cos, relresid, anchor —
    # already the RUNS_METRICS insertion order — but ranks rows by the fitted FOSCTTM.
    records.sort(key=lambda r: (r["dataset"], r["fitted_side"],
                                r["foscttm_fitted_mean"] if r["foscttm_fitted_mean"] != "" else 9.9))
    with open(args.out_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {args.out_csv}: {len(records)} config rows, {len(fieldnames)} columns")


# ============================================================================
# ablations
# ============================================================================

ABLATIONS_METRICS = ["mean_foscttm", "val_foscttm", "branch_accuracy", "branch_accuracy_branched",
           "jvp_rhs_cos_median", "rel_residual_median", "loss_dyn", "loss_align",
           "v_eff_norm_median"]


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


def ablations_mean_sd(values: pd.Series) -> tuple[float, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return float("nan"), float("nan")
    return float(clean.mean()), float(clean.std(ddof=0))


def ablations_main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("per_seed_csv", type=Path)
    parser.add_argument("out_csv", type=Path)
    args = parser.parse_args()

    frame = pd.read_csv(args.per_seed_csv)
    frame["arm"] = frame.apply(arm_label, axis=1)
    keys = [k for k in CONFIG_KEYS if k in frame.columns]
    metrics = [m for m in ABLATIONS_METRICS if m in frame.columns]

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
                mean, sd = ablations_mean_sd(sub[metric])
                row[f"{metric}_mean"] = mean
                row[f"{metric}_sd"] = sd
                if metric == "mean_foscttm":
                    row["foscttm_delta_vs_real"] = (
                        mean - ablations_mean_sd(real[metric])[0] if not real.empty else float("nan")
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


# ============================================================================
# winner
# ============================================================================

def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def winner_as_float(raw):
    """A metric cell, or None where the writer left it blank (baseline rows, no diagnostics)."""
    return float(raw) if raw else None


def cell_of(row, keys):
    return tuple(row.get(k, "") for k in keys)


def winner_main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--paired", required=True,
                        help="<out>_paired.csv from collect_run_diagnostics.py")
    parser.add_argument("--by-config", required=True,
                        help="<out>_by_config.csv from the same call (carries `flag`)")
    parser.add_argument("--keys", required=True,
                        help="Comma-separated --group-by keys that define a cell.")
    parser.add_argument("--top", type=int, default=1,
                        help="How many cells to emit (default 1).")
    args = parser.parse_args()

    keys = [k.strip() for k in args.keys.split(",") if k.strip()]
    paired = read_csv(args.paired)
    by_config = read_csv(args.by_config)

    datasets = sorted({r["dataset"] for r in paired})
    if not datasets:
        print("no datasets in the paired table -- nothing to pick", file=sys.stderr)
        return 2

    flagged: dict[tuple, list[str]] = {}
    for row in by_config:
        flag = (row.get("flag") or "").strip()
        if flag:
            flagged.setdefault(cell_of(row, keys), []).append(f"{row['dataset']}:{flag}")

    deltas: dict[tuple, dict[str, dict]] = {}
    baseline = None
    for row in paired:
        cell = cell_of(row, keys)
        deltas.setdefault(cell, {})[row["dataset"]] = row
        if row.get("is_baseline") == "True":
            baseline = cell

    scored, rejected = [], []
    for cell, per_dataset in deltas.items():
        if cell in flagged:
            rejected.append((cell, f"flagged {'; '.join(flagged[cell])}"))
            continue
        if len(per_dataset) != len(datasets):
            missing = sorted(set(datasets) - set(per_dataset))
            rejected.append((cell, f"missing dataset(s): {', '.join(missing)}"))
            continue
        total, wins, regressed = 0.0, 0, []
        for dataset, row in per_dataset.items():
            delta = winner_as_float(row.get("val_foscttm_delta_mean"))
            if delta is None:
                regressed.append(f"{dataset}: no delta")
                continue
            if delta > 0:
                regressed.append(f"{dataset}: {delta:+.4f}")
            total += delta
            wins += int(winner_as_float(row.get("val_foscttm_delta_win")) or 0)
        if regressed:
            rejected.append((cell, "regressed on " + "; ".join(regressed)))
            continue
        scored.append((total, -wins, cell))

    if not scored:
        print("every cell was rejected; no winner", file=sys.stderr)
        for cell, why in rejected:
            print(f"  reject {dict(zip(keys, cell))}: {why}", file=sys.stderr)
        return 3

    scored.sort()
    print(f"[pick] {len(scored)} cell(s) survived, {len(rejected)} rejected "
          f"({len(datasets)} datasets: {', '.join(datasets)})", file=sys.stderr)
    for why_total, neg_wins, cell in scored[: max(args.top, 3)]:
        marker = "  <- baseline" if cell == baseline else ""
        print(f"    sum dVAL {why_total:+.4f}  wins {-neg_wins}  "
              f"{dict(zip(keys, cell))}{marker}", file=sys.stderr)
    for cell, why in rejected[:8]:
        print(f"  reject {dict(zip(keys, cell))}: {why}", file=sys.stderr)

    chosen = [cell for _, _, cell in scored[: args.top]]
    if chosen[0] == baseline:
        print("[pick] nothing beat the incumbent; freezing the baseline.", file=sys.stderr)
    for rank, cell in enumerate(chosen):
        prefix = "" if args.top == 1 else f"PICK{rank + 1}_"
        for key, value in zip(keys, cell):
            print(f"{prefix}{key}={value}")
    return 0



# ============================================================================
# predictions
# ============================================================================

PREDICTIONS_DIR = Path("data/predictions")


def prediction_rows(directory: Path) -> pd.DataFrame:
    """One row per prediction h5ad, read from the uns stamp the writer left."""
    rows = []
    for path in sorted(directory.glob("*.h5ad")):
        adata = ad.read_h5ad(path, backed="r")
        rows.append({
            "model": adata.uns.get("model", path.stem),
            "dataset": adata.uns.get("dataset", "?"),
            "mean_foscttm": adata.uns.get("mean_foscttm", float("nan")),
            "n_cells": adata.n_obs,
        })
        adata.file.close()
    return pd.DataFrame(rows)


def predictions_main() -> int:
    """FOSCTTM per model x dataset, straight off the saved predictions."""
    parser = argparse.ArgumentParser(
        description="Compare mean FOSCTTM across the saved prediction h5ads.")
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS_DIR)
    args = parser.parse_args()

    frame = prediction_rows(args.predictions)
    if frame.empty:
        print(f"No predictions in {args.predictions} -- runs write them only when "
              f"save_prediction_h5ad is true.")
        return 1
    table = frame.pivot(index="model", columns="dataset", values="mean_foscttm").round(4)
    print("\nFOSCTTM comparison (lower = better)\n")
    print(table.to_string())
    out = args.predictions / "foscttm_summary.csv"
    table.to_csv(out)
    print(f"\nSaved: {out}")
    return 0


# ============================================================================
# dispatch
# ============================================================================

COMMANDS = {"runs": runs_main, "ablations": ablations_main, "winner": winner_main,
            "predictions": predictions_main}


def main() -> int:
    """Route to a subcommand, leaving the rest of argv for its own parser."""
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        print(f"commands: {', '.join(COMMANDS)}")
        return 2
    command = sys.argv.pop(1)
    return COMMANDS[command]() or 0


if __name__ == "__main__":
    sys.exit(main())
