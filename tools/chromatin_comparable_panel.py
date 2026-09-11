#!/usr/bin/env python3
"""Re-score existing Task A tables on one gene set, without retraining.

R2 KOT predicts the kinetic panel; R1 KOT and the unpaired methods predict the
full HVG panel. A median over different genes is not a method comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
RUNS = PROJECT / "cache" / "chromatin" / "runs"
CACHE = PROJECT / "cache" / "chromatin"
RESULTS = PROJECT / "cache" / "results" / "chromatin"

# (label, per-gene csv, evaluation json for Task B)
SOURCES = [
    ("KOT R1 full", RUNS / "r1_bmmc_full_seed42" / "task_a_per_gene_best_align.csv",
     RUNS / "r1_bmmc_full_seed42" / "evaluation_best_align.json"),
    ("KOT R1 noDyn", RUNS / "r1_bmmc_noDyn_seed42" / "task_a_per_gene_best_align.csv",
     RUNS / "r1_bmmc_noDyn_seed42" / "evaluation_best_align.json"),
    ("KOT R2 LSI full", RUNS / "r2lsi_bmmc_relay_full_seed42" / "task_a_per_gene_best_align.csv",
     RUNS / "r2lsi_bmmc_relay_full_seed42" / "evaluation_best_align.json"),
    ("KOT R2 LSI noDyn", RUNS / "r2lsi_bmmc_relay_noDyn_seed42" / "task_a_per_gene_best_align.csv",
     RUNS / "r2lsi_bmmc_relay_noDyn_seed42" / "evaluation_best_align.json"),
    ("scGLUE seed42", RUNS / "final_scglue_bmmc_seed42" / "task_a_per_gene.csv",
     RUNS / "final_scglue_bmmc_seed42" / "evaluation_baseline.json"),
    ("MaxFuse seed42", RUNS / "final_maxfuse_bmmc_seed42" / "task_a_per_gene.csv",
     RUNS / "final_maxfuse_bmmc_seed42" / "evaluation_baseline.json"),
    ("ridge (paired)", CACHE / "bmmc_baseline_ridge_per_gene.csv", None),
    ("mean_only", CACHE / "bmmc_baseline_mean_only_per_gene.csv", None),
]


def load_genes(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "gene" not in frame.columns:
        raise ValueError(f"{path} has no gene column")
    return frame.set_index("gene")


def median_on(frame: pd.DataFrame, genes: pd.Index) -> tuple[float, int]:
    values = pd.to_numeric(frame.reindex(genes)["pearson"], errors="coerce")
    return float(np.nanmedian(values)), int(values.notna().sum())


def task_b(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text())
    block = payload.get("task_b") or {}
    return {
        "n_test_cells": payload.get("n_test_cells", payload.get("n_test_scored")),
        "B_knn_foscttm": block.get("knn_foscttm"),
        "B_knn_top1": block.get("knn_top1"),
        "C_label_transfer": (payload.get("task_c") or {}).get("label_transfer_accuracy"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=RESULTS / "comparable_panel_bmmc.csv")
    args = parser.parse_args()

    tables = []
    for label, csv_path, eval_path in SOURCES:
        if not csv_path.exists():
            print(f"[skip] missing {csv_path}")
            continue
        frame = load_genes(csv_path)
        tables.append((label, frame, eval_path))

    kinetic = None
    for label, frame, _ in tables:
        if label.startswith("KOT R2"):
            kinetic = frame.index if kinetic is None else kinetic.intersection(frame.index)
    hvg = None
    for label, frame, _ in tables:
        if label in {"KOT R1 full", "scGLUE seed42"}:
            hvg = frame.index if hvg is None else hvg.intersection(frame.index)
    in_g = None
    for label, frame, _ in tables:
        if "in_G" in frame.columns:
            in_g = frame.index[frame["in_G"].astype(str).str.lower().eq("true")]

    panels = [("own_panel", None), ("hvg_2000", hvg), ("kinetic_r2", kinetic)]
    if in_g is not None:
        panels.append(("ridge_in_G", in_g))

    rows = []
    for label, frame, eval_path in tables:
        extras = task_b(eval_path)
        for panel_name, genes in panels:
            if genes is None:
                used = frame.index
            else:
                used = genes.intersection(frame.index)
            if len(used) == 0:
                continue
            pearson, n = median_on(frame, used)
            rows.append({
                "method": label, "panel": panel_name, "n_genes": n,
                "A_gene_pearson_median": pearson, **extras,
            })
            print(f"  {label:22s} {panel_name:12s} n={n:4d}  r={pearson:+.4f}")

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
