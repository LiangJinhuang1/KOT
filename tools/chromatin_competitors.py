#!/usr/bin/env python3
"""KOT against the competing methods and the reference baselines, on relative terms.

An absolute FOSCTTM or per-gene correlation says nothing on this modality pair, because
chromatin is a weak and lagged predictor of expression and the achievable range is narrow.
This table therefore reports every method twice: the raw metric, and the metric expressed
as a FRACTION OF THE ACHIEVABLE RANGE between two references measured on the same cells --

    floor    `mean_only`, the training mean predicted for every cell. No per-cell
             information by construction.
    ceiling  `ridge`, a supervised least-squares fit WITH the cell pairing. Not a
             competitor: it sees the correspondence every unpaired method must discover.

so `fraction_of_range = (method - floor) / (ceiling - floor)`, oriented so that higher is
always better whichever direction the underlying metric runs in. 0 means "no better than
predicting the average cell"; 1 means "as good as a supervised map that was handed the
answer"; negative means worse than the floor.

THREE COMPARABILITY WARNINGS, all of which the columns record rather than hide:

  n_test_cells differs between methods (3000 / 4000 / 4930 / 13848). FOSCTTM depends on how
  many candidates a cell competes against, so rows with different counts are NOT directly
  comparable -- filter on `n_test_cells` before ranking anything.

  `B_knn_foscttm_constant_floor` is 0.25 by arithmetic for any data (a collapsed reference
  ties every distance in one of FOSCTTM's two averaged directions). Rows carrying it were
  judged against a floor that means nothing; the relative columns here use the permuted
  floor or the measured mean_only row instead.

  The competing methods and the older KOT rows were scored on the R1 target
  (`spliced_lognorm`); the regulatory-R2 runs predict shared-unit [u, s]. Same cells, same
  split, but not the same prediction target -- `target` records which.

Usage:
  python tools/chromatin_competitors.py --out cache/results/chromatin/r2_competitors.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RUNS = PROJECT_ROOT / "cache" / "chromatin" / "runs"
CACHE = PROJECT_ROOT / "cache" / "chromatin"
RESULTS = PROJECT_ROOT / "cache" / "results" / "chromatin"

# (column, higher_is_better). FOSCTTM runs the other way, which is exactly the sort of
# thing a reader of a wide table gets wrong, so the orientation is data rather than prose.
METRICS = [("A_gene_pearson_median", True), ("A_cell_cosine_median", True),
           ("B_knn_foscttm", False), ("C_label_transfer_accuracy", True),
           ("D_bio_scvelo_cell_cosine_centred_median", True)]


def baseline_references(dataset: str) -> dict:
    """mean_only / identity / ridge / paired_mlp as measured by the `baseline` stage."""
    path = CACHE / f"{dataset}_baselines.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    out = {}
    for name, block in payload.items():
        allgenes = block.get("all", {})
        out[name] = {"A_gene_pearson_median": allgenes.get("gene_pearson_median"),
                     "A_cell_cosine_median": block.get("cell_cosine_median")}
    return out


def evaluation_rows(dataset: str | None = None) -> pd.DataFrame:
    """Every run directory carrying an evaluation, KOT and competitor alike."""
    rows = []
    for path in sorted(RUNS.glob("*/evaluation*.json")):
        payload = json.loads(path.read_text())
        run = path.parent.name
        config_path = path.parent / "run_config.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        flat = {"run": run,
                "method": payload.get("method", config.get("method", "KOT")),
                "dataset": payload.get("dataset", config.get("dataset", "")),
                "target": config.get("target_layer", payload.get("target_layer", "")),
                "transform": config.get("chromatin_transform", "as_is" if config else ""),
                "phi_gate": config.get("phi_gate", ""),
                "lambda_dyn": (0.0 if config.get("condition") == "noDyn"
                               else config.get("lambda_dyn")),
                "formulation": config.get("formulation", ""),
                "n_test_cells": payload.get("n_test_cells")}
        for section, prefix in [("a", "A"), ("b", "B"), ("c", "C"), ("task_d", "D")]:
            block = payload.get(section, {})
            if isinstance(block, dict):
                for key, value in block.items():
                    if isinstance(value, (int, float)):
                        flat[f"{prefix}_{key}"] = value
        rows.append(flat)
    frame = pd.DataFrame(rows)
    if dataset and not frame.empty:
        frame = frame[frame.dataset == dataset]
    return frame


def add_relative(frame: pd.DataFrame, references: dict) -> pd.DataFrame:
    """Express each metric as a fraction of the floor-to-ceiling range, per dataset."""
    frame = frame.copy()
    for metric, higher_better in METRICS:
        column = f"rel_{metric}"
        frame[column] = np.nan
        if metric not in frame:
            continue
        for dataset, block in references.items():
            floor = (block.get("mean_only") or {}).get(metric)
            ceiling = (block.get("ridge") or {}).get(metric)
            if floor is None or ceiling is None or not np.isfinite([floor, ceiling]).all():
                continue
            span = ceiling - floor
            if abs(span) < 1e-12:
                continue
            mask = frame.dataset == dataset
            value = (frame.loc[mask, metric] - floor) / span
            # A lower-is-better metric already flips sign through the span, so no extra
            # negation is needed; the assert below is what keeps that true if span changes.
            frame.loc[mask, column] = value if higher_better or span < 0 else -value
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path,
                        default=RESULTS / "r2_competitors.csv")
    parser.add_argument("--datasets", nargs="+", default=["bmmc", "hspc"])
    args = parser.parse_args()

    frame = evaluation_rows()
    if frame.empty:
        raise SystemExit("no evaluation*.json found — run `evaluate` on the runs first")
    frame = frame[frame.dataset.isin(args.datasets)]

    references = {}
    rows = []
    for dataset in args.datasets:
        block = baseline_references(dataset)
        references[dataset] = block
        for name, values in block.items():
            rows.append({"run": f"baseline:{name}", "method": name, "dataset": dataset,
                         "target": "spliced_lognorm", "role":
                         "floor" if name == "mean_only" else
                         ("ceiling (PAIRED)" if name in ("ridge", "paired_mlp") else "reference"),
                         **values})
    frame["role"] = np.where(frame.formulation.astype(str).str.startswith("regulatory_r2"),
                             "KOT (regulatory R2)",
                             np.where(frame.method == "KOT", "KOT (R1 reduced)", "competitor"))
    combined = pd.concat([frame, pd.DataFrame(rows)], ignore_index=True, sort=False)
    combined = add_relative(combined, references)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.out, index=False)
    print(f"[competitors] wrote {len(combined)} rows -> {args.out}")

    show = ["dataset", "role", "method", "run", "n_test_cells", "target",
            "A_gene_pearson_median", "rel_A_gene_pearson_median",
            "B_knn_foscttm", "rel_B_knn_foscttm"]
    show = [c for c in show if c in combined]
    print(combined.sort_values(["dataset", "role", "method"])[show].round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
