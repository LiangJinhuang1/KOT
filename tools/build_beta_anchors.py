#!/usr/bin/env python3
"""
Build a β-anchor CSV from measured protein half-lives + the ADT→gene mapping.

Inputs
------
--half-lives : CSV with columns gene_symbol, half_life_hours, and optionally
               r2 / cell_type / source. Extract this from Mathieson et al. 2018
               Supplementary Data 2 (per-protein half-lives in hours) for the
               ADT-panel genes.
--mapping    : the ADT→gene mapping CSV (tools/build_adt_mapping.py output).

Output (schema requested):
  protein_name, hgnc_id, gene_symbol, uniprot_id, cell_type, half_life_hours,
  beta_per_hour, quality_score_or_R2, source, anchor_weight

Usage:
  python tools/build_beta_anchors.py \
      --half-lives Datasets/protein_half_lives_tcell.csv \
      --mapping cache/results/mapping/adt_mapping_pbmc.csv \
      --out config/beta_anchors_pbmc.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

LN2 = math.log(2.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--half-lives", required=True, help="CSV: gene_symbol, half_life_hours[, r2, cell_type, source]")
    parser.add_argument("--mapping", required=True, help="ADT→gene mapping CSV")
    parser.add_argument("--out", required=True, help="Output β-anchor CSV")
    parser.add_argument("--default-weight", type=float, default=1.0)
    args = parser.parse_args()

    hl = pd.read_csv(args.half_lives)
    hl["gene_symbol"] = hl["gene_symbol"].astype(str).str.upper()
    mp = pd.read_csv(args.mapping)
    mp = mp[mp["gene_symbol"].astype(str) != ""]
    mp["gene_symbol"] = mp["gene_symbol"].astype(str).str.upper()

    merged = mp.merge(hl, on="gene_symbol", how="inner", suffixes=("", "_hl"))
    rows = []
    for _, r in merged.iterrows():
        half = float(r["half_life_hours"])
        if half <= 0:
            continue
        rows.append({
            "protein_name":        r["adt_name"],
            "hgnc_id":             r.get("hgnc_id", ""),
            "gene_symbol":         r["gene_symbol"],
            "uniprot_id":          r.get("uniprot_id", ""),
            "cell_type":           r.get("cell_type", ""),
            "half_life_hours":     half,
            "beta_per_hour":       LN2 / half,
            "quality_score_or_R2": r.get("r2", r.get("quality_score_or_R2", "")),
            "source":              r.get("source", "Mathieson2018"),
            "anchor_weight":       args.default_weight,
        })

    out = pd.DataFrame(rows, columns=[
        "protein_name", "hgnc_id", "gene_symbol", "uniprot_id", "cell_type",
        "half_life_hours", "beta_per_hour", "quality_score_or_R2", "source", "anchor_weight",
    ])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out)} β-anchors → {args.out}")
    if len(out):
        print(out[["protein_name", "gene_symbol", "half_life_hours", "beta_per_hour"]].to_string(index=False))


if __name__ == "__main__":
    main()
