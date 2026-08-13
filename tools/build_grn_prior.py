#!/usr/bin/env python3
"""
Build a prior gene regulatory network (GRN) for the RegVelo velocity backend.

Writes an edge list  cache/results/grn/grn_prior_<dataset>.csv  with columns
tf, target, weight — consumed by src/data/regvelo_backend.py::load_grn_prior.

Default source is DoRothEA (curated TF->target, confidence levels A/B/C) via
decoupler — no ATAC required, a good first pass for human immune data (PBMC/BMMC).
An ATAC-derived GRN (e.g. for BMMC from its multiome companion) can overwrite this
file later without touching the RegVelo backend.

Usage:
  python tools/build_grn_prior.py --organism human --levels A,B,C \
      --out cache/results/grn/grn_prior_pbmc.csv

Confirmed against decoupler 2.x: dc.op.dorothea(organism=...) returns columns
source/target/weight/confidence (A–E); we keep the requested confidence levels.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import decoupler as dc
import pandas as pd


def build_dorothea(organism: str, levels: list[str]) -> pd.DataFrame:
    # decoupler 2.x: network getters live under dc.op; DoRothEA columns are
    # source (TF), target (gene), weight (signed), confidence (A–E). Filter on
    # the confidence column so this works regardless of the getter's signature.
    net = dc.op.dorothea(organism=organism)
    net = net[net["confidence"].isin(levels)]
    out = pd.DataFrame({
        "tf":     net["source"].astype(str),
        "target": net["target"].astype(str),
        "weight": net["weight"].astype(float),
    })
    return out.drop_duplicates(["tf", "target"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--organism", default="human")
    parser.add_argument("--levels", default="A,B,C",
                        help="DoRothEA confidence levels to include (comma-separated).")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    levels = [s.strip() for s in args.levels.split(",") if s.strip()]
    net = build_dorothea(args.organism, levels)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    net.to_csv(args.out, index=False)
    print(f"[grn] wrote {len(net)} edges, {net['tf'].nunique()} TFs, "
          f"{net['target'].nunique()} targets → {args.out}")


if __name__ == "__main__":
    main()
