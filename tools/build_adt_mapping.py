#!/usr/bin/env python3
"""
Write the ADT→gene mapping table for each CITE-seq dataset.

For every dataset it resolves the protein panel to genes (HGNC + curated map),
checks which resolved genes are present in the RNA matrix, and writes one row per
ADT marker with the alignment / kinetics usage flags:

  cache/results/mapping/adt_mapping_<dataset>.csv

Columns: adt_name, hgnc_id, gene_symbol, uniprot_id, mapping_type,
         present_in_rna, use_for_alignment, use_for_kinetics, excluded_because

Usage:
  python tools/build_adt_mapping.py                      # every rna_protein dataset
  python tools/build_adt_mapping.py --datasets bmmc_cite,pbmc
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import pandas as pd

from src.utils.io import load_yaml
from src.data.preprocessing import load_protein_from_h5ad, load_protein_from_10x_h5
from src.data.adt_gene_map import build_mapping_rows

DATASETS_CONFIG = Path("config/datasets.yaml")
OUT_DIR = Path("cache/results/mapping")


def protein_var_names(meta: dict) -> list[str]:
    path = meta["protein_path"]
    label = meta.get("protein_label", "ADT")
    if str(path).endswith(".h5"):
        protein = load_protein_from_10x_h5(path, label)
    else:
        protein = load_protein_from_h5ad(path, label)
    return list(protein.var_names)


def rna_var_names(meta: dict) -> list[str]:
    return list(ad.read_h5ad(meta["rna_path"], backed="r").var_names)


def dedupe_preserve_order(names: list[str]) -> tuple[list[str], list[str]]:
    """Drop duplicate ADT names (keep first occurrence); return (unique, dropped)."""
    seen: set[str] = set()
    unique, dropped = [], []
    for n in names:
        if n in seen:
            dropped.append(n)
        else:
            seen.add(n)
            unique.append(n)
    return unique, dropped


def write_mapping(name: str, meta: dict) -> None:
    prot_names = protein_var_names(meta)
    unique_names, dropped = dedupe_preserve_order(prot_names)
    if dropped:
        print(f"[{name}] WARNING: dropped {len(dropped)} duplicate ADT name(s): "
              f"{sorted(set(dropped))}")

    rows = build_mapping_rows(unique_names, rna_var_names(meta))   # validated: names unique
    df = pd.DataFrame(rows)
    assert df["adt_name"].is_unique, f"[{name}] duplicate adt_name in output table"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"adt_mapping_{name}.csv"
    df.to_csv(out, index=False)

    # Short summary for a manual double-check.
    n        = len(df)
    align    = int(df["use_for_alignment"].sum())
    kin      = int(df["use_for_kinetics"].sum())
    present  = int(sum(v is True for v in df["present_in_rna"]))
    excluded = int((df["excluded_because"].astype(str).str.len() > 0).sum())
    by_type  = df["mapping_type"].value_counts().to_dict()
    print(f"[{name}] {n} unique ADTs → {out}")
    print(f"    mapping_type:      {by_type}")
    print(f"    use_for_alignment: {align}/{n}")
    print(f"    use_for_kinetics:  {kin}/{n}")
    print(f"    present_in_rna:    {present}/{n}")
    print(f"    excluded:          {excluded}/{n}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=str, default=None,
                        help="Comma-separated dataset names (default: all rna_protein datasets).")
    args = parser.parse_args()

    datasets = load_yaml(DATASETS_CONFIG).get("datasets", {})
    selected = args.datasets.split(",") if args.datasets else list(datasets.keys())

    for name in selected:
        meta = datasets.get(name)
        if meta is None or not meta.get("protein_path"):
            continue
        write_mapping(name, meta)


if __name__ == "__main__":
    main()
