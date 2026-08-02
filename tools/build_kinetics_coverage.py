#!/usr/bin/env python3
"""
Write the ADT → kinetics coverage funnel for CITE-seq datasets.

For each ADT reports which pipeline stages it survives, and a one-row summary
of counts:

  cache/results/mapping/kinetics_coverage_<dataset>.csv          (per ADT)
  cache/results/mapping/kinetics_coverage_<dataset>_summary.csv  (counts)

Stages (supervisor checklist):
  total ADTs
  mapped to HGNC
  present in RNA matrix
  present in spliced/unspliced layers
  retained after count QC (gene in the velocity result file)
  successfully velocity-fitted (scVelo velocity_genes)
  used in kinetic loss (mapped ∧ present ∧ velocity-fitted when that column exists;
                        else mapped ∧ present)

Usage:
  python tools/build_kinetics_coverage.py --datasets pbmc,pbmc_retained,bmmc_cite,bmmc_cite_retained
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import anndata as ad
import pandas as pd

from src.utils.io import load_yaml

DATASETS_CONFIG = Path("config/datasets.yaml")
OUT_DIR = Path("cache/results/mapping")

DETAIL_COLS = [
    "adt_name",
    "gene_symbol",
    "mapped_to_hgnc",
    "present_in_rna",
    "present_in_spliced_unspliced",
    "retained_after_qc",
    "successfully_velocity_fitted",
    "used_in_kinetic_loss",
]

SUMMARY_COLS = [
    "dataset",
    "total_adts",
    "mapped_to_hgnc",
    "present_in_rna",
    "present_in_spliced_unspliced",
    "retained_after_qc",
    "successfully_velocity_fitted",
    "used_in_kinetic_loss",
]


def gene_set_from_var(adata, upper: bool = True) -> set[str]:
    names = [str(g) for g in adata.var_names]
    if upper:
        return {g.upper() for g in names}
    return set(names)


def velocity_gene_set(adata) -> set[str] | None:
    if "velocity_genes" not in adata.var.columns:
        return None
    flags = adata.var["velocity_genes"].to_numpy().astype(bool)
    return {str(g).upper() for g, ok in zip(adata.var_names, flags) if ok}


def mapping_path_for(dataset: str, meta: dict) -> Path:
    configured = meta.get("adt_mapping_csv")
    if configured:
        return Path(configured)
    # retained datasets share the parent panel mapping
    base = dataset.replace("_retained", "")
    return OUT_DIR / f"adt_mapping_{base}.csv"


def build_detail_rows(mapping_df: pd.DataFrame, rna) -> list[dict]:
    rna_genes = gene_set_from_var(rna)
    has_spliced = "spliced" in rna.layers and "unspliced" in rna.layers
    vgenes = velocity_gene_set(rna)

    rows = []
    for _, adt in mapping_df.iterrows():
        gene = str(adt.get("gene_symbol") or "").strip()
        gene_u = gene.upper()
        mapping_type = str(adt.get("mapping_type") or "")
        mapped = bool(gene) and mapping_type not in ("isotype", "unmapped")
        present_rna = bool(gene_u and gene_u in rna_genes)

        in_layers = bool(gene_u and has_spliced and gene_u in rna_genes)
        retained = bool(gene_u and gene_u in rna_genes)
        fitted = bool(gene_u and vgenes is not None and gene_u in vgenes)
        if vgenes is None:
            fitted = retained
            used = bool(mapped and present_rna)
        else:
            used = bool(mapped and present_rna and fitted)

        rows.append({
            "adt_name": adt["adt_name"],
            "gene_symbol": gene,
            "mapped_to_hgnc": mapped,
            "present_in_rna": present_rna,
            "present_in_spliced_unspliced": in_layers,
            "retained_after_qc": retained,
            "successfully_velocity_fitted": fitted,
            "used_in_kinetic_loss": used,
        })
    return rows


def summarize(dataset: str, rows: list[dict]) -> dict:
    def count(key):
        return sum(1 for r in rows if r[key] is True)

    return {
        "dataset": dataset,
        "total_adts": len(rows),
        "mapped_to_hgnc": count("mapped_to_hgnc"),
        "present_in_rna": count("present_in_rna"),
        "present_in_spliced_unspliced": count("present_in_spliced_unspliced"),
        "retained_after_qc": count("retained_after_qc"),
        "successfully_velocity_fitted": count("successfully_velocity_fitted"),
        "used_in_kinetic_loss": count("used_in_kinetic_loss"),
    }


def write_detail(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=DETAIL_COLS)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_summary(summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLS)
        w.writeheader()
        w.writerow(summary)


def run_one(name: str, meta: dict) -> None:
    map_path = mapping_path_for(name, meta)
    if not map_path.exists():
        raise FileNotFoundError(
            f"ADT mapping CSV missing for {name}: {map_path}. "
            "Run tools/build_adt_mapping.py first."
        )
    rna_path = Path(meta["rna_path"])
    if not rna_path.exists():
        raise FileNotFoundError(f"RNA / velocity h5ad missing for {name}: {rna_path}")

    mapping_df = pd.read_csv(map_path)
    rna = ad.read_h5ad(rna_path, backed="r")
    rows = build_detail_rows(mapping_df, rna)
    summary = summarize(name, rows)

    detail_out = OUT_DIR / f"kinetics_coverage_{name}.csv"
    summary_out = OUT_DIR / f"kinetics_coverage_{name}_summary.csv"
    write_detail(rows, detail_out)
    write_summary(summary, summary_out)

    print(f"[{name}] wrote {detail_out}")
    print(f"[{name}] summary → {summary_out}")
    print(
        f"    total={summary['total_adts']}  mapped={summary['mapped_to_hgnc']}  "
        f"in_rna={summary['present_in_rna']}  spliced={summary['present_in_spliced_unspliced']}  "
        f"retained={summary['retained_after_qc']}  "
        f"velocity_fitted={summary['successfully_velocity_fitted']}  "
        f"kinetic={summary['used_in_kinetic_loss']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        type=str,
        default="pbmc,pbmc_retained,bmmc_cite,bmmc_cite_retained",
        help="Comma-separated dataset keys from config/datasets.yaml",
    )
    args = parser.parse_args()

    datasets = load_yaml(DATASETS_CONFIG).get("datasets", {})
    selected = [s.strip() for s in args.datasets.split(",") if s.strip()]
    for name in selected:
        meta = datasets.get(name)
        if meta is None or not meta.get("rna_path"):
            print(f"[{name}] skip (not in datasets.yaml or no rna_path)")
            continue
        run_one(name, meta)


if __name__ == "__main__":
    main()
