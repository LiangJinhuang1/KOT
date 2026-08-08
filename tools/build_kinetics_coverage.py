#!/usr/bin/env python3
"""
ADT → kinetics coverage, recomputed from the exact training inputs.

For each dataset this reconciles three sources so the counts match what training
actually uses (an independent test, not a restatement of the mapping CSV):
  * the ADT→gene mapping CSV  — mapping_type, use_for_alignment, use_for_kinetics
  * the velocity h5ad         — RNA presence, spliced/unspliced, velocity_genes
  * the β-anchor CSV          — which proteins have a degradation-rate anchor

Writes:
  cache/results/mapping/kinetics_coverage_<dataset>.csv          (per ADT)
  cache/results/mapping/kinetics_coverage_<dataset>_summary.csv  (per dataset)

Per-ADT columns:
  adt_name, gene_symbol, mapping_type, rna_present, spliced_present,
  unspliced_present, retained, velocity_fitted, velocity_reliable,
  alignment_active, kinetics_active, kinetics_used, anchor_available

Usage:
  python tools/build_kinetics_coverage.py --datasets pbmc_retained,bmmc_cite_retained
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from src.utils.io import load_yaml
from src.data.adt_gene_map import load_mapping_records

DATASETS_CONFIG = Path("config/datasets.yaml")
TRAINING_CONFIG = Path("config/training.yaml")
OUT_DIR = Path("cache/results/mapping")

DETAIL_COLS = [
    "adt_name", "gene_symbol", "mapping_type",
    "rna_present", "spliced_present", "unspliced_present",
    "retained", "velocity_fitted", "velocity_reliable",
    "alignment_active", "kinetics_active", "kinetics_used", "anchor_available",
]

SUMMARY_COLS = [
    "dataset", "total_adts", "mapped_to_hgnc", "present_in_rna",
    "present_in_spliced_unspliced", "retained_after_qc", "velocity_fitted",
    "used_in_kinetic_loss", "anchored",
]


def mapping_path_for(dataset: str, meta: dict, train_cfg: dict) -> Path:
    # Use exactly the mapping CSV training uses: prefer the training.yaml dataset
    # block, then datasets.yaml, then a dataset-specific file, then the base panel.
    for block in ((train_cfg.get("datasets") or {}).get(dataset), meta):
        if block and block.get("adt_mapping_csv"):
            return Path(block["adt_mapping_csv"])
    specific = OUT_DIR / f"adt_mapping_{dataset}.csv"
    if specific.exists():
        return specific
    return OUT_DIR / f"adt_mapping_{dataset.replace('_retained', '')}.csv"


def anchor_proteins(dataset: str, override: str | None, train_cfg: dict) -> tuple[set[str], str | None]:
    """Set of ADT names that have a β anchor, from the dataset's anchor CSV."""
    path = override
    if path is None:
        block = (train_cfg.get("datasets") or {}).get(dataset) or {}
        path = block.get("beta_anchor_csv")
    if path is None:
        base = dataset.replace("_retained", "")
        cand = Path(f"config/beta_anchors_{base}_matched.csv")
        path = str(cand) if cand.exists() else None
    if not path or not Path(path).exists():
        return set(), None
    df = pd.read_csv(path)
    return {str(p) for p in df["protein_name"]}, str(path)


def velocity_gene_set(rna) -> set[str] | None:
    if "velocity_genes" not in rna.var.columns:
        return None
    flags = rna.var["velocity_genes"].to_numpy().astype(bool)
    return {str(g).upper() for g, ok in zip(rna.var_names, flags) if ok}


def reliable_gene_set(rna, min_r2: float) -> set[str] | None:
    """velocity_genes whose fit quality (r2 / likelihood) clears a threshold."""
    if "velocity_genes" not in rna.var.columns:
        return None
    vg = rna.var["velocity_genes"].to_numpy().astype(bool)
    quality = None
    for col in ("velocity_r2", "fit_r2", "fit_likelihood"):
        if col in rna.var.columns:
            quality = rna.var[col].to_numpy(dtype=float)
            break
    out = set()
    for i, (g, ok) in enumerate(zip(rna.var_names, vg)):
        if not ok:
            continue
        good = True if quality is None else (np.isfinite(quality[i]) and quality[i] >= min_r2)
        if good:
            out.add(str(g).upper())
    return out


def layer_presence(rna, genes_in_var: list[str], layers: tuple[str, ...]) -> dict[str, set[str]]:
    """Per layer, upper-cased genes (among genes_in_var) with > 0 total counts.

    Loads the mapped-gene columns into memory once and reuses them for all layers.
    """
    out = {layer: set() for layer in layers}
    if not genes_in_var:
        return out
    sub = rna[:, genes_in_var].to_memory()   # load only the mapped-gene columns, once
    for layer in layers:
        if layer not in sub.layers:
            continue
        M = sub.layers[layer]
        totals = np.asarray(M.sum(axis=0)).ravel() if sparse.issparse(M) else np.asarray(M).sum(axis=0)
        out[layer] = {str(g).upper() for g, t in zip(sub.var_names, totals) if t > 0}
    return out


def build_detail_rows(records, rna, anchored, min_r2: float) -> list[dict]:
    var_upper = {str(g).upper(): str(g) for g in rna.var_names}
    map_genes = [var_upper[r["gene_symbol"].upper()] for r in records
                 if r["gene_symbol"] and r["gene_symbol"].upper() in var_upper]
    presence = layer_presence(rna, map_genes, ("spliced", "unspliced"))
    spliced, unspliced = presence["spliced"], presence["unspliced"]
    vgenes = velocity_gene_set(rna) or set()
    reliable = reliable_gene_set(rna, min_r2) or set()

    rows = []
    for r in records:
        gene = r["gene_symbol"]
        gu = gene.upper()
        retained = gu in var_upper                      # gene made it into the exact h5ad
        kinetics_active = r["use_for_kinetics"]
        rows.append({
            "adt_name":          r["adt_name"],
            "gene_symbol":       gene,
            "mapping_type":      r["mapping_type"],
            "rna_present":       r["present_in_rna"] is True,   # mapping-CSV belief
            "spliced_present":   gu in spliced,
            "unspliced_present": gu in unspliced,
            "retained":          retained,
            "velocity_fitted":   gu in vgenes,
            "velocity_reliable": gu in reliable,
            "alignment_active":  r["use_for_alignment"],
            "kinetics_active":   kinetics_active,
            "kinetics_used":     bool(kinetics_active and retained),  # training's kinetic mask
            "anchor_available":  r["adt_name"] in anchored,
        })
    return rows


def summarize(dataset: str, rows: list[dict]) -> dict:
    def count(pred):
        return sum(1 for r in rows if pred(r))
    mapped_types = {"one_to_one", "complex_curated"}
    return {
        "dataset": dataset,
        "total_adts": len(rows),
        "mapped_to_hgnc": count(lambda r: r["mapping_type"] in mapped_types),
        "present_in_rna": count(lambda r: r["rna_present"]),
        "present_in_spliced_unspliced": count(lambda r: r["spliced_present"] and r["unspliced_present"]),
        "retained_after_qc": count(lambda r: r["retained"]),
        "velocity_fitted": count(lambda r: r["velocity_fitted"]),
        "used_in_kinetic_loss": count(lambda r: r["kinetics_used"]),
        "anchored": count(lambda r: r["anchor_available"]),
    }


def write_csv(rows: list[dict], path: Path, cols: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def run_one(name: str, meta: dict, train_cfg: dict, anchor_override: str | None, min_r2: float) -> None:
    map_path = mapping_path_for(name, meta, train_cfg)
    if not map_path.exists():
        raise FileNotFoundError(
            f"ADT mapping CSV missing for {name}: {map_path}. Run tools/build_adt_mapping.py first."
        )
    rna_path = Path(meta["rna_path"])
    if not rna_path.exists():
        raise FileNotFoundError(f"RNA / velocity h5ad missing for {name}: {rna_path}")

    records = load_mapping_records(map_path)   # validated + all decision columns
    anchored, anchor_csv = anchor_proteins(name, anchor_override, train_cfg)
    rna = ad.read_h5ad(rna_path, backed="r")
    rows = build_detail_rows(records, rna, anchored, min_r2)
    summary = summarize(name, rows)

    write_csv(rows, OUT_DIR / f"kinetics_coverage_{name}.csv", DETAIL_COLS)
    write_csv([summary], OUT_DIR / f"kinetics_coverage_{name}_summary.csv", SUMMARY_COLS)

    print(f"[{name}] {summary['total_adts']} ADTs  (anchor CSV: {anchor_csv or 'none'})")
    print(f"    mapped={summary['mapped_to_hgnc']}  in_rna={summary['present_in_rna']}  "
          f"spliced/unspliced={summary['present_in_spliced_unspliced']}  "
          f"retained={summary['retained_after_qc']}")
    print(f"    velocity_fitted={summary['velocity_fitted']}  "
          f"used_in_kinetic_loss={summary['used_in_kinetic_loss']}  "
          f"anchored={summary['anchored']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=str,
                        default="pbmc,pbmc_retained,bmmc_cite,bmmc_cite_retained",
                        help="Comma-separated dataset keys from config/datasets.yaml")
    parser.add_argument("--anchor-csv", type=str, default=None,
                        help="Override β-anchor CSV for all selected datasets")
    parser.add_argument("--velocity-min-r2", type=float, default=0.1,
                        help="Quality threshold for velocity_reliable")
    args = parser.parse_args()

    datasets = load_yaml(DATASETS_CONFIG).get("datasets", {})
    train_cfg = load_yaml(TRAINING_CONFIG) if TRAINING_CONFIG.exists() else {}
    selected = [s.strip() for s in args.datasets.split(",") if s.strip()]
    for name in selected:
        meta = datasets.get(name)
        if meta is None or not meta.get("rna_path"):
            print(f"[{name}] skip (not in datasets.yaml or no rna_path)")
            continue
        run_one(name, meta, train_cfg, args.anchor_csv, args.velocity_min_r2)


if __name__ == "__main__":
    main()
