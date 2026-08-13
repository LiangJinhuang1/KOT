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

Every stage is reported SEPARATELY, so a gene's drop-out point is visible and the
velocity flags are not conflated (velocity_gene is scVelo's filter, NOT a fit):
  curated_kinetic_eligible   = biological / curation eligibility (maps to a real gene)
  present_in_rna             = data availability (mapping-CSV belief)
  retained                   = present in the final velocity matrix
  velocity_estimated         = a velocity value exists for the gene
  velocity_gene              = scVelo velocity_genes flag (steady-state/velocity filter)
  dynamical_fit_success      = recover_dynamics actually fit this gene (finite fit_likelihood)
  velocity_quality           = fit quality (likelihood / r2) clears the threshold
  velocity_usable            = velocity_gene & dynamical_fit_success & velocity_quality (strict report)
  final_runtime_kinetic_mask = curated_kinetic_eligible & retained, plus velocity_gene only when
                               kot_kinetics_require_velocity_gene=true for that dataset
  ...plus adt_name, gene_symbol, mapping_type, spliced_present, unspliced_present,
  alignment_active, anchor_available

Usage:
  python tools/build_kinetics_coverage.py --datasets pbmc_retained,bmmc_cite_retained
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import anndata as ad
import numpy as np
from scipy import sparse

from src.utils.io import load_yaml
from src.data.adt_gene_map import load_mapping_records
from src.data.beta_anchor import resolve_beta_anchors

DATASETS_CONFIG = Path("config/datasets.yaml")
TRAINING_CONFIG = Path("config/training.yaml")
OUT_DIR = Path("cache/results/mapping")

DETAIL_COLS = [
    "adt_name", "gene_symbol", "mapping_type",
    "curated_kinetic_eligible", "present_in_rna", "spliced_present", "unspliced_present",
    "retained", "velocity_estimated", "velocity_gene", "dynamical_fit_success",
    "velocity_quality", "velocity_usable", "require_velocity_gene",
    "alignment_active", "final_runtime_kinetic_mask", "strict_velocity_kinetic_mask",
    "anchor_available",
]

SUMMARY_COLS = [
    "dataset", "total_adts", "mapped_to_hgnc", "curated_kinetic_eligible",
    "present_in_rna", "retained_after_qc", "velocity_estimated", "velocity_gene",
    "dynamical_fit_success", "velocity_usable", "require_velocity_gene",
    "final_runtime_kinetic_mask", "strict_velocity_kinetic_mask", "anchored",
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


def anchor_proteins(dataset: str, override: str | None, train_cfg: dict,
                    panel_names: list[str]) -> tuple[set[str], str | None]:
    """ADT names the run ACTUALLY anchors — via the same resolver training uses
    (resolve_beta_anchors: name-normalization + stable_cv_max filter + aggregation),
    so the count matches diagnostics beta_anchor_n, not a raw CSV membership."""
    block = (train_cfg.get("datasets") or {}).get(dataset) or {}
    path = override or block.get("beta_anchor_csv")
    if path is None:
        base = dataset.replace("_retained", "")
        cand = Path(f"config/beta_anchors_{base}_matched.csv")
        path = str(cand) if cand.exists() else None
    if not path or not Path(path).exists():
        return set(), None
    merged = {**(train_cfg.get("defaults") or {}), **block}
    idx, *_ = resolve_beta_anchors(
        path, panel_names,
        target_mean_beta=float(merged.get("beta_anchor_target", 0.5)),
        aggregate=merged.get("beta_anchor_aggregate", "median"),
        stable_cv_max=merged.get("beta_anchor_stable_cv_max"),
        prior_sigma_floor=float(merged.get("beta_anchor_sigma_floor", 0.25)),
    )
    return {panel_names[j] for j in idx}, str(path)


def var_bool_set(rna, col: str) -> set[str]:
    """Upper-cased genes where a boolean var column is True."""
    if col not in rna.var.columns:
        return set()
    flags = rna.var[col].to_numpy().astype(bool)
    return {str(g).upper() for g, ok in zip(rna.var_names, flags) if ok}


def var_column(rna, candidates: tuple[str, ...]):
    """First existing var column (as float array), else None."""
    for col in candidates:
        if col in rna.var.columns:
            return rna.var[col].to_numpy(dtype=float)
    return None


def dynamical_fit_set(rna) -> set[str]:
    """Genes the DYNAMICAL model actually fit (finite fit_likelihood / fit_alpha) —
    distinct from velocity_genes, which is scVelo's steady-state / velocity filter."""
    q = var_column(rna, ("fit_likelihood", "fit_alpha"))
    if q is None:
        return set()
    return {str(g).upper() for g, v in zip(rna.var_names, q) if np.isfinite(v)}


def quality_pass_set(rna, min_quality: float) -> set[str]:
    """Genes whose velocity fit quality (likelihood or r2) clears a threshold."""
    q = var_column(rna, ("fit_likelihood", "fit_r2", "velocity_r2", "r2"))
    if q is None:
        return set()
    return {str(g).upper() for g, v in zip(rna.var_names, q) if np.isfinite(v) and v >= min_quality}


def gene_presence_sets(rna, genes_in_var: list[str]) -> dict[str, set[str]]:
    """Load the mapped-gene columns once; spliced/unspliced by total counts > 0,
    velocity_estimated by having any finite non-zero velocity value."""
    out = {"spliced": set(), "unspliced": set(), "velocity": set()}
    if not genes_in_var:
        return out
    sub = rna[:, genes_in_var].to_memory()
    for layer in ("spliced", "unspliced"):
        if layer in sub.layers:
            M = sub.layers[layer]
            tot = np.asarray(M.sum(axis=0)).ravel() if sparse.issparse(M) else np.asarray(M).sum(axis=0)
            out[layer] = {str(g).upper() for g, t in zip(sub.var_names, tot) if t > 0}
    if "velocity" in sub.layers:
        V = sub.layers["velocity"]
        arr = V.toarray() if sparse.issparse(V) else np.asarray(V)
        has_v = np.isfinite(arr).any(axis=0) & (np.nan_to_num(arr) != 0).any(axis=0)
        out["velocity"] = {str(g).upper() for g, f in zip(sub.var_names, has_v) if f}
    return out


def build_detail_rows(records, rna, anchored, min_quality: float, require_velocity_gene: bool) -> list[dict]:
    var_upper = {str(g).upper(): str(g) for g in rna.var_names}
    map_genes = [var_upper[r["gene_symbol"].upper()] for r in records
                 if r["gene_symbol"] and r["gene_symbol"].upper() in var_upper]
    pres = gene_presence_sets(rna, map_genes)
    spliced, unspliced, vel_est = pres["spliced"], pres["unspliced"], pres["velocity"]
    vgene = var_bool_set(rna, "velocity_genes")
    fit_ok = dynamical_fit_set(rna)
    qual_ok = quality_pass_set(rna, min_quality)
    # scVelo emits per-gene fit_likelihood/fit_alpha; RegVelo (one joint GRN-ODE) does
    # not — for it there is no per-gene "fit failed" mode, so a produced velocity IS the
    # fit and there is no separate quality gate. Detect the backend to avoid reporting
    # dynamical_fit_success / velocity_usable as spuriously zero for RegVelo output.
    has_dynamical = ("fit_likelihood" in rna.var.columns) or ("fit_alpha" in rna.var.columns)

    rows = []
    for r in records:
        gene = r["gene_symbol"]
        gu = gene.upper()
        eligible = r["use_for_kinetics"]
        retained = gu in var_upper
        velocity_gene = gu in vgene
        velocity_estimated = gu in vel_est
        if has_dynamical:
            dyn_fit = gu in fit_ok
            velocity_quality = gu in qual_ok
        else:
            dyn_fit = velocity_estimated
            velocity_quality = velocity_estimated
        if has_dynamical:
            # scVelo: keep the stricter report that separates its velocity filter,
            # per-gene dynamical fit, and fit-quality threshold.
            velocity_usable = bool(velocity_gene and dyn_fit and velocity_quality)
        else:
            # Joint ODE backends such as RegVelo have no per-gene fit columns; a
            # finite non-zero velocity is the backend-specific usable signal.
            velocity_usable = bool(velocity_estimated)
        runtime_mask = bool(eligible and retained and (velocity_gene or not require_velocity_gene))
        strict_mask = bool(eligible and retained and velocity_usable)
        rows.append({
            "adt_name":                   r["adt_name"],
            "gene_symbol":                gene,
            "mapping_type":               r["mapping_type"],
            "curated_kinetic_eligible":   eligible,               # use_for_kinetics (curation)
            "present_in_rna":             r["present_in_rna"] is True,
            "spliced_present":            gu in spliced,
            "unspliced_present":          gu in unspliced,
            "retained":                   retained,               # in the final velocity matrix
            "velocity_estimated":         velocity_estimated,     # a velocity value exists
            "velocity_gene":              velocity_gene,          # scVelo velocity_genes flag
            "dynamical_fit_success":      dyn_fit,                # recover_dynamics fit this gene
            "velocity_quality":           velocity_quality,       # fit quality >= threshold
            "velocity_usable":            velocity_usable,        # derived: all three above
            "require_velocity_gene":       require_velocity_gene,
            "alignment_active":           r["use_for_alignment"],
            "final_runtime_kinetic_mask":  runtime_mask,
            "strict_velocity_kinetic_mask": strict_mask,
            "anchor_available":           r["adt_name"] in anchored,
        })
    return rows


def count_rows(rows: list[dict], pred) -> int:
    return sum(1 for row in rows if pred(row))


def summarize(dataset: str, rows: list[dict]) -> dict:
    mapped_types = {"one_to_one", "complex_curated"}
    return {
        "dataset": dataset,
        "total_adts": len(rows),
        "mapped_to_hgnc": count_rows(rows, lambda r: r["mapping_type"] in mapped_types),
        "curated_kinetic_eligible": count_rows(rows, lambda r: r["curated_kinetic_eligible"]),
        "present_in_rna": count_rows(rows, lambda r: r["present_in_rna"]),
        "retained_after_qc": count_rows(rows, lambda r: r["retained"]),
        "velocity_estimated": count_rows(rows, lambda r: r["velocity_estimated"]),
        "velocity_gene": count_rows(rows, lambda r: r["velocity_gene"]),
        "dynamical_fit_success": count_rows(rows, lambda r: r["dynamical_fit_success"]),
        "velocity_usable": count_rows(rows, lambda r: r["velocity_usable"]),
        "require_velocity_gene": rows[0]["require_velocity_gene"] if rows else None,
        "final_runtime_kinetic_mask": count_rows(rows, lambda r: r["final_runtime_kinetic_mask"]),
        "strict_velocity_kinetic_mask": count_rows(rows, lambda r: r["strict_velocity_kinetic_mask"]),
        "anchored": count_rows(rows, lambda r: r["anchor_available"]),
    }


def require_velocity_gene_for(dataset: str, train_cfg: dict) -> bool:
    defaults = train_cfg.get("defaults") or {}
    block = (train_cfg.get("datasets") or {}).get(dataset) or {}
    merged = {**defaults, **block}
    return bool(merged.get("kot_kinetics_require_velocity_gene", False))


def write_csv(rows: list[dict], path: Path, cols: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def run_one(name: str, meta: dict, train_cfg: dict, anchor_override: str | None, min_quality: float) -> None:
    map_path = mapping_path_for(name, meta, train_cfg)
    if not map_path.exists():
        raise FileNotFoundError(
            f"ADT mapping CSV missing for {name}: {map_path}. Run tools/build_adt_mapping.py first."
        )
    rna_path = Path(meta["rna_path"])
    if not rna_path.exists():
        raise FileNotFoundError(f"RNA / velocity h5ad missing for {name}: {rna_path}")

    records = load_mapping_records(map_path)   # validated + all decision columns
    panel_names = [r["adt_name"] for r in records]
    anchored, anchor_csv = anchor_proteins(name, anchor_override, train_cfg, panel_names)
    rna = ad.read_h5ad(rna_path, backed="r")
    require_velocity_gene = require_velocity_gene_for(name, train_cfg)
    rows = build_detail_rows(records, rna, anchored, min_quality, require_velocity_gene)
    summary = summarize(name, rows)

    write_csv(rows, OUT_DIR / f"kinetics_coverage_{name}.csv", DETAIL_COLS)
    write_csv([summary], OUT_DIR / f"kinetics_coverage_{name}_summary.csv", SUMMARY_COLS)

    print(f"[{name}] {summary['total_adts']} ADTs  (anchor CSV: {anchor_csv or 'none'})")
    print(f"    curated_eligible={summary['curated_kinetic_eligible']}  "
          f"retained={summary['retained_after_qc']}  "
          f"velocity_estimated={summary['velocity_estimated']}")
    print(f"    velocity_gene={summary['velocity_gene']}  "
          f"dynamical_fit_success={summary['dynamical_fit_success']}  "
          f"velocity_usable={summary['velocity_usable']}")
    print(f"    final_runtime_kinetic_mask={summary['final_runtime_kinetic_mask']}  "
          f"anchored={summary['anchored']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=str,
                        default="pbmc_retained,bmmc_cite_retained",
                        help="Comma-separated dataset keys from config/datasets.yaml "
                             "(canonical = the _retained pair; add pbmc,bmmc_cite for the "
                             "deprecated HVG-based variants)")
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
