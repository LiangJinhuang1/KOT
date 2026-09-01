#!/usr/bin/env python3
"""Rebuild the files training treats as source of truth when the panel or gene set moves."""
from __future__ import annotations

from pathlib import Path
from scipy import sparse
import anndata as ad
import argparse
import csv
import decoupler as dc
import gzip
import math
import numpy as np
import pandas as pd
import scanpy as sc
import sys
import tarfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.adt_gene_map import (
    aliases_for_positions,
    gene_alias_lookup,
    load_mapping_records,
    normalize_gene_symbol,
)
from src.data.adt_gene_map import build_mapping_rows, gene_alias_set, normalize_gene_symbol
from src.data.adt_gene_map import gene_alias_set, normalize_gene_symbol, validate_mapping_records
from src.data.beta_anchor import resolve_beta_anchors
from src.data.velocity import read_features, read_layer, validate_layer_shapes, write_h5ad_atomic
from src.data.preprocessing import load_and_preprocess_cached
from src.utils.io import load_yaml


# ============================================================================
# adt_mapping
# ============================================================================

ADT_MAPPING_DATASETS_CONFIG = Path("config/datasets.yaml")


ADT_MAPPING_OUT_DIR = Path("cache/results/mapping")


def protein_var_names(meta: dict) -> list[str]:
    path = meta["protein_path"]
    label = meta.get("protein_label", "ADT")
    if str(path).endswith(".h5"):
        adata = sc.read_10x_h5(path, gex_only=False)
    else:
        adata = ad.read_h5ad(path)
    if "feature_types" not in adata.var:
        raise ValueError(f"{path} is missing var['feature_types']; cannot find {label!r} panel.")
    protein = adata[:, adata.var["feature_types"] == label].copy()
    return list(protein.var_names)


def rna_var_names(meta: dict) -> list[str]:
    adata = ad.read_h5ad(meta["rna_path"], backed="r")
    aliases = gene_alias_set(
        adata.var_names,
        adata.var,
        normalizer=normalize_gene_symbol,
    )
    adata.file.close()
    return sorted(aliases)


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


def adt_mapping_write_mapping(name: str, meta: dict, kinetic_complex_allow=None) -> None:
    prot_names = protein_var_names(meta)
    unique_names, dropped = dedupe_preserve_order(prot_names)
    if dropped:
        print(f"[{name}] WARNING: dropped {len(dropped)} duplicate ADT name(s): "
              f"{sorted(set(dropped))}")

    # complex_curated markers (representative-chain approximations, e.g. CD3→CD3E)
    # are kinetics-eligible only if named here; one_to_one are eligible automatically.
    allow = kinetic_complex_allow or meta.get("kinetic_complex_allow")
    rows = build_mapping_rows(unique_names, rna_var_names(meta), kinetic_complex_allow=allow)
    df = pd.DataFrame(rows)
    if not df["adt_name"].is_unique:
        raise ValueError(f"[{name}] duplicate adt_name in output table")

    ADT_MAPPING_OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = ADT_MAPPING_OUT_DIR / f"adt_mapping_{name}.csv"
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


def adt_mapping_main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=str, default=None,
                        help="Comma-separated dataset names (default: all rna_protein datasets).")
    parser.add_argument("--kinetic-complex", type=str, default=None,
                        help="Comma-separated ADT names or gene symbols allowed for kinetics "
                             "despite being complex_curated (e.g. HLA-DR). Default: none.")
    args = parser.parse_args()

    allow = [s.strip() for s in args.kinetic_complex.split(",")] if args.kinetic_complex else None

    datasets = load_yaml(ADT_MAPPING_DATASETS_CONFIG).get("datasets", {})
    selected = args.datasets.split(",") if args.datasets else list(datasets.keys())

    for name in selected:
        meta = datasets.get(name)
        if meta is None or not meta.get("protein_path"):
            continue
        adt_mapping_write_mapping(name, meta, kinetic_complex_allow=allow)


# ============================================================================
# anchors
# ============================================================================

ANCHORS_LN2 = math.log(2.0)


def anchors_main() -> None:
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
            "beta_per_hour":       ANCHORS_LN2 / half,
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


# ============================================================================
# halflife
# ============================================================================

HALFLIFE_LN2 = math.log(2)


MATHIESON_CELL_TYPES = {
    "Bcells": (
        ["Bcells replicate 1 half_life", "Bcells replicate 2 half_life"],
        ["Bcells replicate 1 R_sq", "Bcells replicate 2 R_sq"],
    ),
    "NK": (
        ["NK cells replicate 1 half_life", "NK cells replicate 2 half_life"],
        ["NK cells replicate 1 R_sq", "NK cells replicate 2 R_sq"],
    ),
    "Monocytes": (
        ["Monocytes replicate 1 half_life", "Monocytes replicate 2 half_life"],
        ["Monocytes replicate 1 R_sq", "Monocytes replicate 2 R_sq"],
    ),
    "Hepatocytes": (
        ["Hepatocytes replicate 1 half_life", "Hepatocytes replicate 2 half_life"],
        ["Hepatocytes replicate 1 R_sq", "Hepatocytes replicate 2 R_sq"],
    ),
}


MATHIESON_SOURCE = "Mathieson2018_NatCommun_SuppData2"


TCELL_SOURCE = "Savitski2018_Cell_Tcells"


def mean_positive(values) -> float | None:
    vals = [float(v) for v in values if pd.notna(v) and float(v) > 0]
    if not vals:
        return None
    return sum(vals) / len(vals)


def load_adt_mapping(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"adt_name", "gene_symbol", "hgnc_id"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    if "uniprot_id" not in df.columns:
        df["uniprot_id"] = ""
    if "use_for_kinetics" not in df.columns:
        df["use_for_kinetics"] = True
    return df


def load_mathieson(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=0)
    return pd.read_csv(path)


def mathieson_rows_for_gene(math_df: pd.DataFrame, gene: str, cell_types: list[str]) -> list[dict]:
    """One row per (gene, cell_type). Multiple gene_name hits are resolved, not
    silently taken as iloc[0]: several rows for one gene can be the same protein
    measured repeatedly (keep and aggregate) or distinct accessions (ambiguous →
    skip). One row per cell_type is produced by aggregating across the kept rows."""
    key = gene.upper()
    hits = math_df.loc[math_df["gene_name"].astype(str).str.upper() == key]
    if hits.empty:
        return []
    # Distinct protein accessions for the same gene → ambiguous; do not guess.
    if "uniprot_id" in hits.columns:
        acc = hits["uniprot_id"].dropna().astype(str).str.strip()
        acc = acc[acc != ""]
        if acc.nunique() > 1:
            print(f"[mathieson] {gene}: ambiguous — {acc.nunique()} distinct uniprot accessions; skipped")
            return []

    out = []
    for ct in cell_types:
        hl_cols, r2_cols = MATHIESON_CELL_TYPES[ct]
        # aggregate across ALL matching rows × replicate columns → one value per cell type
        t_half = mean_positive([hits.iloc[k][c] for k in range(len(hits)) for c in hl_cols])
        if t_half is None:
            continue
        r2 = mean_positive([hits.iloc[k][c] for k in range(len(hits)) for c in r2_cols])
        out.append({
            "cell_type": ct,
            "half_life_hours": t_half,
            "quality_score_or_R2": r2 if r2 is not None else "",
            "source": MATHIESON_SOURCE,
            "anchor_weight": r2 if r2 is not None else 1.0,
        })
    return out


def load_tcell_table(
    path: Path,
    gene_col: str,
    half_life_col: str,
    r2_col: str | None,
) -> dict[str, dict]:
    """Map gene_symbol.upper() → {half_life_hours, quality_score_or_R2, anchor_weight}.

    Expects a CSV (not xlsx). Convert Supplementary Dataset 8 with
    tools/export_tcell_halflife_csv.py if needed.
    """
    path = Path(path)
    if path.suffix.lower() != ".csv":
        raise ValueError(
            f"T-cell table must be a CSV, got {path}. "
            "Expected columns: gene_name, half_life [, R_sq]."
        )
    df = pd.read_csv(path)
    if gene_col not in df.columns or half_life_col not in df.columns:
        raise ValueError(
            f"T-cell table needs columns '{gene_col}' and '{half_life_col}'. "
            f"Got: {list(df.columns)}"
        )
    # Group by gene so duplicate rows for one gene do NOT overwrite each other
    # (a dict assignment would keep only the last). Aggregate to one T-cell row/gene.
    groups: dict[str, list[tuple[float, float | None]]] = {}
    for _, row in df.iterrows():
        gene = str(row[gene_col]).strip().upper()
        if not gene or gene == "NAN":
            continue
        t_half = row[half_life_col]
        if pd.isna(t_half) or float(t_half) <= 0:
            continue
        r2 = float(row[r2_col]) if (r2_col and r2_col in df.columns and pd.notna(row[r2_col])) else None
        groups.setdefault(gene, []).append((float(t_half), r2))

    out: dict[str, dict] = {}
    for gene, measurements in groups.items():
        halfs = [h for h, _ in measurements]
        r2s = [r for _, r in measurements if r is not None]
        r2 = (sum(r2s) / len(r2s)) if r2s else None
        out[gene] = {
            "half_life_hours": sum(halfs) / len(halfs),   # mean over duplicate measurements
            "quality_score_or_R2": r2 if r2 is not None else "",
            "anchor_weight": r2 if r2 is not None else 1.0,
            "source": TCELL_SOURCE,
            "cell_type": "Tcells",
            "n_measurements": len(measurements),
        }
    return out


def build_rows(
    adt_df: pd.DataFrame,
    math_df: pd.DataFrame | None,
    cell_types: list[str],
    tcell_by_gene: dict[str, dict] | None,
    kinetics_only: bool,
) -> list[dict]:
    rows: list[dict] = []
    for _, adt in adt_df.iterrows():
        if kinetics_only and not bool(adt.get("use_for_kinetics", True)):
            continue
        gene = str(adt.get("gene_symbol") or "").strip()
        if not gene:
            continue
        protein = adt["adt_name"]
        hgnc = adt.get("hgnc_id") or ""
        uniprot = adt.get("uniprot_id") or ""

        if math_df is not None:
            for m in mathieson_rows_for_gene(math_df, gene, cell_types):
                rows.append({
                    "protein_name": protein,
                    "hgnc_id": hgnc,
                    "gene_symbol": gene,
                    "uniprot_id": uniprot,
                    "cell_type": m["cell_type"],
                    "half_life_hours": round(m["half_life_hours"], 4),
                    "beta_per_hour": round(HALFLIFE_LN2 / m["half_life_hours"], 8),
                    "quality_score_or_R2": m["quality_score_or_R2"],
                    "source": m["source"],
                    "anchor_weight": m["anchor_weight"],
                })

        if tcell_by_gene is not None:
            t = tcell_by_gene.get(gene.upper())
            if t is not None:
                rows.append({
                    "protein_name": protein,
                    "hgnc_id": hgnc,
                    "gene_symbol": gene,
                    "uniprot_id": uniprot,
                    "cell_type": t["cell_type"],
                    "half_life_hours": round(t["half_life_hours"], 4),
                    "beta_per_hour": round(HALFLIFE_LN2 / t["half_life_hours"], 8),
                    "quality_score_or_R2": t["quality_score_or_R2"],
                    "source": t["source"],
                    "anchor_weight": t["anchor_weight"],
                })
    return rows


def halflife_write_csv(rows: list[dict], out: Path) -> None:
    cols = [
        "protein_name", "hgnc_id", "gene_symbol", "uniprot_id", "cell_type",
        "half_life_hours", "beta_per_hour", "quality_score_or_R2", "source",
        "anchor_weight",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def halflife_main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adt-mapping", type=Path, required=True,
                        help="ADT mapping CSV from tools/build_inputs.py adt-mapping")
    parser.add_argument("--mathieson", type=Path, default=None,
                        help="Mathieson 2018 Supp Data 2 CSV (omit for T-cell-only)")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output anchor half-life CSV "
                             "(use distinct names, e.g. *_mathieson.csv / *_tcell.csv)")
    parser.add_argument(
        "--cell-types",
        type=str,
        default="Bcells,NK,Monocytes",
        help="Comma-separated Mathieson cell types to include "
             f"(choices: {','.join(MATHIESON_CELL_TYPES)})",
    )
    parser.add_argument("--tcell-table", type=Path, default=None,
                        help="Savitski T-cell half-life CSV "
                             "(gene_name, half_life [, R_sq]); omit for Mathieson-only")
    parser.add_argument("--tcell-gene-col", type=str, default="gene_name")
    parser.add_argument("--tcell-halflife-col", type=str, default="half_life")
    parser.add_argument("--tcell-r2-col", type=str, default="R_sq")
    parser.add_argument("--include-alignment-only", action="store_true",
                        help="Also keep ADTs with use_for_kinetics=false")
    args = parser.parse_args()

    if args.mathieson is None and args.tcell_table is None:
        raise ValueError("Provide at least one of --mathieson or --tcell-table")

    cell_types = [c.strip() for c in args.cell_types.split(",") if c.strip()]
    unknown = [c for c in cell_types if c not in MATHIESON_CELL_TYPES]
    if unknown:
        raise ValueError(f"Unknown cell types {unknown}; choose from {list(MATHIESON_CELL_TYPES)}")

    adt_df = load_adt_mapping(args.adt_mapping)

    math_df = None
    if args.mathieson is not None:
        math_df = load_mathieson(args.mathieson)
        print(f"[mathieson] loaded {args.mathieson}")

    tcell_by_gene = None
    if args.tcell_table is not None:
        tcell_by_gene = load_tcell_table(
            args.tcell_table,
            gene_col=args.tcell_gene_col,
            half_life_col=args.tcell_halflife_col,
            r2_col=args.tcell_r2_col,
        )
        print(f"[tcell] loaded half-lives for {len(tcell_by_gene)} genes from {args.tcell_table}")

    rows = build_rows(
        adt_df,
        math_df,
        cell_types=cell_types,
        tcell_by_gene=tcell_by_gene,
        kinetics_only=not args.include_alignment_only,
    )
    ct_order = {c: i for i, c in enumerate(cell_types + ["Tcells"])}
    prot_order = {n: i for i, n in enumerate(adt_df["adt_name"].tolist())}
    rows.sort(key=lambda r: (prot_order.get(r["protein_name"], 10**9), ct_order.get(r["cell_type"], 99)))

    halflife_write_csv(rows, args.out)

    n_prot = len({r["protein_name"] for r in rows})
    n_adt = int((adt_df["gene_symbol"].astype(str).str.len() > 0).sum())
    by_source = {}
    by_ct = {}
    for r in rows:
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
        by_ct[r["cell_type"]] = by_ct.get(r["cell_type"], 0) + 1
    print(f"[anchors] wrote {len(rows)} rows for {n_prot} proteins → {args.out}")
    print(f"[anchors] ADTs with gene symbol: {n_adt}/{len(adt_df)}; "
          f"with ≥1 half-life: {n_prot}")
    print(f"[anchors] rows by source: {by_source}")
    print(f"[anchors] rows by cell_type: {by_ct}")


# ============================================================================
# coverage
# ============================================================================

COVERAGE_DATASETS_CONFIG = Path("config/datasets.yaml")


TRAINING_CONFIG = Path("config/training.yaml")


COVERAGE_OUT_DIR = Path("cache/results/mapping")


DETAIL_COLS = [
    "adt_name", "gene_symbol", "mapping_type", "velocity_backend",
    "curated_kinetic_eligible", "present_in_rna", "spliced_present", "unspliced_present",
    "retained", "velocity_estimated", "velocity_gene", "dynamical_fit_success",
    "velocity_quality", "velocity_usable", "require_velocity_gene",
    "alignment_active", "final_runtime_kinetic_mask", "strict_velocity_kinetic_mask",
    "anchor_available",
]


SUMMARY_COLS = [
    "dataset", "velocity_backend", "total_adts", "mapped_to_hgnc", "curated_kinetic_eligible",
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
    specific = COVERAGE_OUT_DIR / f"adt_mapping_{dataset}.csv"
    if specific.exists():
        return specific
    return COVERAGE_OUT_DIR / f"adt_mapping_{dataset.replace('_retained', '')}.csv"


def anchor_proteins(dataset: str, override: str | None, train_cfg: dict,
                    panel_names: list[str]) -> tuple[set[str], str | None]:
    """ADT names the run ACTUALLY anchors — via the same resolver training uses
    (resolve_beta_anchors: name-normalization, quality/stability filters, aggregation),
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
    stable_cv_cfg = merged.get("beta_anchor_stable_cv_max")
    stable_cv_max = float(stable_cv_cfg) if stable_cv_cfg is not None else None
    min_quality_cfg = merged.get("beta_anchor_min_r2")
    min_quality = float(min_quality_cfg) if min_quality_cfg is not None else None
    idx, *_ = resolve_beta_anchors(
        path, panel_names,
        target_mean_beta=float(merged.get("beta_anchor_target", 0.5)),
        aggregate=merged.get("beta_anchor_aggregate", "median"),
        stable_cv_max=stable_cv_max,
        prior_sigma_floor=float(merged.get("beta_anchor_sigma_floor", 0.25)),
        min_quality=min_quality,
    )
    return {panel_names[j] for j in idx}, str(path)


def var_bool_set(rna, col: str) -> set[str]:
    """Upper-cased gene aliases where a boolean var column is True."""
    if col not in rna.var.columns:
        return set()
    flags = rna.var[col].to_numpy().astype(bool)
    return aliases_for_positions(rna.var_names, rna.var, np.flatnonzero(flags), normalizer=normalize_gene_symbol)


def var_column(rna, candidates: tuple[str, ...]):
    """First existing var column (as float array), else None."""
    for col in candidates:
        if col in rna.var.columns:
            return rna.var[col].to_numpy(dtype=float)
    return None


def velocity_backend_name(rna) -> str:
    backend = rna.uns.get("velocity_backend")
    if backend:
        return str(backend)
    source = rna.uns.get("velocity_source", {})
    if isinstance(source, dict) and source.get("backend"):
        return str(source["backend"])
    if "skeleton" in rna.uns and "regulators" in rna.uns:
        return "regvelo"
    return "scvelo_dynamical"


def dynamical_fit_set(rna) -> set[str]:
    """Genes the scVelo dynamical model fit (finite fit_likelihood / fit_alpha)."""
    q = var_column(rna, ("fit_likelihood", "fit_alpha"))
    if q is None:
        return set()
    return aliases_for_positions(rna.var_names, rna.var, np.flatnonzero(np.isfinite(q)), normalizer=normalize_gene_symbol)


def quality_pass_set(rna, min_quality: float) -> set[str]:
    """Genes whose velocity fit R² clears a threshold. Use an explicit R² column only
    (velocity_r2 / fit_r2 / r2), NOT fit_likelihood — a likelihood is not an R² and is
    not comparable to a --velocity-min-r2 threshold."""
    q = var_column(rna, ("velocity_r2", "fit_r2", "r2"))
    if q is None:
        return set()
    ok = np.isfinite(q) & (q >= min_quality)
    return aliases_for_positions(rna.var_names, rna.var, np.flatnonzero(ok), normalizer=normalize_gene_symbol)


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
            out[layer] = aliases_for_positions(
                sub.var_names, sub.var, np.flatnonzero(tot > 0), normalizer=normalize_gene_symbol,
            )
    if "velocity" in sub.layers:
        V = sub.layers["velocity"]
        arr = V.toarray() if sparse.issparse(V) else np.asarray(V)
        has_v = np.isfinite(arr).any(axis=0) & (np.nan_to_num(arr) != 0).any(axis=0)
        out["velocity"] = aliases_for_positions(
            sub.var_names, sub.var, np.flatnonzero(has_v), normalizer=normalize_gene_symbol,
        )
    return out


def build_detail_rows(records, rna, anchored, min_quality: float, require_velocity_gene: bool) -> list[dict]:
    gene_pos = gene_alias_lookup(rna.var_names, rna.var, normalizer=normalize_gene_symbol)
    map_genes = list(dict.fromkeys(
        str(rna.var_names[gene_pos[normalize_gene_symbol(r["gene_symbol"])]])
        for r in records
        if r["gene_symbol"] and normalize_gene_symbol(r["gene_symbol"]) in gene_pos
    ))
    pres = gene_presence_sets(rna, map_genes)
    spliced, unspliced, vel_est = pres["spliced"], pres["unspliced"], pres["velocity"]
    vgene = var_bool_set(rna, "velocity_genes")
    backend = velocity_backend_name(rna)
    # scVelo's dynamical model has a per-gene fit/fail mode; RegVelo is one joint
    # GRN-ODE backend, so finite non-zero velocity is the usable backend-specific signal.
    has_dynamical = backend.startswith("scvelo")
    fit_ok = dynamical_fit_set(rna) if has_dynamical else set()
    qual_ok = quality_pass_set(rna, min_quality) if has_dynamical else set()

    rows = []
    for r in records:
        gene = r["gene_symbol"]
        gu = normalize_gene_symbol(gene)
        eligible = r["use_for_kinetics"]
        retained = gu in gene_pos
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
            "velocity_backend":           backend,
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
        "velocity_backend": rows[0]["velocity_backend"] if rows else None,
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


def coverage_write_csv(rows: list[dict], path: Path, cols: list[str]) -> None:
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
            f"ADT mapping CSV missing for {name}: {map_path}. Run tools/build_inputs.py adt-mapping first."
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
    rna.file.close()
    summary = summarize(name, rows)

    coverage_write_csv(rows, COVERAGE_OUT_DIR / f"kinetics_coverage_{name}.csv", DETAIL_COLS)
    coverage_write_csv([summary], COVERAGE_OUT_DIR / f"kinetics_coverage_{name}_summary.csv", SUMMARY_COLS)

    print(f"[{name}] {summary['total_adts']} ADTs  (anchor CSV: {anchor_csv or 'none'})")
    print(f"    curated_eligible={summary['curated_kinetic_eligible']}  "
          f"retained={summary['retained_after_qc']}  "
          f"velocity_estimated={summary['velocity_estimated']}")
    print(f"    velocity_gene={summary['velocity_gene']}  "
          f"dynamical_fit_success={summary['dynamical_fit_success']}  "
          f"velocity_usable={summary['velocity_usable']}")
    print(f"    final_runtime_kinetic_mask={summary['final_runtime_kinetic_mask']}  "
          f"anchored={summary['anchored']}")


def coverage_main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=str,
                        default=(
                            "pbmc_retained,bmmc_cite_retained,"
                            "pbmc_regvelo,bmmc_cite_regvelo,"
                            "papalexi_retained,papalexi_regvelo"
                        ),
                        help="Comma-separated dataset keys from config/datasets.yaml "
                             "(default = retained scVelo/RegVelo CITE-seq datasets)")
    parser.add_argument("--anchor-csv", type=str, default=None,
                        help="Override β-anchor CSV for all selected datasets")
    parser.add_argument("--velocity-min-r2", type=float, default=0.1,
                        help="R2 threshold for velocity_quality")
    args = parser.parse_args()

    datasets = load_yaml(COVERAGE_DATASETS_CONFIG).get("datasets", {})
    train_cfg = load_yaml(TRAINING_CONFIG) if TRAINING_CONFIG.exists() else {}
    selected = [s.strip() for s in args.datasets.split(",") if s.strip()]
    for name in selected:
        meta = datasets.get(name)
        if meta is None or not meta.get("rna_path"):
            print(f"[{name}] skip (not in datasets.yaml or no rna_path)")
            continue
        run_one(name, meta, train_cfg, args.anchor_csv, args.velocity_min_r2)


# ============================================================================
# grn
# ============================================================================

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


def grn_main() -> None:
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


# ============================================================================
# papalexi
# ============================================================================

ADT_GENE_MAP = {
    "CD86": "CD86",       # CD86 costimulatory ligand
    "PDL1": "CD274",      # PD-L1
    "PDL2": "PDCD1LG2",   # PD-L2
    "CD366": "HAVCR2",    # TIM-3
}


def normalize_core(barcode: str) -> str:
    """Core 16bp cell barcode: drop a lane prefix (l1_) and a 10x suffix (-1)."""
    return str(barcode).split("_")[-1].split("-")[0]


def load_metadata(path: Path) -> pd.DataFrame:
    """ECCITE metadata indexed by cell id (l{lane}_{16bp}), collisions dropped.

    A core barcode reused across lanes cannot be assigned to one cell under a pooled
    STARsolo run, so every cell sharing such a core is removed.
    """
    meta = pd.read_csv(path, sep="\t", index_col=0)
    meta.index = meta.index.astype(str)
    meta["core"] = [normalize_core(x) for x in meta.index]
    counts = meta["core"].value_counts()
    unique_cores = set(counts[counts == 1].index)
    dropped = len(meta) - len(unique_cores)
    meta = meta[meta["core"].isin(unique_cores)].copy()
    print(f"[papalexi] metadata: {len(meta)} cells kept, {dropped} dropped (cross-lane barcode collisions)")
    return meta


def load_adt_from_tar(tar_path: Path, member: str) -> pd.DataFrame:
    """ADT count matrix (proteins x cells) read straight from the GEO RAW tar."""
    with tarfile.open(tar_path) as tar:
        handle = tar.extractfile(member)
        if handle is None:
            raise FileNotFoundError(f"{member} not found in {tar_path}")
        with gzip.open(handle) as gz:
            df = pd.read_csv(gz, sep="\t", index_col=0)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    print(f"[papalexi] ADT matrix: {df.shape[0]} proteins x {df.shape[1]} cells ({member})")
    return df


def build_rna(velocyto_dir: Path, meta: pd.DataFrame) -> ad.AnnData:
    """RNA velocity input: STARsolo spliced/unspliced/ambiguous filtered to metadata cells."""
    barcodes = pd.read_csv(velocyto_dir / "barcodes.tsv", sep="\t", header=None)[0].astype(str)
    features = read_features(velocyto_dir / "features.tsv")
    spliced = read_layer(velocyto_dir / "spliced.mtx")
    unspliced = read_layer(velocyto_dir / "unspliced.mtx")
    ambiguous = read_layer(velocyto_dir / "ambiguous.mtx")
    expected_shape = (len(barcodes), len(features))
    validate_layer_shapes("papalexi", expected_shape, {
        "spliced": spliced,
        "unspliced": unspliced,
        "ambiguous": ambiguous,
    })

    core_to_cell = dict(zip(meta["core"], meta.index))
    cores = [normalize_core(b) for b in barcodes]
    keep = [i for i, c in enumerate(cores) if c in core_to_cell]
    cell_ids = [core_to_cell[cores[i]] for i in keep]
    print(f"[papalexi] RNA: {len(keep)}/{len(barcodes)} STARsolo barcodes matched to metadata cells")
    if not keep:
        raise ValueError("No STARsolo barcodes matched the metadata core barcodes.")

    keep = np.asarray(keep)
    spliced = spliced[keep].tocsr()
    unspliced = unspliced[keep].tocsr()
    ambiguous = ambiguous[keep].tocsr()

    var = features.copy()
    var.index = pd.Index(var["gene_name"].astype(str).to_numpy(), name=None)
    if not pd.Index(var.index).is_unique:
        duplicated = int(pd.Index(var.index).duplicated().sum())
        print(f"[papalexi] RNA gene symbols include {duplicated} duplicate feature(s); "
              "AnnData will make var_names unique while var['gene_name'] keeps the symbols")
    var["feature_types"] = "GEX"

    counts = (spliced + unspliced).tocsr()
    rna = ad.AnnData(X=counts.copy(), obs=meta.loc[cell_ids].copy(), var=var)
    rna.obs_names = cell_ids
    rna.layers["counts"] = counts
    rna.layers["spliced"] = spliced
    rna.layers["unspliced"] = unspliced
    rna.layers["ambiguous"] = ambiguous
    rna.var_names_make_unique()
    rna.obs.index.name = None
    rna.var.index.name = None
    return rna


def add_umap(rna: ad.AnnData) -> None:
    """Store an X_umap in rna.obsm — the h5ad velocity mode requires one for plotting.

    Computed on a normalized/log copy so rna.X and the raw `counts` layer stay untouched.
    """
    tmp = rna.copy()
    sc.pp.filter_genes(tmp, min_cells=3)
    sc.pp.normalize_total(tmp, target_sum=1e4)
    sc.pp.log1p(tmp)
    sc.pp.highly_variable_genes(tmp, n_top_genes=2000)
    tmp = tmp[:, tmp.var["highly_variable"]].copy()
    sc.pp.scale(tmp, max_value=10)
    sc.tl.pca(tmp, n_comps=30)
    sc.pp.neighbors(tmp, n_neighbors=15, n_pcs=30)
    sc.tl.umap(tmp)
    rna.obsm["X_umap"] = tmp.obsm["X_umap"]
    print(f"[papalexi] computed X_umap ({rna.obsm['X_umap'].shape})")


def build_protein(adt: pd.DataFrame, cell_ids) -> ad.AnnData:
    """Protein h5ad: 4 ADTs x matched cells, tagged feature_types=ADT for the loader."""
    cols = [c for c in cell_ids if c in adt.columns]
    sub = adt[cols].T  # cells x proteins
    protein = ad.AnnData(X=sparse.csr_matrix(sub.to_numpy().astype("float32")))
    protein.obs_names = cols
    protein.var_names = list(adt.index)
    protein.obs.index.name = None
    protein.var.index.name = None
    protein.var["feature_types"] = "ADT"
    print(f"[papalexi] protein: {protein.n_obs} cells x {protein.n_vars} ADTs")
    return protein


def papalexi_write_mapping(path: Path, present_genes: set[str]) -> None:
    """ADT->gene mapping in the adt_mapping_*.csv schema used by the KOT pipeline."""
    present = {normalize_gene_symbol(g) for g in present_genes}
    rows = []
    for adt_name, gene in ADT_GENE_MAP.items():
        in_rna = normalize_gene_symbol(gene) in present
        rows.append({
            "adt_name": adt_name,
            "hgnc_id": "",
            "gene_symbol": gene,
            "uniprot_id": "",
            "mapping_type": "one_to_one",
            "present_in_rna": in_rna,
            "use_for_alignment": True,
            "use_for_kinetics": True,
            "excluded_because": "",
        })
    validate_mapping_records(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"[papalexi] wrote ADT-gene mapping: {path}")


def papalexi_main() -> None:
    parser = argparse.ArgumentParser(description="Build Papalexi ECCITE RNA + protein h5ads.")
    parser.add_argument("--velocyto-dir", type=Path,
                        default=Path("Datasets/Papalexi_GSE153056/starsolo_eccite_pooled/Solo.out/Velocyto/raw"))
    parser.add_argument("--tar", type=Path, default=Path("data/papalexi/GSE153056_RAW.tar"))
    parser.add_argument("--metadata", type=Path, default=Path("data/papalexi/ECCITE_metadata.tsv"))
    parser.add_argument("--adt-member", default="GSM4633615_ECCITE_ADT_counts.tsv.gz")
    parser.add_argument("--rna-out", type=Path,
                        default=Path("Datasets/Papalexi_GSE153056/papalexi_rna_input.h5ad"))
    parser.add_argument("--protein-out", type=Path,
                        default=Path("Datasets/Papalexi_GSE153056/papalexi_protein.h5ad"))
    parser.add_argument("--mapping-out", type=Path,
                        default=Path("cache/results/mapping/adt_mapping_papalexi.csv"))
    args = parser.parse_args()

    meta = load_metadata(args.metadata)
    adt = load_adt_from_tar(args.tar, args.adt_member)

    rna = build_rna(args.velocyto_dir, meta)
    protein = build_protein(adt, list(rna.obs_names))

    common = rna.obs_names.intersection(protein.obs_names)
    print(f"[papalexi] RNA∩protein cells: {len(common)}")
    if len(common) == 0:
        raise ValueError("No cells shared between RNA and protein after matching.")
    rna = rna[common].copy()
    protein = protein[common].copy()
    rna.obs.index.name = None
    rna.var.index.name = None
    protein.obs.index.name = None
    protein.var.index.name = None

    add_umap(rna)

    args.rna_out.parent.mkdir(parents=True, exist_ok=True)
    args.protein_out.parent.mkdir(parents=True, exist_ok=True)
    write_h5ad_atomic(rna, args.rna_out)
    write_h5ad_atomic(protein, args.protein_out)
    print(f"[papalexi] wrote RNA velocity input: {args.rna_out}  {rna.shape}")
    print(f"[papalexi] wrote protein: {args.protein_out}  {protein.shape}")

    papalexi_write_mapping(args.mapping_out, gene_alias_set(rna.var_names, rna.var))



# ============================================================================
# preprocess
# ============================================================================


def preprocess_main() -> None:
    """Build the preprocessed cache for every configured dataset."""
    parser = argparse.ArgumentParser(
        description="Warm cache/preprocessed/ for every dataset in config/datasets.yaml.")
    parser.add_argument("--datasets", default=None,
                        help="Comma-separated subset; default is every configured dataset.")
    args = parser.parse_args()

    configured = load_yaml("config/datasets.yaml").get("datasets", {})
    wanted = set(args.datasets.split(",")) if args.datasets else None
    for name, cfg in configured.items():
        if wanted is not None and name not in wanted:
            continue
        print(f"\n[preprocess] {name}")
        load_and_preprocess_cached(
            rna_path=cfg["rna_path"],
            protein_path=cfg.get("protein_path"),
            protein_label=cfg.get("protein_label", "ADT"),
            atac_path=cfg.get("atac_path"),
            atac_label=cfg.get("atac_label", "ATAC"),
        )


# ============================================================================
# dispatch
# ============================================================================

COMMANDS = {
    "adt-mapping": adt_mapping_main,
    "beta-anchors": anchors_main,
    "halflife-anchors": halflife_main,
    "kinetics-coverage": coverage_main,
    "grn-prior": grn_main,
    "papalexi": papalexi_main,
    "preprocess": preprocess_main,
}


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
