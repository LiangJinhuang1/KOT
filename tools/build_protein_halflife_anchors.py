#!/usr/bin/env python3
"""
Build a protein half-life / β prior CSV for CITE-seq anchor experiments.

Joins the ADT→gene mapping (from tools/build_adt_mapping.py) to literature
half-lives and writes one row per (ADT, cell_type) with:

  protein_name, hgnc_id, gene_symbol, uniprot_id, cell_type,
  half_life_hours, beta_per_hour, quality_score_or_R2, source, anchor_weight

β = ln(2) / t½  (absolute, per hour). For KOT training, rescale to the model's
β units (e.g. relative to a reference β≈0.5) — do not plug beta_per_hour in raw.

Primary source: Mathieson et al., Nat Commun 2018 Supplementary Data 2
  (B-cells / NK / monocytes / hepatocytes / mouse neurons), CSV export of the
  high-quality sheet: Datasets/references/mathieson2018_supp_data2.csv
  (original xlsx: static-content.springer.com .../41467_2018_3106_MOESM5_ESM.xlsx).

Optional T-cell source: processed half-life table from Savitski et al., Cell 2018
  (PXD008630 is raw MS only — use the paper's supplementary xlsx/csv, not PRIDE).
  Pass with --tcell-table; expected columns include a gene name and half-life
  (hours), optionally an R² column. See --tcell-* flags.

Usage (on the cluster, after ADT mapping exists):
  python tools/build_adt_mapping.py --datasets pbmc
  python tools/build_protein_halflife_anchors.py \\
      --adt-mapping cache/results/mapping/adt_mapping_pbmc.csv \\
      --mathieson Datasets/references/mathieson2018_supp_data2.csv \\
      --out cache/results/mapping/pbmc_protein_halflife_anchors.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import pandas as pd

LN2 = math.log(2)

# Mathieson Supp Data 2: (cell_type label, half_life cols, R² cols)
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
    # CSV export of the high-quality sheet (first sheet of the Nature xlsx).
    return pd.read_csv(path)


def mathieson_rows_for_gene(math_df: pd.DataFrame, gene: str, cell_types: list[str]) -> list[dict]:
    key = gene.upper()
    hit = math_df.loc[math_df["gene_name"].astype(str).str.upper() == key]
    if hit.empty:
        return []
    row = hit.iloc[0]
    out = []
    for ct in cell_types:
        hl_cols, r2_cols = MATHIESON_CELL_TYPES[ct]
        t_half = mean_positive([row.get(c) for c in hl_cols])
        if t_half is None:
            continue
        r2 = mean_positive([row.get(c) for c in r2_cols])
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
    """Map gene_symbol.upper() → {half_life_hours, quality_score_or_R2, anchor_weight}."""
    df = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)
    if gene_col not in df.columns or half_life_col not in df.columns:
        raise ValueError(
            f"T-cell table needs columns '{gene_col}' and '{half_life_col}'. "
            f"Got: {list(df.columns)}"
        )
    out: dict[str, dict] = {}
    for _, row in df.iterrows():
        gene = str(row[gene_col]).strip().upper()
        if not gene or gene == "NAN":
            continue
        t_half = row[half_life_col]
        if pd.isna(t_half) or float(t_half) <= 0:
            continue
        r2 = None
        if r2_col and r2_col in df.columns and pd.notna(row[r2_col]):
            r2 = float(row[r2_col])
        out[gene] = {
            "half_life_hours": float(t_half),
            "quality_score_or_R2": r2 if r2 is not None else "",
            "anchor_weight": r2 if r2 is not None else 1.0,
            "source": TCELL_SOURCE,
            "cell_type": "Tcells",
        }
    return out


def build_rows(
    adt_df: pd.DataFrame,
    math_df: pd.DataFrame,
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

        for m in mathieson_rows_for_gene(math_df, gene, cell_types):
            rows.append({
                "protein_name": protein,
                "hgnc_id": hgnc,
                "gene_symbol": gene,
                "uniprot_id": uniprot,
                "cell_type": m["cell_type"],
                "half_life_hours": round(m["half_life_hours"], 4),
                "beta_per_hour": round(LN2 / m["half_life_hours"], 8),
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
                    "beta_per_hour": round(LN2 / t["half_life_hours"], 8),
                    "quality_score_or_R2": t["quality_score_or_R2"],
                    "source": t["source"],
                    "anchor_weight": t["anchor_weight"],
                })
    return rows


def write_csv(rows: list[dict], out: Path) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adt-mapping", type=Path, required=True,
                        help="ADT mapping CSV from tools/build_adt_mapping.py")
    parser.add_argument("--mathieson", type=Path, required=True,
                        help="Mathieson 2018 Supplementary Data 2 CSV (high-quality sheet)")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output anchor half-life CSV")
    parser.add_argument(
        "--cell-types",
        type=str,
        default="Bcells,NK,Monocytes",
        help="Comma-separated Mathieson cell types to include "
             f"(choices: {','.join(MATHIESON_CELL_TYPES)})",
    )
    parser.add_argument("--tcell-table", type=Path, default=None,
                        help="Optional processed T-cell half-life csv/xlsx")
    parser.add_argument("--tcell-gene-col", type=str, default="gene_name")
    parser.add_argument("--tcell-halflife-col", type=str, default="half_life")
    parser.add_argument("--tcell-r2-col", type=str, default="R_sq")
    parser.add_argument("--include-alignment-only", action="store_true",
                        help="Also keep ADTs with use_for_kinetics=false")
    args = parser.parse_args()

    cell_types = [c.strip() for c in args.cell_types.split(",") if c.strip()]
    unknown = [c for c in cell_types if c not in MATHIESON_CELL_TYPES]
    if unknown:
        raise ValueError(f"Unknown cell types {unknown}; choose from {list(MATHIESON_CELL_TYPES)}")

    adt_df = load_adt_mapping(args.adt_mapping)
    math_df = load_mathieson(args.mathieson)

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
    # Stable order: protein panel order, then cell type
    ct_order = {c: i for i, c in enumerate(cell_types + ["Tcells"])}
    prot_order = {n: i for i, n in enumerate(adt_df["adt_name"].tolist())}
    rows.sort(key=lambda r: (prot_order.get(r["protein_name"], 10**9), ct_order.get(r["cell_type"], 99)))

    write_csv(rows, args.out)

    n_prot = len({r["protein_name"] for r in rows})
    n_adt = int((adt_df["gene_symbol"].astype(str).str.len() > 0).sum())
    print(f"[anchors] wrote {len(rows)} rows for {n_prot} proteins → {args.out}")
    print(f"[anchors] ADTs with gene symbol: {n_adt}/{len(adt_df)}; "
          f"with ≥1 half-life: {n_prot}")
    by_ct = {}
    for r in rows:
        by_ct[r["cell_type"]] = by_ct.get(r["cell_type"], 0) + 1
    print(f"[anchors] rows by cell_type: {by_ct}")


if __name__ == "__main__":
    main()
