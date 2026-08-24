
#!/usr/bin/env python3
"""
Diagnose why ADT-target genes are (not) scVelo velocity_genes.

For the genes behind the protein panel, report the scVelo fit stats that decide
`velocity_genes`, so we can tell a gene with *real dynamics but a strict
threshold* (recoverable) from a gene at *transcriptional steady state* (velocity
~ 0, not recoverable, and useless as a kinetic anchor even if forced in).

Run on the cluster, where the retained velocity h5ad and scVelo live:
  python tools/diagnose_velocity_genes.py \
      --rna cache/velocity/<dataset>/<dataset>_scvelo_results_retained.h5ad \
      --coverage cache/results/mapping/kinetics_coverage_pbmc_retained.csv

Reports, per ADT gene: velocity_genes flag, fit r2, likelihood, and the
unspliced fraction; plus how many currently-excluded genes a lower min_r2 admits.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import anndata as ad

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.adt_gene_map import gene_alias_lookup, normalize_gene_symbol


def col(var, names):
    """First matching var column as a float array, else None."""
    for n in names:
        if n in var.columns:
            return var[n].to_numpy(dtype=float)
    return None


def unspliced_fraction(adata):
    """Per-gene mean unspliced / (spliced + unspliced) from the raw layers."""
    if "unspliced" not in adata.layers or "spliced" not in adata.layers:
        return None
    u = np.asarray(adata.layers["unspliced"].sum(axis=0)).ravel()
    s = np.asarray(adata.layers["spliced"].sum(axis=0)).ravel()
    denom = u + s
    return np.divide(u, denom, out=np.zeros_like(u, dtype=float), where=denom > 0)


def adt_genes_from_coverage(path: Path) -> dict[str, dict]:
    out = {}
    with path.open() as handle:
        for r in csv.DictReader(handle):
            g = (r.get("gene_symbol") or "").strip()
            if g:
                out[normalize_gene_symbol(g)] = r
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rna", type=Path, required=True, help="retained velocity h5ad")
    parser.add_argument("--coverage", type=Path, required=True, help="kinetics coverage CSV")
    parser.add_argument("--min-r2-grid", type=str, default="0.01,0.005,0.001,0.0",
                        help="candidate min_r2 thresholds to count admitted genes")
    args = parser.parse_args()

    adata = ad.read_h5ad(args.rna)
    var = adata.var
    vgenes = var["velocity_genes"].to_numpy(dtype=bool) if "velocity_genes" in var else None
    r2 = col(var, ["velocity_r2", "fit_r2", "r2"])
    lik = col(var, ["fit_likelihood", "velocity_score"])
    ufrac = unspliced_fraction(adata)

    cov = adt_genes_from_coverage(args.coverage)
    gene_pos = gene_alias_lookup(var.index.astype(str), var, normalizer=normalize_gene_symbol)

    print(f"{'gene':<10}{'in_var':<7}{'vel_gene':<9}{'r2':>9}{'likelihood':>12}{'u_frac':>8}")
    recoverable, steady, unknown, absent = [], [], [], []
    for g in sorted(cov):
        i = gene_pos.get(g)
        if i is None:
            print(f"{g:<10}{'no':<7}")
            absent.append(g)
            continue
        vg = bool(vgenes[i]) if vgenes is not None else None
        r = r2[i] if r2 is not None else float("nan")
        lk = lik[i] if lik is not None else float("nan")
        uf = ufrac[i] if ufrac is not None else float("nan")
        print(f"{g:<10}{'yes':<7}{str(vg):<9}{r:>9.4f}{lk:>12.4f}{uf:>8.3f}")
        if vg is False:
            if uf < 0.05 or (not np.isnan(r) and r < 1e-3):
                steady.append(g)
            elif np.isnan(r):
                unknown.append(g)
            else:
                recoverable.append(g)

    print("\n--- admitted ADT genes at candidate min_r2 (of currently-excluded) ---")
    grid = [float(x) for x in args.min_r2_grid.split(",")]
    excluded = [g for g in cov if gene_pos.get(g) is not None
                and vgenes is not None and not vgenes[gene_pos[g]]]
    for thr in grid:
        n = sum(1 for g in excluded
                if r2 is not None and not np.isnan(r2[gene_pos[g]]) and r2[gene_pos[g]] >= thr)
        print(f"  min_r2 >= {thr:<7}: {n}/{len(excluded)} excluded ADT genes would be admitted")

    print(f"\nsummary: recoverable(real dynamics, low r2)={len(recoverable)}  "
          f"steady-state(u_frac<0.05 or r2~0)={len(steady)}  "
          f"unknown_missing_r2={len(unknown)}  not-in-var={len(absent)}")
    if recoverable:
        print("  recoverable -> " + ", ".join(recoverable))
    if steady:
        print("  steady-state (forcing them in will not help) -> " + ", ".join(steady))
    if unknown:
        print("  unknown (missing r2; inspect fit columns before relaxing thresholds) -> " + ", ".join(unknown))


if __name__ == "__main__":
    main()
