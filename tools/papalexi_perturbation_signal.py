"""Does Papalexi's CRISPR screen contain a measurable perturbation -> protein effect?

Model-free feasibility check for reframing Papalexi from a FOSCTTM alignment
benchmark into a perturbation-response benchmark. Nothing here touches KOT or any
baseline: it only asks whether the ground truth an evaluation would score against
actually exists in the data.

Three questions, in order of how fatal a "no" would be:

  1. Positive controls. CD86 and PDCD1LG2 are both knockout targets *and* ADT
     targets. Their own protein must drop in their own knockout. If it does not,
     the guide assignments or the ADT matrix are unusable and nothing downstream
     can be trusted.
  2. Trans effects. How many (knockout, protein) pairs move significantly? That
     count is the size of the evaluation set a benchmark would have.
  3. mRNA/protein discordance. Pairs where the protein moves but the target gene's
     mRNA does not are the ones a central-dogma model can be scored on; pairs where
     both move are already explained by copying the mRNA effect.

ADT normalization is deliberately reported three ways. A 4-plex panel has no
isotype controls and its per-cell total is dominated by the four measured
proteins, so library-size normalization is compositional: one protein falling
pushes the other three up. Conclusions that survive all three variants are real.

Run:
  singularity exec --pwd $(pwd) /data/common/images/codedev_v1.0.5.sif \
      python -u tools/papalexi_perturbation_signal.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse, stats

# ADT name in the protein h5ad -> RNA gene symbol encoding it.
ADT_GENE_MAP = {
    "CD86": "CD86",
    "PDL1": "CD274",
    "PDL2": "PDCD1LG2",
    "CD366": "HAVCR2",
}

CONTROL_LABEL = "NT"
MIN_CELLS = 40


def to_dense(x) -> np.ndarray:
    return np.asarray(x.todense()) if sparse.issparse(x) else np.asarray(x)


def load_protein(path: Path) -> ad.AnnData:
    protein = ad.read_h5ad(path)
    print(f"[protein] {protein.n_obs} cells x {protein.n_vars} ADTs: {list(protein.var_names)}")
    unknown = set(map(str, protein.var_names)) - set(ADT_GENE_MAP)
    if unknown:
        raise ValueError(
            f"ADT names {sorted(unknown)} are not in ADT_GENE_MAP {sorted(ADT_GENE_MAP)}; "
            "the protein/mRNA merge would silently drop them. Update ADT_GENE_MAP."
        )
    return protein


def load_metadata(path: Path) -> pd.DataFrame:
    meta = pd.read_csv(path, sep="\t", index_col=0)
    meta.index = meta.index.astype(str)
    print(f"[metadata] {len(meta)} cells, columns: {list(meta.columns)}")
    return meta


def load_target_gene_expression(path: Path, genes: list[str]) -> pd.DataFrame:
    """log1p CP10K expression of `genes`, streamed in row blocks so the 1.4GB
    matrix is never held in memory. X is spliced+unspliced counts, identical to
    the `counts` layer written by tools/build_papalexi.py."""
    backed = ad.read_h5ad(path, backed="r")
    var_names = pd.Index(backed.var_names.astype(str))
    present = [g for g in genes if g in var_names]
    missing = [g for g in genes if g not in var_names]
    if missing:
        print(f"[rna] WARNING: genes absent from the RNA matrix: {missing}")
    cols = [int(var_names.get_loc(g)) for g in present]

    n_obs = backed.n_obs
    totals = np.zeros(n_obs, dtype=np.float64)
    sub = np.zeros((n_obs, len(cols)), dtype=np.float64)
    block = 4000
    for start in range(0, n_obs, block):
        end = min(start + block, n_obs)
        chunk = backed.X[start:end]
        dense_cols = to_dense(chunk[:, cols])
        totals[start:end] = np.asarray(chunk.sum(axis=1)).ravel()
        sub[start:end] = dense_cols

    obs_names = pd.Index(backed.obs_names.astype(str))
    backed.file.close()

    cp10k = sub / np.maximum(totals[:, None], 1.0) * 1e4
    out = pd.DataFrame(np.log1p(cp10k), index=obs_names, columns=present)
    print(f"[rna] target-gene expression: {out.shape[0]} cells x {out.shape[1]} genes")
    return out


def normalize_adt(counts: np.ndarray, rna_totals: np.ndarray) -> dict[str, np.ndarray]:
    """Three ADT normalizations with different compositional exposure.

    raw_log1p    no size factor at all; immune to composition, sensitive to depth.
    rna_size     size factor from total RNA counts, i.e. outside the 4-plex panel,
                 so a drop in one protein cannot inflate the other three.
    clr          the CITE-seq convention, included because reviewers expect it;
                 fully compositional across only 4 features.
    """
    raw_log1p = np.log1p(counts)

    size = np.maximum(rna_totals, 1.0)
    rna_size = np.log1p(counts / size[:, None] * float(np.median(size)))

    logged = np.log1p(counts)
    clr = logged - logged.mean(axis=1, keepdims=True)

    return {"raw_log1p": raw_log1p, "rna_size": rna_size, "clr": clr}


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """BH-adjusted p-values over the whole (knockout, protein) grid.

    ``p * n / rank`` on its own is not the BH q-value. Without the running minimum
    taken from the largest p downwards the result is not monotone in p, so a pair
    can end up with a smaller q than a pair with a smaller p, and a q < alpha cut
    then keeps an inconsistent set. The benchmark's scoring set is chosen by
    exactly that cut, so the step-up matters here.
    """
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    stepped = p[order] * n / np.arange(1, n + 1)
    stepped = np.minimum.accumulate(stepped[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.minimum(stepped, 1.0)
    return q


def effect_table(values: np.ndarray, adt_names: list[str], groups: pd.Series) -> pd.DataFrame:
    """Per (knockout, protein) effect versus the non-targeting control."""
    is_control = (groups == CONTROL_LABEL).to_numpy()
    control = values[is_control]
    targets = [g for g in sorted(groups.unique()) if g != CONTROL_LABEL]

    rows = []
    for target in targets:
        mask = (groups == target).to_numpy()
        if mask.sum() < MIN_CELLS:
            continue
        ko = values[mask]
        for j, adt in enumerate(adt_names):
            a, b = ko[:, j], control[:, j]
            pooled_sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
            _, pval = stats.mannwhitneyu(a, b, alternative="two-sided")
            rows.append({
                "knockout": target,
                "adt": adt,
                "n_ko": int(mask.sum()),
                "delta": float(a.mean() - b.mean()),
                "cohens_d": float((a.mean() - b.mean()) / pooled_sd) if pooled_sd > 0 else np.nan,
                "pval": float(pval),
            })

    df = pd.DataFrame(rows)
    df["qval"] = benjamini_hochberg(df["pval"].to_numpy())
    return df


def rna_effect_table(expr: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """Same contrast on the ADT-target genes' mRNA, for the discordance analysis."""
    gene_to_adt = {gene: adt for adt, gene in ADT_GENE_MAP.items()}
    is_control = (groups == CONTROL_LABEL).to_numpy()
    targets = [g for g in sorted(groups.unique()) if g != CONTROL_LABEL]

    rows = []
    for target in targets:
        mask = (groups == target).to_numpy()
        if mask.sum() < MIN_CELLS:
            continue
        for gene in expr.columns:
            a = expr[gene].to_numpy()[mask]
            b = expr[gene].to_numpy()[is_control]
            rows.append({
                "knockout": target,
                "adt": gene_to_adt[gene],
                "rna_delta": float(a.mean() - b.mean()),
            })
    return pd.DataFrame(rows)


def report(effects: dict[str, pd.DataFrame], rna: pd.DataFrame, out_dir: Path) -> None:
    primary = effects["rna_size"]

    print("\n" + "=" * 78)
    print("1. POSITIVE CONTROLS — a knockout of the gene encoding a measured protein")
    print("=" * 78)
    for knockout, adt in [("CD86", "CD86"), ("PDCD1LG2", "PDL2")]:
        print(f"\n  {knockout} KO -> {adt} protein")
        for label, df in effects.items():
            hit = df[(df["knockout"] == knockout) & (df["adt"] == adt)]
            if hit.empty:
                print(f"    {label:<10} (knockout absent or below {MIN_CELLS} cells)")
                continue
            row = hit.iloc[0]
            print(f"    {label:<10} delta={row['delta']:+.4f}  d={row['cohens_d']:+.3f}  "
                  f"q={row['qval']:.2e}  n={row['n_ko']}")

    print("\n" + "=" * 78)
    print("2. EVALUATION SET SIZE — significant (knockout, protein) pairs, rna_size")
    print("=" * 78)
    for q in (0.05, 0.01):
        for d in (0.1, 0.2, 0.5):
            n = int(((primary["qval"] < q) & (primary["cohens_d"].abs() > d)).sum())
            print(f"  q<{q:<5} |d|>{d:<4} : {n:3d} / {len(primary)} pairs")

    print("\n  Top 20 effects by |Cohen's d|:")
    top = primary.reindex(primary["cohens_d"].abs().sort_values(ascending=False).index).head(20)
    for _, row in top.iterrows():
        print(f"    {row['knockout']:<10} {row['adt']:<6} delta={row['delta']:+.4f}  "
              f"d={row['cohens_d']:+.3f}  q={row['qval']:.2e}  n={int(row['n_ko'])}")

    print("\n" + "=" * 78)
    print("3. mRNA/PROTEIN DISCORDANCE — where a central-dogma model can be scored")
    print("=" * 78)
    merged = primary.merge(rna, on=["knockout", "adt"], how="left")
    sig = merged[(merged["qval"] < 0.05) & (merged["cohens_d"].abs() > 0.1)].copy()
    if sig.empty:
        print("  No significant pairs — the discordance analysis is not defined.")
    else:
        rho = stats.spearmanr(sig["rna_delta"], sig["delta"])
        print(f"  Across {len(sig)} significant pairs, Spearman(mRNA delta, protein delta) = "
              f"{rho.statistic:+.3f} (p={rho.pvalue:.2e})")
        print("\n  Pairs where protein moves but mRNA barely does (|rna_delta| < 0.05):")
        disc = sig[sig["rna_delta"].abs() < 0.05]
        if disc.empty:
            print("    none")
        for _, row in disc.reindex(
            disc["cohens_d"].abs().sort_values(ascending=False).index
        ).iterrows():
            print(f"    {row['knockout']:<10} {row['adt']:<6} protein_delta={row['delta']:+.4f} "
                  f"(d={row['cohens_d']:+.3f})  rna_delta={row['rna_delta']:+.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_dir / "papalexi_perturbation_effects.csv", index=False)
    for label, df in effects.items():
        df.to_csv(out_dir / f"papalexi_perturbation_effects_{label}.csv", index=False)
    print(f"\n[out] wrote effect tables to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protein", type=Path,
                        default=Path("Datasets/Papalexi_GSE153056/papalexi_protein.h5ad"))
    parser.add_argument("--rna", type=Path,
                        default=Path("Datasets/Papalexi_GSE153056/papalexi_rna_input.h5ad"))
    parser.add_argument("--metadata", type=Path, default=Path("data/papalexi/ECCITE_metadata.tsv"))
    parser.add_argument("--out-dir", type=Path, default=Path("cache/results/papalexi_perturbation"))
    args = parser.parse_args()

    protein = load_protein(args.protein)
    meta = load_metadata(args.metadata)
    expr = load_target_gene_expression(args.rna, list(ADT_GENE_MAP.values()))

    cells = protein.obs_names.intersection(meta.index).intersection(expr.index)
    print(f"[join] protein n meta n rna = {len(cells)} cells")
    protein = protein[cells].copy()
    meta = meta.loc[cells]
    expr = expr.loc[cells]

    groups = meta["gene"].astype(str)
    counts = to_dense(protein.X).astype(np.float64)
    adt_names = [str(v) for v in protein.var_names]

    rna_totals = np.asarray(meta["nCount_RNA"], dtype=np.float64)
    variants = normalize_adt(counts, rna_totals)

    print(f"\n[groups] {groups.nunique()} labels, "
          f"{int((groups == CONTROL_LABEL).sum())} control cells")

    effects = {label: effect_table(vals, adt_names, groups) for label, vals in variants.items()}
    report(effects, rna_effect_table(expr, groups), args.out_dir)


if __name__ == "__main__":
    main()
