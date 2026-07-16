"""
Gene-to-protein projection matrix S.

S is a (D_p × D_genes) binary (or weighted) matrix where S[j, i] = 1 if
gene i encodes protein j.  Used in the KOT ODE constraint:

    J_φ(r) · v  =  κ(r) · (α(r) ⊙ S r − β ⊙ φ(r))

For CITE-seq datasets gene names match protein panel names after stripping
common suffixes (.1, -TotalSeqB, etc.).  For cases where no match is found
the implementation falls back to a uniform projection so the kinetic term
still receives a stable input.
"""
from __future__ import annotations

import re
import numpy as np
import scipy.sparse as sp


STRIP_PATTERNS = [
    r"\.1$",
    r"-TotalSeq[A-Z]$",
    r"_TotalSeq[A-Z]$",
    r"_prot$",
    r"-prot$",
]


# CITE-seq ADT markers whose panel name differs from the encoding gene symbol
# (e.g. CD16 → FCGR3A). Used as a fallback when a protein name does not directly
# match a gene. Keys and values are compared through normalize_name, so panel
# suffixes and case are handled uniformly. Markers with no single encoding gene
# (isotype controls, glycan epitopes like CD15) are intentionally absent.
CD_MARKER_TO_GENE = {
    "CD3": "CD3E", "CD8": "CD8A", "CD11a": "ITGAL", "CD11b": "ITGAM",
    "CD11c": "ITGAX", "CD16": "FCGR3A", "CD20": "MS4A1", "CD25": "IL2RA",
    "CD45": "PTPRC", "CD45RA": "PTPRC", "CD45RO": "PTPRC", "CD56": "NCAM1",
    "CD62L": "SELL", "CD127": "IL7R", "CD137": "TNFRSF9", "CD197": "CCR7",
    "CD278": "ICOS", "CD279": "PDCD1", "CD335": "NCR1", "PD-1": "PDCD1",
    "HLA-DR": "HLA-DRA",
}


def normalize_name(name: str) -> str:
    name = name.strip()
    for pat in STRIP_PATTERNS:
        name = re.sub(pat, "", name, flags=re.IGNORECASE)
    return name.upper()


def build_projection_matrix(
    rna_var_names: list[str],
    protein_var_names: list[str],
    weights: dict[str, float] | None = None,
) -> np.ndarray:
    """
    Build S matrix of shape (D_p, D_genes).

    S[j, i] = weight  if gene i encodes protein j,  else 0.
    """
    D_p = len(protein_var_names)
    D_r = len(rna_var_names)
    S = np.zeros((D_p, D_r), dtype=np.float32)

    gene_index = {normalize_name(g): i for i, g in enumerate(rna_var_names)}
    alias_index = {normalize_name(m): normalize_name(g) for m, g in CD_MARKER_TO_GENE.items()}

    matched = 0
    for j, prot in enumerate(protein_var_names):
        key = normalize_name(prot)
        if key not in gene_index and key in alias_index:
            key = alias_index[key]   # remap CITE marker name → encoding gene symbol
        if key in gene_index:
            i = gene_index[key]
            w = weights.get(prot, 1.0) if weights else 1.0
            S[j, i] = float(w)
            matched += 1

    print(f"[projection] Matched {matched}/{D_p} proteins to RNA genes.")
    if matched == 0:
        print("[projection] No name matches — using uniform S (all genes contribute equally).")
        S[:] = 1.0 / D_r
    return S


def projection_matrix_from_adatas(
    rna_adata,
    protein_adata,
    use_mean_expr: bool = False,
    use_explicit_links: bool = True,
):
    """
    Build S of shape (D_p, D_genes).

    Priority:
      1. Explicit links in adata.uns["rna_protein_feature_links"] — used for synthetic
         data and any real dataset that stores a curated gene→protein mapping.
      2. Name-matching via normalize_name — real CITE-seq where protein panel names
         correspond to gene names after stripping TotalSeq suffixes.
    use_mean_expr: weight each match by mean RNA expression.
    """
    rna_names  = list(rna_adata.var_names)
    prot_names = list(protein_adata.var_names)

    weights = None
    if use_mean_expr:
        X = rna_adata.X
        mean_expr = np.asarray(X.mean(axis=0)).ravel() if sp.issparse(X) else X.mean(axis=0)
        weights = {g: float(mean_expr[i]) for i, g in enumerate(rna_names)}

    links = None
    if use_explicit_links:
        links = rna_adata.uns.get("rna_protein_feature_links")
        if links is None:
            links = protein_adata.uns.get("rna_protein_feature_links")

    if links is not None:
        link_genes   = links.get("rna",     links.get("gene",   links.get("source", [])))
        link_prots   = links.get("protein", links.get("target", []))
        link_weights = links.get("weight",  [1.0] * len(link_genes))

        rna_index  = {g: i for i, g in enumerate(rna_names)}
        prot_index = {p: j for j, p in enumerate(prot_names)}

        D_p, D_r = len(prot_names), len(rna_names)
        S = np.zeros((D_p, D_r), dtype=np.float32)
        matched = 0
        for gene, prot, w in zip(link_genes, link_prots, link_weights):
            i = rna_index.get(str(gene))
            j = prot_index.get(str(prot))
            if i is not None and j is not None:
                S[j, i] = float(weights[str(gene)]) if (use_mean_expr and weights) else float(w)
                matched += 1
        if matched > 0:
            print(f"[projection] Used explicit links: {matched}/{D_p} proteins mapped.")
            return S
        print("[projection] Explicit links present but no var_name matches — falling back to name matching.")

    return build_projection_matrix(rna_names, prot_names, weights=weights)
