"""
Gene-to-protein projection matrix S.

S is a (D_p × D_genes) binary (or weighted) matrix where S[j, i] = 1 if
gene i encodes protein j.  Used in the KOT ODE constraint:

    J_φ(r) · v  =  κ(r) · (α(r) ⊙ S r − β ⊙ φ(r))

For CITE-seq datasets gene names match protein panel names after stripping
common suffixes (.1, -TotalSeqB, etc.).  For cases where no match is found
the row is left as all-zeros (protein has no measured mRNA transcript).
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

    matched = 0
    for j, prot in enumerate(protein_var_names):
        key = normalize_name(prot)
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
