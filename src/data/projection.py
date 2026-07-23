"""
Gene-to-protein projection matrix S.

S is a (D_p × D_genes) binary (or weighted) matrix where S[j, i] = 1 if
gene i encodes protein j.  Used in the KOT ODE constraint:

    J_φ(r) · v  =  κ(r) · (α(r) ⊙ S r − β ⊙ φ(r))

For CITE-seq datasets a protein matches a gene by normalized name, or via the
HGNC-backed ADT→gene map (src/data/adt_gene_map.py) when the panel name differs
from the gene symbol (CD16 → FCGR3A). If no match is found the implementation
falls back to a uniform projection so the kinetic term still receives a stable
input.
"""
from __future__ import annotations

import re
import numpy as np
import scipy.sparse as sp

from src.data.adt_gene_map import adt_to_gene


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
    alias_map: dict[str, str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build S matrix of shape (D_p, D_genes) and the per-protein kinetics mask.

    S[j, i] = weight  if gene i encodes protein j,  else 0.
    mask[j] = True     if protein j has a gene→protein link (used in the ODE),
              False    if unmapped (alignment-only, no kinetics — S row stays 0).

    A protein matches a gene either by normalized name, or via alias_map (exact
    ADT name → gene symbol, e.g. from HGNC) when the names differ. Unmapped
    proteins keep an all-zero S row rather than a uniform fallback, so the
    kinetic term is never fed a meaningless "all genes encode this protein" prior.
    """
    D_p = len(protein_var_names)
    D_r = len(rna_var_names)
    S = np.zeros((D_p, D_r), dtype=np.float32)
    mask = np.zeros(D_p, dtype=bool)

    gene_index = {normalize_name(g): i for i, g in enumerate(rna_var_names)}
    alias_map = alias_map or {}

    for j, prot in enumerate(protein_var_names):
        key = normalize_name(prot)
        if key not in gene_index and prot in alias_map:
            key = normalize_name(alias_map[prot])   # ADT marker → encoding gene symbol
        if key in gene_index:
            i = gene_index[key]
            w = weights.get(prot, 1.0) if weights else 1.0
            S[j, i] = float(w)
            mask[j] = True

    print(f"[projection] Matched {int(mask.sum())}/{D_p} proteins to RNA genes "
          f"({int(mask.sum())} used for kinetics, {D_p - int(mask.sum())} alignment-only).")
    return S, mask


def projection_matrix_from_adatas(
    rna_adata,
    protein_adata,
    use_mean_expr: bool = False,
    use_explicit_links: bool = True,
    alias_map: dict[str, str] | None = None,
):
    """
    Build S of shape (D_p, D_genes).

    Returns (S, mask): S is (D_p, D_genes); mask[j] is True when protein j has a
    gene→protein link (participates in the ODE / kinetics), False when it is
    alignment-only.

    Priority:
      1. Explicit links in adata.uns["rna_protein_feature_links"] — used for synthetic
         data and any real dataset that stores a curated gene→protein mapping.
      2. Name-matching via normalize_name / HGNC alias map — real CITE-seq.
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
        mask = np.zeros(D_p, dtype=bool)
        for gene, prot, w in zip(link_genes, link_prots, link_weights):
            i = rna_index.get(str(gene))
            j = prot_index.get(str(prot))
            if i is not None and j is not None:
                S[j, i] = float(weights[str(gene)]) if (use_mean_expr and weights) else float(w)
                mask[j] = True
        if mask.any():
            print(f"[projection] Used explicit links: {int(mask.sum())}/{D_p} proteins mapped.")
            return S, mask
        print("[projection] Explicit links present but no var_name matches — falling back to name matching.")

    # alias_map from a curated mapping CSV if given, else resolve live via HGNC.
    if alias_map is None:
        alias_map = adt_to_gene(prot_names)
    return build_projection_matrix(rna_names, prot_names, weights=weights, alias_map=alias_map)
