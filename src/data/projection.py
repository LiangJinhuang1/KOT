"""
Gene-to-protein projection matrix S, with alignment / kinetics masks and a report.

S is a (D_p × D_genes) binary (or weighted) matrix where S[j, i] = 1 if gene i
encodes protein j.  Used in the KOT ODE constraint:

    J_φ(r) · v  =  κ(r) · (α(r) ⊙ S r − β ⊙ φ(r))

The builder returns four objects, all indexed by the protein panel order:

    S               (D_p, D_genes)  projection matrix
    alignment_mask  (D_p,) bool     proteins that feed the Sinkhorn alignment
    kinetic_mask    (D_p,) bool     proteins that feed the kinetic residual
    mapping_report  list[dict]      one row per protein: mapped gene, mapping type,
                                    alignment / kinetic flags, present_in_rna, reason

Source of truth: if a curated mapping CSV is provided (via `mapping_records`), its
decisions are authoritative — φ still predicts the full panel, Sinkhorn uses the
`use_for_alignment` proteins, the kinetic loss uses the `use_for_kinetics` proteins,
and a protein excluded from kinetics gets an all-zero S row (no uniform fallback).
Without a CSV we fall back to explicit links (synthetic) or HGNC name matching, in
which case every measured protein feeds alignment and the linked ones feed kinetics.
"""
from __future__ import annotations

import re
import numpy as np
import scipy.sparse as sp

from src.data.adt_gene_map import adt_to_gene, gene_alias_lookup


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


def report_row(protein, gene, mapping_type, present, use_align, use_kin, in_s, reason):
    return {
        "protein":           protein,
        "gene_symbol":       gene or "",
        "mapping_type":      mapping_type,
        "present_in_rna":    bool(present),
        "use_for_alignment": bool(use_align),
        "use_for_kinetics":  bool(use_kin),
        "in_S":              bool(in_s),
        "excluded_because":  reason,
    }


def assert_no_orphan_s_rows(S: np.ndarray, kinetic_mask: np.ndarray) -> None:
    """A protein excluded from kinetics must have an all-zero S row (no fallback)."""
    orphan = (S != 0).any(axis=1) & ~kinetic_mask
    if orphan.any():
        raise ValueError(
            f"{int(orphan.sum())} protein(s) have a non-zero S row but kinetic_mask=False"
        )


def build_projection_from_records(
    rna_var_names: list[str],
    protein_var_names: list[str],
    records: list[dict],
    weights: dict[str, float] | None = None,
    require_full_panel: bool = False,
    rna_var=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """
    Build (S, alignment_mask, kinetic_mask, report) from a curated mapping CSV that
    is treated as the source of truth. A protein enters S / the kinetic residual
    only if its record says use_for_kinetics AND its gene is present in the current
    RNA matrix; otherwise its S row stays zero.
    """
    D_p, D_r = len(protein_var_names), len(rna_var_names)
    S = np.zeros((D_p, D_r), dtype=np.float32)
    alignment_mask = np.zeros(D_p, dtype=bool)
    kinetic_mask = np.zeros(D_p, dtype=bool)
    gene_index = gene_alias_lookup(rna_var_names, rna_var, normalizer=normalize_name)
    by_name = {r["adt_name"]: r for r in records}

    report: list[dict] = []
    missing: list[str] = []
    for j, prot in enumerate(protein_var_names):
        rec = by_name.get(prot)
        if rec is None:
            # Panel protein absent from the CSV: φ still predicts it, keep it in
            # alignment, but it cannot enter kinetics (no curated decision / gene).
            missing.append(prot)
            alignment_mask[j] = True
            report.append(report_row(prot, "", "", False, True, False, False,
                                     "not in mapping CSV"))
            continue

        gene = rec["gene_symbol"]
        present = bool(gene) and normalize_name(gene) in gene_index
        alignment_mask[j] = rec["use_for_alignment"]
        reason = rec["excluded_because"]
        effective_kin = bool(rec["use_for_kinetics"] and present)
        if rec["use_for_kinetics"] and not present:
            reason = reason or f"gene {gene!r} absent from current RNA matrix"
            print(f"[projection] WARNING: {prot} use_for_kinetics=True but gene "
                  f"{gene!r} not in RNA matrix — masked from kinetics.")
        if effective_kin:
            i = gene_index[normalize_name(gene)]
            rna_name = str(rna_var_names[i])
            if weights:
                S[j, i] = float(weights.get(rna_name, weights.get(gene, weights.get(prot, 1.0))))
            else:
                S[j, i] = 1.0
            kinetic_mask[j] = True
        report.append(report_row(prot, gene, rec["mapping_type"], present,
                                  alignment_mask[j], effective_kin,
                                  kinetic_mask[j], reason))

    if missing:
        if require_full_panel:
            raise ValueError(
                f"{len(missing)} panel protein(s) are not in the mapping CSV: "
                f"{missing[:10]}{' ...' if len(missing) > 10 else ''}. Rebuild the mapping for "
                f"this exact panel (tools/build_adt_mapping.py) — with require_full_panel, "
                f"partial panel coverage is a hard error (a protein with no mapping cannot align)."
            )
        print(f"[projection] WARNING: {len(missing)} panel protein(s) not in mapping CSV "
              f"(kept in alignment, excluded from kinetics).")
    assert_no_orphan_s_rows(S, kinetic_mask)
    print(f"[projection] CSV source-of-truth: kinetics {int(kinetic_mask.sum())}/{D_p}, "
          f"alignment {int(alignment_mask.sum())}/{D_p}.")
    return S, alignment_mask, kinetic_mask, report


def build_projection_matrix(
    rna_var_names: list[str],
    protein_var_names: list[str],
    weights: dict[str, float] | None = None,
    alias_map: dict[str, str] | None = None,
    rna_var=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """
    Name-matching fallback (no curated CSV): build S by matching each protein to a
    gene via normalized name or alias_map. Every measured protein feeds alignment;
    linked proteins feed kinetics. Unmapped proteins keep an all-zero S row (no
    uniform fallback). Returns (S, alignment_mask, kinetic_mask, report).
    """
    D_p = len(protein_var_names)
    D_r = len(rna_var_names)
    S = np.zeros((D_p, D_r), dtype=np.float32)
    kinetic_mask = np.zeros(D_p, dtype=bool)
    alignment_mask = np.ones(D_p, dtype=bool)   # every measured protein feeds alignment

    gene_index = gene_alias_lookup(rna_var_names, rna_var, normalizer=normalize_name)
    alias_map = alias_map or {}

    report: list[dict] = []
    for j, prot in enumerate(protein_var_names):
        key = normalize_name(prot)
        gene = None
        if key in gene_index:
            gene = prot
        elif prot in alias_map:
            gene = alias_map[prot]
            key = normalize_name(gene)
        if gene is not None and key in gene_index:
            i = gene_index[key]
            rna_name = str(rna_var_names[i])
            w = weights.get(rna_name, weights.get(gene, weights.get(prot, 1.0))) if weights else 1.0
            S[j, i] = float(w)
            kinetic_mask[j] = True
            report.append(report_row(prot, gene, "name_match", True, True, True, True, ""))
        else:
            report.append(report_row(prot, "", "unmapped", False, True, False, False,
                                      "no gene match (alignment-only)"))

    assert_no_orphan_s_rows(S, kinetic_mask)
    print(f"[projection] Matched {int(kinetic_mask.sum())}/{D_p} proteins to RNA genes "
          f"({int(kinetic_mask.sum())} used for kinetics, {D_p - int(kinetic_mask.sum())} alignment-only).")
    return S, alignment_mask, kinetic_mask, report


def projection_matrix_from_adatas(
    rna_adata,
    protein_adata,
    mapping_records: list[dict] | None = None,
    use_mean_expr: bool = False,
    use_explicit_links: bool = True,
    alias_map: dict[str, str] | None = None,
    require_full_panel: bool = False,
):
    """
    Build (S, alignment_mask, kinetic_mask, mapping_report), indexed by the protein
    panel order.

    Priority:
      1. mapping_records — a curated CSV (source of truth). Explicit links and live
         name matching are NOT consulted; the CSV's decisions win.
      2. Explicit links in adata.uns["rna_protein_feature_links"] (synthetic data).
      3. HGNC alias / name matching (real CITE-seq without a CSV).
    use_mean_expr weights each match by mean RNA expression.
    """
    rna_names  = list(rna_adata.var_names)
    prot_names = list(protein_adata.var_names)

    weights = None
    if use_mean_expr:
        X = rna_adata.X
        mean_expr = np.asarray(X.mean(axis=0)).ravel() if sp.issparse(X) else X.mean(axis=0)
        weights = {g: float(mean_expr[i]) for i, g in enumerate(rna_names)}

    # 1. Curated mapping CSV is authoritative.
    if mapping_records is not None:
        return build_projection_from_records(
            rna_names, prot_names, mapping_records, weights,
            require_full_panel=require_full_panel,
            rna_var=rna_adata.var,
        )

    # 2. Explicit curated links (synthetic / stored mapping).
    links = None
    if use_explicit_links:
        links = rna_adata.uns.get("rna_protein_feature_links")
        if links is None:
            links = protein_adata.uns.get("rna_protein_feature_links")

    if links is not None:
        link_genes   = links.get("rna",     links.get("gene",   links.get("source", [])))
        link_prots   = links.get("protein", links.get("target", []))
        link_weights = links.get("weight",  [1.0] * len(link_genes))

        rna_index  = gene_alias_lookup(rna_names, rna_adata.var, normalizer=normalize_name)
        prot_index = {p: j for j, p in enumerate(prot_names)}

        D_p, D_r = len(prot_names), len(rna_names)
        S = np.zeros((D_p, D_r), dtype=np.float32)
        kinetic_mask = np.zeros(D_p, dtype=bool)
        alignment_mask = np.ones(D_p, dtype=bool)
        linked_gene = {}
        for gene, prot, w in zip(link_genes, link_prots, link_weights):
            i = rna_index.get(normalize_name(str(gene)))
            j = prot_index.get(str(prot))
            if i is not None and j is not None:
                rna_name = str(rna_names[i])
                S[j, i] = float(weights[rna_name]) if (use_mean_expr and weights) else float(w)
                kinetic_mask[j] = True
                linked_gene[j] = str(gene)
        if kinetic_mask.any():
            assert_no_orphan_s_rows(S, kinetic_mask)
            report = [
                report_row(p, linked_gene.get(j, ""), "explicit_link",
                           j in linked_gene, True, bool(kinetic_mask[j]),
                           bool(kinetic_mask[j]), "" if kinetic_mask[j] else "no explicit link")
                for j, p in enumerate(prot_names)
            ]
            print(f"[projection] Used explicit links: {int(kinetic_mask.sum())}/{D_p} proteins mapped.")
            return S, alignment_mask, kinetic_mask, report
        print("[projection] Explicit links present but no var_name matches — falling back to name matching.")

    # 3. HGNC alias / name matching.
    if alias_map is None:
        alias_map = adt_to_gene(prot_names)
    return build_projection_matrix(
        rna_names, prot_names, weights=weights, alias_map=alias_map, rna_var=rna_adata.var,
    )
