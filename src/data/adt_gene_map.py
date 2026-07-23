"""
Resolve CITE-seq ADT marker names to their encoding gene symbols.

Antibody panel names (CD16, HLA-DR, CD8a) rarely equal the HGNC gene symbol
(FCGR3A, HLA-DRA, CD8A). We resolve them authoritatively from the HGNC gene
table (approved / previous / alias symbols), plus a small curated map for the
few markers that name a protein *complex* or *isoform epitope* rather than a
single gene, which HGNC cannot resolve to one symbol.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

HGNC_PATH = Path("Datasets/HGNC_set.txt")

# ADT markers naming a complex or isoform epitope → chosen encoding gene.
# HGNC has no single-gene alias for these, so they are curated (documented biology).
MANUAL_ADT_TO_GENE = {
    "CD3":        "CD3E",    # CD3/TCR complex → representative chain
    "CD45":       "PTPRC",
    "CD45RA":     "PTPRC",   # isoform epitope of CD45
    "CD45RO":     "PTPRC",
    "HLA-DR":     "HLA-DRA",
    "IgM":        "IGHM",
    "IgD":        "IGHD",
    "integrinB7": "ITGB7",
}


def alnum(name: str) -> str:
    """Aggressive normalisation for cross-source name matching: keep A–Z0–9 only."""
    return re.sub(r"[^A-Z0-9]", "", name.upper())


def strip_adt_suffix(name: str) -> str:
    """Drop the TotalSeq panel suffix (CD8a_TotalSeqB → CD8a)."""
    return re.sub(r"_TotalSeq[A-Z]$", "", name)


def load_hgnc_records(hgnc_path: Path = HGNC_PATH) -> dict[str, tuple[str, str]]:
    """Map every known gene name (approved / previous / alias) → (symbol, HGNC ID)."""
    records: dict[str, tuple[str, str]] = {}
    with open(hgnc_path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            approved = row["Approved symbol"].strip()
            if not approved:
                continue
            hgnc_id = (row.get("HGNC ID", "") or "").strip()
            names = [approved]
            for col in ("Previous symbols", "Alias symbols"):
                names += [p.strip() for p in re.split(r"[,|]", row.get(col, "") or "") if p.strip()]
            for nm in names:
                records.setdefault(alnum(nm), (approved, hgnc_id))
    return records


def resolve_marker(name: str, records: dict, manual: dict) -> tuple[str | None, str, str]:
    """
    Resolve one ADT marker → (gene_symbol, hgnc_id, mapping_type).

    mapping_type is one of:
      isotype         — isotype control, no target gene
      complex_curated — multi-gene complex / isoform epitope, curated to one gene
      one_to_one      — resolved to a single gene via HGNC
      unmapped        — no single encoding gene (uncurated complex / not found)
    """
    if "control" in name.lower():
        return None, "", "isotype"
    key = alnum(strip_adt_suffix(name))
    if key in manual:
        gene = manual[key]
        rec = records.get(alnum(gene))
        return gene, (rec[1] if rec else ""), "complex_curated"
    if key in records:
        symbol, hgnc_id = records[key]
        return symbol, hgnc_id, "one_to_one"
    return None, "", "unmapped"


def adt_to_gene(protein_names, hgnc_path: Path = HGNC_PATH) -> dict[str, str]:
    """
    Resolve ADT marker names → gene symbols: curated complex/isoform map first,
    then HGNC aliases. Markers with no single encoding gene (isotype controls,
    glycan epitopes, multi-gene complexes like TCR / HLA-A-B-C) are simply absent
    from the result. Returns {} when the HGNC table is unavailable, so callers
    fall back to plain name matching.
    """
    if not hgnc_path.exists():
        return {}
    records = load_hgnc_records(hgnc_path)
    manual = {alnum(k): v for k, v in MANUAL_ADT_TO_GENE.items()}
    mapping: dict[str, str] = {}
    for name in protein_names:
        gene, _, _ = resolve_marker(name, records, manual)
        if gene is not None:
            mapping[name] = gene
    return mapping


def load_mapping_csv(path: Path) -> dict[str, str]:
    """
    {adt_name: gene_symbol} from a mapping CSV written by build_adt_mapping.

    Every row carrying a gene symbol is included; whether that gene is actually
    present in the RNA matrix (hence whether the protein enters kinetics) is
    decided live in build_projection_matrix, so the CSV stays valid even if the
    RNA gene set changes (e.g. after a velocity re-run).
    """
    mapping: dict[str, str] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            gene = (row.get("gene_symbol") or "").strip()
            if gene:
                mapping[row["adt_name"]] = gene
    return mapping


def build_mapping_rows(protein_names, rna_symbols=None, hgnc_path: Path = HGNC_PATH) -> list[dict]:
    """
    One row per ADT marker with its gene resolution and usage flags.

    use_for_alignment is always True (every measured protein feeds the Sinkhorn
    alignment). use_for_kinetics is True only when the marker resolves to a gene
    that is present in the RNA matrix — those are the proteins the ODE / S term
    can constrain. rna_symbols=None leaves present_in_rna blank (resolution only).
    """
    records = load_hgnc_records(hgnc_path) if hgnc_path.exists() else {}
    manual = {alnum(k): v for k, v in MANUAL_ADT_TO_GENE.items()}
    rna_set = {s.upper() for s in rna_symbols} if rna_symbols is not None else None

    rows = []
    for name in protein_names:
        gene, hgnc_id, mapping_type = resolve_marker(name, records, manual)
        present = None if (rna_set is None or gene is None) else (gene.upper() in rna_set)
        if mapping_type == "isotype":
            excluded = "isotype control (no target gene)"
        elif mapping_type == "unmapped":
            excluded = "no single HGNC gene (complex / not found)"
        elif present is False:
            excluded = "gene absent from RNA matrix"
        else:
            excluded = ""
        rows.append({
            "adt_name":          name,
            "hgnc_id":           hgnc_id,
            "gene_symbol":       gene or "",
            "uniprot_id":        "",   # not present in this HGNC export
            "mapping_type":      mapping_type,
            "present_in_rna":    "" if present is None else bool(present),
            "use_for_alignment": True,
            "use_for_kinetics":  gene is not None and present is not False,
            "excluded_because":  excluded,
        })
    return rows
