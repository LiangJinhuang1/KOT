"""ADT panel names to HGNC gene symbols.

CD16 ≠ FCGR3A. HGNC approved/previous/alias plus a curated map for complexes
and isoform epitopes that have no single-gene HGNC alias.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Callable, Iterable

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

# Columns commonly used by 10x / STARsolo / AnnData objects to keep the gene
# symbol even when var_names are Ensembl IDs or have been made unique.
GENE_ALIAS_COLUMNS = (
    "gene_name",
    "gene_symbol",
    "gene_symbols",
    "symbol",
    "feature_name",
    "gene_id",
)


def valid_gene_alias(value) -> str | None:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "na", "<na>"}:
        return None
    return text


def normalize_gene_symbol(value: str) -> str:
    return str(value).strip().upper()


def iter_gene_aliases(var_names: Iterable, var=None):
    for i, name in enumerate(var_names):
        alias = valid_gene_alias(name)
        if alias is not None:
            yield i, alias
    if var is None:
        return
    for col in GENE_ALIAS_COLUMNS:
        if col not in var.columns:
            continue
        for i, value in enumerate(var[col].astype(str)):
            alias = valid_gene_alias(value)
            if alias is not None:
                yield i, alias


def gene_alias_lookup(
    var_names: Iterable,
    var=None,
    normalizer: Callable[[str], str] | None = None,
) -> dict[str, int]:
    lookup: dict[str, int] = {}
    for i, alias in iter_gene_aliases(var_names, var):
        key = normalizer(alias) if normalizer is not None else alias
        lookup.setdefault(key, i)
    return lookup


def gene_alias_set(
    var_names: Iterable,
    var=None,
    normalizer: Callable[[str], str] | None = None,
) -> set[str]:
    return {
        normalizer(alias) if normalizer is not None else alias
        for _, alias in iter_gene_aliases(var_names, var)
    }


def aliases_for_positions(
    var_names: Iterable,
    var,
    positions: Iterable[int],
    normalizer: Callable[[str], str] | None = None,
) -> set[str]:
    keep = set(int(i) for i in positions)
    return {
        normalizer(alias) if normalizer is not None else alias
        for i, alias in iter_gene_aliases(var_names, var) if i in keep
    }


def alnum(name: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", name.upper())


def strip_adt_suffix(name: str) -> str:
    return re.sub(r"_TotalSeq[A-Z]$", "", name)


def load_hgnc_records(hgnc_path: Path = HGNC_PATH) -> dict[str, tuple[str, str]]:
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
    """{} if HGNC is missing so callers fall back to name matching, not a crash."""
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


# Canonical column order of an ADT→gene mapping CSV (from build_mapping_rows).
MAPPING_COLUMNS = [
    "adt_name", "hgnc_id", "gene_symbol", "uniprot_id", "mapping_type",
    "present_in_rna", "use_for_alignment", "use_for_kinetics", "excluded_because",
]


def parse_bool(value) -> bool | None:
    """'' is unknown (present_in_rna), not False."""
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    if s == "":
        return None
    raise ValueError(f"cannot parse boolean from {value!r}")


def validate_mapping_records(records: list[dict]) -> None:
    """Fail here; a malformed CSV would silently distort alignment and kinetics."""
    names = [r["adt_name"] for r in records]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"duplicate adt_name(s) in mapping: {dupes}")
    for r in records:
        name = r["adt_name"]
        if not name:
            raise ValueError("empty adt_name in mapping row")
        if r["use_for_kinetics"]:
            if not r["gene_symbol"]:
                raise ValueError(f"{name}: use_for_kinetics=True but gene_symbol is empty")
        else:
            if not r["excluded_because"]:
                raise ValueError(f"{name}: excluded from kinetics but no excluded_because reason")
        if r["mapping_type"] == "isotype":
            if r["use_for_kinetics"]:
                raise ValueError(f"{name}: isotype control must not have use_for_kinetics=True")


def load_mapping_records(path: Path) -> list[dict]:
    records: list[dict] = []
    with open(path) as f:
        reader = csv.DictReader(f)
        missing = [c for c in MAPPING_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} missing mapping columns: {missing}")
        for row in reader:
            records.append({
                "adt_name":          (row.get("adt_name") or "").strip(),
                "hgnc_id":           (row.get("hgnc_id") or "").strip(),
                "gene_symbol":       (row.get("gene_symbol") or "").strip(),
                "uniprot_id":        (row.get("uniprot_id") or "").strip(),
                "mapping_type":      (row.get("mapping_type") or "").strip(),
                "present_in_rna":    parse_bool(row.get("present_in_rna")),
                "use_for_alignment": bool(parse_bool(row.get("use_for_alignment"))),
                "use_for_kinetics":  bool(parse_bool(row.get("use_for_kinetics"))),
                "excluded_because":  (row.get("excluded_because") or "").strip(),
            })
    validate_mapping_records(records)
    return records


def build_mapping_rows(protein_names, rna_symbols=None, hgnc_path: Path = HGNC_PATH,
                       kinetic_complex_allow=None) -> list[dict]:
    """Kinetics eligibility is intent, not matrix presence: a dropped gene can still
    be recovered. complex_curated is not auto-eligible — that would erase the
    1:1 vs representative-chain distinction. Isotypes never enter Sinkhorn.
    """
    records = load_hgnc_records(hgnc_path) if hgnc_path.exists() else {}
    manual = {alnum(k): v for k, v in MANUAL_ADT_TO_GENE.items()}
    rna_set = {normalize_gene_symbol(s) for s in rna_symbols} if rna_symbols is not None else None
    allow = {alnum(strip_adt_suffix(str(x))) for x in (kinetic_complex_allow or [])}

    rows = []
    for name in protein_names:
        gene, hgnc_id, mapping_type = resolve_marker(name, records, manual)
        present = None if (rna_set is None or gene is None) else (normalize_gene_symbol(gene) in rna_set)
        allowed_complex = (
            mapping_type == "complex_curated"
            and (alnum(strip_adt_suffix(name)) in allow or (gene and alnum(gene) in allow))
        )
        use_for_kinetics = (mapping_type == "one_to_one") or allowed_complex
        if mapping_type == "isotype":
            excluded = "isotype control (no target gene)"
        elif mapping_type == "unmapped":
            excluded = "no single HGNC gene (complex / not found)"
        elif mapping_type == "complex_curated" and not allowed_complex:
            excluded = "complex / representative-chain approximation — not kinetics-eligible unless allow-listed"
        else:
            excluded = ""   # eligible; absence in a matrix is not an exclusion
        rows.append({
            "adt_name":          name,
            "hgnc_id":           hgnc_id,
            "gene_symbol":       gene or "",
            "uniprot_id":        "",   # not present in this HGNC export
            "mapping_type":      mapping_type,
            "present_in_rna":    "" if present is None else bool(present),
            "use_for_alignment": mapping_type != "isotype",   # isotype controls do not inform alignment
            "use_for_kinetics":  use_for_kinetics,   # intent; complex needs allow-listing
            "excluded_because":  excluded,
        })
    validate_mapping_records(rows)   # fail loud if the built table breaks an invariant
    return rows
