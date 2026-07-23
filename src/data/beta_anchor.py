"""
β degradation-rate anchor from measured protein half-lives.

Reads a CSV with (at least) protein_name and either half_life_hours or
beta_per_hour, and resolves it against the protein panel to (anchor indices,
target β, weights). Half-life gives β = ln 2 / t_half (per hour).

Modelling note: KOT's ODE runs in *pseudotime*, not hours, so absolute per-hour
rates are not directly comparable to the model's β. We therefore rescale the
targets so their mean equals `target_mean_beta` — the anchor then imposes the
*relative* degradation pattern (which proteins are fast vs slow) at KOT's own β
scale, and the learned κ absorbs the pseudotime↔time conversion.
"""
from __future__ import annotations

import csv
import math
import re
from pathlib import Path

LN2 = math.log(2.0)


def normalize_protein_name(name: str) -> str:
    """Match ADT names across sources: strip TotalSeq suffix, keep A–Z0–9."""
    name = re.sub(r"_TotalSeq[A-Z]$", "", name)
    return re.sub(r"[^A-Z0-9]", "", name.upper())


def load_beta_anchor_csv(path: Path) -> list[dict]:
    """Rows with a resolved β_per_hour (from the column or ln2/half_life)."""
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            protein = (r.get("protein_name") or "").strip()
            if not protein:
                continue
            beta = (r.get("beta_per_hour") or "").strip()
            half = (r.get("half_life_hours") or "").strip()
            if beta:
                beta_val = float(beta)
            elif half and float(half) > 0:
                beta_val = LN2 / float(half)
            else:
                continue
            rows.append({
                "protein_name": protein,
                "beta_per_hour": beta_val,
                "weight": float(r.get("anchor_weight") or 1.0),
            })
    return rows


def resolve_beta_anchors(path, protein_var_names, target_mean_beta: float = 0.5):
    """
    Map a β-anchor CSV to (indices, betas, weights) against the protein panel.

    Betas are rescaled so their mean equals target_mean_beta (relative anchoring).
    Proteins in the CSV that are not in the panel are skipped.
    """
    rows = load_beta_anchor_csv(Path(path))
    index = {normalize_protein_name(p): j for j, p in enumerate(protein_var_names)}
    # The anchor CSV may hold several rows per protein (one per cell type); collapse
    # them to a single β/weight per protein so each anchor is counted once.
    per_protein: dict[int, list[tuple[float, float]]] = {}
    for r in rows:
        j = index.get(normalize_protein_name(r["protein_name"]))
        if j is None:
            continue
        per_protein.setdefault(j, []).append((r["beta_per_hour"], r["weight"]))
    if not per_protein:
        return [], [], []
    idx = sorted(per_protein)
    betas = [sum(b for b, _ in per_protein[j]) / len(per_protein[j]) for j in idx]
    weights = [sum(w for _, w in per_protein[j]) / len(per_protein[j]) for j in idx]
    mean_beta = sum(betas) / len(betas)
    if mean_beta > 0:
        betas = [b / mean_beta * target_mean_beta for b in betas]
    return idx, betas, weights
