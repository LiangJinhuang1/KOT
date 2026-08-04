"""
β degradation-rate anchor from measured protein half-lives.

Reads a CSV with (at least) protein_name and either half_life_hours or
beta_per_hour (optionally cell_type, quality_score_or_R2, anchor_weight), and
resolves it against the protein panel.

Aggregation across cell-type rows (supervisor):
  - median (default) or mean of β_per_hour
  - optional stability filter: keep only proteins whose CV across cell types
    is ≤ stable_cv_max
  - per-protein σ from cross-cell-type dispersion (for soft Gaussian /
    log-normal priors); single-measurement proteins get prior_sigma_floor

Modelling note: KOT's ODE runs in pseudotime, so absolute per-hour rates are
rescaled so mean(β) = target_mean_beta. Relative fast/slow structure is kept;
learned κ absorbs the time-unit conversion.
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
                "cell_type": (r.get("cell_type") or "").strip(),
            })
    return rows


def aggregate_values(values: list[float], how: str) -> float:
    if how == "mean":
        return sum(values) / len(values)
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = sum(values) / len(values)
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(var)


def resolve_beta_anchors(
    path,
    protein_var_names,
    target_mean_beta: float = 0.5,
    aggregate: str = "median",
    stable_cv_max: float | None = None,
    prior_sigma_floor: float = 0.25,
):
    """
    Map a β-anchor CSV to (indices, betas, weights, sigmas).

    sigmas are relative: σ_i = rel_sigma_i × β*_i with rel_sigma floored at
    prior_sigma_floor, so the log-normal width log_σ = σ/β* is constant across
    proteins and low-β* anchors are not artificially weakened.
    """
    if aggregate not in ("median", "mean"):
        raise ValueError(f"aggregate must be 'median' or 'mean', got {aggregate!r}")

    rows = load_beta_anchor_csv(Path(path))
    index = {normalize_protein_name(p): j for j, p in enumerate(protein_var_names)}
    per_protein: dict[int, list[tuple[float, float]]] = {}
    for r in rows:
        j = index.get(normalize_protein_name(r["protein_name"]))
        if j is None:
            continue
        per_protein.setdefault(j, []).append((r["beta_per_hour"], r["weight"]))
    if not per_protein:
        return [], [], [], []

    kept: list[tuple[int, float, float, float]] = []
    for j in sorted(per_protein):
        betas_h = [b for b, _ in per_protein[j]]
        weights = [w for _, w in per_protein[j]]
        center = aggregate_values(betas_h, aggregate)
        if center <= 0:
            continue
        cv = sample_std(betas_h) / center if center > 0 else 0.0
        if stable_cv_max is not None and len(betas_h) >= 2 and cv > stable_cv_max:
            continue
        weight = aggregate_values(weights, "mean")
        # Relative dispersion → absolute σ after later rescaling (apply scale below).
        rel_sigma = max(cv, prior_sigma_floor) if len(betas_h) >= 2 else prior_sigma_floor
        kept.append((j, center, weight, rel_sigma))

    if not kept:
        return [], [], [], []

    idx = [j for j, _, _, _ in kept]
    betas = [b for _, b, _, _ in kept]
    weights = [w for _, _, w, _ in kept]
    rel_sigmas = [s for _, _, _, s in kept]

    mean_beta = sum(betas) / len(betas)
    scale = (float(target_mean_beta) / mean_beta) if mean_beta > 0 else 1.0
    betas = [b * scale for b in betas]
    # σ is purely relative: σ_i = rel_sigma_i × β*_i, where rel_sigma is already
    # floored at prior_sigma_floor. This keeps the log-normal width (log_σ = σ/β*)
    # constant across proteins, so a low-β* anchor gets the same pull as a high-β*
    # one. An absolute floor (prior_sigma_floor × target_mean) would inflate the
    # relative width of small-β* anchors and make them toothless.
    sigmas = [rel * beta for rel, beta in zip(rel_sigmas, betas)]
    return idx, betas, weights, sigmas
