"""
β degradation-rate anchor from measured protein half-lives.

Reads a CSV with (at least) protein_name and either half_life_hours or
beta_per_hour (optionally cell_type, quality_score_or_R2, anchor_weight), and
resolves it against the protein panel.

Aggregation across cell-type rows:
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
            quality_raw = (r.get("quality_score_or_R2") or "").strip()
            rows.append({
                "protein_name": protein,
                "beta_per_hour": beta_val,
                "weight": float(r.get("anchor_weight") or 1.0),
                "quality": float(quality_raw) if quality_raw else float("nan"),
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


def geometric_mean(values: list[float]) -> float:
    """Geometric mean of positive values (0 if none) — β is log-normal, so the
    physical→model centring should use the geometric, not arithmetic, mean."""
    logs = [math.log(v) for v in values if v > 0]
    return math.exp(sum(logs) / len(logs)) if logs else 0.0


def resolve_beta_anchors(
    path,
    protein_var_names,
    target_mean_beta: float = 0.5,
    aggregate: str = "median",
    stable_cv_max: float | None = None,
    prior_sigma_floor: float = 0.25,
    fixed_scale: float | None = None,
    min_quality: float | None = None,
):
    """
    Map a β-anchor CSV to (indices, betas, weights, sigmas, scale).

    The physical→model scale is c = target_mean_beta / geometric_mean(centers),
    computed from the resolved set (geometric because β is log-normal, and robust
    to a single anchor unlike an arithmetic/global-log centring). For anchor-count
    ablations, compute c ONCE on the full trusted set and pass it as `fixed_scale`
    — otherwise every change of the anchor subset silently redefines the β unit and
    confounds the ablation. The chosen scale is returned so it can be saved/frozen.

    sigmas are relative: σ_i = rel_sigma_i × β*_i (floored at prior_sigma_floor), so
    the log-normal width log_σ = σ/β* is constant across proteins.
    """
    if aggregate not in ("median", "mean"):
        raise ValueError(f"aggregate must be 'median' or 'mean', got {aggregate!r}")

    rows = load_beta_anchor_csv(Path(path))
    index = {normalize_protein_name(p): j for j, p in enumerate(protein_var_names)}
    per_protein: dict[int, list[tuple[float, float, float]]] = {}
    for r in rows:
        j = index.get(normalize_protein_name(r["protein_name"]))
        if j is None:
            continue
        per_protein.setdefault(j, []).append((r["beta_per_hour"], r["weight"], r["quality"]))
    if not per_protein:
        return [], [], [], [], 1.0

    kept: list[tuple[int, float, float, float]] = []
    for j in sorted(per_protein):
        entries = per_protein[j]
        # Quality gate: drop measurements whose R²/quality is below min_quality (missing
        # quality is kept — unknown is not the same as low). Downweighting alone leaves
        # low-confidence half-lives anchoring β; this gate removes them outright.
        if min_quality is not None:
            entries = [e for e in entries if math.isnan(e[2]) or e[2] >= min_quality]
        if not entries:
            continue
        betas_h = [b for b, _, _ in entries]
        weights = [w for _, w, _ in entries]
        center = aggregate_values(betas_h, aggregate)
        if center <= 0:
            continue
        cv = sample_std(betas_h) / center
        # Stability filter: drop proteins whose half-life varies too much across cell types.
        if stable_cv_max is not None and len(betas_h) >= 2 and cv > stable_cv_max:
            continue
        weight = aggregate_values(weights, "mean")
        rel_sigma = max(cv, prior_sigma_floor) if len(betas_h) >= 2 else prior_sigma_floor
        kept.append((j, center, weight, rel_sigma))

    if not kept:
        return [], [], [], [], 1.0

    idx = [j for j, _, _, _ in kept]
    centers = [b for _, b, _, _ in kept]
    weights = [w for _, _, w, _ in kept]
    rel_sigmas = [s for _, _, _, s in kept]

    if fixed_scale is not None:
        scale = float(fixed_scale)          # frozen unit — use for anchor-count ablations
    else:
        # Geometric mean of the TRUSTED set — after the quality + stability filters, before
        # any anchor-count subset — so a dropped low-quality/unstable anchor cannot skew the
        # physical→model β unit.
        g = geometric_mean(centers)
        scale = (float(target_mean_beta) / g) if g > 0 else 1.0
    betas = [c * scale for c in centers]
    sigmas = [rel * beta for rel, beta in zip(rel_sigmas, betas)]
    return idx, betas, weights, sigmas, scale
