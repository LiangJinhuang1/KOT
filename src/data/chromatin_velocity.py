"""Chromatin velocity: the direction the JVP is read along, built from ATAC alone.

Two datasets, two constructions, one output contract — a displacement per cell in
GENE-ACTIVITY coordinates, plus a per-cell confidence and the mask of cells whose
velocity is trustworthy enough to carry the kinetic loss.

  hspc  §7. Real day 0 / day 7 measurements, so the velocity is a measured displacement:
        an entropic OT coupling from day-0 to day-7 ATAC gives each day-0 cell a
        barycentric future state, and v_c is that displacement over seven days. The OT
        COST is computed in LSI coordinates; the DISPLACEMENT is taken in gene-activity
        coordinates. Mixing the two would measure a distance in one space and a motion in
        another. RNA is not read at any point.

  bmmc  §8. No experimental time, so there is no measured velocity and this must not be
        called one. What is built is an ATAC-derived DIRECTIONAL TRAJECTORY FIELD: an
        ATAC-only pseudotime, then a forward-neighbour difference quotient. It inherits
        every weakness of a pseudotime, which is why it is named for what it is.

Confidence is not decoration. A barycentric projection onto a diffuse coupling, or a
difference quotient over neighbours that disagree, produces a direction that carries no
information, and §7 says those cells stay out of the dynamic loss.
"""

from __future__ import annotations

import numpy as np
import scanpy as sc
import torch
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

from src.losses.entropic_ot import row_entropy, sinkhorn_plan

# Day 0 → day 7. The denominator of the HSPC velocity, in days.
HSPC_INTERVAL_DAYS = 7.0

# Lineage grouping for the BMMC per-branch kinetic evaluation. Labels, not velocity
# inputs: the field itself never reads a cell type.
BMMC_LINEAGES = {
    "erythroid": ["Proerythroblast", "Erythroblast", "Normoblast", "MK/E prog"],
    "myeloid": ["CD14+ Mono", "CD16+ Mono", "G/M prog", "cDC2", "pDC", "ID2-hi myeloid prog"],
    "lymphoid": ["Lymph prog", "CD4+ T naive", "CD4+ T activated", "CD8+ T", "CD8+ T naive",
                 "NK", "ILC", "Naive CD20+ B", "Transitional B", "B1 B", "Plasma cell"],
}


def barycentric_displacement(source_activity: np.ndarray, target_activity: np.ndarray,
                             source_lsi: np.ndarray, target_lsi: np.ndarray,
                             interval: float, epsilon: float, n_iterations: int,
                             device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OT in LSI, displacement in gene-activity. Returns velocity, entropy, confidence."""
    plan = sinkhorn_plan(source_lsi, target_lsi, epsilon, n_iterations, device)
    mass = plan.sum(dim=1, keepdim=True).clamp(min=1e-30)
    future = ((plan @ torch.as_tensor(target_activity, device=device)) / mass).cpu().numpy()
    entropy = row_entropy(plan)
    confidence = np.clip(1.0 - entropy / np.log(len(target_activity)), 0.0, 1.0)
    return (future - source_activity) / interval, entropy, confidence.astype(np.float32)


def hspc_velocity(adata: sc.AnnData, epsilon: float, n_iterations: int,
                  confidence_quantile: float, device: torch.device,
                  split: np.ndarray | None = None,
                  source_mask: np.ndarray | None = None) -> dict:
    """§7. Barycentric day-0 → day-7 displacement in gene-activity coordinates.

    Each split fold gets its own OT plan. Training day-0 cells therefore couple only to
    training day-7 cells — a global plan would let held-out day-7 cells write the
    velocities that enter the kinetic loss. Day-7 cells still take part in static
    alignment, which needs no direction.
    """
    day = adata.obs["day"].to_numpy().astype(int)
    activity = np.asarray(adata.obsm["gene_activity"], dtype=np.float32)
    lsi = np.asarray(adata.obsm["lsi"], dtype=np.float32)
    available = np.ones(adata.n_obs, dtype=bool) if source_mask is None \
        else np.asarray(source_mask, dtype=bool)
    folds = np.asarray(["all"] * adata.n_obs if split is None else split)

    velocity = np.zeros_like(activity)
    entropy = np.zeros(adata.n_obs, dtype=np.float32)
    confidence = np.zeros(adata.n_obs, dtype=np.float32)
    dynamic = np.zeros(adata.n_obs, dtype=bool)
    for fold in np.unique(folds):
        source_rows = np.flatnonzero((day == 0) & (folds == fold) & available)
        target_rows = np.flatnonzero((day == 7) & (folds == fold) & available)
        if len(source_rows) < 2 or len(target_rows) < 2:
            print(f"[velocity] hspc fold {fold}: skip "
                  f"({len(source_rows)} day-0, {len(target_rows)} day-7)")
            continue
        moved, ent, conf = barycentric_displacement(
            activity[source_rows], activity[target_rows],
            lsi[source_rows], lsi[target_rows],
            HSPC_INTERVAL_DAYS, epsilon, n_iterations, device)
        velocity[source_rows] = moved
        entropy[source_rows] = ent
        confidence[source_rows] = conf
        cutoff = np.quantile(conf, confidence_quantile)
        dynamic[source_rows] = conf >= cutoff
        print(f"[velocity] hspc fold {fold}: {len(source_rows)} day-0 -> {len(target_rows)} day-7; "
              f"confidence cutoff {cutoff:.4f} keeps {int(dynamic[source_rows].sum())}")
    return {
        "velocity": velocity,
        "confidence": confidence,
        "coupling_entropy": entropy,
        "dynamic_mask": dynamic,
        "norm": np.linalg.norm(velocity, axis=1).astype(np.float32),
    }


def atac_pseudotime(adata: sc.AnnData, n_neighbors: int, root_row: int) -> np.ndarray:
    """Diffusion pseudotime on the ATAC LSI graph, rooted at one ATAC-chosen cell.

    Built on `obsm["lsi"]` and nothing else, so no RNA measurement orients the trajectory.
    """
    graph = sc.AnnData(X=np.zeros((adata.n_obs, 1), dtype=np.float32), obs=adata.obs.copy())
    graph.obsm["X_lsi"] = np.asarray(adata.obsm["lsi"], dtype=np.float32)
    sc.pp.neighbors(graph, n_neighbors=n_neighbors, use_rep="X_lsi")
    sc.tl.diffmap(graph)
    graph.uns["iroot"] = int(root_row)
    sc.tl.dpt(graph)
    return graph.obs["dpt_pseudotime"].to_numpy().astype(np.float32)


def progenitor_root(adata: sc.AnnData) -> int:
    """The cell the ATAC pseudotime is rooted at.

    `ATAC_pseudotime_order` ships with the processed object and is computed from ATAC, so
    using its minimum keeps the root ATAC-derived. A cell-type label would be an RNA
    annotation and §8 forbids orienting the trajectory with RNA.
    """
    order = adata.obs["ATAC_pseudotime_order"].to_numpy().astype(float)
    return int(np.nanargmin(order))


def finite_lsi_mask(lsi: np.ndarray) -> np.ndarray:
    """Cells whose ATAC LSI is usable as a neighbour query.

    NeurIPS `ATAC_lsi_red` is NaN for cells that had too few peaks; pynndescent then
    aborts the whole graph. Those cells get no chromatin velocity, same as HSPC day-7.
    """
    return np.isfinite(lsi).all(axis=1)


def interpolate_pseudotime(train_lsi: np.ndarray, train_tau: np.ndarray,
                           query_lsi: np.ndarray, n_neighbors: int) -> np.ndarray:
    """Held-out pseudotime as a weighted average of train LSI neighbours."""
    n_neighbors = min(n_neighbors, len(train_lsi))
    neighbors = NearestNeighbors(n_neighbors=n_neighbors).fit(train_lsi)
    dists, index = neighbors.kneighbors(query_lsi)
    weights = 1.0 / np.clip(dists, 1e-8, None)
    weights /= weights.sum(axis=1, keepdims=True)
    return (weights * train_tau[index]).sum(axis=1).astype(np.float32)


def train_neighbour_graph(lsi: np.ndarray, train_mask: np.ndarray,
                          n_neighbors: int) -> sparse.csr_matrix:
    """kNN graph whose edges never point at a held-out cell.

    Train–train neighbours come from a train-only Scanpy graph. Held-out cells query
    that same train cloud, so the field used in the kinetic loss cannot see a test cell.
    """
    n_cells = len(lsi)
    train_idx = np.flatnonzero(train_mask)
    held_idx = np.flatnonzero(~train_mask)
    held_idx = held_idx[np.isfinite(lsi[held_idx]).all(axis=1)]
    n_neighbors = min(n_neighbors, len(train_idx))

    graph = sc.AnnData(X=np.zeros((len(train_idx), 1), dtype=np.float32))
    graph.obsm["X_lsi"] = lsi[train_idx]
    sc.pp.neighbors(graph, n_neighbors=n_neighbors, use_rep="X_lsi")
    train_conn = sparse.csr_matrix(graph.obsp["connectivities"])
    rows, cols = train_conn.nonzero()
    full_rows = train_idx[rows]
    full_cols = train_idx[cols]
    data = train_conn.data.astype(np.float32)

    if len(held_idx) > 0:
        query = NearestNeighbors(n_neighbors=n_neighbors).fit(lsi[train_idx])
        dists, neigh = query.kneighbors(lsi[held_idx])
        weights = (1.0 / np.clip(dists, 1e-8, None)).astype(np.float32)
        full_rows = np.concatenate([full_rows, np.repeat(held_idx, n_neighbors)])
        full_cols = np.concatenate([full_cols, train_idx[neigh.ravel()]])
        data = np.concatenate([data, weights.ravel()])

    return sparse.csr_matrix((data, (full_rows, full_cols)), shape=(n_cells, n_cells))


def forward_difference_field(activity: np.ndarray, connectivity: sparse.csr_matrix,
                             pseudotime: np.ndarray, min_forward: int
                             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Difference quotient over forward neighbours of one (possibly train-only) graph."""
    velocity = np.zeros_like(activity)
    confidence = np.zeros(activity.shape[0], dtype=np.float32)
    n_forward = np.zeros(activity.shape[0], dtype=np.int32)
    graph = sparse.csr_matrix(connectivity)
    for row in range(activity.shape[0]):
        span = slice(graph.indptr[row], graph.indptr[row + 1])
        candidates = graph.indices[span]
        weights = graph.data[span]
        delta = pseudotime[candidates] - pseudotime[row]
        # Equal float32 pseudotimes do not define a forward derivative; excluding their
        # near-zero gaps prevents infinite difference quotients.
        ahead = delta > 1e-6
        n_forward[row] = int(ahead.sum())
        if n_forward[row] < min_forward:
            continue
        forward = candidates[ahead]
        weights = weights[ahead]
        steps = (activity[forward] - activity[row]) / delta[ahead, None]
        velocity[row] = (weights[:, None] * steps).sum(axis=0) / weights.sum()
        confidence[row] = direction_coherence(steps, velocity[row], weights)
    return velocity, confidence, n_forward


def bmmc_velocity(adata: sc.AnnData, n_neighbors: int, confidence_quantile: float,
                  min_forward: int, train_mask: np.ndarray | None = None) -> dict:
    """§8. Forward-pseudotime difference quotient over ATAC neighbours.

    An ATAC-DERIVED DIRECTIONAL TRAJECTORY FIELD, not a measured chromatin velocity: the
    time it differentiates against is a pseudotime, so its magnitude has no unit and only
    its direction should be read.

    The field is fitted on the training cells only. Held-out cells receive an interpolated
    pseudotime and a velocity toward later TRAIN neighbours, so test geometry cannot write
    the directions that enter the kinetic loss.
    """
    if train_mask is None:
        train_mask = np.ones(adata.n_obs, dtype=bool)
    lsi = np.asarray(adata.obsm["lsi"], dtype=np.float32)
    activity = np.asarray(adata.obsm["gene_activity"], dtype=np.float32)
    finite = finite_lsi_mask(lsi)
    n_nan = int((~finite).sum())
    if n_nan:
        print(f"[velocity] bmmc: {n_nan}/{adata.n_obs} cells have non-finite LSI; "
              "they are kept in the object but excluded from the trajectory field")
    train_mask = train_mask & finite
    train_idx = np.flatnonzero(train_mask)
    held_idx = np.flatnonzero(finite & ~train_mask)
    assert len(train_idx) > n_neighbors, (
        f"only {len(train_idx)} train cells have finite LSI, need more than {n_neighbors}")

    train_handle = adata[train_idx].copy()
    train_handle.obsm["lsi"] = lsi[train_idx]
    pt_train = atac_pseudotime(train_handle, n_neighbors, progenitor_root(train_handle))
    pseudotime = np.zeros(adata.n_obs, dtype=np.float32)
    pseudotime[train_idx] = pt_train
    if len(held_idx) > 0:
        pseudotime[held_idx] = interpolate_pseudotime(
            lsi[train_idx], pt_train, lsi[held_idx], n_neighbors)

    connectivity = train_neighbour_graph(lsi, train_mask, n_neighbors)
    velocity, confidence, n_forward = forward_difference_field(
        activity, connectivity, pseudotime, min_forward)

    usable = n_forward >= min_forward
    fit_usable = usable & train_mask
    if not fit_usable.any():
        raise ValueError("no training-source cell has enough forward ATAC neighbours")
    direction_confidence = confidence.copy()
    speed_score, speed_cap = speed_reliability(velocity, fit_usable)
    confidence *= speed_score
    print(f"[velocity] bmmc: robust speed cap {speed_cap:.4g}; faster cells are downweighted")
    # Validation/test cells cannot decide either confidence threshold used by L_dyn.
    cutoff = np.quantile(confidence[fit_usable], confidence_quantile)
    dynamic = usable & (confidence >= cutoff)
    print(f"[velocity] bmmc: field on {len(train_idx)} train cells; "
          f"{int(usable.sum())}/{adata.n_obs} with >= {min_forward} forward neighbours; "
          f"combined-confidence cutoff {cutoff:.4f} keeps {int(dynamic.sum())}")
    return {
        "velocity": velocity,
        "confidence": confidence,
        "direction_confidence": direction_confidence,
        "speed_reliability": speed_score,
        "pseudotime": pseudotime,
        "n_forward": n_forward,
        "dynamic_mask": dynamic,
        "norm": np.linalg.norm(velocity, axis=1).astype(np.float32),
        "lineage": bmmc_lineage(adata),
    }


def direction_coherence(steps: np.ndarray, aggregate: np.ndarray,
                        weights: np.ndarray) -> float:
    """How much the contributing neighbours agree with the direction they averaged to.

    Neighbours pointing every way average to a short vector whose direction is noise; this
    is the quantity that separates that case from a genuine local flow, and averaging the
    difference quotients alone cannot.
    """
    scale = np.linalg.norm(aggregate)
    if scale == 0.0:
        return 0.0
    cosines = steps @ aggregate / (np.linalg.norm(steps, axis=1) * scale + 1e-12)
    return float(np.clip((weights * cosines).sum() / weights.sum(), 0.0, 1.0))


def speed_reliability(velocity: np.ndarray, fit_mask: np.ndarray,
                      upper_quantile: float = 0.95) -> tuple[np.ndarray, float]:
    """Downweight numerical derivative explosions without changing the saved field.

    BMMC pseudotime is continuous but locally contains almost tied values. Dividing by
    such a gap can make a handful of otherwise coherent directions thousands of times
    faster than the median; a squared residual then becomes almost entirely those cells.
    The trajectory is directional, so confidence includes a robust speed-reliability
    term fitted on training-source cells only. Normal speeds retain weight one and the
    upper tail decays quadratically.
    """
    norms = np.linalg.norm(velocity, axis=1)
    reference = norms[np.asarray(fit_mask, dtype=bool) & (norms > 0)]
    if len(reference) == 0:
        return np.zeros(len(velocity), dtype=np.float32), 0.0
    cap = max(float(np.quantile(reference, upper_quantile)), 1e-12)
    reliability = np.ones(len(velocity), dtype=np.float32)
    fast = norms > cap
    reliability[fast] = (cap / norms[fast]) ** 2
    reliability[norms == 0] = 0.0
    return reliability, cap


def bmmc_lineage(adata: sc.AnnData) -> np.ndarray:
    """Lineage label per cell for the per-branch kinetic evaluation only."""
    lookup = {name: lineage for lineage, names in BMMC_LINEAGES.items() for name in names}
    if "cell_type" not in adata.obs:
        return np.repeat("unassigned", adata.n_obs)
    return adata.obs["cell_type"].astype(str).map(lookup).fillna("other").to_numpy()


def direction_diversity(velocity: np.ndarray, mask: np.ndarray, seed: int = 0,
                        n_probe: int = 2000) -> dict:
    """How much of the field is one global arrow rather than a per-cell direction.

    A barycentric projection onto a diffuse OT plan hands every source cell the same
    future — the target mean — so every displacement points the same way. The JVP would
    then read ONE direction for the whole dataset and the kinetic term would carry no
    per-cell information, while the norms, the loss and the confidence all looked healthy.
    Nothing else in the pipeline would catch that, so it is measured here.
    """
    rows = np.flatnonzero(mask)
    if len(rows) < 2:
        return {
            "pairwise_cosine_median": float("nan"),
            "energy_in_mean_direction": float("nan"),
        }
    probe = np.random.default_rng(seed).choice(rows, min(n_probe, len(rows)), replace=False)
    sample = velocity[probe]
    unit = sample / np.linalg.norm(sample, axis=1, keepdims=True).clip(1e-12)
    cosines = unit @ unit.T
    pairwise = cosines[~np.eye(len(cosines), dtype=bool)]
    centred = sample - sample.mean(axis=0)
    return {
        "pairwise_cosine_median": float(np.median(pairwise)),
        "energy_in_mean_direction": float(1.0 - (centred ** 2).sum() / (sample ** 2).sum()),
    }


def gauge_normalize(velocity: np.ndarray) -> np.ndarray:
    """Divide by the median non-zero per-cell norm, exactly as the RNA→protein runs do.

    One global scalar, so the relative magnitudes between cells survive and only the unit
    changes. kappa is bounded, so lambda_dyn would otherwise mean something different on
    each dataset — and a pseudotime-derived field's raw scale is arbitrary anyway, being
    set by how finely the pseudotime happens to be spaced.
    """
    norms = np.linalg.norm(velocity, axis=1)
    nonzero = norms > 0
    gauge = float(np.median(norms[nonzero])) if nonzero.any() else 1.0
    print(f"[velocity] gauge-normalized: median non-zero per-cell norm {gauge:.4g} -> 1.0")
    return (velocity / gauge).astype(np.float32)
