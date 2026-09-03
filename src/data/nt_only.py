"""Velocity preprocessing whose every FITTED statistic comes from the control cells alone.

The CRISPR protocol says no knockout cell may enter the preprocessing used for training.
That sentence has to be made precise before it can be implemented, because one reading of
it destroys the experiment:

  FITTED ON CONTROLS   HVG selection, the shared-count gene filter, the PCA basis and the
                       mean it centres on, scVelo's per-gene kinetic parameters, and
                       velocity confidence. A knockout cell influences none of these, so
                       nothing about the perturbations shapes the representation KOT
                       learns in.

  PER CELL, NOT FITTED Size-factor normalisation and log1p. These are functions of one
                       cell's own counts and estimate nothing across cells, so applying
                       them to a knockout cell fits nothing to it.

  NEIGHBOURHOODS       Same PCA basis for every cell, but the GRAPH is not shared. NT
                       cells take neighbours only from NT: a shared kNN graph would let
                       a control's moments be smoothed toward knockout cells, which is
                       leakage into the representation KOT trains on. KO cells keep
                       neighbours from the full graph, including other knockouts --
                       restricting those to NT would turn every KO moment into a smoothed
                       control profile and erase the perturbation the benchmark measures.

`fitted_statistics_report` writes exactly which cells each fitted quantity saw, so the
claim is auditable from the output rather than from this docstring.
"""

from __future__ import annotations

import numpy as np
import scanpy as sc
import scvelo as scv
from scipy import sparse

from src.data.velocity import select_hvgs


# scVelo writes its per-gene dynamical fit into var columns with this prefix, plus the
# uns entry `recover_dynamics`. Both must move from the control-only fit to the full
# object for `scv.tl.velocity` to score every cell against control-fitted kinetics.
FIT_PREFIX = "fit_"
DYNAMICS_UNS_KEY = "recover_dynamics"


def lift_square_sparse(sub, parent_idx: np.ndarray, n: int):
    """Place a k×k graph on `parent_idx` into an n×n matrix, zeros elsewhere."""
    sub = sparse.coo_matrix(sub)
    return sparse.csr_matrix(
        (sub.data, (parent_idx[sub.row], parent_idx[sub.col])),
        shape=(n, n),
    )


def keep_rows(matrix, keep: np.ndarray):
    """Zero every row whose `keep` entry is False."""
    weights = np.asarray(keep, dtype=np.float64)
    return sparse.csr_matrix(sparse.diags(weights) @ sparse.csr_matrix(matrix))


def stitch_neighbor_graphs(control_sub, query_full, fit_mask: np.ndarray):
    """NT rows from the NT-only graph, KO rows from the full-cell graph.

    `control_sub` is k×k on the control cells in the order of `fit_mask`. `query_full`
    is n×n on every cell. After this, no NT row has an edge to a knockout cell.
    """
    n = int(fit_mask.size)
    lifted = lift_square_sparse(control_sub, np.flatnonzero(fit_mask), n)
    query_full = sparse.csr_matrix(query_full, shape=(n, n))
    return keep_rows(query_full, ~fit_mask) + keep_rows(lifted, fit_mask)


def fit_neighbors_split_by_control(adata, fit_mask: np.ndarray, n_neighbors: int,
                                   n_pcs: int) -> None:
    """kNN in the control-fitted PCA: NT→NT only, KO from the full graph."""
    controls = adata[fit_mask].copy()
    sc.pp.neighbors(controls, n_neighbors=n_neighbors, n_pcs=n_pcs, use_rep="X_pca")
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, use_rep="X_pca")
    for key in ("connectivities", "distances"):
        adata.obsp[key] = stitch_neighbor_graphs(
            controls.obsp[key], adata.obsp[key], fit_mask)
    print(f"[nt-only] NT neighbours from {int(fit_mask.sum())} controls; "
          f"KO neighbours from all {adata.n_obs} cells")


def fit_hvg_on_controls(adata, fit_mask: np.ndarray, n_top_genes: int, hvg_flavor: str,
                        retain_genes=None):
    """Choose genes from the control cells, then subset every cell to that choice."""
    controls = adata[fit_mask].copy()
    controls = select_hvgs(controls, n_top_genes, hvg_flavor, retain_genes=retain_genes)
    print(f"[nt-only] HVG fitted on {int(fit_mask.sum())} control cells -> "
          f"{controls.n_vars} genes")
    return adata[:, controls.var_names].copy()


def shared_count_mask(adata, fit_mask: np.ndarray, min_shared_counts: int,
                      retain_genes=None) -> np.ndarray:
    """Genes with enough spliced+unspliced support AMONG CONTROLS.

    scVelo's filter_and_normalize would compute this over every cell, which lets the
    knockouts decide which genes survive.
    """
    controls = adata[fit_mask]
    spliced = controls.layers["spliced"]
    unspliced = controls.layers["unspliced"]
    shared = np.asarray((spliced > 0).sum(axis=0)).ravel() + \
        np.asarray((unspliced > 0).sum(axis=0)).ravel()
    keep = shared >= min_shared_counts
    if retain_genes:
        keep |= np.asarray(adata.var_names.isin(retain_genes))
    print(f"[nt-only] shared-count filter on controls keeps {int(keep.sum())}/{len(keep)} genes")
    return keep


def fit_pca_on_controls(adata, fit_mask: np.ndarray, n_comps: int) -> int:
    """Fit the PCA basis on controls; project every cell through it.

    scanpy centres on the mean of whatever it is given, so the control mean has to be
    subtracted explicitly when the loadings are applied to the knockouts. Letting
    `sc.tl.pca` run on the full matrix would fit both the basis and that mean to the
    knockouts.
    """
    resolved = int(min(n_comps, int(fit_mask.sum()) - 1, adata.n_vars - 1))
    if resolved < 1:
        raise ValueError(f"too few control cells or genes for PCA (n_comps={resolved})")
    controls = adata[fit_mask].copy()
    sc.tl.pca(controls, n_comps=resolved)

    loadings = np.asarray(controls.varm["PCs"], dtype=np.float64)
    dense = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)
    control_mean = np.asarray(
        controls.X.mean(axis=0) if not sparse.issparse(controls.X)
        else controls.X.mean(axis=0)
    ).ravel()
    adata.obsm["X_pca"] = ((dense - control_mean) @ loadings).astype(np.float32)
    adata.varm["PCs"] = loadings.astype(np.float32)
    adata.uns["pca"] = {"variance_ratio": controls.uns["pca"]["variance_ratio"],
                        "variance": controls.uns["pca"]["variance"]}
    adata.uns["nt_only_pca_mean"] = control_mean.astype(np.float32)
    print(f"[nt-only] PCA basis fitted on controls: {resolved} components, "
          f"projected {adata.n_obs} cells")
    return resolved


def fit_dynamics_on_controls(adata, fit_mask: np.ndarray, dynamics_n_jobs: int,
                             fit_all_genes: bool) -> None:
    """Recover per-gene kinetics from controls, then score every cell with them.

    `scv.tl.velocity` reads the `fit_*` var columns, so copying them across is what makes
    a knockout cell's velocity a control-fitted quantity evaluated at that cell rather
    than a fit that saw it.
    """
    controls = adata[fit_mask].copy()
    if fit_all_genes:
        scv.tl.recover_dynamics(controls, var_names="all", n_jobs=dynamics_n_jobs)
    else:
        scv.tl.recover_dynamics(controls, n_jobs=dynamics_n_jobs)

    moved = [column for column in controls.var.columns if column.startswith(FIT_PREFIX)]
    for column in moved:
        adata.var[column] = controls.var[column].reindex(adata.var_names).to_numpy()
    if DYNAMICS_UNS_KEY in controls.uns:
        adata.uns[DYNAMICS_UNS_KEY] = controls.uns[DYNAMICS_UNS_KEY]
    print(f"[nt-only] kinetics fitted on {int(fit_mask.sum())} control cells; "
          f"moved {len(moved)} fit_* columns to all {adata.n_obs} cells")


def fit_velocity_confidence_on_controls(adata, fit_mask: np.ndarray, n_neighbors: int) -> None:
    """Confidence is a summary statistic, so it is computed over controls and broadcast.

    KOT weights the ODE residual by it. Computing it across all cells would let the
    knockouts set the weight their own residual is judged under.
    """
    controls = adata[fit_mask].copy()
    # The full graph sliced to controls leaves cells whose neighbours were knockouts with
    # almost no edges, which scVelo rejects as corrupted. A control-only confidence needs
    # a control-only graph anyway, so it is rebuilt rather than inherited.
    sc.pp.neighbors(controls, n_neighbors=n_neighbors, use_rep="X_pca")
    for key in ("connectivities", "distances"):
        controls.obsp[key] = sparse.csr_matrix(controls.obsp[key])
    scv.tl.velocity_graph(controls)
    scv.tl.velocity_confidence(controls)
    for column in ("velocity_confidence", "velocity_confidence_transition", "velocity_length"):
        if column not in controls.obs:
            continue
        values = np.full(adata.n_obs, float(np.nanmedian(controls.obs[column])))
        values[fit_mask] = controls.obs[column].to_numpy()
        adata.obs[column] = values
    print("[nt-only] velocity confidence fitted on controls; knockouts carry the "
          "control median, never a value of their own")


def fitted_statistics_report(adata, fit_mask: np.ndarray) -> dict:
    """What each fitted quantity actually saw, written beside the data it produced."""
    return {
        "n_cells": int(adata.n_obs),
        "n_fit_cells": int(fit_mask.sum()),
        "n_genes": int(adata.n_vars),
        "fitted_on_controls": ["hvg_selection", "shared_count_filter", "pca_basis",
                               "pca_centring_mean", "gene_kinetics", "velocity_confidence"],
        "per_cell_not_fitted": ["size_factor_normalization", "log1p"],
        "nt_neighbours_drawn_from": "control cells, in the control-fitted PCA basis",
        "ko_neighbours_drawn_from": "all cells, in the control-fitted PCA basis",
    }


def preprocess_nt_only(adata, fit_mask: np.ndarray, n_top_genes: int, hvg_flavor: str,
                       min_shared_counts: int, n_pcs: int, n_neighbors: int,
                       dynamics_n_jobs: int, retain_genes=None):
    """The full control-only pipeline, in the order the current one runs it."""
    adata = fit_hvg_on_controls(adata, fit_mask, n_top_genes, hvg_flavor, retain_genes)
    keep = shared_count_mask(adata, fit_mask, min_shared_counts, retain_genes)
    adata = adata[:, keep].copy()

    # Per cell, so no statistic crosses cells; scVelo's own filter step is skipped
    # because its gene filter was already applied above, on controls.
    scv.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)

    resolved_pcs = fit_pca_on_controls(adata, fit_mask, n_pcs)
    fit_neighbors_split_by_control(adata, fit_mask, n_neighbors, resolved_pcs)
    scv.pp.moments(adata, n_pcs=None, n_neighbors=None)

    fit_dynamics_on_controls(adata, fit_mask, dynamics_n_jobs, bool(retain_genes))
    scv.tl.velocity(adata, mode="dynamical", use_highly_variable=False)
    fit_velocity_confidence_on_controls(adata, fit_mask, n_neighbors)
    adata.uns["velocity_backend"] = "scvelo_dynamical_nt_only"
    adata.uns["nt_only_preprocessing"] = fitted_statistics_report(adata, fit_mask)
    return adata
