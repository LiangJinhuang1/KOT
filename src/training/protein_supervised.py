"""Shared contract for the supervised RNA→protein baselines.

Papalexi is CITE-seq, so the control cells carry genuine paired (RNA, protein)
observations. A diagonal-integration method throws that pairing away; these baselines
use it. Each one fits on the cells the protocol allows (``context["fit_rows"]``, the NT
controls on Papalexi) and then predicts protein from RNA alone for EVERY cell, so the
held-out knockouts are scored without their ADT ever being read.

The return contract is the runner's ``holdout_second`` one: ``aligned[0]`` is the
prediction for all n cells in ADT coordinates, ``aligned[1]`` the observed ADT of the
fitted cells, which runner.py scatters back to full length with NaN elsewhere.
"""

from __future__ import annotations

import numpy as np

from src.data.transforms import rna_size_adt
from src.utils.arrays import matrix_from_adata


def fit_rows_of(context: dict, n_cells: int) -> np.ndarray:
    """Positional rows this run may fit on; every cell when no restriction is set."""
    fit_rows = context.get("fit_rows")
    return np.arange(n_cells) if fit_rows is None else np.asarray(fit_rows)


def rna_features(context: dict, cfg: dict) -> np.ndarray:
    """RNA for every cell in KOT's feature space, so a gap is the map and not the input."""
    return matrix_from_adata(context["rna_adata"], cfg.get("kot_rna_layer"), "RNA")


def adt_targets(context: dict, cfg: dict) -> np.ndarray:
    """Observed ADT for the fitted cells, in the units the prediction will be SCORED in.

    `protein_target_normalization: rna_size` trains against the CRISPR benchmark's own
    normalisation. The default `layer` reads `kot_protein_layer` (CLR in protein.X), which
    is what every non-CRISPR dataset still uses.

    Training on CLR and scoring rna_size deltas cost the ridge baseline more than half its
    Spearman (+0.51 -> +0.20), and no post-hoc calibration recovers it: CLR divides each
    cell by its own 4-protein total, so it is a per-cell row operation that a per-protein
    affine map cannot invert.

    KOT goes through this function too. It did not, and read protein.X directly, so phi
    trained on CLR while every baseline it was ranked against trained on rna_size. On the
    Papalexi panel that ceiling is about +0.37 Spearman for a PERFECT CLR map, and CLR
    closure across only four proteins inverts three of them: phi predicted PDL1 at +0.74
    and CD366/CD86/PDL2 backwards, which cancelled to ~0 in the pooled metric. One
    normalisation per dataset, for every method, is the point of this function.

    Under holdout_second the runner has already cut ``second_adata`` to the fitted cells,
    so its rows line up with ``fit_rows`` without re-indexing.
    """
    second = context["second_adata"]
    if str(cfg.get("protein_target_normalization", "layer")) != "rna_size":
        return matrix_from_adata(second, cfg.get("kot_protein_layer"), "protein")
    counts = matrix_from_adata(second, "counts", "protein")
    rna_totals = rna_size_totals(context, counts.shape[0])
    return rna_size_adt(counts, rna_totals).astype(np.float32)


def protein_target_units(cfg: dict, protein_layer: str | None = None) -> str:
    """Name of the units ``adt_targets`` returns, for logs and for the run stamp.

    Written into every KOT checkpoint so a checkpoint trained on CLR can never be
    re-scored as though it were in the benchmark's units.
    """
    if str(cfg.get("protein_target_normalization", "layer")) == "rna_size":
        return "rna_size"
    return str(protein_layer or "X")


def rna_size_totals(context: dict, n_fit: int) -> np.ndarray:
    """Total RNA counts for the fitted cells, the size factor from outside the ADT panel."""
    rna_adata = context["rna_adata"]
    fit_rows = fit_rows_of(context, rna_adata.n_obs)
    if len(fit_rows) != n_fit:
        raise ValueError(
            f"{n_fit} fitted protein rows against {len(fit_rows)} fitted RNA rows; the "
            "size factor would be taken from the wrong cells.")
    if "nCount_RNA" in rna_adata.obs.columns:
        return np.asarray(rna_adata.obs["nCount_RNA"].to_numpy()[fit_rows], dtype=np.float64)
    counts = matrix_from_adata(rna_adata, "counts", "RNA")
    return np.asarray(counts[fit_rows].sum(axis=1), dtype=np.float64).ravel()


def counts_layer(adata, label: str) -> np.ndarray:
    """Raw integer counts, which sciPENN and scButterfly normalize themselves."""
    return matrix_from_adata(adata, "counts", label)


def check_full_length(predicted: np.ndarray, n_cells: int, n_adt: int, model: str) -> None:
    """External predictors are allowed to filter cells; this benchmark is not.

    A method that silently returns fewer rows than it was given is scored against fewer
    distractors than every other method, and its rows can no longer be mapped back to a
    knockout label. Fail here rather than let a shortened prediction reach the table.
    """
    if predicted.shape != (n_cells, n_adt):
        raise ValueError(
            f"{model} returned a prediction of shape {predicted.shape}, expected "
            f"({n_cells}, {n_adt}). Every cell must be predicted, in input order — a "
            "dropped cell cannot be scored against its knockout label. Check the "
            "wrapper's QC/filter arguments."
        )
