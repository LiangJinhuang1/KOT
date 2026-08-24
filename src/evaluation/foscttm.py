from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

PREDICTIONS_DIR = Path("data/predictions")


def calc_frac_idx(x1_mat: np.ndarray, x2_mat: np.ndarray, batch_size: int = 1000) -> list[float]:
    """
    Fraction of samples closer than the true match, computed in batches.

    Runs on GPU when available (falls back to CPU) — identical result to the
    dense NumPy version, but the O(n²) distance work is the bottleneck at large
    n (e.g. 90k cells), so the device matters. Row-batched to bound memory.
    """
    x1_array = np.asarray(x1_mat, dtype=np.float32)
    x2_array = np.asarray(x2_mat, dtype=np.float32)
    if x1_array.ndim != 2 or x2_array.ndim != 2:
        raise ValueError("FOSCTTM inputs must both be two-dimensional matrices.")
    if x1_array.shape != x2_array.shape:
        raise ValueError(
            "FOSCTTM inputs must have the same paired shape; "
            f"got {x1_array.shape} and {x2_array.shape}."
        )
    if x1_array.shape[0] < 2:
        raise ValueError("FOSCTTM requires at least two paired observations.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    try:
        import torch
    except ModuleNotFoundError:
        return _calc_frac_idx_numpy(x1_array, x2_array, batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x1 = torch.as_tensor(x1_array, dtype=torch.float32, device=device)
    x2 = torch.as_tensor(x2_array, dtype=torch.float32, device=device)
    n = x1.shape[0]
    fracs = torch.empty(n, dtype=torch.float32, device=device)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        dists = torch.cdist(x1[start:end], x2)                          # (batch, n)
        rows = torch.arange(end - start, device=device)
        cols = torch.arange(start, end, device=device)
        true_dists = dists[rows, cols]                                  # (batch,)
        fracs[start:end] = (dists < true_dists[:, None]).sum(dim=1).float() / (n - 1)

    return fracs.cpu().tolist()


def _calc_frac_idx_numpy(
    x1: np.ndarray, x2: np.ndarray, batch_size: int
) -> list[float]:
    """CPU fallback used when PyTorch is not installed."""
    n = x1.shape[0]
    fracs = np.empty(n, dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        dists = cdist(x1[start:end], x2)
        rows = np.arange(end - start)
        cols = np.arange(start, end)
        true_dists = dists[rows, cols]
        fracs[start:end] = np.sum(dists < true_dists[:, None], axis=1) / (n - 1)
    return fracs.tolist()


def calc_domainAveraged_FOSCTTM(x1_mat: np.ndarray, x2_mat: np.ndarray) -> list[float]:
    """Average FOSCTTM over both directions."""
    fracs1 = calc_frac_idx(x1_mat, x2_mat)
    fracs2 = calc_frac_idx(x2_mat, x1_mat)
    return [(f1 + f2) / 2 for f1, f2 in zip(fracs1, fracs2)]


def is_degenerate_embedding(matrix: np.ndarray, tol: float = 1e-8) -> bool:
    """
    True when an embedding has collapsed to (essentially) a single point: every
    cell maps to the same vector. The strict-less-than comparison in calc_frac_idx
    then makes every distance tie with the true match, yielding a misleading
    FOSCTTM of 0.0 — the best possible score for a fully collapsed, useless
    embedding. Detecting this lets evaluate_and_save flag it instead of reporting
    a fake perfect alignment.
    """
    spread = float(np.max(np.std(np.asarray(matrix, dtype=np.float64), axis=0)))
    return spread < tol


def save_prediction_h5ad(
    aligned: list,
    foscttm_scores: list,
    mean_foscttm: float,
    dataset_name: str,
    model_name: str,
    second_label: str,
    obs_names,
) -> None:
    import anndata as ad

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    adata = ad.AnnData(X=aligned[0])
    adata.obs_names = list(obs_names)
    adata.obsm["X_aligned_rna"] = aligned[0].astype("float32")
    adata.obsm[f"X_aligned_{second_label}"] = aligned[1].astype("float32")
    adata.obs["foscttm"] = foscttm_scores
    adata.uns["mean_foscttm"] = mean_foscttm
    adata.uns["model"] = model_name
    adata.uns["dataset"] = dataset_name
    adata.uns["second_label"] = second_label
    path = PREDICTIONS_DIR / f"{model_name}_{dataset_name}.h5ad"
    adata.write_h5ad(path)
    print(f"Saved prediction: {path}")


def evaluate_and_save(
    aligned: list,
    name: str,
    output_dir: Path,
    second_label: str,
    model_name: str = None,
    obs_names=None,
    foscttm_scores=None,
    save_prediction: bool = True,
) -> float:
    degenerate = is_degenerate_embedding(aligned[0]) or is_degenerate_embedding(aligned[1])
    # foscttm_scores may be precomputed (e.g. per-batch alignment, where a global
    # FOSCTTM across independently-aligned batches would be meaningless).
    if foscttm_scores is None:
        foscttm_scores = calc_domainAveraged_FOSCTTM(aligned[0], aligned[1])
    finite_scores = np.asarray(foscttm_scores, dtype=np.float64)
    mean_foscttm = (
        float(np.mean(finite_scores[np.isfinite(finite_scores)]))
        if np.isfinite(finite_scores).any()
        else float("nan")
    )
    if degenerate:
        print(
            f"[{name}] WARNING: aligned embedding collapsed to a single point — "
            f"FOSCTTM={mean_foscttm:.4f} is a degenerate artifact, not real alignment. "
            "Reporting NaN instead."
        )
        mean_foscttm = float("nan")
    else:
        print(f"[{name}] Mean FOSCTTM: {mean_foscttm:.4f}  (lower = better alignment)")

    pd.DataFrame({"foscttm": foscttm_scores}).to_csv(output_dir / "foscttm.csv", index=False)
    np.save(output_dir / "aligned_rna.npy", aligned[0])
    np.save(output_dir / f"aligned_{second_label}.npy", aligned[1])

    # The prediction h5ad path is GLOBAL (data/predictions/), keyed only by model +
    # dataset + seed — so two runs that differ only in hyperparameters write the same
    # file. In a parallel sweep that is a live HDF5 lock collision that kills the run
    # (BlockingIOError), not just an overwrite. Sweeps pass save_prediction=False; the
    # per-run copies under output_dir (aligned_*.npy, foscttm.csv) are unaffected.
    if save_prediction and model_name is not None and obs_names is not None:
        save_prediction_h5ad(
            aligned, foscttm_scores, mean_foscttm,
            name, model_name, second_label, obs_names,
        )

    return mean_foscttm
