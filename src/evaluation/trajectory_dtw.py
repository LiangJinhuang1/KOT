"""Trajectory DTW on pseudotime-binned centroids (O(n_bins²), not O(n_cells²)).

  temporal  predicted order vs true order — the axis FOSCTTM and JVP-cosine miss.
  recon     both ordered by true time — smoothed reconstruction, small warps ok.

Both matrices must live in the same space for cross-model comparison.
"""
import numpy as np


def bin_trajectory(matrix: np.ndarray, pseudotime: np.ndarray, n_bins: int) -> np.ndarray:
    """Occupied-bin centroids only: empty bins would be NaN steps in DTW."""
    t = np.asarray(pseudotime, dtype=np.float64)
    matrix = np.asarray(matrix, dtype=np.float64)
    lo, hi = float(t.min()), float(t.max())
    if hi <= lo:
        return matrix.mean(axis=0, keepdims=True)

    edges = np.linspace(lo, hi, n_bins + 1)
    bin_idx = np.clip(np.digitize(t, edges[1:-1]), 0, n_bins - 1)
    centroids = []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.any():
            centroids.append(matrix[mask].mean(axis=0))
    return np.asarray(centroids)


def dtw_distance(seq_a: np.ndarray, seq_b: np.ndarray) -> float:
    """Path cost / (T_a + T_b) so length is not a score."""
    seq_a = np.asarray(seq_a, dtype=np.float64)
    seq_b = np.asarray(seq_b, dtype=np.float64)
    ta, tb = len(seq_a), len(seq_b)

    cost = np.sqrt(((seq_a[:, None, :] - seq_b[None, :, :]) ** 2).sum(axis=2))  # (ta, tb)
    acc = np.full((ta + 1, tb + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, ta + 1):
        row_cost = cost[i - 1]
        for j in range(1, tb + 1):
            acc[i, j] = row_cost[j - 1] + min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])

    return float(acc[ta, tb]) / (ta + tb)


def trajectory_dtw(
    predicted: np.ndarray,
    observed: np.ndarray,
    pred_pseudotime: np.ndarray,
    obs_pseudotime: np.ndarray,
    n_bins: int = 100,
) -> float:
    """DTW on binned centroids. Both matrices must already share a space."""
    pred_traj = bin_trajectory(predicted, pred_pseudotime, n_bins)
    obs_traj = bin_trajectory(observed, obs_pseudotime, n_bins)
    return dtw_distance(pred_traj, obs_traj)
