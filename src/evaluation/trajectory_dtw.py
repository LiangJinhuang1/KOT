"""
Trajectory DTW (Dynamic Time Warping).

Measures temporal alignment between a predicted and an observed trajectory,
each ordered along its own pseudotime. Cells are first binned into pseudotime
bins (pseudobulk trajectory points) so DTW stays O(n_bins^2) regardless of the
number of cells, then aligned by classic DTW. Lower distance = better temporal
agreement.

Two framings are useful for the KOT benchmark:

  temporal : predicted ordered by PREDICTED pseudotime vs observed ordered by
             TRUE pseudotime — tests whether the model recovers the temporal
             ordering, independent of exact per-cell matching. This is the
             axis that FOSCTTM (matching) and JVP-cosine (physics) do not cover.
  recon    : both ordered by the SAME (true) pseudotime — a temporally smoothed
             reconstruction error, tolerant to small warps.

The two matrices must live in the same space (protein PCA for KOT/totalVI, or a
shared aligned embedding for cross-model comparison) for the distance to be
comparable across models.
"""
import numpy as np


def bin_trajectory(matrix: np.ndarray, pseudotime: np.ndarray, n_bins: int) -> np.ndarray:
    """
    Average feature vectors within equal-width pseudotime bins.

    Returns occupied-bin centroids in ascending time order, shape (n_occupied, D).
    Empty bins are dropped so DTW never sees a NaN trajectory point. A degenerate
    pseudotime (all cells at one time) collapses to a single centroid.
    """
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
    """
    Classic DTW between two (T, D) sequences with per-step Euclidean cost.

    Returns the accumulated cost normalised by (T_a + T_b): the mean per-step
    Euclidean distance along the optimal warping path. The (T_a, T_b) cost matrix
    is vectorised; only the DP recurrence loops, so cost is O(T_a * T_b) — cheap
    once the trajectories are binned.
    """
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
    """
    DTW between a predicted and an observed trajectory in a shared space.

    Parameters
    ----------
    predicted       : (n, D)  predicted feature matrix
    observed        : (n, D)  observed feature matrix
    pred_pseudotime : (n,)    ordering coordinate for the predicted trajectory
    obs_pseudotime  : (n,)    ordering coordinate for the observed trajectory
    n_bins          : pseudotime bins; caps DTW cost at O(n_bins^2)

    Returns
    -------
    dtw_dist : float  normalised DTW distance (lower = better)
    """
    pred_traj = bin_trajectory(predicted, pred_pseudotime, n_bins)
    obs_traj = bin_trajectory(observed, obs_pseudotime, n_bins)
    return dtw_distance(pred_traj, obs_traj)
