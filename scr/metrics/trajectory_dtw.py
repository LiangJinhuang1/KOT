"""
Trajectory DTW (Dynamic Time Warping).

Measures temporal alignment between predicted and true protein trajectories
ordered along pseudo-time. Lower DTW distance = better alignment.
"""
import numpy as np


def order_by_pseudotime(matrix: np.ndarray, pseudotime: np.ndarray) -> np.ndarray:
    """Return rows of matrix sorted by ascending pseudotime."""
    order = np.argsort(pseudotime)
    return matrix[order]


def dtw_distance(seq_a: np.ndarray, seq_b: np.ndarray) -> float:
    """
    Standard DTW distance between two sequences of shape (T, D).
    Cost is squared Euclidean distance per step.
    Returns the normalised total cost (divided by T_a + T_b).
    """
    T_a, T_b = len(seq_a), len(seq_b)
    cost = np.full((T_a, T_b), np.inf)
    cost[0, 0] = np.sum((seq_a[0] - seq_b[0]) ** 2)

    for i in range(1, T_a):
        cost[i, 0] = cost[i - 1, 0] + np.sum((seq_a[i] - seq_b[0]) ** 2)
    for j in range(1, T_b):
        cost[0, j] = cost[0, j - 1] + np.sum((seq_a[0] - seq_b[j]) ** 2)
    for i in range(1, T_a):
        for j in range(1, T_b):
            step = np.sum((seq_a[i] - seq_b[j]) ** 2)
            cost[i, j] = step + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

    return float(cost[-1, -1]) / (T_a + T_b)


def trajectory_dtw(
    predicted: np.ndarray,
    observed: np.ndarray,
    pseudotime: np.ndarray,
) -> float:
    """
    Compute DTW between predicted and observed protein trajectories.

    Parameters
    ----------
    predicted   : (n, D_p)  predicted protein states
    observed    : (n, D_p)  observed protein states
    pseudotime  : (n,)      pseudo-time coordinate for each cell

    Returns
    -------
    dtw_dist : float  normalised DTW distance (lower = better)
    """
    pred_ordered = order_by_pseudotime(predicted, pseudotime)
    obs_ordered  = order_by_pseudotime(observed,  pseudotime)
    return dtw_distance(pred_ordered, obs_ordered)
