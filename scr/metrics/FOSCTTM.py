import numpy as np


def calc_frac_idx(x1_mat: np.ndarray, x2_mat: np.ndarray, batch_size: int = 1000) -> list[float]:
    """
    Fraction of samples closer than the true match, computed in batches.

    Complexity: O(n * batch_size) memory, O(n²) time — no Python loop over cells.
    For n=5000: fast. For n=90000: runs in batches to stay within memory.
    """
    n = x1_mat.shape[0]
    fracs = np.empty(n, dtype=np.float32)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        # (batch, n) squared distances from each query row to all x2 rows
        diff = x1_mat[start:end, None, :] - x2_mat[None, :, :]   # (batch, n, d)
        dists = np.sqrt((diff ** 2).sum(axis=2))                   # (batch, n)
        true_dists = dists[np.arange(end - start), np.arange(start, end)]  # (batch,)
        fracs[start:end] = (dists < true_dists[:, None]).sum(axis=1) / (n - 1)

    return fracs.tolist()


def calc_domainAveraged_FOSCTTM(x1_mat: np.ndarray, x2_mat: np.ndarray) -> list[float]:
    """Average FOSCTTM over both directions."""
    fracs1 = calc_frac_idx(x1_mat, x2_mat)
    fracs2 = calc_frac_idx(x2_mat, x1_mat)
    return [(f1 + f2) / 2 for f1, f2 in zip(fracs1, fracs2)]
