"""kNN alignment: is the true partner inside the top k, for every k?

FOSCTTM reports one number per cell — the fraction of candidates closer than the true
partner. Averaged, it hides the SHAPE of the failure: a method that is either exactly
right or catastrophically wrong and one that is consistently a near-miss can share a
mean. The curve over k separates them, and it is the metric MaxFuse reports as FOSKNN.

It costs nothing new to compute. `calc_frac_idx` already returns, per cell, the fraction
of candidates strictly closer than the true partner; multiplying by (n - 1) turns that
fraction back into the partner's rank, and the curve is the fraction of cells whose rank
falls below each k. The saved `foscttm.csv` cannot be reused for this — it holds the
DOMAIN-AVERAGED score, the mean of both directions, and the mean of two ranks is not a
rank.

WHY k IS A FRACTION. The datasets differ by more than an order of magnitude in cell
count (PBMC 5.5k, BMMC 90k), so "the true partner is in the top 50" means something
different on each. Reporting k as a fraction of n makes the curves comparable, which is
also how MaxFuse states its FOSKNN operating point.
"""

from __future__ import annotations

import numpy as np

from src.evaluation.foscttm import calc_frac_idx

# Cells drawn for one curve. The rank scan is O(n^2) and this box has no GPU, so the
# curve is estimated on a common subsample rather than computed on all of BMMC's 90k
# cells — a fixed n also keeps the two datasets on one k axis.
DEFAULT_SUBSAMPLE = 6000


def true_match_ranks(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Rank of each cell's true partner: how many candidates are strictly closer."""
    n = len(x1)
    fractions = np.asarray(calc_frac_idx(x1, x2), dtype=np.float64)
    return np.rint(fractions * (n - 1)).astype(np.int64)


def subsample_pair(x1: np.ndarray, x2: np.ndarray, n: int, seed: int):
    """The same rows of both modalities, so row i stays the partner of row i."""
    if len(x1) <= n:
        return x1, x2
    idx = np.random.default_rng(seed).choice(len(x1), n, replace=False)
    return x1[idx], x2[idx]


def alignment_curve(x1: np.ndarray, x2: np.ndarray, fractions: np.ndarray, *,
                    subsample: int = DEFAULT_SUBSAMPLE, seed: int = 0) -> np.ndarray:
    """Fraction of cells whose true partner falls inside the top `fractions` of n."""
    a, b = subsample_pair(x1, x2, subsample, seed)
    ranks = true_match_ranks(a, b)
    k_values = np.maximum(1, np.rint(fractions * len(a)).astype(np.int64))
    return np.array([float((ranks < k).mean()) for k in k_values])
