"""One entropic-OT solver, because two of them disagreed.

The chromatin experiment needs an OT PLAN in two places — the day-0 → day-7 coupling that
defines the HSPC velocity, and the transport matcher in Task B — and `geomloss` gives a
divergence, not a plan. Both were written separately, both got the log-domain update
wrong in the same way, and the second copy was still wrong after the first was fixed: its
matcher scored 0.001 top-1 on a cloud whose true partner was obvious. So the solver lives
here once and both call it.

The update that goes wrong: f and g are the FULL potentials, so the softmin must not add
the previous value back on top of the logsumexp. Doing that double-counts them and they
diverge to +-inf within tens of iterations; exp() then returns a plan of NaN (visible), or
the potentials saturate and the plan goes uniform (invisible, and far worse).
"""

from __future__ import annotations

import numpy as np
import torch


def squared_distances(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Pairwise squared Euclidean distances, without torch.cdist.

    On blocks of this size cdist takes the matrix-multiplication path, whose rounding
    leaves tiny negative entries, and its sqrt turns those into NaN. One NaN is enough to
    poison a median or an OT plan, and nothing downstream would say where it came from.
    """
    return (x.pow(2).sum(1)[:, None] + y.pow(2).sum(1)[None, :]
            - 2.0 * (x @ y.T)).clamp(min=0.0)


def sinkhorn_plan(source, target, epsilon: float, n_iterations: int,
                  device: torch.device | None = None) -> torch.Tensor:
    """The entropic OT plan between two point clouds, in the log domain.

    The cost is normalised by its own median so `epsilon` means the same thing whatever
    the dimensionality and scale of the space — 2,000 log-counts and 50 LSI components are
    otherwise orders of magnitude apart and one epsilon cannot serve both.
    """
    device = device or torch.device("cpu")
    x = torch.as_tensor(np.asarray(source), dtype=torch.float32, device=device)
    y = torch.as_tensor(np.asarray(target), dtype=torch.float32, device=device)
    cost = squared_distances(x, y)
    cost = cost / cost.median().clamp(min=1e-12)

    log_a = torch.full((len(x),), -float(np.log(len(x))), device=device)
    log_b = torch.full((len(y),), -float(np.log(len(y))), device=device)
    f = torch.zeros(len(x), device=device)
    g = torch.zeros(len(y), device=device)
    for _ in range(n_iterations):
        f = epsilon * (log_a - torch.logsumexp((g[None, :] - cost) / epsilon, dim=1))
        g = epsilon * (log_b - torch.logsumexp((f[:, None] - cost) / epsilon, dim=0))
    plan = torch.exp((f[:, None] + g[None, :] - cost) / epsilon)
    assert torch.isfinite(plan).all(), "the entropic OT plan contains non-finite entries"
    return plan


def row_entropy(plan: torch.Tensor) -> np.ndarray:
    """Shannon entropy of each row, after normalising the row to a distribution.

    A row spread evenly over every target says "the future is the target mean" and carries
    no information; log(n) is that worst case and 0 is a matched pair.
    """
    weights = plan / plan.sum(dim=1, keepdim=True).clamp(min=1e-30)
    return (-(weights * torch.log(weights.clamp(min=1e-30))).sum(dim=1)).cpu().numpy()
