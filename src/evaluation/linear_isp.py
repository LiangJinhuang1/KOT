"""Linear identity-to-RNA-shift baseline for held-out replicate prediction.

Training examples are replicate-level KO-minus-NT effects, equally weighted.
This estimates known interventions in a new replicate, NOT unseen interventions.
All arrays passed to fit must already use frozen NT-fitted KOT coordinates.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class LinearISP:
    targets: tuple[str, ...]
    coefficients: np.ndarray
    training_rows: np.ndarray

    def predict(self, nt_rna: np.ndarray, target: str) -> np.ndarray:
        if target not in self.targets:
            raise ValueError(f"No training replicate for knockout {target}")
        return np.asarray(nt_rna) + self.coefficients[self.targets.index(target)]


def fit_linear_isp(features, groups, replicates, held_out, *, min_cells=20, alpha=1.0):
    """Ridge regression of RNA effects on one-hot knockout identities, no intercept.

    Only non-held-out cells enter the fit; cell counts determine eligibility but
    do not weight the loss. The fixed alpha is not selected on test responses.
    """
    x = np.asarray(features, dtype=float)
    groups, replicates = np.asarray(groups, str), np.asarray(replicates, str)
    if x.ndim != 2 or len(x) != len(groups) or len(x) != len(replicates):
        raise ValueError("Feature and metadata dimensions do not match")
    if alpha < 0 or not np.isfinite(alpha) or min_cells < 1:
        raise ValueError("Require finite alpha >= 0 and min_cells >= 1")
    if str(held_out) not in replicates:
        raise ValueError("Held-out replicate is absent")
    rows, effects, targets = [], [], []
    for rep in sorted(set(replicates) - {str(held_out)}):
        nt = (replicates == rep) & (groups == "NT")
        if nt.sum() < min_cells:
            continue
        if not np.isfinite(x[nt]).all():
            raise ValueError("Nonfinite training controls")
        for target in sorted(set(groups[replicates == rep]) - {"NT"}):
            ko = (replicates == rep) & (groups == target)
            if ko.sum() < min_cells:
                continue
            if not np.isfinite(x[ko]).all():
                raise ValueError("Nonfinite training knockout RNA")
            targets.append(target)
            effects.append(x[ko].mean(0) - x[nt].mean(0))
            rows.extend(np.flatnonzero(ko | nt))
    if not targets:
        raise ValueError("No sufficiently sampled training effects outside held-out replicate")
    names = tuple(sorted(set(targets)))
    design = np.array([[t == name for name in names] for t in targets], dtype=float)
    coefficients = np.linalg.solve(
        design.T @ design + alpha * np.eye(len(names)), design.T @ np.array(effects)
    )
    return LinearISP(names, coefficients, np.unique(rows))
