"""
Perturbation metrics for CRISPR knockout evaluation.

• Sign accuracy : fraction of proteins where predicted fold-change direction matches ground truth.
• Pearson / Spearman correlation : magnitude agreement of fold-changes.
"""
import numpy as np
from scipy.stats import pearsonr, spearmanr


def fold_change(perturbed: np.ndarray, control: np.ndarray) -> np.ndarray:
    """Log2 fold-change: perturbed vs control, per protein. Shape (D_p,)."""
    eps = 1e-8
    return np.log2((perturbed + eps) / (control + eps))


def sign_accuracy(predicted_fc: np.ndarray, true_fc: np.ndarray) -> float:
    """
    Fraction of proteins where predicted and true fold-change share the same sign.
    Proteins with true_fc == 0 are excluded (no ground-truth direction).

    Parameters
    ----------
    predicted_fc : (D_p,) predicted log2 fold-changes
    true_fc      : (D_p,) ground-truth log2 fold-changes

    Returns
    -------
    accuracy : float in [0, 1]
    """
    mask = true_fc != 0.0
    if mask.sum() == 0:
        return float("nan")
    matches = np.sign(predicted_fc[mask]) == np.sign(true_fc[mask])
    return float(matches.mean())


def pearson_correlation(predicted_fc: np.ndarray, true_fc: np.ndarray) -> float:
    """Pearson r between predicted and true fold-changes."""
    r, _ = pearsonr(predicted_fc, true_fc)
    return float(r)


def spearman_correlation(predicted_fc: np.ndarray, true_fc: np.ndarray) -> float:
    """Spearman ρ between predicted and true fold-changes."""
    rho, _ = spearmanr(predicted_fc, true_fc)
    return float(rho)


def evaluate_perturbation(
    predicted_perturbed: np.ndarray,
    true_perturbed: np.ndarray,
    control: np.ndarray,
) -> dict:
    """
    Full perturbation evaluation from raw expression matrices.

    Parameters
    ----------
    predicted_perturbed : (n_cells, D_p)  model output for knockout cells
    true_perturbed      : (n_cells, D_p)  observed protein levels for knockout cells
    control             : (n_ctrl,  D_p)  observed protein levels for control cells

    Returns
    -------
    dict with keys: sign_accuracy, pearson, spearman
    """
    ctrl_mean = control.mean(axis=0)
    pred_fc   = fold_change(predicted_perturbed.mean(axis=0), ctrl_mean)
    true_fc   = fold_change(true_perturbed.mean(axis=0),      ctrl_mean)

    return {
        "sign_accuracy": sign_accuracy(pred_fc, true_fc),
        "pearson":       pearson_correlation(pred_fc, true_fc),
        "spearman":      spearman_correlation(pred_fc, true_fc),
    }
