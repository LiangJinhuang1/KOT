"""Pure helpers for RegVelo regulon-knockout Task B predictions."""
from __future__ import annotations

import numpy as np


def silence_outgoing(weights, regulator_index: int, cutoff: float = 1e-3):
    """Return a copy with active outgoing weights for one regulator set to zero."""
    values = np.asarray(weights, dtype=float)
    if values.ndim != 2 or not 0 <= regulator_index < values.shape[1]:
        raise ValueError("Invalid RegVelo weight matrix or regulator index")
    result = values.copy()
    active = np.abs(result[:, regulator_index]) > cutoff
    result[active, regulator_index] = 0.0
    return result, active


def counterfactual_query(observed_nt, baseline_fit, perturbed_fit, feature_range,
                         model_indices=None, feature_scales=None):
    """Transfer RegVelo's fitted shift to observed NT RNA in KOT coordinates.

    ``model_indices`` maps each KOT feature to a RegVelo feature. ``feature_scales``
    is fitted on NT cells only and converts the unscaled RegVelo moment to KOT's
    corresponding moment scale.
    """
    observed_nt = np.asarray(observed_nt, dtype=float)
    baseline_fit = np.asarray(baseline_fit, dtype=float)
    perturbed_fit = np.asarray(perturbed_fit, dtype=float)
    feature_range = np.asarray(feature_range, dtype=float)
    if baseline_fit.shape != perturbed_fit.shape or observed_nt.shape[0] != baseline_fit.shape[0]:
        raise ValueError("Observed, baseline, and perturbed profiles have incompatible shapes")
    if feature_range.shape != (baseline_fit.shape[1],):
        raise ValueError("feature_range must have one value per RegVelo feature")
    if model_indices is None:
        model_indices = np.arange(observed_nt.shape[1])
    model_indices = np.asarray(model_indices, dtype=int)
    if model_indices.shape != (observed_nt.shape[1],) or (model_indices < 0).any():
        raise ValueError("model_indices must map every KOT feature")
    if feature_scales is None:
        feature_scales = np.ones(observed_nt.shape[1])
    feature_scales = np.asarray(feature_scales, dtype=float)
    if feature_scales.shape != (observed_nt.shape[1],):
        raise ValueError("feature_scales must have one value per KOT feature")
    model_delta = (perturbed_fit - baseline_fit) * feature_range
    query = observed_nt + model_delta[:, model_indices] * feature_scales
    clipped = query < 0
    return np.maximum(query, 0.0), clipped


def positive_feature_scales(source, target, eps=1e-12):
    """Positive per-feature SD ratios fitted entirely from paired NT cells."""
    source, target = np.asarray(source, float), np.asarray(target, float)
    if source.shape != target.shape or source.ndim != 2:
        raise ValueError("source and target must be equal two-dimensional arrays")
    source_sd, target_sd = source.std(0), target.std(0)
    return np.divide(target_sd, source_sd, out=np.zeros_like(target_sd),
                     where=source_sd > eps)
