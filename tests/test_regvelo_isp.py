import numpy as np
import pytest

from src.evaluation.regvelo_isp import (
    counterfactual_query, positive_feature_scales, silence_outgoing,
)


def test_silences_only_active_outgoing_edges_without_mutating_input():
    weights = np.array([[1.0, 0.2], [0.0001, -2.0], [3.0, 0.0]])
    result, active = silence_outgoing(weights, 0, cutoff=0.001)
    np.testing.assert_array_equal(active, [True, False, True])
    np.testing.assert_allclose(result[:, 0], [0, 0.0001, 0])
    np.testing.assert_allclose(result[:, 1], weights[:, 1])
    np.testing.assert_allclose(weights[:, 0], [1, 0.0001, 3])


def test_counterfactual_transfers_unscaled_shift_and_clips_negative_values():
    observed = np.array([[2.0, 1.0], [3.0, 4.0]])
    base = np.array([[0.5, 0.5], [0.3, 0.1]])
    perturbed = np.array([[0.0, 1.0], [0.1, 0.0]])
    query, clipped = counterfactual_query(observed, base, perturbed, [10.0, 2.0])
    np.testing.assert_allclose(query, [[0.0, 2.0], [1.0, 3.8]])
    np.testing.assert_array_equal(clipped, [[True, False], [False, False]])


def test_counterfactual_rejects_feature_mismatch():
    with pytest.raises(ValueError, match="feature_range"):
        counterfactual_query(np.zeros((2, 3)), np.zeros((2, 3)),
                             np.zeros((2, 3)), np.ones(2))


def test_maps_model_features_and_nt_fitted_scales():
    observed = np.ones((2, 2))
    baseline = np.zeros((2, 3))
    perturbed = np.array([[1., 2., 3.], [1., 2., 3.]])
    query, _ = counterfactual_query(observed, baseline, perturbed, [1., 2., 3.],
                                    model_indices=[2, 0], feature_scales=[2., 4.])
    np.testing.assert_allclose(query, [[19., 5.], [19., 5.]])
    scales = positive_feature_scales([[0., 0.], [2., 4.]], [[0., 0.], [4., 2.]])
    np.testing.assert_allclose(scales, [2., .5])
