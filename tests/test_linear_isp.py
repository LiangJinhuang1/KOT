import numpy as np
import pytest

from src.evaluation.linear_isp import fit_linear_isp


def test_held_out_rna_cannot_change_fit_or_prediction():
    x = np.array([[1., 2.], [3., 6.], [9., 8.], [99., 88.]])
    groups = ['NT', 'A', 'NT', 'A']
    reps = ['train', 'train', 'test', 'test']
    first = fit_linear_isp(x, groups, reps, 'test', min_cells=1, alpha=0)
    x[2:] = np.nan
    second = fit_linear_isp(x, groups, reps, 'test', min_cells=1, alpha=0)
    np.testing.assert_array_equal(first.coefficients, second.coefficients)
    np.testing.assert_array_equal(first.training_rows, [0, 1])
    np.testing.assert_allclose(first.predict([[10., 10.]], 'A'), [[12., 14.]])


def test_replicates_equally_weighted_and_ridge_shrinks():
    x = np.array([[0.], [2.], [0.], [4.], [4.], [4.], [0.], [9.]])
    groups = ['NT', 'A', 'NT', 'A', 'A', 'A', 'NT', 'A']
    reps = ['one', 'one', 'two', 'two', 'two', 'two', 'test', 'test']
    fit = fit_linear_isp(x, groups, reps, 'test', min_cells=1, alpha=1)
    np.testing.assert_allclose(fit.coefficients, [[2.]])  # (2 + 4) / (2 + alpha)
    with pytest.raises(ValueError, match='No training replicate'):
        fit.predict([[0.]], 'unseen')


def test_threshold_and_no_training_failure():
    with pytest.raises(ValueError, match='No sufficiently sampled'):
        fit_linear_isp([[0.], [1.]], ['NT', 'A'], ['one', 'test'], 'test', min_cells=1)
