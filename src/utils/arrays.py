import numpy as np


def to_dense(matrix, dtype=None):
    """Densify a scipy-sparse or array-like into an ndarray, optionally casting dtype."""
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=dtype)
