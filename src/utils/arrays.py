import numpy as np


def to_dense(matrix, dtype=None):
    """Densify a scipy-sparse or array-like into an ndarray, optionally casting dtype."""
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=dtype)


def matrix_from_adata(adata, layer_key: str | None, label: str) -> np.ndarray:
    """Read ``X`` or a named AnnData layer as a dense float32 matrix."""
    if layer_key in (None, "", "X"):
        return to_dense(adata.X, np.float32)
    if layer_key not in adata.layers:
        available = list(adata.layers.keys())
        raise ValueError(f"{label} layer '{layer_key}' not found. Available layers: {available}")
    return to_dense(adata.layers[layer_key], np.float32)
