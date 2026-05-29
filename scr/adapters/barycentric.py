import numpy as np


def apply_barycentric(coupling: np.ndarray, source: np.ndarray) -> np.ndarray:
    """
    Project target cells into source space via barycentric projection.
    Used by OT-based models (SCOT, MOSCOT, Linear ODE) that return a coupling matrix.

    coupling : (n_target, n_source)  OT transport plan
    source   : (n_source, d)         source embeddings
    returns  : (n_target, d)         projected target embeddings
    """
    weights = coupling.sum(axis=1, keepdims=True)
    return (coupling / weights) @ source
