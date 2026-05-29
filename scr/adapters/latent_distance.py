import numpy as np

from scr.metrics.FOSCTTM import calc_domainAveraged_FOSCTTM


def paired_foscttm(latent_x: np.ndarray, latent_y: np.ndarray) -> tuple[list, float]:
    """
    Compute FOSCTTM directly from shared latent embeddings.
    Used by models that co-embed both modalities (GLUE, uniPort, totalVI).
    Both arrays must have the same row order (cell i in x matches cell i in y).
    Returns (per_cell_scores, mean_score).
    """
    scores = calc_domainAveraged_FOSCTTM(latent_x, latent_y)
    return scores, float(np.mean(scores))
