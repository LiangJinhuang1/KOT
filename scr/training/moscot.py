from __future__ import annotations

import numpy as np

from moscot.problems.cross_modality import TranslationProblem


def run_moscot(context: dict, cfg: dict) -> tuple[list, np.ndarray | None]:
    """
    Cross-modality alignment using MOSCOT's TranslationProblem (Gromov-Wasserstein).

    Pure GW (alpha=1, joint_attr=None): aligns RNA and protein via geometry only,
    no shared features required.

    Returns
    -------
    aligned  : [rna_in_protein_space (n, D_p), protein_obs (n, D_p)]
    coupling : (n, n) transport plan
    """
    rna_adata    = context["rna_adata"]
    second_adata = context["second_adata"]
    rna_repr_key    = str(cfg.get("rna_repr",    "X_pca"))
    second_repr_key = str(cfg.get("second_repr", "X_pca"))

    epsilon  = float(cfg.get("epsilon",      0.1))
    tau_a    = float(cfg.get("tau_a",        1.0))
    tau_b    = float(cfg.get("tau_b",        1.0))
    max_iter = int(cfg.get("max_iterations", 100))

    tp = TranslationProblem(rna_adata, second_adata)
    tp = tp.prepare(
        src_attr=rna_repr_key,
        tgt_attr=second_repr_key,
        joint_attr=None,   # pure GW — no shared features between RNA and protein
    )
    tp = tp.solve(
        alpha=1.0,         # pure GW (joint_attr=None makes alpha irrelevant, explicit for clarity)
        epsilon=epsilon,
        tau_a=tau_a,
        tau_b=tau_b,
        max_iterations=max_iter,
    )

    # Barycentric projection: RNA cells mapped into protein PCA space
    rna_translated = np.array(tp.translate("src", "tgt", forward=True))
    protein_obs    = np.array(second_adata.obsm[second_repr_key])

    coupling = np.array(tp.solutions[("src", "tgt")].transport_matrix)

    return [rna_translated, protein_obs], coupling
