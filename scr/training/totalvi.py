from __future__ import annotations

import numpy as np
import anndata as ad
import scipy.sparse as sp
import scvi


def run_totalvi(context: dict, cfg: dict) -> tuple[list, np.ndarray | None]:
    """
    totalVI: paired CITE-seq VAE (upper bound baseline).

    Trained on the true paired RNA + protein data, so both modalities share
    the same joint latent vector per cell → FOSCTTM ≈ 0 by construction.

    Returns
    -------
    aligned  : [latent (n, n_latent), latent (n, n_latent)]  — same array twice
    coupling : None
    """
    rna_adata    = context["rna_adata"]
    second_adata = context["second_adata"]

    n_latent   = int(cfg.get("n_latent",       20))
    max_epochs = int(cfg.get("totalvi_max_epochs", 400))
    seed       = int(cfg.get("seed",           42))

    # raw counts required by totalVI's NB / mixture decoder
    rna_counts = rna_adata.layers["counts"]
    if sp.issparse(rna_counts):
        rna_counts = rna_counts.toarray()
    rna_counts = np.asarray(rna_counts, dtype=np.float32)

    prot_counts = second_adata.layers["counts"]
    if sp.issparse(prot_counts):
        prot_counts = prot_counts.toarray()
    prot_counts = np.asarray(prot_counts, dtype=np.float32)

    # paired AnnData: RNA in X, protein in obsm
    combined = ad.AnnData(
        X=rna_counts,
        obs=rna_adata.obs.copy(),
        var=rna_adata.var.copy(),
    )
    combined.obsm["protein_expression"] = prot_counts

    scvi.settings.seed = seed
    scvi.model.TOTALVI.setup_anndata(
        combined,
        protein_expression_obsm_key="protein_expression",
    )

    model = scvi.model.TOTALVI(combined, n_latent=n_latent)
    model.train(max_epochs=max_epochs)

    latent = model.get_latent_representation()  # (n_cells, n_latent)

    return [latent, latent], None
