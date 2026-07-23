from __future__ import annotations

import numpy as np
import anndata as ad
import pandas as pd
import scvi

from src.utils.arrays import to_dense


PROTEIN_OBSM_KEY = "protein_expression"


def attach_protein_expression(
    adata: ad.AnnData,
    second_adata: ad.AnnData,
    protein: np.ndarray,
) -> None:
    adata.obsm[PROTEIN_OBSM_KEY] = pd.DataFrame(
        protein,
        index=adata.obs_names,
        columns=second_adata.var_names.astype(str),
    )


def train_totalvi_model(
    combined: ad.AnnData,
    cfg: dict,
    seed: int,
):
    n_latent = int(cfg.get("n_latent", 20))
    max_epochs = int(cfg.get("totalvi_max_epochs", 400))

    scvi.settings.seed = seed
    scvi.model.TOTALVI.setup_anndata(
        combined,
        protein_expression_obsm_key=PROTEIN_OBSM_KEY,
    )
    model = scvi.model.TOTALVI(combined, n_latent=n_latent)
    model.train(max_epochs=max_epochs)
    return model


def run_totalvi_latent_oracle(
    rna_adata: ad.AnnData,
    second_adata: ad.AnnData,
    rna_counts: np.ndarray,
    prot_counts: np.ndarray,
    cfg: dict,
    seed: int,
) -> tuple[list, None]:
    combined = ad.AnnData(
        X=rna_counts,
        obs=rna_adata.obs.copy(),
        var=rna_adata.var.copy(),
    )
    attach_protein_expression(combined, second_adata, prot_counts)

    model = train_totalvi_model(combined, cfg, seed)
    latent = model.get_latent_representation()
    return [latent, latent], None


def run_totalvi(context: dict, cfg: dict) -> tuple[list, np.ndarray | None]:
    """
    Paired CITE-seq totalVI upper bound.

    One encoder reads RNA + protein from the same cell, so the joint latent is
    identical for both modalities and FOSCTTM is approximately zero.
    """
    rna_adata    = context["rna_adata"]
    second_adata = context["second_adata"]
    second_label = context.get("second_label", "protein")
    if second_label != "protein":
        raise ValueError("totalVI is only defined for RNA + protein/CITE-seq data.")

    seed = int(cfg.get("seed", 42))
    rna_counts = to_dense(rna_adata.layers["counts"], np.float32)
    prot_counts = to_dense(second_adata.layers["counts"], np.float32)
    return run_totalvi_latent_oracle(
        rna_adata, second_adata, rna_counts, prot_counts, cfg, seed
    )
