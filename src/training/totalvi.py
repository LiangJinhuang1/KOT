from __future__ import annotations

import warnings

import numpy as np
import anndata as ad
import pandas as pd
import scvi
import torch

from src.utils.arrays import to_dense


PROTEIN_OBSM_KEY = "protein_expression"


REDUCE_LR_SCHEDULER = torch.optim.lr_scheduler.ReduceLROnPlateau


def init_dropping_verbose(self, *args, **kwargs) -> None:
    """Replacement ReduceLROnPlateau.__init__ that drops the kwarg torch 2.10 removed.

    The original __init__ is parked on the class as ``kot_original_init`` by
    :func:`drop_reduce_lr_verbose_kwarg`, which is the only thing that installs this.
    """
    kwargs.pop("verbose", None)
    REDUCE_LR_SCHEDULER.kot_original_init(self, *args, **kwargs)


def drop_reduce_lr_verbose_kwarg() -> None:
    """scvi 1.2.0 passes verbose= to torch's ReduceLROnPlateau, which torch 2.10 removed.

    Wrap its __init__ to drop the now-unsupported kwarg. Idempotent, so calling it before
    every totalVI train is safe.
    """
    if getattr(REDUCE_LR_SCHEDULER, "kot_verbose_dropped", False):
        return
    REDUCE_LR_SCHEDULER.kot_original_init = REDUCE_LR_SCHEDULER.__init__
    REDUCE_LR_SCHEDULER.__init__ = init_dropping_verbose
    REDUCE_LR_SCHEDULER.kot_verbose_dropped = True


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

    drop_reduce_lr_verbose_kwarg()
    # scvi warns (and this env escalates it to an error) when the protein layer is not raw
    # integer counts. BMMC's ADT comes pre-normalized from the NeurIPS object; totalVI is the
    # PAIRED upper bound (shared encoder → FOSCTTM ≈ 0 regardless of protein normalization), so
    # silence that specific warning rather than let it abort the run. pbmc/papalexi (raw counts)
    # never trip it.
    warnings.filterwarnings("ignore", message=".*unnormalized count data.*", category=UserWarning)
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
