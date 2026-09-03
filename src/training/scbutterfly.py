"""scButterfly: dual-VAE cross-modality translation, run in its CITE-seq (RNA↔ADT) mode.

Cao et al., Nature Communications 15, 2973 (2024).
https://pmc.ncbi.nlm.nih.gov/articles/PMC10998864/

The package's high-level `Butterfly` wrapper is RNA↔ATAC only (peak binarisation, BCE
decoder, chromosome-split encoder), so this drives `train_model_cite.Model` directly, as
the CITE-seq tutorial does: one ADT "chromosome", MSE decoder, identity output.

Only the R→A direction is used. `test` also runs A→R, which reads ADT; its test ids are
therefore the fitted cells, so no held-out ADT is touched even in the discarded half.
"""

from __future__ import annotations

import sys
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
from scipy.sparse import csr_matrix

# The vendored copy defers scButterfly's module-level `import episcanpy.api`, which calls
# `from anndata import read` and so cannot be imported against this project's anndata at
# all. It must precede any pip-installed scButterfly. See vendor/scbutterfly/README.md.
VENDOR_SCBUTTERFLY_PATH = Path(__file__).resolve().parents[2] / "vendor" / "scbutterfly"
if VENDOR_SCBUTTERFLY_PATH.exists():
    sys.path.insert(0, str(VENDOR_SCBUTTERFLY_PATH))

from scButterfly.train_model_cite import Model
from torch import nn

from src.training.protein_supervised import (
    adt_targets,
    check_full_length,
    counts_layer,
    fit_rows_of,
)
from src.utils.arrays import to_dense


# [r_loss, a_loss, d_loss, kl_div_R, kl_div_A, kl_div_translator]; the package supplies
# no CITE default, so this mirrors the RNA↔ATAC weighting with unit KL terms.
DEFAULT_LOSS_WEIGHT = [1.0, 2.0, 1.0, 1.0, 1.0, 1.0]


def sparse_adata(matrix: np.ndarray, obs_names) -> ad.AnnData:
    """scButterfly calls .toarray() on X, so it must be handed a sparse matrix."""
    adata = ad.AnnData(X=csr_matrix(matrix.astype(np.float32)))
    adata.obs_names = [str(name) for name in obs_names]
    return adata


def normalized_rna(counts: np.ndarray, obs_names) -> ad.AnnData:
    """CP10K + log1p, per cell, so no statistic crosses between cells."""
    adata = sparse_adata(counts, obs_names)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata


def full_length_adt(targets: np.ndarray, fit_rows: np.ndarray, n_cells: int, obs_names):
    """ADT for every row, zero where the protocol forbids reading it.

    scButterfly indexes one full-length matrix by positional id, so the held-out rows
    must exist. They are never in `train_id_a`, `validation_id_a`, or `test_id_a`, so the
    zeros are placeholders the model never sees, not imputed values it could learn from.
    """
    matrix = np.zeros((n_cells, targets.shape[1]), dtype=np.float32)
    matrix[fit_rows] = targets
    return sparse_adata(matrix, obs_names)


def train_validation_ids(fit_rows: np.ndarray, fraction: float, seed: int) -> tuple[list, list]:
    shuffled = np.random.default_rng(seed).permutation(fit_rows)
    n_val = max(1, int(round(fraction * len(shuffled))))
    return shuffled[n_val:].tolist(), shuffled[:n_val].tolist()


def run_scbutterfly(context: dict, cfg: dict) -> tuple[list, None, dict]:
    """Fit the RNA↔ADT translator on the paired fitted cells; translate every cell."""
    rna_adata = context["rna_adata"]
    second_adata = context["second_adata"]
    seed = int(cfg.get("seed", 42))
    output_path = str(context["output_dir"])

    rna_counts = counts_layer(rna_adata, "RNA")
    n_cells, n_genes = rna_counts.shape
    targets = adt_targets(context, cfg).astype(np.float32)
    n_adt = targets.shape[1]
    fit_rows = fit_rows_of(context, n_cells)

    rna_data = normalized_rna(rna_counts, rna_adata.obs_names)
    adt_data = full_length_adt(targets, fit_rows, n_cells, rna_adata.obs_names)

    embed_dim = int(cfg.get("scbutterfly_embed_dim", 128))
    model = Model(
        RNA_data=rna_data,
        ATAC_data=adt_data,
        chrom_list=[n_adt],                       # the ADT panel is one unsplit block
        logging_path=None,
        R_encoder_dim_list=[n_genes, 256, embed_dim],
        A_encoder_dim_list=[n_adt, 32, embed_dim],
        R_decoder_dim_list=[embed_dim, 256, n_genes],
        A_decoder_dim_list=[embed_dim, 32, n_adt],
        A_decoder_act_list=[nn.LeakyReLU(), nn.Identity()],   # ADT is continuous, not binary
        translator_embed_dim=embed_dim,
        translator_input_dim_r=embed_dim,
        translator_input_dim_a=embed_dim,
        dropout_rate=float(cfg.get("scbutterfly_dropout", 0.1)),
        R_noise_rate=float(cfg.get("scbutterfly_rna_noise", 0.5)),
        A_noise_rate=0.0,
    )

    train_ids, validation_ids = train_validation_ids(
        fit_rows, float(cfg.get("scbutterfly_val_fraction", 0.2)), seed)
    model.train(
        loss_weight=[float(w) for w in cfg.get("scbutterfly_loss_weight", DEFAULT_LOSS_WEIGHT)],
        train_id_r=train_ids,
        train_id_a=list(train_ids),
        validation_id_r=validation_ids,
        validation_id_a=list(validation_ids),
        R2R_pretrain_epoch=int(cfg.get("scbutterfly_rna_pretrain_epoch", 100)),
        A2A_pretrain_epoch=int(cfg.get("scbutterfly_adt_pretrain_epoch", 100)),
        translator_epoch=int(cfg.get("scbutterfly_translator_epoch", 200)),
        patience=int(cfg.get("scbutterfly_patience", 50)),
        batch_size=int(cfg.get("scbutterfly_batch_size", 64)),
        a_loss=nn.MSELoss(),
        r_loss=nn.MSELoss(),
        output_path=output_path,
        seed=seed,
    )

    # test_id_a stays on the fitted cells: the A→R half is discarded, and pointing it at
    # the knockouts would read the ADT this protocol holds out.
    _, r2a_predict = model.test(
        test_id_r=list(range(n_cells)),
        test_id_a=fit_rows.tolist(),
        load_model=False,
        output_path=output_path,
        test_cluster=False,
        test_figure=False,
        return_predict=True,
    )

    predicted = to_dense(r2a_predict.X, np.float32)   # scButterfly returns it sparse
    check_full_length(predicted, n_cells, n_adt, "scButterfly")
    print(f"[scbutterfly] trained on {len(fit_rows)} paired cells → translated "
          f"{predicted.shape[0]} cells x {predicted.shape[1]} ADTs")

    diagnostics = {
        "n_fit_cells": int(len(fit_rows)),
        "scbutterfly_embed_dim": embed_dim,
        "scbutterfly_translator_epoch": int(cfg.get("scbutterfly_translator_epoch", 200)),
    }
    return [predicted, targets], None, diagnostics
