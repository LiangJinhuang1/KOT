"""sciPENN: supervised RNA→protein imputation with a transfer-learning encoder.

Lakkis et al., Nature Machine Intelligence 4, 940–952 (2022).
https://doi.org/10.1038/s42256-022-00545-w

sciPENN is a REFERENCE→QUERY method: it fits on paired CITE-seq cells and imputes
protein for query cells that have RNA only. That is exactly the CRISPR protocol here,
with the NT controls as the reference and every cell as the query, so it needs no
protocol adapter — only the guarantee that its QC never drops a query cell.
"""

from __future__ import annotations

import sys
from pathlib import Path

import anndata as ad
import numpy as np
import torch

# The vendored copy separates the PROTEIN normalisation from the gene normalisation;
# upstream ties them to one pair of flags. See vendor/scipenn/README.md.
VENDOR_SCIPENN_PATH = Path(__file__).resolve().parents[2] / "vendor" / "scipenn"
if VENDOR_SCIPENN_PATH.exists():
    sys.path.insert(0, str(VENDOR_SCIPENN_PATH))

from sciPENN.sciPENN_API import sciPENN_API

from src.training.protein_supervised import (
    adt_targets,
    check_full_length,
    counts_layer,
    fit_rows_of,
)


def counts_adata(counts: np.ndarray, obs_names, var_names) -> ad.AnnData:
    adata = ad.AnnData(X=counts.astype(np.float32))
    adata.obs_names = [str(name) for name in obs_names]
    adata.var_names = [str(name) for name in var_names]
    return adata


def run_scipenn(context: dict, cfg: dict) -> tuple[list, None, dict]:
    """Fit sciPENN on the paired fitted cells; impute protein for every cell.

    QC is disabled (min_cells=min_genes=0) and HVG selection is off. Both would filter,
    and sciPENN's HVG step concatenates the query set into the selection, which would
    mix held-out cells into a decision the reference alone should make. The gene panel
    is already the curated 691-gene retained set, so there is nothing to select.
    """
    rna_adata = context["rna_adata"]
    second_adata = context["second_adata"]
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    rna_counts = counts_layer(rna_adata, "RNA")
    fit_rows = fit_rows_of(context, rna_counts.shape[0])
    protein_names = second_adata.var_names.astype(str)

    reference_rna = counts_adata(rna_counts[fit_rows], rna_adata.obs_names[fit_rows],
                                 rna_adata.var_names)
    # Protein is supplied in the units this benchmark SCORES, and sciPENN is told not to
    # normalise it again. Its own normalize_total + log1p is compositional over the 4-plex
    # panel, which no per-protein rescale can undo, and it cost this model most of its
    # signal (primary Spearman +0.031). The RNA side keeps sciPENN's own normalisation.
    targets = adt_targets(context, cfg)
    reference_protein = counts_adata(targets, rna_adata.obs_names[fit_rows], protein_names)
    query_rna = counts_adata(rna_counts, rna_adata.obs_names, rna_adata.var_names)

    api = sciPENN_API(
        gene_trainsets=[reference_rna],
        protein_trainsets=[reference_protein],
        gene_test=query_rna,
        select_hvg=False,
        min_cells=0,
        min_genes=0,
        batch_size=int(cfg.get("scipenn_batch_size", 128)),
        val_split=float(cfg.get("scipenn_val_split", 0.1)),
        # null in the config means auto-detect, which is what a CPU allocation needs.
        use_gpu=bool(cfg.get("scipenn_use_gpu") if cfg.get("scipenn_use_gpu") is not None
                     else torch.cuda.is_available()),
        protein_normalize=False,
    )
    api.train(
        n_epochs=int(cfg.get("scipenn_epochs", 10000)),
        ES_max=int(cfg.get("scipenn_es_max", 12)),
        lr=float(cfg.get("scipenn_lr", 1e-3)),
        weights_dir=str(context["output_dir"] / "scipenn_weights"),
        # Per-run weights are still shared by every seed of that run dir; loading them
        # would make seed 2 a rerun of seed 1's checkpoint rather than a new fit.
        load=False,
    )
    imputed = api.predict()

    predicted = np.asarray(imputed[:, protein_names].X, dtype=np.float32)
    check_full_length(predicted, rna_counts.shape[0], len(protein_names), "sciPENN")
    print(f"[scipenn] reference {len(fit_rows)} paired cells → imputed "
          f"{predicted.shape[0]} cells x {predicted.shape[1]} ADTs")

    diagnostics = {"n_fit_cells": int(len(fit_rows)),
                   "scipenn_epochs": int(cfg.get("scipenn_epochs", 10000))}
    return [predicted, targets.astype(np.float32)], None, diagnostics
