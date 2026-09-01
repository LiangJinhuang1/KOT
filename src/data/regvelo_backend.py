"""RegVelo velocity backend: same contract as run_scvelo (velocity layer, Ms/Mu,
velocity_gene_diagnostics). Extra input is a GRN prior from
tools/build_inputs.py grn-prior. For retained ADT-target runs use keep_dim=True
and filter_on_r2=False so those genes stay available for KOT.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import scvelo as scv
import torch
from regvelo import REGVELOVI
from regvelo.preprocessing import preprocess_data, set_prior_grn
from scipy import sparse

from src.data.adt_gene_map import gene_alias_lookup, gene_alias_set, normalize_gene_symbol
from src.data.velocity import (
    build_velocity_gene_diagnostics,
    gene_layer_presence,
    has_splicing_layers,
    preprocess_for_velocity,
)


def load_grn_prior(path: Path) -> pd.DataFrame:
    """Prior GRN as an edge list with columns tf, target, weight (build_grn_prior.py)."""
    df = pd.read_csv(path)
    missing = {"tf", "target", "weight"} - set(df.columns)
    if missing:
        raise ValueError(f"GRN prior {path} missing columns {sorted(missing)}; expected tf,target,weight.")
    return df


def grn_edges_to_matrix(edges: pd.DataFrame, var_names, var=None) -> pd.DataFrame:
    """Square regulator x target skeleton over the genes present in the matrix."""
    genes = [str(g) for g in var_names]
    index = gene_alias_lookup(var_names, var, normalizer=normalize_gene_symbol)
    W = np.zeros((len(genes), len(genes)), dtype=np.float32)
    for tf, target, weight in zip(edges["tf"], edges["target"], edges["weight"]):
        i = index.get(normalize_gene_symbol(str(tf)))
        j = index.get(normalize_gene_symbol(str(target)))
        if i is not None and j is not None:
            W[i, j] = float(weight)
    return pd.DataFrame(W, index=genes, columns=genes)


def regvelo_log(message: str) -> None:
    print(f"[regvelo] {message}", flush=True)


def run_regvelo(
    adata,
    n_top_genes,
    hvg_flavor,
    min_shared_counts,
    n_pcs,
    n_neighbors,
    grn_prior_csv,
    retain_genes=None,
    batch_size=None,
    max_epochs=None,
):
    if not has_splicing_layers(adata):
        raise ValueError("Missing spliced/unspliced layers; cannot run RegVelo.")
    # Snapshot raw presence before filtering so the retained-gene trace is comparable
    # to the scVelo backend's diagnostics.
    raw_names     = gene_alias_set(adata.var_names, adata.var, normalizer=normalize_gene_symbol)
    raw_spliced   = gene_layer_presence(adata, "spliced")
    raw_unspliced = gene_layer_presence(adata, "unspliced")

    # Identical upstream preprocessing to run_scvelo (HVG ∪ retain → filter/normalize →
    # PCA → neighbours → moments), so the scVelo-vs-RegVelo comparison isolates the
    # velocity model. RegVelo's own min-max step (preprocess_data) runs on top below.
    if int(n_top_genes) < 1:
        raise ValueError("RegVelo preprocessing requires n_top_genes >= 1.")
    adata, _ = preprocess_for_velocity(
        adata, int(n_top_genes), hvg_flavor, min_shared_counts, n_pcs, n_neighbors,
        retain_genes=retain_genes,
    )
    if adata.n_vars < 1:
        raise ValueError("RegVelo preprocessing removed all genes; cannot run RegVelo.")
    regvelo_log(f"shared velocity preprocessing kept {adata.n_vars} genes")

    # Attach the prior GRN. set_prior_grn expects rows=targets, cols=regulators, so
    # transpose (grn_edges_to_matrix builds regulators×targets). keep_dim=True
    # matches the docs note for retaining all preprocessed genes for velocity.
    prior = grn_edges_to_matrix(load_grn_prior(Path(grn_prior_csv)), adata.var_names, adata.var)
    adata = set_prior_grn(adata, prior.T, keep_dim=True)

    n_before_regvelo_scale = adata.n_vars
    adata = preprocess_data(adata, filter_on_r2=False)
    if adata.n_vars < 1:
        raise ValueError("RegVelo preprocess_data removed all genes; cannot run RegVelo.")
    skeleton = adata.uns["skeleton"]
    regulators = list(adata.uns["regulators"])
    n_edges = int(np.count_nonzero(np.asarray(skeleton)))
    if n_edges == 0:
        raise ValueError(
            "RegVelo prior GRN has 0 edges after matching to RNA genes; check "
            "GRN gene symbols and AnnData var aliases."
        )
    regvelo_log(
        f"preprocess_data filter_on_r2=False retained "
        f"{adata.n_vars}/{n_before_regvelo_scale} genes"
    )
    regvelo_log(f"{adata.n_vars} genes, {n_edges} GRN edges after set_prior_grn")

    # Fit the GRN-coupled velocity ODE. W must be a torch Tensor of shape
    # [n_targets, n_regulators] = skeleton.T (skeleton is regulators×targets).
    # Keep full float32 precision: TF32 ("high") makes RegVelo's variance->sqrt go
    # NaN after many steps (invalid Normal scale), which crashes training.
    regvelo_log(
        "pre-train state: "
        f"adata={adata.n_obs} cells x {adata.n_vars} genes, "
        f"skeleton_shape={tuple(np.asarray(skeleton).shape)}, "
        f"regulators={len(regulators)}, edges={n_edges}, "
        f"batch_size={batch_size or 'default'}, max_epochs={max_epochs or 'default'}, "
        f"torch={torch.__version__}, torch_cuda={torch.version.cuda}, "
        f"cuda_env={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}"
    )
    regvelo_log("building W tensor")
    W = torch.tensor(np.asarray(skeleton).T.copy(), dtype=torch.float32)
    regvelo_log(f"W tensor ready: shape={tuple(W.shape)}, dtype={W.dtype}, finite={bool(torch.isfinite(W).all())}")
    regvelo_log("calling REGVELOVI.setup_anndata")
    REGVELOVI.setup_anndata(adata, spliced_layer="Ms", unspliced_layer="Mu")
    regvelo_log("constructing REGVELOVI model")
    reg_vae = REGVELOVI(adata, W=W, regulators=regulators)
    # RegVelo defaults to full-batch (the ODE solver holds a trajectory per cell), which
    # OOMs on large datasets — pass a minibatch size for those.
    train_kwargs = {}
    if batch_size:
        train_kwargs["batch_size"] = int(batch_size)
    if max_epochs:
        train_kwargs["max_epochs"] = int(max_epochs)
    regvelo_log(f"starting REGVELOVI.train with kwargs={train_kwargs}")
    reg_vae.train(**train_kwargs)
    regvelo_log("REGVELOVI.train finished")
    regvelo_log("adding RegVelo outputs to AnnData")
    output_kwargs = {}
    if batch_size:
        output_kwargs["batch_size"] = int(batch_size)
    adata = reg_vae.add_regvelo_outputs_to_adata(adata=adata, **output_kwargs)
    adata.uns["velocity_backend"] = "regvelo"

    # Velocity graph + pseudotime from the RegVelo velocity, so the shared
    # save_plots/save_confidence and KOT's pseudotime gauge behave like the scVelo path.
    scv.tl.velocity_graph(adata)
    scv.tl.velocity_confidence(adata)
    scv.tl.velocity_pseudotime(adata)

    # velocity_genes = genes RegVelo modelled with a finite, non-zero velocity, so the
    # coverage funnel's velocity_gene column stays meaningful across backends.
    velocity_layer = adata.layers["velocity"]
    V = velocity_layer.toarray() if sparse.issparse(velocity_layer) else np.asarray(velocity_layer)
    adata.var["velocity_genes"] = np.isfinite(V).any(axis=0) & (np.nan_to_num(V) != 0).any(axis=0)

    if retain_genes:
        diag = build_velocity_gene_diagnostics(retain_genes, raw_names, raw_spliced, raw_unspliced, adata)
        adata.uns["velocity_gene_diagnostics"] = pd.DataFrame(diag)
        n_elig = int(sum(r["final_kinetics_eligible"] for r in diag))
        print(f"[regvelo] retained-gene trace: {n_elig}/{len(diag)} finally kinetics-eligible")
    return adata
