"""
RegVelo velocity backend for the KOT pipeline.

Drop-in alternative to run_scvelo (src/data/velocity.py): it produces the SAME
output contract — a per-gene `velocity` layer, Ms/Mu moments, and the
velocity_gene_diagnostics table — so everything downstream (gene retention, the
coverage funnel, and KOT's JVP) is unchanged. The only extra input is a prior
gene regulatory network (GRN), built once per dataset by tools/build_grn_prior.py
(DoRothEA to start; an ATAC-derived GRN for BMMC from its multiome companion later).

RegVelo models transcription->splicing as one GRN-coupled ODE and fits every gene
in the network, so the force-retained ADT-target genes get a velocity even where
scVelo's per-gene dynamical model would skip them — the coverage gap we care about.

Confirmed against the container stack (regvelo 0.4.2 / scvi 1.2.0 / torch 2.10) by
reading the source: set_prior_grn(adata, gt_net, keep_dim=True) takes the prior as
rows=targets × cols=regulators and — with keep_dim=True — preserves every gene
(so force-retained ADT genes without a DoRothEA edge survive). We skip sanity_check
(it boolean-indexes the DataFrame skeleton and raises in 0.4.2). Outputs are written
by reg_vae.add_regvelo_outputs_to_adata(adata=adata).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scvelo as scv
import torch
from regvelo import REGVELOVI
from regvelo.preprocessing import set_prior_grn

from src.data.velocity import (
    build_velocity_gene_diagnostics,
    gene_layer_presence,
    has_splicing_layers,
    select_hvgs,
)


def load_grn_prior(path: Path) -> pd.DataFrame:
    """Prior GRN as an edge list with columns tf, target, weight (build_grn_prior.py)."""
    df = pd.read_csv(path)
    missing = {"tf", "target", "weight"} - set(df.columns)
    if missing:
        raise ValueError(f"GRN prior {path} missing columns {sorted(missing)}; expected tf,target,weight.")
    return df


def grn_edges_to_matrix(edges: pd.DataFrame, var_names) -> pd.DataFrame:
    """Square regulator x target skeleton over the genes present in the matrix."""
    genes = [str(g) for g in var_names]
    index = {g: i for i, g in enumerate(genes)}
    W = np.zeros((len(genes), len(genes)), dtype=np.float32)
    for tf, target, weight in zip(edges["tf"], edges["target"], edges["weight"]):
        i = index.get(str(tf))
        j = index.get(str(target))
        if i is not None and j is not None:
            W[i, j] = float(weight)
    return pd.DataFrame(W, index=genes, columns=genes)


def run_regvelo(adata, n_top_genes, hvg_flavor, min_shared_counts, n_pcs, n_neighbors,
                grn_prior_csv, retain_genes=None, batch_size=None):
    if not has_splicing_layers(adata):
        raise ValueError("Missing spliced/unspliced layers; cannot run RegVelo.")
    # Snapshot raw presence before filtering so the retained-gene trace is comparable
    # to the scVelo backend's diagnostics.
    raw_names     = {str(g) for g in adata.var_names}
    raw_spliced   = gene_layer_presence(adata, "spliced")
    raw_unspliced = gene_layer_presence(adata, "unspliced")

    # Same gene set + moments as scVelo: HVG ∪ force-retained ADT targets → Ms/Mu.
    adata = select_hvgs(adata, n_top_genes, hvg_flavor, retain_genes=retain_genes)
    scv.pp.filter_and_normalize(
        adata,
        min_shared_counts=min_shared_counts,
        retain_genes=sorted(retain_genes) if retain_genes else None,
    )
    actual_n_pcs = min(n_pcs, adata.n_obs - 1, adata.n_vars - 1)
    if actual_n_pcs < 1:
        raise ValueError(f"Too few features ({adata.n_vars}) after filtering — cannot run RegVelo.")
    scv.pp.moments(adata, n_pcs=actual_n_pcs, n_neighbors=n_neighbors)

    # Attach the prior GRN. set_prior_grn expects rows=targets, cols=regulators, so
    # transpose (grn_edges_to_matrix builds regulators×targets). keep_dim=True keeps
    # the force-retained ADT genes even when they have no DoRothEA edge — otherwise
    # set_prior_grn prunes edge-less genes and we lose kinetic coverage. It reads
    # adata.layers["Ms"] for its correlation filter, which scv.pp.moments provided.
    prior = grn_edges_to_matrix(load_grn_prior(Path(grn_prior_csv)), adata.var_names)
    adata = set_prior_grn(adata, prior.T, keep_dim=True)
    skeleton = adata.uns["skeleton"]
    regulators = list(adata.uns["regulators"])
    print(f"[regvelo] {adata.n_vars} genes, {int(np.asarray(skeleton).sum())} GRN edges after set_prior_grn")

    # Fit the GRN-coupled velocity ODE. W must be a torch Tensor of shape
    # [n_targets, n_regulators] = skeleton.T (skeleton is regulators×targets).
    # Keep full float32 precision: TF32 ("high") makes RegVelo's variance->sqrt go
    # NaN after many steps (invalid Normal scale), which crashes training.
    W = torch.tensor(np.asarray(skeleton).T.copy(), dtype=torch.float32)
    REGVELOVI.setup_anndata(adata, spliced_layer="Ms", unspliced_layer="Mu")
    reg_vae = REGVELOVI(adata, W=W, regulators=regulators)
    # RegVelo defaults to full-batch (the ODE solver holds a trajectory per cell), which
    # OOMs on large datasets — pass a minibatch size for those (e.g. BMMC's 90k cells).
    train_kwargs = {"batch_size": int(batch_size)} if batch_size else {}
    reg_vae.train(**train_kwargs)
    adata = reg_vae.add_regvelo_outputs_to_adata(adata=adata)

    # Velocity graph + pseudotime from the RegVelo velocity, so the shared
    # save_plots/save_confidence and KOT's pseudotime gauge behave like the scVelo path.
    scv.tl.velocity_graph(adata)
    scv.tl.velocity_confidence(adata)
    scv.tl.velocity_pseudotime(adata)

    # velocity_genes = genes RegVelo modelled with a finite, non-zero velocity, so the
    # coverage funnel's velocity_gene column stays meaningful across backends.
    V = np.asarray(adata.layers["velocity"])
    adata.var["velocity_genes"] = np.isfinite(V).any(axis=0) & (np.nan_to_num(V) != 0).any(axis=0)

    if retain_genes:
        diag = build_velocity_gene_diagnostics(retain_genes, raw_names, raw_spliced, raw_unspliced, adata)
        adata.uns["velocity_gene_diagnostics"] = pd.DataFrame(diag)
        n_elig = int(sum(r["final_kinetics_eligible"] for r in diag))
        print(f"[regvelo] retained-gene trace: {n_elig}/{len(diag)} finally kinetics-eligible")
    return adata
