"""RegVelo trained on NT only, predicting Papalexi TF regulon knockouts.

Its published regulon-KO definition silences a regulator's outgoing TF edges, so a
knockout with no learned outgoing edges cannot be represented and is written to
coverage.csv rather than scored. Counterfactual RNA is converted back into frozen KOT Ms
coordinates for `run_kot_crispr.py task_b`.
"""
from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from regvelo import REGVELOVI
from regvelo.preprocessing import preprocess_data, set_prior_grn

from run_kot_crispr import MIN_KO_CELLS, MIN_REPLICATES, RNA_H5AD
from src.data.adt_gene_map import (
    gene_alias_lookup,
    iter_gene_aliases,
    normalize_gene_symbol,
)
from src.data.regvelo_backend import grn_edges_to_matrix, load_grn_prior
from src.data.velocity import preprocess_for_velocity
from src.evaluation.isp.common import profile_row, usable_perturbations, write_isp_outputs
from src.evaluation.isp.frozen import load_nt_only_run
from src.evaluation.regvelo_isp import (
    counterfactual_query,
    positive_feature_scales,
    silence_outgoing,
)
from src.utils.arrays import to_dense

CONTROL_LABEL = "NT"


def add_arguments(parser) -> None:
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="frozen NT-only KOT run; defines the RNA feature space")
    parser.add_argument("--grn-prior", type=Path,
                        default=Path("cache/results/grn/grn_prior_pbmc.csv"))
    parser.add_argument("--raw-rna", type=Path, default=RNA_H5AD)
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--cutoff", type=float, default=1e-3)


def fitted_spliced(model, adata, seed: int, n_samples: int, batch_size: int) -> np.ndarray:
    """RegVelo's fitted spliced matrix, reseeded so a refit is comparable to the last."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    fitted, _ = model.rgv_expression_fit(
        adata=adata, n_samples=n_samples, batch_size=batch_size,
        return_mean=True, return_numpy=True)
    return np.asarray(fitted, dtype=np.float32)


def set_regulator_weights(model, weights: np.ndarray) -> None:
    """Overwrite the velocity encoder's regulator weights in place."""
    target = model.module.v_encoder.fc1.weight
    with torch.no_grad():
        target.copy_(torch.as_tensor(weights, device=model.device, dtype=target.dtype))


def kot_to_regvelo_indices(frozen_nt, model_adata) -> np.ndarray:
    """Position of every frozen KOT feature inside RegVelo's own gene axis, by alias."""
    lookup = gene_alias_lookup(model_adata.var_names, model_adata.var,
                               normalizer=normalize_gene_symbol)
    aliases = {index: [] for index in range(frozen_nt.n_vars)}
    for index, alias in iter_gene_aliases(frozen_nt.var_names, frozen_nt.var):
        aliases[index].append(normalize_gene_symbol(alias))
    indices = []
    for feature in range(frozen_nt.n_vars):
        match = next((lookup[a] for a in aliases[feature] if a in lookup), None)
        if match is None:
            raise ValueError(
                f"RegVelo preprocessing lost KOT feature {frozen_nt.var_names[feature]}")
        indices.append(match)
    return np.asarray(indices)


def run(args) -> None:
    if args.max_epochs < 1 or args.batch_size < 1 or args.n_samples < 1:
        raise SystemExit("Epochs, batch size, and samples must be positive")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rna = load_nt_only_run(args.run_dir).rna
    groups = rna.obs["gene"].astype(str).to_numpy()
    reps = rna.obs["replicate"].astype(str).to_numpy()
    frozen_nt = rna[groups == CONTROL_LABEL].copy()
    if "Ms" not in frozen_nt.layers or "Mu" not in frozen_nt.layers:
        raise ValueError("Frozen KOT data must contain Ms and Mu")
    observed_nt = to_dense(frozen_nt.layers["Ms"], dtype=np.float32)
    usable = usable_perturbations(groups, reps, MIN_KO_CELLS, MIN_REPLICATES, CONTROL_LABEL)

    # Build the ISP input from raw NT cells so every eligible knockout can be retained
    # as a regulator even when KOT's HVG selection omitted it. No KO cell is loaded into
    # the RegVelo fit or any statistic used by its preprocessing.
    raw = ad.read_h5ad(args.raw_rna)
    raw.obs_names = raw.obs_names.astype(str)
    missing_cells = frozen_nt.obs_names.difference(raw.obs_names)
    if len(missing_cells):
        raise ValueError(f"Raw RNA lacks {len(missing_cells)} frozen NT cells")
    raw_nt = raw[frozen_nt.obs_names].copy()
    del raw
    retain = set(usable)
    for _, alias in iter_gene_aliases(frozen_nt.var_names, frozen_nt.var):
        retain.add(alias)
    nt, _ = preprocess_for_velocity(
        raw_nt, n_top_genes=2000, hvg_flavor="seurat_v3",
        min_shared_counts=20, n_pcs=30, n_neighbors=30, retain_genes=retain)
    model_observed = to_dense(nt.layers["Ms"], dtype=np.float32)
    ranges = model_observed.max(axis=0) - model_observed.min(axis=0)
    ranges[ranges == 0] = 1.0

    # Map all frozen KOT features to RegVelo genes by aliases, then fit a positive
    # unit conversion using NT cells only. Deltas carry no intercept.
    model_indices = kot_to_regvelo_indices(frozen_nt, nt)
    feature_scales = positive_feature_scales(model_observed[:, model_indices], observed_nt)

    prior = grn_edges_to_matrix(load_grn_prior(args.grn_prior), nt.var_names, nt.var)
    # set_prior_grn expects targets x regulators. Correlation filtering sees NT only.
    nt = set_prior_grn(nt, prior.T, keep_dim=True, cor_filter=True)
    nt = preprocess_data(nt, spliced_layer="Ms", unspliced_layer="Mu",
                         min_max_scale=True, filter_on_r2=False)
    skeleton = np.asarray(nt.uns["skeleton"])
    if not skeleton.any():
        raise ValueError("No GRN edges survive NT-only matching/correlation filtering")
    REGVELOVI.setup_anndata(nt, spliced_layer="Ms", unspliced_layer="Mu")
    model = REGVELOVI(nt, W=torch.tensor(skeleton.T.copy(), dtype=torch.float32),
                      regulators=list(nt.uns["regulators"]))
    model.train(max_epochs=args.max_epochs, batch_size=args.batch_size, early_stopping=True)
    model.save(args.out_dir / "model", overwrite=True, save_anndata=True)

    baseline = fitted_spliced(model, nt, args.seed, args.n_samples, args.batch_size)
    original_weights = model.module.v_encoder.fc1.weight.detach().cpu().numpy().copy()
    lookup = gene_alias_lookup(nt.var_names, nt.var, normalizer=normalize_gene_symbol)
    profiles, coverage = [], []
    for target in usable:
        index = lookup.get(normalize_gene_symbol(target))
        record = {"perturbation": target, "supported": False,
                  "reason": "gene_absent", "n_outgoing_edges": 0}
        if index is None:
            coverage.append(record)
            continue
        perturbed_weights, active = silence_outgoing(original_weights, index, args.cutoff)
        record["n_outgoing_edges"] = int(active.sum())
        if not active.any():
            record["reason"] = "no_learned_outgoing_edges"
            coverage.append(record)
            continue
        set_regulator_weights(model, perturbed_weights)
        try:
            perturbed = fitted_spliced(model, nt, args.seed, args.n_samples, args.batch_size)
        finally:
            set_regulator_weights(model, original_weights)
        query, clipped = counterfactual_query(
            observed_nt, baseline, perturbed, ranges, model_indices, feature_scales)
        profiles.append(profile_row(target, query.mean(axis=0)))
        record.update(supported=True, reason="ok", clipped_fraction=float(clipped.mean()))
        coverage.append(record)

    manifest = {
        "model": "RegVelo regulon knockout", "training_cells": "NT only",
        "n_training_cells": int(nt.n_obs), "seed": args.seed,
        "max_epochs": args.max_epochs, "batch_size": args.batch_size,
        "n_samples": args.n_samples, "cutoff": args.cutoff,
        "grn_prior": str(args.grn_prior.resolve()),
        "input_coordinates": "frozen KOT Ms",
        "counterfactual": "observed_NT + unscaled(fit_KO - fit_unperturbed)",
        "unit_conversion": "positive per-feature SD ratio fitted on NT only",
        "intervention": "zero learned outgoing TF weights above cutoff",
        "supported_perturbations": int(sum(r["supported"] for r in coverage)),
        "requested_perturbations": len(coverage),
    }
    write_isp_outputs(args.out_dir, profiles, coverage, manifest, "RegVelo")
    print(pd.DataFrame(coverage).to_string(index=False))
