#!/usr/bin/env python3
r"""How much of the kinetic residual does the velocity actually account for?

The R1 sweep found `zero` scoring as well as `full`, which would mean the velocity's
direction contributes nothing. There are two very different explanations: the map really
is insensitive to chromatin dynamics, or the push-forward term is numerically negligible
next to the state term and the residual is satisfied without it either way.

    residual = J_phi(c) v_c  -  kappa(c)[alpha(c) (*) Gc - gamma (*) phi(c)]
               \___________/    \_________________________________________/
                velocity-dependent            velocity-FREE

Both sides are measured here, masked and scaled exactly as `kinetics_loss` does them, so
the ratio says which of the two explanations holds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import run_kot_chromatin as chromatin
from src.losses.chromatin_laws import law_mask, law_rhs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="best_align")
    parser.add_argument("--n-cells", type=int, default=2000)
    args = parser.parse_args()

    device = torch.device("cpu")
    model, payload, config = chromatin.load_checkpoint(Path(args.run_dir), args.checkpoint,
                                                       device)
    adata = chromatin.load_dataset(payload["dataset"])
    fields = chromatin.load_velocity(payload["dataset"], config["split_seed"])
    splits = pd.read_csv(
        chromatin.split_path(payload["dataset"], config["split_seed"]), index_col=0)
    full_mapping, _, full_kinetic_mask = chromatin.gene_map(
        adata, chromatin.gene_map_path(payload["dataset"]))
    output_columns = adata.var_names.get_indexer(payload["gene_names"])
    assert (output_columns >= 0).all(), "checkpoint output genes do not match dataset"
    mapping = full_mapping[output_columns].tocsr()
    kinetic_mask = full_kinetic_mask[output_columns]

    activity = np.asarray(adata.obsm["gene_activity"], dtype=np.float32)
    split_column = splits["split"].to_numpy()
    side_column = splits["train_side"].to_numpy()
    train_source = (split_column == "train") & (side_column == "atac")
    eligible = train_source & (np.asarray(fields["norm"]) > 0)
    velocity_values, _, dynamic = chromatin.apply_velocity_condition(
        fields["velocity"], fields["confidence"], fields["dynamic_mask"],
        config["condition"], config["seed"], eligible=eligible)
    if config["condition"] == "permG":
        mapping = chromatin.permute_chromatin_projection(mapping, config["seed"])

    keep = np.ones(adata.n_obs, dtype=bool)
    if config.get("subsample") is not None:
        selected = np.random.default_rng(config["seed"]).choice(
            adata.n_obs, config["subsample"], replace=False)
        keep[:] = False
        keep[selected] = True
    moving = np.flatnonzero(train_source & keep & dynamic)
    rows = np.sort(np.random.default_rng(0).choice(
        moving, min(args.n_cells, len(moving)), replace=False))

    chrom = torch.as_tensor(activity[rows], device=device)
    velocity = torch.as_tensor(velocity_values[rows].astype(np.float32), device=device)
    production = torch.as_tensor((activity[rows] @ mapping.toarray().T).astype(np.float32),
                                 device=device)
    measured = chromatin.target_cell_mask(adata, config["target_layer"])
    rna_rows = np.flatnonzero(
        keep & measured & (split_column == "train") & (side_column == "rna"))
    gene_mask = np.zeros(adata.n_vars, dtype=bool)
    gene_mask[output_columns] = True
    target = chromatin.build_targets(
        adata, payload["law"], config["target_layer"], gene_mask)
    scale = chromatin.block_scales(
        torch.as_tensor(target[rna_rows]), payload["law"])
    mask = torch.as_tensor(law_mask(kinetic_mask, payload["law"]))

    prediction, pushforward = torch.func.jvp(model.phi, (chrom,), (velocity,))
    rhs = law_rhs(model, payload["law"], prediction, chrom, production)
    scaled = lambda block: ((block / scale) * mask).norm(dim=1).median().item()

    jvp_norm, rhs_norm = scaled(pushforward), scaled(rhs)
    result = {
        "run_dir": args.run_dir, "dataset": payload["dataset"],
        "condition": config["condition"], "n_cells": int(len(rows)),
        "jvp_norm_median": jvp_norm, "rhs_norm_median": rhs_norm,
        "velocity_share_of_residual": jvp_norm / (jvp_norm + rhs_norm),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
