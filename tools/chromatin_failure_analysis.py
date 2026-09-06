#!/usr/bin/env python3
"""Why does phi collapse? Separate "optimiser failed" from "the objective prefers this".

Three questions, in the order that decides what to fix:

 1. SCALE     Does phi's output span the target's range, or is it squashed toward the mean?
              Spectral norm caps each layer's gain, so a low-magnitude input may simply be
              unable to reach the target scale.

 2. OBJECTIVE Is the collapsed map a BAD point of the Sinkhorn loss, or a good one? If the
              trained phi scores close to what a constant scores, the optimiser is at a
              minimum the objective genuinely likes, and no amount of tuning fixes it —
              the loss does not identify the map. If it scores far worse than a paired
              oracle, the objective is fine and the optimisation is the problem.

 3. HEADROOM  What does the loss look like for a map that IS right (the true paired RNA)?
              That is the value a working run should approach.
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
from src.losses.sinkhorn import sinkhorn_divergence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--n-cells", type=int, default=2048)
    parser.add_argument("--dims", type=int, nargs="+", default=[0, 512, 128, 32, 8],
                        help="target directions to measure the divergence in; 0 = full space")
    args = parser.parse_args()

    device = torch.device("cpu")
    model, payload, config = chromatin.load_checkpoint(Path(args.run_dir), "best_align", device)
    adata = chromatin.load_dataset(payload["dataset"])
    splits = pd.read_csv(chromatin.split_path(payload["dataset"], config["split_seed"]),
                         index_col=0)
    side = splits["train_side"].to_numpy()
    is_train = splits["split"].to_numpy() == "train"
    measured = chromatin.target_cell_mask(adata, config["target_layer"])
    atac_rows = np.flatnonzero(is_train & (side == "atac") & measured)
    rna_rows = np.flatnonzero(is_train & (side == "rna") & measured)

    activity = np.asarray(adata.obsm["gene_activity"], dtype=np.float32)
    output_columns = adata.var_names.get_indexer(payload["gene_names"])
    assert (output_columns >= 0).all(), "checkpoint output genes do not match dataset"
    gene_mask = np.zeros(adata.n_vars, dtype=bool)
    gene_mask[output_columns] = True
    target = chromatin.build_targets(
        adata, payload["law"], config["target_layer"], gene_mask)
    rng = np.random.default_rng(0)
    a_sample = rng.choice(atac_rows, min(args.n_cells, len(atac_rows)), replace=False)
    r_sample = rng.choice(rna_rows, min(args.n_cells, len(rna_rows)), replace=False)

    scale = chromatin.block_scales(torch.as_tensor(target[rna_rows]), payload["law"])
    gauge = chromatin.alignment_gauge(torch.as_tensor(target[rna_rows]), scale)
    align_scale = scale * gauge

    with torch.no_grad():
        predicted = model.phi(torch.as_tensor(activity[a_sample], device=device))
    observed = torch.as_tensor(target[r_sample], device=device)
    paired = torch.as_tensor(target[a_sample], device=device)   # the ATAC cells' OWN RNA

    def loss(block):
        return float(sinkhorn_divergence(block / align_scale, observed / align_scale,
                                         blur=config["sinkhorn_blur"], backend="tensorized"))

    constant = observed.mean(0, keepdim=True).expand_as(predicted)

    # The ordering as a function of DIMENSION. Empirical OT between two independent
    # n-samples of the same distribution does not vanish in high d — it decays like
    # n^(-1/d), which at d=2000 is barely at all — while a point mass at the barycentre
    # sits closer to the cloud than a genuine second sample does. If that is what is
    # happening, the oracle beats the constant at low d and loses at high d.
    basis = torch.linalg.svd(
        (observed - observed.mean(0)) / align_scale, full_matrices=False)[2]
    by_dimension = {}
    for n_dims in args.dims:
        project = (lambda block: block) if n_dims == 0 else (
            lambda block, b=basis[:n_dims].T: block @ b)
        scores = {name: float(sinkhorn_divergence(
                      project(block / align_scale), project(observed / align_scale),
                      blur=config["sinkhorn_blur"], backend="tensorized"))
                  for name, block in [("constant", constant), ("trained_phi", predicted),
                                      ("paired_oracle", paired)]}
        scores["oracle_beats_constant"] = scores["paired_oracle"] < scores["constant"]
        by_dimension[n_dims or predicted.shape[1]] = scores

    results = {
        "run_dir": args.run_dir,
        "by_dimension": by_dimension,
        "phi_output_sd_median": float(predicted.std(0).median()),
        "target_sd_median": float(observed.std(0).median()),
        "sd_ratio_phi_over_target": float(predicted.std(0).median() / observed.std(0).median()),
        "phi_output_mean": float(predicted.mean()),
        "target_mean": float(observed.mean()),
        "sinkhorn_trained_phi": loss(predicted),
        "sinkhorn_constant_map": loss(constant),
        "sinkhorn_paired_oracle": loss(paired),
    }
    span = results["sinkhorn_constant_map"] - results["sinkhorn_paired_oracle"]
    results["fraction_of_headroom_captured"] = (
        (results["sinkhorn_constant_map"] - results["sinkhorn_trained_phi"]) / span
        if abs(span) > 1e-12 else float("nan"))
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
