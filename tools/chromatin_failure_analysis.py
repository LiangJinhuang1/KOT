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

Everything is reported as a function of the number of target directions the divergence is
measured in, because that is the knob `--align-dims` turns. Empirical OT between two
independent n-samples of the same distribution does not vanish in high d — it decays like
n^(-1/d), which at d=2000 is barely at all — while a point mass at the barycentre sits
closer to the cloud than a genuine second sample does. If that is what is happening, the
oracle beats the constant at low d and loses at high d.

`blur` is an absolute length and the gauge that makes it relative is fitted in the FULL
space, so projecting shrinks the cloud without shrinking the blur. `blur_over_pair_distance`
is that ratio in the space each row was actually measured in: a large value says the
divergence has been smoothed until it cannot tell the three maps apart, which is a
different failure from the concentration one and has the opposite fix.
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
from src.evaluation.chromatin_eval import retrieval_metrics, task_a_state
from src.losses.entropic_ot import squared_distances
from src.losses.sinkhorn import sinkhorn_divergence
from src.training.kot import choose_torch_device


def spread_ratio(predicted: torch.Tensor, observed: torch.Tensor) -> float:
    """Median per-gene spread of the map over that of the target — the collapse number.

    A phi that ignores its input still scores a respectable Sinkhorn loss by sitting on the
    target's mean, and the loss curve looks the same either way. This ratio is what tells
    the two apart.
    """
    return float(predicted.std(dim=0).median() / observed.std(dim=0).median())


def median_pair_distance(cloud: torch.Tensor) -> float:
    """Typical distance between two cells — the length `blur` is a fraction of."""
    return float(squared_distances(cloud, cloud).sqrt().median())


def fraction_of_headroom(scores: dict) -> float:
    """Where the trained map sits between the constant floor and the paired ceiling."""
    span = scores["constant"] - scores["paired_oracle"]
    if abs(span) < 1e-12:
        return float("nan")
    return (scores["constant"] - scores["trained_phi"]) / span


def sample_rows(rows: np.ndarray, n_cells: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(rows, min(n_cells, len(rows)), replace=False)


def held_out_metrics(predicted: np.ndarray, observed: np.ndarray, gene_names: list[str],
                     seed: int) -> dict:
    """Task A and Task B on test cells, where the pairing is revealed after the fact.

    Same functions the `evaluate` stage calls, on a capped sample: the point is to read
    the collapse in the reported metrics without paying for a full evaluation of a run
    that is only being diagnosed.
    """
    summary, _ = task_a_state(predicted, observed, gene_names)
    retrieval = retrieval_metrics(predicted, observed, seed)
    return {
        "n_cells": int(len(predicted)),
        "gene_pearson_median": summary["gene_pearson_median"],
        "gene_spearman_median": summary["gene_spearman_median"],
        "cell_cosine_median": summary["cell_cosine_median"],
        "prediction_spread_ratio": spread_ratio(torch.as_tensor(predicted),
                                                torch.as_tensor(observed)),
        "knn_foscttm": retrieval["foscttm"],
        "knn_foscttm_constant_floor": retrieval["foscttm_constant_floor"],
        "knn_top1": retrieval["top1"],
        "knn_partner_diversity": retrieval["partner_diversity"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="best_align")
    parser.add_argument("--n-cells", type=int, default=2048,
                        help="cells per side; the training loss saw sinkhorn_max_points")
    parser.add_argument("--dims", type=int, nargs="+", default=[0, 512, 128, 32, 8],
                        help="target directions to measure the divergence in; 0 = full space")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--out", default=None, help="JSON path (default: inside the run dir)")
    args = parser.parse_args()

    # The gauge probes 1000 random cells, so without a seed the same checkpoint gives a
    # slightly different align_scale on every call and the runs stop being comparable.
    torch.manual_seed(0)
    device = choose_torch_device({"device": args.device})
    model, payload, config = chromatin.load_checkpoint(Path(args.run_dir), args.checkpoint,
                                                       device)
    adata = chromatin.load_dataset(payload["dataset"])
    splits = pd.read_csv(chromatin.split_path(payload["dataset"], config["split_seed"]),
                         index_col=0)
    side = splits["train_side"].to_numpy()
    split = splits["split"].to_numpy()
    measured = chromatin.target_cell_mask(adata, config["target_layer"])
    atac_rows = np.flatnonzero((split == "train") & (side == "atac") & measured)
    rna_rows = np.flatnonzero((split == "train") & (side == "rna") & measured)
    test_rows = np.flatnonzero((split == "test") & measured)

    activity = np.asarray(adata.obsm["gene_activity"], dtype=np.float32)
    output_columns = adata.var_names.get_indexer(payload["gene_names"])
    assert (output_columns >= 0).all(), "checkpoint output genes do not match dataset"
    gene_mask = np.zeros(adata.n_vars, dtype=bool)
    gene_mask[output_columns] = True
    target = chromatin.build_targets(
        adata, payload["law"], config["target_layer"], gene_mask)
    rng = np.random.default_rng(0)
    a_sample = sample_rows(atac_rows, args.n_cells, rng)
    r_sample = sample_rows(rna_rows, args.n_cells, rng)

    scale = chromatin.block_scales(torch.as_tensor(target[rna_rows], device=device),
                                   payload["law"])
    gauge = chromatin.alignment_gauge(torch.as_tensor(target[rna_rows], device=device), scale)
    align_scale = scale * gauge

    with torch.no_grad():
        predicted = model.phi(torch.as_tensor(activity[a_sample], device=device))
    observed = torch.as_tensor(target[r_sample], device=device)
    paired = torch.as_tensor(target[a_sample], device=device)   # the ATAC cells' OWN RNA
    candidates = {"constant": observed.mean(0, keepdim=True).expand_as(predicted),
                  "trained_phi": predicted, "paired_oracle": paired}

    centred = (observed - observed.mean(0)) / align_scale
    total_variance = float(centred.var(dim=0).sum())
    basis = torch.linalg.svd(centred, full_matrices=False)[2]
    blur = config["sinkhorn_blur"]
    by_dimension = {}
    for n_dims in args.dims:
        project = (lambda block: block) if n_dims == 0 else (
            lambda block, b=basis[:n_dims].T: block @ b)
        reference = project(observed / align_scale)
        scores = {name: float(sinkhorn_divergence(project(block / align_scale), reference,
                                                  blur=blur, backend="tensorized"))
                  for name, block in candidates.items()}
        scores["oracle_beats_constant"] = scores["paired_oracle"] < scores["constant"]
        scores["fraction_of_headroom_captured"] = fraction_of_headroom(scores)
        scores["explained_variance"] = 1.0 if n_dims == 0 else float(
            project(centred).var(dim=0).sum() / total_variance)
        scores["median_pair_distance"] = median_pair_distance(reference)
        scores["blur_over_pair_distance"] = blur / scores["median_pair_distance"]
        by_dimension[n_dims or predicted.shape[1]] = scores

    test_sample = sample_rows(test_rows, args.n_cells, rng)
    with torch.no_grad():
        test_predicted = model.phi(torch.as_tensor(activity[test_sample], device=device))
    results = {
        "run_dir": args.run_dir,
        "checkpoint": args.checkpoint,
        "method": "KOT",
        "dataset": payload["dataset"],
        "law": payload["law"],
        "condition": config["condition"],
        "seed": config["seed"],
        "align_dims_trained": config["align_dims"],
        "sinkhorn_max_points_trained": config["sinkhorn_max_points"],
        "sinkhorn_blur": blur,
        "n_cells_per_side": int(len(a_sample)),
        "phi_output_sd_median": float(predicted.std(0).median()),
        "target_sd_median": float(observed.std(0).median()),
        "sd_ratio_phi_over_target": spread_ratio(predicted, observed),
        "phi_output_mean": float(predicted.mean()),
        "target_mean": float(observed.mean()),
        "by_dimension": by_dimension,
        "held_out": held_out_metrics(
            chromatin.spliced_block(test_predicted.cpu().numpy(), payload["law"]),
            chromatin.spliced_block(target[test_sample], payload["law"]),
            list(payload["gene_names"]), config["seed"]),
    }
    out = Path(args.out) if args.out else Path(args.run_dir) / "failure_analysis.json"
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
