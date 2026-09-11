#!/usr/bin/env python3
"""Why does phi collapse? Separate optimiser failure from an objective that prefers collapse.

Five unpaired clouds, all scored the way training's Sinkhorn sees them (ŝ against s
under R2):

  constant            RNA-train mean, expanded
  affine_init         unpaired per-gene location/scale of Gc, before the MLP residual
  trained_phi         the checkpoint
  paired_oracle       each ATAC cell's own RNA (pairing exists, the plan is not shown it)
  permuted_oracle     the same RNA vectors, randomly reassigned

Unpaired Sinkhorn cannot tell paired_oracle from permuted_oracle: they are the same
marginal. Paired FOSCTTM on held-out cells can. That is the empirical ambiguity.

Reported as a function of `--align-dims`, because empirical OT between independent samples
of the same distribution does not vanish in high dimension while a point mass at the
barycentre can sit closer than a genuine second sample.

`blur` is an absolute length; `blur_over_pair_distance` is that length in the space each
row was measured in.
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
from src.data.chromatin import PREPROCESSING_PROTOCOL, chromatin_features
from src.evaluation.chromatin_eval import retrieval_metrics, task_a_state
from src.losses.chromatin_laws import REDUCED, RELAY
from src.losses.entropic_ot import squared_distances
from src.losses.sinkhorn import sinkhorn_divergence
from src.training.kot import choose_torch_device

CANDIDATE_NAMES = (
    "constant", "affine_init", "trained_phi", "paired_oracle", "permuted_oracle",
)


def spread_ratio(predicted: torch.Tensor, observed: torch.Tensor) -> float:
    """Median per-gene spread of the map over that of the target — the collapse number.

    A phi that ignores its input still scores a respectable Sinkhorn loss by sitting on the target's mean.
    """
    return float(predicted.std(dim=0).median() / observed.std(dim=0).median())


def median_pair_distance(cloud: torch.Tensor) -> float:
    """Typical distance between two cells — the length `blur` is a fraction of."""
    return float(squared_distances(cloud, cloud).sqrt().median())


def fraction_of_headroom(scores: dict, name: str = "trained_phi") -> float:
    """Where a map sits between the constant floor and the paired ceiling."""
    span = scores["constant"] - scores["paired_oracle"]
    if abs(span) < 1e-12:
        return float("nan")
    return (scores["constant"] - scores[name]) / span


def sample_rows(rows: np.ndarray, n_cells: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(rows, min(n_cells, len(rows)), replace=False)


def permute_rows(values, rng: np.random.Generator):
    """The same empirical measure, with cell correspondence destroyed.

    Unpaired Sinkhorn cannot see this; paired FOSCTTM can. That is the OT ambiguity.
    """
    order = rng.permutation(len(values))
    if torch.is_tensor(values):
        return values[torch.as_tensor(order, device=values.device)]
    return values[order]


def alignment_block(values, law: str):
    """The coordinates training's Sinkhorn is measured on: spliced RNA under R2."""
    if law == RELAY:
        return chromatin.spliced_block(values, law)
    return values


def gene_projection_for_affine(model, n_genes: int,
                               fallback: torch.Tensor | None) -> torch.Tensor:
    """G in gene × feature coordinates. Relay stores [G; G] on phi; diagnostics need G."""
    if hasattr(model.phi, "gene_projection"):
        projection = model.phi.gene_projection.detach()
        if projection.shape[0] == 2 * n_genes:
            return projection[:n_genes]
        if projection.shape[0] == n_genes:
            return projection
        raise ValueError(
            f"phi.gene_projection has {projection.shape[0]} rows; expected {n_genes} or "
            f"{2 * n_genes}")
    if fallback is None:
        raise ValueError(
            "affine initialization needs G from the checkpoint or the gene-map CSV")
    return fallback


def apply_affine(production: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor,
                 rows: np.ndarray, law: str) -> torch.Tensor:
    source = production[torch.as_tensor(rows, device=production.device)]
    if law == RELAY:
        source = torch.cat([source, source], dim=1)
    return source * scale + bias


def unpaired_alignment_candidates(predicted, observed, paired, rng, affine,
                                  law: str = RELAY) -> dict:
    """The five unpaired clouds, already restricted to the coordinates Sinkhorn sees.

    permuted_oracle is a row shuffle of paired_oracle: same RNA measure, destroyed
    correspondence.
    """
    raw = {
        "constant": observed.mean(0, keepdim=True).expand_as(predicted),
        "affine_init": affine,
        "trained_phi": predicted,
        "paired_oracle": paired,
        "permuted_oracle": permute_rows(paired, rng),
    }
    missing = [name for name in CANDIDATE_NAMES if raw[name] is None]
    if missing:
        raise ValueError(f"failure analysis is missing candidates {missing}")
    return {name: alignment_block(raw[name], law) for name in CANDIDATE_NAMES}


def held_out_metrics(predicted: np.ndarray, observed: np.ndarray, gene_names: list[str],
                     seed: int) -> dict:
    """Task A and retrieval where the pairing is revealed after the fact."""
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
    parser.add_argument("--eval-split", choices=["val", "test"], default="val",
                        help="paired cells for held-out FOSCTTM; val during development")
    parser.add_argument("--n-cells", type=int, default=2048,
                        help="cells per side; the training loss saw sinkhorn_max_points")
    parser.add_argument("--dims", type=int, nargs="+", default=[0, 512, 128, 32, 8],
                        help="target directions to measure the divergence in; 0 = full space")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--out", default=None, help="JSON path (default: inside the run dir)")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = choose_torch_device({"device": args.device})
    model, payload, config = chromatin.load_checkpoint(Path(args.run_dir), args.checkpoint,
                                                       device)
    prepared_seed = (config["split_seed"]
                     if config.get("preprocessing_protocol") == PREPROCESSING_PROTOCOL else None)
    adata = chromatin.load_dataset(payload["dataset"], prepared_seed)
    splits = pd.read_csv(chromatin.split_path(payload["dataset"], config["split_seed"]),
                         index_col=0)
    side = splits["train_side"].to_numpy()
    split = splits["split"].to_numpy()
    measured = chromatin.target_cell_mask(adata, config["target_layer"])
    atac_rows = np.flatnonzero((split == "train") & (side == "atac") & measured)
    rna_rows = np.flatnonzero((split == "train") & (side == "rna") & measured)
    held_rows = chromatin.evaluation_split_rows(split, measured, args.eval_split)

    transform = config.get("chromatin_transform", "as_is")
    features = chromatin_features(adata, transform)
    output_columns = adata.var_names.get_indexer(payload["gene_names"])
    assert (output_columns >= 0).all(), "checkpoint output genes do not match dataset"
    gene_mask = np.zeros(adata.n_vars, dtype=bool)
    gene_mask[output_columns] = True
    law = payload["law"]
    target = chromatin.build_targets(adata, law, config["target_layer"], gene_mask)
    rng = np.random.default_rng(0)
    a_sample = sample_rows(atac_rows, args.n_cells, rng)
    r_sample = sample_rows(rna_rows, args.n_cells, rng)

    target_t = torch.as_tensor(target, device=device)
    atac_t = torch.as_tensor(atac_rows, device=device)
    rna_t = torch.as_tensor(rna_rows, device=device)
    features_t = torch.as_tensor(features, device=device)
    n_genes = len(payload["gene_names"])
    mapping, _, _ = chromatin.gene_map(
        adata, chromatin.gene_map_path(payload["dataset"], prepared_seed))
    fallback_g = torch.as_tensor(mapping[gene_mask].toarray(), dtype=torch.float32,
                                 device=device)
    gene_proj = gene_projection_for_affine(model, n_genes, fallback_g)
    production = features_t @ gene_proj.T
    affine_scale, affine_bias = chromatin.gene_affine_calibration(
        production, target_t, atac_t, rna_t, law)

    with torch.no_grad():
        predicted = model.phi(features_t[a_sample])
    observed = target_t[r_sample]
    paired = target_t[a_sample]
    candidates = unpaired_alignment_candidates(
        predicted, observed, paired, rng,
        affine=apply_affine(production, affine_scale, affine_bias, a_sample, law),
        law=law)

    observed_s = alignment_block(observed, law)
    fit_s = alignment_block(target_t[rna_t], law)
    scale = chromatin.block_scales(fit_s, REDUCED)
    gauge = chromatin.alignment_gauge(fit_s, scale)
    align_scale = scale * gauge

    centred = (observed_s - observed_s.mean(0)) / align_scale
    total_variance = float(centred.var(dim=0).sum())
    basis = torch.linalg.svd(centred, full_matrices=False)[2]
    blur = config["sinkhorn_blur"]
    by_dimension = {}
    for n_dims in args.dims:
        project = (lambda block: block) if n_dims == 0 else (
            lambda block, b=basis[:n_dims].T: block @ b)
        reference = project(observed_s / align_scale)
        scores = {name: float(sinkhorn_divergence(project(block / align_scale), reference,
                                                  blur=blur, backend="tensorized"))
                  for name, block in candidates.items()}
        scores["oracle_beats_constant"] = scores["paired_oracle"] < scores["constant"]
        scores["permuted_oracle_relative_gap"] = (
            abs(scores["permuted_oracle"] - scores["paired_oracle"])
            / max(abs(scores["paired_oracle"]), 1e-12))
        scores["fraction_of_headroom_captured"] = fraction_of_headroom(scores)
        scores["affine_fraction_of_headroom"] = fraction_of_headroom(scores, "affine_init")
        scores["explained_variance"] = 1.0 if n_dims == 0 else float(
            project(centred).var(dim=0).sum() / total_variance)
        scores["median_pair_distance"] = median_pair_distance(reference)
        scores["blur_over_pair_distance"] = blur / scores["median_pair_distance"]
        by_dimension[n_dims or int(observed_s.shape[1])] = scores

    held_sample = sample_rows(held_rows, args.n_cells, rng)
    with torch.no_grad():
        held_predicted = model.phi(features_t[held_sample])
    held_observed = target_t[held_sample]
    held_s = alignment_block(held_observed, law).cpu().numpy()
    gene_names = list(payload["gene_names"])
    held_constant = np.broadcast_to(
        fit_s.mean(0).cpu().numpy(), held_s.shape).copy()
    held_candidates = {
        "constant": held_constant,
        "affine_init": alignment_block(
            apply_affine(production, affine_scale, affine_bias, held_sample, law),
            law).cpu().numpy(),
        "trained_phi": alignment_block(held_predicted, law).cpu().numpy(),
        "paired_oracle": held_s,
        "permuted_oracle": permute_rows(held_s, rng),
    }

    results = {
        "run_dir": args.run_dir,
        "checkpoint": args.checkpoint,
        "eval_split": args.eval_split,
        "chromatin_transform": transform,
        "alignment_block": "spliced" if law == RELAY else "joint",
        "method": "KOT",
        "dataset": payload["dataset"],
        "law": law,
        "condition": config["condition"],
        "seed": config["seed"],
        "align_dims_trained": config["align_dims"],
        "sinkhorn_max_points_trained": config["sinkhorn_max_points"],
        "sinkhorn_blur": blur,
        "n_cells_per_side": int(len(a_sample)),
        "candidates": list(CANDIDATE_NAMES),
        "phi_output_sd_median": float(candidates["trained_phi"].std(0).median()),
        "target_sd_median": float(observed_s.std(0).median()),
        "sd_ratio_phi_over_target": spread_ratio(candidates["trained_phi"], observed_s),
        "by_dimension": by_dimension,
        "held_out": {name: held_out_metrics(held_candidates[name], held_s, gene_names,
                                            config["seed"])
                     for name in CANDIDATE_NAMES},
    }
    out = Path(args.out) if args.out else (
        Path(args.run_dir) / f"failure_analysis_{args.eval_split}.json")
    out.write_text(json.dumps(results, indent=2, default=str))
    print(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
