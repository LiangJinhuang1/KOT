#!/usr/bin/env python3
r"""Is the chromatin velocity field usable, and how much direction signal does it hold?

The kinetic law only sees v_c, and the two datasets build it differently. Three questions share dataset, split, neighbour graph and coherence statistic.

Reads whole datasets; submit these.

Usage:
  python tools/chromatin_velocity_diagnostics.py audit --dataset hspc
  python tools/chromatin_velocity_diagnostics.py epsilon
  python tools/chromatin_velocity_diagnostics.py ceiling --dataset bmmc --run-dir ...
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
from src.data.chromatin_velocity import direction_diversity, hspc_velocity
from src.evaluation.chromatin_eval import field_coherence, row_cosine, task_d_kinetics
from src.losses.chromatin_laws import (LOG1P_MAX, LOG1P_MIN, REDUCED,
                                       kinetics_residual, law_rhs,
                                       linear_rate_to_log1p)
from src.training.kot import choose_torch_device

RESULTS = PROJECT_ROOT / "cache" / "results" / "chromatin"

KEPT_METRICS = ("cell_cosine_median", "cell_cosine_centred_median",
                "cell_cosine_centred_null_p95", "gene_pearson_median")


def scored_metrics(predicted: np.ndarray, reference: np.ndarray) -> dict:
    """The three Task D numbers, on the contiguous float32 the evaluator expects."""
    metrics = task_d_kinetics(np.ascontiguousarray(predicted, np.float32),
                              np.ascontiguousarray(reference, np.float32))
    return {key: metrics[key] for key in KEPT_METRICS}


def project(matrix: np.ndarray, mapping: torch.Tensor, device: torch.device) -> np.ndarray:
    """G applied to every cell's row, on the device: (cells x activity) @ G.T."""
    return (torch.as_tensor(matrix, device=device) @ mapping.T).cpu().numpy()


def neighbour_index(state: np.ndarray, n_neighbors: int, device: torch.device) -> np.ndarray:
    """Each cell's nearest others in RNA space, itself excluded."""
    points = torch.as_tensor(state, device=device)
    distances = torch.cdist(points, points)
    return distances.topk(n_neighbors + 1, largest=False).indices[:, 1:].cpu().numpy()


def neighbour_coherence(field: np.ndarray, index: np.ndarray, seed: int) -> dict:
    """Does the field agree with the mean field of the cell's RNA-space neighbours?

    Raw cosine is shared-mean and uninformative; centred cosine is read against a random-cell comparison.
    """
    local = field[index].mean(axis=1)
    rng = np.random.default_rng(seed)
    distant = field[rng.integers(0, len(field), size=index.shape)].mean(axis=1)
    centred = field - field.mean(axis=0)
    return {
        "neighbour_cosine_median": float(np.median(row_cosine(field, local))),
        "random_cosine_median": float(np.median(row_cosine(field, distant))),
        "neighbour_cosine_centred_median": float(
            np.median(row_cosine(centred, local - local.mean(axis=0)))),
        "random_cosine_centred_median": float(
            np.median(row_cosine(centred, distant - distant.mean(axis=0)))),
    }


def ridge_prediction(train_features: np.ndarray, train_reference: np.ndarray,
                     test_features: np.ndarray, alpha: float,
                     device: torch.device) -> np.ndarray:
    """The best linear read-out of the chromatin field, fitted with the true pairing.

    Float64 because the gram matrix of a velocity field is poorly conditioned.
    """
    features = torch.as_tensor(train_features, device=device, dtype=torch.float64)
    reference = torch.as_tensor(train_reference, device=device, dtype=torch.float64)
    held_out = torch.as_tensor(test_features, device=device, dtype=torch.float64)
    feature_mean, reference_mean = features.mean(dim=0), reference.mean(dim=0)
    centred = features - feature_mean
    gram = centred.T @ centred
    gram += alpha * torch.eye(gram.shape[0], device=device, dtype=torch.float64)
    weights = torch.linalg.solve(gram, centred.T @ (reference - reference_mean))
    predicted = (held_out - feature_mean) @ weights + reference_mean
    return predicted.cpu().numpy().astype(np.float32)


def shuffle_rows(matrix: np.ndarray, seed: int) -> np.ndarray:
    """The same cells, repaired at random. Used to CORRUPT a ridge's training pairs; the
    chance level of a score comes from `task_d_kinetics`' own permutation null."""
    return matrix[np.random.default_rng(seed).permutation(len(matrix))]


def model_pushforward(run_dir: Path, checkpoint: str, activity: np.ndarray,
                      velocity: np.ndarray, production: np.ndarray, rows: np.ndarray,
                      device: torch.device) -> np.ndarray:
    """J_phi(c) v_c from a trained checkpoint, on the same cells as everything else."""
    model, payload, _ = chromatin.load_checkpoint(run_dir, checkpoint, device)
    assert payload["law"] == REDUCED, (
        f"{run_dir} was trained under {payload['law']!r}; this tool scores the R1 path")
    chromatin_rows = torch.as_tensor(activity[rows], device=device)
    velocity_rows = torch.as_tensor(velocity[rows], device=device)
    production_rows = torch.as_tensor(production[rows], device=device)
    residual, prediction = kinetics_residual(model, REDUCED, chromatin_rows, velocity_rows,
                                             production_rows)
    pushforward = residual + law_rhs(model, REDUCED, prediction, chromatin_rows,
                                     production_rows)
    return pushforward.detach().cpu().numpy()


def usable_rows(split_column: np.ndarray, split: str, dynamic_mask: np.ndarray,
                target_covered: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Cells of one split that carry a chromatin velocity, an RNA target, and a reference."""
    rows = np.flatnonzero((split_column == split) & dynamic_mask & target_covered)
    return rows[np.abs(reference[rows]).sum(axis=1) > 0]


def audit(args: argparse.Namespace) -> int:


    device = choose_torch_device({"device": args.device})
    adata = chromatin.load_dataset(args.dataset)
    fields = chromatin.load_velocity(args.dataset, args.split_seed)
    activity = np.asarray(adata.obsm["gene_activity"], dtype=np.float32)
    velocity = fields["velocity"].astype(np.float32)
    dynamic = fields["dynamic_mask"].astype(bool)

    rows = np.flatnonzero(dynamic)
    rows = np.random.default_rng(args.seed).choice(
        rows, min(args.n_cells, len(rows)), replace=False)
    state = torch.as_tensor(activity[rows], device=device)
    print(f"[audit] {args.dataset}: {len(rows)} dynamic cells sampled of {dynamic.sum()}",
          flush=True)

    results = {"dataset": args.dataset, "n_cells": int(len(rows))}
    results["coherence_gene_activity"] = field_coherence(
        velocity[rows], activity[rows], args.n_neighbors, args.seed)
    lsi = np.asarray(adata.obsm["lsi"], dtype=np.float32)[rows]
    results["coherence_lsi"] = field_coherence(velocity[rows], lsi, args.n_neighbors, args.seed)

    pair = torch.cdist(state, state)
    typical = float(pair[pair > 0].median())
    step = np.linalg.norm(velocity[rows], axis=1)
    results["scale"] = {
        "median_cell_distance": typical,
        "median_step_norm": float(np.median(step)),
        "step_over_cell_distance": float(np.median(step) / max(typical, 1e-12)),
    }

    # Whether the barycentric future is just the day-7 population mean.
    if "day" in adata.obs:
        day = adata.obs["day"].to_numpy().astype(int)
        future = activity[rows] + velocity[rows] * 7.0
        population = activity[day == 7].mean(axis=0, keepdims=True)
        toward = population - activity[rows]
        results["contraction"] = {
            "cosine_velocity_vs_pull_to_day7_mean": float(np.median(row_cosine(
                velocity[rows], np.broadcast_to(toward, velocity[rows].shape).copy()))),
            "future_spread_over_day7_spread": float(
                np.median(future.std(axis=0)) / max(float(np.median(activity[day == 7].std(axis=0))), 1e-12)),
        }

    out = RESULTS / f"velocity_audit_{args.dataset}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"[audit] wrote {out}")
    return 0


def epsilon(args: argparse.Namespace) -> int:


    device = choose_torch_device({"device": args.device})
    adata = chromatin.load_dataset("hspc")
    splits = pd.read_csv(chromatin.split_path("hspc", args.split_seed), index_col=0)
    split = splits["split"].to_numpy()
    side = splits["train_side"].to_numpy()
    train_source = (split == "train") & (side == "atac")
    atac_available = (split != "train") | train_source
    activity = np.asarray(adata.obsm["gene_activity"], dtype=np.float32)
    day = adata.obs["day"].to_numpy().astype(int)
    population = activity[day == 7].mean(axis=0, keepdims=True)

    rows = []
    for epsilon in args.epsilons:
        field = hspc_velocity(adata, epsilon, args.ot_iterations, args.confidence_quantile,
                              device, split=split, source_mask=atac_available)
        velocity, dynamic = field["velocity"], field["dynamic_mask"].astype(bool)
        scored = np.flatnonzero(dynamic)
        scored = np.random.default_rng(args.seed).choice(
            scored, min(args.n_cells, len(scored)), replace=False)
        toward = np.broadcast_to(population - activity[scored], velocity[scored].shape).copy()
        block = field_coherence(velocity[scored], activity[scored], args.n_neighbors,
                                args.seed)
        rows.append({
            "epsilon": epsilon,
            "n_dynamic": int(dynamic.sum()),
            "confidence_median": float(np.median(field["confidence"][dynamic])),
            "contraction_cosine": float(np.median(row_cosine(velocity[scored], toward))),
            "coherence_centred": block["neighbour_cosine_centred_median"],
            "random_centred": block["random_cosine_centred_median"],
            **{k: float(v) for k, v in direction_diversity(velocity, dynamic).items()},
        })
        r = rows[-1]
        print(f"  eps {epsilon:<7g} conf {r['confidence_median']:.4f}  "
              f"contraction {r['contraction_cosine']:+.4f}  "
              f"coherence {r['coherence_centred']:+.4f}  "
              f"dynamic {r['n_dynamic']}", flush=True)

    frame = pd.DataFrame(rows)
    out = RESULTS / "velocity_epsilon_sweep_hspc.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(f"\n[sweep] reference for scale: scVelo's own RNA velocity on HSPC scores "
          f"+0.229 centred coherence; the cached field scores +0.175 at contraction 0.989")
    print(f"[sweep] wrote {out}")
    return 0


def ceiling(args: argparse.Namespace) -> int:


    device = choose_torch_device({"device": args.device})
    adata = chromatin.load_dataset(args.dataset)
    splits = pd.read_csv(chromatin.split_path(args.dataset, args.split_seed), index_col=0)
    assert splits.index.equals(adata.obs_names), "split file does not match the dataset"
    fields = chromatin.load_velocity(args.dataset, args.split_seed)
    split_column = splits["split"].to_numpy()
    side_column = splits["train_side"].to_numpy()

    target_layer = chromatin.rna_target_layer(adata, "auto", REDUCED)
    assert target_layer == "spliced_lognorm", (
        f"the reference is a spliced velocity; target {target_layer!r} is a different quantity")
    target_covered = chromatin.target_cell_mask(adata, target_layer)
    gene_fit_rows = np.flatnonzero(target_covered & (split_column == "train")
                                   & (side_column == "rna"))
    gene_mask = chromatin.output_gene_mask(adata, REDUCED, rows=gene_fit_rows,
                                           target_layer=target_layer)
    target = chromatin.build_targets(adata, REDUCED, target_layer, gene_mask)
    reference = chromatin.reference_velocity(adata, args.reference_layer)
    assert reference is not None, f"run `reference --dataset {args.dataset}` first"
    reference = reference[:, np.flatnonzero(gene_mask)]

    full_mapping, _, _ = chromatin.gene_map(adata, chromatin.gene_map_path(args.dataset))
    mapping = torch.as_tensor(full_mapping[gene_mask].toarray().astype(np.float32),
                              device=device)
    activity = np.asarray(adata.obsm["gene_activity"], dtype=np.float32)
    velocity = fields["velocity"].astype(np.float32)
    production = project(activity, mapping, device)
    # Unpaired calibration so the structural row is the affine path phi actually carries.
    atac_rows = np.flatnonzero(target_covered & (split_column == "train")
                               & (side_column == "atac"))
    scale, _ = chromatin.gene_affine_calibration(
        torch.as_tensor(production, device=device), torch.as_tensor(target, device=device),
        torch.as_tensor(atac_rows, device=device),
        torch.as_tensor(gene_fit_rows, device=device), REDUCED)
    scale = scale.cpu().numpy()

    train_rows = usable_rows(split_column, "train", fields["dynamic_mask"], target_covered,
                             reference)
    test_rows = usable_rows(split_column, "test", fields["dynamic_mask"], target_covered,
                            reference)
    covered_genes = np.flatnonzero(
        np.abs(reference[np.concatenate([train_rows, test_rows])]).sum(axis=0) > 0)
    print(f"[ceiling] {args.dataset} on {device}: fit on {len(train_rows)} train cells, "
          f"score on {len(test_rows)} test cells, {len(covered_genes)} genes the reference "
          "covers")

    # Convert the model side to linear rate so the comparison matches reference_agreement; converting the reference needs scVelo Ms which is not stored.
    exponent = np.exp(np.clip(target, LOG1P_MIN, LOG1P_MAX))
    reference_log = reference[:, covered_genes]
    field_push = ((project(velocity, mapping, device) * scale) * exponent)[:, covered_genes]
    observed_state = target[:, covered_genes]

    test_reference = reference_log[test_rows]
    train_reference = reference_log[train_rows]
    predictions = {
        "structural_G_push": field_push[test_rows],
        "ridge_on_G_push": ridge_prediction(field_push[train_rows], train_reference,
                                            field_push[test_rows], args.ridge_alpha, device),
        "ridge_on_raw_field": ridge_prediction(velocity[train_rows], train_reference,
                                               velocity[test_rows], args.ridge_alpha, device),
        "ridge_on_G_push_shuffled_fit": ridge_prediction(
            field_push[train_rows], shuffle_rows(train_reference, args.seed),
            field_push[test_rows], args.ridge_alpha, device),
    }
    for run_dir in args.run_dir:
        predictions[f"trained_jvp:{Path(run_dir).name}"] = model_pushforward(
            Path(run_dir), args.checkpoint, activity, velocity, production, test_rows,
            device)[:, covered_genes] * exponent[test_rows][:, covered_genes]

    index = neighbour_index(observed_state[test_rows], args.n_neighbors, device)
    results = {
        "dataset": args.dataset,
        "split_seed": args.split_seed,
        "reference_layer": args.reference_layer,
        "ridge_alpha": args.ridge_alpha,
        "n_train_cells": int(len(train_rows)),
        "n_test_cells": int(len(test_rows)),
        "n_genes": int(len(covered_genes)),
        "coherence": {
            "reference": neighbour_coherence(test_reference, index, args.seed),
            "chromatin_push": neighbour_coherence(field_push[test_rows], index, args.seed),
        },
        "agreement": {name: scored_metrics(predicted, test_reference)
                      for name, predicted in predictions.items()},
    }

    out = Path(args.out) if args.out else (
        RESULTS / f"velocity_ceiling_{args.dataset}_seed{args.split_seed}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))

    print("\n[ceiling] field coherence on test cells (vs the same cells' RNA-space neighbours)")
    for name, block in results["coherence"].items():
        print(f"  {name:<16} centred: neighbours "
              f"{block['neighbour_cosine_centred_median']:+.4f}"
              f"   random {block['random_cosine_centred_median']:+.4f}")
    # Centred cosine vs permutation null; raw cosine is shared-mean and uninformative.
    print("\n[ceiling] agreement with the RNA reference, median per-cell cosine")
    print(f"  {'row':<44} {'centred':>9} {'null p95':>9} {'gene r':>9} {'raw':>9}")
    for name, block in results["agreement"].items():
        print(f"  {name:<44} {block['cell_cosine_centred_median']:>+9.4f} "
              f"{block['cell_cosine_centred_null_p95']:>+9.4f} "
              f"{block['gene_pearson_median']:>+9.4f} "
              f"{block['cell_cosine_median']:>+9.4f}")
    print(f"\n[ceiling] wrote {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, function in [("audit", audit), ("epsilon", epsilon), ("ceiling", ceiling)]:
        child = sub.add_parser(name)
        child.set_defaults(func=function)
        child.add_argument("--split-seed", type=int, default=0)
        child.add_argument("--seed", type=int, default=0)
        child.add_argument("--n-neighbors", type=int, default=15)
        child.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
        # epsilon is HSPC-only because only that dataset has the OT field.
        if name != "epsilon":
            child.add_argument("--dataset", choices=chromatin.DATASETS, required=True)
        if name in ("audit", "epsilon"):
            child.add_argument("--n-cells", type=int, default=4000)
        if name == "epsilon":
            child.add_argument("--epsilons", type=float, nargs="+",
                               default=[0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002])
            child.add_argument("--ot-iterations", type=int, default=200)
            child.add_argument("--confidence-quantile", type=float, default=0.1)
        if name == "ceiling":
            child.add_argument("--reference-layer", default="velocity_scvelo")
            child.add_argument("--ridge-alpha", type=float, default=100.0)
            child.add_argument("--run-dir", nargs="+", default=[],
                               help="trained checkpoints to score on the ceiling's cells")
            child.add_argument("--checkpoint", default="best_align")
            child.add_argument("--out", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
