#!/usr/bin/env python3
r"""Four checks on whether a trained chromatin run's headline numbers survive a control.

They share dataset, split, checkpoint loading and retrieval metrics.

Reads whole datasets; submit these.

Usage:
  python tools/chromatin_model_checks.py intron-confound --run-dir cache/chromatin/runs/...
  python tools/chromatin_model_checks.py paired-kinetics --dataset bmmc
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
from src.evaluation.chromatin_eval import (cell_type_match_accuracy, column_pearson,
                                           retrieval_metrics)
from src.losses.chromatin_laws import (REDUCED, RELAY, law_mask, law_rhs,
                                       linear_abundance, linear_rate_to_log1p)
from scipy.optimize import nnls
from sklearn.decomposition import PCA

from src.training.kot import choose_torch_device
from src.utils.arrays import to_dense

RESULTS = PROJECT_ROOT / "cache" / "results" / "chromatin"
N_SHUFFLE = 8



def residual_share(args: argparse.Namespace) -> int:


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

def score(predicted: np.ndarray, observed: np.ndarray, labels: np.ndarray | None) -> dict:
    metrics = retrieval_metrics(predicted, observed)
    if labels is not None:
        metrics["cell_type_accuracy"] = cell_type_match_accuracy(predicted, observed, labels)
    return {key: round(float(value), 4) for key, value in metrics.items()}


def scoring_parity(args: argparse.Namespace) -> int:


    device = torch.device("cpu")
    model, payload, config = chromatin.load_checkpoint(Path(args.run_dir), args.checkpoint,
                                                       device)
    adata = chromatin.load_dataset(payload["dataset"])
    splits = pd.read_csv(chromatin.split_path(payload["dataset"], config["split_seed"]),
                         index_col=0)
    split_column = splits["split"].to_numpy()
    measured = chromatin.target_cell_mask(adata, config["target_layer"])
    test = np.flatnonzero((split_column == "test") & measured)
    rna_train = np.flatnonzero((split_column == "train")
                               & (splits["train_side"].to_numpy() == "rna") & measured)

    chromatin_matrix = np.asarray(adata.obsm["gene_activity"], dtype=np.float32)
    output_columns = adata.var_names.get_indexer(payload["gene_names"])
    assert (output_columns >= 0).all(), "checkpoint output genes do not match dataset"
    observed_all = to_dense(adata.layers[config["target_layer"]], np.float32)[:, output_columns]
    with torch.no_grad():
        predicted = chromatin.spliced_block(
            model.phi(torch.as_tensor(chromatin_matrix[test], device=device)).numpy(),
            payload["law"])
    observed = observed_all[test]
    labels = (adata.obs["cell_type"].astype(str).to_numpy()[test]
              if "cell_type" in adata.obs else None)

    # Basis on RNA train half so the map is not scored in a space built from the answer.
    basis = PCA(n_components=args.n_comps, random_state=args.seed).fit(
        observed_all[rna_train])
    sample = np.sort(np.random.default_rng(args.seed).choice(
        len(test), min(args.n_cells, len(test)), replace=False))

    results = {"run_dir": args.run_dir, "dataset": payload["dataset"],
               "condition": config["condition"], "n_test_full": int(len(test)),
               "n_test_sampled": int(len(sample)), "n_comps": args.n_comps}
    for space, (a, b) in {"gene_space": (predicted, observed),
                          "pca_space": (basis.transform(predicted),
                                        basis.transform(observed))}.items():
        results[f"{space}_full"] = score(a, b, labels)
        results[f"{space}_subsampled"] = score(
            a[sample], b[sample], None if labels is None else labels[sample])

    out = Path(args.run_dir) / "scoring_parity.json"
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"\nwrote {out}")
    return 0

def per_gene_zscore(values: np.ndarray) -> np.ndarray:
    """Each gene centred and scaled by its own statistics over these cells.

    Gene length, intron content and mappability are constant across cells, so they live in a gene's mean and scale.
    """
    centred = values - values.mean(axis=0, keepdims=True)
    return centred / values.std(axis=0, keepdims=True).clip(min=1e-6)

def retrieval_pair(predicted: np.ndarray, observed: np.ndarray, seed: int) -> dict:
    """Retrieval as reported, and again with every per-gene constant removed."""
    raw = retrieval_metrics(predicted, observed, seed)
    scored = retrieval_metrics(per_gene_zscore(predicted), per_gene_zscore(observed), seed)
    return {
        "top1": raw["top1"], "foscttm": raw["foscttm"],
        "foscttm_constant_floor": raw["foscttm_constant_floor"],
        "top1_gene_zscored": scored["top1"], "foscttm_gene_zscored": scored["foscttm"],
        "foscttm_constant_floor_gene_zscored": scored["foscttm_constant_floor"],
        "chance_top1": 1.0 / len(observed),
    }


def intron_confound(args: argparse.Namespace) -> int:


    device = choose_torch_device({"device": args.device})
    model, payload, config = chromatin.load_checkpoint(Path(args.run_dir), args.checkpoint,
                                                       device)
    assert payload["law"] == RELAY, "the confound test is about the relay's u block"
    adata = chromatin.load_dataset(payload["dataset"])
    splits = pd.read_csv(chromatin.split_path(payload["dataset"], config["split_seed"]),
                         index_col=0)
    target_layer = payload["target_layer"]
    covered = chromatin.target_cell_mask(adata, target_layer)
    columns = adata.var_names.get_indexer(payload["gene_names"])
    assert (columns >= 0).all(), "checkpoint genes are not in the dataset"
    gene_mask = np.zeros(adata.n_vars, dtype=bool)
    gene_mask[columns] = True
    target = chromatin.build_targets(adata, RELAY, target_layer, gene_mask)
    activity = np.asarray(adata.obsm["gene_activity"], dtype=np.float32)

    test = np.flatnonzero((splits["split"].to_numpy() == "test") & covered)
    rows = np.random.default_rng(config["seed"]).choice(
        test, min(args.n_cells, len(test)), replace=False)
    with torch.no_grad():
        predicted = model.phi(torch.as_tensor(activity[rows], device=device)).cpu().numpy()
    observed = target[rows]
    blocks = {
        "unspliced": (chromatin.unspliced_block(predicted, RELAY),
                      chromatin.unspliced_block(observed, RELAY)),
        "spliced": (chromatin.spliced_block(predicted, RELAY),
                    chromatin.spliced_block(observed, RELAY)),
    }
    print(f"[confound] {args.run_dir}: {len(rows)} held-out cells, "
          f"{len(payload['gene_names'])} genes per block", flush=True)

    results = {"run_dir": args.run_dir, "n_cells": int(len(rows)), "blocks": {}}
    for name, (block_pred, block_obs) in blocks.items():
        gene_r = column_pearson(block_pred, block_obs)
        results["blocks"][name] = {
            "gene_pearson_median": float(np.nanmedian(gene_r)),
            **retrieval_pair(block_pred, block_obs, config["seed"]),
        }

    # Unspliced share as intron-length proxy for the confound.
    u_obs, s_obs = blocks["unspliced"][1], blocks["spliced"][1]
    intron_share = u_obs.mean(axis=0) / np.clip(u_obs.mean(axis=0) + s_obs.mean(axis=0), 1e-9, None)
    gene_r_u = column_pearson(*blocks["unspliced"])
    usable = np.isfinite(gene_r_u) & np.isfinite(intron_share)
    order = np.argsort(intron_share[usable])
    thirds = np.array_split(order, 3)
    results["intron_stratification"] = {
        "spearman_gene_pearson_vs_intron_share": float(
            pd.Series(gene_r_u[usable]).corr(pd.Series(intron_share[usable]), method="spearman")),
        "terciles": [{
            "intron_share_median": float(np.median(intron_share[usable][part])),
            "u_gene_pearson_median": float(np.median(gene_r_u[usable][part])),
            "n_genes": int(len(part)),
        } for part in thirds],
    }

    out = Path(args.out) if args.out else Path(args.run_dir) / "intron_confound.json"
    out.write_text(json.dumps(results, indent=2))

    print(f"\n{'block':<11}{'geneR':>9}{'top1':>9}{'top1 z':>9}{'FOSCTTM':>9}{'FOSCTTM z':>11}"
          f"{'floor z':>9}")
    for name, b in results["blocks"].items():
        print(f"{name:<11}{b['gene_pearson_median']:>+9.4f}{b['top1']:>9.5f}"
              f"{b['top1_gene_zscored']:>9.5f}{b['foscttm']:>9.4f}"
              f"{b['foscttm_gene_zscored']:>11.4f}"
              f"{b['foscttm_constant_floor_gene_zscored']:>9.4f}")
    print(f"chance top1 {results['blocks']['unspliced']['chance_top1']:.5f}")
    strat = results["intron_stratification"]
    print(f"\nu gene-Pearson vs intron share, Spearman: "
          f"{strat['spearman_gene_pearson_vs_intron_share']:+.4f}")
    for i, t in enumerate(strat["terciles"]):
        print(f"  tercile {i + 1}: intron share {t['intron_share_median']:.3f}  "
              f"u gene Pearson {t['u_gene_pearson_median']:+.4f}  ({t['n_genes']} genes)")
    print(f"\n[confound] wrote {out}")
    return 0

def project(matrix: np.ndarray, mapping: torch.Tensor, device: torch.device) -> np.ndarray:
    return (torch.as_tensor(matrix, device=device) @ mapping.T).cpu().numpy()

def usable_rows(split_column: np.ndarray, split: str, dynamic_mask: np.ndarray,
                target_covered: np.ndarray, left: np.ndarray) -> np.ndarray:
    rows = np.flatnonzero((split_column == split) & dynamic_mask & target_covered)
    return rows[np.abs(left[rows]).sum(axis=1) > 0]

def fit_gene_rates(production: np.ndarray, log_state: np.ndarray,
                   target_log_rate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gene-wise non-negative (alpha, gamma) so RHS log-rate ≈ target_log_rate."""
    abundance = np.expm1(np.clip(log_state, 0.0, np.log1p(1e4)))
    scale = np.exp(-np.clip(log_state, 0.0, np.log1p(1e4)))
    n_genes = log_state.shape[1]
    alpha = np.zeros(n_genes, dtype=np.float64)
    gamma = np.zeros(n_genes, dtype=np.float64)
    for gene in range(n_genes):
        design = np.column_stack([
            production[:, gene] * scale[:, gene],
            -abundance[:, gene] * scale[:, gene],
        ])
        target = target_log_rate[:, gene]
        if not np.isfinite(design).all() or not np.isfinite(target).all():
            continue
        if np.linalg.norm(target) < 1e-12 or np.linalg.norm(design) < 1e-12:
            continue
        coeffs, _ = nnls(design, target)
        alpha[gene], gamma[gene] = coeffs
    return alpha.astype(np.float32), gamma.astype(np.float32)

def rhs_log_rate(production: np.ndarray, log_state: np.ndarray,
                 alpha: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """kappa=1 reduced RHS in log1p coordinates, numpy twin of chromatin_laws.reduced_rhs."""
    y = torch.as_tensor(log_state, dtype=torch.float32)
    linear = (torch.as_tensor(alpha) * torch.as_tensor(production)
              - torch.as_tensor(gamma) * linear_abundance(y))
    return linear_rate_to_log1p(linear, y).numpy()

def residual_summary(left: np.ndarray, right: np.ndarray) -> dict:
    delta = left - right
    per_cell = (delta ** 2).mean(axis=1)
    left_norm = np.linalg.norm(left, axis=1)
    right_norm = np.linalg.norm(right, axis=1)
    denom = np.clip(left_norm * right_norm, 1e-12, None)
    cosine = (left * right).sum(axis=1) / denom
    return {
        "mse_mean": float(per_cell.mean()),
        "mse_median": float(np.median(per_cell)),
        "cosine_median": float(np.median(cosine)),
        "left_norm_median": float(np.median(left_norm)),
        "right_norm_median": float(np.median(right_norm)),
    }

def score_field(push: np.ndarray, production: np.ndarray, log_state: np.ndarray,
                alpha: np.ndarray, gamma: np.ndarray, rows: np.ndarray) -> dict:
    right = rhs_log_rate(production[rows], log_state[rows], alpha, gamma)
    return residual_summary(push[rows], right)

def subsample(rows: np.ndarray, limit: int | None, seed: int) -> np.ndarray:
    if limit is None or len(rows) <= limit:
        return rows
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(rows, size=limit, replace=False))

def run_dataset(dataset: str, split_seed: int, reference_layer: str, seed: int,
                max_train: int | None, max_test: int | None,
                device: torch.device) -> dict:
    adata = chromatin.load_dataset(dataset)
    splits = pd.read_csv(chromatin.split_path(dataset, split_seed), index_col=0)
    assert splits.index.equals(adata.obs_names), "split file does not match the dataset"
    fields = chromatin.load_velocity(dataset, split_seed)
    split_column = splits["split"].to_numpy()
    side_column = splits["train_side"].to_numpy()

    target_layer = chromatin.rna_target_layer(adata, "auto", REDUCED)
    assert target_layer == "spliced_lognorm", (
        f"need spliced_lognorm for the reduced paired diagnostic; got {target_layer!r}")
    target_covered = chromatin.target_cell_mask(adata, target_layer)
    gene_fit_rows = np.flatnonzero(target_covered & (split_column == "train")
                                   & (side_column == "rna"))
    gene_mask = chromatin.output_gene_mask(adata, REDUCED, rows=gene_fit_rows,
                                           target_layer=target_layer)
    log_state = chromatin.build_targets(adata, REDUCED, target_layer, gene_mask)
    reference = chromatin.reference_velocity(adata, reference_layer)
    assert reference is not None, f"run `reference --dataset {dataset}` first"
    reference = reference[:, np.flatnonzero(gene_mask)]

    full_mapping, _, _ = chromatin.gene_map(adata, chromatin.gene_map_path(dataset))
    mapping = torch.as_tensor(full_mapping[gene_mask].toarray().astype(np.float32),
                              device=device)
    activity = np.asarray(adata.obsm["gene_activity"], dtype=np.float32)
    velocity = fields["velocity"].astype(np.float32)
    production = project(activity, mapping, device)
    push = project(velocity, mapping, device)

    covered = np.flatnonzero(
        (np.abs(push).sum(axis=0) > 0) & (np.abs(reference).sum(axis=0) > 0)
        & (np.abs(log_state).sum(axis=0) > 0) & (np.abs(production).sum(axis=0) > 0))
    push = push[:, covered]
    production = production[:, covered]
    log_state = log_state[:, covered]
    reference = reference[:, covered]
    reference_log = reference * np.exp(-np.clip(log_state, 0.0, np.log1p(1e4)))

    train_rows = usable_rows(split_column, "train", fields["dynamic_mask"], target_covered,
                             push)
    test_rows = usable_rows(split_column, "test", fields["dynamic_mask"], target_covered,
                            push)
    # Val if present else a train slice for reporting only.
    val_rows = usable_rows(split_column, "val", fields["dynamic_mask"], target_covered, push)
    train_rows = subsample(train_rows, max_train, seed)
    score_rows = val_rows if len(val_rows) >= 100 else test_rows
    score_name = "val" if len(val_rows) >= 100 else "test"
    score_rows = subsample(score_rows, max_test, seed + 1)

    print(f"[paired-kinetics] {dataset} on {device}: fit on {len(train_rows)} train cells, "
          f"score on {len(score_rows)} {score_name} cells, {push.shape[1]} genes")

    fit_targets = {
        "fit_to_chromatin_push": push,
        "fit_to_rna_reference": reference_log,
    }
    rng = np.random.default_rng(seed)
    report = {
        "dataset": dataset,
        "split_seed": split_seed,
        "reference_layer": reference_layer,
        "n_train_cells": int(len(train_rows)),
        "n_score_cells": int(len(score_rows)),
        "score_split": score_name,
        "n_genes": int(push.shape[1]),
        "n_shuffle": N_SHUFFLE,
        "fits": {},
    }

    for fit_name, target in fit_targets.items():
        alpha, gamma = fit_gene_rates(production[train_rows], log_state[train_rows],
                                      target[train_rows])
        active = int(((alpha > 0) | (gamma > 0)).sum())
        block = {
            "n_active_genes": active,
            "alpha_median": float(np.median(alpha)),
            "gamma_median": float(np.median(gamma)),
            "train_correct": score_field(push, production, log_state, alpha, gamma,
                                         train_rows),
            "score_correct": score_field(push, production, log_state, alpha, gamma,
                                         score_rows),
            "score_reverse": score_field(-push, production, log_state, alpha, gamma,
                                          score_rows),
            "score_shuffle": [],
        }
        for draw in range(N_SHUFFLE):
            perm = rng.permutation(len(score_rows))
            shuffled_push = np.zeros_like(push)
            shuffled_push[score_rows] = push[score_rows][perm]
            block["score_shuffle"].append(
                score_field(shuffled_push, production, log_state, alpha, gamma, score_rows))
        shuffle_mse = np.array([row["mse_mean"] for row in block["score_shuffle"]],
                               dtype=np.float64)
        correct_mse = block["score_correct"]["mse_mean"]
        block["shuffle_mse_mean"] = float(shuffle_mse.mean())
        block["shuffle_mse_std"] = float(shuffle_mse.std())
        block["correct_beats_shuffle"] = bool(correct_mse < shuffle_mse.mean())
        block["correct_over_shuffle"] = float(correct_mse / max(shuffle_mse.mean(), 1e-30))
        # Shuffled-fit residual diagnoses under-identification.
        train_perm = rng.permutation(len(train_rows))
        shuffled_train_push = push[train_rows][train_perm]
        alpha_sh, gamma_sh = fit_gene_rates(production[train_rows], log_state[train_rows],
                                            shuffled_train_push)
        block["refit_on_shuffled_train"] = score_field(
            push, production, log_state, alpha_sh, gamma_sh, score_rows)
        # Same shuffled field the wrong direction would see if treated as truth.
        score_perm = rng.permutation(len(score_rows))
        shuffled_score_push = np.zeros_like(push)
        shuffled_score_push[score_rows] = push[score_rows][score_perm]
        block["refit_on_shuffled_eval_shuffled"] = score_field(
            shuffled_score_push, production, log_state, alpha_sh, gamma_sh, score_rows)

        report["fits"][fit_name] = block
        print(f"\n[{fit_name}] active genes {active}/{push.shape[1]}")
        print(f"  score correct mse {correct_mse:.6g}   "
              f"shuffle mean {shuffle_mse.mean():.6g} ± {shuffle_mse.std():.3g}   "
              f"ratio correct/shuffle {block['correct_over_shuffle']:.4f}")
        print(f"  score reverse mse {block['score_reverse']['mse_mean']:.6g}   "
              f"cosine correct {block['score_correct']['cosine_median']:+.4f}   "
              f"shuffle cos {np.median([r['cosine_median'] for r in block['score_shuffle']]):+.4f}")
        print(f"  refit-on-shuffle then eval correct mse "
              f"{block['refit_on_shuffled_train']['mse_mean']:.6g}; "
              f"eval shuffled mse {block['refit_on_shuffled_eval_shuffled']['mse_mean']:.6g}")

    return report


def paired_kinetics(args: argparse.Namespace) -> int:


    device = choose_torch_device({"device": args.device})
    report = run_dataset(args.dataset, args.split_seed, args.reference_layer, args.seed,
                         args.max_train_cells, args.max_test_cells, device)
    out = Path(args.out) if args.out else (
        PROJECT_ROOT / "cache" / "results" / "chromatin"
        / f"paired_kinetics_{args.dataset}_seed{args.split_seed}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\n[paired-kinetics] wrote {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)

    child = sub.add_parser('residual-share')
    child.set_defaults(func=residual_share)
    child.add_argument("--run-dir", required=True)
    child.add_argument("--checkpoint", default="best_align")
    child.add_argument("--n-cells", type=int, default=2000)

    child = sub.add_parser('scoring-parity')
    child.set_defaults(func=scoring_parity)
    child.add_argument("--run-dir", required=True)
    child.add_argument("--checkpoint", default="best_align")
    child.add_argument("--n-cells", type=int, default=1200,
                        help="the cap the subsampled baselines ran at")
    child.add_argument("--n-comps", type=int, default=30)
    child.add_argument("--seed", type=int, default=42)

    child = sub.add_parser('intron-confound')
    child.set_defaults(func=intron_confound)
    child.add_argument("--run-dir", required=True)
    child.add_argument("--checkpoint", default="best_align")
    child.add_argument("--n-cells", type=int, default=2048)
    child.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    child.add_argument("--out", default=None)

    child = sub.add_parser('paired-kinetics')
    child.set_defaults(func=paired_kinetics)
    child.add_argument("--dataset", choices=chromatin.DATASETS, required=True)
    child.add_argument("--split-seed", type=int, default=0)
    child.add_argument("--reference-layer", default="velocity_scvelo")
    child.add_argument("--seed", type=int, default=0)
    child.add_argument("--max-train-cells", type=int, default=None)
    child.add_argument("--max-test-cells", type=int, default=None)
    child.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    child.add_argument("--out", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
