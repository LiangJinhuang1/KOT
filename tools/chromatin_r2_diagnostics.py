#!/usr/bin/env python3
"""Diagnostics for the regulatory R2 runs: input, phi, rates, and agreement with scVelo.

One tool because they share dataset, split, gene panel and G.

Usage:
  python tools/chromatin_r2_diagnostics.py normalisation --dataset bmmc
  python tools/chromatin_r2_diagnostics.py runs --runs r2lsi_bmmc_relay_full_seed42 ...
  python tools/chromatin_r2_diagnostics.py scvelo --runs ... --out cache/results/...
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.linear_model import Ridge

from tools.chromatin_common import (
    Context, centred_norm, context, gate_verdict, load_run, preflight, quantiles,
    run_transform,
)

import run_kot_chromatin as runner
from src.data.chromatin import CHROMATIN_TRANSFORMS, chromatin_features, normalize_log, tfidf_lsi
from src.evaluation.foscttm import calc_frac_idx, permuted_pairing_floor
from src.losses.chromatin_laws import relay_rhs

RIDGE_ALPHA = 100.0
RESULTS = Path("cache/results/chromatin")


def write(frame: pd.DataFrame, out: Path | None, default: str) -> None:
    destination = out or (RESULTS / default)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    print(f"\n[diagnostics] wrote {destination}")


def variants(ctx: Context) -> dict:
    """Alternative normalisations of c, on the same panel and column order.

    `zscore_per_gene` is the control: a per-gene rescaling must come back identical or the screen is broken.
    """
    counts = ctx.adata.obsm["gene_activity_counts"]
    current = np.asarray(ctx.adata.obsm["gene_activity"], dtype=np.float32)
    raw = runner.to_dense(counts, np.float32)
    standardised = (current - current.mean(0)) / current.std(0).clip(min=1e-6)
    return {
        "as_is": current,
        "raw_counts": raw,
        "cp10k_log1p": normalize_log(counts),
        "l2_per_cell": raw / np.linalg.norm(raw, axis=1, keepdims=True).clip(min=1e-6),
        "zscore_per_gene (control)": standardised.astype(np.float32),
        "tfidf_lsi": tfidf_lsi(counts),
    }


def depth_alignment(values: np.ndarray, depth: np.ndarray, q: int = 10) -> dict:
    """PC1 against the original library depth, plus how concentrated the variance is.

    Correlating against each variant's own row sum measures a different quantity per variant; fixing the depth vector separates depth removal from rank truncation.
    """
    centred = torch.as_tensor(np.asarray(values, dtype=np.float32))
    centred = centred - centred.mean(0)
    _, singular, basis = torch.svd_lowrank(centred, q=q)
    scores = centred @ basis
    reference = torch.as_tensor(depth, dtype=torch.float64)
    correlation = abs(float(torch.corrcoef(
        torch.stack([scores[:, 0].double(), reference]))[0, 1]))
    shares = (singular ** 2 / centred.pow(2).sum()).tolist()
    return {"pc1_vs_depth": correlation, "pc1_variance_share": shares[0],
            "rank10_variance_share": float(sum(shares))}


def normalisation(ctx: Context, out: Path | None) -> pd.DataFrame:
    target = torch.as_tensor(ctx.targets())
    source = torch.as_tensor(ctx.rows("train", side="atac"))
    rna = torch.as_tensor(ctx.rows("train", side="rna"))
    validation = ctx.rows("val")
    observed = target[validation]
    sample = torch.tensor(np.sort(np.random.default_rng(0).choice(
        len(validation), min(2000, len(validation)), replace=False)))
    depth = np.asarray(ctx.adata.obsm["gene_activity_counts"].sum(axis=1)).ravel()[validation]
    projection = torch.as_tensor(ctx.projection)

    rows = []
    for name, values in variants(ctx).items():
        chromatin = torch.as_tensor(np.asarray(values, dtype=np.float32))
        regulatory = chromatin @ projection.T
        scale, bias = runner.gene_affine_calibration(regulatory, target, source, rna,
                                                     runner.RELAY)
        prediction = torch.cat([regulatory[validation]] * 2, 1) * scale + bias
        confound = depth_alignment(np.asarray(values)[validation], depth)
        for block, span in ctx.blocks():
            metrics = runner.state_metrics(prediction[:, span], observed[:, span], sample)
            rows.append({"transform": name, "block": block,
                         "foscttm": metrics["foscttm"],
                         "permuted_floor": permuted_pairing_floor(
                             prediction[:, span][sample].numpy(),
                             observed[:, span][sample].numpy()),
                         "gene_pearson_median": metrics["state_pearson_median"],
                         "spread_ratio": metrics["prediction_spread_ratio"], **confound})
        print(f"  {name:26s} u {rows[-2]['foscttm']:.4f}  s {rows[-1]['foscttm']:.4f}  "
              f"pc1_vs_depth {confound['pc1_vs_depth']:.3f}", flush=True)
    frame = pd.DataFrame(rows)
    write(frame, out, f"r2_normalisation_{ctx.dataset}.csv")
    print(frame.pivot_table(index="transform", columns="block",
                            values=["foscttm", "gene_pearson_median"]).round(4).to_string())
    return frame


def rate_health(model, config, chromatin: torch.Tensor, n_genes: int) -> dict:
    rows = torch.tensor(np.sort(np.random.default_rng(0).choice(len(chromatin), 512, False)))
    with torch.no_grad():
        kappa = model.kappa(chromatin[rows])
        alpha = model.g(chromatin[rows] @ model.phi.gene_projection[:n_genes].T)
    anchors = config["gamma_anchors"]
    relative = (model.gamma.detach()[torch.tensor(anchors["indices"])]
                / model.gamma.new_tensor(anchors["gamma"]))
    return {"kappa_median": float(kappa.median()),
            # Pinned bounds can drive the RHS to zero without explaining the data.
            "kappa_at_floor": float((kappa < 1.1e-3).float().mean()),
            "alpha_at_floor": float((alpha < 1.1e-5).float().mean()),
            "anchor_error_percent": float(((relative - 1).abs() * 100).median())}


def runs(names: list[str], out: Path | None) -> pd.DataFrame:
    rows = []
    # Cache by (dataset, split, transform) because the object is large and shared.
    inputs: dict = {}
    for name in names:
        checks = preflight(name)
        history = pd.read_csv(Path("cache/chromatin/runs") / name / "training_loss.csv").iloc[-1]
        model, payload, config = load_run(name)
        key = (payload["dataset"], config["split_seed"],
               config.get("chromatin_transform", "as_is"))
        if key not in inputs:
            built = context(*key)
            inputs[key] = (built.n_genes, torch.as_tensor(built.chromatin()))
            del built
        n_genes, chromatin = inputs[key]
        rows.append({
            "run": name, "dataset": payload["dataset"],
            "transform": config.get("chromatin_transform", "as_is"),
            "phi_gate": config.get("phi_gate", "?"),
            "lambda_dyn": config["lambda_dyn"] if config["condition"] != "noDyn" else 0.0,
            "condition": config["condition"], "gate": gate_verdict(name),
            # Old runs judged against the constant-map floor are not comparable.
            "gate_floor_convention": ("permuted_pairing_0.5"
                                      if "foscttm_permuted_floor" in checks
                                      else "constant_map_0.25_SUPERSEDED"),
            "seed": config.get("seed"), "split_seed": config.get("split_seed"),
            "n_epochs": config.get("n_epochs"),
            "n_output_genes": config.get("n_output_genes"),
            "align_dims": config.get("align_dims"),
            "lambda_gamma_anchor": config.get("lambda_gamma_anchor"),
            "n_gamma_anchors": len(config.get("gamma_anchors", {}).get("indices", [])),
            "best_epoch": payload.get("epoch"),
            "foscttm_s": checks["foscttm"],
            "permuted_floor": checks.get("foscttm_permuted_floor", float("nan")),
            "foscttm_u": checks["unspliced_foscttm"],
            "gene_pearson": checks["state_pearson_median"],
            "spread": checks["prediction_spread_ratio"],
            "jvp_vs_scvelo_centred": checks["jvp_vs_reference_cosine_centred_median"],
            "jvp_null_p95": checks["jvp_vs_reference_cosine_centred_null_p95"],
            "jvp_rhs_cosine": checks["jvp_rhs_cosine_median"],
            "residual_norm": checks["residual_norm_median"],
            **{k: float(history[k]) for k in ("loss_align", "val_align", "loss_dyn",
                                              "loss_held_block", "grad_mag_ratio",
                                              "grad_cosine")},
            **rate_health(model, config, chromatin, n_genes)})
        print(f"  {name:44s} FOSCTTM {rows[-1]['foscttm_s']:.4f}  r "
              f"{rows[-1]['gene_pearson']:.4f}  k@floor {rows[-1]['kappa_at_floor']:.3f}",
              flush=True)
        del model
    frame = pd.DataFrame(rows).assign(
        jvp_margin=lambda d: d.jvp_vs_scvelo_centred - d.jvp_null_p95,
        jvp_ratio_to_null=lambda d: d.jvp_vs_scvelo_centred / d.jvp_null_p95)
    write(frame, out, "r2_runs.csv")
    for title, keep in [
        ("ALIGNMENT", ["transform", "phi_gate", "lambda_dyn", "gate", "foscttm_s",
                       "permuted_floor", "foscttm_u", "gene_pearson", "spread",
                       "val_align", "loss_held_block"]),
        ("JVP vs scVelo", ["transform", "phi_gate", "lambda_dyn", "jvp_vs_scvelo_centred",
                           "jvp_null_p95", "jvp_margin", "jvp_ratio_to_null",
                           "jvp_rhs_cosine", "residual_norm"]),
        ("RATE HEALTH", ["transform", "phi_gate", "lambda_dyn", "loss_dyn", "kappa_median",
                         "kappa_at_floor", "alpha_at_floor", "grad_mag_ratio",
                         "anchor_error_percent"])]:
        print(f"\n--- {title} ---")
        print(frame[keep].round(5).to_string(index=False))
    return frame


def phi(names: list[str], out: Path | None) -> pd.DataFrame:
    rows = []
    cache: dict = {}
    for name in names:
        model, payload, config = load_run(name)
        transform = config.get("chromatin_transform", "as_is")
        key = (payload["dataset"], transform)
        if key not in cache:
            cache[key] = torch.as_tensor(
                chromatin_features(runner.load_dataset(key[0]), transform))
        chromatin = cache[key]
        n_genes = len(payload["gene_names"])
        with torch.no_grad():
            prediction = model.phi(chromatin)
            affine = (chromatin @ model.phi.gene_projection.T) * model.phi.gene_scale \
                + model.phi.gene_bias
            network = prediction - affine
            # "none" removes the switch, so residual_scale() is a plain float there.
            value = model.phi.residual_scale()
            gate = np.atleast_1d(value.detach().numpy() if torch.is_tensor(value) else value)
        # Failed calibration starts more closed so the gate can still turn the gene off.
        uncalibrated = model.phi.gene_scale.detach().numpy() == 0
        separation = (float(gate[uncalibrated].mean() - gate[~uncalibrated].mean())
                      if gate.size > 1 else float("nan"))
        for block, span in [("u", slice(0, n_genes)), ("s", slice(n_genes, None))]:
            rows.append({
                "run": name, "phi_gate": config.get("phi_gate", "?"), "block": block,
                "network_over_prediction": float(
                    centred_norm(network[:, span]) / centred_norm(prediction[:, span])),
                "affine_over_prediction": float(
                    centred_norm(affine[:, span]) / centred_norm(prediction[:, span])),
                "gate_mean": float(gate.mean()), "gate_min": float(gate.min()),
                "gate_max": float(gate.max()),
                "gate_separation_failed_minus_worked": separation,
                "n_uncalibrated_outputs": int(uncalibrated.sum())})
        print(f"  {name:44s} network/pred s "
              f"{rows[-1]['network_over_prediction']:.4f}  gate {gate.mean():+.4f}", flush=True)
        del model
    frame = pd.DataFrame(rows)
    write(frame, out, "r2_phi.csv")
    print(frame.round(4).to_string(index=False))
    return frame


def supervised(ctx: Context, transforms: list[str], out: Path | None) -> pd.DataFrame:
    """Paired ceilings for the state map, per transform."""
    target = ctx.targets()
    train, test = ctx.rows("train"), ctx.rows("test")
    observed = target[test]
    sample = np.sort(np.random.default_rng(0).choice(len(test), min(2000, len(test)), False))
    rows = []
    for transform in transforms:
        chromatin = chromatin_features(ctx.adata, transform)
        regulatory = chromatin @ ctx.projection.T
        predictions = {
            "mean_only (floor)": np.tile(target[train].mean(axis=0), (len(test), 1)),
            "identity (Gc, no model)": np.concatenate([regulatory[test]] * 2, axis=1),
            "ridge (PAIRED ceiling)": Ridge(alpha=RIDGE_ALPHA).fit(
                chromatin[train], target[train]).predict(chromatin[test]).astype(np.float32)}
        for name, prediction in predictions.items():
            for block, span in ctx.blocks():
                piece = np.ascontiguousarray(prediction[:, span])
                truth = np.ascontiguousarray(observed[:, span])
                metrics = runner.state_metrics(torch.as_tensor(piece), torch.as_tensor(truth),
                                               torch.as_tensor(sample))
                rows.append({"transform": transform, "method": name, "block": block,
                             "foscttm": metrics["foscttm"],
                             "permuted_floor": permuted_pairing_floor(piece[sample],
                                                                      truth[sample]),
                             "gene_pearson_median": metrics["state_pearson_median"]})
                print(f"  {transform:12s} {name:24s} {block}  FOSCTTM "
                      f"{rows[-1]['foscttm']:.4f}  r "
                      f"{rows[-1]['gene_pearson_median']:+.4f}", flush=True)
        del chromatin, regulatory
    frame = pd.DataFrame(rows)
    write(frame, out, f"r2_supervised_{ctx.dataset}.csv")
    print(frame.pivot_table(index=["transform", "method"], columns="block",
                            values=["foscttm", "gene_pearson_median"]).round(4).to_string())
    return frame


def scvelo(ctx: Context, names: list[str], transforms: list[str],
           out: Path | None) -> pd.DataFrame:
    """Model du/dt and ds/dt against scVelo's, and the paired ceiling for each block."""
    references = {"u": runner.reference_velocity(ctx.adata, "velocity_scvelo_u"),
                  "s": runner.reference_velocity(ctx.adata, "velocity_scvelo")}
    fields = {t: runner.load_velocity(ctx.dataset, ctx.split_seed, t) for t in transforms}
    rows = []

    for name in names:
        transform = run_transform(name)
        field = fields[transform]
        test = ctx.rows("test", dynamic=field["dynamic_mask"])
        model, payload, config = load_run(name)
        # Refuse a run scored on a different gene panel.
        if list(payload["gene_names"]) != list(ctx.adata.var_names[ctx.columns]):
            raise ValueError(
                f"{name} was trained on {len(payload['gene_names'])} output genes but this "
                f"context has {ctx.n_genes}; they are not comparable")
        chromatin = torch.as_tensor(chromatin_features(ctx.adata, transform))
        velocity = torch.as_tensor(np.asarray(field["velocity"], dtype=np.float32))
        probe = torch.as_tensor(test)
        moving, pushed = torch.func.jvp(model.phi, (chromatin[probe],), (velocity[probe],))
        with torch.no_grad():
            regulatory = chromatin[probe] @ model.phi.gene_projection[:ctx.n_genes].T
            rhs = relay_rhs(moving, model.kappa(chromatin[probe]), model.g(regulatory),
                            model.beta, model.gamma)
        moving_np = moving.detach().numpy()
        for quantity, values in [("J_phi(c) v_c", pushed), ("law RHS", rhs)]:
            for block, take in [("u", runner.unspliced_block), ("s", runner.spliced_block)]:
                if references[block] is None:
                    continue
                metrics = runner.reference_agreement(
                    take(values.detach().numpy(), runner.RELAY),
                    take(moving_np, runner.RELAY), references[block], probe, ctx.columns, "x")
                if not metrics:
                    continue
                rows.append({"source": name, "transform": transform,
                             "lambda_dyn": config["lambda_dyn"]
                             if config["condition"] != "noDyn" else 0.0,
                             "quantity": quantity, "block": block,
                             "cosine_centred": metrics["x_cosine_centred_median"],
                             "null_p95": metrics["x_cosine_centred_null_p95"],
                             "gene_pearson": metrics["x_gene_pearson_median"],
                             "n_genes": metrics["x_n_genes"]})
                print(f"  {name:40s} {quantity:14s} {block}  centred "
                      f"{rows[-1]['cosine_centred']:+.5f}  null "
                      f"{rows[-1]['null_p95']:+.5f}", flush=True)
        del model, chromatin

    for transform in transforms:
        chromatin = chromatin_features(ctx.adata, transform)
        field = fields[transform]
        train = ctx.rows("train", dynamic=field["dynamic_mask"])
        test = ctx.rows("test", dynamic=field["dynamic_mask"])
        raw = np.asarray(field["velocity"], dtype=np.float32)
        for source, matrix in [("ridge on c", chromatin), ("ridge on v_c", raw)]:
            for block, reference in references.items():
                if reference is None:
                    continue
                target = reference[:, ctx.columns]
                covered = np.flatnonzero(np.abs(target).sum(axis=0) > 0)
                predicted = Ridge(alpha=RIDGE_ALPHA).fit(
                    matrix[train], target[train][:, covered]).predict(matrix[test])
                metrics = runner.task_d_kinetics(predicted.astype(np.float32),
                                                 target[test][:, covered])
                rows.append({"source": f"{source} (PAIRED)", "transform": transform,
                             "lambda_dyn": float("nan"), "quantity": "PAIRED ceiling",
                             "block": block,
                             "cosine_centred": metrics["cell_cosine_centred_median"],
                             "null_p95": metrics["cell_cosine_centred_null_p95"],
                             "gene_pearson": metrics["gene_pearson_median"],
                             "n_genes": len(covered)})
                print(f"  {source + ' [' + transform + ']':40s} {'PAIRED':14s} {block}  "
                      f"centred {metrics['cell_cosine_centred_median']:+.5f}  null "
                      f"{metrics['cell_cosine_centred_null_p95']:+.5f}", flush=True)
        del chromatin
    frame = pd.DataFrame(rows).assign(margin=lambda d: d.cosine_centred - d.null_p95)
    write(frame, out, f"r2_scvelo_{ctx.dataset}.csv")
    print(frame.round(5).to_string(index=False))
    return frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("normalisation", "runs", "phi", "supervised", "scvelo"):
        child = sub.add_parser(name)
        child.add_argument("--out", type=Path, default=None)
        if name in ("normalisation", "supervised", "scvelo"):
            child.add_argument("--dataset", choices=runner.DATASETS, default="bmmc")
            child.add_argument("--split-seed", type=int, default=0)
        if name in ("supervised", "scvelo"):
            child.add_argument("--transforms", nargs="+", default=CHROMATIN_TRANSFORMS,
                               choices=CHROMATIN_TRANSFORMS)
        if name in ("runs", "phi", "scvelo"):
            child.add_argument("--runs", nargs="+", required=True,
                               help="run-directory names under cache/chromatin/runs")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "runs":
        runs(args.runs, args.out)
    elif args.command == "phi":
        phi(args.runs, args.out)
    else:
        ctx = context(args.dataset, args.split_seed)
        if args.command == "normalisation":
            normalisation(ctx, args.out)
        elif args.command == "supervised":
            supervised(ctx, args.transforms, args.out)
        else:
            scvelo(ctx, args.runs, args.transforms, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
