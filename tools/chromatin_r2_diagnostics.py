#!/usr/bin/env python3
"""Diagnostics for the regulatory R2 runs: input, phi, rates, and agreement with scVelo.

One tool because they share dataset, split, gene panel and G.

Usage:
  python tools/chromatin_r2_diagnostics.py normalisation --dataset bmmc
  python tools/chromatin_r2_diagnostics.py supervised --dataset bmmc --eval-split val
  python tools/chromatin_r2_diagnostics.py scvelo --dataset bmmc --eval-split val --runs ...
  python tools/chromatin_r2_diagnostics.py runs --runs r2lsi_bmmc_relay_full_seed42 ...
  python tools/chromatin_r2_diagnostics.py manifest
  python tools/chromatin_r2_diagnostics.py summary
"""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge

from tools.chromatin_align_dims import DEFAULT_GEOMETRY, blur_is_usable
from tools.chromatin_common import (
    Context, centred_norm, context, gate_verdict, load_run, preflight,
    run_transform,
)

import run_kot_chromatin as runner
from src.data.chromatin import (
    CHROMATIN_TRANSFORMS, INPUT_SCREEN_TRANSFORMS, chromatin_features,
)
from src.evaluation.foscttm import permuted_pairing_floor
from src.losses.chromatin_laws import relay_rhs
from src.utils.io import load_yaml

RIDGE_ALPHA = 100.0
RESULTS = Path("cache/results/chromatin")
# 51 screen + 16 kinetic + 2 RNA-coord + the dim ladder (blur trains only
# when geometry left the training blur). Velocity lines in the screen file
# are ignored.
DEV_JOB_FILES = (
    Path("jobs/jobs_chromatin_input_lambda_screen.txt"),
    Path("jobs/jobs_chromatin_kinetic_controls.txt"),
    Path("jobs/jobs_chromatin_rna_coords.txt"),
    Path("jobs/jobs_chromatin_align_dims.txt"),
)
GEOMETRY_COLUMNS = [
    "oracle_advantage", "blur_over_pair_distance", "oracle_beats_constant",
    "explained_variance", "neighbour_preservation", "blur_usable",
]
MANIFEST_COLUMNS = [
    # `condition` is not recoverable from the other columns: a full arm and its
    # shuffle control share every one of them, differing only in whether the
    # velocity rows were permuted. Without it a 3-arm manifest has duplicate rows.
    "run_id", "condition", "map_transform", "regulatory_transform", "rna_kinetic_coords",
    "lambda_dyn", "stabilization", "kappa_mode", "phi_gate",
    "lambda_held_block", "lambda_gamma_anchor", "align_dims", "sinkhorn_blur",
    "seed", "status", *GEOMETRY_COLUMNS,
]
SUMMARY_COLUMNS = [
    "run_id", "checkpoint", "align_dims", "sinkhorn_blur",
    "val_s_foscttm", "val_u_foscttm", "val_gene_pearson", "spread",
    "partner_diversity", "jvp_scvelo_centred", "jvp_scvelo_null_p95",
    "jvp_scvelo_margin", "jvp_scvelo_gene_pearson", "jvp_rhs_cosine",
    "residual_norm", "grad_mag_ratio", "grad_cosine", "kappa_median",
    "kappa_floor_fraction", "alpha_floor_fraction", "gamma_anchor_error",
    "best_epoch", "final_epoch", *GEOMETRY_COLUMNS,
]
CHECKPOINTS = ("best_align", "final")
# ChromatinKOT.gamma: softplus(gamma_raw) + 1e-6. Keep the table dump off the
# 1.1G dataset by reading the checkpoint tensor directly.
GAMMA_OFFSET = 1e-6


def write(frame: pd.DataFrame, out: Path | None, default: str) -> None:
    destination = out or (RESULTS / default)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    print(f"\n[diagnostics] wrote {destination}")


def iter_variants(ctx: Context):
    """Yield (name, c) one gauge at a time so BMMC does not hold every matrix at once.

    Named transforms go through `chromatin_features` so TF-IDF/SVD fit on ATAC train only.
    `raw_counts` and `zscore_per_gene` are diagnostic controls, not KOT inputs.
    `zscore_per_gene` must come back identical under a per-gene rescaling or the screen is broken.
    """
    counts = ctx.adata.obsm["gene_activity_counts"]
    current = chromatin_features(ctx.adata, "as_is")
    yield "as_is", current
    for name in INPUT_SCREEN_TRANSFORMS:
        if name == "as_is":
            continue
        yield name, chromatin_features(ctx.adata, name)
    if "batch" in ctx.adata.obs:
        yield "tfidf_lsi_batch", chromatin_features(ctx.adata, "tfidf_lsi_batch")
    raw = runner.to_dense(counts, np.float32)
    yield "raw_counts", raw
    standardised = (current - current.mean(0)) / current.std(0).clip(min=1e-6)
    yield "zscore_per_gene (control)", standardised.astype(np.float32)


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
    for name, values in iter_variants(ctx):
        chromatin = torch.as_tensor(np.asarray(values, dtype=np.float32))
        regulatory = chromatin @ projection.T
        scale, bias = runner.gene_affine_calibration(regulatory, target, source, rna,
                                                     runner.RELAY)
        prediction = torch.cat([regulatory[validation]] * 2, 1) * scale + bias
        confound = depth_alignment(np.asarray(values)[validation], depth)
        # Training's pairing is decided on s; u is reported beside it, not instead of it.
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
        del chromatin, regulatory, prediction, values
    frame = pd.DataFrame(rows)
    write(frame, out, f"r2_normalisation_{ctx.dataset}.csv")
    print(frame.pivot_table(index="transform", columns="block",
                            values=["foscttm", "gene_pearson_median"]).round(4).to_string())
    return frame


def regulatory_input(model, chromatin: torch.Tensor, n_genes: int,
                     projection: np.ndarray | None) -> torch.Tensor:
    """Gc, the alpha head's input. `--phi-affine none` leaves phi without a stored G, so
    the gene map's own projection stands in -- it is the same matrix training built."""
    if hasattr(model.phi, "gene_projection"):
        return chromatin @ model.phi.gene_projection[:n_genes].T
    if projection is None:
        raise ValueError("phi has no gene_projection and no fallback was supplied")
    return chromatin @ torch.as_tensor(projection[:n_genes]).T


def gamma_anchor_error(gamma: torch.Tensor, anchors: dict | None) -> float:
    """Median |γ / γ_anchor − 1| × 100, the same number `runs()` already reports."""
    if not anchors or not anchors.get("indices"):
        return float("nan")
    relative = (gamma.detach()[torch.as_tensor(anchors["indices"], device=gamma.device)]
                / gamma.new_tensor(anchors["gamma"]))
    return float(((relative - 1).abs() * 100).median())


def rate_health(model, config, chromatin: torch.Tensor, n_genes: int,
                projection: np.ndarray | None = None,
                regulatory: torch.Tensor | None = None) -> dict:
    rows = torch.tensor(np.sort(np.random.default_rng(0).choice(len(chromatin), 512, False)))
    alpha_source = chromatin if regulatory is None else regulatory
    with torch.no_grad():
        kappa = model.kappa(chromatin[rows])
        production = regulatory_input(model, alpha_source[rows], n_genes, projection)
        alpha = model.g(runner.alpha_features(model, alpha_source[rows], production))
    return {"kappa_median": float(kappa.median()),
            # Pinned bounds can drive the RHS to zero without explaining the data.
            "kappa_at_floor": float((kappa < 1.1e-3).float().mean()),
            "alpha_at_floor": float((alpha < 1.1e-5).float().mean()),
            "anchor_error_percent": gamma_anchor_error(model.gamma, config.get("gamma_anchors"))}


def runs(names: list[str], out: Path | None) -> pd.DataFrame:
    rows = []
    # Cache by (dataset, split, transform) because the object is large and shared.
    inputs: dict = {}
    for name in names:
        checks = preflight(name)
        model, payload, config = load_run(name)
        history = pd.read_csv(Path("cache/chromatin/runs") / name / "training_loss.csv")
        logged = history_at(history, payload.get("epoch"))
        key = (payload["dataset"], config["split_seed"],
               config.get("chromatin_transform", "as_is"))
        if key not in inputs:
            built = context(*key)
            inputs[key] = (built.n_genes, torch.as_tensor(built.chromatin()),
                           built.projection)
            del built
        n_genes, chromatin, projection = inputs[key]
        regulatory = None
        if config.get("regulatory_transform", "same") not in (None, "same"):
            regulatory = torch.as_tensor(runner.regulatory_features(
                runner.load_dataset(payload["dataset"], config["split_seed"]),
                config.get("chromatin_transform", "as_is"),
                config["regulatory_transform"],
                map_features=chromatin.numpy()))
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
            "foscttm_s": missing_get(checks, "foscttm"),
            "permuted_floor": missing_get(checks, "foscttm_permuted_floor"),
            "foscttm_u": missing_get(checks, "unspliced_foscttm"),
            "gene_pearson": missing_get(checks, "state_pearson_median"),
            "spread": missing_get(checks, "prediction_spread_ratio"),
            "jvp_vs_scvelo_centred": missing_get(
                checks, "jvp_vs_reference_cosine_centred_median"),
            "jvp_null_p95": missing_get(
                checks, "jvp_vs_reference_cosine_centred_null_p95"),
            "jvp_rhs_cosine": missing_get(checks, "jvp_rhs_cosine_median"),
            "residual_norm": missing_get(checks, "residual_norm_median"),
            **{k: float(logged[k]) if k in logged and pd.notna(logged[k]) else float("nan")
               for k in ("loss_align", "val_align", "loss_dyn",
                         "loss_held_block", "grad_mag_ratio", "grad_cosine")},
            **rate_health(model, config, chromatin, n_genes, projection,
                          regulatory=regulatory)})
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
        if not hasattr(model.phi, "gene_projection"):
            rows.append({"run": name, "phi_gate": config.get("phi_gate", "?"), "block": "u",
                         "network_over_prediction": 1.0, "affine_over_prediction": 0.0,
                         "gate_mean": float("nan"), "gate_min": float("nan"),
                         "gate_max": float("nan"),
                         "gate_separation_failed_minus_worked": float("nan"),
                         "n_uncalibrated_outputs": 0})
            print(f"  {name:44s} no affine path (--phi-affine none): the network IS phi",
                  flush=True)
            del model
            continue
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


def supervised(ctx: Context, transforms: list[str], out: Path | None,
               eval_split: str = "val") -> pd.DataFrame:
    """Paired ceilings for the state map, per transform."""
    target = ctx.targets()
    train, scored = ctx.rows("train"), ctx.rows(eval_split)
    observed = target[scored]
    sample = np.sort(np.random.default_rng(0).choice(len(scored), min(2000, len(scored)), False))
    rows = []
    for transform in transforms:
        if transform == "tfidf_lsi_batch" and "batch" not in ctx.adata.obs:
            print(f"  SKIP {transform}: no batch column", flush=True)
            continue
        chromatin = chromatin_features(ctx.adata, transform)
        regulatory = chromatin @ ctx.projection.T
        predictions = {
            "mean_only (floor)": np.tile(target[train].mean(axis=0), (len(scored), 1)),
            "identity (Gc, no model)": np.concatenate([regulatory[scored]] * 2, axis=1),
            "ridge (PAIRED ceiling)": Ridge(alpha=RIDGE_ALPHA).fit(
                chromatin[train], target[train]).predict(chromatin[scored]).astype(np.float32)}
        for name, prediction in predictions.items():
            for block, span in ctx.blocks():
                piece = np.ascontiguousarray(prediction[:, span])
                truth = np.ascontiguousarray(observed[:, span])
                metrics = runner.state_metrics(torch.as_tensor(piece), torch.as_tensor(truth),
                                               torch.as_tensor(sample))
                rows.append({"transform": transform, "method": name, "block": block,
                             "eval_split": eval_split,
                             "foscttm": metrics["foscttm"],
                             "permuted_floor": permuted_pairing_floor(piece[sample],
                                                                      truth[sample]),
                             "gene_pearson_median": metrics["state_pearson_median"]})
                print(f"  {transform:12s} {name:24s} {block}  FOSCTTM "
                      f"{rows[-1]['foscttm']:.4f}  r "
                      f"{rows[-1]['gene_pearson_median']:+.4f}", flush=True)
        del chromatin, regulatory
    frame = pd.DataFrame(rows)
    write(frame, out, f"r2_supervised_{ctx.dataset}_{eval_split}.csv")
    print(frame.pivot_table(index=["transform", "method"], columns="block",
                            values=["foscttm", "gene_pearson_median"]).round(4).to_string())
    return frame


def try_load_velocity(dataset: str, split_seed: int, transform: str):
    """Return the cached field if it exists; diagnostics must not crash a missing gauge."""
    for protocol in (4, 3):
        path = runner.velocity_path(dataset, split_seed, transform, protocol=protocol)
        if path.exists():
            return runner.load_velocity(dataset, split_seed, transform, protocol=protocol)
    print(f"  no velocity cache for {transform!r}; ridge-on-v_c / JVP skipped for that gauge",
          flush=True)
    return None


def scvelo(ctx: Context, names: list[str], transforms: list[str],
           out: Path | None, eval_split: str = "val") -> pd.DataFrame:
    """Model du/dt and ds/dt against scVelo's, and the paired ceiling for each block."""
    references = {"u": runner.reference_velocity(ctx.adata, "velocity_scvelo_u"),
                  "s": runner.reference_velocity(ctx.adata, "velocity_scvelo")}
    fields = {t: try_load_velocity(ctx.dataset, ctx.split_seed, t) for t in transforms}
    rows = []

    for name in names:
        transform = run_transform(name)
        field = fields.get(transform) or try_load_velocity(ctx.dataset, ctx.split_seed, transform)
        if field is None:
            print(f"  SKIP {name}: no velocity cache for {transform}", flush=True)
            continue
        scored = ctx.rows(eval_split, dynamic=field["dynamic_mask"])
        model, payload, config = load_run(name)
        if list(payload["gene_names"]) != list(ctx.adata.var_names[ctx.columns]):
            print(f"  SKIP {name}: gene panel differs from this context", flush=True)
            del model
            continue
        chromatin_np = chromatin_features(ctx.adata, transform)
        regulatory_np = runner.regulatory_features(
            ctx.adata, transform, config.get("regulatory_transform", "same"),
            map_features=chromatin_np)
        chromatin = torch.as_tensor(chromatin_np)
        regulatory = torch.as_tensor(regulatory_np)
        velocity = torch.as_tensor(np.asarray(field["velocity"], dtype=np.float32))
        probe = torch.as_tensor(scored)
        moving, pushed = torch.func.jvp(model.phi, (chromatin[probe],), (velocity[probe],))
        with torch.no_grad():
            if hasattr(model.phi, "gene_projection"):
                production = regulatory[probe] @ model.phi.gene_projection[:ctx.n_genes].T
            else:
                production = regulatory[probe] @ torch.as_tensor(ctx.projection[:ctx.n_genes]).T
            alpha = model.g(runner.alpha_features(model, regulatory[probe], production))
            rhs = relay_rhs(moving, model.kappa(chromatin[probe]), alpha,
                            model.beta, model.gamma,
                            coords=config.get("rna_kinetic_coords"))
        moving_np = moving.detach().numpy()
        for quantity, values in [("J_phi(c) v_c", pushed), ("law RHS", rhs)]:
            for block, take in [("u", runner.unspliced_block), ("s", runner.spliced_block)]:
                if references[block] is None:
                    continue
                metrics = runner.reference_agreement(
                    take(values.detach().numpy(), runner.RELAY),
                    take(moving_np, runner.RELAY), references[block], probe, ctx.columns, "x",
                    coords=config.get("rna_kinetic_coords"))
                if not metrics:
                    continue
                rows.append({"source": name, "transform": transform, "eval_split": eval_split,
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
        del model, chromatin, regulatory

    for transform in transforms:
        if transform == "tfidf_lsi_batch" and "batch" not in ctx.adata.obs:
            print(f"  SKIP {transform}: no batch column", flush=True)
            continue
        chromatin = chromatin_features(ctx.adata, transform)
        field = fields[transform]
        matrices = [("ridge on c", chromatin)]
        if field is not None:
            matrices.append(("ridge on v_c", np.asarray(field["velocity"], dtype=np.float32)))
            train = ctx.rows("train", dynamic=field["dynamic_mask"])
            scored = ctx.rows(eval_split, dynamic=field["dynamic_mask"])
        else:
            train = ctx.rows("train")
            scored = ctx.rows(eval_split)
        for source, matrix in matrices:
            for block, reference in references.items():
                if reference is None:
                    continue
                target = reference[:, ctx.columns]
                covered = np.flatnonzero(np.abs(target).sum(axis=0) > 0)
                predicted = Ridge(alpha=RIDGE_ALPHA).fit(
                    matrix[train], target[train][:, covered]).predict(matrix[scored])
                metrics = runner.task_d_kinetics(predicted.astype(np.float32),
                                                 target[scored][:, covered])
                rows.append({"source": f"{source} (PAIRED)", "transform": transform,
                             "eval_split": eval_split,
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
    write(frame, out, f"r2_scvelo_{ctx.dataset}_{eval_split}.csv")
    print(frame.round(5).to_string(index=False))
    return frame


def yaml_sinkhorn_blur(config_path: Path = Path("config/training.yaml")) -> float:
    """Job lines omit --sinkhorn-blur; train fills it from training.yaml `sinkhorn_reg`."""
    defaults = load_yaml(config_path).get("defaults", {})
    return float(defaults.get("sinkhorn_reg", 0.1))


def train_lines(path: Path) -> list[str]:
    return [line for line in path.read_text().splitlines() if line.startswith("train ")]


def parse_train_line(line: str) -> argparse.Namespace:
    """Reuse the chromatin trainer parser so job-file flags stay in one place."""
    return runner.build_parser().parse_args(shlex.split(line))


def kappa_mode(args: argparse.Namespace | dict) -> str:
    """How the clock was constrained: a constant, a pinned box, or the open r2lsi box."""
    get = args.get if isinstance(args, dict) else lambda key, default=None: getattr(args, key, default)
    if get("fixed_kappa") is not None:
        return "fixed"
    kmin, kmax = get("kappa_min"), get("kappa_max")
    if kmin is not None and kmax is not None and float(kmin) == float(kmax):
        return "pinned"
    return "open"


def run_status(run_dir: Path) -> str:
    if not (run_dir / "checkpoint_best_align.pt").exists():
        if run_dir.exists() and any(run_dir.iterdir()):
            return "incomplete"
        return "missing"
    if (run_dir / "preflight_passed.json").exists():
        return "preflight_passed"
    if (run_dir / "preflight.json").exists() or (run_dir / "preflight_best_align.json").exists():
        return "preflight_failed"
    return "trained"


def intended_row(line: str, sinkhorn_blur: float | None = None) -> dict:
    args = parse_train_line(line)
    run_dir = Path(args.run_dir)
    config = {}
    config_path = run_dir / "run_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
    lambda_dyn = 0.0 if args.condition == "noDyn" else args.lambda_dyn
    if lambda_dyn is None:
        lambda_dyn = config.get("lambda_dyn", 0.0)
    blur = args.sinkhorn_blur
    if blur is None:
        blur = config.get("sinkhorn_blur", sinkhorn_blur if sinkhorn_blur is not None else 0.1)
    return {
        "run_id": run_dir.name,
        "condition": args.condition,
        "map_transform": args.chromatin_transform,
        "regulatory_transform": args.regulatory_transform,
        "rna_kinetic_coords": args.rna_kinetic_coords,
        "lambda_dyn": lambda_dyn,
        "stabilization": args.stabilization,
        "kappa_mode": kappa_mode(args),
        "phi_gate": args.phi_gate,
        "lambda_held_block": args.lambda_held_block,
        "lambda_gamma_anchor": args.lambda_gamma_anchor,
        "align_dims": args.align_dims if args.align_dims is not None else config.get("align_dims", 32),
        "sinkhorn_blur": blur,
        "seed": args.seed if args.seed is not None else config.get("seed"),
        "status": run_status(run_dir),
    }


def load_preflight(run_dir: Path, checkpoint: str) -> dict:
    """Checkpoint-tagged gate file. Missing `final` is empty, not the launch gate.

    Do not route campaign tables through `chromatin_common.preflight`: that helper
    prefers `preflight_passed.json` and raises if `final` has not been scored.
    """
    tagged = run_dir / f"preflight_{checkpoint}.json"
    if tagged.exists():
        return json.loads(tagged.read_text())
    if checkpoint != "best_align":
        return {}
    untagged = run_dir / "preflight.json"
    if untagged.exists():
        return json.loads(untagged.read_text())
    passed = run_dir / "preflight_passed.json"
    if passed.exists():
        return json.loads(passed.read_text())
    return {}


def missing_get(checks: dict, key: str) -> float:
    value = checks.get(key)
    if value is None:
        return float("nan")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")


def history_at(history: pd.DataFrame | None, epoch) -> dict:
    if history is None or history.empty or epoch is None or not np.isfinite(epoch):
        return {}
    if "epoch" not in history.columns:
        return {}
    delta = (history["epoch"] - epoch).abs()
    idx = delta.idxmin()
    # Nearest-row without a bound would pin noDyn epoch 0 onto the first eval.
    if not np.isfinite(delta.loc[idx]) or float(delta.loc[idx]) >= 0.5:
        return {}
    return history.loc[idx].to_dict()


def checkpoint_probe(run_dir: Path, checkpoint: str) -> dict:
    """Epoch and gamma-anchor error without loading the Multiome object."""
    path = run_dir / f"checkpoint_{checkpoint}.pt"
    empty = {"epoch": float("nan"), "gamma_anchor_error": float("nan")}
    if not path.exists():
        return empty
    payload = torch.load(path, map_location="cpu", weights_only=False)
    try:
        config_path = run_dir / "run_config.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        anchors = payload.get("gamma_anchors") or config.get("gamma_anchors")
        state = payload.get("state_dict") or {}
        epoch = payload.get("epoch", float("nan"))
        if "gamma_raw" not in state:
            return {"epoch": epoch, "gamma_anchor_error": float("nan")}
        gamma = torch.nn.functional.softplus(state["gamma_raw"]) + GAMMA_OFFSET
        return {"epoch": epoch, "gamma_anchor_error": gamma_anchor_error(gamma, anchors)}
    finally:
        del payload


def summary_rows(run_dir: Path, history: pd.DataFrame | None = None) -> list[dict]:
    if history is None:
        loss_path = run_dir / "training_loss.csv"
        history = pd.read_csv(loss_path) if loss_path.exists() else pd.DataFrame()
    probes = {name: checkpoint_probe(run_dir, name) for name in CHECKPOINTS}
    rows = []
    for checkpoint in CHECKPOINTS:
        checks = load_preflight(run_dir, checkpoint)
        logged = history_at(history, probes[checkpoint]["epoch"])
        centred = missing_get(checks, "jvp_vs_reference_cosine_centred_median")
        p95 = missing_get(checks, "jvp_vs_reference_cosine_centred_null_p95")
        rows.append({
            "run_id": run_dir.name,
            "checkpoint": checkpoint,
            "val_s_foscttm": missing_get(checks, "foscttm"),
            "val_u_foscttm": missing_get(checks, "unspliced_foscttm"),
            "val_gene_pearson": missing_get(checks, "state_pearson_median"),
            "spread": missing_get(checks, "prediction_spread_ratio"),
            "partner_diversity": missing_get(checks, "partner_diversity"),
            "jvp_scvelo_centred": centred,
            "jvp_scvelo_null_p95": p95,
            "jvp_scvelo_margin": centred - p95,
            "jvp_scvelo_gene_pearson": missing_get(
                checks, "jvp_vs_reference_gene_pearson_median"),
            "jvp_rhs_cosine": missing_get(checks, "jvp_rhs_cosine_median"),
            "residual_norm": missing_get(checks, "residual_norm_median"),
            "grad_mag_ratio": logged.get("grad_mag_ratio", float("nan")),
            "grad_cosine": logged.get("grad_cosine", float("nan")),
            "kappa_median": missing_get(checks, "kappa_median"),
            "kappa_floor_fraction": missing_get(checks, "kappa_at_floor"),
            "alpha_floor_fraction": missing_get(checks, "alpha_at_floor"),
            "gamma_anchor_error": probes[checkpoint]["gamma_anchor_error"],
            "best_epoch": probes["best_align"]["epoch"],
            "final_epoch": probes["final"]["epoch"],
        })
    return rows


def load_geometry_table(path: Path | None = None) -> pd.DataFrame:
    """Spliced-target Sinkhorn geometry. Missing file is empty, not a crash."""
    geometry = path or DEFAULT_GEOMETRY
    if geometry is None or not geometry.exists():
        return pd.DataFrame()
    return pd.read_csv(geometry)


def geometry_lookup(align_dims, blur, geometry: pd.DataFrame) -> dict:
    """One (rank, blur) row. Missing settings stay NaN, not 0."""
    empty = {name: float("nan") for name in GEOMETRY_COLUMNS}
    if geometry is None or geometry.empty or align_dims is None or blur is None:
        return empty
    try:
        dims = int(align_dims)
        sigma = float(blur)
    except (TypeError, ValueError):
        return empty
    if not np.isfinite(dims) or not np.isfinite(sigma):
        return empty
    match = geometry[(geometry["align_dims"] == dims)
                     & np.isclose(geometry["blur"].astype(float), sigma)]
    if match.empty:
        return empty
    if len(match) != 1:
        raise ValueError(
            f"geometry CSV has {len(match)} rows for align_dims={dims} blur={sigma:g}")
    row = match.iloc[0]
    return {
        "oracle_advantage": float(row["oracle_advantage"]),
        "blur_over_pair_distance": float(row["blur_over_pair_distance"]),
        "oracle_beats_constant": bool(row["oracle_beats_constant"]),
        "explained_variance": float(row["explained_variance"]),
        "neighbour_preservation": float(row["neighbour_preservation"]),
        "blur_usable": bool(blur_is_usable(row)),
    }


def attach_geometry(frame: pd.DataFrame, geometry: pd.DataFrame | None = None) -> pd.DataFrame:
    """Join blur geometry onto run tables by (align_dims, sinkhorn_blur)."""
    table = load_geometry_table() if geometry is None else geometry
    out = frame.copy()
    if out.empty:
        for name in GEOMETRY_COLUMNS:
            out[name] = pd.Series(dtype=float)
        return out
    records = [geometry_lookup(row.get("align_dims"), row.get("sinkhorn_blur"), table)
               for row in out.to_dict(orient="records")]
    for name in GEOMETRY_COLUMNS:
        out[name] = [record[name] for record in records]
    return out


def intended_runs(job_files: list[Path], sinkhorn_blur: float | None = None,
                  geometry: pd.DataFrame | None = None) -> pd.DataFrame:
    blur = yaml_sinkhorn_blur() if sinkhorn_blur is None else sinkhorn_blur
    rows = [intended_row(line, sinkhorn_blur=blur)
            for path in job_files for line in train_lines(path)]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=MANIFEST_COLUMNS)
    return attach_geometry(frame, geometry)[MANIFEST_COLUMNS]


def manifest(job_files: list[Path], out: Path | None,
             geometry: pd.DataFrame | None = None) -> pd.DataFrame:
    frame = intended_runs(job_files, geometry=geometry)
    write(frame, out, "r2_final_dev_manifest.csv")
    counts = frame["status"].value_counts().to_dict()
    print(f"{len(frame)} intended runs: {counts}", flush=True)
    return frame


def summary(job_files: list[Path], out: Path | None,
            geometry: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    blur = yaml_sinkhorn_blur()
    for path in job_files:
        for line in train_lines(path):
            identity = intended_row(line, sinkhorn_blur=blur)
            run_dir = Path(parse_train_line(line).run_dir)
            print(f"  {run_dir.name}", flush=True)
            for row in summary_rows(run_dir):
                row["align_dims"] = identity["align_dims"]
                row["sinkhorn_blur"] = identity["sinkhorn_blur"]
                rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    frame = attach_geometry(frame, geometry)[SUMMARY_COLUMNS]
    write(frame, out, "r2_final_dev_summary.csv")
    n_jvp = int(frame["jvp_scvelo_centred"].notna().sum())
    print(f"{len(frame)} rows ({frame['run_id'].nunique()} runs × {len(CHECKPOINTS)} "
          f"checkpoints); {n_jvp} have JVP vs scVelo", flush=True)
    return frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("normalisation", "runs", "phi", "supervised", "scvelo",
                 "manifest", "summary"):
        child = sub.add_parser(name)
        child.add_argument("--out", type=Path, default=None)
        if name in ("normalisation", "supervised", "scvelo"):
            child.add_argument("--dataset", choices=runner.DATASETS, default="bmmc")
            child.add_argument("--split-seed", type=int, default=0)
        if name in ("supervised", "scvelo"):
            child.add_argument("--eval-split", choices=["val", "test"], default="val",
                               help="paired cells; val for development, test once frozen")
            child.add_argument("--transforms", nargs="+", default=list(INPUT_SCREEN_TRANSFORMS),
                               choices=CHROMATIN_TRANSFORMS,
                               help="named chromatin_features gauges; default is the "
                                    "pre-sweep screen (cp10k_linear and tfidf_gene "
                                    "included). Pass tfidf_lsi_batch on BMMC.")
        if name in ("runs", "phi"):
            child.add_argument("--runs", nargs="+", required=True,
                               help="run-directory names under cache/chromatin/runs")
        if name == "scvelo":
            child.add_argument("--runs", nargs="*", default=[],
                               help="run-directory names under cache/chromatin/runs")
        if name in ("manifest", "summary"):
            child.add_argument("--jobs", nargs="+", type=Path, default=list(DEV_JOB_FILES),
                               help="job files whose `train` lines are the intended set")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "runs":
        runs(args.runs, args.out)
    elif args.command == "phi":
        phi(args.runs, args.out)
    elif args.command == "manifest":
        manifest(args.jobs, args.out)
    elif args.command == "summary":
        summary(args.jobs, args.out)
    else:
        ctx = context(args.dataset, args.split_seed)
        if args.command == "normalisation":
            normalisation(ctx, args.out)
        elif args.command == "supervised":
            supervised(ctx, args.transforms, args.out, args.eval_split)
        else:
            scvelo(ctx, args.runs, args.transforms, args.out, args.eval_split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
