#!/usr/bin/env python3
r"""Does chromatin lead transcription, and does that lead reach the model?

Five tests share one lead-score estimator so it cannot drift across copies.

Usage:
  python tools/chromatin_lead.py lag --dataset hspc
  python tools/chromatin_lead.py static-vs-jvp --dataset hspc --run-dir ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.stats import spearmanr, wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import run_kot_chromatin as chromatin
from src.data.chromatin import gene_map
from src.evaluation.chromatin_eval import centred_cosine, column_pearson, row_cosine
from src.losses.chromatin_laws import LOG1P_MAX, LOG1P_MIN, RELAY
from src.training.kot import choose_torch_device
from src.utils.arrays import to_dense

RESULTS = PROJECT_ROOT / "cache" / "results" / "chromatin"


def steady_state_slope(upstream: np.ndarray, downstream: np.ndarray) -> np.ndarray:
    """Per gene, the least-squares ratio k in downstream ~ k * upstream, through zero."""
    numerator = (upstream * downstream).sum(axis=0)
    denominator = (upstream * upstream).sum(axis=0)
    return numerator / np.clip(denominator, 1e-12, None)


def lead_score(upstream: np.ndarray, downstream: np.ndarray,
               rate: np.ndarray) -> np.ndarray:
    """Per gene, corr(k*upstream - downstream, d(downstream)/dt) across cells.

    The residual is positive where upstream sits above the steady-state line, which is where downstream should be rising if upstream leads.
    """
    residual = steady_state_slope(upstream, downstream) * upstream - downstream
    return column_pearson(residual, rate)


def report(name: str, score: np.ndarray, null: np.ndarray) -> dict:
    usable = np.isfinite(score) & np.isfinite(null)
    score, null = score[usable], null[usable]
    block = {
        "median_correlation": float(np.median(score)),
        "fraction_positive": float((score > 0).mean()),
        "median_null_correlation": float(np.median(null)),
        "n_genes": int(len(score)),
    }
    print(f"  {name:<10} corr {block['median_correlation']:+.4f}   "
          f"{block['fraction_positive']:.1%} of genes positive   "
          f"(cell-shuffled null {block['median_null_correlation']:+.4f})")
    return block


def lag(args: argparse.Namespace) -> int:


    adata = chromatin.load_dataset(args.dataset)
    covered = chromatin.target_cell_mask(adata, "spliced_lognorm")
    rate_s = chromatin.reference_velocity(adata, "velocity_scvelo")
    rate_u = chromatin.reference_velocity(adata, "velocity_scvelo_u")
    assert rate_s is not None and rate_u is not None, (
        f"run `reference --dataset {args.dataset}` first: both velocity_scvelo and "
        "velocity_scvelo_u are needed")
    scored = np.flatnonzero(covered & (np.abs(rate_s).sum(axis=1) > 0)
                            & (np.abs(rate_u).sum(axis=1) > 0))
    print(f"[leadlag] {args.dataset}: {len(scored)} cells carrying both reference rates",
          flush=True)

    activity = np.asarray(adata.obsm["gene_activity"], dtype=np.float32)[scored]
    unspliced = chromatin.to_dense(adata.layers["unspliced_lognorm"], np.float32)[scored]
    spliced = chromatin.to_dense(adata.layers["spliced_lognorm"], np.float32)[scored]
    rate_s, rate_u = rate_s[scored], rate_u[scored]

    _, _, kinetic_mask = gene_map(adata, chromatin.gene_map_path(args.dataset))
    detected = ((unspliced > 0).mean(axis=0) >= args.min_detected) \
        & ((spliced > 0).mean(axis=0) >= args.min_detected) \
        & ((activity > 0).mean(axis=0) >= args.min_detected) \
        & (np.abs(rate_s).sum(axis=0) > 0) & (np.abs(rate_u).sum(axis=0) > 0)
    genes = np.flatnonzero(detected & kinetic_mask)
    print(f"[leadlag] {len(genes)} genes in G, detected in >= {args.min_detected:.0%} "
          "of cells, and covered by both reference rates")
    activity, unspliced, spliced = activity[:, genes], unspliced[:, genes], spliced[:, genes]
    rate_s, rate_u = rate_s[:, genes], rate_u[:, genes]

    shuffle = np.random.default_rng(args.seed).permutation(len(scored))
    results = {"dataset": args.dataset, "n_cells": int(len(scored)),
               "n_genes": int(len(genes)), "pairs": {}}
    print("\n[leadlag] does the upstream species predict the downstream species' rate?")
    for label, (up, down, rate) in {
        "u -> s": (unspliced, spliced, rate_s),
        "c -> u": (activity, unspliced, rate_u),
        "c -> s": (activity, spliced, rate_s),
    }.items():
        results["pairs"][label] = report(
            label, lead_score(up, down, rate), lead_score(up, down, rate[shuffle]))

    print("\n[leadlag] u -> s is scVelo's own steady-state model scored against scVelo's "
          "own velocity: it must be clearly positive, or c -> u means nothing.")
    out = PROJECT_ROOT / "cache" / "results" / "chromatin" / f"lead_lag_{args.dataset}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"[leadlag] wrote {out}")
    return 0

def association(x: np.ndarray, y: np.ndarray) -> dict:
    """Pearson and Spearman across genes, with p-values and a gene bootstrap on Spearman."""
    usable = np.isfinite(x) & np.isfinite(y)
    x, y = x[usable], y[usable]
    n = int(len(x))
    if n < 5:
        return {"n_genes": n}
    pr = stats.pearsonr(x, y)
    sr = stats.spearmanr(x, y)
    rng = np.random.default_rng(0)
    boot = np.empty(500)
    for i in range(len(boot)):
        draw = rng.integers(0, n, n)
        boot[i] = stats.spearmanr(x[draw], y[draw]).statistic
    return {
        "n_genes": n,
        "pearson": float(pr.statistic), "pearson_p": float(pr.pvalue),
        "spearman": float(sr.statistic), "spearman_p": float(sr.pvalue),
        "spearman_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
    }

def partial_spearman(x: np.ndarray, y: np.ndarray, controls: np.ndarray) -> dict:
    """Spearman of x and y after regressing rank-transformed controls out of both."""
    usable = np.isfinite(x) & np.isfinite(y) & np.isfinite(controls).all(axis=1)
    if usable.sum() < 10:
        return {"n_genes": int(usable.sum())}
    rank = lambda v: stats.rankdata(v, axis=0)
    design = np.column_stack([np.ones(usable.sum()), rank(controls[usable])])
    rx, ry = rank(x[usable]), rank(y[usable])
    resid = lambda v: v - design @ np.linalg.lstsq(design, v, rcond=None)[0]
    pr = stats.pearsonr(resid(rx), resid(ry))
    return {"n_genes": int(usable.sum()), "partial_spearman": float(pr.statistic),
            "p": float(pr.pvalue)}


def link(args: argparse.Namespace) -> int:


    device = choose_torch_device({"device": "auto"})
    adata = chromatin.load_dataset(args.dataset)
    covered = chromatin.target_cell_mask(adata, "spliced_lognorm")
    rate_s = chromatin.reference_velocity(adata, "velocity_scvelo")
    rate_u = chromatin.reference_velocity(adata, "velocity_scvelo_u")
    assert rate_s is not None and rate_u is not None

    activity = np.asarray(adata.obsm["gene_activity"], dtype=np.float32)
    unspliced = chromatin.to_dense(adata.layers["unspliced_lognorm"], np.float32)
    spliced = chromatin.to_dense(adata.layers["spliced_lognorm"], np.float32)
    G, _, kinetic_mask = gene_map(adata, chromatin.gene_map_path(args.dataset))
    # G identity must be checked, not assumed.
    driver = np.full(adata.n_vars, -1)
    Gc = G.tocoo()
    driver[Gc.row] = Gc.col
    self_driven = int((driver[kinetic_mask] == np.flatnonzero(kinetic_mask)).sum())
    print(f"[link] G: {int(kinetic_mask.sum())} kinetic genes, {self_driven} driven by "
          f"their own activity column", flush=True)

    has_rate = (np.abs(rate_s).sum(axis=1) > 0) & (np.abs(rate_u).sum(axis=1) > 0)
    splits = pd.read_csv(chromatin.split_path(args.dataset, 0), index_col=0)
    split = splits.reindex(adata.obs_names)["split"].to_numpy()
    cell_sets = {
        "all": np.flatnonzero(covered & has_rate),
        "test": np.flatnonzero(covered & has_rate & (split == "test")),
    }
    print(f"[link] cells: " + ", ".join(f"{k} {len(v)}" for k, v in cell_sets.items()),
          flush=True)

    detected = ((unspliced > 0).mean(axis=0) >= args.min_detected) \
        & ((spliced > 0).mean(axis=0) >= args.min_detected) \
        & ((activity > 0).mean(axis=0) >= args.min_detected) \
        & (np.abs(rate_s).sum(axis=0) > 0) & (np.abs(rate_u).sum(axis=0) > 0)
    print(f"[link] detected genes {int(detected.sum())}, "
          f"detected & kinetic {int((detected & kinetic_mask).sum())}", flush=True)

    frame = pd.DataFrame(index=pd.Index(adata.var_names, name="gene"))
    frame["detected"] = detected
    frame["kinetic"] = kinetic_mask
    for tag, rows in cell_sets.items():
        c, u, s = activity[rows], unspliced[rows], spliced[rows]
        ru, rs = rate_u[rows], rate_s[rows]
        shuffle = np.random.default_rng(args.seed).permutation(len(rows))
        frame[f"lead_cu_{tag}"] = lead_score(c, u, ru)
        frame[f"lead_cu_null_{tag}"] = lead_score(c, u, ru[shuffle])
        frame[f"lead_us_{tag}"] = lead_score(u, s, rs)
        frame[f"lead_cs_{tag}"] = lead_score(c, s, rs)
        frame[f"static_cu_{tag}"] = column_pearson(c, u)
        frame[f"static_cs_{tag}"] = column_pearson(c, s)
    test_rows = cell_sets["test"]
    frame["u_mean_test"] = unspliced[test_rows].mean(axis=0)
    frame["u_sd_test"] = unspliced[test_rows].std(axis=0)
    frame["u_detect_test"] = (unspliced[test_rows] > 0).mean(axis=0)
    frame["c_sd_test"] = activity[test_rows].std(axis=0)

    del unspliced, spliced, rate_s, rate_u

    target_u = chromatin.to_dense(adata.layers["unspliced_lognorm"], np.float32)[test_rows]
    target_s = chromatin.to_dense(adata.layers["spliced_lognorm"], np.float32)[test_rows]
    runs = {}
    for run_dir in args.run_dirs:
        model, payload, config = chromatin.load_checkpoint(Path(run_dir), args.checkpoint,
                                                           device)
        assert payload["law"] == RELAY and payload["dataset"] == args.dataset
        columns = adata.var_names.get_indexer(payload["gene_names"])
        assert (columns >= 0).all()
        with torch.no_grad():
            predicted = model.phi(
                torch.as_tensor(activity[test_rows], device=device)).cpu().numpy()
        tag = Path(run_dir).name
        runs[tag] = {"condition": payload["condition"], "epoch": int(payload["epoch"])}
        for block, getter, observed in [
            ("u", chromatin.unspliced_block, target_u),
            ("s", chromatin.spliced_block, target_s),
        ]:
            gene_r = column_pearson(getter(predicted, RELAY), observed[:, columns])
            column = f"model_{block}_{payload['condition']}"
            values = np.full(adata.n_vars, np.nan)
            values[columns] = gene_r
            frame[column] = values
            runs[tag][f"gene_pearson_median_{block}"] = float(np.nanmedian(gene_r))
        print(f"[link] {tag} ({payload['condition']}, epoch {payload['epoch']}): "
              f"u median r {runs[tag]['gene_pearson_median_u']:+.4f}, "
              f"s median r {runs[tag]['gene_pearson_median_s']:+.4f}", flush=True)

    out_csv = PROJECT_ROOT / "cache" / "results" / "chromatin" / f"lead_link_{args.dataset}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_csv)

    conditions = sorted({v["condition"] for v in runs.values()})
    gene_sets = {
        "kinetic_detected": frame["detected"] & frame["kinetic"],
        "detected": frame["detected"],
    }
    results = {"dataset": args.dataset, "runs": runs,
               "n_cells": {k: int(len(v)) for k, v in cell_sets.items()},
               "associations": {}, "partials": {}, "run_agreement": {}}
    for set_name, mask in gene_sets.items():
        block = frame[mask]
        for condition in conditions:
            model = block[f"model_u_{condition}"].to_numpy()
            for predictor in ["lead_cu_all", "lead_cu_test", "lead_cu_null_all",
                              "lead_us_all", "lead_cs_all", "static_cu_test",
                              "static_cu_all", "u_sd_test", "u_detect_test"]:
                key = f"{set_name}|{condition}|{predictor}"
                results["associations"][key] = association(
                    block[predictor].to_numpy(), model)
            for predictor, controls in [
                ("lead_cu_all", ["static_cu_test", "u_sd_test", "u_detect_test"]),
                ("lead_cu_all", ["static_cu_test"]),
            ]:
                key = f"{set_name}|{condition}|{predictor}|ctrl:{'+'.join(controls)}"
                results["partials"][key] = partial_spearman(
                    block[predictor].to_numpy(), model,
                    block[controls].to_numpy())
        for a in conditions:
            for b in conditions:
                if a < b:
                    results["run_agreement"][f"{set_name}|{a} vs {b}"] = association(
                        block[f"model_u_{a}"].to_numpy(), block[f"model_u_{b}"].to_numpy())

    for set_name, mask in gene_sets.items():
        block = frame[mask]
        results.setdefault("lead_summary", {})[set_name] = {
            "n_genes": int(mask.sum()),
            **{col: {"median": float(np.nanmedian(block[col])),
                     "frac_positive": float(np.nanmean(block[col] > 0))}
               for col in ["lead_cu_all", "lead_cu_test", "lead_cu_null_all",
                           "lead_us_all", "static_cu_test"]},
            **{f"model_u_{c}": {"median": float(np.nanmedian(block[f"model_u_{c}"])),
                                "frac_positive": float(np.nanmean(block[f"model_u_{c}"] > 0))}
               for c in conditions},
        }

    out = Path(args.out) if args.out else (
        PROJECT_ROOT / "cache" / "results" / "chromatin" / f"lead_link_{args.dataset}.json")
    out.write_text(json.dumps(results, indent=2))

    print("\n[link] cross-gene association: predictor vs held-out u-block gene Pearson")
    print(f"{'gene set':<18}{'cond':<9}{'predictor':<18}{'n':>5}{'pearson':>10}"
          f"{'spearman':>10}{'p':>11}{'ci95':>20}")
    for key, block in results["associations"].items():
        if "n_genes" not in block or "spearman" not in block:
            continue
        set_name, condition, predictor = key.split("|")
        ci = f"[{block['spearman_ci95'][0]:+.3f},{block['spearman_ci95'][1]:+.3f}]"
        print(f"{set_name:<18}{condition:<9}{predictor:<18}{block['n_genes']:>5}"
              f"{block['pearson']:>+10.4f}{block['spearman']:>+10.4f}"
              f"{block['spearman_p']:>11.2e}{ci:>20}")
    print("\n[link] partial Spearman (lead vs model, controls regressed out of both)")
    for key, block in results["partials"].items():
        if "partial_spearman" not in block:
            continue
        print(f"  {key}  n={block['n_genes']}  "
              f"{block['partial_spearman']:+.4f}  p={block['p']:.2e}")
    print("\n[link] agreement between runs' per-gene u Pearson")
    for key, block in results["run_agreement"].items():
        if "spearman" not in block:
            continue
        print(f"  {key}  n={block['n_genes']}  pearson {block['pearson']:+.4f}  "
              f"spearman {block['spearman']:+.4f}")
    print(f"\n[link] wrote {out} and {out_csv}")
    return 0


def link_controls(args: argparse.Namespace) -> int:


    device = choose_torch_device({"device": "auto"})
    adata = chromatin.load_dataset(args.dataset)
    rate_u = chromatin.reference_velocity(adata, "velocity_scvelo_u")
    rate_s = chromatin.reference_velocity(adata, "velocity_scvelo")
    activity = np.asarray(adata.obsm["gene_activity"], dtype=np.float32)
    unspliced = chromatin.to_dense(adata.layers["unspliced_lognorm"], np.float32)
    spliced = chromatin.to_dense(adata.layers["spliced_lognorm"], np.float32)
    _, _, kinetic_mask = gene_map(adata, chromatin.gene_map_path(args.dataset))

    has_rate = (np.abs(rate_u).sum(axis=1) > 0) & (np.abs(rate_s).sum(axis=1) > 0)
    splits = pd.read_csv(chromatin.split_path(args.dataset, 0), index_col=0)
    split = splits.reindex(adata.obs_names)["split"].to_numpy()
    all_rows = np.flatnonzero(has_rate)
    test_rows = np.flatnonzero(has_rate & (split == "test"))
    print(f"[ctrl] cells: all {len(all_rows)}, test {len(test_rows)}", flush=True)

    c, u, s, ru = activity[all_rows], unspliced[all_rows], spliced[all_rows], rate_u[all_rows]
    rng = np.random.default_rng(args.seed)
    frame = pd.DataFrame(index=pd.Index(adata.var_names, name="gene"))
    frame["kinetic"] = kinetic_mask
    frame["u_detect"] = (u > 0).mean(axis=0)
    frame["s_detect"] = (s > 0).mean(axis=0)
    frame["c_detect"] = (c > 0).mean(axis=0)
    frame["has_rate_u"] = np.abs(ru).sum(axis=0) > 0
    frame["has_rate_s"] = np.abs(rate_s[all_rows]).sum(axis=0) > 0
    frame["lead_cu"] = lead_score(c, u, ru)
    frame["mech"] = column_pearson(-u, ru)
    frame["chrom"] = column_pearson(c, ru)
    frame["cperm"] = lead_score(c[rng.permutation(len(all_rows))], u, ru)
    frame["chrom_perm"] = column_pearson(c[rng.permutation(len(all_rows))], ru)
    frame["static_cu_test"] = column_pearson(activity[test_rows], unspliced[test_rows])
    del c, u, s, ru, rate_s, spliced, unspliced

    target_u = chromatin.to_dense(adata.layers["unspliced_lognorm"], np.float32)[test_rows]
    conditions = []
    for run_dir in args.run_dirs:
        model, payload, config = chromatin.load_checkpoint(Path(run_dir), args.checkpoint,
                                                           device)
        columns = adata.var_names.get_indexer(payload["gene_names"])
        with torch.no_grad():
            predicted = model.phi(
                torch.as_tensor(activity[test_rows], device=device)).cpu().numpy()
        values = np.full(adata.n_vars, np.nan)
        values[columns] = column_pearson(chromatin.unspliced_block(predicted, RELAY),
                                         target_u[:, columns])
        frame[f"model_u_{payload['condition']}"] = values
        conditions.append(payload["condition"])

    out_csv = PROJECT_ROOT / "cache" / "results" / "chromatin" / f"lead_link_controls_{args.dataset}.csv"
    frame.to_csv(out_csv)
    results = {"dataset": args.dataset, "n_cells_all": int(len(all_rows)),
               "n_cells_test": int(len(test_rows)), "strata": {}}

    print("\n[ctrl] lead-score decomposition, by detection threshold (kinetic genes only)")
    print(f"{'min_det':>8}{'n':>6}{'lead_cu':>10}{'mech':>10}{'chrom':>10}{'cperm':>10}"
          f"{'chromperm':>11}{'static_cu':>11}")
    for threshold in [0.05, 0.02, 0.01, 0.0]:
        mask = (frame["u_detect"] >= threshold) & (frame["s_detect"] >= threshold) \
            & (frame["c_detect"] >= threshold) & frame["kinetic"] \
            & frame["has_rate_u"] & frame["has_rate_s"]
        block = frame[mask]
        medians = {col: float(np.nanmedian(block[col]))
                   for col in ["lead_cu", "mech", "chrom", "cperm", "chrom_perm",
                               "static_cu_test"]}
        print(f"{threshold:>8.2f}{int(mask.sum()):>6}" +
              "".join(f"{medians[k]:>+10.4f}" if k != "chrom_perm" else f"{medians[k]:>+11.4f}"
                      for k in ["lead_cu", "mech", "chrom", "cperm"]) +
              f"{medians['chrom_perm']:>+11.4f}{medians['static_cu_test']:>+11.4f}")
        stratum = {"n_genes": int(mask.sum()), "medians": medians, "links": {}}
        for condition in conditions:
            model = block[f"model_u_{condition}"].to_numpy()
            for predictor in ["lead_cu", "chrom", "mech", "static_cu_test"]:
                stratum["links"][f"{condition}|{predictor}"] = association(
                    block[predictor].to_numpy(), model)
        stratum["links"]["data_only|lead_cu vs static_cu"] = association(
            block["lead_cu"].to_numpy(), block["static_cu_test"].to_numpy())
        stratum["links"]["data_only|lead_cu vs mech"] = association(
            block["lead_cu"].to_numpy(), block["mech"].to_numpy())
        stratum["links"]["data_only|chrom vs static_cu"] = association(
            block["chrom"].to_numpy(), block["static_cu_test"].to_numpy())
        results["strata"][f"{threshold:.2f}"] = stratum

    print("\n[ctrl] link tests: predictor vs held-out u Pearson (and data-only pairs)")
    print(f"{'min_det':>8}{'pair':<38}{'n':>6}{'pearson':>10}{'spearman':>10}{'p':>11}")
    for threshold, stratum in results["strata"].items():
        for key, block in stratum["links"].items():
            if "spearman" not in block:
                continue
            print(f"{threshold:>8}{key:<38}{block['n_genes']:>6}"
                  f"{block['pearson']:>+10.4f}{block['spearman']:>+10.4f}"
                  f"{block['spearman_p']:>11.2e}")

    out = PROJECT_ROOT / "cache" / "results" / "chromatin" / f"lead_link_controls_{args.dataset}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n[ctrl] wrote {out} and {out_csv}")
    return 0

def predicted_unspliced(run_dir: Path, checkpoint: str, activity: np.ndarray,
                        rows: np.ndarray, device: torch.device,
                        batch_size: int = 4096) -> tuple[np.ndarray, list[str], dict]:
    """The u half of phi(c) on the given cells, plus the run's gene order and config."""
    model, payload, config = chromatin.load_checkpoint(run_dir, checkpoint, device)
    assert payload["law"] == RELAY, f"{run_dir} is not a relay run; the u block is undefined"
    blocks = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            chunk = torch.as_tensor(activity[rows[start:start + batch_size]], device=device)
            blocks.append(chromatin.unspliced_block(model.phi(chunk).cpu().numpy(), RELAY))
    return np.concatenate(blocks, axis=0), list(payload["gene_names"]), config

def median_bootstrap(values: np.ndarray, seed: int, n_resamples: int = 10000) -> list[float]:
    """Percentile bootstrap interval for a median over genes."""
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(n_resamples, len(values)))
    medians = np.median(values[draws], axis=1)
    return [float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))]

def paired_delta(a: np.ndarray, b: np.ndarray, seed: int) -> dict:
    """Per-gene a - b: median, spread, sign test and a bootstrap interval."""
    delta = a - b
    block = {
        "n_genes": int(len(delta)),
        "median_delta": float(np.median(delta)),
        "mean_delta": float(np.mean(delta)),
        "fraction_positive": float((delta > 0).mean()),
        "bootstrap_ci_median": median_bootstrap(delta, seed),
    }
    if len(delta) >= 6 and np.any(delta != 0):
        block["wilcoxon_p"] = float(wilcoxon(delta).pvalue)
    return block

def subset_report(gene_pearson: dict[str, np.ndarray], keep: np.ndarray, seed: int) -> dict:
    """Every condition's median on this gene subset, plus the paired contrasts."""
    part = {name: values[keep] for name, values in gene_pearson.items()}
    block = {"n_genes": int(keep.sum())}
    for name, values in part.items():
        block[f"{name}_median"] = float(np.median(values))
        block[f"{name}_ci"] = median_bootstrap(values, seed)
    block["full_minus_shuffle"] = paired_delta(part["full"], part["shuffle"], seed)
    block["full_minus_noDyn"] = paired_delta(part["full"], part["noDyn"], seed)
    return block

def stratification_null(gene_pearson: dict[str, np.ndarray], n_high: int, seed: int,
                        n_permutations: int = 10000) -> dict:
    """Is the high-minus-low contrast bigger than an arbitrary split of the same genes?

    Permuting the lead ranking holds every per-gene property fixed and varies only which genes are called high-lead.
    """
    delta = gene_pearson["full"] - gene_pearson["shuffle"]
    rng = np.random.default_rng(seed)
    gap = np.empty(n_permutations)
    for i in range(n_permutations):
        order = rng.permutation(len(delta))
        gap[i] = np.median(delta[order[:n_high]]) - np.median(delta[order[n_high:]])
    return {"null_median_gap": float(np.median(gap)),
            "null_gap_sd": float(gap.std()),
            "null_gap_p2.5": float(np.percentile(gap, 2.5)),
            "null_gap_p97.5": float(np.percentile(gap, 97.5)),
            "samples": gap}


def stratified(args: argparse.Namespace) -> int:


    device = choose_torch_device({"device": args.device})
    conditions = ["full", "shuffle", "noDyn"]
    run_dirs = {name: PROJECT_ROOT / f"{args.run_prefix}_{name}{args.run_suffix}"
                for name in conditions}
    for name, path in run_dirs.items():
        assert path.exists(), f"missing run directory {path}"

    adata = chromatin.load_dataset(args.dataset)
    splits = pd.read_csv(chromatin.split_path(args.dataset, 0), index_col=0)
    assert splits.index.equals(adata.obs_names), "split file does not match this dataset"
    split_column = splits["split"].to_numpy()

    activity = np.asarray(adata.obsm["gene_activity"], dtype=np.float32)
    unspliced = chromatin.to_dense(adata.layers["unspliced_lognorm"], np.float32)
    rate_u = chromatin.reference_velocity(adata, "velocity_scvelo_u")
    rate_s = chromatin.reference_velocity(adata, "velocity_scvelo")
    assert rate_u is not None and rate_s is not None, "run `reference` first"

    # Same cell population as the other tool, then split train/test.
    covered = chromatin.target_cell_mask(adata, "spliced_lognorm") \
        & (np.abs(rate_s).sum(axis=1) > 0) & (np.abs(rate_u).sum(axis=1) > 0)
    lead_rows_all = np.flatnonzero(covered)
    lead_rows_train = np.flatnonzero(covered & (split_column == "train"))
    test_rows = np.flatnonzero(covered & (split_column == "test"))
    print(f"[lens3] {args.dataset}: {len(lead_rows_all)} cells with both reference rates "
          f"({len(lead_rows_train)} train used for lead scores, {len(test_rows)} test used "
          "for per-gene Pearson)", flush=True)

    # Shared gene order or paired deltas are meaningless.
    predictions, gene_names = {}, None
    for name in conditions:
        predicted, names, config = predicted_unspliced(
            run_dirs[name], args.checkpoint, activity, test_rows, device)
        assert config["condition"] == name, (
            f"{run_dirs[name]} was trained under condition {config['condition']!r}")
        if gene_names is None:
            gene_names = names
        assert names == gene_names, "the ablation runs do not share a gene order"
        predictions[name] = predicted
        print(f"[lens3] {name}: {predicted.shape[0]} cells x {predicted.shape[1]} u genes",
              flush=True)

    columns = adata.var_names.get_indexer(gene_names)
    assert (columns >= 0).all(), "checkpoint genes are not in the dataset"
    observed_u = unspliced[np.ix_(test_rows, columns)]
    gene_pearson = {name: column_pearson(predictions[name], observed_u)
                    for name in conditions}

    # Fit on train for prediction; all-cells only for comparability.
    _, _, kinetic_mask = gene_map(adata, chromatin.gene_map_path(args.dataset))
    scores = {}
    for label, rows in {"train": lead_rows_train, "all": lead_rows_all}.items():
        scores[label] = lead_score(activity[np.ix_(rows, columns)],
                                   unspliced[np.ix_(rows, columns)],
                                   rate_u[np.ix_(rows, columns)])
    shuffle = np.random.default_rng(args.seed).permutation(len(lead_rows_train))
    null_scores = lead_score(activity[np.ix_(lead_rows_train, columns)],
                             unspliced[np.ix_(lead_rows_train, columns)],
                             rate_u[np.ix_(lead_rows_train, columns)][shuffle])

    detected = ((unspliced[np.ix_(lead_rows_all, columns)] > 0).mean(axis=0) >= args.min_detected) \
        & ((activity[np.ix_(lead_rows_all, columns)] > 0).mean(axis=0) >= args.min_detected) \
        & (np.abs(rate_u[np.ix_(lead_rows_all, columns)]).sum(axis=0) > 0)
    finite = np.isfinite(scores["train"]) & np.isfinite(scores["all"])
    for values in gene_pearson.values():
        finite &= np.isfinite(values)
    in_kinetic = kinetic_mask[columns]

    results = {
        "dataset": args.dataset, "checkpoint": args.checkpoint,
        "run_dirs": {k: str(v) for k, v in run_dirs.items()},
        "n_test_cells": int(len(test_rows)),
        "n_lead_cells_train": int(len(lead_rows_train)),
        "n_output_genes": int(len(gene_names)),
        "top_fraction": args.top_fraction,
        "gene_sets": {},
    }

    frame = pd.DataFrame({
        "gene": gene_names,
        "lead_train": scores["train"], "lead_all": scores["all"], "lead_null": null_scores,
        "in_kinetic_mask": in_kinetic, "detected": detected,
        **{f"pearson_{name}": gene_pearson[name] for name in conditions},
    })
    frame["delta_full_minus_shuffle"] = frame["pearson_full"] - frame["pearson_shuffle"]
    frame["delta_full_minus_noDyn"] = frame["pearson_full"] - frame["pearson_noDyn"]

    # Two gene universes: kinetic vs all scored.
    universes = {
        "kinetic_mask_genes": finite & detected & in_kinetic,
        "all_output_genes": finite & detected,
    }
    for universe, mask in universes.items():
        keep = np.flatnonzero(mask)
        block = {
            "n_genes": int(len(keep)),
            "lead_train_median": float(np.median(scores["train"][keep])),
            "lead_all_median": float(np.median(scores["all"][keep])),
            "lead_null_median": float(np.median(null_scores[keep])),
            "overall": subset_report({k: v[keep] for k, v in gene_pearson.items()},
                                     np.ones(len(keep), dtype=bool), args.seed),
            "spearman_lead_vs_delta": {},
            "splits": {},
        }
        for ranking in ("train", "all"):
            rho, p = spearmanr(scores[ranking][keep],
                               (gene_pearson["full"] - gene_pearson["shuffle"])[keep])
            block["spearman_lead_vs_delta"][ranking] = {"rho": float(rho), "p": float(p)}

        for ranking in ("train", "all"):
            order = np.argsort(-scores[ranking][keep])
            n_high = max(1, int(round(args.top_fraction * len(keep))))
            high = np.zeros(len(keep), dtype=bool)
            high[order[:n_high]] = True
            low_quartile = np.zeros(len(keep), dtype=bool)
            low_quartile[order[-n_high:]] = True
            subset = {k: v[keep] for k, v in gene_pearson.items()}
            split = {
                "high_lead": subset_report(subset, high, args.seed),
                "low_lead_rest": subset_report(subset, ~high, args.seed),
                "low_lead_bottom_quartile": subset_report(subset, low_quartile, args.seed),
                "high_lead_score_median": float(np.median(scores[ranking][keep][high])),
                "low_lead_rest_score_median": float(np.median(scores[ranking][keep][~high])),
                "low_lead_bottom_quartile_score_median": float(
                    np.median(scores[ranking][keep][low_quartile])),
            }
            observed_gap = (split["high_lead"]["full_minus_shuffle"]["median_delta"]
                            - split["low_lead_rest"]["full_minus_shuffle"]["median_delta"])
            null = stratification_null(subset, n_high, args.seed)
            samples = null.pop("samples")
            null["observed_gap"] = float(observed_gap)
            null["two_sided_p"] = float(
                (np.abs(samples - np.median(samples)) >= abs(observed_gap - np.median(samples))
                 ).mean())
            split["stratification_null"] = null
            block["splits"][f"by_lead_{ranking}"] = split
        results["gene_sets"][universe] = block

    out = Path(args.out)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    frame.to_csv(out.with_suffix(".csv"), index=False)

    for universe, block in results["gene_sets"].items():
        print(f"\n=== {universe}: {block['n_genes']} genes, median c->u lead "
              f"{block['lead_train_median']:+.4f} (train) / {block['lead_all_median']:+.4f} "
              f"(all cells), null {block['lead_null_median']:+.4f}")
        overall = block["overall"]
        print(f"  overall u gene Pearson  full {overall['full_median']:+.4f}  "
              f"shuffle {overall['shuffle_median']:+.4f}  noDyn {overall['noDyn_median']:+.4f}")
        for ranking, split in block["splits"].items():
            print(f"  -- split {ranking}")
            for subset in ("high_lead", "low_lead_rest", "low_lead_bottom_quartile"):
                s = split[subset]
                d = s["full_minus_shuffle"]
                print(f"     {subset:<26} n={s['n_genes']:<4} "
                      f"full {s['full_median']:+.4f}  shuffle {s['shuffle_median']:+.4f}  "
                      f"noDyn {s['noDyn_median']:+.4f}  "
                      f"delta(full-shuffle) {d['median_delta']:+.4f} "
                      f"[{d['bootstrap_ci_median'][0]:+.4f}, {d['bootstrap_ci_median'][1]:+.4f}] "
                      f"{d['fraction_positive']:.0%}+ p={d.get('wilcoxon_p', float('nan')):.3f}")
            null = split["stratification_null"]
            print(f"     high-minus-low gap {null['observed_gap']:+.5f}  "
                  f"random-split null sd {null['null_gap_sd']:.5f}  p={null['two_sided_p']:.3f}")
            rho = block["spearman_lead_vs_delta"][ranking.replace("by_lead_", "")]
            print(f"     Spearman(lead, full-shuffle delta) {rho['rho']:+.4f}  p={rho['p']:.3f}")

    print(f"\n[lens3] wrote {out} and {out.with_suffix('.csv')}")
    return 0

def unpaired_scale(production: np.ndarray, target: np.ndarray, source_rows: np.ndarray,
                   target_rows: np.ndarray) -> np.ndarray:
    """The model's own per-gene calibration: std(target on RNA cells)/std(G c on ATAC cells).

    Reported beside the paired slope because the trained runs never see a paired cell.
    """
    source_std = production[source_rows].std(axis=0)
    target_std = target[target_rows].std(axis=0)
    usable = (source_std >= 1e-3) & (target_std >= 1e-3)
    return np.where(usable, target_std / np.clip(source_std, 1e-12, None), 0.0)

def ridge_prediction(train_features: np.ndarray, train_reference: np.ndarray,
                     test_features: np.ndarray, alpha: float) -> np.ndarray:
    """Best linear read-out of a feature block, fitted on TRAIN with the true pairing.

    Float64 because the gram matrix of a velocity field is poorly conditioned and this row is an upper bound.
    """
    features = np.asarray(train_features, dtype=np.float64)
    reference = np.asarray(train_reference, dtype=np.float64)
    feature_mean, reference_mean = features.mean(axis=0), reference.mean(axis=0)
    centred = features - feature_mean
    gram = centred.T @ centred + alpha * np.eye(centred.shape[1])
    weights = np.linalg.solve(gram, centred.T @ (reference - reference_mean))
    return ((np.asarray(test_features, np.float64) - feature_mean) @ weights
            + reference_mean).astype(np.float32)

def residualise(values: np.ndarray, control: np.ndarray) -> np.ndarray:
    """Per gene, `values` with its own least-squares dependence on `control` removed."""
    a = values - values.mean(axis=0)
    b = control - control.mean(axis=0)
    slope = (a * b).sum(axis=0) / np.clip((b * b).sum(axis=0), 1e-12, None)
    return a - slope * b

def partial_correlation(predicted: np.ndarray, reference: np.ndarray,
                        control: np.ndarray) -> np.ndarray:
    """Per gene, corr(predicted, reference) with `control` projected out of both.

    Partialling u out asks what chromatin adds once the state term both residual and scVelo du/dt contain is gone.
    """
    return column_pearson(residualise(predicted, control), residualise(reference, control))

def raw_scores(predicted: np.ndarray, reference: np.ndarray) -> dict:
    """The four headline numbers, before any null is attached."""
    per_gene = column_pearson(predicted, reference)
    usable = np.isfinite(per_gene)
    return {
        "cell_cosine_median": float(np.median(row_cosine(predicted, reference))),
        "cell_cosine_centred_median": float(np.median(centred_cosine(predicted, reference))),
        "gene_pearson_median": float(np.nanmedian(per_gene)),
        "gene_pearson_fraction_positive": float((per_gene[usable] > 0).mean()),
    }

def score_with_null(predicted: np.ndarray, reference: np.ndarray, seed: int,
                    n_permutations: int) -> dict:
    """Every score beside what the same two matrices give with the cells repaired at random.

    Zero is not the chance level, so each row carries its own permutation null.
    """
    scores = raw_scores(predicted, reference)
    rng = np.random.default_rng(seed)
    draws = [raw_scores(predicted, reference[rng.permutation(len(reference))])
             for _ in range(n_permutations)]
    for key in list(scores):
        values = np.array([draw[key] for draw in draws])
        scores[f"{key}_null_median"] = float(np.median(values))
        scores[f"{key}_null_p95"] = float(np.quantile(values, 0.95))
    scores["n_permutations"] = int(n_permutations)
    return scores

def usable_rows(mask: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Cells that pass a split/coverage mask and actually carry a reference rate."""
    rows = np.flatnonzero(mask)
    return rows[np.abs(reference[rows]).sum(axis=1) > 0]

def print_block(title: str, block: dict) -> None:
    print(f"\n[static-vs-jvp] {title}")
    print(f"  {'row':<34} {'gene r':>8} {'null':>8} {'p95':>8} "
          f"{'r>0':>6} {'cos_c':>8} {'null':>8} {'p95':>8} {'cos':>8}")
    for name, row in block.items():
        print(f"  {name:<34} {row['gene_pearson_median']:>+8.4f} "
              f"{row['gene_pearson_median_null_median']:>+8.4f} "
              f"{row['gene_pearson_median_null_p95']:>+8.4f} "
              f"{row['gene_pearson_fraction_positive']:>6.1%} "
              f"{row['cell_cosine_centred_median']:>+8.4f} "
              f"{row['cell_cosine_centred_median_null_median']:>+8.4f} "
              f"{row['cell_cosine_centred_median_null_p95']:>+8.4f} "
              f"{row['cell_cosine_median']:>+8.4f}")


def static_vs_jvp(args: argparse.Namespace) -> int:


    adata = chromatin.load_dataset(args.dataset)
    splits = pd.read_csv(chromatin.split_path(args.dataset, args.split_seed), index_col=0)
    assert splits.index.equals(adata.obs_names), "split file does not match the dataset"
    split_column = splits["split"].to_numpy()
    side_column = splits["train_side"].to_numpy()
    fields = chromatin.load_velocity(args.dataset, args.split_seed)

    mapping, _, kinetic_mask = gene_map(adata, chromatin.gene_map_path(args.dataset))
    rows, columns = mapping.nonzero()
    # G identity is the assumption the lens rests on; assert it.
    identity_like = bool(len(rows) == int(kinetic_mask.sum()) and (rows == columns).all())
    print(f"[static-vs-jvp] G {mapping.shape}, {mapping.nnz} links, "
          f"identity on the kinetic genes: {identity_like}")

    activity = np.asarray(adata.obsm["gene_activity"], dtype=np.float32)
    velocity = fields["velocity"].astype(np.float32)
    unspliced = to_dense(adata.layers["unspliced_lognorm"], np.float32)
    spliced = to_dense(adata.layers["spliced_lognorm"], np.float32)
    rate_u = chromatin.reference_velocity(adata, "velocity_scvelo_u")
    rate_s = chromatin.reference_velocity(adata, "velocity_scvelo")
    assert rate_u is not None and rate_s is not None, (
        f"run `reference --dataset {args.dataset}` first")

    production = activity @ mapping.toarray().T
    field_push = velocity @ mapping.toarray().T

    detected = ((unspliced > 0).mean(axis=0) >= args.min_detected) \
        & ((activity > 0).mean(axis=0) >= args.min_detected)
    gene_sets = {
        "u_target": np.flatnonzero(kinetic_mask & detected & (np.abs(rate_u).sum(axis=0) > 0)),
        "lead_lag": np.flatnonzero(
            kinetic_mask & detected & ((spliced > 0).mean(axis=0) >= args.min_detected)
            & (np.abs(rate_u).sum(axis=0) > 0) & (np.abs(rate_s).sum(axis=0) > 0)),
    }

    results = {
        "dataset": args.dataset,
        "split_seed": args.split_seed,
        "reference_layer": "velocity_scvelo_u",
        "gene_map_identity_on_kinetic_genes": identity_like,
        "ridge_alpha": args.ridge_alpha,
        "n_genes": {name: int(len(genes)) for name, genes in gene_sets.items()},
        "blocks": {},
    }

    for set_name, genes in gene_sets.items():
        c = production[:, genes]
        v = field_push[:, genes]
        u = unspliced[:, genes]
        s = spliced[:, genes]
        ru = rate_u[:, genes]
        rs = rate_s[:, genes]
        # Clamp log1p inverse to the same range the law uses.
        exponent = np.exp(np.clip(u, LOG1P_MIN, LOG1P_MAX))

        target_covered = chromatin.target_cell_mask(adata, "unspliced_lognorm")
        dynamic = fields["dynamic_mask"]
        train_rows = usable_rows((split_column == "train") & target_covered & dynamic, ru)
        test_rows = usable_rows((split_column == "test") & target_covered & dynamic, ru)
        # Static row needs no velocity; matched comparison stays on dynamic cells.
        test_all_rows = usable_rows((split_column == "test") & target_covered, ru)
        print(f"\n[static-vs-jvp] {set_name}: {len(genes)} genes, fit k on "
              f"{len(train_rows)} dynamic train cells, score on {len(test_rows)} dynamic "
              f"test cells ({len(test_all_rows)} test cells ignoring the dynamic mask)")

        # k from train so held-out is a prediction; unpaired protocol forbids same-cell c and u.
        k_paired = steady_state_slope(c[train_rows], u[train_rows])
        atac_rows = np.flatnonzero((split_column == "train") & (side_column == "atac"))
        rna_rows = np.flatnonzero((split_column == "train") & (side_column == "rna"))
        k_unpaired = unpaired_scale(c, u, atac_rows, rna_rows)
        print(f"[static-vs-jvp] slope k: paired median {np.median(k_paired):.4g}, "
              f"unpaired median {np.median(k_unpaired):.4g}, "
              f"corr {np.corrcoef(k_paired, k_unpaired)[0, 1]:+.3f}")

        predictors = {
            "static_residual_kGc_minus_u": k_paired * c - u,
            "static_residual_unpaired_k": k_unpaired * c - u,
            # Minus-state-alone is the no-chromatin baseline; without it the residual cannot be attributed to chromatin.
            "decomposition_minus_u_only": -u,
            "decomposition_kGc_only": k_paired * c,
            "jvp_like_k_times_Gv": k_paired * v,
            "jvp_like_unpaired_k": k_unpaired * v,
            "raw_chromatin_velocity_Gv": v,
        }
        for label, exponent_applied, reference in [
                ("log1p", False, ru / exponent),
                ("linear", True, ru)]:
            block = {}
            for name, predicted in predictors.items():
                values = predicted * exponent if exponent_applied else predicted
                block[name] = score_with_null(values[test_rows], reference[test_rows],
                                              args.seed, args.n_permutations)
            # Nested blocks: only the increment over [u] is chromatin; only the increment of the field over [Gc,u] is the field.
            ridge_features = {
                "ridge_[u]_no_chromatin_baseline": u,
                "ridge_[Gc]": c,
                "ridge_static_features_[Gc,u]": np.concatenate([c, u], axis=1),
                "ridge_jvp_features_[Gv]": v,
                "ridge_[Gv,u]": np.concatenate([v, u], axis=1),
                "ridge_jvp_features_[Gv,Gc,u]": np.concatenate([v, c, u], axis=1),
            }
            for name, features in ridge_features.items():
                predicted = ridge_prediction(features[train_rows], reference[train_rows],
                                             features[test_rows], args.ridge_alpha)
                block[name] = score_with_null(predicted, reference[test_rows], args.seed,
                                              args.n_permutations)
            # Shuffled-pair ridge is the null for a fitted row.
            shuffled = train_rows[np.random.default_rng(args.seed).permutation(len(train_rows))]
            features = np.concatenate([c, u], axis=1)
            block["ridge_static_shuffled_fit"] = score_with_null(
                ridge_prediction(features[train_rows], reference[shuffled],
                                 features[test_rows], args.ridge_alpha),
                reference[test_rows], args.seed, args.n_permutations)
            results["blocks"][f"{set_name}/{label}/test_dynamic"] = block
            print_block(f"{set_name} genes, {label} coordinates, "
                        f"{len(test_rows)} dynamic test cells", block)

        # Coverage rows need no field.
        wide = {}
        for name, rows_used in [("test_all_cells", test_all_rows),
                                ("all_cells", usable_rows(target_covered, ru))]:
            k_all = steady_state_slope(c[rows_used], u[rows_used])
            reference = (ru / exponent)[rows_used]
            for label, predicted in [("residual", (k_all * c - u)[rows_used]),
                                     ("minus_u_only", (-u)[rows_used]),
                                     ("kGc_only", (k_all * c)[rows_used])]:
                wide[f"{name}:{label}"] = score_with_null(predicted, reference, args.seed,
                                                          args.n_permutations)
        results["blocks"][f"{set_name}/log1p/static_coverage"] = wide
        print_block(f"{set_name} genes, log1p, static rows on wider cell sets", wide)

        # Partial after removing the state term both residual and scVelo du/dt contain.
        partial = {}
        spliced_exponent = np.exp(np.clip(s, LOG1P_MIN, LOG1P_MAX))
        all_rows = usable_rows(target_covered, ru)
        for name, rows_used, feature, reference, control in [
                ("Gc_partial_u/all_cells/log1p", all_rows, c, ru / exponent, u),
                ("Gc_partial_u/all_cells/linear", all_rows, c, ru, u),
                ("Gc_partial_u/test_dynamic/log1p", test_rows, c, ru / exponent, u),
                ("Gv_partial_u/test_dynamic/log1p", test_rows, v, ru / exponent, u),
                ("Gv_partial_u/test_dynamic/linear", test_rows, v, ru, u),
                ("u_partial_s/all_cells/log1p", usable_rows(target_covered, rs), u,
                 rs / spliced_exponent, s)]:
            probe, target_rate = feature[rows_used], reference[rows_used]
            held = control[rows_used]
            scores = partial_correlation(probe, target_rate, held)
            rng = np.random.default_rng(args.seed)
            draws = [np.nanmedian(partial_correlation(
                probe, target_rate[rng.permutation(len(rows_used))], held))
                for _ in range(args.n_permutations)]
            partial[name] = {
                "gene_partial_pearson_median": float(np.nanmedian(scores)),
                "gene_partial_pearson_fraction_positive": float(
                    (scores[np.isfinite(scores)] > 0).mean()),
                "gene_partial_pearson_null_median": float(np.median(draws)),
                "gene_partial_pearson_null_p95": float(np.quantile(draws, 0.95)),
                "n_cells": int(len(rows_used)),
            }
            print(f"[static-vs-jvp] partial {name:<34} "
                  f"{partial[name]['gene_partial_pearson_median']:+.4f} "
                  f"({partial[name]['gene_partial_pearson_fraction_positive']:.1%} positive, "
                  f"null p95 {partial[name]['gene_partial_pearson_null_p95']:+.4f}, "
                  f"n={len(rows_used)})")
        results["blocks"][f"{set_name}/log1p/partial_correlations"] = partial

        # u→s control in this harness so a harness bug fails here too.
        control_rows = usable_rows(target_covered, rs)
        k_us = steady_state_slope(u[control_rows], s[control_rows])
        control_reference = (rs / spliced_exponent)[control_rows]
        # Both halves of u→s should score; if they do, c→u's failing upstream half is about chromatin not the estimator.
        control = {name: score_with_null(predicted[control_rows], control_reference,
                                         args.seed, args.n_permutations)
                   for name, predicted in {
                       "static_residual_ku_minus_s": k_us * u - s,
                       "decomposition_minus_s_only": -s,
                       "decomposition_ku_only": k_us * u,
                   }.items()}
        results["blocks"][f"{set_name}/log1p/positive_control_u_to_s"] = control
        print_block(f"{set_name} genes, u -> s positive control on {len(control_rows)} cells",
                    control)

    out = Path(args.out) if args.out else (
        PROJECT_ROOT / "cache" / "results" / "chromatin"
        / f"static_vs_jvp_{args.dataset}_seed{args.split_seed}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\n[static-vs-jvp] wrote {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)

    child = sub.add_parser('lag')
    child.set_defaults(func=lag)
    child.add_argument("--dataset", choices=chromatin.DATASETS, required=True)
    child.add_argument("--min-detected", type=float, default=0.05,
                        help="drop genes detected in fewer than this fraction of cells")
    child.add_argument("--seed", type=int, default=0)

    child = sub.add_parser('link')
    child.set_defaults(func=link)
    child.add_argument("--dataset", choices=chromatin.DATASETS, default="hspc")
    child.add_argument("--run-dirs", nargs="+", required=True)
    child.add_argument("--checkpoint", default="best_align")
    child.add_argument("--min-detected", type=float, default=0.05)
    child.add_argument("--seed", type=int, default=0)
    child.add_argument("--out", default=None)

    child = sub.add_parser('link-controls')
    child.set_defaults(func=link_controls)
    child.add_argument("--dataset", default="hspc")
    child.add_argument("--run-dirs", nargs="+", required=True)
    child.add_argument("--checkpoint", default="best_align")
    child.add_argument("--seed", type=int, default=0)

    child = sub.add_parser('stratified')
    child.set_defaults(func=stratified)
    child.add_argument("--dataset", choices=chromatin.DATASETS, default="hspc")
    child.add_argument("--run-prefix", default="cache/chromatin/runs/abl_hspc_r2u")
    child.add_argument("--run-suffix", default="_seed42")
    child.add_argument("--checkpoint", default="best_align")
    child.add_argument("--top-fraction", type=float, default=0.25)
    child.add_argument("--min-detected", type=float, default=0.05)
    child.add_argument("--seed", type=int, default=0)
    child.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    child.add_argument("--out", default="cache/results/chromatin/lead_stratified_ablation.json")

    child = sub.add_parser('static-vs-jvp')
    child.set_defaults(func=static_vs_jvp)
    child.add_argument("--dataset", choices=chromatin.DATASETS, default="hspc")
    child.add_argument("--split-seed", type=int, default=0)
    child.add_argument("--min-detected", type=float, default=0.05)
    child.add_argument("--ridge-alpha", type=float, default=100.0)
    child.add_argument("--n-permutations", type=int, default=20)
    child.add_argument("--seed", type=int, default=0)
    child.add_argument("--out", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
