#!/usr/bin/env python3
"""Chromatin run metrics and collapse diagnostics, gathered into readable tables.

  runs      every run's metrics side by side, with the delta against `full`
  collapse  every `failure_analysis.json` as geometry and held-out tables

Usage:
  python tools/summarize_chromatin.py runs --csv cache/results/chromatin/run_summary.csv
  python tools/summarize_chromatin.py collapse
"""

from __future__ import annotations

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization.style import apply_style, figsize, panel_letter, save_figure


RUNS_ROOT = Path("cache/chromatin/runs")

CACHE_ROOT = Path("cache/chromatin")

CORRUPTIONS = ["shuffle", "reverse", "zero", "permG"]

def flatten_baseline(run: Path) -> dict | None:
    """One row for a competing method. Same Task A-C columns as a KOT row.

    `n_test_scored` identifies provenance so a different submission is not mixed in.
    """
    evaluation = run / "evaluation_baseline.json"
    if not evaluation.exists():
        return None
    ev = json.loads(evaluation.read_text())
    row = {"run": run.name, "method": ev["method"], "dataset": ev["dataset"],
           "law": "-", "cond": "-", "lam": None, "seed": ev["seed"],
           "n_test": ev["n_test_scored"], "kind": ev["prediction_kind"],
           "runtime_s": round(ev["runtime_seconds"], 1)}
    a = ev["task_a"]
    row.update({"A_gene_r": a["gene_pearson_median"], "A_gene_rho": a["gene_spearman_median"],
                "A_nrmse": a["gene_nrmse_median"], "A_cell_cos": a["cell_cosine_median"]})
    b = ev.get("task_b", {})
    row.update({"B_foscttm": b.get("knn_foscttm"),
                "B_floor": b.get("knn_foscttm_constant_floor"),
                "B_diversity": b.get("knn_partner_diversity"),
                "B_knn_type": b.get("knn_cell_type_accuracy"),
                "B_type_baseline": b.get("cell_type_majority_baseline"),
                "B_sink_top1": b.get("sinkhorn_top1"),
                "B_sink_foscttm": b.get("sinkhorn_foscttm"),
                "B_fosknn_1pct": b.get("knn_fosknn_frac0.01"),
                "B_fosknn_5pct": b.get("knn_fosknn_frac0.05")})
    c = ev.get("task_c", {})
    row.update({"C_transfer": c.get("label_transfer_accuracy"), "C_ari": c.get("ari"),
                "C_nmi": c.get("nmi"), "C_asw": c.get("cell_type_asw"),
                "C_ilisi": c.get("ilisi")})
    return row

def flatten(run: Path) -> dict | None:
    """One row per evaluated checkpoint: the gate, then Tasks A-D."""
    evaluation = run / "evaluation_best_align.json"
    gate, config = run / "preflight.json", run / "run_config.json"
    if not (evaluation.exists() and config.exists()):
        return None
    ev = json.loads(evaluation.read_text())
    cfg = json.loads(config.read_text())
    row = {
        "run": run.name, "method": "KOT", "dataset": ev["dataset"], "law": ev["law"],
        "cond": ev["condition"], "lam": cfg["lambda_dyn"], "seed": ev["seed"],
        "n_test": ev["n_test_cells"], "kind": "direct",
    }
    if gate.exists():
        g = json.loads(gate.read_text())
        row["spread"] = g.get("prediction_spread_ratio")
        row["jvp_norm"] = g.get("jvp_norm_median")

    a = ev["task_a"]
    row.update({"A_gene_r": a["gene_pearson_median"], "A_gene_rho": a["gene_spearman_median"],
                "A_nrmse": a["gene_nrmse_median"], "A_cell_cos": a["cell_cosine_median"]})
    b = ev["task_b"]
    row.update({"B_foscttm": b["knn_foscttm"], "B_floor": b.get("knn_foscttm_constant_floor"),
                "B_diversity": b.get("knn_partner_diversity"),
                "B_hub_share": b.get("knn_top_partner_share"),
                "B_knn_type": b.get("knn_cell_type_accuracy"),
                "B_sink_top1": b["sinkhorn_top1"],
                "B_sink_foscttm": b.get("sinkhorn_foscttm"),
                "B_sink_type": b.get("sinkhorn_cell_type_accuracy"),
                "B_fosknn_1pct": b.get("knn_fosknn_frac0.01"),
                "B_fosknn_5pct": b.get("knn_fosknn_frac0.05"),
                "B_fosknn_10pct": b.get("knn_fosknn_frac0.1")})
    row["B_within"] = ev.get("task_b_within_group", {}).get("mean")
    c = ev.get("task_c", {})
    row.update({"C_transfer": c.get("label_transfer_accuracy"), "C_ari": c.get("ari"),
                "C_nmi": c.get("nmi"), "C_asw": c.get("cell_type_asw"),
                "C_ilisi": c.get("ilisi"), "C_conn": c.get("graph_connectivity")})
    law = ev["task_d"].get("internal_vs_law", {})
    row.update({"int_D_cos": law.get("cell_cosine_median"),
                "int_D_cos_c": law.get("cell_cosine_centred_median")})
    bio_law = ev["task_d"].get("biological_vs_law", {})
    row.update({"bio_law_D_cos": bio_law.get("cell_cosine_median"),
                "bio_law_D_cos_c": bio_law.get("cell_cosine_centred_median")})
    d = ev["task_d"].get("biological_vs_velocity_scvelo", {})
    row.update({"bio_D_cos": d.get("cell_cosine_median"),
                "bio_D_cos_c": d.get("cell_cosine_centred_median"),
                "bio_D_pos": d.get("cell_cosine_positive_fraction"),
                "bio_D_gene_r": d.get("gene_pearson_median"),
                "bio_D_sign": d.get("sign_agreement"),
                "bio_D_sign_chance": d.get("sign_agreement_chance"),
                "bio_D_resid_sf": d.get("scale_free_residual_median")})
    reg = ev["task_d"].get("biological_vs_velocity_regvelo", {})
    row.update({"bio_regvelo_D_cos": reg.get("cell_cosine_median")})
    return row

def baseline_row(dataset: str) -> dict | None:
    """The ceilings Task A has to be read against."""
    path = CACHE_ROOT / f"{dataset}_baselines.json"
    if not path.exists():
        return None
    b = json.loads(path.read_text())
    names = [name for name in ["mean_only", "identity", "ridge", "paired_mlp"]
             if name in b]
    return {name: {"gene_r_in_G": b[name]["in_G"]["gene_pearson_median"],
                   "gene_r_all": b[name]["all"]["gene_pearson_median"],
                   "cell_cos": b[name]["cell_cosine_median"]}
            for name in names}

def print_block(frame: pd.DataFrame, title: str, columns: list[str]) -> None:
    present = [c for c in columns if c in frame.columns and frame[c].notna().any()]
    if not present:
        return
    print(f"\n--- {title}")
    keys = [c for c in ["dataset", "method", "cond", "lam", "seed"] if c in frame.columns]
    print(frame[keys + present].to_string(index=False,
                                          float_format=lambda v: f"{v:+.4f}"))

def print_ablation_gaps(frame: pd.DataFrame, columns: list[str]) -> None:
    """Each corruption minus `full`, which is what the ablation actually claims."""
    print("\n--- ablation gaps (condition - full); a corruption that changes nothing is a bug "
          "in the control, not a result")
    for dataset, block in frame.groupby("dataset"):
        reference = block[block["cond"] == "full"]
        if reference.empty:
            continue
        base = reference.iloc[0]
        rows = []
        for _, row in block[block["cond"] != "full"].iterrows():
            entry = {"dataset": dataset, "cond": row["cond"]}
            entry.update({c: row[c] - base[c] for c in columns
                          if c in block.columns and pd.notna(row[c]) and pd.notna(base[c])})
            rows.append(entry)
        if rows:
            print(pd.DataFrame(rows).to_string(index=False,
                                               float_format=lambda v: f"{v:+.4f}"))

def print_seed_summary(frame: pd.DataFrame) -> None:
    """Mean and spread across seeds, which is the only form a comparison can be read in.

    A single seed cannot separate a real difference from fit variation, so the per-run
    tables above are for diagnosis and this is the one to compare methods on.
    """
    columns = [c for c in ["A_gene_r", "A_cell_cos", "B_foscttm", "B_floor", "B_diversity",
                           "B_knn_type", "C_ari", "C_ilisi"] if c in frame.columns]
    grouped = frame.groupby(["dataset", "method", "cond", "lam"], dropna=False)
    summary = grouped[columns].agg(["mean", "std", "count"])
    print("\n--- across seeds (mean / sd / n). FOSCTTM: compare against B_floor, the score "
          "a CONSTANT map gets on the same data — 0.5 is a destroyed pairing, not chance")
    with pd.option_context("display.width", 250, "display.max_columns", 60):
        print(summary.round(4).to_string())


def runs(args: argparse.Namespace) -> int:


    rows = []
    for run in sorted(RUNS_ROOT.glob(args.pattern)):
        if not run.is_dir():
            continue
        row = flatten(run) or flatten_baseline(run)
        if row is not None:
            rows.append(row)
    if not rows:
        print(f"no evaluated runs under {RUNS_ROOT}/{args.pattern}")
        return 1
    frame = pd.DataFrame(rows)
    if args.expect_n_test is not None:
        foreign = frame[frame["n_test"] != args.expect_n_test]
        if not foreign.empty:
            print(f"dropping {len(foreign)} row(s) scored on a different cell count "
                  f"(not this submission): {sorted(foreign['run'])}")
            frame = frame[frame["n_test"] == args.expect_n_test]
    frame = frame.sort_values(["dataset", "method", "law", "cond", "lam", "seed"])
    pd.set_option("display.width", 250, "display.max_columns", 60)

    for dataset in frame["dataset"].unique():
        ceilings = baseline_row(dataset)
        if ceilings:
            print(f"\n=== {dataset} ceilings (Task A is meaningless without these)")
            print(pd.DataFrame(ceilings).T.to_string(float_format=lambda v: f"{v:+.4f}"))

    print_block(frame, "health: did phi survive training at all?", ["lam", "spread", "jvp_norm"])
    print_block(frame, "Task A — RNA state from held-out ATAC",
                ["A_gene_r", "A_gene_rho", "A_nrmse", "A_cell_cos"])
    # Informative FOSCTTM lives below the per-run pairing floor.
    print_block(frame, "Task B — cell pairing (informative band is [0, B_floor]; "
                       "B_floor is what a CONSTANT map scores, ~0.25. 0.5 = destroyed pairing)",
                ["B_foscttm", "B_floor", "B_within", "B_diversity", "B_hub_share",
                 "B_knn_type", "B_sink_top1", "B_sink_foscttm", "B_sink_type",
                 "B_fosknn_1pct", "B_fosknn_5pct"])
    print("    top-k is deliberately omitted: chance is 1/n, so it is not comparable "
          "between methods scored on different numbers of cells.")
    print_block(frame, "Task C — joint integration",
                ["C_transfer", "C_ari", "C_nmi", "C_asw", "C_ilisi", "C_conn"])
    print_block(frame, "Task D — J_phi v_true vs scVelo (biological)",
                ["bio_D_cos", "bio_D_cos_c", "bio_D_pos", "bio_D_gene_r", "bio_D_sign",
                 "bio_D_sign_chance", "bio_D_resid_sf", "bio_regvelo_D_cos"])
    print_block(frame, "Task D — J_phi v vs kinetic-law RHS (internal = train v/G, "
                       "bio_law = true v + G_true)",
                ["int_D_cos", "int_D_cos_c", "bio_law_D_cos", "bio_law_D_cos_c"])
    print_ablation_gaps(frame, ["A_gene_r", "B_foscttm", "C_ari", "bio_D_cos",
                                "bio_D_cos_c", "int_D_cos", "spread"])
    print_seed_summary(frame)

    if args.csv:
        frame.to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}")
    return 0

MAP_COLORS = {"constant": "#999999", "trained_phi": "#D55E00", "oracle": "#0072B2"}

COLLAPSE_NAME = re.compile(
    r"collapse_(?P<dataset>hspc|bmmc)_(?P<condition>full|noDyn)"
    r"_d(?P<dims>\d+)_p(?P<pts>\d+)_seed(?P<seed>\d+)"
)

def identity(name: str, result: dict) -> dict:
    """JSON fields first; older analyses omitted them, so the directory name fills in."""
    parsed = COLLAPSE_NAME.search(name)
    missing = parsed.groupdict() if parsed else {}
    seed = result.get("seed")
    if seed is None and "seed" in missing:
        seed = int(missing["seed"])
    return {
        "run": name,
        "dataset": result.get("dataset") or missing.get("dataset"),
        "method": result.get("method", "KOT"),
        "law": result.get("law") or "reduced",
        "condition": result.get("condition") or missing.get("condition"),
        "seed": seed,
    }

def headroom_at_train_dim(result: dict) -> float:
    scores = {int(dim): block for dim, block in result["by_dimension"].items()}
    train_dim = int(result["align_dims_trained"])
    if train_dim in scores:
        return scores[train_dim]["fraction_of_headroom_captured"]
    nearest = min(scores, key=lambda dim: abs(dim - train_dim))
    return scores[nearest]["fraction_of_headroom_captured"]

def geometry_rows(name: str, result: dict) -> list[dict]:
    base = identity(name, result)
    return [{
        **base,
        "train_dims": result["align_dims_trained"],
        "train_pts": result["sinkhorn_max_points_trained"],
        "n_cells": result["n_cells_per_side"],
        "dim": int(dim),
        "explained_var": scores["explained_variance"],
        "blur/dist": scores["blur_over_pair_distance"],
        "constant": scores["constant"],
        "trained_phi": scores["trained_phi"],
        "oracle": scores["paired_oracle"],
        "oracle<const": scores["oracle_beats_constant"],
        "headroom": scores["fraction_of_headroom_captured"],
    } for dim, scores in result["by_dimension"].items()]

def held_out_row(name: str, result: dict) -> dict:
    held = result["held_out"]
    return {
        **identity(name, result),
        "train_dims": result["align_dims_trained"],
        "train_pts": result["sinkhorn_max_points_trained"],
        "sd_ratio": result["sd_ratio_phi_over_target"],
        "spread": held["prediction_spread_ratio"],
        "gene_r": held["gene_pearson_median"],
        "gene_rho": held["gene_spearman_median"],
        "cell_cos": held["cell_cosine_median"],
        "foscttm": held["knn_foscttm"],
        "floor": held["knn_foscttm_constant_floor"],
        "diversity": held["knn_partner_diversity"],
        "headroom": headroom_at_train_dim(result),
        "foscttm_space": "gene",
    }

def keep_row(row: dict, args: argparse.Namespace) -> bool:
    if args.seeds is not None and row.get("seed") not in args.seeds:
        return False
    if row.get("method", "KOT") != "KOT":
        return True
    if args.train_dims is not None and row.get("train_dims") not in args.train_dims:
        return False
    if args.train_pts is not None and row.get("train_pts") != args.train_pts:
        return False
    return True

def competitor_held_out_row(name: str, evaluation: dict) -> dict:
    """Task A is gene-space; Task B FOSCTTM is the method's own latent, not phi(RNA)."""
    task_a, task_b = evaluation["task_a"], evaluation.get("task_b", {})
    return {
        "run": name,
        "dataset": evaluation["dataset"],
        "method": evaluation["method"],
        "law": "-",
        "condition": "-",
        "seed": evaluation["seed"],
        "train_dims": None,
        "train_pts": None,
        "sd_ratio": None,
        "spread": None,
        "gene_r": task_a["gene_pearson_median"],
        "gene_rho": task_a["gene_spearman_median"],
        "cell_cos": task_a["cell_cosine_median"],
        "foscttm": task_b.get("knn_foscttm"),
        "floor": task_b.get("knn_foscttm_constant_floor"),
        "diversity": task_b.get("knn_partner_diversity"),
        "headroom": None,
        "foscttm_space": "latent",
    }

def geometry_figure(frame: pd.DataFrame, held_out: pd.DataFrame, path: Path) -> None:
    """The crossover, and what it bought. One line per map, dimension on a log axis."""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=figsize("full", 2.1))
    # Pick the run scored in the most dimensions so the panel shows the crossover, not one sweep arm's side of it.
    widest = frame["run"].value_counts().idxmax()
    block = frame[frame["run"] == widest].sort_values("dim")
    for name, column in [("constant", "constant"), ("trained_phi", "trained_phi"),
                         ("oracle", "oracle")]:
        axes[0].plot(block["dim"], block[column], marker="o", ms=3,
                     color=MAP_COLORS[name], label=name.replace("_", " "))
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("target directions the divergence is measured in")
    axes[0].set_ylabel("Sinkhorn divergence")
    axes[0].legend(frameon=False, loc="upper left")
    panel_letter(axes[0], "a")

    # Average duplicate JSONs so a twice-analysed config is not read as a sweep effect.
    kot = held_out.dropna(subset=["train_dims"])
    kot = kot[kot["method"] == "KOT"]
    per_config = kot.groupby(["condition", "train_pts", "train_dims"])["foscttm"].mean()
    floor = kot["floor"].mean()
    for (condition, points), group in per_config.groupby(["condition", "train_pts"]):
        axes[1].plot(group.index.get_level_values("train_dims"), group.to_numpy(),
                     marker="o", ms=3, label=f"{condition} {int(points)} pts")
    axes[1].axhline(floor, color=MAP_COLORS["constant"], ls="--", lw=0.8)
    axes[1].annotate("constant-map floor: no pairing information",
                     (kot["train_dims"].min(), floor), xytext=(0, -3),
                     textcoords="offset points", va="top", color=MAP_COLORS["constant"])
    axes[1].set_xscale("log", base=2)
    axes[1].set_ylim(0.0, floor * 1.2)
    axes[1].set_xlabel("align_dims the run was TRAINED with")
    axes[1].set_ylabel("held-out FOSCTTM")
    axes[1].legend(frameon=False, loc="lower left")
    panel_letter(axes[1], "b")
    save_figure(fig, path)


def collapse(args: argparse.Namespace) -> int:


    geometry, held_out = [], []
    for run in sorted(RUNS_ROOT.glob(args.pattern)):
        for path in sorted(run.glob(args.json_name)):
            result = json.loads(path.read_text())
            name = run.name if path.stem == "failure_analysis" else f"{run.name}:{path.stem}"
            geometry.extend(geometry_rows(name, result))
            held_out.append(held_out_row(name, result))
    if args.competitor_pattern:
        for run in sorted(RUNS_ROOT.glob(args.competitor_pattern)):
            path = run / "evaluation_baseline.json"
            if path.exists():
                held_out.append(competitor_held_out_row(run.name, json.loads(path.read_text())))
    geometry = [row for row in geometry if keep_row(row, args)]
    held_out = [row for row in held_out if keep_row(row, args)]
    if not geometry and not held_out:
        print(f"no {args.json_name} under {RUNS_ROOT}/{args.pattern}")
        return 1

    pd.set_option("display.width", 220, "display.max_columns", 40)
    frame = pd.DataFrame(geometry)
    if not frame.empty:
        frame = frame.sort_values(["dataset", "condition", "train_pts", "train_dims", "seed", "run", "dim"])
    held = pd.DataFrame(held_out).sort_values(["dataset", "method", "condition", "train_pts", "train_dims",
                                               "seed", "run"], na_position="last")
    if not frame.empty:
        print("--- geometry of the alignment objective. `oracle<const` False = the loss "
              "prefers a point mass to the truth")
        print(frame.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
        print()
    print("--- held-out cells, pairing revealed. KOT foscttm is gene-space; competitor "
          "foscttm is the method's latent. Only informative below `floor`")
    print(held.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    if args.figure and not frame.empty:
        geometry_figure(frame, held, Path(args.figure))
    if args.csv:
        csv_path = Path(args.csv)
        if not frame.empty:
            frame.to_csv(csv_path, index=False)
            print(f"\nwrote {csv_path}")
        held_name = (csv_path.stem[:-9] + "_heldout" if csv_path.stem.endswith("_geometry")
                     else csv_path.stem + "_heldout")
        held_path = csv_path.with_name(held_name + csv_path.suffix)
        held.to_csv(held_path, index=False)
        print(f"wrote {held_path}")
        kot = held[(held["method"] == "KOT") & held["train_dims"].notna()]
        if not kot.empty:
            keys = ["dataset", "condition", "train_dims", "train_pts"]
            values = [c for c in ["sd_ratio", "spread", "gene_r", "gene_rho", "cell_cos",
                                  "foscttm", "floor", "diversity", "headroom"]
                      if c in kot.columns]
            by_config = kot.groupby(keys, dropna=False)[values].agg(["mean", "std", "count"])
            by_config.columns = [f"{metric}_{stat}" for metric, stat in by_config.columns]
            config_path = csv_path.with_name(
                csv_path.stem.replace("_geometry", "_sweep_by_config") + csv_path.suffix)
            if config_path == csv_path:
                config_path = csv_path.with_name(csv_path.stem + "_by_config" + csv_path.suffix)
            by_config.reset_index().to_csv(config_path, index=False)
            print(f"wrote {config_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)

    child = sub.add_parser('runs')
    child.set_defaults(func=runs)
    child.add_argument("--pattern", default="*", help="glob over cache/chromatin/runs")
    child.add_argument("--csv", default=None, help="also write the full table here")
    child.add_argument("--expect-n-test", type=int, default=None,
                        help="drop rows scored on a different number of cells — they came "
                             "from another submission and are not comparable")

    child = sub.add_parser('collapse')
    child.set_defaults(func=collapse)
    child.add_argument("--pattern", default="collapse_bmmc_*",
                        help="glob over cache/chromatin/runs for KOT failure analyses")
    child.add_argument("--json-name", default="failure_analysis*.json")
    child.add_argument("--competitor-pattern", default="final_*_bmmc_seed*",
                        help="glob for evaluation_baseline.json rows; empty string skips")
    child.add_argument("--train-dims", type=int, nargs="*", default=None,
                        help="keep only these align_dims (KOT rows)")
    child.add_argument("--train-pts", type=int, default=None,
                        help="keep only this Sinkhorn sample size (KOT rows)")
    child.add_argument("--seeds", type=int, nargs="*", default=None,
                        help="keep only these seeds")
    child.add_argument("--csv", default=None, help="write the geometry table here")
    child.add_argument("--figure", default=None, help="write the two diagnostic panels here")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
