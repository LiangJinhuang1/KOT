#!/usr/bin/env python3
"""Every `failure_analysis.json` in one place, as the two tables the diagnosis needs.

GEOMETRY   one row per (run, measurement dimension). The three maps' divergences side by
           side, so `oracle_beats_constant` can be read down the dimension axis: where it
           is False the objective PREFERS a point mass to the truth, and no optimiser
           setting fixes that. `blur/dist` says whether the same row is instead just
           over-smoothed, which has the opposite fix.

HELD OUT   one row per run. What the map is worth on test cells once the pairing is
           revealed, against the floor a constant map scores on the same cells.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.visualization.style import apply_style, figsize, panel_letter, save_figure

RUNS_ROOT = Path("cache/chromatin/runs")
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
    """The crossover, and what it bought. One line per map, dimension on a log axis.

    Left panel is the whole diagnosis: where the grey constant line sits BELOW the blue
    oracle line, minimising the alignment loss is minimising the wrong thing.
    """
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=figsize("full", 2.1))
    # The run measured in the most dimensions, so the panel shows the crossover itself
    # rather than only the side of it a particular sweep arm was scored on.
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

    # Competitors have no train_dims; a run analysed twice is averaged so two JSONs
    # for one configuration do not read as a sweep effect.
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pattern", default="collapse_bmmc_*",
                        help="glob over cache/chromatin/runs for KOT failure analyses")
    parser.add_argument("--json-name", default="failure_analysis*.json")
    parser.add_argument("--competitor-pattern", default="final_*_bmmc_seed*",
                        help="glob for evaluation_baseline.json rows; empty string skips")
    parser.add_argument("--train-dims", type=int, nargs="*", default=None,
                        help="keep only these align_dims (KOT rows)")
    parser.add_argument("--train-pts", type=int, default=None,
                        help="keep only this Sinkhorn sample size (KOT rows)")
    parser.add_argument("--seeds", type=int, nargs="*", default=None,
                        help="keep only these seeds")
    parser.add_argument("--csv", default=None, help="write the geometry table here")
    parser.add_argument("--figure", default=None, help="write the two diagnostic panels here")
    args = parser.parse_args()

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


if __name__ == "__main__":
    raise SystemExit(main())
