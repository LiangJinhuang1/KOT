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
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.visualization.style import apply_style, figsize, panel_letter, save_figure

RUNS_ROOT = Path("cache/chromatin/runs")
MAP_COLORS = {"constant": "#999999", "trained_phi": "#D55E00", "oracle": "#0072B2"}


def geometry_rows(name: str, result: dict) -> list[dict]:
    return [{
        "run": name,
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
        "run": name,
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

    # A run analysed twice contributes two rows for one training configuration; the mean
    # is the run's score, and plotting both would read as a sweep effect that is not one.
    per_config = held_out.groupby(["train_pts", "train_dims"])["foscttm"].mean()
    floor = held_out["floor"].mean()
    for points, group in per_config.groupby("train_pts"):
        axes[1].plot(group.index.get_level_values("train_dims"), group.to_numpy(),
                     marker="o", ms=3, label=f"{points} points")
    axes[1].axhline(floor, color=MAP_COLORS["constant"], ls="--", lw=0.8)
    axes[1].annotate("constant-map floor: no pairing information",
                     (held_out["train_dims"].min(), floor), xytext=(0, -3),
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
    parser.add_argument("--pattern", default="*", help="glob over cache/chromatin/runs")
    parser.add_argument("--json-name", default="failure_analysis*.json")
    parser.add_argument("--csv", default=None, help="also write the geometry table here")
    parser.add_argument("--figure", default=None, help="write the two diagnostic panels here")
    args = parser.parse_args()

    geometry, held_out = [], []
    for run in sorted(RUNS_ROOT.glob(args.pattern)):
        for path in sorted(run.glob(args.json_name)):
            result = json.loads(path.read_text())
            name = run.name if path.stem == "failure_analysis" else f"{run.name}:{path.stem}"
            geometry.extend(geometry_rows(name, result))
            held_out.append(held_out_row(name, result))
    if not geometry:
        print(f"no {args.json_name} under {RUNS_ROOT}/{args.pattern}")
        return 1

    pd.set_option("display.width", 220, "display.max_columns", 40)
    frame = pd.DataFrame(geometry).sort_values(["train_pts", "train_dims", "run", "dim"])
    held = pd.DataFrame(held_out).sort_values(["train_pts", "train_dims", "run"])
    print("--- geometry of the alignment objective. `oracle<const` False = the loss "
          "prefers a point mass to the truth")
    print(frame.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print("\n--- held-out cells, pairing revealed. foscttm is only informative below "
          "`floor`, the score a constant map gets on the same cells")
    print(held.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    if args.figure:
        geometry_figure(frame, held, Path(args.figure))
    if args.csv:
        frame.to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
