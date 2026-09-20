"""Comparing methods on axes FOSCTTM does not cover: rank shape, and cost.

Two panels that need nothing new trained. The kNN alignment curve re-reads the aligned
embeddings; the runtime scaling re-reads `runtime_seconds`, which every run writes.

RUNTIME IS REPORTED, NOT BENCHMARKED. The runs were launched over months on whatever
node was free, and the hardware each one used is not recorded. The panel is therefore an
order-of-magnitude statement — MaxFuse takes tens of minutes on 90k cells where KOT takes
minutes — and the caption has to say so. Presenting it as a controlled timing comparison
would be claiming an experiment that was never run.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.visualization import METHOD_COLORS, dataset_style, method_label
from src.visualization.runs import CACHE_DIR, curated_runs, is_collapsed, read_diagnostics

ALIGNED_RNA = "aligned_rna.npy"


def curated_method_runs(dataset: str, models: list[str]) -> dict[str, Path]:
    """One run directory per model: the median-FOSCTTM seed among curated runs.

    Median rather than best, for the same reason `pick_run` uses it — a grid of every
    method's luckiest seed is not a comparison.
    """
    curated = set(curated_runs())
    found: dict[str, list[tuple[float, Path]]] = {m: [] for m in models}
    for path in CACHE_DIR.rglob(f"*/{dataset}/seed_*/{ALIGNED_RNA}"):
        run_dir = path.parent
        model = run_dir.parent.parent.name
        if model not in found or run_dir.relative_to(CACHE_DIR).parts[0] not in curated:
            continue
        diagnostics = read_diagnostics(run_dir / "diagnostics.json")
        if "mean_foscttm" not in diagnostics or is_collapsed(diagnostics):
            continue
        found[model].append((float(diagnostics["mean_foscttm"]), run_dir))
    out = {}
    for model, hits in found.items():
        if hits:
            out[model] = sorted(hits)[len(hits) // 2][1]
    return out


def collect_runtimes(datasets: list[str], allowed: set | None = None) -> pd.DataFrame:
    """Runtime and cell count per run, for `allowed` run dirs or else the curated set.

    `allowed` exists so a figure can cost the SAME runs it plots elsewhere. Falling back
    to the manifest had Fig. 9 timing the superseded `*_scvelo_*` dirs in panel b while
    panel a drew the canonical ones.
    """
    curated = set(allowed) if allowed is not None else set(curated_runs())
    rows = []
    for path in CACHE_DIR.rglob("*/diagnostics.json"):
        parts = path.relative_to(CACHE_DIR).parts
        if parts[0] not in curated:
            continue
        diagnostics = json.loads(path.read_text())
        dataset, model = diagnostics.get("dataset"), diagnostics.get("model")
        runtime = diagnostics.get("runtime_seconds")
        aligned = path.parent / ALIGNED_RNA
        if dataset not in datasets or not model or not runtime or not aligned.exists():
            continue
        rows.append({"model": model, "dataset": dataset, "runtime": float(runtime),
                     "n_cells": int(np.load(aligned, mmap_mode="r").shape[0])})
    return pd.DataFrame(rows)


def runtime_panel(ax, table: pd.DataFrame, models: list[str]):
    """Median runtime per method, one marker per dataset, methods ordered by cost.

    Not runtime against cell count: with two CITE-seq panels that axis carries exactly
    two values, so a log-log scatter invites a scaling reading that two points on
    unrecorded hardware cannot support. Methods on the y-axis and one log runtime axis
    answers the question the panel is actually for -- which method is expensive -- and
    matches the layout of the Fig. 3a benchmark.

    Median over seeds rather than every point: a method with twelve seeds would
    otherwise dominate a method with one, and the spread being shown would be scheduler
    noise rather than anything about the method.
    """
    # Order by cost on the LARGER panel, not by a median pooled over both: seed counts
    # differ per dataset (MaxFuse has one BMMC run against twelve PBMC ones), so a pooled
    # median ranked the method with the second-highest BMMC runtime as the cheapest.
    present = [m for m in models if not table[table["model"] == m].empty]
    biggest = table.loc[table["n_cells"].idxmax(), "dataset"]

    def cost(model: str) -> float:
        sub = table[(table["model"] == model) & (table["dataset"] == biggest)]
        if sub.empty:
            sub = table[table["model"] == model]
        return float(sub["runtime"].median())

    order = sorted(present, key=cost)
    for row, model in enumerate(order):
        sub = table[table["model"] == model]
        for dataset, group in sub.groupby("dataset"):
            marker, colour, _ = dataset_style(dataset)
            ax.plot(group["runtime"].median(), row, marker=marker, ms=3.6, lw=0,
                    color=colour, markerfacecolor=colour, markeredgecolor="none",
                    markeredgewidth=0, zorder=3)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([method_label(m) for m in order])
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("Runtime (s)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def knn_curve_panel(ax, curves: dict[str, np.ndarray], fractions: np.ndarray):
    """One line per method: chance of the true partner falling inside the top k."""
    for model, values in curves.items():
        color = METHOD_COLORS.get(model, "#767676")
        ax.plot(100 * fractions, values, lw=1.2, color=color, label=method_label(model),
                zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("Neighbourhood size (% of cells)")
    ax.set_ylabel("True partner recovered")
    ax.set_ylim(0, 1.02)
