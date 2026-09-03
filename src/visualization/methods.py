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

from src.visualization import METHOD_COLORS, method_label
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


def collect_runtimes(datasets: list[str]) -> pd.DataFrame:
    """Runtime and cell count for every curated run of every model on `datasets`."""
    curated = set(curated_runs())
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
    """Median runtime against cell count, one line per method, both axes log.

    Median over seeds rather than every point: a method with twelve seeds would
    otherwise dominate a method with one, and the spread being shown would be scheduler
    noise rather than anything about the method.
    """
    for model in models:
        sub = table[table["model"] == model]
        if sub.empty:
            continue
        grouped = sub.groupby("n_cells")["runtime"].median().sort_index()
        color = METHOD_COLORS.get(model, "#767676")
        ax.plot(grouped.index, grouped.values, marker="o", ms=3, lw=1.0, color=color,
                label=method_label(model), zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Cells")
    ax.set_ylabel("Runtime (s)")


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
