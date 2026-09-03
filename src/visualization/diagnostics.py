"""Supplementary panels: how the runs behaved, not how the method scored.

Four questions a referee asks that no main-text panel answers.

  Which checkpoint?   A run writes four (final, best_align, best_dyn, best_total). If
                      the reported number comes from the checkpoint that minimised the
                      reported metric, the metric was selected on. The panel shows what
                      each choice costs on the OTHER metric, which is the honest way to
                      state the trade rather than defending one choice.
  Do the terms fight? `grad_cos_late` is the angle between the alignment and dynamics
                      gradients late in training and `grad_mag_ratio_late` their size
                      ratio. Together they say whether lambda_dyn bought a compromise or
                      simply let one term dominate.
  How often failed?   Collapse, degeneracy and a dead dynamics term are all invisible in
                      a FOSCTTM ranking, and one of them sorts first. Counting them over
                      every run is the only way the reader learns the failure rate.
  Does the anchor do  The beta prior is a stabiliser, not ground truth. Sweeping the
  anything?           number of anchored proteins from 0 upward separates "it steadies
                      training" from "it is doing the fitting".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.visualization import METHOD_COLORS, dataset_label, method_label
from src.visualization.runs import CACHE_DIR, read_diagnostics, run_flags

RESULTS_DIR = Path("cache/results")

# Written by every KOT run, in the order a reader should compare them: the checkpoint
# with no selection first, then the three selected ones.
CHECKPOINTS = ["final", "best_total", "best_align", "best_dyn"]
# "best" is implied by the axis label, and repeating it three times is what makes the
# ticks collide in a third-width panel.
CHECKPOINT_LABELS = {"final": "final", "best_total": "total",
                     "best_align": "align", "best_dyn": "dyn"}

FLAG_LABELS = {"degenerate": "degenerate", "collapsed": "collapsed", "dyn-dead": "dyn-dead"}


def checkpoint_panel(ax, table: pd.DataFrame, metric: str, models: list[str]):
    """One strip per checkpoint, per model: what the selection choice costs.

    Points are runs, and the tick is the median. A bar of means would hide that the
    checkpoints agree on most runs and diverge on a few, which is the whole question.
    """
    for i, checkpoint in enumerate(CHECKPOINTS):
        for j, model in enumerate(models):
            values = table.loc[(table["checkpoint"] == checkpoint)
                               & (table["model"] == model), metric].dropna()
            if values.empty:
                continue
            x = i + (j - (len(models) - 1) / 2) * 0.26
            jitter = np.random.default_rng(i * 10 + j).uniform(-0.05, 0.05, len(values))
            ax.scatter(x + jitter, values, s=3, alpha=0.45, linewidths=0,
                       color=METHOD_COLORS[model], zorder=3,
                       label=method_label(model) if i == 0 else None)
            ax.plot([x - 0.11, x + 0.11], [values.median()] * 2, color="#1A1A1A",
                    lw=1.0, zorder=4)
    ax.set_xticks(range(len(CHECKPOINTS)))
    ax.set_xticklabels([CHECKPOINT_LABELS[c] for c in CHECKPOINTS])
    ax.set_xlabel("Checkpoint selected on")
    ax.set_xlim(-0.6, len(CHECKPOINTS) - 0.4)


def gradient_panel(ax, table: pd.DataFrame):
    """Gradient angle against gradient size ratio, one point per swept configuration.

    Both axes are mechanism, not score: a configuration where the two terms are almost
    orthogonal (cosine near zero) and the ratio is large is one where lambda_dyn changed
    the loss without changing the direction training moved in.
    """
    for dataset, marker, fill in (("bmmc_cite_retained", "o", "none"),
                                  ("pbmc_retained", "s", "full")):
        sub = table[table["dataset"] == dataset]
        if sub.empty:
            continue
        ax.plot(sub["grad_mag_ratio_late_mean"], sub["grad_cos_late_mean"], marker,
                fillstyle=fill, ms=3.6, markeredgewidth=0.8, lw=0,
                color=METHOD_COLORS["kot"], label=dataset_label(dataset), zorder=3)
    ax.set_xscale("log")
    ax.axhline(0, color="0.45", lw=0.7, ls=(0, (4, 2)), zorder=1)
    ax.set_xlabel("Align / dyn gradient ratio")
    ax.set_ylabel("Gradient cosine")


def collect_flags(datasets: list[str]) -> pd.DataFrame:
    """Every run's failure flags, counted per dataset. Unflagged runs count as clean."""
    rows = []
    for path in CACHE_DIR.rglob("*/diagnostics.json"):
        diagnostics = read_diagnostics(path)
        dataset = diagnostics.get("dataset")
        if dataset not in datasets or "mean_foscttm" not in diagnostics:
            continue
        flags = run_flags(diagnostics)
        rows.append({"dataset": dataset, "model": diagnostics.get("model", "?"),
                     "flags": flags.split(",") if flags else ["clean"]})
    return pd.DataFrame(rows)


def flag_panel(ax, flags: pd.DataFrame, colors: dict[str, str]):
    """Share of runs carrying each failure flag, per dataset, with the count annotated."""
    categories = ["clean"] + list(FLAG_LABELS)
    width = 0.8 / flags["dataset"].nunique()
    for i, dataset in enumerate(sorted(flags["dataset"].unique())):
        sub = flags[flags["dataset"] == dataset]
        counts = [int(sub["flags"].apply(lambda f, c=c: c in f).sum()) for c in categories]
        x = [j + i * width for j in range(len(categories))]
        # Short name plus n: the full dataset label makes a legend wide enough to sit
        # over the tallest bars whichever corner it is placed in.
        short = dataset_label(dataset).split()[0]
        ax.bar(x, [c / len(sub) for c in counts], width=width, color=colors[dataset],
               label=f"{short} (n = {len(sub)})", zorder=3)
        for xx, count in zip(x, counts):
            ax.text(xx, count / len(sub), str(count), ha="center", va="bottom",
                    color="0.35")
    ax.set_xticks([j + width / 2 for j in range(len(categories))])
    ax.set_xticklabels([FLAG_LABELS.get(c, c) for c in categories], rotation=45,
                       ha="right")
    ax.set_ylabel("Share of runs")


def anchor_panel(ax, table: pd.DataFrame, metric: str, colors: dict[str, str]):
    """A metric against the number of anchored proteins, one line per dataset."""
    for dataset in sorted(table["dataset"].unique()):
        sub = table[table["dataset"] == dataset].sort_values("beta_anchor_subset_n")
        ax.errorbar(sub["beta_anchor_subset_n"], sub[f"{metric}_mean"],
                    yerr=sub[f"{metric}_sd"], marker="o", ms=3, lw=1.0, elinewidth=0.7,
                    capsize=1.5, color=colors[dataset], label=dataset_label(dataset),
                    zorder=3)
    ax.set_xlabel("Anchored proteins")


def read_by_config(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / name)


def read_checkpoints() -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / "checkpoint_eval_summary.csv")
