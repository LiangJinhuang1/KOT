"""Predicted protein against measured protein — the comparison only phi(r) supports.

With ``kot_use_feature_space`` the RNA-side map lands in the ADT panel itself, so a
prediction can be read off directly instead of being imputed from neighbours in a latent
space. Every other method here would need a kNN step first, and the step, not the model,
would then be carrying part of the result.

WHICH ROWS COUNT. ``protein_eval_per_protein.csv`` pools an evaluation sweep: velocity
ablations, S permutations and the anchor-subset arms all wrote into it. Reading it
without a filter averages corrupted arms into the headline number, so `canonical_rows`
keeps only the configuration config/training.yaml declares and the runs that were not
part of the anchor-subset sweep (`beta_anchor_subset_n` blank), which used the config's
own anchors.

WHAT `in_kinetics` DOES NOT MEAN. Proteins the kinetics term covers score higher than
those it does not, but coverage is not random: a protein enters the kinetic mask only
when its gene has usable velocity, which already selects well-measured genes. The gap is
worth plotting and is not by itself evidence that the ODE caused it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.visualization import METHOD_COLORS

RESULTS_DIR = Path("cache/results")

# The configuration config/training.yaml declares for the real datasets.
CANONICAL = {"lr_beta": 0.001, "lr_warmup_epochs": 300,
             "kot_velocity_ablation": "none", "kot_s_permute": False}

# Covered by the kinetics term, or reached by alignment alone.
COVERAGE_COLORS = {True: METHOD_COLORS["kot"], False: "#767676"}
COVERAGE_LABELS = {True: "in kinetics", False: "alignment only"}


def canonical_rows(dataset: str) -> pd.DataFrame:
    """Per-protein evaluation rows for one dataset, sweep arms excluded."""
    df = pd.read_csv(RESULTS_DIR / "protein_eval_per_protein.csv")
    keep = df["dataset"] == dataset
    for column, value in CANONICAL.items():
        keep &= df[column].astype(type(value)) == value
    return df[keep & df["beta_anchor_subset_n"].isna()].copy()


def per_protein_median(df: pd.DataFrame, metric: str = "spearman") -> pd.DataFrame:
    """One row per protein: median across seeds, the seed range, and its coverage."""
    grouped = df.groupby(["protein", "in_kinetics"])[metric]
    out = grouped.agg(["median", "min", "max", "count"]).reset_index()
    return out.sort_values("median", ascending=False).reset_index(drop=True)


def coverage_strip_panel(ax, frames: dict[str, pd.DataFrame], metric: str = "spearman"):
    """Per-protein score split by kinetics coverage, one column pair per dataset.

    Points are per-protein medians over seeds, not per-seed values: a protein measured
    twelve times is one protein, and plotting every seed would let a well-covered panel
    outvote a sparse one twelve to one.
    """
    positions, labels = [], []
    for i, (name, df) in enumerate(frames.items()):
        per_protein = per_protein_median(df, metric)
        for j, covered in enumerate((True, False)):
            values = per_protein.loc[per_protein["in_kinetics"] == covered, "median"]
            x = 2 * i + j
            jitter = np.random.default_rng(0).normal(0, 0.06, len(values))
            ax.scatter(x + jitter, values, s=3.5, alpha=0.55, linewidths=0,
                       color=COVERAGE_COLORS[covered], zorder=3)
            ax.plot([x - 0.28, x + 0.28], [values.median()] * 2, color="#1A1A1A",
                    lw=1.0, zorder=4)
            positions.append(x)
            labels.append(COVERAGE_LABELS[covered])
        ax.text(2 * i + 0.5, 1.0, name, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=90)
    ax.set_xlim(-0.6, positions[-1] + 0.6)


def ranked_protein_panel(ax, df: pd.DataFrame, metric: str = "spearman",
                         n_label: int = 3):
    """Every protein in rank order, seed range as a whisker, extremes named.

    Naming all of them would need a tick per protein; naming the ends says where the
    range runs and leaves the panel readable.
    """
    per_protein = per_protein_median(df, metric)
    x = np.arange(len(per_protein))
    for covered in (True, False):
        mask = (per_protein["in_kinetics"] == covered).to_numpy()
        ax.vlines(x[mask], per_protein["min"][mask], per_protein["max"][mask],
                  color=COVERAGE_COLORS[covered], lw=0.7, alpha=0.5, zorder=2)
        ax.scatter(x[mask], per_protein["median"][mask], s=5,
                   color=COVERAGE_COLORS[covered], linewidths=0, zorder=3,
                   label=COVERAGE_LABELS[covered])
    # The named markers sit at almost the same height at each end of the ranking, so
    # they are stacked into the empty side of the curve rather than offset uniformly:
    # the leading group downwards, under a curve that has not descended yet, and the
    # trailing group upwards and leftwards, away from the axis edge.
    leaders = [(i, -1, 4, "left") for i in range(n_label)]
    trailers = [(i, 1, -4, "right") for i in range(len(x) - n_label, len(x))]
    for group in (leaders, trailers):
        for step, (i, sign, dx, ha) in enumerate(group):
            ax.annotate(clean_protein(per_protein["protein"][i]),
                        (x[i], per_protein["median"][i]), textcoords="offset points",
                        xytext=(dx, sign * (7 + 9 * step)), color="0.35", ha=ha,
                        arrowprops=dict(arrowstyle="-", lw=0.5, color="0.7",
                                        shrinkA=0, shrinkB=1))
    ax.set_xlim(-1, len(x))
    return per_protein


def clean_protein(name: str) -> str:
    """The marker name without its assay suffix — 'CD4_TotalSeqB' is not a marker."""
    return str(name).split("_TotalSeq")[0].replace("_control", "")


def pooled_calibration_panel(ax, predicted: np.ndarray, observed: np.ndarray, *,
                             gridsize: int = 45, cmap: str = "Greys"):
    """Every cell and every protein at once, each protein standardised first.

    Pooling raw values would let the widest-ranging markers set the axes and smear the
    rest into the origin; standardising per protein asks the question the panel is for,
    which is whether the shape is right, not whether two panels share a scale.
    """
    pred = stats.zscore(predicted, axis=0)
    obs = stats.zscore(observed, axis=0)
    ax.hexbin(obs.ravel(), pred.ravel(), gridsize=gridsize, cmap=cmap, bins="log",
              mincnt=1, linewidths=0, rasterized=True)
    lim = float(np.percentile(np.abs(np.concatenate([obs.ravel(), pred.ravel()])), 99.5))
    ax.plot([-lim, lim], [-lim, lim], color="0.45", lw=0.7, ls=(0, (4, 2)), zorder=4)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    rho = stats.spearmanr(obs.ravel(), pred.ravel()).statistic
    ax.text(0.03, 0.97, f"ρ = {rho:.2f}", transform=ax.transAxes, ha="left", va="top")
    return rho
