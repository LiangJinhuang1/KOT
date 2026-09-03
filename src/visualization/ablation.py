"""The velocity-corruption ladder: how much of the alignment is the velocity?

One KOT arm per corruption level, twelve seeds each, nothing else changed. Ordered by
how much velocity information survives, eight table rows become a dose-response — the
only form in which "the dynamics term uses the velocity" is falsifiable rather than
asserted.

Two arms sit OUTSIDE that order on purpose:

  reverse   the field is not degraded, it points the wrong way. It stays smooth and
            self-consistent, so the ODE residual can still be driven down: the JVP-RHS
            cosine stays high while FOSCTTM collapses. Placed inside the ladder that
            reads as a non-monotonicity in the dose-response; placed beside it, it
            reads as the different failure it is.
  noDyn     the term is absent, not starved. It is the floor the ladder descends
            towards, not a rung on it.

permS appears on real data only and corrupts the OTHER input — it permutes the
ADT→gene links instead of the velocity. It belongs on the real-data panel and nowhere
near the synthetic ladder, because a reader who sees them on one axis will read
"corruption" as one quantity when it is two.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.visualization import METHOD_COLORS, method_label
from src.visualization.style import chance_line

RESULTS_DIR = Path("cache/results")

# Ordered by how much of the true velocity field survives.
LADDER = ["real", "corrupt0.25", "corrupt0.5", "corrupt0.75", "shuffle", "zero"]
OFF_LADDER = ["reverse", "noDyn"]

# The ladder rungs are FRACTIONS CORRUPTED, so the axis they sit on has to be labelled
# "corrupted", not "retained" -- 25% means a quarter of the field replaced, not a
# quarter kept.
ARM_LABELS = {
    "real":        "none",
    "corrupt0.25": "25%",
    "corrupt0.5":  "50%",
    "corrupt0.75": "75%",
    "shuffle":     "shuffle",
    "zero":        "zero",
    "reverse":     "reverse",
    "noDyn":       "no term",
    "permS":       "permuted S",
}

# Markers reused from the Fig. 3 benchmark so one dataset keeps one glyph.
DATASET_MARKERS = {"bmmc_cite_retained": "o", "pbmc_retained": "s"}
DATASET_FILL = {"bmmc_cite_retained": "none", "pbmc_retained": "full"}


def read_arms(name: str, lr_beta: float, warmup: int) -> pd.DataFrame:
    """One ablation summary, restricted to a single training configuration.

    The sweep ran every arm under two (lr_beta, warmup) settings. Pooling them would
    average a configuration difference into the corruption effect the panel is about,
    so the caller names which configuration the figure reports.
    """
    df = pd.read_csv(RESULTS_DIR / name)
    keep = (df["lr_beta"].astype(float) == lr_beta) & (df["lr_warmup_epochs"].astype(int) == warmup)
    return df[keep].copy()


def arm_series(df: pd.DataFrame, model: str, metric: str, arms: list[str]):
    """Mean and sd of `metric` for one model over `arms`, in the order given."""
    rows = df[df["model"] == model].set_index("arm")
    present = [a for a in arms if a in rows.index]
    mean = [float(rows.loc[a, f"{metric}_mean"]) for a in present]
    sd = [float(rows.loc[a, f"{metric}_sd"]) for a in present]
    return present, mean, sd


def ladder_panel(ax, df: pd.DataFrame, metric: str, models: list[str], *,
                 ylabel: str, chance: float | None = None):
    """Dose-response over the velocity ladder, with the off-ladder arms beside it.

    The two groups share a y-axis and are separated by a rule rather than a gap in the
    x values, so the connected ladder cannot be misread as continuing through arms that
    are not degradations of it.
    """
    positions = {arm: i for i, arm in enumerate(LADDER)}
    gap = len(LADDER) + 0.6
    positions.update({arm: gap + i for i, arm in enumerate(OFF_LADDER)})

    for model in models:
        color = METHOD_COLORS[model]
        arms, mean, sd = arm_series(df, model, metric, LADDER)
        x = [positions[a] for a in arms]
        ax.plot(x, mean, color=color, lw=1.2, marker="o", ms=2.6,
                label=method_label(model), zorder=3)
        ax.fill_between(x, [m - s for m, s in zip(mean, sd)],
                        [m + s for m, s in zip(mean, sd)],
                        color=color, alpha=0.18, lw=0, zorder=2)

        arms, mean, sd = arm_series(df, model, metric, OFF_LADDER)
        ax.errorbar([positions[a] for a in arms], mean, yerr=sd, fmt="o", ms=2.6,
                    color=color, lw=0, elinewidth=0.8, capsize=1.5, zorder=3)

    ax.axvline(gap - 0.8, color="0.8", lw=0.7, zorder=1)
    order = LADDER + OFF_LADDER
    ax.set_xticks([positions[a] for a in order])
    ax.set_xticklabels([ARM_LABELS[a] for a in order], rotation=45, ha="right")
    ax.set_xlim(-0.4, positions[OFF_LADDER[-1]] + 0.4)
    ax.set_ylabel(ylabel)
    if chance is not None:
        chance_line(ax, chance)


def real_panel(ax, df: pd.DataFrame, metric: str, datasets: list[str], *, xlabel: str,
               chance: float | None = None):
    """The same corruptions on real CITE-seq, one row per arm, one glyph per dataset.

    Horizontal because the arm names are words: rotated word ticks on two panels of a
    three-panel row cost more height than the panel has.
    """
    # permS first: it corrupts the LINKAGE, not the velocity, so it heads the panel
    # separated by a rule rather than sitting inside a velocity ladder it is not on.
    arms = [a for a in ["permS"] + LADDER + OFF_LADDER if a in set(df["arm"])]
    color = METHOD_COLORS["kot"]
    for dataset in datasets:
        sub = df[df["dataset"] == dataset].set_index("arm")
        present = [a for a in arms if a in sub.index]
        y = [len(arms) - 1 - arms.index(a) for a in present]
        ax.errorbar([float(sub.loc[a, f"{metric}_mean"]) for a in present], y,
                    xerr=[float(sub.loc[a, f"{metric}_sd"]) for a in present],
                    fmt=DATASET_MARKERS[dataset], fillstyle=DATASET_FILL[dataset],
                    ms=3.4, color=color, lw=0, elinewidth=0.8, capsize=1.5,
                    markeredgewidth=0.8, zorder=3)
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels([ARM_LABELS[a] for a in reversed(arms)])
    ax.set_ylim(-0.6, len(arms) - 0.4)
    ax.axhline(len(arms) - 1.5, color="0.8", lw=0.7, zorder=1)
    ax.set_xlabel(xlabel)
    if chance is not None:
        chance_line(ax, chance, axis="x")
