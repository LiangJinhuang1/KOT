"""Kinetic-parameter recovery figures.

The panels here carry the claim no alignment baseline can make: KOT fits a
per-protein degradation rate beta, and those rates can be checked against
independently measured protein half-lives. SCOT, moscot, GLUE and uniPort have
no kinetic parameters at all, so there is nothing to compare them against.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, MaxNLocator, NullFormatter
from scipy.stats import spearmanr

from src.visualization import METHOD_COLORS, dataset_label
from src.visualization.runs import curated_runs, read_diagnostics
from src.visualization.style import (
    STYLE_STATE, apply_style, figsize, ink, panel_letter, save_figure,
)


def clean_marker(name: str) -> str:
    """Strip assay suffixes so an axis shows CD4, not CD4_TotalSeqB (§5.6)."""
    for suffix in ("_TotalSeqB", "_TotalSeqA", "_TotalSeqC", "_control", "_prot"):
        if str(name).endswith(suffix):
            return str(name)[: -len(suffix)]
    return str(name)


def collect_beta(cache_dir: str | Path = "cache/training",
                 model: str = "kot") -> pd.DataFrame:
    """One row per (run, dataset, seed, marker): fitted beta against its literature target.

    Restricted to curated runs when a MANIFEST exists, for the same reason
    :func:`~tools.make_paper_figures.branch_identity_pairs` is: the uncurated sweep
    arms differ in lambda_dyn and kappa bounds, so pooling them reports a median
    over configurations that were never meant to be compared. Falls back to every
    run when there is no manifest, so a fresh checkout still plots.
    """
    curated = curated_runs(cache_dir)
    rows = []
    for dj in Path(cache_dir).rglob("*/diagnostics.json"):
        d = read_diagnostics(dj)
        names = d.get("beta_anchor_names")
        if not names:
            continue
        targets, fitted = d.get("beta_anchor_targets"), d.get("beta_anchor_final")
        if not targets or not fitted:
            continue
        if len(targets) != len(names) or len(fitted) != len(names):
            continue
        # run/model/dataset[/seed_N]/diagnostics.json — the seed level is absent for
        # the deterministic baselines, where parts[3] is the filename, not a seed.
        parts = dj.relative_to(cache_dir).parts
        if len(parts) < 4 or parts[1] != model:
            continue
        run, dataset = parts[0], parts[2]
        if curated and run not in curated:
            continue
        seed = parts[3].replace("seed_", "") if len(parts) > 4 else "-"
        for name, t, f in zip(names, targets, fitted):
            rows.append({"run": run, "dataset": dataset, "seed": seed,
                         "marker": clean_marker(name),
                         "target": float(t), "fitted": float(f)})
    return pd.DataFrame(rows)


def beta_limits(df: pd.DataFrame, datasets) -> tuple[float, float]:
    """Shared fitted-beta y-range for the recovery panels, taken from the data.

    Panels a and b have to share a y-axis to be read against each other, and a
    hardcoded range would silently drop a point that landed outside it while the
    n reported in the corner still counted it.
    """
    fitted = pd.concat([df.loc[df["dataset"] == ds, "fitted"] for ds in datasets])
    lo, hi = float(fitted.min()), float(fitted.max())
    pad = max((hi - lo) * 0.22, 0.05)
    return lo - pad, hi + pad


def spearman(x, y) -> float:
    return float(spearmanr(x, y).statistic)


def recovery_panel(ax, sub: pd.DataFrame, *, title: str, ylim: tuple[float, float],
                   n_label: int = 3, show_ylabel: bool = True):
    """Fitted beta against literature beta, per marker, on a log x-axis."""
    # Average over seeds: one point per marker, not one per (marker, seed).
    g = sub.groupby("marker").agg(target=("target", "median"),
                                  fitted=("fitted", "median"),
                                  n=("fitted", "size")).reset_index()
    rho = spearman(g["target"], g["fitted"])
    color = METHOD_COLORS["kot"]

    ax.scatter(g["target"], g["fitted"], s=9, color=color, alpha=0.8,
               linewidths=0.4, edgecolors="white", zorder=4)

    # Identity is where a perfectly scaled recovery would sit; the gap between
    # the cloud and this line IS the range compression, so it has to be shown.
    lo = min(g["target"].min(), g["fitted"].min()) * 0.7
    hi = max(g["target"].max(), g["fitted"].max()) * 1.4
    ax.plot([lo, hi], [lo, hi], color="0.55", lw=0.7, ls=(0, (4, 2)), zorder=2)

    # Label the extremes with leader lines (§6.9). The lowest-target markers sit
    # almost on top of each other in x, so consecutive labels are stepped
    # outward rather than all placed at the same offset.
    low = g.nsmallest(n_label, "target").sort_values("target")
    high = g.nlargest(n_label, "target").sort_values("target", ascending=False)
    for side, rows_ in (("low", low), ("high", high)):
        for rank, (_, r) in enumerate(rows_.iterrows()):
            # Both labels sit above their point and lean inward, so neither can
            # run off the left spine or collide with the other end's label.
            ax.annotate(
                r["marker"], xy=(r["target"], r["fitted"]),
                xytext=(5 if side == "low" else -5, 9 + rank * 7),
                textcoords="offset points", fontsize=5.5, color=ink(color),
                ha="left" if side == "low" else "right", va="bottom",
                arrowprops=dict(arrowstyle="-", lw=0.4, color="0.65",
                                shrinkA=0, shrinkB=1.5))

    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(*ylim)
    # Explicit decades strictly inside the limits: a tick sitting on the panel
    # edge collides with the neighbouring panel's first tick (§3.4).
    decades = [10.0 ** k for k in range(-4, 4)]
    ticks = ([t for t in decades if lo * 1.6 < t < hi / 1.6]
             or [t for t in decades if lo < t < hi])
    ax.set_xticks(ticks)
    ax.set_xticklabels(["%g" % t for t in ticks])
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=(), numticks=1))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
    ax.set_xlabel("Literature rate")
    if show_ylabel:
        ax.set_ylabel("Fitted $\\beta$")
    ax.set_title(title)
    ax.text(0.97, 0.06, f"$\\rho$ = {rho:.2f}\nn = {len(g)}", transform=ax.transAxes,
            fontsize=6, color="0.25", ha="right", va="bottom", linespacing=1.4)
    return rho, g


def plot_beta_recovery(df: pd.DataFrame, save_path: str | Path,
                       *, n_label: int = 1, datasets=("bmmc_cite_retained", "pbmc_retained"),
                       backend_pairs=(("bmmc_cite_retained", "bmmc_cite_regvelo"),
                                      ("pbmc_retained", "pbmc_regvelo"))):
    """Three panels: recovery on two datasets, then the velocity-backend comparison."""
    apply_style()
    fig, axes = plt.subplots(1, 4, figsize=figsize("full", 2.1), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.08, wspace=0.07)

    ylim = beta_limits(df, datasets)
    notes = []
    for i, ds in enumerate(datasets):
        sub = df[df["dataset"] == ds]
        rho, g = recovery_panel(axes[i], sub, title=dataset_label(ds), ylim=ylim,
                                n_label=n_label, show_ylabel=(i == 0))
        panel_letter(axes[i], "ab"[i])
        notes.append(f"{dataset_label(ds)}: rho={rho:.3f}, {len(g)} markers, "
                     f"target CV {g.target.std()/g.target.mean():.2f} vs "
                     f"fitted CV {g.fitted.std()/g.fitted.mean():.2f}")

    # Panel c: does a regulatory-informed velocity backend change the recovery?
    ax = axes[2]
    # Both lines are KOT, so both carry KOT's colour and the dataset is carried by
    # marker shape — open circle BMMC, filled square PBMC, exactly as in fig 3a.
    # Reusing MODALITY_COLORS here would have made the same purple/gold mean
    # "RNA vs protein" in one figure and "BMMC vs PBMC" in this one.
    labels = {"bmmc_cite_retained": "BMMC", "pbmc_retained": "PBMC"}
    markers = {"bmmc_cite_retained": "o", "pbmc_retained": "s"}
    color = ink(METHOD_COLORS["kot"])
    for scvelo_ds, regvelo_ds in backend_pairs:
        pts = []
        for x, ds in [(0, scvelo_ds), (1, regvelo_ds)]:
            sub = df[df["dataset"] == ds]
            if sub.empty:
                pts = []
                break
            g = sub.groupby("marker")[["target", "fitted"]].median()
            pts.append((x, spearman(g["target"], g["fitted"])))
        if len(pts) != 2:
            continue
        xs, ys = zip(*pts)
        marker = markers.get(scvelo_ds, "o")
        ax.plot(xs, ys, "-", color=color, lw=1.0, zorder=4)
        ax.plot(xs, ys, marker, color=color, ms=3.4, zorder=5,
                markerfacecolor=color if marker == "s" else "white")
        ax.annotate(labels.get(scvelo_ds, scvelo_ds), xy=(xs[1], ys[1]),
                    xytext=(4, 0), textcoords="offset points",
                    fontsize=6, color=color, va="center", ha="left")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["scVelo", "RegVelo"])
    ax.tick_params(axis="x", pad=2)
    ax.set_xlim(-0.25, 1.55)
    ax.set_ylabel("Rank recovery $\\rho$")
    ax.set_title("Velocity backend")
    panel_letter(ax, "c")

    # d: the seed spread panels a-b cannot show. BMMC carries the larger marker panel,
    # so it is the one worth resolving marker by marker.
    ax = axes[3]
    table = beta_spread_panel(ax, df, datasets[0])
    ax.set_ylabel(r"$\beta$")
    ax.set_title(f"{dataset_label(datasets[0])}, per marker")
    ax.legend(loc="lower right", frameon=False)
    panel_letter(ax, "d")
    notes.append(f"{dataset_label(datasets[0])}: {len(table)} markers with "
                 f"{int(table['seeds'].median())} seeds each")

    save_figure(fig, Path(save_path).with_suffix(""))
    return notes


def beta_spread_panel(ax, df: pd.DataFrame, dataset: str, *, n_label: int = 0):
    """Every marker's fitted beta and its across-seed range, against the target.

    The recovery scatter in panels a-b shows one point per marker and so cannot say
    whether a marker sits where it does reliably or only on average. Here each marker
    keeps its seed range, sorted by the literature target, and the target itself is
    drawn as a separate series -- the vertical gap between the two is the error the
    rank correlation summarises into a single number.

    No marker names by default. The claim is the COMPRESSION between the two series --
    literature beta spans two orders of magnitude where fitted beta spans a factor of
    two -- and panels a-b already name the markers worth naming.
    """
    grouped = df[df["dataset"] == dataset].groupby("marker")
    table = grouped.agg(target=("target", "median"), fitted=("fitted", "median"),
                        lo=("fitted", "min"), hi=("fitted", "max"),
                        seeds=("fitted", "size"))
    table = table.sort_values("target").reset_index()
    x = np.arange(len(table))

    ax.vlines(x, table["lo"], table["hi"], color=METHOD_COLORS["kot"], lw=0.7,
              alpha=0.55, zorder=2)
    ax.scatter(x, table["fitted"], s=4, color=METHOD_COLORS["kot"], linewidths=0,
               zorder=3, label="fitted")
    ax.scatter(x, table["target"], s=4, marker="_", color="#767676", linewidths=0.9,
               zorder=3, label="literature")
    # The ends of the range only: naming 52 markers needs a tick per marker, and the
    # panel's claim is the compression between the two series, not any one marker.
    ends = [(0, (2, -8), "left"), (len(table) - 1, (-2, 6), "right")][:n_label]
    for i, offset, ha in ends:
        ax.annotate(clean_marker(table["marker"][i]), (x[i], table["target"][i]),
                    textcoords="offset points", xytext=offset, ha=ha,
                    fontsize=STYLE_STATE["ladder"][2], color="0.35")
    ax.set_xlim(-1, len(table))
    ax.set_xlabel("Marker, by literature rate")
    ax.set_yscale("log")
    return table
