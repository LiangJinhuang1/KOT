"""Fitted kinetic rates versus the literature-derived priors used in training.

These panels measure agreement with anchor targets, not independent recovery.
Unanchored-protein comparisons live in control_effects.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, MaxNLocator, NullFormatter
from scipy.stats import spearmanr

from src.visualization import METHOD_COLORS, dataset_label, dataset_style
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


# The runs the reported tables are built from: setting B (lambda_dyn 1000, beta learning
# rate 1e-3, 300-epoch dynamics ramp) at twelve seeds. The curated MANIFEST entries
# (bmmc_scvelo_kot and siblings) are the earlier canonical lambda_dyn=100 runs at five
# seeds, so reading those made this figure describe a different model than tables 01 and
# 09 -- KOT scored 0.178 FOSCTTM on BMMC there against 0.129 in the table. Table 09 takes
# its scVelo arm from the ablation launch and its RegVelo arm from the full-panel launch;
# both are setting B, and the figure pairs them the same way so the two agree row for row.
REPORTED_BETA_RUNS = {
    "bmmc_cite_retained": "run_20260829_022744_shuf_bmmc_scv_cfgB_warm300_beta1p0e-3_realVel",
    "pbmc_retained": "run_20260829_022744_shuf_pbmc_scv_cfgB_warm300_beta1p0e-3_realVel",
    "bmmc_cite_regvelo": "run_20260828_101018_full_bmmc_rgv_S2cfgB_warm300_beta1p0e-3_ramp300",
    "pbmc_regvelo": "run_20260828_101018_full_pbmc_rgv_S2cfgB_warm300_beta1p0e-3_ramp300",
}


def collect_beta(cache_dir: str | Path = "cache/training",
                 model: str = "kot",
                 runs: dict[str, str] | None = None) -> pd.DataFrame:
    """One row per (run, dataset, seed, marker): fitted beta against its literature target.

    `runs` pins each dataset to the single run the paper reports it from, defaulting to
    :data:`REPORTED_BETA_RUNS`. Pinning rather than pooling matters here because the
    sweep arms differ in lambda_dyn, beta learning rate and kappa bounds, so pooling
    reports a median over configurations that were never meant to be compared -- and
    because seed counts differ between launches, which silently changes the n this
    figure prints. Falls back to the curated MANIFEST runs, then to every run, when the
    pinned directories are absent, so a fresh or partial checkout still plots.
    """
    runs = REPORTED_BETA_RUNS if runs is None else runs
    pinned = {ds: run for ds, run in runs.items() if (Path(cache_dir) / run).is_dir()}
    if pinned and len(pinned) < len(runs):
        missing = ", ".join(sorted(set(runs) - set(pinned)))
        print(f"[kinetics] reported beta runs missing for {missing}; those panels are dropped")
    curated = {} if pinned else curated_runs(cache_dir)
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
        if pinned:
            if pinned.get(dataset) != run:
                continue
        elif curated and run not in curated:
            continue
        seed = parts[3].replace("seed_", "") if len(parts) > 4 else "-"
        for name, t, f in zip(names, targets, fitted):
            rows.append({"run": run, "dataset": dataset, "seed": seed,
                         "marker": clean_marker(name),
                         "target": float(t), "fitted": float(f)})
    return pd.DataFrame(rows)


def beta_limits(df: pd.DataFrame, datasets) -> tuple[float, float]:
    """One square range covering target AND fitted, shared by panels a and b.

    Square and log on both axes so the identity line is a true 45-degree diagonal:
    with a linear y-axis it stood almost vertical and read as an arbitrary guide
    rather than as "perfect recovery". Panels a and b share it so they can be read
    against each other, and taking it from the data means a point can never land
    outside the frame while the n in the corner still counts it.
    """
    values = pd.concat([df.loc[df["dataset"] == ds, col]
                        for ds in datasets for col in ("target", "fitted")])
    lo, hi = float(values.min()), float(values.max())
    return lo * 0.7, hi * 1.4


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
    lo, hi = ylim
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
                textcoords="offset points", fontsize=6, color=ink(color),
                ha="left" if side == "low" else "right", va="bottom",
                arrowprops=dict(arrowstyle="-", lw=0.4, color="0.65",
                                shrinkA=0, shrinkB=1.5))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    # Explicit decades strictly inside the limits: a tick sitting on the panel
    # edge collides with the neighbouring panel's first tick (§3.4).
    decades = [10.0 ** k for k in range(-4, 4)]
    ticks = ([t for t in decades if lo * 1.6 < t < hi / 1.6]
             or [t for t in decades if lo < t < hi])
    ax.set_xticks(ticks)
    ax.set_xticklabels(["%g" % t for t in ticks])
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=(), numticks=1))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks(ticks)
    ax.set_yticklabels(["%g" % t for t in ticks])
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(), numticks=1))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"Anchor target $\beta$")
    if show_ylabel:
        ax.set_ylabel(r"Predicted $\beta$")
    ax.set_title(title)
    # Bottom right stays: it is the corner the identity line and the fitted points
    # both leave empty. The top-left move put this box straight on the dashed line.
    ax.text(0.97, 0.06, f"$\\rho$ = {rho:.2f}\nn = {len(g)}", transform=ax.transAxes,
            fontsize=6, color="0.25", ha="right", va="bottom", linespacing=1.4)
    return rho, g


def plot_beta_recovery(df: pd.DataFrame, save_path: str | Path,
                       *, n_label: int = 1, datasets=("bmmc_cite_retained", "pbmc_retained"),
                       backend_pairs=(("bmmc_cite_retained", "bmmc_cite_regvelo"),
                                      ("pbmc_retained", "pbmc_regvelo"))):
    """Anchor agreement on two datasets, backend sensitivity, and rate compression."""
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=figsize("full", 4.6), layout="constrained")
    axes = axes.ravel()
    fig.get_layout_engine().set(w_pad=0.08, wspace=0.07)

    ylim = beta_limits(df, datasets)
    labels = {"bmmc_cite_retained": "BMMC", "pbmc_retained": "PBMC"}
    notes = []
    for i, ds in enumerate(datasets):
        sub = df[df["dataset"] == ds]
        # "Rank agreement", not the dataset name: rho is a rank statistic, and panel d
        # exists precisely because the magnitudes are not recovered.
        rho, g = recovery_panel(axes[i], sub,
                                title=f"Rank agreement, {labels.get(ds, dataset_label(ds))}",
                                ylim=ylim, n_label=n_label, show_ylabel=(i == 0))
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
    for scvelo_ds, regvelo_ds in backend_pairs:
        pts, seed_rhos = [], []
        for x, ds in [(0, scvelo_ds), (1, regvelo_ds)]:
            sub = df[df["dataset"] == ds]
            if sub.empty:
                pts = []
                break
            # Correlate within each seed, then summarise. Collapsing the seed dimension
            # first (one median profile, one rho) hid the spread entirely and drew a
            # crossing steeper than the seeds support -- PBMC fell 0.122 that way
            # against 0.053 across seeds.
            rhos = [spearman(g["target"], g["fitted"])
                    for _, seed_rows in sub.groupby("seed")
                    for g in [seed_rows.groupby("marker")[["target", "fitted"]].median()]]
            rhos = np.asarray([r for r in rhos if np.isfinite(r)], dtype=float)
            if not rhos.size:
                pts = []
                break
            pts.append((x, float(rhos.mean()),
                        float(rhos.std(ddof=1)) if rhos.size > 1 else 0.0, rhos.size))
            seed_rhos.append(rhos)
        if len(pts) != 2:
            continue
        xs, ys, _, ns = zip(*pts)
        # One label covers both backends, so it may not quote a single backend's seed
        # count: the scVelo and RegVelo arms come from different launches and a launch
        # that lost seeds would otherwise be reported with its sibling's n.
        seed_n = ns[0] if len(set(ns)) == 1 else "/".join(str(n) for n in ns)
        marker, color, short = dataset_style(scvelo_ds)
        # Every seed as its own point, and nothing else. A mean marker or an error bar
        # reads as a summary the panel has not earned with five runs; the raw spread
        # says the same thing and shows the two backends' seeds interleaving.
        for i, (x, rhos) in enumerate(zip(xs, seed_rhos)):
            offsets = np.linspace(-0.06, 0.06, len(rhos)) if len(rhos) > 1 else [0.0]
            # Label once per dataset, not once per backend, or the key lists each twice.
            ax.plot(x + np.asarray(offsets), rhos, marker, color=color, ms=2.6,
                    markerfacecolor=color, markeredgewidth=0, alpha=0.8,
                    linestyle="none", zorder=5,
                    label=f"{short} (n={seed_n})" if i == 0 else None)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["scVelo", "RegVelo"])
    ax.tick_params(axis="x", pad=2)
    ax.set_xlim(-0.25, 1.75)
    ax.legend(loc="upper right", frameon=False, handletextpad=0.4,
              labelspacing=0.25, fontsize=STYLE_STATE["ladder"][2])
    ax.set_xlabel("Velocity backend")
    ax.set_ylabel(r"Spearman $\rho$, predicted vs target $\beta$")
    ax.set_title("Backend sensitivity")
    panel_letter(ax, "c")

    # d: the seed spread panels a-b cannot show. BMMC carries the larger marker panel,
    # so it is the one worth resolving marker by marker.
    ax = axes[3]
    tables = beta_spread_panel(ax, df, list(datasets))
    ax.set_ylabel(r"$\beta$")
    ax.set_title("Ranks recovered, magnitudes not")
    # The grey target series is registered while the first dataset is drawn, so it
    # lands between the two predicted entries; push it to the end.
    handles, legend_labels = ax.get_legend_handles_labels()
    order = ([i for i, t in enumerate(legend_labels) if t != "anchor target"]
             + [i for i, t in enumerate(legend_labels) if t == "anchor target"])
    ax.legend([handles[i] for i in order], [legend_labels[i] for i in order],
              loc="lower right", frameon=False)
    panel_letter(ax, "d")
    for ds, table in tables.items():
        notes.append(f"{dataset_label(ds)}: {len(table)} proteins with "
                     f"{int(table['seeds'].median())} run/seed observations each")

    save_figure(fig, Path(save_path).with_suffix(""))
    return notes


def beta_spread_panel(ax, df: pd.DataFrame, datasets, *, n_label: int = 0):
    """Every protein's predicted beta and its across-seed range, against the target.

    The recovery scatter in panels a-b shows one point per protein and so cannot say
    whether a protein sits where it does reliably or only on average. Here each protein
    keeps its seed range, ordered by the literature target, and the target itself is
    drawn as a separate series -- the vertical gap between the two is the error the
    rank correlation summarises into a single number.

    Both datasets share the panel. They have very different protein counts (BMMC 52,
    PBMC 10), so x is the rank as a FRACTION of each panel rather than an index; the
    two then span the same width and their compression can be read against each other.
    Dataset is carried by marker shape, matching panel c.

    No protein names by default. The claim is the COMPRESSION between the two series --
    literature beta spans two orders of magnitude where predicted beta spans a factor
    of two -- and panels a-b already name the proteins worth naming.
    """
    tables = {}
    for ds in datasets:
        grouped = df[df["dataset"] == ds].groupby("marker")
        table = grouped.agg(target=("target", "median"), fitted=("fitted", "median"),
                            lo=("fitted", "min"), hi=("fitted", "max"),
                            seeds=("fitted", "size"))
        table = table.sort_values("target").reset_index()
        if table.empty:
            continue
        tables[ds] = table
        x = (np.arange(len(table)) / max(len(table) - 1, 1)) if len(table) > 1 else np.array([0.5])
        shape, color, short = dataset_style(ds)
        ax.vlines(x, table["lo"], table["hi"], color=color, lw=0.7,
                  alpha=0.45, zorder=2)
        ax.scatter(x, table["fitted"], s=7, marker=shape, color=color,
                   linewidths=0.5, edgecolors="white", zorder=3,
                   label=f"predicted, {short}")
        ax.scatter(x, table["target"], s=6, marker="_", color="#767676", linewidths=0.9,
                   zorder=3, label="anchor target" if ds == datasets[0] else None)
        ends = [(0, (2, -8), "left"), (len(table) - 1, (-2, 6), "right")][:n_label]
        for i, offset, ha in ends:
            ax.annotate(clean_marker(table["marker"][i]), (x[i], table["target"][i]),
                        textcoords="offset points", xytext=offset, ha=ha,
                        fontsize=STYLE_STATE["ladder"][2], color="0.35")
    ax.set_xlim(-0.04, 1.04)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xticklabels(["low", "", "high"])
    ax.set_xlabel(r"Protein, ordered by target $\beta$")
    ax.set_yscale("log")
    return tables
