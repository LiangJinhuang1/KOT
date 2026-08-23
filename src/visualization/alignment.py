"""Per-run alignment figures, built to ML-conference publication standard.

Every figure here goes through :func:`src.visualization.style.save_figure`, so
each one lands as an embeddable vector PDF alongside a 300-dpi PNG.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from sklearn.neighbors import NearestNeighbors

from src.visualization import (
    FOSCTTM_CHANCE,
    MODALITY_COLORS,
    MODALITY_MARKERS,
    state_color_map,
    state_label,
)
from src.visualization.style import (
    SCATTER_STYLE,
    apply_style,
    chance_line,
    embedding_axes,
    figsize,
    ink,
    panel_letter,
    save_figure,
)
from src.visualization.reduce import (
    after_alignment_xy,
    axis_labels,
    before_alignment_xy,
    reduction_title,
)


def prefer_time_coloring(obs) -> bool:
    return "state" in obs.columns and "time" in obs.columns and obs["state"].nunique() == 1


def scatter_state_panel(ax, xy, obs, *, show_legend: bool = False, cbar: bool = False):
    """Colour an embedding by cell state, or by time when there is only one state."""
    if prefer_time_coloring(obs):
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=obs["time"].values,
                        cmap="viridis", **SCATTER_STYLE)
        if cbar:
            cb = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
            cb.set_label("Pseudotime")
            cb.outline.set_visible(False)
        return None

    states = obs["state"].values
    colors = state_color_map(states)
    for state, color in colors.items():
        mask = states == state
        ax.scatter(xy[mask, 0], xy[mask, 1], c=color, label=state_label(state), **SCATTER_STYLE)
    if show_legend and colors:
        ax.legend(markerscale=4, loc="best")
    return colors


def scatter_dual_modality_panel(ax, xy_rna, xy_prot, obs, *, second_label="Protein"):
    """Co-embedding coloured by state, with modality carried by marker shape."""
    if prefer_time_coloring(obs):
        time = obs["time"].values
        ax.scatter(xy_rna[:, 0], xy_rna[:, 1], c=time, cmap="viridis",
                   marker=MODALITY_MARKERS["RNA"], **SCATTER_STYLE)
        ax.scatter(xy_prot[:, 0], xy_prot[:, 1], c=time, cmap="viridis",
                   marker=MODALITY_MARKERS.get(second_label, "^"), **SCATTER_STYLE)
        return

    states = obs["state"].values
    colors = state_color_map(states)
    for state, color in colors.items():
        mask = states == state
        ax.scatter(xy_rna[mask, 0], xy_rna[mask, 1], c=color,
                   marker=MODALITY_MARKERS["RNA"], **SCATTER_STYLE)
        ax.scatter(xy_prot[mask, 0], xy_prot[mask, 1], c=color,
                   marker=MODALITY_MARKERS.get(second_label, "^"), **SCATTER_STYLE)

    # Two palettes, two legends (§4.6): state carries hue, modality carries shape.
    state_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=c,
               markersize=3.5, label=state_label(s)) for s, c in colors.items()
    ]
    shape_handles = [
        Line2D([0], [0], marker=MODALITY_MARKERS["RNA"], color="none",
               markerfacecolor="0.45", markersize=3.5, label="RNA"),
        Line2D([0], [0], marker=MODALITY_MARKERS.get(second_label, "^"), color="none",
               markerfacecolor="0.45", markersize=3.5, label=second_label),
    ]
    first = ax.legend(handles=state_handles, loc="upper left", title=None)
    ax.add_artist(first)
    ax.legend(handles=shape_handles, loc="lower right")


def coembedding_plot_kwargs(cfg: dict) -> dict:
    reduction = str(cfg.get("viz_reduction", "pca")).lower()
    return {
        "reduction": reduction,
        "rna_obsm_key": cfg.get("viz_before_rna_obsm"),
        "second_obsm_key": cfg.get("viz_before_second_obsm"),
        "random_state": int(cfg.get("viz_random_state", cfg.get("seed", 42))),
        "umap_neighbors": int(cfg.get("viz_umap_neighbors", 15)),
        "umap_min_dist": float(cfg.get("viz_umap_min_dist", 0.1)),
        "tsne_perplexity": float(cfg.get("viz_tsne_perplexity", 30)),
    }


def modality_mixing(xy_rna: np.ndarray, xy_prot: np.ndarray, *, k: int = 15,
                    max_cells: int = 4000, random_state: int = 0) -> float | None:
    """Fraction of each cell's k nearest co-embedding neighbours from the other modality.

    1.0 = perfectly mixed, 0.0 = the two modalities occupy disjoint regions. This
    is the number the "after alignment, by modality" panel is really asking about,
    so it is reported on the panel rather than left to the reader's eye.
    """
    rng = np.random.default_rng(random_state)
    n_r, n_p = len(xy_rna), len(xy_prot)
    if n_r == 0 or n_p == 0:
        return None
    idx_r = rng.choice(n_r, min(n_r, max_cells // 2), replace=False)
    idx_p = rng.choice(n_p, min(n_p, max_cells // 2), replace=False)

    pts = np.vstack([xy_rna[idx_r], xy_prot[idx_p]])
    is_prot = np.concatenate([np.zeros(len(idx_r), bool), np.ones(len(idx_p), bool)])
    k_eff = int(min(k, len(pts) - 1))
    if k_eff < 1:
        return None

    nn = NearestNeighbors(n_neighbors=k_eff + 1).fit(pts)
    neigh = nn.kneighbors(pts, return_distance=False)[:, 1:]
    cross = (is_prot[neigh] != is_prot[:, None]).mean(axis=1)
    # Expected cross-fraction under perfect mixing is the other class's share,
    # so normalise by it to keep 1.0 meaning "as mixed as possible".
    share = np.where(is_prot, len(idx_r), len(idx_p)) / len(pts)
    return float(np.mean(cross / share))


def plot_coembedding(
    x,
    y,
    aligned_rna,
    aligned_protein,
    obs,
    save_dir,
    model_name,
    *,
    reduction: str = "pca",
    rna_adata=None,
    second_adata=None,
    rna_obsm_key: str | None = None,
    second_obsm_key: str | None = None,
    random_state: int = 42,
    umap_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    tsne_perplexity: float = 30.0,
    second_label: str = "Protein",
):
    """Four panels, two claims: the modalities start apart (row 1) and end
    together while keeping their biology (row 2)."""
    apply_style()
    save_dir = Path(save_dir)
    reduce_kwargs = dict(
        random_state=random_state,
        umap_neighbors=umap_neighbors,
        umap_min_dist=umap_min_dist,
        tsne_perplexity=tsne_perplexity,
    )

    xy_rna_before, xy_prot_before, before_label = before_alignment_xy(
        x, y, reduction=reduction, rna_adata=rna_adata, second_adata=second_adata,
        rna_obsm_key=rna_obsm_key, second_obsm_key=second_obsm_key, **reduce_kwargs,
    )
    xy_rna_after, xy_prot_after = after_alignment_xy(
        aligned_rna, aligned_protein, reduction=reduction, **reduce_kwargs,
    )

    before_x, before_y = axis_labels(
        before_label if (rna_obsm_key or second_obsm_key) else reduction
    )
    after_x, after_y = axis_labels("pca" if reduction == "precomputed" else reduction)

    fig, axes = plt.subplots(2, 2, figsize=figsize("full", 4.6))

    # --- Row 1: before alignment. Separate coordinate systems (§1.2) ---------
    for col, (xy, label) in enumerate([(xy_rna_before, "RNA"), (xy_prot_before, second_label)]):
        ax = axes[0, col]
        scatter_state_panel(ax, xy, obs, show_legend=(col == 0))
        ax.set_title(f"{label} before alignment")
        embedding_axes(ax, before_x, before_y)
        panel_letter(ax, "ab"[col])

    # --- Row 2: after alignment. One shared coordinate system ----------------
    ax = axes[1, 0]
    ax.scatter(xy_rna_after[:, 0], xy_rna_after[:, 1], c=MODALITY_COLORS["RNA"],
               marker=MODALITY_MARKERS["RNA"], label="RNA", **SCATTER_STYLE)
    ax.scatter(xy_prot_after[:, 0], xy_prot_after[:, 1],
               c=MODALITY_COLORS.get(second_label, MODALITY_COLORS["Protein"]),
               marker=MODALITY_MARKERS.get(second_label, "^"), label=second_label, **SCATTER_STYLE)
    mixing = modality_mixing(xy_rna_after, xy_prot_after, random_state=random_state)
    title = "Co-embedding, by modality"
    if mixing is not None:
        title += f"  (mixing {mixing:.2f})"
    ax.set_title(title)
    ax.legend(markerscale=4, loc="best")
    embedding_axes(ax, after_x, after_y)
    panel_letter(ax, "c")

    ax = axes[1, 1]
    scatter_dual_modality_panel(ax, xy_rna_after, xy_prot_after, obs,
                                second_label=second_label)
    ax.set_title("Co-embedding, by cell state")
    embedding_axes(ax, after_x, after_y)
    panel_letter(ax, "d")

    method = reduction_title(before_label if (rna_obsm_key or second_obsm_key) else reduction)
    n_cells = len(xy_rna_after)
    fig.text(0.5, -0.005,
             f"{method} projection · n = {n_cells:,} cells · a and b are separate "
             f"coordinate systems and are not directly comparable",
             ha="center", va="top", fontsize=6, color="0.35")

    fig.tight_layout(pad=0.4, h_pad=1.2, w_pad=1.0)
    return save_figure(fig, save_dir / "coembedding")


def plot_coupling(coupling, time, save_dir, model_name):
    """Transport plan sorted by pseudotime; a diagonal band means the plan respects time."""
    apply_style()
    save_dir = Path(save_dir)
    order = np.argsort(time)
    C = coupling[np.ix_(order, order)]

    fig, ax = plt.subplots(figsize=figsize("half", 2.3))
    im = ax.imshow(C, aspect="auto", cmap="Blues", interpolation="none",
                   rasterized=True)
    cb = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("Transport mass")
    cb.outline.set_visible(False)
    ax.set_title("Transport plan, cells sorted by pseudotime")
    ax.set_xlabel("Protein cells (early → late)")
    ax.set_ylabel("RNA cells (early → late)")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout(pad=0.3)
    return save_figure(fig, save_dir / "coupling")


def plot_foscttm_sorted(foscttm_csv, save_dir, model_name):
    """Empirical CDF of per-cell FOSCTTM against the chance reference.

    Plotted as an ECDF rather than a sorted-rank curve so the x-axis carries the
    metric itself and the 0.5 chance line is readable on it.
    """
    apply_style()
    save_dir = Path(save_dir)
    scores = pd.read_csv(foscttm_csv)["foscttm"].to_numpy()
    scores = scores[np.isfinite(scores)]
    mean_score = float(np.mean(scores))
    below_chance = float(np.mean(scores < FOSCTTM_CHANCE))

    xs = np.sort(scores)
    ys = np.arange(1, len(xs) + 1) / len(xs)

    fig, ax = plt.subplots(figsize=figsize("half", 2.0))
    ax.plot(xs, ys, color=MODALITY_COLORS["RNA"], lw=1.3, solid_capstyle="round")
    chance_line(ax, FOSCTTM_CHANCE, axis="x", label="chance")
    ax.axvline(mean_score, color="0.15", lw=0.7, ls=":")
    # Label sits left of its own line: to the right it would run straight through
    # the chance reference and read as annotating that instead.
    ax.annotate(f"mean {mean_score:.3f}", xy=(mean_score, 0.06),
                xytext=(-4, 0), textcoords="offset points",
                fontsize=6, color="0.15", va="center", ha="right")

    ax.set_xlim(0, max(0.55, float(xs.max()) * 1.02))
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("FOSCTTM (lower = better)")
    ax.set_ylabel("Fraction of cells")
    ax.set_title(f"{below_chance:.0%} of cells align better than chance")
    ax.margins(x=0.02)

    fig.tight_layout(pad=0.3)
    return save_figure(fig, save_dir / "foscttm_sorted")


def density(values: np.ndarray, grid: np.ndarray):
    """Gaussian KDE of ``values`` over ``grid``, or None if the input is degenerate.

    A KDE needs spread to estimate: fewer than three points, or a constant
    vector, has none, so those return None and the caller skips the ridge.
    """
    values = values[np.isfinite(values)]
    if len(values) < 3 or np.allclose(values, values[0]):
        return None
    return gaussian_kde(values)(grid)


def plot_foscttm_distribution(foscttm_csv, obs, save_dir, model_name):
    """Per-cell FOSCTTM as a ridgeline by cell state.

    Overlaid translucent histograms went muddy wherever two states shared support
    (§6.4); offsetting them vertically keeps every distribution readable and makes
    the per-state ordering the visible message.
    """
    apply_style()
    save_dir = Path(save_dir)
    scores = pd.read_csv(foscttm_csv)["foscttm"].to_numpy()
    scores = scores[np.isfinite(scores)]
    mean_score = float(np.mean(scores))

    has_states = obs is not None and "state" in obs.columns and obs["state"].nunique() > 1
    grid = np.linspace(0, max(0.55, float(scores.max()) * 1.02), 256)

    if has_states:
        states = np.asarray(obs["state"].values)[: len(scores)]
        colors = state_color_map(states)
        groups = [(s, scores[states == s], c) for s, c in colors.items()
                  if (states == s).sum() >= 3]
    else:
        groups = [("All cells", scores, MODALITY_COLORS["RNA"])]

    height = 0.62 + 0.42 * len(groups)
    fig, ax = plt.subplots(figsize=figsize("half", height))

    step = 1.0
    peak = 0.0
    for i, (_, vals, _) in enumerate(groups):
        d = density(vals, grid)
        if d is not None:
            peak = max(peak, float(d.max()))
    scale = (step * 0.9) / peak if peak > 0 else 1.0

    for i, (name, vals, color) in enumerate(reversed(groups)):
        base = i * step
        d = density(vals, grid)
        if d is None:
            continue
        ax.fill_between(grid, base, base + d * scale, color=color, alpha=0.75,
                        linewidth=0, zorder=3 + i)
        ax.plot(grid, base + d * scale, color=ink(color), lw=0.8, zorder=3 + i)
        ax.axhline(base, color="0.85", lw=0.5, zorder=2)
        ax.text(-0.012, base + step * 0.18, f"{state_label(name)}  (n={len(vals):,})",
                transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=6, color=ink(color))

    chance_line(ax, FOSCTTM_CHANCE, axis="x", label="chance")
    ax.axvline(mean_score, color="0.15", lw=0.7, ls=":", zorder=20)
    # Every distinct mark has to be identifiable from the figure alone (§2.1).
    ax.text(mean_score, 1.0, "mean ", transform=ax.get_xaxis_transform(),
            fontsize=6, color="0.15", ha="right", va="top", zorder=21)

    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(grid[0], grid[-1])
    ax.set_ylim(-0.08 * step, len(groups) * step + 0.05)
    ax.set_xlabel("FOSCTTM (lower = better)")
    ax.set_title("Alignment quality by cell state")

    fig.tight_layout(pad=0.3)
    return save_figure(fig, save_dir / "foscttm_distribution")


def settling_epoch(epochs: np.ndarray, values: np.ndarray,
                   tol: float = 0.5) -> float | None:
    """First epoch after which the loss stays within ``tol`` (relative) of its final value.

    Computed rather than asserted: a hard-coded "converges by epoch 40" in the
    title is a claim about the data, and on a log axis a slow tail is visible
    enough that a wrong number is obvious to a reviewer (§1.4).
    """
    values = np.asarray(values, dtype=float)
    if len(values) < 3:
        return None
    final = float(values[-1])
    if not np.isfinite(final) or final <= 0:
        return None
    outside = np.abs(values - final) > tol * final
    if not outside.any():
        return float(epochs[0])
    last_bad = int(np.max(np.flatnonzero(outside)))
    if last_bad + 1 >= len(epochs):
        return None
    return float(epochs[last_bad + 1])


def plot_training_loss(loss_csv, save_dir, model_name):
    """Both objectives on one log-scaled panel, direct-labelled.

    Log y because the losses fall by two orders of magnitude in the first ~40
    epochs and then flatten; on a linear axis most of the range and most of the
    epochs carry no visible information (§3.2).
    """
    apply_style()
    save_dir = Path(save_dir)
    df = pd.read_csv(loss_csv)

    series = [
        ("Sinkhorn alignment", "align", MODALITY_COLORS["RNA"]),
        ("Kinetics ODE", "dyn", MODALITY_COLORS["Protein"]),
    ]
    series = [(lbl, col, c) for lbl, col, c in series if col in df.columns]

    fig, ax = plt.subplots(figsize=figsize("half", 2.0))
    x_max = float(df["epoch"].max())
    settle_epochs = []
    for label, col, color in series:
        vals = df[col].to_numpy(dtype=float)
        positive = vals[vals > 0]
        floor = float(positive.min()) * 0.5 if len(positive) else 1e-12
        safe = np.clip(vals, floor, None)
        ax.plot(df["epoch"], safe, color=ink(color), lw=1.1)
        # Direct end-of-line labels in preference to a legend box (§6.3).
        ax.annotate(label, xy=(x_max, safe[-1]),
                    xytext=(3, 0), textcoords="offset points",
                    fontsize=6, color=ink(color), va="center", ha="left")
        settle_epochs.append(settling_epoch(df["epoch"].to_numpy(), safe))

    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    settled = [e for e in settle_epochs if e is not None]
    if settled:
        ax.set_title(f"Both objectives settle by epoch {max(settled):g}")
    else:
        ax.set_title("Training objectives")
    ax.set_xlim(left=0)
    ax.margins(x=0.22)

    fig.tight_layout(pad=0.3)
    return save_figure(fig, save_dir / "training_loss")
