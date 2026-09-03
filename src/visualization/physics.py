"""Per-cell physics: not how well the ODE holds, but WHERE it holds.

Every KOT run writes `per_cell_diagnostics.npz` — seventeen arrays in the same row order
as `aligned_rna.npy` — and the paper so far reports one number out of it, the median
JVP-RHS cosine. The median cannot say whether a run fits most cells well and a few
lineages not at all, or fits every cell mediocrely; those are different results with the
same summary, and only one of them is a problem worth reporting.

Painting the per-cell cosine onto the co-embedding is the same device MaxFuse uses for
its spatial panels, with embedding coordinates in place of tissue coordinates and a
physics residual in place of a protein stain.

Two plotting rules the panels here depend on, both consequences of the data rather than
taste:

  Robust limits.  The cosine has a long low tail. Scaling the colour map to the full
                  range spends most of it on a handful of cells and leaves the bulk
                  indistinguishable, so limits come from percentiles.
  Draw order.     Low-cosine cells are the interesting ones and the rare ones. Plotted
                  in array order they are buried under the bulk, which reads as a
                  cleaner fit than the run achieved.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import stats

PER_CELL_FILE = "per_cell_diagnostics.npz"

# Sequential, perceptually uniform and CVD-safe. Deliberately outside the state,
# modality and method palettes: a continuous diagnostic is a fourth kind of thing, and
# reusing a categorical hue for it would tie two unrelated meanings to one colour.
PHYSICS_CMAP = "viridis"


def load_per_cell(run: Path) -> dict[str, np.ndarray]:
    """The per-cell diagnostic arrays a run wrote, as a plain dict."""
    with np.load(run / PER_CELL_FILE) as z:
        return {k: z[k] for k in z.files}


def robust_limits(values: np.ndarray, lo: float = 2.0, hi: float = 98.0):
    """Colour limits that survive a long tail."""
    return float(np.percentile(values, lo)), float(np.percentile(values, hi))


def paint_panel(ax, xy: np.ndarray, values: np.ndarray, *, emphasize: str = "low",
                cmap: str = PHYSICS_CMAP, size: float = 1.2):
    """Embedding coloured by a per-cell scalar, rare end drawn on top.

    Returns the mappable so the caller places one colourbar per figure rather than one
    per panel.
    """
    lo, hi = robust_limits(values)
    order = np.argsort(values)
    if emphasize == "low":
        order = order[::-1]
    return ax.scatter(xy[order, 0], xy[order, 1], c=values[order], cmap=cmap,
                      vmin=lo, vmax=hi, s=size, linewidths=0, rasterized=True)


def group_violin_panel(ax, values: np.ndarray, groups, order: list[str],
                       colors: dict[str, str]):
    """One violin per group, medians ticked, groups with too few cells dropped.

    Violins rather than the Fig. 3b ridges: eight distributions have to sit beside two
    other panels here, and a ridge stack needs the full panel height a ridge figure
    gives it.
    """
    present = [g for g in order if (np.asarray(groups) == g).sum() >= 20]
    data = [values[np.asarray(groups) == g] for g in present]
    parts = ax.violinplot(data, positions=range(len(present)), widths=0.8,
                          showextrema=False, showmedians=True)
    for body, group in zip(parts["bodies"], present):
        body.set_facecolor(colors[group])
        body.set_alpha(0.75)
        body.set_linewidth(0)
    parts["cmedians"].set_color("#1A1A1A")
    parts["cmedians"].set_linewidth(0.9)
    ax.set_xticks(range(len(present)))
    # Upright, not 45 degrees: eight lineage names at 45 collide in a third-width
    # panel, and the geometric check catches it every time.
    ax.set_xticklabels(present, rotation=90)
    return present


def pair_panel(ax, x: np.ndarray, y: np.ndarray, *, gridsize: int = 40,
               cmap: str = "Greys"):
    """Two per-cell arrays against each other, as density, with their rank correlation.

    A scatter of 90k cells is a solid block; the question the panel asks — whether the
    cells the ODE fits are the cells that pair correctly — is about where the mass sits,
    which is what a hexbin shows and a scatter hides.
    """
    # Log counts: the joint density spans several orders of magnitude, and on a linear
    # count scale every bin outside the mode renders as white.
    ax.hexbin(x, y, gridsize=gridsize, cmap=cmap, bins="log", mincnt=1, linewidths=0,
              rasterized=True)
    rho = stats.spearmanr(x, y).statistic
    ax.text(0.97, 0.95, f"ρ = {rho:.2f}", transform=ax.transAxes, ha="right", va="top")
    return rho
