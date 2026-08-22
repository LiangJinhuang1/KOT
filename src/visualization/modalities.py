"""Input-modality overview figure for the synthetic datasets."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from src.visualization import state_color_map, state_label
from src.visualization.style import (
    apply_style, embedding_axes, figsize, panel_letter, save_figure,
)

_PT = dict(s=1.6, alpha=0.55, linewidths=0, rasterized=True)


def get_2d_embedding(adata):
    if "X_umap" in adata.obsm:
        return adata.obsm["X_umap"], "UMAP"
    if "X_pca" in adata.obsm:
        return adata.obsm["X_pca"][:, :2], "PC"
    xy = PCA(n_components=2).fit_transform(np.array(adata.X))
    return xy, "PC"


def plot_modalities(rna_adata, protein_adata, save_dir, *, second_label="Protein"):
    """Both modalities carry the same branching structure and the same time ordering.

    Row 1 establishes the state structure, row 2 the temporal ordering; the two
    columns are the two modalities. Panels within a row share a colour system.
    """
    apply_style()
    save_dir = Path(save_dir)

    fig, axes = plt.subplots(2, 2, figsize=figsize("full", 4.4))
    letters = [["a", "b"], ["c", "d"]]
    scatter_for_bar = None

    for col, (adata, label) in enumerate([(rna_adata, "RNA"), (protein_adata, second_label)]):
        xy, coord = get_2d_embedding(adata)
        states = adata.obs["state"].values
        times = adata.obs["time"].values

        ax = axes[0, col]
        colors = state_color_map(states)
        for state, color in colors.items():
            mask = states == state
            ax.scatter(xy[mask, 0], xy[mask, 1], c=color, label=state_label(state), **_PT)
        ax.set_title(f"{label} — cell state")
        if col == 0:
            ax.legend(markerscale=4, loc="best")
        embedding_axes(ax, f"{coord} 1", f"{coord} 2")
        panel_letter(ax, letters[0][col])

        ax = axes[1, col]
        scatter_for_bar = ax.scatter(xy[:, 0], xy[:, 1], c=times, cmap="viridis", **_PT)
        ax.set_title(f"{label} — pseudotime")
        embedding_axes(ax, f"{coord} 1", f"{coord} 2")
        panel_letter(ax, letters[1][col])

    # One shared colourbar for the row rather than one per panel (§3.4).
    if scatter_for_bar is not None:
        cb = fig.colorbar(scatter_for_bar, ax=axes[1, :].tolist(),
                          fraction=0.025, pad=0.015)
        cb.set_label("Pseudotime")
        cb.outline.set_visible(False)

    n = len(rna_adata)
    fig.text(0.5, -0.005, f"Synthetic linked-ODE data · n = {n:,} cells per modality",
             ha="center", va="top", fontsize=6, color="0.35")

    return save_figure(fig, save_dir / "input_modalities")
