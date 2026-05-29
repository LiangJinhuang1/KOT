import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA

from scr.visualization import STATE_COLORS


def get_2d_embedding(adata):
    if "X_umap" in adata.obsm:
        return adata.obsm["X_umap"], "UMAP"
    if "X_pca" in adata.obsm:
        return adata.obsm["X_pca"][:, :2], "PC"
    xy = PCA(n_components=2).fit_transform(np.array(adata.X))
    return xy, "PC"


def plot_modalities(rna_adata, protein_adata, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(8, 7))

    for col, (adata, label) in enumerate([(rna_adata, "RNA"), (protein_adata, "Protein")]):
        xy, coord_label = get_2d_embedding(adata)
        states = adata.obs["state"].values
        times = adata.obs["time"].values

        ax = axes[0, col]
        for state, color in STATE_COLORS.items():
            mask = states == state
            ax.scatter(xy[mask, 0], xy[mask, 1], c=color, s=3, alpha=0.7,
                       label=state, rasterized=True)
        ax.set_title(f"{label} — Cell State")
        ax.set_xlabel(f"{coord_label} 1")
        ax.set_ylabel(f"{coord_label} 2")
        if col == 0:
            ax.legend(markerscale=2, frameon=False, fontsize=8)

        ax = axes[1, col]
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=times, cmap="viridis",
                        s=3, alpha=0.7, rasterized=True)
        plt.colorbar(sc, ax=ax, label="Pseudotime")
        ax.set_title(f"{label} — Pseudotime")
        ax.set_xlabel(f"{coord_label} 1")
        ax.set_ylabel(f"{coord_label} 2")

    fig.suptitle("Synthetic Data: Input Modalities", fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = save_dir / "input_modalities.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
