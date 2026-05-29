import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from pathlib import Path

from scr.visualization import STATE_COLORS, MODALITY_COLORS


def plot_coembedding(x, y, aligned_rna, aligned_protein, obs, save_dir, model_name):
    """
    6-panel alignment summary figure:
      Row 1 — Before alignment: RNA PCA | Protein PCA  (colored by state)
      Row 2 — After alignment:  RNA PCA | Protein PCA  (colored by state, joint PCA)
      Row 3 — Co-embedding:     by modality            | by state (both modalities overlaid)
    """
    save_dir = Path(save_dir)
    n = len(aligned_rna)
    states = obs["state"].values

    # Before: separate PCAs (different feature spaces)
    xy_rna_before = PCA(n_components=2).fit_transform(x)
    xy_prot_before = PCA(n_components=2).fit_transform(y)

    # After: joint PCA on concatenated aligned embeddings (same space)
    xy_joint = PCA(n_components=2).fit_transform(np.vstack([aligned_rna, aligned_protein]))
    xy_rna_after = xy_joint[:n]
    xy_prot_after = xy_joint[n:]

    fig, axes = plt.subplots(3, 2, figsize=(9, 12))

    # --- Row 1: before alignment ---
    for col, (xy, label) in enumerate([(xy_rna_before, "RNA"), (xy_prot_before, "Protein")]):
        ax = axes[0, col]
        for state, color in STATE_COLORS.items():
            mask = states == state
            ax.scatter(xy[mask, 0], xy[mask, 1], c=color, s=3, alpha=0.7,
                       label=state, rasterized=True)
        ax.set_title(f"{label} — Before Alignment")
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        if col == 0:
            ax.legend(markerscale=2, frameon=False, fontsize=8)

    # --- Row 2: after alignment, SCOT style (separate, colored by state) ---
    for col, (xy, label) in enumerate([(xy_rna_after, "RNA"), (xy_prot_after, "Protein")]):
        ax = axes[1, col]
        for state, color in STATE_COLORS.items():
            mask = states == state
            ax.scatter(xy[mask, 0], xy[mask, 1], c=color, s=3, alpha=0.7,
                       rasterized=True)
        ax.set_title(f"{label} — After Alignment")
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")

    # --- Row 3 left: co-embedding colored by modality ---
    ax = axes[2, 0]
    ax.scatter(xy_rna_after[:, 0], xy_rna_after[:, 1],
               c=MODALITY_COLORS["RNA"], s=3, alpha=0.6, label="RNA", rasterized=True)
    ax.scatter(xy_prot_after[:, 0], xy_prot_after[:, 1],
               c=MODALITY_COLORS["Protein"], s=3, alpha=0.6, label="Protein", rasterized=True)
    ax.set_title("Co-embedding — Modality")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend(markerscale=2, frameon=False, fontsize=8)

    # --- Row 3 right: co-embedding colored by state (circles=RNA, triangles=protein) ---
    ax = axes[2, 1]
    for state, color in STATE_COLORS.items():
        mask = states == state
        ax.scatter(xy_rna_after[mask, 0], xy_rna_after[mask, 1],
                   c=color, s=3, alpha=0.6, marker="o", rasterized=True)
        ax.scatter(xy_prot_after[mask, 0], xy_prot_after[mask, 1],
                   c=color, s=3, alpha=0.6, marker="^", rasterized=True)
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                      markersize=6, label=s) for s, c in STATE_COLORS.items()]
    handles += [
        Line2D([0], [0], marker="o", color="grey", markersize=6, label="○ RNA"),
        Line2D([0], [0], marker="^", color="grey", markersize=6, label="△ Protein"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=7, ncol=2)
    ax.set_title("Co-embedding — Cell State")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")

    fig.suptitle(f"{model_name} — Alignment Summary", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = save_dir / "coembedding.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_coupling(coupling, time, save_dir, model_name):
    save_dir = Path(save_dir)
    order = np.argsort(time)
    C = coupling[np.ix_(order, order)]

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(C, aspect="auto", cmap="Blues", interpolation="none")
    plt.colorbar(im, ax=ax, label="Coupling weight")
    ax.set_title(f"{model_name} — Coupling Matrix\n(sorted by pseudotime)")
    ax.set_xlabel("Protein cells")
    ax.set_ylabel("RNA cells")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    path = save_dir / "coupling.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_foscttm_sorted(foscttm_csv, save_dir, model_name):
    save_dir = Path(save_dir)
    scores = pd.read_csv(foscttm_csv)["foscttm"].values
    mean_score = float(np.mean(scores))

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(np.arange(len(scores)), np.sort(scores), "r--",
            label=f"{model_name} alignment FOSCTTM\naverage value: {mean_score:.4f}")
    ax.set_xlabel("Cells")
    ax.set_ylabel("Sorted FOSCTTM")
    ax.set_title(f"{model_name} — Sorted FOSCTTM")
    ax.legend(frameon=True, fontsize=8)
    plt.tight_layout()
    path = save_dir / "foscttm_sorted.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_foscttm_distribution(foscttm_csv, obs, save_dir, model_name):
    save_dir = Path(save_dir)
    scores = pd.read_csv(foscttm_csv)["foscttm"].values
    mean_score = float(np.mean(scores))

    fig, ax = plt.subplots(figsize=(5, 3.5))
    if obs is not None and "state" in obs.columns:
        for state, color in STATE_COLORS.items():
            mask = (obs["state"] == state).values
            ax.hist(scores[mask], bins=30, alpha=0.6, color=color,
                    label=state, density=True)
        ax.legend(frameon=False, fontsize=8)
    else:
        ax.hist(scores, bins=30, color=MODALITY_COLORS["RNA"], alpha=0.75, density=True)

    ax.axvline(mean_score, color="black", linestyle="--", linewidth=1,
               label=f"Mean = {mean_score:.3f}")
    ax.set_xlabel("FOSCTTM (lower = better)")
    ax.set_ylabel("Density")
    ax.set_title(f"{model_name} — Per-cell FOSCTTM")
    ax.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    path = save_dir / "foscttm_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_training_loss(loss_csv, save_dir, model_name):
    save_dir = Path(save_dir)
    df = pd.read_csv(loss_csv)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    axes[0].plot(df["epoch"], df["align"], label="Sinkhorn align")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{model_name} — Alignment Loss")
    axes[0].legend(frameon=False)

    axes[1].plot(df["epoch"], df["dyn"], color="orange", label="Kinetics ODE")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title(f"{model_name} — Dynamics Loss")
    axes[1].legend(frameon=False)

    plt.tight_layout()
    path = save_dir / "training_loss.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
