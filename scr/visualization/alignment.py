import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from pathlib import Path

from scr.visualization import MODALITY_COLORS, state_color_map
from scr.visualization.reduce import (
    after_alignment_xy,
    axis_labels,
    before_alignment_xy,
    reduction_title,
)


def prefer_time_coloring(obs) -> bool:
    return "state" in obs.columns and "time" in obs.columns and obs["state"].nunique() == 1


def scatter_state_panel(ax, xy, obs, *, show_legend: bool = False):
    if prefer_time_coloring(obs):
        time = obs["time"].values
        sc = ax.scatter(
            xy[:, 0], xy[:, 1], c=time, s=3, alpha=0.75,
            cmap="viridis", rasterized=True,
        )
        if show_legend:
            plt.colorbar(sc, ax=ax, label="time", fraction=0.046, pad=0.04)
        return

    states = obs["state"].values
    colors = state_color_map(states)
    for state, color in colors.items():
        mask = states == state
        ax.scatter(
            xy[mask, 0], xy[mask, 1], c=color, s=3, alpha=0.7,
            label=state, rasterized=True,
        )
    if show_legend and colors:
        ax.legend(markerscale=2, frameon=False, fontsize=8)


def scatter_dual_modality_panel(ax, xy_rna, xy_prot, obs):
    if prefer_time_coloring(obs):
        time = obs["time"].values
        ax.scatter(xy_rna[:, 0], xy_rna[:, 1], c=time, s=3, alpha=0.6,
                   cmap="viridis", marker="o", rasterized=True)
        ax.scatter(xy_prot[:, 0], xy_prot[:, 1], c=time, s=3, alpha=0.6,
                   cmap="viridis", marker="^", rasterized=True)
        handles = [
            Line2D([0], [0], marker="o", color="grey", markersize=6, label="○ RNA"),
            Line2D([0], [0], marker="^", color="grey", markersize=6, label="△ Protein"),
            Line2D([0], [0], marker="s", color="w", markerfacecolor="grey",
                   markersize=6, label="color = time"),
        ]
        ax.legend(handles=handles, frameon=False, fontsize=7)
        return

    states = obs["state"].values
    colors = state_color_map(states)
    for state, color in colors.items():
        mask = states == state
        ax.scatter(xy_rna[mask, 0], xy_rna[mask, 1],
                   c=color, s=3, alpha=0.6, marker="o", rasterized=True)
        ax.scatter(xy_prot[mask, 0], xy_prot[mask, 1],
                   c=color, s=3, alpha=0.6, marker="^", rasterized=True)
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
               markersize=6, label=s) for s, c in colors.items()
    ]
    handles += [
        Line2D([0], [0], marker="o", color="grey", markersize=6, label="○ RNA"),
        Line2D([0], [0], marker="^", color="grey", markersize=6, label="△ Protein"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=7, ncol=2)


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
):
    save_dir = Path(save_dir)
    n = len(aligned_rna)
    reduce_kwargs = dict(
        random_state=random_state,
        umap_neighbors=umap_neighbors,
        umap_min_dist=umap_min_dist,
        tsne_perplexity=tsne_perplexity,
    )

    xy_rna_before, xy_prot_before, before_label = before_alignment_xy(
        x, y,
        reduction=reduction,
        rna_adata=rna_adata,
        second_adata=second_adata,
        rna_obsm_key=rna_obsm_key,
        second_obsm_key=second_obsm_key,
        **reduce_kwargs,
    )
    xy_rna_after, xy_prot_after = after_alignment_xy(
        aligned_rna, aligned_protein,
        reduction=reduction,
        **reduce_kwargs,
    )

    x_label, y_label = axis_labels(
        before_label if rna_obsm_key or second_obsm_key else reduction
    )
    after_x_label, after_y_label = axis_labels(
        "pca" if reduction == "precomputed" else reduction
    )

    fig, axes = plt.subplots(3, 2, figsize=(9, 12))

    # --- Row 1: before alignment ---
    for col, (xy, label) in enumerate([(xy_rna_before, "RNA"), (xy_prot_before, "Protein")]):
        ax = axes[0, col]
        scatter_state_panel(ax, xy, obs, show_legend=(col == 0))
        ax.set_title(f"{label} — Before Alignment")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

    # --- Row 2: after alignment ---
    for col, (xy, label) in enumerate([(xy_rna_after, "RNA"), (xy_prot_after, "Protein")]):
        ax = axes[1, col]
        scatter_state_panel(ax, xy, obs)
        ax.set_title(f"{label} — After Alignment")
        ax.set_xlabel(after_x_label)
        ax.set_ylabel(after_y_label)

    # --- Row 3 left: co-embedding colored by modality ---
    ax = axes[2, 0]
    ax.scatter(xy_rna_after[:, 0], xy_rna_after[:, 1],
               c=MODALITY_COLORS["RNA"], s=3, alpha=0.6, label="RNA", rasterized=True)
    ax.scatter(xy_prot_after[:, 0], xy_prot_after[:, 1],
               c=MODALITY_COLORS["Protein"], s=3, alpha=0.6, label="Protein", rasterized=True)
    ax.set_title("Co-embedding — Modality")
    ax.set_xlabel(after_x_label)
    ax.set_ylabel(after_y_label)
    ax.legend(markerscale=2, frameon=False, fontsize=8)

    # --- Row 3 right: co-embedding colored by state (circles=RNA, triangles=protein) ---
    ax = axes[2, 1]
    scatter_dual_modality_panel(ax, xy_rna_after, xy_prot_after, obs)
    ax.set_title("Co-embedding — Cell State")
    ax.set_xlabel(after_x_label)
    ax.set_ylabel(after_y_label)

    method_note = reduction_title(
        before_label if (rna_obsm_key or second_obsm_key) else reduction
    )
    fig.suptitle(
        f"{model_name} — Alignment Summary ({method_note})",
        fontsize=13,
        fontweight="bold",
    )
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
        states = obs["state"].values
        colors = state_color_map(states)
        for state, color in colors.items():
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
