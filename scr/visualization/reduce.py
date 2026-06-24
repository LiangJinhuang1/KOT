from __future__ import annotations

import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

VIZ_REDUCTIONS = ("pca", "umap", "tsne", "precomputed")


def axis_labels(reduction: str) -> tuple[str, str]:
    reduction = reduction.lower()
    if reduction == "pca":
        return "PC 1", "PC 2"
    if reduction == "umap":
        return "UMAP 1", "UMAP 2"
    if reduction == "tsne":
        return "t-SNE 1", "t-SNE 2"
    if reduction == "precomputed":
        return "Dim 1", "Dim 2"
    raise ValueError(f"Unknown viz_reduction '{reduction}'. Use: {VIZ_REDUCTIONS}")


def reduction_title(reduction: str) -> str:
    return {
        "pca": "PCA",
        "umap": "UMAP",
        "tsne": "t-SNE",
        "precomputed": "precomputed",
    }[reduction.lower()]


def obsm_xy(adata, key: str, label: str) -> np.ndarray:
    if key not in adata.obsm:
        available = list(adata.obsm.keys())
        raise ValueError(f"{label} obsm key '{key}' not found. Available: {available}")
    xy = np.asarray(adata.obsm[key], dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] < 2:
        raise ValueError(f"{label} obsm['{key}'] must be (n_cells, >=2); got {xy.shape}.")
    return xy[:, :2]


def fit_transform_2d(
    data: np.ndarray,
    reduction: str,
    *,
    random_state: int = 42,
    umap_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    tsne_perplexity: float = 30.0,
) -> np.ndarray:
    reduction = reduction.lower()
    data = np.asarray(data, dtype=np.float64)
    n = data.shape[0]

    if reduction == "pca":
        n_components = min(2, data.shape[1], n)
        return PCA(n_components=n_components, random_state=random_state).fit_transform(data)

    if reduction == "umap":
        n_neighbors = min(int(umap_neighbors), n - 1)
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=max(2, n_neighbors),
            min_dist=float(umap_min_dist),
            random_state=random_state,
        )
        return reducer.fit_transform(data)

    if reduction == "tsne":
        perplexity = min(float(tsne_perplexity), max(1.0, (n - 1) / 3.0))
        return TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=random_state,
            init="pca",
            learning_rate="auto",
        ).fit_transform(data)

    raise ValueError(f"Unknown viz_reduction '{reduction}'. Use: {VIZ_REDUCTIONS}")


def before_alignment_xy(
    x: np.ndarray,
    y: np.ndarray,
    *,
    reduction: str,
    rna_adata=None,
    second_adata=None,
    rna_obsm_key: str | None = None,
    second_obsm_key: str | None = None,
    random_state: int = 42,
    umap_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    tsne_perplexity: float = 30.0,
) -> tuple[np.ndarray, np.ndarray, str]:
    reduction = reduction.lower()
    fit_method = "pca" if reduction == "precomputed" else reduction
    if reduction == "precomputed" and not rna_obsm_key and not second_obsm_key:
        raise ValueError(
            "viz_reduction=precomputed requires viz_before_rna_obsm and/or "
            "viz_before_second_obsm."
        )

    kwargs = dict(
        random_state=random_state,
        umap_neighbors=umap_neighbors,
        umap_min_dist=umap_min_dist,
        tsne_perplexity=tsne_perplexity,
    )

    if rna_obsm_key:
        xy_rna = obsm_xy(rna_adata, rna_obsm_key, "RNA")
    else:
        xy_rna = fit_transform_2d(x, fit_method, **kwargs)

    if second_obsm_key:
        xy_second = obsm_xy(second_adata, second_obsm_key, "Protein")
    else:
        xy_second = fit_transform_2d(y, fit_method, **kwargs)

    label = reduction if reduction != "precomputed" else "precomputed"
    return xy_rna, xy_second, label


def after_alignment_xy(
    aligned_rna: np.ndarray,
    aligned_protein: np.ndarray,
    *,
    reduction: str,
    random_state: int = 42,
    umap_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    tsne_perplexity: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    reduction = "pca" if reduction.lower() == "precomputed" else reduction.lower()
    n = len(aligned_rna)
    stacked = np.vstack([aligned_rna, aligned_protein])
    xy_joint = fit_transform_2d(
        stacked,
        reduction,
        random_state=random_state,
        umap_neighbors=umap_neighbors,
        umap_min_dist=umap_min_dist,
        tsne_perplexity=tsne_perplexity,
    )
    return xy_joint[:n], xy_joint[n:]
