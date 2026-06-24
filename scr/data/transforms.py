import numpy as np
import scanpy as sc
from scipy.sparse import diags, issparse, csr_matrix
from sklearn.decomposition import TruncatedSVD


# --- utilities ---

def dense_matrix(matrix):
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return np.asarray(matrix)


def clr_transform(matrix):
    values = dense_matrix(matrix).astype(float)
    log_values = np.log1p(values)
    return (log_values - log_values.mean(axis=1, keepdims=True)).astype("float32")


# --- QC: gene/feature filtering only, no cell filtering ---

def qc_rna(adata, min_cells=3):
    sc.pp.filter_genes(adata, min_cells=min_cells)


def qc_protein(adata, min_cells=1):
    sc.pp.filter_genes(adata, min_cells=min_cells)


def qc_atac(adata, min_cells=3):
    sc.pp.filter_genes(adata, min_cells=min_cells)


# --- normalization ---

def normalize_rna(adata, target_sum=1e4, n_top_genes=2000, flavor="seurat_v3"):
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    if not n_top_genes or n_top_genes <= 0 or adata.n_vars < 50:
        print(f"[preprocess] Skipping RNA HVG selection for {adata.n_vars} genes.")
        return
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=min(n_top_genes, adata.n_vars),
        flavor=flavor,
        layer="counts",
    )


def add_log_velocity_layers(adata, target_sum=1e4, velocity_scale=1.0):
    """
    Add velocity layers in the same coordinates as normalize_total + log1p RNA.

    If x_j = log(1 + target_sum * c_j / sum_k c_k), then this stores dx_j/dt
    for each available raw velocity layer. velocity_scale converts velocities
    into the same raw units as adata.layers["counts"] when needed.
    """
    counts = dense_matrix(adata.layers["counts"]).astype("float32")
    total = counts.sum(axis=1, keepdims=True)
    total_safe = np.maximum(total, 1e-8)
    normalized = target_sum * counts / total_safe

    for source in ("true_velocity", "velocity"):
        if source not in adata.layers:
            continue

        velocity = dense_matrix(adata.layers[source]).astype("float32") * float(velocity_scale)
        total_velocity = velocity.sum(axis=1, keepdims=True)
        normalized_velocity = target_sum * (
            velocity * total_safe - counts * total_velocity
        ) / (total_safe ** 2)
        log_velocity = normalized_velocity / (1.0 + normalized)

        target = f"{source}_log"
        adata.layers[target] = log_velocity.astype("float32")
        adata.uns.setdefault("log_velocity_layers", {})[target] = {
            "source": source,
            "transform": "d/dt log1p(normalize_total(counts))",
            "target_sum": float(target_sum),
            "velocity_scale": float(velocity_scale),
        }


def normalize_protein(adata):
    adata.layers["counts"] = adata.X.copy()
    adata.X = clr_transform(adata.X)


def normalize_atac(adata):
    adata.layers["counts"] = adata.X.copy()
    X = adata.X if issparse(adata.X) else csr_matrix(adata.X)
    row_sums = np.asarray(X.sum(axis=1)).ravel()
    tf = diags(1.0 / row_sums) @ X
    n_cells = X.shape[0]
    df = np.asarray((X > 0).sum(axis=0)).ravel()
    idf = np.log1p(n_cells / (df + 1.0))
    adata.X = (tf @ diags(idf)).astype("float32")


# --- dimensionality reduction ---

def reduce_rna(adata, n_pcs=30, n_neighbors=30):
    use_hvg = "highly_variable" in adata.var.columns
    n_comps = min(n_pcs, adata.n_vars - 1, adata.n_obs - 1)
    sc.tl.pca(adata, n_comps=n_comps, use_highly_variable=use_hvg)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_comps)
    sc.tl.umap(adata)
    sc.tl.leiden(adata)


def reduce_protein(adata, n_pcs=10):
    n_comps = min(n_pcs, adata.n_vars - 1, adata.n_obs - 1)
    sc.tl.pca(adata, n_comps=n_comps)


def reduce_atac(adata, n_components=30, n_neighbors=30):
    n_comps = min(n_components + 1, min(adata.n_obs, adata.n_vars) - 1)
    svd = TruncatedSVD(n_components=n_comps, random_state=0)
    X_lsi = svd.fit_transform(adata.X)
    adata.obsm["X_lsi"] = X_lsi[:, 1:].astype("float32")
    adata.uns["lsi"] = {"variance_ratio": svd.explained_variance_ratio_[1:].tolist()}
    sc.pp.neighbors(adata, use_rep="X_lsi", n_neighbors=n_neighbors)
    sc.tl.umap(adata)
