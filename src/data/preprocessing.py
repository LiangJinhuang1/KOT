import os
import hashlib
import anndata as ad
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix

from src.data.transforms import (
    qc_rna,
    qc_protein,
    qc_atac,
    normalize_rna,
    add_log_velocity_layers,
    normalize_protein,
    normalize_atac,
    reduce_rna,
    reduce_protein,
    reduce_atac,
)


def write_h5ad_atomic(adata, path: str) -> None:
    """Write to a per-process temp file then os.replace into place.

    os.replace is atomic on the filesystem, so concurrent jobs that share a preprocessing
    cache key (e.g. many runs of the same dataset packed on one GPU) never see a partial
    file: readers get either the old or a complete new h5ad, and racing writers just
    redundantly recompute — the last replace wins with a valid file.
    """
    tmp = f"{path}.tmp.{os.getpid()}"
    adata.write_h5ad(tmp)
    os.replace(tmp, path)


# --- UMAP utilities ---

def add_umap_from_csv(adata, umap_path, key="X_umap"):
    print(f"Reading UMAP coordinates: {umap_path}")
    umap_df = pd.read_csv(umap_path, index_col=0)
    coord_cols = [
        col for col in umap_df.columns
        if col.lower() in ["x", "y", "umap1", "umap2", "umap_1", "umap_2"]
    ]
    if len(coord_cols) < 2 and umap_df.shape[1] >= 2:
        coord_cols = umap_df.columns[:2].tolist()
    if len(coord_cols) >= 2:
        coord_df = umap_df[coord_cols[:2]].apply(pd.to_numeric, errors="coerce")
        coord_df = coord_df.reindex(adata.obs_names)
        missing_count = int(coord_df.isna().any(axis=1).sum())
        if missing_count < adata.n_obs:
            adata.obsm[key] = coord_df.to_numpy(dtype="float32")
            print(f"Stored UMAP in adata.obsm['{key}']; missing cells: {missing_count}")
        else:
            raise ValueError("UMAP coordinates were not aligned to cell names.")
    else:
        raise ValueError("UMAP coordinate columns were not found.")


# --- loaders ---

def load_rna_from_h5ad(path, raw_layer=None):
    print(f"Loading RNA from: {path}")
    adata = sc.read_h5ad(path)
    adata.var_names_make_unique()
    if raw_layer:
        if raw_layer not in adata.layers:
            available = list(adata.layers.keys())
            raise ValueError(f"RNA raw_layer '{raw_layer}' not found. Available layers: {available}")
        adata.X = adata.layers[raw_layer].copy()
        print(f"Using layer '{raw_layer}' as X for normalization.")
    if "feature_types" in adata.var.columns:
        rna_mask = adata.var["feature_types"].isin(["GEX", "Gene Expression"])
        if not rna_mask.all():
            adata = adata[:, rna_mask].copy()
    return adata


def load_protein_from_h5ad(path, label="ADT"):
    print(f"Loading protein ({label}) from: {path}")
    adata = sc.read_h5ad(path)
    # Subset to the protein features BEFORE de-duplicating names: in a joint
    # RNA+ADT object an ADT marker (e.g. CD86) collides with the gene of the same
    # name, so make_unique would rename it CD86-1 and break gene→protein matching.
    mask = adata.var["feature_types"] == label
    protein = adata[:, mask].copy()
    protein.var_names_make_unique()
    return protein


def load_protein_from_10x_h5(path, label="Antibody Capture"):
    print(f"Loading protein ({label}) from 10x h5: {path}")
    adata = sc.read_10x_h5(path, gex_only=False)
    mask = adata.var["feature_types"] == label
    protein = adata[:, mask].copy()
    protein.var_names_make_unique()
    return protein


def load_atac_from_shareseq(path):
    print(f"Loading ATAC from SHARE-seq counts: {path}")
    df = pd.read_csv(path, sep="\t", index_col=0)
    adata = ad.AnnData(X=csr_matrix(df.values.T.astype("float32")))
    adata.obs_names = list(df.columns)
    adata.var_names = list(df.index)
    adata.var["feature_types"] = "ATAC"
    return adata


def load_atac_from_h5ad(path, label="ATAC"):
    print(f"Loading ATAC ({label}) from: {path}")
    adata = sc.read_h5ad(path)
    mask = adata.var["feature_types"] == label
    atac = adata[:, mask].copy()
    atac.var_names_make_unique()
    return atac


def load_protein_from_obsm(rna_adata, obsm_key="protein_counts"):
    print(f"Extracting protein from obsm['{obsm_key}']")
    protein_names = rna_adata.uns.get(
        "protein_names",
        [f"protein_{i}" for i in range(rna_adata.obsm[obsm_key].shape[1])],
    )
    protein_adata = ad.AnnData(
        X=rna_adata.obsm[obsm_key].copy(),
        obs=rna_adata.obs.copy(),
    )
    protein_adata.var_names = protein_names
    protein_adata.var["feature_types"] = "Protein"
    return protein_adata


# --- alignment ---

def align_common_obs(rna_adata, protein_adata, atac_adata):
    common = rna_adata.obs_names
    if protein_adata is not None:
        common = common[common.isin(protein_adata.obs_names)]
    if atac_adata is not None:
        common = common[common.isin(atac_adata.obs_names)]

    if not rna_adata.obs_names.equals(common):
        rna_adata = rna_adata[common].copy()
    if protein_adata is not None and not protein_adata.obs_names.equals(common):
        protein_adata = protein_adata[common].copy()
    if atac_adata is not None and not atac_adata.obs_names.equals(common):
        atac_adata = atac_adata[common].copy()

    return rna_adata, protein_adata, atac_adata


# --- per-modality preprocessing orchestrators ---

def preprocess_rna(
    adata,
    min_cells=3,
    n_top_genes=2000,
    n_pcs=30,
    n_neighbors=30,
    add_log_velocity_layer=False,
    log_velocity_scale=1.0,
):
    print(f"Preprocessing RNA: {adata.shape}")
    qc_rna(adata, min_cells=min_cells)
    normalize_rna(adata, n_top_genes=n_top_genes)
    if add_log_velocity_layer:
        add_log_velocity_layers(adata, velocity_scale=log_velocity_scale)
    reduce_rna(adata, n_pcs=n_pcs, n_neighbors=n_neighbors)
    print(f"RNA after preprocessing: {adata.shape}")
    return adata


def preprocess_protein(adata, min_cells=1, n_pcs=10):
    print(f"Preprocessing Protein: {adata.shape}")
    qc_protein(adata, min_cells=min_cells)
    normalize_protein(adata)
    reduce_protein(adata, n_pcs=n_pcs)
    print(f"Protein after preprocessing: {adata.shape}")
    return adata


def preprocess_atac(adata, min_cells=3, n_components=30, n_neighbors=30):
    print(f"Preprocessing ATAC: {adata.shape}")
    qc_atac(adata, min_cells=min_cells)
    normalize_atac(adata)
    reduce_atac(adata, n_components=n_components, n_neighbors=n_neighbors)
    print(f"ATAC after preprocessing: {adata.shape}")
    return adata


# --- cached pipeline ---

def load_and_preprocess_cached(
    rna_path,
    protein_path=None,
    protein_label="ADT",
    protein_obsm_key=None,
    atac_path=None,
    atac_label="ATAC",
    rna_raw_layer=None,
    rna_umap_path=None,
    cache_dir="cache/preprocessed",
    force_recompute=False,
    cache_version=None,
    rna_min_cells=3,
    rna_n_top_genes=2000,
    rna_n_pcs=30,
    rna_n_neighbors=30,
    add_log_velocity_layer=False,
    log_velocity_scale=1.0,
    protein_min_cells=1,
    protein_n_pcs=10,
    atac_min_cells=3,
    atac_n_components=30,
    atac_n_neighbors=30,
):
    key_parts = os.path.abspath(rna_path)
    if rna_raw_layer:
        key_parts += f"|rna_raw_layer={rna_raw_layer}"
    if protein_path:
        key_parts += f"|{os.path.abspath(protein_path)}|protein_label={protein_label}"
    if protein_obsm_key:
        key_parts += f"|protein_obsm_key={protein_obsm_key}"
    if atac_path:
        key_parts += f"|{os.path.abspath(atac_path)}|atac_label={atac_label}"
    if cache_version:
        key_parts += f"|cache_version={cache_version}"
    if rna_umap_path:
        key_parts += f"|rna_umap_path={os.path.abspath(rna_umap_path)}"
    key_parts += (
        "|preprocess_schema=2"
        f"|rna_min_cells={int(rna_min_cells)}"
        f"|rna_n_top_genes={int(rna_n_top_genes)}"
        f"|rna_n_pcs={int(rna_n_pcs)}"
        f"|rna_n_neighbors={int(rna_n_neighbors)}"
        f"|log_velocity={bool(add_log_velocity_layer)}:{float(log_velocity_scale)}"
        f"|protein_min_cells={int(protein_min_cells)}"
        f"|protein_n_pcs={int(protein_n_pcs)}"
        f"|atac_min_cells={int(atac_min_cells)}"
        f"|atac_n_components={int(atac_n_components)}"
        f"|atac_n_neighbors={int(atac_n_neighbors)}"
    )
    cache_key = hashlib.md5(key_parts.encode("utf-8")).hexdigest()[:12]
    base_name = os.path.splitext(os.path.basename(rna_path))[0]
    safe_base = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in base_name)
    cache_prefix = os.path.join(cache_dir, f"{safe_base}_{cache_key}")
    rna_cache = f"{cache_prefix}.rna.h5ad"
    protein_cache = f"{cache_prefix}.protein.h5ad"
    atac_cache = f"{cache_prefix}.atac.h5ad"

    os.makedirs(cache_dir, exist_ok=True)
    print(f"[cache] Prefix: {cache_prefix}")

    if not force_recompute and os.path.exists(rna_cache):
        print(f"[cache] Loading RNA from {rna_cache}")
        rna_adata = sc.read_h5ad(rna_cache)
        protein_adata = sc.read_h5ad(protein_cache) if os.path.exists(protein_cache) else None
        atac_adata = sc.read_h5ad(atac_cache) if os.path.exists(atac_cache) else None
        if rna_umap_path:
            if not os.path.exists(rna_umap_path):
                raise FileNotFoundError(f"RNA UMAP path not found: {rna_umap_path}")
            if "X_umap" not in rna_adata.obsm:
                add_umap_from_csv(rna_adata, rna_umap_path)
                write_h5ad_atomic(rna_adata, rna_cache)
        return rna_adata, protein_adata, atac_adata

    rna_adata = load_rna_from_h5ad(rna_path, raw_layer=rna_raw_layer)

    if protein_path:
        ext = os.path.splitext(protein_path)[-1].lower()
        if ext == ".h5":
            protein_adata = load_protein_from_10x_h5(protein_path, protein_label)
        else:
            protein_adata = load_protein_from_h5ad(protein_path, protein_label)
    elif protein_obsm_key:
        protein_adata = load_protein_from_obsm(rna_adata, protein_obsm_key)
    else:
        protein_adata = None

    if atac_path:
        if str(atac_path).endswith(".h5ad"):
            atac_adata = load_atac_from_h5ad(atac_path, atac_label)
        else:
            atac_adata = load_atac_from_shareseq(atac_path)
    else:
        atac_adata = None

    rna_adata, protein_adata, atac_adata = align_common_obs(rna_adata, protein_adata, atac_adata)

    if rna_umap_path:
        if not os.path.exists(rna_umap_path):
            raise FileNotFoundError(f"RNA UMAP path not found: {rna_umap_path}")
        add_umap_from_csv(rna_adata, rna_umap_path)

    rna_adata = preprocess_rna(
        rna_adata,
        min_cells=rna_min_cells,
        n_top_genes=rna_n_top_genes,
        n_pcs=rna_n_pcs,
        n_neighbors=rna_n_neighbors,
        add_log_velocity_layer=add_log_velocity_layer,
        log_velocity_scale=log_velocity_scale,
    )
    if protein_adata is not None:
        protein_adata = preprocess_protein(
            protein_adata,
            min_cells=protein_min_cells,
            n_pcs=protein_n_pcs,
        )
    if atac_adata is not None:
        atac_adata = preprocess_atac(
            atac_adata,
            min_cells=atac_min_cells,
            n_components=atac_n_components,
            n_neighbors=atac_n_neighbors,
        )

    print(f"[cache] Writing RNA to {rna_cache}")
    write_h5ad_atomic(rna_adata, rna_cache)
    if protein_adata is not None:
        print(f"[cache] Writing Protein to {protein_cache}")
        write_h5ad_atomic(protein_adata, protein_cache)
    elif os.path.exists(protein_cache):
        os.remove(protein_cache)
    if atac_adata is not None:
        print(f"[cache] Writing ATAC to {atac_cache}")
        write_h5ad_atomic(atac_adata, atac_cache)
    elif os.path.exists(atac_cache):
        os.remove(atac_cache)

    return rna_adata, protein_adata, atac_adata
