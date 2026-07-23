from __future__ import annotations

import argparse
import gc
import re
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv
from scipy import sparse
from scipy.io import mmread

from src.data.transforms import reduce_rna
from src.data.adt_gene_map import load_mapping_csv
from src.utils.io import load_yaml

DEFAULT_CONFIG = Path("config/velocity.yaml")


# --- label / path utilities ---

def safe_label(value: str) -> str:
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in value)


def make_output_dir(output_root: Path, label: str) -> Path:
    output_dir = output_root / safe_label(label)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def as_path(config: dict, key: str, required: bool = False) -> Path | None:
    value = config.get(key)
    if value in (None, ""):
        if required:
            raise ValueError(f"Missing required config value: {key}")
        return None
    return Path(value)


# --- STARsolo / velocyto loading ---

def read_features(path: Path) -> pd.DataFrame:
    features = pd.read_csv(path, sep="\t", header=None)
    if features.shape[1] == 1:
        features.columns = ["gene_id"]
        features["gene_name"] = features["gene_id"]
    else:
        features = features.iloc[:, :2]
        features.columns = ["gene_id", "gene_name"]
        first_is_ensembl = features["gene_id"].astype(str).str.startswith("ENSG").mean() > 0.5
        second_is_ensembl = features["gene_name"].astype(str).str.startswith("ENSG").mean() > 0.5
        if second_is_ensembl and not first_is_ensembl:
            features = features.rename(columns={"gene_id": "gene_name", "gene_name": "gene_id"})
    return features


def parse_sample(sample: str) -> dict[str, str]:
    parts = sample.split(".")[0].split("_")
    site_num = parts[0].removeprefix("site") if len(parts) > 0 else ""
    donor_num = parts[1].removeprefix("donor") if len(parts) > 1 else ""
    return {
        "sample": sample,
        "site": parts[0] if len(parts) > 0 else "",
        "donor": parts[1] if len(parts) > 1 else "",
        "batch": f"s{int(site_num)}d{int(donor_num)}" if site_num.isdigit() and donor_num.isdigit() else "",
        "assay": "multiome_gex" if "multiome_gex" in sample else "cite",
    }


def read_layer(path: Path) -> sparse.csr_matrix:
    return mmread(path).tocsr().T.tocsr()


def load_velocyto_dir(sample: str, velocyto_dir: Path) -> ad.AnnData:
    barcodes = pd.read_csv(velocyto_dir / "barcodes.tsv", sep="\t", header=None)[0].astype(str)
    features = read_features(velocyto_dir / "features.tsv")

    spliced = read_layer(velocyto_dir / "spliced.mtx")
    unspliced = read_layer(velocyto_dir / "unspliced.mtx")
    ambiguous = read_layer(velocyto_dir / "ambiguous.mtx")

    expected_shape = (len(barcodes), len(features))
    for layer_name, matrix in {"spliced": spliced, "unspliced": unspliced, "ambiguous": ambiguous}.items():
        if matrix.shape != expected_shape:
            raise ValueError(f"{sample} {layer_name} shape {matrix.shape} != expected {expected_shape}")

    obs = pd.DataFrame(index=[f"{sample}:{barcode}" for barcode in barcodes])
    obs["barcode"] = barcodes.to_numpy()
    for key, value in parse_sample(sample).items():
        obs[key] = value

    var = features.copy()
    var.index = var["gene_name"].astype(str)
    if not pd.Index(var.index).is_unique:
        var.index = var["gene_id"].astype(str)

    adata = ad.AnnData(X=(spliced + unspliced).tocsr(), obs=obs, var=var)
    adata.layers["spliced"] = spliced
    adata.layers["unspliced"] = unspliced
    adata.layers["ambiguous"] = ambiguous
    return adata


def normalize_barcode(name: str) -> str:
    return str(name).split("-")[0]


def infer_best_batch(sample_barcodes: pd.Series, parsed_batch: str, original: ad.AnnData) -> str:
    sample_core = set(sample_barcodes.astype(str).map(normalize_barcode))
    candidate_batches = [parsed_batch]
    if "batch" in original.obs:
        candidate_batches.extend([b for b in original.obs["batch"].astype(str).unique() if b not in candidate_batches])

    best_batch = parsed_batch
    best_overlap = -1
    for batch in candidate_batches:
        if "batch" in original.obs:
            original_names = original.obs_names[original.obs["batch"].astype(str).to_numpy() == batch]
        else:
            original_names = original.obs_names
        original_core = set(pd.Index(original_names).map(normalize_barcode))
        overlap = len(sample_core.intersection(original_core))
        if overlap > best_overlap:
            best_batch = batch
            best_overlap = overlap

    print(f"  inferred batch for {parsed_batch}: {best_batch} (raw barcode overlap {best_overlap})")
    return best_batch


def make_velocity_obs_names(barcodes: pd.Series, batch: str, original_obs_names: pd.Index) -> list[str]:
    candidates = []
    for barcode in barcodes.astype(str):
        core = normalize_barcode(barcode)
        if re.search(r"-\d+$", barcode):
            candidates.append([f"{barcode}-{batch}", barcode, f"{core}-1-{batch}", f"{core}-{batch}"])
        else:
            candidates.append([f"{barcode}-1-{batch}", f"{barcode}-{batch}", barcode])

    best = [item[0] for item in candidates]
    best_matched = -1
    for candidate_index in range(max(len(item) for item in candidates)):
        proposed = [
            item[candidate_index] if candidate_index < len(item) else item[0]
            for item in candidates
        ]
        matched = pd.Index(proposed).intersection(original_obs_names).size
        if matched > best_matched:
            best = proposed
            best_matched = matched

    print(f"  matched cells with chosen name format: {best_matched}")
    return best


def read_report(report_path: Path) -> pd.DataFrame:
    report = pd.read_csv(report_path, sep="\t")
    required = {"sample", "velocyto_dir", "barcodes", "features", "spliced", "unspliced", "ambiguous"}
    missing = required.difference(report.columns)
    if missing:
        raise ValueError(f"Missing columns in {report_path}: {sorted(missing)}")
    ok_rows = (report[["barcodes", "features", "spliced", "unspliced", "ambiguous"]] == "ok").all(axis=1)
    if not ok_rows.all():
        bad = report.loc[~ok_rows, "sample"].tolist()
        raise ValueError(f"Some STARsolo outputs are incomplete: {bad}")
    return report


def make_gene_index(var: pd.DataFrame) -> pd.Index:
    if "gene_id" in var.columns:
        gene_ids = var["gene_id"].astype(str)
        if gene_ids.notna().any():
            return pd.Index(gene_ids)
    return pd.Index(var.index.astype(str))


def build_integrated_h5ad(report_path: Path, original_h5ad: Path, output_path: Path, assay: str) -> None:
    report = read_report(report_path)
    assay_rows = [(row, parse_sample(row.sample)) for row in report.itertuples(index=False)
                  if parse_sample(row.sample)["assay"] == assay]

    if not assay_rows:
        raise ValueError(f"No {assay} STARsolo outputs found in report.")

    print(f"Reading original processed h5ad: {original_h5ad}")
    original = sc.read_h5ad(original_h5ad)
    original.var_names_make_unique()

    feature_types = original.var.get("feature_types", pd.Series(dtype=str)).astype(str)
    gex_mask = feature_types == "GEX"
    original_gex = original[:, gex_mask].copy() if gex_mask.any() else original
    original_gene_index = make_gene_index(original_gex.var)

    velocity_parts: list[ad.AnnData] = []
    for row, parsed in assay_rows:
        batch = parsed["batch"]
        if not batch:
            raise ValueError(f"Could not infer batch from sample name: {row.sample}")
        print(f"Loading velocity sample {row.sample} -> batch {batch}")
        sample_velocity = load_velocyto_dir(row.sample, Path(row.velocyto_dir))
        batch = infer_best_batch(sample_velocity.obs["barcode"], batch, original_gex)
        sample_velocity.obs_names = make_velocity_obs_names(
            sample_velocity.obs["barcode"], batch, original_gex.obs_names,
        )
        sample_velocity.obs["batch"] = batch
        velocity_parts.append(sample_velocity)

    velocity = ad.concat(velocity_parts, join="inner")
    velocity.var_names_make_unique()
    velocity_gene_index = make_gene_index(velocity.var)
    velocity.var["_merge_gene_key"] = velocity_gene_index

    common_cells = original_gex.obs_names.intersection(velocity.obs_names)
    print(f"Original cells: {original_gex.n_obs}")
    print(f"Velocity cells: {velocity.n_obs}")
    print(f"Matched cells:  {len(common_cells)}")
    if len(common_cells) == 0:
        print("Original obs_names examples:", original_gex.obs_names[:10].tolist())
        print("Velocity obs_names examples:", velocity.obs_names[:10].tolist())
        raise ValueError("No cells matched between original h5ad and velocity matrices.")

    velocity_gene_keys = pd.Index(velocity.var["_merge_gene_key"].astype(str))
    velocity_key_to_pos = {}
    for pos, key in enumerate(velocity_gene_keys):
        if key not in velocity_key_to_pos:
            velocity_key_to_pos[key] = pos

    common_gene_keys = pd.Index([key for key in original_gene_index if key in velocity_key_to_pos])
    print(f"Original genes: {original_gex.n_vars}")
    print(f"Velocity genes: {velocity.n_vars}")
    print(f"Matched genes:  {len(common_gene_keys)}")
    if len(common_gene_keys) == 0:
        raise ValueError("No genes matched between original h5ad and velocity matrices.")

    original_gene_mask = original_gene_index.isin(set(common_gene_keys))
    original_subset = original_gex[common_cells, original_gene_mask].copy()
    original_subset.var["_merge_gene_key"] = original_gene_index[original_gene_mask].to_numpy()
    del original_gex
    gc.collect()

    velocity_positions = [velocity_key_to_pos[key] for key in original_subset.var["_merge_gene_key"].astype(str)]
    velocity_subset = velocity[common_cells, velocity_positions].copy()
    del velocity
    gc.collect()

    for layer in ["spliced", "unspliced", "ambiguous"]:
        original_subset.layers[layer] = velocity_subset.layers[layer]
    del velocity_subset
    gc.collect()

    original_subset.uns["velocity_source"] = {
        "starsolo_report": str(report_path),
        "original_h5ad": str(original_h5ad),
        "matched_cells": int(len(common_cells)),
        "matched_genes": int(len(common_gene_keys)),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    original_subset.write_h5ad(output_path)
    print(f"Wrote integrated {assay} velocity h5ad: {output_path}")
    print(original_subset)


# --- scVelo helpers ---

def choose_color(adata):
    for column in ["cell_type", "leiden", "batch", "Samplename", "Site"]:
        if column in adata.obs:
            return column
    return None


def has_splicing_layers(adata) -> bool:
    return "spliced" in adata.layers and "unspliced" in adata.layers


def read_velocity_loom(path: Path):
    read_legacy = getattr(scv, "read", None)
    if read_legacy is not None:
        return read_legacy(str(path), cache=True)
    return sc.read(path)


def clean_loom_barcodes(adata) -> None:
    adata.obs_names = adata.obs_names.str.split(":").str[-1].str.replace("x", "-1", regex=False)


def load_shareseq_counts(counts_path: Path) -> ad.AnnData:
    df = pd.read_csv(counts_path, sep="\t", index_col=0)
    values = np.nan_to_num(df.values.T.astype("float32"), nan=0.0, posinf=0.0, neginf=0.0)
    adata = ad.AnnData(X=sparse.csr_matrix(values))
    adata.obs_names = list(df.columns)
    adata.var_names = list(df.index)
    adata.layers["counts"] = adata.X.copy()
    return adata


def merge_shareseq_velocity(rna: ad.AnnData, ldata: ad.AnnData) -> ad.AnnData:
    obs = ldata.obs_names
    if obs.str.contains(":").any():
        obs = obs.str.split(":").str[-1]
    obs = obs.str.rstrip("x")
    ldata.obs_names = obs
    ldata.var_names_make_unique()
    ldata.obs_names_make_unique()

    common_cells = rna.obs_names.intersection(ldata.obs_names)
    print(f"RNA cells: {rna.n_obs}  Loom cells: {ldata.n_obs}  Common: {len(common_cells)}")
    if len(common_cells) == 0:
        print("RNA obs_names examples: ", rna.obs_names[:3].tolist())
        print("Loom obs_names examples:", ldata.obs_names[:3].tolist())
        raise ValueError("No common cells between counts and loom — barcode format mismatch.")

    common_genes = rna.var_names.intersection(ldata.var_names)
    print(f"RNA genes: {rna.n_vars}  Loom genes: {ldata.n_vars}  Common: {len(common_genes)}")

    merged = rna[common_cells, common_genes].copy()
    ldata_sub = ldata[common_cells, common_genes]
    for layer in ["spliced", "unspliced", "ambiguous"]:
        if layer in ldata_sub.layers:
            merged.layers[layer] = ldata_sub.layers[layer]
    return merged


def split_rna_from_10x(path: Path):
    adata = sc.read_10x_h5(path, gex_only=False)
    adata.var_names_make_unique()
    if "feature_types" not in adata.var:
        raise ValueError("10x input is missing var['feature_types']; cannot split RNA.")
    rna = adata[:, adata.var["feature_types"] == "Gene Expression"].copy()
    rna.layers["counts"] = rna.X.copy()
    sc.pp.filter_genes(rna, min_counts=1)
    return rna


def select_hvgs(adata, n_top_genes: int, hvg_flavor: str, retain_genes=None):
    if not n_top_genes:
        return adata
    # seurat_v3 requires raw counts; seurat/cell_ranger use log-normalized X
    hvg_layer = "counts" if (hvg_flavor == "seurat_v3" and "counts" in adata.layers) else None
    hvg_genes = min(n_top_genes, max(1, adata.n_vars - 1))
    sc.pp.highly_variable_genes(adata, layer=hvg_layer, n_top_genes=hvg_genes, flavor=hvg_flavor)
    keep = adata.var["highly_variable"].to_numpy().copy()
    if retain_genes:
        # Force-retain ADT-target genes so KOT's S/kinetics can use them, even
        # when they are not among the most variable genes.
        forced = adata.var_names.isin(list(retain_genes))
        n_added = int((forced & ~keep).sum())
        keep = keep | forced
        print(f"[velocity] retained {n_added} ADT-target genes on top of "
              f"{int(adata.var['highly_variable'].sum())} HVGs → {int(keep.sum())} total")
    return adata[:, keep].copy()


def run_scvelo(adata, n_top_genes, hvg_flavor, min_shared_counts, n_pcs, n_neighbors,
               velocity_mode, dynamics_n_jobs, retain_genes=None):
    if not has_splicing_layers(adata):
        raise ValueError("Missing spliced/unspliced layers; cannot calculate RNA velocity.")
    scv.utils.show_proportions(adata)
    adata = select_hvgs(adata, n_top_genes, hvg_flavor, retain_genes=retain_genes)
    scv.pp.filter_and_normalize(adata, min_shared_counts=min_shared_counts)
    actual_n_pcs = min(n_pcs, adata.n_obs - 1, adata.n_vars - 1)
    if actual_n_pcs < 1:
        raise ValueError(f"Too few features ({adata.n_vars}) after filtering — cannot run PCA.")
    sc.tl.pca(adata, n_comps=actual_n_pcs)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=actual_n_pcs)
    # scVelo velocity_graph requires CSR format; newer scanpy may use other sparse types
    for key in ("connectivities", "distances"):
        if key in adata.obsp:
            adata.obsp[key] = sparse.csr_matrix(adata.obsp[key])
    scv.pp.moments(adata, n_pcs=actual_n_pcs, n_neighbors=n_neighbors)
    if velocity_mode == "dynamical":
        scv.tl.recover_dynamics(adata, n_jobs=dynamics_n_jobs)
    scv.tl.velocity(adata, mode=velocity_mode)
    scv.tl.velocity_graph(adata)
    scv.tl.velocity_confidence(adata)
    scv.tl.velocity_pseudotime(adata)
    if velocity_mode == "dynamical":
        scv.tl.latent_time(adata)
    return adata


# --- loaders ---

def load_pbmc(pbmc_counts: Path, loom: Path):
    print(f"Reading PBMC 10x counts: {pbmc_counts}")
    rna = split_rna_from_10x(pbmc_counts)
    sc.pp.normalize_total(rna, target_sum=1e4)
    sc.pp.log1p(rna)
    reduce_rna(rna)
    print(f"Reading PBMC velocity loom: {loom}")
    ldata = read_velocity_loom(loom)
    clean_loom_barcodes(ldata)
    adata = scv.utils.merge(rna, ldata)
    adata.obsm["X_umap"] = rna[adata.obs_names].obsm["X_umap"]
    adata.obs["leiden"] = rna.obs.loc[adata.obs_names, "leiden"].astype(str).to_numpy()
    return adata


def load_h5ad(path: Path, umap_key: str | None):
    print(f"Reading velocity h5ad: {path}")
    adata = sc.read_h5ad(path)
    adata.var_names_make_unique()
    if umap_key and umap_key in adata.obsm:
        adata.obsm["X_umap"] = adata.obsm[umap_key]
    elif "X_umap" not in adata.obsm:
        raise ValueError("No UMAP found. Set umap_key in velocity.yaml or provide X_umap.")
    return adata


# --- saving ---

def save_summary(adata, output_dir: Path, label: str) -> None:
    path = output_dir / f"{label}_summary.txt"
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"shape: {adata.shape}\n")
        handle.write(f"layers: {list(adata.layers.keys())}\n")
        handle.write(f"obs columns: {list(adata.obs.columns)}\n")
        handle.write(f"var columns: {list(adata.var.columns)}\n")
        handle.write(f"obsm keys: {list(adata.obsm.keys())}\n")
        handle.write(f"uns keys: {list(adata.uns.keys())}\n")
    print(f"Wrote {path}")


def save_confidence(adata, output_dir: Path, label: str) -> None:
    columns = ["velocity_length", "velocity_confidence", "velocity_confidence_transition"]
    existing = [c for c in columns if c in adata.obs]
    if existing:
        path = output_dir / f"{label}_velocity_confidence.csv"
        adata.obs[existing].to_csv(path)
        print(f"Wrote {path}")


def save_plots(adata, output_dir: Path, label: str) -> None:
    scv.settings.figdir = str(output_dir)
    color = choose_color(adata)
    scv.pl.proportions(adata, save=f"{label}_proportions.png")
    scv.pl.velocity_embedding(adata, basis="umap", color=color, save=f"{label}_embedding.png")
    scv.pl.velocity_embedding_grid(adata, basis="umap", color=color, save=f"{label}_grid.png")
    scv.pl.velocity_embedding_stream(adata, basis="umap", color=color, save=f"{label}_stream.png")
    scv.pl.scatter(adata, color="velocity_pseudotime", cmap="gnuplot", save=f"{label}_pseudotime.png")
    if "latent_time" in adata.obs:
        scv.pl.scatter(adata, color="latent_time", cmap="gnuplot", save=f"{label}_latent_time.png")
    plt.close("all")


# --- pipeline orchestration ---

def build_run_config(config: dict, dataset_name: str) -> dict:
    datasets = config.get("datasets", {})
    if dataset_name not in datasets:
        available = ", ".join(sorted(datasets)) or "(none)"
        raise KeyError(f"Unknown dataset '{dataset_name}'. Available datasets: {available}")
    run_config = dict(config.get("defaults", {}))
    run_config.update(datasets[dataset_name] or {})
    run_config.setdefault("label", dataset_name)
    return run_config


def execute_run(run_config: dict) -> None:
    scv.settings.verbosity = 3
    mode = run_config.get("mode")
    label = str(run_config["label"])
    output_root = as_path(run_config, "output_root", required=True)
    output_dir = make_output_dir(output_root, label)

    if mode == "starsolo_h5ad":
        velocity_input_path = output_dir / f"{label}_velocity_input.h5ad"
        assay = "multiome_gex" if run_config.get("assay") == "multiome" else run_config.get("assay", "cite")
        build_integrated_h5ad(
            as_path(run_config, "report", required=True),
            as_path(run_config, "original_h5ad", required=True),
            velocity_input_path,
            assay,
        )
        adata = load_h5ad(velocity_input_path, run_config.get("umap_key"))
    elif mode == "h5ad":
        adata = load_h5ad(as_path(run_config, "input_h5ad", required=True), run_config.get("umap_key"))
        merged_path = output_dir / f"{label}_merged_velocity_input.h5ad"
        adata.write_h5ad(merged_path)
        print(f"Wrote {merged_path}")
    elif mode == "pbmc_10x_loom":
        adata = load_pbmc(
            as_path(run_config, "pbmc_counts", required=True),
            as_path(run_config, "loom", required=True),
        )
        merged_path = output_dir / f"{label}_merged_velocity_input.h5ad"
        adata.write_h5ad(merged_path)
        print(f"Wrote {merged_path}")
    elif mode == "shareseq_loom":
        counts_path = as_path(run_config, "counts", required=True)
        print(f"Loading SHARE-seq counts: {counts_path}")
        rna = load_shareseq_counts(counts_path)
        print(f"Loaded RNA: {rna.shape}")
        sc.pp.normalize_total(rna, target_sum=1e4)
        sc.pp.log1p(rna)
        reduce_rna(rna)
        ldata = read_velocity_loom(as_path(run_config, "loom", required=True))
        adata = merge_shareseq_velocity(rna, ldata)
        adata.obsm["X_umap"] = rna[adata.obs_names].obsm["X_umap"]
        adata.obs["leiden"] = rna.obs.loc[adata.obs_names, "leiden"].astype(str).to_numpy()
        merged_path = output_dir / f"{label}_merged_velocity_input.h5ad"
        adata.write_h5ad(merged_path)
        print(f"Wrote {merged_path}")
    else:
        raise ValueError(f"Unsupported mode: {mode}. Expected starsolo_h5ad, h5ad, pbmc_10x_loom, or shareseq_loom.")

    save_summary(adata, output_dir, f"{label}_merged")

    retain_genes = None
    retain_csv = run_config.get("retain_genes_csv")
    if retain_csv and Path(retain_csv).exists():
        retain_genes = set(load_mapping_csv(Path(retain_csv)).values())
        print(f"[velocity] force-retaining {len(retain_genes)} ADT-target genes from {retain_csv}")

    adata = run_scvelo(
        adata,
        int(run_config.get("n_top_genes", 2000) or 0),
        str(run_config.get("hvg_flavor", "seurat_v3")),
        int(run_config.get("min_shared_counts", 20)),
        int(run_config.get("n_pcs", 30)),
        int(run_config.get("n_neighbors", 30)),
        str(run_config.get("velocity_mode", "dynamical")),
        int(run_config.get("dynamics_n_jobs", 1)),
        retain_genes=retain_genes,
    )

    save_plots(adata, output_dir, label)
    save_confidence(adata, output_dir, label)

    # Retained runs write to a separate _retained file so the original velocity
    # result (and its downstream runs) stay reproducible.
    suffix = "_retained" if retain_genes else ""
    result_path = output_dir / f"{label}_scvelo_results{suffix}.h5ad"
    adata.write_h5ad(result_path)
    print(f"Wrote {result_path}")


def run_dataset(dataset_name: str, config_path: Path = DEFAULT_CONFIG) -> None:
    config = load_yaml(Path(config_path))
    config.setdefault("defaults", {})
    config.setdefault("datasets", {})
    run_config = build_run_config(config, dataset_name)
    execute_run(run_config)


# --- CLI ---

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate and save RNA velocity.")
    parser.add_argument("dataset", nargs="?", help="Dataset key in velocity.yaml.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--mode", choices=["h5ad", "pbmc_10x_loom", "starsolo_h5ad", "shareseq_loom"], default=None)
    parser.add_argument("--counts", type=Path, default=None, help="SHARE-seq rna.counts.txt.gz path")
    parser.add_argument("--input-h5ad", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--original-h5ad", type=Path, default=None)
    parser.add_argument("--assay", choices=["cite", "multiome"], default=None)
    parser.add_argument("--pbmc-counts", type=Path, default=None)
    parser.add_argument("--loom", type=Path, default=None)
    parser.add_argument("--umap-key", default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument("--n-top-genes", type=int, default=None)
    parser.add_argument("--hvg-flavor", default=None)
    parser.add_argument("--min-shared-counts", type=int, default=None)
    parser.add_argument("--n-pcs", type=int, default=None)
    parser.add_argument("--n-neighbors", type=int, default=None)
    parser.add_argument("--velocity-mode", choices=["dynamical", "stochastic", "deterministic"], default=None)
    parser.add_argument("--dynamics-n-jobs", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(Path(args.config))
    config.setdefault("defaults", {})
    config.setdefault("datasets", {})

    if args.list:
        for name in sorted(config.get("datasets", {})):
            mode = config["datasets"][name].get("mode", "")
            print(f"{name}\t{mode}")
        return

    if not args.dataset:
        raise ValueError("Please specify a dataset key, or use --list to see available datasets.")

    run_config = build_run_config(config, args.dataset)
    cli_overrides = {
        "mode": args.mode,
        "input_h5ad": args.input_h5ad,
        "counts": args.counts,
        "report": args.report,
        "original_h5ad": args.original_h5ad,
        "assay": args.assay,
        "pbmc_counts": args.pbmc_counts,
        "loom": args.loom,
        "umap_key": args.umap_key,
        "output_root": args.output_root,
        "label": args.label,
        "n_top_genes": args.n_top_genes,
        "hvg_flavor": args.hvg_flavor,
        "min_shared_counts": args.min_shared_counts,
        "n_pcs": args.n_pcs,
        "n_neighbors": args.n_neighbors,
        "velocity_mode": args.velocity_mode,
        "dynamics_n_jobs": args.dynamics_n_jobs,
        "random_state": args.random_state,
    }
    for key, value in cli_overrides.items():
        if value is not None:
            run_config[key] = value
    execute_run(run_config)
