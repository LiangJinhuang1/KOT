from __future__ import annotations

import argparse
import gc
import re
from pathlib import Path

import anndata as ad
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.io import mmread


DEFAULT_REPORT = Path(
    "/data2/core-med1-telem/common/Thesis_Jin/Datasets/BMMC/"
    "bmmc_starsolo_output_report.tsv"
)
DEFAULT_OUT_DIR = Path("/data2/core-med1-telem/common/Thesis_Jin/Datasets/BMMC/velocity_h5ad")
DEFAULT_CITE_H5AD = Path(
    "/data2/core-med1-telem/common/Thesis_Jin/Datasets/BMMC/"
    "GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad"
)
DEFAULT_MULTIOME_H5AD = Path(
    "/data2/core-med1-telem/common/Thesis_Jin/Datasets/BMMC/"
    "GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad"
)


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
    parsed = {
        "sample": sample,
        "site": parts[0] if len(parts) > 0 else "",
        "donor": parts[1] if len(parts) > 1 else "",
        "batch": f"s{int(site_num)}d{int(donor_num)}" if site_num.isdigit() and donor_num.isdigit() else "",
        "assay": "multiome_gex" if "multiome_gex" in sample else "cite",
    }
    return parsed


def read_layer(path: Path) -> sparse.csr_matrix:
    matrix = mmread(path).tocsr()
    return matrix.T.tocsr()


def load_velocyto_dir(sample: str, velocyto_dir: Path) -> ad.AnnData:
    barcodes = pd.read_csv(velocyto_dir / "barcodes.tsv", sep="\t", header=None)[0].astype(str)
    features = read_features(velocyto_dir / "features.tsv")

    spliced = read_layer(velocyto_dir / "spliced.mtx")
    unspliced = read_layer(velocyto_dir / "unspliced.mtx")
    ambiguous = read_layer(velocyto_dir / "ambiguous.mtx")

    expected_shape = (len(barcodes), len(features))
    for layer_name, matrix in {
        "spliced": spliced,
        "unspliced": unspliced,
        "ambiguous": ambiguous,
    }.items():
        if matrix.shape != expected_shape:
            raise ValueError(
                f"{sample} {layer_name} has shape {matrix.shape}, expected {expected_shape}"
            )

    obs = pd.DataFrame(index=[f"{sample}:{barcode}" for barcode in barcodes])
    obs["barcode"] = barcodes.to_numpy()
    for key, value in parse_sample(sample).items():
        obs[key] = value

    var = features.copy()
    var.index = var["gene_name"].astype(str)
    var_names = pd.Index(var.index)
    if not var_names.is_unique:
        var.index = var["gene_id"].astype(str)

    adata = ad.AnnData(X=(spliced + unspliced).tocsr(), obs=obs, var=var)
    adata.layers["spliced"] = spliced
    adata.layers["unspliced"] = unspliced
    adata.layers["ambiguous"] = ambiguous
    return adata


def normalize_barcode(name: str) -> str:
    first = str(name).split("-")[0]
    return first


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


def split_modalities(adata: ad.AnnData) -> tuple[ad.AnnData, ad.AnnData | None, ad.AnnData | None]:
    if "feature_types" not in adata.var:
        return adata, None, None
    feature_types = adata.var["feature_types"].astype(str)
    gex_mask = feature_types == "GEX"
    adt_mask = feature_types == "ADT"
    atac_mask = feature_types == "ATAC"
    gex = adata[:, gex_mask].copy() if gex_mask.any() else adata
    adt = adata[:, adt_mask].copy() if adt_mask.any() else None
    atac = adata[:, atac_mask].copy() if atac_mask.any() else None
    return gex, adt, atac


def build_integrated_h5ad(
    report_path: Path,
    original_h5ad: Path,
    output_path: Path,
    assay: str,
) -> None:
    report = read_report(report_path)
    assay_rows = []
    for row in report.itertuples(index=False):
        parsed = parse_sample(row.sample)
        if parsed["assay"] == assay:
            assay_rows.append((row, parsed))

    if not assay_rows:
        raise ValueError(f"No {assay} STARsolo outputs found in report.")

    print(f"Reading original processed h5ad: {original_h5ad}")
    original = sc.read_h5ad(original_h5ad)
    original.var_names_make_unique()
    original_gex, original_adt, original_atac = split_modalities(original)
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
            sample_velocity.obs["barcode"],
            batch,
            original_gex.obs_names,
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
    print(f"Matched cells: {len(common_cells)}")
    if len(common_cells) == 0:
        print("Original obs_names examples:")
        print(original_gex.obs_names[:10].tolist())
        print("Velocity obs_names examples:")
        print(velocity.obs_names[:10].tolist())
        raise ValueError("No cells matched between original h5ad and velocity matrices.")

    velocity_gene_keys = pd.Index(velocity.var["_merge_gene_key"].astype(str))
    velocity_key_to_pos = {}
    for pos, key in enumerate(velocity_gene_keys):
        if key not in velocity_key_to_pos:
            velocity_key_to_pos[key] = pos

    common_gene_keys = pd.Index([key for key in original_gene_index if key in velocity_key_to_pos])
    print(f"Original genes: {original_gex.n_vars}")
    print(f"Velocity genes: {velocity.n_vars}")
    print(f"Matched genes: {len(common_gene_keys)}")
    if len(common_gene_keys) == 0:
        print("Original gene key examples:")
        print(original_gene_index[:10].tolist())
        print("Velocity var examples:")
        print(velocity.var.head(10).to_string())
        print("Velocity gene key examples:")
        print(pd.Index(velocity.var["_merge_gene_key"])[:10].tolist())
        raise ValueError("No genes matched between original h5ad and velocity matrices.")

    original_gene_mask = original_gene_index.isin(set(common_gene_keys))
    original_subset = original_gex[common_cells, original_gene_mask].copy()
    original_subset.var["_merge_gene_key"] = original_gene_index[original_gene_mask].to_numpy()
    del original_gex
    gc.collect()

    velocity_positions = [
        velocity_key_to_pos[key]
        for key in original_subset.var["_merge_gene_key"].astype(str)
    ]
    velocity_subset = velocity[common_cells, velocity_positions].copy()
    del velocity
    gc.collect()

    for layer in ["spliced", "unspliced", "ambiguous"]:
        original_subset.layers[layer] = velocity_subset.layers[layer]
    del velocity_subset
    gc.collect()

    if original_adt is not None:
        adt_subset = original_adt[common_cells].copy()
        original_subset.obsm["protein_counts"] = adt_subset.X.copy()
        original_subset.uns["protein_names"] = adt_subset.var_names.astype(str).to_list()
        print(f"Attached protein matrix: {adt_subset.n_vars} features")
        del adt_subset, original_adt
        gc.collect()

    if original_atac is not None:
        atac_subset = original_atac[common_cells].copy()
        original_subset.obsm["atac_peaks"] = atac_subset.X.copy()
        original_subset.uns["atac_peak_names"] = atac_subset.var_names.astype(str).to_list()
        print(f"Attached ATAC peak matrix: {atac_subset.n_vars} peaks")
        del atac_subset, original_atac
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


def build_h5ads(report_path: Path, out_dir: Path, assay: str) -> None:
    report = read_report(report_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    adatas: list[ad.AnnData] = []
    for row in report.itertuples(index=False):
        sample = row.sample
        parsed = parse_sample(sample)
        if assay != "all" and parsed["assay"] != assay:
            continue

        print(f"Loading {sample}")
        adata = load_velocyto_dir(sample, Path(row.velocyto_dir))
        sample_path = out_dir / f"{sample}.velocity.h5ad"
        adata.write_h5ad(sample_path)
        print(f"  wrote {sample_path} {adata.shape}")
        adatas.append(adata)

    if not adatas:
        raise ValueError(f"No samples matched assay={assay}")

    merged = ad.concat(adatas, join="inner", label="sample_batch", keys=[a.obs["sample"].iloc[0] for a in adatas])
    merged_path = out_dir / f"bmmc_{assay}.velocity.h5ad"
    merged.write_h5ad(merged_path)
    print(f"Wrote merged AnnData: {merged_path} {merged.shape}")
    print(f"Layers: {list(merged.layers.keys())}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Add STARsolo velocity layers to processed BMMC h5ad files.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--dataset", choices=["cite", "multiome"], default="cite")
    parser.add_argument("--original-h5ad", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    default_inputs = {
        "cite": DEFAULT_CITE_H5AD,
        "multiome": DEFAULT_MULTIOME_H5AD,
    }
    default_outputs = {
        "cite": DEFAULT_OUT_DIR / "GSE194122_openproblems_neurips2021_cite_BMMC_processed.velocity.h5ad",
        "multiome": DEFAULT_OUT_DIR / "GSE194122_openproblems_neurips2021_multiome_BMMC_processed.velocity.h5ad",
    }
    original_h5ad = args.original_h5ad or default_inputs[args.dataset]
    output = args.output or default_outputs[args.dataset]

    assay = "multiome_gex" if args.dataset == "multiome" else "cite"
    build_integrated_h5ad(args.report, original_h5ad, output, assay)


if __name__ == "__main__":
    main()
