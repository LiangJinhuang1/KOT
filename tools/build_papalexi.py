"""Build Papalexi ECCITE-seq RNA-velocity input + protein h5ads for KOT alignment.

The pooled STARsolo run (8 lanes) produces spliced/unspliced/ambiguous counts over
raw 16bp barcodes. The GEO ECCITE files carry the per-lane cell ids (l{lane}_{16bp}),
the 4-protein checkpoint ADT panel (CD86, PD-L1, PD-L2, CD366/TIM3) and the cell
annotations (perturbation, replicate, cell cycle).

Cells are matched on the core 16bp barcode. The ~1.4% of core barcodes reused across
lanes are ambiguous under a pooled STARsolo run (their velocity counts sum two cells),
so they are dropped. RNA X is spliced+unspliced (self-consistent with the velocity
layers and stored as a raw `counts` layer for seurat_v3 HVG), so no GEO cDNA matrix is
needed. The result mirrors the bmmc_cite_retained contract: an RNA h5ad with
spliced/unspliced layers for scVelo, plus a matched ADT protein h5ad.

Run on the cluster where the STARsolo output lives:
  PYTHONPATH=. python -m tools.build_papalexi \
      --velocyto-dir Datasets/Papalexi_GSE153056/starsolo_eccite_pooled/Solo.out/Velocyto/raw \
      --tar data/papalexi/GSE153056_RAW.tar \
      --metadata data/papalexi/ECCITE_metadata.tsv
"""

from __future__ import annotations

import argparse
import gzip
import tarfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

from src.data.adt_gene_map import gene_alias_set, normalize_gene_symbol, validate_mapping_records
from src.data.velocity import read_features, read_layer, validate_layer_shapes, write_h5ad_atomic

# ADT (protein) -> RNA gene symbol for the 4-marker checkpoint panel.
ADT_GENE_MAP = {
    "CD86": "CD86",       # CD86 costimulatory ligand
    "PDL1": "CD274",      # PD-L1
    "PDL2": "PDCD1LG2",   # PD-L2
    "CD366": "HAVCR2",    # TIM-3
}


def normalize_core(barcode: str) -> str:
    """Core 16bp cell barcode: drop a lane prefix (l1_) and a 10x suffix (-1)."""
    return str(barcode).split("_")[-1].split("-")[0]


def load_metadata(path: Path) -> pd.DataFrame:
    """ECCITE metadata indexed by cell id (l{lane}_{16bp}), collisions dropped.

    A core barcode reused across lanes cannot be assigned to one cell under a pooled
    STARsolo run, so every cell sharing such a core is removed.
    """
    meta = pd.read_csv(path, sep="\t", index_col=0)
    meta.index = meta.index.astype(str)
    meta["core"] = [normalize_core(x) for x in meta.index]
    counts = meta["core"].value_counts()
    unique_cores = set(counts[counts == 1].index)
    dropped = len(meta) - len(unique_cores)
    meta = meta[meta["core"].isin(unique_cores)].copy()
    print(f"[papalexi] metadata: {len(meta)} cells kept, {dropped} dropped (cross-lane barcode collisions)")
    return meta


def load_adt_from_tar(tar_path: Path, member: str) -> pd.DataFrame:
    """ADT count matrix (proteins x cells) read straight from the GEO RAW tar."""
    with tarfile.open(tar_path) as tar:
        handle = tar.extractfile(member)
        if handle is None:
            raise FileNotFoundError(f"{member} not found in {tar_path}")
        with gzip.open(handle) as gz:
            df = pd.read_csv(gz, sep="\t", index_col=0)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    print(f"[papalexi] ADT matrix: {df.shape[0]} proteins x {df.shape[1]} cells ({member})")
    return df


def build_rna(velocyto_dir: Path, meta: pd.DataFrame) -> ad.AnnData:
    """RNA velocity input: STARsolo spliced/unspliced/ambiguous filtered to metadata cells."""
    barcodes = pd.read_csv(velocyto_dir / "barcodes.tsv", sep="\t", header=None)[0].astype(str)
    features = read_features(velocyto_dir / "features.tsv")
    spliced = read_layer(velocyto_dir / "spliced.mtx")
    unspliced = read_layer(velocyto_dir / "unspliced.mtx")
    ambiguous = read_layer(velocyto_dir / "ambiguous.mtx")
    expected_shape = (len(barcodes), len(features))
    validate_layer_shapes("papalexi", expected_shape, {
        "spliced": spliced,
        "unspliced": unspliced,
        "ambiguous": ambiguous,
    })

    core_to_cell = dict(zip(meta["core"], meta.index))
    cores = [normalize_core(b) for b in barcodes]
    keep = [i for i, c in enumerate(cores) if c in core_to_cell]
    cell_ids = [core_to_cell[cores[i]] for i in keep]
    print(f"[papalexi] RNA: {len(keep)}/{len(barcodes)} STARsolo barcodes matched to metadata cells")
    if not keep:
        raise ValueError("No STARsolo barcodes matched the metadata core barcodes.")

    keep = np.asarray(keep)
    spliced = spliced[keep].tocsr()
    unspliced = unspliced[keep].tocsr()
    ambiguous = ambiguous[keep].tocsr()

    var = features.copy()
    var.index = pd.Index(var["gene_name"].astype(str).to_numpy(), name=None)
    if not pd.Index(var.index).is_unique:
        duplicated = int(pd.Index(var.index).duplicated().sum())
        print(f"[papalexi] RNA gene symbols include {duplicated} duplicate feature(s); "
              "AnnData will make var_names unique while var['gene_name'] keeps the symbols")
    var["feature_types"] = "GEX"

    counts = (spliced + unspliced).tocsr()
    rna = ad.AnnData(X=counts.copy(), obs=meta.loc[cell_ids].copy(), var=var)
    rna.obs_names = cell_ids
    rna.layers["counts"] = counts
    rna.layers["spliced"] = spliced
    rna.layers["unspliced"] = unspliced
    rna.layers["ambiguous"] = ambiguous
    rna.var_names_make_unique()
    rna.obs.index.name = None
    rna.var.index.name = None
    return rna


def add_umap(rna: ad.AnnData) -> None:
    """Store an X_umap in rna.obsm — the h5ad velocity mode requires one for plotting.

    Computed on a normalized/log copy so rna.X and the raw `counts` layer stay untouched.
    """
    tmp = rna.copy()
    sc.pp.filter_genes(tmp, min_cells=3)
    sc.pp.normalize_total(tmp, target_sum=1e4)
    sc.pp.log1p(tmp)
    sc.pp.highly_variable_genes(tmp, n_top_genes=2000)
    tmp = tmp[:, tmp.var["highly_variable"]].copy()
    sc.pp.scale(tmp, max_value=10)
    sc.tl.pca(tmp, n_comps=30)
    sc.pp.neighbors(tmp, n_neighbors=15, n_pcs=30)
    sc.tl.umap(tmp)
    rna.obsm["X_umap"] = tmp.obsm["X_umap"]
    print(f"[papalexi] computed X_umap ({rna.obsm['X_umap'].shape})")


def build_protein(adt: pd.DataFrame, cell_ids) -> ad.AnnData:
    """Protein h5ad: 4 ADTs x matched cells, tagged feature_types=ADT for the loader."""
    cols = [c for c in cell_ids if c in adt.columns]
    sub = adt[cols].T  # cells x proteins
    protein = ad.AnnData(X=sparse.csr_matrix(sub.to_numpy().astype("float32")))
    protein.obs_names = cols
    protein.var_names = list(adt.index)
    protein.obs.index.name = None
    protein.var.index.name = None
    protein.var["feature_types"] = "ADT"
    print(f"[papalexi] protein: {protein.n_obs} cells x {protein.n_vars} ADTs")
    return protein


def write_mapping(path: Path, present_genes: set[str]) -> None:
    """ADT->gene mapping in the adt_mapping_*.csv schema used by the KOT pipeline."""
    present = {normalize_gene_symbol(g) for g in present_genes}
    rows = []
    for adt_name, gene in ADT_GENE_MAP.items():
        in_rna = normalize_gene_symbol(gene) in present
        rows.append({
            "adt_name": adt_name,
            "hgnc_id": "",
            "gene_symbol": gene,
            "uniprot_id": "",
            "mapping_type": "one_to_one",
            "present_in_rna": in_rna,
            "use_for_alignment": True,
            "use_for_kinetics": True,
            "excluded_because": "",
        })
    validate_mapping_records(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"[papalexi] wrote ADT-gene mapping: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Papalexi ECCITE RNA + protein h5ads.")
    parser.add_argument("--velocyto-dir", type=Path,
                        default=Path("Datasets/Papalexi_GSE153056/starsolo_eccite_pooled/Solo.out/Velocyto/raw"))
    parser.add_argument("--tar", type=Path, default=Path("data/papalexi/GSE153056_RAW.tar"))
    parser.add_argument("--metadata", type=Path, default=Path("data/papalexi/ECCITE_metadata.tsv"))
    parser.add_argument("--adt-member", default="GSM4633615_ECCITE_ADT_counts.tsv.gz")
    parser.add_argument("--rna-out", type=Path,
                        default=Path("Datasets/Papalexi_GSE153056/papalexi_rna_input.h5ad"))
    parser.add_argument("--protein-out", type=Path,
                        default=Path("Datasets/Papalexi_GSE153056/papalexi_protein.h5ad"))
    parser.add_argument("--mapping-out", type=Path,
                        default=Path("cache/results/mapping/adt_mapping_papalexi.csv"))
    args = parser.parse_args()

    meta = load_metadata(args.metadata)
    adt = load_adt_from_tar(args.tar, args.adt_member)

    rna = build_rna(args.velocyto_dir, meta)
    protein = build_protein(adt, list(rna.obs_names))

    common = rna.obs_names.intersection(protein.obs_names)
    print(f"[papalexi] RNA∩protein cells: {len(common)}")
    if len(common) == 0:
        raise ValueError("No cells shared between RNA and protein after matching.")
    rna = rna[common].copy()
    protein = protein[common].copy()
    rna.obs.index.name = None
    rna.var.index.name = None
    protein.obs.index.name = None
    protein.var.index.name = None

    add_umap(rna)

    args.rna_out.parent.mkdir(parents=True, exist_ok=True)
    args.protein_out.parent.mkdir(parents=True, exist_ok=True)
    write_h5ad_atomic(rna, args.rna_out)
    write_h5ad_atomic(protein, args.protein_out)
    print(f"[papalexi] wrote RNA velocity input: {args.rna_out}  {rna.shape}")
    print(f"[papalexi] wrote protein: {args.protein_out}  {protein.shape}")

    write_mapping(args.mapping_out, gene_alias_set(rna.var_names, rna.var))


if __name__ == "__main__":
    main()
