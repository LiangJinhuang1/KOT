"""Chromatin→RNA inputs: the paired Multiome objects the chromatin KOT reads.

Two datasets, one shape on disk, so every downstream stage is dataset-agnostic:

  hspc  GSE209878 (MultiVelo), mobilised CD34+ HSPCs, 10x Multiome.
        3423-MV-1 = day 0, 3423-MV-2 = day 7 — two SEPARATE runs, so the two days
        share no barcodes and were peak-called independently. The union-peak step
        below exists for exactly that reason: without it, day-0 and day-7 ATAC live
        in different coordinate systems and no OT cost between them means anything.

  bmmc  GSE194122 (NeurIPS 2021), 13 site×donor batches. Gene activity and LSI ship
        with the processed object, so nothing has to be recomputed for it.

Coordinate systems are kept apart on purpose, and this is the invariant the whole
experiment rests on:

  obsm["gene_activity"]  phi's INPUT and the space the JVP is taken in. Chromatin
                         velocity is a displacement in these coordinates.
  obsm["lsi"]            geometry ONLY — neighbours, the OT cost, pseudotime. Never
                         an input to phi and never a direction the JVP is read along.

Both matrices are written on ONE normalisation, so no later stage re-decides the units the
velocity was measured in — but "normalise it" is not unconditional. BMMC's gene activity
arrives already library-normalised and log1p'd from the NeurIPS organisers, HSPC's arrives
as raw peak counts, and applying the transform to both put BMMC in a log-of-a-log space
while HSPC sat in a log one. `already_normalised` asks the data which case it is.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.utils.extmath import randomized_svd

from src.utils.arrays import to_dense
from src.data.chromatin_map import (
    build_chromatin_projection,
    chromatin_map_rows,
    load_chromatin_map,
    write_chromatin_map,
)

# Columns of a cellranger-arc features.tsv: id, name, feature_type, chrom, start, end.
FEATURE_COLUMNS = ["feature_id", "feature_name", "feature_type", "chrom", "start", "end"]
GENE_FEATURE = "Gene Expression"
PEAK_FEATURE = "Peaks"

# Rows of the .mtx are parsed in blocks this size. 4e8 entries at 64-bit would not fit;
# blocks keep the parse flat in memory and cost one extra pass over a gzip stream.
MTX_BLOCK_ROWS = 20_000_000

HSPC_RUNS = {
    "d0": {
        "prefix": "GSE209878_3423-MV-1",
        "loom": "GSM6403408_3423-MV-1_gex_possorted_bam_0E7KE.loom",
        "peaks": "GSM6403409_3423-MV-1_atac_peak_annotation.tsv.gz",
        "day": 0,
    },
    "d7": {
        "prefix": "GSE209878_3423-MV-2",
        "loom": "GSM6403410_3423-MV-2_gex_possorted_bam_ICXFB.loom",
        "peaks": "GSM6403411_3423-MV-2_atac_peak_annotation.tsv.gz",
        "day": 7,
    },
}


def read_features(path: Path) -> pd.DataFrame:
    """The features.tsv of a cellranger-arc filtered matrix, in row order."""
    frame = pd.read_csv(path, sep="\t", header=None, names=FEATURE_COLUMNS, dtype=str)
    frame["start"] = pd.to_numeric(frame["start"], errors="coerce")
    frame["end"] = pd.to_numeric(frame["end"], errors="coerce")
    return frame


def merge_intervals(chrom: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Assign every interval to one merged interval of the union set.

    Overlapping intervals collapse into a single block, so each input interval lands in
    exactly one block and mapping counts onto the union is a plain sum — no interval is
    split and no count is double-assigned. Returns the block index per input row.
    """
    order = np.lexsort((start, chrom))
    block = np.empty(len(order), dtype=np.int64)
    current = -1
    current_chrom = None
    current_end = -1
    for position in order:
        same_chrom = chrom[position] == current_chrom
        if not same_chrom or start[position] > current_end:
            current += 1
            current_chrom = chrom[position]
            current_end = end[position]
        else:
            current_end = max(current_end, end[position])
        block[position] = current
    return block


def stream_mtx(path: Path, row_map: dict[str, tuple[np.ndarray, int]],
               n_cols: int) -> dict[str, sparse.csr_matrix]:
    """Read a MatrixMarket .mtx(.gz) once, folding rows into the requested groups.

    `row_map` gives, per output group, a length-n_rows array of destination indices (-1
    for "drop") AND the group's width. The width is explicit because the largest index a
    group actually USES is not its size: the last few hundred genes carry no peak, and
    sizing gene activity by its maximum used index would silently return a matrix
    narrower than the gene panel.

    A 4e8-entry matrix is never held whole: each block is reduced into its groups and
    discarded. Duplicated destinations (many peaks, one gene) sum, which is what gene
    activity is.
    """
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        header = handle.readline()
        assert header.startswith("%%MatrixMarket"), f"{path} is not MatrixMarket"
        line = handle.readline()
        while line.startswith("%"):
            line = handle.readline()
        n_rows, n_cells, n_entries = (int(value) for value in line.split())
        assert n_cells == n_cols, f"{path} has {n_cells} columns, expected {n_cols}"
        for indices, width in row_map.values():
            assert len(indices) == n_rows, f"{path} has {n_rows} rows, row_map has {len(indices)}"
            assert indices.max() < width, f"{path}: destination index exceeds width {width}"

        totals = {name: sparse.csr_matrix((n_cells, width), dtype=np.float32)
                  for name, (_, width) in row_map.items()}
        seen = 0
        for block in pd.read_csv(handle, sep=" ", header=None, chunksize=MTX_BLOCK_ROWS,
                                 names=["row", "col", "value"], dtype=np.int64):
            seen += len(block)
            source = block["row"].to_numpy() - 1
            cells = block["col"].to_numpy() - 1
            values = block["value"].to_numpy(dtype=np.float32)
            for name, (indices, width) in row_map.items():
                target = indices[source]
                keep = target >= 0
                totals[name] = totals[name] + sparse.coo_matrix(
                    (values[keep], (cells[keep], target[keep])),
                    shape=(n_cells, width),
                ).tocsr()
        assert seen == n_entries, f"{path}: read {seen} entries, header declared {n_entries}"

    return totals


def reindex_columns(matrix, source_names: pd.Index, target_names: pd.Index):
    """Put `matrix`'s columns on `target_names`, zero-filling names it does not carry.

    A selection matrix rather than fancy indexing into a lil: the assignment form is
    minutes on a 70k x 20k block and this is one sparse product.
    """
    position = pd.Series(np.arange(len(source_names)), index=source_names)
    columns = position.reindex(target_names).to_numpy()
    present = np.flatnonzero(~np.isnan(columns))
    mapper = sparse.csr_matrix(
        (np.ones(len(present), dtype=np.float32),
         (columns[present].astype(np.int64), present)),
        shape=(len(source_names), len(target_names)),
    )
    return (sparse.csr_matrix(matrix, dtype=np.float32) @ mapper).tocsr()


def peak_to_gene_rows(features: pd.DataFrame, annotation: pd.DataFrame,
                      genes: pd.Index) -> np.ndarray:
    """Destination gene index for every feature row, -1 where the row is not a peak.

    Gene activity is the Signac definition: a gene's promoter and gene-body peaks.
    cellranger-arc reports distance 0 for a peak overlapping the gene body, so the 2 kb
    window keeps those and the promoter, and drops the far distal peaks — those are the
    links a chromatin→RNA map should have to LEARN, not be handed by annotation.
    """
    linked = annotation[annotation["peak_type"].isin(["promoter", "distal"])]
    linked = linked[linked["distance"].abs() <= 2000]
    gene_position = pd.Series(np.arange(len(genes)), index=genes)
    peak_name = (linked["chrom"].astype(str) + ":"
                 + linked["start"].astype(str) + "-" + linked["end"].astype(str))
    target = gene_position.reindex(linked["gene"].astype(str)).to_numpy()
    usable = ~np.isnan(target)
    peak_target = pd.Series(target[usable], index=peak_name.to_numpy()[usable])
    peak_target = peak_target.groupby(level=0).first()

    rows = np.full(len(features), -1, dtype=np.int64)
    peak_rows = np.flatnonzero(features["feature_type"].to_numpy() == PEAK_FEATURE)
    mapped = peak_target.reindex(features["feature_name"].to_numpy()[peak_rows])
    rows[peak_rows] = np.nan_to_num(mapped.to_numpy(), nan=-1.0).astype(np.int64)
    return rows


def normalize_log(matrix, target_sum: float = 1e4) -> np.ndarray:
    """Per-cell library-size normalisation then log1p, as a dense float32 matrix.

    Both modalities go through the same transform so that a displacement in gene-activity
    space and a displacement in RNA space are read in comparable units.
    """
    counts = sparse.csr_matrix(matrix, dtype=np.float32)
    totals = np.asarray(counts.sum(axis=1)).ravel()
    totals[totals == 0] = 1.0
    scaled = sparse.diags(target_sum / totals) @ counts
    return np.log1p(scaled.toarray()).astype(np.float32)


def lsi_embedding(peaks: sparse.csr_matrix, n_components: int, seed: int) -> np.ndarray:
    """TF-IDF then truncated SVD, first component dropped.

    Component 1 of an ATAC LSI is sequencing depth, not biology; keeping it would make
    the OT cost and the pseudotime root partly a depth ranking.
    """
    binary = peaks.copy()
    binary.data = np.ones_like(binary.data, dtype=np.float32)
    cell_totals = np.asarray(binary.sum(axis=1)).ravel()
    cell_totals[cell_totals == 0] = 1.0
    peak_totals = np.asarray(binary.sum(axis=0)).ravel()
    idf = np.log(1.0 + binary.shape[0] / np.maximum(peak_totals, 1.0))
    tfidf = sparse.diags(1.0 / cell_totals) @ binary @ sparse.diags(idf)
    tfidf.data = np.log1p(tfidf.data * 1e4)
    left, values, _ = randomized_svd(tfidf.tocsr(), n_components=n_components,
                                     random_state=seed)
    return (left * values)[:, 1:].astype(np.float32)


def align_loom_to_barcodes(loom_path: Path, barcodes: pd.Index,
                           genes: pd.Index) -> dict[str, sparse.csr_matrix]:
    """Spliced/unspliced from a velocyto loom, on the filtered matrix's cells and genes.

    velocyto writes `<bam tag>:<16-mer>x`; the ARC matrix writes `<16-mer>-1`. The core
    barcode is the only part both agree on.
    """
    loom = ad.read_loom(loom_path, sparse=True, var_names="Gene", obs_names="CellID",
                        cleanup=False)
    loom.var_names_make_unique()
    core = pd.Index([name.split(":")[-1].rstrip("x") for name in loom.obs_names])
    loom.obs_names = core
    wanted_cells = pd.Index([name.split("-")[0] for name in barcodes])
    assert wanted_cells.isin(core).all(), (
        f"{loom_path.name}: {int((~wanted_cells.isin(core)).sum())} filtered cells absent "
        "from the loom")
    subset = loom[wanted_cells.to_numpy(), :]
    return {name: reindex_columns(subset.layers[name], subset.var_names, genes)
            for name in ["spliced", "unspliced"]}


def build_hspc(source_dir: Path, output_path: Path, n_top_genes: int, seed: int,
               n_lsi: int, min_peak_cells: int, map_path: Path) -> ad.AnnData:
    """Assemble day 0 + day 7 into one object with shared gene and peak coordinates."""
    per_run = []
    peak_frames = []
    for run_name, spec in HSPC_RUNS.items():
        prefix = source_dir / spec["prefix"]
        features = read_features(Path(f"{prefix}_features.tsv.gz"))
        barcodes = pd.Index(pd.read_csv(f"{prefix}_barcodes.tsv.gz", header=None)[0].astype(str))
        annotation = pd.read_csv(source_dir / spec["peaks"], sep="\t")
        genes = pd.Index(features.loc[features["feature_type"] == GENE_FEATURE,
                                      "feature_name"].to_numpy())
        genes = pd.Index(pd.Series(genes).drop_duplicates().to_numpy())

        gene_rows = np.full(len(features), -1, dtype=np.int64)
        gene_mask = features["feature_type"].to_numpy() == GENE_FEATURE
        gene_position = pd.Series(np.arange(len(genes)), index=genes)
        gene_rows[gene_mask] = gene_position.reindex(
            features["feature_name"].to_numpy()[gene_mask]).to_numpy()
        activity_rows = peak_to_gene_rows(features, annotation, genes)
        peak_mask = features["feature_type"].to_numpy() == PEAK_FEATURE
        peak_rows = np.full(len(features), -1, dtype=np.int64)
        peak_rows[peak_mask] = np.arange(int(peak_mask.sum()))

        print(f"[chromatin] {run_name}: {len(barcodes)} cells, {len(genes)} genes, "
              f"{int(peak_mask.sum())} peaks, "
              f"{int((activity_rows >= 0).sum())} peaks linked to a gene")
        blocks = stream_mtx(
            Path(f"{prefix}_matrix.mtx.gz"),
            {"rna": (gene_rows, len(genes)),
             "activity": (activity_rows, len(genes)),
             "peaks": (peak_rows, int(peak_mask.sum()))},
            n_cols=len(barcodes),
        )
        loom_layers = align_loom_to_barcodes(source_dir / spec["loom"], barcodes, genes)

        obs = pd.DataFrame(
            {"day": spec["day"], "run": run_name},
            index=pd.Index([f"{barcode}-{run_name}" for barcode in barcodes]),
        )
        per_run.append({
            "obs": obs, "genes": genes, "rna": blocks["rna"], "activity": blocks["activity"],
            "peaks": blocks["peaks"], "layers": loom_layers,
            "peaks_per_gene": np.bincount(activity_rows[activity_rows >= 0],
                                          minlength=len(genes)),
        })
        peak_frames.append(features.loc[peak_mask])

    assert per_run[0]["genes"].equals(per_run[1]["genes"]), \
        "the two HSPC runs use different gene references"
    genes = per_run[0]["genes"]

    all_peaks = pd.concat(peak_frames, ignore_index=True)
    block = merge_intervals(all_peaks["chrom"].to_numpy().astype(str),
                            all_peaks["start"].to_numpy(), all_peaks["end"].to_numpy())
    print(f"[chromatin] union peaks: {block.max() + 1} from {len(all_peaks)} per-run peaks")
    offset = 0
    union_parts = []
    for part in per_run:
        width = part["peaks"].shape[1]
        target = block[offset:offset + width]
        offset += width
        mapper = sparse.csr_matrix(
            (np.ones(width, dtype=np.float32), (np.arange(width), target)),
            shape=(width, block.max() + 1),
        )
        union_parts.append(part["peaks"] @ mapper)

    obs = pd.concat([part["obs"] for part in per_run])
    rna = sparse.vstack([part["rna"] for part in per_run]).tocsr()
    activity = sparse.vstack([part["activity"] for part in per_run]).tocsr()
    union_peaks = sparse.vstack(union_parts).tocsr()
    spliced = sparse.vstack([part["layers"]["spliced"] for part in per_run]).tocsr()
    unspliced = sparse.vstack([part["layers"]["unspliced"] for part in per_run]).tocsr()

    prevalence = np.asarray((union_peaks > 0).sum(axis=0)).ravel()
    keep_peaks = prevalence >= min_peak_cells
    print(f"[chromatin] LSI on {int(keep_peaks.sum())} peaks seen in >= {min_peak_cells} cells")

    adata = ad.AnnData(X=rna, obs=obs, var=pd.DataFrame(index=genes))
    adata.layers["counts"] = rna
    adata.layers["spliced"] = spliced
    adata.layers["unspliced"] = unspliced
    adata.obsm["gene_activity_counts"] = activity
    adata.obsm["lsi"] = lsi_embedding(union_peaks[:, keep_peaks], n_lsi, seed)
    adata.uns["gene_activity_names"] = genes.to_numpy()
    activity_total = np.asarray(activity.sum(axis=0)).ravel()
    adata.uns["chromatin_source"] = {
        "dataset": "hspc",
        "n_atac_features": int(block.max() + 1),
        "n_atac_features_lsi": int(keep_peaks.sum()),
        "n_rna_genes_full": int(len(genes)),
        "n_overlap_activity_rna": int((activity_total > 0).sum()),
    }
    peaks_per_gene = sum(part["peaks_per_gene"] for part in per_run)
    attach_optional_cell_types(adata, source_dir / "cell_types.csv")
    return finalize_panel(adata, n_top_genes, output_path, peaks_per_gene,
                          "cellranger_arc promoter+gene_body <=2kb", map_path)


def build_bmmc(processed_path: Path, velocity_path: Path | None, output_path: Path,
               n_top_genes: int, map_path: Path) -> ad.AnnData:
    """Assemble BMMC from the processed object, whose gene activity and LSI already ship.

    `velocity_path` is the scVelo cache built from the STARsolo velocyto output. It
    covers only the cells STARsolo was run for, so spliced/unspliced are written for
    those cells and left absent elsewhere — the relay law's own audit refuses the
    dataset when they are missing, rather than this builder inventing them.
    """
    processed = sc.read_h5ad(processed_path)
    gex = processed[:, processed.var["feature_types"].astype(str) == "GEX"].copy()
    genes = pd.Index(gex.var_names.astype(str))
    activity_names = pd.Index(np.asarray(processed.uns["ATAC_gene_activity_var_names"]).astype(str))

    shared = genes.intersection(activity_names)
    print(f"[chromatin] bmmc: {gex.n_obs} cells, {len(genes)} RNA genes, "
          f"{len(activity_names)} activity features, {len(shared)} shared")

    adata = ad.AnnData(
        X=sparse.csr_matrix(gex.layers["counts"]),
        obs=gex.obs.copy(),
        var=pd.DataFrame(index=genes),
    )
    adata.layers["counts"] = adata.X
    adata.obsm["gene_activity_counts"] = reindex_columns(
        processed.obsm["ATAC_gene_activity"], activity_names, genes)
    adata.obsm["lsi"] = select_bmmc_lsi(processed)
    adata.uns["gene_activity_names"] = genes.to_numpy()
    adata.uns["chromatin_source"] = {
        "dataset": "bmmc",
        "n_atac_features": int((processed.var["feature_types"].astype(str) == "ATAC").sum()),
        "n_rna_genes_full": int(len(genes)),
        "n_overlap_activity_rna": int(len(shared)),
    }

    if velocity_path is not None and Path(velocity_path).exists():
        velocity = sc.read_h5ad(velocity_path)
        attach_splicing(adata, velocity)
    # The NeurIPS object ships gene activity already built, so the peaks behind each gene
    # are not recoverable here — recorded as unknown rather than guessed.
    return finalize_panel(adata, n_top_genes, output_path, None,
                          "neurips_signac_prebuilt", map_path)


def select_bmmc_lsi(processed: ad.AnnData) -> np.ndarray:
    """ATAC LSI with the sequencing-depth component removed.

    NeurIPS ships `ATAC_lsi_red` (depth already dropped) and `ATAC_lsi_full` (component 1
    is depth). HSPC's own LSI drops that component for the same reason: keeping it makes
    the OT cost and the pseudotime a depth ranking. Prefer the reduced matrix; if a
    downstream object only has the full one, drop column 0 here.
    """
    if "ATAC_lsi_red" in processed.obsm:
        lsi = np.asarray(processed.obsm["ATAC_lsi_red"], dtype=np.float32)
        source = "ATAC_lsi_red"
    else:
        full = np.asarray(processed.obsm["ATAC_lsi_full"], dtype=np.float32)
        assert full.shape[1] > 1, "ATAC LSI has no component to drop"
        print("[chromatin] ATAC_lsi_red absent — dropping component 0 of ATAC_lsi_full (depth)")
        lsi, source = full[:, 1:].astype(np.float32), "ATAC_lsi_full[:, 1:]"
    n_nan = int((~np.isfinite(lsi).all(axis=1)).sum())
    print(f"[chromatin] BMMC LSI from {source}: {lsi.shape[1]} components, "
          f"{n_nan}/{len(lsi)} cells have NaN (dropped from the velocity graph later)")
    return lsi


def attach_optional_cell_types(adata: ad.AnnData, path: Path) -> None:
    """Join a barcode → cell_type table if the caller dropped one next to the raw files.

    GEO's HSPC matrices have no cell-type column, so Task B/C cell-type metrics stay
    undefined until this file exists. Matching tries the AnnData names first, then the
    run suffix stripped (`AAAC...-1-d0` → `AAAC...-1`).
    """
    if not path.exists():
        return
    frame = pd.read_csv(path)
    assert "cell_type" in frame.columns, f"{path} needs a cell_type column"
    key = "barcode" if "barcode" in frame.columns else frame.columns[0]
    mapping = pd.Series(frame["cell_type"].astype(str).to_numpy(), index=frame[key].astype(str))
    labelled = mapping.reindex(adata.obs_names.astype(str))
    if labelled.isna().all():
        stripped = pd.Index([name.rsplit("-", 1)[0] for name in adata.obs_names.astype(str)])
        labelled = pd.Series(mapping.reindex(stripped).to_numpy(), index=adata.obs_names)
    n_matched = int(labelled.notna().sum())
    assert n_matched > 0, f"{path} matched no barcodes in this object"
    adata.obs["cell_type"] = labelled
    print(f"[chromatin] cell_type: {n_matched}/{adata.n_obs} cells labelled from {path}")


def attach_splicing(adata: ad.AnnData, velocity: ad.AnnData) -> None:
    """Write spliced/unspliced for the cells and genes the velocity cache covers.

    Cells outside the cache keep zero rows and are flagged `has_splicing = False`, so the
    relay law can select on the flag instead of guessing which zeros are real.
    """
    rows = adata.obs_names.get_indexer(velocity.obs_names)
    covered = rows >= 0
    print(f"[chromatin] splicing: {int(covered.sum())}/{adata.n_obs} cells covered by "
          f"{velocity.n_vars} velocity genes")
    lifter = sparse.csr_matrix(
        (np.ones(int(covered.sum()), dtype=np.float32),
         (rows[covered], np.flatnonzero(covered))),
        shape=(adata.n_obs, velocity.n_obs),
    )
    for name in ["spliced", "unspliced"]:
        on_genes = reindex_columns(velocity.layers[name], velocity.var_names, adata.var_names)
        adata.layers[name] = (lifter @ on_genes).tocsr()
    adata.obs["has_splicing"] = False
    adata.obs.iloc[rows[covered], adata.obs.columns.get_loc("has_splicing")] = True


def already_normalised(matrix) -> bool:
    """True when a matrix holds normalised values rather than raw counts.

    The NeurIPS BMMC object ships `ATAC_gene_activity` ALREADY library-normalised and
    log1p'd — 70% of its non-zeros lie strictly between 0 and 1 and the maximum is 4.7.
    Running normalize_total + log1p over that again puts phi's input, `G c` and the
    chromatin velocity into a log-of-a-log space whose "library size" is a sum of logs,
    and leaves the two datasets on different transforms so nothing cross-dataset compares.
    HSPC's activity is genuine integer peak counts and does need the transform, so the
    question is asked of the data rather than hard-coded per dataset.
    """
    values = sparse.csr_matrix(matrix).data
    if len(values) == 0:
        return False
    return not np.allclose(values, np.round(values))


def finalize_panel(adata: ad.AnnData, n_top_genes: int, output_path: Path,
                   peaks_per_gene: np.ndarray | None, peak_definition: str,
                   map_path: Path) -> ad.AnnData:
    """Decide the panel, write the gene map that records the decision, and save.

    The panel is every highly variable expressed gene — NOT only the ones chromatin
    supports. Genes without chromatin support stay in phi's output and in Task A, and are
    kept out of the kinetics term by the map's `use_for_kinetics` flag, exactly as a
    protein with no curated gene stays in the RNA→protein alignment and out of its ODE.
    """
    counts = adata.copy()
    sc.pp.normalize_total(counts, target_sum=1e4)
    sc.pp.log1p(counts)
    sc.pp.highly_variable_genes(counts, n_top_genes=n_top_genes)

    activity = adata.obsm["gene_activity_counts"]
    activity_total = np.asarray(activity.sum(axis=0)).ravel()
    activity_detected = np.asarray((activity > 0).sum(axis=0)).ravel() / adata.n_obs
    rna = sparse.csr_matrix(adata.layers["counts"])
    rna_total = np.asarray(rna.sum(axis=0)).ravel()
    rna_detected = np.asarray((rna > 0).sum(axis=0)).ravel() / adata.n_obs
    has_us = "spliced" in adata.layers and "unspliced" in adata.layers
    usable_us = (usable_splicing_genes(adata) if has_us
                 else np.zeros(adata.n_vars, dtype=bool))

    rows = chromatin_map_rows(adata.var_names, activity_total, activity_detected,
                              rna_total, rna_detected,
                              counts.var["highly_variable"].to_numpy(), usable_us,
                              peaks_per_gene, peak_definition)
    write_chromatin_map(rows, map_path)
    usable = np.array([row["use_for_alignment"] for row in rows])
    kinetic = np.array([row["use_for_kinetics"] for row in rows])
    print(f"[chromatin] panel: {int(usable.sum())} genes for alignment, "
          f"{int(kinetic.sum())} with chromatin support for kinetics, "
          f"{int((usable & usable_us).sum())} with usable u/s; wrote {map_path}")

    panel = adata[:, usable].copy()
    panel.obsm["gene_activity_counts"] = adata.obsm["gene_activity_counts"][:, usable]
    pre_normalised = already_normalised(panel.obsm["gene_activity_counts"])
    print(f"[chromatin] gene activity arrives "
          f"{'already normalised — transformed once, not twice' if pre_normalised else 'as raw counts — normalising'}")
    panel.obsm["gene_activity"] = (
        to_dense(panel.obsm["gene_activity_counts"], np.float32) if pre_normalised
        else normalize_log(panel.obsm["gene_activity_counts"]))
    panel.layers["rna_lognorm"] = sparse.csr_matrix(normalize_log(panel.layers["counts"]))
    for name in ["spliced", "unspliced"]:
        if name in panel.layers:
            panel.layers[f"{name}_lognorm"] = sparse.csr_matrix(normalize_log(panel.layers[name]))
    panel.uns["gene_activity_names"] = panel.var_names.to_numpy()
    panel.uns["chromatin_map_csv"] = str(map_path)
    panel.uns["gene_activity_pre_normalised"] = bool(pre_normalised)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.write_h5ad(output_path)
    print(f"[chromatin] wrote {output_path}: {panel.n_obs} cells x {panel.n_vars} genes")
    return panel


def audit_report(adata: ad.AnnData) -> dict:
    """Everything §3 asks to see before a single parameter is fitted.

    Deliberately reads the built object rather than the raw files: what the audit
    describes is then exactly what training will read, not what the builder intended.
    """
    source = dict(adata.uns["chromatin_source"])
    has_spliced = "spliced" in adata.layers
    has_unspliced = "unspliced" in adata.layers
    has_both_splicing_layers = has_spliced and has_unspliced
    report = {
        "dataset": source["dataset"],
        "n_paired_cells": int(adata.n_obs),
        "n_atac_features": int(source["n_atac_features"]),
        "n_rna_genes_full": int(source["n_rna_genes_full"]),
        "n_overlap_activity_rna": int(source["n_overlap_activity_rna"]),
        "n_panel_genes": int(adata.n_vars),
        "gene_activity_dim": int(adata.obsm["gene_activity"].shape[1]),
        "lsi_dim": int(adata.obsm["lsi"].shape[1]),
        "has_spliced_layer": has_spliced,
        "has_unspliced_layer": has_unspliced,
        "n_genes_usable_us": (int(usable_splicing_genes(adata).sum())
                               if has_both_splicing_layers else 0),
        "n_cells_with_splicing": int(adata.obs["has_splicing"].sum())
                                 if "has_splicing" in adata.obs
                                 else int(adata.n_obs * has_both_splicing_layers),
        "has_cell_type": "cell_type" in adata.obs,
        "n_cells_with_cell_type": int(adata.obs["cell_type"].notna().sum())
                                  if "cell_type" in adata.obs else 0,
    }
    if "day" in adata.obs:
        report["cells_per_day"] = adata.obs["day"].value_counts().sort_index().to_dict()
    for column in ["Site", "DonorID", "batch", "cell_type"]:
        if column in adata.obs:
            report[f"cells_per_{column}"] = adata.obs[column].value_counts().to_dict()
    return report


def usable_splicing_genes(adata: ad.AnnData, min_shared_counts: int = 20,
                          rows: np.ndarray | None = None) -> np.ndarray:
    """Genes carrying enough spliced AND unspliced signal for the relay law to fit.

    scVelo's own gene filter: a gene needs both layers, so a gene with unspliced counts
    nowhere gives the u-block nothing to predict and would only add a constant column.
    When `rows` is supplied, eligibility is learned from the RNA training population only;
    validation and test counts must not decide the relay model's output panel.
    """
    spliced_matrix = sparse.csr_matrix(adata.layers["spliced"])
    unspliced_matrix = sparse.csr_matrix(adata.layers["unspliced"])
    if rows is not None:
        spliced_matrix = spliced_matrix[rows]
        unspliced_matrix = unspliced_matrix[rows]
    spliced = np.asarray(spliced_matrix.sum(axis=0)).ravel()
    unspliced = np.asarray(unspliced_matrix.sum(axis=0)).ravel()
    return (spliced >= min_shared_counts) & (unspliced >= min_shared_counts)


def require_relay_inputs(adata: ad.AnnData) -> None:
    """Hard fail when the relay law is asked for and the counts it needs are absent."""
    missing = [name for name in ["spliced", "unspliced"] if name not in adata.layers]
    if missing:
        raise ValueError(
            f"--law relay needs {missing} counts and this dataset has none. Build them "
            "(velocyto/STARsolo) rather than approximating them.")
    usable = int(usable_splicing_genes(adata).sum())
    if usable == 0:
        raise ValueError("--law relay: no panel gene has usable spliced AND unspliced counts")


def stratify_labels(adata: ad.AnnData) -> np.ndarray:
    """The strata the 70/10/20 split is balanced on.

    HSPC splits by day. BMMC splits by batch crossed with cell type where that leaves a
    stratum big enough to divide; rare types fall back to batch alone so a stratum of two
    cells does not decide a whole fold.
    """
    if "day" in adata.obs:
        return adata.obs["day"].astype(str).to_numpy()
    # BMMC donor/site are nuisance strata, never time. Prefer the two explicit columns;
    # `batch` is only a compatibility fallback for older processed objects where it is
    # the already-combined site×donor label.
    if {"Site", "DonorID"}.issubset(adata.obs.columns):
        batch = adata.obs["Site"].astype(str) + "|" + adata.obs["DonorID"].astype(str)
    elif "batch" in adata.obs:
        batch = adata.obs["batch"].astype(str)
    else:
        raise ValueError("BMMC splitting needs Site+DonorID or a combined batch column")
    if "cell_type" not in adata.obs:
        return batch.to_numpy()
    population = adata.obs["cell_type"].fillna("unlabelled").astype(str)
    crossed = batch + "|" + population
    counts = crossed.value_counts()
    rare = counts.index[counts < 10]
    return crossed.where(~crossed.isin(rare), batch).to_numpy()


def make_splits(adata: ad.AnnData, seed: int, val_fraction: float,
                test_fraction: float) -> pd.DataFrame:
    """The one split every method and ablation reuses, plus the disjoint source/target halves.

    Two separate jobs, in this order. First a stratified 70/10/20 over PAIRED cells,
    because Tasks A and B need the pairing to score against. Then the training pairing is
    destroyed: the 70% is cut in half by barcode, one half contributing ATAC only and the
    other RNA only, so KOT never sees both modalities of one training cell. The halving is
    stratified too — otherwise the ATAC half and the RNA half are different populations and
    the alignment has a batch effect to explain rather than a mapping to learn.
    """
    if not (0.0 <= val_fraction < 1.0 and 0.0 <= test_fraction < 1.0
            and val_fraction + test_fraction < 1.0):
        raise ValueError("val_fraction and test_fraction must be non-negative and sum to < 1")
    rng = np.random.default_rng(seed)
    strata = stratify_labels(adata)
    split = np.empty(adata.n_obs, dtype=object)
    side = np.full(adata.n_obs, "", dtype=object)
    for level in np.unique(strata):
        members = np.flatnonzero(strata == level)
        order = rng.permutation(members)
        n_test = int(round(test_fraction * len(order)))
        n_val = int(round(val_fraction * len(order)))
        split[order[:n_test]] = "test"
        split[order[n_test:n_test + n_val]] = "val"
        train = order[n_test + n_val:]
        split[train] = "train"
        half = len(train) // 2
        side[train[:half]] = "atac"
        side[train[half:]] = "rna"

    assert np.isin(split, ["train", "val", "test"]).all(), "some cells were not split"
    frame = pd.DataFrame({"split": split, "train_side": side}, index=adata.obs_names)
    atac_train = set(frame.index[(frame["split"] == "train") & (frame["train_side"] == "atac")])
    rna_train = set(frame.index[(frame["split"] == "train") & (frame["train_side"] == "rna")])
    assert len(atac_train & rna_train) == 0, "ATAC and RNA training sets share a cell"
    return frame


def gene_map(adata: ad.AnnData, map_path: Path
             ) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    """G plus its two masks, read from the gene-map CSV this dataset was built with."""
    return build_chromatin_projection(
        load_chromatin_map(map_path), adata.var_names,
        pd.Index(np.asarray(adata.uns["gene_activity_names"]).astype(str)))
