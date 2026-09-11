"""Chromatin→RNA G, built the way `src/data/projection.py` builds the RNA→protein S.

G occupies exactly the slot S occupies in the law:

    RNA→protein   J_phi(r) v_r ~ kappa(r) [alpha(r) (*) S r - beta  (*) phi(r)]
    chromatin→RNA J_phi(c) v_c ~ kappa(c) [alpha(c) (*) G c - gamma (*) phi(c)]

so it is given the same structure, for the same reasons:

  A CSV IS THE SOURCE OF TRUTH.  Which genes take part is decided once, written down, and
  read back — not recomputed inside a training run where nobody can see it.

  TWO INDEPENDENT MASKS.  `use_for_alignment` and `use_for_kinetics` are separate. A gene
  with RNA signal but no chromatin support is still predicted by phi and still scored in
  Task A; it simply cannot appear in a transcription law that has no chromatin to
  transcribe. Collapsing the two would silently shrink the panel to the kinetic subset.

  A ZERO ROW, NEVER A FALLBACK.  A gene excluded from kinetics gets an all-zero G row, and
  `assert_no_orphan_g_rows` refuses the alternative. A uniform or nearest-gene fallback
  would put invented production into the residual and it would look like signal.

What the two tables catch differs, and it is worth being plain about it. S encodes
knowledge a person had to look up (PDL1 is read off CD274). G's correspondence is name
identity, so the pairing is not where the risk lives — the risk is silent gene loss, and
the fact that the two datasets do not build gene activity the same way: BMMC's ships from
the NeurIPS organisers (Signac), HSPC's is aggregated here from cellranger-arc promoter and
gene-body peaks. `peak_definition` and `n_peaks` put that difference in the file instead of
in a docstring.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

CHROMATIN_MAP_COLUMNS = [
    "gene", "activity_feature", "mapping_type", "n_peaks", "peak_definition",
    "activity_total", "activity_detected_frac", "rna_total", "rna_detected_frac",
    "highly_variable", "has_usable_us", "in_G", "use_for_alignment", "use_for_kinetics",
    "excluded_because",
]

IDENTITY = "identity"
NO_CHROMATIN = "no_chromatin_support"


def chromatin_map_rows(genes: pd.Index, activity_total: np.ndarray,
                       activity_detected: np.ndarray, rna_total: np.ndarray,
                       rna_detected: np.ndarray, highly_variable: np.ndarray,
                       usable_us: np.ndarray, peaks_per_gene: np.ndarray | None,
                       peak_definition: str) -> list[dict]:
    """One row per candidate gene, with the decision and the reason for it.

    Alignment takes every highly variable gene that is actually expressed — phi predicts
    it whether or not chromatin explains it, which is what makes Task A a real test rather
    than a test restricted to the genes the law already covers. Kinetics additionally
    requires chromatin support, because that is the term's input.
    """
    rows = []
    for position, gene in enumerate(genes):
        has_chromatin = activity_total[position] > 0
        has_rna = rna_total[position] > 0
        align = bool(highly_variable[position] and has_rna)
        kinetic = bool(align and has_chromatin)
        if not has_rna:
            reason = "no RNA counts"
        elif not highly_variable[position]:
            reason = "not highly variable"
        elif not has_chromatin:
            reason = "no chromatin support in the gene-activity matrix"
        else:
            reason = ""
        rows.append({
            "gene": str(gene),
            "activity_feature": str(gene) if has_chromatin else "",
            "mapping_type": IDENTITY if has_chromatin else NO_CHROMATIN,
            "n_peaks": int(peaks_per_gene[position]) if peaks_per_gene is not None else -1,
            "peak_definition": peak_definition,
            "activity_total": float(activity_total[position]),
            "activity_detected_frac": float(activity_detected[position]),
            "rna_total": float(rna_total[position]),
            "rna_detected_frac": float(rna_detected[position]),
            "highly_variable": bool(highly_variable[position]),
            "has_usable_us": bool(usable_us[position]),
            "in_G": kinetic,
            "use_for_alignment": align,
            "use_for_kinetics": kinetic,
            "excluded_because": reason,
        })
    return rows


def write_chromatin_map(rows: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=CHROMATIN_MAP_COLUMNS).to_csv(path, index=False)
    return path


def load_chromatin_map(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    missing = set(CHROMATIN_MAP_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    for column in ["highly_variable", "has_usable_us", "in_G", "use_for_alignment",
                   "use_for_kinetics"]:
        frame[column] = frame[column].astype(str).str.lower().isin(["true", "1"])
    return frame


def assert_no_orphan_g_rows(matrix: sparse.csr_matrix, kinetic_mask: np.ndarray) -> None:
    """A gene excluded from kinetics must have an all-zero G row — no fallback."""
    populated = np.asarray((matrix != 0).sum(axis=1)).ravel() > 0
    orphan = populated & ~kinetic_mask
    if orphan.any():
        raise ValueError(
            f"{int(orphan.sum())} gene(s) have a non-zero G row but use_for_kinetics=False")


def build_chromatin_projection(frame: pd.DataFrame, panel_genes: pd.Index,
                               activity_names: pd.Index, require_full_panel: bool = True
                               ) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    """(G, alignment_mask, kinetic_mask) for the panel this run actually loaded.

    G is (n_panel_genes x n_activity_features) — a gene's row selects the chromatin
    feature that drives it, and is empty when nothing does.
    """
    records = frame.set_index("gene")
    missing = [gene for gene in panel_genes if gene not in records.index]
    if missing and require_full_panel:
        raise ValueError(
            f"{len(missing)} panel gene(s) are absent from the chromatin map: "
            f"{missing[:10]}{' ...' if len(missing) > 10 else ''}. Rebuild it for this exact "
            "panel (tools/build_inputs.py chromatin-gene-map).")

    activity_position = pd.Series(np.arange(len(activity_names)), index=activity_names)
    alignment_mask = np.zeros(len(panel_genes), dtype=bool)
    kinetic_mask = np.zeros(len(panel_genes), dtype=bool)
    rows, columns = [], []
    for position, gene in enumerate(panel_genes):
        record = records.loc[gene]
        alignment_mask[position] = bool(record["use_for_alignment"])
        feature = record["activity_feature"]
        if not (record["use_for_kinetics"] and feature in activity_position.index):
            continue
        kinetic_mask[position] = True
        rows.append(position)
        columns.append(int(activity_position[feature]))

    matrix = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, columns)),
        shape=(len(panel_genes), len(activity_names)),
    )
    assert_no_orphan_g_rows(matrix, kinetic_mask)
    print(f"[chromatin-map] G {matrix.shape}, {matrix.nnz} links: "
          f"alignment {int(alignment_mask.sum())}/{len(panel_genes)}, "
          f"kinetics {int(kinetic_mask.sum())}/{len(panel_genes)}")
    return matrix, alignment_mask, kinetic_mask


def permute_chromatin_projection(matrix: sparse.csr_matrix, seed: int) -> sparse.csr_matrix:
    """permG: shuffle which chromatin feature feeds which gene, and nothing else.

    Shape, both dimensionalities, the non-zero count and which genes have a row all
    survive; only the identity behind each link is destroyed. Permuting the COLUMNS does
    that — permuting rows would move the empty rows too, and a gene excluded from kinetics
    would acquire production out of nowhere.
    """
    rng = np.random.default_rng(seed)
    permuted = matrix[:, rng.permutation(matrix.shape[1])].tocsr()
    assert permuted.shape == matrix.shape and permuted.nnz == matrix.nnz
    populated = np.asarray((permuted != 0).sum(axis=1)).ravel()
    original = np.asarray((matrix != 0).sum(axis=1)).ravel()
    assert np.array_equal(populated > 0, original > 0), "permG moved which genes have a row"
    return permuted


GENE_MAP_MODES = ["curated", "diagonal_full", "coaccess", "genomic", "peak-genomic"]


def gene_loci_from_features(path: Path) -> pd.DataFrame:
    """Gene intervals from a cellranger-arc features.tsv, indexed by gene name.

    HSPC's features file is GRCh38, so BMMC can reuse it as a symbol lookup. Genes
    absent from the table keep their curated G row and pick up no genomic neighbours.
    """
    frame = pd.read_csv(
        path, sep="\t", header=None,
        names=["feature_id", "feature_name", "feature_type", "chrom", "start", "end"],
        dtype={"feature_id": str, "feature_name": str, "feature_type": str, "chrom": str})
    genes = frame.loc[frame["feature_type"] == "Gene Expression"].copy()
    genes["start"] = pd.to_numeric(genes["start"], errors="coerce")
    genes["end"] = pd.to_numeric(genes["end"], errors="coerce")
    genes = genes.dropna(subset=["start", "end", "chrom"])
    genes = genes.drop_duplicates("feature_name", keep="first")
    return genes.set_index("feature_name")[["chrom", "start", "end"]]


def row_normalise_links(rows, columns, data, shape) -> sparse.csr_matrix:
    """One weighted-mean row per gene so degree cannot masquerade as production."""
    matrix = sparse.csr_matrix(
        (np.asarray(data, dtype=np.float32),
         (np.asarray(rows, dtype=np.int64), np.asarray(columns, dtype=np.int64))),
        shape=shape)
    totals = np.asarray(matrix.sum(axis=1)).ravel()
    inv = np.zeros_like(totals, dtype=np.float32)
    live = totals > 0
    inv[live] = 1.0 / totals[live]
    return sparse.diags(inv) @ matrix


def genomic_projection(projection, output_genes: pd.Index, feature_names: pd.Index,
                       loci: pd.DataFrame, max_bp: float, decay_bp: float
                       ) -> sparse.csr_matrix:
    """Off-diagonal G from gene-locus proximity. No RNA, no pairing, no peak counts.

    A gene keeps its curated chromatin feature and also reads activity features whose
    gene-body midpoints sit on the same chromosome within `max_bp`, weighted
    exp(-d / decay_bp). The features.tsv has no strand, so this is not a TSS
    window. It is the gene–gene genomic graph that is possible on BMMC, where
    peaks were never stored.
    """
    if decay_bp <= 0 or max_bp < 0:
        raise ValueError("genomic G needs max_bp >= 0 and decay_bp > 0")
    projection = sparse.csr_matrix(projection, dtype=np.float32)
    n_genes, n_features = projection.shape
    if len(output_genes) != n_genes or len(feature_names) != n_features:
        raise ValueError(
            f"genomic G names do not match projection {projection.shape}: "
            f"{len(output_genes)} output genes, {len(feature_names)} features")
    feature_position = pd.Series(np.arange(n_features), index=feature_names)
    tss = (loci["start"].astype(float) + loci["end"].astype(float)) / 2.0
    by_chrom: dict[str, list[tuple[float, int, str]]] = {}
    for name in feature_names:
        if name not in tss.index:
            continue
        chrom = str(loci.loc[name, "chrom"])
        by_chrom.setdefault(chrom, []).append(
            (float(tss.loc[name]), int(feature_position[name]), str(name)))
    rows, columns, data = [], [], []
    existing = projection.tolil()
    for gene_i, gene in enumerate(output_genes):
        own = existing.rows[gene_i]
        if not own:
            continue
        rows.extend([gene_i] * len(own))
        columns.extend(own)
        data.extend([1.0] * len(own))
        if gene not in tss.index:
            continue
        chrom = str(loci.loc[gene, "chrom"])
        centre = float(tss.loc[gene])
        for neighbour_tss, column, neighbour in by_chrom.get(chrom, []):
            distance = abs(neighbour_tss - centre)
            if neighbour == gene or distance > max_bp:
                continue
            rows.append(gene_i)
            columns.append(column)
            data.append(float(np.exp(-distance / decay_bp)))
    return row_normalise_links(rows, columns, data, projection.shape)


def peak_genomic_projection(projection, output_genes: pd.Index, feature_names: pd.Index,
                            loci: pd.DataFrame, annotation: pd.DataFrame,
                            decay_bp: float = 50_000
                            ) -> sparse.csr_matrix:
    """Peak-level genomic G, using overlapping gene-activity features as peak proxies.

    cellranger peak annotation is a peak→gene graph. Distal peaks were dropped from
    gene activity (2 kb promoter/body only), so they cannot be recovered as counts.
    A distal peak assigned to gene i that *overlaps another panel gene's locus*
    still has a proxy: that gene's activity column. Self-overlaps are skipped —
    the curated identity already carries weight 1 — and off-diagonal links decay
    as exp(-|peak distance| / decay_bp) so a gene with many distal peaks cannot
    drown its own activity. Peaks that sit on no panel gene stay invisible.
    """
    if decay_bp <= 0:
        raise ValueError("peak-genomic G needs decay_bp > 0")
    projection = sparse.csr_matrix(projection, dtype=np.float32)
    n_genes, n_features = projection.shape
    if len(output_genes) != n_genes or len(feature_names) != n_features:
        raise ValueError("peak-genomic G names do not match the projection")
    required = {"chrom", "start", "end", "gene"}
    missing = required.difference(annotation.columns)
    if missing:
        raise ValueError(f"peak annotation is missing {sorted(missing)}")
    gene_row = pd.Series(np.arange(n_genes), index=output_genes)
    feature_position = pd.Series(np.arange(n_features), index=feature_names)
    feature_loci = loci.reindex(feature_names).dropna(subset=["chrom", "start", "end"])
    existing = projection.tolil()
    allowed = {str(output_genes[i]) for i in range(n_genes) if existing.rows[i]}
    rows, columns, data = [], [], []
    for gene_i in range(n_genes):
        own = existing.rows[gene_i]
        if not own:
            continue
        rows.extend([gene_i] * len(own))
        columns.extend(own)
        data.extend([1.0] * len(own))

    peak_type = annotation["peak_type"] if "peak_type" in annotation.columns \
        else pd.Series(["distal"] * len(annotation), index=annotation.index)
    columns_needed = ["chrom", "start", "end", "gene"]
    if "distance" in annotation.columns:
        columns_needed.append("distance")
    linked = annotation.loc[peak_type.isin(["promoter", "distal"]), columns_needed].copy()
    linked["chrom"] = linked["chrom"].astype(str)
    linked["start"] = pd.to_numeric(linked["start"], errors="coerce")
    linked["end"] = pd.to_numeric(linked["end"], errors="coerce")
    if "distance" in linked.columns:
        linked["distance"] = pd.to_numeric(linked["distance"], errors="coerce").fillna(0.0)
    else:
        linked["distance"] = 0.0
    linked = linked.dropna(subset=["chrom", "start", "end", "gene"])
    for chrom, peaks in linked.groupby("chrom", sort=False):
        on_chrom = feature_loci[feature_loci["chrom"].astype(str) == str(chrom)]
        if on_chrom.empty:
            continue
        feat_start = on_chrom["start"].to_numpy(dtype=np.float64)
        feat_end = on_chrom["end"].to_numpy(dtype=np.float64)
        feat_index = feature_position.reindex(on_chrom.index).to_numpy()
        feat_names = on_chrom.index.to_numpy()
        peak_start = peaks["start"].to_numpy(dtype=np.float64)
        peak_end = peaks["end"].to_numpy(dtype=np.float64)
        peak_genes = peaks["gene"].astype(str).to_numpy()
        peak_dist = np.abs(peaks["distance"].to_numpy(dtype=np.float64))
        overlap = ((peak_start[:, None] < feat_end[None, :])
                   & (feat_start[None, :] < peak_end[:, None]))
        peak_ix, feat_ix = np.nonzero(overlap)
        for peak_i, feat_i in zip(peak_ix.tolist(), feat_ix.tolist()):
            gene = peak_genes[peak_i]
            if gene not in allowed or feat_names[feat_i] == gene:
                continue
            rows.append(int(gene_row[gene]))
            columns.append(int(feat_index[feat_i]))
            data.append(float(np.exp(-peak_dist[peak_i] / decay_bp)))
    return row_normalise_links(rows, columns, data, projection.shape)


def widen_projection(projection, activity, mode: str, neighbours: int = 10,
                     seed: int = 0, output_genes: pd.Index | None = None,
                     feature_names: pd.Index | None = None,
                     loci: pd.DataFrame | None = None,
                     annotation: pd.DataFrame | None = None,
                     genomic_bp: float = 100_000,
                     genomic_decay_bp: float = 50_000):
    """Alternative G matrices, all built WITHOUT the cell correspondence.

    The curated G turns out to be a 0/1 DIAGONAL selection matrix -- one activity feature
    per gene, 719 of 2000 genes with none -- so `Gc` is just the gene's own activity and
    phi's affine Jacobian is diag(gene_scale * mask). A diagonal Jacobian can only rescale
    v_c elementwise; it cannot mix genes. That matters because a freely fitted DENSE map
    from v_c to the kinetic flux reaches cosine 0.40 while the trained model reaches ~0.

    curated        the map as built: diagonal, masked to genes with chromatin support.
    diagonal_full  diagonal over EVERY gene, no mask. Isolates what dropping the 719
                   unsupported genes costs, separately from the lack of mixing.
    coaccess       each gene also links to the `neighbours` activity features it is most
                   correlated with ACROSS ATAC CELLS ONLY. Correlation between two
                   chromatin features needs no RNA, so this stays unpaired; it is the
                   cheapest way to give G off-diagonal structure that means something.
    genomic        off-diagonal links from gene-body midpoint proximity.
    peak-genomic   cellranger peak→gene links, using overlapping activity features as
                   the peak's proxy, with identity held at weight 1 and distal
                   links decayed by |distance|. Needs the annotation table; HSPC only.

    Rows are re-normalised to sum to 1 so `Gc` stays a weighted mean of activities rather
    than growing with the number of links -- otherwise a widened G would hand the alpha
    head a per-gene constant proportional to its degree.
    """
    if mode not in GENE_MAP_MODES:
        raise ValueError(f"unknown gene-map mode {mode!r}; expected {GENE_MAP_MODES}")
    projection = sparse.csr_matrix(projection, dtype=np.float32)
    if mode == "curated":
        return projection
    n_genes, n_features = projection.shape
    if mode in ("diagonal_full", "coaccess"):
        if output_genes is None or feature_names is None:
            raise ValueError(f"{mode} G needs output gene names and feature names")
        output_genes, feature_names = pd.Index(output_genes), pd.Index(feature_names)
        if len(output_genes) != n_genes or len(feature_names) != n_features:
            raise ValueError("G shape does not match output gene and feature names")
        own_columns = feature_names.get_indexer(output_genes)
    if mode == "diagonal_full":
        rows = np.flatnonzero(own_columns >= 0)
        return sparse.csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, own_columns[rows])),
            shape=projection.shape)
    if mode == "genomic":
        if loci is None or output_genes is None or feature_names is None:
            raise ValueError("genomic G needs output gene names, feature names, and loci")
        return genomic_projection(
            projection, pd.Index(output_genes), pd.Index(feature_names), loci,
            genomic_bp, genomic_decay_bp)
    if mode == "peak-genomic":
        if loci is None or annotation is None or output_genes is None or feature_names is None:
            raise ValueError("peak-genomic G needs loci, peak annotation, and gene names")
        return peak_genomic_projection(
            projection, pd.Index(output_genes), pd.Index(feature_names), loci,
            annotation, decay_bp=genomic_decay_bp)

    values = np.asarray(activity, dtype=np.float32)
    centred = values - values.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(centred, axis=0)
    usable = norms > 1e-8
    normalised = np.zeros_like(centred)
    normalised[:, usable] = centred[:, usable] / norms[usable]
    rows, columns, data = [], [], []
    existing = projection.tolil()
    for gene in range(n_genes):
        own = existing.rows[gene]
        anchor = own[0] if own else own_columns[gene]
        if anchor < 0 or not usable[anchor]:
            continue
        correlation = normalised.T @ normalised[:, anchor]
        correlation[~usable] = -np.inf
        top = np.argpartition(-correlation, min(neighbours, len(correlation) - 1))
        top = top[:neighbours + 1]
        keep = np.unique(np.concatenate([np.array([anchor]), top]))
        weights = np.clip(correlation[keep], 0.0, None)
        if weights.sum() <= 0:
            keep, weights = np.array([anchor]), np.array([1.0], dtype=np.float32)
        rows.extend([gene] * len(keep))
        columns.extend(keep.tolist())
        data.extend((weights / weights.sum()).tolist())
    return sparse.csr_matrix((np.asarray(data, dtype=np.float32), (rows, columns)),
                            shape=projection.shape)
