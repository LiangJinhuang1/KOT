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
