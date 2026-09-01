"""MaxFuse baseline (Chen et al., Nat Biotechnol 2024). Closest published method
to KOT's S without kinetics: shared features come from this run's adt_mapping_csv,
not the package's conversion table. Second modality is row-permuted before fit.
The seed does not pin neighbour graphs (scanpy / pynndescent), so identical
reruns are not bitwise equal.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

VENDOR_MAXFUSE_PATH = Path(__file__).resolve().parents[2] / "vendor" / "maxfuse"
if VENDOR_MAXFUSE_PATH.exists():
    sys.path.insert(0, str(VENDOR_MAXFUSE_PATH))

import maxfuse as mf

from src.data.adt_gene_map import load_mapping_records
from src.utils.arrays import matrix_from_adata

STATIC_TOL = 1e-5

# Author-recommended CITE-seq settings, from the package's own
# docs/citeseq_pbmc_evaluate.ipynb. Fixed rather than configurable: they describe how
# MaxFuse is meant to be run on this modality pair, and the SVD widths are clamped to
# each panel's size anyway (clamp_components). What a different dataset actually needs
# to move -- batching, embedding width, filter strength -- is in config/training.yaml.
MATCHING_RATIO = 3
N_NEIGHBORS = 15
LEIDEN_RESOLUTION = 2
RESOLUTION_TOL = 0.1
GRAPH_SVD_COMPONENTS = 30
INIT_SVD_COMPONENTS_RNA = 25
INIT_SVD_COMPONENTS_SECOND = 20
SMOOTHING_WEIGHT = 0.7
CCA_MAX_ITER = 2000


def shared_feature_pairs(mapping_records: list[dict], rna_names, protein_names):
    """One column per alignment link, not per protein: MaxFuse matches column-wise."""
    rna_index = {str(name): j for j, name in enumerate(rna_names)}
    protein_index = {str(name): j for j, name in enumerate(protein_names)}
    gene_cols, protein_cols, dropped = [], [], []
    for record in mapping_records:
        if not record["use_for_alignment"]:
            continue
        gene, adt = record["gene_symbol"], record["adt_name"]
        if gene in rna_index and adt in protein_index:
            gene_cols.append(rna_index[gene])
            protein_cols.append(protein_index[adt])
        else:
            dropped.append(adt if gene in rna_index else f"{adt}->{gene}")
    if not gene_cols:
        raise ValueError(
            "No use_for_alignment link in adt_mapping_csv survives in both matrices, so "
            "MaxFuse has no shared features to start from. Check that the mapping CSV was "
            "built against this dataset's retained gene set."
        )
    return np.array(gene_cols), np.array(protein_cols), dropped


def scale_columns(values: np.ndarray) -> np.ndarray:
    """Static columns stay 0: dividing by ~0 is noise, and callers drop them."""
    values = np.asarray(values, dtype=np.float32)
    sd = values.std(axis=0)
    centred = values - values.mean(axis=0, keepdims=True)
    return np.divide(centred, sd, out=np.zeros_like(centred),
                     where=sd > STATIC_TOL).astype(np.float32)


def non_static_columns(values: np.ndarray) -> np.ndarray:
    """Mask of features that vary: a constant column breaks the SVD and carries nothing."""
    return values.std(axis=0) > STATIC_TOL


def paired_shared_blocks(rna_values: np.ndarray, protein_values: np.ndarray,
                         gene_cols: np.ndarray, protein_cols: np.ndarray):
    """The two shared blocks, scaled and pruned TOGETHER.

    A pair is dropped when either side is static, never one side alone: the blocks are
    matched column by column, so pruning them independently would silently pair gene i
    with a different protein.
    """
    rna_shared = rna_values[:, gene_cols]
    protein_shared = protein_values[:, protein_cols]
    keep = non_static_columns(rna_shared) & non_static_columns(protein_shared)
    return scale_columns(rna_shared[:, keep]), scale_columns(protein_shared[:, keep]), keep


def clamp_components(requested: int, n_features: int, label: str) -> int:
    """A component count the data can actually supply.

    The tutorials assume a wide CITE-seq panel; a short ADT panel cannot supply
    that many singular vectors, and MaxFuse then fails inside the SVD. Clamping
    here turns a crash into a printed, recorded decision.
    """
    allowed = max(1, min(int(requested), int(n_features) - 1))
    if allowed != int(requested):
        print(f"[maxfuse] {label}: {requested} -> {allowed} "
              f"(only {n_features} features available)")
    return allowed


def permuted_rows(n_rows: int, seed: int, enabled: bool):
    """Row order for the second modality, and the inverse that undoes it."""
    if not enabled:
        return None, None
    permutation = np.random.default_rng(seed).permutation(n_rows)
    return permutation, np.argsort(permutation)


def save_matching(output_dir, fusor, permutation, second_label: str) -> int:
    """Persist the pivot matching (cells, not a dense coupling).

    MaxFuse returns a sparse set of matched pairs, so the honest artifact is the pair list
    -- a dense n1 x n2 coupling would be 8 GB on bmmc_cite and mostly zero. Indices into
    the second modality are mapped back through the permutation, so the file is readable
    against the caller's own cell order.
    """
    rows, cols, scores = fusor.get_matching(target="pivot")
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    if permutation is not None:
        # get_matching indexes the PERMUTED arr2, whose row i is the caller's
        # permutation[i]; report the caller's row.
        cols = permutation[cols]
    if output_dir is not None:
        np.savez(Path(output_dir) / "maxfuse_matching.npz", rna_rows=rows,
                 **{f"{second_label}_rows": cols}, scores=np.asarray(scores, float))
    return len(rows)


def run_maxfuse(context: dict, cfg: dict):
    """Fit MaxFuse and return [rna_embedding, second_embedding] in the caller's order."""
    rna_adata = context["rna_adata"]
    second_adata = context["second_adata"]
    second_label = context["second_label"]
    output_dir = context.get("output_dir")

    seed = int(cfg.get("seed", 42))
    # The same matrices KOT reads in feature space: MaxFuse works on features, not on the
    # PCA reprs the runner hands over in `x`/`y`, and a baseline fed a different matrix
    # would differ from KOT by its input before it differed by its method.
    rna_layer = cfg.get("maxfuse_rna_layer") or cfg.get("kot_rna_layer")
    second_layer = cfg.get("maxfuse_second_layer") or cfg.get("kot_protein_layer")
    rna_values = matrix_from_adata(rna_adata, rna_layer, "RNA")
    second_values = matrix_from_adata(second_adata, second_layer, second_label)

    mapping_csv = cfg.get("adt_mapping_csv")
    if not mapping_csv or not Path(mapping_csv).exists():
        raise FileNotFoundError(
            f"maxfuse needs the curated ADT->gene mapping (adt_mapping_csv={mapping_csv!r}). "
            f"Build it first:\n"
            f"  PYTHONPATH=. python tools/build_inputs.py adt-mapping --datasets <dataset>"
        )
    mapping_records = load_mapping_records(Path(mapping_csv))
    gene_cols, protein_cols, dropped = shared_feature_pairs(
        mapping_records, rna_adata.var_names, second_adata.var_names)

    rna_shared, second_shared, kept = paired_shared_blocks(
        rna_values, second_values, gene_cols, protein_cols)
    rna_active = scale_columns(rna_values[:, non_static_columns(rna_values)])
    second_active = scale_columns(second_values[:, non_static_columns(second_values)])
    print(f"[maxfuse] shared features: {int(kept.sum())} linked pairs "
          f"({len(gene_cols)} curated, {len(gene_cols) - int(kept.sum())} static); "
          f"active: RNA {rna_active.shape[1]}, {second_label} {second_active.shape[1]}")
    if dropped:
        print(f"[maxfuse] {len(dropped)} curated link(s) absent from one matrix: "
              f"{sorted(set(dropped))[:8]}{' ...' if len(dropped) > 8 else ''}")

    # Diagonal integration: the second modality is shuffled so no step can read the
    # pairing, and the embedding is put back in the caller's order at the end.
    permute_second = bool(cfg.get("maxfuse_permute_second", True))
    permutation, inverse_permutation = permuted_rows(
        second_shared.shape[0], seed, permute_second)
    if permutation is not None:
        second_shared = second_shared[permutation]
        second_active = second_active[permutation]
        print(f"[maxfuse] permuted {second_label} rows before fitting (seed={seed})")

    graph_svd1 = clamp_components(GRAPH_SVD_COMPONENTS, rna_active.shape[1],
                                  "graph svd (RNA)")
    graph_svd2 = clamp_components(GRAPH_SVD_COMPONENTS, second_active.shape[1],
                                  f"graph svd ({second_label})")
    init_svd1 = clamp_components(INIT_SVD_COMPONENTS_RNA, rna_shared.shape[1],
                                 "init svd (RNA)")
    init_svd2 = clamp_components(INIT_SVD_COMPONENTS_SECOND, second_shared.shape[1],
                                 f"init svd ({second_label})")
    cca_components = clamp_components(cfg.get("maxfuse_cca_components", 20),
                                      min(rna_active.shape[1], second_active.shape[1]),
                                      "cca components")

    # MaxFuse builds its neighbour index as pynndescent.NNDescent(...) with no
    # random_state (vendor/maxfuse/maxfuse/graph.py), which makes it draw from numpy's
    # GLOBAL RNG -- the reason two runs at one seed used to disagree. Seeding the global
    # RNG here reaches that call without patching vendored code, and nothing else in this
    # module relies on global numpy randomness.
    np.random.seed(seed)
    fusor = mf.model.Fusor(
        shared_arr1=rna_shared,
        shared_arr2=second_shared,
        active_arr1=rna_active,
        active_arr2=second_active,
        labels1=None,
        labels2=None,
    )
    fusor.split_into_batches(
        max_outward_size=int(cfg.get("maxfuse_max_outward_size", 5000)),
        matching_ratio=MATCHING_RATIO,
        metacell_size=int(cfg.get("maxfuse_metacell_size", 2)),
        seed=seed,
    )
    fusor.construct_graphs(
        n_neighbors1=N_NEIGHBORS,
        n_neighbors2=N_NEIGHBORS,
        svd_components1=graph_svd1,
        svd_components2=graph_svd2,
        resolution1=LEIDEN_RESOLUTION,
        resolution2=LEIDEN_RESOLUTION,
        resolution_tol=RESOLUTION_TOL,
        leiden_seed=seed,
    )
    fusor.find_initial_pivots(
        wt1=SMOOTHING_WEIGHT,
        wt2=SMOOTHING_WEIGHT,
        svd_components1=init_svd1,
        svd_components2=init_svd2,
    )
    fusor.refine_pivots(
        wt1=SMOOTHING_WEIGHT,
        wt2=SMOOTHING_WEIGHT,
        svd_components1=graph_svd1,
        svd_components2=graph_svd2,
        cca_components=cca_components,
        n_iters=int(cfg.get("maxfuse_n_iters", 3)),
        cca_max_iter=CCA_MAX_ITER,
    )
    fusor.filter_bad_matches(
        target="pivot",
        filter_prop=float(cfg.get("maxfuse_filter_prop", 0.3)),
    )

    n_pivots = save_matching(output_dir, fusor, permutation, second_label)
    rna_embedding, second_embedding = fusor.get_embedding(
        active_arr1=fusor.active_arr1, active_arr2=fusor.active_arr2)
    rna_embedding = np.asarray(rna_embedding, dtype=np.float32)
    second_embedding = np.asarray(second_embedding, dtype=np.float32)
    if inverse_permutation is not None:
        second_embedding = second_embedding[inverse_permutation]

    diagnostics = {
        "maxfuse_shared_features": int(kept.sum()),
        "maxfuse_curated_links": int(len(gene_cols)),
        "maxfuse_active_rna_features": int(rna_active.shape[1]),
        f"maxfuse_active_{second_label}_features": int(second_active.shape[1]),
        "maxfuse_cca_components": int(cca_components),
        "maxfuse_n_pivots": int(n_pivots),
        # Matched pairs surviving the filter, expanded from metacells back to cells. The
        # fraction of RNA cells they cover says how much of the data the CCA was fitted on
        # -- a low value means the embedding is an extrapolation for most cells.
        "maxfuse_pivot_fraction": float(n_pivots / max(rna_shared.shape[0], 1)),
        "maxfuse_permute_second": bool(permute_second),
        "maxfuse_permutation_seed": int(seed),
        "maxfuse_rna_layer": str(rna_layer),
        f"maxfuse_{second_label}_layer": str(second_layer),
    }
    print(f"[maxfuse] {n_pivots} pivots, embedding dim {rna_embedding.shape[1]}")
    # No coupling: MaxFuse produces a sparse pair list, saved above as maxfuse_matching.npz.
    return [rna_embedding, second_embedding], None, diagnostics
