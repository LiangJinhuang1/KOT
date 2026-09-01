#!/usr/bin/env python3
"""Checks on the velocity field the runs consume, not on the model.

backend-cosine: two ablations may differ because the field differed.
genes: transcriptional steady-state is not recoverable as a kinetic anchor.
regeneration: a rebuilt h5ad that does not match the cache invalidates every number.
shuffle-strength: cosine near 1 in PC space means the null barely moved.
"""
from __future__ import annotations

from pathlib import Path
import anndata as ad
import argparse
import csv
import h5py
import numpy as np
import pandas as pd
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.adt_gene_map import gene_alias_lookup, normalize_gene_symbol
from src.training.kot import velocity_row_permutation
from src.data.adt_gene_map import load_mapping_records
from src.data.preprocessing import load_and_preprocess_cached
from src.data.projection import projection_matrix_from_adatas
from src.training.kot import (
    align_velocity_fields,
    per_gene_cosine,
    build_velocity_weight,
    compute_velocity_backend_agreement,
    matrix_from_adata,
    resolve_velocity_matrix,
)
from src.utils.io import load_yaml


# ============================================================================
# backend_cosine
# ============================================================================

DATASETS_CONFIG = Path("config/datasets.yaml")


TRAINING_CONFIG = Path("config/training.yaml")


DEFAULT_OUTPUT = Path("cache/results/velocity_backend_cosine.csv")


DEFAULT_PAIRS = ["pbmc_retained:pbmc_regvelo", "bmmc_cite_retained:bmmc_cite_regvelo"]


BINARY_MASKS = {
    "all_shared_genes":      {},
    "velocity_genes":        {"kot_velocity_rna_reliability": True},
    "s_linked_genes":        {"kot_velocity_s_linked_only": True},
    "velocity_and_s_linked": {"kot_velocity_rna_reliability": True,
                              "kot_velocity_s_linked_only": True},
}


RELIABILITY_COLUMNS = {"scvelo_r2": "fit_r2", "scvelo_likelihood": "fit_likelihood"}


def merged_config(dataset_name: str) -> tuple[dict, dict]:
    """Training defaults + this dataset's overrides -- the same merge the runner does, so
    the preprocessing cache key, and therefore the gene set, matches the training runs."""
    train_cfg = load_yaml(TRAINING_CONFIG)
    datasets_meta = load_yaml(DATASETS_CONFIG)["datasets"]
    if dataset_name not in datasets_meta:
        raise ValueError(f"{DATASETS_CONFIG} has no dataset {dataset_name!r}")
    overrides = train_cfg.get("datasets", {}).get(dataset_name) or {}
    return {**train_cfg.get("defaults", {}), **overrides}, datasets_meta[dataset_name]


def load_pair_inputs(cfg: dict, meta: dict):
    """Preprocessed RNA + protein for one dataset, using the shared preprocessing cache."""
    rna_adata, protein_adata, _atac = load_and_preprocess_cached(
        rna_path=meta["rna_path"],
        protein_path=meta.get("protein_path"),
        protein_label=meta.get("protein_label", "ADT"),
        protein_obsm_key=meta.get("protein_obsm_key"),
        rna_raw_layer=meta.get("rna_raw_layer"),
        cache_version=cfg.get("preprocessing_cache_version"),
        rna_min_cells=int(cfg.get("rna_min_cells", 3)),
        rna_n_top_genes=int(cfg.get("rna_n_top_genes", 2000)),
        rna_n_pcs=int(cfg.get("rna_n_pcs", 30)),
        rna_n_neighbors=int(cfg.get("rna_n_neighbors", 30)),
        protein_min_cells=int(cfg.get("protein_min_cells", 1)),
        protein_n_pcs=int(cfg.get("protein_n_pcs", 10)),
    )
    return rna_adata, protein_adata


def reliability_weight(rna_adata, column: str) -> np.ndarray | None:
    """A per-gene reliability vector from the RNA var, floored at 0, or None if absent.

    Two things have to be neutralised, both of which are present in real scVelo output:

    fit_r2 goes negative for genes whose fit is worse than a horizontal line (446 of 1318
    on pbmc). A negative weight would flip those coordinates' sign and turn a disagreement
    into an agreement, so the floor is a correctness requirement, not tidying.

    fit_likelihood is NaN for genes that were never fit (10 of 1318). NaN does not stay
    put: 0 * NaN is NaN, so a single unfit gene propagates through the masked product and
    makes every per-cell cosine NaN. A gene with no fit has no evidence of reliability,
    which is a weight of zero.
    """
    if column not in rna_adata.var.columns:
        return None
    values = np.asarray(rna_adata.var[column].values, dtype=np.float32)
    return np.clip(np.nan_to_num(values, nan=0.0), 0.0, None)


def kinetic_gene_mask(S: np.ndarray, kin_mask: np.ndarray) -> np.ndarray:
    """RNA genes that reach the kinetic protein coordinates through S.

    Restricted to the kinetic rows explicitly. It happens to equal the column support of
    the whole S -- a non-zero row and kinetic_mask are the same condition, by
    assert_no_orphan_s_rows -- but saying it in terms of the kinetic panel is what the
    quantity actually means, and it stays correct if that invariant is ever relaxed.
    Uses abs().sum() so two entries that cancel in a column still count as present.
    """
    return np.abs(S[kin_mask]).sum(axis=0) > 0


def projection_for(cfg: dict, rna_adata, protein_adata):
    """S in feature space, built exactly as run_kot builds it (mapping CSV wins)."""
    mapping_csv = cfg.get("adt_mapping_csv")
    mapping_records = load_mapping_records(Path(mapping_csv)) if mapping_csv else None
    S_np, _align, kin_mask, _report = projection_matrix_from_adatas(
        rna_adata,
        protein_adata,
        mapping_records=mapping_records,
        use_mean_expr=False,
        use_explicit_links=bool(cfg.get("kot_use_explicit_links", True)),
        require_full_panel=bool(cfg.get("kot_require_full_panel_mapping", True))
        and mapping_records is not None,
    )
    return S_np, kin_mask


def per_gene_diagonal(rna_adata, velocity: np.ndarray, compare_path: str, layer: str,
                     schemes: dict) -> pd.DataFrame:
    """One row per shared gene: how much the two backends agree about THAT gene.

    This is the diagonal of the cross-backend gene-by-gene product. The per-cell cosine
    reported elsewhere asks whether a cell's whole velocity vector points the same way in
    both; this asks which GENES the two providers disagree about, which is what a
    difference in the kinetics has to be traced back to.

    The per-gene weights are reported as COLUMNS, not applied. A per-gene weight scales
    that gene's column identically in both fields, and cosine is scale-invariant, so
    multiplying by it cannot change this number -- only the per-cell cosine, where a
    weight changes how much each gene contributes to a cell's direction. Reporting the
    weights alongside lets agreement be read against reliability without implying a
    correction that did not happen.
    """
    other = ad.read_h5ad(compare_path)
    v_self, v_other, cells, genes = align_velocity_fields(rna_adata, velocity, other, layer)
    del other
    frame = pd.DataFrame({
        "gene": list(genes),
        "cos": per_gene_cosine(v_self, v_other),
        "norm_self": np.linalg.norm(v_self, axis=0),
        "norm_other": np.linalg.norm(v_other, axis=0),
        "n_cells": len(cells),
    })
    gene_index = pd.Index(rna_adata.var_names).get_indexer(genes)
    for name, weight in schemes.items():
        frame[f"w_{name}"] = 1.0 if weight is None else weight[gene_index]
    return frame


def weight_schemes(rna_adata, S_np: np.ndarray, kin_mask: np.ndarray, cfg: dict) -> dict:
    """name -> per-gene weight vector (None means unweighted), in report order."""
    schemes = {}
    for mask_name, flags in BINARY_MASKS.items():
        schemes[mask_name] = build_velocity_weight(
            rna_adata, S_np, {**cfg, **flags}, use_feature_space=True)
    s_linked = kinetic_gene_mask(S_np, kin_mask).astype(np.float32)
    for label, column in RELIABILITY_COLUMNS.items():
        weight = reliability_weight(rna_adata, column)
        if weight is not None:
            schemes[f"s_linked_x_{label}"] = s_linked * weight
    return schemes


def rows_for_pair(name: str, compare_name: str) -> list[dict]:
    """One row per mask: the two backends' per-cell cosine over the genes that mask keeps."""
    cfg, meta = merged_config(name)
    _compare_cfg, compare_meta = merged_config(compare_name)
    rna_adata, protein_adata = load_pair_inputs(cfg, meta)
    S_np, kin_mask = projection_for(cfg, rna_adata, protein_adata)

    # Feature space is what makes the gene axis meaningful: a PCA representation has no
    # gene to mask or to match against the other file.
    D_r = matrix_from_adata(rna_adata, cfg.get("kot_rna_layer"), "RNA").shape[1]
    velocity, layer = resolve_velocity_matrix(
        rna_adata, cfg.get("velocity_layer"), D_r, use_feature_space=True,
    )
    print(f"[pair] {name} vs {compare_name} | velocity layer '{layer}' "
          f"| {rna_adata.n_obs} cells x {rna_adata.n_vars} genes")

    mask_cfg = {
        **cfg,
        "kot_velocity_compare_h5ad": compare_meta["rna_path"],
        "kot_velocity_compare_layer": cfg.get("kot_velocity_compare_layer", "velocity"),
        "kot_velocity_compare_label": compare_name,
    }
    schemes = weight_schemes(rna_adata, S_np, kin_mask, cfg)
    genes_frame = per_gene_diagonal(
        rna_adata, velocity, compare_meta["rna_path"],
        str(cfg.get("kot_velocity_compare_layer", "velocity")), schemes)
    genes_frame.insert(0, "dataset", name)
    genes_frame.insert(1, "compare_dataset", compare_name)

    rows = []
    for mask_name, weight in schemes.items():
        out = compute_velocity_backend_agreement(rna_adata, velocity, mask_cfg, weight)
        rows.append({
            "dataset": name,
            "compare_dataset": compare_name,
            "mask": mask_name,
            "weighting": "continuous" if mask_name.startswith("s_linked_x_") else "binary",
            "genes_in_mask": out["velocity_backend_kot_n_genes"],
            "shared_genes": out["velocity_backend_n_genes"],
            "n_cells": out["velocity_backend_n_cells"],
            "cos_median": out["velocity_backend_cos_kot_median"],
            "cos_mean": out["velocity_backend_cos_kot_mean"],
            "cos_q25": out["velocity_backend_cos_kot_q25"],
            "cos_q75": out["velocity_backend_cos_kot_q75"],
            # Magnitude: cosine is scale-invariant, so the two together are the whole
            # difference between the fields. >1 means scVelo's vectors are the longer.
            "norm_ratio_median": out["velocity_backend_norm_ratio_kot_median"],
            "norm_ratio_q25": out["velocity_backend_norm_ratio_kot_q25"],
            "norm_ratio_q75": out["velocity_backend_norm_ratio_kot_q75"],
            "zero_frac_masked": out["velocity_backend_kot_zero_frac"],
            "zero_frac_all_genes": out["velocity_backend_zero_frac"],
            "status": out["velocity_backend_status"],
        })
    return rows, genes_frame


def backend_cosine_main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", default=",".join(DEFAULT_PAIRS),
                        help="Comma-separated dataset:compare_dataset pairs.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output CSV path.")
    args = parser.parse_args()

    rows, gene_frames = [], []
    for pair in args.pairs.split(","):
        name, _, compare_name = pair.partition(":")
        if not compare_name:
            raise ValueError(f"--pairs expects dataset:compare_dataset, got {pair!r}")
        pair_rows, pair_genes = rows_for_pair(name.strip(), compare_name.strip())
        rows.extend(pair_rows)
        gene_frames.append(pair_genes)

    frame = pd.DataFrame(rows)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)
    genes_path = out_path.with_name(out_path.stem + "_per_gene.csv")
    genes = pd.concat(gene_frames, ignore_index=True)
    genes.to_csv(genes_path, index=False)
    print(f"\n{frame.to_string(index=False)}\n\nwrote {out_path}")
    print(f"wrote {genes_path}: {len(genes)} genes")
    for label, mask in (("all shared genes", genes["cos"].notna()),
                        ("KOT S-linked genes", genes["w_s_linked_genes"] > 0)):
        sub = genes.loc[mask]
        cos_q25, cos_med, cos_q75 = sub["cos"].quantile([0.25, 0.50, 0.75])
        ratio = (sub["norm_self"] / sub["norm_other"].replace(0.0, float("nan"))).dropna()
        r25, rmed, r75 = ratio.quantile([0.25, 0.50, 0.75])
        print(f"  per-gene {label:<20} n={len(sub):>5}  "
              f"cos={cos_med:+.3f} [{cos_q25:+.3f},{cos_q75:+.3f}]  "
              f"|v_sc|/|v_reg|={rmed:.2f} [{r25:.2f},{r75:.2f}]")


# ============================================================================
# genes
# ============================================================================

def col(var, names):
    """First matching var column as a float array, else None."""
    for n in names:
        if n in var.columns:
            return var[n].to_numpy(dtype=float)
    return None


def unspliced_fraction(adata):
    """Per-gene mean unspliced / (spliced + unspliced) from the raw layers."""
    if "unspliced" not in adata.layers or "spliced" not in adata.layers:
        return None
    u = np.asarray(adata.layers["unspliced"].sum(axis=0)).ravel()
    s = np.asarray(adata.layers["spliced"].sum(axis=0)).ravel()
    denom = u + s
    return np.divide(u, denom, out=np.zeros_like(u, dtype=float), where=denom > 0)


def adt_genes_from_coverage(path: Path) -> dict[str, dict]:
    out = {}
    with path.open() as handle:
        for r in csv.DictReader(handle):
            g = (r.get("gene_symbol") or "").strip()
            if g:
                out[normalize_gene_symbol(g)] = r
    return out


def genes_main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rna", type=Path, required=True, help="retained velocity h5ad")
    parser.add_argument("--coverage", type=Path, required=True, help="kinetics coverage CSV")
    parser.add_argument("--min-r2-grid", type=str, default="0.01,0.005,0.001,0.0",
                        help="candidate min_r2 thresholds to count admitted genes")
    args = parser.parse_args()

    adata = ad.read_h5ad(args.rna)
    var = adata.var
    vgenes = var["velocity_genes"].to_numpy(dtype=bool) if "velocity_genes" in var else None
    r2 = col(var, ["velocity_r2", "fit_r2", "r2"])
    lik = col(var, ["fit_likelihood", "velocity_score"])
    ufrac = unspliced_fraction(adata)

    cov = adt_genes_from_coverage(args.coverage)
    gene_pos = gene_alias_lookup(var.index.astype(str), var, normalizer=normalize_gene_symbol)

    print(f"{'gene':<10}{'in_var':<7}{'vel_gene':<9}{'r2':>9}{'likelihood':>12}{'u_frac':>8}")
    recoverable, steady, unknown, absent = [], [], [], []
    for g in sorted(cov):
        i = gene_pos.get(g)
        if i is None:
            print(f"{g:<10}{'no':<7}")
            absent.append(g)
            continue
        vg = bool(vgenes[i]) if vgenes is not None else None
        r = r2[i] if r2 is not None else float("nan")
        lk = lik[i] if lik is not None else float("nan")
        uf = ufrac[i] if ufrac is not None else float("nan")
        print(f"{g:<10}{'yes':<7}{str(vg):<9}{r:>9.4f}{lk:>12.4f}{uf:>8.3f}")
        if vg is False:
            if uf < 0.05 or (not np.isnan(r) and r < 1e-3):
                steady.append(g)
            elif np.isnan(r):
                unknown.append(g)
            else:
                recoverable.append(g)

    print("\n--- admitted ADT genes at candidate min_r2 (of currently-excluded) ---")
    grid = [float(x) for x in args.min_r2_grid.split(",")]
    excluded = [g for g in cov if gene_pos.get(g) is not None
                and vgenes is not None and not vgenes[gene_pos[g]]]
    for thr in grid:
        n = sum(1 for g in excluded
                if r2 is not None and not np.isnan(r2[gene_pos[g]]) and r2[gene_pos[g]] >= thr)
        print(f"  min_r2 >= {thr:<7}: {n}/{len(excluded)} excluded ADT genes would be admitted")

    print(f"\nsummary: recoverable(real dynamics, low r2)={len(recoverable)}  "
          f"steady-state(u_frac<0.05 or r2~0)={len(steady)}  "
          f"unknown_missing_r2={len(unknown)}  not-in-var={len(absent)}")
    if recoverable:
        print("  recoverable -> " + ", ".join(recoverable))
    if steady:
        print("  steady-state (forcing them in will not help) -> " + ", ".join(steady))
    if unknown:
        print("  unknown (missing r2; inspect fit columns before relaxing thresholds) -> " + ", ".join(unknown))


# ============================================================================
# regeneration
# ============================================================================

def index_names(group) -> np.ndarray:
    """The obs/var index of an h5ad group, as plain strings."""
    key = group.attrs.get("_index", "_index")
    if isinstance(key, bytes):
        key = key.decode()
    values = group[key][:]
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in values])


def read_axes(path: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    with h5py.File(path, "r") as handle:
        obs = index_names(handle["obs"])
        var = index_names(handle["var"])
        layers = sorted(handle["layers"].keys()) if "layers" in handle else []
    return obs, var, layers


def regeneration_main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("regenerated", help="new velocity .h5ad")
    parser.add_argument("cache", help="preprocessed .rna.h5ad built from the old one")
    args = parser.parse_args()

    src_obs, src_var, src_layers = read_axes(args.regenerated)
    cch_obs, cch_var, cch_layers = read_axes(args.cache)

    print(f"regenerated : {len(src_obs):7d} cells  {len(src_var):6d} genes")
    print(f"cache       : {len(cch_obs):7d} cells  {len(cch_var):6d} genes")

    same_cells = set(src_obs) == set(cch_obs)
    missing_genes = set(cch_var) - set(src_var)
    missing_layers = set(cch_layers) - set(src_layers)

    print(f"cells identical      : {same_cells}")
    print(f"cache genes ⊆ source : {not missing_genes}  (missing {len(missing_genes)})")
    print(f"cache layers ⊆ source: {not missing_layers}  (missing {sorted(missing_layers)})")

    if same_cells and not missing_genes and not missing_layers:
        print("\nPASS — the regeneration reproduces the input behind the cache.")
        return 0
    print("\nFAIL — the regenerated file differs from what the cache was built from.")
    if not same_cells:
        print(f"  cells only in cache : {len(set(cch_obs) - set(src_obs))}")
        print(f"  cells only in source: {len(set(src_obs) - set(cch_obs))}")
    if missing_genes:
        print(f"  genes dropped: {sorted(missing_genes)[:10]}")
    return 1



# ============================================================================
# shuffle_strength
# ============================================================================


def unit_rows(matrix: np.ndarray) -> np.ndarray:
    """Rows rescaled to unit length, dropping the cells that carry no velocity."""
    norms = np.linalg.norm(matrix, axis=1)
    keep = norms > 0
    return matrix[keep] / norms[keep][:, None]


def report_space(name: str, matrix: np.ndarray, seeds: list) -> None:
    """cos(v_i, v_assigned) pooled over one permutation per seed, plus the field mean."""
    units = unit_rows(matrix)
    cosines = np.concatenate([
        (units * units[velocity_row_permutation(units.shape[0], seed)]).sum(1)
        for seed in seeds
    ])
    mean_direction = units.mean(0)
    to_mean = units @ (mean_direction / np.linalg.norm(mean_direction))
    print(f"  {name:11s} cos(real_i, assigned_i)  mean={cosines.mean():+.3f} "
          f"median={np.median(cosines):+.3f}  |cos|>0.5: {100 * np.mean(np.abs(cosines) > 0.5):.1f}%")
    print(f"  {name:11s} cos(real_i, field mean)  mean={to_mean.mean():+.3f} "
          f"median={np.median(to_mean):+.3f}")


def shuffle_strength_main() -> None:
    """Report the permutation's strength in PC space and, for reference, in gene space.

    A linear projection does not preserve cosines, so the gene-space number describes a
    vector the model never sees -- the PC-space one is the control's real strength.
    """
    parser = argparse.ArgumentParser(
        description="How different is a permuted velocity from the real one, per cell?")
    parser.add_argument("rna", help="velocity h5ad carrying layers['velocity'] and varm['PCs']")
    parser.add_argument("label", help="panel name for the printed header")
    parser.add_argument("seeds", help="comma-separated permutation seeds")
    args = parser.parse_args()

    seeds = [int(seed) for seed in args.seeds.split(",")]
    adata = ad.read_h5ad(args.rna)
    velocity = np.nan_to_num(np.asarray(adata.layers["velocity"], dtype=np.float32), nan=0.0)
    in_pc_space = velocity @ np.asarray(adata.varm["PCs"], dtype=np.float32)

    print(f"{args.label}: {velocity.shape[0]} cells x {velocity.shape[1]} genes -> "
          f"{in_pc_space.shape[1]} PCs, over {len(seeds)} seeds")
    report_space("PC (model)", in_pc_space, seeds)
    report_space("gene", velocity, seeds)


# ============================================================================
# dispatch
# ============================================================================

COMMANDS = {
    "backend-cosine": backend_cosine_main,
    "genes": genes_main,
    "regeneration": regeneration_main,
    "shuffle-strength": shuffle_strength_main,
}


def main() -> int:
    """Route to a subcommand, leaving the rest of argv for its own parser."""
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        print(f"commands: {', '.join(COMMANDS)}")
        return 2
    command = sys.argv.pop(1)
    return COMMANDS[command]() or 0


if __name__ == "__main__":
    sys.exit(main())
