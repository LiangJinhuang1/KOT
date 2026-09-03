#!/usr/bin/env python3
"""KOT on the Papalexi GSE153056 CRISPR screen — the experiment, not the exploration.

`tools/papalexi.py` is the earlier exploratory pair (signal + benchmark) and stays as it
is. This is the CRISPR experiment proper, and it is built in the order the evaluation has
to exist before a model result means anything:

  effects   the observed response table, written BEFORE any model is trained
  gate      the sanity gate: can NT-supervised models predict these responses at all?
  preprocess  velocity inputs whose every fitted statistic comes from NT cells alone
  predict   a checkpoint's two perturbation predictions: phi and its Jacobian
  competitors  put the Task A competitors on the SAME per-replicate delta format
  evaluate  the full metric suite over the four effect sets, with unit-level CIs

Panel (4 ADTs, and the transcript each one is read off):
  CD86   -> CD86        PDL1  -> CD274
  PDL2   -> PDCD1LG2    CD366 -> HAVCR2

THE THRESHOLD IS FROZEN HERE. A knockout is usable when it has at least
MIN_KO_CELLS cells in each of at least MIN_REPLICATES replicates
(src.evaluation.perturbation.sufficient_knockouts, the same rule the benchmark's scoring
set uses). It is fixed before any model is scored so that no later result can move it.
"""
from __future__ import annotations

from pathlib import Path
from scipy import stats
import anndata as ad
import argparse
import h5py
import numpy as np
import pandas as pd
import sys

from anndata.io import read_elem
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.data.nt_only import preprocess_nt_only
from src.losses.jvp_physics import jacobian_vector_product
from src.data.adt_gene_map import load_mapping_records
from src.evaluation.crispr_metrics import (
    EFFECT_SETS,
    annotate_effect_sets,
    bootstrap_over_units,
    confirmed_knockouts,
    metric_suite,
    per_group_correlation,
    response_cosine,
)
from src.evaluation.perturbation import (
    positive_scale,
    replicate_effects,
    sufficient_knockouts,
)
from src.training.protein_regression import run_mlp, run_ridge
from src.training.protein_supervised import protein_target_units
from tools.evaluate_checkpoints import build_model_and_tensors, load_dataset_for_run
from tools.papalexi import (
    ADT_GENE_MAP,
    CONTROL_LABEL,
    load_metadata,
    load_protein,
    load_target_gene_expression,
    normalize_adt,
    to_dense,
)

# Default inputs and outputs, named once: they are shared by five subcommands, and a
# subcommand pointed at a different protein file than the effects table it is scored
# against would produce a table nothing else in the pipeline could be joined to.
PROTEIN_H5AD = Path("Datasets/Papalexi_GSE153056/papalexi_protein.h5ad")
METADATA_TSV = Path("data/papalexi/ECCITE_metadata.tsv")
RNA_H5AD = Path("Datasets/Papalexi_GSE153056/papalexi_rna_input.h5ad")
NT_ONLY_H5AD = Path("cache/velocity/papalexi_nt_only/papalexi_scvelo_results_nt_only.h5ad")
CRISPR_DIR = Path("cache/results/crispr")
EFFECTS_CSV = CRISPR_DIR / "crispr_effects_observed.csv"
MULTIPERT_SUBSET_CSV = Path("config/papalexi_multipert_subset.csv")

# Frozen before model evaluation. Raising either after seeing a result would be choosing
# the benchmark to suit the answer.
MIN_KO_CELLS = 20
MIN_REPLICATES = 2
REPLICATE_COLUMN = "replicate"

# A phi whose control-cell spread is below this fraction of the observed spread has
# collapsed. Rescaling it onto the observed units divides numerical dust by a very small
# number, which is how the zero-velocity control arm -- sd_ratio 0.002, i.e. a constant
# phi -- came to post the highest primary Spearman of any ablation. Below the floor the
# protein is dropped from scoring instead.
MIN_CALIBRATION_SD_RATIO = 0.01

# A self effect is a knockout of the gene the protein is read off. A CRISPR indel can
# destroy the protein while leaving the transcript intact, so these say nothing about an
# RNA->protein map and are counted separately, never scored with the rest.
SELF_PAIRS = {(gene, adt) for adt, gene in ADT_GENE_MAP.items()}


# The gate is deliberately generous: these models are handed the NT cells' RNA<->protein
# PAIRING, which KOT never gets. A pass says the responses are predictable from KO RNA at
# all; it says nothing about whether KOT can do it.
GATE_PREDICTORS = ("zero_change", "cognate_mrna", "ridge", "mlp")
N_PERMUTATIONS = 2000


def load_rna_layer(path: Path, layer: str) -> ad.AnnData:
    """Read one layer plus obs/var names, not all eleven.

    The retained Papalexi object carries Ms, Mu, velocity, spliced, unspliced and more at
    ~858MB; a plain read_h5ad of it is the difference between fitting in memory and not.
    """
    with h5py.File(path, "r") as handle:
        matrix = read_elem(handle[f"layers/{layer}"])
        obs_names = read_elem(handle["obs"]).index
        var_names = read_elem(handle["var"]).index
    dense = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
    rna = ad.AnnData(X=np.asarray(dense, dtype=np.float32))
    rna.obs_names = obs_names.astype(str)
    rna.var_names = var_names.astype(str)
    rna.layers[layer] = rna.X.copy()
    return rna


def supervised_cell_predictions(rna: ad.AnnData, protein: ad.AnnData, fit_rows: np.ndarray,
                                layer: str, seed: int) -> dict[str, np.ndarray]:
    """Per-cell protein predicted by the two NT-supervised models, for every cell."""
    context = {
        "rna_adata": rna,
        "second_adata": protein[fit_rows].copy(),
        "second_label": "protein",
        "fit_rows": fit_rows,
        "output_dir": Path("."),
    }
    cfg = {"seed": seed, "kot_rna_layer": layer, "kot_protein_layer": None}
    return {"ridge": np.asarray(run_ridge(context, cfg)[0][0]),
            "mlp": np.asarray(run_mlp(context, cfg)[0][0])}


def permutation_null(predicted: np.ndarray, observed: np.ndarray, seed: int) -> tuple[float, float]:
    """Chance Spearman for this many pairs: keep the values, break the pairing."""
    rng = np.random.default_rng(seed)
    finite = np.isfinite(predicted) & np.isfinite(observed)
    predicted, observed = predicted[finite], observed[finite]
    draws = [stats.spearmanr(predicted, observed[rng.permutation(len(observed))]).statistic
             for _ in range(N_PERMUTATIONS)]
    return float(np.mean(draws)), float(np.std(draws))


def score_predictor(predicted: np.ndarray, observed: np.ndarray, seed: int) -> dict:
    finite = np.isfinite(predicted) & np.isfinite(observed)
    predicted, observed = predicted[finite], observed[finite]
    if len(predicted) < 3 or np.std(predicted) == 0:
        # zero_change is constant by construction: Spearman is undefined, MAE is not, and
        # MAE is the metric that makes "predict nothing" a meaningful reference.
        return {"n_pairs": int(len(predicted)), "spearman": np.nan, "pearson": np.nan,
                "sign_acc": np.nan, "mae": float(np.mean(np.abs(predicted - observed))),
                "null_mean": np.nan, "null_sd": np.nan, "z_vs_null": np.nan}
    null_mean, null_sd = permutation_null(predicted, observed, seed)
    spearman = float(stats.spearmanr(predicted, observed).statistic)
    return {
        "n_pairs": int(len(predicted)),
        "spearman": spearman,
        "pearson": float(stats.pearsonr(predicted, observed).statistic),
        "sign_acc": float(np.mean(np.sign(predicted) == np.sign(observed))),
        "mae": float(np.mean(np.abs(predicted - observed))),
        "null_mean": null_mean,
        "null_sd": null_sd,
        "z_vs_null": float((spearman - null_mean) / null_sd) if null_sd > 0 else np.nan,
    }


def pooled_effects(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """One row per (perturbation, protein): the mean over replicates.

    Equal weight per replicate, matching how the replicate correlations are computed, so
    a perturbation with more cells in one batch does not quietly become that batch's
    estimate and the pooled and replicate-level rows stay on the same footing.
    """
    return frame.groupby(["perturbation", "protein"], as_index=False)[columns].mean()


def print_gate_table(title: str, scores: dict[str, dict]) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    print(f"  {'predictor':<16}{'spearman':>10}{'null':>16}{'z':>8}{'sign':>8}"
          f"{'MAE':>10}{'n':>6}")
    for name, score in scores.items():
        null = ("       n/a      " if np.isnan(score["null_mean"])
                else f"{score['null_mean']:+.3f}+/-{score['null_sd']:.3f}")
        spearman = "     n/a" if np.isnan(score["spearman"]) else f"{score['spearman']:+.3f}"
        z = "     n/a" if np.isnan(score["z_vs_null"]) else f"{score['z_vs_null']:+.1f}"
        sign = "     n/a" if np.isnan(score["sign_acc"]) else f"{score['sign_acc']:.3f}"
        print(f"  {name:<16}{spearman:>10}{null:>16}{z:>8}{sign:>8}"
              f"{score['mae']:>10.4f}{score['n_pairs']:>6}")


def gate_verdict(pooled: dict[str, dict], threshold: float) -> int:
    """PASS if either NT-supervised model is clearly above chance on the pooled effects.

    The spec's rule: if BOTH are random, the dataset or the effect calculation is broken
    and KOT must not be tuned to compensate. This returns a non-zero exit code in that
    case so a pipeline stops here rather than proceeding to train.
    """
    supervised = {name: pooled[name] for name in ("ridge", "mlp") if name in pooled}
    passing = {name: score for name, score in supervised.items()
               if np.isfinite(score["z_vs_null"]) and score["z_vs_null"] >= threshold}
    print("\n" + "=" * 78)
    print(f"GATE VERDICT — pass needs z >= {threshold:.1f} vs the permutation null")
    print("=" * 78)
    for name, score in supervised.items():
        mark = "PASS" if name in passing else "fail"
        print(f"  {name:<16}z={score['z_vs_null']:+.1f}  spearman={score['spearman']:+.3f}"
              f"   {mark}")
    if passing:
        print("\n  PASS: KO RNA carries information about these four protein responses.")
        print("  This does NOT mean KOT will find it -- these models were handed the NT")
        print("  pairing that KOT is denied. It means the evaluation is not broken.")
        return 0
    print("\n  STOP: both NT-supervised models are at chance. Debug the dataset and the")
    print("  effect calculation before training KOT. Do not tune KOT against this.")
    return 1


def replicate_effect_correlations(effects: pd.DataFrame) -> pd.DataFrame:
    """Spearman between replicates over their shared (perturbation, protein) effects.

    This is the reproducibility of the thing being predicted. A model cannot beat it, so
    it belongs in the report before any model number does.
    """
    wide = effects.pivot_table(index=["perturbation", "protein"], columns="replicate",
                               values="delta_protein")
    replicates = list(wide.columns)
    rows = []
    for i, first in enumerate(replicates):
        for second in replicates[i + 1:]:
            both = wide[[first, second]].dropna()
            if len(both) < 3:
                continue
            rho = stats.spearmanr(both[first], both[second])
            rows.append({"replicate_a": first, "replicate_b": second, "n_effects": len(both),
                         "spearman": float(rho.statistic), "pvalue": float(rho.pvalue)})
    return pd.DataFrame(rows)


def report(effects: pd.DataFrame, groups: pd.Series, replicates: pd.Series,
           usable: list[str]) -> None:
    knockouts = sorted(set(groups.unique()) - {CONTROL_LABEL})
    per_perturbation = groups[groups != CONTROL_LABEL].value_counts()

    print("\n" + "=" * 78)
    print("1. DESIGN")
    print("=" * 78)
    print(f"  perturbations                : {len(knockouts)}")
    print(f"  control ({CONTROL_LABEL}) cells            : {int((groups == CONTROL_LABEL).sum())}")
    print(f"  replicates                   : {replicates.nunique()} "
          f"({', '.join(sorted(replicates.unique()))})")
    print(f"  cells per perturbation       : median {int(per_perturbation.median())}, "
          f"range {int(per_perturbation.min())}-{int(per_perturbation.max())}")

    spread = (groups[groups != CONTROL_LABEL]
              .groupby(groups[groups != CONTROL_LABEL])
              .apply(lambda s: replicates.loc[s.index].nunique()))
    print(f"  replicates per perturbation  : median {int(spread.median())}, "
          f"range {int(spread.min())}-{int(spread.max())}")

    print("\n" + "=" * 78)
    print(f"2. FROZEN THRESHOLD — >={MIN_KO_CELLS} KO cells in >={MIN_REPLICATES} replicates")
    print("=" * 78)
    print(f"  usable perturbations         : {len(usable)}/{len(knockouts)}")
    dropped = [k for k in knockouts if k not in usable]
    if dropped:
        print(f"  dropped                      : {dropped}")
        for knockout in dropped:
            counts = replicates[(groups == knockout).to_numpy()].value_counts()
            print(f"      {knockout}: cells per replicate {counts.to_dict()}")

    print("\n" + "=" * 78)
    print("3. REPLICATE REPRODUCIBILITY — the ceiling any model is measured against")
    print("=" * 78)
    correlations = replicate_effect_correlations(effects[effects["perturbation"].isin(usable)])
    if correlations.empty:
        print("  no replicate pair shares enough effects to correlate")
    for row in correlations.itertuples():
        print(f"  {row.replicate_a} vs {row.replicate_b}: spearman={row.spearman:+.3f} "
              f"over {row.n_effects} (perturbation, protein) effects  (p={row.pvalue:.2e})")

    print("\n" + "=" * 78)
    print("4. EFFECT COUNTS")
    print("=" * 78)
    scored = effects[effects["perturbation"].isin(usable)].copy()
    scored["is_self"] = [
        (perturbation, protein) in SELF_PAIRS
        for perturbation, protein in zip(scored["perturbation"], scored["protein"])
    ]
    trans = scored[~scored["is_self"]]
    self_effects = scored[scored["is_self"]]
    print(f"  usable trans effects (replicate-level) : {len(trans)}")
    print(f"  usable trans (perturbation, protein)   : "
          f"{trans.groupby(['perturbation', 'protein']).ngroups}")
    print(f"  self effects (replicate-level)         : {len(self_effects)}")
    print(f"  self (perturbation, protein) pairs     : "
          f"{sorted(set(zip(self_effects['perturbation'], self_effects['protein'])))}")


def effects_main() -> None:
    parser = argparse.ArgumentParser(description="Observed CRISPR response table.")
    parser.add_argument("--protein", type=Path,
                        default=PROTEIN_H5AD)
    parser.add_argument("--rna", type=Path,
                        default=RNA_H5AD)
    parser.add_argument("--metadata", type=Path, default=METADATA_TSV)
    parser.add_argument("--out", type=Path,
                        default=EFFECTS_CSV)
    parser.add_argument("--normalization", default="rna_size",
                        choices=["rna_size", "raw_log1p", "clr"],
                        help="must match the benchmark's --normalization")
    args = parser.parse_args()

    protein = load_protein(args.protein)
    meta = load_metadata(args.metadata)
    expression = load_target_gene_expression(args.rna, list(ADT_GENE_MAP.values()))

    cells = protein.obs_names.intersection(meta.index).intersection(expression.index)
    print(f"[join] protein n meta n rna = {len(cells)} cells")
    protein, meta, expression = protein[cells].copy(), meta.loc[cells], expression.loc[cells]

    adt_names = [str(name) for name in protein.var_names]
    groups = meta["gene"].astype(str)
    replicates = meta[REPLICATE_COLUMN].astype(str)
    values = normalize_adt(to_dense(protein.X).astype(np.float64),
                           np.asarray(meta["nCount_RNA"], dtype=np.float64))[args.normalization]

    effects = replicate_effects(values, adt_names, expression, ADT_GENE_MAP,
                                groups, replicates, CONTROL_LABEL)
    usable = sufficient_knockouts(groups, replicates, CONTROL_LABEL,
                                  MIN_KO_CELLS, MIN_REPLICATES)
    # The rule travels with the table: a reader should not have to re-derive which rows
    # the thresholds admit, and no downstream script should re-implement it.
    effects["passes_threshold"] = (
        effects["perturbation"].isin(usable) & (effects["n_ko_cells"] >= MIN_KO_CELLS)
    )
    effects["is_self_effect"] = [
        (perturbation, protein_name) in SELF_PAIRS
        for perturbation, protein_name in zip(effects["perturbation"], effects["protein"])
    ]

    report(effects, groups, replicates, usable)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    effects.to_csv(args.out, index=False)
    print(f"\n[out] {len(effects)} rows -> {args.out}")


def gate_main() -> None:
    parser = argparse.ArgumentParser(
        description="Sanity gate: is KO RNA predictive of these protein responses at all?")
    parser.add_argument("--effects", type=Path,
                        default=EFFECTS_CSV,
                        help="written by `run_kot_crispr.py effects`")
    parser.add_argument("--protein", type=Path,
                        default=PROTEIN_H5AD)
    parser.add_argument("--rna", type=Path,
                        default=NT_ONLY_H5AD)
    parser.add_argument("--metadata", type=Path, default=METADATA_TSV)
    parser.add_argument("--rna-layer", default="Ms",
                        help="the feature space KOT trains on, so the gate tests the same input")
    parser.add_argument("--out-dir", type=Path, default=CRISPR_DIR)
    parser.add_argument("--normalization", default="rna_size",
                        choices=["rna_size", "raw_log1p", "clr"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--z-threshold", type=float, default=3.0,
                        help="standard deviations above the permutation null to count as a pass")
    args = parser.parse_args()

    observed = pd.read_csv(args.effects)
    usable = observed[observed["passes_threshold"] & ~observed["is_self_effect"]].copy()
    print(f"[gate] {len(usable)} usable trans effects over "
          f"{usable['perturbation'].nunique()} perturbations, "
          f"{usable['protein'].nunique()} proteins")

    protein = load_protein(args.protein)
    meta = load_metadata(args.metadata)
    rna = load_rna_layer(args.rna, args.rna_layer)
    print(f"[gate] RNA features: {rna.shape} from layer '{args.rna_layer}'")

    cells = protein.obs_names.intersection(meta.index).intersection(rna.obs_names)
    protein, meta, rna = protein[cells].copy(), meta.loc[cells], rna[cells].copy()
    adt_names = [str(name) for name in protein.var_names]
    groups = meta["gene"].astype(str)
    replicates = meta[REPLICATE_COLUMN].astype(str)

    # The models are fitted against the SAME ADT units the observed table is in, so the
    # predicted delta needs no rescaling before it is compared with the observed one.
    values = normalize_adt(to_dense(protein.X).astype(np.float64),
                           np.asarray(meta["nCount_RNA"], dtype=np.float64))[args.normalization]
    protein.X = values.astype(np.float32)

    fit_rows = np.nonzero((groups == CONTROL_LABEL).to_numpy())[0]
    print(f"[gate] training ridge and MLP on {len(fit_rows)} paired {CONTROL_LABEL} cells")
    predictions = supervised_cell_predictions(rna, protein, fit_rows, args.rna_layer, args.seed)

    # No cognate RNA is needed for the PREDICTED deltas: they are read from the models'
    # per-cell protein. The observed table already carries delta_cognate_rna, which is
    # where the cognate-mRNA passthrough reference comes from.
    empty_rna = pd.DataFrame(index=pd.Index(cells))
    predicted_tables = {
        name: replicate_effects(matrix, adt_names, empty_rna, ADT_GENE_MAP,
                                groups, replicates, CONTROL_LABEL)
        for name, matrix in predictions.items()
    }

    merged = usable[["perturbation", "replicate", "protein", "delta_protein",
                     "delta_cognate_rna"]].copy()
    merged["zero_change"] = 0.0
    merged["cognate_mrna"] = merged["delta_cognate_rna"]
    for name, table in predicted_tables.items():
        merged = merged.merge(
            table[["perturbation", "replicate", "protein", "delta_protein"]]
            .rename(columns={"delta_protein": name}),
            on=["perturbation", "replicate", "protein"], how="left")

    replicate_scores = {
        name: score_predictor(merged[name].to_numpy(),
                              merged["delta_protein"].to_numpy(), args.seed)
        for name in GATE_PREDICTORS
    }
    pooled_observed = pooled_effects(merged, ["delta_protein"])
    pooled_scores = {}
    for name in GATE_PREDICTORS:
        pooled = pooled_effects(merged, [name]).merge(
            pooled_observed, on=["perturbation", "protein"])
        pooled_scores[name] = score_predictor(pooled[name].to_numpy(),
                                              pooled["delta_protein"].to_numpy(), args.seed)

    print_gate_table("REPLICATE-LEVEL EFFECTS — one row per (perturbation, replicate, protein)",
                     replicate_scores)
    print_gate_table("POOLED OVER REPLICATES — one row per (perturbation, protein)",
                     pooled_scores)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out_dir / "crispr_gate_predictions.csv", index=False)
    summary = pd.concat([
        pd.DataFrame(replicate_scores).T.assign(level="replicate"),
        pd.DataFrame(pooled_scores).T.assign(level="pooled"),
    ]).rename_axis("predictor").reset_index()
    summary.to_csv(args.out_dir / "crispr_gate_summary.csv", index=False)
    print(f"\n[out] {args.out_dir}/crispr_gate_predictions.csv, crispr_gate_summary.csv")

    raise SystemExit(gate_verdict(pooled_scores, args.z_threshold))


def preprocess_main() -> None:
    """Build the NT-only velocity cache, written BESIDE the existing one, never over it.

    A separate path keeps every finished run reproducible and makes the leakage question
    answerable by running the same model on both caches.
    """
    parser = argparse.ArgumentParser(
        description="Velocity preprocessing fitted on NT cells only.")
    parser.add_argument("--rna", type=Path,
                        default=RNA_H5AD)
    parser.add_argument("--metadata", type=Path, default=METADATA_TSV)
    parser.add_argument("--retain-genes-csv", type=Path,
                        default=Path("cache/results/mapping/adt_mapping_papalexi.csv"))
    parser.add_argument("--out", type=Path,
                        default=NT_ONLY_H5AD)
    parser.add_argument("--n-top-genes", type=int, default=2000)
    parser.add_argument("--hvg-flavor", default="seurat_v3")
    parser.add_argument("--min-shared-counts", type=int, default=20)
    parser.add_argument("--n-pcs", type=int, default=30)
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--dynamics-n-jobs", type=int, default=8)
    args = parser.parse_args()

    adata = ad.read_h5ad(args.rna)
    adata.obs_names = adata.obs_names.astype(str)
    meta = load_metadata(args.metadata)
    shared = adata.obs_names.intersection(meta.index)
    adata = adata[shared].copy()
    meta = meta.loc[shared]
    groups = meta["gene"].astype(str)
    adata.obs["gene"] = groups.to_numpy()
    adata.obs[REPLICATE_COLUMN] = meta[REPLICATE_COLUMN].astype(str).to_numpy()

    fit_mask = (groups == CONTROL_LABEL).to_numpy()
    print(f"[preprocess] {adata.n_obs} cells, {int(fit_mask.sum())} {CONTROL_LABEL} controls "
          f"fitted, {int((~fit_mask).sum())} knockout cells transformed only")

    # Same source of truth and same column the existing velocity pipeline uses, so the
    # NT-only cache force-retains exactly the ADT target genes the current one does.
    retain_genes = sorted({record["gene_symbol"]
                           for record in load_mapping_records(args.retain_genes_csv)
                           if record["gene_symbol"]})
    print(f"[preprocess] force-retaining ADT target genes: {retain_genes}")

    processed = preprocess_nt_only(
        adata, fit_mask,
        n_top_genes=args.n_top_genes, hvg_flavor=args.hvg_flavor,
        min_shared_counts=args.min_shared_counts, n_pcs=args.n_pcs,
        n_neighbors=args.n_neighbors, dynamics_n_jobs=args.dynamics_n_jobs,
        retain_genes=retain_genes,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    processed.write_h5ad(args.out)
    print(f"\n[out] {processed.shape} -> {args.out}")
    for key, value in processed.uns["nt_only_preprocessing"].items():
        print(f"  {key}: {value}")


def group_means(matrix: torch.Tensor, mask: np.ndarray) -> torch.Tensor:
    return matrix[torch.as_tensor(mask, device=matrix.device)].mean(dim=0, keepdim=True)


def benchmark_units_adt(protein_path: Path, metadata_path: Path, normalization: str,
                        obs_names) -> np.ndarray:
    """Observed ADT in the units the effects table uses, row-aligned to `obs_names`."""
    protein = load_protein(protein_path)
    protein.obs_names = protein.obs_names.astype(str)
    meta = load_metadata(metadata_path)
    rows = pd.Index([str(name) for name in obs_names])
    protein, meta = protein[rows].copy(), meta.loc[rows]
    return normalize_adt(to_dense(protein.X).astype(np.float64),
                         np.asarray(meta["nCount_RNA"], dtype=np.float64))[normalization]


def calibrated_phi(model, features: torch.Tensor, observed: np.ndarray,
                   control_mask: np.ndarray) -> np.ndarray:
    """phi for every cell rescaled onto the observed table's units, and the scale used.

    A residual per-protein correction, not a units conversion: phi now trains against the
    same normalisation the benchmark scores (protein_target_normalization), so this only
    absorbs the scale a distribution match leaves behind. It cannot rescue a phi trained
    in another normalisation -- CLR is a per-cell row operation and no per-protein factor
    inverts it, which is what check_checkpoint_units refuses. The scale is
    fitted on CONTROL cells only, is strictly POSITIVE, and carries no intercept: an
    intercept cancels in a difference, and a signed slope would let a model that predicts
    a protein backwards be flipped the right way round for free. The competitors go
    through the identical step, so the two are on the same footing.
    """
    with torch.no_grad():
        predicted = model.phi(features).cpu().numpy().astype(np.float64)
    scale = positive_scale(predicted, observed.astype(np.float64),
                           np.nonzero(control_mask)[0], MIN_CALIBRATION_SD_RATIO)
    return predicted * scale, scale


def perturbation_predictions(model, features: torch.Tensor, groups: pd.Series,
                             replicates: pd.Series, adt_names: list[str],
                             min_cells: int, phi_values: np.ndarray,
                             phi_slope: np.ndarray) -> pd.DataFrame:
    """The two predictions this benchmark asks a checkpoint for, per (perturbation, replicate).

    delta_p_phi       mean phi(r_KO) - mean phi(r_NT). The full nonlinear response of the
                      learned map, which is what the benchmark scores.
    delta_p_jacobian  J_phi(mean r_NT) . (mean r_KO - mean r_NT). The LOCAL linear response
                      of the same map, from one forward-mode pass. It tests the map's
                      DIFFERENTIAL directly under a real intervention, which is the object
                      KOT's ODE residual actually constrains -- phi can score well while
                      its Jacobian is wrong, and only this separates the two.

    Both are taken within a replicate against that replicate's own controls, so a batch
    shift shared by a knockout and its controls cancels in each.
    """
    is_control = (groups == CONTROL_LABEL).to_numpy()
    rows = []
    for replicate in sorted(replicates.unique()):
        in_replicate = (replicates == replicate).to_numpy()
        control_mask = is_control & in_replicate
        if control_mask.sum() < 2:
            continue
        control_r = group_means(features, control_mask)
        control_phi = phi_values[control_mask].mean(axis=0)
        for target in sorted(set(groups.unique()) - {CONTROL_LABEL}):
            ko_mask = (groups == target).to_numpy() & in_replicate
            if ko_mask.sum() < min_cells:
                continue
            phi_shift = phi_values[ko_mask].mean(axis=0) - control_phi
            delta_r = group_means(features, ko_mask) - control_r
            jacobian_shift = (jacobian_vector_product(model.phi, control_r, delta_r)
                              .squeeze(0).detach().cpu().numpy() * phi_slope)
            for j, adt in enumerate(adt_names):
                rows.append({
                    "perturbation": target,
                    "replicate": str(replicate),
                    "protein": adt,
                    "n_ko_cells": int(ko_mask.sum()),
                    "delta_p_phi": float(phi_shift[j]),
                    "delta_p_jacobian": float(jacobian_shift[j]),
                })
    return pd.DataFrame(rows)


def energy_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Multivariate energy distance, normalised by the reference cloud's own spread.

    E = 2E|X-Y| - E|X-X'| - E|Y-Y'|, divided by E|Y-Y'| so it is scale-free: 0 means the
    two clouds are the same distribution, 1 means they are as far apart as the reference
    is wide. It is sensitive to location, scale and shape of the joint cloud, and WEAKLY
    sensitive to the between-protein correlation structure: independently permuting each
    protein of a real sample destroys every correlation and still scores ~0.01 here.
    `correlation_gap` is what covers that.
    """
    cross = float(np.mean(np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)))
    within_a = float(np.mean(np.linalg.norm(a[:, None, :] - a[None, :, :], axis=-1)))
    within_b = float(np.mean(np.linalg.norm(b[:, None, :] - b[None, :, :], axis=-1)))
    if within_b <= 0:
        return float("nan")
    return (2.0 * cross - within_a - within_b) / within_b


def correlation_gap(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute difference between the two protein-protein correlation matrices.

    The dependence structure energy_distance cannot see. A phi that reproduces each
    protein's margin while getting their joint behaviour wrong is not reproducing the
    panel, and on a 4-plex panel that is most of what there is to get right.
    """
    upper = np.triu_indices(a.shape[1], k=1)
    return float(np.mean(np.abs(np.corrcoef(a, rowvar=False)[upper]
                                - np.corrcoef(b, rowvar=False)[upper])))


def observational_metrics(model, features: torch.Tensor, protein: np.ndarray,
                          control_mask: np.ndarray, adt_names: list[str], seed: int,
                          max_cells: int = 1500) -> pd.DataFrame:
    """How well phi reproduces the OBSERVED protein DISTRIBUTION of the fitted cells.

    Distributional, not per-cell, because KOT is unpaired: it never learns which control
    cell goes with which control protein, so a paired per-cell correlation is a metric it
    structurally cannot win, and scoring it that way made every arm read ~0.02 and emptied
    the observational half of the ablation question. What KOT does claim is that phi(r)
    lands on the protein distribution, and that is what these measure:

      sd_ratio     sd(phi_j) / sd(observed_j). The collapse detector -- a phi pinned to a
                   constant reads 0 here whatever its correlation says.
      mean_gap     |mean(phi_j) - mean(obs_j)| / sd(obs_j). Location match.
      wasserstein  1-D optimal-transport distance per protein, over sd(obs_j). Full
                   marginal match, not just the first two moments.
      panel_energy Energy distance over all four proteins at once: location, scale and
                   shape of the joint cloud.
      panel_corr_gap Mean |difference| between the predicted and observed protein-protein
                   correlations, which energy distance is nearly blind to.

    `obs_spearman` is kept but is PAIRED, so it is fair only between paired methods
    (ridge, MLP, totalVI) and must not be used to rank KOT against them.
    """
    rows_all = np.nonzero(control_mask)[0]
    rng = np.random.default_rng(seed)
    rows = (rng.choice(rows_all, max_cells, replace=False)
            if len(rows_all) > max_cells else rows_all)
    with torch.no_grad():
        predicted = model.phi(
            features[torch.as_tensor(rows, device=features.device)]
        ).cpu().numpy().astype(np.float64)
    observed = protein[rows].astype(np.float64)

    panel_energy = energy_distance(predicted, observed)
    panel_corr = correlation_gap(predicted, observed)
    records = []
    for j, adt in enumerate(adt_names):
        pred, obs = predicted[:, j], observed[:, j]
        sd = float(np.std(obs))
        records.append({
            "protein": adt,
            "n_control_cells": int(len(rows)),
            "sd_ratio": float(np.std(pred) / sd) if sd > 0 else np.nan,
            "mean_gap": float(abs(np.mean(pred) - np.mean(obs)) / sd) if sd > 0 else np.nan,
            "wasserstein": float(stats.wasserstein_distance(pred, obs) / sd) if sd > 0 else np.nan,
            "panel_energy": panel_energy,
            "panel_corr_gap": panel_corr,
            "obs_spearman_paired": float(stats.spearmanr(pred, obs).statistic),
        })
    return pd.DataFrame(records)


def discover_seed_checkpoints(run_dir: Path, checkpoint: str,
                              dataset: str | None = None,
                              model: str | None = None
                              ) -> tuple[list[tuple[Path, int]], str, str]:
    """Find one model/dataset's seed checkpoints below a run root.

    Dataset and model filters are applied before ambiguity checks. Their authoritative
    names come from <root>/<model>/<dataset>/seed_<n>/ rather than the root config, whose
    default model/dataset may differ from the checkpoint being loaded.
    """
    found, datasets, models = [], set(), set()
    for path in sorted(run_dir.glob(f"*/*/seed_*/checkpoint_{checkpoint}.pt")):
        path_dataset = path.parents[1].name
        path_model = path.parents[2].name
        if dataset is not None and path_dataset != dataset:
            continue
        if model is not None and path_model != model:
            continue
        found.append((path, int(path.parent.name.split("_")[-1])))
        datasets.add(path_dataset)
        models.add(path_model)
    if len(datasets) > 1:
        raise SystemExit(
            f"{run_dir} holds checkpoints for {sorted(datasets)}; pass --dataset to say "
            "which one to score.")
    if len(models) > 1:
        raise SystemExit(
            f"{run_dir} holds checkpoints for {sorted(models)}; pass --model to say "
            "which one to score.")
    return (found, datasets.pop() if datasets else "",
            models.pop() if models else "")


def check_checkpoint_units(checkpoints: list[tuple[Path, int]], run_cfg: dict,
                           allow_unstamped: bool) -> None:
    """Refuse a checkpoint whose phi was trained in units this scorer is not scoring.

    `protein_units` is stamped into every checkpoint by run_kot. A CLR checkpoint and an
    rna_size one have identical shapes and load into the same architecture without
    complaint, so nothing else here can tell them apart -- and scoring a CLR phi against
    the rna_size benchmark ranks a units mismatch, not the map. Checkpoints written
    before the stamp existed are CLR; they must be retrained, not re-scored.
    """
    wanted = protein_target_units(run_cfg, run_cfg.get("kot_protein_layer"))
    unstamped, mismatched = [], []
    for path, _ in checkpoints:
        stamped = torch.load(path, map_location="cpu").get("protein_units")
        if stamped is None:
            unstamped.append(path)
        elif stamped != wanted:
            mismatched.append((path, stamped))
    if mismatched:
        first, stamped = mismatched[0]
        raise SystemExit(
            f"{len(mismatched)} checkpoint(s) were trained to predict '{stamped}' but this "
            f"run's config asks for '{wanted}' (e.g. {first}). Retrain them; phi cannot be "
            "rescaled from one to the other.")
    if unstamped and not allow_unstamped:
        raise SystemExit(
            f"{len(unstamped)} checkpoint(s) carry no protein_units stamp, so they predate "
            f"the units fix and were trained on CLR while this config asks for '{wanted}' "
            f"(e.g. {unstamped[0]}). A CLR phi scored against the rna_size benchmark tops "
            "out near spearman +0.37 whatever it learned, and CLR closure inverts three of "
            "the four proteins. Retrain, or pass --allow-unstamped-checkpoints to score "
            "them anyway and label the result as CLR-trained.")
    if unstamped:
        print(f"[predict] WARNING: {len(unstamped)} unstamped (CLR-trained) checkpoint(s) "
              f"scored against '{wanted}' units on request; the result is not comparable "
              "with the NT-supervised baselines.")


def observational_output_path(out: Path) -> Path:
    """A sibling path that cannot silently overwrite the perturbation predictions."""
    if "predictions" in out.stem:
        stem = out.stem.replace("predictions", "observational", 1)
    else:
        stem = f"{out.stem}_observational"
    return out.with_name(stem + out.suffix)

def predict_main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-checkpoint phi and Jacobian perturbation predictions.")
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="a KOT RUN ROOT (the directory holding training_resume_*.yaml)")
    parser.add_argument("--checkpoint", default="best_align")
    parser.add_argument("--dataset", default=None,
                        help="dataset name in the run config; inferred when omitted")
    parser.add_argument("--model", default=None,
                        help="model directory under the run root; inferred when omitted")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="default: every seed found under the run root")
    parser.add_argument("--out", type=Path, default=None,
                        help="default: <run-dir>/crispr_predictions_<checkpoint>.csv")
    parser.add_argument("--protein", type=Path,
                        default=PROTEIN_H5AD)
    parser.add_argument("--metadata", type=Path,
                        default=METADATA_TSV)
    parser.add_argument("--normalization", default="rna_size",
                        choices=["rna_size", "raw_log1p", "clr"],
                        help="must match the observed effects table; phi is calibrated "
                             "onto these units before its deltas are taken")
    parser.add_argument("--device", default="auto")
    # The two control arms randomise S and v per seed, so the checkpoint tool refuses to
    # rebuild them. Neither quantity is READ here: phi and its JVP are functions of r
    # alone, and S and v only enter the ODE right-hand side, which this tool never forms.
    # `real` is therefore safe for those arms and changes nothing for the others -- what it
    # must NOT be taken to mean is that the arm was trained with the true S or velocity.
    parser.add_argument("--mapping", default="as-trained", choices=["as-trained", "real"],
                        help="use `real` for the kot_s_permute (permS) arm")
    parser.add_argument("--velocity", default="as-trained", choices=["as-trained", "real"],
                        help="use `real` for the kot_velocity_shuffle (shuffleVel) arm")
    parser.add_argument("--allow-unstamped-checkpoints", action="store_true",
                        help="score checkpoints written before phi's training units were "
                             "stamped. They are CLR-trained, so their deltas are not "
                             "comparable with the rna_size benchmark or its baselines.")
    args = parser.parse_args()

    checkpoints, found_dataset, found_model = discover_seed_checkpoints(
        args.run_dir, args.checkpoint, dataset=args.dataset, model=args.model)
    if args.seeds is not None:
        checkpoints = [(path, seed) for path, seed in checkpoints if seed in set(args.seeds)]
    if not checkpoints:
        raise SystemExit(
            f"no checkpoint_{args.checkpoint}.pt under {args.run_dir}. Point --run-dir at "
            "the run ROOT (the one holding training_resume_*.yaml), not a seed directory.")

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else ("cpu" if args.device == "auto" else args.device))
    rna_adata, protein_adata, run_cfg, dataset_name = load_dataset_for_run(
        args.run_dir, args.dataset or found_dataset)
    model_name = found_model
    if not model_name:
        raise SystemExit(f"could not infer a model from checkpoints under {args.run_dir}")
    check_checkpoint_units(checkpoints, run_cfg, args.allow_unstamped_checkpoints)
    built = build_model_and_tensors(run_cfg, rna_adata, protein_adata, model_name, device,
                                    velocity_mode=args.velocity, mapping_mode=args.mapping)
    model, features = built[0], built[1]

    groups = rna_adata.obs["gene"].astype(str)
    replicates = rna_adata.obs[REPLICATE_COLUMN].astype(str)
    adt_names = [str(name) for name in protein_adata.var_names]
    print(f"[predict] {dataset_name} / {model_name} / {args.checkpoint}: "
          f"{features.shape[0]} cells, {len(adt_names)} ADTs, "
          f"{len(checkpoints)} seed(s)")

    protein_values = built[3].cpu().numpy()
    control_mask = (groups == CONTROL_LABEL).to_numpy()
    # The benchmark's own ADT units. With protein_target_normalization=rna_size this is
    # the same normalisation phi trained against, so the scale below is a residual
    # correction rather than a units conversion -- which is the point: a per-protein
    # rescale can correct a scale, and cannot undo a per-cell transform like CLR.
    benchmark_adt = benchmark_units_adt(args.protein, args.metadata, args.normalization,
                                        rna_adata.obs_names)

    frames, observational = [], []
    for path, seed in checkpoints:
        state = torch.load(path, map_location="cpu")["state_dict"]
        model.load_state_dict({key: value.to(device) for key, value in state.items()})
        model.eval()
        phi_values, phi_slope = calibrated_phi(model, features, benchmark_adt, control_mask)
        predictions = perturbation_predictions(model, features, groups, replicates,
                                               adt_names, MIN_KO_CELLS, phi_values,
                                               phi_slope)
        predictions["seed"] = seed
        observed = observational_metrics(model, features, protein_values,
                                         control_mask, adt_names, seed)
        observed["seed"] = seed
        observational.append(observed)
        agreement = stats.spearmanr(predictions["delta_p_phi"],
                                    predictions["delta_p_jacobian"],
                                    nan_policy="omit")
        # Never silent: a protein the calibration floor rejected leaves NaN deltas, and a
        # reader comparing arms has to see that this one predicted nothing for it rather
        # than find a smaller n in the summary and have to work out why.
        collapsed = [adt for adt, factor in zip(adt_names, phi_slope)
                     if not np.isfinite(factor)]
        note = f"   COLLAPSED phi, not scored: {collapsed}" if collapsed else ""
        print(f"  seed {seed:<5} {len(predictions)} rows   "
              f"phi vs Jacobian spearman={agreement.statistic:+.3f}   "
              f"NT energy={observed['panel_energy'].iloc[0]:.3f} "
              f"corr_gap={observed['panel_corr_gap'].iloc[0]:.3f} "
              f"sd_ratio={observed['sd_ratio'].mean():.3f}{note}")
        frames.append(predictions)

    combined = pd.concat(frames, ignore_index=True)
    combined["run_dir"] = str(args.run_dir)
    combined["checkpoint"] = args.checkpoint
    combined["dataset"] = dataset_name
    combined["model"] = model_name
    combined["velocity_ablation"] = arm_label(str(args.run_dir), model_name)

    out = args.out or args.run_dir / f"crispr_predictions_{args.checkpoint}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out, index=False)

    # Written beside the perturbation predictions and tagged the same way, so the two
    # halves of item 28's question can be joined on (run_dir, checkpoint, seed).
    obs_table = pd.concat(observational, ignore_index=True)
    for column in ("run_dir", "checkpoint", "dataset", "model"):
        obs_table[column] = combined[column].iloc[0]
    obs_out = observational_output_path(out)
    obs_table.to_csv(obs_out, index=False)
    print(f"[out] {len(combined)} rows over {len(frames)} seed(s) -> {out}")
    print(f"[out] NT observational metrics -> {obs_out}")


BOOTSTRAP_METRIC = "spearman"
ARM_PATTERN = "crisprabl_"


def arm_label(run_dir: str, model: str) -> str:
    """The ablation arm a prediction came from.

    Read from the run directory, because the arm lives in three different config keys
    (kot_velocity_ablation, kot_velocity_shuffle, kot_s_permute) plus the model name, and
    deriving it from any one of them silently merges arms.
    """
    name = Path(str(run_dir)).name
    if ARM_PATTERN in name:
        return name.split(ARM_PATTERN, 1)[1]
    return model


def score_arm_seeds(merged: pd.DataFrame, column: str, n_boot: int,
                    seed: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Per (arm, effect set, seed) metrics, and the arm-level consensus with CIs.

    Seeds are scored SEPARATELY and then summarised, so the spread across seeds is a real
    error bar rather than something hidden by averaging predictions first. The bootstrap
    CI is taken on the arm's consensus prediction (the per-effect mean over seeds), over
    PERTURBATIONS -- 24 experiments, not 20,000 cells.
    """
    per_seed, consensus, breakdown = [], [], {}
    for (arm, effect_set), block in merged.groupby(["arm", "effect_set"]):
        for seed_value, seed_block in block.groupby("seed"):
            pooled = pooled_effects(seed_block, [column, "delta_protein"])
            per_seed.append({"arm": arm, "effect_set": effect_set, "seed": seed_value,
                             "predictor": column,
                             **metric_suite(pooled[column].to_numpy(),
                                            pooled["delta_protein"].to_numpy())})
        agreed = (block.groupby(["perturbation", "replicate", "protein"], as_index=False)
                  [[column, "delta_protein"]].mean())
        pooled = pooled_effects(agreed, [column, "delta_protein"])
        row = {"arm": arm, "effect_set": effect_set, "predictor": column,
               "n_seeds": int(block["seed"].nunique()),
               **metric_suite(pooled[column].to_numpy(), pooled["delta_protein"].to_numpy())}
        row["pert_boot_lo"], row["pert_boot_hi"] = bootstrap_over_units(
            pooled, "perturbation", column, "delta_protein", BOOTSTRAP_METRIC, n_boot, seed)
        row["rep_boot_lo"], row["rep_boot_hi"] = bootstrap_over_units(
            agreed, "replicate", column, "delta_protein", BOOTSTRAP_METRIC, n_boot, seed)
        cosines = response_cosine(pooled, column, "delta_protein")
        row["response_cosine_mean"] = float(cosines["cosine"].mean())
        row["response_cosine_median"] = float(cosines["cosine"].median())
        consensus.append(row)

        # The three per-set tables item 27 asks to be SAVED rather than summarised: which
        # proteins the model gets right, which perturbations, and the response shape of
        # each one. A pooled correlation hides all three.
        tag = f"{column}_{arm}_{effect_set}"
        breakdown[f"{tag}_per_protein"] = per_group_correlation(
            pooled, "protein", column, "delta_protein")
        breakdown[f"{tag}_per_perturbation"] = per_group_correlation(
            agreed, "perturbation", column, "delta_protein")
        breakdown[f"{tag}_response_cosine"] = cosines
    return pd.DataFrame(per_seed), pd.DataFrame(consensus), breakdown


def print_arm_table(consensus: pd.DataFrame, per_seed: pd.DataFrame,
                    observational: pd.DataFrame | None) -> None:
    """Item 28's question in one table: same observational quality, different perturbational?"""
    for effect_set in EFFECT_SETS:
        block = consensus[consensus["effect_set"] == effect_set]
        if block.empty:
            continue
        print("\n" + "=" * 96)
        print(f"{'PRIMARY' if effect_set == 'primary' else 'SECONDARY: ' + effect_set.upper()}"
              f" — {int(block['n'].max())} effects")
        print("=" * 96)
        print(f"  {'arm':<12}{'predictor':<18}{'NT energy':>10}{'corr gap':>9}{'sd_ratio':>9}"
              f"{'spearman':>10}{'+/- seed':>10}{'95% CI (pert)':>18}{'NRMSE':>8}"
              f"{'sign':>7}{'cos':>7}")
        for row in block.sort_values(["predictor", "arm"]).itertuples():
            spread = per_seed[(per_seed["arm"] == row.arm)
                              & (per_seed["effect_set"] == effect_set)
                              & (per_seed["predictor"] == row.predictor)]["spearman"].std()
            energy, corr_gap, sd_ratio = "n/a", "n/a", "n/a"
            if observational is not None:
                match = observational[observational["arm"] == row.arm]
                if len(match):
                    energy = f"{match['panel_energy'].mean():.3f}"
                    corr_gap = f"{match['panel_corr_gap'].mean():.3f}"
                    sd_ratio = f"{match['sd_ratio'].mean():.3f}"
            ci = (f"[{row.pert_boot_lo:+.2f},{row.pert_boot_hi:+.2f}]"
                  if np.isfinite(row.pert_boot_lo) else "       n/a       ")
            spearman = "     n/a" if not np.isfinite(row.spearman) else f"{row.spearman:+.3f}"
            print(f"  {row.arm:<12}{row.predictor:<18}{energy:>10}{corr_gap:>9}{sd_ratio:>9}"
                  f"{spearman:>10}{spread:>10.3f}{ci:>18}{row.nrmse:>8.3f}"
                  f"{row.sign_acc:>7.3f}{row.response_cosine_mean:>7.3f}")


SPEC_EFFECT_COLUMNS = ["perturbation", "replicate", "protein", "model",
                       "velocity_ablation", "delta_p_observed", "delta_p_phi",
                       "delta_p_jacobian", "delta_cognate_rna", "is_self",
                       "is_posttranscriptional", "n_cells"]

SPEC_METRICS = ["spearman", "pearson", "mae", "nrmse", "sign_acc"]


def write_effects_predicted(merged: pd.DataFrame, out_dir: Path) -> Path:
    """crispr_effects_predicted.csv, one row per scored (perturbation, replicate, protein).

    Every prediction beside the observation it is judged against and the two flags that
    say which subset it belongs to, so the per-effect record is readable without joining
    anything back to the observed table.
    """
    table = merged.rename(columns={
        "delta_protein": "delta_p_observed",
        "in_self": "is_self",
        "in_post_transcriptional": "is_posttranscriptional",
        "n_ko_cells": "n_cells",
    })
    table = table.drop_duplicates(
        subset=["perturbation", "replicate", "protein", "model", "velocity_ablation", "seed"])
    out = out_dir / "crispr_effects_predicted.csv"
    table.reindex(columns=SPEC_EFFECT_COLUMNS + ["seed"]).to_csv(out, index=False)
    return out


def write_summary(consensus: pd.DataFrame, out_dir: Path) -> Path:
    """crispr_summary.csv, phi and Jacobian side by side on one row per model+condition.

    The per-arm table carries one row PER PREDICTOR; the spec wants them paired, because
    the question it is built to answer -- does the differential of the map beat the map --
    is only readable when both sit on the same line.
    """
    wide = consensus.pivot_table(
        index=["model", "condition", "effect_subset"],
        columns="predictor", values=SPEC_METRICS + ["n"], aggfunc="first")
    wide.columns = [f"{metric}_{predictor.replace('delta_p_', '')}"
                    for metric, predictor in wide.columns]
    wide = wide.reset_index().rename(columns={"sign_acc_phi": "sign_phi",
                                              "sign_acc_jacobian": "sign_jacobian"})
    wide["n_effects"] = wide.get("n_phi", wide.get("n_jacobian"))
    columns = ["model", "condition", "effect_subset", "n_effects"] + \
        [f"{m}_{p}" for p in ("phi", "jacobian")
         for m in ("spearman", "pearson", "mae", "nrmse", "sign")]
    out = out_dir / "crispr_summary.csv"
    wide.reindex(columns=columns).to_csv(out, index=False)
    return out


def evaluate_main() -> None:
    parser = argparse.ArgumentParser(
        description="Metrics per ablation arm over the four CRISPR effect sets.")
    parser.add_argument("--effects", type=Path,
                        default=EFFECTS_CSV)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--observational", type=Path, nargs="*", default=None,
                        help="crispr_observational_*.csv for the same checkpoints; joined "
                             "on the arm so observational and perturbational quality sit "
                             "in one table")
    parser.add_argument("--columns", nargs="+", default=["delta_p_phi", "delta_p_jacobian"])
    parser.add_argument("--out-dir", type=Path, default=Path("cache/results/crispr/evaluation"))
    parser.add_argument("--perturbation-subset", type=Path, default=None,
                        help="CSV with knockout,status; confirmed rows are scored as a "
                             "separate MultiPert-compat evaluation "
                             f"(example: {MULTIPERT_SUBSET_CSV})")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    observed = pd.read_csv(args.effects)
    if args.perturbation_subset is not None:
        keep, skipped = confirmed_knockouts(args.perturbation_subset)
        print(f"[evaluate] MultiPert-compat subset: {len(keep)} confirmed, "
              f"excluded {skipped}")
        observed = observed[observed["perturbation"].isin(keep)].copy()
    observed = annotate_effect_sets(observed)
    print(f"[evaluate] effect sets from {args.effects}")
    for effect_set in EFFECT_SETS:
        member = observed[observed[f"in_{effect_set}"]]
        print(f"  {effect_set:<22}{member.groupby(['perturbation', 'protein']).ngroups:>4} "
              "(perturbation, protein) pairs")
    print(f"  post-transcriptional cuts: |delta_protein| >= "
          f"{observed.attrs['post_transcriptional_protein_cut']:.4f} and "
          f"|delta_cognate_rna| <= {observed.attrs['post_transcriptional_rna_cut']:.4f}")

    predictions = pd.concat([pd.read_csv(path) for path in args.predictions],
                            ignore_index=True)
    predictions["arm"] = [arm_label(r, m) for r, m in
                          zip(predictions["run_dir"], predictions["model"])]
    keys = ["perturbation", "replicate", "protein"]
    carried = ["arm", "seed", "model", "velocity_ablation"] + list(args.columns)
    merged = observed.merge(predictions[keys + carried], on=keys, how="inner")
    per_effect = merged.copy()
    # One row per (effect set, arm, seed, effect); an arm is never pooled with another.
    merged = pd.concat([merged[merged[f"in_{name}"]].assign(effect_set=name)
                        for name in EFFECT_SETS], ignore_index=True)
    print(f"[evaluate] arms: {sorted(predictions['arm'].unique())}")

    observational = None
    if args.observational:
        observational = pd.concat([pd.read_csv(path) for path in args.observational],
                                  ignore_index=True)
        observational["arm"] = [arm_label(r, m) for r, m in
                                zip(observational["run_dir"], observational["model"])]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    breakdown_dir = args.out_dir / "breakdown"
    breakdown_dir.mkdir(exist_ok=True)
    seed_rows, consensus_rows = [], []
    for column in args.columns:
        per_seed, consensus, breakdown = score_arm_seeds(merged, column, args.n_boot,
                                                         args.seed)
        seed_rows.append(per_seed)
        consensus_rows.append(consensus)
        for name, frame in breakdown.items():
            frame.to_csv(breakdown_dir / f"{name}.csv", index=False)
    per_seed = pd.concat(seed_rows, ignore_index=True)
    consensus = pd.concat(consensus_rows, ignore_index=True)
    per_seed.to_csv(args.out_dir / "crispr_metrics_per_seed.csv", index=False)
    consensus.to_csv(args.out_dir / "crispr_metrics_summary.csv", index=False)
    if observational is not None:
        observational.to_csv(args.out_dir / "crispr_observational.csv", index=False)
    merged.to_csv(args.out_dir / "crispr_effects_scored.csv", index=False)

    spec = consensus.rename(columns={"arm": "condition", "effect_set": "effect_subset"})
    spec["model"] = spec["condition"].map(
        merged.drop_duplicates("arm").set_index("arm")["model"])
    effects_path = write_effects_predicted(per_effect, args.out_dir)
    summary_path = write_summary(spec, args.out_dir)

    print_arm_table(consensus, per_seed, observational)
    print(f"\n[out] {args.out_dir}/crispr_metrics_summary.csv (+ per_seed, observational)")
    print(f"[out] {effects_path}")
    print(f"[out] {summary_path}")
    print(f"[out] {breakdown_dir}/ per-protein, per-perturbation and response-cosine tables")


def competitor_deltas(predicted: np.ndarray, observed: np.ndarray, adt_names: list[str],
                      groups: pd.Series, replicates: pd.Series,
                      control_mask: np.ndarray) -> pd.DataFrame:
    """One competitor's per-replicate predicted deltas, in the observed table's units.

    The per-protein scale is fitted on CONTROL cells only and is strictly positive, so
    it corrects units without granting a sign flip. KOT goes through the identical step.
    """
    rows = np.nonzero(control_mask)[0]
    scale = positive_scale(predicted.astype(np.float64), observed.astype(np.float64), rows,
                           MIN_CALIBRATION_SD_RATIO)
    calibrated = predicted * scale
    empty_rna = pd.DataFrame(index=groups.index)
    return replicate_effects(calibrated, adt_names, empty_rna, ADT_GENE_MAP,
                             groups, replicates, CONTROL_LABEL)


def competitors_main() -> None:
    parser = argparse.ArgumentParser(
        description="Task A competitors in the same delta format as `predict`.")
    parser.add_argument("--predictions-dir", type=Path, default=Path("data/predictions"),
                        help="holds {model}_{dataset}_seed_{seed}.h5ad")
    parser.add_argument("--dataset", default="papalexi_nt_only")
    parser.add_argument("--models", nargs="+",
                        default=["ridge", "mlp", "scipenn", "scbutterfly", "totalvi"])
    parser.add_argument("--effects", type=Path,
                        default=EFFECTS_CSV,
                        help="supplies the cognate-mRNA passthrough reference")
    parser.add_argument("--multipert", type=Path,
                        default=Path("cache/results/multipert_pair_deltas.csv"))
    parser.add_argument("--protein", type=Path,
                        default=PROTEIN_H5AD)
    parser.add_argument("--metadata", type=Path, default=METADATA_TSV)
    parser.add_argument("--normalization", default="rna_size",
                        choices=["rna_size", "raw_log1p", "clr"])
    parser.add_argument("--out", type=Path,
                        default=Path("cache/results/crispr/crispr_predictions_competitors.csv"))
    args = parser.parse_args()

    protein = load_protein(args.protein)
    meta = load_metadata(args.metadata)
    cells = protein.obs_names.intersection(meta.index)
    protein, meta = protein[cells].copy(), meta.loc[cells]
    adt_names = [str(name) for name in protein.var_names]
    observed = normalize_adt(to_dense(protein.X).astype(np.float64),
                             np.asarray(meta["nCount_RNA"], dtype=np.float64))[args.normalization]
    groups = meta["gene"].astype(str)
    replicates = meta[REPLICATE_COLUMN].astype(str)
    control_mask = (groups == CONTROL_LABEL).to_numpy()

    frames = []
    for model in args.models:
        paths = sorted(args.predictions_dir.glob(f"{model}_{args.dataset}*.h5ad"))
        if not paths:
            print(f"[competitors] {model}: no prediction h5ad found, skipping")
            continue
        for path in paths:
            adata = ad.read_h5ad(path)
            adata.obs_names = adata.obs_names.astype(str)
            shared = pd.Index(cells).intersection(adata.obs_names)
            aligned = np.asarray(adata[shared].obsm["X_aligned_rna"], dtype=np.float32)
            if aligned.shape[1] != len(adt_names):
                print(f"[competitors] {path.name}: {aligned.shape[1]} columns against "
                      f"{len(adt_names)} ADTs — not a direct predictor, skipping")
                break
            order = pd.Index(cells).get_indexer(shared)
            table = competitor_deltas(aligned, observed[order], adt_names,
                                      groups.iloc[order], replicates.iloc[order],
                                      control_mask[order])
            seed = path.stem.split("_seed_")[-1] if "_seed_" in path.stem else "0"
            table["arm"] = model
            table["seed"] = int(seed)
            frames.append(table)
        print(f"[competitors] {model}: {len(paths)} file(s)")

    # The two mandatory simple references. Neither needs a model, and both are computed
    # from the observed table so they cannot drift from the effects they are compared to.
    effects = pd.read_csv(args.effects)
    for arm, column in (("zero_change", None), ("cognate_mrna", "delta_cognate_rna")):
        reference = effects[["perturbation", "replicate", "protein", "n_ko_cells"]].copy()
        reference["delta_protein"] = 0.0 if column is None else effects[column]
        reference["arm"], reference["seed"] = arm, 0
        frames.append(reference)
        print(f"[competitors] {arm}: {len(reference)} rows")

    if args.multipert.exists():
        # MultiPert predicts one delta per (perturbation, protein); broadcast it to the
        # replicates so it joins, and label the regime it was trained under.
        pairs = pd.read_csv(args.multipert).rename(columns={"knockout": "perturbation",
                                                            "adt": "protein"})
        keys = effects[["perturbation", "replicate", "protein", "n_ko_cells"]]
        merged = keys.merge(pairs[["perturbation", "protein", "pred_delta"]],
                            on=["perturbation", "protein"], how="inner")
        merged = merged.rename(columns={"pred_delta": "delta_protein"})
        merged["arm"], merged["seed"] = "multipert_paired_all_cells", 0
        frames.append(merged)
        print(f"[competitors] multipert: {len(merged)} rows [paired_all_cells regime]")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.rename(columns={"delta_protein": "delta_p_phi"})
    combined["delta_p_jacobian"] = np.nan          # only KOT has a differential to take
    combined["run_dir"] = ""
    combined["model"] = combined["arm"]
    combined["velocity_ablation"] = "none"      # competitors have no velocity term
    combined["checkpoint"] = "competitor"
    combined["dataset"] = args.dataset
    args.out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.out, index=False)
    print(f"\n[out] {len(combined)} rows, arms {sorted(combined['arm'].unique())} -> {args.out}")


COMMANDS = {"effects": effects_main, "gate": gate_main, "preprocess": preprocess_main,
            "predict": predict_main, "competitors": competitors_main,
            "evaluate": evaluate_main}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        print(f"commands: {', '.join(COMMANDS)}")
        return 2
    COMMANDS[sys.argv.pop(1)]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
