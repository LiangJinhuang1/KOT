"""Score Papalexi runs on CRISPR perturbation-response, not FOSCTTM.

Paper protocol:

  train     φ is fitted on NT cells only (fit_obs_key=gene, fit_obs_values=[NT]).
            Knockout transcriptomes are held out of the loss.
  predict   KOT: Δ from φ(r) directly (feature-space ADT map).
            Latent baselines: kNN imputation from NT neighbours only, so the
            protein atlas cannot leak the knockout protein state.
  contrast  delta(knockout, protein) = mean_KO - mean_NT.
  score     Spearman / Pearson / sign agreement on a scoring set fixed from the
            observed data. Self-pairs are excluded: a CRISPR indel destroys the
            protein while leaving the transcript intact.

The baseline that matters is mRNA passthrough. A method earns its place only by
beating it, and the pairs that decide the comparison are the post-transcriptional
ones (CMTM6 and CUL3 move PD-L1 protein while CD274 mRNA does not).

Run (GPU strongly preferred; the kNN is O(n^2) per method-seed):
  sbatch --export=ALL,RUN_CMD='PYTHONPATH=. python -u tools/papalexi_perturbation_benchmark.py' \
      slurm/train_slurm.sh
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch

from src.evaluation.perturbation import (
    bootstrap_spearman_gap,
    build_scoring_set,
    column_order_warning,
    group_deltas,
    predict_protein,
    score,
)
from tools.papalexi_perturbation_signal import (
    ADT_GENE_MAP,
    CONTROL_LABEL,
    MIN_CELLS,
    effect_table,
    load_metadata,
    load_protein,
    load_target_gene_expression,
    normalize_adt,
    rna_effect_table,
    to_dense,
)

PREDICTION_RE = re.compile(
    r"^(?P<model>.+?)_(?P<dataset>papalexi_(?:retained|regvelo))"
    r"(?:_seed_(?P<seed>\d+))?\.h5ad$"
)

# Post-transcriptional regulators of PD-L1: protein moves, CD274 mRNA does not.
# These are the pairs on which mRNA passthrough cannot score above chance.
DISCRIMINATING_PAIRS = [("CMTM6", "PDL1"), ("CUL3", "PDL1")]


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def discover_runs(pred_dir: Path) -> list[dict]:
    runs = []
    for path in sorted(pred_dir.glob("*papalexi*.h5ad")):
        match = PREDICTION_RE.match(path.name)
        if match is None:
            print(f"[runs] skipping unparseable name: {path.name}")
            continue
        runs.append({
            "path": path,
            "model": match.group("model"),
            "dataset": match.group("dataset"),
            "seed": match.group("seed"),
        })
    print(f"[runs] {len(runs)} prediction files")
    return runs


def evaluate_run(run: dict, observed_adt: pd.DataFrame, groups_all: pd.Series,
                 adt_names: list[str], scoring: pd.DataFrame, k: int,
                 device, n_perm: int, seed: int,
                 direct_models: set[str]) -> tuple[dict, pd.DataFrame]:
    adata = ad.read_h5ad(run["path"])
    cells = pd.Index(adata.obs_names.astype(str))
    protein_key = next((key for key in adata.obsm if key.startswith("X_aligned_")
                        and key != "X_aligned_rna"), None)
    if protein_key is None:
        raise KeyError(f"{run['path'].name} has no aligned protein embedding: {list(adata.obsm)}")

    emb_rna = np.asarray(adata.obsm["X_aligned_rna"], dtype=np.float32)
    emb_protein = np.asarray(adata.obsm[protein_key], dtype=np.float32)

    shared = cells.intersection(observed_adt.index)
    if len(shared) != len(cells):
        keep = cells.isin(observed_adt.index)
        emb_rna, emb_protein, cells = emb_rna[keep], emb_protein[keep], cells[keep]
    values = observed_adt.loc[cells].to_numpy(dtype=np.float32)
    groups = groups_all.loc[cells]
    control_mask = (groups == CONTROL_LABEL).to_numpy()

    spread = float(np.max(np.std(emb_rna, axis=0)))
    if spread < 1e-8:
        print(f"    [degenerate] RNA embedding collapsed to a point (spread={spread:.2e})")
        return {**{key: run[key] for key in ("model", "dataset", "seed")},
                "spearman": np.nan, "pearson": np.nan, "sign_acc": np.nan,
                "n_pairs": 0, "degenerate": True, "predictor": "",
                "gap_vs_mrna_mean": np.nan, "gap_vs_mrna_lo": np.nan,
                "gap_vs_mrna_hi": np.nan}, pd.DataFrame()

    # Which predictor to use is a property of the MODEL (kot_use_feature_space puts
    # phi(r) in the protein panel), not something to infer from the embedding.
    warning = column_order_warning(emb_protein, values, adt_names)
    if warning is not None:
        print(f"    [order] WARNING: {warning}")
    predicted_values, predictor = predict_protein(
        emb_rna, emb_protein, values, control_mask, k, device,
        use_direct=run["model"] in direct_models)
    print(f"    predictor={predictor}")
    predicted = group_deltas(predicted_values, adt_names, groups, CONTROL_LABEL, MIN_CELLS)

    merged = scoring.merge(predicted.rename(columns={"delta": "pred_delta"}),
                           on=["knockout", "adt"], how="left")
    result = {**{key: run[key] for key in ("model", "dataset", "seed")},
              **score(merged["pred_delta"].to_numpy(), merged["obs_delta"].to_numpy()),
              "degenerate": False, "predictor": predictor}

    rng = np.random.default_rng(seed)
    null_scores = []
    for _ in range(n_perm):
        shuffled = predicted_values[rng.permutation(len(predicted_values))]
        null_pred = group_deltas(shuffled, adt_names, groups, CONTROL_LABEL, MIN_CELLS)
        null_merged = scoring.merge(null_pred.rename(columns={"delta": "pred_delta"}),
                                    on=["knockout", "adt"], how="left")
        null_scores.append(score(null_merged["pred_delta"].to_numpy(),
                                 null_merged["obs_delta"].to_numpy())["spearman"])
    result["shuffled_alignment_spearman_mean"] = float(np.nanmean(null_scores))
    result["shuffled_alignment_spearman_sd"] = float(np.nanstd(null_scores))

    merged["model"] = run["model"]
    merged["dataset"] = run["dataset"]
    merged["seed"] = run["seed"]
    merged["predictor"] = predictor
    return result, merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protein", type=Path,
                        default=Path("Datasets/Papalexi_GSE153056/papalexi_protein.h5ad"))
    parser.add_argument("--rna", type=Path,
                        default=Path("Datasets/Papalexi_GSE153056/papalexi_rna_input.h5ad"))
    parser.add_argument("--metadata", type=Path, default=Path("data/papalexi/ECCITE_metadata.tsv"))
    parser.add_argument("--predictions", type=Path, default=Path("data/predictions"))
    parser.add_argument("--out-dir", type=Path, default=Path("cache/results/papalexi_perturbation"))
    parser.add_argument("--normalization", default="rna_size",
                        choices=["rna_size", "raw_log1p", "clr"],
                        help="ADT size factor; rna_size avoids the compositional artifact "
                             "of normalizing a 4-plex panel by its own total")
    parser.add_argument("--k", type=int, default=30, help="neighbours for knn_control imputation")
    parser.add_argument(
        "--direct-models", default="kot",
        help="Comma-separated models whose aligned RNA embedding is already in ADT "
             "coordinates, so Delta is read from phi(r) directly instead of imputed by "
             "kNN. Everything else uses knn_control.")
    parser.add_argument("--q-max", type=float, default=0.05)
    parser.add_argument("--d-min", type=float, default=0.1)
    parser.add_argument("--n-perm", type=int, default=10)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"[device] {device}")

    protein = load_protein(args.protein)
    meta = load_metadata(args.metadata)
    expr = load_target_gene_expression(args.rna, list(ADT_GENE_MAP.values()))

    cells = protein.obs_names.intersection(meta.index).intersection(expr.index)
    protein, meta, expr = protein[cells].copy(), meta.loc[cells], expr.loc[cells]
    adt_names = [str(v) for v in protein.var_names]
    groups = meta["gene"].astype(str)

    variants = normalize_adt(to_dense(protein.X).astype(np.float64),
                             np.asarray(meta["nCount_RNA"], dtype=np.float64))
    observed_adt = pd.DataFrame(variants[args.normalization], index=cells, columns=adt_names)

    observed_effects = effect_table(observed_adt.to_numpy(), adt_names, groups)
    scoring = build_scoring_set(observed_effects, ADT_GENE_MAP, args.q_max, args.d_min)

    rna_effects = rna_effect_table(expr, groups).rename(columns={"rna_delta": "pred_delta"})
    mrna_null = scoring.merge(rna_effects, on=["knockout", "adt"], how="left")
    obs = mrna_null["obs_delta"].to_numpy()
    mrna_pred = mrna_null["pred_delta"].to_numpy()
    mrna_score = score(mrna_pred, obs)
    print(f"\n[null: mRNA passthrough] spearman={mrna_score['spearman']:+.3f}  "
          f"pearson={mrna_score['pearson']:+.3f}  sign_acc={mrna_score['sign_acc']:.3f}")

    rng = np.random.default_rng(args.seed)
    label_null = [
        score(mrna_pred, obs[rng.permutation(len(obs))])["spearman"]
        for _ in range(200)
    ]
    print(f"[null: shuffled pairing] spearman = {np.nanmean(label_null):+.3f} "
          f"+/- {np.nanstd(label_null):.3f}  (chance level for {len(obs)} pairs)")

    direct_models = {m.strip() for m in args.direct_models.split(",") if m.strip()}
    print(f"[predictor] direct (phi(r)): {sorted(direct_models)}; everything else knn_control")

    results, per_pair = [], []
    for run in discover_runs(args.predictions):
        tag = f"{run['model']}/{run['dataset']}" + (f"/seed_{run['seed']}" if run["seed"] else "")
        print(f"\n[run] {tag}")
        result, merged = evaluate_run(run, observed_adt, groups, adt_names, scoring,
                                      args.k, device, args.n_perm, args.seed,
                                      direct_models)
        if not result["degenerate"]:
            gap, lo, hi = bootstrap_spearman_gap(
                merged["pred_delta"].to_numpy(), mrna_pred, obs, args.n_boot, args.seed)
            result.update({"gap_vs_mrna_mean": gap, "gap_vs_mrna_lo": lo, "gap_vs_mrna_hi": hi})
            print(f"    spearman={result['spearman']:+.3f}  pearson={result['pearson']:+.3f}  "
                  f"sign_acc={result['sign_acc']:.3f}  "
                  f"shuffled={result['shuffled_alignment_spearman_mean']:+.3f}")
            print(f"    gap vs mRNA passthrough: {gap:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
            per_pair.append(merged)
        results.append(result)

    table = pd.DataFrame(results)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_dir / "perturbation_benchmark_runs.csv", index=False)

    summary = (table[~table["degenerate"]]
               .groupby(["model", "dataset"])
               .agg(spearman_mean=("spearman", "mean"), spearman_sd=("spearman", "std"),
                    sign_acc_mean=("sign_acc", "mean"), sign_acc_sd=("sign_acc", "std"),
                    gap_mean=("gap_vs_mrna_mean", "mean"), n_seeds=("spearman", "count"))
               .sort_values("spearman_mean", ascending=False)
               .reset_index())
    summary.to_csv(args.out_dir / "perturbation_benchmark_summary.csv", index=False)

    print("\n" + "=" * 78)
    print(f"SUMMARY — Spearman(predicted, observed) over {len(obs)} scoring pairs")
    print("=" * 78)
    print(f"  {'mRNA passthrough (null)':<32} {mrna_score['spearman']:+.3f}   "
          f"sign_acc={mrna_score['sign_acc']:.3f}")
    print(f"  {'shuffled pairing (chance)':<32} {np.nanmean(label_null):+.3f} "
          f"+/- {np.nanstd(label_null):.3f}")
    print("-" * 78)
    for row in summary.itertuples():
        sd = "" if np.isnan(row.spearman_sd) else f" +/- {row.spearman_sd:.3f}"
        print(f"  {row.model + ' / ' + row.dataset:<32} {row.spearman_mean:+.3f}{sd}   "
              f"sign_acc={row.sign_acc_mean:.3f}   n={row.n_seeds}")

    if per_pair:
        pairs = pd.concat(per_pair, ignore_index=True)
        pairs.to_csv(args.out_dir / "perturbation_benchmark_per_pair.csv", index=False)

        print("\n" + "=" * 78)
        print("DISCRIMINATING PAIRS — post-transcriptional, mRNA passthrough predicts ~0")
        print("=" * 78)
        for knockout, adt in DISCRIMINATING_PAIRS:
            sel = pairs[(pairs["knockout"] == knockout) & (pairs["adt"] == adt)]
            if sel.empty:
                print(f"\n  {knockout} -> {adt}: not in the scoring set")
                continue
            print(f"\n  {knockout} -> {adt}   observed delta = {sel['obs_delta'].iloc[0]:+.4f}")
            for (model, dataset), grp in sel.groupby(["model", "dataset"]):
                mean = grp["pred_delta"].mean()
                agrees = np.sign(mean) == np.sign(sel["obs_delta"].iloc[0])
                print(f"    {model + ' / ' + dataset:<34} predicted {mean:+.4f}  "
                      f"{'sign ok' if agrees else 'SIGN WRONG'}")

    print(f"\n[out] wrote tables to {args.out_dir}")


if __name__ == "__main__":
    main()
