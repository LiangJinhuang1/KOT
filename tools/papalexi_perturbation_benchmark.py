"""Score every Papalexi run on perturbation-response prediction instead of FOSCTTM.

FOSCTTM asks each method to pair individual cells across modalities. With a 4-plex
ADT panel that pairing is unidentifiable, so every method lands near the 0.5 chance
level and the ranking that comes out is noise. This script asks a question the same
data can actually answer: given the learned cross-modal alignment, what does each
CRISPR knockout do to the four checkpoint proteins?

The quantity estimated is a group mean difference over 100-1300 cells per knockout
rather than a per-cell assignment, so the noise floor drops by roughly sqrt(n).

Protocol, identical for every method so none is favoured:

  predict   For RNA cell i, take its k nearest cells in the protein-side embedding
            (excluding i itself, which would leak the answer) and average their
            *observed* ADT values. This is ordinary cross-modal imputation, the
            natural downstream use of a diagonal integration method.
  contrast  delta(knockout, protein) = mean over knockout cells - mean over NT cells,
            computed the same way on predicted and on observed values.
  score     Spearman / Pearson / sign agreement between predicted and observed delta
            over a scoring set fixed in advance from the observed data alone.

Self-pairs are excluded from scoring. A CRISPR indel destroys the protein while
leaving the transcript intact (CD86 knockout moves CD86 protein by -0.43 and CD86
mRNA by +0.03), so "this protein was knocked out" is information that does not exist
in the transcriptome. Scoring any RNA->protein model on those pairs asks an
unanswerable question. They still serve to validate the ground truth, which is what
tools/papalexi_perturbation_signal.py uses them for.

The baseline that matters is mRNA passthrough: predict the protein effect by copying
the observed log-fold-change of the protein's own encoding gene. It needs no
cross-modal model at all, and on this dataset it is strong (Spearman ~0.64), because
the largest effects run through the IFN-gamma axis where mRNA and protein move
together. A method earns its place only by beating it, and the pairs that decide the
comparison are the post-transcriptional ones (CMTM6 and CUL3 move PD-L1 protein in
opposite directions while CD274 mRNA does not move at all), where passthrough is
guaranteed to predict zero.

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
from scipy import stats

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


def knn_indices(source: np.ndarray, target: np.ndarray, k: int, device: torch.device,
                batch: int = 1024) -> np.ndarray:
    """Row i -> indices of its k nearest rows in `target`, with i itself masked out.

    Row i of source and row i of target are the same cell, so leaving it in would let
    a method that aligns directly in protein feature space read off the answer.
    """
    src = torch.as_tensor(np.asarray(source), dtype=torch.float32, device=device)
    tgt = torch.as_tensor(np.asarray(target), dtype=torch.float32, device=device)
    n = src.shape[0]
    out = torch.empty((n, k), dtype=torch.long, device=device)
    for start in range(0, n, batch):
        end = min(start + batch, n)
        dist = torch.cdist(src[start:end], tgt)
        rows = torch.arange(end - start, device=device)
        dist[rows, torch.arange(start, end, device=device)] = float("inf")
        out[start:end] = torch.topk(dist, k, dim=1, largest=False).indices
    return out.cpu().numpy()


def gather_mean(values: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Mean of `values` over each row's neighbour set."""
    return values[idx].mean(axis=1)


def group_deltas(values: np.ndarray, adt_names: list[str], groups: pd.Series) -> pd.DataFrame:
    """Mean shift versus the non-targeting control, per (knockout, protein)."""
    is_control = (groups == CONTROL_LABEL).to_numpy()
    control_mean = values[is_control].mean(axis=0)
    rows = []
    for target in sorted(set(groups.unique()) - {CONTROL_LABEL}):
        mask = (groups == target).to_numpy()
        if mask.sum() < MIN_CELLS:
            continue
        delta = values[mask].mean(axis=0) - control_mean
        for j, adt in enumerate(adt_names):
            rows.append({"knockout": target, "adt": adt, "delta": float(delta[j])})
    return pd.DataFrame(rows)


def build_scoring_set(observed: pd.DataFrame, q_max: float, d_min: float) -> pd.DataFrame:
    """Pairs with a real observed effect, minus the unanswerable self-pairs.

    Fixed from the observed data alone, before any method is scored.
    """
    df = observed.copy()
    df["is_self_pair"] = [
        ADT_GENE_MAP[adt] == knockout
        for adt, knockout in zip(df["adt"], df["knockout"])
    ]
    significant = (df["qval"] < q_max) & (df["cohens_d"].abs() > d_min)
    kept = df[significant & ~df["is_self_pair"]].copy()
    dropped = df[significant & df["is_self_pair"]]
    print(f"[scoring set] {len(kept)} pairs (q<{q_max}, |d|>{d_min}), "
          f"{len(dropped)} self-pairs excluded: "
          f"{', '.join(f'{r.knockout}->{r.adt}' for r in dropped.itertuples())}")
    return kept[["knockout", "adt", "delta", "cohens_d"]].rename(columns={"delta": "obs_delta"})


def score(pred: np.ndarray, obs: np.ndarray) -> dict:
    finite = np.isfinite(pred) & np.isfinite(obs)
    pred, obs = pred[finite], obs[finite]
    if len(pred) < 3 or np.std(pred) == 0:
        return {"spearman": np.nan, "pearson": np.nan, "sign_acc": np.nan, "n_pairs": len(pred)}
    return {
        "spearman": float(stats.spearmanr(pred, obs).statistic),
        "pearson": float(stats.pearsonr(pred, obs).statistic),
        "sign_acc": float(np.mean(np.sign(pred) == np.sign(obs))),
        "n_pairs": int(len(pred)),
    }


def bootstrap_spearman_gap(pred: np.ndarray, reference: np.ndarray, obs: np.ndarray,
                           n_boot: int, seed: int) -> tuple[float, float, float]:
    """Paired bootstrap over scoring-set pairs for spearman(pred) - spearman(reference).

    Paired because both predictors are scored on the same resampled pairs, which is
    what makes the difference interpretable with only ~20 points.
    """
    rng = np.random.default_rng(seed)
    n = len(obs)
    gaps = np.full(n_boot, np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.std(pred[idx]) == 0 or np.std(reference[idx]) == 0 or np.std(obs[idx]) == 0:
            continue
        gaps[b] = (stats.spearmanr(pred[idx], obs[idx]).statistic
                   - stats.spearmanr(reference[idx], obs[idx]).statistic)
    lo, hi = np.nanpercentile(gaps, [2.5, 97.5])
    return float(np.nanmean(gaps)), float(lo), float(hi)


def verify_protein_column_order(aligned_protein: np.ndarray, observed: np.ndarray,
                                adt_names: list[str]) -> None:
    """The imputation reads protein values from our own normalized matrix, so a column
    mismatch against the saved embedding would silently scramble the four channels."""
    n_cols = aligned_protein.shape[1]
    if n_cols != len(adt_names):
        print(f"    [order] aligned protein is a {n_cols}-d latent space, not the "
              f"{len(adt_names)} ADT channels — column-order check does not apply")
        return
    corr = np.array([
        [abs(stats.spearmanr(aligned_protein[:, i], observed[:, j]).statistic)
         for j in range(n_cols)]
        for i in range(n_cols)
    ])
    # A constant embedding column correlates with nothing, so its whole row is NaN
    # and nanargmax would raise instead of reporting.
    if np.isnan(corr).all(axis=1).any():
        print("    [order] WARNING: an aligned protein column is constant — the "
              "column-order check is not defined for it")
        return
    if not np.all(np.nanargmax(corr, axis=1) == np.arange(n_cols)):
        print(f"    [order] WARNING: aligned protein columns do not match {adt_names} "
              f"in order; argmax per row = {np.nanargmax(corr, axis=1).tolist()}")


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
                 device: torch.device, n_perm: int, seed: int) -> tuple[dict, pd.DataFrame]:
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

    spread = float(np.max(np.std(emb_rna, axis=0)))
    if spread < 1e-8:
        print(f"    [degenerate] RNA embedding collapsed to a point (spread={spread:.2e})")
        # Same keys as a scored run, so the summary groupby below still finds its
        # columns when every run in a batch turns out to be degenerate.
        return {**{key: run[key] for key in ("model", "dataset", "seed")},
                "spearman": np.nan, "pearson": np.nan, "sign_acc": np.nan,
                "n_pairs": 0, "degenerate": True,
                "gap_vs_mrna_mean": np.nan, "gap_vs_mrna_lo": np.nan,
                "gap_vs_mrna_hi": np.nan}, pd.DataFrame()

    verify_protein_column_order(emb_protein, values, adt_names)

    idx = knn_indices(emb_rna, emb_protein, k, device)
    predicted = group_deltas(gather_mean(values, idx), adt_names, groups)

    merged = scoring.merge(predicted.rename(columns={"delta": "pred_delta"}),
                           on=["knockout", "adt"], how="left")
    result = {**{key: run[key] for key in ("model", "dataset", "seed")},
              **score(merged["pred_delta"].to_numpy(), merged["obs_delta"].to_numpy()),
              "degenerate": False}

    # Chance level for this exact scoring set: keep the neighbour graph, break the
    # correspondence between a neighbour and its protein readout.
    rng = np.random.default_rng(seed)
    null_scores = []
    for _ in range(n_perm):
        shuffled = values[rng.permutation(len(values))]
        null_pred = group_deltas(gather_mean(shuffled, idx), adt_names, groups)
        null_merged = scoring.merge(null_pred.rename(columns={"delta": "pred_delta"}),
                                    on=["knockout", "adt"], how="left")
        null_scores.append(score(null_merged["pred_delta"].to_numpy(),
                                 null_merged["obs_delta"].to_numpy())["spearman"])
    result["shuffled_alignment_spearman_mean"] = float(np.nanmean(null_scores))
    result["shuffled_alignment_spearman_sd"] = float(np.nanstd(null_scores))

    merged["model"] = run["model"]
    merged["dataset"] = run["dataset"]
    merged["seed"] = run["seed"]
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
    parser.add_argument("--k", type=int, default=30, help="neighbours for cross-modal imputation")
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
    scoring = build_scoring_set(observed_effects, args.q_max, args.d_min)

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

    results, per_pair = [], []
    for run in discover_runs(args.predictions):
        tag = f"{run['model']}/{run['dataset']}" + (f"/seed_{run['seed']}" if run["seed"] else "")
        print(f"\n[run] {tag}")
        result, merged = evaluate_run(run, observed_adt, groups, adt_names, scoring,
                                      args.k, device, args.n_perm, args.seed)
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
