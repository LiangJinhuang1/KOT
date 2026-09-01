#!/usr/bin/env python3
"""Papalexi effect table and ranking share one scoring set, or the null drifts."""
from __future__ import annotations

from pathlib import Path
from scipy import sparse, stats
import anndata as ad
import argparse
import json
import numpy as np
import pandas as pd
import re
import sys
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.perturbation import (
    bootstrap_spearman_gap,
    build_scoring_set,
    calibrate_direct,
    column_order_warning,
    group_deltas,
    predict_protein,
    score,
)
from src.evaluation.protocol import read_protocol


# ============================================================================
# signal
# ============================================================================

ADT_GENE_MAP = {
    "CD86": "CD86",
    "PDL1": "CD274",
    "PDL2": "PDCD1LG2",
    "CD366": "HAVCR2",
}


CONTROL_LABEL = "NT"


MIN_CELLS = 40


def to_dense(x) -> np.ndarray:
    return np.asarray(x.todense()) if sparse.issparse(x) else np.asarray(x)


def load_protein(path: Path) -> ad.AnnData:
    protein = ad.read_h5ad(path)
    print(f"[protein] {protein.n_obs} cells x {protein.n_vars} ADTs: {list(protein.var_names)}")
    unknown = set(map(str, protein.var_names)) - set(ADT_GENE_MAP)
    if unknown:
        raise ValueError(
            f"ADT names {sorted(unknown)} are not in ADT_GENE_MAP {sorted(ADT_GENE_MAP)}; "
            "the protein/mRNA merge would silently drop them. Update ADT_GENE_MAP."
        )
    return protein


def load_metadata(path: Path) -> pd.DataFrame:
    meta = pd.read_csv(path, sep="\t", index_col=0)
    meta.index = meta.index.astype(str)
    print(f"[metadata] {len(meta)} cells, columns: {list(meta.columns)}")
    return meta


def load_target_gene_expression(path: Path, genes: list[str]) -> pd.DataFrame:
    """log1p CP10K expression of `genes`, streamed in row blocks so the 1.4GB
    matrix is never held in memory. X is spliced+unspliced counts, identical to
    the `counts` layer written by tools/build_inputs.py papalexi."""
    backed = ad.read_h5ad(path, backed="r")
    var_names = pd.Index(backed.var_names.astype(str))
    present = [g for g in genes if g in var_names]
    missing = [g for g in genes if g not in var_names]
    if missing:
        print(f"[rna] WARNING: genes absent from the RNA matrix: {missing}")
    cols = [int(var_names.get_loc(g)) for g in present]

    n_obs = backed.n_obs
    totals = np.zeros(n_obs, dtype=np.float64)
    sub = np.zeros((n_obs, len(cols)), dtype=np.float64)
    block = 4000
    for start in range(0, n_obs, block):
        end = min(start + block, n_obs)
        chunk = backed.X[start:end]
        dense_cols = to_dense(chunk[:, cols])
        totals[start:end] = np.asarray(chunk.sum(axis=1)).ravel()
        sub[start:end] = dense_cols

    obs_names = pd.Index(backed.obs_names.astype(str))
    backed.file.close()

    cp10k = sub / np.maximum(totals[:, None], 1.0) * 1e4
    out = pd.DataFrame(np.log1p(cp10k), index=obs_names, columns=present)
    print(f"[rna] target-gene expression: {out.shape[0]} cells x {out.shape[1]} genes")
    return out


def normalize_adt(counts: np.ndarray, rna_totals: np.ndarray) -> dict[str, np.ndarray]:
    """Three ADT normalizations with different compositional exposure.

    raw_log1p    no size factor at all; immune to composition, sensitive to depth.
    rna_size     size factor from total RNA counts, i.e. outside the 4-plex panel,
                 so a drop in one protein cannot inflate the other three.
    clr          the CITE-seq convention, included because reviewers expect it;
                 fully compositional across only 4 features.
    """
    raw_log1p = np.log1p(counts)

    size = np.maximum(rna_totals, 1.0)
    rna_size = np.log1p(counts / size[:, None] * float(np.median(size)))

    logged = np.log1p(counts)
    clr = logged - logged.mean(axis=1, keepdims=True)

    return {"raw_log1p": raw_log1p, "rna_size": rna_size, "clr": clr}


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """BH-adjusted p-values over the whole (knockout, protein) grid.

    ``p * n / rank`` on its own is not the BH q-value. Without the running minimum
    taken from the largest p downwards the result is not monotone in p, so a pair
    can end up with a smaller q than a pair with a smaller p, and a q < alpha cut
    then keeps an inconsistent set. The benchmark's scoring set is chosen by
    exactly that cut, so the step-up matters here.
    """
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    stepped = p[order] * n / np.arange(1, n + 1)
    stepped = np.minimum.accumulate(stepped[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.minimum(stepped, 1.0)
    return q


def effect_table(values: np.ndarray, adt_names: list[str], groups: pd.Series) -> pd.DataFrame:
    """Per (knockout, protein) effect versus the non-targeting control."""
    is_control = (groups == CONTROL_LABEL).to_numpy()
    control = values[is_control]
    targets = [g for g in sorted(groups.unique()) if g != CONTROL_LABEL]

    rows = []
    for target in targets:
        mask = (groups == target).to_numpy()
        if mask.sum() < MIN_CELLS:
            continue
        ko = values[mask]
        for j, adt in enumerate(adt_names):
            a, b = ko[:, j], control[:, j]
            pooled_sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
            _, pval = stats.mannwhitneyu(a, b, alternative="two-sided")
            rows.append({
                "knockout": target,
                "adt": adt,
                "n_ko": int(mask.sum()),
                "delta": float(a.mean() - b.mean()),
                "cohens_d": float((a.mean() - b.mean()) / pooled_sd) if pooled_sd > 0 else np.nan,
                "pval": float(pval),
            })

    df = pd.DataFrame(rows)
    df["qval"] = benjamini_hochberg(df["pval"].to_numpy())
    return df


def rna_effect_table(expr: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """Same contrast on the ADT-target genes' mRNA, for the discordance analysis."""
    gene_to_adt = {gene: adt for adt, gene in ADT_GENE_MAP.items()}
    is_control = (groups == CONTROL_LABEL).to_numpy()
    targets = [g for g in sorted(groups.unique()) if g != CONTROL_LABEL]

    rows = []
    for target in targets:
        mask = (groups == target).to_numpy()
        if mask.sum() < MIN_CELLS:
            continue
        for gene in expr.columns:
            a = expr[gene].to_numpy()[mask]
            b = expr[gene].to_numpy()[is_control]
            rows.append({
                "knockout": target,
                "adt": gene_to_adt[gene],
                "rna_delta": float(a.mean() - b.mean()),
            })
    return pd.DataFrame(rows)


def report(effects: dict[str, pd.DataFrame], rna: pd.DataFrame, out_dir: Path) -> None:
    primary = effects["rna_size"]

    print("\n" + "=" * 78)
    print("1. POSITIVE CONTROLS — a knockout of the gene encoding a measured protein")
    print("=" * 78)
    for knockout, adt in [("CD86", "CD86"), ("PDCD1LG2", "PDL2")]:
        print(f"\n  {knockout} KO -> {adt} protein")
        for label, df in effects.items():
            hit = df[(df["knockout"] == knockout) & (df["adt"] == adt)]
            if hit.empty:
                print(f"    {label:<10} (knockout absent or below {MIN_CELLS} cells)")
                continue
            row = hit.iloc[0]
            print(f"    {label:<10} delta={row['delta']:+.4f}  d={row['cohens_d']:+.3f}  "
                  f"q={row['qval']:.2e}  n={row['n_ko']}")

    print("\n" + "=" * 78)
    print("2. EVALUATION SET SIZE — significant (knockout, protein) pairs, rna_size")
    print("=" * 78)
    for q in (0.05, 0.01):
        for d in (0.1, 0.2, 0.5):
            n = int(((primary["qval"] < q) & (primary["cohens_d"].abs() > d)).sum())
            print(f"  q<{q:<5} |d|>{d:<4} : {n:3d} / {len(primary)} pairs")

    print("\n  Top 20 effects by |Cohen's d|:")
    top = primary.reindex(primary["cohens_d"].abs().sort_values(ascending=False).index).head(20)
    for _, row in top.iterrows():
        print(f"    {row['knockout']:<10} {row['adt']:<6} delta={row['delta']:+.4f}  "
              f"d={row['cohens_d']:+.3f}  q={row['qval']:.2e}  n={int(row['n_ko'])}")

    print("\n" + "=" * 78)
    print("3. mRNA/PROTEIN DISCORDANCE — where a central-dogma model can be scored")
    print("=" * 78)
    merged = primary.merge(rna, on=["knockout", "adt"], how="left")
    sig = merged[(merged["qval"] < 0.05) & (merged["cohens_d"].abs() > 0.1)].copy()
    if sig.empty:
        print("  No significant pairs — the discordance analysis is not defined.")
    else:
        rho = stats.spearmanr(sig["rna_delta"], sig["delta"])
        print(f"  Across {len(sig)} significant pairs, Spearman(mRNA delta, protein delta) = "
              f"{rho.statistic:+.3f} (p={rho.pvalue:.2e})")
        print("\n  Pairs where protein moves but mRNA barely does (|rna_delta| < 0.05):")
        disc = sig[sig["rna_delta"].abs() < 0.05]
        if disc.empty:
            print("    none")
        for _, row in disc.reindex(
            disc["cohens_d"].abs().sort_values(ascending=False).index
        ).iterrows():
            print(f"    {row['knockout']:<10} {row['adt']:<6} protein_delta={row['delta']:+.4f} "
                  f"(d={row['cohens_d']:+.3f})  rna_delta={row['rna_delta']:+.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_dir / "papalexi_perturbation_effects.csv", index=False)
    for label, df in effects.items():
        df.to_csv(out_dir / f"papalexi_perturbation_effects_{label}.csv", index=False)
    print(f"\n[out] wrote effect tables to {out_dir}")


def signal_main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protein", type=Path,
                        default=Path("Datasets/Papalexi_GSE153056/papalexi_protein.h5ad"))
    parser.add_argument("--rna", type=Path,
                        default=Path("Datasets/Papalexi_GSE153056/papalexi_rna_input.h5ad"))
    parser.add_argument("--metadata", type=Path, default=Path("data/papalexi/ECCITE_metadata.tsv"))
    parser.add_argument("--out-dir", type=Path, default=Path("cache/results/papalexi_perturbation"))
    args = parser.parse_args()

    protein = load_protein(args.protein)
    meta = load_metadata(args.metadata)
    expr = load_target_gene_expression(args.rna, list(ADT_GENE_MAP.values()))

    cells = protein.obs_names.intersection(meta.index).intersection(expr.index)
    print(f"[join] protein n meta n rna = {len(cells)} cells")
    protein = protein[cells].copy()
    meta = meta.loc[cells]
    expr = expr.loc[cells]

    groups = meta["gene"].astype(str)
    counts = to_dense(protein.X).astype(np.float64)
    adt_names = [str(v) for v in protein.var_names]

    rna_totals = np.asarray(meta["nCount_RNA"], dtype=np.float64)
    variants = normalize_adt(counts, rna_totals)

    print(f"\n[groups] {groups.nunique()} labels, "
          f"{int((groups == CONTROL_LABEL).sum())} control cells")

    effects = {label: effect_table(vals, adt_names, groups) for label, vals in variants.items()}
    report(effects, rna_effect_table(expr, groups), args.out_dir)


# ============================================================================
# benchmark
# ============================================================================

PREDICTION_RE = re.compile(
    r"^(?P<model>.+?)_(?P<dataset>papalexi_(?:retained|regvelo))"
    r"(?:_seed_(?P<seed>\d+))?\.h5ad$"
)


DISCRIMINATING_PAIRS = [("CMTM6", "PDL1"), ("CUL3", "PDL1")]


PROVENANCE_NAME = "provenance.json"


RESULT_KEYS = ("model", "dataset", "seed")


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def discover_runs(pred_dir: Path, direct_models: set[str]) -> list[dict]:
    """Every parseable prediction in `pred_dir`, with the protocol each one was run under.

    The estimator and the ranking eligibility both come from the file's own stamp. Only a
    file written before stamping existed falls back to `direct_models`, and says so.
    """
    runs = []
    for path in sorted(pred_dir.glob("*papalexi*.h5ad")):
        match = PREDICTION_RE.match(path.name)
        if match is None:
            print(f"[runs] skipping unparseable name: {path.name}")
            continue
        backed = ad.read_h5ad(path, backed="r")
        protocol = read_protocol(backed.uns)
        backed.file.close()
        model = match.group("model")
        unstamped = protocol["oos_mode"] == "unstamped"
        if unstamped:
            protocol["predictor"] = "direct" if model in direct_models else "latent"
        runs.append({
            "path": path,
            "model": model,
            "dataset": match.group("dataset"),
            "seed": match.group("seed"),
            "protocol": protocol,
            "unstamped": unstamped,
        })
    if not runs:
        raise FileNotFoundError(
            f"no parseable *papalexi*.h5ad predictions in {pred_dir}. An empty table is a "
            "silent way to drop a method from the comparison, so this is an error: check "
            "--predictions, or that the runs wrote their h5ad (save_prediction_h5ad)."
        )
    print(f"[runs] {len(runs)} prediction files")
    return runs


def is_ranked(run: dict) -> bool:
    """A run belongs in the method ranking only if it was truly held out of the knockouts."""
    protocol = run["protocol"]
    return protocol["out_of_distribution"] and not protocol["paired_oracle"]


def exclusion_reason(run: dict) -> str:
    protocol = run["protocol"]
    if protocol["paired_oracle"]:
        return "paired oracle (encodes the cell's own protein)"
    if run["unstamped"]:
        return "unstamped: predates protocol stamping, trained on every cell"
    return "in-distribution: no fit restriction was applied"


def degenerate_result(run: dict) -> dict:
    """Same keys as a scored run, so the summary groupby still finds its columns."""
    return {**{key: run[key] for key in RESULT_KEYS},
            "spearman": np.nan, "pearson": np.nan, "sign_acc": np.nan,
            "n_pairs": 0, "degenerate": True, "predictor": "",
            "spearman_uncalibrated": np.nan,
            "gap_vs_mrna_mean": np.nan, "gap_vs_mrna_lo": np.nan,
            "gap_vs_mrna_hi": np.nan}


def shuffled_null(predicted_values: np.ndarray, adt_names: list[str], groups: pd.Series,
                  scoring: pd.DataFrame, n_perm: int, seed: int) -> tuple[float, float]:
    """Chance level for this scoring set: keep the values, break the cell/knockout link."""
    rng = np.random.default_rng(seed)
    null_scores = []
    for _ in range(n_perm):
        shuffled = predicted_values[rng.permutation(len(predicted_values))]
        null_pred = group_deltas(shuffled, adt_names, groups, CONTROL_LABEL, MIN_CELLS)
        merged = scoring.merge(null_pred.rename(columns={"delta": "pred_delta"}),
                               on=["knockout", "adt"], how="left")
        null_scores.append(score(merged["pred_delta"].to_numpy(),
                                 merged["obs_delta"].to_numpy())["spearman"])
    return float(np.nanmean(null_scores)), float(np.nanstd(null_scores))


def scored_deltas(values: np.ndarray, adt_names: list[str], groups: pd.Series,
                  scoring: pd.DataFrame) -> pd.DataFrame:
    """Scoring-set pairs with this predictor's delta attached."""
    predicted = group_deltas(values, adt_names, groups, CONTROL_LABEL, MIN_CELLS)
    return scoring.merge(predicted.rename(columns={"delta": "pred_delta"}),
                         on=["knockout", "adt"], how="left")


def evaluate_run(run: dict, observed_adt: pd.DataFrame, groups_all: pd.Series,
                 adt_names: list[str], scoring: pd.DataFrame, k: int,
                 device, n_perm: int, seed: int) -> tuple[dict, pd.DataFrame]:
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
        return degenerate_result(run), pd.DataFrame()

    use_direct = run["protocol"]["predictor"] == "direct"
    warning = column_order_warning(emb_protein, values, adt_names)
    if warning is not None:
        print(f"    [order] WARNING: {warning}")
    predicted_values, predictor = predict_protein(
        emb_rna, emb_protein, values, control_mask, k, device, use_direct=use_direct)

    # The direct prediction lives in the units KOT trained on, the target in the units
    # this script fixed. Calibrating per protein on NT cells is what makes the pooled
    # Spearman a statement about the map and not about the two normalisations.
    raw_merged = scored_deltas(predicted_values, adt_names, groups, scoring)
    if use_direct:
        predicted_values, scale = calibrate_direct(predicted_values, values, control_mask)
        predictor = "direct_calibrated"
        print(f"    calibration scale per ADT: "
              f"{dict(zip(adt_names, np.round(scale, 3).tolist()))}")
    merged = scored_deltas(predicted_values, adt_names, groups, scoring)

    null_mean, null_sd = shuffled_null(predicted_values, adt_names, groups, scoring,
                                       n_perm, seed)
    print(f"    predictor={predictor}")
    result = {
        **{key: run[key] for key in RESULT_KEYS},
        **score(merged["pred_delta"].to_numpy(), merged["obs_delta"].to_numpy()),
        "degenerate": False,
        "predictor": predictor,
        # What the same prediction scores before calibration, so the fix is auditable
        # rather than something the reader has to take on faith.
        "spearman_uncalibrated": score(raw_merged["pred_delta"].to_numpy(),
                                       raw_merged["obs_delta"].to_numpy())["spearman"],
        "shuffled_alignment_spearman_mean": null_mean,
        "shuffled_alignment_spearman_sd": null_sd,
    }

    merged["model"] = run["model"]
    merged["dataset"] = run["dataset"]
    merged["seed"] = run["seed"]
    merged["predictor"] = predictor
    merged["ranked"] = is_ranked(run)
    return result, merged


def protocol_columns(run: dict) -> dict:
    """The stamp fields that belong in every output row."""
    protocol = run["protocol"]
    return {
        "ranked": is_ranked(run),
        "oos_mode": protocol["oos_mode"],
        "out_of_distribution": protocol["out_of_distribution"],
        "paired_oracle": protocol["paired_oracle"],
        "n_fit_cells": protocol["n_fit_cells"],
        "exclusion_reason": "" if is_ranked(run) else exclusion_reason(run),
    }


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    scored = table[~table["degenerate"]]
    return (scored
            .groupby(["model", "dataset", "ranked", "exclusion_reason"], dropna=False)
            .agg(spearman_mean=("spearman", "mean"), spearman_sd=("spearman", "std"),
                 sign_acc_mean=("sign_acc", "mean"), sign_acc_sd=("sign_acc", "std"),
                 gap_mean=("gap_vs_mrna_mean", "mean"), n_seeds=("spearman", "count"))
            .sort_values("spearman_mean", ascending=False)
            .reset_index())


def provenance_record(args, runs: list[dict]) -> dict:
    return {
        "predictions_dir": str(args.predictions),
        "normalization": args.normalization,
        "q_max": args.q_max,
        "d_min": args.d_min,
        "k": args.k,
        "files": sorted(run["path"].name for run in runs),
        "models": sorted({run["model"] for run in runs}),
        "ranked_models": sorted({run["model"] for run in runs if is_ranked(run)}),
    }


def check_provenance(out_dir: Path, record: dict, force: bool) -> None:
    """Refuse to overwrite a table that describes different inputs.

    The tables carry no record of what produced them, so a default invocation used to
    overwrite a finished comparison with one built from a different predictions directory
    -- or, once a method's files had been moved away, with one silently missing it.
    """
    path = out_dir / PROVENANCE_NAME
    if force or not path.exists():
        return
    previous = json.loads(path.read_text())
    changed = [key for key in ("predictions_dir", "normalization", "models")
               if previous.get(key) != record[key]]
    if not changed:
        return
    lines = "\n".join(f"    {key}: {previous.get(key)!r} -> {record[key]!r}"
                      for key in changed)
    raise SystemExit(
        f"{out_dir} holds a comparison built from different inputs:\n{lines}\n"
        "Write to a different --out-dir, or pass --force to overwrite it."
    )


def print_ranking(title: str, rows: pd.DataFrame, mrna_score: dict,
                  label_null: list[float], n_pairs: int) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    print(f"  {'mRNA passthrough (null)':<32} {mrna_score['spearman']:+.3f}   "
          f"sign_acc={mrna_score['sign_acc']:.3f}")
    print(f"  {'shuffled pairing (chance)':<32} {np.nanmean(label_null):+.3f} "
          f"+/- {np.nanstd(label_null):.3f}  ({n_pairs} pairs)")
    print("-" * 78)
    for row in rows.itertuples():
        sd = "" if np.isnan(row.spearman_sd) else f" +/- {row.spearman_sd:.3f}"
        print(f"  {row.model + ' / ' + row.dataset:<32} {row.spearman_mean:+.3f}{sd}   "
              f"sign_acc={row.sign_acc_mean:.3f}   n={row.n_seeds}")


def print_discriminating_pairs(pairs: pd.DataFrame) -> None:
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
            tag = "" if grp["ranked"].all() else "  [not ranked]"
            print(f"    {model + ' / ' + dataset:<34} predicted {mean:+.4f}  "
                  f"{'sign ok' if agrees else 'SIGN WRONG'}{tag}")


def benchmark_main() -> None:
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
        help="Fallback only, for prediction files written before the protocol stamp "
             "existed: models whose aligned RNA embedding is in ADT coordinates. Stamped "
             "files carry their own predictor and ignore this.")
    parser.add_argument("--q-max", type=float, default=0.05)
    parser.add_argument("--d-min", type=float, default=0.1)
    parser.add_argument("--n-perm", type=int, default=10)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an out-dir whose provenance describes other inputs")
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
    runs = discover_runs(args.predictions, direct_models)
    record = provenance_record(args, runs)
    check_provenance(args.out_dir, record, args.force)

    unstamped = [run["path"].name for run in runs if run["unstamped"]]
    if unstamped:
        print(f"\n[protocol] {len(unstamped)} unstamped file(s); treated as "
              f"in-distribution and excluded from the ranking: {sorted(unstamped)}")
    print(f"[protocol] ranked: {record['ranked_models']}  |  "
          f"reference only: {sorted(set(record['models']) - set(record['ranked_models']))}")

    results, per_pair = [], []
    for run in runs:
        tag = f"{run['model']}/{run['dataset']}" + (f"/seed_{run['seed']}" if run["seed"] else "")
        print(f"\n[run] {tag}  [{run['protocol']['oos_mode']}]")
        result, merged = evaluate_run(run, observed_adt, groups, adt_names, scoring,
                                      args.k, device, args.n_perm, args.seed)
        result.update(protocol_columns(run))
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
    summary = summarize(table)
    summary.to_csv(args.out_dir / "perturbation_benchmark_summary.csv", index=False)

    ranked = summary[summary["ranked"]]
    reference = summary[~summary["ranked"]]
    print_ranking(f"METHOD RANKING — Spearman(predicted, observed) over {len(obs)} pairs",
                  ranked, mrna_score, label_null, len(obs))
    if not reference.empty:
        print("\n" + "=" * 78)
        print("NOT RANKED — ceilings and in-distribution references, not competitors")
        print("=" * 78)
        for row in reference.itertuples():
            print(f"  {row.model + ' / ' + row.dataset:<32} {row.spearman_mean:+.3f}   "
                  f"sign_acc={row.sign_acc_mean:.3f}   n={row.n_seeds}")
            print(f"  {'':<32} {row.exclusion_reason}")

    if per_pair:
        pairs = pd.concat(per_pair, ignore_index=True)
        pairs.to_csv(args.out_dir / "perturbation_benchmark_per_pair.csv", index=False)
        print_discriminating_pairs(pairs)

    (args.out_dir / PROVENANCE_NAME).write_text(json.dumps(record, indent=2))
    print(f"\n[out] wrote tables and {PROVENANCE_NAME} to {args.out_dir}")


# ============================================================================
# dispatch
# ============================================================================

COMMANDS = {"signal": signal_main, "benchmark": benchmark_main}


def main() -> int:
    """Route to a subcommand, leaving the rest of argv for its own parser."""
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        print(f"commands: {', '.join(COMMANDS)}")
        return 2
    command = sys.argv.pop(1)
    COMMANDS[command]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
