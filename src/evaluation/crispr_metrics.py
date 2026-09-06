"""Effect sets and metrics for the CRISPR perturbation benchmark.

TWO RULES SHAPE THIS FILE.

The primary set is chosen from the DESIGN, never from the answers: every sufficiently
sampled non-self (perturbation, protein) effect is in it, whatever its size or p-value.
Selecting on significance would define the benchmark by the effects that happened to
reach it, and would drop exactly the small effects a cross-modal model has most to prove
on. Significance is a SECONDARY view of the same predictions, reported beside the primary
one, never in place of it.

Uncertainty is resampled over EXPERIMENTAL UNITS -- perturbations, or replicates -- not
over cells. Thousands of cells from one perturbation are one experiment, and bootstrapping
them would report a confidence interval many times narrower than the experiment supports.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# Frozen definitions of the secondary subsets. Quantiles, not gene names: which pairs are
# post-transcriptional is a property of the measured effects, and hard-coding CMTM6 and
# CUL3 would bake in the answer the analysis is supposed to find.
POST_TRANSCRIPTIONAL_PROTEIN_QUANTILE = 0.75
POST_TRANSCRIPTIONAL_RNA_QUANTILE = 0.25
SIGNIFICANCE_Q_MAX = 0.05


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """BH step-up q-values, monotone in p."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    stepped = p[order] * len(p) / np.arange(1, len(p) + 1)
    stepped = np.minimum.accumulate(stepped[::-1])[::-1]
    q = np.empty(len(p))
    q[order] = np.minimum(stepped, 1.0)
    return q


def pooled_pair_effects(table: pd.DataFrame) -> pd.DataFrame:
    """One row per (perturbation, protein): effects combined across replicates.

    The protein effect is the mean over replicates and its SE is combined in quadrature,
    so the pair carries a single effect and a single uncertainty for the subset rules to
    read.
    """
    grouped = table.groupby(["perturbation", "protein"])
    pooled = grouped.agg(
        delta_protein=("delta_protein", "mean"),
        delta_cognate_rna=("delta_cognate_rna", "mean"),
        n_replicates=("replicate", "nunique"),
    ).reset_index()
    combined_se = grouped["protein_effect_se"].apply(
        lambda se: float(np.sqrt(np.sum(np.square(se))) / len(se)))
    pooled["protein_effect_se"] = combined_se.to_numpy()
    return pooled


def annotate_effect_sets(effects: pd.DataFrame) -> pd.DataFrame:
    """Add the membership flags every downstream table and figure reads.

    Every set is a property of the (perturbation, protein) PAIR, then broadcast to that
    pair's replicate rows. Deciding membership per replicate would let a pair count as
    post-transcriptional in one batch and not another, which is a statement about noise
    rather than about the biology the subset is supposed to isolate.
    """
    table = effects.copy()
    primary = table["passes_threshold"] & ~table["is_self_effect"]
    table["in_primary"] = primary

    pooled = pooled_pair_effects(table[primary])
    z = pooled["delta_protein"] / pooled["protein_effect_se"].replace(0.0, np.nan)
    pooled["pvalue"] = 2.0 * stats.norm.sf(np.abs(z))
    pooled["qvalue"] = benjamini_hochberg(pooled["pvalue"].fillna(1.0).to_numpy())

    # Post-transcriptional: the protein moves a lot while its own transcript barely does.
    # Thresholds are quantiles OF THE PRIMARY SET, so the subset is defined by the data in
    # front of it rather than by a list of genes someone already believed in.
    protein_cut = pooled["delta_protein"].abs().quantile(POST_TRANSCRIPTIONAL_PROTEIN_QUANTILE)
    rna_cut = pooled["delta_cognate_rna"].abs().quantile(POST_TRANSCRIPTIONAL_RNA_QUANTILE)
    pooled["is_significant"] = pooled["qvalue"] < SIGNIFICANCE_Q_MAX
    pooled["is_post_transcriptional"] = (
        (pooled["delta_protein"].abs() >= protein_cut)
        & (pooled["delta_cognate_rna"].abs() <= rna_cut)
    )

    keys = ["perturbation", "protein"]
    flags = pooled[keys + ["qvalue", "is_significant", "is_post_transcriptional"]]
    table = table.merge(flags, on=keys, how="left")
    table["in_significant"] = primary & table["is_significant"].eq(True)
    table["in_post_transcriptional"] = primary & table["is_post_transcriptional"].eq(True)
    table["in_self"] = table["is_self_effect"] & table["passes_threshold"]

    table.attrs["post_transcriptional_protein_cut"] = float(protein_cut)
    table.attrs["post_transcriptional_rna_cut"] = float(rna_cut)
    return table


EFFECT_SETS = ("primary", "significant", "post_transcriptional", "self")


def confirmed_knockouts(path: Path) -> tuple[list[str], list[str]]:
    """Gene targets marked confirmed in a MultiPert-compat subset file, plus the rest.

    The skipped names are returned so the caller can print the exclusion rather than
    dropping them silently. Status is the file's own column; this does not guess.
    """
    table = pd.read_csv(path, comment="#")
    status = table["status"].astype(str)
    names = table["knockout"].astype(str)
    return names[status == "confirmed"].tolist(), names[status != "confirmed"].tolist()


def metric_suite(predicted: np.ndarray, observed: np.ndarray) -> dict:
    """Every headline metric for one set of (perturbation, protein) effects.

    NRMSE is normalised by the SD of the observed effects, so it answers "how large is the
    error against the spread the model had to explain", and a constant predictor scores 1.
    """
    finite = np.isfinite(predicted) & np.isfinite(observed)
    predicted, observed = predicted[finite], observed[finite]
    if len(predicted) < 3:
        return {key: np.nan for key in
                ("spearman", "pearson", "mae", "nrmse", "sign_acc")} | {"n": len(predicted)}
    residual = predicted - observed
    observed_sd = float(np.std(observed))
    constant = np.std(predicted) == 0
    return {
        "n": int(len(predicted)),
        "spearman": np.nan if constant else float(stats.spearmanr(predicted, observed).statistic),
        "pearson": np.nan if constant else float(stats.pearsonr(predicted, observed).statistic),
        "mae": float(np.mean(np.abs(residual))),
        "nrmse": (float(np.sqrt(np.mean(residual ** 2)) / observed_sd)
                  if observed_sd > 0 else np.nan),
        "sign_acc": float(np.mean(np.sign(predicted) == np.sign(observed))),
    }


def response_cosine(frame: pd.DataFrame, predicted_column: str,
                    observed_column: str, min_proteins: int = 2) -> pd.DataFrame:
    """Cosine between the predicted and observed 4-protein response of each perturbation.

    The panel-level metrics pool every (perturbation, protein) pair into one correlation,
    which a model can win by ranking perturbations correctly while getting each one's
    PATTERN across the four proteins wrong. This asks the other question: given this
    knockout, is the shape of the response right?
    """
    rows = []
    for perturbation, group in frame.groupby("perturbation"):
        pred = group[predicted_column].to_numpy(dtype=float)
        obs = group[observed_column].to_numpy(dtype=float)
        finite = np.isfinite(pred) & np.isfinite(obs)
        pred, obs = pred[finite], obs[finite]
        if len(pred) < min_proteins:
            cosine = np.nan
        else:
            norms = np.linalg.norm(pred) * np.linalg.norm(obs)
            cosine = float(pred @ obs / norms) if norms > 0 else np.nan
        rows.append({"perturbation": perturbation, "n_proteins": int(len(pred)),
                     "cosine": cosine})
    return pd.DataFrame(rows)


def per_group_correlation(frame: pd.DataFrame, group_column: str, predicted_column: str,
                          observed_column: str) -> pd.DataFrame:
    """Spearman and Pearson within each protein, or within each perturbation."""
    rows = []
    for name, group in frame.groupby(group_column):
        pred = group[predicted_column].to_numpy(dtype=float)
        obs = group[observed_column].to_numpy(dtype=float)
        finite = np.isfinite(pred) & np.isfinite(obs)
        pred, obs = pred[finite], obs[finite]
        if len(pred) < 3 or np.std(pred) == 0 or np.std(obs) == 0:
            rows.append({group_column: name, "n": len(pred),
                         "spearman": np.nan, "pearson": np.nan})
            continue
        rows.append({group_column: name, "n": int(len(pred)),
                     "spearman": float(stats.spearmanr(pred, obs).statistic),
                     "pearson": float(stats.pearsonr(pred, obs).statistic)})
    return pd.DataFrame(rows)


def score_effect_subsets(effects: pd.DataFrame, predicted_column: str,
                        observed_column: str, n_boot: int, seed: int,
                        unit_columns: tuple[str, ...] = ("perturbation",)) -> pd.DataFrame:
    """One row per effect subset: the metric suite, response cosine and unit-level CIs.

    ONE scorer for every Task B in-silico perturbation model. The linear ISP and the
    generic `task_b` adapter each grew their own, and they diverged: the adapter scored
    all 100 (perturbation, protein) pairs -- self effects and under-sampled ones included
    -- while the linear runner scored the four frozen subsets, so RegVelo and the linear
    baseline could not be put in the same table. Item 26 fixes the subsets before any
    model is scored; a scorer that ignores them is measuring a different benchmark.

    `effects` must already carry the `in_<subset>` flags from `annotate_effect_sets`.
    Cosines are taken WITHIN a replicate when the table has one, because the shape of a
    response is a property of a single experiment rather than of an average over batches.
    """
    rows = []
    for subset in EFFECT_SETS:
        block = effects[effects[f"in_{subset}"].fillna(False).astype(bool)]
        if block.empty:
            continue
        row = {"effect_subset": subset,
               **metric_suite(block[predicted_column].to_numpy(dtype=float),
                              block[observed_column].to_numpy(dtype=float))}
        if "replicate" in block:
            cosines = pd.concat([
                response_cosine(part, predicted_column, observed_column)
                for _, part in block.groupby("replicate")
            ])
        else:
            cosines = response_cosine(block, predicted_column, observed_column)
        row["response_cosine_mean"] = float(cosines["cosine"].mean())
        for metric in ("pearson", "spearman"):
            for unit in unit_columns:
                lo, hi = bootstrap_over_units(block, unit, predicted_column,
                                              observed_column, metric, n_boot, seed)
                row[f"{metric}_{unit}_lo"], row[f"{metric}_{unit}_hi"] = lo, hi
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_over_units(frame: pd.DataFrame, unit_column: str, predicted_column: str,
                         observed_column: str, metric: str, n_boot: int,
                         seed: int) -> tuple[float, float]:
    """Percentile CI for one metric, resampling whole UNITS with replacement.

    `unit_column` is `perturbation` or `replicate`. Every row of a resampled unit comes
    along with it, so the interval reflects how many independent experiments there are --
    24 perturbations or 3 replicates -- and not how many cells were sequenced.
    """
    # Each unit's row positions are resolved ONCE, then every draw is a numpy gather.
    # Filtering and concatenating the frame inside the loop instead cost 24 boolean masks
    # and a concat per draw -- roughly five million pandas operations across a full table,
    # which is why the evaluation step took longer than the training it was scoring.
    predicted = frame[predicted_column].to_numpy(dtype=float)
    observed = frame[observed_column].to_numpy(dtype=float)
    labels = frame[unit_column].to_numpy()
    units = pd.unique(labels)
    rows_per_unit = [np.nonzero(labels == unit)[0] for unit in units]

    rng = np.random.default_rng(seed)
    draws = np.full(n_boot, np.nan)
    for b in range(n_boot):
        drawn = rng.integers(0, len(units), len(units))
        rows = np.concatenate([rows_per_unit[i] for i in drawn])
        draws[b] = metric_suite(predicted[rows], observed[rows])[metric]
    if np.isnan(draws).all():
        return float("nan"), float("nan")
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def bootstrap_correlation_difference(
    frame: pd.DataFrame,
    unit_column: str,
    full_column: str,
    ablation_column: str,
    observed_column: str,
    metric: str,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
    """Paired bootstrap mean and CI for correlation(full) - correlation(ablation).

    One draw of whole experimental units is shared by the two correlations. Complete
    cases are selected jointly before resampling, so neither arm can gain a different
    effective sample within a draw because it has a missing prediction. The return order
    is ``(mean, lower, upper)`` for the bootstrap distribution.
    """
    if metric not in {"spearman", "pearson"}:
        raise ValueError("metric must be 'spearman' or 'pearson'")
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1")

    full = frame[full_column].to_numpy(dtype=float)
    ablation = frame[ablation_column].to_numpy(dtype=float)
    observed = frame[observed_column].to_numpy(dtype=float)
    labels = frame[unit_column].to_numpy()
    finite = (np.isfinite(full) & np.isfinite(ablation) & np.isfinite(observed)
              & pd.notna(labels))
    full, ablation, observed, labels = (
        values[finite] for values in (full, ablation, observed, labels)
    )
    units = pd.unique(labels)
    if len(units) == 0:
        return float("nan"), float("nan"), float("nan")
    rows_per_unit = [np.nonzero(labels == unit)[0] for unit in units]

    rng = np.random.default_rng(seed)
    draws = np.full(n_boot, np.nan)
    for b in range(n_boot):
        drawn = rng.integers(0, len(units), len(units))
        rows = np.concatenate([rows_per_unit[i] for i in drawn])
        full_corr = metric_suite(full[rows], observed[rows])[metric]
        ablation_corr = metric_suite(ablation[rows], observed[rows])[metric]
        draws[b] = full_corr - ablation_corr

    if np.isnan(draws).all():
        return float("nan"), float("nan"), float("nan")
    mean = np.nanmean(draws)
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return float(mean), float(lo), float(hi)
