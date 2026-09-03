"""Perturbation Δ, not pairing. Knockouts must not be kNN neighbours of each other."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from scipy import stats


def group_deltas(values: np.ndarray, adt_names: list[str], groups: pd.Series,
                 control_label: str, min_cells: int) -> pd.DataFrame:
    """Mean shift versus the control, per (knockout, protein)."""
    is_control = (groups == control_label).to_numpy()
    control_mean = values[is_control].mean(axis=0)
    rows = []
    for target in sorted(set(groups.unique()) - {control_label}):
        mask = (groups == target).to_numpy()
        if mask.sum() < min_cells:
            continue
        delta = values[mask].mean(axis=0) - control_mean
        for j, adt in enumerate(adt_names):
            rows.append({"knockout": target, "adt": adt, "delta": float(delta[j])})
    return pd.DataFrame(rows)


def build_scoring_set(observed: pd.DataFrame, adt_gene_map: dict[str, str],
                      q_max: float, d_min: float) -> pd.DataFrame:
    """Pairs with a real observed effect, minus the unanswerable self-pairs.

    A CRISPR indel destroys the protein while leaving the transcript intact, so
    scoring an RNA→protein map on (CD86 KO → CD86 protein) asks an unanswerable
    question. Those pairs still validate the ground truth; they are not a metric.
    """
    df = observed.copy()
    df["is_self_pair"] = [
        adt_gene_map[adt] == knockout
        for adt, knockout in zip(df["adt"], df["knockout"])
    ]
    significant = (df["qval"] < q_max) & (df["cohens_d"].abs() > d_min)
    kept = df[significant & ~df["is_self_pair"]].copy()
    dropped = df[significant & df["is_self_pair"]]
    print(f"[scoring set] {len(kept)} pairs (q<{q_max}, |d|>{d_min}), "
          f"{len(dropped)} self-pairs excluded: "
          f"{', '.join(f'{r.knockout}->{r.adt}' for r in dropped.itertuples())}")
    return kept[["knockout", "adt", "delta", "cohens_d"]].rename(columns={"delta": "obs_delta"})


def sufficient_knockouts(groups: pd.Series, replicates: pd.Series, control_label: str,
                         min_cells: int, min_replicates: int) -> list[str]:
    """Knockouts sampled well enough to carry an effect estimate at all.

    The rule is `min_cells` cells in EACH of at least `min_replicates` replicates, not
    that many cells in total that happen to span that many replicates. A knockout with
    200 cells in one replicate and 2 in another has one replicate-level effect estimate,
    not two, and cannot contribute a replicate correlation.

    Sufficiency is a property of the DESIGN -- how many cells, spread over how many
    replicates -- so it can be fixed before any model is scored. Selecting instead on
    whether the observed effect reached significance would define the benchmark by its
    own answers, and would drop every perturbation whose true effect is small, which is
    where a cross-modal model has the most to prove.

    ONE definition, shared by the observed-response table (run_kot_crispr.py) and the
    benchmark scoring set (tools/papalexi.py). Two copies would be two thresholds.
    """
    kept = []
    for target in sorted(set(groups.unique()) - {control_label}):
        mask = (groups == target).to_numpy()
        per_replicate = replicates[mask].value_counts()
        if int((per_replicate >= min_cells).sum()) >= min_replicates:
            kept.append(target)
    return kept


def replicate_effects(protein: np.ndarray, adt_names: list[str], cognate_rna: pd.DataFrame,
                      adt_gene_map: dict[str, str], groups: pd.Series, replicates: pd.Series,
                      control_label: str) -> pd.DataFrame:
    """Observed response per (perturbation, replicate, protein), against that replicate's NT.

    The control is taken WITHIN the replicate, not pooled across all of them. A batch
    shift shared by the knockout and its own controls then cancels, which is the whole
    reason the replicate is in the key; pooling the controls would leave that shift in
    every effect and make the replicates look more consistent than they are.

    `protein_effect_se` is the Welch standard error of the difference in means, so an
    effect from 20 cells is visibly less certain than one from 1,000 rather than sitting
    in the table looking equally solid.
    """
    gene_of = {adt: adt_gene_map[adt] for adt in adt_names}
    is_control = (groups == control_label).to_numpy()
    rows = []
    for replicate in sorted(replicates.unique()):
        in_replicate = (replicates == replicate).to_numpy()
        control_mask = is_control & in_replicate
        if control_mask.sum() < 2:
            continue
        for target in sorted(set(groups.unique()) - {control_label}):
            ko_mask = (groups == target).to_numpy() & in_replicate
            if ko_mask.sum() < 2:
                continue
            for j, adt in enumerate(adt_names):
                ko, nt = protein[ko_mask, j], protein[control_mask, j]
                gene = gene_of[adt]
                cognate = float("nan")
                if gene in cognate_rna.columns:
                    values = cognate_rna[gene].to_numpy()
                    cognate = float(values[ko_mask].mean() - values[control_mask].mean())
                rows.append({
                    "perturbation": target,
                    "replicate": str(replicate),
                    "protein": adt,
                    "n_ko_cells": int(ko_mask.sum()),
                    "n_nt_cells": int(control_mask.sum()),
                    "delta_protein": float(ko.mean() - nt.mean()),
                    "delta_cognate_rna": cognate,
                    "protein_effect_se": float(np.sqrt(ko.var(ddof=1) / len(ko)
                                                       + nt.var(ddof=1) / len(nt))),
                })
    return pd.DataFrame(rows)


def build_sufficiency_scoring_set(observed: pd.DataFrame, adt_gene_map: dict[str, str],
                                  knockouts: list[str]) -> pd.DataFrame:
    """Every non-self (knockout, protein) pair for the sufficiently sampled knockouts.

    Same self-pair exclusion as `build_scoring_set` and the same output columns; what
    differs is that nothing here consults a p-value or an effect size.
    """
    df = observed[observed["knockout"].isin(knockouts)].copy()
    df["is_self_pair"] = [
        adt_gene_map[adt] == knockout
        for adt, knockout in zip(df["adt"], df["knockout"])
    ]
    kept = df[~df["is_self_pair"]].copy()
    dropped = df[df["is_self_pair"]]
    print(f"[scoring set] {len(kept)} pairs over {len(knockouts)} sufficiently sampled "
          f"knockouts, {len(dropped)} self-pairs excluded: "
          f"{', '.join(f'{r.knockout}->{r.adt}' for r in dropped.itertuples())}")
    return kept[["knockout", "adt", "delta", "cohens_d"]].rename(columns={"delta": "obs_delta"})


def finite_pairs(*arrays: np.ndarray) -> np.ndarray:
    """Mask of positions where every input is finite.

    One definition for `score` and `bootstrap_spearman_gap` alike. They disagreed before:
    score dropped non-finite pairs, the bootstrap kept them, so a resample containing one
    produced a NaN gap that np.nanmean silently discarded -- leaving the CI averaged over
    only those resamples that happened to miss the missing pair, which is a biased subset,
    not a wider interval.
    """
    mask = np.ones(len(arrays[0]), dtype=bool)
    for array in arrays:
        mask &= np.isfinite(array)
    return mask


def score(pred: np.ndarray, obs: np.ndarray) -> dict:
    finite = finite_pairs(pred, obs)
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

    Paired because both predictors are scored on the same resampled pairs, which is what
    makes the difference interpretable with only ~20 points. Pairs that either predictor
    cannot score are dropped ONCE, up front, so both arms see the identical pair set.
    """
    finite = finite_pairs(pred, reference, obs)
    pred, reference, obs = pred[finite], reference[finite], obs[finite]
    rng = np.random.default_rng(seed)
    n = len(obs)
    if n < 3:
        return float("nan"), float("nan"), float("nan")
    gaps = np.full(n_boot, np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.std(pred[idx]) == 0 or np.std(reference[idx]) == 0 or np.std(obs[idx]) == 0:
            continue
        gaps[b] = (stats.spearmanr(pred[idx], obs[idx]).statistic
                   - stats.spearmanr(reference[idx], obs[idx]).statistic)
    lo, hi = np.nanpercentile(gaps, [2.5, 97.5])
    return float(np.nanmean(gaps)), float(lo), float(hi)


def column_order_warning(aligned_protein: np.ndarray, observed: np.ndarray,
                         adt_names: list[str]) -> str | None:
    """A warning when the aligned protein columns do not line up with the observed ADTs.

    DIAGNOSTIC ONLY. It must never decide which predictor runs: a normalisation
    mismatch can make a column correlate more with a neighbour protein than with
    itself, and using that as a dispatch silently swapped KOT to kNN. Column
    ORDER was never the problem.

    Returns None when the check passes, or when it does not apply because the embedding
    is a latent space with no ADT columns to order.
    """
    n_adt = len(adt_names)
    if aligned_protein.shape[1] != n_adt:
        return None
    # Held-out cells have no protein embedding under the holdout_second protocol; scoring
    # their NaN rows would make every correlation NaN and turn this into a false alarm.
    rows = np.isfinite(aligned_protein).all(axis=1)
    aligned_protein, observed = aligned_protein[rows], observed[rows]
    corr = np.array([
        [abs(stats.spearmanr(aligned_protein[:, i], observed[:, j]).statistic)
         for j in range(n_adt)]
        for i in range(n_adt)
    ])
    if np.isnan(corr).all(axis=1).any():
        return "an aligned protein column is constant; column-order check undefined"
    argmax = np.nanargmax(corr, axis=1)
    if np.all(argmax == np.arange(n_adt)):
        return None
    return (f"aligned protein columns do not best-match {adt_names} in order; "
            f"argmax per row = {argmax.tolist()} (a normalisation difference alone can "
            f"cause this; it does not by itself mean the columns are misordered)")


def knn_indices(source: np.ndarray, target: np.ndarray, k: int,
                allowed_neighbors: np.ndarray, device: torch.device,
                batch: int = 1024) -> np.ndarray:
    """Row i → k nearest rows in `target` among `allowed_neighbors`.

    `allowed_neighbors` is the CRISPR atlas: control cells. Query cells may be
    knockouts; they must not retrieve other knockout protein profiles.
    """
    if int(allowed_neighbors.sum()) < k:
        raise ValueError(
            f"k={k} neighbours requested but only {int(allowed_neighbors.sum())} "
            "allowed neighbour cells"
        )
    src = torch.as_tensor(np.asarray(source), dtype=torch.float32, device=device)
    tgt = torch.as_tensor(np.asarray(target), dtype=torch.float32, device=device)
    allowed = torch.as_tensor(np.asarray(allowed_neighbors), dtype=torch.bool, device=device)
    # Disallowed rows are masked to inf after cdist, but a NaN row (a held-out cell with
    # no protein embedding) can poison cdist's matmul path for every column, so it is
    # zeroed before the distance rather than after.
    tgt = torch.where(allowed[:, None], tgt, torch.zeros_like(tgt))
    n = src.shape[0]
    out = torch.empty((n, k), dtype=torch.long, device=device)
    for start in range(0, n, batch):
        end = min(start + batch, n)
        dist = torch.cdist(src[start:end], tgt)
        dist[:, ~allowed] = float("inf")
        rows = torch.arange(end - start, device=device)
        dist[rows, torch.arange(start, end, device=device)] = float("inf")
        out[start:end] = torch.topk(dist, k, dim=1, largest=False).indices
    return out.cpu().numpy()


def gather_mean(values: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Mean of `values` over each row's neighbour set."""
    return values[idx].mean(axis=1)


def affine_calibration(predicted: np.ndarray, observed: np.ndarray,
                       rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-protein least-squares (a, b) mapping `predicted` onto `observed` over `rows`.

    A column whose prediction is constant over `rows` has no identifiable scale, so it is
    left alone (a=1, b=0) rather than collapsed to zero.
    """
    p, o = predicted[rows], observed[rows]
    p_mean, o_mean = p.mean(axis=0), o.mean(axis=0)
    p_centred, o_centred = p - p_mean, o - o_mean
    variance = (p_centred * p_centred).sum(axis=0)
    a = np.ones(predicted.shape[1], dtype=np.float64)
    movable = variance > 0
    a[movable] = (p_centred[:, movable] * o_centred[:, movable]).sum(axis=0) / variance[movable]
    return a, o_mean - a * p_mean


def positive_scale(predicted: np.ndarray, observed: np.ndarray, rows: np.ndarray,
                   min_sd_ratio: float = 0.0) -> np.ndarray:
    """Per-protein POSITIVE scale, sd(observed)/sd(predicted), fitted on `rows`.

    A unit correction, not a fit. `affine_calibration`'s least-squares slope can come out
    NEGATIVE, and that is not a rescale: it inverts the prediction. Granting that to an
    ablation lets the ablation undo itself -- the reversed-velocity arm predicts protein
    backwards, and a signed slope flips it back, which turned its Spearman from -0.12 into
    +0.42 and made it beat the unablated model. A model has to get the direction right on
    its own; only its scale is allowed to be corrected.

    A protein whose prediction is constant over `rows` has no identifiable scale and is
    left at 1 rather than blown up to infinity.

    `min_sd_ratio` guards the case in between, which is worse than either: a prediction
    that is not quite constant. The zero-velocity control arm's phi has sd 0.002x the
    observed spread, so this divides its numerical dust by 0.002 and hands the scorer a
    ranked prediction -- that degenerate arm then posted the HIGHEST primary Spearman of
    any ablation. A column below the ratio is scaled to NaN so it is dropped from scoring
    and reported as absent, rather than scored as though it had predicted something. The
    default is 0.0, which is the historical behaviour; the CRISPR benchmark sets it.
    """
    predicted_sd = predicted[rows].std(axis=0)
    observed_sd = observed[rows].std(axis=0)
    scale = np.ones(predicted.shape[1], dtype=np.float64)
    movable = predicted_sd > 0
    scale[movable] = observed_sd[movable] / predicted_sd[movable]
    if min_sd_ratio > 0:
        degenerate = movable & (predicted_sd < min_sd_ratio * observed_sd)
        scale[degenerate] = np.nan
    return scale


def calibrate_direct(predicted: np.ndarray, observed: np.ndarray,
                     control_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-protein affine map from φ units (CLR) into the benchmark's units.

    Without this, pooling pairs into one Spearman ranks a scale mismatch, not
    the map. kNN baselines already sit in observed units. Fit on control cells
    only; group_deltas subtracts the control mean so the intercept cannot
    manufacture a knockout effect.
    """
    rows = np.nonzero(control_mask)[0]
    a, b = affine_calibration(predicted.astype(np.float64), observed.astype(np.float64), rows)
    return (predicted * a + b).astype(np.float32), a


def predict_protein(emb_rna: np.ndarray, emb_protein: np.ndarray,
                    observed: np.ndarray, control_mask: np.ndarray,
                    k: int, device: torch.device, use_direct: bool) -> tuple[np.ndarray, str]:
    """Predicted protein per cell: phi(r) directly, or kNN imputation from control cells.

    `use_direct` is decided by the CALLER from the run's stamped protocol
    (kot_use_feature_space puts phi(r) in the protein panel), never inferred from the
    data. Inferring it meant a normalisation quirk in one of four proteins could silently
    swap the estimator, which is not a decision that should rest on a correlation.

    The shape is still checked: a direct prediction that is not in ADT coordinates would
    be scored against proteins it does not represent.
    """
    if not use_direct:
        # A run under the holdout_second protocol has no protein embedding for its
        # held-out cells (NaN by construction, src/evaluation/protocol.scatter_rows), so
        # those rows are not available as neighbours whatever the control mask says.
        available = control_mask & np.isfinite(emb_protein).all(axis=1)
        idx = knn_indices(emb_rna, emb_protein, k, available, device)
        return gather_mean(observed, idx).astype(np.float32), "knn_control"
    if emb_rna.shape[1] != observed.shape[1]:
        raise ValueError(
            f"direct prediction needs the RNA embedding in ADT coordinates, but it has "
            f"{emb_rna.shape[1]} columns against {observed.shape[1]} ADTs"
        )
    return np.asarray(emb_rna, dtype=np.float32), "direct"
