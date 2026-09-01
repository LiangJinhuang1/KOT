"""CRISPR perturbation-response scoring for unpaired RNA→protein maps.

The paper question is not cell-level pairing (FOSCTTM). It is: given a map
trained without the knockout transcriptomes, does predicted protein Δ(knockout,
protein) recover the observed checkpoint-protein fold-changes?

Two predictors, used on different methods:

  direct        φ(r) itself, when the RNA-side embedding is already in ADT
                coordinates (KOT with kot_use_feature_space).
  knn_control   impute observed ADT from k nearest *control* cells in the
                protein embedding. Knockout cells are not allowed as neighbours,
                so the atlas cannot leak the perturbation protein state.
"""
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
    """Paired bootstrap over scoring-set pairs for spearman(pred) - spearman(reference)."""
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


def column_order_warning(aligned_protein: np.ndarray, observed: np.ndarray,
                         adt_names: list[str]) -> str | None:
    """A warning when the aligned protein columns do not line up with the observed ADTs.

    DIAGNOSTIC ONLY. It must never decide which predictor runs, and the reason is
    concrete: KOT's protein-side normalisation disagrees with the benchmark's rna_size
    normalisation enough that PDL2's column correlates 0.10 with its own observed values
    and 0.44 with CD86. Used as a dispatch, that made every KOT run fall back to kNN
    without saying so. Column ORDER was never the problem.

    Returns None when the check passes, or when it does not apply because the embedding
    is a latent space with no ADT columns to order.
    """
    n_adt = len(adt_names)
    if aligned_protein.shape[1] != n_adt:
        return None
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


def predict_protein(emb_rna: np.ndarray, emb_protein: np.ndarray,
                    observed: np.ndarray, control_mask: np.ndarray,
                    k: int, device: torch.device, use_direct: bool) -> tuple[np.ndarray, str]:
    """Predicted protein per cell: phi(r) directly, or kNN imputation from control cells.

    `use_direct` is decided by the CALLER from the run's configuration
    (kot_use_feature_space puts phi(r) in the protein panel), never inferred from the
    data. Inferring it meant a normalisation quirk in one of four proteins could silently
    swap the estimator, which is not a decision that should rest on a correlation.

    The shape is still checked: a direct prediction that is not in ADT coordinates would
    be scored against proteins it does not represent.
    """
    if not use_direct:
        idx = knn_indices(emb_rna, emb_protein, k, control_mask, device)
        return gather_mean(observed, idx).astype(np.float32), "knn_control"
    if emb_rna.shape[1] != observed.shape[1]:
        raise ValueError(
            f"direct prediction needs the RNA embedding in ADT coordinates, but it has "
            f"{emb_rna.shape[1]} columns against {observed.shape[1]} ADTs"
        )
    return np.asarray(emb_rna, dtype=np.float32), "direct"
