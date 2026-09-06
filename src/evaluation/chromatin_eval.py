"""Tasks A-D for the chromatin→RNA experiment, and the rule that keeps them honest.

§13 is the reason this module exists separately from the training script. A corrupted
run must be scored twice, against two different things, and the two answers mean
different things:

  INTERNAL   score the checkpoint against the law it was TRAINED with — its own velocity,
             its own G. This asks "did the optimiser satisfy the objective it was given?"
             A shuffle run can and should look self-consistent here.

  BIOLOGICAL score the checkpoint against the TRUE velocity and the TRUE G, whatever it
             was trained on. This asks "did it learn the chromatin→RNA dynamics?" This is
             the number that answers the experiment's question.

Scoring a corrupted checkpoint only against its corrupted law makes every ablation look
successful. Every function here that takes a velocity or a G takes it as an argument, so
the caller has to say which of the two tracks it is on.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.stats import rankdata
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

from src.evaluation.foscttm import calc_domainAveraged_FOSCTTM
from src.evaluation.knn_alignment import alignment_curve
from src.losses.entropic_ot import sinkhorn_plan

RETRIEVAL_DEPTHS = [1, 5, 10]
# MaxFuse-style FOSKNN: k as a fraction of n, so HSPC (~5k) and BMMC (~14k) share an axis.
FOSKNN_FRACTIONS = (0.01, 0.05, 0.10)


def column_pearson(predicted: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Pearson per column, NaN where a column is constant and the correlation is undefined."""
    a = predicted - predicted.mean(axis=0)
    b = observed - observed.mean(axis=0)
    denominator = np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denominator > 0, (a * b).sum(axis=0) / denominator, np.nan)


def column_spearman(predicted: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Spearman per column, as Pearson on ranks — one ranking pass instead of n_genes calls."""
    return column_pearson(rankdata(predicted, axis=0), rankdata(observed, axis=0))


def row_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine per row, zero where either row has no length."""
    scale = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(scale > 0, (a * b).sum(axis=1) / scale, 0.0)


def row_pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return row_cosine(a - a.mean(axis=1, keepdims=True), b - b.mean(axis=1, keepdims=True))


def column_nrmse(predicted: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """RMSE per gene over that gene's own spread, so genes are comparable."""
    error = np.sqrt(((predicted - observed) ** 2).mean(axis=0))
    spread = observed.std(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(spread > 0, error / spread, np.nan)


def task_a_state(predicted: np.ndarray, observed: np.ndarray,
                 gene_names: list[str]) -> tuple[dict, pd.DataFrame]:
    """§14. Predict held-out RNA from held-out ATAC, then reveal the pairing and score.

    Per-gene and per-cell are different questions and both are reported: a map can get
    every cell's overall profile roughly right (high cell cosine) while ranking cells
    wrongly for every individual gene (low gene correlation), and that failure is exactly
    what a chromatin→RNA claim would be wrong about.
    """
    per_gene = pd.DataFrame({
        "gene": gene_names,
        "spearman": column_spearman(predicted, observed),
        "pearson": column_pearson(predicted, observed),
        "nrmse": column_nrmse(predicted, observed),
    })
    summary = {
        "gene_spearman_median": float(np.nanmedian(per_gene["spearman"])),
        "gene_pearson_median": float(np.nanmedian(per_gene["pearson"])),
        "gene_nrmse_median": float(np.nanmedian(per_gene["nrmse"])),
        "cell_cosine_median": float(np.median(row_cosine(predicted, observed))),
        "cell_pearson_median": float(np.median(row_pearson(predicted, observed))),
        "n_genes": int(predicted.shape[1]),
        "n_cells": int(predicted.shape[0]),
    }
    return summary, per_gene


def fosknn_scores(predicted: np.ndarray, observed: np.ndarray, seed: int) -> dict:
    """Fraction of cells whose true partner sits inside the top {1,5,10}% of candidates."""
    values = alignment_curve(predicted, observed, np.asarray(FOSKNN_FRACTIONS), seed=seed)
    return {f"fosknn_frac{frac:g}": float(score)
            for frac, score in zip(FOSKNN_FRACTIONS, values)}


def retrieval_metrics(predicted: np.ndarray, observed: np.ndarray, seed: int = 0) -> dict:
    """Where a cell's true partner sits in the ranking of every candidate partner.

    FOSCTTM is reported WITH the score a zero-information predictor gets on the same data.
    A constant prediction scores 0.25, not 0.5: the ranking of observed cells by distance
    to a single point is identical for every query, so one direction of the average is 0.5
    and the other is 0. 0.5 is what a DESTROYED pairing gives, not what an uninformative
    map gives, and reading 0.25 as "half way to chance" credits a collapsed map.

    top-k is reported too, but it depends strongly on how many candidates there are
    (chance is 1/n), so it is only comparable between methods scored on the same cells.
    `partner_diversity` catches the hub degeneracy those rates hide: a map whose cloud sits
    beside a handful of observed cells answers most queries with the same few partners and
    still posts a plausible top-k.
    """
    neighbors = NearestNeighbors(n_neighbors=min(len(observed), max(RETRIEVAL_DEPTHS)))
    neighbors.fit(observed)
    ranked = neighbors.kneighbors(predicted, return_distance=False)
    truth = np.arange(len(predicted))[:, None]
    metrics = {f"top{depth}": float((ranked[:, :depth] == truth).any(axis=1).mean())
               for depth in RETRIEVAL_DEPTHS}
    metrics["n_candidates"] = int(len(observed))
    partners = ranked[:, 0]
    counts = np.bincount(partners, minlength=len(observed))
    metrics["partner_diversity"] = float(len(np.unique(partners)) / len(predicted))
    metrics["top_partner_share"] = float(counts.max() / len(predicted))
    metrics["foscttm"] = float(np.mean(calc_domainAveraged_FOSCTTM(predicted, observed)))
    constant = np.tile(predicted.mean(axis=0), (len(predicted), 1)).astype(predicted.dtype)
    metrics["foscttm_constant_floor"] = float(
        np.mean(calc_domainAveraged_FOSCTTM(constant, observed)))
    metrics.update(fosknn_scores(predicted, observed, seed))
    return metrics


def majority_class_accuracy(labels: np.ndarray) -> float:
    """What a predictor that always answers the commonest cell type would score."""
    _, counts = np.unique(labels, return_counts=True)
    return float(counts.max() / len(labels))


def cell_type_match_accuracy(predicted: np.ndarray, observed: np.ndarray,
                             labels: np.ndarray) -> float:
    """How often a cell's nearest partner is at least the RIGHT KIND of cell.

    Exact retrieval is a hard ceiling on a population where many cells are genuinely
    interchangeable; getting the cell type right is the weaker claim that still has to
    hold for the map to be useful.
    """
    nearest = NearestNeighbors(n_neighbors=1).fit(observed)
    partner = nearest.kneighbors(predicted, return_distance=False).ravel()
    return float((labels[partner] == labels).mean())


def sinkhorn_assignment(predicted: np.ndarray, observed: np.ndarray,
                        epsilon: float = 0.05, n_iterations: int = 200) -> np.ndarray:
    """Row-argmax of an entropic OT plan: matching that accounts for the whole population.

    Nearest neighbour lets one popular RNA cell be the answer for hundreds of ATAC cells.
    Transport cannot: the plan's marginals hold, so a cell only wins a partner no
    better-placed cell is claiming. The two matchers together separate "phi lands in the
    right neighbourhood" from "phi orders cells within it".
    """
    plan = sinkhorn_plan(predicted, observed, epsilon, n_iterations)
    return plan.argmax(dim=1).cpu().numpy()


def sinkhorn_plan_retrieval(plan: np.ndarray) -> dict:
    """Partner ranks induced by transport mass, averaged over both modalities.

    Distance-based FOSCTTM is a property of the embedding and is therefore identical for
    kNN and Sinkhorn. A Sinkhorn-specific score must rank candidates by coupling mass.
    """
    scores = np.asarray(plan)
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1] or len(scores) < 2:
        raise ValueError("Sinkhorn retrieval needs a square plan with at least two cells")
    n = len(scores)
    truth = np.diag(scores)
    row_rank = (scores > truth[:, None]).sum(axis=1)
    column_rank = (scores > truth[None, :]).sum(axis=0)
    metrics = {
        "foscttm": float(np.mean((row_rank + column_rank) / (2.0 * (n - 1)))),
    }
    for depth in RETRIEVAL_DEPTHS:
        k = min(depth, n)
        metrics[f"top{depth}"] = float(
            0.5 * ((row_rank < k).mean() + (column_rank < k).mean()))
    for fraction in FOSKNN_FRACTIONS:
        k = max(1, int(round(fraction * n)))
        metrics[f"fosknn_frac{fraction:g}"] = float(
            0.5 * ((row_rank < k).mean() + (column_rank < k).mean()))
    return metrics


def task_b_pairing(predicted: np.ndarray, observed: np.ndarray, labels: np.ndarray | None,
                   max_sinkhorn_cells: int, seed: int) -> dict:
    """§15. Both matchers on the same held-out populations, with the pairing hidden.

    `labels` is None for a dataset with no cell-type annotation (HSPC ships none), and the
    cell-type metrics are then absent rather than reported as a meaningless 1.0.
    """
    metrics = {f"knn_{name}": value for name, value in
               retrieval_metrics(predicted, observed, seed).items()}
    if labels is not None:
        metrics["knn_cell_type_accuracy"] = cell_type_match_accuracy(predicted, observed, labels)
        # Below this, the matcher is worse than always answering the commonest type.
        metrics["cell_type_majority_baseline"] = majority_class_accuracy(labels)

    rows = np.arange(len(predicted))
    if len(rows) > max_sinkhorn_cells:
        rows = np.sort(np.random.default_rng(seed).choice(rows, max_sinkhorn_cells, False))
    assigned = predicted[rows]
    partners = observed[rows]
    plan = sinkhorn_plan(assigned, partners, epsilon=0.05, n_iterations=200).cpu().numpy()
    metrics.update({f"sinkhorn_{name}": value
                    for name, value in sinkhorn_plan_retrieval(plan).items()})
    metrics["sinkhorn_n_cells"] = int(len(rows))
    assignment = plan.argmax(axis=1)
    if labels is not None:
        metrics["sinkhorn_cell_type_accuracy"] = float(
            (labels[rows][assignment] == labels[rows]).mean())
    return metrics


def foscttm_within(predicted: np.ndarray, observed: np.ndarray,
                   groups: np.ndarray) -> dict:
    """FOSCTTM computed inside each group, which is the harder and more honest question.

    Across the whole population a map can score well by separating days or donors, which
    is a batch effect it did not have to learn anything about chromatin to find. Within a
    day or a donor that shortcut is gone.
    """
    scores = {}
    for level in np.unique(groups):
        rows = np.flatnonzero(groups == level)
        if len(rows) < 10:
            continue
        scores[str(level)] = float(np.mean(
            calc_domainAveraged_FOSCTTM(predicted[rows], observed[rows])))
    scores["mean"] = float(np.mean(list(scores.values()))) if scores else float("nan")
    return scores


def knn_label_transfer(source: np.ndarray, source_labels: np.ndarray,
                       target: np.ndarray, target_labels: np.ndarray,
                       n_neighbors: int) -> float:
    """Accuracy of labelling the mapped ATAC cells from their RNA neighbours."""
    neighbors = NearestNeighbors(n_neighbors=n_neighbors).fit(source)
    indices = neighbors.kneighbors(target, return_distance=False)
    voted = pd.DataFrame(source_labels[indices]).mode(axis=1)[0].to_numpy()
    return float((voted == target_labels).mean())


def inverse_simpson_mixing(embedding: np.ndarray, modality: np.ndarray,
                           n_neighbors: int) -> float:
    """iLISI: the effective number of modalities in each cell's neighbourhood, averaged.

    1 means a cell only ever sees its own modality, 2 means the two are locally
    interchangeable. Reported, not optimised — §16 is explicit that the method must not be
    tuned around this number.
    """
    neighbors = NearestNeighbors(n_neighbors=n_neighbors).fit(embedding)
    indices = neighbors.kneighbors(embedding, return_distance=False)
    labels = pd.factorize(modality)[0][indices]
    counts = np.stack([(labels == value).mean(axis=1) for value in np.unique(labels)], axis=1)
    return float(np.mean(1.0 / (counts ** 2).sum(axis=1)))


def graph_connectivity(embedding: np.ndarray, labels: np.ndarray, n_neighbors: int) -> float:
    """Per label, the share of its cells in its largest connected kNN component.

    An integration that scatters one cell type into disconnected islands has not put the
    modalities in the same space, however well the global mixing scores.
    """
    scores = []
    for level in np.unique(labels):
        rows = np.flatnonzero(labels == level)
        if len(rows) < n_neighbors + 1:
            continue
        neighbors = NearestNeighbors(n_neighbors=n_neighbors).fit(embedding[rows])
        graph = neighbors.kneighbors_graph(embedding[rows], mode="connectivity")
        _, membership = connected_components(sparse.csr_matrix(graph), directed=False)
        scores.append(np.bincount(membership).max() / len(rows))
    return float(np.mean(scores)) if scores else float("nan")


def task_c_integration(mapped: np.ndarray, observed: np.ndarray, labels: np.ndarray,
                       cluster: Callable[[np.ndarray], np.ndarray], n_neighbors: int) -> dict:
    """§16. Both modalities in one space, scored for mixing AND for structure.

    `cluster` is run on the JOINT embedding, not on the observed RNA. Clustering the RNA
    alone and duplicating the labels across both halves produces an ARI and an NMI that
    cannot see phi at all: they measure how well Leiden recovers cell types in the RNA,
    which is a property of the dataset and identical for every one of the 18 conditions.

    Secondary by design: a map that collapses every cell onto one point mixes the
    modalities perfectly and preserves nothing, so mixing is only meaningful beside ARI,
    NMI, ASW and connectivity.
    """
    joint = np.vstack([mapped, observed])
    modality = np.array(["atac"] * len(mapped) + ["rna"] * len(observed))
    joint_labels = np.concatenate([labels, labels])
    joint_clusters = cluster(joint)
    return {
        "label_transfer_accuracy": knn_label_transfer(observed, labels, mapped, labels,
                                                      n_neighbors),
        "ari": float(adjusted_rand_score(joint_labels, joint_clusters)),
        "nmi": float(normalized_mutual_info_score(joint_labels, joint_clusters)),
        "cell_type_asw": float((silhouette_score(joint, joint_labels) + 1.0) / 2.0),
        "ilisi": inverse_simpson_mixing(joint, modality, n_neighbors),
        "graph_connectivity": graph_connectivity(joint, joint_labels, n_neighbors),
    }


def by_group(scorer, predicted: np.ndarray, observed: np.ndarray,
             groups: np.ndarray, min_cells: int = 50) -> dict:
    """Run a metric separately per group, for the per-lineage read §8 asks for.

    A pooled cosine over a branching population can be carried entirely by one large
    lineage: BMMC is 40% T/NK cells, so a model that gets lymphoid right and erythroid
    backwards still scores positively overall. Per lineage, that cancellation is visible.
    """
    scores = {}
    for level in np.unique(groups):
        rows = np.flatnonzero(groups == level)
        if len(rows) >= min_cells:
            scores[str(level)] = scorer(predicted[rows], observed[rows])
    return scores


def task_d_kinetics(pushforward: np.ndarray, reference: np.ndarray) -> dict:
    """§17. The Jacobian push-forward against an RNA velocity computed from RNA alone.

    The reference never enters training, so agreement here is a prediction and not a fit.
    Direction and magnitude are reported apart: kappa is bounded, so the model can only be
    asked to get the direction right, and the norm ratio says how far off the scale is
    without letting that contaminate the cosine.
    """
    cosine = row_cosine(pushforward, reference)
    # The same cosine with each side's population mean removed. A phi that has collapsed
    # produces a near-constant push-forward, and a constant field correlates with ANY
    # other field's mean direction — so the raw cosine can rank a destroyed control above
    # the real one while measuring nothing per-cell. The centred value is what says
    # whether the agreement is about individual cells.
    centred = row_cosine(pushforward - pushforward.mean(axis=0),
                         reference - reference.mean(axis=0))
    predicted_norm = np.linalg.norm(pushforward, axis=1)
    reference_norm = np.linalg.norm(reference, axis=1)
    usable = reference_norm > 0
    scale_free = np.sqrt(np.clip(1.0 - cosine ** 2, 0.0, 1.0))
    nonzero = reference != 0
    positive_share = float((reference[nonzero] > 0).mean()) if nonzero.any() else 0.0
    ceiling = float(nonzero.mean())
    ratio = (np.median(predicted_norm[usable] / reference_norm[usable])
             if usable.any() else np.nan)
    residual = (np.median(np.linalg.norm(pushforward - reference, axis=1)[usable]
                          / reference_norm[usable])
                if usable.any() else np.nan)
    return {
        "cell_cosine_median": float(np.median(cosine)),
        "cell_cosine_centred_median": float(np.median(centred)),
        "cell_cosine_positive_fraction": float((cosine > 0).mean()),
        "cell_cosine_centred_positive_fraction": float((centred > 0).mean()),
        "scale_free_residual_median": float(np.median(scale_free)),
        "gene_pearson_median": float(np.nanmedian(column_pearson(pushforward, reference))),
        "gene_spearman_median": float(np.nanmedian(column_spearman(pushforward, reference))),
        "sign_agreement": float((np.sign(pushforward) == np.sign(reference)).mean()),
        "sign_agreement_ceiling": ceiling,
        "sign_agreement_chance": ceiling * (positive_share ** 2 + (1 - positive_share) ** 2),
        "sign_agreement_constant_positive": ceiling * positive_share,
        "normalized_residual": float(residual),
        "normalized_residual_is_informative": bool(0.1 < ratio < 10.0) if np.isfinite(ratio) else False,
        "norm_ratio_median": float(ratio) if np.isfinite(ratio) else float("nan"),
        "n_cells": int(len(cosine)),
    }


def task_d_tracks(jvp_train: np.ndarray, rhs_train: np.ndarray,
                  jvp_true: np.ndarray, rhs_true: np.ndarray) -> dict:
    """§13. Internal self-consistency vs biological law-RHS, kept as two named answers.

    INTERNAL   J_phi v_train vs the RHS the run was trained with (its own G, its own v).
    BIOLOGICAL J_phi v_true vs the RHS under G_true, whatever the run was trained on.

    Agreement with an RNA velocity is a separate, later comparison and must not replace
    either of these — scoring a shuffled checkpoint only against shuffled training law
    used to make every ablation look successful.
    """
    return {
        "internal_vs_law": task_d_kinetics(jvp_train, rhs_train),
        "biological_vs_law": task_d_kinetics(jvp_true, rhs_true),
    }
