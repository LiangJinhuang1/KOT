from __future__ import annotations

from pathlib import Path

import numpy as np

from moscot.problems.cross_modality import TranslationProblem


def assert_context_matches_moscot_inputs(
    context: dict,
    rna_adata,
    second_adata,
    rna_repr_key: str,
    second_repr_key: str,
) -> None:
    assert np.allclose(context["x"], rna_adata.obsm[rna_repr_key])
    assert np.allclose(context["y"], second_adata.obsm[second_repr_key])


def coupling_metrics(coupling: np.ndarray, threshold: float) -> dict:
    values = np.asarray(coupling, dtype=np.float64)
    mass = float(values.sum())
    probs = values[values > 0] / mass
    entropy = float(-np.sum(probs * np.log(probs)))
    effective_support = float(np.exp(entropy))
    total_entries = int(values.size)
    row_mass = values.sum(axis=1)
    col_mass = values.sum(axis=0)

    return {
        "coupling_mass": mass,
        "coupling_entropy": entropy,
        "coupling_effective_support": effective_support,
        "coupling_effective_sparsity": float(1.0 - effective_support / total_entries),
        "coupling_nonzero_fraction": float(np.mean(values > threshold)),
        "coupling_threshold": threshold,
        "coupling_row_mass_min": float(row_mass.min()),
        "coupling_row_mass_max": float(row_mass.max()),
        "coupling_col_mass_min": float(col_mass.min()),
        "coupling_col_mass_max": float(col_mass.max()),
    }


def valid_trace_values(values) -> np.ndarray:
    trace = np.asarray(values, dtype=np.float64).reshape(-1)
    return trace[np.isfinite(trace) & (trace != -1.0)]


def trace_metrics(prefix: str, values) -> dict:
    trace = valid_trace_values(values)
    metrics = {f"{prefix}_trace_length": int(trace.size)}
    if trace.size > 0:
        metrics[f"{prefix}_trace_final"] = float(trace[-1])
        metrics[f"{prefix}_trace_min"] = float(trace.min())
    return metrics


def save_trace(output_dir, filename: str, values) -> None:
    if output_dir is None:
        return
    path = Path(output_dir) / filename
    np.save(path, np.asarray(values))


def solution_metrics(solution, coupling: np.ndarray, cfg: dict) -> dict:
    threshold = float(cfg.get("coupling_sparsity_threshold", 1.0e-12))
    metrics = {
        "moscot_epsilon": float(cfg.get("epsilon", 0.1)),
        "moscot_tau_a": float(cfg.get("tau_a", 1.0)),
        "moscot_tau_b": float(cfg.get("tau_b", 1.0)),
        "moscot_max_iterations": int(cfg.get("max_iterations", 100)),
        "moscot_converged": bool(solution.converged),
        "moscot_cost": float(solution.cost),
    }
    metrics.update(coupling_metrics(coupling, threshold))

    cost_trace = getattr(solution, "_costs", None)
    error_trace = getattr(solution, "_errors", None)
    if cost_trace is not None:
        metrics.update(trace_metrics("moscot_cost", cost_trace))
    if error_trace is not None:
        metrics.update(trace_metrics("moscot_error", error_trace))
    return metrics


def run_moscot(context: dict, cfg: dict) -> tuple[list, np.ndarray | None, None, dict]:
    """
    Cross-modality alignment using MOSCOT's TranslationProblem (Gromov-Wasserstein).

    Pure GW (alpha=1, joint_attr=None): aligns RNA and protein via geometry only,
    no shared features required.

    Returns
    -------
    aligned  : [rna_in_protein_space (n, D_p), protein_obs (n, D_p)]
    coupling : (n, n) transport plan
    """
    rna_adata = context["rna_adata"]
    second_adata = context["second_adata"]
    rna_repr_key = str(cfg.get("rna_repr", "X_pca"))
    second_repr_key = str(cfg.get("second_repr", "X_pca"))

    epsilon = float(cfg.get("epsilon", 0.1))
    tau_a = float(cfg.get("tau_a", 1.0))
    tau_b = float(cfg.get("tau_b", 1.0))
    max_iter = int(cfg.get("max_iterations", 100))

    if bool(cfg.get("moscot_validate_inputs", True)):
        assert_context_matches_moscot_inputs(
            context, rna_adata, second_adata, rna_repr_key, second_repr_key
        )

    tp = TranslationProblem(rna_adata, second_adata)
    tp = tp.prepare(
        src_attr=rna_repr_key,
        tgt_attr=second_repr_key,
        joint_attr=None,   # pure GW: no shared features between RNA and protein
    )
    tp = tp.solve(
        alpha=1.0,         # pure GW (joint_attr=None makes alpha irrelevant)
        epsilon=epsilon,
        tau_a=tau_a,
        tau_b=tau_b,
        max_iterations=max_iter,
    )

    # Barycentric projection: RNA cells mapped into protein PCA space
    rna_translated = np.array(tp.translate("src", "tgt", forward=True))
    protein_obs = np.array(second_adata.obsm[second_repr_key])

    solution = tp.solutions[("src", "tgt")]
    coupling = np.array(solution.transport_matrix)
    diagnostics = solution_metrics(solution, coupling, cfg)

    output_dir = context.get("output_dir")
    cost_trace = getattr(solution, "_costs", None)
    error_trace = getattr(solution, "_errors", None)
    if cost_trace is not None:
        save_trace(output_dir, "moscot_cost_trace.npy", cost_trace)
    if error_trace is not None:
        save_trace(output_dir, "moscot_error_trace.npy", error_trace)

    return [rna_translated, protein_obs], coupling, None, diagnostics
