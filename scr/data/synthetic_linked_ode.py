from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


DEFAULT_LINKED_ODE_SEED = 42

# Staged synthetic, per the v2 design:
#   oracle (v2a-oracle): trunk only, no noise, native counts -> implementation sanity gate.
#   clean  (v2a-clean) : trunk only, Poisson + small noise   -> observation-noise robustness.
#   branch (v2b-branch): mirror-symmetric bifurcation        -> OT-degenerate branch swap that
#                        only the dynamics term can disambiguate (the proof-of-mechanism stage).
STAGE_SETTINGS = {
    "oracle": {
        "branch": False,
        "noise": 0.0,
        "dynamic_alpha": False,
        "dynamic_kappa": False,
    },
    "clean": {
        "branch": False,
        "noise": 0.05,            # moderate observation noise
        "dynamic_alpha": True,    # time-varying α → model must learn it (non-trivial ODE)
        "dynamic_kappa": True,    # time-varying κ → model must learn it
    },
    "branch": {
        "branch": True,
        "noise": 0.005,
        "dynamic_alpha": False,   # constant α/κ keeps the A<->B protein mirror exact
        "dynamic_kappa": False,   # (linearity-preserved swap degeneracy → OT genuinely ambiguous)
    },
}

STAGE_LABELS = {
    "oracle": "v2a-oracle",
    "clean": "v2a-clean",
    "branch": "v2b-branch",
}


def stage_output_paths(stage: str) -> tuple[str, str]:
    """Stage-specific h5ad paths so the three stages coexist instead of overwriting."""
    rna = f"cache/velocity/synthetic_linked_ode/{stage}/rna.h5ad"
    protein = f"cache/synthetic_linked_ode/{stage}/protein.h5ad"
    return rna, protein


def linked_ode_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate native-coordinate RNA/protein data satisfying the linked KOT ODE."
    )
    parser.add_argument("--stage", choices=sorted(STAGE_SETTINGS), default="oracle")
    parser.add_argument("--seed", type=int, default=DEFAULT_LINKED_ODE_SEED)
    parser.add_argument("--n-cells", type=int, default=5000)
    parser.add_argument("--n-rna-genes", type=int, default=12)
    parser.add_argument("--n-proteins", type=int, default=10)
    parser.add_argument("--t-max", type=float, default=10.0)
    parser.add_argument("--t-bif", type=float, default=5.0)
    parser.add_argument("--rna-count-scale", type=float, default=80.0)
    parser.add_argument("--protein-count-scale", type=float, default=40.0)
    parser.add_argument(
        "--out-rna",
        default=None,
        help="Defaults to a stage-specific path under cache/velocity/synthetic_linked_ode/<stage>/.",
    )
    parser.add_argument(
        "--out-protein",
        default=None,
        help="Defaults to a stage-specific path under cache/synthetic_linked_ode/<stage>/.",
    )
    return parser


def positive_counts(values: np.ndarray, scale: float, rng: np.random.Generator) -> np.ndarray:
    rates = np.clip(values * scale, 0.0, None)
    return rng.poisson(rates).astype("float32")


def log1p_scaled(values: np.ndarray, scale: float) -> np.ndarray:
    return np.log1p(np.clip(values, 0.0, None) * scale)


def log1p_scaled_velocity(values: np.ndarray, velocity: np.ndarray, scale: float) -> np.ndarray:
    denominator = 1.0 + np.clip(values, 0.0, None) * scale
    return scale * velocity / np.maximum(denominator, 1e-12)


def noisy_positive(values: np.ndarray, noise: float, rng: np.random.Generator) -> np.ndarray:
    if noise <= 0:
        return np.asarray(values, dtype=np.float32)
    perturbed = values + rng.normal(0.0, noise, size=values.shape)
    return np.clip(perturbed, 0.0, None).astype("float32")


def linked_feature_names(n_rna_genes: int, n_proteins: int) -> tuple[list[str], list[str]]:
    rna_names = [f"Linked_RNA_{i:02d}" for i in range(n_rna_genes)]
    protein_names = [f"Linked_Protein_{i:02d}" for i in range(n_proteins)]
    return rna_names, protein_names


def linked_feature_links(rna_names: list[str], protein_names: list[str]) -> dict[str, list]:
    n_links = min(len(rna_names), len(protein_names))
    return {
        "rna": rna_names[:n_links],
        "protein": protein_names[:n_links],
        "weight": [1.0] * n_links,
        "sign": [1] * n_links,
    }


def make_time_branch(
    n_cells: int,
    t_max: float,
    t_bif: float,
    branch: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not branch:
        time = np.linspace(0.0, t_max, n_cells)
        branch_sign = np.zeros(n_cells, dtype=np.float64)
        branch_label = np.array(["Trunk"] * n_cells, dtype=object)
        return time, branch_sign, branch_label

    n_progenitor = n_cells // 3
    n_branch_a = (n_cells - n_progenitor) // 2
    n_branch_b = n_cells - n_progenitor - n_branch_a

    time_progenitor = np.linspace(0.0, t_bif, n_progenitor)
    time_a = np.linspace(t_bif, t_max, n_branch_a)
    time_b = np.linspace(t_bif, t_max, n_branch_b)

    time = np.concatenate([time_progenitor, time_a, time_b])
    branch_sign = np.concatenate([
        np.zeros(n_progenitor),
        np.ones(n_branch_a),
        -np.ones(n_branch_b),
    ]).astype(np.float64)
    branch_label = np.array(
        ["Progenitor"] * n_progenitor
        + ["Branch_A"] * n_branch_a
        + ["Branch_B"] * n_branch_b,
        dtype=object,
    )
    return time, branch_sign, branch_label


def smooth_positive_features(
    time: np.ndarray,
    branch_sign: np.ndarray,
    n_features: int,
    t_max: float,
    t_bif: float,
    rng: np.random.Generator,
    branch_enabled: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth positive features with a *native-space mirror-symmetric* branch split.

    Both branches share a continuation s_cont(t); post-bifurcation they split as
        Branch_A:  s_cont(t) * (1 + c_j * rho(t))
        Branch_B:  s_cont(t) * (1 - c_j * rho(t))
    with branch_sign in {0, +1, -1} and rho the post-bifurcation ramp. Because the
    split is additive about s_cont in *native* space (not log space), s_A + s_B = 2 s_cont
    exactly. The protein ODE is linear in (s, p) with constant kappa/alpha/beta, so this
    symmetry is inherited exactly by the protein paths (p_A + p_B = 2 p_cont). The pooled
    static marginal is therefore swap-invariant (A<->B) and pure OT is 2-fold degenerate;
    only the velocity sign (d(c_j*rho)/dt), carried RNA->protein through the ODE, breaks
    the tie. |c_j| < 1 keeps both branches strictly positive.
    """
    time_scaled = time / t_max
    post_bif = np.clip((time - t_bif) / max(t_max - t_bif, 1e-12), 0.0, None)
    d_post_bif = np.where(time >= t_bif, 1.0 / max(t_max - t_bif, 1e-12), 0.0)

    base = rng.uniform(-0.15, 0.25, size=n_features)
    slope = rng.uniform(0.05, 0.35, size=n_features)
    amplitude = rng.uniform(0.03, 0.15, size=n_features)
    frequency = rng.uniform(0.8, 1.8, size=n_features)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=n_features)
    branch_coeff = (
        rng.uniform(-0.6, 0.6, size=n_features) if branch_enabled else np.zeros(n_features)
    )

    angle = 2.0 * np.pi * frequency[None, :] * time_scaled[:, None] + phase[None, :]
    log_cont = (
        base[None, :]
        + slope[None, :] * time_scaled[:, None]
        + amplitude[None, :] * np.sin(angle)
    )
    cont = np.exp(log_cont)                                  # shared continuation s_cont(t)
    ramp = branch_sign[:, None] * branch_coeff[None, :] * post_bif[:, None]
    values = cont * (1.0 + ramp)

    dlog_cont_dt = (
        slope[None, :] / t_max
        + amplitude[None, :]
        * np.cos(angle)
        * (2.0 * np.pi * frequency[None, :] / t_max)
    )
    dcont_dt = cont * dlog_cont_dt
    dramp_dt = branch_sign[:, None] * branch_coeff[None, :] * d_post_bif[:, None]
    derivative = dcont_dt * (1.0 + ramp) + cont * dramp_dt
    return values.astype("float64"), derivative.astype("float64")


def alpha_values(
    time: np.ndarray,
    branch_sign: np.ndarray,
    n_proteins: int,
    t_max: float,
    t_bif: float,
    rng: np.random.Generator,
    dynamic_alpha: bool,
    branch_enabled: bool,
) -> np.ndarray:
    base = np.ones(n_proteins, dtype=np.float64)
    if not dynamic_alpha:
        return np.broadcast_to(base[None, :], (time.shape[0], n_proteins)).copy()

    time_scaled = time / t_max
    post_bif = np.clip((time - t_bif) / max(t_max - t_bif, 1e-12), 0.0, None)
    amplitude = rng.uniform(0.02, 0.08, size=n_proteins)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=n_proteins)
    branch_loading = (
        rng.uniform(-0.12, 0.12, size=n_proteins) if branch_enabled else np.zeros(n_proteins)
    )
    log_alpha = (
        np.log(base)[None, :]
        + amplitude[None, :] * np.sin(2.0 * np.pi * time_scaled[:, None] + phase[None, :])
        + branch_sign[:, None] * branch_loading[None, :] * post_bif[:, None]
    )
    return np.exp(log_alpha)


def kappa_values(
    time: np.ndarray,
    branch_sign: np.ndarray,
    t_max: float,
    t_bif: float,
    dynamic_kappa: bool,
    branch_enabled: bool,
) -> np.ndarray:
    base = 1.0
    if not dynamic_kappa:
        return np.full(time.shape[0], base, dtype=np.float64)

    time_scaled = time / t_max
    post_bif = np.clip((time - t_bif) / max(t_max - t_bif, 1e-12), 0.0, None)
    branch_term = 0.10 * branch_sign * post_bif if branch_enabled else 0.0
    log_kappa = np.log(base) + 0.08 * np.sin(2.0 * np.pi * time_scaled) + branch_term
    return np.exp(log_kappa)


def beta_values(n_proteins: int, rng: np.random.Generator) -> np.ndarray:
    return np.full(n_proteins, 0.5, dtype=np.float64)


def protein_rhs(
    protein: np.ndarray,
    linked_rna: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    kappa: np.ndarray,
) -> np.ndarray:
    return kappa[:, None] * (alpha * linked_rna - beta[None, :] * protein)


def linked_ode_debug_summary(
    rna_adata: ad.AnnData,
    protein_adata: ad.AnnData,
) -> dict:
    linked_rna = rna_adata.layers["spliced_mean"][:, :protein_adata.n_vars]
    protein = rna_adata.obsm["protein_mean"]
    alpha = rna_adata.obsm["true_alpha"]
    beta = rna_adata.obsm["true_beta"]
    kappa = rna_adata.obsm["true_kappa"]
    rhs = kappa * (alpha * linked_rna - beta * protein)
    true_velocity = rna_adata.obsm["true_protein_velocity_mean"]
    metadata = dict(rna_adata.uns["synthetic_linked_ode"])
    return {
        "stage": metadata["stage"],
        "seed": int(metadata["seed"]),
        "n_cells": int(rna_adata.n_obs),
        "n_rna_genes": int(rna_adata.n_vars),
        "n_proteins": int(protein_adata.n_vars),
        "n_distractor_rna_genes": int(rna_adata.n_vars - protein_adata.n_vars),
        "native_rna_layer": metadata["native_rna_layer"],
        "native_protein_layer": metadata["native_protein_layer"],
        "native_velocity_layer": metadata["native_velocity_layer"],
        "transformed_rna_layer": metadata["transformed_rna_layer"],
        "transformed_protein_layer": metadata["transformed_protein_layer"],
        "transformed_velocity_layer": metadata["transformed_velocity_layer"],
        "alpha_min": float(alpha.min()),
        "alpha_max": float(alpha.max()),
        "beta_min": float(beta.min()),
        "beta_max": float(beta.max()),
        "kappa_min": float(kappa.min()),
        "kappa_max": float(kappa.max()),
        "protein_rhs_max_abs_err": float(np.max(np.abs(rhs - true_velocity))),
        "rna_protein_link_count": len(rna_adata.uns["rna_protein_links"]["rna"]),
    }


def print_linked_ode_debug_summary(summary: dict) -> None:
    print("---- synthetic_linked_ode debug ----")
    print(f"  stage/seed:        {summary['stage']} / {summary['seed']}")
    print(f"  cells:             {summary['n_cells']}")
    print(
        "  features:          "
        f"RNA={summary['n_rna_genes']}  protein={summary['n_proteins']}  "
        f"distractor_RNA={summary['n_distractor_rna_genes']}"
    )
    print(
        "  native layers:     "
        f"rna={summary['native_rna_layer']}  "
        f"protein={summary['native_protein_layer']}  "
        f"velocity={summary['native_velocity_layer']}"
    )
    print(
        "  transformed:       "
        f"rna={summary['transformed_rna_layer']}  "
        f"protein={summary['transformed_protein_layer']}  "
        f"velocity={summary['transformed_velocity_layer']}"
    )
    print(f"  alpha min/max:     {summary['alpha_min']:.6g} / {summary['alpha_max']:.6g}")
    print(f"  beta min/max:      {summary['beta_min']:.6g} / {summary['beta_max']:.6g}")
    print(f"  kappa min/max:     {summary['kappa_min']:.6g} / {summary['kappa_max']:.6g}")
    print(f"  linked features:   {summary['rna_protein_link_count']}")
    print(f"  ODE RHS max error: {summary['protein_rhs_max_abs_err']:.6g}")


def interpolate_path(
    query_time: float,
    path_time: np.ndarray,
    path_values: np.ndarray,
) -> np.ndarray:
    return np.array([
        np.interp(query_time, path_time, path_values[:, j])
        for j in range(path_values.shape[1])
    ])


def protein_rhs_at(
    query_time: float,
    y_value: np.ndarray,
    time: np.ndarray,
    linked_rna: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    kappa: np.ndarray,
) -> np.ndarray:
    s_value = interpolate_path(query_time, time, linked_rna)
    alpha_value = interpolate_path(query_time, time, alpha)
    kappa_value = np.interp(query_time, time, kappa)
    return kappa_value * (alpha_value * s_value - beta * y_value)


def solve_protein_path(
    time: np.ndarray,
    linked_rna: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    kappa: np.ndarray,
    initial_protein: np.ndarray,
) -> np.ndarray:
    protein = np.zeros((time.shape[0], beta.shape[0]), dtype=np.float64)
    protein[0] = initial_protein
    for i in range(time.shape[0] - 1):
        t0 = time[i]
        t1 = time[i + 1]
        h = t1 - t0
        y0 = protein[i]

        k1 = protein_rhs_at(t0, y0, time, linked_rna, alpha, beta, kappa)
        k2 = protein_rhs_at(
            t0 + 0.5 * h,
            y0 + 0.5 * h * k1,
            time,
            linked_rna,
            alpha,
            beta,
            kappa,
        )
        k3 = protein_rhs_at(
            t0 + 0.5 * h,
            y0 + 0.5 * h * k2,
            time,
            linked_rna,
            alpha,
            beta,
            kappa,
        )
        k4 = protein_rhs_at(t1, y0 + h * k3, time, linked_rna, alpha, beta, kappa)
        protein[i + 1] = np.clip(y0 + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0, 0.0, None)
    return protein


def solve_protein(
    time: np.ndarray,
    branch_label: np.ndarray,
    linked_rna: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    kappa: np.ndarray,
    t_bif: float,
    branch_enabled: bool,
) -> np.ndarray:
    if not branch_enabled:
        initial = alpha[0] * linked_rna[0] / beta
        return solve_protein_path(time, linked_rna, alpha, beta, kappa, initial)

    protein = np.zeros((time.shape[0], beta.shape[0]), dtype=np.float64)
    progenitor_idx = np.where(branch_label == "Progenitor")[0]
    progenitor_order = progenitor_idx[np.argsort(time[progenitor_idx])]
    initial = alpha[progenitor_order[0]] * linked_rna[progenitor_order[0]] / beta
    protein[progenitor_order] = solve_protein_path(
        time[progenitor_order],
        linked_rna[progenitor_order],
        alpha[progenitor_order],
        beta,
        kappa[progenitor_order],
        initial,
    )
    branch_initial = protein[progenitor_order[-1]]

    for label in ["Branch_A", "Branch_B"]:
        branch_idx = np.where(branch_label == label)[0]
        order = branch_idx[np.argsort(time[branch_idx])]
        protein[order] = solve_protein_path(
            time[order],
            linked_rna[order],
            alpha[order],
            beta,
            kappa[order],
            branch_initial,
        )
        bif_mask = np.isclose(time[order], t_bif)
        if bif_mask.any():
            protein[order[bif_mask]] = branch_initial
    return protein


def make_umap(time: np.ndarray, branch_sign: np.ndarray, t_max: float, t_bif: float) -> np.ndarray:
    x = 5.0 * time / t_max
    y = np.zeros_like(x)
    post_bif = np.clip((time - t_bif) / max(t_max - t_bif, 1e-12), 0.0, None)
    y += branch_sign * 2.0 * post_bif
    return np.vstack([x, y]).T.astype("float32")


def make_rna_adata(
    rna_native: np.ndarray,
    rna_counts: np.ndarray,
    rna_velocity: np.ndarray,
    protein_native: np.ndarray,
    protein_velocity: np.ndarray,
    rna_log1p: np.ndarray,
    rna_log1p_velocity: np.ndarray,
    protein_log1p: np.ndarray,
    protein_log1p_velocity: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    kappa: np.ndarray,
    time: np.ndarray,
    branch_label: np.ndarray,
    rna_names: list[str],
    protein_names: list[str],
    links: dict[str, list],
    stage: str,
    seed: int,
    t_max: float,
    t_bif: float,
) -> ad.AnnData:
    cell_ids = [f"linked_cell_{i:05d}" for i in range(rna_native.shape[0])]
    adata = ad.AnnData(X=rna_counts.astype("float32"))
    adata.obs_names = cell_ids
    adata.var_names = rna_names
    adata.var["feature_type"] = "rna"
    adata.var["feature_types"] = "GEX"
    adata.layers["spliced"] = rna_counts.astype("float32")
    adata.layers["spliced_mean"] = rna_native.astype("float32")
    adata.layers["true_velocity_mean"] = rna_velocity.astype("float32")
    adata.layers["spliced_log1p"] = rna_log1p.astype("float32")
    adata.layers["true_velocity_log1p"] = rna_log1p_velocity.astype("float32")
    adata.obsm["protein_mean"] = protein_native.astype("float32")
    adata.obsm["protein_log1p"] = protein_log1p.astype("float32")
    adata.obsm["true_protein_velocity_mean"] = protein_velocity.astype("float32")
    adata.obsm["true_protein_velocity_log1p"] = protein_log1p_velocity.astype("float32")
    adata.obsm["true_alpha"] = alpha.astype("float32")
    adata.obsm["true_beta"] = np.broadcast_to(beta[None, :], alpha.shape).astype("float32")
    adata.obsm["true_kappa"] = kappa[:, None].astype("float32")
    adata.obsm["true_pairing"] = np.arange(rna_native.shape[0], dtype=np.int64)[:, None]
    adata.obsm["X_umap"] = make_umap(time, branch_sign_from_labels(branch_label), t_max, t_bif)
    adata.obs = pd.DataFrame({
        "true_time": time,
        "time": time,
        "true_branch": branch_label,
        "state": branch_label,
        "true_pairing": np.arange(rna_native.shape[0], dtype=int),
        "true_kappa": kappa,
    }, index=cell_ids)
    adata.uns["protein_names"] = protein_names
    adata.uns["rna_protein_links"] = links
    adata.uns["rna_protein_feature_links"] = links
    adata.uns["anchor_indices"] = list(range(len(protein_names)))
    adata.uns["anchor_betas"] = beta.astype(float).tolist()
    adata.uns["synthetic_linked_ode"] = {
        "stage": stage,
        "stage_label": STAGE_LABELS.get(stage, stage),
        "seed": seed,
        "equation": "dp_j/dt = kappa(t,b) * (alpha_j(t,b) * s_j(t,b) - beta_j * p_j(t,b))",
        "branch_construction": (
            "native mirror: s_{A,B} = s_cont * (1 +/- c_j * rho); linear ODE => "
            "p_{A,B} = p_cont +/- f, swap-invariant static marginal (OT-degenerate), "
            "velocity sign disambiguates"
        ),
        "n_rna_genes": len(rna_names),
        "n_proteins": len(protein_names),
        "n_distractor_rna_genes": max(len(rna_names) - len(protein_names), 0),
        "alpha_is_constant": bool(np.allclose(alpha, alpha[0])),
        "beta_is_known": True,
        "kappa_is_constant": bool(np.allclose(kappa, kappa[0])),
        "native_rna_layer": "spliced_mean",
        "native_protein_layer": "protein_mean",
        "native_velocity_layer": "true_velocity_mean",
        "native_protein_velocity_layer": "true_protein_velocity_mean",
        "transformed_rna_layer": "spliced_log1p",
        "transformed_protein_layer": "protein_log1p",
        "transformed_velocity_layer": "true_velocity_log1p",
        "transformed_protein_velocity_layer": "true_protein_velocity_log1p",
    }
    return adata


def branch_sign_from_labels(branch_label: np.ndarray) -> np.ndarray:
    branch_sign = np.zeros(branch_label.shape[0], dtype=np.float64)
    branch_sign[branch_label == "Branch_A"] = 1.0
    branch_sign[branch_label == "Branch_B"] = -1.0
    return branch_sign


def make_protein_adata(
    rna_adata: ad.AnnData,
    protein_native: np.ndarray,
    protein_counts: np.ndarray,
    protein_velocity: np.ndarray,
    protein_log1p: np.ndarray,
    protein_log1p_velocity: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    kappa: np.ndarray,
    protein_names: list[str],
    links: dict[str, list],
) -> ad.AnnData:
    protein_adata = ad.AnnData(X=protein_counts.astype("float32"))
    protein_adata.obs_names = rna_adata.obs_names.copy()
    protein_adata.var_names = protein_names
    protein_adata.var["feature_types"] = "Protein"
    protein_adata.obs = rna_adata.obs.copy()
    protein_adata.layers["protein"] = protein_counts.astype("float32")
    protein_adata.layers["protein_mean"] = protein_native.astype("float32")
    protein_adata.layers["true_protein_velocity_mean"] = protein_velocity.astype("float32")
    protein_adata.layers["protein_log1p"] = protein_log1p.astype("float32")
    protein_adata.layers["true_protein_velocity_log1p"] = protein_log1p_velocity.astype("float32")
    protein_adata.layers["true_alpha"] = alpha.astype("float32")
    protein_adata.layers["true_beta"] = np.broadcast_to(beta[None, :], alpha.shape).astype("float32")
    protein_adata.layers["true_kappa"] = np.broadcast_to(kappa[:, None], alpha.shape).astype("float32")
    protein_adata.uns["rna_protein_links"] = links
    protein_adata.uns["rna_protein_feature_links"] = links
    protein_adata.uns["anchor_indices"] = list(range(len(protein_names)))
    protein_adata.uns["anchor_betas"] = beta.astype(float).tolist()
    protein_adata.uns["synthetic_linked_ode"] = rna_adata.uns["synthetic_linked_ode"]
    return protein_adata


def generate_linked_ode_data(
    stage: str = "oracle",
    seed: int = DEFAULT_LINKED_ODE_SEED,
    n_cells: int = 5000,
    n_rna_genes: int = 12,
    n_proteins: int = 10,
    t_max: float = 10.0,
    t_bif: float = 5.0,
    rna_count_scale: float = 80.0,
    protein_count_scale: float = 40.0,
) -> tuple[ad.AnnData, ad.AnnData]:
    settings = STAGE_SETTINGS[stage]
    rng = np.random.default_rng(seed)
    time, branch_sign, branch_label = make_time_branch(
        n_cells, t_max, t_bif, bool(settings["branch"]),
    )
    rna_names, protein_names = linked_feature_names(n_rna_genes, n_proteins)
    links = linked_feature_links(rna_names, protein_names)

    rna_native, rna_velocity = smooth_positive_features(
        time,
        branch_sign,
        n_rna_genes,
        t_max,
        t_bif,
        rng,
        bool(settings["branch"]),
    )
    linked_rna = rna_native[:, :n_proteins]
    alpha = alpha_values(
        time,
        branch_sign,
        n_proteins,
        t_max,
        t_bif,
        rng,
        bool(settings["dynamic_alpha"]),
        bool(settings["branch"]),
    )
    kappa = kappa_values(
        time,
        branch_sign,
        t_max,
        t_bif,
        bool(settings["dynamic_kappa"]),
        bool(settings["branch"]),
    )
    beta = beta_values(n_proteins, rng)
    protein_native = solve_protein(
        time,
        branch_label,
        linked_rna,
        alpha,
        beta,
        kappa,
        t_bif,
        bool(settings["branch"]),
    )
    protein_velocity = protein_rhs(protein_native, linked_rna, alpha, beta, kappa)
    rna_log1p = log1p_scaled(rna_native, rna_count_scale)
    protein_log1p = log1p_scaled(protein_native, protein_count_scale)
    rna_log1p_velocity = log1p_scaled_velocity(rna_native, rna_velocity, rna_count_scale)
    protein_log1p_velocity = log1p_scaled_velocity(
        protein_native,
        protein_velocity,
        protein_count_scale,
    )

    noise = float(settings["noise"])
    rna_noisy = noisy_positive(rna_native, noise, rng)
    protein_noisy = noisy_positive(protein_native, noise, rng)
    rna_counts = positive_counts(rna_noisy, rna_count_scale, rng)
    protein_counts = positive_counts(protein_noisy, protein_count_scale, rng)
    if stage == "oracle":
        rna_counts = rna_native.astype("float32")
        protein_counts = protein_native.astype("float32")

    rna_adata = make_rna_adata(
        rna_native,
        rna_counts,
        rna_velocity,
        protein_native,
        protein_velocity,
        rna_log1p,
        rna_log1p_velocity,
        protein_log1p,
        protein_log1p_velocity,
        alpha,
        beta,
        kappa,
        time,
        branch_label,
        rna_names,
        protein_names,
        links,
        stage,
        seed,
        t_max,
        t_bif,
    )
    protein_adata = make_protein_adata(
        rna_adata,
        protein_native,
        protein_counts,
        protein_velocity,
        protein_log1p,
        protein_log1p_velocity,
        alpha,
        beta,
        kappa,
        protein_names,
        links,
    )
    return rna_adata, protein_adata


def save_linked_ode(
    out_rna: str | Path = "cache/velocity/synthetic_linked_ode/synthetic_linked_ode.h5ad",
    out_protein: str | Path = "cache/synthetic_linked_ode/protein.h5ad",
    stage: str = "oracle",
    seed: int = DEFAULT_LINKED_ODE_SEED,
    n_cells: int = 5000,
    n_rna_genes: int = 12,
    n_proteins: int = 10,
    t_max: float = 10.0,
    t_bif: float = 5.0,
    rna_count_scale: float = 80.0,
    protein_count_scale: float = 40.0,
) -> tuple[ad.AnnData, ad.AnnData]:
    rna_adata, protein_adata = generate_linked_ode_data(
        stage=stage,
        seed=seed,
        n_cells=n_cells,
        n_rna_genes=n_rna_genes,
        n_proteins=n_proteins,
        t_max=t_max,
        t_bif=t_bif,
        rna_count_scale=rna_count_scale,
        protein_count_scale=protein_count_scale,
    )

    out_rna = Path(out_rna)
    out_protein = Path(out_protein)
    out_rna.parent.mkdir(parents=True, exist_ok=True)
    out_protein.parent.mkdir(parents=True, exist_ok=True)
    rna_adata.write_h5ad(out_rna)
    protein_adata.write_h5ad(out_protein)
    summary = linked_ode_debug_summary(rna_adata, protein_adata)
    print(f"Saved linked ODE RNA:     {out_rna}")
    print(f"Saved linked ODE protein: {out_protein}")
    print(f"stage={stage} seed={seed}")
    print_linked_ode_debug_summary(summary)
    return rna_adata, protein_adata


def main() -> None:
    args = linked_ode_parser().parse_args()
    default_rna, default_protein = stage_output_paths(args.stage)
    save_linked_ode(
        out_rna=args.out_rna or default_rna,
        out_protein=args.out_protein or default_protein,
        stage=args.stage,
        seed=args.seed,
        n_cells=args.n_cells,
        n_rna_genes=args.n_rna_genes,
        n_proteins=args.n_proteins,
        t_max=args.t_max,
        t_bif=args.t_bif,
        rna_count_scale=args.rna_count_scale,
        protein_count_scale=args.protein_count_scale,
    )


if __name__ == "__main__":
    main()
