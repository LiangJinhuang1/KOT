#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scr.data.projection import projection_matrix_from_adatas
from scr.data.synthetic import PARAMS, build_protein_adata, generate_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit whether the old synthetic matches the native feature-wise KOT ODE."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-cells", type=int, default=5000)
    parser.add_argument("--noise", type=float, default=0.03)
    parser.add_argument("--n-rna-genes", type=int, default=30)
    parser.add_argument("--n-proteins", type=int, default=10)
    return parser.parse_args()


def dense_matrix(value) -> np.ndarray:
    if hasattr(value, "toarray"):
        value = value.toarray()
    return np.asarray(value, dtype=np.float64)


def derivative_by_state(values: np.ndarray, time: np.ndarray, state: np.ndarray) -> np.ndarray:
    derivative = np.zeros_like(values, dtype=np.float64)
    for label in sorted(set(state.tolist())):
        idx = np.where(state == label)[0]
        order = idx[np.argsort(time[idx])]
        edge_order = 2 if len(order) >= 3 else 1
        derivative[order] = np.gradient(values[order], time[order], axis=0, edge_order=edge_order)
    return derivative


def delayed_spliced_values(time: np.ndarray, state: np.ndarray, spliced: np.ndarray) -> np.ndarray:
    delayed = np.zeros_like(spliced, dtype=np.float64)
    progenitor_idx = np.where(state == "Progenitor")[0]
    progenitor_order = progenitor_idx[np.argsort(time[progenitor_idx])]
    progenitor_time = time[progenitor_order]
    progenitor_spliced = spliced[progenitor_order]

    for label in sorted(set(state.tolist())):
        idx = np.where(state == label)[0]
        order = idx[np.argsort(time[idx])]
        if label == "Progenitor":
            path_time = progenitor_time
            path_spliced = progenitor_spliced
        else:
            path_time = np.concatenate([progenitor_time[:-1], time[order]])
            path_spliced = np.concatenate([progenitor_spliced[:-1], spliced[order]])
        delayed[order] = np.interp(time[order] - PARAMS["TAU"], path_time, path_spliced)
    return delayed


def vector_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    residual = target - prediction
    target_flat = target.reshape(-1)
    prediction_flat = prediction.reshape(-1)
    residual_flat = residual.reshape(-1)
    target_norm = np.linalg.norm(target_flat)
    prediction_norm = np.linalg.norm(prediction_flat)
    cosine = float(
        np.dot(target_flat, prediction_flat)
        / max(target_norm * prediction_norm, 1e-12)
    )
    return {
        "mae": float(np.mean(np.abs(residual_flat))),
        "rmse": float(np.sqrt(np.mean(residual_flat ** 2))),
        "relative_l2": float(np.linalg.norm(residual_flat) / max(target_norm, 1e-12)),
        "cosine": cosine,
        "target_norm_mean": float(np.linalg.norm(target, axis=1).mean())
        if target.ndim == 2 else float(np.mean(np.abs(target))),
        "prediction_norm_mean": float(np.linalg.norm(prediction, axis=1).mean())
        if prediction.ndim == 2 else float(np.mean(np.abs(prediction))),
    }


def print_metrics(name: str, metrics: dict[str, float]) -> None:
    print(f"\n{name}")
    print("-" * len(name))
    for key in ["relative_l2", "cosine", "mae", "rmse", "target_norm_mean", "prediction_norm_mean"]:
        print(f"{key:>22s}: {metrics[key]:.6g}")


def build_protein_from_rna(adata, seed: int):
    protein_df = adata.obsm["protein_expression"]
    return build_protein_adata(
        adata,
        list(protein_df.index),
        protein_df.values,
        adata.obsm["protein_mean"].values,
        list(protein_df.columns),
        seed,
    )


def main() -> None:
    args = parse_args()
    adata = generate_data(
        n_cells=args.n_cells,
        noise=args.noise,
        seed=args.seed,
        n_rna_genes=args.n_rna_genes,
        n_proteins=args.n_proteins,
    )
    protein_adata = build_protein_from_rna(adata, args.seed)

    time = np.asarray(adata.obs["time"].values, dtype=np.float64)
    state = np.asarray(adata.obs["state"].values)
    scalar_spliced = np.asarray(adata.obs["spliced_rna"].values, dtype=np.float64)
    scalar_protein = np.asarray(adata.obs["protein"].values, dtype=np.float64)

    spliced_mean = dense_matrix(adata.layers["spliced_mean"])
    true_velocity_mean = dense_matrix(adata.layers["true_velocity_mean"])
    protein_mean = dense_matrix(protein_adata.layers["protein_mean"])

    protein_derivative_fd = derivative_by_state(protein_mean, time, state)
    spliced_derivative_fd = derivative_by_state(spliced_mean, time, state)
    scalar_protein_derivative_fd = derivative_by_state(
        scalar_protein.reshape(-1, 1), time, state,
    ).reshape(-1)

    delayed_spliced = delayed_spliced_values(time, state, scalar_spliced)
    scalar_delayed_rhs = PARAMS["K_S"] * delayed_spliced - PARAMS["GAMMA_P"] * scalar_protein

    S = projection_matrix_from_adatas(adata, protein_adata, use_explicit_links=True)
    Sr = (S @ spliced_mean.T).T
    feature_kot_rhs = PARAMS["K_S"] * Sr - PARAMS["GAMMA_P"] * protein_mean

    scalar_current_rhs = PARAMS["K_S"] * scalar_spliced - PARAMS["GAMMA_P"] * scalar_protein
    scalar_delay_gap = scalar_current_rhs - scalar_delayed_rhs

    print("Old synthetic ODE audit")
    print(f"seed={args.seed} n_cells={args.n_cells} noise={args.noise}")
    print(f"TAU={PARAMS['TAU']} K_S={PARAMS['K_S']} GAMMA_P={PARAMS['GAMMA_P']}")
    print(f"RNA shape={spliced_mean.shape} protein shape={protein_mean.shape}")

    print_metrics(
        "RNA velocity layer vs finite-difference d(spliced_mean)/dt",
        vector_metrics(spliced_derivative_fd, true_velocity_mean),
    )
    print_metrics(
        "Simulator scalar delayed protein ODE vs finite-difference dp/dt",
        vector_metrics(scalar_protein_derivative_fd, scalar_delayed_rhs),
    )
    print_metrics(
        "Scalar current-source no-delay ODE vs finite-difference dp/dt",
        vector_metrics(scalar_protein_derivative_fd, scalar_current_rhs),
    )
    print_metrics(
        "Feature KOT no-delay linked ODE vs finite-difference d(protein_mean)/dt",
        vector_metrics(protein_derivative_fd, feature_kot_rhs),
    )

    print("\nDelay/source mismatch")
    print("---------------------")
    print(f"mean |current_rhs - delayed_rhs|: {np.mean(np.abs(scalar_delay_gap)):.6g}")
    print(f"rmse current_rhs - delayed_rhs:  {np.sqrt(np.mean(scalar_delay_gap ** 2)):.6g}")
    print(
        "mean |delayed_rhs|:             "
        f"{np.mean(np.abs(scalar_delayed_rhs)):.6g}"
    )


if __name__ == "__main__":
    main()
