#!/usr/bin/env python3
"""
Run the MOSCOT epsilon/max_iterations grid on the staged synthetic experiment.

Example:
    python tools/run_moscot_sweep.py --stage clean --scale mean
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import load_and_preprocess_cached
from src.evaluation.foscttm import evaluate_and_save
from src.training.moscot import run_moscot
from src.training.runner import (
    DATASETS_CONFIG,
    SCALE_LAYERS,
    STAGED_DATASET,
    build_context,
    get_repr,
    save_diagnostics,
    split_model_result,
)
from src.data.synthetic_linked_ode import stage_output_paths
from src.utils.io import load_yaml


TRAINING_CONFIG = Path("config/training.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MOSCOT 4 x 4 sweep.")
    parser.add_argument("--config", type=Path, default=TRAINING_CONFIG)
    parser.add_argument("--dataset", type=str, default=STAGED_DATASET)
    parser.add_argument("--stage", choices=["oracle", "clean", "branch"], default=None)
    parser.add_argument("--scale", choices=["mean", "log"], default="mean")
    parser.add_argument("--epsilons", nargs="+", type=float, default=None)
    parser.add_argument("--max-iterations", nargs="+", type=int, default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    return parser.parse_args()


def float_token(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def list_from_config(values, fallback, caster) -> list:
    source = fallback if values is None else values
    return [caster(value) for value in source]


def effective_dataset_and_config(args, train_cfg: dict, datasets: dict) -> tuple[dict, dict]:
    if args.dataset not in datasets:
        raise ValueError(f"Dataset '{args.dataset}' is not defined in {DATASETS_CONFIG}.")

    defaults = dict(train_cfg.get("defaults", {}) or {})
    stage_overrides = train_cfg.get("stage_overrides", {}) or {}
    dataset_cfg = dict(datasets[args.dataset])
    overrides = dict((train_cfg.get("datasets", {}) or {}).get(args.dataset, {}) or {})

    if args.dataset == STAGED_DATASET and args.stage is not None:
        rna_path, protein_path = stage_output_paths(args.stage)
        dataset_cfg = {**dataset_cfg, "rna_path": rna_path, "protein_path": protein_path}
        overrides.update(stage_overrides.get(args.stage, {}))
        overrides.update(SCALE_LAYERS[args.scale])
        overrides["preprocessing_cache_version"] = (
            f"synthetic_linked_ode_{args.stage}_{args.scale}"
        )

    run_cfg = {**defaults, **overrides, "model": "moscot"}
    return dataset_cfg, run_cfg


def load_context(name: str, dataset: dict, run_cfg: dict):
    rna_adata, protein_adata, atac_adata = load_and_preprocess_cached(
        rna_path=dataset["rna_path"],
        protein_path=dataset.get("protein_path"),
        protein_label=dataset.get("protein_label", "ADT"),
        protein_obsm_key=dataset.get("protein_obsm_key"),
        atac_path=dataset.get("atac_path"),
        atac_label=dataset.get("atac_label", "ATAC"),
        rna_raw_layer=dataset.get("rna_raw_layer"),
        rna_umap_path=dataset.get("rna_umap_path"),
        force_recompute=bool(run_cfg.get("force_preprocess", False)),
        cache_version=run_cfg.get("preprocessing_cache_version"),
        rna_min_cells=int(run_cfg.get("rna_min_cells", 3)),
        rna_n_top_genes=int(run_cfg.get("rna_n_top_genes", 2000)),
        rna_n_pcs=int(run_cfg.get("rna_n_pcs", 30)),
        rna_n_neighbors=int(run_cfg.get("rna_n_neighbors", 30)),
        add_log_velocity_layer=bool(run_cfg.get("add_log_velocity_layer", False)),
        log_velocity_scale=float(run_cfg.get("log_velocity_scale", 1.0)),
        protein_min_cells=int(run_cfg.get("protein_min_cells", 1)),
        protein_n_pcs=int(run_cfg.get("protein_n_pcs", 10)),
        atac_min_cells=int(run_cfg.get("atac_min_cells", 3)),
        atac_n_components=int(run_cfg.get("atac_n_components", 30)),
        atac_n_neighbors=int(run_cfg.get("atac_n_neighbors", 30)),
    )

    modality_pair = str(run_cfg.get("modality_pair", "rna_protein"))
    rna_repr_key = str(run_cfg.get("rna_repr", "X_pca"))
    second_repr_key = str(run_cfg.get("second_repr", "X_pca"))
    x = get_repr(rna_adata, rna_repr_key, "RNA")

    if modality_pair == "rna_protein":
        second_adata = protein_adata
        second_label = "protein"
        y = get_repr(second_adata, second_repr_key, "Protein")
    elif modality_pair == "rna_atac":
        second_adata = atac_adata
        second_label = "atac"
        y = get_repr(second_adata, second_repr_key, "ATAC")
    else:
        raise ValueError(f"Unknown modality_pair: '{modality_pair}'.")

    print(f"[moscot_sweep] dataset={name} modality_pair={modality_pair}")
    print(f"[moscot_sweep] RNA {rna_repr_key}: {x.shape}")
    print(f"[moscot_sweep] {second_label.upper()} {second_repr_key}: {y.shape}")

    context = build_context(x, y, rna_adata, second_adata, second_label, modality_pair)
    return context, rna_adata, second_label, modality_pair, rna_repr_key, second_repr_key


def write_summary(output_dir: Path, diagnostics: dict, run_cfg: dict) -> None:
    with open(output_dir / "summary.txt", "w") as f:
        f.write(f"dataset:        {diagnostics['dataset']}\n")
        f.write("model:          moscot\n")
        f.write("sweep:          epsilon x max_iterations\n")
        f.write(f"epsilon:        {diagnostics['moscot_epsilon']}\n")
        f.write(f"max_iterations: {diagnostics['moscot_max_iterations']}\n")
        f.write(f"modality_pair:  {run_cfg.get('modality_pair', 'rna_protein')}\n")
        f.write(f"rna_repr:       {run_cfg.get('rna_repr', 'X_pca')}\n")
        f.write(f"second_repr:    {run_cfg.get('second_repr', 'X_pca')}\n")
        f.write(f"mean_foscttm:   {diagnostics['mean_foscttm']:.6f}\n")
        f.write(f"runtime_sec:    {diagnostics['runtime_seconds']:.6f}\n")
        f.write(f"moscot_cost:    {diagnostics['moscot_cost']}\n")
        f.write(f"moscot_converged: {diagnostics['moscot_converged']}\n")
        f.write(f"coupling_entropy: {diagnostics['coupling_entropy']}\n")
        f.write(f"coupling_effective_sparsity: {diagnostics['coupling_effective_sparsity']}\n")


def run_grid(
    dataset_name: str,
    context: dict,
    rna_adata,
    second_label: str,
    run_cfg: dict,
    run_root: Path,
    epsilons: list[float],
    max_iterations_values: list[int],
) -> pd.DataFrame:
    rows = []

    for epsilon in epsilons:
        for max_iterations in max_iterations_values:
            label = f"epsilon_{float_token(epsilon)}_maxiter_{max_iterations}"
            output_dir = run_root / "moscot_sweep" / dataset_name / label
            output_dir.mkdir(parents=True, exist_ok=True)

            grid_cfg = {
                **run_cfg,
                "epsilon": float(epsilon),
                "max_iterations": int(max_iterations),
            }
            context["output_dir"] = output_dir

            print(
                f"[moscot_sweep] epsilon={epsilon:g} "
                f"max_iterations={max_iterations}"
            )
            start_time = time.perf_counter()
            result = run_moscot(context, grid_cfg)
            runtime_seconds = time.perf_counter() - start_time
            aligned, coupling, loss_df, diagnostics = split_model_result(result)

            result_name = f"{dataset_name}_{label}"
            mean_foscttm = evaluate_and_save(
                aligned,
                result_name,
                output_dir,
                second_label,
                model_name=None,
                obs_names=rna_adata.obs_names,
            )

            if coupling is not None:
                np.save(output_dir / "coupling.npy", coupling)
            if loss_df is not None:
                loss_df.to_csv(output_dir / "training_loss.csv", index=False)

            diagnostics = {
                "dataset": dataset_name,
                "model": "moscot",
                "sweep_label": label,
                "runtime_seconds": float(runtime_seconds),
                **diagnostics,
                "mean_foscttm": float(mean_foscttm),
            }
            save_diagnostics(output_dir, diagnostics)
            write_summary(output_dir, diagnostics, grid_cfg)
            rows.append(diagnostics)

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    train_cfg = load_yaml(args.config)
    datasets = load_yaml(DATASETS_CONFIG).get("datasets", {})
    dataset, run_cfg = effective_dataset_and_config(args, train_cfg, datasets)

    epsilons = list_from_config(
        args.epsilons,
        run_cfg.get("moscot_sweep_epsilons", [0.01, 0.05, 0.1, 0.5]),
        float,
    )
    max_iterations_values = list_from_config(
        args.max_iterations,
        run_cfg.get("moscot_sweep_max_iterations", [100, 200, 500, 1000]),
        int,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_root = Path(run_cfg.get("output_root", "cache/training"))
    run_root = (
        args.run_dir
        if args.run_dir is not None
        else base_root / f"run_{timestamp}_moscot_sweep"
    )
    run_root.mkdir(parents=True, exist_ok=True)
    run_cfg["output_root"] = str(run_root)
    shutil.copy(args.config, run_root / "training.yaml")

    print(f"[moscot_sweep] Output root: {run_root}")
    print(f"[moscot_sweep] epsilons: {epsilons}")
    print(f"[moscot_sweep] max_iterations: {max_iterations_values}")

    (
        context,
        rna_adata,
        second_label,
        modality_pair,
        rna_repr_key,
        second_repr_key,
    ) = load_context(args.dataset, dataset, run_cfg)
    run_cfg.update({
        "modality_pair": modality_pair,
        "rna_repr": rna_repr_key,
        "second_repr": second_repr_key,
    })

    results = run_grid(
        args.dataset,
        context,
        rna_adata,
        second_label,
        run_cfg,
        run_root,
        epsilons,
        max_iterations_values,
    )
    results.to_csv(run_root / "moscot_sweep_results.csv", index=False)
    print(f"[moscot_sweep] Saved aggregate table: {run_root / 'moscot_sweep_results.csv'}")


if __name__ == "__main__":
    main()
