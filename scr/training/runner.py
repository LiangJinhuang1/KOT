from __future__ import annotations

import numpy as np
from pathlib import Path

from scr.ultils.io import load_yaml
from scr.data.preprocessing import load_and_preprocess_cached
from scr.training.scot import run_scot
from scr.training.moscot import run_moscot
from scr.training.glue import run_glue
from scr.training.uniport import run_uniport
from scr.training.totalvi import run_totalvi
from scr.training.linear_ode import run_linear_ode
from scr.training.kot import run_kot
from scr.evaluation.foscttm import evaluate_and_save
from scr.visualization.alignment import plot_coembedding, plot_coupling, plot_foscttm_distribution, plot_foscttm_sorted, plot_training_loss
from scr.visualization.results import save_comparison_table

DATASETS_CONFIG = Path("config/datasets.yaml")
TRAINING_CONFIG = Path("config/training.yaml")

# Registry: model name → alignment function
# Each function accepts (context, cfg) and returns (aligned, coupling).
# context keys:
#   x            — RNA repr np.ndarray (n_cells × n_pcs)
#   y            — second modality repr np.ndarray
#   rna_adata    — full RNA AnnData (needed by GLUE, uniPort, totalVI)
#   second_adata — full second modality AnnData
#   second_label — "protein" or "atac"
MODELS = {
    "scot":        run_scot,
    "moscot":      run_moscot,
    "glue":        run_glue,
    "uniport":     run_uniport,
    "totalvi":     run_totalvi,
    "linear_ode":  run_linear_ode,
    "kot":         run_kot,
    "kot_nodyn":   run_kot,
    "kot_anchor":  run_kot,
}

# Per-model config overrides injected automatically at runtime
MODEL_OVERRIDES = {
    "kot_nodyn":  {"lambda_dyn": 0.0},
    "kot_anchor": {"use_anchor": True},
}


def get_repr(adata, obsm_key: str, label: str) -> np.ndarray:
    if adata is None:
        raise ValueError(f"{label} AnnData is None — check dataset config.")
    if obsm_key not in adata.obsm:
        available = list(adata.obsm.keys())
        raise ValueError(f"Key '{obsm_key}' not found in {label} obsm. Available: {available}")
    return np.array(adata.obsm[obsm_key])


def build_context(x, y, rna_adata, second_adata, second_label, modality_pair):
    return {
        "x": x,
        "y": y,
        "rna_adata": rna_adata,
        "second_adata": second_adata,
        "second_label": second_label,
        "modality_pair": modality_pair,
    }


def model_names_from_cfg(run_cfg: dict) -> list[str]:
    models = run_cfg.get("models")
    if models is None:
        return [str(run_cfg.get("model", "scot"))]
    if isinstance(models, str):
        return [models]
    return [str(model) for model in models]


def run_one(name: str, dataset: dict, run_cfg: dict) -> None:
    model_name = str(run_cfg.get("model", "scot"))
    if model_name not in MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(MODELS)}")

    output_dir = Path(run_cfg["output_root"]) / model_name / name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 50}")
    print(f"[{model_name}] Training on [{name}]")
    print(f"{'=' * 50}")

    rna_adata, protein_adata, atac_adata = load_and_preprocess_cached(
        rna_path=dataset["rna_path"],
        protein_path=dataset.get("protein_path"),
        protein_label=dataset.get("protein_label", "ADT"),
        atac_path=dataset.get("atac_path"),
        atac_label=dataset.get("atac_label", "ATAC"),
    )

    modality_pair = str(run_cfg.get("modality_pair", "rna_protein"))
    rna_repr_key = str(run_cfg.get("rna_repr", "X_pca"))
    second_repr_key = str(run_cfg.get("second_repr", "X_pca"))

    x = get_repr(rna_adata, rna_repr_key, "RNA")

    if modality_pair == "rna_protein":
        if protein_adata is None:
            print(f"[{name}] No protein data — skipping.")
            return
        y = get_repr(protein_adata, second_repr_key, "Protein")
        second_adata = protein_adata
        second_label = "protein"
    elif modality_pair == "rna_atac":
        if atac_adata is None:
            print(f"[{name}] No ATAC data — skipping.")
            return
        y = get_repr(atac_adata, second_repr_key, "ATAC")
        second_adata = atac_adata
        second_label = "atac"
    else:
        raise ValueError(f"Unknown modality_pair: '{modality_pair}'. Use 'rna_protein' or 'rna_atac'.")

    print(f"RNA repr ({rna_repr_key}): {x.shape}")
    print(f"{second_label.upper()} repr ({second_repr_key}): {y.shape}")

    context = build_context(x, y, rna_adata, second_adata, second_label, modality_pair)

    align_fn = MODELS[model_name]
    result = align_fn(context, run_cfg)
    aligned, coupling = result[0], result[1]
    loss_df = result[2] if len(result) > 2 else None

    mean_foscttm = evaluate_and_save(
        aligned, name, output_dir, second_label,
        model_name=model_name,
        obs_names=rna_adata.obs_names,
    )

    if coupling is not None:
        np.save(output_dir / "coupling.npy", coupling)
    if loss_df is not None:
        loss_csv = output_dir / "training_loss.csv"
        loss_df.to_csv(loss_csv, index=False)
        plot_training_loss(loss_csv, output_dir, model_name)

    with open(output_dir / "summary.txt", "w") as f:
        f.write(f"dataset:        {name}\n")
        f.write(f"model:          {model_name}\n")
        f.write(f"modality_pair:  {modality_pair}\n")
        f.write(f"rna_repr:       {rna_repr_key} {x.shape}\n")
        f.write(f"second_repr:    {second_repr_key} {y.shape}\n")
        f.write(f"mean_foscttm:   {mean_foscttm:.6f}\n")

    has_ground_truth = {"state", "time"}.issubset(rna_adata.obs.columns)
    if has_ground_truth:
        plot_coembedding(x, y, aligned[0], aligned[1], rna_adata.obs, output_dir, model_name)
        plot_foscttm_distribution(
            output_dir / "foscttm.csv", rna_adata.obs, output_dir, model_name
        )
        plot_foscttm_sorted(output_dir / "foscttm.csv", output_dir, model_name)
        if coupling is not None:
            plot_coupling(coupling, rna_adata.obs["time"].values, output_dir, model_name)

    print(f"Results saved to {output_dir}/")


def run_training(config_path: Path = TRAINING_CONFIG) -> None:
    train_cfg = load_yaml(config_path)
    datasets = load_yaml(DATASETS_CONFIG).get("datasets", {})
    defaults = train_cfg.get("defaults", {})
    trained_models = []

    for name, overrides in train_cfg.get("datasets", {}).items():
        if name not in datasets:
            print(f"[train] Warning: '{name}' not in {DATASETS_CONFIG}, skipping.")
            continue
        base_cfg = {**defaults, **(overrides or {})}
        for model_name in model_names_from_cfg(base_cfg):
            run_cfg = {**base_cfg, "model": model_name, **MODEL_OVERRIDES.get(model_name, {})}
            run_one(name, datasets[name], run_cfg)
            if model_name not in trained_models:
                trained_models.append(model_name)

    save_comparison_table(model_names=trained_models or None)
