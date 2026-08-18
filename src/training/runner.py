from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from src.utils.io import load_yaml, save_yaml
from src.data.preprocessing import load_and_preprocess_cached
from src.training.scot import run_scot
from src.training.moscot import run_moscot
from src.training.glue import run_glue
from src.training.uniport import run_uniport
from src.training.totalvi import run_totalvi
from src.training.linear_ode import run_linear_ode
from src.training.kot import run_kot, matrix_from_adata
from src.evaluation.foscttm import evaluate_and_save, calc_domainAveraged_FOSCTTM
from src.visualization.alignment import (
    coembedding_plot_kwargs,
    plot_coembedding,
    plot_coupling,
    plot_foscttm_distribution,
    plot_foscttm_sorted,
    plot_training_loss,
)
from src.visualization.results import save_comparison_table

DATASETS_CONFIG = Path("config/datasets.yaml")
TRAINING_CONFIG = Path("config/training.yaml")


class Tee:
    """Duplicate writes to multiple streams (e.g., stdout + file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

    def isatty(self):
        return False

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
    "totalvi":        run_totalvi,
    "linear_ode":  run_linear_ode,
    "kot":            run_kot,
    "kot_nodyn":      run_kot,
    "kot_fixedkappa": run_kot,
    "kot_fixedalpha": run_kot,
    "kot_oracle":     run_kot,
    "kot_oracle_learnalpha":       run_kot,
    "kot_oracle_learnalpha_kappa": run_kot,
    "kot_unbounded":    run_kot,
    "kot_unbounded_sn": run_kot,
    "kot_anchor":     run_kot,
}

# Per-model config overrides injected automatically at runtime
MODEL_OVERRIDES = {
    "kot_nodyn":       {"lambda_dyn": 0.0},
    "kot_fixedkappa": {
        "kot_fixed_kappa": 0.6931471805599453,   # κ fixed (mirror of fixedalpha's fixed α)
        "kot_fixed_alpha": None,                  # α learned
        "kot_alpha_min": 1.0e-3,                  # learned-α box mirrors fixedalpha's learned-κ box [1e-3, 1.5]
        "kot_alpha_max": 1.5,
        "lambda_kappa_prior": 0.0,                # κ is fixed → no prior pull
        "g_freeze_epochs": 0,
        "phi_spectral_norm": False,               # match fixedalpha (φ-side confound control)
        "g_dims": [256, 128],
    },
    "kot_fixedalpha": {
        "kot_fixed_alpha": 0.6931471805599453,
        "kot_fixed_kappa": None,
        "kot_kappa_min": 1.0e-3,
        "kot_kappa_max": 1.5,
        "kot_kappa_prior": 0.6931471805599453,
        "lambda_kappa_prior": 0.01,
        "kot_alpha_min": 1.0e-6,
        "kot_alpha_max": None,
        "g_freeze_epochs": 0,
        "phi_spectral_norm": False,
        "g_dims": [256, 128],
    },
    "kot_oracle": {
        "kot_fixed_alpha": 1.0,
        "kot_fixed_kappa": 1.0,
        "kot_fixed_beta": 0.5,
        "kot_kappa_min": 1.0,
        "kot_kappa_max": 1.0,
        "use_anchor": False,
        "kot_anchor_indices": [],
        "kot_anchor_betas": [],
        "lambda_prior": 0.0,
        "lambda_kappa_prior": 0.0,
        "kot_alpha_min": 1.0,
        "kot_alpha_max": 1.0,
        "g_freeze_epochs": 0,
        "phi_spectral_norm": False,
        "g_dims": [256, 128],
    },
    "kot_oracle_learnalpha": {
        "kot_fixed_alpha": None,
        "kot_fixed_kappa": 1.0,
        "kot_fixed_beta": 0.5,
        "kot_kappa_min": 1.0,
        "kot_kappa_max": 1.0,
        "use_anchor": False,
        "kot_anchor_indices": [],
        "kot_anchor_betas": [],
        "lambda_prior": 0.0,
        "lambda_kappa_prior": 0.0,
        "kot_alpha_min": 1.0e-3,
        "kot_alpha_max": 1.5,
        "g_freeze_epochs": 0,
        "phi_spectral_norm": False,
        "g_dims": [256, 128],
    },
    "kot_oracle_learnalpha_kappa": {
        "kot_fixed_alpha": None,
        "kot_fixed_kappa": None,
        "kot_fixed_beta": 0.5,
        "kot_kappa_min": 1.0e-3,
        "kot_kappa_max": 1.5,
        "use_anchor": False,
        "kot_anchor_indices": [],
        "kot_anchor_betas": [],
        "lambda_prior": 0.0,
        "lambda_kappa_prior": 0.0,
        "kot_alpha_min": 1.0e-3,
        "kot_alpha_max": 1.5,
        "g_freeze_epochs": 0,
        "phi_spectral_norm": False,
        "g_dims": [256, 128],
    },
    "kot_unbounded": {
        "kot_kappa_min": 1.0e-6,
        "kot_kappa_max": None,
        "kot_alpha_min": 1.0e-6,
        "kot_alpha_max": None,
        "lambda_kappa_prior": 0.0,
        "g_freeze_epochs": 0,
        "phi_spectral_norm": False,
        "g_dims": [256, 128],
    },
    "kot_unbounded_sn": {
        "kot_kappa_min": 1.0e-6,
        "kot_kappa_max": None,
        "kot_alpha_min": 1.0e-6,
        "kot_alpha_max": None,
        "lambda_kappa_prior": 0.0,
        "g_freeze_epochs": 0,
        "phi_spectral_norm": True,
        "g_dims": [256, 128],
    },
    "kot_anchor":      {"use_anchor": True},
}

# Name of the staged synthetic dataset that --stage / --scale apply to.
STAGED_DATASET = "synthetic_linked_ode"

# KOT feature layers per observation scale.
#   mean = native ODE state (linear ODE holds exactly → true oracle)
#   log  = log1p-normalized (realistic; linear ODE only approximate)
SCALE_LAYERS = {
    "mean": {
        "kot_rna_layer":     "spliced_mean",
        "kot_protein_layer": "protein_mean",
        "velocity_layer":    "true_velocity_mean",
    },
    "log": {
        "kot_rna_layer":     "spliced_log1p",
        "kot_protein_layer": "protein_log1p",
        "velocity_layer":    "true_velocity_log1p",
    },
}

# Baselines whose likelihood needs integer counts; the noise-free oracle stage has none.
COUNT_MODELS = {"glue", "totalvi"}


def stage_data_paths(stage: str) -> tuple[str, str]:
    """Stage-specific h5ad paths (mirror of synthetic_linked_ode.stage_output_paths)."""
    rna = f"cache/velocity/synthetic_linked_ode/{stage}/rna.h5ad"
    protein = f"cache/synthetic_linked_ode/{stage}/protein.h5ad"
    return rna, protein


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


def json_ready_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def save_diagnostics(output_dir: Path, diagnostics: dict) -> None:
    if not diagnostics:
        return
    path = output_dir / "diagnostics.json"
    clean = {str(key): json_ready_value(value) for key, value in diagnostics.items()}
    if path.exists():
        with open(path) as f:
            existing = json.load(f)
        if isinstance(existing, dict):
            clean = {**existing, **clean}
    with open(path, "w") as f:
        json.dump(clean, f, indent=2, sort_keys=True)


def split_model_result(result):
    aligned = result[0]
    coupling = result[1]
    loss_df = None
    diagnostics = {}

    for extra in result[2:]:
        if extra is None:
            continue
        if isinstance(extra, dict):
            diagnostics.update(extra)
        else:
            loss_df = extra

    return aligned, coupling, loss_df, diagnostics


# O(n²) OT baselines that cannot solve the full 90k-cell problem. When
# align_batch_key is set they are run within each biological batch instead.
BATCH_SPLIT_MODELS = {"scot", "moscot", "linear_ode"}


def aggregate_batch_diagnostics(batch_diags: list[dict]) -> dict:
    """Mean of the numeric diagnostics across per-batch runs."""
    out = {}
    keys = set().union(*[d.keys() for d in batch_diags]) if batch_diags else set()
    for k in keys:
        vals = [d[k] for d in batch_diags if isinstance(d.get(k), (int, float, bool))]
        if vals:
            out[k] = float(np.mean(vals))
    out["n_batches"] = len(batch_diags)
    return out


def run_alignment_per_batch(align_fn, x, y, rna_adata, second_adata,
                            second_label, modality_pair, run_cfg, batch_labels, output_dir):
    """
    Run an alignment adapter independently within each batch and reassemble.

    FOSCTTM is computed per batch (a global score across independently-aligned
    batches would be meaningless) and returned as one per-cell array in the
    original cell order. Coupling / loss traces are not combined across batches
    (adapter trace files reflect the last batch).
    """
    n = x.shape[0]
    aligned_rna = None
    aligned_second = None
    foscttm = np.empty(n, dtype=np.float64)
    batch_diags = []

    for b in np.unique(batch_labels):
        idx = np.where(batch_labels == b)[0]
        sub_context = build_context(
            x[idx], y[idx], rna_adata[idx].copy(), second_adata[idx].copy(),
            second_label, modality_pair,
        )
        sub_context["output_dir"] = output_dir
        aligned_b, _, _, diag_b = split_model_result(align_fn(sub_context, run_cfg))
        rb = np.asarray(aligned_b[0], dtype=np.float32)
        pb = np.asarray(aligned_b[1], dtype=np.float32)
        if aligned_rna is None:
            aligned_rna = np.zeros((n, rb.shape[1]), dtype=np.float32)
            aligned_second = np.zeros((n, pb.shape[1]), dtype=np.float32)
        aligned_rna[idx] = rb
        aligned_second[idx] = pb
        foscttm[idx] = np.asarray(calc_domainAveraged_FOSCTTM(rb, pb))
        batch_diags.append(diag_b)
        print(f"    [batch {b}] {len(idx)} cells | FOSCTTM={foscttm[idx].mean():.4f}")

    return [aligned_rna, aligned_second], foscttm.tolist(), aggregate_batch_diagnostics(batch_diags)


def model_names_from_cfg(run_cfg: dict) -> list[str]:
    models = run_cfg.get("models")
    if models is None:
        return [str(run_cfg.get("model", "scot"))]
    if isinstance(models, str):
        return [models]
    return [str(model) for model in models]


def resolve_models(spec, model_groups: dict) -> list[str]:
    """Resolve a --models spec into a list of model names.

    spec may be a group name (looked up in model_groups), a comma-separated
    string of model names, or an already-resolved list.
    """
    if isinstance(spec, list):
        names = spec
    elif spec in model_groups:
        names = model_groups[spec]
    else:
        names = [part.strip() for part in str(spec).split(",") if part.strip()]
    return [str(name) for name in names]


def seed_list_from_cfg(run_cfg: dict) -> tuple[list[int], bool]:
    """Return (seeds, is_multi_seed). is_multi_seed True if 'seeds' is explicitly set."""
    seeds = run_cfg.get("seeds")
    if seeds is None:
        return [int(run_cfg.get("seed", 42))], False
    if isinstance(seeds, int):
        return [seeds], True
    return [int(seed) for seed in seeds], True


def seeded_models_from_cfg(run_cfg: dict, all_models: list[str]) -> set[str]:
    """Set of models that should run across all configured seeds."""
    seeded = run_cfg.get("seeded_models")
    if seeded is None:
        return set(all_models) if "seeds" in run_cfg else set()
    if isinstance(seeded, str):
        return {seeded}
    return {str(model) for model in seeded}


def coembedding_input_matrices(x, y, rna_adata, second_adata, cfg):
    mode = str(cfg.get("viz_input", "auto")).lower()
    if mode == "repr":
        return x, y
    if mode == "feature" or (mode == "auto" and cfg.get("kot_use_feature_space")):
        return (
            matrix_from_adata(rna_adata, cfg.get("kot_rna_layer"), "RNA"),
            matrix_from_adata(second_adata, cfg.get("kot_protein_layer"), "Protein"),
        )
    return x, y


def run_one(name: str, dataset: dict, run_cfg: dict) -> None:
    model_name = str(run_cfg.get("model", "scot"))
    if model_name not in MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(MODELS)}")

    run_seed = run_cfg.get("run_seed")
    run_label = f"seed_{run_seed}" if run_seed is not None else None
    output_dir = Path(run_cfg["output_root"]) / model_name / name
    if run_label is not None:
        output_dir = output_dir / run_label

    seed_tag = f" | seed={run_seed}" if run_seed is not None else ""
    # Resume support: a finished run leaves summary.txt → skip it (and its preprocessing).
    if (output_dir / "summary.txt").exists():
        print(f"[{model_name}] already complete on [{name}]{seed_tag} — skipping")
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 50}")
    if run_label is None:
        print(f"[{model_name}] Training on [{name}]")
    else:
        print(f"[{model_name}] Training on [{name}] | seed={run_seed}")
    print(f"{'=' * 50}")

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
        if protein_adata is None:
            raise ValueError(f"Dataset '{name}' is configured for rna_protein but has no protein data.")
        y = get_repr(protein_adata, second_repr_key, "Protein")
        second_adata = protein_adata
        second_label = "protein"
    elif modality_pair == "rna_atac":
        if atac_adata is None:
            raise ValueError(f"Dataset '{name}' is configured for rna_atac but has no ATAC data.")
        y = get_repr(atac_adata, second_repr_key, "ATAC")
        second_adata = atac_adata
        second_label = "atac"
    else:
        raise ValueError(f"Unknown modality_pair: '{modality_pair}'. Use 'rna_protein' or 'rna_atac'.")

    print(f"[runner] RNA repr ({rna_repr_key}): {x.shape}")
    print(f"[runner] {second_label.upper()} repr ({second_repr_key}): {y.shape}")

    context = build_context(x, y, rna_adata, second_adata, second_label, modality_pair)
    context["output_dir"] = output_dir

    align_fn = MODELS[model_name]
    batch_key = run_cfg.get("align_batch_key")
    per_batch = (
        batch_key is not None
        and model_name in BATCH_SPLIT_MODELS
        and batch_key in rna_adata.obs.columns
    )
    start_time = time.perf_counter()
    if per_batch:
        batch_labels = np.asarray(rna_adata.obs[batch_key].values)
        print(f"[{model_name}] per-batch alignment over '{batch_key}' "
              f"({len(np.unique(batch_labels))} batches)")
        aligned, batch_foscttm, diagnostics = run_alignment_per_batch(
            align_fn, x, y, rna_adata, second_adata, second_label, modality_pair,
            run_cfg, batch_labels, output_dir,
        )
        coupling, loss_df = None, None
    else:
        aligned, coupling, loss_df, diagnostics = split_model_result(align_fn(context, run_cfg))
        batch_foscttm = None
    runtime_seconds = time.perf_counter() - start_time
    diagnostics = {
        "dataset": name,
        "model": model_name,
        "runtime_seconds": float(runtime_seconds),
        **diagnostics,
    }
    if run_seed is not None:
        diagnostics["seed"] = int(run_seed)

    result_name = f"{name}_{run_label}" if run_label is not None else name
    mean_foscttm = evaluate_and_save(
        aligned, result_name, output_dir, second_label,
        model_name=model_name,
        obs_names=rna_adata.obs_names,
        foscttm_scores=batch_foscttm,
    )
    diagnostics["mean_foscttm"] = float(mean_foscttm)
    save_diagnostics(output_dir, diagnostics)

    if coupling is not None:
        np.save(output_dir / "coupling.npy", coupling)
    if loss_df is not None:
        loss_csv = output_dir / "training_loss.csv"
        loss_df.to_csv(loss_csv, index=False)
        plot_training_loss(loss_csv, output_dir, model_name)

    with open(output_dir / "summary.txt", "w") as f:
        f.write(f"dataset:        {name}\n")
        f.write(f"model:          {model_name}\n")
        if run_seed is not None:
            f.write(f"seed:           {run_seed}\n")
        f.write(f"modality_pair:  {modality_pair}\n")
        f.write(f"rna_repr:       {rna_repr_key} {x.shape}\n")
        f.write(f"second_repr:    {second_repr_key} {y.shape}\n")
        f.write(f"mean_foscttm:   {mean_foscttm:.6f}\n")
        f.write(f"runtime_sec:    {runtime_seconds:.6f}\n")
        for key in (
            "moscot_converged",
            "moscot_cost",
            "coupling_entropy",
            "coupling_effective_sparsity",
        ):
            if key in diagnostics:
                f.write(f"{key}: {diagnostics[key]}\n")

    # FOSCTTM plots are modality-agnostic (distribution degrades to a plain
    # histogram without ground-truth branches), so they run on real data too.
    plot_foscttm_distribution(
        output_dir / "foscttm.csv", rna_adata.obs, output_dir, model_name
    )
    plot_foscttm_sorted(output_dir / "foscttm.csv", output_dir, model_name)

    # Coembedding (colours by state/time) and coupling (needs time) require the
    # synthetic ground-truth columns.
    has_ground_truth = {"state", "time"}.issubset(rna_adata.obs.columns)
    if has_ground_truth:
        viz_x, viz_y = coembedding_input_matrices(x, y, rna_adata, second_adata, run_cfg)
        plot_coembedding(
            viz_x, viz_y, aligned[0], aligned[1], rna_adata.obs, output_dir, model_name,
            rna_adata=rna_adata,
            second_adata=second_adata,
            **coembedding_plot_kwargs(run_cfg),
        )
        if coupling is not None:
            plot_coupling(coupling, rna_adata.obs["time"].values, output_dir, model_name)

    print(f"[runner] results saved to {output_dir}/")


def run_training(
    config_path: Path = TRAINING_CONFIG,
    stage: str | None = None,
    scale: str | None = None,
    run_dir: Path | None = None,
    models: str | None = None,
    datasets_filter: str | None = None,
    set_overrides: dict | None = None,
) -> None:
    train_cfg = load_yaml(config_path)
    datasets = load_yaml(DATASETS_CONFIG).get("datasets", {})
    defaults = train_cfg.get("defaults", {})
    stage_overrides_cfg = train_cfg.get("stage_overrides", {})
    cli_overrides = dict(set_overrides or {})

    # Reject typo'd --set keys: a plain dict merge would silently inject dead keys
    # (e.g. `beta_anchor_prio=gaussian`), so validate against every key that appears
    # in the config's defaults, per-dataset overrides, stage overrides, and the
    # runtime keys the runner sets itself.
    known_keys = set(defaults)
    for ov in train_cfg.get("datasets", {}).values():
        known_keys.update((ov or {}).keys())
    for ov in stage_overrides_cfg.values():
        known_keys.update((ov or {}).keys())
    for layers in SCALE_LAYERS.values():
        known_keys.update(layers.keys())
    for ov in MODEL_OVERRIDES.values():
        known_keys.update(ov.keys())
    known_keys.update({"model", "seed", "run_seed", "models", "preprocessing_cache_version"})
    unknown = set(cli_overrides) - known_keys
    if unknown:
        raise ValueError(
            f"--set has unknown config key(s): {sorted(unknown)}. "
            f"Check for typos; known keys include: {sorted(known_keys)}"
        )

    # --datasets restricts the run to a comma-separated subset of the config's
    # `datasets:` section (which otherwise trains every dataset listed there).
    selected_datasets = (
        set(name.strip() for name in datasets_filter.split(",")) if datasets_filter else None
    )

    # Resolve which models to run: --models (group name or comma-list) if given,
    # else the config's default_models group. Overrides any per-dataset model list.
    model_groups = train_cfg.get("model_groups", {})
    models_spec = models if models is not None else train_cfg.get("default_models")
    selected_models = (
        resolve_models(models_spec, model_groups) if models_spec is not None else None
    )

    # Fresh run → new timestamped dir. Resume (--run-dir) → reuse the existing dir;
    # run_one then skips any (model, seed) that already has a summary.txt.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_root = Path(defaults.get("output_root", "cache/training"))
    run_root = Path(run_dir) if run_dir is not None else base_root / f"run_{timestamp}"
    run_root.mkdir(parents=True, exist_ok=True)
    defaults["output_root"] = str(run_root)

    # Snapshot config + log under names that don't clobber the original run when resuming.
    snapshot_name = f"training_resume_{timestamp}.yaml" if run_dir is not None else "training.yaml"
    log_name = f"run_resume_{timestamp}.log" if run_dir is not None else "run.log"
    # Snapshot the ACTUAL config used, not the original file: train_cfg already carries
    # the modified defaults (output_root), and cli_overrides record what --set changed.
    # shutil.copy(config_path) would save the pre-override file — misleading for resumes
    # and for reading back what a run actually used.
    config_snapshot = run_root / snapshot_name
    snapshot_cfg = dict(train_cfg)
    if cli_overrides:
        snapshot_cfg["_cli_overrides"] = dict(cli_overrides)
    save_yaml(snapshot_cfg, config_snapshot)

    log_path = run_root / log_name
    log_file = open(log_path, "w", buffering=1)
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)

    print(f"[runner] Run timestamp: {timestamp}")
    print(f"[runner] Output root:   {run_root}{'  (resume)' if run_dir is not None else ''}")
    print(f"[runner] Config saved:  {config_snapshot}")
    print(f"[runner] Log file:      {log_path}")
    if cli_overrides:
        print(f"[runner] --set overrides: {cli_overrides}")

    trained_models = []
    result_seeded_models = set()
    trained_datasets = []

    for name, overrides in train_cfg.get("datasets", {}).items():
        if selected_datasets is not None and name not in selected_datasets:
            continue
        if name not in datasets:
            raise ValueError(f"Dataset '{name}' is not defined in {DATASETS_CONFIG}.")
        if name not in trained_datasets:
            trained_datasets.append(name)
        dataset = datasets[name]
        overrides = dict(overrides or {})

        # --stage / --scale override the staged synthetic: data path, KOT feature
        # layers, per-stage models/anchors (stage_overrides), and the cache version.
        if name == STAGED_DATASET and stage is not None:
            eff_scale = scale or "mean"
            rna_path, protein_path = stage_data_paths(stage)
            dataset = {**dataset, "rna_path": rna_path, "protein_path": protein_path}
            overrides.update(stage_overrides_cfg.get(stage, {}))
            overrides.update(SCALE_LAYERS[eff_scale])
            overrides["preprocessing_cache_version"] = f"synthetic_linked_ode_{stage}_{eff_scale}"
            print(f"[runner] stage={stage} scale={eff_scale} | rna={rna_path}")

        base_cfg = {**defaults, **overrides}

        # Fail fast on a missing anchor CSV: a wrong path (e.g. a typo'd filename)
        # would otherwise fall through to a silent no-anchor run.
        effective_csv = cli_overrides.get("beta_anchor_csv", base_cfg.get("beta_anchor_csv"))
        if effective_csv and not Path(effective_csv).exists():
            raise FileNotFoundError(
                f"beta_anchor_csv for dataset '{name}' does not exist: {effective_csv}"
            )

        if selected_models is not None:
            base_cfg["models"] = selected_models
        all_models = model_names_from_cfg(base_cfg)

        # The oracle stage has no integer counts → NB-likelihood baselines can't run there.
        if stage == "oracle":
            all_models = [m for m in all_models if m not in COUNT_MODELS]

        seeded = seeded_models_from_cfg(base_cfg, all_models)
        result_seeded_models.update(seeded)

        multi_seeds, is_multi_seed = seed_list_from_cfg(base_cfg)
        default_seeds = [int(base_cfg.get("seed", 42))]

        for model_name in all_models:
            use_run_seed = is_multi_seed and model_name in seeded
            seeds = multi_seeds if model_name in seeded else default_seeds
            for seed in seeds:
                # Order: dataset defaults → --set → per-model overrides (kot_nodyn
                # still forces lambda_dyn=0 even if --set lambda_dyn=N).
                run_cfg = {
                    **base_cfg,
                    **cli_overrides,
                    "model": model_name,
                    "seed": seed,
                    **MODEL_OVERRIDES.get(model_name, {}),
                }
                if use_run_seed:
                    run_cfg["run_seed"] = seed
                run_one(name, dataset, run_cfg)
            if model_name not in trained_models:
                trained_models.append(model_name)

    for name in trained_datasets:
        save_comparison_table(
            cache_dir=run_root,
            dataset=name,
            save_dir=run_root / "results",
            model_names=trained_models or None,
            seeded_models=sorted(result_seeded_models) or None,
        )


INT_PATTERN = re.compile(r"[+-]?\d+$")
FLOAT_PATTERN = re.compile(r"[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?$")


def parse_set_value(raw: str):
    """Parse a --set value: bool / int / float / else string."""
    key = raw.strip()
    low = key.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none"):
        return None
    if INT_PATTERN.match(key):
        return int(key)
    if FLOAT_PATTERN.match(key):
        return float(key)
    return key


def parse_set_overrides(items: list[str] | None) -> dict:
    """Parse repeated --set key=value into a dict (last wins on duplicate keys)."""
    if not items:
        return {}
    out = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got: {item!r}")
        key, raw = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"--set missing key in: {item!r}")
        out[key] = parse_set_value(raw)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run KOT / baseline training from config.")
    parser.add_argument("--config", type=Path, default=TRAINING_CONFIG,
                        help="Training config YAML.")
    parser.add_argument("--stage", choices=["oracle", "clean", "branch"], default=None,
                        help="Staged synthetic: overrides data path, feature layers, "
                             "anchor indices, and the preprocessing cache version.")
    parser.add_argument("--scale", choices=["mean", "log"], default=None,
                        help="KOT feature scale: mean (ODE-exact) or log (log1p, realistic).")
    parser.add_argument("--models", type=str, default=None,
                        help="Models to run: a group name from the config's model_groups "
                             "(e.g. baselines, kot, all) or a comma-separated list of model "
                             "names. Defaults to the config's default_models.")
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="Resume into an existing run_* dir: only run (model, seed) "
                             "outputs that are missing, instead of a new timestamped dir.")
    parser.add_argument("--datasets", type=str, default=None,
                        help="Comma-separated subset of the config's datasets to run "
                             "(e.g. bmmc_cite). Defaults to every dataset in the config.")
    parser.add_argument(
        "--set",
        dest="set_overrides",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Override a config key (repeatable), e.g. --set lambda_dyn=10. "
             "Applied after dataset defaults; per-model overrides (e.g. kot_nodyn "
             "forces lambda_dyn=0) still win.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training(
        config_path=args.config,
        stage=args.stage,
        scale=args.scale,
        run_dir=args.run_dir,
        models=args.models,
        datasets_filter=args.datasets,
        set_overrides=parse_set_overrides(args.set_overrides),
    )
