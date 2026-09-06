#!/usr/bin/env python3
"""Re-evaluate saved KOT checkpoints. Rank on val_foscttm; mean_foscttm is in-sample."""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from glob import glob
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from src.utils.io import load_yaml
from src.utils.arrays import to_dense
from src.data.preprocessing import load_and_preprocess_cached
from src.data.synthetic_linked_ode import stage_output_paths
from src.data.projection import projection_matrix_from_adatas
from src.data.adt_gene_map import load_mapping_records
from src.data.splits import selection_validation_mask
from src.models.KOT import KOTModel
from src.training.protein_supervised import adt_targets
from src.training.registry import MODEL_OVERRIDES
from src.training.kot import (
    FixedAlpha,
    FixedKappa,
    apply_velocity_ablation,
    build_effective_velocity,
    build_velocity_weight,
    compute_velocity_backend_agreement,
    DiagInputs,
    compute_full_diagnostics,
    compute_per_cell_diagnostics,
    matrix_from_adata,
    resolve_velocity_ablation,
    set_fixed_beta,
    to_index_tensor,
    to_tensor,
    validation_metrics,
)


# Shared across run folders: keyed by the full preprocess args, never by dataset
# name (two runs can differ in rna_n_top_genes). Downstream only reads these objects.
PREPROCESSED_CACHE: dict = {}

DATASETS_CONFIG = Path("config/datasets.yaml")
DEFAULT_RUN_DIR = Path("cache/training")
DEFAULT_OUTPUT = Path("cache/results/checkpoint_eval_summary.csv")

CHECKPOINT_NAMES = ["best_align", "best_val_align", "best_dyn", "best_total", "final"]

STAGED_DATASET = "synthetic_linked_ode"
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


# ---------------------------------------------------------------------------
# Loading per-run data + reproducing model architecture
# ---------------------------------------------------------------------------

def config_path_for_run(run_folder: Path) -> Path:
    """Config snapshot for a run: training.yaml, or the latest training_resume_*.yaml.

    Resumed runs carry only training_resume_<timestamp>.yaml (identical dataset
    config), so fall back to the most recent resume snapshot when the original
    training.yaml is absent.
    """
    primary = run_folder / "training.yaml"
    if primary.exists():
        return primary
    resumes = sorted(run_folder.glob("training_resume_*.yaml"))
    if resumes:
        return resumes[-1]
    raise FileNotFoundError(f"No training.yaml or training_resume_*.yaml in {run_folder}")


def staged_run_args(run_folder: Path) -> tuple[str | None, str | None]:
    """Recover --stage/--scale from run.log or resume logs, last entry wins."""
    log_paths = [run_folder / "run.log", *sorted(run_folder.glob("run_resume_*.log"))]
    stage = None
    scale = None
    for log_path in log_paths:
        if not log_path.exists():
            continue
        for line in log_path.read_text(errors="replace").splitlines():
            match = re.search(r"\[runner\]\s+stage=([^\s]+)\s+scale=([^\s]+)", line)
            if match:
                stage, scale = match.group(1), match.group(2)
                continue
            match = re.search(
                r"synthetic_linked_ode/(oracle|clean|branch)/rna\.h5ad",
                line,
            )
            if match:
                stage = match.group(1)
    return stage, scale


def parse_cli_overrides(run_folder: Path) -> dict:
    """Recover runner --set overrides from original/resume logs, last value wins."""
    overrides = {}
    log_paths = [run_folder / "run.log", *sorted(run_folder.glob("run_resume_*.log"))]
    for log_path in log_paths:
        if not log_path.exists():
            continue
        for line in log_path.read_text(errors="replace").splitlines():
            match = re.search(r"--set overrides:\s*(\{.*\})", line)
            if match:
                overrides.update(ast.literal_eval(match.group(1)))
    return overrides


def apply_staged_dataset_overrides(
    run_folder: Path,
    dataset_name: str,
    dataset_meta: dict,
    run_cfg: dict,
    cfg: dict,
) -> tuple[dict, dict]:
    """Mirror runner.py's runtime --stage/--scale injection for saved runs."""
    if dataset_name != STAGED_DATASET:
        return dataset_meta, run_cfg

    stage, scale = staged_run_args(run_folder)
    if not stage:
        return dataset_meta, run_cfg
    eff_scale = scale or "mean"
    if eff_scale not in SCALE_LAYERS:
        raise ValueError(f"Unsupported synthetic scale in {run_folder}: {eff_scale!r}")

    rna_path, protein_path = stage_output_paths(stage)
    dataset_meta = {**dataset_meta, "rna_path": rna_path, "protein_path": protein_path}
    run_cfg = dict(run_cfg)
    run_cfg.update((cfg.get("stage_overrides") or {}).get(stage, {}))
    run_cfg.update(SCALE_LAYERS[eff_scale])
    run_cfg["preprocessing_cache_version"] = f"synthetic_linked_ode_{stage}_{eff_scale}"
    return dataset_meta, run_cfg


def dataset_config_for_run(run_folder: Path, dataset_name: str):
    """Merged training config + dataset metadata for one dataset in a run."""
    cfg_path = config_path_for_run(run_folder)
    cfg = load_yaml(cfg_path)
    datasets_meta = load_yaml(DATASETS_CONFIG).get("datasets", {})

    defaults = cfg.get("defaults", {})
    datasets_section = cfg.get("datasets", {})
    if dataset_name not in datasets_section:
        raise ValueError(f"{cfg_path} has no dataset block for {dataset_name!r}")
    if dataset_name not in datasets_meta:
        raise ValueError(f"{DATASETS_CONFIG} has no dataset metadata for {dataset_name!r}")
    overrides = datasets_section[dataset_name] or {}
    run_cfg = {**defaults, **overrides}
    dataset_meta, run_cfg = apply_staged_dataset_overrides(
        run_folder, dataset_name, datasets_meta[dataset_name], run_cfg, cfg,
    )
    run_cfg.update(parse_cli_overrides(run_folder))
    return run_cfg, dataset_meta, dataset_name


def load_dataset_for_run(run_folder: Path, dataset_name: str | None = None):
    """Load preprocessed RNA + protein AnnData using this run's config snapshot."""
    if dataset_name is None:
        cfg = load_yaml(config_path_for_run(run_folder))
        datasets_section = cfg.get("datasets", {})
        if not datasets_section:
            raise ValueError(f"{config_path_for_run(run_folder)} has no datasets section")
        dataset_name = next(iter(datasets_section.keys()))

    run_cfg, dataset_meta, name = dataset_config_for_run(run_folder, dataset_name)
    rna_adata, protein_adata = load_preprocessed_cached(run_cfg, dataset_meta)
    return rna_adata, protein_adata, run_cfg, name


def load_preprocessed_cached(run_cfg: dict, dataset_meta: dict):
    """The run's preprocessed RNA + protein AnnData, loaded once per distinct config."""
    kwargs = dict(
        rna_path=dataset_meta["rna_path"],
        protein_path=dataset_meta.get("protein_path"),
        protein_label=dataset_meta.get("protein_label", "ADT"),
        protein_obsm_key=dataset_meta.get("protein_obsm_key"),
        atac_path=None,
        rna_raw_layer=dataset_meta.get("rna_raw_layer"),
        rna_umap_path=dataset_meta.get("rna_umap_path"),
        force_recompute=False,
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
    key = tuple(sorted((name, str(value)) for name, value in kwargs.items()))
    if key not in PREPROCESSED_CACHE:
        rna_adata, protein_adata, _ = load_and_preprocess_cached(**kwargs)
        PREPROCESSED_CACHE[key] = (rna_adata, protein_adata)
    return PREPROCESSED_CACHE[key]


def cfg_with_model_overrides(run_cfg: dict, model_name: str) -> dict:
    return {**run_cfg, **MODEL_OVERRIDES.get(model_name, {})}


def build_model_and_tensors(
    run_cfg: dict, rna_adata, protein_adata, model_name: str, device,
    velocity_mode: str = "as-trained", mapping_mode: str = "as-trained",
    with_velocity_backend: bool = True,
):
    """Reproduce KOTModel architecture + input tensors (R, V, P, S) for evaluation."""
    run_cfg = cfg_with_model_overrides(run_cfg, model_name)
    use_feature_space = bool(run_cfg.get("kot_use_feature_space", False))
    rna_layer = run_cfg.get("kot_rna_layer")
    velocity_layer = run_cfg.get("kot_velocity_layer") or run_cfg.get("velocity_layer")
    use_explicit_links = bool(run_cfg.get("kot_use_explicit_links", True))

    if use_feature_space:
        x = matrix_from_adata(rna_adata, rna_layer, "RNA")
        # Same contract run_kot trains against, so a re-evaluated checkpoint is scored
        # in the units its phi actually learned. Reading protein_adata.X here while
        # run_kot honoured protein_target_normalization would compare an rna_size phi
        # against CLR observations and report a model that does not exist.
        y = adt_targets({"rna_adata": rna_adata, "second_adata": protein_adata}, run_cfg)
    else:
        x = np.array(rna_adata.obsm["X_pca"])
        y = np.array(protein_adata.obsm["X_pca"])

    D_r = x.shape[1]
    D_p = y.shape[1]

    mapping_records = None
    mapping_csv = run_cfg.get("adt_mapping_csv")
    if mapping_csv:
        if not Path(mapping_csv).exists():
            raise FileNotFoundError(f"adt_mapping_csv missing: {mapping_csv}")
        mapping_records = load_mapping_records(Path(mapping_csv))
    require_full_panel = (
        bool(run_cfg.get("kot_require_full_panel_mapping", True))
        and mapping_records is not None
    )
    S_np, align_mask, kin_mask, _report = projection_matrix_from_adatas(
        rna_adata,
        protein_adata,
        mapping_records=mapping_records,
        use_mean_expr=False,
        use_explicit_links=use_explicit_links,
        require_full_panel=require_full_panel,
    )
    # The RNA->protein map is its own control axis, independent of the velocity: run_kot
    # applies the permutation in exactly this position, before the PC projection, and
    # mapping_mode="real" asks for the TRUE map whatever the run trained on. Everything
    # that follows from S -- S_model, the velocity-gene kinetic filter, the RNA-side
    # velocity weight -- follows this one choice.
    if mapping_mode == "as-trained" and run_cfg.get("kot_s_permute", False):
        # Seeded off the model seed, but these tensors are built once per (run, model)
        # outside the seed loop, so the run's own permutation cannot be rebuilt here.
        raise ValueError(
            "This run trained with kot_s_permute=true; its S is per-seed and cannot be "
            "reproduced here. Use --mapping real to score it against the true mapping "
            "instead (written as diagnostics_<ckpt>_realmap.json)."
        )
    if use_feature_space:
        S_model = S_np.copy()
    elif "PCs" in rna_adata.varm and "PCs" in protein_adata.varm:
        S_model = S_np @ np.array(rna_adata.varm["PCs"])
        protein_PCs = np.array(protein_adata.varm["PCs"])
        S_model = protein_PCs.T @ S_model
    else:
        raise ValueError("Representation mode needs PCs in .varm for both RNA and protein.")

    V_raw = None
    velocity_candidates = [velocity_layer] if velocity_layer else ["true_velocity", "velocity"]
    for key in velocity_candidates:
        if key in rna_adata.layers:
            V_raw = np.nan_to_num(to_dense(rna_adata.layers[key], np.float32), nan=0.0)
            break
    if V_raw is None:
        raise ValueError(f"No velocity layer found (tried {velocity_candidates})")

    if V_raw.shape[1] == D_r:
        V = V_raw.astype(np.float32)
    elif not use_feature_space and "PCs" in rna_adata.varm:
        V = V_raw @ np.array(rna_adata.varm["PCs"])
    else:
        raise ValueError(f"Velocity shape {V_raw.shape} incompatible with RNA {x.shape}")

    if bool(run_cfg.get("kot_kinetics_require_velocity_gene", False)) and "velocity_genes" in rna_adata.var:
        v_genes = np.asarray(rna_adata.var["velocity_genes"].values, dtype=bool)
        has_stable = (S_np.astype(bool) & v_genes[None, :]).any(axis=1)
        kin_mask = kin_mask & has_stable

    csv_source_of_truth = mapping_records is not None
    use_kinetics_mask = (
        bool(run_cfg.get("kot_use_kinetics_mask", True)) or csv_source_of_truth
    ) and use_feature_space
    kin_mask_np = kin_mask.astype(np.float32) if use_kinetics_mask else None

    align_cols = None
    if use_feature_space and not align_mask.all():
        align_cols = torch.as_tensor(np.nonzero(align_mask)[0], dtype=torch.long, device=device)

    R_t = to_tensor(x, device)
    P_t = to_tensor(y, device)
    S_t = to_tensor(S_model, device)
    mask_t = to_tensor(kin_mask_np, device) if kin_mask_np is not None else None
    # Effective velocity, built the same way run_kot builds it (RNA-side weight, then gauge
    # normalisation). The diagnostics take this single vector — re-evaluating a checkpoint
    # on a raw or differently-scaled velocity would report physics numbers that do not match
    # the run that produced the checkpoint.
    # Reproduce the run's own velocity by default: a control arm trained against a broken
    # field, so re-evaluating it on the real one would overwrite its physics numbers with
    # a different quantity under the same name. velocity_mode="real" deliberately asks for
    # that comparison and is written under its own checkpoint name.
    if velocity_mode == "as-trained":
        if resolve_velocity_ablation(run_cfg) == "shuffle":
            # The permutation is drawn per model seed, but these tensors are built once
            # per (run, model) outside the seed loop, so the run's own field cannot be
            # rebuilt here. Refuse rather than evaluate against a different permutation
            # and write the result under the run's own diagnostics name. reverse and zero
            # carry no seed, so they reproduce exactly and fall through.
            raise ValueError(
                "This run trained with the velocity shuffle control; its velocity is "
                "per-seed and cannot be reproduced here. Use --velocity real to score it "
                "against the true velocity instead (written as "
                "diagnostics_<ckpt>_realvel.json)."
            )
        V, _ = apply_velocity_ablation(V, None, run_cfg)
    velocity_weight_np = build_velocity_weight(rna_adata, S_np, run_cfg, use_feature_space)
    V_eff_t = to_tensor(build_effective_velocity(V, velocity_weight_np, run_cfg), device)
    # This tool overwrites each run's diagnostics_<ckpt>.json, so it has to reproduce the
    # cross-backend velocity cosine too — otherwise re-evaluating a run silently strips it.
    # Checkpoint-independent (the velocity is fixed input), so it is computed once here.
    # Reads the comparison h5ad once here; other tools discard it.
    velocity_backend = (
        compute_velocity_backend_agreement(rna_adata, V, run_cfg, velocity_weight_np)
        if with_velocity_backend else {}
    )

    phi_dims        = list(run_cfg.get("phi_dims", [1024, 512, 256]))
    kappa_dims      = list(run_cfg.get("kappa_dims", [64, 32]))
    g_dims          = list(run_cfg.get("g_dims", [256, 128]))
    phi_init_gain   = float(run_cfg.get("phi_init_gain", 0.1))
    activation      = str(run_cfg.get("kot_activation", "gelu"))
    init_method     = str(run_cfg.get("kot_init", "orthogonal"))
    phi_spectral_norm = bool(run_cfg.get("phi_spectral_norm", False))
    kappa_min = float(run_cfg.get("kot_kappa_min", 1e-6))
    kappa_max_cfg = run_cfg.get("kot_kappa_max")
    kappa_max = float(kappa_max_cfg) if kappa_max_cfg is not None else None
    alpha_min = float(run_cfg.get("kot_alpha_min", 1e-6))
    alpha_max_cfg = run_cfg.get("kot_alpha_max")
    alpha_max = float(alpha_max_cfg) if alpha_max_cfg is not None else None

    model = KOTModel(
        D_r, D_p,
        phi_dims=phi_dims,
        kappa_dims=kappa_dims,
        g_dims=g_dims,
        phi_init_gain=phi_init_gain,
        activation=activation,
        init_method=init_method,
        phi_spectral_norm=phi_spectral_norm,
        kappa_min=kappa_min,
        kappa_max=kappa_max,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
    ).to(device)

    fixed_kappa_value = run_cfg.get("kot_fixed_kappa")
    if fixed_kappa_value is not None:
        model.kappa = FixedKappa(float(fixed_kappa_value)).to(device)
    fixed_alpha_value = run_cfg.get("kot_fixed_alpha")
    if fixed_alpha_value is not None:
        model.g = FixedAlpha(fixed_alpha_value, D_p).to(device)
    fixed_beta_value = run_cfg.get("kot_fixed_beta")
    if fixed_beta_value is not None:
        set_fixed_beta(model, fixed_beta_value, D_p, device)

    # The same validation slice the run itself used, so a re-evaluation scores the
    # cells the run held out and not a fresh sample. Every input to the draw has to be
    # read back from the run's own config: with val_split_per_seed the split is keyed
    # on the run's SEED, and with val_stratify_by it is drawn per stratum, so passing
    # the bare val_split_seed here would silently rebuild a different set of cells.
    # diagnostics.json records val_split_digest; that is what this can be checked against.
    # Only the fit-eligible random validation slice may populate val diagnostics.
    # Group-held-out rows (Papalexi KO cells) are an OOD test set, not validation data.
    val_mask = selection_validation_mask(rna_adata.obs, run_cfg, run_cfg.get("seed"))
    val_idx_t = (
        to_index_tensor(np.nonzero(val_mask)[0], device) if val_mask.any() else None
    )

    return (model, R_t, V_eff_t, P_t, S_t, rna_adata.obs, mask_t, align_cols,
            velocity_backend, val_idx_t)


# ---------------------------------------------------------------------------
# Per-checkpoint evaluation
# ---------------------------------------------------------------------------

def evaluate_one_checkpoint(
    model,
    ckpt_path: Path,
    R_t,
    V_eff_t,
    P_t,
    S_t,
    rna_obs,
    mask_t=None,
    align_cols=None,
    velocity_backend=None,
    val_idx=None,
):
    """Load checkpoint into model, recompute diagnostics + per-cell arrays."""
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Move state_dict to the model's device.
    target_device = R_t.device
    state_dict = {k: v.to(target_device) for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(state_dict)

    diag_inputs = DiagInputs(R_t, V_eff_t, P_t, S_t, rna_obs, mask_t, align_cols, val_idx)
    diagnostics = compute_full_diagnostics(model, diag_inputs)
    per_cell = compute_per_cell_diagnostics(model, diag_inputs)
    mean_foscttm = float(np.mean(per_cell["foscttm"]))

    if velocity_backend:
        diagnostics.update(velocity_backend)
    diagnostics.update(validation_metrics(per_cell))
    diagnostics["checkpoint_epoch"]  = int(ckpt["epoch"])
    diagnostics["checkpoint_losses"] = ckpt.get("losses", {})
    diagnostics["mean_foscttm"]      = mean_foscttm
    return diagnostics, per_cell


# ---------------------------------------------------------------------------
# Walk and evaluate
# ---------------------------------------------------------------------------

def artifact_suffix(velocity_mode: str, mapping_mode: str) -> str:
    """Name the diagnostics file after the inputs it was scored against.

    A run's own numbers live in diagnostics_<ckpt>.json; scoring a control arm against a
    real input answers a different question, so it gets its own name rather than
    overwriting them. The two axes are independent, hence two independent tags.
    """
    return (("_realvel" if velocity_mode == "real" else "")
            + ("_realmap" if mapping_mode == "real" else ""))


def evaluate_run(run_folder: Path, device, save_percell: bool,
                 velocity_mode: str = "as-trained",
                 mapping_mode: str = "as-trained") -> list[dict]:
    """Evaluate all checkpoints across all (model, seed) folders in one run."""
    print(f"\n=== {run_folder.name} ===")

    rows = []
    data_cache = {}
    for model_dir in sorted(p for p in run_folder.iterdir() if p.is_dir()):
        model_name = model_dir.name
        if model_name in {"cache", "results"} or model_name.startswith("."):
            continue

        for dataset_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            dataset_name = dataset_dir.name
            if dataset_name not in data_cache:
                data_cache[dataset_name] = load_dataset_for_run(run_folder, dataset_name)
            rna_adata, protein_adata, run_cfg, _ = data_cache[dataset_name]

            (model, R_t, V_eff_t, P_t, S_t, rna_obs, mask_t, align_cols,
             velocity_backend, val_idx_t) = build_model_and_tensors(
                run_cfg, rna_adata, protein_adata, model_name, device, velocity_mode,
                mapping_mode,
            )

            for seed_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")):
                tail = seed_dir.name.split("_", 1)[1]
                if not tail.isdigit():
                    continue
                seed = int(tail)

                for ckpt_name in CHECKPOINT_NAMES:
                    ckpt_path = seed_dir / f"checkpoint_{ckpt_name}.pt"
                    if not ckpt_path.exists():
                        continue

                    diagnostics, per_cell = evaluate_one_checkpoint(
                        model,
                        ckpt_path,
                        R_t,
                        V_eff_t,
                        P_t,
                        S_t,
                        rna_obs,
                        mask_t,
                        align_cols,
                        velocity_backend,
                        val_idx_t,
                    )

                    # Per-checkpoint artifacts
                    suffix = artifact_suffix(velocity_mode, mapping_mode)
                    diag_out  = seed_dir / f"diagnostics_{ckpt_name}{suffix}.json"
                    with open(diag_out, "w") as f:
                        json.dump(diagnostics, f, indent=2)
                    if save_percell:
                        pd.DataFrame({"foscttm": per_cell["foscttm"]}).to_csv(
                            seed_dir / f"foscttm_{ckpt_name}.csv", index=False,
                        )

                    # Summary row
                    row = {
                        "run":                run_folder.name,
                        "model":              model_name,
                        "dataset":            dataset_name,
                        "seed":               seed,
                        "checkpoint":         ckpt_name,
                        "checkpoint_epoch":   diagnostics["checkpoint_epoch"],
                        "mean_foscttm":       diagnostics["mean_foscttm"],
                        "val_foscttm":        diagnostics["val_foscttm"],
                        "val_foscttm_in_full_pool": diagnostics["val_foscttm_in_full_pool"],
                        "time_mae":           diagnostics["time_mae"],
                        "time_spearman":      diagnostics["time_spearman"],
                        "traj_dtw_temporal":  diagnostics["traj_dtw_temporal"],
                        "traj_dtw_recon":     diagnostics["traj_dtw_recon"],
                        "phi_variance_ratio": diagnostics["phi_variance_ratio"],
                        "jvp_rhs_cos":        diagnostics["jvp_rhs_cos"],
                        "jvp_norm":           diagnostics["jvp_norm"],
                        "rhs_norm":           diagnostics["rhs_norm"],
                        "kappa_mean":         diagnostics["kappa_mean"],
                        "kappa_std":          diagnostics["kappa_std"],
                        "beta_mean":          diagnostics["beta_mean"],
                        "beta_std":           diagnostics["beta_std"],
                    }
                    # Include training-time losses if present (align, dyn, anchor, total)
                    losses = diagnostics.get("checkpoint_losses", {})
                    if isinstance(losses, dict):
                        for k, v in losses.items():
                            if isinstance(v, (int, float)):
                                row[f"loss_{k}"] = float(v)

                    rows.append(row)
                    print(f"  ✓ {model_name}/{dataset_name}/seed_{seed:>4}/{ckpt_name:<11} "
                          f"FOSCTTM={diagnostics['mean_foscttm']:.4f}  "
                          f"cos={diagnostics['jvp_rhs_cos']:+.3f}  "
                          f"κ={diagnostics['kappa_mean']:.3f}  β={diagnostics['beta_mean']:.3f}")

    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def collect_run_folders(args) -> list[Path]:
    if not args.runs:
        return sorted(DEFAULT_RUN_DIR.glob("run_*"))

    folders: list[Path] = []
    for pattern in args.runs:
        # Direct path?
        p = Path(pattern)
        if p.exists() and p.is_dir():
            folders.append(p)
            continue
        # Glob pattern
        matches = sorted(Path(m) for m in glob(pattern))
        folders.extend([m for m in matches if m.is_dir()])
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for f in folders:
        if f not in seen:
            seen.add(f)
            unique.append(f)
    return unique


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs", nargs="*", default=None,
        help="Run folder paths or glob patterns (default: all cache/training/run_*)",
    )
    parser.add_argument(
        "--output", default=str(DEFAULT_OUTPUT),
        help=f"Output CSV path for summary (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--save-percell", action="store_true",
        help="Also write foscttm_<checkpoint>.csv per (model, seed). Off by default to save disk.",
    )
    parser.add_argument(
        "--velocity", choices=["as-trained", "real"], default="as-trained",
        help="Which velocity field to score against. 'real' ignores "
             "kot_velocity_ablation and asks whether a control-trained model fits the "
             "TRUE velocity; results go to diagnostics_<ckpt>_realvel.json so the run's "
             "own numbers are preserved.",
    )
    parser.add_argument(
        "--mapping", choices=["as-trained", "real"], default="as-trained",
        help="Which RNA->protein map S to score against. 'real' ignores kot_s_permute "
             "and asks whether a permuted-map run fits the TRUE mapping -- the only way "
             "to score that control, since its S is drawn per seed. Independent of "
             "--velocity; results go to diagnostics_<ckpt>_realmap.json.",
    )
    args = parser.parse_args()

    run_folders = collect_run_folders(args)
    if not run_folders:
        print("No run folders found.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Found {len(run_folders)} run folder(s).")

    all_rows = []
    for run_folder in run_folders:
        rows = evaluate_run(run_folder, device, args.save_percell, args.velocity,
                            args.mapping)
        all_rows.extend(rows)

    if not all_rows:
        print("\nNo rows produced.")
        return

    df = pd.DataFrame(all_rows)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print()
    print(f"✓ Wrote {len(df)} rows to {out_path}")
    print(f"  runs:        {df['run'].nunique()}")
    print(f"  models:      {sorted(df['model'].unique())}")
    print(f"  datasets:    {sorted(df['dataset'].unique())}")
    print(f"  checkpoints: {sorted(df['checkpoint'].unique())}")
    print()
    print("Preview (best mean_foscttm per checkpoint):")
    preview = df.sort_values("mean_foscttm").groupby("checkpoint").head(3)
    print(preview[["run", "model", "dataset", "seed", "checkpoint", "checkpoint_epoch",
                   "mean_foscttm", "jvp_rhs_cos", "kappa_mean", "beta_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
