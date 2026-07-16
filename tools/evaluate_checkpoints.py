#!/usr/bin/env python3
"""
Re-evaluate every saved KOT checkpoint and write per-checkpoint diagnostics
plus a single compact summary CSV across ALL runs (v1, v2, v3, anything in
cache/training/run_*).

For each (run, model, seed, checkpoint) it loads `checkpoint_<name>.pt`,
restores the model, recomputes FOSCTTM + full diagnostics, and writes:

  <run>/<model>/<dataset>/<seed>/diagnostics_<checkpoint>.json
  <run>/<model>/<dataset>/<seed>/foscttm_<checkpoint>.csv   (optional)

Plus one summary file with all rows:

  cache/results/checkpoint_eval_summary.csv

Usage:
  python evaluate_checkpoints.py                       # all runs
  python evaluate_checkpoints.py --runs cache/training/run_20260617_085640
  python evaluate_checkpoints.py --runs "cache/training/run_20260617_*"
  python evaluate_checkpoints.py --save-percell        # also dump per-cell CSVs
  python evaluate_checkpoints.py --output my_summary.csv

The summary CSV has one row per (run, model, seed, checkpoint) and includes:
  mean_foscttm, time_mae, time_spearman, jvp_rhs_cos, kappa_mean, beta_mean,
  phi_variance_ratio, checkpoint_epoch, training-time loss values, etc.
"""

from __future__ import annotations

import argparse
import json
import traceback
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.utils.io import load_yaml
from src.data.preprocessing import load_and_preprocess_cached
from src.data.projection import projection_matrix_from_adatas
from src.models.KOT import KOTModel
from src.training.kot import (
    FixedKappa,
    compute_full_diagnostics,
    compute_per_cell_diagnostics,
    dense_array,
    matrix_from_adata,
    to_tensor,
)


DATASETS_CONFIG = Path("config/datasets.yaml")
DEFAULT_RUN_DIR = Path("cache/training")
DEFAULT_OUTPUT = Path("cache/results/checkpoint_eval_summary.csv")

CHECKPOINT_NAMES = ["best_align", "best_dyn", "best_total", "final"]

# Mirror runner.MODEL_OVERRIDES so we apply the right kappa for kot_fixedkappa.
MODEL_OVERRIDES = {
    "kot_nodyn":      {"lambda_dyn": 0.0},
    "kot_fixedkappa": {"kot_fixed_kappa": 0.6931471805599453},
    "kot_anchor":     {"use_anchor": True},
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


def load_dataset_for_run(run_folder: Path):
    """Load preprocessed RNA + protein AnnData using this run's config snapshot."""
    cfg_path = config_path_for_run(run_folder)
    cfg = load_yaml(cfg_path)
    datasets_meta = load_yaml(DATASETS_CONFIG).get("datasets", {})

    defaults = cfg.get("defaults", {})
    datasets_section = cfg.get("datasets", {})

    if not datasets_section:
        raise ValueError(f"{cfg_path} has no datasets section")

    name = next(iter(datasets_section.keys()))
    overrides = datasets_section[name] or {}
    run_cfg = {**defaults, **overrides}
    dataset_meta = datasets_meta[name]

    rna_adata, protein_adata, _ = load_and_preprocess_cached(
        rna_path=dataset_meta["rna_path"],
        protein_path=dataset_meta.get("protein_path"),
        protein_label=dataset_meta.get("protein_label", "ADT"),
        protein_obsm_key=dataset_meta.get("protein_obsm_key"),
        atac_path=None,
        rna_raw_layer=dataset_meta.get("rna_raw_layer"),
        rna_umap_path=dataset_meta.get("rna_umap_path"),
        force_recompute=False,
        cache_version=run_cfg.get("preprocessing_cache_version"),
        add_log_velocity_layer=bool(run_cfg.get("add_log_velocity_layer", False)),
        log_velocity_scale=float(run_cfg.get("log_velocity_scale", 1.0)),
    )
    return rna_adata, protein_adata, run_cfg, name


def build_model_and_tensors(
    run_cfg: dict, rna_adata, protein_adata, model_name: str, device,
):
    """Reproduce KOTModel architecture + input tensors (R, V, P, S) for evaluation."""
    use_feature_space = bool(run_cfg.get("kot_use_feature_space", False))
    rna_layer = run_cfg.get("kot_rna_layer")
    protein_layer = run_cfg.get("kot_protein_layer")
    velocity_layer = run_cfg.get("kot_velocity_layer") or run_cfg.get("velocity_layer")
    use_explicit_links = bool(run_cfg.get("kot_use_explicit_links", True))

    if use_feature_space:
        x = matrix_from_adata(rna_adata, rna_layer, "RNA")
        y = matrix_from_adata(protein_adata, protein_layer, "protein")
    else:
        x = np.array(rna_adata.obsm["X_pca"])
        y = np.array(protein_adata.obsm["X_pca"])

    D_r = x.shape[1]
    D_p = y.shape[1]

    S_np = projection_matrix_from_adatas(
        rna_adata, protein_adata,
        use_mean_expr=False,
        use_explicit_links=use_explicit_links,
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
            V_raw = np.nan_to_num(dense_array(rna_adata.layers[key]), nan=0.0)
            break
    if V_raw is None:
        raise ValueError(f"No velocity layer found (tried {velocity_candidates})")

    if V_raw.shape[1] == D_r:
        V = V_raw.astype(np.float32)
    elif not use_feature_space and "PCs" in rna_adata.varm:
        V = V_raw @ np.array(rna_adata.varm["PCs"])
    else:
        raise ValueError(f"Velocity shape {V_raw.shape} incompatible with RNA {x.shape}")

    R_t = to_tensor(x, device)
    P_t = to_tensor(y, device)
    V_t = to_tensor(V, device)
    S_t = to_tensor(S_model, device)

    phi_dims        = list(run_cfg.get("phi_dims", [1024, 512, 256]))
    kappa_dims      = list(run_cfg.get("kappa_dims", [64, 32]))
    g_dims          = list(run_cfg.get("g_dims", [256, 128]))
    phi_init_gain   = float(run_cfg.get("phi_init_gain", 0.1))
    activation      = str(run_cfg.get("kot_activation", "gelu"))
    init_method     = str(run_cfg.get("kot_init", "orthogonal"))
    phi_spectral_norm = bool(run_cfg.get("phi_spectral_norm", False))

    model = KOTModel(
        D_r, D_p,
        phi_dims=phi_dims,
        kappa_dims=kappa_dims,
        g_dims=g_dims,
        phi_init_gain=phi_init_gain,
        activation=activation,
        init_method=init_method,
        phi_spectral_norm=phi_spectral_norm,
    ).to(device)

    # Apply model-specific overrides (matches runner.MODEL_OVERRIDES)
    overrides = MODEL_OVERRIDES.get(model_name, {})
    fixed_kappa_value = overrides.get("kot_fixed_kappa", run_cfg.get("kot_fixed_kappa"))
    if fixed_kappa_value is not None:
        model.kappa = FixedKappa(float(fixed_kappa_value)).to(device)

    return model, R_t, V_t, P_t, S_t, rna_adata.obs


# ---------------------------------------------------------------------------
# Per-checkpoint evaluation
# ---------------------------------------------------------------------------

def evaluate_one_checkpoint(model, ckpt_path: Path, R_t, V_t, P_t, S_t, rna_obs):
    """Load checkpoint into model, recompute diagnostics + per-cell arrays."""
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Move state_dict to the model's device.
    target_device = R_t.device
    state_dict = {k: v.to(target_device) for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(state_dict)

    diagnostics = compute_full_diagnostics(model, R_t, V_t, P_t, S_t, rna_obs)
    per_cell    = compute_per_cell_diagnostics(model, R_t, V_t, P_t, S_t, rna_obs)
    mean_foscttm = float(np.mean(per_cell["foscttm"]))

    diagnostics["checkpoint_epoch"]  = int(ckpt["epoch"])
    diagnostics["checkpoint_losses"] = ckpt.get("losses", {})
    diagnostics["mean_foscttm"]      = mean_foscttm
    return diagnostics, per_cell


# ---------------------------------------------------------------------------
# Walk and evaluate
# ---------------------------------------------------------------------------

def evaluate_run(run_folder: Path, device, save_percell: bool) -> list[dict]:
    """Evaluate all checkpoints across all (model, seed) folders in one run."""
    print(f"\n=== {run_folder.name} ===")
    rna_adata, protein_adata, run_cfg, dataset_name = load_dataset_for_run(run_folder)

    rows = []
    for model_dir in sorted(p for p in run_folder.iterdir() if p.is_dir()):
        model_name = model_dir.name
        if model_name in {"cache"}:
            continue

        dataset_dir = model_dir / dataset_name
        if not dataset_dir.exists():
            continue

        try:
            model, R_t, V_t, P_t, S_t, rna_obs = build_model_and_tensors(
                run_cfg, rna_adata, protein_adata, model_name, device,
            )
        except Exception as e:
            print(f"  [{model_name}] build_model FAILED: {e}")
            continue

        for seed_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")):
            try:
                seed = int(seed_dir.name.split("_", 1)[1])
            except ValueError:
                continue

            for ckpt_name in CHECKPOINT_NAMES:
                ckpt_path = seed_dir / f"checkpoint_{ckpt_name}.pt"
                if not ckpt_path.exists():
                    continue

                try:
                    diagnostics, per_cell = evaluate_one_checkpoint(
                        model, ckpt_path, R_t, V_t, P_t, S_t, rna_obs,
                    )
                except Exception as e:
                    print(f"  [{model_name}/seed_{seed}/{ckpt_name}] eval FAILED: {e}")
                    continue

                # Per-checkpoint artifacts
                diag_out  = seed_dir / f"diagnostics_{ckpt_name}.json"
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
                    "seed":               seed,
                    "checkpoint":         ckpt_name,
                    "checkpoint_epoch":   diagnostics["checkpoint_epoch"],
                    "mean_foscttm":       diagnostics["mean_foscttm"],
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
                print(f"  ✓ {model_name}/seed_{seed:>4}/{ckpt_name:<11} "
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
        try:
            rows = evaluate_run(run_folder, device, args.save_percell)
        except Exception:
            print(f"\n[{run_folder.name}] FAILED:")
            traceback.print_exc()
            continue
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
    print(f"  checkpoints: {sorted(df['checkpoint'].unique())}")
    print()
    print("Preview (best mean_foscttm per checkpoint):")
    preview = df.sort_values("mean_foscttm").groupby("checkpoint").head(3)
    print(preview[["run", "model", "seed", "checkpoint", "checkpoint_epoch",
                   "mean_foscttm", "jvp_rhs_cos", "kappa_mean", "beta_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
