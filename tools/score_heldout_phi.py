"""Held-out φ(r) Spearman from a checkpoint, without loading the full RNA object.

protein_eval_per_protein.csv has no kot_nodyn rows. The nodyn cfgB runs kept
best_align checkpoints; scoring them on the validation barcodes is an evaluation,
not a new training run.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import yaml

from src.data.preprocessing import preprocessed_cache_prefix
from src.data.splits import selection_validation_mask
from src.models.KOT import KOTModel
from src.visualization.h5ad_obs import (
    read_dense_rows, read_obs_frame, read_obs_names, read_var_names,
)
from src.visualization.prediction import NODYN_EVAL_CSV, NODYN_RUNS
from tools.evaluate_protein_prediction import column_spearman

CHECKPOINT = "checkpoint_best_align.pt"
PHI_BATCH = 2048


def dense_key(layer: str | None, *, default: str) -> str:
    name = default if layer in (None, "", "None") else str(layer)
    return "X" if name in ("X", "x") else f"layers/{name}"


def cache_pair(payload: dict) -> tuple[Path, Path]:
    paths = payload["dataset_paths"]
    cfg = payload["run_cfg"]
    prefix = preprocessed_cache_prefix(
        paths["rna_path"],
        protein_path=paths.get("protein_path"),
        protein_label=paths.get("protein_label", "ADT"),
        protein_obsm_key=paths.get("protein_obsm_key"),
        rna_raw_layer=paths.get("rna_raw_layer"),
        rna_umap_path=paths.get("rna_umap_path"),
        cache_version=cfg.get("preprocessing_cache_version"),
        rna_min_cells=int(cfg.get("rna_min_cells", 3)),
        rna_n_top_genes=int(cfg.get("rna_n_top_genes", 2000)),
        rna_n_pcs=int(cfg.get("rna_n_pcs", 30)),
        rna_n_neighbors=int(cfg.get("rna_n_neighbors", 30)),
        add_log_velocity_layer=bool(cfg.get("add_log_velocity_layer", False)),
        log_velocity_scale=float(cfg.get("log_velocity_scale", 1.0)),
        protein_min_cells=int(cfg.get("protein_min_cells", 1)),
        protein_n_pcs=int(cfg.get("protein_n_pcs", 10)),
    )
    rna = Path(f"{prefix}.rna.h5ad")
    protein = Path(f"{prefix}.protein.h5ad")
    if not rna.exists() or not protein.exists():
        raise FileNotFoundError(f"preprocessed cache missing for {rna} / {protein}")
    return rna, protein


def model_from_config(cfg: dict, d_rna: int, d_protein: int) -> KOTModel:
    kappa_max = cfg.get("kot_kappa_max")
    alpha_max = cfg.get("kot_alpha_max")
    return KOTModel(
        d_rna, d_protein,
        phi_dims=list(cfg.get("phi_dims", [1024, 512, 256])),
        kappa_dims=list(cfg.get("kappa_dims", [64, 32])),
        g_dims=list(cfg.get("g_dims", [256, 128])),
        phi_init_gain=float(cfg.get("phi_init_gain", 0.1)),
        activation=str(cfg.get("kot_activation", "gelu")),
        init_method=str(cfg.get("kot_init", "orthogonal")),
        phi_spectral_norm=bool(cfg.get("phi_spectral_norm", False)),
        kappa_min=float(cfg.get("kot_kappa_min", 1e-6)),
        kappa_max=None if kappa_max is None else float(kappa_max),
        alpha_min=float(cfg.get("kot_alpha_min", 1e-6)),
        alpha_max=None if alpha_max is None else float(alpha_max),
    )


def score_seed(seed_dir: Path, proteins: np.ndarray, rna: np.ndarray,
               protein: np.ndarray) -> pd.DataFrame:
    payload = yaml.safe_load((seed_dir / "run_config.yaml").read_text())
    cfg = payload["run_cfg"]
    ckpt_path = seed_dir / CHECKPOINT
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    model = model_from_config(cfg, rna.shape[1], protein.shape[1])
    state = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    model.load_state_dict(state)
    model.eval()
    rna_t = torch.as_tensor(rna)
    parts = []
    with torch.no_grad():
        for start in range(0, rna_t.shape[0], PHI_BATCH):
            parts.append(model.phi(rna_t[start:start + PHI_BATCH]).cpu().numpy())
    pred = np.concatenate(parts, axis=0).astype(np.float64)
    rho = column_spearman(pred, protein.astype(np.float64))
    seed = int(cfg.get("run_seed", cfg.get("seed")))
    return pd.DataFrame({
        "dataset": payload["dataset"],
        "seed": seed,
        "protein": proteins,
        "spearman": rho,
        "arm": "nodyn",
        "n_eval_cells": int(rna.shape[0]),
        "held_out": True,
        "checkpoint": "best_align",
    })


def validation_obs(rna_cache: Path, cfg: dict) -> pd.DataFrame:
    column = cfg.get("val_stratify_by")
    return read_obs_frame(rna_cache, [column] if column else [])


def score_nodyn_dataset(dataset: str) -> pd.DataFrame:
    root = NODYN_RUNS[dataset]
    seed_dirs = sorted((root / "kot_nodyn" / dataset).glob("seed_*"))
    if not seed_dirs:
        raise FileNotFoundError(f"no nodyn seeds under {root}")
    payload = yaml.safe_load((seed_dirs[0] / "run_config.yaml").read_text())
    rna_cache, protein_cache = cache_pair(payload)
    cfg = payload["run_cfg"]
    obs = validation_obs(rna_cache, cfg)
    protein_names = read_obs_names(protein_cache)
    rna_names = obs.index.to_numpy()
    if len(rna_names) != len(protein_names) or not np.array_equal(rna_names, protein_names):
        raise ValueError(f"{dataset}: RNA and protein cache barcodes are not paired")
    val = selection_validation_mask(obs, cfg, cfg.get("seed"))
    if not val.any():
        raise ValueError(f"{dataset}: validation slice is empty")
    val_rows = np.nonzero(val)[0]
    proteins = read_var_names(protein_cache)
    rna = read_dense_rows(rna_cache, dense_key(cfg.get("kot_rna_layer"), default="Ms"),
                          val_rows)
    protein = read_dense_rows(protein_cache, dense_key(cfg.get("kot_protein_layer"),
                                                       default="X"),
                              val_rows)
    if rna.shape[0] != protein.shape[0]:
        raise ValueError("held-out RNA and protein row counts differ")
    blocks = []
    for seed_dir in seed_dirs:
        if not (seed_dir / CHECKPOINT).exists():
            print(f"[nodyn] skip {seed_dir.name}: no {CHECKPOINT}")
            continue
        blocks.append(score_seed(seed_dir, proteins, rna, protein))
        print(f"[nodyn] {dataset} {seed_dir.name}: {len(proteins)} proteins")
    if not blocks:
        raise FileNotFoundError(f"no scored nodyn seeds for {dataset}")
    return pd.concat(blocks, ignore_index=True)


def nodyn_rows(dataset: str, reference: pd.DataFrame) -> pd.DataFrame:
    """Nodyn Spearman rows aligned to a KOT protein_eval table.

    Coverage flags come from the KOT table: this evaluation does not rebuild
    the kinetic mask, and inventing one would split the pairing.
    """
    scored = pd.read_csv(NODYN_EVAL_CSV) if NODYN_EVAL_CSV.exists() else pd.DataFrame()
    if not scored.empty:
        scored = scored[scored["dataset"] == dataset].copy()
    need = reference["seed"].nunique()
    if scored.empty or scored["seed"].nunique() < need:
        computed = score_nodyn_dataset(dataset)
        others = pd.DataFrame()
        if NODYN_EVAL_CSV.exists():
            others = pd.read_csv(NODYN_EVAL_CSV)
            others = others[others["dataset"] != dataset]
        scored = computed
        NODYN_EVAL_CSV.parent.mkdir(parents=True, exist_ok=True)
        pd.concat([others, scored], ignore_index=True).to_csv(NODYN_EVAL_CSV, index=False)
    coverage = reference[["protein", "in_kinetics"]].drop_duplicates()
    if coverage.duplicated("protein").any():
        raise ValueError("A protein has two kinetics-coverage flags")
    merged = scored.merge(coverage, on="protein", how="inner", validate="many_to_one")
    merged["dataset"] = dataset
    merged = merged[merged["seed"].isin(set(reference["seed"]))].copy()
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", nargs="*", default=list(NODYN_RUNS),
                        help="Datasets whose nodyn cfgB checkpoints to score")
    args = parser.parse_args()
    others = pd.DataFrame()
    if NODYN_EVAL_CSV.exists():
        others = pd.read_csv(NODYN_EVAL_CSV)
        others = others[~others["dataset"].isin(args.dataset)]
    blocks = [score_nodyn_dataset(dataset) for dataset in args.dataset]
    scored = pd.concat(blocks, ignore_index=True)
    NODYN_EVAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([others, scored], ignore_index=True).to_csv(NODYN_EVAL_CSV, index=False)
    print(f"wrote {NODYN_EVAL_CSV}: {len(scored)} rows")


if __name__ == "__main__":
    main()
