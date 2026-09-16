"""Map lineage labels onto a run's aligned arrays using saved cell IDs.

Equal cell counts do not prove equal order. MaxFuse (and any holdout_second
method) writes only the fitted cells, so a protein cache of the full dataset
cannot be aligned by position. The cache is the one the run's config hashes to,
not the most recently written file with a similar name.
"""
from __future__ import annotations

from pathlib import Path

import json
import numpy as np
import pandas as pd
import yaml

from src.data.preprocessing import preprocessed_cache_prefix
from src.data.splits import fitted_row_indices
from src.visualization import lineage
from src.visualization.h5ad_obs import read_obs_frame, read_obs_names

PREPROCESSED_DIR = Path("cache/preprocessed")
DATASETS_YAML = Path("config/datasets.yaml")


def aligned_cell_positions(n_aligned: int, n_full: int,
                           fit_rows: np.ndarray | None) -> np.ndarray | None:
    """Positions in the preprocessing cache for each row of aligned_*.npy.

    Full-length embeddings keep cache order. A holdout that dropped cells keeps
    the fitted rows, in that order. Any other length is unrecoverable.
    """
    if n_aligned == n_full:
        return np.arange(n_full)
    if fit_rows is not None and n_aligned == len(fit_rows):
        return np.asarray(fit_rows, dtype=np.int64)
    return None


def run_config_payload(run: Path) -> dict | None:
    path = run / "run_config.yaml"
    if not path.exists():
        return None
    return yaml.safe_load(path.read_text())


def protein_cache_from_run(run: Path) -> Path | None:
    """The protein h5ad this run trained on, or None if that file cannot be named."""
    payload = run_config_payload(run)
    if payload is not None:
        paths = payload.get("dataset_paths") or {}
        cfg = payload.get("run_cfg") or {}
        rna_path = paths.get("rna_path")
        protein_path = paths.get("protein_path")
        if rna_path:
            try:
                prefix = preprocessed_cache_prefix(
                    rna_path,
                    protein_path=protein_path,
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
            except OSError as exc:
                print(f"[lineage] cannot hash sources for {run}: {exc}")
                return None
            candidate = Path(f"{prefix}.protein.h5ad")
            if candidate.exists():
                return candidate
            print(f"[lineage] hashed protein cache missing: {candidate}")
            return None
    return unique_protein_cache_for_dataset(dataset_name_of(run))


def unique_protein_cache_for_dataset(dataset: str | None) -> Path | None:
    """The unique preprocessed protein cache whose stem matches datasets.yaml.

    Several hashes can share a stem. Refusing then is the point: mtime is not
    an identity.
    """
    if not dataset:
        return None
    if not DATASETS_YAML.exists():
        return None
    datasets = yaml.safe_load(DATASETS_YAML.read_text()).get("datasets", {})
    spec = datasets.get(dataset) or {}
    rna_path = spec.get("rna_path")
    if not rna_path:
        return None
    stem = Path(rna_path).stem
    hits = sorted(PREPROCESSED_DIR.glob(f"{stem}_*.protein.h5ad"))
    if len(hits) == 1:
        return hits[0]
    print(f"[lineage] {dataset}: {len(hits)} protein caches match {stem}_*; "
          "not choosing by modification time")
    return None


def dataset_name_of(run: Path) -> str | None:
    payload = run_config_payload(run)
    if payload and payload.get("dataset"):
        return str(payload["dataset"])
    diagnostics = run / "diagnostics.json"
    if diagnostics.exists():
        return json.loads(diagnostics.read_text()).get("dataset")
    return None


def lineages_for_run(run: Path, n_aligned: int) -> pd.Series | None:
    """Coarse lineage per aligned row, or None with the reason printed."""
    cache = protein_cache_from_run(run)
    if cache is None:
        print(f"[lineage] {run}: no protein cache keyed to this run")
        return None
    obs = read_obs_frame(cache, ["cell_type"])
    payload = run_config_payload(run)
    fit_obs = obs
    rna_cache = cache.with_name(cache.name.replace(".protein.h5ad", ".rna.h5ad"))
    if rna_cache.exists():
        rna_names = read_obs_names(rna_cache)
        if len(rna_names) != len(obs) or not np.array_equal(rna_names, obs.index.to_numpy()):
            print(f"[lineage] {run.name}: RNA and protein cache barcodes differ")
            return None
        stratify = None if payload is None else (payload.get("run_cfg") or {}).get("val_stratify_by")
        fit_obs = read_obs_frame(rna_cache, [stratify] if stratify else [])
    fit_rows = None
    if payload is not None:
        cfg = payload.get("run_cfg") or {}
        fit_rows = fitted_row_indices(fit_obs, cfg, cfg.get("run_seed", cfg.get("seed")))
    positions = aligned_cell_positions(n_aligned, len(obs), fit_rows)
    if positions is None:
        n_fit = 0 if fit_rows is None else len(fit_rows)
        print(f"[lineage] {run.name}: aligned n={n_aligned} matches neither "
              f"cache n={len(obs)} nor fitted n={n_fit}")
        return None
    return pd.Series(obs["cell_type"].to_numpy()[positions],
                     dtype=str).map(lineage).reset_index(drop=True)
