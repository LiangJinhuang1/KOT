from pathlib import Path

import anndata as ad

from src.data.synthetic import DEFAULT_SYNTHETIC_SEED, save_synthetic
from src.utils.io import load_yaml
from src.data.velocity import run_dataset as run_velocity
from src.training.runner import run_training
from src.visualization.modalities import plot_modalities

TRAINING_CONFIG = Path("config/training.yaml")
SYNTHETIC_RNA = Path("cache/velocity/synthetic/synthetic_scvelo_results.h5ad")
SYNTHETIC_PROTEIN = Path("cache/synthetic/protein.h5ad")

VELOCITY_DATASETS = [
    # "pbmc",
    # "bmmc_cite",
    # "bmmc_multiome",
    # "shareseq_skin",
    # "shareseq_brain",
    # "shareseq_lung",
]


def velocity_result(name: str) -> Path:
    if name.endswith("_retained"):
        base = name.removesuffix("_retained")
        return Path("cache/velocity") / base / f"{base}_scvelo_results_retained.h5ad"
    return Path("cache/velocity") / name / f"{name}_scvelo_results.h5ad"


def synthetic_seed() -> int:
    defaults = load_yaml(TRAINING_CONFIG).get("defaults", {})
    return int(defaults.get("synthetic_seed", DEFAULT_SYNTHETIC_SEED))


if __name__ == "__main__":
    seed = synthetic_seed()
    if not SYNTHETIC_RNA.exists() or not SYNTHETIC_PROTEIN.exists():
        print(f"[synthetic] Creating cached synthetic data with seed={seed}")
        save_synthetic(seed=seed)
    else:
        print(f"[synthetic] Using cached synthetic data: {SYNTHETIC_RNA}")

    rna_adata = ad.read_h5ad(SYNTHETIC_RNA)
    protein_adata = ad.read_h5ad(SYNTHETIC_PROTEIN)
    plot_modalities(rna_adata, protein_adata, save_dir="cache/results/synthetic")

    for name in VELOCITY_DATASETS:
        if velocity_result(name).exists():
            print(f"[velocity] Skipping {name} — result already exists.")
        else:
            run_velocity(name)

    run_training()
