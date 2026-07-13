"""
Stage 2 — RNA Velocity
Runs scVelo for every dataset listed in VELOCITY_DATASETS.
Skips datasets whose _scvelo_results.h5ad already exists.
"""
from pathlib import Path
from src.data.velocity import run_dataset as run_velocity

VELOCITY_DATASETS = [
    # "bmmc_cite",
    # "bmmc_multiome",
    # "pbmc",
    # "shareseq_skin",
    # "shareseq_brain",
    "shareseq_lung",
]


def velocity_result(name: str) -> Path:
    return Path("cache/velocity") / name / f"{name}_scvelo_results.h5ad"


for name in VELOCITY_DATASETS:
    if velocity_result(name).exists():
        print(f"[02_compute_velocity] Skipping {name} — result already exists.")
    else:
        print(f"[02_compute_velocity] Running {name}")
        run_velocity(name)
