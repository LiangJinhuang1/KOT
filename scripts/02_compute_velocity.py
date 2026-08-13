"""
Stage 2 — RNA Velocity
Runs RNA velocity for every dataset listed in VELOCITY_DATASETS.
Skips datasets whose configured result h5ad already exists.
"""
from src.data.velocity import result_path_for_dataset, run_dataset as run_velocity

VELOCITY_DATASETS = [
    # "bmmc_cite",
    # "bmmc_multiome",
    # "pbmc",
    # "shareseq_skin",
    # "shareseq_brain",
    "shareseq_lung",
]


for name in VELOCITY_DATASETS:
    if result_path_for_dataset(name).exists():
        print(f"[02_compute_velocity] Skipping {name} — result already exists.")
    else:
        print(f"[02_compute_velocity] Running {name}")
        run_velocity(name)
