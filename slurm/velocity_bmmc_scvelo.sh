#!/bin/bash
#SBATCH --account=core-med1-telem
#SBATCH --partition=jobs-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=08:00:00
#SBATCH --job-name=vel_bmmc_scv
#SBATCH --output=logs/velocity_%j.log
#SBATCH --error=logs/velocity_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
CONTAINER="${CONTAINER:-/data/common/images/codedev_v1.0.5.sif}"
singularity exec --pwd "$(pwd)" "${CONTAINER}" /bin/bash -lc \
  "python -m src.data.velocity bmmc_cite_retained --dynamics-n-jobs 16"
