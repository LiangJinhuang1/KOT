#!/bin/bash
#SBATCH --account=core-med1-telem
#SBATCH --partition=jobs-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=128GB
#SBATCH --time=02:00:00
#SBATCH --job-name=vel_backend_cos
#SBATCH --output=logs/slurm%j.out
#SBATCH --error=logs/slurm%j.err
#
# scVelo-vs-RegVelo per-cell velocity cosine under each KOT gene mask.
#
# WHY SLURM AND NOT THE LOGIN NODE. compute_velocity_backend_agreement reads the
# comparison h5ad whole, and bmmc_cite_regvelo_results_retained.h5ad is 4.7 GB against
# ~7 GB free on the login node. No GPU is used -- this is a numpy einsum over the shared
# genes -- so the request is memory and nothing else.
#
# Usage: sbatch slurm/velocity_backend_cosine.sh [pairs]
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

PAIRS="${1:-bmmc_cite_retained:bmmc_cite_regvelo}"
OUT="${2:-cache/results/velocity_backend_cosine_bmmc.csv}"

singularity exec --pwd "$(pwd)" /data/common/images/codedev_v1.0.5.sif /bin/bash -lc \
  "PYTHONPATH=. MPLBACKEND=Agg python -u tools/velocity_backend_cosine.py \
     --pairs '${PAIRS}' --output '${OUT}'"
