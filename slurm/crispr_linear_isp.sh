#!/bin/bash
#SBATCH --account=core-med1-telem
#SBATCH --partition=jobs-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=crispr_linear_isp
#SBATCH --output=logs/crispr_linear_isp_%A_%a.log
#SBATCH --array=0-2%3
# Submit: sbatch slurm/crispr_linear_isp.sh FROZEN_KOT_RUN OUTPUT_ROOT
# Evaluates seen knockouts in held-out replicates, not unseen knockout genes.
set -euo pipefail
# SLURM executes a spool copy, so use the submission directory in a batch allocation.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    cd "$SLURM_SUBMIT_DIR"
else
    cd "$(dirname "$0")/.."
fi
run_dir="${1:?Provide frozen NT-only KOT run directory}"
out_root="${2:?Provide a new output root}"
seeds=(42 123 2026)
seed="${seeds[${SLURM_ARRAY_TASK_ID:-0}]}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
singularity exec --pwd "$PWD" /data/common/images/codedev_v1.0.5.sif \
    python -u run_crispr_isp.py linear --run-dir "$run_dir" --seed "$seed" \
    --out-dir "$out_root/seed_$seed"
