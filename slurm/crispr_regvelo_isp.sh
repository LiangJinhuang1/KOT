#!/bin/bash
#SBATCH --account=core-med1-telem
#SBATCH --partition=jobs-gpu-long
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=crispr_regvelo_isp
#SBATCH --output=logs/crispr_regvelo_isp_%A_%a.log
#SBATCH --array=0-2%3
# Submit: sbatch slurm/crispr_regvelo_isp.sh FROZEN_KOT_RUN NEW_OUTPUT_ROOT
# Uses /data/common/images/codedev_v1.0.5.sif, where RegVelo is already installed.
# RegVelo and every learned preprocessing statistic see NT cells only.
# Its published regulon-KO definition silences outgoing TF edges; non-TF knockouts
# without learned outgoing edges are excluded and written to coverage.csv.
# Counterfactual RNA is converted back into frozen KOT Ms coordinates, then scored at
# RNA level and through the unchanged NT-only KOT checkpoint.
set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then cd "$SLURM_SUBMIT_DIR"; else cd "$(dirname "$0")/.."; fi
run_dir="${1:?Provide frozen NT-only KOT run directory}"
out_root="${2:?Provide a new output root}"
seeds=(42 123 2026)
seed="${seeds[${SLURM_ARRAY_TASK_ID:-0}]}"
fold="$out_root/seed_$seed"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
container=/data/common/images/codedev_v1.0.5.sif
singularity exec --nv --pwd "$PWD" "$container" python -u run_crispr_isp.py regvelo \
    --run-dir "$run_dir" --seed "$seed" --out-dir "$fold" \
    --max-epochs "${REGVELO_MAX_EPOCHS:-500}" \
    --batch-size "${REGVELO_BATCH_SIZE:-2048}" \
    --n-samples "${REGVELO_N_SAMPLES:-1}"
singularity exec --nv --pwd "$PWD" "$container" python -u run_kot_crispr.py task_b \
    --predicted-rna "$fold/predicted_rna.csv" --isp-model regvelo \
    --run-dir "$run_dir" --seed "$seed" --checkpoint best_align --out-dir "$fold"
