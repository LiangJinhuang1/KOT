#!/bin/bash
#
# Hyperparameter sweep V3 — alternating phase optimization + fixed kappa.
#
# Submit ONCE with: sbatch sweep_slurm_v3.sh
#
# WHEN TO USE THIS:
#   Only after looking at v2 results. If kot_fixedkappa beat kot_nodyn, this is
#   worth running: it combines the collapse-prevention (fixed κ) with a
#   curriculum (Sinkhorn → Dynamics → Joint). If kot_fixedkappa was flat vs
#   kot_nodyn, do NOT run this — pivot to mean-scale coordinates instead.
#
# PRE-REQUISITE (one-time):
#   kot.py must support phase config. See instructions in the message that
#   accompanied this file. The keys to read are:
#       phase1_epochs, phase2_epochs, phase3_epochs, phase3_lr_scale
#   If these keys are missing from kot.py, this sweep will silently fall back
#   to joint training and v3 will look identical to v2.
#
# WHAT THIS SWEEPS:
#   phase3_lr_scale ∈ {0.01, 0.1, 0.5}
#   With phase split fixed at 200 / 100 / 200, kot_fixed_kappa = ln(2).
#   3 sweeps × 12 seeds × 1 model = 36 trainings.
#
# DO NOT run in parallel with sweep_slurm.sh / sweep_slurm_v2.sh — all three
# sed-edit config/training.yaml.
#
#SBATCH --account=core-med1-telem
#SBATCH --partition=jobs-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64GB
#SBATCH --time=07:00:00
#SBATCH --job-name=kot_sweep_v3
#SBATCH --output=logs/slurm%j.log
#SBATCH --error=logs/slurm%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

echo "===== Slurm Job Info ====="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Workdir: $(pwd)"
echo "Start time: $(date)"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK:-1}"
[ -f main.py ] || { echo "ERROR: main.py not found in $(pwd)"; exit 1; }
[ -f config/training.yaml ] || { echo "ERROR: config/training.yaml not found"; exit 1; }

CONTAINER="/data/common/images/codedev_v1.0.5.sif"

export PREPROCESS_THREADS="${SLURM_CPUS_PER_TASK:-1}"

srun --cpu-bind=none singularity exec --nv --pwd "$(pwd)" "${CONTAINER}" /bin/bash -lc '
set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${PREPROCESS_THREADS}"
export OPENBLAS_NUM_THREADS="${PREPROCESS_THREADS}"
export MKL_NUM_THREADS="${PREPROCESS_THREADS}"
export NUMEXPR_NUM_THREADS="${PREPROCESS_THREADS}"
export VECLIB_MAXIMUM_THREADS="${PREPROCESS_THREADS}"
echo "Inside container on $(hostname)"

# ----------------------------------------------------------------------------
# Phase budget — kept fixed across this sweep. Tweak here if you want.
# ----------------------------------------------------------------------------
sed -i "s/^  phase1_epochs:.*/  phase1_epochs: 200/" config/training.yaml
sed -i "s/^  phase2_epochs:.*/  phase2_epochs: 100/" config/training.yaml
sed -i "s/^  phase3_epochs:.*/  phase3_epochs: 200/" config/training.yaml

# ----------------------------------------------------------------------------
# Hyperparameter grid — only phase3_lr_scale varies in v3.
# ----------------------------------------------------------------------------
PHASE3_LR_SCALE_VALUES=(0.01 0.1 0.5)

run_count=0
total_runs=${#PHASE3_LR_SCALE_VALUES[@]}

for lr_scale in "${PHASE3_LR_SCALE_VALUES[@]}"; do
  run_count=$(( run_count + 1 ))
  echo ""
  echo "================================================================"
  echo "  Sweep v3 run ${run_count}/${total_runs}: phase3_lr_scale=${lr_scale}"
  echo "  $(date)"
  echo "================================================================"
  sed -i "s/^  phase3_lr_scale:.*/  phase3_lr_scale: ${lr_scale}/" config/training.yaml
  python -u main.py
done

echo ""
echo "Sweep v3 complete: ${run_count} runs."
echo "Results in cache/training/run_<timestamp>/"
'

echo "End time: $(date)"
