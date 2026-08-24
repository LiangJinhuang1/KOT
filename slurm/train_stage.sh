#!/bin/bash
#
# Staged synthetic training. One job = one (stage, scale) setting.
# Usage:
#   sbatch train_stage.sh
#   sbatch --export=ALL,STAGE=clean,SCALE=log   train_stage.sh
#   sbatch --export=ALL,STAGE=branch,SCALE=mean train_stage.sh
#   sbatch --export=ALL,CONFIG=config/training_full_sweep.yaml,STAGE=oracle,SCALE=mean train_stage.sh
#   sbatch --export=ALL,CONFIG=config/training_full_sweep.yaml,STAGE=clean,SCALE=mean train_stage.sh
#
# Generate the data first (once per stage; produces both mean and log layers):
#   python -m src.data.synthetic_linked_ode --stage oracle
#   python -m src.data.synthetic_linked_ode --stage clean
#   python -m src.data.synthetic_linked_ode --stage branch
#
#SBATCH --account=core-med1-telem
#SBATCH --partition=jobs-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=128GB
#SBATCH --time=6:00:00
#SBATCH --job-name=kot_stage
#SBATCH --output=logs/slurm%j.log
#SBATCH --error=logs/slurm%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

STAGE="${STAGE:-clean}"
SCALE="${SCALE:-mean}"
CONFIG="${CONFIG:-config/training.yaml}"
MODELS="${MODELS:-}"            # model group or comma-list; empty = config default_models
RUN_DIR="${RUN_DIR:-}"          # set to an existing run_* dir to resume (run only missing parts)

echo "===== Slurm Job Info ====="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Workdir: $(pwd)"
echo "Config: ${CONFIG}"
echo "Stage: ${STAGE}  Scale: ${SCALE}  Models: ${MODELS:-<default>}  ResumeDir: ${RUN_DIR:-<new>}"
echo "Start time: $(date)"

CONTAINER="/data/common/images/codedev_v1.0.5.sif"
RUN_CMD="python -u -m src.training.runner --config ${CONFIG} --stage ${STAGE} --scale ${SCALE}"
if [ -n "${MODELS}" ]; then
  RUN_CMD="${RUN_CMD} --models ${MODELS}"
fi
if [ -n "${RUN_DIR}" ]; then
  RUN_CMD="${RUN_CMD} --run-dir ${RUN_DIR}"
fi
export RUN_CMD
export PREPROCESS_THREADS="${SLURM_CPUS_PER_TASK:-8}"

srun --cpu-bind=none singularity exec --nv --pwd "$(pwd)" "${CONTAINER}" /bin/bash -lc '
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${PREPROCESS_THREADS}"
export OPENBLAS_NUM_THREADS="${PREPROCESS_THREADS}"
export MKL_NUM_THREADS="${PREPROCESS_THREADS}"
export NUMEXPR_NUM_THREADS="${PREPROCESS_THREADS}"
export VECLIB_MAXIMUM_THREADS="${PREPROCESS_THREADS}"
echo "Inside container on $(hostname)"
echo "Run command: ${RUN_CMD}"
eval "${RUN_CMD}"
'

echo "End time: $(date)"
