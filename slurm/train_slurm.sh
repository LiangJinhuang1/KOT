#!/bin/bash
#
# Single-job entry point: run any RUN_CMD in the project container.
#
# Usage:
#   sbatch slurm/train_slurm.sh
#   sbatch --export=ALL,RUN_CMD='PYTHONPATH=. python -u -m src.training.runner ...' slurm/train_slurm.sh
#
# GPU if allocated, else CPU, so numpy-only jobs work:
#   sbatch --partition=jobs-cpu --gres=none --mem=96GB --time=08:00:00 \
#     --export=ALL,RUN_CMD='...' slurm/train_slurm.sh
#
# Cluster requires an exact GPU model (generic gpu:1 is rejected):
#   gpu:b200:1       → srvcore3
#   gpu:a100:1       → srvdgx1
#   gpu:a100_10gb:1  → srvdrai2
#
#SBATCH --account=core-med1-telem
#SBATCH --partition=jobs-gpu
#SBATCH --gres=gpu:b200:1
# --cpus-per-task, not --cpus-per-gpu: the two are mutually exclusive, and the per-gpu form leaves SLURM_CPUS_PER_TASK unset.
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=07:00:00
#SBATCH --job-name=scot_synthetic
#SBATCH --requeue
# Requeued attempts reopen these files; without append each retry truncates the log.
#SBATCH --open-mode=append
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

# GPU if allocated else CPU so numpy-only jobs work; without a GPU skip --nv and the CUDA preflight.
if [ -n "${SLURM_JOB_GPUS:-}${SLURM_GPUS_ON_NODE:-}" ]; then
  NV_FLAG="--nv"
  echo "GPU allocation: ${SLURM_JOB_GPUS:-${SLURM_GPUS_ON_NODE}}"
else
  NV_FLAG=""
  export SKIP_CUDA_PREFLIGHT=1
  echo "No GPU in this allocation: running CPU-only, CUDA preflight skipped."
fi

CUDA_PREFLIGHT_REQUEUE="${CUDA_PREFLIGHT_REQUEUE:-1}"
CUDA_PREFLIGHT_MAX_REQUEUES="${CUDA_PREFLIGHT_MAX_REQUEUES:-12}"
# Fabric-manager errors do not clear on immediate requeue. 0 = immediate.
CUDA_REQUEUE_DELAY_SEC="${CUDA_REQUEUE_DELAY_SEC:-600}"
echo "CUDA preflight requeue: ${CUDA_PREFLIGHT_REQUEUE} (max ${CUDA_PREFLIGHT_MAX_REQUEUES}, current restart ${SLURM_RESTART_COUNT:-0})"

# Preserve Slurm's GPU mask; clearing it can give Singularity/PyTorch an inconsistent device view.
if [ -z "${SLURM_JOB_ID:-}" ]; then
  unset CUDA_VISIBLE_DEVICES
  unset NVIDIA_VISIBLE_DEVICES
fi
echo "Batch CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Batch SLURM_JOB_GPUS: ${SLURM_JOB_GPUS:-<unset>}"
if [ "${FORCE_CONTAINER_SLURM_GPU_IDS:-0}" = "1" ] && [ -n "${SLURM_JOB_GPUS:-}" ]; then
  export SINGULARITYENV_CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"
  export APPTAINERENV_CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"
  export SINGULARITYENV_NVIDIA_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"
  export APPTAINERENV_NVIDIA_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"
  echo "Container CUDA mask override: ${SLURM_JOB_GPUS}"
fi
if [ -n "${CONTAINER_CUDA_VISIBLE_DEVICES:-}" ]; then
  export SINGULARITYENV_CUDA_VISIBLE_DEVICES="${CONTAINER_CUDA_VISIBLE_DEVICES}"
  export APPTAINERENV_CUDA_VISIBLE_DEVICES="${CONTAINER_CUDA_VISIBLE_DEVICES}"
  export SINGULARITYENV_NVIDIA_VISIBLE_DEVICES="${CONTAINER_CUDA_VISIBLE_DEVICES}"
  export APPTAINERENV_NVIDIA_VISIBLE_DEVICES="${CONTAINER_CUDA_VISIBLE_DEVICES}"
  echo "Container CUDA mask override: ${CONTAINER_CUDA_VISIBLE_DEVICES}"
fi

CONTAINER="${CONTAINER:-/data/common/images/codedev_v1.0.5.sif}"
export CONTAINER

RUN_CMD="${RUN_CMD:-python -u main.py}"
export RUN_CMD
export PREPROCESS_THREADS="${SLURM_CPUS_PER_TASK:-1}"

set +e
srun --cpu-bind=none singularity exec ${NV_FLAG} --pwd "$(pwd)" "${CONTAINER}" /bin/bash -lc '
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${PREPROCESS_THREADS}"
export OPENBLAS_NUM_THREADS="${PREPROCESS_THREADS}"
export MKL_NUM_THREADS="${PREPROCESS_THREADS}"
export NUMEXPR_NUM_THREADS="${PREPROCESS_THREADS}"
export VECLIB_MAXIMUM_THREADS="${PREPROCESS_THREADS}"
# Keep JIT / library caches off the home quota; write to project space.
export CACHE_ROOT="$(pwd)/cache/.env_cache"
export NUMBA_CACHE_DIR="${CACHE_ROOT}/numba"
export MPLCONFIGDIR="${CACHE_ROOT}/mpl"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export NUMBA_CACHE_DIR MPLCONFIGDIR XDG_CACHE_HOME
mkdir -p "${NUMBA_CACHE_DIR}" "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"
echo "Inside container on $(hostname)" >&2
echo "Container: ${SINGULARITY_CONTAINER:-unknown}" >&2
echo "Threads: ${PREPROCESS_THREADS}" >&2
echo "PYTORCH_ALLOC_CONF: ${PYTORCH_ALLOC_CONF}" >&2
echo "SLURM_JOB_GPUS: ${SLURM_JOB_GPUS:-<unset>}" >&2
echo "SLURM_STEP_GPUS: ${SLURM_STEP_GPUS:-<unset>}" >&2
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}" >&2
nvidia-smi -L >&2 || true
python -c "import os; print(\"python:\", os.sys.executable); print(\"cwd:\", os.getcwd())" >&2
if [ "${SKIP_CUDA_PREFLIGHT:-0}" != "1" ]; then
  python slurm/cuda_preflight.py >&2 || exit $?
else
echo "Skipping CUDA preflight because SKIP_CUDA_PREFLIGHT=1" >&2
fi
echo "Run command: ${RUN_CMD}"
eval "${RUN_CMD}"
'
run_status=$?
set -e

if [ "${run_status}" -ne 0 ]; then
  echo "Container command exited with status ${run_status}" >&2

  # Requeue only preflight codes; do not hide a real RUN_CMD error.
  if [ "${SKIP_CUDA_PREFLIGHT:-0}" != "1" ] && [ "${CUDA_PREFLIGHT_REQUEUE}" = "1" ]; then
    restart_count="${SLURM_RESTART_COUNT:-0}"
    if [ "${run_status}" = "43" ] || [ "${run_status}" = "44" ] || [ "${run_status}" = "45" ]; then
      if [ "${restart_count}" -lt "${CUDA_PREFLIGHT_MAX_REQUEUES}" ]; then
        echo "CUDA preflight failed on $(hostname); requeueing job ${SLURM_JOB_ID} (${restart_count}/${CUDA_PREFLIGHT_MAX_REQUEUES})." >&2
        scontrol requeue "${SLURM_JOB_ID}"
        if [ "${CUDA_REQUEUE_DELAY_SEC}" -gt 0 ]; then
          scontrol update JobId="${SLURM_JOB_ID}" StartTime=now+"${CUDA_REQUEUE_DELAY_SEC}" || true
          echo "Held until now+${CUDA_REQUEUE_DELAY_SEC}s so the node has time to recover." >&2
        fi
        exit 0
      fi
      echo "CUDA preflight still failed after ${restart_count} restarts; giving up." >&2
    fi
  fi

  exit "${run_status}"
fi

echo "End time: $(date)"
