#!/bin/bash
#
# THE single-job entry point for this repo: it runs any RUN_CMD inside the project
# container. Per-tool wrapper scripts are not needed -- slurm/README.md carries the exact
# command for each one, and slurm/parallel_train.sh is the many-runs-at-once counterpart.
#
# Usage:
#   sbatch slurm/train_slurm.sh
#   sbatch --export=ALL,RUN_CMD='PYTHONPATH=. python -u -m src.training.runner ...' slurm/train_slurm.sh
#
# GPU is used when the allocation has one and skipped when it does not, so a CPU-only tool
# submits through the same script:
#   sbatch --partition=jobs-cpu --gres=none --mem=96GB --time=08:00:00 \
#     --export=ALL,RUN_CMD='...' slurm/train_slurm.sh
#
# GPU (cluster requires an exact model — generic gpu:1 is rejected):
#   gpu:b200:1       → srvcore3
#   gpu:a100:1       → srvdgx1
#   gpu:a100_10gb:1  → srvdrai2
# Override at submit time: sbatch --gres=gpu:a100:1 slurm/train_slurm.sh
# Requeue on broken CUDA init: CUDA_PREFLIGHT_REQUEUE=1 CUDA_PREFLIGHT_MAX_REQUEUES=12
#
#SBATCH --account=core-med1-telem
#SBATCH --partition=jobs-gpu
#SBATCH --gres=gpu:b200:1
# --cpus-per-task, not --cpus-per-gpu: the two are mutually exclusive at submit time, so
# the per-gpu form makes every `sbatch --cpus-per-task=N ... train_slurm.sh` a fatal error.
# It also leaves SLURM_CPUS_PER_TASK unset, which silently drops the thread count below to 1.
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=07:00:00
#SBATCH --job-name=scot_synthetic
#SBATCH --requeue
# Requeued attempts reopen these files; without append, each retry TRUNCATES the
# log and the record of how many times CUDA failed, and when, is destroyed.
#SBATCH --open-mode=append
#SBATCH --output=logs/slurm%j.log
#SBATCH --error=logs/slurm%j.err

set -euo pipefail

# 让脚本从你提交作业时所在目录运行
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

echo "===== Slurm Job Info ====="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Workdir: $(pwd)"
echo "Start time: $(date)"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK:-1}"
[ -f main.py ] || { echo "ERROR: main.py not found in $(pwd)"; exit 1; }

# GPU if the allocation has one, CPU otherwise. This script is the entry point for every
# job in the repo, including numpy-only tools submitted with --gres=none, so it must not
# demand a device they never asked for: without a GPU both `singularity --nv` and the CUDA
# preflight are skipped rather than failing the job.
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
# Seconds to hold a requeued job before it may start again. Error 802 (fabric manager
# down) does not clear in the seconds an immediate requeue takes, so an undelayed loop
# spends all 12 attempts inside a few minutes and gives up while the fault is still
# being fixed. Spacing them turns the same 12 into ~2 hours of coverage. 0 = immediate.
CUDA_REQUEUE_DELAY_SEC="${CUDA_REQUEUE_DELAY_SEC:-600}"
echo "CUDA preflight requeue: ${CUDA_PREFLIGHT_REQUEUE} (max ${CUDA_PREFLIGHT_MAX_REQUEUES}, current restart ${SLURM_RESTART_COUNT:-0})"

# Preserve Slurm's GPU mask inside real allocations. Clearing it here can make
# Singularity/PyTorch see an inconsistent device view on multi-GPU nodes.
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

# 你指定使用的容器镜像
CONTAINER="${CONTAINER:-/data/common/images/codedev_v1.0.5.sif}"
export CONTAINER

# 默认运行命令：调用仓库根目录下的 main.py 做批量预处理测试（无缓冲输出）
RUN_CMD="${RUN_CMD:-python -u main.py}"
export RUN_CMD
export PREPROCESS_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# 通过 srun + singularity 启动训练（CPU 预处理）
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
# Keep JIT / library caches off the (quota-limited) home dir — write to project space.
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
  # Shared with slurm/parallel_train.sh: one definition of "is this allocation
  # usable", one set of exit codes for the requeue branch below.
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

  # Exit codes from the CUDA preflight only. Requeueing releases this bad B200
  # allocation and lets Slurm try again later, without hiding real RUN_CMD errors.
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
