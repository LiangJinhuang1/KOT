#!/bin/bash
#
# Usage:
#   sbatch slurm/train_slurm.sh
#   sbatch --export=ALL,RUN_CMD='PYTHONPATH=. python -u -m src.training.runner ...' slurm/train_slurm.sh
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
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=128GB
#SBATCH --time=07:00:00
#SBATCH --job-name=scot_synthetic
#SBATCH --requeue
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

CUDA_PREFLIGHT_REQUEUE="${CUDA_PREFLIGHT_REQUEUE:-1}"
CUDA_PREFLIGHT_MAX_REQUEUES="${CUDA_PREFLIGHT_MAX_REQUEUES:-12}"
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
srun --cpu-bind=none singularity exec --nv --pwd "$(pwd)" "${CONTAINER}" /bin/bash -lc '
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
python - <<'"'"'PY'"'"' >&2
import os
import re
import subprocess
import sys
import torch

print("torch:", torch.__version__, "cuda build:", torch.version.cuda)
print("cuda env:", os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"))
try:
    smi = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,compute_cap,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    gpu_info = smi.stdout.strip() or smi.stderr.strip()
    print("nvidia-smi query:", gpu_info or "<empty>")
except Exception as exc:
    gpu_info = ""
    print("nvidia-smi query failed:", repr(exc))

def parse_cuda_version(version):
    if not version:
        return ()
    match = re.match(r"^(\d+)\.(\d+)", str(version))
    return tuple(map(int, match.groups())) if match else ()

if "B200" in gpu_info and parse_cuda_version(torch.version.cuda) < (12, 8):
    print(
        "ERROR: B200/Blackwell requires a CUDA 12.8+ PyTorch runtime; "
        f"this container has torch.version.cuda={torch.version.cuda!r}. "
        "Use a newer CUDA 12.8+ / PyTorch container.",
        file=sys.stderr,
    )
    sys.exit(42)

try:
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count()
    print("cuda available:", cuda_available)
    print("cuda device_count:", device_count)
    if not cuda_available or device_count < 1:
        print(
            "ERROR: GPU was requested but PyTorch cannot initialize CUDA. "
            "Requeue for another B200 allocation, fix the B200 node, or set "
            "SKIP_CUDA_PREFLIGHT=1 only for CPU-only jobs.",
            file=sys.stderr,
        )
        sys.exit(44)
    torch.empty(1, device="cuda")
    print("cuda smoke allocation: ok")
except Exception as exc:
    print("CUDA preflight failed:", repr(exc))
    print("ERROR: refusing to start RUN_CMD with broken CUDA.", file=sys.stderr)
    sys.exit(45)
PY
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
        exit 0
      fi
      echo "CUDA preflight still failed after ${restart_count} restarts; giving up." >&2
    fi
  fi

  exit "${run_status}"
fi

echo "End time: $(date)"
