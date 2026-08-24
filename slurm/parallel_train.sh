#!/bin/bash
#
# Pack many small training runs onto the allocated GPUs. Each KOT run peaks at ~4-5 GB,
# so a 183 GB B200 fits 20-30 concurrently; the real limits are host RAM (each run loads
# its h5ad) and CPU cores (preprocessing + per-batch Python), so this asks for more of
# both than train_slurm.sh and caps concurrency with MAX_PARALLEL.
#
# Jobs are pinned round-robin across every visible GPU via CUDA_VISIBLE_DEVICES, so the
# script adapts to whatever --gres gives it: raise or lower the gres count and the
# distribution follows. MAX_PARALLEL is the TOTAL across GPUs, not per GPU.
#
# Usage:
#   1) Put one runner arg-string per line in a jobs file, e.g. jobs.txt:
#        --datasets pbmc_retained --models kot
#        --datasets pbmc_retained --models scot
#        --datasets bmmc_cite_retained --models kot --set lambda_dyn=1
#   2) sbatch --export=ALL,JOBS_FILE=jobs.txt,MAX_PARALLEL=12 slurm/parallel_train.sh
#      (optional: ,ENABLE_MPS=1 for true concurrent kernels via NVIDIA MPS)
#
# The preprocessing cache write is atomic (src/data/preprocessing.py), so several runs of the
# SAME dataset starting cold is safe — but warming the cache first (run one job per dataset)
# still avoids the redundant recompute.
#
#SBATCH --account=core-med1-telem
#SBATCH --partition=jobs-gpu
#SBATCH --gres=gpu:b200:2
#SBATCH --cpus-per-gpu=32
#SBATCH --mem=256GB
#SBATCH --time=07:00:00
#SBATCH --job-name=kot_parallel
#SBATCH --requeue
#SBATCH --output=logs/slurm%j.log
#SBATCH --error=logs/slurm%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

: "${JOBS_FILE:?set JOBS_FILE=path/to/jobs.txt (one runner arg-string per line)}"
MAX_PARALLEL="${MAX_PARALLEL:-12}"
ENABLE_MPS="${ENABLE_MPS:-0}"
AUTO_GPU_THROTTLE="${AUTO_GPU_THROTTLE:-1}"
MIN_FREE_GPU_MB="${MIN_FREE_GPU_MB:-18000}"
GPU_POLL_SEC="${GPU_POLL_SEC:-5}"
[ -f "${JOBS_FILE}" ] || { echo "JOBS_FILE not found: ${JOBS_FILE}"; exit 1; }

echo "===== Parallel training ====="
echo "Node: $(hostname) | JOBS_FILE=${JOBS_FILE} | MAX_PARALLEL=${MAX_PARALLEL} | MPS=${ENABLE_MPS} | AUTO_GPU_THROTTLE=${AUTO_GPU_THROTTLE} | MIN_FREE_GPU_MB=${MIN_FREE_GPU_MB}"
echo "Jobs: $(grep -cve '^[[:space:]]*$' "${JOBS_FILE}")"

# One stamp for the whole sweep, fixed at job start. jobs.txt carries the literal
# token @STAMP@ in its --run-dir values and this is what replaces it, so a jobs
# file generated weeks ago still lands in a directory named for when it ran.
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
echo "Run stamp: ${RUN_STAMP}  (run dirs: cache/training/run_${RUN_STAMP}_*)"

CONTAINER="${CONTAINER:-/data/common/images/codedev_v1.0.5.sif}"
export CONTAINER JOBS_FILE MAX_PARALLEL ENABLE_MPS RUN_STAMP
export AUTO_GPU_THROTTLE MIN_FREE_GPU_MB GPU_POLL_SEC
# --cpus-per-gpu does not set SLURM_CPUS_PER_TASK, so that alone would silently
# fall back to 8 and leave most of the allocation idle — more so now that two GPUs
# means twice the cores. CPUS_ON_NODE reflects what was actually granted.
export PREPROCESS_THREADS="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-8}}"
echo "CPU budget: ${PREPROCESS_THREADS} (cpus_per_task=${SLURM_CPUS_PER_TASK:-unset}, cpus_on_node=${SLURM_CPUS_ON_NODE:-unset})"

srun --cpu-bind=none singularity exec --nv --pwd "$(pwd)" "${CONTAINER}" /bin/bash -lc '
set -uo pipefail
export PYTHONUNBUFFERED=1
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export CACHE_ROOT="$(pwd)/cache/.env_cache"
export NUMBA_CACHE_DIR="${CACHE_ROOT}/numba" MPLCONFIGDIR="${CACHE_ROOT}/mpl" XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
mkdir -p "${NUMBA_CACHE_DIR}" "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"
# Keep per-process CPU thread pools small so N processes do not oversubscribe the cores.
THREADS_PER=$(( PREPROCESS_THREADS / MAX_PARALLEL )); [ "${THREADS_PER}" -lt 1 ] && THREADS_PER=1
export OMP_NUM_THREADS="${THREADS_PER}" OPENBLAS_NUM_THREADS="${THREADS_PER}" MKL_NUM_THREADS="${THREADS_PER}"
# Round-robin target for each launched job. Counting the devices rather than trusting
# the gres number keeps this correct if the allocation differs from the request.
N_GPUS="$(nvidia-smi -L 2>/dev/null | grep -c "^GPU" || echo 1)"
[ "${N_GPUS}" -ge 1 ] || N_GPUS=1
echo "Inside container on $(hostname); threads/proc=${THREADS_PER}; GPUs=${N_GPUS}" >&2
nvidia-smi -L >&2 || true

if [ "${ENABLE_MPS}" = "1" ]; then
  export CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps-${SLURM_JOB_ID}"
  export CUDA_MPS_LOG_DIRECTORY="/tmp/nvidia-mps-log-${SLURM_JOB_ID}"
  mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
  nvidia-cuda-mps-control -d && echo "MPS daemon started" >&2 || echo "MPS start failed; continuing without" >&2
fi

idx=0
running=0
while IFS= read -r job || [ -n "${job}" ]; do
  case "${job}" in ""|\#*) continue ;; esac
  # Resolve the placeholder now, so every line in this sweep shares one stamp.
  job="${job//@STAMP@/${RUN_STAMP}}"

  # Which GPU this job will use; also the one the throttle should be checking.
  gpu=$(( idx % N_GPUS ))

  while true; do
    [ "${running}" -lt "${MAX_PARALLEL}" ] || {
      wait -n
      running=$((running-1))
      continue
    }

    if [ "${AUTO_GPU_THROTTLE}" = "1" ]; then
      free_mb="$(nvidia-smi --id="${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | awk "{print int(\$1); exit}")"
      [ -n "${free_mb}" ] || free_mb=0
      if [ "${free_mb}" -lt "${MIN_FREE_GPU_MB}" ]; then
        if [ "${running}" -gt 0 ]; then
          wait -n
          running=$((running-1))
        else
          echo "[throttle] gpu${gpu} free ${free_mb} MB < ${MIN_FREE_GPU_MB} MB; sleeping ${GPU_POLL_SEC}s" >&2
          sleep "${GPU_POLL_SEC}"
        fi
        continue
      fi
    fi
    break
  done

  idx=$((idx+1))
  log="logs/par_${SLURM_JOB_ID}_$(printf %03d ${idx}).log"
  if [ "${AUTO_GPU_THROTTLE}" = "1" ]; then
    echo "[launch ${idx}] gpu${gpu} free_mb=${free_mb} ${job} -> ${log}" >&2
  else
    echo "[launch ${idx}] gpu${gpu} ${job} -> ${log}" >&2
  fi
  ( CUDA_VISIBLE_DEVICES="${gpu}" python -u -m src.training.runner ${job} ) > "${log}" 2>&1 &
  running=$((running+1))
done < "${JOBS_FILE}"
wait
echo "all ${idx} jobs finished" >&2

if [ "${ENABLE_MPS}" = "1" ]; then echo quit | nvidia-cuda-mps-control || true; fi
'
echo "End time: $(date)"
