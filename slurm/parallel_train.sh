#!/bin/bash
#
# Pack many training runs onto the allocated GPUs. Host RAM and CPU cores, not
# GPU memory, are the real limits, so this asks for more of both than
# train_slurm.sh and caps concurrency with MAX_PARALLEL.
#
# Round-robin across visible GPUs; MAX_PARALLEL is the total, not per GPU.
#
# Usage:
#   1) Put one runner arg-string per line in a jobs file, e.g. jobs.txt:
#        --datasets pbmc_retained --models kot
#        --datasets pbmc_retained --models scot
#        --datasets bmmc_cite_retained --models kot --set lambda_dyn=1
#   2) sbatch --export=ALL,JOBS_FILE=jobs.txt,MAX_PARALLEL=16 slurm/parallel_train.sh
#      (optional: ,ENABLE_MPS=1 for true concurrent kernels via NVIDIA MPS)
#
# Cache writes are atomic, so cold starts of the same dataset are safe; warming first still avoids redundant recompute.
#
#SBATCH --account=core-med1-telem
#SBATCH --partition=jobs-gpu
# Ask for 3: a stuck fabric-manager card fails CUDA init for the whole allocation; CUDA_VISIBLE_DEVICES does not mask it. Slurm hands out low indices first.
#SBATCH --gres=gpu:b200:3
# 16 matches DefCpuPerGPU; 32/GPU hits the per-user CPU QOS with two 3-GPU jobs.
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=512GB
#SBATCH --time=07:00:00
#SBATCH --job-name=kot_parallel
#SBATCH --requeue
# Requeued attempts reopen these files; without append each retry truncates the log.
#SBATCH --open-mode=append
#SBATCH --output=logs/slurm%j.log
#SBATCH --error=logs/slurm%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

: "${JOBS_FILE:?set JOBS_FILE=path/to/jobs.txt (one runner arg-string per line)}"
MAX_PARALLEL="${MAX_PARALLEL:-16}"
ENABLE_MPS="${ENABLE_MPS:-0}"
AUTO_GPU_THROTTLE="${AUTO_GPU_THROTTLE:-1}"
MIN_FREE_GPU_MB="${MIN_FREE_GPU_MB:-18000}"
GPU_POLL_SEC="${GPU_POLL_SEC:-5}"
# Check CUDA once before any job starts, and hand a broken allocation back instead of feeding every job into it.
CUDA_PREFLIGHT_REQUEUE="${CUDA_PREFLIGHT_REQUEUE:-1}"
CUDA_PREFLIGHT_MAX_REQUEUES="${CUDA_PREFLIGHT_MAX_REQUEUES:-12}"
# Fabric-manager errors do not clear on immediate requeue. 0 = immediate.
CUDA_REQUEUE_DELAY_SEC="${CUDA_REQUEUE_DELAY_SEC:-600}"
[ -f "${JOBS_FILE}" ] || { echo "JOBS_FILE not found: ${JOBS_FILE}"; exit 1; }

echo "===== Parallel training ====="
echo "Node: $(hostname) | JOBS_FILE=${JOBS_FILE} | MAX_PARALLEL=${MAX_PARALLEL} | MPS=${ENABLE_MPS} | AUTO_GPU_THROTTLE=${AUTO_GPU_THROTTLE} | MIN_FREE_GPU_MB=${MIN_FREE_GPU_MB}"
# Count lines starting with -- (the same test the launch loop uses); wrapped comments would otherwise be counted as jobs.
echo "Jobs: $(grep -ce '^--' "${JOBS_FILE}")"

# One stamp for the whole sweep, so a jobs file generated earlier still lands in a directory named for when it ran.
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
echo "Run stamp: ${RUN_STAMP}  (run dirs: cache/training/run_${RUN_STAMP}_*)"

CONTAINER="${CONTAINER:-/data/common/images/codedev_v1.0.5.sif}"
export CONTAINER JOBS_FILE MAX_PARALLEL ENABLE_MPS RUN_STAMP
export AUTO_GPU_THROTTLE MIN_FREE_GPU_MB GPU_POLL_SEC
# --cpus-per-gpu does not set SLURM_CPUS_PER_TASK, so that alone would silently fall back to 8.
export PREPROCESS_THREADS="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-8}}"
echo "CPU budget: ${PREPROCESS_THREADS} (cpus_per_task=${SLURM_CPUS_PER_TASK:-unset}, cpus_on_node=${SLURM_CPUS_ON_NODE:-unset})"
echo "CUDA preflight requeue: ${CUDA_PREFLIGHT_REQUEUE} (max ${CUDA_PREFLIGHT_MAX_REQUEUES}, current restart ${SLURM_RESTART_COUNT:-0})"

set +e
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
# Count devices rather than trusting the gres number if the allocation differs from the request.
N_GPUS="$(nvidia-smi -L 2>/dev/null | grep -c "^GPU" || echo 1)"
[ "${N_GPUS}" -ge 1 ] || N_GPUS=1
echo "Inside container on $(hostname); threads/proc=${THREADS_PER}; GPUs=${N_GPUS}" >&2
nvidia-smi -L >&2 || true

# Before the first job, not once per job, so the requeue branch can see the exit code.
USABLE_FILE="logs/gpu_usable_${SLURM_JOB_ID}.txt"
GPU_IDS=""
if [ "${SKIP_CUDA_PREFLIGHT:-0}" != "1" ]; then
  python slurm/cuda_preflight.py --emit-usable "${USABLE_FILE}" >&2 || exit $?
  GPU_IDS="$(cat "${USABLE_FILE}" 2>/dev/null)"
fi
# No usable list: fall back to every device nvidia-smi listed.
[ -n "${GPU_IDS}" ] || GPU_IDS="$(seq -s, 0 $(( N_GPUS - 1 )))"
IFS="," read -r -a GPU_ARRAY <<< "${GPU_IDS}"
N_GPUS="${#GPU_ARRAY[@]}"
echo "Scheduling across GPUs [${GPU_IDS}] (${N_GPUS} usable)" >&2

if [ "${ENABLE_MPS}" = "1" ]; then
  export CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps-${SLURM_JOB_ID}"
  export CUDA_MPS_LOG_DIRECTORY="/tmp/nvidia-mps-log-${SLURM_JOB_ID}"
  mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
  nvidia-cuda-mps-control -d && echo "MPS daemon started" >&2 || echo "MPS start failed; continuing without" >&2
fi

idx=0
running=0
worker_fail=0
wait_one() {
  wait -n || worker_fail=1
  running=$((running-1))
}
while IFS= read -r job || [ -n "${job}" ]; do
  # A job line starts with --. Matching on that, not skipping # lines, catches a wrapped comment: only its first line carries the #.
  case "${job}" in
    --*) ;;
    ""|\#*) continue ;;
    *) echo "[skip] not a job line: ${job}" >&2; continue ;;
  esac
  # Resolve the placeholder now, so every line in this sweep shares one stamp.
  job="${job//@STAMP@/${RUN_STAMP}}"

  # Index into the usable list so a rejected device is never scheduled.
  gpu="${GPU_ARRAY[$(( idx % N_GPUS ))]}"

  while true; do
    [ "${running}" -lt "${MAX_PARALLEL}" ] || {
      wait_one
      continue
    }

    if [ "${AUTO_GPU_THROTTLE}" = "1" ]; then
      free_mb="$(nvidia-smi --id="${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | awk "{print int(\$1); exit}")"
      [ -n "${free_mb}" ] || free_mb=0
      if [ "${free_mb}" -lt "${MIN_FREE_GPU_MB}" ]; then
        if [ "${running}" -gt 0 ]; then
          wait_one
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
  # moscot/JAX raises if its CUDA plugin is present but cannot initialize. Export inside the subshell: a VAR=value word after CUDA_VISIBLE_DEVICES=... is parsed as a command.
  ( export CUDA_VISIBLE_DEVICES="${gpu}"
    case "${job}" in *"--models moscot"*) export JAX_PLATFORMS=cpu ;; esac
    python -u -m src.training.runner ${job}
  ) > "${log}" 2>&1 &
  running=$((running+1))
done < "${JOBS_FILE}"
while [ "${running}" -gt 0 ]; do wait_one; done
if [ "${worker_fail}" -ne 0 ]; then
  echo "one or more workers failed; see logs/par_${SLURM_JOB_ID}_*.log" >&2
  exit 1
fi
echo "all ${idx} jobs finished" >&2

if [ "${ENABLE_MPS}" = "1" ]; then echo quit | nvidia-cuda-mps-control || true; fi
'
run_status=$?
set -e

# 44/45 are this allocation (requeue); 42 is the wrong container (do not requeue). Preflight precedes the first launch, so requeue costs no completed work.
if [ "${run_status}" -ne 0 ]; then
  echo "Container command exited with status ${run_status}" >&2
  if [ "${SKIP_CUDA_PREFLIGHT:-0}" != "1" ] && [ "${CUDA_PREFLIGHT_REQUEUE}" = "1" ]; then
    restart_count="${SLURM_RESTART_COUNT:-0}"
    if [ "${run_status}" = "44" ] || [ "${run_status}" = "45" ]; then
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
