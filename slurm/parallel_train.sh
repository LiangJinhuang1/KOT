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
#   2) sbatch --export=ALL,JOBS_FILE=jobs.txt,MAX_PARALLEL=16 slurm/parallel_train.sh
#      (optional: ,ENABLE_MPS=1 for true concurrent kernels via NVIDIA MPS)
#
# The preprocessing cache write is atomic (src/data/preprocessing.py), so several runs of the
# SAME dataset starting cold is safe — but warming the cache first (run one job per dataset)
# still avoids the redundant recompute.
#
#SBATCH --account=core-med1-telem
#SBATCH --partition=jobs-gpu
# 3, not 4, and this is a workaround rather than a preference. On 2026-08-26 srvcore3
# held one B200 stuck at fabric.state="In Progress" (the other three read
# "Completed, Success"), and ANY allocation containing it fails CUDA init with error
# 802 for the whole process -- masking it with CUDA_VISIBLE_DEVICES does NOT help: a
# 4-GPU allocation reported cuda.is_available()=False even with only device 0 visible,
# which is why the preflight throws the whole allocation back instead of excluding one
# card. Slurm hands out the low indices first, so asking for 3 lands on the healthy
# ones. Raise this back to 4+ once nvidia-fabricmanager has been restarted on the node.
#SBATCH --gres=gpu:b200:3
# 16, not 32: jobs-gpu-qos caps one user at 168 CPUs, and 32/GPU made two 3-GPU
# jobs (96+96) hit QOSMaxCpuPerUserLimit. 16 matches the partition DefCpuPerGPU.
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=512GB
#SBATCH --time=07:00:00
#SBATCH --job-name=kot_parallel
#SBATCH --requeue
# Requeued attempts reopen these files; without append, each retry TRUNCATES the
# log and the record of how many times CUDA failed, and when, is destroyed.
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
# Same contract as slurm/train_slurm.sh: check CUDA once, before any job starts, and
# hand a broken allocation back to Slurm instead of feeding every job into it. Job
# 4206192 lost all 16 arms to a node whose nvidia-smi looked perfect.
CUDA_PREFLIGHT_REQUEUE="${CUDA_PREFLIGHT_REQUEUE:-1}"
CUDA_PREFLIGHT_MAX_REQUEUES="${CUDA_PREFLIGHT_MAX_REQUEUES:-12}"
# Seconds to hold a requeued job before it may start again. Error 802 (fabric manager
# down) does not clear in the seconds an immediate requeue takes, so an undelayed loop
# spends all 12 attempts inside a few minutes and gives up while the fault is still
# being fixed. Spacing them turns the same 12 into ~2 hours of coverage. 0 = immediate.
CUDA_REQUEUE_DELAY_SEC="${CUDA_REQUEUE_DELAY_SEC:-600}"
[ -f "${JOBS_FILE}" ] || { echo "JOBS_FILE not found: ${JOBS_FILE}"; exit 1; }

echo "===== Parallel training ====="
echo "Node: $(hostname) | JOBS_FILE=${JOBS_FILE} | MAX_PARALLEL=${MAX_PARALLEL} | MPS=${ENABLE_MPS} | AUTO_GPU_THROTTLE=${AUTO_GPU_THROTTLE} | MIN_FREE_GPU_MB=${MIN_FREE_GPU_MB}"
# Count what the launch loop will actually accept -- lines starting with `--`, the
# same test the case below applies. Counting non-blank lines instead reports the
# header comments as jobs: tier f announced "Jobs: 27" for a file holding 16.
echo "Jobs: $(grep -ce '^--' "${JOBS_FILE}")"

# One stamp for the whole sweep, fixed at job start. jobs.txt carries the literal
# token @STAMP@ in its --run-dir values and this is what replaces it, so a jobs
# file generated weeks ago still lands in a directory named for when it ran.
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
echo "Run stamp: ${RUN_STAMP}  (run dirs: cache/training/run_${RUN_STAMP}_*)"

CONTAINER="${CONTAINER:-/data/common/images/codedev_v1.0.5.sif}"
export CONTAINER JOBS_FILE MAX_PARALLEL ENABLE_MPS RUN_STAMP
export AUTO_GPU_THROTTLE MIN_FREE_GPU_MB GPU_POLL_SEC
# --cpus-per-gpu does not set SLURM_CPUS_PER_TASK, so that alone would silently
# fall back to 8 and leave most of the allocation idle — more so now that four GPUs
# means four times the cores. CPUS_ON_NODE reflects what was actually granted.
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
# Round-robin target for each launched job. Counting the devices rather than trusting
# the gres number keeps this correct if the allocation differs from the request.
N_GPUS="$(nvidia-smi -L 2>/dev/null | grep -c "^GPU" || echo 1)"
[ "${N_GPUS}" -ge 1 ] || N_GPUS=1
echo "Inside container on $(hostname); threads/proc=${THREADS_PER}; GPUs=${N_GPUS}" >&2
nvidia-smi -L >&2 || true

# Before the first job, not once per job. Its exit code is propagated out of the
# container so the requeue branch below can see it, and it reports WHICH devices are
# usable so a single wedged card costs one GPU rather than the whole allocation.
USABLE_FILE="logs/gpu_usable_${SLURM_JOB_ID}.txt"
GPU_IDS=""
if [ "${SKIP_CUDA_PREFLIGHT:-0}" != "1" ]; then
  python slurm/cuda_preflight.py --emit-usable "${USABLE_FILE}" >&2 || exit $?
  GPU_IDS="$(cat "${USABLE_FILE}" 2>/dev/null)"
fi
# Skipping the preflight (or a preflight that wrote nothing) falls back to every device
# nvidia-smi listed, which is what this script did before it had a preflight.
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
  # A job line is a runner argument string and therefore starts with `--`. Matching on
  # that instead of skipping `#` lines is what catches a WRAPPED comment: only its first
  # line carries the `#`, so job 4206192 launched the continuation lines of its own
  # "# Collect:" header as two jobs. Anything else is announced rather than dropped
  # silently, so a malformed jobs file is visible in the log.
  case "${job}" in
    --*) ;;
    ""|\#*) continue ;;
    *) echo "[skip] not a job line: ${job}" >&2; continue ;;
  esac
  # Resolve the placeholder now, so every line in this sweep shares one stamp.
  job="${job//@STAMP@/${RUN_STAMP}}"

  # Which GPU this job will use; also the one the throttle should be checking. Indexes
  # into the USABLE list, so a device the preflight rejected is never scheduled onto.
  gpu="${GPU_ARRAY[$(( idx % N_GPUS ))]}"

  # moscot is the only JAX consumer here, and JAX RAISES when its CUDA plugin is
  # present but cannot initialize -- which is what a wedged GPU looks like, and it
  # killed the moscot arms of job 4206192 while torch was failing on the same node.
  # Pinning it to CPU costs moscot its (unverified) GPU path and buys immunity to
  # every GPU-side failure, which is the trade the baselines want: they are
  # deterministic, run once per config, and are not the method under test.
  jax_platforms=""
  case "${job}" in *"--models moscot"*) jax_platforms="JAX_PLATFORMS=cpu" ;; esac

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
  ( CUDA_VISIBLE_DEVICES="${gpu}" ${jax_platforms} python -u -m src.training.runner ${job} ) > "${log}" 2>&1 &
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

# 44/45 are preflight verdicts on THIS allocation, so another one may well be fine;
# 42 is the container being wrong for the hardware, which requeueing cannot fix.
# Nothing has run at that point -- the preflight precedes the first launch -- so the
# requeue costs no completed work and the fresh RUN_STAMP starts a clean sweep.
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
