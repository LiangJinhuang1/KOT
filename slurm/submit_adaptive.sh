#!/bin/bash
#
# Submit a jobs file on however many GPUs are actually free, and step down on a
# preflight failure instead of retrying the same size.
# parallel_train.sh requeues at the same GPU count; a preflight failure is evidence to ask for less.
#
# Usage:
#   bash slurm/submit_adaptive.sh jobs.txt
#   bash slurm/submit_adaptive.sh jobs.txt --dependency afterok:12345
#   bash slurm/submit_adaptive.sh jobs.txt --max-gpus 4 --min-gpus 1 --max-parallel 16
#   bash slurm/submit_adaptive.sh jobs.txt --gpu-order b200,a100
#
# b200 and a100 are separate pools; walk --gpu-order so a full pool does not hide an idle one.
# Only b200, a100 and a100_10gb pass the submit filter; a100_10gb matches no real gres.
#
#   nohup bash slurm/submit_adaptive.sh jobs.txt > logs/adaptive.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."

JOBS_FILE="${1:?usage: submit_adaptive.sh <jobs_file> [flags]}"; shift || true
MAX_GPUS=4               # the most this account may ask for
MIN_GPUS=1
MAX_PARALLEL=16
NODE=srvcore3
GRES=b200
GPU_ORDER=b200,a100
DEPENDENCY=""
WAIT_FREE_SEC=300        # how long to wait for a free GPU before giving up

while [ $# -gt 0 ]; do
  case "$1" in
    --max-gpus)     MAX_GPUS="$2";     shift 2 ;;
    --min-gpus)     MIN_GPUS="$2";     shift 2 ;;
    --max-parallel) MAX_PARALLEL="$2"; shift 2 ;;
    --node)         NODE="$2";         shift 2 ;;
    --gres)         GRES="$2"; GPU_ORDER="$2"; shift 2 ;;
    --gpu-order)    GPU_ORDER="$2";    shift 2 ;;
    --dependency)   DEPENDENCY="$2";   shift 2 ;;
    --wait-free)    WAIT_FREE_SEC="$2";shift 2 ;;
    *) echo "unknown flag: $1" >&2; exit 1 ;;
  esac
done
[ -f "${JOBS_FILE}" ] || { echo "no such jobs file: ${JOBS_FILE}" >&2; exit 1; }

# Configured minus allocated, summed over every node of this type. A node with nothing allocated has no AllocTRES gres entry, hence :-0.
free_of_type() {
  local type="$1" total=0 node cfg alloc
  for node in $(sinfo -h -N -o "%N" | sort -u); do
    cfg=$(scontrol show node "${node}" 2>/dev/null | sed -n 's/.*CfgTRES=\([^ ]*\).*/\1/p' \
          | tr ',' '\n' | grep -oE "gres/gpu:${type}=[0-9]+" | head -1 | cut -d= -f2)
    [ -z "${cfg}" ] && continue
    alloc=$(scontrol show node "${node}" 2>/dev/null | sed -n 's/.*AllocTRES=\([^ ]*\).*/\1/p' \
          | tr ',' '\n' | grep -oE "gres/gpu:${type}=[0-9]+" | head -1 | cut -d= -f2)
    total=$(( total + cfg - ${alloc:-0} ))
  done
  echo "${total}"
}

free_gpus() { free_of_type "${GRES}"; }

# First type in --gpu-order with at least MIN_GPUS free.
pick_gpu_type() {
  local type
  for type in $(echo "${GPU_ORDER}" | tr ',' ' '); do
    if [ "$(free_of_type "${type}")" -ge "${MIN_GPUS}" ]; then echo "${type}"; return; fi
  done
  echo ""
}

# Wait for at least MIN_GPUS so the first request is sized to what is free now.
waited=0
while :; do
  chosen=$(pick_gpu_type)
  [ -n "${chosen}" ] && break
  if [ "${waited}" -ge "${WAIT_FREE_SEC}" ]; then
    echo "no GPU free in [${GPU_ORDER}] after ${WAIT_FREE_SEC}s; giving up." >&2
    for t in $(echo "${GPU_ORDER}" | tr ',' ' '); do
      echo "  ${t}: $(free_of_type "${t}") free" >&2
    done
    exit 1
  fi
  sleep 30; waited=$(( waited + 30 ))
done
GRES="${chosen}"

free=$(free_gpus)
n=$(( free < MAX_GPUS ? free : MAX_GPUS ))
echo "[adaptive] chose ${GRES} from [${GPU_ORDER}]: ${free} free; starting at ${n} GPU(s), floor ${MIN_GPUS}."

while [ "${n}" -ge "${MIN_GPUS}" ]; do
  dep_flag=(); [ -n "${DEPENDENCY}" ] && dep_flag=(--dependency="${DEPENDENCY}")
  # CUDA_PREFLIGHT_REQUEUE=0 so this script owns the step-down instead of retrying the same size.
  jid=$(sbatch --parsable "${dep_flag[@]}" \
        --gres="gpu:${GRES}:${n}" \
        --export=ALL,JOBS_FILE="${JOBS_FILE}",MAX_PARALLEL="${MAX_PARALLEL}",CUDA_PREFLIGHT_REQUEUE=0 \
        slurm/parallel_train.sh)
  echo "[adaptive] submitted ${jid} with ${n} GPU(s) for ${JOBS_FILE}"

  while true; do
    state=$(sacct -j "${jid}" --format=State -X -n -P 2>/dev/null | head -1 | tr -d ' ')
    case "${state}" in
      PENDING|RUNNING|REQUEUED|CONFIGURING|COMPLETING|"")
        if grep -q "cuda available: True" "logs/slurm${jid}.err" 2>/dev/null; then
          echo "[adaptive] ${jid} cleared the CUDA preflight on ${n} GPU(s) -- training."
          exit 0
        fi
        sleep 15 ;;
      *)
        code=$(sacct -j "${jid}" --format=ExitCode -X -n -P 2>/dev/null | head -1 | cut -d: -f1)
        if [ "${code}" = "44" ] || [ "${code}" = "45" ]; then
          echo "[adaptive] ${jid} failed the preflight (exit ${code}) on ${n} GPU(s); stepping down."
          n=$(( n - 1 ))
        else
          echo "[adaptive] ${jid} ended ${state} (exit ${code}) -- not a preflight verdict, stopping." >&2
          exit 1
        fi
        break ;;
    esac
  done
done

echo "[adaptive] exhausted down to ${MIN_GPUS} GPU(s) without clearing the preflight." >&2
exit 1
