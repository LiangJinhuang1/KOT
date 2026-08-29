#!/bin/bash
#
# Submit a jobs file on however many GPUs are actually free, and step DOWN on a
# preflight failure instead of retrying the same size.
#
# WHY THIS EXISTS. slurm/parallel_train.sh requeues itself when the CUDA preflight
# fails, which is right when the allocation was unlucky -- but it asks for the SAME
# number of GPUs every time. On 2026-08-28 that lost a whole night: srvcore3 has one
# card stuck in fabric init (error 802), and while exactly 3 of 8 were free, a 3-GPU
# request was FORCED onto it. All 12 requeues drew the same broken card and the job
# died 8 h later having trained nothing. Asking for 2 succeeded on the first try.
#
# The rule this encodes: the request size is not a constant, it is whatever is free
# right now, capped at what you are allowed to ask for -- and a preflight failure is
# evidence to ask for LESS, not to ask again.
#
# Usage:
#   bash slurm/submit_adaptive.sh jobs_sweep_D.txt
#   bash slurm/submit_adaptive.sh jobs_sweep_DH.txt --dependency afterok:12345
#   bash slurm/submit_adaptive.sh jobs.txt --max-gpus 4 --min-gpus 1 --max-parallel 16
#
# Run it in the background -- it polls until the job clears its preflight:
#   nohup bash slurm/submit_adaptive.sh jobs_sweep_D.txt > logs/adaptive_D.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."

JOBS_FILE="${1:?usage: submit_adaptive.sh <jobs_file> [flags]}"; shift || true
MAX_GPUS=4               # the most this account may ask for
MIN_GPUS=1
MAX_PARALLEL=16
NODE=srvcore3
GRES=b200
DEPENDENCY=""
WAIT_FREE_SEC=300        # how long to wait for a free GPU before giving up

while [ $# -gt 0 ]; do
  case "$1" in
    --max-gpus)     MAX_GPUS="$2";     shift 2 ;;
    --min-gpus)     MIN_GPUS="$2";     shift 2 ;;
    --max-parallel) MAX_PARALLEL="$2"; shift 2 ;;
    --node)         NODE="$2";         shift 2 ;;
    --gres)         GRES="$2";         shift 2 ;;
    --dependency)   DEPENDENCY="$2";   shift 2 ;;
    --wait-free)    WAIT_FREE_SEC="$2";shift 2 ;;
    *) echo "unknown flag: $1" >&2; exit 1 ;;
  esac
done
[ -f "${JOBS_FILE}" ] || { echo "no such jobs file: ${JOBS_FILE}" >&2; exit 1; }

# Free GPUs of this type on this node: configured minus allocated. A node with nothing
# allocated has no gres entry in AllocTRES at all, hence the :-0 defaults.
free_gpus() {
  local cfg alloc
  cfg=$(scontrol show node "${NODE}" 2>/dev/null | tr ',' '\n' \
        | grep -oE "gres/gpu:${GRES}=[0-9]+" | head -1 | cut -d= -f2)
  alloc=$(scontrol show node "${NODE}" 2>/dev/null \
        | sed -n 's/.*AllocTRES=\([^ ]*\).*/\1/p' | tr ',' '\n' \
        | grep -oE "gres/gpu:${GRES}=[0-9]+" | head -1 | cut -d= -f2)
  echo $(( ${cfg:-0} - ${alloc:-0} ))
}

# Wait for at least MIN_GPUS to be free, so the first request is sized to reality
# rather than to a number picked before anyone else's job finished.
waited=0
while [ "$(free_gpus)" -lt "${MIN_GPUS}" ]; do
  if [ "${waited}" -ge "${WAIT_FREE_SEC}" ]; then
    echo "no ${GRES} GPU free on ${NODE} after ${WAIT_FREE_SEC}s; giving up." >&2
    exit 1
  fi
  sleep 30; waited=$(( waited + 30 ))
done

free=$(free_gpus)
n=$(( free < MAX_GPUS ? free : MAX_GPUS ))
echo "[adaptive] ${free} ${GRES} free on ${NODE}; starting at ${n} GPU(s), floor ${MIN_GPUS}."

while [ "${n}" -ge "${MIN_GPUS}" ]; do
  dep_flag=(); [ -n "${DEPENDENCY}" ] && dep_flag=(--dependency="${DEPENDENCY}")
  # CUDA_PREFLIGHT_REQUEUE=0 so a bad allocation EXITS (code 44) instead of requeueing
  # at the same size -- stepping down is this script's job, not the batch script's.
  jid=$(sbatch --parsable "${dep_flag[@]}" \
        --gres="gpu:${GRES}:${n}" \
        --export=ALL,JOBS_FILE="${JOBS_FILE}",MAX_PARALLEL="${MAX_PARALLEL}",CUDA_PREFLIGHT_REQUEUE=0 \
        slurm/parallel_train.sh)
  echo "[adaptive] submitted ${jid} with ${n} GPU(s) for ${JOBS_FILE}"

  # Poll until the job either clears its preflight (success: it is training) or exits.
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
