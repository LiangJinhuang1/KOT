#!/bin/bash
#
# Run the whole hyperparameter search end to end, freezing each round's winner into the
# next one's job lines. Four rounds:
#
#   k0  lr warmup x run length                        12 jobs
#   k   joint lr_phi x lr_alpha_kappa x lr_beta x lambda_dyn   162 jobs
#   i   the fine 1..1000 lambda ladder, on round k's top TWO lr triples   28 jobs
#   fin the winner and the incumbent on the 8 seeds the search never saw   4 jobs
#
# Each round is its own sbatch, so the 7 h wall limit applies per round rather than to
# the chain. Between rounds this collects the round, applies tools/pick_sweep_winner.py,
# and generates the next jobs file from what it picked -- so a round's job lines carry
# the whole resolved config and nothing is inherited implicitly.
#
# The chain STOPS rather than guessing if a round dies, produces no usable cell, or
# comes back with every cell flagged. Restart it from a later round with --from.
#
# Usage:
#   nohup setsid bash slurm/run_sweep_chain.sh > logs/chain.log 2>&1 &
#   bash slurm/run_sweep_chain.sh --adopt-k0 4206208     # reuse an in-flight round 0
#   bash slurm/run_sweep_chain.sh --from k               # skip ahead
set -euo pipefail
cd "$(dirname "$0")/.."

CONTAINER="${CONTAINER:-/data/common/images/codedev_v1.0.5.sif}"
MAX_PARALLEL="${MAX_PARALLEL:-32}"
POLL_SEC="${POLL_SEC:-60}"
SEARCH_SEEDS="42,123,2026,6"
HOLDOUT_SEEDS="9,11,17,21,33,77,88,101"   # the 8 in config/training.yaml the search never sees
STATE="cache/results/chain_state.env"
ADOPT_K0=""
FROM="k0"

while [ $# -gt 0 ]; do
  case "$1" in
    --adopt-k0) ADOPT_K0="$2"; shift 2 ;;
    --from)     FROM="$2";     shift 2 ;;
    --max-parallel) MAX_PARALLEL="$2"; shift 2 ;;
    -h|--help)  sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 1 ;;
  esac
done

mkdir -p cache/results logs
# stderr, not stdout: submit() and stamp_of() return their value on stdout via $( ),
# and a progress line printed there would be captured as part of that value.
say() { echo "[chain $(date +%H:%M:%S)] $*" >&2; }
die() { echo "[chain $(date +%H:%M:%S)] STOP: $*" >&2; exit 1; }

# Persist each frozen setting as it is decided, so --from can resume without re-running
# the round that decided it.
remember() { printf '%s=%q\n' "$1" "$2" >> "${STATE}"; }
[ -f "${STATE}" ] && . "${STATE}"

py() { singularity exec --bind /data2 --pwd "$(pwd)" "${CONTAINER}" /bin/bash -lc "$1"; }

# Wait for a job to leave the queue, then insist it actually succeeded. A requeue (the
# CUDA-802 path) keeps the id, so polling the id rather than sleeping a fixed time is
# what makes this survive a bad allocation.
wait_for() {
  local jobid="$1" state
  say "waiting on job ${jobid}"
  while squeue -j "${jobid}" -h -o "%T" 2>/dev/null | grep -qE "RUNNING|PENDING|CONFIGURING|COMPLETING|REQUEUED"; do
    sleep "${POLL_SEC}"
  done
  state=$(sacct -j "${jobid}" -X -n -o State%30 2>/dev/null | head -1 | tr -d ' ')
  say "job ${jobid} finished: ${state}"
  case "${state}" in
    COMPLETED) ;;
    *) die "job ${jobid} ended ${state}; see logs/slurm${jobid}.log" ;;
  esac
}

submit() {
  local file="$1" jobid
  local n; n=$(grep -ce '^--' "${file}")
  # The generator checks this per file; the chain concatenates files, so check again.
  local u; u=$(grep '^--' "${file}" | grep -o '\-\-run-dir [^ ]*$' | sort -u | wc -l)
  [ "${n}" -eq "${u}" ] || die "${file}: ${n} jobs but ${u} unique run-dirs (collision)"
  say "submitting ${file} (${n} jobs, MAX_PARALLEL=${MAX_PARALLEL})"
  jobid=$(sbatch --parsable --export=ALL,JOBS_FILE="${file}",MAX_PARALLEL="${MAX_PARALLEL}" \
                 slurm/parallel_train.sh)
  echo "${jobid}"
}

# The stamp the launcher chose, so the collect glob finds this round's dirs and not an
# earlier round's. Appended logs can hold several; the last one is this attempt's.
stamp_of() {
  local jobid="$1" stamp
  stamp=$(grep -h "^Run stamp:" "logs/slurm${jobid}.log" 2>/dev/null | tail -1 | awk '{print $3}')
  [ -n "${stamp}" ] || die "no run stamp in logs/slurm${jobid}.log"
  echo "${stamp}"
}

# collect <tier> <stamp> <group-by> <paired-vs> <out-stem>
collect() {
  local tier="$1" stamp="$2" group="$3" paired="$4" out="$5"
  say "collecting tier ${tier} (stamp ${stamp}) -> ${out}"
  py "python tools/collect_run_diagnostics.py \
        --runs 'cache/training/run_${stamp}_sw${tier}_*' \
        --rank-by val_foscttm --group-by '${group}' --paired-vs '${paired}' \
        --out '$(pwd)/${out}.csv'" || die "collect failed for tier ${tier}"
  [ -s "${out}_paired.csv" ] || die "collect wrote no paired table for tier ${tier}"
}

# pick <group-by> <out-stem> <top> -> key=value lines on stdout
pick() {
  local group="$1" out="$2" top="${3:-1}"
  python3 tools/pick_sweep_winner.py --paired "${out}_paired.csv" \
      --by-config "${out}_by_config.csv" --keys "${group}" --top "${top}" \
    || die "no usable cell in ${out}_paired.csv (every cell flagged or regressed)"
}

run_round() {   # run_round <tier> <jobs-file> ; echoes the stamp
  local tier="$1" file="$2" jobid
  jobid=$(submit "${file}")
  remember "JOBID_${tier}" "${jobid}"
  wait_for "${jobid}"
  stamp_of "${jobid}"
}

order="k0 k i fin"
started=0
for round in ${order}; do
  [ "${round}" = "${FROM}" ] && started=1
  [ "${started}" -eq 1 ] || { say "skipping ${round} (--from ${FROM})"; continue; }

  case "${round}" in

  k0)
    say "===== ROUND k0: lr warmup x run length ====="
    if [ -n "${ADOPT_K0}" ]; then
      # Do NOT regenerate the jobs file here: parallel_train.sh streams it with an open
      # fd for the whole run (`done < ${JOBS_FILE}`), so truncating it under a live job
      # would corrupt what is left to read.
      say "adopting in-flight job ${ADOPT_K0} for round k0 (leaving its jobs file alone)"
      wait_for "${ADOPT_K0}"
      STAMP_K0=$(stamp_of "${ADOPT_K0}")
    else
      bash slurm/make_sweep_jobs.sh --tier k0 --seeds "${SEARCH_SEEDS}"
      STAMP_K0=$(run_round k0 jobs_sweep_k0.txt)
    fi
    remember STAMP_K0 "${STAMP_K0}"
    collect k0 "${STAMP_K0}" "lr_warmup_epochs,n_epochs" \
            "lr_warmup_epochs=20,n_epochs=500" cache/results/chain_k0
    picked=$(pick 'lr_warmup_epochs,n_epochs' cache/results/chain_k0 1)
    eval "${picked}"
    WARMUP="${lr_warmup_epochs}"; EPOCHS="${n_epochs}"
    remember WARMUP "${WARMUP}"; remember EPOCHS "${EPOCHS}"
    say "FROZEN: lr_warmup_epochs=${WARMUP}  n_epochs=${EPOCHS}"
    ;;

  k)
    say "===== ROUND k: joint lr x lambda_dyn ====="
    : "${WARMUP:?round k needs the warmup from round k0 -- run --from k0, or set it in ${STATE}}"
    bash slurm/make_sweep_jobs.sh --tier k --seeds "${SEARCH_SEEDS}" \
         --warmup "${WARMUP}" --n-epochs "${EPOCHS}"
    STAMP_K=$(run_round k jobs_sweep_k.txt)
    remember STAMP_K "${STAMP_K}"
    collect k "${STAMP_K}" "lr_phi,lr_alpha_kappa,lr_beta,lambda_dyn" \
            "lr_phi=0.001,lr_alpha_kappa=0.001,lr_beta=0.001,lambda_dyn=100" \
            cache/results/chain_k
    picked=$(pick 'lr_phi,lr_alpha_kappa,lr_beta,lambda_dyn' cache/results/chain_k 2)
    eval "${picked}"
    for n in 1 2; do
      for key in lr_phi lr_alpha_kappa lr_beta lambda_dyn; do
        remember "PICK${n}_${key}" "$(eval echo "\$PICK${n}_${key}")"
      done
    done
    say "FROZEN lr triple 1: phi=${PICK1_lr_phi} ak=${PICK1_lr_alpha_kappa} beta=${PICK1_lr_beta} (lambda ${PICK1_lambda_dyn})"
    say "runner-up  triple 2: phi=${PICK2_lr_phi} ak=${PICK2_lr_alpha_kappa} beta=${PICK2_lr_beta} (lambda ${PICK2_lambda_dyn})"
    ;;

  i)
    say "===== ROUND i: fine lambda ladder on the top two lr triples ====="
    : "${PICK1_lr_phi:?round i needs the pick from round k -- run --from k}"
    for n in 1 2; do
      eval "p=\$PICK${n}_lr_phi; a=\$PICK${n}_lr_alpha_kappa; b=\$PICK${n}_lr_beta"
      bash slurm/make_sweep_jobs.sh --tier i --seeds "${SEARCH_SEEDS}" \
           --warmup "${WARMUP}" --n-epochs "${EPOCHS}" \
           --lr-phi "${p}" --lr-alpha-kappa "${a}" --lr-beta "${b}" \
           --tag-suffix "_t${n}" --out "jobs_sweep_i_t${n}.txt"
    done
    { cat jobs_sweep_i_t1.txt; grep '^--' jobs_sweep_i_t2.txt; } > jobs_sweep_i.txt
    STAMP_I=$(run_round i jobs_sweep_i.txt)
    remember STAMP_I "${STAMP_I}"
    collect i "${STAMP_I}" "lr_phi,lr_alpha_kappa,lr_beta,lambda_dyn" \
            "lr_phi=${PICK1_lr_phi},lr_alpha_kappa=${PICK1_lr_alpha_kappa},lr_beta=${PICK1_lr_beta},lambda_dyn=${PICK1_lambda_dyn}" \
            cache/results/chain_i
    picked=$(pick 'lr_phi,lr_alpha_kappa,lr_beta,lambda_dyn' cache/results/chain_i 1)
    eval "${picked}"
    for key in lr_phi lr_alpha_kappa lr_beta lambda_dyn; do
      remember "FINAL_${key}" "$(eval echo "\$${key}")"
    done
    FINAL_lr_phi="${lr_phi}"; FINAL_lr_alpha_kappa="${lr_alpha_kappa}"
    FINAL_lr_beta="${lr_beta}"; FINAL_lambda_dyn="${lambda_dyn}"
    say "FROZEN: phi=${lr_phi} ak=${lr_alpha_kappa} beta=${lr_beta} lambda_dyn=${lambda_dyn}"
    ;;

  fin)
    say "===== ROUND fin: winner vs incumbent on the 8 unseen seeds ====="
    : "${FINAL_lr_phi:?round fin needs the pick from round i -- run --from i}"
    # The headline number must not come from seeds the search chose on. These 8 are the
    # rest of config/training.yaml's 12; the search only ever saw the first 4.
    bash slurm/make_sweep_jobs.sh --tier i --seeds "${HOLDOUT_SEEDS}" \
         --warmup "${WARMUP}" --n-epochs "${EPOCHS}" \
         --lr-phi "${FINAL_lr_phi}" --lr-alpha-kappa "${FINAL_lr_alpha_kappa}" \
         --lr-beta "${FINAL_lr_beta}" --lambda-dyn "${FINAL_lambda_dyn}" \
         --tag-suffix "_final" --out jobs_sweep_fin_raw.txt
    # tier i emits its whole ladder; the confirmation only wants the chosen lambda and
    # the incumbent (config default: warmup 0, lr 1e-3 across heads, lambda_dyn 100).
    {
      echo "# Confirmation round: the frozen winner and the incumbent, on the 8 seeds"
      echo "# the search never saw. Read as an absolute number, not a paired delta."
      grep '^--' jobs_sweep_fin_raw.txt | grep -- "--set lambda_dyn=${FINAL_lambda_dyn} "
      for ds in pbmc_retained bmmc_cite_retained; do
        short=${ds/_retained/_scv}; short=${short/bmmc_cite/bmmc}
        echo "--datasets ${ds} --models kot_main --seeds ${HOLDOUT_SEEDS} --no-predictions --set lr_warmup_epochs=0 --set n_epochs=500 --set dyn_warmup_epochs=50 --set lr_phi=1.0e-3 --set lr_alpha_kappa=1.0e-3 --set lr_beta=1.0e-3 --set lambda_dyn=100 --set early_stopping_patience=100 --run-dir cache/training/run_@STAMP@_swfin_${short}_incumbent"
      done
    } > jobs_sweep_fin.txt
    STAMP_FIN=$(run_round fin jobs_sweep_fin.txt)
    remember STAMP_FIN "${STAMP_FIN}"
    say "collecting the confirmation as absolute numbers (no paired baseline)"
    py "python tools/collect_run_diagnostics.py \
          --runs 'cache/training/run_${STAMP_FIN}_sw*' --rank-by val_foscttm --aggregate \
          --hparams lr_phi,lr_alpha_kappa,lr_beta,lambda_dyn,lr_warmup_epochs \
          --out '$(pwd)/cache/results/chain_fin.csv'" || die "final collect failed"
    ;;
  esac
done

say "===== CHAIN COMPLETE ====="
say "frozen config: warmup=${WARMUP:-?} n_epochs=${EPOCHS:-?} phi=${FINAL_lr_phi:-?} \
ak=${FINAL_lr_alpha_kappa:-?} beta=${FINAL_lr_beta:-?} lambda_dyn=${FINAL_lambda_dyn:-?}"
say "tables: cache/results/chain_{k0,k,i,fin}*.csv   state: ${STATE}"
