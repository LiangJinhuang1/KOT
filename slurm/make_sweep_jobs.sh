#!/bin/bash
#
# Generate a hyperparameter-sweep jobs file for slurm/parallel_train.sh.
#
# This is NOT slurm/make_jobs.sh: that one generates the fixed paper runs (all three
# KOT arms, 12 seeds, canonical weights). This one generates a SEARCH -- one arm
# (kot_main), 4 seeds, one run-dir per config, and --no-predictions on every line
# so parallel jobs cannot collide on the global data/predictions/ HDF5 path.
#
# Tiers a-d were the open grid (100/100/52/26 jobs), run before anything was fixed;
# they are done and were deleted -- `git log slurm/make_sweep_jobs.sh` has them, and
# each run's own run_config.yaml is the actual record of what it ran. What they settled
# is now locked below and on every job line: the lr shape (warmup 10 / start 0.1 /
# cosine to 0.01), the per-head rates (phi 1e-3, alpha/kappa 1e-4, beta 1e-3), the
# kinetics ramp (dyn_warmup 50; tier d scanned arrival at 0/50/100 and 100 hurt
# time_spearman), and the kappa prior, Sinkhorn blur, gauge and batch size.
#
# Two axes are left, so there are two tiers:
#
#   bash slurm/make_sweep_jobs.sh --tier e                              # lr x length (16 jobs)
#   bash slurm/make_sweep_jobs.sh --tier f --lr-mult 3 --epochs 1000:200  # lambda_dyn (6 jobs)
#
# lr comes FIRST and shares a grid with run length because the two are confounded: a
# learning rate set too low makes 1000 epochs look better than 500 for a reason that has
# nothing to do with how long the model needs. lambda_dyn can safely follow, because the
# always-binding gradient clip (see below) makes it unusually independent of lr -- it
# changes the DIRECTION of a fixed-size step, not the step size.
#
#   sbatch --export=ALL,JOBS_FILE=jobs_sweep_e.txt,MAX_PARALLEL=16 slurm/parallel_train.sh
#
# Each generated file carries the exact command to read it, with the in-grid control
# already filled in. Read a round PAIRED, not as a mean+-sd ranking: seed sd on real
# CITE-seq is the same size as the gaps between configs, every cell runs the same four
# seeds, and differencing within a seed is most of what makes those gaps resolvable.
#
# Search defaults to the first 4 seeds (42,123,2026,6). config/training.yaml keeps
# the full 12 for the later paper runs. After a winner is chosen, drop --seeds
# (or run make_jobs.sh) so those 12 are used. The headline number should then be
# reported on the 8 seeds the search did NOT see, not on all 12.
set -euo pipefail

STAMP="@STAMP@"          # resolved by parallel_train.sh at submit time
ROOT="cache/training"
TIER=""
OUT=""
SEEDS="42,123,2026,6"    # search uses 4; config keeps 12 for later paper runs
MODELS="kot_main"        # one arm during search; the 3-arm group is for the paper runs
# Canonical scVelo retained only. RegVelo / Papalexi are ablations re-run at the frozen
# config once the search is over, never search arms.
DATASETS="pbmc_retained bmmc_cite_retained"
LR_MULT=""               # tier f: the lr multiplier chosen in tier e
EPOCHS=""                # tier f: the n_epochs:patience chosen in tier e

while [ $# -gt 0 ]; do
  case "$1" in
    --tier)     TIER="$2";     shift 2 ;;
    --out)      OUT="$2";      shift 2 ;;
    --seeds)    SEEDS="$2";    shift 2 ;;
    --models)   MODELS="$2";   shift 2 ;;
    --datasets) DATASETS="${2//,/ }"; shift 2 ;;
    --lr-mult)  LR_MULT="$2";  shift 2 ;;
    --epochs)   EPOCHS="$2";   shift 2 ;;
    -h|--help)  sed -n '2,38p' "$0"; exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 1 ;;
  esac
done

case "${TIER}" in
  e|f) ;;
  "")  echo "--tier e|f is required" >&2; exit 1 ;;
  *)   echo "unknown tier: ${TIER} (expected e or f)" >&2; exit 1 ;;
esac
OUT="${OUT:-jobs_sweep_${TIER}.txt}"
[ "${TIER}" != "f" ] || { [ -n "${LR_MULT}" ] && [ -n "${EPOCHS}" ]; } || {
  echo "--tier f needs --lr-mult <multiplier> and --epochs <n_epochs:patience> from tier e" >&2
  exit 1; }

# Settled by tier d and held fixed for both rounds: the lr SHAPE (one multiplicative
# warmup->cosine over every head) and the kinetics ramp. Tier d also settled the RATIOS
# between the three head rates -- phi 1e-3, alpha/kappa 1e-4, beta 1e-3, scanned in both
# directions -- but never the overall scale they sit at, which is tier e's axis.
SCHED="--set lr_warmup_epochs=10 --set lr_warmup_start_factor=0.1 --set lr_min_factor=0.01"
WARMUP="--set dyn_warmup_epochs=50"
EPOCHS="${EPOCHS:-500:100}"

# The three head rates at a multiplier on those ratios. One definition, so tier f's
# control cell is bit-identical to the tier-e cell it was chosen from.
head_rates() {
  awk -v m="$1" 'BEGIN{printf "--set lr_phi=%.2e --set lr_alpha_kappa=%.2e --set lr_beta=%.2e", \
                        m * 1.0e-3, m * 1.0e-4, m * 1.0e-3}'
}

# Short filesystem-safe tag for a value: 5.0e-4 -> 5p0e-4, true -> true, 0.05 -> 0p05.
tag() { echo "${1//./p}"; }

# One job line. `emit <cfg-tag> <dataset> <extra --set flags...>`
emit() {
  local cfgtag="$1"; shift
  local dataset="$1"; shift
  local short="${dataset/_retained/_scv}"; short="${short/_regvelo/_rgv}"
  short="${short/bmmc_cite/bmmc}"
  local seed_flag=""
  [ -n "${SEEDS}" ] && seed_flag="--seeds ${SEEDS} "
  echo "--datasets ${dataset} --models ${MODELS} ${seed_flag}--no-predictions $* --run-dir ${ROOT}/run_${STAMP}_sw${TIER}_${short}_${cfgtag}"
}

# The command that reads this round, with the in-grid control as the paired baseline.
# lr values are compared as the floats run_config.yaml stores them as, so the tier-f
# baseline is 0.001 / 0.0001 / 0.001 -- the multiplier=1.0 cell.
COLLECT_RUNS="--runs 'cache/training/run_<stamp>_sw${TIER}_*'"
case "${TIER}" in
  e) COLLECT="python tools/collect_run_diagnostics.py ${COLLECT_RUNS} \\
             --group-by lr_phi,n_epochs \\
             --paired-vs lr_phi=0.001,n_epochs=500" ;;
  f) COLLECT="python tools/collect_run_diagnostics.py ${COLLECT_RUNS} \\
             --group-by lambda_dyn \\
             --paired-vs lambda_dyn=100" ;;
esac

{
  echo "# Sweep tier ${TIER} -- generated by slurm/make_sweep_jobs.sh on $(date +%Y-%m-%d)."
  echo "# seeds=${SEEDS:-from config/training.yaml}  models=${MODELS}  datasets=${DATASETS}${LAM:+  lambda_dyn=${LAM}}"
  echo "#"
  echo "# Submit:  sbatch --export=ALL,JOBS_FILE=${OUT},MAX_PARALLEL=16 slurm/parallel_train.sh"
  echo "# Collect: ${COLLECT}"
  echo

  case "${TIER}" in

    e)
      # Shared lr multiplier x run length, as ONE grid. Both are open and confounded:
      # an lr set too low makes 1000 epochs beat 500 for a reason that has nothing to do
      # with how long the model needs, so scanning length at a wrong lr measures the lr.
      #
      # Why lr is the axis that matters most here: clip_grad_norm_(max_norm=1.0) runs on
      # every step and the PRE-clip norm sits 100-6000x above that budget, so the clip
      # binds always and the post-clip gradient has norm exactly 1.0. lr is therefore the
      # entire step size. It has never been scanned in this configuration -- tier b's
      # {5e-4, 1e-3, 2e-3} predates both the lr schedule and batch_size=8192.
      #
      # NOTE this is NOT the batch-size scaling rule. batch_size only splits the kinetics
      # term into accumulated chunks (kinetics_step weights each by len_b/n) and there is
      # ONE optimiser step per epoch, so the effective batch is all n cells at any
      # batch_size. Nothing about 1024 -> 8192 calls for 8x the lr.
      #
      # The grid leans UP: tier d measured lr_beta=5e-3 recovering beta anchors 11x
      # better than 1e-3 for 0.014 in jvp cos, so the evidence points above 1.0, and 0.3
      # is carried only so an optimum below the default is still visible.
      # Multiplier 1.0 at 500 epochs is the current canonical config: the paired baseline.
      echo "# ===== shared lr multiplier x run length ====="
      for ds in ${DATASETS}; do
        for len in "500:100" "1000:200"; do
          IFS=: read -r ep pat <<< "${len}"
          for mult in 0.3 1.0 3.0 10.0; do
            emit "lrx$(tag "${mult}")_ep${ep}" "${ds}" ${SCHED} ${WARMUP} \
                 $(head_rates "${mult}") \
                 --set n_epochs=${ep} --set early_stopping_patience=${pat}
          done
        done
      done
      ;;

    f)
      # lambda_dyn ceiling at the tier-e winner. It goes second because the clip makes it
      # near-orthogonal to lr: with the post-clip gradient pinned at norm 1.0, raising
      # lambda_dyn does not lengthen the step, it re-splits a fixed-length step between
      # the alignment and kinetics directions. lr sets how far, lambda_dyn sets where.
      # Tier a did scan it, but before the lr schedule existed, so those numbers are stale.
      # lambda_dyn=100 is the dataset canonical: the in-grid control.
      IFS=: read -r ep pat <<< "${EPOCHS}"
      echo "# ===== lambda_dyn ceiling ====="
      for ds in ${DATASETS}; do
        for lam in 30 100 300; do
          emit "lam${lam}_lrx$(tag "${LR_MULT}")_ep${ep}" "${ds}" ${SCHED} ${WARMUP} \
               $(head_rates "${LR_MULT}") \
               --set lambda_dyn=${lam} \
               --set n_epochs=${ep} --set early_stopping_patience=${pat}
        done
      done
      ;;

  esac
} > "${OUT}"

n_jobs=$(grep -c '^--' "${OUT}")
n_dirs=$(grep '^--' "${OUT}" | grep -o '\-\-run-dir [^ ]*$' | sort -u | wc -l)
echo "Wrote ${OUT}: ${n_jobs} jobs, ${n_dirs} unique run-dirs"
[ "${n_jobs}" -eq "${n_dirs}" ] || {
  echo "ERROR: run-dir collision -- jobs would overwrite each other" >&2
  grep '^--' "${OUT}" | grep -o '\-\-run-dir [^ ]*$' | sort | uniq -d >&2
  exit 1
}
echo "Submit:  sbatch --export=ALL,JOBS_FILE=${OUT},MAX_PARALLEL=16 slurm/parallel_train.sh"
