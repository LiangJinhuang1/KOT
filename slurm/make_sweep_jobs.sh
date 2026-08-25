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
# Two tiers remain:
#
#   bash slurm/make_sweep_jobs.sh --tier e       # KOT: lr x length x lambda_dyn (50 jobs)
#   bash slurm/make_sweep_jobs.sh --tier base    # baseline sweeps (80 jobs, mostly 1 run each)
#
# Tier e is one 3-D grid rather than a chain of one-axis rounds. The three knobs are
# pairwise coupled, and sequencing them has already cost us once: tier a chose
# lambda_dyn at an lr and schedule that tiers b and d later changed, so that answer had
# to be discarded. One grid, one submission, no conditioning.
#
# Tier base exists because a tuned method against default-configured baselines is the
# easiest thing for a reviewer to attack, and three of the baselines are deterministic
# so sweeping them is nearly free.
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

while [ $# -gt 0 ]; do
  case "$1" in
    --tier)     TIER="$2";     shift 2 ;;
    --out)      OUT="$2";      shift 2 ;;
    --seeds)    SEEDS="$2";    shift 2 ;;
    --models)   MODELS="$2";   shift 2 ;;
    --datasets) DATASETS="${2//,/ }"; shift 2 ;;
    -h|--help)  sed -n '2,40p' "$0"; exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 1 ;;
  esac
done

case "${TIER}" in
  e|base) ;;
  "")  echo "--tier e|base is required" >&2; exit 1 ;;
  *)   echo "unknown tier: ${TIER} (expected e or base)" >&2; exit 1 ;;
esac
OUT="${OUT:-jobs_sweep_${TIER}.txt}"


# Settled by tier d and held fixed for both rounds: the lr SHAPE (one multiplicative
# warmup->cosine over every head) and the kinetics ramp. Tier d also settled the RATIOS
# between the three head rates -- phi 1e-3, alpha/kappa 1e-4, beta 1e-3, scanned in both
# directions -- but never the overall scale they sit at, which is tier e's axis.
SCHED="--set lr_warmup_epochs=10 --set lr_warmup_start_factor=0.1 --set lr_min_factor=0.01"
WARMUP="--set dyn_warmup_epochs=50"

# The three head rates at a multiplier on those ratios. One definition, so tier f's
# control cell is bit-identical to the tier-e cell it was chosen from.
head_rates() {
  awk -v m="$1" 'BEGIN{printf "--set lr_phi=%.2e --set lr_alpha_kappa=%.2e --set lr_beta=%.2e", \
                        m * 1.0e-3, m * 1.0e-4, m * 1.0e-3}'
}

# Short filesystem-safe tag for a value: 5.0e-4 -> 5p0e-4, true -> true, 0.05 -> 0p05.
tag() { echo "${1//./p}"; }

# Same as emit, but names the model on the line. The baselines tier mixes several
# methods in one file, where the tier-wide MODELS default does not apply.
emit_model() {
  local model="$1"; shift
  local saved="${MODELS}"
  MODELS="${model}"
  emit "$@"
  MODELS="${saved}"
}

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
  e)    COLLECT="python tools/collect_run_diagnostics.py ${COLLECT_RUNS} \\
             --group-by lr_phi,lambda_dyn,n_epochs \\
             --paired-vs lr_phi=0.001,lambda_dyn=1000,n_epochs=500" ;;
  base) COLLECT="python tools/collect_run_diagnostics.py ${COLLECT_RUNS} --aggregate" ;;
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
      # lr x run length x lambda_dyn, as ONE 3-D grid rather than a chain of rounds.
      # All three are open and pairwise coupled, and the chain has already burned us
      # twice: tier a picked lambda_dyn at an lr and schedule that later changed, so its
      # answer had to be thrown away. 48 jobs ends the conditioning problem outright.
      #
      # lr is in it because clip_grad_norm_(max_norm=1.0) runs on every step and the
      # PRE-clip norm sits 100-6000x above that budget -- the clip always binds, the
      # post-clip gradient has norm exactly 1.0, and lr is therefore the entire step
      # size. It has never been scanned here: tier b's {5e-4, 1e-3, 2e-3} predates both
      # the schedule and batch_size=8192. The grid leans UP because tier d measured
      # lr_beta=5e-3 recovering beta anchors 11x better than 1e-3.
      #
      # NOT a batch-size rescale: batch_size only splits the kinetics term into chunks
      # that kinetics_step weights by len_b/n, and there is ONE optimiser step per
      # epoch, so the effective batch is all n cells at any batch_size.
      #
      # lambda_dyn spans 300-3000 because 1000 beat the canonical 100 on pbmc across 5
      # seeds (0.18 vs 0.21-0.23) -- but it won at the EDGE of tier a's grid, and an
      # optimum on a grid boundary is a direction, not an optimum. 3000 is here to
      # bracket it; if 3000 wins, the range has to move again.
      echo "# ===== lr multiplier x run length x lambda_dyn ====="
      for ds in ${DATASETS}; do
        for len in "500:100" "1000:200"; do
          IFS=: read -r ep pat <<< "${len}"
          for lam in 300 1000 3000; do
            for mult in 0.3 1.0 3.0 10.0; do
              emit "lrx$(tag "${mult}")_lam${lam}_ep${ep}" "${ds}" ${SCHED} ${WARMUP} \
                   $(head_rates "${mult}") --set lambda_dyn=${lam} \
                   --set n_epochs=${ep} --set early_stopping_patience=${pat}
            done
          done
        done
      done
      echo
      # Reproduction anchor, 2 jobs: the current canonical config (lambda_dyn=100,
      # multiplier 1.0, 500 epochs). It is outside the grid on purpose -- its job is to
      # re-derive the known 0.21-0.23 pbmc / 0.14-0.19 bmmc numbers under today's code.
      # If it does not, the grid is measuring a pipeline change, not a hyperparameter.
      echo "# ===== reproduction anchor (canonical lambda_dyn=100) ====="
      for ds in ${DATASETS}; do
        emit "anchor_lam100_ep500" "${ds}" ${SCHED} ${WARMUP} $(head_rates 1.0) \
             --set lambda_dyn=100 --set n_epochs=500 --set early_stopping_patience=100
      done
      ;;

    base)
      # Baseline sweeps, so the comparison is method-vs-method rather than
      # tuned-vs-default. Split by cost, not by importance:
      #
      # scot / moscot / linear_ode are deterministic -- config/training.yaml leaves them
      # out of seeded_models, so they run ONCE per config. A 16-cell moscot grid is 16
      # runs, not 16x12, which makes this the cheapest fairness fix available. It also
      # matters most: moscot is the method KOT currently ties on pbmc, and the existing
      # tools/run_moscot_sweep.py only ever ran on the SYNTHETIC staged dataset -- on
      # real CITE-seq moscot has only ever run at epsilon=0.1, max_iterations=100.
      #
      # glue / uniport ARE seeded, so they are screened at 3 seeds on one axis each.
      # Screening only has to find large differences; precision is for the final
      # confirmation, which must use the same seed count as KOT's headline number.
      #
      # totalvi is deliberately absent: config/training.yaml calls it the paired-latent
      # performance CEILING, not a competitor. Tuning it raises the ceiling, which is a
      # different claim.
      echo "# ===== moscot: epsilon x max_iterations (deterministic, 1 run/config) ====="
      for ds in ${DATASETS}; do
        for eps in 0.01 0.05 0.1 0.5; do
          for iters in 100 200 500 1000; do
            emit_model moscot "moscot_eps$(tag "${eps}")_it${iters}" "${ds}" \
                 --set epsilon=${eps} --set max_iterations=${iters}
          done
        done
      done
      echo
      echo "# ===== scot: k x eps (deterministic) ====="
      # SCOT's published protocol is itself a grid over k and eps with unsupervised
      # selection, so a single fixed (k=110, eps=5e-3) is not its recommended usage.
      for ds in ${DATASETS}; do
        for k in 50 110 200; do
          for eps in 1.0e-3 5.0e-3 1.0e-2; do
            emit_model scot "scot_k${k}_eps$(tag "${eps}")" "${ds}" \
                 --set k=${k} --set eps=${eps}
          done
        done
      done
      echo
      echo "# ===== linear_ode: lambda_ode x sinkhorn_reg (deterministic) ====="
      for ds in ${DATASETS}; do
        for lode in 0.1 1.0 10.0; do
          for blur in 0.05 0.1 0.2; do
            emit_model linear_ode "lode$(tag "${lode}")_blur$(tag "${blur}")" "${ds}" \
                 --set lambda_ode=${lode} --set sinkhorn_reg=${blur}
          done
        done
      done
      echo
      # 3 seeds, not 4: a screen only has to separate large differences. emit reads
      # SEEDS at call time, and this is the last block in the tier.
      SEEDS="42,123,2026"
      echo "# ===== glue / uniport: 3-seed screen, one axis each ====="
      for ds in ${DATASETS}; do
        for la in 0.01 0.05 0.2; do
          emit_model glue "glue_lalign$(tag "${la}")" "${ds}" --set lam_align=${la}
        done
        for lot in 0.5 1.0 2.0; do
          emit_model uniport "uniport_lot$(tag "${lot}")" "${ds}" --set lambda_ot=${lot}
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
