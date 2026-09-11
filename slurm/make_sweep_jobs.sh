#!/bin/bash
#
# Generate a hyperparameter-sweep jobs file for slurm/parallel_train.sh.
# Unlike make_jobs.sh (paper runs), this is a search: one arm, search seeds,
# one run-dir per config, --no-predictions so parallel jobs cannot collide
# on the global data/predictions/ HDF5 path.
#
# Rank on val_foscttm: the validation slice is held out of both loss terms.
# mean_foscttm is on cells the run trained on, so ranking by it ranks on the
# test set. Paper runs put those cells back (--set val_holdout_from_training=false).
#
# Usage (each round freezes the previous winner as flags):
#   bash slurm/make_sweep_jobs.sh --tier f                       # warmup length × early-stopping target
#   bash slurm/make_sweep_jobs.sh --tier k0                      # warmup length × run length
#   bash slurm/make_sweep_jobs.sh --tier k  --warmup <w> --n-epochs <e>
#   bash slurm/make_sweep_jobs.sh --tier i  --warmup <w> <lr flags>
#   bash slurm/make_sweep_jobs.sh --tier j  <same flags>
#   bash slurm/make_sweep_jobs.sh --tier base
#
#   sbatch --export=ALL,JOBS_FILE=jobs_sweep_f.txt,MAX_PARALLEL=16 slurm/parallel_train.sh
#
# f first: warmup and early-stopping change what later rounds measure.
# g and h are superseded by k and kept so old jobs files stay readable.
# Read a round paired within seed, not as a mean±sd ranking.
# Headline confirmation uses seeds the search did not rank on.
set -euo pipefail

STAMP="@STAMP@"          # resolved by parallel_train.sh at submit time
ROOT="cache/training"
TIER=""
OUT=""
SEEDS="42,123,2026,6"    # search seeds; config keeps 12 for later paper runs
MODELS="kot_main"        # one arm during search; the 3-arm group is for the paper runs
# Canonical scVelo retained only. RegVelo / Papalexi are post-search ablations.
DATASETS="pbmc_retained bmmc_cite_retained"

# Defaults are config defaults; a later round overrides what it has not settled.
WARMUP_EPOCHS=10         # lr warmup length; round f's axis
LR_PHI="1.0e-3"
LR_ALPHA_KAPPA="1.0e-3"
LR_BETA="1.0e-3"
DYN_WARMUP=50            # kinetics ramp length; round j's axis
LAMBDA_DYN=100           # canonical weight; round i's axis
PATIENCE=100
N_EPOCHS=500             # run length; round k0's second axis
# Suffix so one tier can be generated twice and concatenated without run-dir collisions.
TAG_SUFFIX=""
# Overridable so the grid can be re-run further out if the winner lands on the edge.
WARMUP_GRID="20 50 100"
EPOCHS_GRID="500 1000"

while [ $# -gt 0 ]; do
  case "$1" in
    --tier)            TIER="$2";            shift 2 ;;
    --out)             OUT="$2";             shift 2 ;;
    --seeds)           SEEDS="$2";           shift 2 ;;
    --models)          MODELS="$2";          shift 2 ;;
    --datasets)        DATASETS="${2//,/ }"; shift 2 ;;
    --warmup)          WARMUP_EPOCHS="$2";   shift 2 ;;
    --lr-phi)          LR_PHI="$2";          shift 2 ;;
    --lr-alpha-kappa)  LR_ALPHA_KAPPA="$2";  shift 2 ;;
    --lr-beta)         LR_BETA="$2";         shift 2 ;;
    --dyn-warmup)      DYN_WARMUP="$2";      shift 2 ;;
    --lambda-dyn)      LAMBDA_DYN="$2";      shift 2 ;;
    --patience)        PATIENCE="$2";        shift 2 ;;
    --n-epochs)        N_EPOCHS="$2";        shift 2 ;;
    --tag-suffix)      TAG_SUFFIX="$2";      shift 2 ;;
    --warmup-grid)     WARMUP_GRID="$2";     shift 2 ;;
    --epochs-grid)     EPOCHS_GRID="$2";     shift 2 ;;
    -h|--help)  sed -n '2,26p' "$0"; exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 1 ;;
  esac
done

case "${TIER}" in
  f|g|h|i|j|k0|k|base) ;;
  "")  echo "--tier f|g|h|i|j|k0|k|base is required" >&2; exit 1 ;;
  *)   echo "unknown tier: ${TIER} (expected f, g, h, i, j, k0, k or base)" >&2; exit 1 ;;
esac
OUT="${OUT:-jobs_sweep_${TIER}.txt}"

# One multiplicative factor on all three heads so their ratios hold. Warmup 0 is plain cosine.
sched() {
  echo "--set lr_warmup_epochs=$1 --set lr_warmup_start_factor=0.1 --set lr_min_factor=0.01"
}

# Named rates rather than a remembered multiplier, so a job line needs no lookup.
head_rates() {
  echo "--set lr_phi=$1 --set lr_alpha_kappa=$2 --set lr_beta=$3"
}

# Same %g form run_config.yaml and collect_run_diagnostics.py use, so --paired-vs matches.
gnum() { awk -v v="$1" 'BEGIN{printf "%g", v}'; }

# Filesystem-safe tag: 5.0e-4 -> 5p0e-4.
tag() { echo "${1//./p}"; }

# Names the model on the line; the baselines tier mixes methods so MODELS does not apply.
emit_model() {
  local model="$1"; shift
  local saved="${MODELS}"
  MODELS="${model}"
  emit "$@"
  MODELS="${saved}"
}

emit() {
  local cfgtag="$1"; shift
  local dataset="$1"; shift
  local short="${dataset/_retained/_scv}"; short="${short/_regvelo/_rgv}"
  short="${short/bmmc_cite/bmmc}"
  local seed_flag=""
  [ -n "${SEEDS}" ] && seed_flag="--seeds ${SEEDS} "
  echo "--datasets ${dataset} --models ${MODELS} ${seed_flag}--no-predictions $* --run-dir ${ROOT}/run_${STAMP}_sw${TIER}_${short}_${cfgtag}${TAG_SUFFIX}"
}

# Collect command with the in-grid control named so the jobs file is explicit about ranking.
COLLECT_RUNS="--runs 'cache/training/run_<stamp>_sw${TIER}_*'"
case "${TIER}" in
  f)    COLLECT="python tools/collect_run_diagnostics.py ${COLLECT_RUNS} \\
             --rank-by val_foscttm --group-by lr_warmup_epochs,early_stopping_monitor \\
             --paired-vs lr_warmup_epochs=0,early_stopping_monitor=train_align" ;;
  g)    COLLECT="python tools/collect_run_diagnostics.py ${COLLECT_RUNS} \\
             --rank-by val_foscttm --group-by lr_alpha_kappa,lr_beta \\
             --paired-vs lr_alpha_kappa=0.001,lr_beta=0.001" ;;
  h)    COLLECT="python tools/collect_run_diagnostics.py ${COLLECT_RUNS} \\
             --rank-by val_foscttm --group-by lr_phi \\
             --paired-vs lr_phi=0.001" ;;
  i)    COLLECT="python tools/collect_run_diagnostics.py ${COLLECT_RUNS} \\
             --rank-by val_foscttm --group-by lambda_dyn \\
             --paired-vs lambda_dyn=100" ;;
  j)    COLLECT="python tools/collect_run_diagnostics.py ${COLLECT_RUNS} \\
             --runs 'cache/training/run_<stamp-of-round-i>_swi_*' \\
             --rank-by val_foscttm --group-by lambda_dyn,dyn_warmup_epochs \\
             --paired-vs lambda_dyn=100,dyn_warmup_epochs=$(gnum "${DYN_WARMUP}")" ;;
  k0)   COLLECT="python tools/collect_run_diagnostics.py ${COLLECT_RUNS} \\
             --rank-by val_foscttm --group-by lr_warmup_epochs,n_epochs \\
             --paired-vs lr_warmup_epochs=20,n_epochs=500" ;;
  k)    COLLECT="python tools/collect_run_diagnostics.py ${COLLECT_RUNS} \\
             --rank-by val_foscttm --group-by lr_phi,lr_alpha_kappa,lr_beta,lambda_dyn \\
             --paired-vs lr_phi=0.001,lr_alpha_kappa=0.001,lr_beta=0.001,lambda_dyn=100" ;;
  # Baselines have no out-of-sample embedding, so they can only be ranked on mean_foscttm.
  base) COLLECT="python tools/collect_run_diagnostics.py ${COLLECT_RUNS} \
             --rank-by mean_foscttm --aggregate" ;;
esac

{
  echo "# Sweep tier ${TIER} -- generated by slurm/make_sweep_jobs.sh on $(date +%Y-%m-%d)."
  echo "# seeds=${SEEDS:-from config/training.yaml}  models=${MODELS}  datasets=${DATASETS}"
  echo "# frozen: warmup=${WARMUP_EPOCHS}  lr=(phi ${LR_PHI}, alpha/kappa ${LR_ALPHA_KAPPA}, beta ${LR_BETA})"
  echo "#         dyn_warmup=${DYN_WARMUP}  lambda_dyn=${LAMBDA_DYN}  patience=${PATIENCE}"
  echo "#         n_epochs=${N_EPOCHS}"
  # Split is not a sweep axis but decides what val_foscttm means; per-seed from k0 on.
  echo "#         split: per-seed (val_split_per_seed), stratified where configured"
  echo "#"
  echo "# Submit:  sbatch --export=ALL,JOBS_FILE=${OUT},MAX_PARALLEL=16 slurm/parallel_train.sh"
  # COLLECT wraps; a continuation without `#` would be a live job line.
  echo "# Collect:"
  echo "${COLLECT}" | sed 's/^/#   /'
  echo

  case "${TIER}" in

    f)
      # Warmup length × early-stopping target. Patience is shared so the round varies what is watched, not how long.
      echo "# ===== lr warmup x early-stopping monitor ====="
      for ds in ${DATASETS}; do
        for warm in 0 1 10 20; do
          for monitor in train_align val_align; do
            emit "warm${warm}_${monitor}" "${ds}" $(sched "${warm}") \
                 --set dyn_warmup_epochs="${DYN_WARMUP}" \
                 $(head_rates "${LR_PHI}" "${LR_ALPHA_KAPPA}" "${LR_BETA}") \
                 --set lambda_dyn="${LAMBDA_DYN}" \
                 --set early_stopping_monitor="${monitor}" \
                 --set early_stopping_patience="${PATIENCE}"
          done
        done
      done
      ;;

    g)
      # alpha/kappa × beta with phi fixed (superseded by k; kept so old jobs files stay readable).
      echo "# ===== lr: alpha/kappa x beta, phi fixed at ${LR_PHI} ====="
      for ds in ${DATASETS}; do
        for ak in 1.0e-4 3.0e-4 1.0e-3 3.0e-3; do
          for lb in 3.0e-4 1.0e-3 3.0e-3 1.0e-2; do
            emit "ak$(tag "${ak}")_b$(tag "${lb}")" "${ds}" $(sched "${WARMUP_EPOCHS}") \
                 --set dyn_warmup_epochs="${DYN_WARMUP}" \
                 $(head_rates "${LR_PHI}" "${ak}" "${lb}") \
                 --set lambda_dyn="${LAMBDA_DYN}" \
                 --set early_stopping_patience="${PATIENCE}"
          done
        done
      done
      ;;

    h)
      # phi rate (superseded by the joint round). phi is the alignment map FOSCTTM rides on.
      echo "# ===== lr: phi, at alpha/kappa=${LR_ALPHA_KAPPA} beta=${LR_BETA} ====="
      for ds in ${DATASETS}; do
        for lp in 1.0e-4 3.0e-4 1.0e-3 3.0e-3 1.0e-2; do
          emit "phi$(tag "${lp}")" "${ds}" $(sched "${WARMUP_EPOCHS}") \
               --set dyn_warmup_epochs="${DYN_WARMUP}" \
               $(head_rates "${lp}" "${LR_ALPHA_KAPPA}" "${LR_BETA}") \
               --set lambda_dyn="${LAMBDA_DYN}" \
               --set early_stopping_patience="${PATIENCE}"
        done
      done
      ;;

    i)
      # Fine lambda_dyn ladder. It is a mixing coefficient: the clip binds, so it splits a fixed-norm step.
      echo "# ===== lambda_dyn ====="
      for ds in ${DATASETS}; do
        for lam in 1 50 100 200 300 500 1000; do
          emit "lam${lam}" "${ds}" $(sched "${WARMUP_EPOCHS}") \
               --set dyn_warmup_epochs="${DYN_WARMUP}" \
               $(head_rates "${LR_PHI}" "${LR_ALPHA_KAPPA}" "${LR_BETA}") \
               --set lambda_dyn="${lam}" \
               --set early_stopping_patience="${PATIENCE}"
        done
      done
      ;;

    j)
      # Ramp × weight; checkpoint after the ramp so a climbing weight cannot hand back a low-dynamics model.
      echo "# ===== lambda ramp x lambda_dyn ====="
      for ds in ${DATASETS}; do
        for ramp in 0 100; do
          for lam in 1 50 100 200 300 500 1000; do
            emit "ramp${ramp}_lam${lam}" "${ds}" $(sched "${WARMUP_EPOCHS}") \
                 --set dyn_warmup_epochs="${ramp}" \
                 $(head_rates "${LR_PHI}" "${LR_ALPHA_KAPPA}" "${LR_BETA}") \
                 --set lambda_dyn="${lam}" \
                 --set early_stopping_patience="${PATIENCE}"
          done
        done
      done
      ;;

    k0)
      # Warmup length × run length. The two are not separable: warmup sets the cosine span as n_epochs - warmup.
      echo "# ===== lr warmup x run length ====="
      for ds in ${DATASETS}; do
        for warm in ${WARMUP_GRID}; do
          for epochs in ${EPOCHS_GRID}; do
            emit "warm${warm}_ep${epochs}" "${ds}" $(sched "${warm}") \
                 --set n_epochs="${epochs}" \
                 --set dyn_warmup_epochs="${DYN_WARMUP}" \
                 $(head_rates "${LR_PHI}" "${LR_ALPHA_KAPPA}" "${LR_BETA}") \
                 --set lambda_dyn="${LAMBDA_DYN}" \
                 --set early_stopping_patience="${PATIENCE}"
          done
        done
      done
      ;;

    k)
      # Joint head rates × lambda_dyn. The clip binds, so rates are step sizes and lambda_dyn only splits a fixed-norm step.
      echo "# ===== joint: lr_phi x lr_alpha_kappa x lr_beta x lambda_dyn ====="
      for ds in ${DATASETS}; do
        for lp in 3.0e-4 1.0e-3 3.0e-3; do
          for ak in 1.0e-4 1.0e-3 3.0e-3; do
            for lb in 1.0e-3 3.0e-3 1.0e-2; do
              for lam in 1 100 1000; do
                emit "phi$(tag "${lp}")_ak$(tag "${ak}")_b$(tag "${lb}")_lam${lam}" "${ds}" \
                     $(sched "${WARMUP_EPOCHS}") \
                     --set n_epochs="${N_EPOCHS}" \
                     --set dyn_warmup_epochs="${DYN_WARMUP}" \
                     $(head_rates "${lp}" "${ak}" "${lb}") \
                     --set lambda_dyn="${lam}" \
                     --set early_stopping_patience="${PATIENCE}"
              done
            done
          done
        done
      done
      ;;

    base)
      # Method-vs-method, not tuned-vs-default. totalvi is a paired-latent ceiling, not a competitor.
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
      # SCOT's protocol is a grid over k and eps with unsupervised selection, not a single fixed pair.
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
      # emit reads SEEDS at call time; this is the last block in the tier.
      SEEDS="42,123,2026"
      echo "# ===== glue / uniport: 3-seed screen, one axis each ====="
      for ds in ${DATASETS}; do
        for la in 0.01 0.05 0.2; do
          emit_model glue "glue_lalign$(tag "${la}")" "${ds}" --set lam_align=${la}
        done
        for lg in 0.01 0.05; do          # config holds lam_graph=0.02
          emit_model glue "glue_lgraph$(tag "${lg}")" "${ds}" --set lam_graph=${lg}
        done
        for lot in 0.5 1.0 2.0; do
          emit_model uniport "uniport_lot$(tag "${lot}")" "${ds}" --set lambda_ot=${lot}
        done
        for lkl in 0.02 0.2; do          # config holds lambda_kl=0.05
          emit_model uniport "uniport_lkl$(tag "${lkl}")" "${ds}" --set lambda_kl=${lkl}
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
