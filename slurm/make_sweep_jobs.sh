#!/bin/bash
#
# Generate a hyperparameter-sweep jobs file for slurm/parallel_train.sh.
#
# This is NOT slurm/make_jobs.sh: that one generates the fixed paper runs (all three
# KOT arms, 12 seeds, canonical weights). This one generates a SEARCH -- one arm
# (kot_main), 4 seeds, one run-dir per config, and --no-predictions on every line
# so parallel jobs cannot collide on the global data/predictions/ HDF5 path.
#
# RANKED ON HELD-OUT CELLS. Every search round is read on val_foscttm: the fixed
# validation slice (config/training.yaml val_fraction -- 20% of the small panels, 10%
# of BMMC) is held out of both loss terms, and mean_foscttm is measured on cells the
# run trained on, so ranking configs by it is ranking them on the test set. The final
# paper runs put those cells back (--set val_holdout_from_training=false) and report
# mean_foscttm unchanged.
#
# Tiers a-e were the open grid, run before anything was fixed; they are done and were
# deleted. `git log slurm/make_sweep_jobs.sh` has them, and each run's own
# run_config.yaml is the actual record of what it ran.
#
# FOUR ROUNDS, in order. Each freezes what the previous one settled and takes it as a
# flag, so a round's job lines carry the whole resolved config and nothing is implicit:
#
#   bash slurm/make_sweep_jobs.sh --tier f                       # lr warmup x early stopping (16 jobs)
#   bash slurm/make_sweep_jobs.sh --tier k0                      # warmup x run length      (12 jobs)
#   bash slurm/make_sweep_jobs.sh --tier k  --warmup <w> --n-epochs <e>
#                                                                # joint lr x lambda_dyn   (162 jobs)
#   bash slurm/make_sweep_jobs.sh --tier i  --warmup <w> <lr flags>  # fine lambda ladder   (14 jobs)
#   bash slurm/make_sweep_jobs.sh --tier j  <same flags>          # lambda ramp x lambda_dyn (28 jobs)
#   bash slurm/make_sweep_jobs.sh --tier base                    # baselines               (80 jobs)
#
# Rounds g and h are SUPERSEDED by k and kept only so old jobs files stay readable.
# They split the three head rates over two sequential rounds; k runs all three against
# lambda_dyn in one grid, because the gradient clip binds every step and makes those
# axes interact (see the tier-k comment). Round i still runs, after k, to put the fine
# 1..1000 lambda ladder on the LR triple k settles.
#
# Round f comes first because the two things it varies change what every later round
# measures. It also settles an open question about early stopping: across the 66 runs
# that carry a run_config.yaml, the train_align criterion fired ONCE -- the training
# align loss is still falling at epoch 500 and best_align sits a median of 28 epochs
# before the end -- so patience on it is inert and `final` and `best_align` are nearly
# the same model. val_align (the Sinkhorn on the held-out cells, still label-free) is
# the criterion that can actually fire, and round f is where the two are compared.
# Patience is 100 in BOTH arms so the round varies what is watched, not how long.
#
#   sbatch --export=ALL,JOBS_FILE=jobs_sweep_f.txt,MAX_PARALLEL=16 slurm/parallel_train.sh
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

# Frozen-so-far settings, each overridable by the round that has not settled it yet.
# The defaults are the config defaults -- lr 1e-3 on every head, which is where round f
# runs and where round g's grid is centred.
WARMUP_EPOCHS=10         # lr warmup length; round f's axis, frozen from round g on
LR_PHI="1.0e-3"
LR_ALPHA_KAPPA="1.0e-3"
LR_BETA="1.0e-3"
DYN_WARMUP=50            # kinetics ramp length (settled by tier d); round j's axis
LAMBDA_DYN=100           # canonical weight; round i's axis
PATIENCE=100
N_EPOCHS=500             # run length; round k0's second axis, frozen from round k on
# Appended to every cfg tag, so one tier can be generated twice at different frozen
# settings and the two files concatenated without their run-dirs colliding. The chain
# driver uses it to put round i's lambda ladder on two different lr triples at once.
TAG_SUFFIX=""
# Round k0's two axes, overridable so the round can be re-run further out when its
# winner lands on the edge of the grid -- which it did twice: round f topped out at
# warmup 20 of {0,1,10,20}, and round k0 at 100 of {20,50,100}.
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
    -h|--help)  sed -n '2,55p' "$0"; exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 1 ;;
  esac
done

case "${TIER}" in
  f|g|h|i|j|k0|k|base) ;;
  "")  echo "--tier f|g|h|i|j|k0|k|base is required" >&2; exit 1 ;;
  *)   echo "unknown tier: ${TIER} (expected f, g, h, i, j, k0, k or base)" >&2; exit 1 ;;
esac
OUT="${OUT:-jobs_sweep_${TIER}.txt}"

# The lr SHAPE around the warmup length: linear from 0.1 up over WARMUP_EPOCHS, then
# cosine down to 0.01, as one multiplicative factor on all three heads so their ratios
# hold for the whole run. With warmup 0 the start factor is unused and this is the plain
# cosine the code has always run.
sched() {
  echo "--set lr_warmup_epochs=$1 --set lr_warmup_start_factor=0.1 --set lr_min_factor=0.01"
}

# The three head rates, named explicitly rather than as a multiplier on a remembered
# ratio, so a job line says what it ran without a lookup.
head_rates() {
  echo "--set lr_phi=$1 --set lr_alpha_kappa=$2 --set lr_beta=$3"
}

# %g, the same form run_config.yaml round-trips through and collect_run_diagnostics.py
# formats cells with -- so a --paired-vs value written here matches the column exactly.
gnum() { awk -v v="$1" 'BEGIN{printf "%g", v}'; }

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
  echo "--datasets ${dataset} --models ${MODELS} ${seed_flag}--no-predictions $* --run-dir ${ROOT}/run_${STAMP}_sw${TIER}_${short}_${cfgtag}${TAG_SUFFIX}"
}

# The command that reads this round, with the in-grid control as the paired baseline.
# Rounds are ranked on val_foscttm; --rank-by picks that automatically once the runs
# carry one, and it is named here so the generated file is explicit about it.
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
  # mean_foscttm, NOT val_foscttm, and deliberately so: scot, moscot and linear_ode
  # build a coupling over the two full point sets, so they have no out-of-sample
  # embedding and cannot be scored on held-out cells at all. Selecting them on the full
  # metric is the only option available; it favours the baselines, which keeps the KOT
  # comparison conservative. Said out loud here so it is a choice, not a silent fallback.
  base) COLLECT="python tools/collect_run_diagnostics.py ${COLLECT_RUNS} \
             --rank-by mean_foscttm --aggregate" ;;
esac

{
  echo "# Sweep tier ${TIER} -- generated by slurm/make_sweep_jobs.sh on $(date +%Y-%m-%d)."
  echo "# seeds=${SEEDS:-from config/training.yaml}  models=${MODELS}  datasets=${DATASETS}"
  echo "# frozen: warmup=${WARMUP_EPOCHS}  lr=(phi ${LR_PHI}, alpha/kappa ${LR_ALPHA_KAPPA}, beta ${LR_BETA})"
  echo "#         dyn_warmup=${DYN_WARMUP}  lambda_dyn=${LAMBDA_DYN}  patience=${PATIENCE}"
  echo "#         n_epochs=${N_EPOCHS}"
  # The split regime is not a sweep axis, but it decides what val_foscttm MEANS, so a
  # jobs file says which one it ran under. Tier f and everything before it used one
  # fixed split for every seed; from tier k0 on the split is redrawn per model seed
  # (and stratified by cell_type on BMMC), so val_foscttm from the two eras must not
  # be pooled into one paired table. See config/training.yaml val_split_per_seed.
  echo "#         split: per-seed (val_split_per_seed), stratified where configured"
  echo "#"
  echo "# Submit:  sbatch --export=ALL,JOBS_FILE=${OUT},MAX_PARALLEL=16 slurm/parallel_train.sh"
  # Every line of it: COLLECT wraps across lines, and a continuation without its own
  # `#` is a live job line to anything reading this file.
  echo "# Collect:"
  echo "${COLLECT}" | sed 's/^/#   /'
  echo

  case "${TIER}" in

    f)
      # Round 0: the lr warmup length x what early stopping watches.
      #
      # warmup is in EPOCHS, and an epoch here is ONE optimiser step -- the kinetics
      # minibatches accumulate into a single step, so a 500-epoch run takes 500 steps.
      # A 1-epoch warmup is therefore literally one step at 0.1x lr; it is in the grid
      # as the control the request asked for, not because it is expected to move
      # anything. 20 epochs is 4% of the run, 10 is what tiers b-d ran at.
      #
      # The monitor axis is not a duration knob: patience is 100 in both arms, so the
      # only difference is whether the criterion is the training align loss (which has
      # never fired) or the held-out Sinkhorn (which can). Both write best_align AND
      # best_val_align checkpoints, so the round also answers which checkpoint to keep.
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
      # Round 1a: learning rate, with phi held at 1e-3 and the other two heads scanned
      # as multiples of it. Splitting the three rates over two rounds is the request's
      # own shape ("keep one of those learning rates constant at 1e-3"), and it is also
      # what keeps the round at 16 cells instead of 64.
      #
      # lr matters more than a grid this size usually would: clip_grad_norm_(max_norm=1)
      # runs on every step and the PRE-clip norm sits 100-6000x above that budget, so
      # the clip always binds, the post-clip gradient has norm exactly 1, and lr is the
      # entire step size.
      #
      # The beta axis leans up because tier d measured lr_beta=5e-3 recovering the beta
      # anchors 11x better than 1e-3, and beta has a travel cap of ~0.14 from init at a
      # shared rate. The alpha/kappa axis leans down because that is where tier d put it
      # (1e-4); this round re-tests that under the round-f schedule rather than assuming
      # it, since the cell (0.0001, 0.001) is in the grid.
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
      # Round 1b: the phi rate, at the alpha/kappa and beta rates round g settled.
      # phi is the alignment map and therefore the head the FOSCTTM actually rides on,
      # which is the reason it gets a round to itself rather than a third grid axis.
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
      # Round 2: the kinetics weight at the frozen lr.
      #
      # 1-500 is the requested range. 1000 is added because it is the incumbent: tier a
      # measured lambda_dyn=1000 beating the canonical 100 on pbmc across 5 seeds (0.18
      # vs 0.21-0.23) -- at the EDGE of that grid, which is a direction rather than an
      # optimum, but leaving it out would mean the round cannot see the value it has to
      # beat. If 1000 wins again the range has to move up, not down.
      #
      # lambda_dyn is a MIXING coefficient, not a scale: the clip binds every step, so
      # it decides how a fixed-norm step is split between alignment and kinetics. That
      # is what grad_mag_ratio_late in the table measures, and it is how to tell a real
      # win from one bought by letting the dynamics go slack.
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
      # Round 3: the kinetics RAMP against the same lambda_dyn values, which is the
      # ablation figure -- lambda_dyn on x, one line per ramp length.
      #
      # Only the ramp lengths round i did NOT run are here; the collect command pools
      # this tier's runs with round i's, which supply the dyn_warmup=${DYN_WARMUP} row.
      # Re-running that row would just be the same 14 jobs a second time.
      #
      # The ramp interacts with model selection: checkpointing and early stopping only
      # start once the ramp is done (run_kot gates on it), because while the kinetics
      # weight is still climbing the align loss is the lowest it will ever be, and a
      # best_align recorded there would hand back a model carrying far less dynamics
      # than it trained with -- one that ranks WELL on FOSCTTM. A ramp of 0 has no such
      # dead zone; a ramp of 100 spends a fifth of a 500-epoch run in one.
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
      # Round 0b: how long to warm up, and how long to run.
      #
      # Round f ranked warm20 first on BOTH datasets (bmmc 0.424 -> 0.208, pbmc 0.450 ->
      # 0.384) at the EDGE of its 0/1/10/20 grid, so that is a direction, not an optimum.
      # This round walks it out to 100 and pairs it against n_epochs, because the two are
      # not separable: lr_warmup_epochs sets the cosine's span as n_epochs - warmup, so
      # warm20 at 500 epochs and warm20 at 1000 are different schedules, not the same
      # schedule run longer.
      #
      # n_epochs is here because it is the only real length knob. Round f showed early
      # stopping is inert at patience=100 -- it fired in 1 of 16 runs, and the other 15
      # ran all 500 epochs on every seed -- and the training align loss is still falling
      # at 500. Patience stays at ${PATIENCE} in every cell so the round varies the run
      # length, not what ends it, and every cell is a fixed-length run.
      #
      # The early_stopping_monitor axis from round f is NOT repeated. It cost half that
      # round and returned nothing: 6 of its 8 config pairs came back identical to 6
      # decimal places, because with patience=100 the criterion never got to fire.
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
      # Round 1: the three head rates and lambda_dyn, JOINTLY, replacing rounds g/h/i.
      #
      # Rounds g -> h -> i are coordinate descent over exactly the axes that are least
      # separable here. clip_grad_norm_(max_norm=1) binds on every step (pre-clip norms
      # sit 100-6000x above the budget), so the post-clip gradient has norm exactly 1:
      # the head rates ARE the step sizes, and lambda_dyn only decides how that
      # fixed-norm step splits between alignment and kinetics. lr_beta / lr_alpha_kappa
      # and lambda_dyn therefore push on the same quantity from two sides, and sweeping
      # one with the others frozen finds the optimum of a slice, not of the surface.
      #
      # Running them together is affordable because they are cheap. Measured on round f
      # at 16-way on 2 GPUs: 7.3 min per 4-seed pbmc job, 28.5 min per bmmc job. The
      # 81-cell grid is ~3 h of an allocation that may run 7.
      #
      # The grid is the config incumbent (1e-3, 1e-3, 1e-3, 100) with one coarse step
      # either side, so the control is IN the grid and --paired-vs has a baseline.
      # beta leans high because tier d measured lr_beta=5e-3 recovering the beta anchors
      # 11x better than 1e-3; alpha/kappa leans low because tier d put it at 1e-4.
      # lambda_dyn spans its whole 1-1000 range in three log steps here -- the fine
      # ladder (1/50/100/200/300/500/1000) is round i's job, run afterwards at the LR
      # triple this round settles, where the extra resolution can actually be read.
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
      # glue / uniport ARE seeded, so they are screened at 3 seeds, one axis at a time.
      # Screening only has to find large differences; precision is for the final
      # confirmation, which must use the same 12 seeds as KOT's headline number.
      #
      # Two axes each rather than one. uniPort gets lambda_kl because config/training.yaml
      # moved it off the package default (0.05, to stop the VAE collapsing on low-dim PCA
      # inputs), and a hand-picked value nobody scanned is exactly what a reviewer asks
      # about; the range stays near it rather than climbing to the value that collapses.
      # GLUE gets lam_graph for the same reason. The second axis skips the value the
      # config already holds, so no cell is run twice.
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
