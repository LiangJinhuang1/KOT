#!/bin/bash
# Submit chromatin→RNA experiments in protocol order.
#
# Usage:
#   bash slurm/submit_chromatin_phase.sh pilot
#   bash slurm/submit_chromatin_phase.sh r1-ablation
#   bash slurm/submit_chromatin_phase.sh relay
#   bash slurm/submit_chromatin_phase.sh development
#   bash slurm/submit_chromatin_phase.sh final-central
#   bash slurm/submit_chromatin_phase.sh final-all
#
# An optional second argument is a comma-separated seed list. The pilot deliberately
# accepts one seed only. Later phases refuse to submit until all four pilot pass markers
# exist, so shuffle/reverse/zero/permG and R2 cannot accidentally precede the gate.
set -euo pipefail
cd "$(dirname "$0")/.."

phase="${1:?usage: $0 PHASE [seed_csv]}"
seed_csv="${2:-}"
pilot_seed=42
# v1 used a frozen affine JVP and a permissive FOSCTTM < 0.40 gate. Never let those
# stale pass markers unlock experiments trained under the corrected protocol.
pilot_tag="pilot_v2"
dev_seeds="42,123,2026"
final_seeds="42,123,2026,6,9,11,17,21,33,77,88,101"

pilot_dir() {
  local dataset="$1" condition="$2"
  echo "cache/chromatin/runs/${pilot_tag}_${dataset}_reduced_${condition}_seed${pilot_seed}"
}

require_pilot() {
  local dataset condition marker missing=0
  for dataset in hspc bmmc; do
    for condition in full noDyn; do
      marker="$(pilot_dir "$dataset" "$condition")/preflight_passed.json"
      if [[ ! -s "$marker" ]]; then
        echo "missing passed pilot: $marker" >&2
        missing=1
      fi
    done
  done
  if [[ "$missing" -ne 0 ]]; then
    echo "Run 'bash slurm/submit_chromatin_phase.sh pilot' and wait for all four jobs." >&2
    exit 2
  fi
}

require_evaluated() {
  local tag="$1" law="$2" conditions="$3" seeds="$4"
  local dataset seed condition path missing=0
  IFS=',' read -r -a required_seeds <<< "$seeds"
  for dataset in hspc bmmc; do
    for seed in "${required_seeds[@]}"; do
      for condition in $conditions; do
        path="cache/chromatin/runs/${tag}_${dataset}_${law}_${condition}_seed${seed}/evaluation_best_align.json"
        if [[ ! -s "$path" ]]; then
          echo "missing evaluated prerequisite: $path" >&2
          missing=1
        fi
      done
    done
  done
  if [[ "$missing" -ne 0 ]]; then
    echo "Wait for the prerequisite phase and evaluate every checkpoint before continuing." >&2
    exit 2
  fi
}

submit_one() {
  local tag="$1" dataset="$2" law="$3" condition="$4" seed="$5" run_dir
  if [[ "$tag" == "pilot" ]]; then
    run_dir="$(pilot_dir "$dataset" "$condition")"
  else
    run_dir="cache/chromatin/runs/${tag}_${dataset}_${law}_${condition}_seed${seed}"
  fi
  if [[ -s "$run_dir/evaluation_best_align.json" ]]; then
    echo "[skip] evaluated: $run_dir"
    return
  fi
  local command
  command="PYTHONPATH=. python -u run_kot_chromatin.py train --dataset ${dataset} --law ${law} --condition ${condition} --seed ${seed} --lambda-dyn 1 --run-dir ${run_dir} && PYTHONPATH=. python -u run_kot_chromatin.py evaluate --run-dir ${run_dir}"
  sbatch --job-name="chr_${dataset}_${law}_${condition}_${seed}" \
    --export=ALL,RUN_CMD="$command" slurm/train_slurm.sh
}

submit_grid() {
  local tag="$1" seeds="$2" reduced_conditions="$3" relay_conditions="$4"
  local dataset seed condition
  IFS=',' read -r -a seed_array <<< "$seeds"
  for dataset in hspc bmmc; do
    for seed in "${seed_array[@]}"; do
      for condition in $reduced_conditions; do
        submit_one "$tag" "$dataset" reduced "$condition" "$seed"
      done
      for condition in $relay_conditions; do
        submit_one "$tag" "$dataset" relay "$condition" "$seed"
      done
    done
  done
}

case "$phase" in
  pilot)
    seeds="${seed_csv:-$pilot_seed}"
    if [[ "$seeds" == *,* ]]; then
      echo "pilot accepts exactly one seed" >&2
      exit 2
    fi
    if [[ "$seeds" != "$pilot_seed" ]]; then
      echo "pilot seed is frozen at $pilot_seed so later phases can verify one location" >&2
      exit 2
    fi
    for dataset in hspc bmmc; do
      for condition in full noDyn; do
        submit_one pilot "$dataset" reduced "$condition" "$pilot_seed"
      done
    done
    ;;
  r1-ablation)
    require_pilot
    [[ -z "$seed_csv" || "$seed_csv" == "$pilot_seed" ]] || {
      echo "r1-ablation is the frozen one-seed pilot phase (seed $pilot_seed)" >&2; exit 2; }
    submit_grid r1pilot "$pilot_seed" "shuffle reverse zero permG" ""
    ;;
  relay)
    require_pilot
    require_evaluated r1pilot reduced "shuffle reverse zero permG" "$pilot_seed"
    [[ -z "$seed_csv" || "$seed_csv" == "$pilot_seed" ]] || {
      echo "relay is the frozen one-seed pilot phase (seed $pilot_seed)" >&2; exit 2; }
    submit_grid r2pilot "$pilot_seed" "" "full shuffle noDyn"
    ;;
  development)
    require_pilot
    require_evaluated r1pilot reduced "shuffle reverse zero permG" "$pilot_seed"
    require_evaluated r2pilot relay "full shuffle noDyn" "$pilot_seed"
    submit_grid dev "${seed_csv:-$dev_seeds}" \
      "full shuffle reverse zero noDyn permG" "full shuffle noDyn"
    ;;
  final-central)
    require_pilot
    require_evaluated dev reduced "full shuffle reverse zero noDyn permG" "$dev_seeds"
    require_evaluated dev relay "full shuffle noDyn" "$dev_seeds"
    submit_grid finalcentral "${seed_csv:-$final_seeds}" \
      "full shuffle noDyn" "full shuffle noDyn"
    ;;
  final-all)
    require_pilot
    require_evaluated dev reduced "full shuffle reverse zero noDyn permG" "$dev_seeds"
    require_evaluated dev relay "full shuffle noDyn" "$dev_seeds"
    submit_grid finalall "${seed_csv:-$final_seeds}" \
      "full shuffle reverse zero noDyn permG" "full shuffle noDyn"
    ;;
  *)
    echo "unknown phase: $phase" >&2
    exit 2
    ;;
esac
