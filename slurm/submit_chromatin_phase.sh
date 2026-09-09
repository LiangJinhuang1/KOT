#!/bin/bash
# Regulatory R2 only. Shuffle waits for both gates. Existing runs are preserved.
# Usage: bash slurm/submit_chromatin_phase.sh pilot|shuffle [hspc|bmmc|both] [--dry-run]
set -euo pipefail
cd "$(dirname "$0")/.."

phase="${1:?usage: $0 pilot|shuffle [hspc|bmmc|both] [--dry-run]}"
dataset_arg="${2:-both}"
dry_run="${3:-}"
tag="regulatory_r2_v1"
anchor="config/gamma_anchors_k562_timelapse.csv"

case "$phase" in
  pilot) conditions="full noDyn" ;;
  shuffle) conditions="shuffle" ;;
  *) echo "Retired or unknown phase '$phase'; use pilot or shuffle." >&2; exit 2 ;;
esac
case "$dataset_arg" in
  hspc|bmmc) datasets="$dataset_arg" ;;
  both) datasets="hspc bmmc" ;;
  *) echo "Unknown dataset '$dataset_arg'" >&2; exit 2 ;;
esac
if [[ -n "$dry_run" && "$dry_run" != "--dry-run" ]] || [[ $# -gt 3 ]]; then
  echo "usage: $0 pilot|shuffle [hspc|bmmc|both] [--dry-run]" >&2
  exit 2
fi
[[ -s "$anchor" ]] || { echo "Missing RNA anchors: $anchor" >&2; exit 2; }

run_directory() {
  echo "cache/chromatin/runs/${tag}_${1}_relay_${2}_seed42"
}

for dataset in $datasets; do
  if [[ "$phase" == "shuffle" ]]; then
    for condition in full noDyn; do
      marker="$(run_directory "$dataset" "$condition")/preflight_passed.json"
      [[ -s "$marker" ]] || { echo "Missing passed R2 pilot: $marker" >&2; exit 2; }
    done
  fi
  for condition in $conditions; do
    run_dir="$(run_directory "$dataset" "$condition")"
    [[ ! -e "$run_dir" ]] || { echo "Preserving existing run: $run_dir" >&2; exit 2; }
  done
done

for dataset in $datasets; do
  for condition in $conditions; do
    run_dir="$(run_directory "$dataset" "$condition")"
    command="PYTHONPATH=. python -u run_kot_chromatin.py check-r2 --dataset ${dataset} --gamma-anchor-csv ${anchor} && PYTHONPATH=. python -u run_kot_chromatin.py train --dataset ${dataset} --law relay --condition ${condition} --seed 42 --split-seed 0 --align-block spliced --lambda-held-block 1 --lambda-dyn 1 --gamma-anchor-csv ${anchor} --lambda-gamma-anchor 1 --run-dir ${run_dir}"
    if [[ "$dry_run" == "--dry-run" ]]; then
      printf '%s\n' "$command"
    else
      mkdir -p logs
      sbatch --job-name="chr_r2_${dataset}_${condition}" \
        --export=ALL,RUN_CMD="$command" slurm/train_slurm.sh
    fi
  done
done
