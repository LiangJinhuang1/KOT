#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

stamp="${1:?usage: score_papalexi_crispr_lambda_sweep.sh <run-stamp>}"
lambdas=(100 1000)
predictions=()
observational=()

for cfg in cfgB cfgC; do
  for lambda in "${lambdas[@]}"; do
    run_dir="cache/training/run_${stamp}_crisprabl_${cfg}_lam${lambda}"
    python -u run_kot_crispr.py predict \
      --run-dir "${run_dir}" \
      --dataset papalexi_nt_only \
      --model kot \
      --checkpoint best_val_align
    predictions+=("${run_dir}/crispr_predictions_best_val_align.csv")
    observational+=("${run_dir}/crispr_observational_best_val_align.csv")
  done
done

gauge_run="cache/training/run_${stamp}_crisprabl_cfgB_lam1000_gaugeOn"
python -u run_kot_crispr.py predict \
  --run-dir "${gauge_run}" \
  --dataset papalexi_nt_only \
  --model kot \
  --checkpoint best_val_align
predictions+=("${gauge_run}/crispr_predictions_best_val_align.csv")
observational+=("${gauge_run}/crispr_observational_best_val_align.csv")

python -u run_kot_crispr.py evaluate \
  --effects cache/results/crispr/crispr_effects_observed.csv \
  --predictions "${predictions[@]}" \
  --observational "${observational[@]}" \
  --out-dir "cache/results/crispr/lambda_sweep_${stamp}"
