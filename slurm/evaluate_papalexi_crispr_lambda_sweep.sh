#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

stamp="${1:?usage: evaluate_papalexi_crispr_lambda_sweep.sh <run-stamp>}"
cases=(cfgB_lam100 cfgB_lam1000 cfgC_lam100 cfgC_lam1000 cfgB_lam1000_gaugeOn)
predictions=()
observational=()
for label in "${cases[@]}"; do
  run_dir="cache/training/run_${stamp}_crisprabl_${label}"
  predictions+=("${run_dir}/crispr_predictions_best_val_align.csv")
  observational+=("${run_dir}/crispr_observational_best_val_align.csv")
done
python -u run_kot_crispr.py evaluate \
  --effects cache/results/crispr/crispr_effects_observed.csv \
  --predictions "${predictions[@]}" \
  --observational "${observational[@]}" \
  --out-dir "cache/results/crispr/lambda_sweep_${stamp}"
