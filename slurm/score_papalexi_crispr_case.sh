#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

stamp="${1:?usage: score_papalexi_crispr_case.sh <run-stamp> <case-index>}"
case_index="${2:?usage: score_papalexi_crispr_case.sh <run-stamp> <case-index>}"
cases=(cfgB_lam100 cfgB_lam1000 cfgC_lam100 cfgC_lam1000 cfgB_lam1000_gaugeOn)
if [ "${case_index}" -lt 0 ] || [ "${case_index}" -ge "${#cases[@]}" ]; then
  echo "invalid case index: ${case_index}" >&2
  exit 2
fi
label="${cases[${case_index}]}"
run_dir="cache/training/run_${stamp}_crisprabl_${label}"
python -u run_kot_crispr.py predict \
  --run-dir "${run_dir}" \
  --dataset papalexi_nt_only \
  --model kot \
  --checkpoint best_val_align \
  --device cuda
