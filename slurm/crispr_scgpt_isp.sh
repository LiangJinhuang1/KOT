#!/bin/bash
#SBATCH --account=core-med1-telem
#SBATCH --partition=jobs-gpu-long
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=crispr_scgpt_isp
#SBATCH --output=logs/crispr_scgpt_isp_%A_%a.log
#SBATCH --array=0-2%3
# Submit: sbatch slurm/crispr_scgpt_isp.sh FROZEN_KOT_RUN NEW_OUTPUT_ROOT
# scGPT lives in cache/.scgpt_venv, a --system-site-packages venv over
# codedev_v1.0.5.sif. Never install it into the container: it pins scvi-tools 0.20.3
# against the container's 1.2.0 and would break RegVelo.
# Weights are the scGPT_human release (33M cells) mirrored on HuggingFace, fetched to
# cache/scgpt: args.json, vocab.json, best_model.pt (208 MB) from
#   https://huggingface.co/MohamedMabrouk/scGPT/resolve/main/<file>
# scGPT reuses the GEARS PertData path, so the same three Dataverse assets must be
# staged in the per-seed cache; see slurm/crispr_gears_isp.sh for the curl commands.
set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then cd "$SLURM_SUBMIT_DIR"; else cd "$(dirname "$0")/.."; fi
run_dir="${1:?Provide frozen NT-only KOT run directory}"
out_root="${2:?Provide a new output root}"
seeds=(42 123 2026)
seed="${seeds[${SLURM_ARRAY_TASK_ID:-0}]}"
fold="$out_root/seed_$seed"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
container=/data/common/images/codedev_v1.0.5.sif
data_dir="cache/scgpt_data/seed_$seed"
mkdir -p "$data_dir"
for asset in gene2go_all.pkl essential_all_data_pert_genes.pkl go_essential_all.tar.gz; do
    [[ -s "$data_dir/$asset" ]] || cp "cache/gears/$asset" "$data_dir/"
done
[[ -d "$data_dir/go_essential_all" ]] || tar xzf "$data_dir/go_essential_all.tar.gz" -C "$data_dir/"
singularity exec --nv --pwd "$PWD" "$container" cache/.scgpt_venv/bin/python -u \
    run_crispr_isp.py scgpt \
    --run-dir "$run_dir" --seed "$seed" --out-dir "$fold" \
    --data-dir "$data_dir" --epochs "${SCGPT_EPOCHS:-15}" --device cuda
singularity exec --nv --pwd "$PWD" "$container" python -u run_kot_crispr.py task_b \
    --predicted-rna "$fold/predicted_rna.csv" --isp-model scgpt \
    --run-dir "$run_dir" --checkpoint best_align --seed "$seed" \
    --n-boot 1000 --out-dir "$fold"
