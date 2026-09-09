#!/bin/bash
#SBATCH --account=core-med1-telem
#SBATCH --partition=jobs-gpu-long
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=crispr_gears_isp
#SBATCH --output=logs/crispr_gears_isp_%A_%a.log
#SBATCH --array=0-2%3
# Submit: sbatch slurm/crispr_gears_isp.sh FROZEN_KOT_RUN NEW_OUTPUT_ROOT
# Never pip-install GEARS into the container; its resolver wants numpy 2.5 and would break KOT and RegVelo.
# GEARS downloads from Dataverse with a User-Agent that gets 403; fetch with curl:
#   curl -sL -o cache/gears/gene2go_all.pkl \
#     https://dataverse.harvard.edu/api/access/datafile/6153417
#   curl -sL -o cache/gears/essential_all_data_pert_genes.pkl \
#     https://dataverse.harvard.edu/api/access/datafile/6934320
#   curl -sL -o cache/gears/go_essential_all.tar.gz \
#     https://dataverse.harvard.edu/api/access/datafile/6934319
# new_data_process needs more memory than the login node has, which is why this is a batch job.
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
# Each seed its own GEARS cache because new_data_process reuses a processed dataset in place.
mkdir -p "cache/gears/seed_$seed"
for asset in gene2go_all.pkl essential_all_data_pert_genes.pkl go_essential_all.tar.gz; do
    [[ -s "cache/gears/seed_$seed/$asset" ]] || cp "cache/gears/$asset" "cache/gears/seed_$seed/"
done
# The GO graph is read from the extracted directory, not the tarball.
[[ -d "cache/gears/seed_$seed/go_essential_all" ]] || \
    tar xzf "cache/gears/seed_$seed/go_essential_all.tar.gz" -C "cache/gears/seed_$seed/"
singularity exec --nv --pwd "$PWD" "$container" cache/.gears_venv/bin/python -u \
    run_crispr_isp.py gears \
    --run-dir "$run_dir" --seed "$seed" --out-dir "$fold" \
    --data-dir "cache/gears/seed_$seed" \
    --epochs "${GEARS_EPOCHS:-20}" --device cuda
singularity exec --nv --pwd "$PWD" "$container" python -u run_kot_crispr.py task_b \
    --predicted-rna "$fold/predicted_rna.csv" --isp-model gears \
    --run-dir "$run_dir" --checkpoint best_align --seed "$seed" \
    --n-boot 1000 --out-dir "$fold"
