#!/bin/bash
#SBATCH --account=core-med1-telem
#SBATCH --partition=jobs-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=192GB
#SBATCH --time=04:00:00
#SBATCH --job-name=protein_eval
#SBATCH --output=logs/slurm%j.out
#SBATCH --error=logs/slurm%j.err
#
# Held-out protein-prediction quality across the kinetics-ablation runs.
# CPU + memory only: phi is a forward pass and the JVP is one more, but bmmc_cite is
# 89708 cells x 134 proteins and its preprocessed AnnData does not fit the login node.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
singularity exec --pwd "$(pwd)" /data/common/images/codedev_v1.0.5.sif /bin/bash -lc \
  "PYTHONPATH=. MPLBACKEND=Agg python -u tools/evaluate_protein_prediction.py \
     --runs 'cache/training/run_20260829_022744_shuf_*_realVel' \
     --runs 'cache/training/run_20260829_192024_shuf2_*_shufVel' \
     --runs 'cache/training/run_20260831_222311_kinabl_*' \
     --velocity real --out-prefix cache/results/protein_eval"
