#!/bin/bash
#SBATCH --account=core-med1-telem
#SBATCH --partition=jobs-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=03:00:00
#SBATCH --job-name=papalexi_bench
#SBATCH --output=logs/slurm%j.out
#SBATCH --error=logs/slurm%j.err
#
# CRISPR perturbation-response benchmark on the OOD predictions. GPU because the
# knn_control path is O(n^2) per method-seed over 20156 cells; KOT itself takes the
# direct phi(r) path and needs none of it, but the baselines in the same directory do.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
for cfg in cfgC cfgB; do
  echo "########## ${cfg} ##########"
  singularity exec --nv --pwd "$(pwd)" /data/common/images/codedev_v1.0.5.sif /bin/bash -lc \
    "PYTHONPATH=. MPLBACKEND=Agg python -u tools/papalexi_perturbation_benchmark.py \
       --predictions data/predictions_papalexi_${cfg} \
       --out-dir cache/results/papalexi_perturbation_${cfg}"
done
