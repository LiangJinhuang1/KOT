#!/bin/bash
#SBATCH --account=core-med1-telem
#SBATCH --partition=jobs-gpu
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192GB
#SBATCH --time=04:00:00
#SBATCH --job-name=eval_realvel
#SBATCH --output=logs/realvel_%j.log
#SBATCH --error=logs/realvel_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

# Scores the shuffled arm's checkpoints against the TRUE velocity field: if a
# shuffled-trained model fits the real field as well as the permuted one it trained on,
# phi fitted the field's shared structure and not the per-cell correspondence. This is
# the physics check the JVP-RHS cosine cannot provide. Override RUNS to narrow it.
singularity exec --nv --pwd "$(pwd)" /data/common/images/codedev_v1.0.5.sif /bin/bash -lc \
  "PYTHONPATH=. python -u tools/evaluate_checkpoints.py \
     --runs '${RUNS:-cache/training/run_*_shuf*_*shufVel*}' \
     --velocity real \
     --output cache/results/realvel_summary.csv"
