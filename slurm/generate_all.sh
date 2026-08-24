#!/bin/bash
#
# Generate the staged synthetic linked-ODE data inside the project container.
# Each stage writes one RNA + one protein h5ad (both mean and log layers inside).
#
# Usage:
#   sbatch generate_all.sh                                # all three stages
#   sbatch --export=ALL,STAGES="clean branch" generate_all.sh   # only some stages
#
# (Generation is CPU-only; the GPU request just mirrors the known-good partition.)
#
#SBATCH --account=core-med1-telem
#SBATCH --partition=jobs-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64GB
#SBATCH --time=2:00:00
#SBATCH --job-name=gen_synth
#SBATCH --output=logs/slurm%j.log
#SBATCH --error=logs/slurm%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

STAGES="${STAGES:-oracle clean branch}"
export STAGES

echo "Generating stages: ${STAGES}"
echo "Node: $(hostname)  Start: $(date)"

CONTAINER="/data/common/images/codedev_v1.0.5.sif"

srun --cpu-bind=none singularity exec --nv --pwd "$(pwd)" "${CONTAINER}" /bin/bash -lc '
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"
for STAGE in ${STAGES}; do
  echo "===== generating stage: ${STAGE} ====="
  python -m src.data.synthetic_linked_ode --stage "${STAGE}"
done
'

echo "Done: ${STAGES}  End: $(date)"
