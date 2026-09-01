#!/bin/bash
#SBATCH --account=core-med1-telem
#SBATCH --partition=jobs-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=192GB
#SBATCH --time=01:00:00
#SBATCH --job-name=probe_shuf
#SBATCH --output=logs/probeshuf_%j.log
#SBATCH --error=logs/probeshuf_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

# The preprocessing cache is keyed by a config hash, so a pinned path quietly goes stale
# the moment the velocity is regenerated -- this script probed the superseded BMMC field
# for weeks that way. Resolve the newest cache instead, and fail loudly if there is none.
newest_cache() {
  local match
  match=$(ls -t cache/preprocessed/"$1"_*.rna.h5ad 2>/dev/null | head -1)
  [ -n "${match}" ] || { echo "no preprocessed cache for $1 -- run preprocessing first" >&2; exit 1; }
  echo "${match}"
}
PBMC=$(newest_cache pbmc_scvelo_results_retained)
BMMC=$(newest_cache bmmc_cite_scvelo_results_retained)
echo "probing: ${PBMC} and ${BMMC}"
# The twelve seeds config/training.yaml gives both panels.
SEEDS=42,123,2026,6,9,11,17,21,33,77,88,101

singularity exec --pwd "$(pwd)" /data/common/images/codedev_v1.0.5.sif /bin/bash -lc \
  "PYTHONPATH=. python -u tools/velocity_checks.py shuffle-strength ${PBMC} PBMC ${SEEDS} && \
   PYTHONPATH=. python -u tools/velocity_checks.py shuffle-strength ${BMMC} BMMC ${SEEDS}"
