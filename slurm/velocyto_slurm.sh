#!/bin/bash
#
# Run velocyto on SHARE-seq BAM files to produce spliced/unspliced loom files.
# Output loom files are written to Datasets/SHARE_seq/velocyto/.
#
# Usage:
#   sbatch --array=0-2 velocyto_slurm.sh         # tissue samples (skin, brain, lung)
#   sbatch --array=0-5 velocyto_slurm.sh         # all samples
#
#SBATCH --account=core-med1-telem
#SBATCH --partition=jobs-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=06:00:00
#SBATCH --job-name=velocyto_shareseq
#SBATCH --output=logs/velocyto%A_%a.log
#SBATCH --error=logs/velocyto%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs Datasets/SHARE_seq/velocyto

CONTAINER="/data/common/images/codedev_v1.0.5.sif"

# Paths — update GTF paths if they differ on the cluster.
MM10_GTF="/data/common/references/mm10/Mus_musculus.GRCm38.99.gtf"
HG19_GTF="/data/common/references/hg19/Homo_sapiens.GRCh37.87.gtf"

BAM_DIR="Datasets/SHARE_seq/bam"
RAW_DIR="Datasets/SHARE_seq/Raw_Files"
OUT_DIR="Datasets/SHARE_seq/velocyto"

# Each entry: "bam_stem  rna_counts_gz  gtf_var"
SAMPLES=(
    "skin.late.anagen.rna.norg     GSM4156608_skin.late.anagen.rna.counts.txt.gz    MM10_GTF"
    "brain.rna.norg                GSM4156610_brain.rna.counts.txt.gz               MM10_GTF"
    "lung.rna.norg                 GSM4156611_lung.rna.counts.txt.gz                MM10_GTF"
    "GM12878.rep1.rna.noRG         GSM4156601_GM12878.rep1.rna.counts.txt.gz        HG19_GTF"
    "GM12878.rep2.rna.norg         GSM4156602_GM12878.rep2.rna.counts.txt.gz        HG19_GTF"
    "GM12878.rep3.rna.norg         GSM4156603_GM12878.rep3.rna.counts.txt.gz        HG19_GTF"
)

read -r BAM_STEM COUNTS_GZ GTF_VAR <<< "${SAMPLES[$SLURM_ARRAY_TASK_ID]}"
BAM="${BAM_DIR}/${BAM_STEM}.bam"
COUNTS="${RAW_DIR}/${COUNTS_GZ}"

echo "===== velocyto ====="
echo "Sample : ${BAM_STEM}"
echo "BAM    : ${BAM}"
echo "Counts : ${COUNTS}"
echo "Node   : $(hostname)"
echo "Start  : $(date)"

# Extract cell barcodes from the counts file header (tab-separated, first field = "gene").
BARCODES_FILE="${OUT_DIR}/${BAM_STEM}.barcodes.txt"
zcat "${COUNTS}" | head -1 | cut -f2- | tr '\t' '\n' > "${BARCODES_FILE}"
echo "Barcodes written: $(wc -l < "${BARCODES_FILE}")"

export BAM BARCODES_FILE GTF_VAR OUT_DIR BAM_STEM MM10_GTF HG19_GTF

srun --cpu-bind=none singularity exec --pwd "$(pwd)" "${CONTAINER}" /bin/bash -lc '
set -euo pipefail
GTF="${!GTF_VAR}"
echo "GTF: ${GTF}"

# Sort and index if not already done.
SORTED_BAM="${BAM%.bam}.sorted.bam"
if [ ! -f "${SORTED_BAM}" ]; then
    echo "Sorting BAM..."
    samtools sort -@ "${SLURM_CPUS_PER_TASK:-1}" -o "${SORTED_BAM}" "${BAM}"
    samtools index "${SORTED_BAM}"
else
    echo "Sorted BAM already exists: ${SORTED_BAM}"
fi

velocyto run \
    -b "${BARCODES_FILE}" \
    -o "${OUT_DIR}" \
    -@ "${SLURM_CPUS_PER_TASK:-1}" \
    "${GTF}" \
    "${SORTED_BAM}"

# velocyto names the loom after the BAM stem; rename to the expected path.
LOOM_IN="$(find "${OUT_DIR}" -name "*.loom" -newer "${BARCODES_FILE}" | head -1)"
LOOM_OUT="${OUT_DIR}/${BAM_STEM}.loom"
[ -f "${LOOM_IN}" ] && mv "${LOOM_IN}" "${LOOM_OUT}" && echo "Loom: ${LOOM_OUT}"
'

echo "End: $(date)"
