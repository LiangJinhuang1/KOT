#!/bin/bash
#SBATCH --account=core-med1-telem
#SBATCH --partition=jobs-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=12:00:00
#SBATCH --job-name=STARsolo_BMMC_array
#SBATCH --output=logs/starsolo_bmmc_%A_%a.log

set -euo pipefail

STAR_EXEC="${STAR_EXEC:-/data2/core-med1-telem/common/Thesis_Jin/STAR}"
SAMTOOLS_EXEC="${SAMTOOLS_EXEC:-/data2/core-med1-telem/common/Thesis_Jin/samtools-1.20/samtools}"
BAM_ROOT="${BAM_ROOT:-/data2/core-med1-telem/common/Thesis_Jin/Datasets/BMMC/bam}"
OUT_ROOT="${OUT_ROOT:-/data2/core-med1-telem/common/Thesis_Jin/Datasets/BMMC/counts}"
BAM_LIST="${BAM_LIST:-/data2/core-med1-telem/common/Thesis_Jin/Datasets/BMMC/bmmc_bam_files.txt}"
GENOME_DIR="${GENOME_DIR:-/data2/core-med1-telem/common/Thesis_Jin/Datasets/references/refdata-gex-GRCh38-2024-A/star_2.7.11b}"
GTF_FILE="${GTF_FILE:-/data2/core-med1-telem/common/Thesis_Jin/Datasets/references/refdata-gex-GRCh38-2024-A/genes/genes.gtf}"
CITE_WHITELIST="${CITE_WHITELIST:-/data2/core-med1-telem/common/Thesis_Jin/Datasets/BMMC/3M-february-2018.txt}"
MULTIOME_WHITELIST="${MULTIOME_WHITELIST:-/data2/core-med1-telem/common/Thesis_Jin/Datasets/BMMC/737K-arc-v1.txt}"

mkdir -p "$OUT_ROOT" logs "$(dirname "$BAM_LIST")"

if [ ! -s "$BAM_LIST" ]; then
    find "$BAM_ROOT" \( -name "*.bam" -o -name "*.bam.1" \) | sort > "$BAM_LIST"
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"
BAM_PATH=$(sed -n "${TASK_ID}p" "$BAM_LIST")

if [ -z "$BAM_PATH" ]; then
    echo "No BAM found for task index $TASK_ID in $BAM_LIST"
    exit 1
fi

ID=$(basename "$BAM_PATH")
ID=${ID%.1}
ID=${ID%.bam}

if [ -n "${WHITELIST:-}" ]; then
    SELECTED_WHITELIST="$WHITELIST"
elif [[ "$ID" == *multiome_gex* ]]; then
    SELECTED_WHITELIST="$MULTIOME_WHITELIST"
else
    SELECTED_WHITELIST="$CITE_WHITELIST"
fi

if [ ! -s "$SELECTED_WHITELIST" ]; then
    echo "Barcode whitelist not found: $SELECTED_WHITELIST"
    if [[ "$ID" == *multiome_gex* ]]; then
        echo "Multiome GEX needs the 10x ARC GEX whitelist: 737K-arc-v1.txt"
        echo "Find it in a Cell Ranger/Cell Ranger ARC installation under lib/python/cellranger/barcodes/."
    fi
    exit 1
fi

echo "========================================================"
echo "STARsolo BMMC task: $TASK_ID"
echo "Input BAM: $BAM_PATH"
echo "Sample ID: $ID"
echo "Barcode whitelist: $SELECTED_WHITELIST"
echo "Output prefix: ${OUT_ROOT}/${ID}_"
echo "Started: $(date)"

"$STAR_EXEC" \
    --runThreadN "${SLURM_CPUS_PER_TASK:-16}" \
    --genomeDir "$GENOME_DIR" \
    --sjdbGTFfile "$GTF_FILE" \
    --readFilesIn "$BAM_PATH" \
    --readFilesType SAM SE \
    --readFilesCommand "$SAMTOOLS_EXEC" view -F 0x100 \
    --soloType CB_UMI_Simple \
    --soloInputSAMattrBarcodeSeq CR UR \
    --soloInputSAMattrBarcodeQual CY UY \
    --soloCBwhitelist "$SELECTED_WHITELIST" \
    --soloFeatures Gene Velocyto \
    --soloCBstart 1 \
    --soloCBlen 16 \
    --soloUMIstart 17 \
    --soloUMIlen 12 \
    --soloCellFilter EmptyDrops_CR \
    --outFileNamePrefix "${OUT_ROOT}/${ID}_" \
    --outSAMtype None \
    --clipAdapterType CellRanger4 \
    --readFilesSAMattrKeep None

echo "Completed: $(date)"
