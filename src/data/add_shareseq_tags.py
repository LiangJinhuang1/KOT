"""
Add CB (cell barcode) and UB (UMI) tags to SHARE-seq BAM files.

SHARE-seq read name format:
  {illumina_id}_{barcode}_{umi}
  e.g. A00794:80:HFTTTDRXX:2:2248:25536:17127_R1.34,R2.70,R3.39,P1.53_CACGATATCA

Usage:
  python src/data/add_shareseq_tags.py input.bam output.bam
"""
import argparse
import pysam


def add_tags(in_bam: str, out_bam: str) -> None:
    n_tagged = 0
    n_skipped = 0

    with pysam.AlignmentFile(in_bam, "rb") as infile:
        with pysam.AlignmentFile(out_bam, "wb", header=infile.header) as outfile:
            for read in infile:
                parts = read.query_name.split("_")
                if len(parts) >= 3:
                    read.set_tag("CB", parts[-2])
                    read.set_tag("UB", parts[-1])
                    n_tagged += 1
                else:
                    n_skipped += 1
                outfile.write(read)

    print(f"Done. Tagged: {n_tagged:,}  Skipped: {n_skipped:,}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Add CB/UB tags to SHARE-seq BAM.")
    parser.add_argument("in_bam", help="Input sorted BAM")
    parser.add_argument("out_bam", help="Output tagged BAM")
    args = parser.parse_args()
    add_tags(args.in_bam, args.out_bam)


if __name__ == "__main__":
    main()
