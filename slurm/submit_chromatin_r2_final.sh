#!/bin/bash
#
# R2 final: 12 seeds x 3 arms, one frozen config, from
# jobs/jobs_chromatin_r2_final_12seed.txt.
#
# Usage: bash slurm/submit_chromatin_r2_final.sh pilot|shuffle [--dry-run|--skip-gate-check]
#
#   pilot            -> the 24 full/noDyn runs
#   shuffle          -> the 12 shuffle controls, once every pilot run has passed
#   --skip-gate-check -> launch shuffle anyway. Corruption arms are exempt from the
#     biological gate by construction (preflight_verdict only enforces it for
#     full/noDyn), so a shuffle run is still interpretable when a pilot failed; the
#     per-run gate outcome is recorded in the manifest either way. Used here because
#     5/12 noDyn runs miss the centred-JVP-vs-null check, which an arm with
#     lambda_dyn=0 has no dynamics to satisfy in the first place.
#
# One sbatch per line through train_slurm.sh. NOT parallel_train.sh: that skips any
# line not starting with `--` and invokes the RNA->protein runner, so chromatin lines
# would be silently dropped rather than run.
#
# Each job chains `preflight --checkpoint final` after training. Training writes only
# preflight_best_align.json, so without this the `final` half of the summary table is
# blank -- which is exactly what happened to r2_final_dev_summary.csv (72 empty rows).
# Scoring `final` cannot clobber the launch gate: write_preflight_result returns early
# for any checkpoint that is not best_align.
set -euo pipefail
cd "$(dirname "$0")/.."

phase="${1:?usage: $0 pilot|shuffle [--dry-run] [--skip-gate-check]}"
shift
dry_run=""
skip_gate_check=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)         dry_run="--dry-run" ;;
    --skip-gate-check) skip_gate_check=1 ;;
    *) echo "usage: $0 pilot|shuffle [--dry-run] [--skip-gate-check]" >&2; exit 2 ;;
  esac
  shift
done
jobs_file="jobs/jobs_chromatin_r2_final_12seed.txt"
anchor="config/gamma_anchors_k562_timelapse.csv"

case "$phase" in
  pilot)   want="full noDyn" ;;
  shuffle) want="shuffle" ;;
  *) echo "unknown phase '$phase'; use pilot or shuffle" >&2; exit 2 ;;
esac
[[ -s "$jobs_file" ]] || { echo "Missing jobs file: $jobs_file" >&2; exit 2; }
[[ -s "$anchor" ]]    || { echo "Missing RNA anchors: $anchor" >&2; exit 2; }

line_condition() { sed -n 's/.*--condition \([^ ]*\).*/\1/p' <<<"$1"; }
line_run_dir()   { sed -n 's/.*--run-dir \([^ ]*\).*/\1/p'   <<<"$1"; }

wanted() {
  local condition="$1" candidate
  for candidate in $want; do
    [[ "$condition" == "$candidate" ]] && return 0
  done
  return 1
}

# The shuffle control is only interpretable against arms that cleared the gate, so it
# waits for every pilot run rather than for a representative one.
if [[ "$phase" == "shuffle" && "$skip_gate_check" != "1" ]]; then
  missing=0
  while IFS= read -r line; do
    [[ "$line" == train\ * ]] || continue
    condition="$(line_condition "$line")"
    [[ "$condition" == "full" || "$condition" == "noDyn" ]] || continue
    marker="$(line_run_dir "$line")/preflight_passed.json"
    [[ -s "$marker" ]] || { echo "Pilot not passed: $marker" >&2; missing=$((missing + 1)); }
  done < "$jobs_file"
  [[ "$missing" -eq 0 ]] || {
    echo "$missing pilot run(s) have not passed; shuffle waits." >&2
    exit 2
  }
fi

submitted=0
while IFS= read -r line; do
  [[ "$line" == train\ * ]] || continue
  condition="$(line_condition "$line")"
  wanted "$condition" || continue
  run_dir="$(line_run_dir "$line")"
  [[ -n "$run_dir" ]] || { echo "No --run-dir in: $line" >&2; exit 2; }
  # Existing directories are preserved, matching submit_chromatin_phase.sh: a rerun
  # that silently retrains is how a table stops matching the checkpoints on disk.
  [[ ! -e "$run_dir" ]] || { echo "Preserving existing run: $run_dir" >&2; continue; }

  command="PYTHONPATH=. python -u run_kot_chromatin.py ${line} && PYTHONPATH=. python -u run_kot_chromatin.py preflight --run-dir ${run_dir} --checkpoint final"
  if [[ "$dry_run" == "--dry-run" ]]; then
    printf '%s\n' "$command"
  else
    mkdir -p logs
    sbatch --job-name="r2fin_${condition}_$(basename "$run_dir")" \
      --export=ALL,RUN_CMD="$command" slurm/train_slurm.sh
  fi
  submitted=$((submitted + 1))
done < "$jobs_file"

echo "[$phase] ${submitted} run(s) $([[ "$dry_run" == "--dry-run" ]] && echo listed || echo submitted)"
