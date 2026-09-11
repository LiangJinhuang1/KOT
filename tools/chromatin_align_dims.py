#!/usr/bin/env python3
"""align_dims ladder on one winning formulation, after a spliced-target blur check.

d in {8, 16, 32, 64}; 32 is the winner itself, so three new trains. Blur is
assessed from tools/chromatin_alignment_geometry.py on the actual spliced
shared-RNA target. If the current blur is usable, it is left alone; otherwise
at most two alternatives, still on the winner, not a dim x blur grid.

Usage, after the screen JVP refresh and the geometry job:
  PYTHONPATH=. python tools/chromatin_align_dims.py jobs \\
      --winners cache/results/chromatin/align_dims_winners.json \\
      --geometry cache/results/chromatin/alignment_geometry_bmmc_relay_spliced_shared_lognorm_spliced_train_only_v1.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tools.chromatin_kinetic_controls import (
    apply_override, flag_value, load_json, stem_from_winner, train_tokens,
)
from tools.chromatin_rna_coords import load_winner

ALIGN_DIMS = (8, 16, 32, 64)
CURRENT_DIMS = 32
CURRENT_BLUR = 0.1
# blur as a fraction of the median pairwise distance in the projected space.
# Below ~0.05 the kernel is essentially unregularised; above 1 it is larger
# than a typical pair, so the oracle and a constant start to collapse together.
BLUR_DIST_LO = 0.05
BLUR_DIST_HI = 1.0
ADVANTAGE_FLOOR = 1.2
MAX_BLUR_ALTERNATIVES = 2
TARGET_BLUR_OVER_DISTANCE = 0.25

DEFAULT_GEOMETRY = Path(
    "cache/results/chromatin/"
    "alignment_geometry_bmmc_relay_spliced_shared_lognorm_spliced_train_only_v1.csv")


def blur_is_usable(row: pd.Series) -> bool:
    """Oracle beats a constant, and blur is a sensible fraction of pair distance."""
    return bool(row["oracle_beats_constant"]) \
        and BLUR_DIST_LO <= float(row["blur_over_pair_distance"]) <= BLUR_DIST_HI \
        and float(row["oracle_advantage"]) >= ADVANTAGE_FLOOR


def assess_blur(frame: pd.DataFrame, align_dims: int = CURRENT_DIMS,
                current_blur: float = CURRENT_BLUR) -> dict:
    """Keep the training blur, or name at most two replacements at the same rank."""
    at = frame[(frame["align_dims"] == align_dims)
               & (frame["blur"].astype(float) == float(current_blur))]
    if at.empty:
        return {"keep": True, "alternatives": [],
                "reason": f"no geometry row for dims {align_dims} blur {current_blur:g}; "
                          "leaving the training blur"}
    row = at.iloc[0]
    if blur_is_usable(row):
        return {"keep": True, "alternatives": [],
                "reason": (f"dims {align_dims} blur {current_blur:g}: oracle advantage "
                           f"{float(row['oracle_advantage']):.2f}x, blur/dist "
                           f"{float(row['blur_over_pair_distance']):.2f}")}
    usable = frame[(frame["align_dims"] == align_dims)
                   & frame["oracle_beats_constant"]
                   & (frame["blur"].astype(float) != float(current_blur))].copy()
    usable = usable[usable.apply(blur_is_usable, axis=1)]
    if usable.empty:
        return {"keep": True, "alternatives": [],
                "reason": f"current blur {current_blur:g} is outside the usable band "
                          "and no alternative on this rank is either; leaving it"}
    usable = usable.assign(
        dist=(usable["blur_over_pair_distance"] - TARGET_BLUR_OVER_DISTANCE).abs())
    picks = usable.sort_values(["dist", "oracle_advantage"],
                               ascending=[True, False]).head(MAX_BLUR_ALTERNATIVES)
    alternatives = [float(value) for value in picks["blur"]]
    return {"keep": False, "alternatives": alternatives,
            "reason": (f"dims {align_dims} blur {current_blur:g} is not usable "
                       f"(advantage {float(row['oracle_advantage']):.2f}x, blur/dist "
                       f"{float(row['blur_over_pair_distance']):.2f}); "
                       f"alternatives {alternatives}")}


def append_geometry_flags(tokens: list[str], align_dims: int, blur: float) -> list[str]:
    extra = ["--align-dims", *flag_value(align_dims),
             "--sinkhorn-blur", *flag_value(blur)]
    return tokens + extra


def dim_overrides(winner_dims: int) -> list[dict]:
    return [{"tag": f"d{dims}", "align_dims": dims}
            for dims in ALIGN_DIMS if dims != winner_dims]


def blur_overrides(blurs: list[float], winner_dims: int) -> list[dict]:
    return [{"tag": f"d{winner_dims}_b{blur:g}", "align_dims": winner_dims,
             "sinkhorn_blur": blur}
            for blur in blurs]


def align_dim_jobs(winner: dict, decision: dict) -> list[str]:
    config = winner["config"]
    stem = stem_from_winner(winner["run"])
    winner_dims = int(config.get("align_dims") or CURRENT_DIMS)
    winner_blur = float(config.get("sinkhorn_blur") if config.get("sinkhorn_blur") is not None
                        else CURRENT_BLUR)
    lines = []
    for override in dim_overrides(winner_dims) + blur_overrides(
            decision.get("alternatives") or [], winner_dims):
        tag, updated = apply_override(config, override)
        run_dir = f"cache/chromatin/runs/r2dim_{stem}_{tag}_seed42"
        blur = float(updated.get("sinkhorn_blur") if updated.get("sinkhorn_blur") is not None
                     else winner_blur)
        dims = int(updated["align_dims"])
        lines.append(" ".join(append_geometry_flags(
            train_tokens(updated, run_dir), dims, blur)))
    return lines


def job_file_text(winner: dict, decision: dict) -> str:
    n_dim = len(dim_overrides(int(winner["config"].get("align_dims") or CURRENT_DIMS)))
    n_blur = len(decision.get("alternatives") or [])
    header = f"""# align_dims ladder on one winning formulation.
# Winner: {winner['run']}
# Blur: {decision['reason']}
#
# Generated by tools/chromatin_align_dims.py. d in {{8,16,32,64}}; {CURRENT_DIMS}
# is the winner, so {n_dim} new dim trains. Blur alternatives: {n_blur}
# (at most {MAX_BLUR_ALTERNATIVES}, and only if the current blur is not usable).
# Development evaluate: --eval-split val. Do not score BMMC test.
#
# Submit from the repo root after the winner's velocity cache exists:
#   while IFS= read -r line; do
#     [[ "$line" =~ ^train ]] || continue
#     name=${{line##*--run-dir cache/chromatin/runs/}}
#     sbatch --job-name="$name" \\
#       --export=ALL,RUN_CMD="PYTHONPATH=. python -u run_kot_chromatin.py $line" \\
#       slurm/train_slurm.sh
#   done < jobs/jobs_chromatin_align_dims.txt
"""
    return header + "\n" + "\n".join(align_dim_jobs(winner, decision)) + "\n"


def jobs_main(args: argparse.Namespace) -> int:
    winner = load_winner(args)
    if args.geometry is None or not args.geometry.exists():
        raise FileNotFoundError(
            "need the spliced-target geometry CSV before writing dim/blur trains")
    decision = assess_blur(pd.read_csv(args.geometry))
    print(f"[align-dims] {decision['reason']}")
    text = job_file_text(winner, decision)
    args.out.write_text(text)
    trains = [line for line in text.splitlines() if line.startswith("train ")]
    print(f"[align-dims] winner {winner['run']}; wrote {len(trains)} trains to {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    jobs = sub.add_parser("jobs", help="write the dim ladder, and blur only if needed")
    jobs.add_argument("--winners", type=Path, default=None,
                      help="select JSON; uses the alignment pick when two exist")
    jobs.add_argument("--run-dir", type=Path, default=None,
                      help="one completed run to copy identity from")
    jobs.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY,
                      help="spliced-target CSV from chromatin_alignment_geometry.py")
    jobs.add_argument("--out", type=Path,
                      default=Path("jobs/jobs_chromatin_align_dims.txt"))
    jobs.set_defaults(func=jobs_main)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "jobs" and args.winners is None and args.run_dir is None:
        raise SystemExit("jobs needs --winners or --run-dir")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
