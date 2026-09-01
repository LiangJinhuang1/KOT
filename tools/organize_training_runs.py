#!/usr/bin/env python3
"""Symlink finished run_* dirs into analysis batches. Default is links, not copies."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

STAGE_ORDER = ["clean", "oracle", "branch"]

KNOWN_RUN_STAGES = {
    "run_20260701_141322": "clean",
    "run_20260701_171218": "clean",
    "run_20260701_183237": "clean",
    "run_20260701_185533": "oracle",
    "run_20260701_200635_moscot_sweep": "clean",
    "run_oracle_kotoracles_mean_20260701_205724": "oracle",
    "run_oracle_mean_moscot_sweep": "oracle",
    "run_oracle_baselines_mean": "oracle",
    "run_branch_mean_20260701_190000": "branch",
    "run_branch_mean_20260701_205803": "branch",
}


def infer_stage(run_folder: Path) -> str:
    name = run_folder.name
    if name in KNOWN_RUN_STAGES:
        return KNOWN_RUN_STAGES[name]
    if "branch" in name:
        return "branch"
    if "oracle" in name:
        return "oracle"
    return "clean"


def model_names(run_folder: Path) -> list[str]:
    skip = {"cache", "results", "moscot_sweep"}
    return sorted(
        child.name
        for child in run_folder.iterdir()
        if child.is_dir() and child.name not in skip
    )


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    if copy:
        shutil.copytree(src, dst, symlinks=True)
        return
    os.symlink(src.resolve(), dst)


def write_index(output_dir: Path, rows: list[dict]) -> None:
    lines = ["stage,run,models,source_path,link_path"]
    for row in rows:
        models = " ".join(row["models"])
        lines.append(
            f"{row['stage']},{row['run']},\"{models}\","
            f"{row['source']},{row['dest']}"
        )
    (output_dir / "INDEX.csv").write_text("\n".join(lines) + "\n")

    readme = [
        "Training run layout",
        "===================",
        "",
        "runs/<stage>/<run_name>/  — one folder per Slurm job",
        "  <model>/synthetic_linked_ode/seed_<n>/  — metrics, plots, logs",
        "  training.yaml, run.log, moscot_sweep/, results/",
        "",
        "Stages: clean, oracle, branch",
        "",
        "Rows:",
    ]
    for row in rows:
        readme.append(
            f"  {row['stage']:6}  {row['run']}  "
            f"({len(row['models'])} models: {', '.join(row['models']) or '—'})"
        )
    (output_dir / "README.txt").write_text("\n".join(readme) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("cache/training"))
    parser.add_argument("--output", type=Path, default=Path("analysis/jul2026/runs"))
    parser.add_argument("--copy", action="store_true",
                        help="Copy files instead of symlinking (uses more disk).")
    args = parser.parse_args()

    batch_dir = args.output.parent
    batch_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for run_folder in sorted(args.base.glob("run_*")):
        if not run_folder.is_dir():
            continue
        stage = infer_stage(run_folder)
        dest = args.output / stage / run_folder.name
        link_or_copy(run_folder, dest, args.copy)
        rows.append({
            "stage": stage,
            "run": run_folder.name,
            "models": model_names(run_folder),
            "source": run_folder.resolve(),
            "dest": dest.resolve(),
        })
        print(f"{run_folder.name}  ->  runs/{stage}/{run_folder.name}")

    write_index(batch_dir, rows)
    print(f"\nWrote {batch_dir / 'INDEX.csv'}")
    print(f"Wrote {batch_dir / 'README.txt'}")
    print(f"Organized {len(rows)} runs under {args.output}/")


if __name__ == "__main__":
    main()
