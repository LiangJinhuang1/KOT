#!/usr/bin/env python3
"""Backfill protocol stamps. Unstamped files read as in-distribution; re-running
a 12-seed sweep to recover a fact already in run_config.yaml is not worth a GPU.
A prediction with no matching run dir is left alone.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import anndata as ad

from src.evaluation.protocol import PROTOCOL_KEY, predictor_kind, resolve_oos_mode
from src.utils.io import load_yaml

PREDICTION_RE = re.compile(
    r"^(?P<model>.+?)_(?P<dataset>[a-z0-9_]+?)(?:_seed_(?P<seed>\d+))?\.h5ad$"
)


def run_dir_for(run_roots: list[Path], model: str, dataset: str, seed: str | None) -> Path | None:
    """The one run directory that produced this prediction, if it is among `run_roots`."""
    leaf = Path(model) / dataset / (f"seed_{seed}" if seed else "")
    matches = [root / leaf for root in run_roots if (root / leaf / "run_config.yaml").exists()]
    if len(matches) > 1:
        raise ValueError(
            f"{model}/{dataset}/seed_{seed} exists under {len(matches)} run roots: "
            f"{[str(m) for m in matches]}. Pass only the run root that wrote these "
            "predictions -- guessing which one would stamp a file with another run's "
            "protocol."
        )
    return matches[0] if matches else None


def stamp_from_run(run_dir: Path, model: str, n_cells: int) -> dict:
    """The stamp implied by a finished run's own config and diagnostics."""
    run_cfg = load_yaml(run_dir / "run_config.yaml")["run_cfg"]
    fit_key = str(run_cfg.get("fit_obs_key") or "")
    diagnostics = json.loads((run_dir / "diagnostics.json").read_text())
    # n_train_cells is the split's own count, written by run_kot. Baselines do not write
    # it, so a restricted baseline reports its fitted count as unknown rather than wrong.
    n_fit = diagnostics.get("n_train_cells", -1) if fit_key else n_cells
    mode = resolve_oos_mode(model)
    return {
        "oos_mode": mode,
        "fit_obs_key": fit_key,
        "fit_obs_values": [str(v) for v in (run_cfg.get("fit_obs_values") or [])],
        "n_fit_cells": int(n_fit),
        "n_cells": int(n_cells),
        "predictor": predictor_kind(run_cfg),
        "out_of_distribution": bool(fit_key),
        "paired_oracle": mode == "paired",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, action="append", required=True,
                        help="Run root that produced these predictions (repeatable).")
    parser.add_argument("--pattern", default="*.h5ad")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    stamped, skipped = 0, []
    for path in sorted(args.predictions.glob(args.pattern)):
        match = PREDICTION_RE.match(path.name)
        if match is None:
            skipped.append((path.name, "unparseable name"))
            continue
        model, dataset, seed = match.group("model", "dataset", "seed")
        run_dir = run_dir_for(args.run_root, model, dataset, seed)
        if run_dir is None:
            skipped.append((path.name, "no matching run dir"))
            continue
        adata = ad.read_h5ad(path)
        stamp = stamp_from_run(run_dir, model, adata.n_obs)
        print(f"[stamp] {path.name}: {stamp['oos_mode']} | "
              f"ood={stamp['out_of_distribution']} | predictor={stamp['predictor']} | "
              f"fit={stamp['n_fit_cells']}/{stamp['n_cells']}")
        if not args.dry_run:
            adata.uns[PROTOCOL_KEY] = stamp
            adata.write_h5ad(path)
        stamped += 1

    print(f"\n[out] stamped {stamped} file(s){' (dry run)' if args.dry_run else ''}")
    for name, why in skipped:
        print(f"[skip] {name}: {why} — left unstamped, i.e. read as in-distribution")


if __name__ == "__main__":
    main()
