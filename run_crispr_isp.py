#!/usr/bin/env python
"""Task B in-silico perturbation: predict knockout RNA, then score it through frozen KOT.

One entry point; the selected module is imported only after the name is read, because competitor stacks cannot share a process.
"""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path

from src.evaluation.isp import MODELS


def main() -> None:
    # Two-stage parser so unused competitor modules are not imported.
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("model", choices=sorted(MODELS))
    selected, remaining = selector.parse_known_args()
    module = importlib.import_module(MODELS[selected.model])

    parser = argparse.ArgumentParser(
        prog=f"run_crispr_isp.py {selected.model}",
        description=module.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    module.add_arguments(parser)
    args = parser.parse_args(remaining)

    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error("Output directory must be empty (preserve previous experiments)")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    module.run(args)


if __name__ == "__main__":
    main()
