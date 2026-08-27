#!/usr/bin/env python3
"""Choose one config cell from a finished sweep round, for the chain driver to freeze.

The rule this applies is the one the round tables are meant to be read by, written down
so an unattended chain applies it the same way every time:

  1. A flagged cell is not a result. degenerate / collapsed / dyn-dead are invisible in
     the FOSCTTM ranking and one of them sorts FIRST, so they are dropped before any
     comparison -- and dropped if they are flagged on EITHER dataset, because a config
     that kills the dynamics on one panel has not won on the other.
  2. A cell must not regress on any dataset. tools/collect_run_diagnostics.py says it
     outright: "A cell that wins on one dataset and loses on the other has not won."
  3. Among survivors, rank by the SUM of the paired val_foscttm deltas over datasets,
     tie-broken by total seeds moved the right way.

val_foscttm, never mean_foscttm: the latter is measured on cells the run trained on and
choosing by it is choosing on the test set.

The baseline cell has delta 0 everywhere, so it always survives rules 1-2 and is the
natural fallback: if nothing beats the incumbent, the incumbent is what gets frozen.

stdout is `key=value` lines for the caller to eval; the reasoning goes to stderr.
"""
from __future__ import annotations

import argparse
import csv
import sys


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(raw):
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def cell_of(row, keys):
    return tuple(row.get(k, "") for k in keys)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--paired", required=True,
                        help="<out>_paired.csv from collect_run_diagnostics.py")
    parser.add_argument("--by-config", required=True,
                        help="<out>_by_config.csv from the same call (carries `flag`)")
    parser.add_argument("--keys", required=True,
                        help="Comma-separated --group-by keys that define a cell.")
    parser.add_argument("--top", type=int, default=1,
                        help="How many cells to emit (default 1).")
    args = parser.parse_args()

    keys = [k.strip() for k in args.keys.split(",") if k.strip()]
    paired = read_csv(args.paired)
    by_config = read_csv(args.by_config)

    datasets = sorted({r["dataset"] for r in paired})
    if not datasets:
        print("no datasets in the paired table -- nothing to pick", file=sys.stderr)
        return 2

    flagged: dict[tuple, list[str]] = {}
    for row in by_config:
        flag = (row.get("flag") or "").strip()
        if flag:
            flagged.setdefault(cell_of(row, keys), []).append(f"{row['dataset']}:{flag}")

    deltas: dict[tuple, dict[str, dict]] = {}
    baseline = None
    for row in paired:
        cell = cell_of(row, keys)
        deltas.setdefault(cell, {})[row["dataset"]] = row
        if row.get("is_baseline") == "True":
            baseline = cell

    scored, rejected = [], []
    for cell, per_dataset in deltas.items():
        if cell in flagged:
            rejected.append((cell, f"flagged {'; '.join(flagged[cell])}"))
            continue
        if len(per_dataset) != len(datasets):
            missing = sorted(set(datasets) - set(per_dataset))
            rejected.append((cell, f"missing dataset(s): {', '.join(missing)}"))
            continue
        total, wins, regressed = 0.0, 0, []
        for dataset, row in per_dataset.items():
            delta = as_float(row.get("val_foscttm_delta_mean"))
            if delta is None:
                regressed.append(f"{dataset}: no delta")
                continue
            if delta > 0:
                regressed.append(f"{dataset}: {delta:+.4f}")
            total += delta
            wins += int(as_float(row.get("val_foscttm_delta_win")) or 0)
        if regressed:
            rejected.append((cell, "regressed on " + "; ".join(regressed)))
            continue
        scored.append((total, -wins, cell))

    if not scored:
        print("every cell was rejected; no winner", file=sys.stderr)
        for cell, why in rejected:
            print(f"  reject {dict(zip(keys, cell))}: {why}", file=sys.stderr)
        return 3

    scored.sort()
    print(f"[pick] {len(scored)} cell(s) survived, {len(rejected)} rejected "
          f"({len(datasets)} datasets: {', '.join(datasets)})", file=sys.stderr)
    for why_total, neg_wins, cell in scored[: max(args.top, 3)]:
        marker = "  <- baseline" if cell == baseline else ""
        print(f"    sum dVAL {why_total:+.4f}  wins {-neg_wins}  "
              f"{dict(zip(keys, cell))}{marker}", file=sys.stderr)
    for cell, why in rejected[:8]:
        print(f"  reject {dict(zip(keys, cell))}: {why}", file=sys.stderr)

    chosen = [cell for _, _, cell in scored[: args.top]]
    if chosen[0] == baseline:
        print("[pick] nothing beat the incumbent; freezing the baseline.", file=sys.stderr)
    for rank, cell in enumerate(chosen):
        prefix = "" if args.top == 1 else f"PICK{rank + 1}_"
        for key, value in zip(keys, cell):
            print(f"{prefix}{key}={value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
