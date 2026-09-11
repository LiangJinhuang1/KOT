#!/usr/bin/env python3
"""Does a chromatin velocity field point forward in the pseudotime it was built from?

The field is a local time derivative over forward neighbours, so `<v_i, a_j - a_i>` should
be positive exactly when `tau_j > tau_i`. Measured on the shipped bmmc field it is not:
sign agreement 0.41-0.45 against a chance level of 0.50. The cause is the estimator --
each neighbour difference is divided by its own `tau` gap, gaps run from 1.4e-4 to a 9.2e-3
median, and the `> 1e-6` guard lets the small ones through with ~1e4 weight.

This scores any set of cached fields on the same cells so a fix can be shown to work:

  pair_sign_agreement  NOT a forwardness test and kept only as a sanity check. The pairs
                       are restricted to FORWARD neighbours, so sign(tau_j - tau_i) is
                       always +1 and v_i is a positive combination of those same
                       differences -- the statistic is ~1 by construction (measured 0.995
                       for every variant, including the broken field). Read
                       cell_correlation_median instead.
  cell_correlation     THE forwardness statistic: per cell, the correlation between
                       <v_i, a_j - a_i> and tau_j - tau_i over its forward neighbours,
                       median across cells. A field that is a genuine time derivative
                       points further along the further ahead the neighbour is, so this
                       must be POSITIVE. The quotient estimator gives -0.268 -- inverted,
                       because dividing by a near-zero gap makes the closest neighbour
                       dominate -- and the regression estimator gives +0.267.
  weight_concentration fraction of cells whose single largest neighbour weight exceeds
                       half the total -- the failure mode itself.

Reads the whole dataset; submit it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import run_kot_chromatin as runner
from src.data.chromatin import chromatin_features
from src.data.chromatin_velocity import (atac_pseudotime, finite_lsi_mask,
                                         interpolate_pseudotime, progenitor_root,
                                         train_neighbour_graph)

RESULTS = PROJECT_ROOT / "cache" / "results" / "chromatin"


def forwardness(velocity: np.ndarray, activity: np.ndarray, pseudotime: np.ndarray,
                graph, rows: np.ndarray) -> dict:
    agree, total, correlations, concentrated = 0, 0, [], 0
    for row in rows:
        span = slice(graph.indptr[row], graph.indptr[row + 1])
        candidates = graph.indices[span]
        gaps = pseudotime[candidates] - pseudotime[row]
        ahead = gaps > 1e-6
        if ahead.sum() < 3:
            continue
        forward = candidates[ahead]
        projection = (activity[forward] - activity[row]) @ velocity[row]
        gap = gaps[ahead]
        agree += int((np.sign(projection) == np.sign(gap)).sum())
        total += len(gap)
        if projection.std() > 0 and gap.std() > 0:
            correlations.append(float(np.corrcoef(projection, gap)[0, 1]))
        weights = 1.0 / np.clip(gap, 1e-12, None)
        concentrated += int(weights.max() > 0.5 * weights.sum())
    return {
        "pair_sign_agreement": agree / max(total, 1),
        "cell_correlation_median": float(np.median(correlations)) if correlations else float("nan"),
        "cell_correlation_positive_fraction": float(np.mean(np.array(correlations) > 0))
        if correlations else float("nan"),
        "weight_concentration": concentrated / max(len(rows), 1),
        "n_cells": int(len(rows)), "n_pairs": int(total),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="bmmc", choices=runner.DATASETS)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--transform", default="tfidf_lsi")
    parser.add_argument("--tags", nargs="+", default=["", "regression", "rawscale",
                                                     "regression_rawscale", "k60"])
    parser.add_argument("--n-cells", type=int, default=1500)
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=RESULTS / "r2_velocity_forwardness.json")
    args = parser.parse_args()

    adata = runner.load_dataset(args.dataset)
    splits = __import__("pandas").read_csv(runner.split_path(args.dataset, args.split_seed),
                                           index_col=0)
    split = splits["split"].to_numpy()
    side = splits["train_side"].to_numpy()
    train_mask = (split == "train") & (side == "atac")
    activity = chromatin_features(adata, args.transform)

    # Rebuild the same pseudotime and graph the field was constructed on.
    lsi = np.asarray(adata.obsm["lsi"], dtype=np.float32)
    finite = finite_lsi_mask(lsi)
    train_idx = np.flatnonzero(train_mask & finite)
    held_idx = np.flatnonzero(~train_mask & finite)
    handle = adata[train_idx].copy()
    handle.obsm["lsi"] = lsi[train_idx]
    pt_train = atac_pseudotime(handle, args.n_neighbors, progenitor_root(handle))
    pseudotime = np.zeros(adata.n_obs, dtype=np.float32)
    pseudotime[train_idx] = pt_train
    if len(held_idx):
        pseudotime[held_idx] = interpolate_pseudotime(lsi[train_idx], pt_train,
                                                      lsi[held_idx], args.n_neighbors)
    graph = train_neighbour_graph(lsi, train_mask & finite, args.n_neighbors).tocsr()

    results = {}
    rng = np.random.default_rng(args.seed)
    for tag in args.tags:
        try:
            field = runner.load_velocity(args.dataset, args.split_seed, args.transform, tag)
        except (AssertionError, ValueError) as error:
            print(f"  {tag or 'quotient':22s} unavailable: {error}")
            continue
        dynamic = np.asarray(field["dynamic_mask"], dtype=bool)
        rows = np.flatnonzero(dynamic & finite)
        rows = np.sort(rng.choice(rows, min(args.n_cells, len(rows)), replace=False))
        entry = forwardness(np.asarray(field["velocity"], dtype=np.float32), activity,
                            pseudotime, graph, rows)
        entry["gauge_normalized"] = bool(field.get("gauge_normalized", True))
        entry["estimator"] = str(field.get("velocity_estimator", "quotient"))
        results[tag or "quotient"] = entry
        print(f"  {tag or 'quotient':22s} sign {entry['pair_sign_agreement']:.4f}  "
              f"cell-corr {entry['cell_correlation_median']:+.4f}  "
              f"concentration {entry['weight_concentration']:.3f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\n[forwardness] chance is 0.5000; wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
