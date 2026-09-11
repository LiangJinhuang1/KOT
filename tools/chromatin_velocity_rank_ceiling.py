#!/usr/bin/env python3
"""Does the velocity field's RANK cap how well any model could ever do?

Every intervention tried so far leaves Task D at 1.05-1.15x its permutation null, and a
supervised linear map from `v_c` to the true scVelo velocity -- handed the answer -- reaches
only 1.29x. That is a ceiling on the INPUT, not on the model. `tfidf_lsi` returns a rank-50
reconstruction, so `v_c` built in that space is exactly rank 50 and only ~2.9% of scVelo's
per-cell variance lies in its span.

This measures the ceiling per cached field, so raising the LSI rank can be judged before
any training run is launched:

  ridge_ratio_to_null  a paired ridge from v_c to the scVelo reference on held-out cells,
                       as a ratio to its own permutation null. This is the best ANY linear
                       pushforward could do. A model cannot beat it, so if it does not rise
                       with rank, raising the rank cannot help.
  variance_captured    fraction of the reference's per-cell variance that lies inside the
                       field's own span -- the geometric version of the same question.
  effective_rank       singular values needed for 99% of the field's energy. Confirms
                       whether a field really is rank-limited.

Reads whole datasets; submit it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.chromatin_common import context

import run_kot_chromatin as runner
from src.evaluation.chromatin_eval import task_d_kinetics

RESULTS = PROJECT_ROOT / "cache" / "results" / "chromatin"


def effective_rank(values: np.ndarray, share: float = 0.99) -> int:
    centred = values - values.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centred, compute_uv=False)
    energy = np.cumsum(singular ** 2) / max((singular ** 2).sum(), 1e-12)
    return int(np.searchsorted(energy, share) + 1)


def variance_captured(field: np.ndarray, reference: np.ndarray, rank: int) -> float:
    """Reference variance inside the field's top-`rank` cell-space directions.

    Taking the FULL row space measures nothing: with as many sampled cells as features the
    span is everything and the answer is 1.0000 for every field, which is what the first
    version of this returned. The question only has content once the span is truncated.
    """
    centred = field - field.mean(axis=0, keepdims=True)
    basis, _, _ = np.linalg.svd(centred, full_matrices=False)
    basis = basis[:, :rank]
    target = reference - reference.mean(axis=0, keepdims=True)
    projected = basis @ (basis.T @ target)
    return float((projected ** 2).sum() / max((target ** 2).sum(), 1e-12))


def ceiling(field: np.ndarray, reference: np.ndarray, train: np.ndarray,
            test: np.ndarray, rank: int, ridge: float = 100.0) -> dict:
    """Best cosine a paired linear map from the field can reach on held-out cells.

    The field is projected to `rank` feature-space directions FIRST. Fitting 2000 features
    on 2000 training cells is ill-conditioned, and the first version of this did exactly
    that: it returned a ceiling of 0.944 for a field on which the trained model scored
    1.128, i.e. the model beat its own upper bound, which is impossible. Reducing rank
    before the ridge makes the estimate a bound again.
    """
    basis = np.linalg.svd(field[train] - field[train].mean(axis=0, keepdims=True),
                          full_matrices=False)[2][:rank].T
    reduced = field @ basis
    x, y = reduced[train], reference[train]
    gram = x.T @ x + ridge * np.eye(x.shape[1], dtype=np.float64)
    weights = np.linalg.solve(gram, x.T @ y)
    field = reduced
    metrics = task_d_kinetics((field[test] @ weights).astype(np.float32),
                              reference[test].astype(np.float32))
    centred = metrics["cell_cosine_centred_median"]
    null = metrics["cell_cosine_centred_null_p95"]
    return {"ridge_centred": centred, "ridge_null_p95": null,
            "ridge_ratio_to_null": centred / null if abs(null) > 1e-9 else float("nan"),
            "ridge_gene_pearson": metrics["gene_pearson_median"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="bmmc", choices=runner.DATASETS)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--fields", nargs="+",
                        default=["as_is:", "cp10k_log1p:", "tfidf_lsi:",
                                 "tfidf_lsi:regression", "tfidf_lsi:k60"],
                        help="transform:tag pairs naming cached velocity fields")
    parser.add_argument("--n-cells", type=int, default=4000)
    parser.add_argument("--rank", type=int, default=50,
                        help="project the field to this many directions before fitting, so "
                             "the ridge is well conditioned and the result is a real bound")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=RESULTS / "r2_velocity_rank_ceiling.json")
    args = parser.parse_args()

    ctx = context(args.dataset, args.split_seed)
    reference = runner.reference_velocity(ctx.adata, "velocity_scvelo")
    if reference is None:
        raise SystemExit("scVelo reference absent — run `reference` first")
    reference = reference[:, ctx.columns].astype(np.float64)
    covered = np.flatnonzero(np.abs(reference).sum(axis=0) > 0)
    rng = np.random.default_rng(args.seed)

    rows = []
    for spec in args.fields:
        transform, _, tag = spec.partition(":")
        try:
            field = runner.load_velocity(args.dataset, args.split_seed, transform, tag)
        except (AssertionError, ValueError) as error:
            print(f"  {spec:26s} unavailable: {str(error)[:70]}")
            continue
        dynamic = np.asarray(field["dynamic_mask"], dtype=bool)
        train = ctx.rows("train", dynamic=dynamic)
        test = ctx.rows("test", dynamic=dynamic)
        train = np.sort(rng.choice(train, min(args.n_cells, len(train)), replace=False))
        test = np.sort(rng.choice(test, min(args.n_cells, len(test)), replace=False))
        values = np.asarray(field["velocity"], dtype=np.float64)
        both = np.concatenate([train, test])
        entry = {"field": spec,
                 "effective_rank": effective_rank(values[both]),
                 "n_features": int(values.shape[1]),
                 "fit_rank": args.rank,
                 "variance_captured": variance_captured(values[test],
                                                        reference[test][:, covered],
                                                        args.rank),
                 **ceiling(values, reference[:, covered], train, test, args.rank)}
        rows.append(entry)
        print(f"  {spec:26s} rank {entry['effective_rank']:4d}/{entry['n_features']}  "
              f"ceiling {entry['ridge_ratio_to_null']:.3f}x null  "
              f"variance-captured {entry['variance_captured']:.4f}", flush=True)

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"\n[rank] the trained model reaches 1.05-1.15x; wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
