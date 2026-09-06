#!/usr/bin/env python3
"""Does KOT lose cell pairing to the baselines, or lose to how it was scored?

Two asymmetries crept into the baseline comparison, and both fall on KOT:

  CELL COUNT  KOT was scored on all 13,848 test cells, the subsampled baselines on 1,200.
              FOSCTTM is a fraction, so it looks scale-free — but in a population with 22
              cell types, more candidates means more cells that are LEGITIMATELY closer to
              a query than its own partner. Bigger test set, harder task.

  SPACE       KOT was scored in the 2,000-dim gene space it predicts into; every latent
              baseline was scored in its own 8-50 dim embedding. Nearest-neighbour
              retrieval in 2,000 dimensions is a different problem from retrieval in 30.

This re-scores ONE KOT checkpoint four ways — {full, 1200 cells} x {gene space, 30-dim
PCA} — so the size of each effect is measured rather than argued about. The PCA basis is
fitted on the RNA TRAINING half, never on the test cells.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import run_kot_chromatin as chromatin
from src.evaluation.chromatin_eval import cell_type_match_accuracy, retrieval_metrics
from src.utils.arrays import to_dense


def score(predicted: np.ndarray, observed: np.ndarray, labels: np.ndarray | None) -> dict:
    metrics = retrieval_metrics(predicted, observed)
    if labels is not None:
        metrics["cell_type_accuracy"] = cell_type_match_accuracy(predicted, observed, labels)
    return {key: round(float(value), 4) for key, value in metrics.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="best_align")
    parser.add_argument("--n-cells", type=int, default=1200,
                        help="the cap the subsampled baselines ran at")
    parser.add_argument("--n-comps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cpu")
    model, payload, config = chromatin.load_checkpoint(Path(args.run_dir), args.checkpoint,
                                                       device)
    adata = chromatin.load_dataset(payload["dataset"])
    splits = pd.read_csv(chromatin.split_path(payload["dataset"], config["split_seed"]),
                         index_col=0)
    split_column = splits["split"].to_numpy()
    measured = chromatin.target_cell_mask(adata, config["target_layer"])
    test = np.flatnonzero((split_column == "test") & measured)
    rna_train = np.flatnonzero((split_column == "train")
                               & (splits["train_side"].to_numpy() == "rna") & measured)

    chromatin_matrix = np.asarray(adata.obsm["gene_activity"], dtype=np.float32)
    output_columns = adata.var_names.get_indexer(payload["gene_names"])
    assert (output_columns >= 0).all(), "checkpoint output genes do not match dataset"
    observed_all = to_dense(adata.layers[config["target_layer"]], np.float32)[:, output_columns]
    with torch.no_grad():
        predicted = chromatin.spliced_block(
            model.phi(torch.as_tensor(chromatin_matrix[test], device=device)).numpy(),
            payload["law"])
    observed = observed_all[test]
    labels = (adata.obs["cell_type"].astype(str).to_numpy()[test]
              if "cell_type" in adata.obs else None)

    # Fitted on the RNA training half: a basis fitted on the test cells would be scoring
    # the map in a space built from the answer.
    basis = PCA(n_components=args.n_comps, random_state=args.seed).fit(
        observed_all[rna_train])
    sample = np.sort(np.random.default_rng(args.seed).choice(
        len(test), min(args.n_cells, len(test)), replace=False))

    results = {"run_dir": args.run_dir, "dataset": payload["dataset"],
               "condition": config["condition"], "n_test_full": int(len(test)),
               "n_test_sampled": int(len(sample)), "n_comps": args.n_comps}
    for space, (a, b) in {"gene_space": (predicted, observed),
                          "pca_space": (basis.transform(predicted),
                                        basis.transform(observed))}.items():
        results[f"{space}_full"] = score(a, b, labels)
        results[f"{space}_subsampled"] = score(
            a[sample], b[sample], None if labels is None else labels[sample])

    out = Path(args.run_dir) / "scoring_parity.json"
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
