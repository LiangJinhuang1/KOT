#!/usr/bin/env python3
r"""Choose align_dims and blur before training, against a criterion that means something.

The oracle (source cell's own RNA) must beat a constant, and that margin has to be read against whether the projected space still carries cell identity.

Reads the whole dataset; submit rather than run on the login node.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import run_kot_chromatin as chromatin
from src.losses.chromatin_laws import REDUCED, RELAY
from src.losses.entropic_ot import squared_distances
from src.losses.sinkhorn import sinkhorn_divergence
from src.training.kot import choose_torch_device, subsample_rows


def median_pair_distance(cloud: torch.Tensor) -> float:
    """Typical distance between two cells — the length `blur` is a fraction of."""
    return float(squared_distances(cloud, cloud).sqrt().median())


def neighbour_preservation(full: torch.Tensor, projected: torch.Tensor,
                           n_neighbors: int) -> float:
    """Share of each cell's full-space nearest neighbours that survive the projection.

    Training matches distributions in the projected space, but verdicts are read in the full one.
    """
    def neighbours(points: torch.Tensor) -> torch.Tensor:
        return torch.cdist(points, points).topk(n_neighbors + 1, largest=False).indices[:, 1:]

    kept = [len(set(a.tolist()) & set(b.tolist()))
            for a, b in zip(neighbours(full), neighbours(projected))]
    return float(np.mean(kept) / n_neighbors)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=chromatin.DATASETS, required=True)
    parser.add_argument("--law", choices=[REDUCED, RELAY], default=REDUCED)
    parser.add_argument("--rna-target", choices=["auto", "spliced", "rna", "unspliced"],
                        default="auto")
    parser.add_argument("--align-block", choices=["joint", "spliced"], default="joint")
    parser.add_argument("--dims", type=int, nargs="+",
                        default=[0, 256, 128, 64, 32, 16, 8, 4])
    parser.add_argument("--blurs", type=float, nargs="+",
                        default=[0.5, 0.2, 0.1, 0.05, 0.02, 0.01])
    parser.add_argument("--n-points", type=int, default=2048)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = choose_torch_device({"device": args.device})
    torch.manual_seed(args.seed)
    adata = chromatin.load_dataset(args.dataset)
    splits = pd.read_csv(chromatin.split_path(args.dataset, args.split_seed), index_col=0)
    assert splits.index.equals(adata.obs_names), "split file does not match the dataset"
    split_column = splits["split"].to_numpy()
    side_column = splits["train_side"].to_numpy()

    target_layer = chromatin.rna_target_layer(adata, args.rna_target, args.law)
    covered = chromatin.target_cell_mask(adata, target_layer)
    rna_rows_np = np.flatnonzero(covered & (split_column == "train") & (side_column == "rna"))
    atac_rows_np = np.flatnonzero(covered & (split_column == "train") & (side_column == "atac"))
    gene_mask = chromatin.output_gene_mask(adata, args.law, rows=rna_rows_np,
                                           target_layer=target_layer)
    target = torch.as_tensor(
        chromatin.build_targets(adata, args.law, target_layer, gene_mask), device=device)
    n_output_genes = int(gene_mask.sum())
    rna_rows = torch.as_tensor(rna_rows_np, device=device)
    atac_rows = torch.as_tensor(atac_rows_np, device=device)
    print(f"[geometry] {args.dataset} {args.law} target '{target_layer}' "
          f"({target.shape[1]} columns, {n_output_genes} genes); "
          f"{len(atac_rows)} source cells, {len(rna_rows)} target cells", flush=True)

    columns = chromatin.alignment_columns(args.law, args.align_block, n_output_genes, device)
    fit_target = target[rna_rows]
    aligned_fit = fit_target if columns is None else fit_target[:, columns]
    align_scale = chromatin.alignment_scale(
        aligned_fit, args.law if columns is None else REDUCED)

    # Source cells carry their own RNA because the split is an artefact of paired multiome.
    source = subsample_rows(len(atac_rows), args.n_points, device, atac_rows)
    sink = subsample_rows(len(rna_rows), args.n_points, device, rna_rows)
    assert len(set(source.tolist()) & set(sink.tolist())) == 0, "source and sink overlap"
    observed = target[sink]
    candidates = {
        "constant": observed.mean(dim=0, keepdim=True).expand(len(source), -1),
        "paired_oracle": target[source],
    }

    rows = []
    for n_dims in args.dims:
        basis = chromatin.alignment_basis(target, align_scale, rna_rows, n_dims, columns)
        reference = chromatin.align_view(observed, columns, align_scale, basis)
        full_reference = chromatin.align_view(observed, columns, align_scale, None)
        preserved = 1.0 if basis is None else neighbour_preservation(
            full_reference, reference, args.n_neighbors)
        views = {name: chromatin.align_view(block, columns, align_scale, basis)
                 for name, block in candidates.items()}
        centred = chromatin.align_view(fit_target, columns, align_scale, None)
        centred = centred - centred.mean(dim=0)
        explained = 1.0 if basis is None else float(
            (centred @ basis).var(dim=0).sum() / centred.var(dim=0).sum())
        pair_distance = median_pair_distance(reference)
        for blur in args.blurs:
            scores = {name: float(sinkhorn_divergence(view, reference, blur=blur,
                                                      backend="tensorized"))
                      for name, view in views.items()}
            rows.append({
                "align_dims": n_dims or target.shape[1],
                "blur": blur,
                "constant": scores["constant"],
                "paired_oracle": scores["paired_oracle"],
                "oracle_beats_constant": scores["paired_oracle"] < scores["constant"],
                # Ratio vs constant is the signal the pairing has to ride.
                "oracle_advantage": scores["constant"] / max(scores["paired_oracle"], 1e-12),
                "explained_variance": explained,
                "neighbour_preservation": preserved,
                "blur_over_pair_distance": blur / max(pair_distance, 1e-12),
            })
            flag = "" if rows[-1]["oracle_beats_constant"] else "   CONSTANT WINS"
            print(f"  dims {rows[-1]['align_dims']:>5}  blur {blur:<6g} "
                  f"constant {scores['constant']:.5g}  oracle {scores['paired_oracle']:.5g}  "
                  f"advantage {rows[-1]['oracle_advantage']:>8.2f}x  "
                  f"var {explained:.1%}  nbr {preserved:.1%}  "
                  f"blur/dist {rows[-1]['blur_over_pair_distance']:.3f}"
                  f"{flag}", flush=True)

    frame = pd.DataFrame(rows)
    out = Path(args.out) if args.out else (
        PROJECT_ROOT / "cache" / "results" / "chromatin"
        / f"alignment_geometry_{args.dataset}_{args.law}_{target_layer}_{args.align_block}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)

    usable = frame[frame["oracle_beats_constant"]]
    print(f"\n[geometry] {len(usable)}/{len(frame)} settings where the truth beats a constant")
    if len(usable):
        # Both halves or neither, so a decisive objective in a collapsed neighbourhood is visible.
        ranked = usable.assign(
            score=usable["oracle_advantage"] * usable["neighbour_preservation"])
        best = ranked.loc[ranked["score"].idxmax()]
        print(f"[geometry] best advantage x neighbour preservation: "
              f"align_dims {int(best['align_dims'])}, blur {best['blur']:g} -> "
              f"{best['oracle_advantage']:.2f}x with {best['neighbour_preservation']:.1%} "
              f"of neighbourhoods kept ({best['explained_variance']:.1%} of the variance)")
    print(f"[geometry] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
