#!/usr/bin/env python3
"""
Post-hoc trajectory DTW from saved KOT checkpoints.

For every (run, model, seed, checkpoint) it loads `checkpoint_<name>.pt`,
restores the model, maps RNA -> protein with phi, and computes the trajectory
DTW distance against the observed protein trajectory along pseudotime:

  traj_dtw_temporal : phi(r) ordered by PREDICTED pseudotime (nearest-neighbour
                      protein match) vs observed protein ordered by TRUE
                      pseudotime — recovery of the temporal ordering.
  traj_dtw_recon    : both ordered by TRUE pseudotime — warp-tolerant
                      reconstruction score.

This is the axis FOSCTTM (matching) and JVP-cosine (physics) do not cover, so it
is written to its own summary rather than folded into the FOSCTTM eval. Results
are saved per run (so they travel with the run when it is copied) plus one
combined aggregate:

  <run_folder>/trajectory_dtw_summary.csv     (one file per run)
  cache/results/trajectory_dtw_summary.csv    (all runs combined; --output)

Usage:
  python tools/compute_trajectory_dtw.py                       # all runs
  python tools/compute_trajectory_dtw.py --runs cache/training/run_20260617_085640
  python tools/compute_trajectory_dtw.py --runs "cache/training/run_20260617_*"
  python tools/compute_trajectory_dtw.py --n-bins 50 --output my_dtw.csv

Reuses load_dataset_for_run / build_model_and_tensors / collect_run_folders from
evaluate_checkpoints.py, so the model architecture and input tensors match those
used at training time exactly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch

from src.evaluation.trajectory_dtw import trajectory_dtw
from src.training.kot import nearest_protein_match, resolve_pseudotime
from tools.evaluate_checkpoints import (
    CHECKPOINT_NAMES,
    build_model_and_tensors,
    collect_run_folders,
    load_dataset_for_run,
)


DEFAULT_OUTPUT = Path("cache/results/trajectory_dtw_summary.csv")


def dtw_for_checkpoint(model, ckpt_path: Path, R_t, P_t, rna_obs, n_bins: int):
    """Load a checkpoint into model and compute both trajectory DTW variants."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = {k: v.to(R_t.device) for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(state_dict)
    model.eval()

    pseudotime, _ = resolve_pseudotime(rna_obs)
    if pseudotime is None:
        return int(ckpt["epoch"]), float("nan"), float("nan")

    with torch.no_grad():
        phi = model.phi(R_t)
        nn_idx = nearest_protein_match(phi, P_t)[0].cpu().numpy()

    phi_np = phi.cpu().numpy()
    p_np = P_t.cpu().numpy()
    t_pred = pseudotime[nn_idx]

    temporal = trajectory_dtw(phi_np, p_np, t_pred, pseudotime, n_bins=n_bins)
    recon = trajectory_dtw(phi_np, p_np, pseudotime, pseudotime, n_bins=n_bins)
    return int(ckpt["epoch"]), temporal, recon


def dtw_for_run(run_folder: Path, device, n_bins: int) -> list[dict]:
    """Compute trajectory DTW for every checkpoint under one run folder."""
    print(f"\n=== {run_folder.name} ===")

    rows = []
    data_cache = {}
    for model_dir in sorted(p for p in run_folder.iterdir() if p.is_dir()):
        model_name = model_dir.name
        if model_name in {"cache", "results"} or model_name.startswith("."):
            continue

        for dataset_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            dataset_name = dataset_dir.name
            if dataset_name not in data_cache:
                try:
                    data_cache[dataset_name] = load_dataset_for_run(run_folder, dataset_name)
                except Exception as e:
                    print(f"  [{dataset_name}] load_dataset FAILED: {e}")
                    continue
            rna_adata, protein_adata, run_cfg, _ = data_cache[dataset_name]

            try:
                model, R_t, _V_t, P_t, _S_t, rna_obs, _mask_t, _align_cols = build_model_and_tensors(
                    run_cfg, rna_adata, protein_adata, model_name, device,
                )
            except Exception as e:
                print(f"  [{model_name}/{dataset_name}] build_model FAILED: {e}")
                continue

            for seed_dir in sorted(
                p for p in dataset_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")
            ):
                tail = seed_dir.name.split("_", 1)[1]
                if not tail.isdigit():
                    continue
                seed = int(tail)
                for ckpt_name in CHECKPOINT_NAMES:
                    ckpt_path = seed_dir / f"checkpoint_{ckpt_name}.pt"
                    if not ckpt_path.exists():
                        continue

                    epoch, temporal, recon = dtw_for_checkpoint(
                        model, ckpt_path, R_t, P_t, rna_obs, n_bins,
                    )
                    rows.append({
                        "run":                run_folder.name,
                        "model":              model_name,
                        "dataset":            dataset_name,
                        "seed":               seed,
                        "checkpoint":         ckpt_name,
                        "checkpoint_epoch":   epoch,
                        "traj_dtw_temporal":  temporal,
                        "traj_dtw_recon":     recon,
                    })
                    print(f"  ✓ {model_name}/{dataset_name}/seed_{seed:>4}/{ckpt_name:<11} "
                          f"DTW_temporal={temporal:.4f}  DTW_recon={recon:.4f}")

    if rows:
        per_run_out = run_folder / "trajectory_dtw_summary.csv"
        pd.DataFrame(rows).to_csv(per_run_out, index=False)
        print(f"  → wrote {len(rows)} rows to {per_run_out}")

    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs", nargs="*", default=None,
        help="Run folder paths or glob patterns (default: all cache/training/run_*)",
    )
    parser.add_argument(
        "--output", default=str(DEFAULT_OUTPUT),
        help=f"Output CSV path for the DTW summary (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--n-bins", type=int, default=100,
        help="Pseudotime bins for the trajectory (default: 100).",
    )
    args = parser.parse_args()

    run_folders = collect_run_folders(args)
    if not run_folders:
        print("No run folders found.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Found {len(run_folders)} run folder(s). n_bins={args.n_bins}")

    all_rows = []
    for run_folder in run_folders:
        all_rows.extend(dtw_for_run(run_folder, device, args.n_bins))

    if not all_rows:
        print("\nNo rows produced.")
        return

    df = pd.DataFrame(all_rows)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print()
    print(f"✓ Wrote {len(df)} rows to {out_path}")
    print("Preview (best traj_dtw_temporal per checkpoint):")
    preview = df.sort_values("traj_dtw_temporal").groupby("checkpoint").head(3)
    print(preview[["run", "model", "dataset", "seed", "checkpoint", "checkpoint_epoch",
                   "traj_dtw_temporal", "traj_dtw_recon"]].to_string(index=False))


if __name__ == "__main__":
    main()
