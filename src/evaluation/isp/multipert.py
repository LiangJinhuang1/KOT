"""MultiPert predicted KO RNA, transferred into frozen NT-only KOT coordinates.

MultiPert predicts a perturbed cell from a control cell plus a GEARS gene-ontology
embedding, so it is usable as an in-silico perturbation model even though this project
first ran it as a Task A translator. Its prediction lives in its own 5,000-HVG normalised
space; the predicted shift is transferred onto observed NT cells in frozen-KOT
coordinates, then written in KOT feature order for `run_kot_crispr.py task_b`.

SUPERVISION: MultiPert trains on cells of every knockout it scores, and it reads control
ADT alongside control RNA. That is strictly more supervision than the linear ISP (which
sees only other replicates) and far more than KOT itself. Label it accordingly; it is not
a like-for-like competitor.
"""
from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch

from run_kot_crispr import CONTROL_LABEL, MIN_KO_CELLS, MIN_REPLICATES
from src.evaluation.isp.common import (
    apply_mean_shift,
    profile_row,
    usable_perturbations,
    write_isp_outputs,
)
from src.evaluation.isp.frozen import load_frozen_checkpoint
from src.evaluation.regvelo_isp import positive_feature_scales
from src.utils.arrays import to_dense


def add_arguments(parser) -> None:
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="frozen NT-only KOT run root")
    parser.add_argument("--checkpoint", default="best_align")
    parser.add_argument("--multipert-dir", type=Path,
                        default=Path("cache/multipert/output/seed_1"),
                        help="directory holding predict.h5ad and test_control.h5ad")
    parser.add_argument("--perturbation-column", default="perturb")


def predicted_shift(predict: ad.AnnData, control: ad.AnnData,
                    column: str) -> dict[str, np.ndarray]:
    """Per-perturbation mean shift in MultiPert's own normalised gene space.

    predict and control are paired cell for cell, so the difference of their means is the
    model's predicted perturbation effect with its own reconstruction offset included.
    That offset is a property of the model and is left in deliberately.
    """
    groups = predict.obs[column].astype(str).to_numpy()
    if not np.array_equal(groups, control.obs[column].astype(str).to_numpy()):
        raise ValueError("predict.h5ad and test_control.h5ad are not cell-paired")
    predicted = to_dense(predict.X, dtype=np.float64)
    observed = to_dense(control.X, dtype=np.float64)
    return {target: predicted[groups == target].mean(0) - observed[groups == target].mean(0)
            for target in np.unique(groups)}


def run(args) -> None:
    device = torch.device("cpu")
    frozen = load_frozen_checkpoint(args.run_dir, args.checkpoint, args.seed, device)
    rna = frozen.run.rna
    kot = frozen.features.detach().cpu().numpy().astype(np.float64)

    predict = ad.read_h5ad(args.multipert_dir / "predict.h5ad")
    control = ad.read_h5ad(args.multipert_dir / "test_control.h5ad")
    shifts = predicted_shift(predict, control, args.perturbation_column)

    # KOT feature -> MultiPert column, where MultiPert has that gene at all.
    lookup = pd.Index(predict.var_names.astype(str))
    model_indices = lookup.get_indexer(pd.Index(rna.var_names.astype(str)))
    covered = model_indices >= 0
    if not covered.any():
        raise ValueError("MultiPert and the checkpoint share no genes")
    safe_indices = np.where(covered, model_indices, 0)

    groups = rna.obs["gene"].astype(str).to_numpy()
    replicates = rna.obs["replicate"].astype(str).to_numpy()
    nt = groups == CONTROL_LABEL
    # Scale conversion is fitted on NT cells only, matching the RegVelo runner. The
    # preprocessed matrix carries the same cells in the same order as the checkpoint's
    # input, so the two sides of the ratio are the same NT cells in two coordinate systems.
    preprocessed = ad.read_h5ad(args.multipert_dir / "preprocessed_rna.h5ad")
    if list(preprocessed.obs_names) != list(rna.obs_names):
        raise ValueError("MultiPert preprocessed cells do not match the checkpoint's cells")
    multipert_nt = to_dense(preprocessed.X, dtype=np.float64)[nt][:, safe_indices]
    scales = positive_feature_scales(multipert_nt, kot[nt])

    usable = usable_perturbations(groups, replicates, MIN_KO_CELLS, MIN_REPLICATES,
                                  CONTROL_LABEL)
    profiles, coverage = [], []
    for target in usable:
        record = {"perturbation": target, "supported": target in shifts,
                  "n_genes_covered": int(covered.sum()),
                  "n_genes_total": int(len(covered))}
        if target not in shifts:
            record["reason"] = "no MultiPert prediction"
            coverage.append(record)
            continue
        # Features MultiPert does not model receive a zero shift rather than a guess.
        shift = np.where(covered, shifts[target][safe_indices], 0.0)
        query, clipped = apply_mean_shift(kot[nt], shift, scales)
        profiles.append(profile_row(target, query.mean(0)))
        record.update(reason="ok", clipped_fraction=float(clipped.mean()))
        coverage.append(record)

    manifest = {
        "model": "MultiPert perturbation prediction", "seed": args.seed,
        "multipert_dir": str(args.multipert_dir.resolve()),
        "input_coordinates": "frozen KOT Ms",
        "counterfactual": "observed_NT + (predict - test_control) mean shift",
        "unit_conversion": "positive per-feature SD ratio fitted on control cells only",
        "genes_covered": int(covered.sum()), "genes_total": int(len(covered)),
        "supervision": "trains on cells of every knockout it scores; reads control ADT",
        "supported_perturbations": int(len(profiles)),
        "requested_perturbations": len(usable),
    }
    write_isp_outputs(args.out_dir, profiles, coverage, manifest, "MultiPert")
    print(f"[multipert-isp] {len(profiles)}/{len(usable)} perturbations, "
          f"{covered.sum()}/{len(covered)} KOT features covered")
