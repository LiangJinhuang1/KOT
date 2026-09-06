"""Leave-one-replicate-out linear RNA ISP, scored through frozen NT-only KOT.

Unlike the other four runners this one scores end to end rather than writing predicted
RNA for `run_kot_crispr.py task_b`: its protocol is per replicate fold, so the effects
and summary tables are produced here.

SUPERVISION: the ISP reads KO RNA from OTHER replicates; KOT itself stays NT-only. The
regime is a held-out replicate of a SEEN perturbation, not an unseen knockout gene.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch

from run_kot_crispr import (
    EFFECTS_CSV,
    METADATA_TSV,
    MIN_KO_CELLS,
    MIN_REPLICATES,
    PROTEIN_H5AD,
    benchmark_units_adt,
    calibrated_phi,
)
from src.evaluation.crispr_metrics import (
    annotate_effect_sets,
    metric_suite,
    score_effect_subsets,
)
from src.evaluation.isp.common import usable_perturbations
from src.evaluation.isp.frozen import load_frozen_checkpoint
from src.evaluation.linear_isp import fit_linear_isp

CONTROL_LABEL = "NT"


def add_arguments(parser) -> None:
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="best_align")
    parser.add_argument("--effects", type=Path, default=EFFECTS_CSV)
    parser.add_argument("--protein", type=Path, default=PROTEIN_H5AD)
    parser.add_argument("--metadata", type=Path, default=METADATA_TSV)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--n-boot", type=int, default=1000)


def panel_adt(protein_path: Path, metadata_path: Path, panel, obs_names):
    """Benchmark-unit ADT with its columns reordered onto the checkpoint's panel."""
    source = ad.read_h5ad(protein_path, backed="r")
    order = pd.Index(source.var_names).get_indexer(panel)
    source.file.close()
    if (order < 0).any():
        raise ValueError("Checkpoint protein panel differs from benchmark")
    return benchmark_units_adt(protein_path, metadata_path, "rna_size", obs_names)[:, order]


def run(args) -> None:
    if args.n_boot < 1:
        raise SystemExit("--n-boot must be positive")
    # Real velocity only initializes model tensors; prediction uses frozen phi.
    # No kinetic fidelity metric is computed here.
    device = torch.device("cpu")
    frozen = load_frozen_checkpoint(args.run_dir, args.checkpoint, args.seed, device)
    model, features = frozen.model, frozen.features
    rna, protein = frozen.run.rna, frozen.run.protein
    model.load_state_dict(torch.load(frozen.path, map_location=device)["state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    x = features.detach().cpu().numpy()
    groups = rna.obs["gene"].astype(str).to_numpy()
    if "replicate" not in rna.obs or rna.obs["replicate"].isna().any():
        raise ValueError("Complete replicate metadata is required")
    reps = rna.obs["replicate"].astype(str).to_numpy()
    if len(set(reps)) < MIN_REPLICATES:
        raise ValueError("Too few biological replicates")
    observed = annotate_effect_sets(pd.read_csv(args.effects, dtype={"replicate": str}))
    if observed.duplicated(["perturbation", "replicate", "protein"]).any():
        raise ValueError("Duplicate observed effect keys")

    adt = panel_adt(args.protein, args.metadata, protein.var_names, rna.obs_names)
    nt_all = groups == CONTROL_LABEL
    _, slope = calibrated_phi(model, features[nt_all], adt[nt_all],
                              np.ones(nt_all.sum(), dtype=bool))
    if not np.isfinite(slope).all():
        raise ValueError("Collapsed calibrated phi")

    usable = usable_perturbations(groups, reps, MIN_KO_CELLS, MIN_REPLICATES, CONTROL_LABEL)
    protein_rows, rna_rows, profiles, splits, exclusions = [], [], [], [], []
    for rep in sorted(set(reps)):
        fit = fit_linear_isp(x, groups, reps, rep, alpha=args.alpha, min_cells=MIN_KO_CELLS)
        nt = (reps == rep) & (groups == CONTROL_LABEL)
        if nt.sum() < MIN_KO_CELLS:
            exclusions.append({"replicate": rep, "reason": "insufficient NT cells"})
            continue
        training_ids = rna.obs_names[fit.training_rows].tolist()
        test_ids = rna.obs_names[reps == rep].tolist()
        assert not set(training_ids) & set(test_ids)
        splits.append({"replicate": rep, "training_cell_ids": training_ids,
                       "test_cell_ids": test_ids, "trained_targets": list(fit.targets)})
        np.savez_compressed(args.out_dir / f"linear_fold_{len(splits)}.npz",
                            targets=fit.targets, coefficients=fit.coefficients, held_out=rep)
        with torch.no_grad():
            baseline = model.phi(features[nt]).mean(0).numpy()
        for target in usable:
            ko = (reps == rep) & (groups == target)
            if ko.sum() < MIN_KO_CELLS or target not in fit.targets:
                exclusions.append({"replicate": rep, "perturbation": target,
                                   "reason": "insufficient test or training KO cells"})
                continue
            # Inference consumes NT RNA and identity only. KO RNA is accessed for scoring below.
            for label in ("linear", "zero_change"):
                query = fit.predict(x[nt], target) if label == "linear" else x[nt].copy()
                delta = query.mean(0) - x[nt].mean(0)
                actual = x[ko].mean(0) - x[nt].mean(0)
                key = {"model": label, "perturbation": target, "replicate": rep,
                       "seed": args.seed}
                profiles.append({**key, **{f"feature_{i}": float(v)
                                           for i, v in enumerate(query.mean(0))}})
                for i, (pred, obs) in enumerate(zip(delta, actual)):
                    rna_rows.append({**key, "feature": i, "predicted_delta": float(pred),
                                     "observed_delta": float(obs)})
                # Average translated cells, rather than phi(mean RNA), for nonlinear KOT.
                with torch.no_grad():
                    dp = (model.phi(torch.as_tensor(query, dtype=features.dtype))
                          .mean(0).numpy() - baseline) * slope
                for j, protein_name in enumerate(protein.var_names):
                    protein_rows.append({**key, "protein": str(protein_name),
                                         "delta_p_phi": float(dp[j]),
                                         "delta_r_norm": float(np.linalg.norm(actual)),
                                         "predicted_delta_r_norm": float(np.linalg.norm(delta)),
                                         "n_cells": int(ko.sum())})
    if not protein_rows:
        raise ValueError("No evaluable folds")

    effects = pd.DataFrame(protein_rows).merge(
        observed, on=["perturbation", "replicate", "protein"], how="left",
        validate="many_to_one")
    if effects["delta_protein"].isna().any():
        raise ValueError("Some predictions lack observed protein effects")
    effects["supervision"] = "ISP: other-replicate KO RNA; KOT: NT unpaired"
    effects["evaluation_regime"] = "held_out_replicate_seen_perturbation"
    rna_effects = pd.DataFrame(rna_rows)
    summaries = []
    for label, block in effects.groupby("model"):
        scored = score_effect_subsets(block, "delta_p_phi", "delta_protein",
                                      args.n_boot, args.seed,
                                      unit_columns=("perturbation", "replicate"))
        for row in scored.to_dict("records"):
            summaries.append({"model": label, "seed": args.seed, "space": "protein",
                              "phi_estimator": "mean_of_phi", **row})
        rna_block = rna_effects[rna_effects.model == label]
        summaries.append({"model": label, "seed": args.seed, "space": "rna",
                          "effect_subset": "all_eligible",
                          **metric_suite(rna_block.predicted_delta,
                                         rna_block.observed_delta)})

    effects.to_csv(args.out_dir / "crispr_task_b_effects_predicted.csv", index=False)
    rna_effects.to_csv(args.out_dir / "crispr_task_b_rna_effects.csv", index=False)
    pd.DataFrame(profiles).to_csv(args.out_dir / "predicted_rna_profiles.csv", index=False)
    pd.DataFrame(summaries).to_csv(args.out_dir / "crispr_task_b_summary.csv", index=False)
    manifest = {
        "protocol": "leave_one_replicate_out_seen_perturbations", "alpha": args.alpha,
        "checkpoint": str(frozen.path.resolve()), "seed": args.seed,
        "checkpoint_sha256": hashlib.sha256(frozen.path.read_bytes()).hexdigest(),
        "feature_space": "frozen KOT input coordinates", "n_features": x.shape[1],
        "min_ko_cells": MIN_KO_CELLS, "min_replicates": MIN_REPLICATES,
        "nt_preprocessing": "from frozen checkpoint cache; shared NT controls across folds",
        "splits": splits, "exclusions": exclusions,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(pd.DataFrame(summaries).to_string(index=False))
