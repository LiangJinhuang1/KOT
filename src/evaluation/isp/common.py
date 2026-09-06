"""Pieces every Task B in-silico-perturbation runner shares.

Kept to numpy/pandas so the GEARS and scGPT virtualenvs, which cannot import the KOT
stack, can still use it. Anything that reads a frozen checkpoint lives in frozen.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.regvelo_isp import counterfactual_query


def usable_perturbations(groups, replicates, min_ko_cells: int, min_replicates: int,
                         control_label: str) -> list[str]:
    """Knockouts sampled deeply enough in enough replicates, control excluded.

    Eligibility depends only on cell counts, so every runner settles it before a model
    is fitted and no arm can widen the scoring set after seeing its own result.
    """
    counts = pd.crosstab(groups, replicates)
    eligible = counts.index[(counts >= min_ko_cells).sum(axis=1) >= min_replicates]
    return sorted(set(eligible) - {control_label})


def profile_row(target: str, values) -> dict:
    """One predicted_rna.csv row: the knockout plus its profile in KOT feature order."""
    return {"perturbation": target,
            **{f"feature_{j}": float(value) for j, value in enumerate(values)}}


def apply_mean_shift(observed_nt, shift, feature_scales):
    """Add one predicted mean shift to every observed NT cell, in KOT coordinates.

    RegVelo produces a fitted pair per cell; MultiPert and scGPT produce a single mean
    shift per perturbation. The latter enter counterfactual_query as that shift broadcast
    over the NT block, against a zero baseline, so both go through one transfer path.
    """
    rows = np.broadcast_to(shift, (len(observed_nt), len(shift)))
    return counterfactual_query(observed_nt, np.zeros_like(rows), rows,
                                np.ones(len(shift)), np.arange(len(shift)), feature_scales)


def write_isp_outputs(out_dir: Path, profiles: list[dict], coverage: list[dict],
                      manifest: dict, model_label: str) -> None:
    """The three files `run_crispr_isp.py` hands to `run_kot_crispr.py task_b`."""
    if not profiles:
        raise ValueError(
            f"{model_label} predicted none of the sufficiently sampled perturbations")
    pd.DataFrame(profiles).to_csv(out_dir / "predicted_rna.csv", index=False)
    pd.DataFrame(coverage).to_csv(out_dir / "coverage.csv", index=False)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[out] {out_dir / 'predicted_rna.csv'}")
