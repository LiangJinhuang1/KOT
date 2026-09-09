"""Shared RNA coordinates and external RNA-decay anchors for regulatory R2."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse

SHARED_SPLICED = "spliced_shared_lognorm"


def shared_splicing_targets(adata, columns=None) -> np.ndarray:
    """log1p(u), log1p(s) under ONE factor 1e4 / sum(u+s), before gene selection.

    This is a fixed coordinate convention, not a library-size dynamical model.
    Source-cell RNA never enters the ODE. Raw and existing layers stay unchanged.
    """
    u = sparse.csr_matrix(adata.layers["unspliced"], dtype=np.float32)
    s = sparse.csr_matrix(adata.layers["spliced"], dtype=np.float32)
    if any(not np.isfinite(x.data).all() or (x.data < 0).any() for x in (u, s)):
        raise ValueError("Shared RNA normalization needs finite nonnegative counts")
    totals = np.asarray((u + s).sum(axis=1)).ravel()
    factor = sparse.diags(1e4 / np.maximum(totals, 1.0))
    selected = slice(None) if columns is None else columns
    return np.concatenate([
        np.log1p((factor @ x)[:, selected].toarray()) for x in (u, s)
    ], axis=1).astype(np.float32)


def load_gamma_anchors(path: str, gene_names, *, target: float = 0.5,
                       hours_per_model_time: float | None = None) -> dict:
    """External RNA decay; never reuse protein half-life priors.

    Default: relative rates with whole-table geometric mean set to target in model
    time, BEFORE panel intersection. Explicit hours_per_model_time uses physical
    rates. Neither option makes the per-cell kappa identifiable as elapsed time.
    """
    frame = pd.read_csv(path)
    required = {"gene_symbol", "molecule", "source"}
    if not required.issubset(frame.columns):
        raise ValueError(f"RNA anchor CSV requires {sorted(required)} and gamma_per_hour or half_life_hours")
    if not frame["molecule"].astype(str).str.lower().isin(["rna", "mrna"]).all():
        raise ValueError("Gamma anchors must measure RNA decay, not protein turnover")
    if frame.empty or frame[list(required)].isna().any().any():
        raise ValueError("RNA anchor table and gene/source fields must be nonempty")
    if frame[list(required)].astype(str).apply(lambda col: col.str.strip().eq("")).any().any():
        raise ValueError("RNA anchor gene/source fields must be nonempty")
    if frame["gene_symbol"].astype(str).str.fullmatch(r"\d+(?:\.\d+)?").any():
        raise ValueError("RNA anchors contain ambiguous numeric gene symbols; rebuild the table")
    if "gamma_per_hour" in frame:
        rates = pd.to_numeric(frame["gamma_per_hour"], errors="raise").to_numpy(float)
    elif "half_life_hours" in frame:
        half = pd.to_numeric(frame["half_life_hours"], errors="raise").to_numpy(float)
        if not np.isfinite(half).all() or (half <= 0).any():
            raise ValueError("RNA half-lives must be finite and positive")
        rates = np.log(2.0) / half
    else:
        raise ValueError("RNA anchor CSV needs gamma_per_hour or half_life_hours")
    weights = (pd.to_numeric(frame["anchor_weight"], errors="raise").to_numpy(float)
               if "anchor_weight" in frame else np.ones(len(frame)))
    if not np.isfinite(rates).all() or (rates <= 0).any():
        raise ValueError("RNA decay rates must be finite and positive")
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Anchor weights must be finite and positive")
    frame = frame.assign(rate=rates, weight=weights)
    if frame["gene_symbol"].duplicated().any():
        raise ValueError("Aggregate repeated RNA measurements to one row per gene before fitting")
    conversion = (float(hours_per_model_time) if hours_per_model_time is not None
                  else float(target) / float(np.exp(np.log(rates).mean())))
    if not np.isfinite(conversion) or conversion <= 0:
        raise ValueError("Gamma time conversion must be finite and positive")
    indexed = frame.set_index("gene_symbol")
    indices = [i for i, name in enumerate(gene_names) if name in indexed.index]
    if not indices:
        raise ValueError("No RNA-decay anchors match the RNA training gene panel")
    matched = indexed.loc[[gene_names[i] for i in indices]]
    return {
        "indices": indices, "gene_names": list(matched.index),
        "gamma": (matched["rate"].to_numpy() * conversion).tolist(),
        "weights": matched["weight"].tolist(), "source": matched["source"].tolist(),
        "path": str(Path(path).resolve()),
        "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "hours_per_model_time": conversion,
        "time_convention": "physical_conversion" if hours_per_model_time is not None else "relative_geomean",
        "table_n_genes": len(frame),
    }


def gamma_anchor_loss(gamma: torch.Tensor, anchors: dict) -> torch.Tensor:
    """Weighted squared log-rate error on gamma only; beta is splicing."""
    targets = gamma.new_tensor(anchors["gamma"])
    weights = gamma.new_tensor(anchors["weights"])
    residual = gamma[anchors["indices"]].log() - targets.log()
    return (weights * residual.square()).sum() / weights.sum()
