"""Fit restriction, stamped on the output so scoring does not guess from filenames.

A method may not use a held-out cell's second-modality measurement at fit time.
oos is declared on the model registry:

  native          restriction in the loss; still maps every cell (KOT).
  holdout_second  full RNA, second modality restricted (OT baselines; they still
                  see held-out RNA).
  paired          both modalities of the same cell — a leakage ceiling.

Orthogonal to the mode: `uses_fit_pairing` says whether the method read WHICH fitted
RNA goes with WHICH fitted protein. A supervised translator does and KOT does not, so
the two sit in the same protocol but not on the same footing.

`supervision_regime` collapses those two axes into the one label a results table has to
carry. Methods in different regimes were not given the same information, and a table
that ranks them in one list implies they were.
"""

from __future__ import annotations

import numpy as np

from src.training.registry import (
    DIRECT,
    HOLDOUT_SECOND,
    LATENT,
    MODELS as REGISTERED_MODELS,
    NATIVE,
    PAIRED,
    PAIRED_FIT_MODELS,
)

PROTOCOL_KEY = "kot_protocol"

# Pre-stamp files had no fit restriction, so they are in-distribution.
LEGACY_PROTOCOL = {
    "oos_mode": "unstamped",
    "fit_obs_key": "",
    "fit_obs_values": [],
    "n_fit_cells": -1,
    "n_cells": -1,
    "predictor": "",
    "out_of_distribution": False,
    "paired_oracle": False,
    "uses_fit_pairing": False,
}


# What a run was actually shown, across both axes. Ordered weakest to strongest: a
# method lower in this list was given strictly less than one above it.
UNPAIRED_FIT = "unpaired_fit"
PAIRED_FIT = "paired_fit"
UNPAIRED_ALL_CELLS = "unpaired_all_cells"
PAIRED_ALL_CELLS = "paired_all_cells"

SUPERVISION_DESCRIPTION = {
    UNPAIRED_FIT: ("both modalities of the fitted cells as unpaired distributions; "
                   "no perturbed cell's protein"),
    PAIRED_FIT: ("the fitted cells' (RNA, protein) PAIRING; no perturbed cell's protein"),
    UNPAIRED_ALL_CELLS: ("both modalities of every cell, unpaired, including the "
                         "perturbations it is scored on"),
    PAIRED_ALL_CELLS: ("the (RNA, protein) pairing of every cell, including the "
                       "perturbations it is scored on"),
}


def supervision_regime(protocol: dict) -> str:
    """What this run was shown, as one label for the results table.

    Two runs can share an `oos_mode` and still not be comparable: holdout_second covers
    both an unpaired OT method and a supervised regression, and an unrestricted run has
    seen the very perturbations it is scored on. Naming the regime is what stops a
    reader inferring that one number beat another under equal conditions.
    """
    if not protocol["out_of_distribution"]:
        return PAIRED_ALL_CELLS if (protocol["uses_fit_pairing"] or protocol["paired_oracle"]) \
            else UNPAIRED_ALL_CELLS
    return PAIRED_FIT if protocol["uses_fit_pairing"] else UNPAIRED_FIT


def resolve_oos_mode(model_name: str) -> str:
    if model_name.startswith("kot"):
        return NATIVE
    spec = REGISTERED_MODELS.get(model_name)
    if spec is None:
        raise KeyError(
            f"model '{model_name}' has no declared out-of-sample capability. Add it to "
            f"MODELS in src.training.registry -- a model whose capability is unknown "
            "cannot be run under a fit_obs_key protocol without silently deciding "
            "what it means."
        )
    return spec["oos"]


def predictor_kind(run_cfg: dict, model_name: str) -> str:
    """Whether the aligned RNA output is already in second-modality coordinates.

    Every registered model declares this, because a dataset-level
    `kot_use_feature_space` otherwise stamped the latent baselines run beside KOT as
    direct ADT predictors. Only the kot_* family reads the config switch, since for
    them it genuinely is one.
    """
    if model_name.startswith("kot"):
        return DIRECT if bool(run_cfg.get("kot_use_feature_space", False)) else LATENT
    spec = REGISTERED_MODELS.get(model_name)
    if spec is None or "predictor" not in spec:
        raise KeyError(
            f"model '{model_name}' does not declare what its aligned RNA output is. Add "
            f"'predictor': DIRECT or LATENT to MODELS in src.training.registry. Guessing "
            "from the config is what stamped every baseline on papalexi_retained as a "
            "direct ADT predictor, because that dataset sets kot_use_feature_space for KOT."
        )
    return spec["predictor"]


def plan_fit_restriction(run_cfg: dict, model_name: str, fit_rows: np.ndarray | None,
                         second_label: str) -> tuple[np.ndarray | None, str]:
    """Fit rows this model will honour. A paired oracle returns None so it is not
    credited with a holdout it cannot keep.
    """
    mode = resolve_oos_mode(model_name)
    if fit_rows is None or mode != PAIRED:
        return fit_rows, mode
    if not bool(run_cfg.get("allow_paired_oracle", False)):
        raise ValueError(
            f"'{model_name}' encodes both modalities of the same cell, so a cell with no "
            f"{second_label} cannot be embedded and the fit restriction "
            f"({run_cfg.get('fit_obs_key')} in {list(run_cfg.get('fit_obs_values', []))}) "
            "cannot be honoured. It is a paired oracle on this benchmark, not a "
            "cross-modal predictor. Set allow_paired_oracle=true to run it anyway; its "
            "predictions are then stamped in-distribution and reported as a leakage "
            "ceiling instead of being ranked."
        )
    return None, mode


def restrict_second_modality(context: dict, fit_rows: np.ndarray) -> dict:
    """Keep `fit_rows` so ALS can recover x[fit_rows]."""
    restricted = dict(context)
    restricted["y"] = context["y"][fit_rows]
    restricted["second_adata"] = context["second_adata"][fit_rows].copy()
    restricted["fit_rows"] = fit_rows
    return restricted


def scatter_rows(values: np.ndarray, rows: np.ndarray, n_total: int) -> np.ndarray:
    """Place `values` at `rows`; NaN elsewhere so a forgotten mask cannot look like
    the origin.
    """
    out = np.full((n_total, values.shape[1]), np.nan, dtype=np.float32)
    out[rows] = values
    return out


def protocol_stamp(run_cfg: dict, model_name: str, n_cells: int,
                   fit_rows: np.ndarray | None) -> dict:
    """Scalars and lists only: h5ad uns cannot store None."""
    mode = resolve_oos_mode(model_name)
    restricted = fit_rows is not None
    return {
        "oos_mode": mode,
        "fit_obs_key": str(run_cfg.get("fit_obs_key") or ""),
        "fit_obs_values": [str(v) for v in (run_cfg.get("fit_obs_values") or [])],
        "n_fit_cells": int(len(fit_rows)) if restricted else int(n_cells),
        "n_cells": int(n_cells),
        "predictor": predictor_kind(run_cfg, model_name),
        "out_of_distribution": bool(restricted),
        "paired_oracle": mode == PAIRED,
        "uses_fit_pairing": model_name in PAIRED_FIT_MODELS,
    }


def read_protocol(uns) -> dict:
    if PROTOCOL_KEY not in uns:
        return dict(LEGACY_PROTOCOL)
    stamp = dict(uns[PROTOCOL_KEY])
    return {
        "oos_mode": str(stamp["oos_mode"]),
        "fit_obs_key": str(stamp["fit_obs_key"]),
        "fit_obs_values": [str(v) for v in stamp["fit_obs_values"]],
        "n_fit_cells": int(stamp["n_fit_cells"]),
        "n_cells": int(stamp["n_cells"]),
        "predictor": str(stamp["predictor"]),
        "out_of_distribution": bool(stamp["out_of_distribution"]),
        "paired_oracle": bool(stamp["paired_oracle"]),
        # Added after the first stamped runs; those files predate supervised baselines.
        "uses_fit_pairing": bool(stamp.get("uses_fit_pairing", False)),
    }
