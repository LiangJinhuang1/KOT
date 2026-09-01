"""Trainable models: import path plus `oos` capability.

protocol.resolve_oos_mode reads `oos`; the runner reads module/function.
Unlisted `kot_*` names are still native; listed names must still carry `oos`.
"""

from __future__ import annotations

# Re-exported by protocol.py. Do not import protocol here (cycle with runner).
NATIVE = "native"
HOLDOUT_SECOND = "holdout_second"
PAIRED = "paired"


def kot_spec() -> dict:
    return {
        "module": "src.training.kot",
        "function": "run_kot",
        "oos": NATIVE,
        "extra": "kot",
    }


MODELS = {
    "scot": {
        "module": "src.training.scot",
        "function": "run_scot",
        "oos": HOLDOUT_SECOND,
        "batch_split": True,
    },
    "moscot": {
        "module": "src.training.moscot",
        "function": "run_moscot",
        "oos": HOLDOUT_SECOND,
        "extra": "baselines",
        "batch_split": True,
    },
    "glue": {
        "module": "src.training.glue",
        "function": "run_glue",
        "oos": HOLDOUT_SECOND,
        "extra": "baselines",
        "counts": True,
    },
    "uniport": {
        "module": "src.training.uniport",
        "function": "run_uniport",
        "oos": HOLDOUT_SECOND,
    },
    "maxfuse": {
        "module": "src.training.maxfuse",
        "function": "run_maxfuse",
        "oos": HOLDOUT_SECOND,
        "extra": "baselines",
    },
    "totalvi": {
        "module": "src.training.totalvi",
        "function": "run_totalvi",
        "oos": PAIRED,
        "extra": "baselines",
        "counts": True,
    },
    "linear_ode": {
        "module": "src.training.linear_ode",
        "function": "run_linear_ode",
        "oos": HOLDOUT_SECOND,
        "batch_split": True,
    },
    "kot": kot_spec(),
    "kot_nodyn": kot_spec(),
    "kot_fixedkappa": kot_spec(),
    "kot_fixedalpha": kot_spec(),
    "kot_oracle": kot_spec(),
    "kot_oracle_learnalpha": kot_spec(),
    "kot_oracle_learnalpha_kappa": kot_spec(),
    "kot_unbounded": kot_spec(),
    "kot_unbounded_sn": kot_spec(),
    "kot_noanchor": kot_spec(),
}

COUNT_MODELS = {name for name, spec in MODELS.items() if spec.get("counts")}
# O(n²) OT methods: with align_batch_key they run per biological batch.
BATCH_SPLIT_MODELS = {name for name, spec in MODELS.items() if spec.get("batch_split")}

MODEL_OVERRIDES = {
    "kot_nodyn": {"lambda_dyn": 0.0},
    "kot_fixedkappa": {
        "kot_fixed_kappa": 0.6931471805599453,
        "kot_fixed_alpha": None,
        "kot_alpha_min": 1.0e-3,                  # learned-α box mirrors kot_fixedalpha's learned-κ box
        "kot_alpha_max": 1.5,
        "lambda_kappa_prior": 0.0,
        "g_freeze_epochs": 0,
        "phi_spectral_norm": False,               # match kot_fixedalpha (φ-side confound)
        "g_dims": [256, 128],
    },
    "kot_fixedalpha": {
        "kot_fixed_alpha": 0.6931471805599453,
        "kot_fixed_kappa": None,
        "kot_kappa_min": 1.0e-3,
        "kot_kappa_max": 1.5,
        "kot_kappa_prior": 0.6931471805599453,
        "lambda_kappa_prior": 0.01,
        "kot_alpha_min": 1.0e-6,
        "kot_alpha_max": None,
        "g_freeze_epochs": 0,
        "phi_spectral_norm": False,
        "g_dims": [256, 128],
    },
    "kot_oracle": {
        "kot_fixed_alpha": 1.0,
        "kot_fixed_kappa": 1.0,
        "kot_fixed_beta": 0.5,
        "kot_kappa_min": 1.0,
        "kot_kappa_max": 1.0,
        "use_anchor": False,
        "kot_anchor_indices": [],
        "kot_anchor_betas": [],
        "lambda_prior": 0.0,
        "lambda_kappa_prior": 0.0,
        "kot_alpha_min": 1.0,
        "kot_alpha_max": 1.0,
        "g_freeze_epochs": 0,
        "phi_spectral_norm": False,
        "g_dims": [256, 128],
    },
    "kot_oracle_learnalpha": {
        "kot_fixed_alpha": None,
        "kot_fixed_kappa": 1.0,
        "kot_fixed_beta": 0.5,
        "kot_kappa_min": 1.0,
        "kot_kappa_max": 1.0,
        "use_anchor": False,
        "kot_anchor_indices": [],
        "kot_anchor_betas": [],
        "lambda_prior": 0.0,
        "lambda_kappa_prior": 0.0,
        "kot_alpha_min": 1.0e-3,
        "kot_alpha_max": 1.5,
        "g_freeze_epochs": 0,
        "phi_spectral_norm": False,
        "g_dims": [256, 128],
    },
    "kot_oracle_learnalpha_kappa": {
        "kot_fixed_alpha": None,
        "kot_fixed_kappa": None,
        "kot_fixed_beta": 0.5,
        "kot_kappa_min": 1.0e-3,
        "kot_kappa_max": 1.5,
        "use_anchor": False,
        "kot_anchor_indices": [],
        "kot_anchor_betas": [],
        "lambda_prior": 0.0,
        "lambda_kappa_prior": 0.0,
        "kot_alpha_min": 1.0e-3,
        "kot_alpha_max": 1.5,
        "g_freeze_epochs": 0,
        "phi_spectral_norm": False,
        "g_dims": [256, 128],
    },
    "kot_unbounded": {
        "kot_kappa_min": 1.0e-6,
        "kot_kappa_max": None,
        "kot_alpha_min": 1.0e-6,
        "kot_alpha_max": None,
        "lambda_kappa_prior": 0.0,
        "g_freeze_epochs": 0,
        "phi_spectral_norm": False,
        "g_dims": [256, 128],
    },
    "kot_unbounded_sn": {
        "kot_kappa_min": 1.0e-6,
        "kot_kappa_max": None,
        "kot_alpha_min": 1.0e-6,
        "kot_alpha_max": None,
        "lambda_kappa_prior": 0.0,
        "g_freeze_epochs": 0,
        "phi_spectral_norm": True,
        "g_dims": [256, 128],
    },
    # Default `kot` already anchors; use_anchor=True was a no-op (210/210 paired runs).
    "kot_noanchor": {
        "use_anchor": False,
        "kot_anchor_indices": [],
        "kot_anchor_betas": [],
        "beta_anchor_csv": None,
        "lambda_prior": 0.0,
    },
}

# Stale --models names whose overrides were no-ops. Fail instead of duplicating `kot`.
RETIRED_MODELS = {
    "kot_anchor": (
        "kot_anchor applied {'use_anchor': True}, which is already the default, so "
        "it produced results bit-identical to `kot` in 210/210 paired runs. Use "
        "`kot` for the anchored arm and `kot_noanchor` for the anchor-free ablation."
    ),
}


def extra_for(model_name: str) -> str | None:
    """pip extra for a missing import, if this model has one."""
    spec = MODELS.get(model_name)
    if spec is not None and spec.get("extra"):
        return spec["extra"]
    if model_name.startswith("kot"):
        return MODELS["kot"]["extra"]
    return None
