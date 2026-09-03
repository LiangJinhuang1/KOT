"""
KOT training loop: Sinkhorn alignment plus kinetic ODE optimisation.

The objective follows the paper terms:
  L = L_sinkhorn + lambda_dyn * L_kinetics + lambda_prior * L_anchor
      + lambda_reg * L_reg,
with L_anchor active only when an explicit anchor set H is configured.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import NamedTuple

import anndata as ad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch
import torch.nn.functional as F
from torch.func import jvp as torch_jvp

from src.utils.arrays import matrix_from_adata, to_dense
from src.models.KOT import KOTModel
from src.losses.sinkhorn import sinkhorn_divergence
from src.losses.jvp_physics import kinetics_loss
from src.data.projection import projection_matrix_from_adatas
from src.data.adt_gene_map import load_mapping_records
from src.data.beta_anchor import resolve_beta_anchors
from src.training.protein_supervised import adt_targets, protein_target_units
from src.data.splits import (
    group_holdout_mask,
    held_out_mask,
    selection_validation_mask,
    resolve_split_seed,
    split_digest,
)
from src.evaluation.foscttm import calc_domainAveraged_FOSCTTM
from src.evaluation.trajectory_dtw import trajectory_dtw
from src.visualization import FOSCTTM_CHANCE, METHOD_COLORS, MODALITY_COLORS
from src.visualization.style import (
    apply_style, chance_line, figsize, ink, panel_letter, save_figure,
)


def to_tensor(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(arr, dtype=torch.float32, device=device)


def to_index_tensor(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    """Row indices for gathering; int64 because torch indexing requires it."""
    return torch.tensor(arr, dtype=torch.long, device=device)


def choose_torch_device(cfg: dict) -> torch.device:
    requested = str(cfg.get("device", "auto")).lower()
    gpu_allocated = bool(
        os.environ.get("SLURM_JOB_GPUS")
        or os.environ.get("SLURM_STEP_GPUS")
        or os.environ.get("CUDA_VISIBLE_DEVICES")
    )
    if requested == "cpu":
        return torch.device("cpu")
    if requested not in {"auto", "cuda", "gpu"}:
        raise ValueError(f"Unsupported device setting: {requested!r}. Expected auto, cuda, or cpu.")
    if not torch.cuda.is_available():
        if requested in {"cuda", "gpu"} or gpu_allocated:
            raise RuntimeError(
                "A GPU allocation is present, but torch.cuda.is_available() is false. "
                "This usually means the allocated node/GPU failed CUDA initialization."
            )
        return torch.device("cpu")
    torch.empty(1, device="cuda")
    return torch.device("cuda")


def beta_anchor_prior_loss(
    beta: torch.Tensor,
    indices: torch.Tensor,
    targets: torch.Tensor,
    sigmas: torch.Tensor | None,
    weights: torch.Tensor | None,
    lambda_prior: float,
    prior_kind: str,
) -> torch.Tensor:
    """Soft β prior: l2 | gaussian | log_normal (not hard-fixed).

    A WEIGHTED MEAN over anchors, not a sum — so a 24-anchor set does not impose 6x
    the total prior force of a 4-anchor set (which would confound anchor-count
    ablations). With weights: Σ(w·r²)/(Σw+ε); without: mean(r²).
    """
    pred = beta[indices]
    if prior_kind == "l2":
        sq = (pred - targets).pow(2)
        scale = 1.0
    else:
        if sigmas is None:
            raise ValueError(f"prior_kind={prior_kind!r} requires per-anchor sigmas")
        if prior_kind == "gaussian":
            resid = (pred - targets) / sigmas.clamp(min=1e-6)
        elif prior_kind == "log_normal":
            # σ stored on β-scale → approximate log-space width as σ/β*
            log_sigma = (sigmas / targets.clamp(min=1e-6)).clamp(min=1e-3)
            resid = (
                torch.log(pred.clamp(min=1e-6)) - torch.log(targets.clamp(min=1e-6))
            ) / log_sigma
        else:
            raise ValueError(f"Unknown beta_anchor_prior {prior_kind!r}; use l2|gaussian|log_normal")
        sq = resid.pow(2)
        scale = 0.5
    if weights is not None:
        loss = (weights * sq).sum() / (weights.sum() + 1e-8)
    else:
        loss = sq.mean()
    return lambda_prior * scale * loss


def kot_param_groups(model, lr: float, lr_phi, lr_alpha_kappa, lr_beta) -> list[dict]:
    """One Adam group per head, so phi, (alpha, kappa) and beta can take different steps.

    alpha and kappa share a group deliberately: they are the two heads of the same
    kinetics term and trade against each other (kappa absorbs the pseudotime scale that
    alpha is measured on), so separate step sizes would change WHICH head wins that
    trade rather than just how fast both converge.

    A head with no parameters is dropped -- the FixedKappa / FixedAlpha ablations
    register buffers, not parameters. `base_lr` is the unscheduled rate; both the
    scheduler and phase3_lr_scale multiply it rather than the current `lr`, so neither
    compounds on the other's output.
    """
    heads = [
        ("phi",         lr_phi,         list(model.phi.parameters())),
        ("alpha_kappa", lr_alpha_kappa,
         list(model.g.parameters()) + list(model.kappa.parameters())),
        ("beta",        lr_beta,        [model.beta_raw]),
    ]
    groups = []
    for name, head_lr, params in heads:
        if not params:
            continue
        resolved = float(lr if head_lr is None else head_lr)
        groups.append({"name": name, "params": params, "lr": resolved, "base_lr": resolved})
    return groups


def lr_schedule_factor(epoch_index: int, warmup_epochs: int, n_epochs: int,
                       start_factor: float, min_factor: float) -> float:
    """Multiplier on every group's base lr: start_factor -> 1 -> min_factor.

    Linear warmup over the first `warmup_epochs`, then a cosine down to `min_factor`
    over the remaining n_epochs - warmup_epochs. `epoch_index` is 0-based, which is
    what LambdaLR passes.

    ONE factor for ALL groups, applied multiplicatively: the ratios between the
    per-head learning rates then hold for the whole run. A per-group schedule would
    silently re-weight the heads against each other as training progressed, which is
    exactly the confound the separate rates are meant to control.

    warmup_epochs=0 with start_factor=1 and min_factor=0 reproduces the
    CosineAnnealingLR(T_max=n_epochs) this replaced, epoch for epoch.
    """
    if warmup_epochs > 0 and epoch_index < warmup_epochs:
        return start_factor + (1.0 - start_factor) * epoch_index / warmup_epochs
    span = max(1, n_epochs - warmup_epochs)
    progress = min(1.0, max(0.0, (epoch_index - warmup_epochs) / span))
    return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def dyn_weight_factor(epoch: int, dyn_warmup_epochs: int) -> float:
    """Multiplier on lambda_dyn for a 1-based `epoch`: a linear ramp to 1 over W epochs.

    The kinetics weight has exactly ONE free parameter -- when it arrives. Its ceiling
    is lambda_dyn and its floor is 0, so W is the whole schedule: the term is at
    lambda_dyn/W on the first epoch and at full strength on epoch W, not W+1. W=0 is
    joint training from epoch 1.

    Top-level and single-purpose so the curriculum can be tested without standing up a
    training run, the same reason lr_schedule_factor is.
    """
    if dyn_warmup_epochs <= 0:
        return 1.0
    return min(1.0, epoch / dyn_warmup_epochs)


def phase_lr_factor(optimiser) -> float:
    """The factor the groups' current lrs sit at, relative to their base_lr.

    Both lr mechanisms are multiplicative and uniform across the heads -- the LambdaLR
    factor in scheduler mode, phase3_lr_scale in phase mode -- so one group answers for
    all of them, and reading the factor back out of the optimiser keeps the call sites
    from having to know which of the two is live. In phase mode it is 1.0 until phase 3
    and phase3_lr_scale after; with a scheduler it is lr_schedule_factor of the epoch
    that has been stepped to.
    """
    group = optimiser.param_groups[0]
    return float(group["lr"] / group["base_lr"])


def format_group_lrs(optimiser, factor: float) -> str:
    """`lr=1.00e-03` when the heads share a rate, else one field per head.

    Takes the schedule factor rather than reading pg["lr"]: the scheduler has already
    advanced by the time an epoch is logged, so reading the group would report the rate
    the NEXT epoch will use. base_lr × factor is the rate this epoch actually stepped
    with, and it is correct in phase mode too (factor = phase3_lr_scale there).
    """
    lrs = {pg["name"]: pg["base_lr"] * factor for pg in optimiser.param_groups}
    if len(set(f"{v:.6e}" for v in lrs.values())) == 1:
        return f"lr={next(iter(lrs.values())):.2e}"
    return " ".join(f"lr[{name}]={value:.2e}" for name, value in lrs.items())


def build_kot_optimiser(model, lr: float, n_epochs: int, use_phases: bool, cfg: dict):
    optimiser = torch.optim.Adam(kot_param_groups(
        model, lr, cfg.get("lr_phi"), cfg.get("lr_alpha_kappa"), cfg.get("lr_beta"),
    ))
    if use_phases:
        # Phase mode runs its own lr control (phase3_lr_scale); stacking a decay on
        # top would make the phase-3 rate depend on where the cosine happened to be.
        return optimiser, None
    warmup_epochs = int(cfg.get("lr_warmup_epochs", 0))
    start_factor  = float(cfg.get("lr_warmup_start_factor", 1.0))
    min_factor    = float(cfg.get("lr_min_factor", 0.0))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimiser,
        lr_lambda=lambda e: lr_schedule_factor(
            e, warmup_epochs, n_epochs, start_factor, min_factor,
        ),
    )
    return optimiser, scheduler


def array_debug_stats(values: np.ndarray) -> dict:
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def optional_obsm_debug_stats(adata, key: str) -> dict | None:
    if key not in adata.obsm:
        return None
    return array_debug_stats(np.asarray(adata.obsm[key], dtype=np.float32))


def kot_data_debug_summary(
    rna_adata,
    rna_layer: str | None,
    protein_layer: str | None,
    velocity_layer: str,
    x: np.ndarray,
    y: np.ndarray,
    velocity: np.ndarray,
    s_model: np.ndarray,
) -> dict:
    links = rna_adata.uns.get("rna_protein_links", {})
    link_count = len(links.get("rna", [])) if isinstance(links, dict) else 0
    synthetic_meta = dict(rna_adata.uns.get("synthetic_linked_ode", {}))
    return {
        "rna_layer": rna_layer or "X",
        "protein_layer": protein_layer or "X",
        "velocity_layer": velocity_layer,
        "rna_shape": [int(v) for v in x.shape],
        "protein_shape": [int(v) for v in y.shape],
        "velocity_shape": [int(v) for v in velocity.shape],
        "s_matrix_shape": [int(v) for v in s_model.shape],
        "s_matrix_nonzero": int(np.count_nonzero(s_model)),
        "link_count": int(link_count),
        "synthetic_linked_ode": synthetic_meta,
        "true_alpha": optional_obsm_debug_stats(rna_adata, "true_alpha"),
        "true_beta": optional_obsm_debug_stats(rna_adata, "true_beta"),
        "true_kappa": optional_obsm_debug_stats(rna_adata, "true_kappa"),
        "true_protein_velocity": optional_obsm_debug_stats(
            rna_adata,
            "true_protein_velocity_mean",
        ),
    }


def print_kot_data_debug(summary: dict) -> None:
    print("[kot] data debug:")
    print(
        "[kot]   layers:   "
        f"rna={summary['rna_layer']}  protein={summary['protein_layer']}  "
        f"velocity={summary['velocity_layer']}"
    )
    print(
        "[kot]   shapes:   "
        f"rna={summary['rna_shape']}  protein={summary['protein_shape']}  "
        f"velocity={summary['velocity_shape']}"
    )
    print(
        "[kot]   S matrix: "
        f"shape={summary['s_matrix_shape']}  nonzero={summary['s_matrix_nonzero']}  "
        f"links={summary['link_count']}"
    )
    synthetic_meta = summary["synthetic_linked_ode"]
    if synthetic_meta:
        print(
            "[kot]   synthetic: "
            f"stage={synthetic_meta.get('stage')}  "
            f"distractors={synthetic_meta.get('n_distractor_rna_genes')}  "
            f"native_rna={synthetic_meta.get('native_rna_layer')}  "
            f"native_protein={synthetic_meta.get('native_protein_layer')}"
        )
    for key in ["true_alpha", "true_beta", "true_kappa", "true_protein_velocity"]:
        stats = summary[key]
        if stats is not None:
            print(
                f"[kot]   {key:21s}"
                f"min={stats['min']:.6g}  max={stats['max']:.6g}  "
                f"mean={stats['mean']:.6g}  std={stats['std']:.6g}"
            )


class FixedKappa(torch.nn.Module):
    """Constant positive kappa used to test the kinetic objective without kappa collapse."""

    def __init__(self, value: float):
        super().__init__()
        self.register_buffer("value", torch.tensor(float(value), dtype=torch.float32))

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        value = self.value.to(device=r.device, dtype=r.dtype).reshape(1, 1)
        return value.expand(r.shape[0], 1)


class FixedAlpha(torch.nn.Module):
    """Constant positive alpha used to test KOT without g/alpha collapse."""

    def __init__(self, value: float | list[float], d_protein: int):
        super().__init__()
        if isinstance(value, list):
            alpha = torch.tensor([float(v) for v in value], dtype=torch.float32)
            if alpha.numel() != d_protein:
                raise ValueError(
                    "kot_fixed_alpha list length must match protein dimension "
                    f"({d_protein})."
                )
        else:
            alpha = torch.full((d_protein,), float(value), dtype=torch.float32)
        self.register_buffer("value", alpha.reshape(1, d_protein))

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        value = self.value.to(device=r.device, dtype=r.dtype)
        return value.expand(r.shape[0], value.shape[1])


def fixed_vector_tensor(value: float | list[float], d_protein: int, device: torch.device) -> torch.Tensor:
    if isinstance(value, list):
        vector = torch.tensor([float(v) for v in value], dtype=torch.float32, device=device)
        if vector.numel() != d_protein:
            raise ValueError(
                "Fixed beta list length must match protein dimension "
                f"({d_protein})."
            )
        return vector
    return torch.full((d_protein,), float(value), dtype=torch.float32, device=device)


def inverse_softplus(values: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.expm1(values))


def set_fixed_beta(model, value: float | list[float], d_protein: int, device: torch.device) -> torch.Tensor:
    target = fixed_vector_tensor(value, d_protein, device)
    if bool(torch.any(target <= 1e-6)):
        raise ValueError("kot_fixed_beta values must be greater than 1e-6.")
    raw_target = inverse_softplus(target - 1e-6)
    with torch.no_grad():
        model.beta_raw.copy_(raw_target)
    model.beta_raw.requires_grad = False
    return target


def velocity_row_permutation(n_cells: int, seed: int) -> np.ndarray:
    """The row permutation behind the velocity control, seeded by the run seed.

    Returns the permutation rather than a permuted matrix because everything that is a
    property of a cell's own velocity has to move with it -- the velocity vector and
    the confidence that weights its residual -- and one shared permutation is the only
    way those stay together. Top-level so the control can be tested without a training
    run, and seeded off the model seed so each seed draws its own permutation and the
    control carries the same seed variance as the arm it is compared against.
    """
    return np.random.default_rng(seed).permutation(n_cells)


VELOCITY_ABLATIONS = ("none", "shuffle", "reverse", "zero")


def resolve_velocity_ablation(cfg: dict) -> str:
    """Which velocity control this run uses, with the legacy boolean folded in.

    kot_velocity_shuffle predates the enum and is written into every run dir and jobs
    file of the permutation control, so it stays readable as an alias rather than
    silently doing nothing on an old config. Asking for two different controls at once
    is a contradiction the config cannot resolve, so it is an error, not a precedence
    rule.
    """
    ablation = str(cfg.get("kot_velocity_ablation", "none")).lower()
    if ablation not in VELOCITY_ABLATIONS:
        raise ValueError(
            f"kot_velocity_ablation must be one of {list(VELOCITY_ABLATIONS)}, "
            f"got {ablation!r}"
        )
    if bool(cfg.get("kot_velocity_shuffle", False)):
        if ablation not in ("none", "shuffle"):
            raise ValueError(
                f"kot_velocity_shuffle=true contradicts "
                f"kot_velocity_ablation={ablation!r}; set only one of them."
            )
        return "shuffle"
    return ablation


def apply_velocity_ablation(velocity: np.ndarray, confidence: np.ndarray | None,
                            cfg: dict) -> tuple[np.ndarray, np.ndarray | None]:
    """Break one piece of L_dyn; everything else in the run stays identical.

      shuffle  another cell's real v (confidence rides the shuffle so weight
               and vector stay coupled)
      reverse  v → −v (arrow of time only)
      zero     v → 0, residual is the static half of the ODE

    Print is the audit trail: JVP-RHS cosine cannot be, the RHS contains no v.
    """
    ablation = resolve_velocity_ablation(cfg)
    corrupt = float(cfg.get("kot_velocity_corrupt", 0.0))
    if not 0.0 <= corrupt <= 1.0:
        raise ValueError(f"kot_velocity_corrupt must be in [0, 1], got {corrupt}")
    if corrupt > 0.0 and ablation != "none":
        raise ValueError(
            f"kot_velocity_corrupt={corrupt} contradicts kot_velocity_ablation="
            f"{ablation!r}: the mixture already spans real (x=0) to shuffled (x=1)."
        )
    n = velocity.shape[0]
    if corrupt > 0.0:
        print(f"[kot] CORRUPTED velocity (control): v <- {1 - corrupt:.2f}*v + "
              f"{corrupt:.2f}*v[perm] over {n} cells (x={corrupt})")
        return corrupt_velocity(velocity, confidence, corrupt, int(cfg.get("seed", 42)))
    if ablation == "none":
        return velocity, confidence
    if ablation == "shuffle":
        perm = velocity_row_permutation(n, int(cfg.get("seed", 42)))
        print(f"[kot] SHUFFLED velocity (control): rows permuted across {n} cells, "
              f"{int((perm == np.arange(n)).sum())} left in place")
        return velocity[perm], (None if confidence is None else confidence[perm])
    if ablation == "reverse":
        print(f"[kot] REVERSED velocity (control): v -> -v over {n} cells, "
              f"per-cell norms and the gauge unchanged")
        return -velocity, confidence
    print(f"[kot] ZEROED velocity (control): J_phi.v = 0 over {n} cells, "
          f"kinetics residual keeps only the state term")
    return np.zeros_like(velocity), confidence


def median_norm_ratio(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Scalar putting `candidate`'s median per-cell norm back onto `reference`'s.

    Medians rather than means so a few very fast cells cannot set the scale for all of
    them, and nonzero rows only, for the same reason build_effective_velocity uses them:
    a cell with no velocity carries no length to match and would drag the median down.
    Returns 1.0 when either side has no nonzero row, which leaves the field untouched.
    """
    ref = np.linalg.norm(reference, axis=1)
    cand = np.linalg.norm(candidate, axis=1)
    ref, cand = ref[ref > DIAG_EPS], cand[cand > DIAG_EPS]
    if ref.size == 0 or cand.size == 0:
        return 1.0
    return float(np.median(ref) / np.median(cand))


def corrupt_velocity(velocity: np.ndarray, confidence: np.ndarray | None, fraction: float,
                     seed: int) -> tuple[np.ndarray, np.ndarray | None]:
    """x=0/1 are the real and shuffle arms (same perm, same seed). Rescale the
    mixture's median norm: without it, x=0.5 is shorter by ~√2 as well as
    wrongly directed, so a FOSCTTM dip is unreadable. Not the global velocity
    gauge — that also rescales the real arm and on synthetic collapsed branch
    accuracy (jobs/jobs_synth_gauge_pilot.txt).
    """
    perm = velocity_row_permutation(velocity.shape[0], seed)
    mixed = (1.0 - fraction) * velocity + fraction * velocity[perm]
    mixed = mixed * median_norm_ratio(velocity, mixed)
    if confidence is None:
        return mixed.astype(velocity.dtype), None
    mixed_conf = (1.0 - fraction) * confidence + fraction * confidence[perm]
    return mixed.astype(velocity.dtype), mixed_conf.astype(confidence.dtype)


def branch_accuracy_scores(state_true: np.ndarray, state_pred: np.ndarray) -> dict:
    """Fraction of cells whose nearest protein match carries the same branch label.

    The confusion matrix beside this holds the same information, but as a nested list it
    is invisible to tools/collect_run_diagnostics.py, which only columns flat scalars --
    so the number a sweep is actually read on has to be its own key.

    Two numbers, because they answer different questions. `branch_accuracy` is over every
    labelled cell and matches the confusion matrix. `branch_accuracy_branched` keeps only
    cells on a real branch (label starting "Branch_"), which is the metric the v2b-branch
    stage is built for: the A<->B swap is the degeneracy that static OT genuinely cannot
    resolve, and progenitor cells sit before the bifurcation, so including them dilutes
    the measurement toward 1.0 with cells that were never ambiguous. On a trunk-only
    stage every label is "Trunk", so the first is trivially 1.0 and the second is NaN --
    which is the honest report, not a failure.
    """
    if len(state_true) == 0:
        return {"branch_accuracy": float("nan"), "branch_accuracy_branched": float("nan"),
                "branch_n_branched": 0}
    branched = np.char.startswith(state_true.astype(str), "Branch_")
    matched = state_true == state_pred
    return {
        "branch_accuracy": float(np.mean(matched)),
        "branch_accuracy_branched": (
            float(np.mean(matched[branched])) if branched.any() else float("nan")
        ),
        "branch_n_branched": int(branched.sum()),
    }


def permute_s_rows(S: np.ndarray, seed: int) -> np.ndarray:
    """Give each mapped protein another mapped protein's gene row.

    The control for the RNA->protein map itself: S keeps its shape, its values and its
    per-row sparsity, so a gap against the real arm measures which gene feeds which
    protein and not how much structure S carries.

    Only rows that carry a link take part. A protein has a non-zero S row exactly when
    it is kinetically active (`assert_no_orphan_s_rows`), so permuting inside that set
    leaves the kinetic mask untouched, leaves the count that survives the velocity-gene
    filter unchanged, and leaves the gene set behind the RNA-side velocity weight
    (a column property of S) identical -- the mapping is the only thing that moves.
    """
    mapped = np.nonzero((S != 0).any(axis=1))[0]
    perm = np.random.default_rng(seed).permutation(len(mapped))
    permuted = S.copy()
    permuted[mapped] = S[mapped[perm]]
    return permuted


def apply_s_permutation(S: np.ndarray, cfg: dict) -> np.ndarray:
    """S under the mapping control, or S itself. Seeded off the run seed like the
    velocity shuffle, so each seed draws its own broken mapping and the control carries
    the same seed variance as the arm it is compared against."""
    if not bool(cfg.get("kot_s_permute", False)):
        return S
    permuted = permute_s_rows(S, int(cfg.get("seed", 42)))
    mapped = np.nonzero((S != 0).any(axis=1))[0]
    kept = int((permuted[mapped] == S[mapped]).all(axis=1).sum())
    print(f"[kot] PERMUTED S rows (control): RNA->protein map reassigned across "
          f"{len(mapped)} mapped proteins, {kept} left in place")
    return permuted


def resolve_velocity_confidence(rna_adata, n_cells: int) -> np.ndarray:
    """Per-cell scVelo velocity_confidence, or all-ones when the column is absent."""
    if "velocity_confidence" in rna_adata.obs.columns:
        return np.array(rna_adata.obs["velocity_confidence"].values, dtype=np.float32)
    return np.ones(n_cells, dtype=np.float32)


def resolve_velocity_matrix(rna_adata, velocity_layer: str | None, d_rna: int,
                            use_feature_space: bool) -> tuple[np.ndarray, str]:
    """The RNA velocity layer, projected into the KOT input's coordinate system.

    Takes the configured layer when one is named, else the first of
    `true_velocity` / `velocity` that exists. In representation mode a
    gene-space velocity is carried into PCA coordinates by the same PCs the
    RNA input uses; in feature space it must already match the gene axis.
    """
    candidates = [velocity_layer] if velocity_layer else ["true_velocity", "velocity"]
    for key in candidates:
        if key in rna_adata.layers:
            raw = np.nan_to_num(to_dense(rna_adata.layers[key], np.float32), nan=0.0)
            print(f"[kot] Using velocity layer: '{key}'")
            break
    else:
        named = f"layer '{velocity_layer}' not found" if velocity_layer else "requires a velocity layer"
        raise ValueError(f"KOT {named}. Available layers: {list(rna_adata.layers.keys())}")

    if raw.shape[1] == d_rna:
        return raw.astype(np.float32), key
    if not use_feature_space and "PCs" in rna_adata.varm:
        return raw @ np.array(rna_adata.varm["PCs"]), key
    raise ValueError(
        f"KOT velocity shape {raw.shape} is incompatible with a {d_rna}-column KOT input."
    )


def build_velocity_weight(rna_adata, S_np: np.ndarray, cfg: dict, use_feature_space: bool) -> np.ndarray | None:
    """RNA genes whose velocity can enter the JVP, or None for unweighted velocity."""
    if not use_feature_space:
        return None

    weights = []
    if bool(cfg.get("kot_velocity_rna_reliability", False)) and "velocity_genes" in rna_adata.var:
        weights.append(np.asarray(rna_adata.var["velocity_genes"].values, dtype=np.float32))
    if bool(cfg.get("kot_velocity_s_linked_only", False)):
        weights.append((S_np != 0).any(axis=0).astype(np.float32))
    if not weights:
        return None

    velocity_weight = weights[0].copy()
    for weight in weights[1:]:
        velocity_weight *= weight
    return velocity_weight


# --- Distribution summaries for the physics diagnostics ------------------------------
# A global mean hides the shape of the distribution it came from: a mean JVP/RHS cosine of
# 0.2 means something very different if every cell sits at 0.2 than if half the cells are
# at +0.9 and half at -0.5. Every per-cell / per-protein quantity below is therefore
# reported as mean/std/min/max plus a quantile ladder, so box plots can be drawn straight
# from the JSON without reloading the per-cell arrays.
DIAG_QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
STAT_SUFFIXES = ("mean", "std", "min", "max", "median") + tuple(
    f"q{int(round(q * 100)):02d}" for q in DIAG_QUANTILES
)
# Floor for every ratio / log denominator. Cells whose kept velocity genes are all zero
# give a zero JVP and can give a zero RHS, and the ratio must stay finite instead of
# blowing the whole diagnostic up.
DIAG_EPS = 1e-8

# Per-cell physics quantities produced by `jvp_rhs_per_cell`, in report order.
JVP_DIAG_PREFIXES = (
    "jvp_norm", "rhs_norm", "residual_norm", "jvp_rhs_cos",
    "norm_ratio", "log_norm_ratio", "rel_residual",
)
# Learned-parameter distributions reported alongside them.
PARAM_DIAG_PREFIXES = ("kappa", "alpha")


def summarize_distribution(values, prefix: str) -> dict:
    """mean/std/min/max/median + DIAG_QUANTILES of a 1-D array, keyed `{prefix}_{stat}`.

    Non-finite entries are dropped first (a degenerate cell can make a ratio inf or NaN);
    when nothing finite is left every stat is NaN, so a broken run logs NaN rather than
    aborting the diagnostic. `std` uses ddof=0, matching the `unbiased=False` the older
    scalar stats used, so the values stay comparable with runs from before this change.
    """
    arr = np.asarray(values, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {f"{prefix}_{s}": float("nan") for s in STAT_SUFFIXES}
    stats = {
        "mean":   float(arr.mean()),
        "std":    float(arr.std()),
        "min":    float(arr.min()),
        "max":    float(arr.max()),
        "median": float(np.median(arr)),
    }
    for q, value in zip(DIAG_QUANTILES, np.quantile(arr, DIAG_QUANTILES)):
        stats[f"q{int(round(q * 100)):02d}"] = float(value)
    return {f"{prefix}_{s}": stats[s] for s in STAT_SUFFIXES}


def head_bounds(head) -> tuple[float | None, float | None]:
    """(min, max) of a BoundedPositiveMLP head, or (None, None) for one with no bounds.

    Two heads have none: an unbounded softplus head, and a Fixed* head, which is a
    registered buffer rather than something trained. Saturation asks whether a LEARNED
    head is being pinned by its limits instead of by the data, so a fixed head has
    nothing to report -- and reporting nothing is what keeps kot_oracle / kot_fixedkappa
    / kot_fixedalpha from dying on an attribute their head never had.
    """
    return getattr(head, "min_value", None), getattr(head, "max_value", None)


def bounded_head_saturation(values: np.ndarray, min_value: float | None,
                            max_value: float | None,
                            prefix: str, edge_frac: float = 0.05) -> dict:
    """How much of a BoundedPositiveMLP head sits at its limits instead of being learned.

    The head is min + span·sigmoid(raw), so an output whose pre-activation never moved off
    zero sits exactly at the midpoint, and the sigmoid gradient vanishes at both ends. Mass
    at the midpoint means those entries are still at init; mass at an edge means the bound,
    not the data, is setting them. Reported because a single scalar bound says
    nothing about the bulk. Empty for an unbounded (softplus) head.
    """
    if max_value is None or min_value is None:
        return {}
    span = max_value - min_value
    flat = np.asarray(values).reshape(-1)
    return {
        f"{prefix}_frac_at_midpoint":  float(np.isclose(flat, min_value + 0.5 * span,
                                                        rtol=0, atol=1e-3 * span).mean()),
        f"{prefix}_frac_near_ceiling": float((flat > max_value - edge_frac * span).mean()),
        f"{prefix}_frac_near_floor":   float((flat < min_value + edge_frac * span).mean()),
    }


def constant_output_fraction(values: np.ndarray, tol: float = 1e-3) -> float:
    """Fraction of per-protein outputs that do not vary across cells — a dead head that
    returns the same number for every cell, so its protein carries no cell-state signal."""
    return float(((values.max(axis=0) - values.min(axis=0)) < tol).mean())


def jvp_rhs_per_cell(dphi_dv: torch.Tensor, rhs: torch.Tensor) -> dict:
    """Per-cell views of how well J_φ(r)·v matches the ODE right-hand side.

    The cosine alone only says the two sides point the same way — it is blind to a φ whose
    direction is right but whose speed is wrong. These add the magnitude view:

      jvp_norm       ‖J_φ(r_i)·v_i‖                       LHS speed
      rhs_norm       ‖κ(α⊙Sr_i − β⊙φ_i)‖                  RHS speed
      residual_norm  ‖LHS − RHS‖                          unnormalised miss
      norm_ratio     ‖LHS‖ / (‖RHS‖ + ε)                  1 = matched speeds
      log_norm_ratio log((‖LHS‖+ε) / (‖RHS‖+ε))           symmetric around 0
      rel_residual   ‖LHS−RHS‖ / (‖LHS‖+‖RHS‖+ε)          bounded [0, 1], 0 = perfect fit

    Both sides must already be masked by the caller when only mapped proteins count.
    """
    jvp_n = dphi_dv.norm(dim=1)
    rhs_n = rhs.norm(dim=1)
    res_n = (dphi_dv - rhs).norm(dim=1)
    per_cell = {
        "jvp_norm":       jvp_n,
        "rhs_norm":       rhs_n,
        "residual_norm":  res_n,
        "jvp_rhs_cos":    F.cosine_similarity(dphi_dv, rhs, dim=1, eps=1e-8),
        "norm_ratio":     jvp_n / (rhs_n + DIAG_EPS),
        "log_norm_ratio": torch.log((jvp_n + DIAG_EPS) / (rhs_n + DIAG_EPS)),
        "rel_residual":   res_n / (jvp_n + rhs_n + DIAG_EPS),
    }
    return {name: t.detach().cpu().numpy() for name, t in per_cell.items()}


def jvp_and_rhs(model, r: torch.Tensor, v: torch.Tensor, S_t: torch.Tensor,
                mask: torch.Tensor | None):
    """One forward-mode pass giving both sides of the ODE, masked to the kinetic panel.

    Returns (phi_r, dphi_dv, rhs, kappa, alpha) — φ(r), the LHS J_φ(r)·v, the RHS
    κ(α⊙Sr − β⊙φ), and the two learned parameter fields the caller summarises.
    Every physics diagnostic reads the two sides from here rather than rebuilding
    them, so a change to the ODE cannot leave one diagnostic on the old form.
    JVP needs autograd: do not call inside torch.no_grad().
    """
    phi_r, dphi_dv = torch_jvp(model.phi, (r,), (v,))
    kappa = model.kappa(r)
    alpha = model.g(r)
    rhs = kappa * (alpha * (S_t @ r.T).T - model.beta * phi_r)
    if mask is not None:
        dphi_dv = dphi_dv * mask
        rhs = rhs * mask
    return phi_r, dphi_dv, rhs, kappa, alpha


def jvp_diag_keys() -> list[str]:
    """Every key `compute_jvp_rhs_diagnostics` returns.

    The per-epoch history columns and the NaN row used on epochs where the diagnostic does
    not run are both built from this, so neither can drift out of sync with the diagnostic.
    """
    keys = [
        f"{prefix}_{stat}"
        for prefix in JVP_DIAG_PREFIXES + PARAM_DIAG_PREFIXES
        for stat in STAT_SUFFIXES
    ]
    # Legacy scalar aliases (== the matching *_mean stat) kept for the existing plots,
    # checkpoint CSVs and tools/collect_run_diagnostics.py.
    keys += ["jvp_rhs_cos", "jvp_norm", "rhs_norm"]
    return keys


def velocity_norm_diagnostics(v_eff) -> dict:
    """‖v_eff‖ after RNA-side zeros. A all-zero cell can only push the residual toward 0."""
    if isinstance(v_eff, torch.Tensor):
        norms = v_eff.norm(dim=1).detach().cpu().numpy()
    else:
        norms = np.linalg.norm(np.asarray(v_eff, dtype=np.float64), axis=1)
    out = summarize_distribution(norms, "v_eff_norm")
    out["v_eff_nonzero_frac"] = float(np.mean(norms > DIAG_EPS)) if norms.size else float("nan")
    out["v_eff_n_cells"] = int(norms.size)
    return out


def velocity_backend_skip(out: dict, reason: str) -> dict:
    """Record a velocity-backend comparison that could not be made, and say why."""
    print(f"[kot] velocity-backend comparison skipped: {reason}")
    out["velocity_backend_status"] = reason
    return out


def align_velocity_fields(rna_adata, velocity: np.ndarray, other, layer: str):
    """Both velocity fields on the (cell, gene) grid the two files share.

    Matched BY NAME, never by position: the two AnnData objects were filtered and ordered
    independently, so comparing row i to row i would silently pair different cells.
    Returns (v_self, v_other, cells, genes); empty indexes mean nothing is shared and the
    caller reports that rather than dividing by zero.

    The comparison field is NaN-scrubbed here because backends leave NaN for genes they
    could not fit, and one NaN poisons an entire row's dot product.
    """
    self_cells = pd.Index(rna_adata.obs_names)
    self_genes = pd.Index(rna_adata.var_names)
    cells = self_cells.intersection(pd.Index(other.obs_names))
    genes = self_genes.intersection(pd.Index(other.var_names))
    if len(cells) == 0 or len(genes) == 0:
        return np.zeros((0, 0)), np.zeros((0, 0)), cells, genes
    v_self = velocity[self_cells.get_indexer(cells)][:, self_genes.get_indexer(genes)]
    v_other = np.nan_to_num(to_dense(other.layers[layer], np.float32), nan=0.0)
    v_other = v_other[pd.Index(other.obs_names).get_indexer(cells)][
        :, pd.Index(other.var_names).get_indexer(genes)
    ]
    return v_self, v_other, cells, genes


def log_norm_ratio_stats(a: np.ndarray, b: np.ndarray, axis: int, prefix: str) -> dict:
    """How much LONGER one field's vectors are than the other's, as a ratio.

    Cosine answers only half of "how different are these two velocity fields": it is
    scale-invariant, so two fields that agree perfectly in direction but differ 10x in
    magnitude score 1.0. The difference decomposes exactly into direction and magnitude --
    for unit-scaled vectors ||a - b||^2 = 2(1 - cos), so once the cosine is known the only
    remaining information is this ratio.

    Summarised in the LOG domain and exponentiated back, so that 2x and 0.5x are equal and
    opposite rather than one being pulled toward the mean by the other. Rows where either
    side is empty carry no ratio and are dropped rather than counted as 1.0.
    """
    norm_a = np.linalg.norm(a, axis=axis)
    norm_b = np.linalg.norm(b, axis=axis)
    usable = (norm_a > DIAG_EPS) & (norm_b > DIAG_EPS)
    if not usable.any():
        return {f"{prefix}_{k}": float("nan") for k in ("median", "q25", "q75")} | {
            f"{prefix}_n": 0}
    log_ratio = np.log(norm_a[usable]) - np.log(norm_b[usable])
    q25, median, q75 = np.quantile(log_ratio, (0.25, 0.50, 0.75))
    return {f"{prefix}_median": float(np.exp(median)), f"{prefix}_q25": float(np.exp(q25)),
            f"{prefix}_q75": float(np.exp(q75)), f"{prefix}_n": int(usable.sum())}


def per_gene_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine per GENE (column) across cells -- the diagonal of the cross-backend
    gene-by-gene product, i.e. how much the two providers agree about each gene's velocity
    over the whole population. The per-cell cosine asks the transposed question."""
    denom = np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0)
    return (a * b).sum(axis=0) / (denom + DIAG_EPS)


def compute_velocity_backend_agreement(rna_adata, velocity: np.ndarray, cfg: dict,
                                       velocity_weight: np.ndarray | None = None) -> dict:
    """How different the two velocity files were, so a FOSCTTM gap between
    scVelo and RegVelo ablations is not blamed on KOT. `*_kot_*` is after the
    RNA-side weight (genes that never enter J_φ·v). Missing compare path → NaN.
    """
    compare_path = cfg.get("kot_velocity_compare_h5ad")
    layer = str(cfg.get("kot_velocity_compare_layer", "velocity"))
    out = summarize_distribution([], "velocity_backend_cos")
    out.update(summarize_distribution([], "velocity_backend_cos_kot"))
    out.update({
        "velocity_backend_kot_n_genes": 0,
        "velocity_backend_compare_h5ad": str(compare_path) if compare_path else None,
        "velocity_backend_compare_layer": layer,
        "velocity_backend_label": cfg.get("kot_velocity_compare_label"),
        "velocity_backend_n_cells": 0,
        "velocity_backend_n_genes": 0,
        "velocity_backend_status": "not configured",
    })
    if not compare_path:
        return out

    if velocity.shape[1] != rna_adata.n_vars:
        return velocity_backend_skip(
            out,
            f"velocity has {velocity.shape[1]} columns but the AnnData has "
            f"{rna_adata.n_vars} genes (representation mode: no gene axis to match on)"
        )
    if not Path(compare_path).exists():
        return velocity_backend_skip(out, f"'{compare_path}' not found")

    other = ad.read_h5ad(compare_path)

    if layer not in other.layers:
        return velocity_backend_skip(
            out, f"layer '{layer}' missing from '{compare_path}' "
                 f"(has: {list(other.layers.keys())})")

    self_genes = pd.Index(rna_adata.var_names)
    v_self, v_other, cells, genes = align_velocity_fields(
        rna_adata, velocity, other, layer)
    del other                          # free the comparison AnnData before the einsum
    if len(cells) == 0 or len(genes) == 0:
        return velocity_backend_skip(
            out, f"no shared cells/genes with '{compare_path}' "
                 f"({len(cells)} cells, {len(genes)} genes)")

    cos = per_cell_cosine(v_self, v_other)
    out.update(summarize_distribution(cos, "velocity_backend_cos"))
    # The same comparison restricted to the genes KOT actually pushes through the
    # Jacobian. The weight is indexed over the run's full gene axis, so it is subset to
    # the shared genes the same way the two velocity matrices were.
    if velocity_weight is None:
        weight_shared = np.ones(len(genes), dtype=np.float32)
    else:
        weight_shared = velocity_weight[self_genes.get_indexer(genes)]
    cos_kot = per_cell_cosine(v_self * weight_shared[None, :],
                              v_other * weight_shared[None, :])
    out.update(summarize_distribution(cos_kot, "velocity_backend_cos_kot"))
    out["velocity_backend_kot_n_genes"] = int((weight_shared != 0).sum())
    # The magnitude half of the difference, on the same vectors the cosine used.
    out.update(log_norm_ratio_stats(v_self, v_other, 1, "velocity_backend_norm_ratio"))
    out.update(log_norm_ratio_stats(v_self * weight_shared[None, :],
                                    v_other * weight_shared[None, :], 1,
                                    "velocity_backend_norm_ratio_kot"))
    out["velocity_backend_n_cells"] = int(len(cells))
    out["velocity_backend_n_genes"] = int(len(genes))
    # Cells where one backend produced no velocity at all: their cosine is a forced 0 and
    # says nothing about agreement, so report how many rows are in that state. Reported
    # for BOTH gene sets -- a cell can carry velocity somewhere in the transcriptome and
    # none at all on the handful of genes the mask keeps, and those forced zeros drag the
    # masked median toward 0 while the unmasked fraction still reads 0%.
    out["velocity_backend_zero_frac"] = zero_norm_fraction(v_self, v_other)
    out["velocity_backend_kot_zero_frac"] = zero_norm_fraction(
        v_self * weight_shared[None, :], v_other * weight_shared[None, :])
    out["velocity_backend_status"] = "ok"
    print(f"[kot] velocity-backend cos vs {compare_path}: "
          f"median={out['velocity_backend_cos_median']:+.3f} "
          f"mean={out['velocity_backend_cos_mean']:+.3f} "
          f"[q25={out['velocity_backend_cos_q25']:+.3f}, q75={out['velocity_backend_cos_q75']:+.3f}] "
          f"over {len(cells)} cells x {len(genes)} genes")
    print(f"[kot] velocity-backend cos (KOT-masked): "
          f"median={out['velocity_backend_cos_kot_median']:+.3f} "
          f"mean={out['velocity_backend_cos_kot_mean']:+.3f} "
          f"[q25={out['velocity_backend_cos_kot_q25']:+.3f}, "
          f"q75={out['velocity_backend_cos_kot_q75']:+.3f}] "
          f"over {out['velocity_backend_kot_n_genes']}/{len(genes)} genes in the JVP")
    return out


def zero_norm_fraction(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of rows where either side has no direction at all, so the cosine is a
    forced 0 that means "no evidence" rather than "orthogonal"."""
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return float(np.mean(denom <= DIAG_EPS))


def per_cell_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine, with a floored denominator so an all-zero row gives 0, not NaN.

    A cell whose kept genes are all zero in either field has no direction to compare; a
    forced 0 says "no evidence" and is counted by velocity_backend_zero_frac, whereas a
    NaN would silently drop the cell out of every quantile below.
    """
    dot = np.einsum("ij,ij->i", a, b).astype(np.float64)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return dot / (denom + DIAG_EPS)


def build_effective_velocity(
    velocity: np.ndarray, velocity_weight: np.ndarray | None, cfg: dict, quiet: bool = False,
) -> np.ndarray:
    """The exact velocity fed to the JVP: RNA-side weight applied, then gauge-normalized.

    Order matters. The weight zeroes the genes whose velocity may not enter J_φ·v, and the
    gauge scalar must be measured from the vector that actually reaches the Jacobian — not
    from the raw field — so it is computed after the weight. The gauge uses the median of
    the NONZERO per-cell norms, which is robust both to outliers and to the zeros the weight
    itself introduces.

    Training and every diagnostic must call this rather than re-deriving V_eff from a raw
    velocity plus a separate weight: a checkpoint re-evaluated on a differently-scaled
    velocity reports physics numbers that do not match the run that produced it.
    """
    gauge_eps = 1e-8
    v_eff = velocity if velocity_weight is None else velocity * velocity_weight[None, :]
    if bool(cfg.get("kot_velocity_gauge_normalize", False)):
        cell_norms = np.linalg.norm(v_eff, axis=1)
        nonzero = cell_norms > gauge_eps
        gauge = float(np.median(cell_norms[nonzero])) if nonzero.any() else 1.0
        v_eff = v_eff / (gauge + gauge_eps)
        if not quiet:
            print(f"[kot] velocity gauge-normalized: median nonzero per-cell norm {gauge:.4g} -> 1.0")
    return v_eff.astype(np.float32)


class DiagInputs(NamedTuple):
    """The fixed inputs every end-of-training diagnostic reads.

    These never change within a run — only the model weights do — so they travel
    as one bundle rather than as a long argument list repeated at each call site.
    `V_eff_t` is the exact velocity training pushed through the Jacobian (see
    `build_effective_velocity`), so a checkpoint is always scored on the same
    vector that produced it. `val_idx` is the fixed validation slice
    (src/data/splits.py); None means no validation split was configured.
    """
    R_t: torch.Tensor
    V_eff: torch.Tensor
    P_t: torch.Tensor
    S_t: torch.Tensor
    rna_obs: pd.DataFrame
    mask: torch.Tensor | None = None
    align_cols: torch.Tensor | None = None
    val_idx: torch.Tensor | None = None


def compute_jvp_rhs_diagnostics(
    model,
    R_t: torch.Tensor,
    V_eff: torch.Tensor,
    S_t: torch.Tensor,
    subset: int = 512,
    mask: torch.Tensor | None = None,
) -> dict:
    """JVP/RHS agreement and κ/α distributions on a small subset.

    `V_eff` is the exact velocity fed to the JVP in training (RNA-side weight applied and
    gauge-normalized upstream); this receives it directly so the diagnostic uses the same
    vector as the loss. JVP requires autograd (forward-mode AD): do not call inside
    torch.no_grad(). When mask is given, the cosine/norms are over the mapped (kinetics)
    proteins only, so unmapped proteins do not dilute the physics fit.

    Every quantity is reported as a full distribution across cells (mean/std/min/max/median
    plus DIAG_QUANTILES) — see `jvp_rhs_per_cell` for what each one measures. The bare
    `jvp_rhs_cos` / `jvp_norm` / `rhs_norm` keys are kept as aliases of the corresponding
    `*_mean` so the existing plots and checkpoint CSVs keep working.
    """
    was_training = model.training
    model.eval()

    m = min(subset, R_t.shape[0])
    _, dphi_dv, rhs, kappa, alpha = jvp_and_rhs(model, R_t[:m], V_eff[:m], S_t, mask)

    per_cell = jvp_rhs_per_cell(dphi_dv, rhs)
    out = {}
    for name in JVP_DIAG_PREFIXES:
        out.update(summarize_distribution(per_cell[name], name))
    # κ is one scalar per cell; α is per (cell, protein) and is summarised over every
    # entry, matching what the older alpha_mean/alpha_std reported.
    kappa_np = kappa.detach().cpu().numpy()
    alpha_np = alpha.detach().cpu().numpy()
    out.update(summarize_distribution(kappa_np, "kappa"))
    out.update(summarize_distribution(alpha_np, "alpha"))
    # Where α and κ sit relative to their bounds, and the per-protein α the panel learned.
    out.update(bounded_head_saturation(kappa_np, *head_bounds(model.kappa), "kappa"))
    out.update(bounded_head_saturation(alpha_np, *head_bounds(model.g), "alpha"))
    out["alpha_frac_constant_across_cells"] = constant_output_fraction(alpha_np)
    out["alpha_per_protein_mean"] = alpha_np.mean(axis=0).tolist()
    # Back-compat aliases for the existing plots / CSVs / collect_run_diagnostics.py.
    out["jvp_rhs_cos"] = out["jvp_rhs_cos_mean"]
    out["jvp_norm"]    = out["jvp_norm_mean"]
    out["rhs_norm"]    = out["rhs_norm_mean"]

    if was_training:
        model.train()
    return out


def clone_state(model) -> dict:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def kappa_prior_loss(kappa: torch.Tensor, target: torch.Tensor,
                     lambda_kappa_prior: float) -> torch.Tensor:
    """Log-space pull of κ toward its configured target."""
    return lambda_kappa_prior * torch.log(kappa / target).pow(2).mean()


def phi_output(model, R_t: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model.phi(R_t).cpu().numpy()


def resolve_pseudotime(rna_obs: pd.DataFrame) -> tuple[np.ndarray | None, str | None]:
    """Prefer synthetic `time`; else scVelo. (None, None) if neither."""
    for key in ("time", "velocity_pseudotime", "latent_time"):
        if key in rna_obs.columns:
            return np.asarray(rna_obs[key].values, dtype=np.float64), key
    return None, None


def flat_gradient(grads, params) -> torch.Tensor:
    """One flat vector from a per-parameter gradient list.

    ``torch.autograd.grad(..., allow_unused=True)`` returns None for a parameter the loss
    did not touch; those become zeros so the two losses' gradients stay the same length
    and remain comparable.
    """
    return torch.cat([
        (g if g is not None else torch.zeros_like(p)).reshape(-1)
        for g, p in zip(grads, params)
    ])


def compute_grad_interaction(
    model,
    probe_batches,
    R_t: torch.Tensor,
    V_eff_t: torch.Tensor,
    SR_t: torch.Tensor,
    P_t: torch.Tensor,
    conf_t: torch.Tensor,
    blur: float,
    backend: str,
    mask: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
    align_cols: torch.Tensor | None = None,
    lambda_dyn: float = 1.0,
) -> dict:
    """cos(∇L_align, ∇L_dyn) on φ: <0 is conflict. Norms are unweighted so they
    compare across λ_dyn; mag_ratio uses the epoch's effective λ (0 in warmup).
    eval() so the extra forwards do not move spectral-norm stats.
    """
    was_training = model.training
    model.eval()
    phi_params = [p for p in model.phi.parameters() if p.requires_grad]

    align_norms, dyn_norms, cosines = [], [], []
    dyn_norms_scaled, mag_ratios = [], []
    for idx, pidx in probe_batches:
        phi_r = model.phi(R_t[idx])
        phi_a = phi_r if align_cols is None else phi_r[:, align_cols]
        P_a = (P_t[pidx] if align_cols is None else P_t[pidx][:, align_cols])
        loss_align = sinkhorn_divergence(phi_a, P_a, blur=blur, backend=backend)
        g_align = flat_gradient(
            torch.autograd.grad(loss_align, phi_params, allow_unused=True), phi_params)

        loss_dyn, _ = kinetics_loss(
            model.phi, model.kappa, model.g,
            R_t[idx], V_eff_t[idx], SR_t[idx], model.beta, conf_t[idx],
            mask=mask, scale=scale,
        )
        g_dyn = flat_gradient(
            torch.autograd.grad(loss_dyn, phi_params, allow_unused=True), phi_params)

        na = float(g_align.norm())
        nd = float(g_dyn.norm())
        nd_scaled = float(lambda_dyn) * nd
        align_norms.append(na)
        dyn_norms.append(nd)
        dyn_norms_scaled.append(nd_scaled)
        mag_ratios.append(nd_scaled / (na + DIAG_EPS))
        cosines.append(float((g_align @ g_dyn) / (g_align.norm() * g_dyn.norm() + 1e-12)))

    if was_training:
        model.train()
    return {
        "grad_align_norm":      float(np.mean(align_norms)),
        "grad_dyn_norm":        float(np.mean(dyn_norms)),
        "grad_dyn_norm_scaled": float(np.mean(dyn_norms_scaled)),
        "grad_mag_ratio":       float(np.mean(mag_ratios)),
        "grad_mag_ratio_std":   float(np.std(mag_ratios)),
        "grad_cos_mean":        float(np.mean(cosines)),
        "grad_cos_std":         float(np.std(cosines)),
        "grad_lambda_dyn":      float(lambda_dyn),
        "grad_align_norms":      align_norms,
        "grad_dyn_norms":        dyn_norms,
        "grad_dyn_norms_scaled": dyn_norms_scaled,
        "grad_mag_ratios":       mag_ratios,
        "grad_cosines":          cosines,
    }


# Rows per cdist call are sized to this, not to a fixed count: the block is
# (chunk × m), so a chunk that is cheap on a small panel OOMs a large one.
CDIST_BLOCK_ELEMS = 32 * 1024 * 1024   # 128 MB at fp32


def nearest_protein_match(
    phi: torch.Tensor, protein: torch.Tensor, chunk: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Row-chunked nearest neighbour of each φ(r) in protein space.

    Returns (argmin index, min distance) per query row without ever materialising
    the full (n, m) distance matrix, so a large panel does not OOM.

    `chunk` defaults to whatever keeps one (chunk x m) block near
    `CDIST_BLOCK_ELEMS`, and halves on CUDA OOM down to a single row before
    giving up and finishing the block on CPU -- this runs after the checkpoints
    are written, so it must not be what throws away a finished run.
    """
    n, m = phi.shape[0], protein.shape[0]
    if chunk is None:
        chunk = max(1, min(n, CDIST_BLOCK_ELEMS // max(m, 1)))
    idx = torch.empty(n, dtype=torch.long, device=phi.device)
    vals = torch.empty(n, dtype=phi.dtype, device=phi.device)
    start = 0
    while start < n:
        end = min(start + chunk, n)
        try:
            mins = torch.cdist(phi[start:end], protein).min(dim=1)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if chunk > 1:
                chunk = max(1, chunk // 2)
                continue
            # One row at a time still will not fit: the GPU is full of other jobs,
            # not of this block. Finish on CPU rather than lose the run.
            mins = torch.cdist(phi[start:end].cpu(), protein.cpu()).min(dim=1)
            idx[start:end] = mins.indices.to(idx.device)
            vals[start:end] = mins.values.to(vals.device)
            start = end
            continue
        idx[start:end] = mins.indices
        vals[start:end] = mins.values
        start = end
    return idx, vals


def compute_per_cell_diagnostics(model, inputs: DiagInputs) -> dict:
    """FOSCTTM on the alignment panel only — isotypes must not re-enter the metric.
    Physics arrays are stored so a scatter against FOSCTTM does not rerun the model.
    """
    R_t, V_eff, P_t, S_t, rna_obs, mask, align_cols, val_idx = inputs
    model.eval()

    phi_r, dphi_dv, rhs, kappa, alpha = jvp_and_rhs(model, R_t, V_eff, S_t, mask)

    physics_per_cell = jvp_rhs_per_cell(dphi_dv, rhs)
    dyn_per_cell = (dphi_dv - rhs).pow(2).sum(dim=1).detach().cpu().numpy()

    # α is per protein; mean over masked-out proteins is meaningless.
    kappa_per_cell = kappa.detach().reshape(-1).cpu().numpy()
    if mask is not None:
        mask_row = mask.reshape(1, -1)
        alpha_per_cell = (
            (alpha * mask_row).sum(dim=1) / mask_row.sum().clamp(min=1.0)
        ).detach().cpu().numpy()
    else:
        alpha_per_cell = alpha.mean(dim=1).detach().cpu().numpy()
    v_eff_norm_per_cell = V_eff.norm(dim=1).detach().cpu().numpy()

    # Alignment panel only: excluded features must not re-enter the metric.
    phi_a = phi_r if align_cols is None else phi_r[:, align_cols]
    P_a   = P_t   if align_cols is None else P_t[:, align_cols]
    nn_idx_t, min_dists = nearest_protein_match(phi_a, P_a)
    align_per_cell = min_dists.detach().cpu().numpy()
    nn_idx = nn_idx_t.cpu().numpy()

    pseudotime, _ = resolve_pseudotime(rna_obs)
    if pseudotime is not None:
        time_err = np.abs(pseudotime - pseudotime[nn_idx])
    else:
        time_err = np.full(len(nn_idx), np.nan)

    if "state" in rna_obs.columns:
        state_true = np.asarray(rna_obs["state"].values)
        state_pred = state_true[nn_idx]
        branch_match = (state_true == state_pred).astype(int)
    else:
        state_true = np.array([], dtype="U1")
        state_pred = np.array([], dtype="U1")
        branch_match = np.full(len(nn_idx), -1)

    phi_np = phi_a.detach().cpu().numpy()
    protein_np = P_a.cpu().numpy()
    foscttm = np.asarray(calc_domainAveraged_FOSCTTM(phi_np, protein_np))

    # Validation FOSCTTM, computed WITHIN the held-out slice: each val cell is ranked
    # against the other val cells only. FOSCTTM is a fraction, so on a random subsample
    # it estimates the same quantity as the full-set metric — unbiased, just noisier.
    # `foscttm_val_in_full_pool` keeps the other view, those same cells ranked against
    # all n, which is where a held-out cell's generalisation gap shows up.
    if val_idx is None:
        foscttm_val = np.zeros(0, dtype=float)
        foscttm_val_in_full_pool = np.zeros(0, dtype=float)
        foscttm_train_in_full_pool = np.zeros(0, dtype=float)
    else:
        val_rows = val_idx.cpu().numpy()
        foscttm_val = np.asarray(
            calc_domainAveraged_FOSCTTM(phi_np[val_rows], protein_np[val_rows])
        )
        foscttm_val_in_full_pool = foscttm[val_rows]
        # The mirror image: the cells the run actually FITTED, scored in the same full
        # pool. Training is always the complement of the val mask (see run_kot), so this
        # is exact. Reported directly instead of being reconstructed downstream from
        # mean_foscttm and val_foscttm_in_full_pool, which is arithmetically identical
        # but leaves nothing in the file to check against.
        train_rows_mask = np.ones(foscttm.size, dtype=bool)
        train_rows_mask[val_rows] = False
        foscttm_train_in_full_pool = foscttm[train_rows_mask]

    return {
        "foscttm":     foscttm,
        "foscttm_val": foscttm_val,
        "foscttm_val_in_full_pool": foscttm_val_in_full_pool,
        "foscttm_train_in_full_pool": foscttm_train_in_full_pool,
        "time_err":    time_err,
        "branch_match": branch_match,
        "state_true":  state_true,
        "state_pred":  state_pred,
        "dyn":         dyn_per_cell,
        "align":       align_per_cell,
        "kappa":       kappa_per_cell,
        "alpha_cell_mean": alpha_per_cell,
        "v_eff_norm":  v_eff_norm_per_cell,
        # jvp_norm / rhs_norm / residual_norm / jvp_rhs_cos / norm_ratio /
        # log_norm_ratio / rel_residual
        **physics_per_cell,
    }


def plot_grad_interaction(rows: list, output_path: Path) -> None:
    """How the alignment and kinetics objectives interact on phi, across training.

    a  intrinsic gradient magnitude of each objective on phi
    b  the lambda-weighted ratio: which objective actually drives phi's update
    c  cosine between the two gradients: do they cooperate or fight

    The gradients are probed at a handful of epochs, not every epoch, so the
    samples are drawn as markers with a light connector rather than a solid
    line, which would imply continuity the data does not have (§6.3).
    """
    df = pd.DataFrame(rows)
    g = df.groupby("epoch")
    ep    = g["grad_cos"].mean().index.to_numpy()
    an_m  = g["grad_align_norm"].mean().to_numpy()
    dn_m  = g["grad_dyn_norm"].mean().to_numpy()
    cos_m = g["grad_cos"].mean().to_numpy()
    cos_s = g["grad_cos"].std(ddof=0).fillna(0.0).to_numpy()
    # Runs written before the lambda-scaled columns existed only carry the two raw
    # norms; fall back to the unweighted ratio and say so on the panel rather than
    # failing to plot the run at all.
    has_scaled = "grad_dyn_norm_scaled" in df.columns
    dns_m = g["grad_dyn_norm_scaled"].mean().to_numpy() if has_scaled else None
    if "grad_mag_ratio" in df.columns:
        rat_m = g["grad_mag_ratio"].mean().to_numpy()
    else:
        rat_m = np.divide(dns_m if has_scaled else dn_m, an_m,
                          out=np.zeros_like(an_m), where=an_m > 0)

    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=figsize("full", 1.9))
    probe = dict(marker="o", ms=2.6, lw=0.7, ls=(0, (3, 2)))

    ax = axes[0]
    ax.plot(ep, an_m, color=ink(METHOD_COLORS["kot"]), **probe)
    ax.plot(ep, dn_m, color=ink(MODALITY_COLORS["Protein"]), **probe)
    if has_scaled:
        ax.plot(ep, dns_m, color="0.45", marker="s", ms=2.0, lw=0.7, ls=(0, (1, 2)))
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"$\phi$-gradient norm")
    ax.set_title("Gradient magnitude")
    labelled = [
        ("alignment", an_m, ink(METHOD_COLORS["kot"])),
        ("kinetics", dn_m, ink(MODALITY_COLORS["Protein"])),
    ]
    if has_scaled:
        labelled.append((r"kinetics $\times\ \lambda$", dns_m, "0.45"))
    for label, ys, color in labelled:
        ax.annotate(label, xy=(ep[-1], ys[-1]), xytext=(3, 0),
                    textcoords="offset points", fontsize=6, color=color,
                    va="center", ha="left")
    ax.set_xlim(0, float(ep[-1]) * 1.38)
    panel_letter(ax, "a")

    # A ratio of exactly 0 (lambda_dyn = 0 during warmup) cannot be drawn on a log
    # axis; mask those epochs rather than letting matplotlib drop them silently.
    ax = axes[1]
    ratio_plot = np.where(rat_m > 0, rat_m, np.nan)
    ax.axhline(1.0, color="0.45", lw=0.7, ls=(0, (4, 2)))
    ax.text(0.99, 1.0, " equal ", transform=ax.get_yaxis_transform(),
            fontsize=6, color="0.45", ha="right", va="top")
    ax.plot(ep, ratio_plot, color=ink(METHOD_COLORS["kot"]), **probe)
    ax.set_yscale("log")
    ax.set_xlim(0, float(ep[-1]) * 1.04)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("kinetics / alignment")
    ax.set_title(r"Which objective drives $\phi$")
    if not has_scaled:
        ax.text(0.02, -0.42, r"unweighted ($\lambda$ not recorded)",
                transform=ax.transAxes, fontsize=6, color="0.45",
                ha="left", va="top")
    panel_letter(ax, "b")

    ax = axes[2]
    ax.axhline(0.0, color="0.45", lw=0.7, ls=(0, (4, 2)))
    ax.fill_between(ep, cos_m - cos_s, cos_m + cos_s,
                    color=METHOD_COLORS["kot"], alpha=0.18, linewidth=0)
    ax.plot(ep, cos_m, color=ink(METHOD_COLORS["kot"]), **probe)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlim(0, float(ep[-1]) * 1.04)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("gradient cosine")
    ax.set_title("Cooperate or conflict")
    ax.text(0.99, 0.97, "> 0 cooperate", transform=ax.transAxes,
            fontsize=6, color="0.35", ha="right", va="top")
    ax.text(0.99, 0.03, "< 0 conflict", transform=ax.transAxes,
            fontsize=6, color="0.35", ha="right", va="bottom")
    panel_letter(ax, "c")

    fig.text(0.5, -0.02,
             f"Gradients probed at {len(ep)} epochs; band in c is 1 s.d. over probe batches",
             ha="center", va="top", fontsize=6, color="0.35")
    fig.tight_layout(pad=0.5, w_pad=2.4)
    save_figure(fig, Path(output_path).with_suffix(""))


def plot_foscttm_diagnostics(per_cell: dict, output_path: Path) -> None:
    """What per-cell alignment failure co-varies with.

    One panel per candidate explanation, each plotted against per-cell FOSCTTM
    so the reader can see which quantity actually tracks alignment quality.
    """
    foscttm      = per_cell["foscttm"]
    time_err     = per_cell["time_err"]
    branch_match = per_cell["branch_match"]
    cos_pc       = per_cell["jvp_rhs_cos"]
    dyn_pc       = per_cell["dyn"]
    align_pc     = per_cell["align"]

    apply_style()
    # Constrained layout, not tight_layout: it accounts for tick-label extents when
    # spacing panels, which is what stops a stacked pair's outermost y ticks from
    # meeting. Pruning the end ticks removes the remaining edge case.
    fig, axes = plt.subplots(2, 3, figsize=figsize("full", 3.8),
                             layout="constrained")
    fig.get_layout_engine().set(h_pad=0.06, w_pad=0.05, hspace=0.06, wspace=0.05)
    axes = axes.ravel()
    pt = dict(s=1.2, alpha=0.30, linewidths=0, rasterized=True,
              color=METHOD_COLORS["kot"])

    ax = axes[0]
    if (branch_match >= 0).any():
        correct = foscttm[branch_match == 1]
        wrong   = foscttm[branch_match == 0]
        bp = ax.boxplot([correct, wrong],
                        labels=[f"Correct\n(n={len(correct):,})",
                                f"Wrong\n(n={len(wrong):,})"],
                        widths=0.5, showfliers=False, patch_artist=True)
        for patch, color in zip(bp["boxes"], [METHOD_COLORS["kot"], "#767676"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.45)
            patch.set_linewidth(0.6)
        for key in ("whiskers", "caps", "medians"):
            for line in bp[key]:
                line.set_color("0.25")
                line.set_linewidth(0.7)
        ax.set_ylabel("FOSCTTM")
        ax.set_xlabel("Branch assignment")
        ax.set_title("Alignment vs branch error")
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
    else:
        ax.text(0.5, 0.5, "no ground-truth branch", ha="center", va="center",
                fontsize=6, color="0.45", transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("Alignment vs branch error")
    chance_line(ax, FOSCTTM_CHANCE, axis="y", label="chance")
    panel_letter(ax, "a")

    panels = [
        ("Pseudotime error", time_err,  None,  "b", False),
        ("ODE consistency\ncos(J$_\\phi$v, f)", cos_pc, (-1.05, 1.05), "c", False),
        ("ODE residual", dyn_pc, None, "d", True),
        ("Distance to nearest\nprotein cell", align_pc, None, "e", False),
    ]
    for ax, (ylabel, values, ylim, letter, log_y) in zip(axes[1:5], panels):
        ax.scatter(foscttm, values, **pt)
        ax.set_xlabel("FOSCTTM")
        ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(*ylim)
        if log_y:
            ax.set_yscale("log")
        finite = np.isfinite(foscttm) & np.isfinite(values)
        if finite.sum() > 2:
            r = float(np.corrcoef(foscttm[finite], values[finite])[0, 1])
            ax.text(0.97, 0.05, f"r = {r:+.2f}", transform=ax.transAxes,
                    fontsize=6, color="0.25", ha="right", va="bottom")
        ax.margins(x=0.04, y=0.10)
        if not log_y:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
        panel_letter(ax, letter)

    axes[5].axis("off")
    axes[5].text(0.06, 0.92,
                 "Each point is one cell.\nx-axis is per-cell FOSCTTM\n"
                 "throughout (lower = better).\n\n"
                 "r is the Pearson correlation\nwith FOSCTTM.",
                 transform=axes[5].transAxes, fontsize=6, color="0.35",
                 ha="left", va="top", linespacing=1.5)

    save_figure(fig, Path(output_path).with_suffix(""))


def compute_full_diagnostics(model, inputs: DiagInputs, subset: int = 1024) -> dict:
    """Time, branch, variance, JVP. val_idx is dropped: this is all-cell."""
    R_t, V_eff, P_t, S_t, rna_obs, mask, align_cols = inputs[:7]
    model.eval()

    with torch.no_grad():
        phi_all = model.phi(R_t)
        beta = model.beta.detach()
        phi_metric = phi_all if align_cols is None else phi_all[:, align_cols]
        P_metric = P_t if align_cols is None else P_t[:, align_cols]

    nn_idx_t, _ = nearest_protein_match(phi_metric, P_metric)  # row-chunked
    nn_idx = nn_idx_t.cpu().numpy()

    pseudotime, _ = resolve_pseudotime(rna_obs)
    if pseudotime is not None:
        t_true = pseudotime
        t_pred = t_true[nn_idx]
        time_mae = float(np.mean(np.abs(t_true - t_pred)))
        rank_true = pd.Series(t_true).rank().to_numpy()
        rank_pred = pd.Series(t_pred).rank().to_numpy()
        if rank_true.std() < 1e-12 or rank_pred.std() < 1e-12:
            time_spearman = float("nan")
        else:
            time_spearman = float(np.corrcoef(rank_true, rank_pred)[0, 1])
    else:
        t_true = None
        time_mae = float("nan")
        time_spearman = float("nan")

    # Branch labels exist only on the synthetic.
    if "state" in rna_obs.columns:
        state_true = np.asarray(rna_obs["state"].values)
        state_pred = state_true[nn_idx]
        labels = sorted(set(state_true.tolist()))
        label_to_idx = {label: i for i, label in enumerate(labels)}
        conf = np.zeros((len(labels), len(labels)), dtype=int)
        for t_label, p_label in zip(state_true, state_pred):
            conf[label_to_idx[t_label], label_to_idx[p_label]] += 1
        conf_mat = conf.tolist()
        branch_scores = branch_accuracy_scores(state_true, state_pred)
    else:
        labels = []
        conf_mat = []
        branch_scores = branch_accuracy_scores(np.array([]), np.array([]))

    phi_np = phi_metric.cpu().numpy()
    p_np = P_metric.cpu().numpy()
    var_phi = float(phi_np.var(axis=0).mean())
    var_p   = float(p_np.var(axis=0).mean())
    phi_var_ratio = var_phi / max(var_p, 1e-12)

    # temporal = predicted order; recon = both on true time (warp-tolerant).
    if t_true is not None:
        traj_dtw_temporal = trajectory_dtw(phi_np, p_np, t_pred, t_true)
        traj_dtw_recon    = trajectory_dtw(phi_np, p_np, t_true, t_true)
    else:
        traj_dtw_temporal = float("nan")
        traj_dtw_recon    = float("nan")

    jvp = compute_jvp_rhs_diagnostics(
        model,
        R_t,
        V_eff,
        S_t,
        subset=subset,
        mask=mask,
    )

    beta_np = beta.cpu().numpy()

    out = {
        "time_mae":                  time_mae,
        "time_spearman":             time_spearman,
        "traj_dtw_temporal":         traj_dtw_temporal,
        "traj_dtw_recon":            traj_dtw_recon,
        "branch_confusion_matrix":   conf_mat,
        "branch_confusion_labels":   labels,
        **branch_scores,
        "phi_variance":              var_phi,
        "p_variance":                var_p,
        "phi_variance_ratio":        phi_var_ratio,
        "beta_values":               beta_np.tolist(),
    }
    # JVP/RHS norms, ratios, cosine and κ/α — mean/std/min/max/median + quantiles each,
    # plus the legacy jvp_rhs_cos / jvp_norm / rhs_norm scalars.
    out.update(jvp)
    # β is per protein, not per cell: the distribution is over the panel.
    out.update(summarize_distribution(beta_np, "beta"))
    # ‖v_eff‖ is checkpoint-independent (the velocity is fixed input), but it belongs in
    # every diagnostics JSON so each file explains the physics numbers next to it.
    out.update(velocity_norm_diagnostics(V_eff))
    return out


def anchor_diagnostics(beta: torch.Tensor, use_anchor: bool, anchor_indices: list,
                       anchor_target: torch.Tensor | None, kin_mask, protein_names: list,
                       prior_kind: str, aggregate: str, anchor_scale,
                       subset_n: int | None) -> dict:
    """What the β-anchor asked for and what β actually did, per anchored protein.

    Judged on the anchored proteins alone: the global beta_mean averages the
    unanchored majority and hides whether anchored β moved toward its target.
    `beta_anchor_*_in_kinetic` intersects the anchor set with the runtime kinetic
    mask, because an anchored protein outside that mask constrains nothing.

    An anchor-free run reports the same keys with empty values, so every run's
    diagnostics.json has one shape and a sweep table never has ragged columns.
    """
    out = {
        "beta_anchor_prior":            prior_kind if use_anchor else None,
        "beta_anchor_aggregate":        aggregate if use_anchor else None,
        "beta_anchor_n":                int(len(anchor_indices)) if use_anchor else 0,
        "beta_anchor_scale":            float(anchor_scale) if anchor_scale is not None else None,
        # The ablation's axis: how many anchors were REQUESTED (null = every active one)
        # and the seed the subset was drawn with. Recorded even on the count=0 rung,
        # where the prior is off, so the ladder groups on one column.
        "beta_anchor_subset_n":         subset_n,
        "beta_anchor_in_kinetic_mask":  [],
        "beta_anchor_names_in_kinetic": [],
        "beta_anchor_n_in_kinetic":     0,
        "beta_anchor_indices":          [],
        "beta_anchor_names":            [],
        "beta_anchor_targets":          [],
        "beta_anchor_final":            [],
        "beta_anchor_abs_err":          [],
        "beta_anchor_mean_abs_err":     None,
        "beta_anchor_final_mean":       None,
    }
    if not use_anchor:
        return out

    if anchor_indices:
        in_mask = [bool(kin_mask[j]) for j in anchor_indices]
        out["beta_anchor_in_kinetic_mask"] = in_mask
        out["beta_anchor_names_in_kinetic"] = [
            protein_names[j] for j, ok in zip(anchor_indices, in_mask) if ok
        ]
        out["beta_anchor_n_in_kinetic"] = int(sum(in_mask))
        print(f"[kot] anchors in runtime kinetic mask: "
              f"{out['beta_anchor_n_in_kinetic']}/{len(anchor_indices)}")

    targets = anchor_target.cpu().numpy()
    final = beta.detach().cpu().numpy()[anchor_indices]
    abs_err = [abs(float(f) - float(t)) for f, t in zip(final, targets)]
    out.update({
        "beta_anchor_indices":      [int(j) for j in anchor_indices],
        "beta_anchor_names":        [protein_names[j] for j in anchor_indices],
        "beta_anchor_targets":      [float(t) for t in targets],
        "beta_anchor_final":        [float(f) for f in final],
        "beta_anchor_abs_err":      abs_err,
        "beta_anchor_mean_abs_err": float(sum(abs_err) / len(abs_err)),
        "beta_anchor_final_mean":   float(final.mean()),
    })
    return out


def print_distribution(diagnostics: dict, label: str, prefix: str, fmt: str = "+.4f") -> None:
    """One line per distribution: mean, median and the q05..q95 ladder.

    The physics diagnostics are reported as a median and quantiles rather than a mean
    alone because a handful of cells with near-zero velocity dominate the mean and hide
    what the bulk of the cells are doing.
    """
    print(
        f"[kot]   {label:<20s} mean={diagnostics[f'{prefix}_mean']:{fmt}}  "
        f"med={diagnostics[f'{prefix}_median']:{fmt}}  "
        f"q05={diagnostics[f'{prefix}_q05']:{fmt}}  q25={diagnostics[f'{prefix}_q25']:{fmt}}  "
        f"q75={diagnostics[f'{prefix}_q75']:{fmt}}  q95={diagnostics[f'{prefix}_q95']:{fmt}}"
    )


def print_end_of_training(diagnostics: dict, beta: torch.Tensor, use_anchor: bool,
                          anchor_target: torch.Tensor | None, anchor_loss: float,
                          stop_epoch: int, n_epochs: int) -> None:
    """The end-of-run report: fit quality, parameter distributions, anchor recovery."""
    print("[kot] end-of-training diagnostics:")
    print(f"[kot]   time MAE / Spearman: {diagnostics['time_mae']:.4f} / {diagnostics['time_spearman']:.4f}")
    print(f"[kot]   φ variance ratio:    {diagnostics['phi_variance_ratio']:.4f}")
    print(f"[kot]   JVP·RHS cos:         {diagnostics['jvp_rhs_cos']:.4f}")
    print(f"[kot]   |JVP| / |RHS|:       {diagnostics['jvp_norm']:.4f} / {diagnostics['rhs_norm']:.4f}")

    print("[kot]   per-cell distributions (across cells):")
    print_distribution(diagnostics, "JVP·RHS cos", "jvp_rhs_cos")
    print_distribution(diagnostics, "|JVP|", "jvp_norm", ".4g")
    print_distribution(diagnostics, "|RHS|", "rhs_norm", ".4g")
    print_distribution(diagnostics, "|JVP|/|RHS|", "norm_ratio", ".4g")
    print_distribution(diagnostics, "log |JVP|/|RHS|", "log_norm_ratio")
    print_distribution(diagnostics, "rel. residual", "rel_residual", ".4f")
    print_distribution(diagnostics, "|v_eff|", "v_eff_norm", ".4g")
    print(f"[kot]   v_eff nonzero cells:  "
          f"{diagnostics['v_eff_nonzero_frac'] * 100:.1f}% of {diagnostics['v_eff_n_cells']}")
    if diagnostics.get("velocity_backend_status") == "ok":
        print_distribution(diagnostics, "cos(v, other backend)", "velocity_backend_cos")
    print("[kot]   parameter distributions:")
    print_distribution(diagnostics, "κ", "kappa", ".4g")
    print_distribution(diagnostics, "α", "alpha", ".4g")
    print_distribution(diagnostics, "β", "beta", ".4g")
    print(f"[kot]   κ mean/std [min,max]: {diagnostics['kappa_mean']:.4f} / {diagnostics['kappa_std']:.4f}  "
          f"[{diagnostics['kappa_min']:.4f}, {diagnostics['kappa_max']:.4f}]")
    print(f"[kot]   α mean/std [min,max]: {diagnostics['alpha_mean']:.4f} / {diagnostics['alpha_std']:.4f}  "
          f"[{diagnostics['alpha_min']:.4f}, {diagnostics['alpha_max']:.4f}]")
    print(f"[kot]   β mean/std:          {diagnostics['beta_mean']:.4f} / {diagnostics['beta_std']:.4f}")
    print(f"[kot]   anchor loss:         {anchor_loss:.6f}")
    print(f"[kot]   branch labels:       {diagnostics['branch_confusion_labels']}")
    print("[kot]   branch confusion:")
    for row, label in zip(diagnostics["branch_confusion_matrix"], diagnostics["branch_confusion_labels"]):
        print(f"[kot]     {label:>12s}: {row}")
    print(f"[kot]   final β: {beta.detach().cpu().numpy()}")
    if use_anchor:
        print(
            f"[kot]   anchor recovery: mean|β−β*|={diagnostics['beta_anchor_mean_abs_err']:.4f}  "
            f"anchored β mean={diagnostics['beta_anchor_final_mean']:.4f}  "
            f"(target mean={float(anchor_target.cpu().numpy().mean()):.4f})"
        )
        for name, tgt, fin, err in zip(
            diagnostics["beta_anchor_names"],
            diagnostics["beta_anchor_targets"],
            diagnostics["beta_anchor_final"],
            diagnostics["beta_anchor_abs_err"],
        ):
            print(f"[kot]     {name:<18} β*={tgt:.4f}  β={fin:.4f}  |Δ|={err:.4f}")
    print(f"[kot] stopped at epoch {stop_epoch} / {n_epochs}")


def validation_metrics(per_cell: dict) -> dict:
    """The two validation FOSCTTM means for one checkpoint (None when no split).

    val_foscttm is the number a sweep ranks on; val_foscttm_in_full_pool is the same
    cells scored against all n, so the pair says how much of a difference is the
    smaller candidate pool rather than the model.

    The counts describe what these FOSCTTM means were MEASURED over, so they are None
    when nothing was -- end-of-training diagnostics run on the fitted cells alone and
    drop val_idx, which used to make this report 0 fitted cells for a run that fitted
    thousands. The split's own counts are n_train_cells / n_held_out_cells.
    """
    within = per_cell["foscttm_val"]
    in_pool = per_cell["foscttm_val_in_full_pool"]
    on_train = per_cell["foscttm_train_in_full_pool"]
    return {
        "val_foscttm": float(np.mean(within)) if within.size else None,
        "val_foscttm_in_full_pool": float(np.mean(in_pool)) if in_pool.size else None,
        "val_n_cells": int(within.size) if within.size else None,
        # FOSCTTM on the cells this run fitted, same pool as val_foscttm_in_full_pool
        # so the two are directly comparable. This is what a sweep ranks on when the
        # run trains on the tuning subset (fit_tuning_subset).
        "train_foscttm": float(np.mean(on_train)) if on_train.size else None,
        "train_n_cells": int(on_train.size) if on_train.size else None,
    }


def write_run_artifacts(output_dir, diagnostics: dict, best_align_full: dict,
                        velocity_diag: dict, checkpoints: list, best_align_state,
                        model, diag_inputs: DiagInputs, model_name: str, seed: int,
                        grad_interaction_rows: list, protein_units: str) -> None:
    """Everything a finished run leaves on disk, in one place.

    Writes diagnostics.json, one .pt per recorded checkpoint, a per-checkpoint
    diagnostics/FOSCTTM pair plus the summary CSV, the best_align per-cell arrays
    and their scatter panel, and the gradient-interaction trace.

    The model is left on the best_align state, so whatever the caller does next
    matches the `aligned` matrices it returns.
    """
    out = Path(output_dir)

    for name, state, ep, lo in checkpoints:
        if state is None:
            continue
        ckpt_path = out / f"checkpoint_{name}.pt"
        torch.save(
            {
                "state_dict": state,
                "epoch":      ep,
                "losses":     lo,
                "model_name": model_name,
                "seed":       seed,
                # The units phi was trained to predict. A scorer that rescales phi onto
                # observed ADT has to know this: a CLR checkpoint and an rna_size one
                # have identical shapes, so without the stamp the two are
                # indistinguishable and the wrong one is silently reported.
                "protein_units": protein_units,
            },
            ckpt_path,
        )
        print(f"[kot] Saved checkpoint: {ckpt_path} (epoch {ep})")

    # Every checkpoint, not just best_align. best_align reuses diagnostics already
    # computed above and caches per-cell for the scatter — once, not three times.
    per_ckpt_rows = []
    best_align_per_cell = None
    for name, state, ep, lo in checkpoints:
        if state is None:
            continue
        model.load_state_dict(state)
        if name == "best_align":
            d = dict(best_align_full)
        else:
            d = compute_full_diagnostics(model, diag_inputs)
        pc = compute_per_cell_diagnostics(model, diag_inputs)
        if name == "best_align":
            best_align_per_cell = pc
        d.update(velocity_diag)
        mean_f = float(np.mean(pc["foscttm"]))
        val_metrics = validation_metrics(pc)
        d.update(val_metrics)
        # diagnostics.json is best_align — runner and collect_run_diagnostics read it.
        if name == "best_align":
            diagnostics.update(val_metrics)
        d["mean_foscttm"]      = mean_f
        d["checkpoint"]        = name
        d["checkpoint_epoch"]  = ep
        d["checkpoint_losses"] = lo
        # collect_run_diagnostics columns only scalars; checkpoint_losses is a dict.
        d.update({f"loss_{key}": value for key, value in lo.items()})
        if name == "best_align":
            diagnostics.update({f"loss_{key}": value for key, value in lo.items()})
        with open(out / f"diagnostics_{name}.json", "w") as f:
            json.dump(d, f, indent=2)
        pd.DataFrame({"foscttm": pc["foscttm"]}).to_csv(
            out / f"foscttm_{name}.csv", index=False,
        )
        row = {
            "checkpoint":         name,
            "epoch":              ep,
            "mean_foscttm":       mean_f,
            "val_foscttm":        val_metrics["val_foscttm"],
            "val_foscttm_in_full_pool": val_metrics["val_foscttm_in_full_pool"],
            "time_mae":           d["time_mae"],
            "time_spearman":      d["time_spearman"],
            "traj_dtw_temporal":  d["traj_dtw_temporal"],
            "traj_dtw_recon":     d["traj_dtw_recon"],
            "phi_variance_ratio": d["phi_variance_ratio"],
            "jvp_rhs_cos":        d["jvp_rhs_cos"],
            "jvp_norm":           d["jvp_norm"],
            "rhs_norm":           d["rhs_norm"],
        }
        # Distributions in the summary CSV so a box plot needs nothing else.
        for prefix in JVP_DIAG_PREFIXES + PARAM_DIAG_PREFIXES + ("beta", "v_eff_norm"):
            for stat in STAT_SUFFIXES:
                row[f"{prefix}_{stat}"] = d[f"{prefix}_{stat}"]
        row["v_eff_nonzero_frac"] = d["v_eff_nonzero_frac"]
        row["velocity_backend_cos_mean"]   = d["velocity_backend_cos_mean"]
        row["velocity_backend_cos_median"] = d["velocity_backend_cos_median"]
        row["velocity_backend_cos_q25"]    = d["velocity_backend_cos_q25"]
        row["velocity_backend_cos_q75"]    = d["velocity_backend_cos_q75"]
        per_ckpt_rows.append(row)
        val_text = (
            "" if val_metrics["val_foscttm"] is None
            else (f"  val={val_metrics['val_foscttm']:.4f} "
                  f"(n={val_metrics['val_n_cells']}, "
                  f"in full pool {val_metrics['val_foscttm_in_full_pool']:.4f})")
        )
        print(f"[kot] Eval {name}: FOSCTTM={mean_f:.4f}{val_text}  "
              f"cos={d['jvp_rhs_cos']:+.3f} (med {d['jvp_rhs_cos_median']:+.3f})  "
              f"ratio med={d['norm_ratio_median']:.3f}  "
              f"relres med={d['rel_residual_median']:.3f}  "
              f"κ={d['kappa_mean']:.3f} (med {d['kappa_median']:.3f}) "
              f"[{d['kappa_min']:.3f},{d['kappa_max']:.3f}]  "
              f"α={d['alpha_mean']:.3f} (med {d['alpha_median']:.3f}) "
              f"[{d['alpha_min']:.3f},{d['alpha_max']:.3f}]  "
              f"β={d['beta_mean']:.3f} (med {d['beta_median']:.3f})")
    if per_ckpt_rows:
        pd.DataFrame(per_ckpt_rows).to_csv(
            out / "checkpoint_eval_summary.csv", index=False,
        )

    # After the loop: that is where best_align's val FOSCTTM is written.
    diag_path = out / "diagnostics.json"
    with open(diag_path, "w") as f:
        json.dump(diagnostics, f, indent=2)
    print(f"[kot] Saved diagnostics: {diag_path}")

    # Same checkpoint as the returned `aligned`.
    if best_align_state is not None:
        model.load_state_dict(best_align_state)

    # Recompute only if best_align was never a checkpoint.
    if best_align_per_cell is not None:
        per_cell = best_align_per_cell
    else:
        per_cell = compute_per_cell_diagnostics(model, diag_inputs)
    per_cell_path = out / "per_cell_diagnostics.npz"
    np.savez(per_cell_path, **per_cell)
    plot_path = out / "foscttm_diagnostics.png"
    plot_foscttm_diagnostics(per_cell, plot_path)
    print(f"[kot] Saved per-cell arrays: {per_cell_path}")
    print(f"[kot] Saved FOSCTTM diagnostics plot: {plot_path}")

    # Per-batch alignment-vs-kinetics gradient interaction (epoch × probe batch).
    if grad_interaction_rows:
        gi_path = out / "grad_interaction.csv"
        pd.DataFrame(grad_interaction_rows).to_csv(gi_path, index=False)
        gi_plot = out / "grad_interaction.png"
        plot_grad_interaction(grad_interaction_rows, gi_plot)
        print(f"[kot] Saved grad-interaction diagnostics: {gi_path} + {gi_plot}")


def select_anchor_entries(keep: list[int], indices: list, betas: list,
                          weights: list, sigmas: list) -> tuple[list, list, list, list]:
    """Take positions `keep` out of the parallel anchor lists.

    A list whose length does not match `indices` was not supplied per anchor by a
    manual config, so it passes through untouched and `anchor_tensors` falls back to
    its defaults for it. One helper because both anchor filters -- the kinetic-mask
    intersection and the anchor-count ablation -- have to keep the four lists aligned.
    """
    picked = [[v[k] for k in keep] if len(v) == len(indices) else v
              for v in (betas, weights, sigmas)]
    return [indices[k] for k in keep], *picked


def anchor_subset_indices(n_active: int, keep_n: int, seed: int) -> list[int]:
    """Which active anchors survive an anchor-count ablation, seeded by the run seed.

    Returns positions into the active anchor list, drawn as the head of ONE permutation
    so the ladder is nested within a seed: the keep_n=3 set is a subset of the keep_n=5
    set, and a step down the ladder removes anchors instead of re-drawing a different
    set. Seeded off the run seed, so each
    seed keeps a different subset -- the arm then carries the variance of WHICH anchors
    were kept, not the luck of one pick, and is comparable to the full-anchor arm whose
    spread is training noise alone.
    """
    if not 0 <= keep_n <= n_active:
        raise ValueError(
            f"beta_anchor_subset_n={keep_n} is outside 0..{n_active}, the number of "
            f"anchors active in this run. Pick a rung the dataset actually has."
        )
    return sorted(int(k) for k in np.random.default_rng(seed).permutation(n_active)[:keep_n])


def anchor_tensors(indices: list, betas: list, sigmas: list, weights: list,
                   default_beta: float, prior_sigma_floor: float, target_beta: float,
                   device: torch.device):
    """The β-anchor spec as (indices, targets, sigmas, weights) tensors.

    Targets fall back to a single `default_beta` for every anchor when no
    per-protein β* was resolved, and sigmas to a flat floor when the CSV carried
    no cross-cell-type dispersion. Weights stay None when none were given, which
    `beta_anchor_prior_loss` reads as an unweighted mean.
    """
    if betas and len(betas) != len(indices):
        raise ValueError("kot_anchor_betas must have the same length as kot_anchor_indices.")

    idx_t = torch.tensor(indices, dtype=torch.long, device=device)
    if betas:
        target = torch.tensor(betas, dtype=torch.float32, device=device)
    else:
        target = torch.full((idx_t.numel(),), default_beta, device=device)

    if sigmas and len(sigmas) == len(indices):
        sigma_t = torch.tensor(sigmas, dtype=torch.float32, device=device)
    else:
        sigma_t = torch.full((idx_t.numel(),), prior_sigma_floor * target_beta, device=device)

    weight_t = None
    if weights and len(weights) == len(indices):
        weight_t = torch.tensor(weights, dtype=torch.float32, device=device)
    return idx_t, target, sigma_t, weight_t


def subsample_rows(n_rows: int, size: int, device, rows: torch.Tensor | None) -> torch.Tensor:
    """`size` cells drawn without replacement from `rows` (None = the first n_rows)."""
    picks = torch.randperm(n_rows, device=device)[:size]
    return picks if rows is None else rows[picks]


def shuffled_batches(n_rows: int, batch_size: int, device, rows: torch.Tensor | None):
    """`rows` in a fresh random order, split into batches (None = the first n_rows)."""
    order = torch.randperm(n_rows, device=device)
    if rows is not None:
        order = rows[order]
    return order.split(batch_size)


def sinkhorn_on_rows(model, R_t: torch.Tensor, P_t: torch.Tensor, rna_idx, protein_idx,
                     align_cols: torch.Tensor | None, blur: float, backend: str):
    """φ(R) vs observed protein as a Sinkhorn divergence, on the given rows.

    One definition for both the training step and the held-out check, so the two
    numbers in training_loss.csv are the same estimator on different cells and the
    gap between them means what it looks like. `rna_idx`/`protein_idx` are None for
    the full-batch case, which avoids gathering the whole (n, D_r) matrix.
    """
    phi_align = model.phi(R_t if rna_idx is None else R_t[rna_idx])   # φ predicts full panel
    P_align = P_t if protein_idx is None else P_t[protein_idx]
    if align_cols is not None:                       # only use_for_alignment proteins
        phi_align = phi_align[:, align_cols]
        P_align = P_align[:, align_cols]
    return sinkhorn_divergence(phi_align, P_align, blur=blur, backend=backend)


def alignment_step(model, R_t: torch.Tensor, P_t: torch.Tensor, n_train: int, device,
                   max_points: int | None, align_cols: torch.Tensor | None,
                   blur: float, backend: str, lam_align: float,
                   train_idx: torch.Tensor | None) -> float:
    """One Sinkhorn step on φ, backward already applied. Returns the raw loss.

    Full-batch when small (exact); a fresh random minibatch each epoch once
    `max_points` caps it, so a large panel stays within GPU memory instead of
    building an n×n cost matrix. RNA and protein
    cells are sampled INDEPENDENTLY — the data is unpaired, and drawing the same
    rows would put each cell's true partner in the batch and leak the alignment.

    Every draw comes from `train_idx` (None = every cell), so a held-out
    validation cell never reaches the loss.

    Backward runs here so the φ/Sinkhorn graph is freed before the kinetics loop.
    """
    if max_points is not None and n_train > max_points:
        rna_idx = subsample_rows(n_train, max_points, device, train_idx)
        protein_idx = subsample_rows(n_train, max_points, device, train_idx)
    else:
        rna_idx = train_idx
        protein_idx = train_idx
    loss_align = sinkhorn_on_rows(
        model, R_t, P_t, rna_idx, protein_idx, align_cols, blur, backend,
    )
    (lam_align * loss_align).backward()
    return loss_align.item()


def validation_align_loss(model, R_t: torch.Tensor, P_t: torch.Tensor,
                          val_idx: torch.Tensor, device, max_points: int | None,
                          align_cols: torch.Tensor | None, blur: float,
                          backend: str) -> float:
    """The same Sinkhorn divergence, on the held-out cells. No gradient, no pairing.

    This is the only UNSUPERVISED held-out signal a KOT run has: a Sinkhorn
    divergence compares two point clouds and never looks at which RNA cell belongs
    to which protein cell, so watching it costs no label information — unlike
    val_foscttm, which does and is therefore reserved for ranking configs offline.

    model.eval() is not optional here: with phi_spectral_norm on, a forward pass in
    training mode advances the power iteration, so a monitoring call would perturb
    the very run it is monitoring.
    """
    was_training = model.training
    model.eval()
    with torch.no_grad():
        n_val = val_idx.numel()
        if max_points is not None and n_val > max_points:
            rna_idx = subsample_rows(n_val, max_points, device, val_idx)
            protein_idx = subsample_rows(n_val, max_points, device, val_idx)
        else:
            rna_idx = val_idx
            protein_idx = val_idx
        loss = sinkhorn_on_rows(
            model, R_t, P_t, rna_idx, protein_idx, align_cols, blur, backend,
        )
    if was_training:
        model.train()
    return float(loss)


def kinetics_step(model, R_t: torch.Tensor, V_eff_t: torch.Tensor, SR_t: torch.Tensor,
                  conf_t: torch.Tensor, n_train: int, batch_size: int, device,
                  mask_t: torch.Tensor | None, protein_scale: torch.Tensor | None,
                  lam_dyn: float, kappa_target_t: torch.Tensor | None,
                  lambda_kappa_prior: float,
                  train_idx: torch.Tensor | None) -> tuple[float, float]:
    """The ODE residual and the κ prior, accumulated over batches. Returns both losses.

    Shuffles once on-GPU and splits; each batch's JVP graph is freed by its own
    backward() rather than every graph living until the end of the epoch. Batches
    are cell-weighted (len_b/n_train) so the accumulated gradient equals one
    backward() on the mean over the TRAINING cells — held-out cells are absent from
    the shuffle, so they contribute neither a gradient nor a denominator. κ from
    each batch is reused for the prior instead of a second κ(R_t) pass.

    With the kinetics term off, the κ prior still needs its own pass — but only
    while κ is trainable, since otherwise it has nowhere to go.
    """
    dyn_val = 0.0
    kappa_prior_val = 0.0
    prior_on = lambda_kappa_prior > 0.0 and kappa_target_t is not None

    if lam_dyn > 0.0:
        for idx in shuffled_batches(n_train, batch_size, device, train_idx):
            weight = idx.numel() / n_train
            loss_dyn_b, kappa_b = kinetics_loss(
                model.phi, model.kappa, model.g,
                R_t[idx], V_eff_t[idx], SR_t[idx], model.beta, conf_t[idx],
                mask=mask_t, scale=protein_scale,
            )
            batch_obj = (lam_dyn * weight) * loss_dyn_b
            dyn_val += weight * loss_dyn_b.item()
            if prior_on:
                kp_b = kappa_prior_loss(kappa_b, kappa_target_t, lambda_kappa_prior)
                batch_obj = batch_obj + weight * kp_b
                kappa_prior_val += weight * kp_b.item()
            batch_obj.backward()
        return dyn_val, kappa_prior_val

    if prior_on and any(param.requires_grad for param in model.kappa.parameters()):
        for idx in shuffled_batches(n_train, batch_size, device, train_idx):
            weight = idx.numel() / n_train
            kp_b = kappa_prior_loss(model.kappa(R_t[idx]), kappa_target_t, lambda_kappa_prior)
            (weight * kp_b).backward()
            kappa_prior_val += weight * kp_b.item()
    return dyn_val, kappa_prior_val


def run_kot(context: dict, cfg: dict) -> tuple[list, np.ndarray | None, pd.DataFrame]:
    """
    Full KOT training + alignment.

    Returns
    -------
    aligned  : [rna_aligned (n, D_p), protein_obs (n, D_p)]
    coupling : None; KOT currently optimises the Sinkhorn divergence directly
    loss_df  : per-epoch loss history
    """
    device = choose_torch_device(cfg)

    x = context["x"]             # runner RNA representation; may be replaced below
    y = context["y"]             # runner second-modality representation; may be replaced below
    rna_adata    = context["rna_adata"]
    second_adata = context["second_adata"]   # protein AnnData

    model_name   = str(cfg.get("model",       "kot"))
    n_epochs     = int(cfg.get("n_epochs",     500))
    batch_size   = int(cfg.get("batch_size",   256))
    lr           = float(cfg.get("lr",         1e-3))
    lambda_dyn   = float(cfg.get("lambda_dyn", 1.0))
    lambda_reg   = float(cfg.get("lambda_reg", 1e-4))
    blur         = float(cfg.get("sinkhorn_reg", 0.1))
    # KeOps often cannot load CUDA; tensorized stays on GPU. See sinkhorn.py.
    sinkhorn_backend = str(cfg.get("sinkhorn_backend", "auto"))
    seed         = int(cfg.get("seed",         42))
    # None = full batch. Large n×n would not fit; small/synthetic stays exact.
    sinkhorn_max_points = cfg.get("sinkhorn_max_points")
    if sinkhorn_max_points is not None:
        sinkhorn_max_points = int(sinkhorn_max_points)
    phi_dims     = list(cfg.get("phi_dims",    [1024, 512, 256]))
    kappa_dims   = list(cfg.get("kappa_dims",  [64, 32]))
    g_dims       = list(cfg.get("g_dims",      [256, 128]))
    phi_init_gain = float(cfg.get("phi_init_gain", 0.1))
    activation = str(cfg.get("kot_activation", "gelu"))
    init_method = str(cfg.get("kot_init", "orthogonal"))
    use_explicit_links = bool(cfg.get("kot_use_explicit_links", True))
    use_anchor   = bool(cfg.get("use_anchor",        False))
    anchor_beta  = float(cfg.get("kot_anchor_beta",  0.5))
    anchor_indices = list(cfg.get("kot_anchor_indices", []))
    anchor_betas = list(cfg.get("kot_anchor_betas", []))
    # β-anchor from measured half-lives: overrides the manual indices/betas. Betas
    # are rescaled to the model's β range (relative degradation pattern). Soft prior
    # (l2 / gaussian / log_normal) — never hard-fixed unless kot_fixed_beta is set.
    anchor_source = "manual config (kot_anchor_indices)"
    anchor_sigmas_list: list[float] = []
    anchor_weights_list: list[float] = []
    beta_anchor_csv = cfg.get("beta_anchor_csv")
    beta_anchor_prior = str(cfg.get("beta_anchor_prior", "log_normal")).lower()
    beta_anchor_aggregate = str(cfg.get("beta_anchor_aggregate", "median")).lower()
    stable_cv_cfg = cfg.get("beta_anchor_stable_cv_max")
    stable_cv_max = float(stable_cv_cfg) if stable_cv_cfg is not None else None
    min_anchor_quality_cfg = cfg.get("beta_anchor_min_r2")
    min_anchor_quality = float(min_anchor_quality_cfg) if min_anchor_quality_cfg is not None else None
    prior_sigma_floor = float(cfg.get("beta_anchor_sigma_floor", 0.25))
    beta_warmstart = bool(cfg.get("kot_beta_warmstart_from_anchor", False))
    # Anchor-count ablation: keep only this many of the anchors that actually act
    # (null = keep all, which is the untouched code path). Applied after the kinetic-mask
    # intersection below, so the number asked for is the number of priors in the loss.
    anchor_subset_cfg = cfg.get("beta_anchor_subset_n")
    anchor_subset_n = int(anchor_subset_cfg) if anchor_subset_cfg is not None else None
    anchor_scale = None
    if beta_anchor_csv and Path(beta_anchor_csv).exists():
        # beta_anchor_fixed_scale freezes the physical→model β unit (compute it once on
        # the full anchor set) so anchor-count ablations do not redefine the unit.
        idx, betas, weights, sigmas, anchor_scale = resolve_beta_anchors(
            beta_anchor_csv,
            list(second_adata.var_names),
            target_mean_beta=float(cfg.get("beta_anchor_target", 0.5)),
            aggregate=beta_anchor_aggregate,
            stable_cv_max=stable_cv_max,
            prior_sigma_floor=prior_sigma_floor,
            fixed_scale=cfg.get("beta_anchor_fixed_scale"),
            min_quality=min_anchor_quality,
        )
        if idx:
            anchor_indices, anchor_betas = idx, betas
            anchor_weights_list = weights
            anchor_sigmas_list = sigmas
            anchor_source = (
                f"paper half-lives {beta_anchor_csv} "
                f"(aggregate={beta_anchor_aggregate}, prior={beta_anchor_prior}, "
                f"stable_cv_max={stable_cv_max}, scale={anchor_scale:.4g} "
                f"[{'frozen' if cfg.get('beta_anchor_fixed_scale') is not None else 'geom-mean'}])"
            )
    # Control: permute the target values across the anchored proteins (same set of
    # β* values, wrong assignment). If β still sits at these shuffled targets and
    # FOSCTTM is unchanged, the dynamics does not constrain β (we are asserting, not
    # recovering); if β drifts back or FOSCTTM worsens, the data prefers the true pattern.
    if bool(cfg.get("kot_beta_anchor_shuffle", False)) and anchor_betas:
        perm = np.random.default_rng(int(cfg.get("seed", 42))).permutation(len(anchor_betas))
        anchor_betas = [anchor_betas[i] for i in perm]
        print(f"[kot] SHUFFLED anchor targets (control): β* permuted across {len(anchor_betas)} proteins")
    lambda_prior = float(cfg.get("lambda_prior",     1.0))
    fixed_kappa_value = cfg.get("kot_fixed_kappa")
    fixed_alpha_value = cfg.get("kot_fixed_alpha")
    fixed_beta_value = cfg.get("kot_fixed_beta")
    phi_spectral_norm = bool(cfg.get("phi_spectral_norm", False))
    kappa_min = float(cfg.get("kot_kappa_min", 1e-6))
    kappa_max_cfg = cfg.get("kot_kappa_max")
    kappa_max = float(kappa_max_cfg) if kappa_max_cfg is not None else None
    alpha_min = float(cfg.get("kot_alpha_min", 1e-6))
    alpha_max_cfg = cfg.get("kot_alpha_max")
    alpha_max = float(alpha_max_cfg) if alpha_max_cfg is not None else None
    kappa_prior_target_cfg = cfg.get("kot_kappa_prior")
    kappa_prior_target = (
        float(kappa_prior_target_cfg) if kappa_prior_target_cfg is not None else None
    )
    lambda_kappa_prior = float(cfg.get("lambda_kappa_prior", 0.0))
    allow_empty_kinetic_mask = bool(cfg.get("kot_allow_empty_kinetic_mask", False))
    allow_empty_anchor_after_mask = bool(
        cfg.get("kot_allow_empty_anchor_after_kinetic_mask", False)
    )
    g_freeze_epochs = int(cfg.get("g_freeze_epochs", 0))
    dyn_warmup_epochs       = int(cfg.get("dyn_warmup_epochs",       n_epochs // 2))
    # Per-head rates (null = fall back to `lr`) and the shared warmup->cosine schedule.
    # Read here only to print and to record in diagnostics; build_kot_optimiser takes
    # them straight off cfg so there is one place that resolves them.
    lr_phi          = cfg.get("lr_phi")
    lr_alpha_kappa  = cfg.get("lr_alpha_kappa")
    lr_beta         = cfg.get("lr_beta")
    lr_warmup_epochs      = int(cfg.get("lr_warmup_epochs", 0))
    lr_warmup_start_factor = float(cfg.get("lr_warmup_start_factor", 1.0))
    lr_min_factor          = float(cfg.get("lr_min_factor", 0.0))
    early_stopping_patience = int(cfg.get("early_stopping_patience", 100))
    early_stopping_monitor  = str(cfg.get("early_stopping_monitor", "train_align"))
    if early_stopping_monitor not in ("train_align", "val_align"):
        raise ValueError(
            f"early_stopping_monitor must be 'train_align' or 'val_align', "
            f"got {early_stopping_monitor!r}."
        )
    monitor_is_val = early_stopping_monitor == "val_align"
    # The held-out Sinkhorn is a sampling interval when it is only being logged, but the
    # criterion itself has to be measured every epoch or `patience` would silently count
    # in units of val_every instead of epochs.
    val_every = 1 if monitor_is_val else int(cfg.get("val_every", 5))

    # Alternating-phase optimization. If any phase{1,2,3}_epochs > 0, phase mode
    # is enabled and n_epochs is overridden by the sum. Otherwise behavior is
    # joint training with optional dyn_warmup.
    phase1_epochs   = int(cfg.get("phase1_epochs",   0))
    phase2_epochs   = int(cfg.get("phase2_epochs",   0))
    phase3_epochs   = int(cfg.get("phase3_epochs",   0))
    phase3_lr_scale = float(cfg.get("phase3_lr_scale", 0.1))
    use_phases = (phase1_epochs + phase2_epochs + phase3_epochs) > 0
    if use_phases:
        n_epochs = phase1_epochs + phase2_epochs + phase3_epochs
        print(f"[kot] Phases: P1={phase1_epochs} (Sinkhorn) "
              f"P2={phase2_epochs} (Dynamics) "
              f"P3={phase3_epochs} (Joint @ lr×{phase3_lr_scale})")
    elif dyn_warmup_epochs >= n_epochs:
        # Checkpointing and early stopping only start once the ramp is done (see
        # monitor_align below), so a ramp that outlasts the run leaves best_align
        # unrecorded: no early stopping, no model selection, and the run silently
        # returns its final state instead. Nothing downstream errors on that, so it has
        # to be caught here.
        raise ValueError(
            f"dyn_warmup_epochs ({dyn_warmup_epochs}) must be shorter than n_epochs "
            f"({n_epochs}), or the ramp never finishes and no best_align checkpoint is "
            f"ever recorded. Shorten the ramp or raise n_epochs."
        )

    use_feature_space = bool(cfg.get("kot_use_feature_space", False))
    rna_layer = cfg.get("kot_rna_layer")
    protein_layer = cfg.get("kot_protein_layer")
    velocity_layer = cfg.get("kot_velocity_layer") or cfg.get("velocity_layer")

    # The protein target goes through the SAME contract as the supervised baselines
    # (src/training/protein_supervised.adt_targets), so `protein_target_normalization`
    # means one thing for every method on a dataset. Reading protein_adata.X directly
    # trained phi on CLR while ridge, MLP, sciPENN and scButterfly trained on the
    # benchmark's rna_size units, and CLR is a per-cell row operation that the
    # per-protein rescale in the CRISPR scorer cannot invert: a PERFECT CLR map scores
    # spearman +0.37 on the Papalexi primary set against ridge's +0.51, so phi was
    # ranked against a ceiling it could not reach. Worse, CLR closure forces the four
    # predicted deltas onto one compositional axis -- phi got PDL1 right (+0.74) and
    # the other three exactly backwards.
    if use_feature_space:
        x = matrix_from_adata(rna_adata, rna_layer, "RNA")
        y = adt_targets(context, cfg)
        print(
            "[kot] Using feature-space inputs: "
            f"RNA={x.shape} ({rna_layer or 'X'}), protein={y.shape} "
            f"({protein_target_units(cfg, protein_layer)})"
        )

    n, D_r = x.shape
    D_p = y.shape[1]

    # Gene→protein S matrix. In feature-space mode this is the paper's S.
    # In representation mode it is projected into the same PCA coordinates as phi.
    mapping_csv = cfg.get("adt_mapping_csv")
    mapping_records = None
    if mapping_csv:
        # A configured mapping CSV is the source of truth. If it is missing we do NOT
        # silently fall back to live HGNC matching — fail loud and tell the user to
        # build it, so a public run never trains on an ad-hoc, unaudited mapping.
        if not Path(mapping_csv).exists():
            raise FileNotFoundError(
                f"adt_mapping_csv '{mapping_csv}' not found. Build the mapping first:\n"
                f"  PYTHONPATH=. python tools/build_inputs.py adt-mapping --datasets <dataset>"
            )
        mapping_records = load_mapping_records(Path(mapping_csv))   # validated, source of truth
        print(f"[kot] ADT→gene mapping from CSV (source of truth): {mapping_csv} "
              f"({len(mapping_records)} rows)")
    # For public data with a CSV, require the mapping to cover the exact panel: a
    # protein with no mapping cannot align anyway, so partial coverage is an error.
    require_full_panel = bool(cfg.get("kot_require_full_panel_mapping", True)) and mapping_records is not None
    S_np, align_mask, kin_mask, mapping_report = projection_matrix_from_adatas(
        rna_adata,
        second_adata,
        mapping_records=mapping_records,
        use_mean_expr=False,
        use_explicit_links=use_explicit_links,
        require_full_panel=require_full_panel,
    )
    # Control for the RNA->protein map: reassign gene rows across mapped proteins. It has
    # to happen here, on the feature-space S, because S_model below is projected through
    # the PCs -- permuting rows after that would shuffle protein PCA components, which
    # are not proteins and carry no mapping to destroy.
    S_np = apply_s_permutation(S_np, cfg)
    # Kinetics mask over the protein axis: only in feature space is each protein a
    # row of φ; in representation mode the protein-PCA axis mixes proteins, so no
    # per-protein masking (all components participate).
    if use_feature_space:
        S_model = S_np.copy()
    elif "PCs" in rna_adata.varm:
        S_model = S_np @ np.array(rna_adata.varm["PCs"])  # (n_proteins, D_r)
        if "PCs" in second_adata.varm:
            protein_PCs = np.array(second_adata.varm["PCs"])  # (n_proteins, D_p)
            S_model = protein_PCs.T @ S_model                 # (D_p, D_r)
        else:
            raise ValueError("KOT representation mode requires protein PCA components in .varm['PCs'].")
    else:
        raise ValueError("KOT representation mode requires RNA PCA components in .varm['PCs'].")

    # RNA velocity in the same coordinate system as the KOT RNA input, plus the per-cell
    # scVelo confidence that weights its residual in the kinetics loss.
    V, selected_velocity_layer = resolve_velocity_matrix(
        rna_adata, velocity_layer, D_r, use_feature_space,
    )
    conf = resolve_velocity_confidence(rna_adata, n)
    # Kinetics-term control: shuffle / reverse / zero the velocity, each breaking one
    # kind of information in L_dyn (see apply_velocity_ablation). "none" is the real arm.
    V, conf = apply_velocity_ablation(V, conf, cfg)

    # (Velocity gauge normalisation now happens on the EFFECTIVE velocity V_eff below,
    # after the RNA-side weight is applied — the gauge must be measured from the exact
    # vector fed to the JVP, not from all raw velocity coordinates.)

    data_debug = kot_data_debug_summary(
        rna_adata,
        rna_layer,
        protein_layer,
        selected_velocity_layer,
        x,
        y,
        V,
        S_model,
    )
    print_kot_data_debug(data_debug)
    output_dir = context.get("output_dir")

    # Per-protein mapping report: mapped gene, mapping type, alignment / kinetic
    # flags, RNA presence, and exclusion reason — one row per panel protein.
    if output_dir is not None and mapping_report:
        report_path = Path(output_dir) / "mapping_report.csv"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(mapping_report).to_csv(report_path, index=False)
        print(f"[kot] wrote mapping report: {report_path}")

    R_t    = to_tensor(x,     device)   # (n, D_r)
    P_t    = to_tensor(y,     device)   # (n, D_p)
    S_t    = to_tensor(S_model, device)   # (D_p, D_r)
    conf_t = to_tensor(conf,  device)   # (n,)

    # ---- Held-out cells (see src/data/splits.py) ----
    # Training exclusion, checkpoint-selection validation, and OOD test membership are
    # separate concepts. In particular, Papalexi KO protein is test-only and cannot enter
    # validation_align_loss or select best_val_align.
    val_fraction   = float(cfg.get("val_fraction", 0.0))
    val_split_seed = resolve_split_seed(
        int(cfg.get("val_split_seed", 20260825)),
        seed,
        bool(cfg.get("val_split_per_seed", False)),
    )
    val_stratify_by = cfg.get("val_stratify_by") or None
    fit_obs_key = cfg.get("fit_obs_key") or None

    held_out_np = held_out_mask(rna_adata.obs, cfg, seed)
    val_mask_np = selection_validation_mask(rna_adata.obs, cfg, seed)
    group_held_out_np = group_holdout_mask(rna_adata.obs, cfg)
    n_val = int(val_mask_np.sum())
    n_group_held_out = int(group_held_out_np.sum())
    val_holdout = bool(cfg.get("val_holdout_from_training", True)) and n_val > 0
    val_idx_t = (
        to_index_tensor(np.nonzero(val_mask_np)[0], device) if n_val > 0 else None
    )
    # None means "every cell": it keeps the full-batch alignment path from gathering the
    # whole (n, D_r) matrix once an epoch just to re-index it in the original order.
    has_held_out = bool(held_out_np.any())
    train_idx_t = (
        to_index_tensor(np.nonzero(~held_out_np)[0], device) if has_held_out else None
    )
    n_held_out = int(held_out_np.sum())
    n_train = n - n_held_out

    # Kinetics never reads observed protein, so it can run on held-out RNA.
    # Without that, phi on held-out cells is only a distribution match and
    # predicted deltas collapse. Alignment stays on fitted cells: matching
    # held-out RNA to control protein would train phi to erase the perturbation.
    kinetics_on_held_out = bool(cfg.get("kot_kinetics_on_held_out", False))
    dyn_idx_t = None if kinetics_on_held_out else train_idx_t
    n_dyn = n if kinetics_on_held_out else n_train
    if kinetics_on_held_out and has_held_out:
        print(f"[kot] kinetics term on all {n} cells ({n_train} fitted + {n_held_out} "
              f"held out); alignment stays on the {n_train} fitted cells")
    if n_val > 0:
        held = ("held out of training" if val_holdout else "IN training (val is in-sample)")
        strat = f" | stratified by {val_stratify_by}" if val_stratify_by is not None else ""
        print(f"[kot] checkpoint-selection validation: {n_val}/{n - n_group_held_out} "
              f"fit-eligible cells, {held} | split_seed={val_split_seed}{strat} "
              f"digest={split_digest(rna_adata.obs_names, val_mask_np)}")
    else:
        print("[kot] checkpoint-selection validation: off (val_fraction=0)")
    if n_group_held_out > 0:
        print(f"[kot] OOD test holdout: {n_group_held_out}/{n} cells excluded from "
              "training and checkpoint selection"
              f" | fit_obs {fit_obs_key} in {list(cfg.get('fit_obs_values', []))}")
    if monitor_is_val and (n_val == 0 or not val_holdout):
        raise ValueError(
            "early_stopping_monitor=val_align needs fit-eligible validation cells held "
            "out from training: set val_fraction>0 and val_holdout_from_training=true, "
            "or monitor train_align. Group/OOD test cells cannot select a checkpoint."
        )
    # Optionally drop proteins whose linked gene has unusable RNA velocity: those
    # add noise to the JVP but no real dynamics signal. A protein stays active only
    # if at least one of its linked genes is a scVelo velocity_gene.
    if bool(cfg.get("kot_kinetics_require_velocity_gene", False)) and "velocity_genes" in rna_adata.var:
        v_genes = np.asarray(rna_adata.var["velocity_genes"].values, dtype=bool)   # (D_r,)
        has_stable = (S_np.astype(bool) & v_genes[None, :]).any(axis=1)
        dropped = int((kin_mask & ~has_stable).sum())
        kin_mask = kin_mask & has_stable
        print(f"[kot] velocity-gene filter: dropped {dropped} proteins with no stable-velocity gene")

    # RNA-side velocity weight (D_r,): which genes' velocity may enter J_φ·v. Masking the
    # protein outputs alone cannot stop off-target velocity leaking through the Jacobian,
    # which mixes all RNA coords. Two optional restrictions, combined multiplicatively:
    #  - kot_velocity_rna_reliability: keep only scVelo velocity_genes (fit reliability).
    #  - kot_velocity_s_linked_only:   keep only S-linked genes (those coding for a panel
    #    protein). The ODE's RHS (Sr) already uses only these, so restricting the LHS to
    #    them puts both sides on the same genes — the control for scVelo (sparse) vs
    #    RegVelo (dense) velocity, independent of how many genes each backend fills in.
    velocity_weight_np = build_velocity_weight(rna_adata, S_np, cfg, use_feature_space)
    if velocity_weight_np is not None:
        print(
            f"[kot] RNA-side velocity weight: {int(velocity_weight_np.sum())}/"
            f"{len(velocity_weight_np)} genes kept in JVP"
        )

    # Effective velocity actually fed to the JVP (see build_effective_velocity). Training
    # and diagnostics both receive this same V_eff_t; neither reconstructs it from raw V
    # plus a separate weight.
    V_eff = build_effective_velocity(V, velocity_weight_np, cfg)
    V_eff_t = to_tensor(V_eff, device)   # (n, D_r)

    # Velocity-input diagnostics. V_eff is fixed for the whole run, so this is computed once
    # and merged into every diagnostics JSON: the ‖v_eff‖ distribution and the fraction of
    # cells with any velocity left after the RNA-side weight, plus (when
    # kot_velocity_compare_h5ad is set) the per-cell cosine against the other backend's
    # velocity field, which is the only way to see how much of a scVelo-vs-RegVelo result
    # gap comes from the input rather than from KOT.
    velocity_diag = velocity_norm_diagnostics(V_eff)
    velocity_diag.update(
        compute_velocity_backend_agreement(rna_adata, V, cfg, velocity_weight_np))
    print(f"[kot] ‖v_eff‖ per cell: median={velocity_diag['v_eff_norm_median']:.4g} "
          f"mean={velocity_diag['v_eff_norm_mean']:.4g} "
          f"[q25={velocity_diag['v_eff_norm_q25']:.4g}, q75={velocity_diag['v_eff_norm_q75']:.4g}]  "
          f"nonzero={velocity_diag['v_eff_nonzero_frac'] * 100:.1f}% of {velocity_diag['v_eff_n_cells']} cells")

    # Kinetics mask (default on in feature space): only kinetic_mask=True proteins
    # enter L_dyn. When a mapping CSV is the source of truth, its use_for_kinetics
    # decisions are authoritative — the mask always comes from the CSV, and
    # kot_use_kinetics_mask cannot silently widen the residual back to all D_p.
    csv_source_of_truth = mapping_records is not None
    use_kinetics_mask = (
        bool(cfg.get("kot_use_kinetics_mask", True)) or csv_source_of_truth
    ) and use_feature_space
    if use_kinetics_mask:
        mask_t = to_tensor(kin_mask.astype(np.float32), device)   # (D_p,)
    else:
        mask_t = None
    mask_source = "mapping CSV (use_for_kinetics)" if csv_source_of_truth else "name/link matching"
    print(f"[kot] kinetics_mask={'on' if mask_t is not None else 'off'} (source: {mask_source}) "
          f"| active={int(kin_mask.sum()) if use_feature_space else D_p}/{D_p}")
    if use_feature_space and use_kinetics_mask and int(kin_mask.sum()) == 0:
        message = (
            "kinetic mask has 0 active proteins; no protein has a usable RNA gene "
            "link for the ODE residual."
        )
        if lambda_dyn != 0.0 and not allow_empty_kinetic_mask:
            raise ValueError(
                f"{message} Fix the mapping/RNA features, or set "
                "kot_allow_empty_kinetic_mask=true for an explicit "
                "alignment-only run."
            )
        if lambda_dyn != 0.0:
            print(f"[kot] WARNING: {message} Disabling dynamics for this run.")
            lambda_dyn = 0.0
        else:
            print(f"[kot] WARNING: {message} Dynamics already disabled.")
    # Alignment mask over the protein axis: Sinkhorn compares only use_for_alignment
    # proteins; φ still predicts the full panel. Per-protein only in feature space.
    if use_feature_space and not align_mask.all():
        align_np = np.nonzero(align_mask)[0]
        align_cols = torch.as_tensor(align_np, dtype=torch.long, device=device)
        print(f"[kot] alignment_mask on | {int(align_mask.sum())}/{D_p} proteins in Sinkhorn")
    else:
        align_np = None
        align_cols = None
    # Diagnostics on fitted cells only: an all-cell FOSCTTM uses held-out
    # proteins as distractors. S / masks are per-protein, not sliced.
    if train_idx_t is None:
        diag_inputs = DiagInputs(R_t, V_eff_t, P_t, S_t, rna_adata.obs, mask_t,
                                 align_cols, None)
    else:
        eval_rows = train_idx_t.cpu().numpy()
        diag_inputs = DiagInputs(R_t[train_idx_t], V_eff_t[train_idx_t],
                                 P_t[train_idx_t], S_t, rna_adata.obs.iloc[eval_rows],
                                 mask_t, align_cols, None)
        print(f"[kot] diagnostics evaluated on the {len(eval_rows)} fitted cells only")

    # Optional per-protein normalisation (off by default). The scale is floored at
    # the median std so it only DOWN-weights high-abundance proteins; without the
    # floor, near-constant proteins get a tiny std and their residual explodes.
    #
    # Measured on the FITTED cells. It is only D_p numbers, but they are a statistic of
    # the protein matrix and they enter the loss, so taking them over every cell would
    # let a held-out knockout's ADT reach training -- the one thing the protocol forbids
    # (src/evaluation/protocol.py).
    if bool(cfg.get("kot_kinetics_normalize", False)):
        std = (P_t if train_idx_t is None else P_t[train_idx_t]).std(dim=0)
        protein_scale = std.clamp(min=float(std.median()) + 1e-6)   # (D_p,)
    else:
        protein_scale = None
    print(f"[kot] kinetics_normalize={'on' if protein_scale is not None else 'off'}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = KOTModel(
        D_r, D_p,
        phi_dims=phi_dims,
        kappa_dims=kappa_dims,
        g_dims=g_dims,
        phi_init_gain=phi_init_gain,
        activation=activation,
        init_method=init_method,
        phi_spectral_norm=phi_spectral_norm,
        kappa_min=kappa_min,
        kappa_max=kappa_max,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
    ).to(device)
    if fixed_kappa_value is not None:
        fixed_kappa_value = float(fixed_kappa_value)
        model.kappa = FixedKappa(fixed_kappa_value).to(device)
        print(f"[kot] Fixed kappa enabled: {fixed_kappa_value:.6g}")
    else:
        fixed_kappa_value = None
    if fixed_alpha_value is not None:
        model.g = FixedAlpha(fixed_alpha_value, D_p).to(device)
        print(f"[kot] Fixed alpha enabled: {fixed_alpha_value}")
    fixed_beta_target = None
    if fixed_beta_value is not None:
        fixed_beta_target = set_fixed_beta(model, fixed_beta_value, D_p, device)
        print(f"[kot] Fixed beta enabled: {fixed_beta_target.detach().cpu().numpy()}")

    # Resolve the anchor set BEFORE the warm-start. The kinetic-mask intersection prunes
    # anchors that do not enter the ODE; doing it here means the warm-start only initializes
    # β for the anchors that actually survive (previously it warm-started every resolved
    # anchor, then the prior silently dropped the inactive ones — the init was left stale).
    anchor_idx_t = None
    anchor_target = None
    anchor_sigma_t = None
    anchor_weight_t = None
    # An anchor-free arm (kot_noanchor) never enters the block below, so a subset_n set on
    # its command line is inert. Recording the requested number anyway would put a k=N row
    # in a --group-by table for a run that had no anchors to thin -- report only what ran.
    anchor_subset_applied = False
    if use_anchor:
        if not anchor_indices:
            message = "use_anchor=true but no kot_anchor_indices are configured."
            if allow_empty_anchor_after_mask:
                print(f"[kot] WARNING: {message} Disabling β-anchor prior for this run.")
                use_anchor = False
            else:
                raise ValueError(f"{message} Disable use_anchor or provide anchors.")
        # Restrict the β prior to proteins ACTIVE in the kinetic mask (resolved anchors ∩
        # kinetic mask). β only enters the ODE for kinetic-active proteins, so anchoring
        # an inactive one does nothing — previously the full resolved set was used.
        if use_anchor and use_feature_space:
            n_from_csv = len(anchor_indices)
            keep = [k for k, j in enumerate(anchor_indices) if bool(kin_mask[j])]
            anchor_indices, anchor_betas, anchor_weights_list, anchor_sigmas_list = (
                select_anchor_entries(keep, anchor_indices, anchor_betas,
                                      anchor_weights_list, anchor_sigmas_list)
            )
            print(f"[kot] anchors: {n_from_csv} from csv, {n_from_csv - len(anchor_indices)} "
                  f"excluded (not in kinetic mask), {len(anchor_indices)} active priors")
            if not anchor_indices:
                message = "No anchored proteins fall in the kinetic mask; the β prior would be empty."
                if allow_empty_anchor_after_mask:
                    print(f"[kot] WARNING: {message} Disabling β-anchor prior for this run.")
                    use_anchor = False
                else:
                    raise ValueError(message)
        # Anchor-count ablation. After the kinetic-mask intersection so the count
        # asked for is the count that reaches the loss, and after resolve_beta_anchors
        # so every rung shares one β scale.
        if use_anchor and anchor_subset_n is not None:
            anchor_subset_applied = True
            n_active = len(anchor_indices)
            keep = anchor_subset_indices(n_active, anchor_subset_n, seed)
            anchor_indices, anchor_betas, anchor_weights_list, anchor_sigmas_list = (
                select_anchor_entries(keep, anchor_indices, anchor_betas,
                                      anchor_weights_list, anchor_sigmas_list)
            )
            kept_names = [second_adata.var_names[j] for j in anchor_indices]
            print(f"[kot] ANCHOR-COUNT ABLATION: kept {len(anchor_indices)}/{n_active} active "
                  f"anchors (subset seed={seed}): {kept_names}")
            if not anchor_indices:
                print("[kot] beta_anchor_subset_n=0 — β prior disabled for this run.")
                use_anchor = False
        if use_anchor:
            anchor_idx_t, anchor_target, anchor_sigma_t, anchor_weight_t = anchor_tensors(
                anchor_indices, anchor_betas, anchor_sigmas_list, anchor_weights_list,
                anchor_beta, prior_sigma_floor,
                float(cfg.get("beta_anchor_target", anchor_beta)), device,
            )

    # Warm-start β at the resolved anchor targets: β starts where the anchor wants
    # it (β = softplus(β_raw)+1e-6), stays trainable, and the soft prior holds it
    # there. From the default init the prior alone cannot travel that far.
    if beta_warmstart and use_anchor and fixed_beta_value is None and anchor_indices and anchor_betas:
        with torch.no_grad():
            ws_idx = torch.tensor(anchor_indices, dtype=torch.long, device=device)
            ws_tgt = torch.tensor(anchor_betas, dtype=torch.float32, device=device)
            model.beta_raw[ws_idx] = inverse_softplus((ws_tgt - 1e-6).clamp(min=1e-6))
        print(f"[kot] warm-started β at {len(anchor_indices)} anchor targets (β stays trainable)")

    initial_beta = model.beta.detach().cpu().numpy().copy()

    network_params = (
        list(model.phi.parameters())
        + list(model.kappa.parameters())
        + list(model.g.parameters())
    )
    optimiser, scheduler = build_kot_optimiser(model, lr, n_epochs, use_phases, cfg)

    # SR_t = S·r for every cell (fixed projection, computed once). V_eff_t (weight + gauge
    # already applied above) is the other per-cell quantity the kinetics term reuses each
    # epoch, so the loop only indexes into GPU tensors (no DataLoader, no per-batch S·r).
    SR_t = R_t @ S_t.T                                  # (n, D_p)
    kappa_target_t = (
        torch.tensor(kappa_prior_target, dtype=R_t.dtype, device=device)
        if kappa_prior_target is not None else None
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # Fixed probe batches for the alignment-vs-kinetics gradient-interaction diagnostic. Same
    # (RNA idx, unpaired protein idx) sets every diagnostic call, so ∇L_align / ∇L_dyn norms
    # and their cosine are comparable across epochs. On by default (kot_grad_interaction_diag);
    # the code default stays False so a config that predates the flag behaves as it did.
    grad_diag_on = bool(cfg.get("kot_grad_interaction_diag", False))
    probe_batches = []
    if grad_diag_on:
        probe_gen = torch.Generator(device=device).manual_seed(seed + 1)
        n_probe = int(cfg.get("kot_grad_interaction_batches", 4))
        probe_size = min(batch_size, n_train)
        for _ in range(n_probe):
            # Training rows only: the diagnostic describes the gradients the run takes,
            # and held-out cells take none.
            picks  = torch.randperm(n_train, generator=probe_gen, device=device)[:probe_size]
            ppicks = torch.randperm(n_train, generator=probe_gen, device=device)[:probe_size]
            idx  = picks  if train_idx_t is None else train_idx_t[picks]
            pidx = ppicks if train_idx_t is None else train_idx_t[ppicks]
            probe_batches.append((idx, pidx))
        print(f"[kot] grad-interaction diag: {n_probe} fixed batches of {probe_size} cells")
    grad_diag_nan = {"grad_align_norm": np.nan, "grad_dyn_norm": np.nan,
                     "grad_dyn_norm_scaled": np.nan, "grad_mag_ratio": np.nan,
                     "grad_mag_ratio_std": np.nan, "grad_lambda_dyn": np.nan,
                     "grad_cos_mean": np.nan, "grad_cos_std": np.nan,
                     "grad_align_norms": [], "grad_dyn_norms": [],
                     "grad_dyn_norms_scaled": [], "grad_mag_ratios": [], "grad_cosines": []}
    grad_interaction_rows = []
    # Same keys the JVP diagnostic returns, all NaN — used on the epochs between diagnostic
    # runs so every history column has one entry per epoch.
    jvp_diag_nan = {key: float("nan") for key in jvp_diag_keys()}

    print(f"[kot] Training {n_epochs} epochs | n={n} (align {n_train}, dyn {n_dyn}, "
          f"val {n_val}) | "
          f"D_r={D_r} | D_p={D_p} | device={device}")

    print(f"[kot] model: {model_name}")
    print(f"[kot] bounds: κ∈{(kappa_min, kappa_max)}  α∈{(alpha_min, alpha_max)}")
    print(f"[kot] κ-prior: target={kappa_prior_target} λ_κ={lambda_kappa_prior}  |  "
          f"φ_spectral_norm={phi_spectral_norm}  g_freeze={g_freeze_epochs}  "
          f"dyn_warmup={dyn_warmup_epochs}  early_stop_patience={early_stopping_patience}"
          f" on {early_stopping_monitor}")
    print(f"[kot] lr: base={lr:.2e}  φ={lr_phi or 'base'}  α/κ={lr_alpha_kappa or 'base'}  "
          f"β={lr_beta or 'base'}  |  warmup={lr_warmup_epochs}ep "
          f"×{lr_warmup_start_factor}→1→×{lr_min_factor} (cosine)")
    if any(v is not None for v in (fixed_kappa_value, fixed_alpha_value, fixed_beta_value)):
        print(f"[kot] fixed params: κ={fixed_kappa_value}  α={fixed_alpha_value}  β={fixed_beta_value}")
    print(f"[kot] use_anchor: {use_anchor}")
    if use_anchor:
        protein_names = list(second_adata.var_names)
        targets = anchor_target.cpu().numpy()
        sigmas_np = anchor_sigma_t.cpu().numpy()
        print(f"[kot]   anchor source: {anchor_source}")
        print(
            f"[kot]   anchoring {len(anchor_indices)} proteins "
            f"(prior={beta_anchor_prior}, λ_prior={lambda_prior}, "
            f"mean β*={float(targets.mean()):.3f}):"
        )
        for j, b, s in zip(anchor_indices, targets, sigmas_np):
            print(f"[kot]     [{j:>3}] {protein_names[j]:<18} β*={b:.4f}  σ={s:.4f}")
    else:
        print(f"[kot]   anchor disabled (λ_prior={lambda_prior})")
    print(f"[kot] initial β: {initial_beta}")

    # History columns are derived from the diagnostic helpers rather than listed by hand,
    # so adding a statistic in one place cannot leave the CSV missing a column.
    history_beta_keys = [f"beta_{stat}" for stat in STAT_SUFFIXES]
    history_diag_keys = jvp_diag_keys()
    history_grad_keys = [
        "grad_align_norm", "grad_dyn_norm", "grad_dyn_norm_scaled",
        "grad_mag_ratio", "grad_mag_ratio_std", "grad_cos_mean", "grad_cos_std",
    ]
    loss_history = {
        "epoch": [], "align": [], "val_align": [], "dyn": [], "anchor": [],
        "kappa_prior": [], "reg": [], "total": [], "lambda_dyn_eff": [], "lr_factor": [],
    }
    loss_history["grad_norm"] = []
    for key in history_beta_keys + history_diag_keys + history_grad_keys:
        loss_history[key] = []

    best_align = float("inf")
    best_align_epoch = 0
    best_align_state = None
    best_align_losses = None

    best_val_align = float("inf")
    best_val_align_epoch = 0
    best_val_align_state = None
    best_val_align_losses = None

    best_dyn = float("inf")
    best_dyn_epoch = 0
    best_dyn_state = None
    best_dyn_losses = None

    best_total = float("inf")
    best_total_epoch = 0
    best_total_state = None
    best_total_losses = None

    epochs_since_improvement = 0
    stop_epoch = n_epochs
    current_phase = 0   # for use_phases mode; tracks transitions

    # phi at initialisation, before any gradient step. This is the only honest
    # "before alignment" view on real data: RNA features and protein features have
    # no shared coordinate system, so the two modalities cannot be co-embedded
    # until phi maps one into the other's space. phi_init(R) and the observed
    # protein DO share that space, so a before/after pair drawn from them is a
    # like-for-like comparison. One forward pass, so it costs nothing.
    #
    # Sliced to the alignment panel exactly as aligned_rna.npy is (see where
    # `aligned` is built from align_np below), so the before/after pair spans the
    # same protein columns. Without the slice the two arrays have different widths
    # on every dataset that sets use_for_alignment.
    if output_dir is not None:
        phi_init = phi_output(model, R_t)
        if align_np is not None:
            phi_init = phi_init[:, align_np]
        np.save(Path(output_dir) / "phi_init.npy", phi_init)
        print(f"[kot] Saved untrained phi output: {Path(output_dir) / 'phi_init.npy'} "
              f"{phi_init.shape}")

    train_start = time.perf_counter()
    for epoch in range(1, n_epochs + 1):
        model.train()

        # --- Phase management (use_phases) or dyn_warmup (joint mode) ---
        if use_phases:
            if epoch <= phase1_epochs:
                new_phase = 1
            elif epoch <= phase1_epochs + phase2_epochs:
                new_phase = 2
            else:
                new_phase = 3
            if new_phase != current_phase:
                current_phase = new_phase
                phi_train = current_phase in (1, 3)
                dyn_train = current_phase in (2, 3)
                for p in model.phi.parameters():   p.requires_grad = phi_train
                for p in model.kappa.parameters(): p.requires_grad = dyn_train
                for p in model.g.parameters():     p.requires_grad = dyn_train
                model.beta_raw.requires_grad = dyn_train and fixed_beta_value is None
                if current_phase == 3:
                    # Each head's own base_lr, so phase 3 preserves the ratios between
                    # the per-head rates instead of flattening all three onto `lr`.
                    for pg in optimiser.param_groups:
                        pg["lr"] = pg["base_lr"] * phase3_lr_scale
                print(f"[kot] >>> Phase {current_phase} at epoch {epoch} "
                      f"(phi={'train' if phi_train else 'frozen'}, "
                      f"κ/α/β={'train' if dyn_train else 'frozen'}, "
                      f"{format_group_lrs(optimiser, phase_lr_factor(optimiser))})")
            lam_align_eff = 1.0        if current_phase in (1, 3) else 0.0
            lam_dyn_eff   = lambda_dyn if current_phase in (2, 3) else 0.0
        else:
            lam_align_eff = 1.0
            lam_dyn_eff = lambda_dyn * dyn_weight_factor(epoch, dyn_warmup_epochs)
        if not use_phases and g_freeze_epochs > 0:
            g_train = epoch > g_freeze_epochs
            for p in model.g.parameters():
                p.requires_grad = g_train
            if epoch == 1 or epoch == g_freeze_epochs + 1:
                state = "trainable" if g_train else "frozen"
                print(f"[kot] g/alpha head is {state} at epoch {epoch}")

        optimiser.zero_grad(set_to_none=True)

        # --- Sinkhorn alignment step (differentiable through phi) ---
        align_val = 0.0
        if lam_align_eff > 0.0:
            align_val = alignment_step(
                model, R_t, P_t, n_train, device, sinkhorn_max_points, align_cols,
                blur, sinkhorn_backend, lam_align_eff, train_idx_t,
            )

        # --- Kinetics step + kappa prior (per-batch, gradients accumulated) ---
        # dyn_idx_t, not train_idx_t: the ODE residual reads no observed protein, so it
        # may govern φ on held-out cells too (see where dyn_idx_t is built).
        dyn_val, kappa_prior_val = kinetics_step(
            model, R_t, V_eff_t, SR_t, conf_t, n_dyn, batch_size, device,
            mask_t, protein_scale, lam_dyn_eff, kappa_target_t, lambda_kappa_prior,
            dyn_idx_t,
        )

        # --- β anchor prior (on β directly; only when β is trainable and weighted) ---
        anchor_val = 0.0
        if anchor_target is not None and model.beta_raw.requires_grad and lambda_prior > 0.0:
            loss_anchor = beta_anchor_prior_loss(
                model.beta, anchor_idx_t, anchor_target,
                anchor_sigma_t, anchor_weight_t, lambda_prior, beta_anchor_prior,
            )
            loss_anchor.backward()
            anchor_val = loss_anchor.item()

        # --- weight decay on the network parameters (skip the backward when off) ---
        reg_val = 0.0
        if lambda_reg > 0.0:
            loss_reg = lambda_reg * sum(param.pow(2).sum() for param in network_params)
            loss_reg.backward()
            reg_val = loss_reg.item()

        total_val = (lam_align_eff * align_val + lam_dyn_eff * dyn_val
                     + anchor_val + kappa_prior_val + reg_val)

        # One clip over every head. Per-head budgets added knobs without changing
        # recovery. The return is the PRE-clip norm: it shows the clip binds
        # every step rather than catching spikes.
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0))
        # Read BEFORE the scheduler advances: this is the factor the step above used,
        # and logging the post-step value would report the next epoch's rate instead.
        lr_factor = phase_lr_factor(optimiser)
        optimiser.step()
        if scheduler is not None:
            scheduler.step()
        if epoch == 1 and device.type == "cuda":
            print(f"[kot] GPU memory after epoch 1: "
                  f"allocated {torch.cuda.memory_allocated(device) / 1e9:.2f} GB, "
                  f"peak {torch.cuda.max_memory_allocated(device) / 1e9:.2f} GB "
                  f"(batch_size={batch_size}, n={n})")

        beta_detached = model.beta.detach()
        # β is a (D_p,) parameter, so the full distribution costs nothing to summarise and
        # is logged every epoch — unlike the JVP block, which only runs periodically.
        beta_stats = summarize_distribution(beta_detached.cpu().numpy(), "beta")
        beta_mean_val = beta_stats["beta_mean"]
        beta_std_val  = beta_stats["beta_std"]

        # Periodic JVP-based diagnostics: every 50 epochs (and epoch 1) on a 512-cell subset.
        if epoch % 50 == 0 or epoch == 1:
            diag = compute_jvp_rhs_diagnostics(
                model,
                R_t,
                V_eff_t,
                S_t,
                subset=512,
                mask=mask_t,
            )
            if grad_diag_on:
                gi = compute_grad_interaction(
                    model, probe_batches, R_t, V_eff_t, SR_t, P_t, conf_t,
                    blur, sinkhorn_backend, mask=mask_t, scale=protein_scale,
                    align_cols=align_cols,
                    # The λ actually in force this epoch, so the scaled norm and the
                    # magnitude ratio describe the real update, not the configured weight.
                    lambda_dyn=lam_dyn_eff,
                )
                for b, (na, nd, nds, ratio, c) in enumerate(zip(
                        gi["grad_align_norms"], gi["grad_dyn_norms"],
                        gi["grad_dyn_norms_scaled"], gi["grad_mag_ratios"],
                        gi["grad_cosines"])):
                    grad_interaction_rows.append({
                        "epoch": epoch, "batch": b,
                        "grad_align_norm": na, "grad_dyn_norm": nd,
                        "grad_dyn_norm_scaled": nds, "grad_mag_ratio": ratio,
                        "lambda_dyn_eff": float(lam_dyn_eff), "grad_cos": c,
                    })
            else:
                gi = grad_diag_nan
        else:
            diag = dict(jvp_diag_nan)
            gi = grad_diag_nan

        # Held-out Sinkhorn: the same estimator as `align`, on cells no loss term saw.
        # The gap between the two curves is what says φ has started fitting the training
        # cells rather than the geometry. NaN in between, so every column has one entry
        # per epoch; the last epoch is always measured so the final state is comparable.
        if val_idx_t is not None and (
            epoch % val_every == 0 or epoch == 1 or epoch == n_epochs
        ):
            val_align_val = validation_align_loss(
                model, R_t, P_t, val_idx_t, device, sinkhorn_max_points,
                align_cols, blur, sinkhorn_backend,
            )
        else:
            val_align_val = float("nan")

        loss_history["epoch"].append(epoch)
        loss_history["align"].append(align_val)
        loss_history["val_align"].append(val_align_val)
        loss_history["dyn"].append(dyn_val)
        loss_history["anchor"].append(anchor_val)
        loss_history["kappa_prior"].append(kappa_prior_val)
        loss_history["reg"].append(reg_val)
        loss_history["total"].append(total_val)
        loss_history["lambda_dyn_eff"].append(float(lam_dyn_eff))
        loss_history["lr_factor"].append(float(lr_factor))
        loss_history["grad_norm"].append(grad_norm)
        for key in history_beta_keys:
            loss_history[key].append(beta_stats[key])
        for key in history_diag_keys:
            loss_history[key].append(diag[key])
        for key in history_grad_keys:
            loss_history[key].append(gi[key])

        if epoch % 50 == 0 or epoch == 1:
            # The held-out Sinkhorn rides next to the training one, so the gap between
            # them is readable in the log itself rather than only in training_loss.csv.
            # Empty on an epoch the val check did not land on (val_every > 1).
            val_text = "" if not math.isfinite(val_align_val) else f"  val={val_align_val:.6f}"
            print(f"[kot] epoch {epoch:4d}  align={align_val:.6f}{val_text}  "
                  f"dyn={dyn_val:.6f}  anchor={anchor_val:.6f}  "
                  f"kprior={kappa_prior_val:.6f}  "
                  f"λ_dyn={lam_dyn_eff:.3f}  {format_group_lrs(optimiser, lr_factor)}  "
                  f"β={beta_mean_val:.3f}±{beta_std_val:.3f}  "
                  f"κ={diag['kappa_mean']:.3f}±{diag['kappa_std']:.3f} "
                  f"[{diag['kappa_min']:.3f},{diag['kappa_max']:.3f}]  "
                  f"α={diag['alpha_mean']:.3f} "
                  f"[{diag['alpha_min']:.3f},{diag['alpha_max']:.3f}]  "
                  f"cos={diag['jvp_rhs_cos']:.3f}  total={total_val:.6f}  "
                  f"[{time.perf_counter() - train_start:.0f}s, {epoch / (time.perf_counter() - train_start):.1f} ep/s]")
            # Medians and the IQR say whether the means above describe the typical cell or
            # are being dragged by a tail — the reason these distributions are logged at all.
            print(f"[kot]   physics (median [q25,q75]): "
                  f"cos={diag['jvp_rhs_cos_median']:+.3f} "
                  f"[{diag['jvp_rhs_cos_q25']:+.3f},{diag['jvp_rhs_cos_q75']:+.3f}]  "
                  f"|JVP|={diag['jvp_norm_median']:.3e}  |RHS|={diag['rhs_norm_median']:.3e}  "
                  f"ratio={diag['norm_ratio_median']:.3f} "
                  f"[{diag['norm_ratio_q25']:.3f},{diag['norm_ratio_q75']:.3f}]  "
                  f"logratio={diag['log_norm_ratio_median']:+.3f}  "
                  f"relres={diag['rel_residual_median']:.3f} "
                  f"[{diag['rel_residual_q25']:.3f},{diag['rel_residual_q75']:.3f}]")
            if grad_diag_on:
                print(f"[kot]   grad-interaction: ||∇align||={gi['grad_align_norm']:.3e}  "
                      f"||∇dyn||={gi['grad_dyn_norm']:.3e}  "
                      f"λ||∇dyn||={gi['grad_dyn_norm_scaled']:.3e}  "
                      f"ratio={gi['grad_mag_ratio']:.3e}±{gi['grad_mag_ratio_std']:.1e}  "
                      f"cos(∇align,∇dyn)={gi['grad_cos_mean']:+.3f}±{gi['grad_cos_std']:.3f}")

        # Track minima for all three checkpoints.
        current_align = align_val
        current_dyn   = dyn_val
        current_total = total_val
        current_losses = {
            "align":  current_align,
            "val_align": val_align_val,
            "dyn":    current_dyn,
            "anchor": anchor_val,
            "kappa_prior": kappa_prior_val,
            "total":  current_total,
        }

        if lam_dyn_eff > 0.0 and current_dyn < best_dyn:
            best_dyn = current_dyn
            best_dyn_epoch = epoch
            best_dyn_state = clone_state(model)
            best_dyn_losses = dict(current_losses)

        if current_total < best_total:
            best_total = current_total
            best_total_epoch = epoch
            best_total_state = clone_state(model)
            best_total_losses = dict(current_losses)

        # Early stopping: only monitor epochs where BOTH terms are at full strength.
        # In phased runs, the dynamics-only phase sets lam_align_eff=0 and must not
        # become "best_align" simply because align_val is 0. In joint runs the mirror
        # trap: while the kinetics weight is still ramping, the align loss is the lowest
        # it will ever be precisely because nothing is pulling against it, so an
        # unguarded monitor would checkpoint inside the ramp and hand back a model with
        # far less dynamics than it trained with -- one that ranks WELL on FOSCTTM.
        monitor_align = lam_align_eff > 0.0 and (use_phases or epoch > dyn_warmup_epochs)
        if monitor_align:
            if current_align < best_align:
                best_align = current_align
                best_align_epoch = epoch
                best_align_state = clone_state(model)
                best_align_losses = dict(current_losses)
            # The held-out twin of best_align. Recorded only on epochs where the val
            # Sinkhorn was actually measured (NaN compares false, so the guard is the
            # comparison itself), and it is the checkpoint that can differ from `final`
            # -- the training align loss is still falling at the end of a 500-epoch run,
            # so best_align almost never does.
            if val_align_val < best_val_align:
                best_val_align = val_align_val
                best_val_align_epoch = epoch
                best_val_align_state = clone_state(model)
                best_val_align_losses = dict(current_losses)
            # Patience counts epochs since the CONFIGURED signal last improved. val_every
            # is 1 whenever val_align is the criterion, so the unit is epochs either way.
            improved_epoch = best_val_align_epoch if monitor_is_val else best_align_epoch
            if improved_epoch == epoch:
                epochs_since_improvement = 0
            elif not use_phases or current_phase == 3:
                epochs_since_improvement += 1
                if epochs_since_improvement >= early_stopping_patience:
                    stop_epoch = epoch
                    best_value = best_val_align if monitor_is_val else best_align
                    print(f"[kot] Early stopping at epoch {epoch}: best "
                          f"{early_stopping_monitor}={best_value:.6f} at epoch "
                          f"{improved_epoch}")
                    break

    # Snapshot the literal final state before any restoration.
    final_state  = clone_state(model)
    final_losses = dict(current_losses)

    # Uniform checkpoint table: (name, state, epoch, losses). best_val_align is the
    # only entry that can be absent (no validation split), and write_run_artifacts
    # skips a None state, so nothing downstream has to special-case it.
    checkpoints = [
        ("best_align", best_align_state, best_align_epoch, best_align_losses),
        ("best_val_align", best_val_align_state, best_val_align_epoch, best_val_align_losses),
        ("best_dyn",   best_dyn_state,   best_dyn_epoch,   best_dyn_losses),
        ("best_total", best_total_state, best_total_epoch, best_total_losses),
        ("final",      final_state,      stop_epoch,       final_losses),
    ]

    print("[kot] best checkpoints:")
    for name, state, ep, lo in checkpoints:
        if state is None:
            print(f"[kot]   {name:>10s}: (not recorded)")
        else:
            print(f"[kot]   {name:>10s}: epoch {ep:4d}  "
                  f"align={lo['align']:.6f}  dyn={lo['dyn']:.6f}  total={lo['total']:.6f}")

    if best_align_state is not None:
        model.load_state_dict(best_align_state)
        print(f"[kot] restored best_align state from epoch {best_align_epoch} (align={best_align:.6f})")

    # Final aligned representations
    model.eval()
    with torch.no_grad():
        phi_final = phi_output(model, R_t)              # (n, D_p)
        p_np = y                                         # (n, D_p) observed
        loss_anchor_final = (
            beta_anchor_prior_loss(
                model.beta,
                anchor_idx_t,
                anchor_target,
                anchor_sigma_t,
                anchor_weight_t,
                lambda_prior,
                beta_anchor_prior,
            )
            if anchor_target is not None
            else torch.tensor(0.0, device=device)
        )

    # End-of-training diagnostics: time, branch confusion, variance, JVP/RHS.
    diagnostics = compute_full_diagnostics(model, diag_inputs)
    # Snapshot the raw best_align full-diagnostics (before the anchor augmentation below) so
    # the per-checkpoint loop reuses them for best_align — the model is on the same restored
    # best_align state, so recomputing the JVP + NN match there is pure waste.
    best_align_full = dict(diagnostics)
    # Velocity-input stats (‖v_eff‖ distribution, non-zero fraction, cross-backend cosine).
    # compute_full_diagnostics already fills the ‖v_eff‖ half from the tensor it is given;
    # this adds the backend comparison, which needs the AnnData and so lives out here.
    diagnostics.update(velocity_diag)
    best_align_full.update(velocity_diag)
    diagnostics["anchor_loss_final"] = float(loss_anchor_final.item())
    diagnostics.update(anchor_diagnostics(
        model.beta, use_anchor, anchor_indices, anchor_target, kin_mask,
        list(second_adata.var_names), beta_anchor_prior, beta_anchor_aggregate, anchor_scale,
        anchor_subset_n if anchor_subset_applied else None,
    ))
    diagnostics["stop_epoch"]        = int(stop_epoch)
    diagnostics["best_align_epoch"]  = int(best_align_epoch) if best_align_state is not None else None
    diagnostics["best_align"]        = float(best_align) if best_align_state is not None else None
    diagnostics["best_val_align_epoch"] = (
        int(best_val_align_epoch) if best_val_align_state is not None else None
    )
    diagnostics["best_val_align"]    = (
        float(best_val_align) if best_val_align_state is not None else None
    )
    diagnostics["val_fraction"]      = val_fraction
    diagnostics["val_holdout"]       = val_holdout
    diagnostics["val_split_seed"]    = val_split_seed
    diagnostics["val_split_digest"]  = (
        split_digest(rna_adata.obs_names, val_mask_np) if n_val > 0 else None
    )
    diagnostics["group_holdout_digest"] = (
        split_digest(rna_adata.obs_names, group_held_out_np)
        if n_group_held_out > 0 else None
    )
    # The RESOLVED seed, not the configured base: with val_split_per_seed on they
    # differ, and the resolved one is what the digest can be reproduced from.
    diagnostics["val_split_seed_resolved"] = int(val_split_seed) if n_val > 0 else None
    diagnostics["val_stratify_by"]   = val_stratify_by if n_val > 0 else None
    diagnostics["fit_obs_key"]       = fit_obs_key
    diagnostics["fit_obs_values"]    = (
        list(cfg.get("fit_obs_values", [])) if fit_obs_key is not None else None
    )
    # The split's own counts, as opposed to validation_metrics' train_n_cells /
    # val_n_cells, which count only what a FOSCTTM was measured over.
    diagnostics["n_train_cells"]     = int(n_train)
    diagnostics["n_held_out_cells"]  = int(n_held_out)
    diagnostics["n_reported_cells"]  = int(n_val)
    diagnostics["n_group_holdout_cells"] = int(n_group_held_out)
    # The alignment term saw n_train cells; the kinetics term saw n_dyn. They differ only
    # under kot_kinetics_on_held_out, and a run cannot be interpreted without knowing it.
    diagnostics["kot_kinetics_on_held_out"] = bool(kinetics_on_held_out)
    diagnostics["n_kinetics_cells"]  = int(n_dyn)
    diagnostics["early_stopping_monitor"] = early_stopping_monitor
    diagnostics["best_dyn_epoch"]    = int(best_dyn_epoch) if best_dyn_state is not None else None
    diagnostics["best_dyn"]          = float(best_dyn) if best_dyn_state is not None else None
    diagnostics["best_total_epoch"]  = int(best_total_epoch) if best_total_state is not None else None
    diagnostics["best_total"]        = float(best_total) if best_total_state is not None else None
    diagnostics["fixed_kappa"]       = fixed_kappa_value is not None
    diagnostics["fixed_kappa_value"] = (
        float(fixed_kappa_value) if fixed_kappa_value is not None else None
    )
    diagnostics["fixed_alpha"]       = fixed_alpha_value is not None
    diagnostics["fixed_alpha_value"] = fixed_alpha_value
    diagnostics["fixed_beta"]        = fixed_beta_value is not None
    diagnostics["fixed_beta_value"]  = fixed_beta_value
    diagnostics["model_name"]        = model_name
    diagnostics["seed"]              = int(seed)
    diagnostics["kappa_bounds"]      = [kappa_min, kappa_max]
    diagnostics["alpha_bounds"]      = [alpha_min, alpha_max]
    diagnostics["lambda_kappa_prior"] = float(lambda_kappa_prior)
    diagnostics["kappa_prior_target"] = kappa_prior_target
    diagnostics["g_freeze_epochs"]   = int(g_freeze_epochs)
    diagnostics["dyn_warmup_epochs"] = int(dyn_warmup_epochs)
    diagnostics["lr_phi"]            = lr_phi
    diagnostics["lr_alpha_kappa"]    = lr_alpha_kappa
    diagnostics["lr_beta"]           = lr_beta
    diagnostics["lr_warmup_epochs"]  = int(lr_warmup_epochs)
    diagnostics["lr_warmup_start_factor"] = float(lr_warmup_start_factor)
    diagnostics["lr_min_factor"]     = float(lr_min_factor)
    # Median pre-clip gradient norm. The clip binds every step and acts as a
    # renormaliser — worth knowing, not tunable.
    diagnostics["grad_norm_median"] = float(np.median(loss_history["grad_norm"]))
    # Gradient balance, summarised out of grad_interaction.csv so a sweep can rank on it.
    # grad_mag_ratio = ||lambda_dyn*grad_dyn|| / ||grad_align||: because the clip binds on
    # every step, lambda_dyn is a MIXING coefficient, not a scale -- it decides how a
    # fixed-norm step is split between the two terms, and this ratio is that split
    # measured directly. Near 1 means balanced; <<1 means the kinetics term is along for
    # the ride. grad_cos is the angle between the two: strongly negative means they are
    # fighting, so a FOSCTTM win at high lambda_dyn was bought against alignment.
    # Taken over the last third of the logged epochs -- the early ones are dominated by
    # initialisation and by the lambda_dyn ramp still being partway up.
    if grad_interaction_rows:
        epochs = sorted({row["epoch"] for row in grad_interaction_rows})
        tail = set(epochs[len(epochs) * 2 // 3:])
        late = [row for row in grad_interaction_rows if row["epoch"] in tail]
        diagnostics["grad_mag_ratio_late"] = float(np.median([r["grad_mag_ratio"] for r in late]))
        diagnostics["grad_cos_late"] = float(np.median([r["grad_cos"] for r in late]))
    diagnostics["data_debug"]        = data_debug

    print_end_of_training(
        diagnostics, model.beta, use_anchor, anchor_target,
        float(loss_anchor_final.item()), stop_epoch, n_epochs,
    )


    if output_dir is not None:
        write_run_artifacts(
            output_dir, diagnostics, best_align_full, velocity_diag, checkpoints,
            best_align_state, model, diag_inputs, model_name, seed, grad_interaction_rows,
            protein_target_units(cfg, protein_layer),
        )

    # Benchmark FOSCTTM is computed on `aligned`. Restrict it to the alignment panel
    # (use_for_alignment proteins) so the benchmark is consistent with training and
    # with the other methods — features excluded from the Sinkhorn must not re-enter
    # evaluation. The full-panel φ prediction is saved separately for downstream use.
    if align_np is not None:
        if output_dir is not None:
            np.save(Path(output_dir) / "phi_full_panel.npy", phi_final)
            np.save(Path(output_dir) / "protein_full_panel.npy", p_np)
        aligned = [phi_final[:, align_np], p_np[:, align_np]]
    else:
        aligned = [phi_final, p_np]
    return aligned, None, pd.DataFrame(loss_history)
