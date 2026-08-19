"""
KOT training loop: Sinkhorn alignment plus kinetic ODE optimisation.

The objective follows the paper terms:
  L = L_sinkhorn + lambda_dyn * L_kinetics + lambda_prior * L_anchor
      + lambda_reg * L_reg,
with L_anchor active only when an explicit anchor set H is configured.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.func import jvp as torch_jvp

from src.utils.arrays import to_dense
from src.models.KOT import KOTModel
from src.losses.sinkhorn import sinkhorn_divergence
from src.losses.jvp_physics import kinetics_loss
from src.data.projection import projection_matrix_from_adatas
from src.data.adt_gene_map import load_mapping_records
from src.data.beta_anchor import resolve_beta_anchors
from src.evaluation.foscttm import calc_domainAveraged_FOSCTTM
from src.evaluation.trajectory_dtw import trajectory_dtw


def to_tensor(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(arr, dtype=torch.float32, device=device)


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
    """Soft β prior: l2 | gaussian | log_normal (supervisor: not hard-fixed).

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


def build_kot_optimiser(model, lr: float, n_epochs: int, use_phases: bool):
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = None if use_phases else torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=n_epochs,
    )
    return optimiser, scheduler


def matrix_from_adata(adata, layer_key: str | None, label: str) -> np.ndarray:
    if layer_key in (None, "", "X"):
        return to_dense(adata.X, np.float32)
    if layer_key not in adata.layers:
        available = list(adata.layers.keys())
        raise ValueError(f"KOT {label} layer '{layer_key}' not found. Available layers: {available}")
    return to_dense(adata.layers[layer_key], np.float32)


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


def compute_jvp_rhs_diagnostics(
    model,
    R_t: torch.Tensor,
    V_eff: torch.Tensor,
    S_t: torch.Tensor,
    subset: int = 512,
    mask: torch.Tensor | None = None,
) -> dict:
    """JVP/RHS cosine + norms and kappa stats on a small subset.

    `V_eff` is the exact velocity fed to the JVP in training (RNA-side weight applied and
    gauge-normalized upstream); this receives it directly so the diagnostic uses the same
    vector as the loss. JVP requires autograd (forward-mode AD): do not call inside
    torch.no_grad(). When mask is given, the cosine/norms are over the mapped (kinetics)
    proteins only, so unmapped proteins do not dilute the physics fit.
    """
    was_training = model.training
    model.eval()

    m = min(subset, R_t.shape[0])
    r = R_t[:m]
    v = V_eff[:m]

    phi_r, dphi_dv = torch_jvp(model.phi, (r,), (v,))
    kappa = model.kappa(r)
    alpha = model.g(r)
    Sr = (S_t @ r.T).T
    beta = model.beta
    rhs = kappa * (alpha * Sr - beta * phi_r)

    if mask is not None:
        dphi_dv = dphi_dv * mask
        rhs = rhs * mask
    cos = F.cosine_similarity(dphi_dv, rhs, dim=1, eps=1e-8)

    out = {
        "jvp_rhs_cos": float(cos.mean().detach().cpu()),
        "jvp_norm":    float(dphi_dv.norm(dim=1).mean().detach().cpu()),
        "rhs_norm":    float(rhs.norm(dim=1).mean().detach().cpu()),
        "kappa_mean":  float(kappa.mean().detach().cpu()),
        "kappa_std":   float(kappa.std(unbiased=False).detach().cpu()),
        "kappa_min":   float(kappa.min().detach().cpu()),
        "kappa_max":   float(kappa.max().detach().cpu()),
        "alpha_mean":  float(alpha.mean().detach().cpu()),
        "alpha_std":   float(alpha.std(unbiased=False).detach().cpu()),
        "alpha_min":   float(alpha.min().detach().cpu()),
        "alpha_max":   float(alpha.max().detach().cpu()),
    }

    if was_training:
        model.train()
    return out


def resolve_pseudotime(rna_obs: pd.DataFrame) -> tuple[np.ndarray | None, str | None]:
    """Pseudotime coordinate for a run.

    Prefers ground-truth 'time' (synthetic); falls back to scVelo
    'velocity_pseudotime' / 'latent_time' on real datasets. Returns (values, key),
    or (None, None) when no temporal coordinate is available.
    """
    for key in ("time", "velocity_pseudotime", "latent_time"):
        if key in rna_obs.columns:
            return np.asarray(rna_obs[key].values, dtype=np.float64), key
    return None, None


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
) -> dict:
    """Per-batch interaction between the alignment and kinetics objectives on the shared φ.

    φ is the only parameter block both losses touch (alignment via the Sinkhorn on φ(r); the
    kinetics residual via J_φ·v and β⊙φ). For each FIXED probe batch we take ∇L_align and
    ∇L_dyn w.r.t. the φ parameters and report each gradient's L2 norm plus the cosine between
    them: cos < 0 means the two objectives pull φ in opposing directions (gradient conflict),
    cos > 0 means they cooperate; the norms show which objective dominates φ's update. Raw
    (un-λ-weighted) losses, so the norms reflect the intrinsic gradient magnitudes. Run in
    eval() so the extra forwards do not perturb φ's spectral-norm estimate.
    """
    was_training = model.training
    model.eval()
    phi_params = [p for p in model.phi.parameters() if p.requires_grad]

    def flat(grads):
        return torch.cat([
            (g if g is not None else torch.zeros_like(p)).reshape(-1)
            for g, p in zip(grads, phi_params)
        ])

    align_norms, dyn_norms, cosines = [], [], []
    for idx, pidx in probe_batches:
        phi_r = model.phi(R_t[idx])
        phi_a = phi_r if align_cols is None else phi_r[:, align_cols]
        P_a = (P_t[pidx] if align_cols is None else P_t[pidx][:, align_cols])
        loss_align = sinkhorn_divergence(phi_a, P_a, blur=blur, backend=backend)
        g_align = flat(torch.autograd.grad(loss_align, phi_params, allow_unused=True))

        loss_dyn, _ = kinetics_loss(
            model.phi, model.kappa, model.g,
            R_t[idx], V_eff_t[idx], SR_t[idx], model.beta, conf_t[idx],
            mask=mask, scale=scale,
        )
        g_dyn = flat(torch.autograd.grad(loss_dyn, phi_params, allow_unused=True))

        na = float(g_align.norm())
        nd = float(g_dyn.norm())
        align_norms.append(na)
        dyn_norms.append(nd)
        cosines.append(float((g_align @ g_dyn) / (g_align.norm() * g_dyn.norm() + 1e-12)))

    if was_training:
        model.train()
    return {
        "grad_align_norm": float(np.mean(align_norms)),
        "grad_dyn_norm":   float(np.mean(dyn_norms)),
        "grad_cos_mean":   float(np.mean(cosines)),
        "grad_cos_std":    float(np.std(cosines)),
        "grad_align_norms": align_norms,
        "grad_dyn_norms":   dyn_norms,
        "grad_cosines":     cosines,
    }


def nearest_protein_match(
    phi: torch.Tensor, protein: torch.Tensor, chunk: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Row-chunked nearest neighbour of each φ(r) in protein space.

    Returns (argmin index, min distance) per query row without ever materialising
    the full (n, m) distance matrix, so large-n datasets (e.g. bmmc_cite at ~90k
    cells, where a dense n×n matrix is ~32 GB) do not OOM.
    """
    n = phi.shape[0]
    idx = torch.empty(n, dtype=torch.long, device=phi.device)
    vals = torch.empty(n, dtype=phi.dtype, device=phi.device)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        mins = torch.cdist(phi[start:end], protein).min(dim=1)
        idx[start:end] = mins.indices
        vals[start:end] = mins.values
    return idx, vals


def compute_per_cell_diagnostics(
    model,
    R_t: torch.Tensor,
    V_eff: torch.Tensor,
    P_t: torch.Tensor,
    S_t: torch.Tensor,
    rna_obs: pd.DataFrame,
    mask: torch.Tensor | None = None,
    align_cols: torch.Tensor | None = None,
) -> dict:
    """Per-cell arrays for FOSCTTM-correlated diagnostic plots.

    FOSCTTM and the nearest-neighbour matching are computed on the alignment panel
    (align_cols, e.g. use_for_alignment proteins) so features excluded from the
    Sinkhorn (isotype controls) do not re-enter evaluation. φ still predicts the
    full panel; only the metric is restricted.
    """
    model.eval()

    # All-cell JVP and RHS (single forward-mode pass) on the exact training velocity.
    phi_r, dphi_dv = torch_jvp(model.phi, (R_t,), (V_eff,))
    kappa = model.kappa(R_t)
    alpha = model.g(R_t)
    Sr = (S_t @ R_t.T).T
    beta = model.beta
    rhs = kappa * (alpha * Sr - beta * phi_r)
    if mask is not None:
        dphi_dv = dphi_dv * mask
        rhs = rhs * mask
    residual = dphi_dv - rhs

    cos_per_cell = F.cosine_similarity(dphi_dv, rhs, dim=1, eps=1e-8).detach().cpu().numpy()
    dyn_per_cell = residual.pow(2).sum(dim=1).detach().cpu().numpy()

    # Nearest-neighbor matching + FOSCTTM on the alignment panel (φ predicts all D_p,
    # but excluded features must not re-enter the geometric metric).
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

    foscttm = np.asarray(
        calc_domainAveraged_FOSCTTM(phi_a.detach().cpu().numpy(), P_a.cpu().numpy())
    )

    return {
        "foscttm":     foscttm,
        "time_err":    time_err,
        "branch_match": branch_match,
        "state_true":  state_true,
        "state_pred":  state_pred,
        "jvp_rhs_cos": cos_per_cell,
        "dyn":         dyn_per_cell,
        "align":       align_per_cell,
    }


def plot_grad_interaction(rows: list, output_path: Path) -> None:
    """2-panel: φ-gradient norms of the two objectives, and their cosine, vs epoch.

    Left: ||∇L_align|| and ||∇L_dyn|| on φ (log scale) — which objective drives φ's update.
    Right: cos(∇L_align, ∇L_dyn), mean ± std over the probe batches, with the 0 conflict line
    (< 0 = the objectives pull φ in opposing directions).
    """
    df = pd.DataFrame(rows)
    g = df.groupby("epoch")
    ep    = g["grad_cos"].mean().index.to_numpy()
    an_m  = g["grad_align_norm"].mean().to_numpy()
    dn_m  = g["grad_dyn_norm"].mean().to_numpy()
    cos_m = g["grad_cos"].mean().to_numpy()
    cos_s = g["grad_cos"].std(ddof=0).fillna(0.0).to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(ep, an_m, "-o", label=r"$\|\nabla L_{align}\|$", color="tab:blue")
    axes[0].plot(ep, dn_m, "-o", label=r"$\|\nabla L_{dyn}\|$", color="tab:red")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel(r"$\phi$-gradient L2 norm")
    axes[0].set_title("Objective gradient magnitude on φ")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].axhline(0.0, color="grey", lw=1, ls="--")
    axes[1].plot(ep, cos_m, "-o", color="tab:purple")
    axes[1].fill_between(ep, cos_m - cos_s, cos_m + cos_s, color="tab:purple", alpha=0.2)
    axes[1].set_ylim(-1.05, 1.05)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel(r"$\cos(\nabla L_{align}, \nabla L_{dyn})$")
    axes[1].set_title("Alignment–kinetics gradient conflict\n(<0 conflict, >0 cooperate)")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_foscttm_diagnostics(per_cell: dict, output_path: Path) -> None:
    """5-panel figure: FOSCTTM vs branch_match, time error, JVP/RHS cosine, dyn, align."""
    foscttm      = per_cell["foscttm"]
    time_err     = per_cell["time_err"]
    branch_match = per_cell["branch_match"]
    cos_pc       = per_cell["jvp_rhs_cos"]
    dyn_pc       = per_cell["dyn"]
    align_pc     = per_cell["align"]

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    # 1. FOSCTTM by branch match (correct vs incorrect); needs ground-truth branch.
    if (branch_match >= 0).any():
        correct = foscttm[branch_match == 1]
        wrong   = foscttm[branch_match == 0]
        axes[0].boxplot([correct, wrong], labels=["Match", "Mismatch"])
        axes[0].set_ylabel("FOSCTTM")
        axes[0].set_title(
            f"FOSCTTM by branch match\n"
            f"match={len(correct)} ({len(correct)/len(foscttm):.1%}), "
            f"mismatch={len(wrong)}"
        )
    else:
        axes[0].text(0.5, 0.5, "no ground-truth branch", ha="center", va="center")
        axes[0].set_title("FOSCTTM by branch match")

    # 2. FOSCTTM vs time abs error
    axes[1].scatter(foscttm, time_err, alpha=0.3, s=4)
    axes[1].set_xlabel("FOSCTTM")
    axes[1].set_ylabel("|t_pred - t_true|")
    axes[1].set_title("Time error vs FOSCTTM")

    # 3. FOSCTTM vs JVP/RHS cosine
    axes[2].scatter(foscttm, cos_pc, alpha=0.3, s=4)
    axes[2].set_xlabel("FOSCTTM")
    axes[2].set_ylabel("cos(J_φ·v, rhs)")
    axes[2].set_ylim(-1.05, 1.05)
    axes[2].set_title("ODE alignment vs FOSCTTM")

    # 4. FOSCTTM vs per-cell dynamic residual
    axes[3].scatter(foscttm, dyn_pc, alpha=0.3, s=4)
    axes[3].set_xlabel("FOSCTTM")
    axes[3].set_ylabel("‖J_φ·v − rhs‖²")
    axes[3].set_yscale("log")
    axes[3].set_title("Dyn residual vs FOSCTTM")

    # 5. FOSCTTM vs per-cell align proxy (NN distance to observed protein)
    axes[4].scatter(foscttm, align_pc, alpha=0.3, s=4)
    axes[4].set_xlabel("FOSCTTM")
    axes[4].set_ylabel("min_j ‖φ(r) − p_j‖")
    axes[4].set_title("Align proxy vs FOSCTTM")

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def compute_full_diagnostics(
    model,
    R_t: torch.Tensor,
    V_eff: torch.Tensor,
    P_t: torch.Tensor,
    S_t: torch.Tensor,
    rna_obs: pd.DataFrame,
    subset: int = 1024,
    mask: torch.Tensor | None = None,
    align_cols: torch.Tensor | None = None,
) -> dict:
    """End-of-training diagnostics: time, branch confusion, variance, plus JVP stats."""
    model.eval()

    with torch.no_grad():
        phi_all = model.phi(R_t)
        beta = model.beta.detach()
        phi_metric = phi_all if align_cols is None else phi_all[:, align_cols]
        P_metric = P_t if align_cols is None else P_t[:, align_cols]

    # Nearest-neighbor in protein space → predicted state/time of cell i is taken from the
    # observed cell whose protein is closest to φ(r_i). Row-chunked to bound memory.
    nn_idx_t, _ = nearest_protein_match(phi_metric, P_metric)
    nn_idx = nn_idx_t.cpu().numpy()

    # Ground-truth 'time' (synthetic) or scVelo pseudotime (real); None if neither.
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

    # Branch confusion needs ground-truth branch labels (synthetic only).
    if "state" in rna_obs.columns:
        state_true = np.asarray(rna_obs["state"].values)
        state_pred = state_true[nn_idx]
        labels = sorted(set(state_true.tolist()))
        label_to_idx = {label: i for i, label in enumerate(labels)}
        conf = np.zeros((len(labels), len(labels)), dtype=int)
        for t_label, p_label in zip(state_true, state_pred):
            conf[label_to_idx[t_label], label_to_idx[p_label]] += 1
        conf_mat = conf.tolist()
    else:
        labels = []
        conf_mat = []

    phi_np = phi_metric.cpu().numpy()
    p_np = P_metric.cpu().numpy()
    var_phi = float(phi_np.var(axis=0).mean())
    var_p   = float(p_np.var(axis=0).mean())
    phi_var_ratio = var_phi / max(var_p, 1e-12)

    # Trajectory DTW in protein space. `temporal` orders φ(r) by the model's
    # predicted pseudotime (t_pred) against observed protein ordered by true
    # pseudotime — it scores recovery of the temporal ordering. `recon` orders
    # both by true pseudotime — a warp-tolerant reconstruction score. Uses the
    # resolved pseudotime, so it runs on real data too; NaN only if no time exists.
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

    return {
        "time_mae":                  time_mae,
        "time_spearman":             time_spearman,
        "traj_dtw_temporal":         traj_dtw_temporal,
        "traj_dtw_recon":            traj_dtw_recon,
        "branch_confusion_matrix":   conf_mat,
        "branch_confusion_labels":   labels,
        "phi_variance":              var_phi,
        "p_variance":                var_p,
        "phi_variance_ratio":        phi_var_ratio,
        "jvp_rhs_cos":               jvp["jvp_rhs_cos"],
        "jvp_norm":                  jvp["jvp_norm"],
        "rhs_norm":                  jvp["rhs_norm"],
        "kappa_mean":                jvp["kappa_mean"],
        "kappa_std":                 jvp["kappa_std"],
        "kappa_min":                 jvp["kappa_min"],
        "kappa_max":                 jvp["kappa_max"],
        "alpha_mean":                jvp["alpha_mean"],
        "alpha_std":                 jvp["alpha_std"],
        "alpha_min":                 jvp["alpha_min"],
        "alpha_max":                 jvp["alpha_max"],
        "beta_mean":                 float(beta_np.mean()),
        "beta_std":                  float(beta_np.std()),
        "beta_values":               beta_np.tolist(),
    }


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

    # Hyperparameters
    model_name   = str(cfg.get("model",       "kot"))
    n_epochs     = int(cfg.get("n_epochs",     500))
    batch_size   = int(cfg.get("batch_size",   256))
    lr           = float(cfg.get("lr",         1e-3))
    lambda_dyn   = float(cfg.get("lambda_dyn", 1.0))
    lambda_reg   = float(cfg.get("lambda_reg", 1e-4))
    blur         = float(cfg.get("sinkhorn_reg", 0.1))
    # geomloss backend for the alignment Sinkhorn: "auto" (default) uses the linear
    # KeOps path once the batch is large, "tensorized" forces the O(N²) PyTorch path
    # (no pykeops). See src/losses/sinkhorn.py.
    sinkhorn_backend = str(cfg.get("sinkhorn_backend", "auto"))
    seed         = int(cfg.get("seed",         42))
    # Cap the alignment Sinkhorn to a random minibatch of this many cells each
    # epoch. None → full-batch (exact; keeps small/synthetic runs unchanged). Set
    # it for large real datasets whose n×n distance matrix would not fit in GPU.
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
    early_stopping_patience = int(cfg.get("early_stopping_patience", 100))

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

    use_feature_space = bool(cfg.get("kot_use_feature_space", False))
    rna_layer = cfg.get("kot_rna_layer")
    protein_layer = cfg.get("kot_protein_layer")
    velocity_layer = cfg.get("kot_velocity_layer") or cfg.get("velocity_layer")

    if use_feature_space:
        x = matrix_from_adata(rna_adata, rna_layer, "RNA")
        y = matrix_from_adata(second_adata, protein_layer, "protein")
        print(
            "[kot] Using feature-space inputs: "
            f"RNA={x.shape} ({rna_layer or 'X'}), protein={y.shape} ({protein_layer or 'X'})"
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
                f"  PYTHONPATH=. python tools/build_adt_mapping.py --datasets <dataset>"
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

    # RNA velocity in the same coordinate system as the KOT RNA input.
    V_raw = None
    selected_velocity_layer = None
    velocity_candidates = [velocity_layer] if velocity_layer else ["true_velocity", "velocity"]
    for key in velocity_candidates:
        if key in rna_adata.layers:
            V_raw = np.nan_to_num(to_dense(rna_adata.layers[key], np.float32), nan=0.0)
            selected_velocity_layer = key
            print(f"[kot] Using velocity layer: '{key}'")
            break
    if velocity_layer and V_raw is None:
        available = list(rna_adata.layers.keys())
        raise ValueError(f"KOT velocity layer '{velocity_layer}' not found. Available layers: {available}")
    if V_raw is None:
        available = list(rna_adata.layers.keys())
        raise ValueError(f"KOT requires a velocity layer. Available layers: {available}")

    if V_raw.shape[1] == D_r:
        V = V_raw.astype(np.float32)
    elif not use_feature_space and "PCs" in rna_adata.varm:
        V = V_raw @ np.array(rna_adata.varm["PCs"])
    else:
        raise ValueError(
            f"KOT velocity shape {V_raw.shape} is incompatible with KOT input shape {x.shape}."
        )

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

    if "velocity_confidence" in rna_adata.obs.columns:
        conf = np.array(rna_adata.obs["velocity_confidence"].values, dtype=np.float32)
    else:
        conf = np.ones(n, dtype=np.float32)

    R_t    = to_tensor(x,     device)   # (n, D_r)
    P_t    = to_tensor(y,     device)   # (n, D_p)
    S_t    = to_tensor(S_model, device)   # (D_p, D_r)
    conf_t = to_tensor(conf,  device)   # (n,)
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

    # Effective velocity actually fed to the JVP: apply the RNA-side weight FIRST, then
    # gauge-normalize on THAT vector — the gauge scalar must come from the exact velocity
    # passed to the Jacobian, not the raw field. Use the median of NONZERO per-cell norms
    # (robust to outliers and to the zeros the weight introduces). Training and diagnostics
    # both receive this same V_eff_t; neither reconstructs it from raw V + a separate weight.
    gauge_eps = 1e-8
    V_eff = V if velocity_weight_np is None else V * velocity_weight_np[None, :]
    if bool(cfg.get("kot_velocity_gauge_normalize", False)):
        cell_norms = np.linalg.norm(V_eff, axis=1)
        nonzero = cell_norms > gauge_eps
        gauge = float(np.median(cell_norms[nonzero])) if nonzero.any() else 1.0
        V_eff = V_eff / (gauge + gauge_eps)
        print(f"[kot] velocity gauge-normalized: median nonzero per-cell norm {gauge:.4g} -> 1.0")
    V_eff_t = to_tensor(V_eff.astype(np.float32), device)   # (n, D_r)

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
    # Optional per-protein normalisation (off by default). The scale is floored at
    # the median std so it only DOWN-weights high-abundance proteins; without the
    # floor, near-constant proteins get a tiny std and their residual explodes.
    if bool(cfg.get("kot_kinetics_normalize", False)):
        std = P_t.std(dim=0)
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
            anchor_indices = [anchor_indices[k] for k in keep]
            if len(anchor_betas) == n_from_csv:
                anchor_betas = [anchor_betas[k] for k in keep]
            if len(anchor_weights_list) == n_from_csv:
                anchor_weights_list = [anchor_weights_list[k] for k in keep]
            if len(anchor_sigmas_list) == n_from_csv:
                anchor_sigmas_list = [anchor_sigmas_list[k] for k in keep]
            print(f"[kot] anchors: {n_from_csv} from csv, {n_from_csv - len(anchor_indices)} "
                  f"excluded (not in kinetic mask), {len(anchor_indices)} active priors")
            if not anchor_indices:
                message = "No anchored proteins fall in the kinetic mask; the β prior would be empty."
                if allow_empty_anchor_after_mask:
                    print(f"[kot] WARNING: {message} Disabling β-anchor prior for this run.")
                    use_anchor = False
                else:
                    raise ValueError(message)
        if use_anchor:
            anchor_idx_t = torch.tensor(anchor_indices, dtype=torch.long, device=device)
            if anchor_betas:
                if len(anchor_betas) != len(anchor_indices):
                    raise ValueError(
                        "kot_anchor_betas must have the same length as kot_anchor_indices."
                    )
                anchor_target = torch.tensor(anchor_betas, dtype=torch.float32, device=device)
            else:
                anchor_target = torch.full((anchor_idx_t.numel(),), anchor_beta, device=device)
            if anchor_sigmas_list and len(anchor_sigmas_list) == len(anchor_indices):
                anchor_sigma_t = torch.tensor(anchor_sigmas_list, dtype=torch.float32, device=device)
            else:
                floor = prior_sigma_floor * float(cfg.get("beta_anchor_target", anchor_beta))
                anchor_sigma_t = torch.full((anchor_idx_t.numel(),), floor, device=device)
            if anchor_weights_list and len(anchor_weights_list) == len(anchor_indices):
                anchor_weight_t = torch.tensor(anchor_weights_list, dtype=torch.float32, device=device)

    # Warm-start β at the resolved anchor targets: β starts where the anchor wants
    # it (β = softplus(β_raw)+1e-6), stays trainable, and the soft prior holds it
    # there. This sidesteps the bounded travel of β from its default ~0.72 init,
    # which the diagnostics showed the prior alone cannot overcome.
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
    optimiser, scheduler = build_kot_optimiser(model, lr, n_epochs, use_phases)

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
    # and their cosine are comparable across epochs. Off by default (kot_grad_interaction_diag).
    grad_diag_on = bool(cfg.get("kot_grad_interaction_diag", False))
    probe_batches = []
    if grad_diag_on:
        probe_gen = torch.Generator(device=device).manual_seed(seed + 1)
        n_probe = int(cfg.get("kot_grad_interaction_batches", 4))
        probe_size = min(batch_size, n)
        for _ in range(n_probe):
            idx = torch.randperm(n, generator=probe_gen, device=device)[:probe_size]
            pidx = torch.randperm(n, generator=probe_gen, device=device)[:probe_size]
            probe_batches.append((idx, pidx))
        print(f"[kot] grad-interaction diag: {n_probe} fixed batches of {probe_size} cells")
    grad_diag_nan = {"grad_align_norm": np.nan, "grad_dyn_norm": np.nan,
                     "grad_cos_mean": np.nan, "grad_cos_std": np.nan,
                     "grad_align_norms": [], "grad_dyn_norms": [], "grad_cosines": []}
    grad_interaction_rows = []

    print(f"[kot] Training {n_epochs} epochs | n={n} | D_r={D_r} | D_p={D_p} | device={device}")

    print(f"[kot] model: {model_name}")
    print(f"[kot] bounds: κ∈{(kappa_min, kappa_max)}  α∈{(alpha_min, alpha_max)}")
    print(f"[kot] κ-prior: target={kappa_prior_target} λ_κ={lambda_kappa_prior}  |  "
          f"φ_spectral_norm={phi_spectral_norm}  g_freeze={g_freeze_epochs}  "
          f"dyn_warmup={dyn_warmup_epochs}  early_stop_patience={early_stopping_patience}")
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

    loss_history = {
        "epoch": [], "align": [], "dyn": [], "anchor": [], "kappa_prior": [], "reg": [], "total": [],
        "beta_mean": [], "beta_std": [],
        "kappa_mean": [], "kappa_std": [], "kappa_min": [], "kappa_max": [],
        "alpha_mean": [], "alpha_std": [], "alpha_min": [], "alpha_max": [],
        "jvp_rhs_cos": [], "jvp_norm": [], "rhs_norm": [],
        "grad_align_norm": [], "grad_dyn_norm": [], "grad_cos_mean": [], "grad_cos_std": [],
    }

    best_align = float("inf")
    best_align_epoch = 0
    best_align_state = None
    best_align_losses = None

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
                    for pg in optimiser.param_groups:
                        pg["lr"] = lr * phase3_lr_scale
                print(f"[kot] >>> Phase {current_phase} at epoch {epoch} "
                      f"(phi={'train' if phi_train else 'frozen'}, "
                      f"κ/α/β={'train' if dyn_train else 'frozen'}, "
                      f"lr={optimiser.param_groups[0]['lr']:.2e})")
            lam_align_eff = 1.0        if current_phase in (1, 3) else 0.0
            lam_dyn_eff   = lambda_dyn if current_phase in (2, 3) else 0.0
        else:
            lam_align_eff = 1.0
            lam_dyn_eff = (lambda_dyn * min(1.0, (epoch - 1) / dyn_warmup_epochs)
                           if dyn_warmup_epochs > 0 else lambda_dyn)
        effective_lambda_dyn = lam_dyn_eff   # keep diagnostic prints/logging compatible
        if not use_phases and g_freeze_epochs > 0:
            g_train = epoch > g_freeze_epochs
            for p in model.g.parameters():
                p.requires_grad = g_train
            if epoch == 1 or epoch == g_freeze_epochs + 1:
                state = "trainable" if g_train else "frozen"
                print(f"[kot] g/alpha head is {state} at epoch {epoch}")

        optimiser.zero_grad(set_to_none=True)

        # --- Sinkhorn alignment step (differentiable through phi) ---
        # Full-batch when small (exact); a fresh random minibatch each epoch when
        # sinkhorn_max_points caps it, so large real datasets (e.g. bmmc_cite at
        # ~90k cells) stay within GPU memory instead of building an n×n matrix.
        # Backward immediately so the φ/Sinkhorn graph is freed before the kinetics loop.
        align_val = 0.0
        align_was_computed = False
        if lam_align_eff > 0.0:
            if sinkhorn_max_points is not None and n > sinkhorn_max_points:
                # Sample RNA and protein cells INDEPENDENTLY — the data is unpaired, so the
                # protein batch must not be the same cells as the RNA batch (that would put
                # each cell's true partner in the batch and leak the alignment).
                rna_idx     = torch.randperm(n, device=device)[:sinkhorn_max_points]
                protein_idx = torch.randperm(n, device=device)[:sinkhorn_max_points]
                phi_align = model.phi(R_t[rna_idx])          # (b, D_p), φ predicts full panel
                P_align = P_t[protein_idx]
            else:
                phi_align = model.phi(R_t)                   # (n, D_p)
                P_align = P_t
            if align_cols is not None:                       # compare only use_for_alignment proteins
                phi_align = phi_align[:, align_cols]
                P_align = P_align[:, align_cols]
            loss_align = sinkhorn_divergence(phi_align, P_align, blur=blur, backend=sinkhorn_backend)
            (lam_align_eff * loss_align).backward()
            align_val = loss_align.item()
            align_was_computed = True

        # --- Kinetics step (per-batch, gradients accumulated) ---
        # Shuffle once on-GPU and split; each batch's JVP graph is freed by its own
        # backward() rather than all graphs living until the end of the epoch. Cell-weighted
        # (len_b/n) so the accumulated gradient equals one backward() on the full-data mean.
        # κ from each batch is reused for the κ prior instead of a second κ(R_t) pass.
        dyn_val = 0.0
        kappa_prior_val = 0.0
        dyn_was_computed = False
        if lam_dyn_eff > 0.0:
            dyn_was_computed = True
            for idx in torch.randperm(n, device=device).split(batch_size):
                weight = idx.numel() / n
                loss_dyn_b, kappa_b = kinetics_loss(
                    model.phi, model.kappa, model.g,
                    R_t[idx], V_eff_t[idx], SR_t[idx], model.beta, conf_t[idx],
                    mask=mask_t, scale=protein_scale,
                )
                batch_obj = (lam_dyn_eff * weight) * loss_dyn_b
                dyn_val += weight * loss_dyn_b.item()
                if lambda_kappa_prior > 0.0 and kappa_target_t is not None:
                    kp_b = lambda_kappa_prior * torch.log(kappa_b / kappa_target_t).pow(2).mean()
                    batch_obj = batch_obj + weight * kp_b
                    kappa_prior_val += weight * kp_b.item()
                batch_obj.backward()
        elif (
            lambda_kappa_prior > 0.0
            and kappa_target_t is not None
            and any(param.requires_grad for param in model.kappa.parameters())
        ):
            for idx in torch.randperm(n, device=device).split(batch_size):
                weight = idx.numel() / n
                kappa_b = model.kappa(R_t[idx])
                kp_b = lambda_kappa_prior * torch.log(kappa_b / kappa_target_t).pow(2).mean()
                (weight * kp_b).backward()
                kappa_prior_val += weight * kp_b.item()

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

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()
        if scheduler is not None:
            scheduler.step()
        if epoch == 1 and device.type == "cuda":
            print(f"[kot] GPU memory after epoch 1: "
                  f"allocated {torch.cuda.memory_allocated(device) / 1e9:.2f} GB, "
                  f"peak {torch.cuda.max_memory_allocated(device) / 1e9:.2f} GB "
                  f"(batch_size={batch_size}, n={n})")

        beta_detached = model.beta.detach()
        beta_mean_val = float(beta_detached.mean().cpu())
        beta_std_val  = float(beta_detached.std(unbiased=False).cpu())

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
                )
                for b, (na, nd, c) in enumerate(zip(
                        gi["grad_align_norms"], gi["grad_dyn_norms"], gi["grad_cosines"])):
                    grad_interaction_rows.append({
                        "epoch": epoch, "batch": b,
                        "grad_align_norm": na, "grad_dyn_norm": nd, "grad_cos": c,
                    })
            else:
                gi = grad_diag_nan
        else:
            diag = {"jvp_rhs_cos": np.nan, "jvp_norm": np.nan, "rhs_norm": np.nan,
                    "kappa_mean": np.nan, "kappa_std": np.nan,
                    "kappa_min": np.nan, "kappa_max": np.nan,
                    "alpha_mean": np.nan, "alpha_std": np.nan,
                    "alpha_min": np.nan, "alpha_max": np.nan}
            gi = grad_diag_nan

        loss_history["epoch"].append(epoch)
        loss_history["align"].append(align_val)
        loss_history["dyn"].append(dyn_val)
        loss_history["anchor"].append(anchor_val)
        loss_history["kappa_prior"].append(kappa_prior_val)
        loss_history["reg"].append(reg_val)
        loss_history["total"].append(total_val)
        loss_history["beta_mean"].append(beta_mean_val)
        loss_history["beta_std"].append(beta_std_val)
        loss_history["kappa_mean"].append(diag["kappa_mean"])
        loss_history["kappa_std"].append(diag["kappa_std"])
        loss_history["kappa_min"].append(diag["kappa_min"])
        loss_history["kappa_max"].append(diag["kappa_max"])
        loss_history["alpha_mean"].append(diag["alpha_mean"])
        loss_history["alpha_std"].append(diag["alpha_std"])
        loss_history["alpha_min"].append(diag["alpha_min"])
        loss_history["alpha_max"].append(diag["alpha_max"])
        loss_history["jvp_rhs_cos"].append(diag["jvp_rhs_cos"])
        loss_history["jvp_norm"].append(diag["jvp_norm"])
        loss_history["rhs_norm"].append(diag["rhs_norm"])
        loss_history["grad_align_norm"].append(gi["grad_align_norm"])
        loss_history["grad_dyn_norm"].append(gi["grad_dyn_norm"])
        loss_history["grad_cos_mean"].append(gi["grad_cos_mean"])
        loss_history["grad_cos_std"].append(gi["grad_cos_std"])

        if epoch % 50 == 0 or epoch == 1:
            print(f"[kot] epoch {epoch:4d}  align={align_val:.6f}  "
                  f"dyn={dyn_val:.6f}  anchor={anchor_val:.6f}  "
                  f"kprior={kappa_prior_val:.6f}  "
                  f"λ_dyn={effective_lambda_dyn:.3f}  lr={optimiser.param_groups[0]['lr']:.2e}  "
                  f"β={beta_mean_val:.3f}±{beta_std_val:.3f}  "
                  f"κ={diag['kappa_mean']:.3f}±{diag['kappa_std']:.3f} "
                  f"[{diag['kappa_min']:.3f},{diag['kappa_max']:.3f}]  "
                  f"α={diag['alpha_mean']:.3f} "
                  f"[{diag['alpha_min']:.3f},{diag['alpha_max']:.3f}]  "
                  f"cos={diag['jvp_rhs_cos']:.3f}  total={total_val:.6f}  "
                  f"[{time.perf_counter() - train_start:.0f}s, {epoch / (time.perf_counter() - train_start):.1f} ep/s]")
            if grad_diag_on:
                print(f"[kot]   grad-interaction: ||∇align||={gi['grad_align_norm']:.3e}  "
                      f"||∇dyn||={gi['grad_dyn_norm']:.3e}  "
                      f"cos(∇align,∇dyn)={gi['grad_cos_mean']:+.3f}±{gi['grad_cos_std']:.3f}")

        # Track minima for all three checkpoints.
        current_align = align_val
        current_dyn   = dyn_val
        current_total = total_val
        current_losses = {
            "align":  current_align,
            "dyn":    current_dyn,
            "anchor": anchor_val,
            "kappa_prior": kappa_prior_val,
            "total":  current_total,
        }

        if dyn_was_computed and current_dyn < best_dyn:
            best_dyn = current_dyn
            best_dyn_epoch = epoch
            best_dyn_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_dyn_losses = dict(current_losses)

        if current_total < best_total:
            best_total = current_total
            best_total_epoch = epoch
            best_total_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_total_losses = dict(current_losses)

        # Early stopping: only monitor epochs with a real alignment loss. In phased
        # runs, the dynamics-only phase sets lam_align_eff=0 and must not become
        # "best_align" simply because align_val is 0.
        monitor_align = align_was_computed and (use_phases or epoch > dyn_warmup_epochs)
        if monitor_align:
            if current_align < best_align:
                best_align = current_align
                best_align_epoch = epoch
                best_align_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                best_align_losses = dict(current_losses)
                epochs_since_improvement = 0
            elif not use_phases or current_phase == 3:
                epochs_since_improvement += 1
                if epochs_since_improvement >= early_stopping_patience:
                    stop_epoch = epoch
                    print(f"[kot] Early stopping at epoch {epoch}: "
                          f"best align={best_align:.6f} at epoch {best_align_epoch}")
                    break

    # Snapshot the literal final state before any restoration.
    final_state  = {k: v.detach().clone() for k, v in model.state_dict().items()}
    final_losses = dict(current_losses)

    # Uniform checkpoint table: (name, state, epoch, losses) for all four.
    checkpoints = [
        ("best_align", best_align_state, best_align_epoch, best_align_losses),
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
        phi_final = model.phi(R_t).cpu().numpy()        # (n, D_p)
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
    diagnostics = compute_full_diagnostics(
        model,
        R_t,
        V_eff_t,
        P_t,
        S_t,
        rna_adata.obs,
        subset=1024,
        mask=mask_t,
        align_cols=align_cols,
    )
    # Snapshot the raw best_align full-diagnostics (before the anchor augmentation below) so
    # the per-checkpoint loop reuses them for best_align — the model is on the same restored
    # best_align state, so recomputing the JVP + NN match there is pure waste.
    best_align_full = dict(diagnostics)
    diagnostics["anchor_loss_final"] = float(loss_anchor_final.item())
    diagnostics["beta_anchor_prior"] = beta_anchor_prior if use_anchor else None
    diagnostics["beta_anchor_aggregate"] = beta_anchor_aggregate if use_anchor else None
    diagnostics["beta_anchor_n"] = int(len(anchor_indices)) if use_anchor else 0
    diagnostics["beta_anchor_scale"] = float(anchor_scale) if anchor_scale is not None else None
    # Anchors intersected with the actual runtime kinetic mask: an anchored protein
    # only constrains the dynamics if it is in the kinetic term. Report which, by name.
    if use_anchor and anchor_indices:
        protein_names = list(second_adata.var_names)
        anchor_in_mask = [bool(kin_mask[j]) for j in anchor_indices]
        diagnostics["beta_anchor_in_kinetic_mask"] = anchor_in_mask
        diagnostics["beta_anchor_names_in_kinetic"] = [
            protein_names[j] for j, ok in zip(anchor_indices, anchor_in_mask) if ok
        ]
        diagnostics["beta_anchor_n_in_kinetic"] = int(sum(anchor_in_mask))
        print(f"[kot] anchors in runtime kinetic mask: "
              f"{diagnostics['beta_anchor_n_in_kinetic']}/{len(anchor_indices)}")
    else:
        diagnostics["beta_anchor_in_kinetic_mask"] = []
        diagnostics["beta_anchor_names_in_kinetic"] = []
        diagnostics["beta_anchor_n_in_kinetic"] = 0
    # Per-anchor recovery: judge the anchor on the proteins it constrains, not the
    # global beta_mean (which averages the ~130 unanchored proteins and hides whether
    # anchored β actually moved toward its target).
    if use_anchor:
        final_beta_np = model.beta.detach().cpu().numpy()
        anchor_targets_np = anchor_target.cpu().numpy()
        anchor_final_np = final_beta_np[anchor_indices]
        protein_names = list(second_adata.var_names)
        anchor_abs_err = [
            abs(float(f) - float(t)) for f, t in zip(anchor_final_np, anchor_targets_np)
        ]
        diagnostics["beta_anchor_indices"] = [int(j) for j in anchor_indices]
        diagnostics["beta_anchor_names"] = [protein_names[j] for j in anchor_indices]
        diagnostics["beta_anchor_targets"] = [float(t) for t in anchor_targets_np]
        diagnostics["beta_anchor_final"] = [float(f) for f in anchor_final_np]
        diagnostics["beta_anchor_abs_err"] = anchor_abs_err
        diagnostics["beta_anchor_mean_abs_err"] = float(sum(anchor_abs_err) / len(anchor_abs_err))
        diagnostics["beta_anchor_final_mean"] = float(anchor_final_np.mean())
    else:
        diagnostics["beta_anchor_indices"] = []
        diagnostics["beta_anchor_names"] = []
        diagnostics["beta_anchor_targets"] = []
        diagnostics["beta_anchor_final"] = []
        diagnostics["beta_anchor_abs_err"] = []
        diagnostics["beta_anchor_mean_abs_err"] = None
        diagnostics["beta_anchor_final_mean"] = None
    diagnostics["stop_epoch"]        = int(stop_epoch)
    diagnostics["best_align_epoch"]  = int(best_align_epoch) if best_align_state is not None else None
    diagnostics["best_align"]        = float(best_align) if best_align_state is not None else None
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
    diagnostics["data_debug"]        = data_debug

    print("[kot] end-of-training diagnostics:")
    print(f"[kot]   time MAE / Spearman: {diagnostics['time_mae']:.4f} / {diagnostics['time_spearman']:.4f}")
    print(f"[kot]   φ variance ratio:    {diagnostics['phi_variance_ratio']:.4f}")
    print(f"[kot]   JVP·RHS cos:         {diagnostics['jvp_rhs_cos']:.4f}")
    print(f"[kot]   |JVP| / |RHS|:       {diagnostics['jvp_norm']:.4f} / {diagnostics['rhs_norm']:.4f}")
    print(f"[kot]   κ mean/std [min,max]: {diagnostics['kappa_mean']:.4f} / {diagnostics['kappa_std']:.4f}  "
          f"[{diagnostics['kappa_min']:.4f}, {diagnostics['kappa_max']:.4f}]")
    print(f"[kot]   α mean/std [min,max]: {diagnostics['alpha_mean']:.4f} / {diagnostics['alpha_std']:.4f}  "
          f"[{diagnostics['alpha_min']:.4f}, {diagnostics['alpha_max']:.4f}]")
    print(f"[kot]   β mean/std:          {diagnostics['beta_mean']:.4f} / {diagnostics['beta_std']:.4f}")
    print(f"[kot]   anchor loss:         {loss_anchor_final.item():.6f}")
    print(f"[kot]   branch labels:       {diagnostics['branch_confusion_labels']}")
    print("[kot]   branch confusion:")
    for row, label in zip(diagnostics["branch_confusion_matrix"], diagnostics["branch_confusion_labels"]):
        print(f"[kot]     {label:>12s}: {row}")
    print(f"[kot]   final β: {model.beta.detach().cpu().numpy()}")
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

    if output_dir is not None:
        out = Path(output_dir)

        # 1. Save best_align diagnostics (downstream FOSCTTM eval uses this).
        diag_path = out / "diagnostics.json"
        with open(diag_path, "w") as f:
            json.dump(diagnostics, f, indent=2)
        print(f"[kot] Saved diagnostics: {diag_path}")

        # 2. Save the 4 checkpoint state dicts to disk.
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
                },
                ckpt_path,
            )
            print(f"[kot] Saved checkpoint: {ckpt_path} (epoch {ep})")

        # 3. Evaluate FOSCTTM + diagnostics at EACH checkpoint (not just best_align).
        # best_align reuses the full-diagnostics already computed above and caches its
        # per-cell result for the scatter panel below — computed once, not three times.
        per_ckpt_rows = []
        best_align_per_cell = None
        for name, state, ep, lo in checkpoints:
            if state is None:
                continue
            model.load_state_dict(state)
            if name == "best_align":
                d = dict(best_align_full)
            else:
                d = compute_full_diagnostics(
                    model,
                    R_t,
                    V_eff_t,
                    P_t,
                    S_t,
                    rna_adata.obs,
                    mask=mask_t,
                    align_cols=align_cols,
                )
            pc = compute_per_cell_diagnostics(
                model,
                R_t,
                V_eff_t,
                P_t,
                S_t,
                rna_adata.obs,
                mask=mask_t,
                align_cols=align_cols,
            )
            if name == "best_align":
                best_align_per_cell = pc
            mean_f = float(np.mean(pc["foscttm"]))
            d["mean_foscttm"]      = mean_f
            d["checkpoint"]        = name
            d["checkpoint_epoch"]  = ep
            d["checkpoint_losses"] = lo
            with open(out / f"diagnostics_{name}.json", "w") as f:
                json.dump(d, f, indent=2)
            pd.DataFrame({"foscttm": pc["foscttm"]}).to_csv(
                out / f"foscttm_{name}.csv", index=False,
            )
            per_ckpt_rows.append({
                "checkpoint":         name,
                "epoch":              ep,
                "mean_foscttm":       mean_f,
                "time_mae":           d["time_mae"],
                "time_spearman":      d["time_spearman"],
                "traj_dtw_temporal":  d["traj_dtw_temporal"],
                "traj_dtw_recon":     d["traj_dtw_recon"],
                "phi_variance_ratio": d["phi_variance_ratio"],
                "jvp_rhs_cos":        d["jvp_rhs_cos"],
                "jvp_norm":           d["jvp_norm"],
                "rhs_norm":           d["rhs_norm"],
                "kappa_mean":         d["kappa_mean"],
                "kappa_std":          d["kappa_std"],
                "kappa_min":          d["kappa_min"],
                "kappa_max":          d["kappa_max"],
                "alpha_mean":         d["alpha_mean"],
                "alpha_std":          d["alpha_std"],
                "alpha_min":          d["alpha_min"],
                "alpha_max":          d["alpha_max"],
                "beta_mean":          d["beta_mean"],
                "beta_std":           d["beta_std"],
            })
            print(f"[kot] Eval {name}: FOSCTTM={mean_f:.4f}  "
                  f"cos={d['jvp_rhs_cos']:+.3f}  "
                  f"κ={d['kappa_mean']:.3f} [{d['kappa_min']:.3f},{d['kappa_max']:.3f}]  "
                  f"α={d['alpha_mean']:.3f} [{d['alpha_min']:.3f},{d['alpha_max']:.3f}]  "
                  f"β={d['beta_mean']:.3f}")
        if per_ckpt_rows:
            pd.DataFrame(per_ckpt_rows).to_csv(
                out / "checkpoint_eval_summary.csv", index=False,
            )

        # 4. Restore best_align state so downstream artifacts (foscttm.csv, plots)
        # reflect the same checkpoint as the returned `aligned`.
        if best_align_state is not None:
            model.load_state_dict(best_align_state)

        # 5. Per-cell arrays + the FOSCTTM-vs-diagnostics scatter panel for best_align.
        # Reuse the per-cell result already computed for best_align in the loop above (same
        # restored state); only recompute in the edge case where best_align was never a
        # checkpoint (best_align_per_cell stays None).
        if best_align_per_cell is not None:
            per_cell = best_align_per_cell
        else:
            per_cell = compute_per_cell_diagnostics(
                model,
                R_t,
                V_eff_t,
                P_t,
                S_t,
                rna_adata.obs,
                mask=mask_t,
                align_cols=align_cols,
            )
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
