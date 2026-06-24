"""
KOT training loop: Sinkhorn alignment plus kinetic ODE optimisation.

The objective follows the paper terms:
  L = L_sinkhorn + lambda_dyn * L_kinetics + lambda_prior * L_anchor
      + lambda_reg * L_reg,
with L_anchor active only when an explicit anchor set H is configured.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.func import jvp as torch_jvp
from torch.utils.data import TensorDataset, DataLoader

from scr.models.KOT import KOTModel
from scr.losses.sinkhorn import sinkhorn_divergence
from scr.losses.jvp_physics import kinetics_loss
from scr.data.projection import projection_matrix_from_adatas
from scr.metrics.FOSCTTM import calc_domainAveraged_FOSCTTM


def to_tensor(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(arr, dtype=torch.float32, device=device)


def build_kot_optimiser(model, lr: float, n_epochs: int, use_phases: bool):
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = None if use_phases else torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=n_epochs,
    )
    return optimiser, scheduler


def dense_array(matrix) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def matrix_from_adata(adata, layer_key: str | None, label: str) -> np.ndarray:
    if layer_key in (None, "", "X"):
        return dense_array(adata.X)
    if layer_key not in adata.layers:
        available = list(adata.layers.keys())
        raise ValueError(f"KOT {label} layer '{layer_key}' not found. Available layers: {available}")
    return dense_array(adata.layers[layer_key])


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
    print("---- kot data debug ----")
    print(
        "  layers:        "
        f"rna={summary['rna_layer']}  protein={summary['protein_layer']}  "
        f"velocity={summary['velocity_layer']}"
    )
    print(
        "  shapes:        "
        f"rna={summary['rna_shape']}  protein={summary['protein_shape']}  "
        f"velocity={summary['velocity_shape']}"
    )
    print(
        "  S matrix:      "
        f"shape={summary['s_matrix_shape']}  nonzero={summary['s_matrix_nonzero']}  "
        f"links={summary['link_count']}"
    )
    synthetic_meta = summary["synthetic_linked_ode"]
    if synthetic_meta:
        print(
            "  synthetic:     "
            f"stage={synthetic_meta.get('stage')}  "
            f"distractors={synthetic_meta.get('n_distractor_rna_genes')}  "
            f"native_rna={synthetic_meta.get('native_rna_layer')}  "
            f"native_protein={synthetic_meta.get('native_protein_layer')}"
        )
    for key in ["true_alpha", "true_beta", "true_kappa", "true_protein_velocity"]:
        stats = summary[key]
        if stats is not None:
            print(
                f"  {key:21s}"
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


def compute_jvp_rhs_diagnostics(
    model,
    R_t: torch.Tensor,
    V_t: torch.Tensor,
    S_t: torch.Tensor,
    subset: int = 512,
) -> dict:
    """JVP/RHS cosine + norms and kappa stats on a small subset.

    JVP requires autograd to be enabled (forward-mode AD). Do not call inside
    torch.no_grad().
    """
    was_training = model.training
    model.eval()

    m = min(subset, R_t.shape[0])
    r = R_t[:m]
    v = V_t[:m]

    phi_r, dphi_dv = torch_jvp(model.phi, (r,), (v,))
    kappa = model.kappa(r)
    alpha = model.g(r)
    Sr = (S_t @ r.T).T
    beta = model.beta
    rhs = kappa * (alpha * Sr - beta * phi_r)

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


def compute_per_cell_diagnostics(
    model,
    R_t: torch.Tensor,
    V_t: torch.Tensor,
    P_t: torch.Tensor,
    S_t: torch.Tensor,
    rna_obs: pd.DataFrame,
) -> dict:
    """Per-cell arrays for FOSCTTM-correlated diagnostic plots."""
    model.eval()

    # All-cell JVP and RHS (single forward-mode pass).
    phi_r, dphi_dv = torch_jvp(model.phi, (R_t,), (V_t,))
    kappa = model.kappa(R_t)
    alpha = model.g(R_t)
    Sr = (S_t @ R_t.T).T
    beta = model.beta
    rhs = kappa * (alpha * Sr - beta * phi_r)
    residual = dphi_dv - rhs

    cos_per_cell = F.cosine_similarity(dphi_dv, rhs, dim=1, eps=1e-8).detach().cpu().numpy()
    dyn_per_cell = residual.pow(2).sum(dim=1).detach().cpu().numpy()

    # Nearest-neighbor matching in protein space.
    dists = torch.cdist(phi_r, P_t)
    align_per_cell = dists.min(dim=1).values.detach().cpu().numpy()
    nn_idx = dists.argmin(dim=1).cpu().numpy()

    t_true = np.asarray(rna_obs["time"].values, dtype=np.float64)
    t_pred = t_true[nn_idx]
    time_err = np.abs(t_true - t_pred)

    state_true = np.asarray(rna_obs["state"].values)
    state_pred = state_true[nn_idx]
    branch_match = (state_true == state_pred).astype(int)

    phi_np = phi_r.detach().cpu().numpy()
    p_np = P_t.cpu().numpy()
    foscttm = np.asarray(calc_domainAveraged_FOSCTTM(phi_np, p_np))

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


def plot_foscttm_diagnostics(per_cell: dict, output_path: Path) -> None:
    """5-panel figure: FOSCTTM vs branch_match, time error, JVP/RHS cosine, dyn, align."""
    import matplotlib.pyplot as plt

    foscttm      = per_cell["foscttm"]
    time_err     = per_cell["time_err"]
    branch_match = per_cell["branch_match"]
    cos_pc       = per_cell["jvp_rhs_cos"]
    dyn_pc       = per_cell["dyn"]
    align_pc     = per_cell["align"]

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    # 1. FOSCTTM by branch match (correct vs incorrect)
    correct = foscttm[branch_match == 1]
    wrong   = foscttm[branch_match == 0]
    axes[0].boxplot([correct, wrong], labels=["Match", "Mismatch"])
    axes[0].set_ylabel("FOSCTTM")
    axes[0].set_title(
        f"FOSCTTM by branch match\n"
        f"match={len(correct)} ({len(correct)/len(foscttm):.1%}), "
        f"mismatch={len(wrong)}"
    )

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
    V_t: torch.Tensor,
    P_t: torch.Tensor,
    S_t: torch.Tensor,
    rna_obs: pd.DataFrame,
    subset: int = 1024,
) -> dict:
    """End-of-training diagnostics: time, branch confusion, variance, plus JVP stats."""
    model.eval()

    with torch.no_grad():
        phi_all = model.phi(R_t)
        beta = model.beta.detach()

    # Nearest-neighbor in protein space → predicted state/time of cell i is taken from the
    # observed cell whose protein is closest to φ(r_i).
    dists = torch.cdist(phi_all, P_t)             # (n, n)
    nn_idx = dists.argmin(dim=1).cpu().numpy()

    t_true = np.asarray(rna_obs["time"].values, dtype=np.float64)
    t_pred = t_true[nn_idx]
    time_mae = float(np.mean(np.abs(t_true - t_pred)))

    rank_true = pd.Series(t_true).rank().to_numpy()
    rank_pred = pd.Series(t_pred).rank().to_numpy()
    if rank_true.std() < 1e-12 or rank_pred.std() < 1e-12:
        time_spearman = float("nan")
    else:
        time_spearman = float(np.corrcoef(rank_true, rank_pred)[0, 1])

    state_true = np.asarray(rna_obs["state"].values)
    state_pred = state_true[nn_idx]
    labels = sorted(set(state_true.tolist()))
    label_to_idx = {label: i for i, label in enumerate(labels)}
    conf_mat = np.zeros((len(labels), len(labels)), dtype=int)
    for t_label, p_label in zip(state_true, state_pred):
        conf_mat[label_to_idx[t_label], label_to_idx[p_label]] += 1

    phi_np = phi_all.cpu().numpy()
    p_np = P_t.cpu().numpy()
    var_phi = float(phi_np.var(axis=0).mean())
    var_p   = float(p_np.var(axis=0).mean())
    phi_var_ratio = var_phi / max(var_p, 1e-12)

    jvp = compute_jvp_rhs_diagnostics(model, R_t, V_t, S_t, subset=subset)

    beta_np = beta.cpu().numpy()

    return {
        "time_mae":                  time_mae,
        "time_spearman":             time_spearman,
        "branch_confusion_matrix":   conf_mat.tolist(),
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    seed         = int(cfg.get("seed",         42))
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
    lambda_prior = float(cfg.get("lambda_prior",     1.0))
    fixed_kappa_value = cfg.get("kot_fixed_kappa")
    fixed_alpha_value = cfg.get("kot_fixed_alpha")
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
    S_np = projection_matrix_from_adatas(
        rna_adata,
        second_adata,
        use_mean_expr=False,
        use_explicit_links=use_explicit_links,
    )
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
            V_raw = np.nan_to_num(dense_array(rna_adata.layers[key]), nan=0.0)
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

    if "velocity_confidence" in rna_adata.obs.columns:
        conf = np.array(rna_adata.obs["velocity_confidence"].values, dtype=np.float32)
    else:
        conf = np.ones(n, dtype=np.float32)

    # Move to device
    R_t    = to_tensor(x,     device)   # (n, D_r)
    P_t    = to_tensor(y,     device)   # (n, D_p)
    V_t    = to_tensor(V,     device)   # (n, D_r)
    S_t    = to_tensor(S_model, device)   # (D_p, D_r)
    conf_t = to_tensor(conf,  device)   # (n,)

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

    initial_beta = model.beta.detach().cpu().numpy().copy()

    network_params = (
        list(model.phi.parameters())
        + list(model.kappa.parameters())
        + list(model.g.parameters())
    )
    optimiser, scheduler = build_kot_optimiser(model, lr, n_epochs, use_phases)
    loader = DataLoader(
        TensorDataset(R_t, V_t, conf_t, torch.arange(n, device=device)),
        batch_size=batch_size, shuffle=True,
    )

    print(f"[kot] Training {n_epochs} epochs | n={n} | D_r={D_r} | D_p={D_p} | device={device}")

    anchor_idx_t = None
    anchor_target = None
    if use_anchor:
        if not anchor_indices:
            raise ValueError("use_anchor=true requires at least one kot_anchor_indices entry.")
        anchor_idx_t = torch.tensor(anchor_indices, dtype=torch.long, device=device)
        if anchor_betas:
            if len(anchor_betas) != len(anchor_indices):
                raise ValueError(
                    "kot_anchor_betas must have the same length as kot_anchor_indices."
                )
            anchor_target = torch.tensor(anchor_betas, dtype=torch.float32, device=device)
        else:
            anchor_target = torch.full((anchor_idx_t.numel(),), anchor_beta, device=device)
        print(f"[kot] Anchor prior enabled for {anchor_idx_t.numel()} proteins.")

    print("model:", model_name)
    print("use_anchor:", use_anchor)
    print("anchor_indices:", anchor_indices)
    print("anchor_target:", anchor_target.cpu().numpy() if anchor_target is not None else None)
    print("lambda_prior:", lambda_prior)
    print("fixed_kappa:", fixed_kappa_value)
    print("fixed_alpha:", fixed_alpha_value)
    print("kappa_bounds:", (kappa_min, kappa_max))
    print("alpha_bounds:", (alpha_min, alpha_max))
    print("lambda_kappa_prior:", lambda_kappa_prior)
    print("kappa_prior_target:", kappa_prior_target)
    print("g_freeze_epochs:", g_freeze_epochs)
    print("phi_spectral_norm:", phi_spectral_norm)
    print("initial beta:", initial_beta)
    print(f"[kot] dyn_warmup_epochs={dyn_warmup_epochs} | early_stopping_patience={early_stopping_patience}")

    loss_history = {
        "epoch": [], "align": [], "dyn": [], "anchor": [], "kappa_prior": [], "reg": [], "total": [],
        "beta_mean": [], "beta_std": [],
        "kappa_mean": [], "kappa_std": [], "kappa_min": [], "kappa_max": [],
        "alpha_mean": [], "alpha_std": [], "alpha_min": [], "alpha_max": [],
        "jvp_rhs_cos": [], "jvp_norm": [], "rhs_norm": [],
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
                model.beta_raw.requires_grad = dyn_train
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

        # --- Sinkhorn alignment step (full batch, differentiable through phi) ---
        phi_all = model.phi(R_t)                        # (n, D_p)
        loss_align = sinkhorn_divergence(phi_all, P_t, blur=blur)

        # --- Kinetics step (micro-batches) ---
        loss_dyn_total = torch.tensor(0.0, device=device)
        for r_b, v_b, c_b, _ in loader:
            loss_dyn_total = loss_dyn_total + kinetics_loss(
                model.phi, model.kappa, model.g,
                r_b, v_b, S_t, model.beta, c_b,
            )
        loss_dyn_total = loss_dyn_total / len(loader)

        loss_anchor = (
            lambda_prior * (model.beta[anchor_idx_t] - anchor_target).pow(2).sum()
            if anchor_target is not None
            else torch.tensor(0.0, device=device)
        )
        loss_kappa_prior = torch.tensor(0.0, device=device)
        if lambda_kappa_prior > 0.0 and kappa_prior_target is not None:
            kappa_for_prior = model.kappa(R_t)
            target = torch.tensor(kappa_prior_target, dtype=R_t.dtype, device=device)
            loss_kappa_prior = (
                lambda_kappa_prior * torch.log(kappa_for_prior / target).pow(2).mean()
            )
        loss_reg = lambda_reg * sum(param.pow(2).sum() for param in network_params)
        loss = (
            lam_align_eff * loss_align
            + lam_dyn_eff * loss_dyn_total
            + loss_anchor
            + loss_kappa_prior
            + loss_reg
        )

        optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()
        if scheduler is not None:
            scheduler.step()

        beta_detached = model.beta.detach()
        beta_mean_val = float(beta_detached.mean().cpu())
        beta_std_val  = float(beta_detached.std(unbiased=False).cpu())

        # Periodic JVP-based diagnostics: every 50 epochs (and epoch 1) on a 512-cell subset.
        if epoch % 50 == 0 or epoch == 1:
            diag = compute_jvp_rhs_diagnostics(model, R_t, V_t, S_t, subset=512)
        else:
            diag = {"jvp_rhs_cos": np.nan, "jvp_norm": np.nan, "rhs_norm": np.nan,
                    "kappa_mean": np.nan, "kappa_std": np.nan,
                    "kappa_min": np.nan, "kappa_max": np.nan,
                    "alpha_mean": np.nan, "alpha_std": np.nan,
                    "alpha_min": np.nan, "alpha_max": np.nan}

        loss_history["epoch"].append(epoch)
        loss_history["align"].append(loss_align.item())
        loss_history["dyn"].append(loss_dyn_total.item())
        loss_history["anchor"].append(loss_anchor.item())
        loss_history["kappa_prior"].append(loss_kappa_prior.item())
        loss_history["reg"].append(loss_reg.item())
        loss_history["total"].append(loss.item())
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

        if epoch % 50 == 0 or epoch == 1:
            print(f"  epoch {epoch:4d}  align={loss_align.item():.6f}  "
                  f"dyn={loss_dyn_total.item():.6f}  anchor={loss_anchor.item():.6f}  "
                  f"kprior={loss_kappa_prior.item():.6f}  "
                  f"λ_dyn={effective_lambda_dyn:.3f}  lr={optimiser.param_groups[0]['lr']:.2e}  "
                  f"β={beta_mean_val:.3f}±{beta_std_val:.3f}  "
                  f"κ={diag['kappa_mean']:.3f}±{diag['kappa_std']:.3f} "
                  f"[{diag['kappa_min']:.3f},{diag['kappa_max']:.3f}]  "
                  f"α={diag['alpha_mean']:.3f} "
                  f"[{diag['alpha_min']:.3f},{diag['alpha_max']:.3f}]  "
                  f"cos={diag['jvp_rhs_cos']:.3f}  total={loss.item():.6f}")

        # Track minima for all three checkpoints.
        current_align = loss_align.item()
        current_dyn   = loss_dyn_total.item()
        current_total = loss.item()
        current_losses = {
            "align":  current_align,
            "dyn":    current_dyn,
            "anchor": loss_anchor.item(),
            "kappa_prior": loss_kappa_prior.item(),
            "total":  current_total,
        }

        if current_dyn < best_dyn:
            best_dyn = current_dyn
            best_dyn_epoch = epoch
            best_dyn_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_dyn_losses = dict(current_losses)

        if current_total < best_total:
            best_total = current_total
            best_total_epoch = epoch
            best_total_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_total_losses = dict(current_losses)

        # Early stopping: only monitor after warmup completes (kinetics term active at full weight).
        if epoch > dyn_warmup_epochs:
            if current_align < best_align:
                best_align = current_align
                best_align_epoch = epoch
                best_align_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                best_align_losses = dict(current_losses)
                epochs_since_improvement = 0
            else:
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

    print("[kot] Best checkpoints:")
    for name, state, ep, lo in checkpoints:
        if state is None:
            print(f"  {name:>10s}: (not recorded)")
        else:
            print(f"  {name:>10s}: epoch {ep:4d}  "
                  f"align={lo['align']:.6f}  dyn={lo['dyn']:.6f}  total={lo['total']:.6f}")

    if best_align_state is not None:
        model.load_state_dict(best_align_state)
        print(f"[kot] Restored best_align state from epoch {best_align_epoch} (align={best_align:.6f})")

    # Final aligned representations
    model.eval()
    with torch.no_grad():
        phi_final = model.phi(R_t).cpu().numpy()        # (n, D_p)
        p_np = y                                         # (n, D_p) observed
        loss_anchor_final = (
            lambda_prior * (model.beta[anchor_idx_t] - anchor_target).pow(2).sum()
            if anchor_target is not None
            else torch.tensor(0.0, device=device)
        )

    # End-of-training diagnostics: time, branch confusion, variance, JVP/RHS.
    diagnostics = compute_full_diagnostics(
        model, R_t, V_t, P_t, S_t, rna_adata.obs, subset=1024,
    )
    diagnostics["anchor_loss_final"] = float(loss_anchor_final.item())
    diagnostics["stop_epoch"]        = int(stop_epoch)
    diagnostics["best_align_epoch"]  = int(best_align_epoch)
    diagnostics["best_align"]        = float(best_align)
    diagnostics["best_dyn_epoch"]    = int(best_dyn_epoch)
    diagnostics["best_dyn"]          = float(best_dyn)
    diagnostics["best_total_epoch"]  = int(best_total_epoch)
    diagnostics["best_total"]        = float(best_total)
    diagnostics["fixed_kappa"]       = fixed_kappa_value is not None
    diagnostics["fixed_kappa_value"] = (
        float(fixed_kappa_value) if fixed_kappa_value is not None else None
    )
    diagnostics["fixed_alpha"]       = fixed_alpha_value is not None
    diagnostics["fixed_alpha_value"] = fixed_alpha_value
    diagnostics["model_name"]        = model_name
    diagnostics["seed"]              = int(seed)
    diagnostics["kappa_bounds"]      = [kappa_min, kappa_max]
    diagnostics["alpha_bounds"]      = [alpha_min, alpha_max]
    diagnostics["lambda_kappa_prior"] = float(lambda_kappa_prior)
    diagnostics["kappa_prior_target"] = kappa_prior_target
    diagnostics["g_freeze_epochs"]   = int(g_freeze_epochs)
    diagnostics["data_debug"]        = data_debug

    print("---- end-of-training diagnostics ----")
    print(f"  time_mae:           {diagnostics['time_mae']:.4f}")
    print(f"  time_spearman:      {diagnostics['time_spearman']:.4f}")
    print(f"  phi_variance_ratio: {diagnostics['phi_variance_ratio']:.4f}")
    print(f"  jvp_rhs_cos:        {diagnostics['jvp_rhs_cos']:.4f}")
    print(f"  jvp_norm / rhs_norm:{diagnostics['jvp_norm']:.4f} / {diagnostics['rhs_norm']:.4f}")
    print(f"  kappa mean/std:     {diagnostics['kappa_mean']:.4f} / {diagnostics['kappa_std']:.4f}")
    print(f"  kappa min/max:      {diagnostics['kappa_min']:.4f} / {diagnostics['kappa_max']:.4f}")
    print(f"  alpha mean/std:     {diagnostics['alpha_mean']:.4f} / {diagnostics['alpha_std']:.4f}")
    print(f"  alpha min/max:      {diagnostics['alpha_min']:.4f} / {diagnostics['alpha_max']:.4f}")
    print(f"  beta mean/std:      {diagnostics['beta_mean']:.4f} / {diagnostics['beta_std']:.4f}")
    print(f"  branch labels:      {diagnostics['branch_confusion_labels']}")
    print(f"  branch confusion:")
    for row, label in zip(diagnostics["branch_confusion_matrix"], diagnostics["branch_confusion_labels"]):
        print(f"    {label:>12s}: {row}")
    print("anchor_loss:", loss_anchor_final.item())
    print("final beta:", model.beta.detach().cpu().numpy())
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
        per_ckpt_rows = []
        for name, state, ep, lo in checkpoints:
            if state is None:
                continue
            model.load_state_dict(state)
            d  = compute_full_diagnostics(model, R_t, V_t, P_t, S_t, rna_adata.obs)
            pc = compute_per_cell_diagnostics(model, R_t, V_t, P_t, S_t, rna_adata.obs)
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
        per_cell = compute_per_cell_diagnostics(model, R_t, V_t, P_t, S_t, rna_adata.obs)
        per_cell_path = out / "per_cell_diagnostics.npz"
        np.savez(per_cell_path, **per_cell)
        plot_path = out / "foscttm_diagnostics.png"
        plot_foscttm_diagnostics(per_cell, plot_path)
        print(f"[kot] Saved per-cell arrays: {per_cell_path}")
        print(f"[kot] Saved FOSCTTM diagnostics plot: {plot_path}")

    aligned = [phi_final, p_np]
    return aligned, None, pd.DataFrame(loss_history)
