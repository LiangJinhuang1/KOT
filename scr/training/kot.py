"""
KOT training loop: Sinkhorn alignment plus kinetic ODE optimisation.

The objective follows the paper terms:
  L = L_sinkhorn + lambda_dyn * L_kinetics + lambda_prior * L_anchor
      + lambda_reg * L_reg,
with L_anchor active only when an explicit anchor set H is configured.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

from scr.models.KOT import KOTModel
from scr.losses.sinkhorn import sinkhorn_divergence
from scr.losses.jvp_physics import kinetics_loss
from scr.data.projection import projection_matrix_from_adatas


def to_tensor(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(arr, dtype=torch.float32, device=device)


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
    phi_spectral_norm = bool(cfg.get("phi_spectral_norm", False))
    dyn_warmup_epochs       = int(cfg.get("dyn_warmup_epochs",       n_epochs // 2))
    early_stopping_patience = int(cfg.get("early_stopping_patience", 100))
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
    velocity_candidates = [velocity_layer] if velocity_layer else ["true_velocity", "velocity"]
    for key in velocity_candidates:
        if key in rna_adata.layers:
            V_raw = np.nan_to_num(dense_array(rna_adata.layers[key]), nan=0.0)
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
    ).to(device)

    initial_beta = model.beta.detach().cpu().numpy().copy()

    network_params = (
        list(model.phi.parameters())
        + list(model.kappa.parameters())
        + list(model.g.parameters())
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=n_epochs)
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
    print("initial beta:", initial_beta)
    print(f"[kot] dyn_warmup_epochs={dyn_warmup_epochs} | early_stopping_patience={early_stopping_patience}")

    loss_history = {"epoch": [], "align": [], "dyn": [], "anchor": [], "reg": [], "total": []}

    best_align = float("inf")
    best_epoch = 0
    best_state = None
    epochs_since_improvement = 0
    stop_epoch = n_epochs

    for epoch in range(1, n_epochs + 1):
        model.train()

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
        loss_reg = lambda_reg * sum(param.pow(2).sum() for param in network_params)
        if dyn_warmup_epochs > 0:
            effective_lambda_dyn = lambda_dyn * min(1.0, (epoch - 1) / dyn_warmup_epochs)
        else:
            effective_lambda_dyn = lambda_dyn
        loss = loss_align + effective_lambda_dyn * loss_dyn_total + loss_anchor + loss_reg

        optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()
        scheduler.step()

        loss_history["epoch"].append(epoch)
        loss_history["align"].append(loss_align.item())
        loss_history["dyn"].append(loss_dyn_total.item())
        loss_history["anchor"].append(loss_anchor.item())
        loss_history["reg"].append(loss_reg.item())
        loss_history["total"].append(loss.item())

        if epoch % 50 == 0 or epoch == 1:
            print(f"  epoch {epoch:4d}  align={loss_align.item():.6f}  "
                  f"dyn={loss_dyn_total.item():.6f}  λ_dyn={effective_lambda_dyn:.3f}  "
                  f"lr={optimiser.param_groups[0]['lr']:.2e}  total={loss.item():.6f}")

        # Early stopping: only monitor after warmup completes (kinetics term active at full weight).
        if epoch > dyn_warmup_epochs:
            current_align = loss_align.item()
            if current_align < best_align:
                best_align = current_align
                best_epoch = epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
                if epochs_since_improvement >= early_stopping_patience:
                    stop_epoch = epoch
                    print(f"[kot] Early stopping at epoch {epoch}: "
                          f"best align={best_align:.6f} at epoch {best_epoch}")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[kot] Restored best state from epoch {best_epoch} (align={best_align:.6f})")

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

    print("anchor_loss:", loss_anchor_final.item())
    print("final beta:", model.beta.detach().cpu().numpy())
    print(f"[kot] stopped at epoch {stop_epoch} / {n_epochs}")

    aligned = [phi_final, p_np]
    return aligned, None, pd.DataFrame(loss_history)
