"""
KOT training loop: alternating Sinkhorn alignment + kinetic ODE optimisation.

Two-phase alternating descent per epoch:
  1. Sinkhorn step  — freeze model, update transport plan T via geomloss
                      (full-batch, no gradient through T)
  2. Kinetics step  — fix T, optimise φ, κ, α on micro-batches
                       loss = λ_align · L_sinkhorn + λ_dyn · L_ode + λ_reg · ‖θ‖²
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


def run_kot(context: dict, cfg: dict) -> tuple[list, np.ndarray | None]:
    """
    Full KOT training + alignment.

    Returns
    -------
    aligned  : [rna_aligned (n, D_p), protein_obs (n, D_p)]
    coupling : (n, n) final OT plan (dense, may be large)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = context["x"]             # (n, D_r) RNA PCA
    y = context["y"]             # (n, D_p) protein PCA
    rna_adata    = context["rna_adata"]
    second_adata = context["second_adata"]   # protein AnnData

    n, D_r = x.shape
    D_p = y.shape[1]

    # Hyperparameters
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
    use_explicit_links = bool(cfg.get("kot_use_explicit_links", False))

    # Gene→protein S matrix projected into joint PCA space → (D_p, D_r)
    # Step 1: S_np (n_proteins, n_genes) @ RNA_PCs (n_genes, D_r) → (n_proteins, D_r)
    # Step 2: protein_PCs.T (D_p, n_proteins) @ (n_proteins, D_r) → (D_p, D_r)
    # Both steps needed because φ maps to protein PCA space, not full protein space.
    S_np = projection_matrix_from_adatas(
        rna_adata,
        second_adata,
        use_mean_expr=False,
        use_explicit_links=use_explicit_links,
    )
    if "PCs" in rna_adata.varm:
        S_pca = S_np @ np.array(rna_adata.varm["PCs"])  # (n_proteins, D_r)
    else:
        S_pca = S_np.copy()
        print("[kot] Warning: no RNA PCA components — S matrix in gene space.")
    if "PCs" in second_adata.varm:
        protein_PCs = np.array(second_adata.varm["PCs"])  # (n_proteins, D_p)
        S_pca = protein_PCs.T @ S_pca                      # (D_p, D_r)
    else:
        S_pca = np.zeros((D_p, D_r), dtype=np.float32)
        print("[kot] Warning: no protein PCA components — S matrix is zero.")

    # RNA velocity in PCA space: real data → "velocity" (scVelo), synthetic → "true_velocity"
    V_raw = None
    for key in ["true_velocity", "velocity"]:
        if key in rna_adata.layers:
            V_raw = np.nan_to_num(np.array(rna_adata.layers[key]), nan=0.0)
            print(f"[kot] Using velocity layer: '{key}'")
            break
    if V_raw is not None:
        if "PCs" in rna_adata.varm:
            V = V_raw @ np.array(rna_adata.varm["PCs"])
        else:
            V = np.zeros_like(x)
        if "velocity_confidence" in rna_adata.obs.columns:
            conf = np.array(rna_adata.obs["velocity_confidence"].values, dtype=np.float32)
        else:
            conf = np.ones(n, dtype=np.float32)
    else:
        print("[kot] Warning: no velocity layer — kinetics loss is zero.")
        V = np.zeros_like(x)
        conf = np.zeros(n, dtype=np.float32)
        lambda_dyn = 0.0

    # Move to device
    R_t    = to_tensor(x,     device)   # (n, D_r)
    P_t    = to_tensor(y,     device)   # (n, D_p)
    V_t    = to_tensor(V,     device)   # (n, D_r)
    S_t    = to_tensor(S_pca, device)   # (D_p, D_r)
    conf_t = to_tensor(conf,  device)   # (n,)

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = KOTModel(
        D_r, D_p,
        phi_dims=phi_dims,
        kappa_dims=kappa_dims,
        g_dims=g_dims,
        phi_init_gain=phi_init_gain,
    ).to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=lambda_reg)
    loader = DataLoader(
        TensorDataset(R_t, V_t, conf_t, torch.arange(n, device=device)),
        batch_size=batch_size, shuffle=True,
    )

    print(f"[kot] Training {n_epochs} epochs | n={n} | D_r={D_r} | D_p={D_p} | device={device}")

    loss_history = {"epoch": [], "align": [], "dyn": [], "total": []}

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

        loss = loss_align + lambda_dyn * loss_dyn_total

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        loss_history["epoch"].append(epoch)
        loss_history["align"].append(loss_align.item())
        loss_history["dyn"].append(loss_dyn_total.item())
        loss_history["total"].append(loss.item())

        if epoch % 50 == 0 or epoch == 1:
            print(f"  epoch {epoch:4d}  align={loss_align.item():.6f}  "
                  f"dyn={loss_dyn_total.item():.6f}  total={loss.item():.6f}")

    # Final aligned representations
    model.eval()
    with torch.no_grad():
        phi_final = model.phi(R_t).cpu().numpy()        # (n, D_p)
        p_np = y                                         # (n, D_p) observed

    aligned = [phi_final, p_np]
    return aligned, None, pd.DataFrame(loss_history)
