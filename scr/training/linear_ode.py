import numpy as np
from sklearn.preprocessing import normalize

from scr.models.LinearODE import solve_linear_ode


def sinkhorn_coupling(x: np.ndarray, y: np.ndarray, reg: float = 0.1, n_iter: int = 100) -> np.ndarray:
    """Sinkhorn algorithm for entropic OT coupling between x and y."""
    n, m = x.shape[0], y.shape[0]
    C = np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)  # (n, m) cost
    C = C / C.max()
    K = np.exp(-C / reg)
    u = np.ones(n) / n
    for _ in range(n_iter):
        v = (1.0 / m) / (K.T @ u)
        u = (1.0 / n) / (K @ v)
    coupling = np.diag(u) @ K @ np.diag(v)
    return coupling


def run_linear_ode(context: dict, cfg: dict) -> tuple[list, np.ndarray]:
    """
    Linear ODE convex baseline: ṗ = Ar − B·p solved via alternating least squares.
    Alignment uses Sinkhorn OT to get soft correspondences between cells.

    Relevant cfg keys:
      lambda_ode    — ODE regularization weight (default 1.0)
      anchor_betas  — list of degradation rates for anchor proteins (default: all 1.0)
      sinkhorn_reg  — Sinkhorn entropic regularization (default 0.1)
      n_als_outer   — outer ALS iterations (default 5)
    """
    x = context["x"]             # (n, D_r) RNA repr
    y = context["y"]             # (m, D_p) protein repr
    rna_adata = context["rna_adata"]

    lambda_ode = float(cfg.get("lambda_ode", 1.0))
    sinkhorn_reg = float(cfg.get("sinkhorn_reg", 0.1))
    n_outer = int(cfg.get("n_als_outer", 5))

    # RNA velocity in gene space: real data → "velocity" (scVelo), synthetic → "true_velocity"
    V_raw = None
    for key in ["velocity", "true_velocity"]:
        if key in rna_adata.layers:
            V_raw = np.nan_to_num(np.array(rna_adata.layers[key]), nan=0.0)
            print(f"[linear_ode] Using velocity layer: '{key}'")
            break
    if V_raw is None:
        print("[linear_ode] Warning: no velocity layer found — ODE term will be zero.")
        X_mat = rna_adata.X.toarray() if hasattr(rna_adata.X, "toarray") else rna_adata.X
        V_raw = np.zeros_like(np.array(X_mat))

    # Project velocity into PCA space using existing PCA components
    if "PCs" in rna_adata.varm:
        PCs = np.array(rna_adata.varm["PCs"])              # (D_genes, n_pcs)
        V = V_raw @ PCs                                     # (n, n_pcs)
    else:
        V = V_raw[:, :x.shape[1]]

    D_p = y.shape[1]
    anchor_betas = cfg.get("anchor_betas") or [1.0] * D_p
    B = np.array(anchor_betas, dtype=float)
    if len(B) != D_p:
        B = np.ones(D_p)

    # Normalize inputs
    x_norm = normalize(x, norm="l2")
    y_norm = normalize(y, norm="l2")

    # x_norm and y_norm live in different spaces (RNA PCA vs protein PCA).
    # Start from uniform coupling; after the first ALS step W maps RNA → protein
    # space, and subsequent Sinkhorn calls operate in the shared protein space.
    n, m = x_norm.shape[0], y_norm.shape[0]
    coupling = np.outer(np.ones(n) / n, np.ones(m) / m)
    W = None
    for _ in range(n_outer):
        # Soft-matched protein targets: P_soft[i] = Σ_j π_ij · y_j
        P_soft = coupling @ y_norm          # (n, D_p)
        W, _ = solve_linear_ode(x_norm, V, P_soft, B, lambda_ode=lambda_ode)
        x_mapped = (W @ x_norm.T).T         # (n, D_p)
        coupling = sinkhorn_coupling(x_mapped, y_norm, reg=sinkhorn_reg)

    x_aligned = (W @ x_norm.T).T           # (n, D_p) — RNA mapped to protein space
    aligned = [x_aligned, y_norm]
    return aligned, coupling
