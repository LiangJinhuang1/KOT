import numpy as np
import torch
from sklearn.preprocessing import normalize

from src.utils.arrays import to_dense
from src.models.LinearODE import solve_linear_ode


def linear_ode_device(cfg: dict) -> torch.device:
    """GPU when available (and not forced off) — the n×n Sinkhorn is the only O(n²) step."""
    if str(cfg.get("device", "auto")).lower() == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sinkhorn_coupling(x: np.ndarray, y: np.ndarray, reg: float = 0.1,
                      n_iter: int = 100, device: torch.device | None = None) -> np.ndarray:
    """Entropic-OT coupling between x and y, on `device` (GPU when available).

    The cost is ||x_i-y_j||² = ||x_i||²+||y_j||²-2 x_i·y_j, computed directly as an (n, m)
    matrix without the (n, m, d) broadcast that would allocate terabytes for large n. On a
    B200 the n×n Sinkhorn iterations are trivial even at ~90k cells.
    """
    device = device if device is not None else torch.device("cpu")
    xt = torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)
    yt = torch.as_tensor(np.asarray(y), dtype=torch.float32, device=device)
    n, m = xt.shape[0], yt.shape[0]
    C = (xt * xt).sum(1)[:, None] + (yt * yt).sum(1)[None, :] - 2.0 * (xt @ yt.t())
    C.clamp_(min=0.0)
    cost_max = C.max()
    if cost_max > 0:
        C = C / cost_max
    K = torch.exp(-C / reg)
    u = torch.full((n,), 1.0 / n, dtype=torch.float32, device=device)
    for _ in range(n_iter):
        v = (1.0 / m) / (K.t() @ u)
        u = (1.0 / n) / (K @ v)
    coupling = (u[:, None] * K) * v[None, :]
    return coupling.cpu().numpy()


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
    velocity_layer = cfg.get("linear_ode_velocity_layer") or cfg.get("velocity_layer")

    V_raw = None
    velocity_candidates = [velocity_layer] if velocity_layer else ["velocity", "true_velocity"]
    for key in velocity_candidates:
        if key in rna_adata.layers:
            V_raw = np.nan_to_num(to_dense(rna_adata.layers[key]), nan=0.0)
            print(f"[linear_ode] Using velocity layer: '{key}'")
            break
    if velocity_layer and V_raw is None:
        available = list(rna_adata.layers.keys())
        raise ValueError(
            f"Linear ODE velocity layer '{velocity_layer}' not found. Available layers: {available}"
        )
    if V_raw is None:
        available = list(rna_adata.layers.keys())
        raise ValueError(f"Linear ODE requires a velocity layer. Available layers: {available}")

    # Project velocity into PCA space using existing PCA components
    if "PCs" in rna_adata.varm:
        PCs = np.array(rna_adata.varm["PCs"])              # (D_genes, n_pcs)
        V = V_raw @ PCs                                     # (n, n_pcs)
    else:
        V = V_raw[:, :x.shape[1]]

    device = linear_ode_device(cfg)

    D_p = y.shape[1]
    anchor_betas = cfg.get("anchor_betas") or [1.0] * D_p
    B = np.array(anchor_betas, dtype=float)
    if len(B) != D_p:
        raise ValueError(f"Linear ODE anchor_betas length {len(B)} does not match protein dimension {D_p}.")

    # Normalize the FULL inputs — the aligned output must cover every cell (obs_names).
    x_norm = normalize(x, norm="l2")
    y_norm = normalize(y, norm="l2")

    # The Sinkhorn coupling is O(n²); estimate W on a capped SUBSAMPLE (sinkhorn_max_points,
    # the same cap KOT uses), then apply the tiny (D_p×D_r) map W to ALL cells. Paired
    # RNA/protein use the same indices so pairing is preserved. Datasets below the cap (e.g.
    # pbmc) run in full; None = no cap.
    max_points = cfg.get("sinkhorn_max_points")
    n_full = x_norm.shape[0]
    if max_points is not None and n_full > int(max_points):
        rng = np.random.default_rng(int(cfg.get("seed", 42)))
        sub = rng.choice(n_full, size=int(max_points), replace=False)
        print(f"[linear_ode] estimating W on {int(max_points)}/{n_full} sampled cells for the "
              f"O(n²) OT (device={device})")
    else:
        sub = np.arange(n_full)
    xs, ys, Vs = x_norm[sub], y_norm[sub], V[sub]

    # ALS on the subsample: alternate soft-matching (Sinkhorn OT) with the ODE least-squares.
    # Start from uniform coupling; after the first step W maps RNA → protein space and later
    # Sinkhorn calls operate in the shared protein space.
    n, m = xs.shape[0], ys.shape[0]
    coupling = np.outer(np.ones(n) / n, np.ones(m) / m)
    W = None
    for _ in range(n_outer):
        P_soft = coupling @ ys              # (n_sub, D_p)
        W, _ = solve_linear_ode(xs, Vs, P_soft, B, lambda_ode=lambda_ode)
        x_mapped = (W @ xs.T).T             # (n_sub, D_p)
        coupling = sinkhorn_coupling(x_mapped, ys, reg=sinkhorn_reg, device=device)

    # Apply the learned map to EVERY cell so the output matches obs_names / full FOSCTTM.
    x_aligned = (W @ x_norm.T).T           # (n_full, D_p)
    aligned = [x_aligned, y_norm]
    return aligned, coupling
