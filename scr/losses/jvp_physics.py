"""
Kinetic ODE residual loss using forward-mode AD (JVP).

ODE:  dφ/dt = κ(r) · (α(r) ⊙ S·r − β ⊙ φ(r))

The JVP J_φ(r)·v gives dφ/dt for free in a single forward pass.
"""
import torch
try:
    from torch.func import jvp as torch_jvp
except ImportError:
    from functorch import jvp as torch_jvp


def kinetics_loss(
    phi_theta,
    kappa_psi,
    g_omega,
    r: torch.Tensor,
    v: torch.Tensor,
    S: torch.Tensor,
    beta: torch.Tensor,
    confidence: torch.Tensor,
) -> torch.Tensor:
    """
    Weighted ODE residual.

    L = (1/B) Σ_i c_i · ‖ J_φ(r_i)·v_i − κ(r_i)·(α(r_i) ⊙ S·r_i − β ⊙ φ(r_i)) ‖²

    Parameters
    ----------
    phi_theta  : callable  (B, D_r) → (B, D_p)
    kappa_psi  : callable  (B, D_r) → (B, 1)
    g_omega    : callable  (B, D_r) → (B, D_p)
    r          : (B, D_r)  RNA states
    v          : (B, D_r)  RNA velocity
    S          : (D_p, D_r) gene-to-protein projection  (fixed, not learned)
    beta       : (D_p,)    degradation rates  (anchor-fixed)
    confidence : (B,)      scVelo velocity_confidence in [0, 1]
    """
    # JVP: phi_r = φ(r),  dphi_dv = J_φ(r)·v   — one forward pass
    phi_r, dphi_dv = torch_jvp(phi_theta, (r,), (v,))

    kappa = kappa_psi(r)               # (B, 1)
    alpha = g_omega(r)                 # (B, D_p)

    # S·r: (D_p, D_r) @ (D_r, B) → (D_p, B) → (B, D_p)
    Sr = (S @ r.T).T                   # (B, D_p)

    rhs = kappa * (alpha * Sr - beta * phi_r)  # (B, D_p)
    residual = dphi_dv - rhs                   # (B, D_p)

    per_cell = residual.norm(dim=1).pow(2)     # (B,)
    return (confidence * per_cell).mean()
