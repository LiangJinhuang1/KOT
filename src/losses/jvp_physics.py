"""
Kinetic ODE residual loss using forward-mode AD (JVP).

ODE:  dφ/dt = κ(r) · (α(r) ⊙ S·r − β ⊙ φ(r))

The JVP J_φ(r)·v gives dφ/dt for free in a single forward pass.
"""
import torch
from torch.func import jvp as torch_jvp


def jacobian_vector_product(phi_theta, r: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """J_phi(r) · direction, in one forward-mode pass.

    Used for the CRISPR Jacobian prediction: evaluated at the control mean with
    direction = mean(r_KO) - mean(r_NT), it is the LOCAL linear response of the learned
    map to an actual intervention. Nothing about it is fitted to the knockout -- it is the
    differential of a map trained on controls, read in the direction the data moved.

    JVP needs autograd, so do not call this inside torch.no_grad().
    """
    return torch_jvp(phi_theta, (r,), (direction,))[1]


def kinetics_loss(
    phi_theta,
    kappa_psi,
    g_omega,
    r: torch.Tensor,
    v_eff: torch.Tensor,
    sr: torch.Tensor,
    beta: torch.Tensor,
    confidence: torch.Tensor,
    mask: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean residual over linked proteins, confidence-weighted.

    Average (not sum) so λ_dyn is comparable across panel sizes; divide by
    per-protein scale so high-abundance proteins do not dominate. `v_eff` already
    carries the RNA-side reliability weight.
    """
    phi_r, dphi_dv = torch_jvp(phi_theta, (r,), (v_eff,))
    kappa = kappa_psi(r)
    alpha = g_omega(r)
    residual = dphi_dv - kappa * (alpha * sr - beta * phi_r)
    if scale is not None:
        residual = residual / scale
    if mask is not None:
        residual = residual * mask
        n_active = mask.sum().clamp(min=1.0)
    else:
        n_active = float(residual.shape[1])
    per_cell = residual.pow(2).sum(dim=1) / n_active
    return (confidence * per_cell).mean(), kappa
