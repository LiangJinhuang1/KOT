"""
Kinetic ODE residual loss using forward-mode AD (JVP).

ODE:  dφ/dt = κ(r) · (α(r) ⊙ S·r − β ⊙ φ(r))

The JVP J_φ(r)·v gives dφ/dt for free in a single forward pass.
"""
import torch
from torch.func import jvp as torch_jvp


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
    """
    Velocity-confidence-weighted ODE residual, averaged over the active proteins.

    L = (1/B) Σ_i c_i · mean_{j: m_j=1} ( [ J_φ(r_i)·v_i − κ(α⊙Sr_i − β⊙φ_i) ]_j / s_j )²

    Only proteins with mask==1 (a gene→protein link, i.e. usable for kinetics)
    enter the loss; the residual is averaged over those active proteins (not
    summed) so the term is comparable across datasets with different numbers of
    mapped proteins. Dividing by the per-protein scale s_j keeps high-abundance
    proteins from dominating.

    `v_eff` and `sr` are precomputed by the caller once for the whole dataset and
    sliced per batch: `v_eff` already has the RNA-side reliability weight applied
    (velocity_weight ⊙ v, so noise from genes without a usable fit cannot leak
    through the Jacobian push-forward), and `sr` = S·r is the fixed gene→protein
    projection. κ(r) is returned so the caller can reuse it for the κ prior without
    a second forward pass over the batch.

    Parameters
    ----------
    phi_theta  : callable  (B, D_r) → (B, D_p)
    kappa_psi  : callable  (B, D_r) → (B, 1)
    g_omega    : callable  (B, D_r) → (B, D_p)
    r          : (B, D_r)  RNA states
    v_eff      : (B, D_r)  RNA velocity, velocity_weight already applied
    sr         : (B, D_p)  precomputed S·r  (S is fixed, not learned)
    beta       : (D_p,)    degradation rates  (anchor-fixed)
    confidence : (B,)      scVelo velocity_confidence in [0, 1]
    mask       : (D_p,)    1 for proteins used in kinetics, 0 for alignment-only.
                           None = every protein is active.
    scale      : (D_p,)    per-protein normaliser (e.g. observed-protein std).
                           None = no per-protein normalisation.

    Returns
    -------
    (loss, kappa) : the scalar residual loss and κ(r) (B, 1) for prior reuse.
    """
    # JVP: phi_r = φ(r),  dphi_dv = J_φ(r)·v_eff   — one forward pass
    phi_r, dphi_dv = torch_jvp(phi_theta, (r,), (v_eff,))

    kappa = kappa_psi(r)               # (B, 1)
    alpha = g_omega(r)                 # (B, D_p)

    rhs = kappa * (alpha * sr - beta * phi_r)  # (B, D_p)
    residual = dphi_dv - rhs                   # (B, D_p)
    if scale is not None:
        residual = residual / scale            # per-protein normalisation
    if mask is not None:
        residual = residual * mask             # keep only active (mapped) proteins
        n_active = mask.sum().clamp(min=1.0)
    else:
        n_active = float(residual.shape[1])

    per_cell = residual.pow(2).sum(dim=1) / n_active   # mean over active proteins
    return (confidence * per_cell).mean(), kappa
