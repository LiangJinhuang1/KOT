"""
KOT neural network components.

Three learnable maps:
  phi_theta  : R^D_r → R^D_p   RNA → protein
  kappa_psi  : R^D_r → R^1     RNA → time-scale κ > 0  (Softplus)
  g_omega    : R^D_r → R^D_p   RNA → translation efficiency α > 0  (Softplus)

β (degradation rates) is a learnable parameter, kept positive via Softplus.
All networks use Xavier normal init with gain=0.1 for stable JVP computation.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def build_mlp(in_dim: int, out_dim: int, hidden_dims: list[int], init_gain: float = 1.0) -> nn.Sequential:
    """Build a GELU MLP with Xavier normal initialisation."""
    dims = [in_dim] + list(hidden_dims) + [out_dim]
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        lin = nn.Linear(dims[i], dims[i + 1])
        nn.init.xavier_normal_(lin.weight, gain=init_gain)
        nn.init.zeros_(lin.bias)
        layers.append(lin)
        if i < len(dims) - 2:
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


class PhiTheta(nn.Module):
    """RNA → protein mapping."""

    def __init__(self, d_rna: int, d_protein: int, hidden_dims: list[int], init_gain: float = 0.1):
        super().__init__()
        self.net = build_mlp(d_rna, d_protein, hidden_dims, init_gain=init_gain)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return self.net(r)


class KappaPsi(nn.Module):
    """RNA → scalar kinetic time-scale κ > 0."""

    def __init__(self, d_rna: int, hidden_dims: list[int]):
        super().__init__()
        self.net = build_mlp(d_rna, 1, hidden_dims, init_gain=0.1)
        self.softplus = nn.Softplus()

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return self.softplus(self.net(r)) + 1e-6


class GOmega(nn.Module):
    """RNA → per-protein translation efficiency α > 0."""

    def __init__(self, d_rna: int, d_protein: int, hidden_dims: list[int]):
        super().__init__()
        self.net = build_mlp(d_rna, d_protein, hidden_dims, init_gain=0.1)
        self.softplus = nn.Softplus()

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return self.softplus(self.net(r)) + 1e-6


class KOTModel(nn.Module):
    """
    Container for all three KOT networks plus learnable degradation rates β.

    Usage
    -----
    model = KOTModel(d_rna, d_protein)
    phi_r   = model.phi(r)
    kappa_r = model.kappa(r)
    alpha_r = model.g(r)
    beta    = model.beta          # (D_p,) positive tensor
    """

    def __init__(
        self,
        d_rna: int,
        d_protein: int,
        phi_dims: list[int] = (1024, 512, 256),
        kappa_dims: list[int] = (64, 32),
        g_dims: list[int] = (256, 128),
        phi_init_gain: float = 0.1,
    ):
        super().__init__()
        self.phi   = PhiTheta(d_rna, d_protein, list(phi_dims), init_gain=phi_init_gain)
        self.kappa = KappaPsi(d_rna, list(kappa_dims))
        self.g     = GOmega(d_rna, d_protein, list(g_dims))

        self.beta_raw  = nn.Parameter(torch.rand(d_protein) * 0.1)
        self.softplus  = nn.Softplus()

        self.d_rna     = d_rna
        self.d_protein = d_protein

    @property
    def beta(self) -> torch.Tensor:
        return self.softplus(self.beta_raw) + 1e-6

    def predict(self, r: torch.Tensor) -> torch.Tensor:
        return self.phi(r)
