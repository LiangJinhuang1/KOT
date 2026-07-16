"""
KOT neural network components.

Three learnable maps:
  phi_theta  : R^D_r → R^D_p   RNA → protein
  kappa_psi  : R^D_r → R^1     RNA → bounded positive time-scale κ
  g_omega    : R^D_r → R^D_p   RNA → bounded positive translation efficiency α

β (degradation rates) is a learnable parameter, kept positive via Softplus.
All networks use configurable weight initialization and hidden activations.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def activation_module(name: str) -> nn.Module:
    name = name.lower()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "softplus":
        return nn.Softplus()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unknown KOT activation '{name}'. Use gelu, silu, tanh, softplus, or relu.")


def build_mlp(
    in_dim: int,
    out_dim: int,
    hidden_dims: list[int],
    init_gain: float = 1.0,
    activation: str = "gelu",
    init_method: str = "orthogonal",
    use_spectral_norm: bool = False,
) -> nn.Sequential:
    """Build an MLP with configurable weight initialization."""
    dims = [in_dim] + list(hidden_dims) + [out_dim]
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        lin = nn.Linear(dims[i], dims[i + 1])
        if init_method == "orthogonal":
            nn.init.orthogonal_(lin.weight, gain=init_gain)
        elif init_method == "xavier":
            nn.init.xavier_normal_(lin.weight, gain=init_gain)
        else:
            raise ValueError("KOT init_method must be 'orthogonal' or 'xavier'.")
        nn.init.zeros_(lin.bias)
        if use_spectral_norm:
            lin = nn.utils.spectral_norm(lin)
        layers.append(lin)
        if i < len(dims) - 2:
            layers.append(activation_module(activation))
    return nn.Sequential(*layers)


class PhiTheta(nn.Module):
    """RNA → protein mapping."""

    def __init__(
        self,
        d_rna: int,
        d_protein: int,
        hidden_dims: list[int],
        init_gain: float = 0.1,
        activation: str = "gelu",
        init_method: str = "orthogonal",
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        self.net = build_mlp(
            d_rna,
            d_protein,
            hidden_dims,
            init_gain=init_gain,
            activation=activation,
            init_method=init_method,
            use_spectral_norm=use_spectral_norm,
        )

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return self.net(r)


class BoundedPositiveMLP(nn.Module):
    """RNA → positive output in [min_value, max_value], or unbounded above (softplus)
    when max_value is None. Used for the kinetic time-scale κ (out_dim=1) and the
    per-protein translation efficiency α (out_dim=D_p)."""

    def __init__(
        self,
        d_rna: int,
        out_dim: int,
        hidden_dims: list[int],
        activation: str = "gelu",
        init_method: str = "orthogonal",
        min_value: float = 1e-6,
        max_value: float | None = None,
    ):
        super().__init__()
        self.net = build_mlp(
            d_rna,
            out_dim,
            hidden_dims,
            init_gain=0.1,
            activation=activation,
            init_method=init_method,
        )
        self.softplus = nn.Softplus()
        self.min_value = float(min_value)
        self.max_value = float(max_value) if max_value is not None else None

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        raw = self.net(r)
        if self.max_value is None:
            return self.softplus(raw) + self.min_value
        span = self.max_value - self.min_value
        return self.min_value + span * torch.sigmoid(raw)


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
        activation: str = "gelu",
        init_method: str = "orthogonal",
        phi_spectral_norm: bool = False,
        kappa_min: float = 1e-6,
        kappa_max: float | None = None,
        alpha_min: float = 1e-6,
        alpha_max: float | None = None,
    ):
        super().__init__()
        self.phi   = PhiTheta(
            d_rna,
            d_protein,
            list(phi_dims),
            init_gain=phi_init_gain,
            activation=activation,
            init_method=init_method,
            use_spectral_norm=phi_spectral_norm,
        )
        self.kappa = BoundedPositiveMLP(
            d_rna,
            1,
            list(kappa_dims),
            activation=activation,
            init_method=init_method,
            min_value=kappa_min,
            max_value=kappa_max,
        )
        self.g     = BoundedPositiveMLP(
            d_rna,
            d_protein,
            list(g_dims),
            activation=activation,
            init_method=init_method,
            min_value=alpha_min,
            max_value=alpha_max,
        )

        self.beta_raw  = nn.Parameter(torch.rand(d_protein) * 0.1)
        self.softplus  = nn.Softplus()

        self.d_rna     = d_rna
        self.d_protein = d_protein

    @property
    def beta(self) -> torch.Tensor:
        return self.softplus(self.beta_raw) + 1e-6

    def predict(self, r: torch.Tensor) -> torch.Tensor:
        return self.phi(r)
