"""The chromatin KOT network: the frozen KOT building blocks, wired for a second rate.

`src/models/KOT.py` is imported unchanged — the same PhiTheta and the same bounded
positive head that the RNA→protein runs use, so a difference between the two experiments
cannot come from a different architecture.

One thing genuinely differs. The RNA→protein law needs a single per-protein rate (beta,
degradation). The relay law needs two per-gene rates, and they mean different things:

    beta   splicing:    unspliced -> spliced
    gamma  degradation: spliced -> nothing

The reduced law needs only gamma, and its gamma plays exactly the role the frozen model's
beta plays. Both are parameters rather than networks for the same reason as there: a rate
that varies per cell is not identifiable against a per-cell kappa.

Every other constructor default matches `config/training.yaml`, which is where a run
actually reads them from — the defaults here only make the class usable on its own.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.losses.chromatin_laws import RELAY
from src.models.KOT import BoundedPositiveMLP, PhiTheta


class GeneAffineResidualPhi(PhiTheta):
    """An unpaired, gene-aware affine initialisation plus a nonlinear residual.

    Gene activity and RNA already have an explicit correspondence G. Sending those
    coordinates only through a 256-dimensional bottleneck discards that information and
    lets a distributional loss settle on the target mean. The gene-aware affine path
    preserves G and is initialised from the two *disjoint* training marginals; the inherited
    MLP then
    learns deviations from that biologically meaningful starting point.
    """

    def __init__(self, d_input: int, d_output: int, hidden_dims: list[int],
                 projection: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor,
                 residual_weight: float = 1.0, **kwargs):
        super().__init__(d_input, d_output, hidden_dims, **kwargs)
        if projection.shape != (d_output, d_input):
            raise ValueError(
                f"phi projection has shape {tuple(projection.shape)}, expected "
                f"{(d_output, d_input)}")
        if scale.shape != (d_output,) or bias.shape != (d_output,):
            raise ValueError("phi affine scale and bias must have one value per output")
        # Gene identity is structural, but the marginal calibration is only an
        # INITIALISATION. Keeping scale/bias as buffers freezes most of J_phi to the
        # affine shortcut: the dynamics loss can then fit alpha/kappa/gamma while barely
        # changing the biological push-forward. They are part of phi and must train.
        self.register_buffer("gene_projection", projection.detach().clone().float())
        self.gene_scale = nn.Parameter(scale.detach().clone().float())
        self.gene_bias = nn.Parameter(bias.detach().clone().float())
        self.residual_weight = float(residual_weight)
        self.residual_gate = nn.Parameter(torch.zeros(()))

    def forward(self, chromatin: torch.Tensor) -> torch.Tensor:
        projected = chromatin @ self.gene_projection.T
        base = projected * self.gene_scale + self.gene_bias
        residual_scale = self.residual_weight * torch.tanh(self.residual_gate)
        return base + residual_scale * self.net(chromatin)


class ChromatinKOT(nn.Module):
    """phi: chromatin → RNA (or → [unspliced, spliced]), with kappa, alpha, beta, gamma."""

    def __init__(
        self,
        n_genes: int,
        law: str,
        n_input_features: int | None = None,
        phi_dims: list[int] = (1024, 512, 256),
        kappa_dims: list[int] = (64, 32),
        g_dims: list[int] = (256, 128),
        phi_init_gain: float = 0.1,
        activation: str = "silu",
        init_method: str = "xavier",
        phi_spectral_norm: bool = True,
        kappa_min: float = 1e-3,
        kappa_max: float | None = 1.5,
        alpha_min: float = 1e-5,
        alpha_max: float | None = 3.0,
        phi_projection: torch.Tensor | None = None,
        phi_scale: torch.Tensor | None = None,
        phi_bias: torch.Tensor | None = None,
        phi_residual_weight: float = 1.0,
    ):
        super().__init__()
        self.law = law
        self.n_genes = n_genes
        self.n_input_features = n_genes if n_input_features is None else n_input_features
        phi_output = 2 * n_genes if law == RELAY else n_genes
        phi_kwargs = dict(
            init_gain=phi_init_gain, activation=activation, init_method=init_method,
            use_spectral_norm=phi_spectral_norm,
        )
        if phi_projection is None:
            self.phi = PhiTheta(self.n_input_features, phi_output, list(phi_dims), **phi_kwargs)
        else:
            if phi_scale is None or phi_bias is None:
                raise ValueError("gene-aware phi needs projection, scale, and bias")
            self.phi = GeneAffineResidualPhi(
                self.n_input_features, phi_output, list(phi_dims), phi_projection,
                phi_scale, phi_bias, residual_weight=phi_residual_weight, **phi_kwargs)
        self.kappa = BoundedPositiveMLP(
            self.n_input_features, 1, list(kappa_dims), activation=activation,
            init_method=init_method,
            min_value=kappa_min, max_value=kappa_max,
        )
        self.g = BoundedPositiveMLP(
            self.n_input_features, n_genes, list(g_dims), activation=activation,
            init_method=init_method,
            min_value=alpha_min, max_value=alpha_max,
        )
        self.gamma_raw = nn.Parameter(torch.rand(n_genes) * 0.1)
        if law == RELAY:
            self.beta_raw = nn.Parameter(torch.rand(n_genes) * 0.1)
        self.softplus = nn.Softplus()

    @property
    def gamma(self) -> torch.Tensor:
        """RNA removal rate, positive."""
        return self.softplus(self.gamma_raw) + 1e-6

    @property
    def beta(self) -> torch.Tensor:
        """Splicing rate, positive. Exists only under the relay law."""
        return self.softplus(self.beta_raw) + 1e-6

    def predict(self, chromatin: torch.Tensor) -> torch.Tensor:
        return self.phi(chromatin)
