"""The chromatin KOT network: the frozen KOT building blocks, wired for a second rate.

`src/models/KOT.py` is imported unchanged — the same PhiTheta and the same bounded
positive head that the RNA→protein runs use, so a difference between the two experiments
cannot come from a different architecture.

One thing genuinely differs. The RNA→protein law needs a single per-protein rate (beta,
degradation). The relay law needs two per-gene rates, and they mean different things:

    beta   splicing:    unspliced -> spliced
    gamma  degradation: spliced -> nothing

Both rates are per-gene parameters. The transcription head receives Gc as regulatory
input and produces alpha directly; chromatin is not a molecular substrate in the ODE.

Every other constructor default matches `config/training.yaml`, which is where a run
actually reads them from — the defaults here only make the class usable on its own.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.losses.chromatin_laws import RELAY
from src.models.KOT import BoundedPositiveMLP, PhiTheta

# How much of the nonlinear path reaches the output, and whether that is learned.
PHI_GATES = ["scalar-zero", "none", "per-gene"]
# What the transcription head reads: the gene's own activity through G, or all of c.
ALPHA_INPUTS = ["gc", "full"]


def residual_gate_init(scale: torch.Tensor) -> torch.Tensor:
    """Per-gene gate, opened widest where the affine path cannot carry the gene.

    `gene_affine_calibration` leaves scale == 0 wherever a marginal could not be matched
    — 207 of 992 BMMC outputs — and those genes are a constant bias plus whatever the
    network adds. They start nearly open; a gene that already has a usable affine path
    starts near half, so the warm start survives where it means something.
    """
    return torch.where(scale == 0, 1.5, 0.5)


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
                 residual_weight: float = 1.0, gate: str = "scalar-zero",
                 trainable_projection: bool = False, **kwargs):
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
        # G is normally structural: a fixed, curated peak->gene correspondence. Making it
        # trainable tests the one architectural thing scGLUE does differently -- its
        # guidance graph is a soft prior that training updates, not a frozen matrix -- and
        # the velocity ceiling says the frozen push carries almost no direction signal.
        if trainable_projection:
            self.gene_projection = nn.Parameter(projection.detach().clone().float())
        else:
            self.register_buffer("gene_projection", projection.detach().clone().float())
        self.gene_scale = nn.Parameter(scale.detach().clone().float())
        self.gene_bias = nn.Parameter(bias.detach().clone().float())
        self.residual_weight = float(residual_weight)
        self.gate = gate
        if gate == "scalar-zero":
            self.residual_gate = nn.Parameter(torch.zeros(()))
        elif gate == "per-gene":
            self.residual_gate = nn.Parameter(residual_gate_init(scale))

    def residual_scale(self) -> torch.Tensor | float:
        """How much of the network reaches the output.

        "scalar-zero" multiplies the network by tanh(0) = 0 at step 1, which also zeroes
        the gradient on every network weight; the gate then only moves if the output of a
        still-random network happens to help, so it does not. Measured over 500 BMMC
        epochs it ended at -0.077, meaning the nonlinear path trained throttled by ~12x
        for the whole run while the affine path had full gradient from the start. The
        other two modes exist to measure that: "none" removes the switch, "per-gene"
        starts it open and lets each gene decide.
        """
        if self.gate == "none":
            return self.residual_weight
        return self.residual_weight * torch.tanh(self.residual_gate)

    def forward(self, chromatin: torch.Tensor) -> torch.Tensor:
        projected = chromatin @ self.gene_projection.T
        base = projected * self.gene_scale + self.gene_bias
        return base + self.residual_scale() * self.net(chromatin)


class ChromatinKOT(nn.Module):
    """phi: chromatin → [unspliced, spliced], with kappa, alpha, beta, gamma."""

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
        alpha_max: float | None = None,
        phi_projection: torch.Tensor | None = None,
        phi_scale: torch.Tensor | None = None,
        phi_bias: torch.Tensor | None = None,
        phi_residual_weight: float = 1.0,
        phi_gate: str = "scalar-zero",
        phi_trainable_projection: bool = False,
        alpha_input: str = "gc",
    ):
        super().__init__()
        if law != RELAY:
            raise ValueError("The reduced chromatin law is retired; use relay R2.")
        self.law = law
        self.n_genes = n_genes
        self.n_input_features = n_genes if n_input_features is None else n_input_features
        phi_output = 2 * n_genes
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
                phi_scale, phi_bias, residual_weight=phi_residual_weight,
                gate=phi_gate, trainable_projection=phi_trainable_projection,
                **phi_kwargs)
        self.kappa = BoundedPositiveMLP(
            self.n_input_features, 1, list(kappa_dims), activation=activation,
            init_method=init_method,
            min_value=kappa_min, max_value=kappa_max,
        )
        # alpha's input width. G is a 0/1 DIAGONAL selection matrix, so `Gc` is just each
        # gene's own activity and the whole right-hand side becomes per-gene: RHS_g depends
        # on c_g alone. A paired ridge that mixes across genes predicts scVelo velocity at
        # ~7.5x its null from c; the per-gene form cannot express that map at all. "full"
        # hands alpha the whole chromatin vector so it can.
        self.alpha_input = alpha_input
        alpha_width = self.n_input_features if alpha_input == "full" else n_genes
        self.g = BoundedPositiveMLP(
            alpha_width, n_genes, list(g_dims), activation=activation,
            init_method=init_method,
            min_value=alpha_min, max_value=alpha_max,
        )
        self.gamma_raw = nn.Parameter(torch.rand(n_genes) * 0.1)
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
