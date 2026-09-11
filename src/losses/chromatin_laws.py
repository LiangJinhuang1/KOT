"""Regulatory R2: chromatin determines transcription rate, not substrate abundance.

    du/dt = kappa(c) [alpha(Gc) - beta*u]
    ds/dt = kappa(c) [beta*u - gamma*s]

kappa is a per-cell clock: it scales the splicing flux without changing its
direction. A scaled L2 residual can therefore be driven to ~0 by shrinking
kappa without J_phi(c) v_c ever pointing the same way as the ODE. Training
matches the kappa-free flux to the push-forward; kappa is not a switch that
turns the law off.

Both RNA states use a common library convention. The default predicts log1p of
shared CP10K abundances and chain-rules the linear RHS before comparing it with
J_phi(c) v_c. That chain rule is not the same as putting the cell-dependent
library map \(\tilde x = x/T\) through the ODE: the missing term is
\(\dot{\tilde x} = \dot x/T - \tilde x\,\dot T/T\). Panel-shared CP10K plus that
compositional term, and globally scaled linear counts, are explicit alternatives.
The primary transport output is s; u is an auxiliary kinetic state. The reduced
law and the former multiplicative relay are no longer executable.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.func import jvp as torch_jvp

from src.losses.entropic_ot import squared_distances

# Single-block label retained for historical state-only analysis utilities.
REDUCED = "reduced"
RELAY = "relay"
LAWS = [RELAY]
RELAY_CONDITIONS = ["full", "shuffle", "noDyn"]
FORMULATION = "regulatory_r2_shared_rna_v1"
LOG1P_MIN = 0.0
LOG1P_MAX = float(np.log1p(1e4))
CP10K_TARGET = 1.0e4
SHARED_LOG1P = "shared_log1p"
PANEL_LOG1P_COMPOSITIONAL = "panel_log1p_compositional"
GLOBAL_LINEAR = "global_linear"
RNA_KINETIC_COORDS = (SHARED_LOG1P, PANEL_LOG1P_COMPOSITIONAL, GLOBAL_LINEAR)


def conditions_for_law(law: str) -> list[str]:
    if law != RELAY:
        raise ValueError("The reduced chromatin law is retired; use relay R2.")
    return list(RELAY_CONDITIONS)


def split_us(prediction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Internal layout remains [u, s]; alignment selects the s half."""
    if prediction.ndim != 2 or prediction.shape[1] % 2:
        raise ValueError("R2 requires equally sized unspliced and spliced blocks")
    return prediction.chunk(2, dim=1)


def resolve_kinetic_coords(name: str | None) -> str:
    coords = SHARED_LOG1P if name in (None, "") else name
    if coords not in RNA_KINETIC_COORDS:
        raise ValueError(
            f"rna_kinetic_coords must be one of {RNA_KINETIC_COORDS}, got {name!r}")
    return coords


def coords_use_log1p(coords: str | None) -> bool:
    return resolve_kinetic_coords(coords) != GLOBAL_LINEAR


def coords_use_compositional(coords: str | None) -> bool:
    return resolve_kinetic_coords(coords) == PANEL_LOG1P_COMPOSITIONAL


def model_kinetic_coords(model) -> str:
    return resolve_kinetic_coords(getattr(model, "kinetic_coords", None))


def law_coordinates(values: torch.Tensor, coords: str | None = None) -> torch.Tensor:
    """Bound the law's view; alignment sees raw predictions."""
    if coords_use_log1p(coords):
        return values.clamp(min=LOG1P_MIN, max=LOG1P_MAX)
    return values.clamp(min=0.0)


def linear_abundance(log_values: torch.Tensor) -> torch.Tensor:
    """Linear CP10K abundance from a log1p prediction. Shared-log1p convention."""
    return torch.expm1(law_coordinates(log_values, SHARED_LOG1P))


def law_abundance(prediction: torch.Tensor, coords: str | None = None) -> torch.Tensor:
    bounded = law_coordinates(prediction, coords)
    if coords_use_log1p(coords):
        return torch.expm1(bounded)
    return bounded


def linear_rate_to_log1p(linear_rate: torch.Tensor, log_values: torch.Tensor) -> torch.Tensor:
    return linear_rate * torch.exp(-law_coordinates(log_values, SHARED_LOG1P))


def linear_rate_to_state(linear_rate: torch.Tensor, prediction: torch.Tensor,
                         coords: str | None = None) -> torch.Tensor:
    """dx/dt in linear abundance → the coordinate phi actually predicts."""
    if coords_use_log1p(coords):
        return linear_rate * torch.exp(-law_coordinates(prediction, coords))
    return linear_rate


def state_rate_to_linear(rate, prediction, coords: str | None = None):
    """Coordinate rate → linear abundance rate, for scVelo comparison."""
    if coords_use_log1p(coords):
        return linear_from_log1p_rate(rate, prediction)
    return rate


def linear_from_log1p_rate(rate, log_values):
    """A log1p-space rate as a LINEAR one: dx/dt = (dy/dt)(1 + x) = (dy/dt) e^y.

    scVelo reports ds/dt in linear units while phi's JVP is a log1p rate, so one side has
    to move before any comparison. Converting the model side is the exact direction --
    scVelo's library normalisation differs from CP10K by a per-cell scalar, which is
    gene-independent and cancels in a per-cell cosine, whereas converting the reference
    would need scVelo's own Ms, which is not stored.

    This lives here, and is called by BOTH scoring paths, because it previously existed
    only inside `reference_agreement`: `evaluate` compared the raw log1p rate against the
    linear reference and reported -0.003 where the preflight reported +0.164 on the same
    checkpoint. Two scorings of one quantity must not be able to disagree.
    """
    import numpy as np_module
    if isinstance(rate, torch.Tensor):
        return rate * torch.exp(log_values.clamp(min=LOG1P_MIN, max=LOG1P_MAX))
    return rate * np_module.exp(np_module.clip(log_values, LOG1P_MIN, LOG1P_MAX))


def relay_flux(prediction: torch.Tensor, alpha: torch.Tensor,
               beta: torch.Tensor, gamma: torch.Tensor,
               coords: str | None = None) -> torch.Tensor:
    """Splicing/transcription direction before the per-cell clock.

    kappa multiplies this vector in `relay_rhs`. Direction losses must read the
    flux, not the clocked RHS: a positive per-cell kappa cannot change the angle,
    and a kappa near zero makes cosine(J_phi v, kappa * flux) undefined.

    Under panel-shared CP10K the library sum of modeled genes is the constant
    CP10K_TARGET, so \(\dot T/T = \kappa/c \sum(\alpha - \gamma s)\) is a common
    compositional mode. It belongs in the kappa-free flux: it changes direction,
    not only scale.
    """
    coords = resolve_kinetic_coords(coords)
    u_hat, s_hat = split_us(prediction)
    u, s = law_abundance(u_hat, coords), law_abundance(s_hat, coords)
    du = alpha - beta * u
    ds = beta * u - gamma * s
    if coords_use_compositional(coords):
        dlog_library = (alpha - gamma * s).sum(dim=1, keepdim=True) / CP10K_TARGET
        du = du - u * dlog_library
        ds = ds - s * dlog_library
    return torch.cat([
        linear_rate_to_state(du, u_hat, coords),
        linear_rate_to_state(ds, s_hat, coords),
    ], dim=1)


def relay_rhs(prediction: torch.Tensor, kappa: torch.Tensor, alpha: torch.Tensor,
              beta: torch.Tensor, gamma: torch.Tensor,
              coords: str | None = None) -> torch.Tensor:
    """Shared-unit splicing equations; alpha is already a transcription rate."""
    return kappa * relay_flux(prediction, alpha, beta, gamma, coords=coords)


def law_rhs(model, law: str, prediction: torch.Tensor, chromatin: torch.Tensor,
            production_input: torch.Tensor) -> torch.Tensor:
    """G selects regulatory inputs to alpha; it is never an ODE multiplier."""
    if law != RELAY:
        raise ValueError("The reduced chromatin law is retired; use relay R2.")
    return relay_rhs(prediction, model.kappa(chromatin), model.g(production_input),
                     model.beta, model.gamma, coords=model_kinetic_coords(model))


def kinetics_residual(model, law: str, chromatin: torch.Tensor, velocity: torch.Tensor,
                      production_input: torch.Tensor
                      ) -> tuple[torch.Tensor, torch.Tensor]:
    """Single forward-mode pass; retain autograd for training through the JVP."""
    prediction, pushforward = torch_jvp(model.phi, (chromatin,), (velocity,))
    return pushforward - law_rhs(model, law, prediction, chromatin, production_input), prediction


def law_mask(kinetic_mask, law: str):
    if law != RELAY:
        raise ValueError("The reduced chromatin law is retired; use relay R2.")
    return np.concatenate([kinetic_mask, kinetic_mask]).astype(np.float32)


def kinetics_losses(model, law: str, chromatin: torch.Tensor, velocity: torch.Tensor,
                    production_input: torch.Tensor, confidence: torch.Tensor,
                    scale: torch.Tensor, mask: torch.Tensor,
                    residual_weight: float = 1.0, direction_weight: float = 0.0
                    ) -> dict[str, torch.Tensor]:
    """Confidence-weighted residual and kappa-free direction terms.

    `residual` is the historical scaled L2 of J_phi v - kappa * flux. Shrinking
    kappa (and the Jacobian) drives it down even when the two vectors are
    orthogonal. `direction` is 1 - cosine(J_phi v, flux) after the same per-gene
    scale and support mask; a per-cell kappa cannot change it, and collapsing
    both sides to ~0 leaves cosine ~0 so the term stays large.

    A transcription-head bias is not evidence for an unsupported G row.
    """
    if residual_weight < 0 or direction_weight < 0:
        raise ValueError("kinetics residual_weight and direction_weight must be >= 0")
    if residual_weight == 0.0 and direction_weight == 0.0:
        raise ValueError("kinetics loss needs a positive residual_weight or direction_weight")
    if law != RELAY:
        raise ValueError("The reduced chromatin law is retired; use relay R2.")
    prediction, pushforward = torch_jvp(model.phi, (chromatin,), (velocity,))
    flux = relay_flux(prediction, model.g(production_input), model.beta, model.gamma,
                      coords=model_kinetic_coords(model))
    residual = pushforward - model.kappa(chromatin) * flux
    n_active = mask.sum().clamp(min=1.0)
    per_cell = ((residual / scale) * mask).pow(2).sum(dim=1) / n_active
    residual_term = (confidence * per_cell).mean()
    cosine = torch.nn.functional.cosine_similarity(
        (pushforward / scale) * mask, (flux / scale) * mask, dim=1, eps=1e-8)
    direction_term = (confidence * (1.0 - cosine)).mean()
    total = residual_weight * residual_term + direction_weight * direction_term
    return {"total": total, "residual": residual_term, "direction": direction_term}


def kinetics_loss(model, law: str, chromatin: torch.Tensor, velocity: torch.Tensor,
                  production_input: torch.Tensor, confidence: torch.Tensor,
                  scale: torch.Tensor, mask: torch.Tensor,
                  residual_weight: float = 1.0, direction_weight: float = 0.0
                  ) -> torch.Tensor:
    """Scalar used by the trainer; see `kinetics_losses` for the two terms."""
    return kinetics_losses(
        model, law, chromatin, velocity, production_input, confidence, scale, mask,
        residual_weight=residual_weight, direction_weight=direction_weight)["total"]


def alignment_gauge(target: torch.Tensor, scale: torch.Tensor, n_probe: int = 1000) -> float:
    """Typical scaled target distance; fit on RNA training cells only."""
    rows = torch.randperm(len(target), device=target.device)[:n_probe]
    probe = target[rows] / scale
    return float(squared_distances(probe, probe).sqrt().median().clamp(min=1e-6))


def block_scales(target: torch.Tensor, law: str) -> torch.Tensor:
    """Each gene's RNA training standard deviation, independently in each block."""
    return target.std(dim=0, correction=0).clamp(min=1e-3)
