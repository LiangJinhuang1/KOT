"""Regulatory R2: chromatin determines transcription rate, not substrate abundance.

    du/dt = kappa(c) [alpha(Gc) - beta*u]
    ds/dt = kappa(c) [beta*u - gamma*s]

Both RNA states use a common, fixed per-cell library factor. Phi predicts their
log1p values, so the linear RHS is chain-ruled before comparison with J_phi(c) v_c.
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


def conditions_for_law(law: str) -> list[str]:
    if law != RELAY:
        raise ValueError("The reduced chromatin law is retired; use relay R2.")
    return list(RELAY_CONDITIONS)


def split_us(prediction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Internal layout remains [u, s]; alignment selects the s half."""
    if prediction.ndim != 2 or prediction.shape[1] % 2:
        raise ValueError("R2 requires equally sized unspliced and spliced blocks")
    return prediction.chunk(2, dim=1)


def law_coordinates(log_values: torch.Tensor) -> torch.Tensor:
    """Bound the law's view to CP10K support; alignment sees raw predictions."""
    return log_values.clamp(min=LOG1P_MIN, max=LOG1P_MAX)


def linear_abundance(log_values: torch.Tensor) -> torch.Tensor:
    return torch.expm1(law_coordinates(log_values))


def linear_rate_to_log1p(linear_rate: torch.Tensor, log_values: torch.Tensor) -> torch.Tensor:
    return linear_rate * torch.exp(-law_coordinates(log_values))


def relay_rhs(prediction: torch.Tensor, kappa: torch.Tensor, alpha: torch.Tensor,
              beta: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """Shared-unit splicing equations; alpha is already a transcription rate."""
    u_hat, s_hat = split_us(prediction)
    u, s = linear_abundance(u_hat), linear_abundance(s_hat)
    return torch.cat([
        kappa * linear_rate_to_log1p(alpha - beta * u, u_hat),
        kappa * linear_rate_to_log1p(beta * u - gamma * s, s_hat),
    ], dim=1)


def law_rhs(model, law: str, prediction: torch.Tensor, chromatin: torch.Tensor,
            production_input: torch.Tensor) -> torch.Tensor:
    """G selects regulatory inputs to alpha; it is never an ODE multiplier."""
    if law != RELAY:
        raise ValueError("The reduced chromatin law is retired; use relay R2.")
    return relay_rhs(prediction, model.kappa(chromatin), model.g(production_input),
                     model.beta, model.gamma)


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


def kinetics_loss(model, law: str, chromatin: torch.Tensor, velocity: torch.Tensor,
                  production_input: torch.Tensor, confidence: torch.Tensor,
                  scale: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Confidence-weighted residual over supported genes, scaled by RNA train spread.

    A transcription-head bias is not evidence for an unsupported G row.
    """
    residual, _ = kinetics_residual(model, law, chromatin, velocity, production_input)
    per_cell = ((residual / scale) * mask).pow(2).sum(dim=1) / mask.sum().clamp(min=1.0)
    return (confidence * per_cell).mean()


def alignment_gauge(target: torch.Tensor, scale: torch.Tensor, n_probe: int = 1000) -> float:
    """Typical scaled target distance; fit on RNA training cells only."""
    rows = torch.randperm(len(target), device=target.device)[:n_probe]
    probe = target[rows] / scale
    return float(squared_distances(probe, probe).sqrt().median().clamp(min=1e-6))


def block_scales(target: torch.Tensor, law: str) -> torch.Tensor:
    """Each gene's RNA training standard deviation, independently in each block."""
    return target.std(dim=0, correction=0).clamp(min=1e-3)
