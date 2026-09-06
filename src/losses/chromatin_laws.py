"""The two chromatin→RNA kinetic laws, and the residual the JVP is scored against.

The frozen RNA→protein law reads

    J_phi(r) v_r  ~  kappa(r) [alpha(r) (*) S r  -  beta (*) phi(r)]

and the reduced chromatin law is the same sentence with the modalities moved along one
step: chromatin replaces RNA, RNA replaces protein, transcription replaces translation,
G c replaces S r, and RNA removal replaces protein degradation.

  R1 (reduced)  accessible chromatin makes RNA, existing RNA disappears:

      J_phi(c) v_c  ~  kappa(c) [alpha(c) (*) G c  -  gamma (*) phi(c)]

  R2 (relay)    the step R1 skips is that transcription makes UNSPLICED RNA:

      Phi(c) = [u, s]
      du = kappa(c) [alpha(c) (*) G c  -  beta (*) u]
      ds = kappa(c) [beta (*) u        -  gamma (*) s]

No more biology than that. R2 is not a better model because it has more parameters; it is
a different claim about which quantity chromatin drives directly, and the point of running
both is that the same push-forward has to hold under either.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.func import jvp as torch_jvp

from src.losses.entropic_ot import squared_distances

REDUCED = "reduced"
RELAY = "relay"
LAWS = [REDUCED, RELAY]

# §12: R1 gets the full corruption ladder; R2 only the three that still have a
# well-defined relay residual (no reverse/zero/permG until R1 has earned them).
REDUCED_CONDITIONS = ["full", "shuffle", "reverse", "zero", "noDyn", "permG"]
RELAY_CONDITIONS = ["full", "shuffle", "noDyn"]


def conditions_for_law(law: str) -> list[str]:
    """The training corruptions this law is allowed to run."""
    return list(RELAY_CONDITIONS if law == RELAY else REDUCED_CONDITIONS)


def split_us(prediction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The relay prediction's unspliced and spliced halves."""
    n_genes = prediction.shape[1] // 2
    return prediction[:, :n_genes], prediction[:, n_genes:]


def reduced_rhs(prediction: torch.Tensor, kappa: torch.Tensor, alpha: torch.Tensor,
                production_input: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """kappa(c) [alpha(c) (*) G c - gamma (*) phi(c)]."""
    return kappa * (alpha * production_input - gamma * prediction)


def relay_rhs(prediction: torch.Tensor, kappa: torch.Tensor, alpha: torch.Tensor,
              production_input: torch.Tensor, beta: torch.Tensor,
              gamma: torch.Tensor) -> torch.Tensor:
    """The stacked [du, ds] of the relay law."""
    u_hat, s_hat = split_us(prediction)
    production = alpha * production_input
    du = production - beta * u_hat
    ds = beta * u_hat - gamma * s_hat
    return torch.cat([kappa * du, kappa * ds], dim=1)


def law_rhs(model, law: str, prediction: torch.Tensor, chromatin: torch.Tensor,
            production_input: torch.Tensor) -> torch.Tensor:
    """The right-hand side the Jacobian push-forward is compared against."""
    kappa = model.kappa(chromatin)
    alpha = model.g(chromatin)
    if law == REDUCED:
        return reduced_rhs(prediction, kappa, alpha, production_input, model.gamma)
    return relay_rhs(prediction, kappa, alpha, production_input, model.beta, model.gamma)


def kinetics_residual(model, law: str, chromatin: torch.Tensor, velocity: torch.Tensor,
                      production_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """J_phi(c) v_c - RHS, and the prediction, from a single forward-mode pass.

    Forward-mode gives the Jacobian-vector product and phi(c) together, so the law's RHS
    costs no extra forward pass. Needs autograd: never call this under `torch.no_grad()`.
    """
    prediction, pushforward = torch_jvp(model.phi, (chromatin,), (velocity,))
    return pushforward - law_rhs(model, law, prediction, chromatin, production_input), prediction


def law_mask(kinetic_mask, law: str):
    """The per-gene kinetic mask, stacked for the relay law's [u, s] output."""
    stacked = np.concatenate([kinetic_mask, kinetic_mask]) if law == RELAY else kinetic_mask
    return stacked.astype(np.float32)


def kinetics_loss(model, law: str, chromatin: torch.Tensor, velocity: torch.Tensor,
                  production_input: torch.Tensor, confidence: torch.Tensor,
                  scale: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Confidence-weighted mean squared residual over the genes G actually covers.

    Mean over the ACTIVE genes rather than all of them, so lambda_dyn means the same thing
    whatever fraction of the panel chromatin supports, and divided by a per-gene scale so a
    handful of abundant genes do not become the whole loss. `confidence` is the velocity's
    own reliability, so a cell whose direction was barely determined contributes little.

    The mask is not cosmetic: a gene with an all-zero G row has no production term, so its
    residual reduces to J_phi v + kappa*gamma*phi and the only way to satisfy it is to
    flatten phi on that gene. Left in, it would pull the map toward zero on exactly the
    genes chromatin cannot explain.
    """
    residual, _ = kinetics_residual(model, law, chromatin, velocity, production_input)
    per_cell = ((residual / scale) * mask).pow(2).sum(dim=1) / mask.sum().clamp(min=1.0)
    return (confidence * per_cell).mean()


def alignment_gauge(target: torch.Tensor, scale: torch.Tensor, n_probe: int = 1000) -> float:
    """The typical distance between two scaled target cells.

    Sinkhorn's `blur` is an absolute length, so it only means "5% of the way between two
    cells" if the cloud it is applied to has a known diameter. Over a 1281-gene panel the
    unscaled distance between two cells is ~50, and a blur of 0.05 there is exact OT with
    almost no entropic smoothing — the gradient becomes a hard assignment and the
    alignment barely moves. Dividing by this gauge puts every dataset and every panel size
    on one scale, so a single blur setting means the same thing across the experiment.
    """
    rows = torch.randperm(len(target), device=target.device)[:n_probe]
    probe = target[rows] / scale
    return float(squared_distances(probe, probe).sqrt().median().clamp(min=1e-6))


def block_scales(target: torch.Tensor, law: str) -> torch.Tensor:
    """Per-gene scale, computed independently for u and s so blocks weigh equally.

    Two jobs at once, and an earlier version did only the second. Per gene, so a handful
    of abundant genes do not become the whole loss — that is what `kinetics_loss` promises
    and it has to hold under both laws, or lambda_dyn and blur mean different things for
    R1 and R2 and §11's "directly comparable" fails on the optimisation geometry. Per
    block, because unspliced counts are several-fold smaller than spliced and an unscaled
    cost over [u, s] would be almost entirely a cost on s.
    """
    if law == REDUCED:
        return target.std(dim=0, correction=0).clamp(min=1e-3)
    # Scaling every coordinate by its own within-block standard deviation gives every
    # u gene and every s gene unit variance. Since the blocks have the same number of
    # genes, their total squared contribution is then equal. The previous implementation
    # divided these scales by each block's mean; that restored the original between-block
    # magnitude difference and let the larger spliced block dominate again.
    u, s = split_us(target)
    return torch.cat([
        u.std(dim=0, correction=0).clamp(min=1e-3),
        s.std(dim=0, correction=0).clamp(min=1e-3),
    ])
