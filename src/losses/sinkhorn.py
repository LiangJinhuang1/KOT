"""
Unbiased Sinkhorn divergence via geomloss.

Install: pip install geomloss pykeops
"""
import torch
from geomloss import SamplesLoss


_loss_cache: dict[float, SamplesLoss] = {}


def sinkhorn_divergence(x: torch.Tensor, y: torch.Tensor, blur: float = 0.05) -> torch.Tensor:
    """
    Unbiased Sinkhorn divergence S_blur(x, y).

    S(x,y) = OT(x,y) - OT(x,x)/2 - OT(y,y)/2
    Zero iff the two empirical distributions are identical.

    Parameters
    ----------
    x    : (n, d) predicted protein embeddings φ_θ(r)
    y    : (m, d) observed protein measurements
    blur : entropic regularisation (= sqrt(ε) in geomloss convention)
    """
    if blur not in _loss_cache:
        _loss_cache[blur] = SamplesLoss("sinkhorn", blur=blur, backend="tensorized")
    return _loss_cache[blur](x, y)
