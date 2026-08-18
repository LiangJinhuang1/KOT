"""
Unbiased Sinkhorn divergence via geomloss.

Install: pip install geomloss pykeops
"""
import torch
from geomloss import SamplesLoss


LOSS_CACHE: dict[tuple[float, str], SamplesLoss] = {}


def sinkhorn_divergence(
    x: torch.Tensor, y: torch.Tensor, blur: float = 0.05, backend: str = "auto"
) -> torch.Tensor:
    """
    Unbiased Sinkhorn divergence S_blur(x, y).

    S(x,y) = OT(x,y) - OT(x,x)/2 - OT(y,y)/2
    Zero iff the two empirical distributions are identical.

    Parameters
    ----------
    x    : (n, d) predicted protein embeddings φ_θ(r)
    y    : (m, d) observed protein measurements
    blur : entropic regularisation (= sqrt(ε) in geomloss convention)
    backend : geomloss compute backend.
        "tensorized" — stores the full n×m cost matrix: O(N²) memory, pure PyTorch,
                       only scales to ~5k points per measure.
        "online"     — KeOps map-reduce, O(N) memory, works in any dimension.
        "multiscale" — KeOps block-sparse, O(N) but only efficient in 1–3D.
        "auto"       — geomloss heuristic: tensorized when small, online when large.
        The linear backends ("online"/"multiscale") need pykeops.
    """
    key = (blur, backend)
    if key not in LOSS_CACHE:
        LOSS_CACHE[key] = SamplesLoss("sinkhorn", blur=blur, backend=backend)
    return LOSS_CACHE[key](x, y)
