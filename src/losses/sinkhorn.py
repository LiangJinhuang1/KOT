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
    """Unbiased S = OT(x,y) − OT(x,x)/2 − OT(y,y)/2.

    tensorized is O(N²) GPU PyTorch. online/multiscale need KeOps, which often
    cannot load CUDA and then runs on CPU — slower, not faster, on a GPU batch.
    """
    key = (blur, backend)
    if key not in LOSS_CACHE:
        LOSS_CACHE[key] = SamplesLoss("sinkhorn", blur=blur, backend=backend)
    return LOSS_CACHE[key](x, y)
