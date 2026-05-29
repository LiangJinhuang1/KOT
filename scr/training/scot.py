import numpy as np

from scr.models.SCOT import SCOTv2


def run_scot(context: dict, cfg: dict) -> tuple[list, np.ndarray]:
    x, y = context["x"], context["y"]
    scot = SCOTv2([x, y])
    aligned = scot.align(
        normalize=bool(cfg.get("normalize", False)),
        norm=str(cfg.get("norm", "l2")),
        k=int(cfg.get("k", 20)),
        mode=str(cfg.get("mode", "connectivity")),
        metric=str(cfg.get("metric", "correlation")),
        eps=float(cfg.get("eps", 0.01)),
        rho=float(cfg.get("rho", 1.0)),
        projMethod=str(cfg.get("proj_method", "barycentric")),
        Lambda=float(cfg.get("lambda", 1.0)),
        out_dim=int(cfg.get("out_dim", 30)),
        nits_plan=int(cfg.get("nits_plan", 500)),
        nits_sinkhorn=int(cfg.get("nits_sinkhorn", 500)),
    )
    coupling = scot.couplings[0].numpy()
    return aligned, coupling
