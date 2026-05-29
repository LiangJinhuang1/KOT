from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

# uniPort uses np.Inf which was removed in NumPy 2.0
if not hasattr(np, "Inf"):
    np.Inf = np.inf


def _import_uniport():
    try:
        import uniport as up
    except ModuleNotFoundError:
        vendor_path = Path(__file__).resolve().parents[2] / "vendor" / "uniport"
        if vendor_path.exists():
            sys.path.insert(0, str(vendor_path))
            import uniport as up
        else:
            raise
    return up


def _dense_float32(matrix) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def _minmax01(matrix) -> np.ndarray:
    values = _dense_float32(matrix)
    mins = np.nanmin(values, axis=0, keepdims=True)
    maxs = np.nanmax(values, axis=0, keepdims=True)
    denom = maxs - mins
    scaled = np.divide(
        values - mins,
        denom,
        out=np.zeros_like(values, dtype=np.float32),
        where=denom > 0,
    )
    return np.clip(scaled, 0.0, 1.0).astype(np.float32)


def _sinkhorn_coupling(
    x: np.ndarray,
    y: np.ndarray,
    reg: float = 0.1,
    n_iter: int = 200,
) -> np.ndarray:
    x = _dense_float32(x)
    y = _dense_float32(y)
    n, m = x.shape[0], y.shape[0]
    cost = np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1, dtype=np.float32)
    cost_max = float(np.max(cost))
    if cost_max > 0:
        cost = cost / cost_max

    kernel = np.exp(-cost / max(reg, 1e-8)).astype(np.float32)
    kernel = np.maximum(kernel, 1e-30)
    u = np.ones(n, dtype=np.float32) / n
    a = np.ones(n, dtype=np.float32) / n
    b = np.ones(m, dtype=np.float32) / m
    for _ in range(n_iter):
        v = b / np.maximum(kernel.T @ u, 1e-30)
        u = a / np.maximum(kernel @ v, 1e-30)
    return (u[:, None] * kernel * v[None, :]).astype(np.float32)


def run_uniport(context: dict, cfg: dict) -> tuple[list, np.ndarray | None]:
    """
    uniPort diagonal integration (mode='d'): aligns RNA and protein with no
    shared features, using a VAE + unbalanced OT plan in the latent space.

    Returns
    -------
    aligned  : [rna_latent (n, latent_dim), second_latent (n, latent_dim)]
    coupling : (n, n) OT plan
    """
    rna_adata    = context["rna_adata"].copy()
    second_adata = context["second_adata"].copy()
    second_label = context["second_label"]

    lambda_s   = float(cfg.get("lambda_s",   0.5))
    lambda_kl  = float(cfg.get("lambda_kl",  0.5))
    lambda_ot  = float(cfg.get("lambda_ot",  1.0))
    reg        = float(cfg.get("reg",        0.1))
    reg_m      = float(cfg.get("reg_m",      1.0))
    iteration  = int(cfg.get("iteration",    30000))
    batch_size = int(cfg.get("batch_size",   256))
    lr         = float(cfg.get("lr",         2e-4))
    seed       = int(cfg.get("seed",         42))
    n_latent   = int(cfg.get("uniport_n_latent", cfg.get("n_latent", 16)))
    hidden_dim = int(cfg.get("uniport_hidden_dim", 1024))
    sinkhorn_reg = float(cfg.get("uniport_sinkhorn_reg", cfg.get("sinkhorn_reg", reg)))
    sinkhorn_iter = int(cfg.get("uniport_sinkhorn_iter", 200))
    max_coupling_entries = int(cfg.get("uniport_max_coupling_entries", 25_000_000))

    # uniPort requires integer domain_id and string source columns
    rna_adata.obs["domain_id"] = 0
    rna_adata.obs["source"]    = "rna"
    second_adata.obs["domain_id"] = 1
    second_adata.obs["source"]    = second_label
    rna_adata.obsm["_uniport_scaled"] = _minmax01(rna_adata.X)
    second_adata.obsm["_uniport_scaled"] = _minmax01(second_adata.X)

    outdir = tempfile.mkdtemp(prefix="uniport_")
    up = _import_uniport()

    up.Run(
        adatas=[rna_adata, second_adata],
        mode="d",                              # diagonal: no shared genes
        use_rep=["_uniport_scaled", "_uniport_scaled"],
        out="latent",
        save_OT=False,
        lambda_s=lambda_s,
        lambda_kl=lambda_kl,
        lambda_ot=lambda_ot,
        reg=reg,
        reg_m=reg_m,
        iteration=iteration,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        enc=[["fc", hidden_dim, 1, "relu"], ["fc", n_latent, "", ""]],
        loss_type="MSE",
        outdir=outdir,
        verbose=False,
    )

    rna_latent    = np.array(rna_adata.obsm["latent"])
    second_latent = np.array(second_adata.obsm["latent"])
    n_entries = rna_latent.shape[0] * second_latent.shape[0]
    if n_entries <= max_coupling_entries:
        coupling = _sinkhorn_coupling(
            rna_latent,
            second_latent,
            reg=sinkhorn_reg,
            n_iter=sinkhorn_iter,
        )
    else:
        print(
            "[uniPort] Skipping dense final coupling: "
            f"{n_entries:,} entries exceeds uniport_max_coupling_entries="
            f"{max_coupling_entries:,}."
        )
        coupling = None

    return [rna_latent, second_latent], coupling
