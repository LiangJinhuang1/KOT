from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

# uniPort uses np.Inf which was removed in NumPy 2.0
if not hasattr(np, "Inf"):
    np.Inf = np.inf

VENDOR_UNIPORT_PATH = Path(__file__).resolve().parents[2] / "vendor" / "uniport"
if VENDOR_UNIPORT_PATH.exists():
    sys.path.insert(0, str(VENDOR_UNIPORT_PATH))

import uniport as up

from src.utils.arrays import to_dense

UNIPORT_INPUT_KEY = "uniport_input"


def minmax01(matrix) -> np.ndarray:
    values = to_dense(matrix, np.float32)
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


def uniport_input_matrix(adata, key: str | None) -> np.ndarray:
    if key is None or key == "X":
        return minmax01(adata.X)
    if key not in adata.obsm:
        available = list(adata.obsm.keys())
        raise ValueError(
            f"uniPort representation '{key}' not found in obsm. Available: {available}"
        )
    return minmax01(adata.obsm[key])


def relabel_obs_names(adata, prefix: str) -> None:
    original_obs_names = adata.obs_names.astype(str).to_numpy()
    adata.obs["original_obs_name"] = original_obs_names
    adata.obs_names = [f"{prefix}_{idx}" for idx in range(adata.n_obs)]


def permuted_adata(adata, seed: int, enabled: bool, relabel_ids: bool):
    if not enabled:
        return adata, None, None
    permutation = np.random.default_rng(seed).permutation(adata.n_obs)
    inverse_permutation = np.argsort(permutation)
    shuffled = adata[permutation].copy()
    if relabel_ids:
        relabel_obs_names(shuffled, "uniport_second")
    return shuffled, permutation, inverse_permutation


def save_permutation(output_dir, permutation: np.ndarray | None) -> None:
    if output_dir is None or permutation is None:
        return
    np.save(Path(output_dir) / "uniport_second_permutation.npy", permutation)


def sinkhorn_coupling(
    x: np.ndarray,
    y: np.ndarray,
    reg: float = 0.1,
    n_iter: int = 200,
) -> np.ndarray:
    x = to_dense(x, np.float32)
    y = to_dense(y, np.float32)
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


def run_uniport(context: dict, cfg: dict) -> tuple[list, np.ndarray | None, dict]:
    """
    uniPort diagonal integration (mode='d'): aligns RNA and protein with no
    shared features, using a VAE + unbalanced OT plan in the latent space.
    The inputs come from the same representation keys used by runner.py, and
    the second modality is permuted by default so diagonal mode remains unpaired.

    Returns
    -------
    aligned  : [rna_latent (n, latent_dim), second_latent (n, latent_dim)]
    coupling : (n, n) OT plan
    """
    rna_adata    = context["rna_adata"].copy()
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
    rna_rep = cfg.get("uniport_rna_rep", cfg.get("rna_repr", "X"))
    second_rep = cfg.get("uniport_second_rep", cfg.get("second_repr", "X"))
    permute_second = bool(cfg.get("uniport_permute_second", True))
    relabel_second_ids = bool(cfg.get("uniport_relabel_second_ids", permute_second))
    second_adata, second_permutation, inverse_second_permutation = permuted_adata(
        context["second_adata"].copy(),
        seed,
        permute_second,
        relabel_second_ids,
    )
    save_permutation(context.get("output_dir"), second_permutation)

    # uniPort requires integer domain_id and string source columns
    rna_adata.obs["domain_id"] = 0
    rna_adata.obs["source"]    = "rna"
    second_adata.obs["domain_id"] = 1
    second_adata.obs["source"]    = second_label
    rna_adata.obsm[UNIPORT_INPUT_KEY] = uniport_input_matrix(rna_adata, rna_rep)
    second_adata.obsm[UNIPORT_INPUT_KEY] = uniport_input_matrix(second_adata, second_rep)
    print(f"[uniport] Input reps: RNA={rna_rep}, {second_label}={second_rep}")
    if inverse_second_permutation is not None:
        print("[uniport] Permuted second modality before diagonal integration.")
    if relabel_second_ids and inverse_second_permutation is not None:
        print("[uniport] Rewrote second modality obs_names for ID-leakage sanity check.")

    with tempfile.TemporaryDirectory(prefix="uniport_") as outdir:
        up.Run(
            adatas=[rna_adata, second_adata],
            mode="d",                              # diagonal: no shared genes
            use_rep=[UNIPORT_INPUT_KEY, UNIPORT_INPUT_KEY],
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
    if inverse_second_permutation is not None:
        second_latent = second_latent[inverse_second_permutation]

    n_entries = rna_latent.shape[0] * second_latent.shape[0]
    if n_entries <= max_coupling_entries:
        coupling = sinkhorn_coupling(
            rna_latent,
            second_latent,
            reg=sinkhorn_reg,
            n_iter=sinkhorn_iter,
        )
    else:
        print(
            "[uniport] Skipping dense final coupling: "
            f"{n_entries:,} entries exceeds uniport_max_coupling_entries="
            f"{max_coupling_entries:,}."
        )
        coupling = None

    diagnostics = {
        "uniport_permute_second": bool(permute_second),
        "uniport_relabel_second_ids": bool(relabel_second_ids),
        "uniport_permutation_seed": int(seed),
        "uniport_rna_rep": str(rna_rep),
        "uniport_second_rep": str(second_rep),
        "uniport_latent_dim": int(n_latent),
        "uniport_iterations": int(iteration),
    }
    return [rna_latent, second_latent], coupling, diagnostics
