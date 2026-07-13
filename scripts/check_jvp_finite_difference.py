#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.KOT import KOTModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare torch.func.jvp against central finite differences for KOT phi."
    )
    parser.add_argument("--eps", type=float, nargs="+", default=[1e-2, 1e-3, 1e-4])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--d-rna", type=int, default=30)
    parser.add_argument("--d-protein", type=int, default=10)
    parser.add_argument("--phi-dims", type=int, nargs="+", default=[128, 64])
    parser.add_argument("--activation", type=str, default="silu")
    parser.add_argument("--init", type=str, default="xavier")
    parser.add_argument("--phi-init-gain", type=float, default=0.5)
    parser.add_argument("--spectral-norm", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, choices=["float32", "float64"], default="float64")
    parser.add_argument("--fail-rel-median", type=float, default=None)
    return parser.parse_args()


def tensor_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    return torch.float64


def build_model(args: argparse.Namespace, dtype: torch.dtype, device: torch.device) -> KOTModel:
    torch.manual_seed(args.seed)
    model = KOTModel(
        args.d_rna,
        args.d_protein,
        phi_dims=args.phi_dims,
        kappa_dims=[16],
        g_dims=[16],
        phi_init_gain=args.phi_init_gain,
        activation=args.activation,
        init_method=args.init,
        phi_spectral_norm=args.spectral_norm,
    )
    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model


def normalized_directions(batch_size: int, d_rna: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    direction = torch.randn(batch_size, d_rna, dtype=dtype, device=device)
    scale = direction.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return direction / scale


def compute_jvp(phi, r: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if hasattr(torch, "func") and hasattr(torch.func, "jvp"):
        return torch.func.jvp(phi, (r,), (v,))
    return torch.autograd.functional.jvp(phi, r, v, create_graph=False, strict=False)


def summarize_error(jvp_value: torch.Tensor, fd_value: torch.Tensor) -> dict[str, float]:
    diff = fd_value - jvp_value
    abs_err = diff.norm(dim=1)
    denom = jvp_value.norm(dim=1).clamp_min(1e-12)
    rel_err = abs_err / denom
    cos = F.cosine_similarity(fd_value, jvp_value, dim=1, eps=1e-12)
    return {
        "abs_mean": float(abs_err.mean().detach().cpu()),
        "abs_max": float(abs_err.max().detach().cpu()),
        "rel_mean": float(rel_err.mean().detach().cpu()),
        "rel_median": float(rel_err.median().detach().cpu()),
        "rel_max": float(rel_err.max().detach().cpu()),
        "cos_mean": float(cos.mean().detach().cpu()),
        "cos_min": float(cos.min().detach().cpu()),
        "jvp_norm_mean": float(jvp_value.norm(dim=1).mean().detach().cpu()),
        "fd_norm_mean": float(fd_value.norm(dim=1).mean().detach().cpu()),
    }


def print_table(rows: list[dict[str, float]]) -> None:
    header = (
        "eps        rel_mean    rel_median  rel_max     abs_mean    "
        "cos_mean   cos_min    jvp_norm   fd_norm"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['eps']:<10.1e}"
            f"{row['rel_mean']:<12.3e}"
            f"{row['rel_median']:<12.3e}"
            f"{row['rel_max']:<12.3e}"
            f"{row['abs_mean']:<12.3e}"
            f"{row['cos_mean']:<11.6f}"
            f"{row['cos_min']:<11.6f}"
            f"{row['jvp_norm_mean']:<11.3e}"
            f"{row['fd_norm_mean']:<11.3e}"
        )


def main() -> None:
    args = parse_args()
    dtype = tensor_dtype(args.dtype)
    device = torch.device(args.device)

    model = build_model(args, dtype, device)
    torch.manual_seed(args.seed + 1)
    r = torch.randn(args.batch_size, args.d_rna, dtype=dtype, device=device)
    v = normalized_directions(args.batch_size, args.d_rna, dtype, device)

    phi_r, jvp_value = compute_jvp(model.phi, r, v)
    rows = []
    for eps in args.eps:
        fd_value = (model.phi(r + eps * v) - model.phi(r - eps * v)) / (2.0 * eps)
        row = summarize_error(jvp_value, fd_value)
        row["eps"] = eps
        rows.append(row)

    print("JVP finite-difference check for KOT phi")
    print(f"dtype={args.dtype} device={device} batch={args.batch_size}")
    print(f"phi_r mean={float(phi_r.mean().detach().cpu()):.6e}")
    print_table(rows)

    if args.fail_rel_median is not None:
        best_median = min(row["rel_median"] for row in rows)
        if best_median > args.fail_rel_median:
            raise SystemExit(
                f"JVP finite-difference check failed: best rel_median={best_median:.3e} "
                f"> {args.fail_rel_median:.3e}"
            )


if __name__ == "__main__":
    main()
