#!/usr/bin/env python3
"""Why does alignment learn while the kinetics do not? Measure phi's Jacobian.

The two objectives ask phi for different things, and the thing that makes the first one
work is what makes the second impossible:

  ALIGNMENT constrains phi's VALUES. It is satisfied by any map whose output cloud sits
  where the RNA cloud sits, and a per-gene affine map fitted from the two marginals
  already does that. Hence FOSCTTM 0.21 from a calibration with zero gradient steps.

  KINETICS constrains phi's DERIVATIVE. J_phi(c) v_c has to equal a right-hand side that
  varies per cell through kappa(c), alpha(Gc) and u(c).

For an AFFINE phi -- which is what these runs converge to, the network contributing under
3% of the output -- the Jacobian is a CONSTANT matrix M = diag(gene_scale) G. The
pushforward is then M v_c: a fixed linear function of the velocity, carrying no per-cell
modulation of its own. Every cell's predicted rate is determined by its velocity alone.
So the kinetics term is being asked to match a cell-varying target with a map that
structurally cannot vary per cell except through v_c.

This tool measures three things that decide whether that is the explanation:

  constant_jacobian_ceiling  the best cosine ANY constant-Jacobian phi could reach, found
                             by least-squares fitting one matrix M from v_c to the flux
                             WITH the pairing. If the trained cosine is already near this,
                             the affine phi is at its structural limit and only
                             nonlinearity can help.
  jacobian_variation         how much J_phi(c) v actually varies across cells for a fixed
                             probe direction v. ~0 confirms phi is affine in practice.
  flux_variation             how much the target flux varies across cells, i.e. how much
                             variation the pushforward would have to produce.

Reads the whole dataset; submit it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.chromatin_common import context, load_run, run_transform

import run_kot_chromatin as runner
from src.data.chromatin import chromatin_features
from src.losses.chromatin_laws import RELAY, relay_flux

RESULTS = PROJECT_ROOT / "cache" / "results" / "chromatin"


def row_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = (left * right).sum(axis=1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return numerator / np.clip(denominator, 1e-12, None)


def constant_jacobian_ceiling(velocity: np.ndarray, flux: np.ndarray,
                              train: np.ndarray, test: np.ndarray,
                              ridge: float = 1.0) -> dict:
    """Best cosine reachable by ANY single matrix M with pushforward M v_c.

    Fitted WITH the pairing, so it is an upper bound no unpaired affine phi can beat. If
    the trained model sits at this number, the ceiling is the architecture and not the
    optimiser.
    """
    x, y = velocity[train], flux[train]
    gram = x.T @ x + ridge * np.eye(x.shape[1], dtype=x.dtype)
    weights = np.linalg.solve(gram, x.T @ y)
    predicted = velocity[test] @ weights
    cosine = row_cosine(predicted, flux[test])
    return {"cosine_median": float(np.median(cosine)),
            "cosine_mean": float(np.mean(cosine)),
            "n_train": int(len(train)), "n_test": int(len(test))}


def diagonal_jacobian_ceiling(velocity: np.ndarray, flux: np.ndarray,
                              train: np.ndarray, test: np.ndarray) -> dict:
    """Best cosine reachable by a DIAGONAL Jacobian, which is what phi actually has.

    G is a 0/1 diagonal selection matrix -- one activity feature per gene, 719 of 2000
    genes with none -- so phi's affine Jacobian is diag(gene_scale * mask) and the
    pushforward is an elementwise rescaling of v_c. Fitting the best per-gene scalar
    m_j = sum_i v_ij flux_ij / sum_i v_ij^2 gives the ceiling that architecture allows.
    The gap between this and the dense ceiling is the price of G being diagonal.
    """
    x, y = velocity[train], flux[train]
    width = min(x.shape[1], y.shape[1])
    scale = (x[:, :width] * y[:, :width]).sum(0) / np.clip((x[:, :width] ** 2).sum(0), 1e-12, None)
    predicted = velocity[test][:, :width] * scale
    cosine = row_cosine(predicted, flux[test][:, :width])
    return {"cosine_median": float(np.median(cosine)), "n_genes": int(width)}


def spread(values: np.ndarray) -> float:
    """Per-cell variation as a fraction of the overall magnitude."""
    centred = values - values.mean(axis=0, keepdims=True)
    return float(np.linalg.norm(centred) / max(np.linalg.norm(values), 1e-12))


def audit(name: str, ctx, references: dict, n_cells: int, seed: int) -> dict:
    model, payload, config = load_run(name)
    transform = config.get("chromatin_transform", "as_is")
    field = runner.load_velocity(ctx.dataset, ctx.split_seed, transform)
    dynamic = np.asarray(field["dynamic_mask"], dtype=bool)
    rows = ctx.rows("test", dynamic=dynamic)
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(rows, min(n_cells, len(rows)), replace=False))
    train_rows = ctx.rows("train", dynamic=dynamic)
    train_rows = np.sort(rng.choice(train_rows, min(n_cells, len(train_rows)), replace=False))

    chromatin = torch.as_tensor(chromatin_features(ctx.adata, transform))
    velocity = torch.as_tensor(np.asarray(field["velocity"], dtype=np.float32))
    n_genes = ctx.n_genes

    def pushforward_and_flux(index):
        probe = torch.as_tensor(index)
        moving, pushed = torch.func.jvp(model.phi, (chromatin[probe],), (velocity[probe],))
        with torch.no_grad():
            if hasattr(model.phi, "gene_projection"):
                regulatory = chromatin[probe] @ model.phi.gene_projection[:n_genes].T
            else:
                regulatory = chromatin[probe] @ torch.as_tensor(ctx.projection[:n_genes]).T
            flux = relay_flux(moving, model.g(regulatory), model.beta, model.gamma)
        return pushed.detach().numpy(), flux.numpy()

    pushed_test, flux_test = pushforward_and_flux(rows)
    pushed_train, flux_train = pushforward_and_flux(train_rows)

    # How much does J_phi(c) itself move? Push ONE fixed direction through every cell's
    # Jacobian: an affine phi returns the same vector for all of them.
    probe_direction = velocity[torch.as_tensor(rows)].mean(0, keepdim=True)
    fixed = probe_direction.repeat(len(rows), 1)
    _, pushed_fixed = torch.func.jvp(model.phi, (chromatin[torch.as_tensor(rows)],), (fixed,))
    pushed_fixed = pushed_fixed.detach().numpy()

    velocity_test = velocity[torch.as_tensor(rows)].numpy()
    velocity_train = velocity[torch.as_tensor(train_rows)].numpy()
    result = {
        "run": name, "transform": transform, "phi_gate": config.get("phi_gate"),
        "condition": config["condition"],
        "dyn_direction_weight": config.get("dyn_direction_weight"),
        "dyn_residual_weight": config.get("dyn_residual_weight"),
        "trained_cosine_median": float(np.median(row_cosine(pushed_test, flux_test))),
        # ~0 means phi is affine in practice: one fixed direction maps to one fixed output.
        "jacobian_variation": spread(pushed_fixed),
        "flux_variation": spread(flux_test),
        "pushforward_variation": spread(pushed_test),
        "diagonal_jacobian_ceiling": diagonal_jacobian_ceiling(
            np.concatenate([velocity_train, velocity_test]),
            np.concatenate([flux_train, flux_test]),
            np.arange(len(velocity_train)),
            np.arange(len(velocity_train), len(velocity_train) + len(velocity_test))),
        "constant_jacobian_ceiling": constant_jacobian_ceiling(
            np.concatenate([velocity_train, velocity_test]),
            np.concatenate([flux_train, flux_test]),
            np.arange(len(velocity_train)),
            np.arange(len(velocity_train), len(velocity_train) + len(velocity_test))),
    }
    del model
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--dataset", choices=runner.DATASETS, default="bmmc")
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--n-cells", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=RESULTS / "r2_jacobian_ceiling.json")
    args = parser.parse_args()

    ctx = context(args.dataset, args.split_seed)
    results = []
    for name in args.runs:
        results.append(audit(name, ctx, {}, args.n_cells, args.seed))
        entry = results[-1]
        print(f"  {name:44s} cos {entry['trained_cosine_median']:+.4f}  "
              f"dense-ceiling {entry['constant_jacobian_ceiling']['cosine_median']:+.4f}  "
              f"diag-ceiling {entry['diagonal_jacobian_ceiling']['cosine_median']:+.4f}  "
              f"J-variation {entry['jacobian_variation']:.4f}  "
              f"flux-variation {entry['flux_variation']:.4f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\n[jacobian] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
