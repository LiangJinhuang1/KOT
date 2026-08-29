"""How different is a permuted velocity from the real one, per cell?

If cos(v_i, v_perm(i)) is near 1 the permutation barely perturbs anything and the control
is weak; near 0 means it is a genuine randomisation.

Measured in the PC space the JVP actually consumes, not in gene space: the velocity is
carried into PCA coordinates by `resolve_velocity_matrix` before it ever reaches
J_phi(r)v, and a linear projection does not preserve cosines, so the gene-space number
describes a vector the model never sees. The gene-space number is printed alongside for
reference.

The field's own directional concentration is printed too, because it is what sets the
floor: a field whose cells largely point the same way cannot be scrambled hard by any
permutation, and the same shared direction is what lets the JVP-RHS cosine stay high on a
shuffled arm.

Usage: probe_shuffle_strength.py <rna.h5ad> <label> <seed,seed,...>
"""
import sys

import anndata as ad
import numpy as np

from src.training.kot import velocity_row_permutation


def unit_rows(matrix: np.ndarray) -> np.ndarray:
    """Rows rescaled to unit length, dropping the cells that carry no velocity."""
    norms = np.linalg.norm(matrix, axis=1)
    keep = norms > 0
    return matrix[keep] / norms[keep][:, None]


def report_space(name: str, matrix: np.ndarray, seeds: list[int]) -> None:
    """cos(v_i, v_assigned) pooled over one permutation per seed, plus the field mean."""
    units = unit_rows(matrix)
    cosines = np.concatenate([
        (units * units[velocity_row_permutation(units.shape[0], seed)]).sum(1)
        for seed in seeds
    ])
    mean_direction = units.mean(0)
    to_mean = units @ (mean_direction / np.linalg.norm(mean_direction))
    print(f"  {name:11s} cos(real_i, assigned_i)  mean={cosines.mean():+.3f} "
          f"median={np.median(cosines):+.3f}  |cos|>0.5: {100 * np.mean(np.abs(cosines) > 0.5):.1f}%")
    print(f"  {name:11s} cos(real_i, field mean)  mean={to_mean.mean():+.3f} "
          f"median={np.median(to_mean):+.3f}")


path, label, seed_arg = sys.argv[1], sys.argv[2], sys.argv[3]
seeds = [int(seed) for seed in seed_arg.split(",")]

adata = ad.read_h5ad(path)
velocity = np.nan_to_num(np.asarray(adata.layers["velocity"], dtype=np.float32), nan=0.0)
in_pc_space = velocity @ np.asarray(adata.varm["PCs"], dtype=np.float32)

print(f"{label}: {velocity.shape[0]} cells x {velocity.shape[1]} genes -> "
      f"{in_pc_space.shape[1]} PCs, over {len(seeds)} seeds")
report_space("PC (model)", in_pc_space, seeds)
report_space("gene", velocity, seeds)
