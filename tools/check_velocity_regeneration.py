#!/usr/bin/env python3
"""
Compare a regenerated velocity file against the preprocessed cache built from the
file it replaces.

The preprocessed cache is the velocity file after QC, normalization and PCA, so the
cells must match exactly and its genes must be a subset of the source's. Anything
else means the regeneration did not reproduce the input the cache was built from,
and every number measured against that cache is on different data.

Usage:
  python tools/check_velocity_regeneration.py \
      cache/velocity/bmmc_cite/bmmc_cite_scvelo_results_retained.h5ad \
      cache/preprocessed/bmmc_cite_scvelo_results_retained_<hash>.rna.h5ad

The <hash> is a preprocessing-cache key and changes whenever preprocessing or its
input does; take it from the run you are checking (`grep '[cache] Prefix:' <run>/*.log`)
rather than from an example, which goes stale the first time the velocity is rebuilt.
"""
from __future__ import annotations

import argparse
import sys

import h5py
import numpy as np


def index_names(group) -> np.ndarray:
    """The obs/var index of an h5ad group, as plain strings."""
    key = group.attrs.get("_index", "_index")
    if isinstance(key, bytes):
        key = key.decode()
    values = group[key][:]
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in values])


def read_axes(path: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    with h5py.File(path, "r") as handle:
        obs = index_names(handle["obs"])
        var = index_names(handle["var"])
        layers = sorted(handle["layers"].keys()) if "layers" in handle else []
    return obs, var, layers


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("regenerated", help="new velocity .h5ad")
    parser.add_argument("cache", help="preprocessed .rna.h5ad built from the old one")
    args = parser.parse_args()

    src_obs, src_var, src_layers = read_axes(args.regenerated)
    cch_obs, cch_var, cch_layers = read_axes(args.cache)

    print(f"regenerated : {len(src_obs):7d} cells  {len(src_var):6d} genes")
    print(f"cache       : {len(cch_obs):7d} cells  {len(cch_var):6d} genes")

    same_cells = set(src_obs) == set(cch_obs)
    missing_genes = set(cch_var) - set(src_var)
    missing_layers = set(cch_layers) - set(src_layers)

    print(f"cells identical      : {same_cells}")
    print(f"cache genes ⊆ source : {not missing_genes}  (missing {len(missing_genes)})")
    print(f"cache layers ⊆ source: {not missing_layers}  (missing {sorted(missing_layers)})")

    if same_cells and not missing_genes and not missing_layers:
        print("\nPASS — the regeneration reproduces the input behind the cache.")
        return 0
    print("\nFAIL — the regenerated file differs from what the cache was built from.")
    if not same_cells:
        print(f"  cells only in cache : {len(set(cch_obs) - set(src_obs))}")
        print(f"  cells only in source: {len(set(src_obs) - set(cch_obs))}")
    if missing_genes:
        print(f"  genes dropped: {sorted(missing_genes)[:10]}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
