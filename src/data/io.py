"""I/O helpers shared by preprocessing and velocity pipelines."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import h5py


def is_valid_h5ad(path: str | Path) -> bool:
    """Return whether *path* looks like a complete AnnData HDF5 file."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return False
    if not h5py.is_hdf5(path):
        return False
    with h5py.File(path, "r") as handle:
        return "obs" in handle and "var" in handle


def write_h5ad_atomic(adata, output_path: str | Path) -> None:
    """Write an H5AD file atomically and validate it before publication.

    A UUID avoids temporary-name collisions between threads in the same process,
    while ``os.replace`` ensures cluster workers never observe a partial cache.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(
        f".{output_path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    try:
        adata.write_h5ad(tmp_path)
        if not is_valid_h5ad(tmp_path):
            raise OSError(f"Temporary H5AD failed validation after write: {tmp_path}")
        os.replace(tmp_path, output_path)
    finally:
        tmp_path.unlink(missing_ok=True)
