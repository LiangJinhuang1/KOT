"""Read AnnData obs/var/layers from HDF5 without loading X.

Figure code that only needs barcodes or a validation slice must not open the
full RNA cache: that object is large enough to get the login node killed.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def decode_strings(values) -> np.ndarray:
    return np.asarray([
        value.decode() if isinstance(value, (bytes, bytearray)) else str(value)
        for value in values
    ])


def labels_from_codes(categories, codes) -> np.ndarray:
    """Map AnnData categorical codes; -1 is unlabelled, not the last category."""
    categories = np.asarray(decode_strings(categories))
    codes = np.asarray(codes)
    out = np.full(len(codes), "", dtype=object)
    valid = codes >= 0
    out[valid] = categories[codes[valid]]
    return out.astype(str)


def read_categorical(node) -> np.ndarray:
    return labels_from_codes(node["categories"][:], node["codes"][:])


def read_obs_column(obs, name: str) -> np.ndarray:
    node = obs[name]
    if isinstance(node, h5py.Group) and node.attrs.get("encoding-type") == "categorical":
        return read_categorical(node)
    values = np.asarray(node[:])
    if values.dtype == object:
        return decode_strings(values)
    return values


def read_obs_frame(path: str | Path, columns: list[str]) -> pd.DataFrame:
    """obs as a DataFrame indexed by cell ID. X is not read."""
    path = Path(path)
    with h5py.File(path, "r") as handle:
        obs = handle["obs"]
        names = decode_strings(obs["_index"][:])
        data = {column: read_obs_column(obs, column) for column in columns}
    return pd.DataFrame(data, index=pd.Index(names, name="cell"))


def read_obs_names(path: str | Path) -> np.ndarray:
    path = Path(path)
    with h5py.File(path, "r") as handle:
        return decode_strings(handle["obs"]["_index"][:])


def read_var_names(path: str | Path) -> np.ndarray:
    path = Path(path)
    with h5py.File(path, "r") as handle:
        return decode_strings(handle["var"]["_index"][:])


def read_dense_rows(path: str | Path, key: str, rows: np.ndarray) -> np.ndarray:
    """Slice a dense dataset (`X` or `layers/<name>`) by integer rows."""
    rows = np.asarray(rows, dtype=np.int64)
    with h5py.File(path, "r") as handle:
        node = handle[key]
        if not isinstance(node, h5py.Dataset):
            raise ValueError(f"{path} {key} is not a dense array")
        return np.asarray(node[rows], dtype=np.float32)
