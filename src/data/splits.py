"""
Fixed validation split for hyperparameter selection.

KOT trains without labels, so nothing a run computes about itself can say which
hyperparameters are the right ones -- only the pairing can, and the pairing is
what the headline FOSCTTM is reported on. Tuning on the full set is therefore
tuning on the test set. This module carves a fixed slice of cells out of the
training pool to tune against instead: held out of both loss terms during the
search, scored with the pairing, and never used for anything but ranking
configurations. The final runs put those cells back into training and report
the full-set metric unchanged.

The assignment is a hash of the cell barcode, not a shuffled index: a cell's
side of the split depends on its own barcode and nothing else, so it is stable
across datasets versions, preprocessing caches, model arms, seeds and
checkpoints, and two runs agree on it without either one loading a file.
`split_digest` goes into diagnostics.json so that agreement is checkable after
the fact rather than assumed.
"""

from __future__ import annotations

import hashlib

import numpy as np

# blake2b digest_size=8 -> a uint64 per cell; dividing by its span gives the
# uniform [0, 1) value the fraction thresholds against.
UINT64_SPAN = float(2 ** 64)


def cell_hash_fractions(obs_names, split_seed: int) -> np.ndarray:
    """A fixed uniform [0, 1) value per cell, keyed on its barcode and the seed."""
    fractions = np.empty(len(obs_names), dtype=np.float64)
    for position, name in enumerate(obs_names):
        digest = hashlib.blake2b(f"{split_seed}:{name}".encode(), digest_size=8).digest()
        fractions[position] = int.from_bytes(digest, "big") / UINT64_SPAN
    return fractions


def validation_mask(obs_names, fraction: float, split_seed: int) -> np.ndarray:
    """Boolean (n,) mask over cells, True for the validation slice.

    A threshold on the per-cell hash, not the first `round(fraction * n)` cells
    of a shuffle: a count has to be filled from some order, so dropping a
    handful of cells upstream (a new preprocessing cache, a different gene
    filter) would shift the boundary and move OTHER cells across the split. With
    a threshold, every cell's side is decided by its barcode alone and the
    realized count is binomial around `fraction * n` (±0.6% of n at n=5000).
    """
    if not 0.0 <= fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1), got {fraction}")
    if fraction == 0.0:
        return np.zeros(len(obs_names), dtype=bool)
    return cell_hash_fractions(obs_names, split_seed) < fraction


def split_digest(obs_names, mask: np.ndarray) -> str:
    """Digest of the selected barcodes: two runs used the same split iff it matches."""
    selected = "\n".join(sorted(str(name) for name, keep in zip(obs_names, mask) if keep))
    return hashlib.blake2b(selected.encode(), digest_size=8).hexdigest()
