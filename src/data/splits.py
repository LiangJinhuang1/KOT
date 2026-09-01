"""
Validation split for hyperparameter selection.

KOT trains without labels, so nothing a run computes about itself can say which
hyperparameters are the right ones -- only the pairing can, and the pairing is
what the headline FOSCTTM is reported on. Tuning on the full set is therefore
tuning on the test set. This module carves a slice of cells out of the training
pool to tune against instead: held out of both loss terms during the search,
scored with the pairing, and never used for anything but ranking configurations.
The final runs put those cells back into training and report the full-set metric
unchanged.

Two axes control how the slice is drawn.

`split_seed` -- what the per-cell hash is keyed on. With `val_split_per_seed`
the run's MODEL seed is mixed into it, so the four seeds of a config are four
different splits and averaging over them averages over splits as well. The
mixing is a hash of both, not a sum, so (base 1, seed 2) and (base 2, seed 1)
cannot collide onto the same split. Crucially the config is NOT an input: two
configs at the same model seed still get the identical split, which is what
keeps the paired comparison across configs valid. With it off, one split serves
every run, as before.

`strata` -- optional per-cell labels. Without them a cell is validation iff its
own hash falls below `fraction`, so each stratum's share is binomial around
`fraction`: fine at n=5000, and capable of handing a 18-cell cell type zero
validation cells. With them, every stratum contributes its own `fraction` share
exactly (to rounding). The cost is that a cell's side then depends on which
other cells share its stratum, so the assignment no longer survives cells being
dropped upstream -- acceptable precisely when the split is redrawn per seed
anyway, and the reason stratification and per-seed drawing arrived together.

`split_digest` goes into diagnostics.json either way, so agreement between two
runs is checkable after the fact rather than assumed.
"""

from __future__ import annotations

import hashlib

import numpy as np

# blake2b digest_size=8 -> a uint64 per cell; dividing by its span gives the
# uniform [0, 1) value the fraction thresholds against.
UINT64_SPAN = float(2 ** 64)


def resolve_split_seed(base_seed: int, model_seed: int | None, per_seed: bool) -> int:
    """The seed the split is keyed on, given the config's base and the run's seed.

    Hashed rather than added: base + model_seed maps (1, 2) and (2, 1) to the same
    number, which would hand two runs the same split while their diagnostics claim
    different ones.
    """
    if not per_seed or model_seed is None:
        return int(base_seed)
    digest = hashlib.blake2b(f"{int(base_seed)}:{int(model_seed)}".encode(),
                             digest_size=8).digest()
    return int.from_bytes(digest, "big")


def cell_hash_fractions(obs_names, split_seed: int) -> np.ndarray:
    """A fixed uniform [0, 1) value per cell, keyed on its barcode and the seed."""
    fractions = np.empty(len(obs_names), dtype=np.float64)
    for position, name in enumerate(obs_names):
        digest = hashlib.blake2b(f"{split_seed}:{name}".encode(), digest_size=8).digest()
        fractions[position] = int.from_bytes(digest, "big") / UINT64_SPAN
    return fractions


def stratified_mask(fractions: np.ndarray, strata: np.ndarray,
                     fraction: float) -> np.ndarray:
    """Give every stratum its own `fraction` share: its lowest-hash cells.

    Ranking inside the stratum rather than thresholding is what makes the share
    exact instead of binomial. Rounding is half-up on the count, so a stratum needs
    only 1/(2*fraction) cells to contribute one -- 5 cells at fraction 0.1 -- rather
    than being dropped from validation entirely.
    """
    mask = np.zeros(len(fractions), dtype=bool)
    for level in np.unique(strata):
        members = np.nonzero(strata == level)[0]
        count = int(np.floor(fraction * len(members) + 0.5))
        if count <= 0:
            continue
        # Stable sort so cells with an identical hash (never seen at 64 bits, but
        # free to guarantee) resolve by position rather than by numpy's whim.
        ordered = members[np.argsort(fractions[members], kind="stable")]
        mask[ordered[:count]] = True
    return mask


def validation_mask(obs_names, fraction: float, split_seed: int,
                    strata=None) -> np.ndarray:
    """Boolean (n,) mask over cells, True for the validation slice.

    Unstratified, this is a threshold on the per-cell hash, not the first
    `round(fraction * n)` cells of a shuffle: a count has to be filled from some
    order, so dropping a handful of cells upstream (a new preprocessing cache, a
    different gene filter) would shift the boundary and move OTHER cells across the
    split. With a threshold, every cell's side is decided by its barcode alone and
    the realized count is binomial around `fraction * n` (±0.6% of n at n=5000).

    Passing `strata` trades that invariance for exact per-stratum shares; see the
    module docstring.
    """
    if not 0.0 <= fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1), got {fraction}")
    if fraction == 0.0:
        return np.zeros(len(obs_names), dtype=bool)
    fractions = cell_hash_fractions(obs_names, split_seed)
    if strata is None:
        return fractions < fraction
    strata = np.asarray(strata)
    if len(strata) != len(obs_names):
        raise ValueError(
            f"strata has {len(strata)} entries for {len(obs_names)} cells"
        )
    return stratified_mask(fractions, strata, fraction)


def split_digest(obs_names, mask: np.ndarray) -> str:
    """Digest of the selected barcodes: two runs used the same split iff it matches."""
    selected = "\n".join(sorted(str(name) for name, keep in zip(obs_names, mask) if keep))
    return hashlib.blake2b(selected.encode(), digest_size=8).hexdigest()


def held_out_mask(rna_obs, cfg: dict, model_seed: int | None) -> np.ndarray:
    """True for cells held out of the loss.

    Union of two independent restrictions:

      * the random / stratified val_fraction slice (optionally inverted by
        fit_tuning_subset, same rule as before)
      * cells whose ``fit_obs_key`` value is outside ``fit_obs_values``

    The group restriction is never inverted: Papalexi's CRISPR experiment trains on
    NT cells and must not silently put knockout transcriptomes back into the loss
    when fit_tuning_subset is on.
    """
    val_fraction = float(cfg.get("val_fraction", 0.0))
    split_seed = resolve_split_seed(
        int(cfg.get("val_split_seed", 20260825)),
        model_seed,
        bool(cfg.get("val_split_per_seed", False)),
    )
    stratify_by = cfg.get("val_stratify_by") or None
    strata = None
    if stratify_by is not None:
        if stratify_by not in rna_obs.columns:
            raise ValueError(
                f"val_stratify_by={stratify_by!r} is not an obs column. "
                f"Available: {sorted(rna_obs.columns)}"
            )
        strata = rna_obs[stratify_by].astype(str).to_numpy()

    mask = validation_mask(rna_obs.index, val_fraction, split_seed, strata=strata)
    if bool(cfg.get("fit_tuning_subset", False)) and val_fraction > 0.0:
        mask = ~mask

    fit_key = cfg.get("fit_obs_key") or None
    if fit_key is None:
        return mask
    if fit_key not in rna_obs.columns:
        raise ValueError(
            f"fit_obs_key={fit_key!r} is not an obs column. "
            f"Available: {sorted(rna_obs.columns)}"
        )
    allowed = cfg.get("fit_obs_values")
    if not allowed:
        raise ValueError("fit_obs_key is set but fit_obs_values is empty")
    allowed_mask = rna_obs[fit_key].astype(str).isin([str(v) for v in allowed]).to_numpy()
    if not allowed_mask.any():
        raise ValueError(
            f"fit_obs_values {list(allowed)} matched 0 cells in obs[{fit_key!r}]"
        )
    return mask | ~allowed_mask


def fitted_row_indices(rna_obs, cfg: dict, model_seed: int | None) -> np.ndarray | None:
    """Positional indices of the cells a run FITS, from its resolved config.

    One definition shared by training (src/training/kot.py, which restricts the loss and
    the diagnostics) and by evaluation (src/training/runner.py, which restricts the
    benchmark FOSCTTM). Two copies of this rule would be two chances for the reported
    number to describe a different cell set than the model was fitted to.

    Takes the obs DataFrame rather than obs_names because a stratified split needs the
    stratum column: bmmc_cite_retained sets val_stratify_by=cell_type, and a mask built
    without it lands on different cells than the one training used.

    None means every cell -- no split configured, or the split deliberately returned to
    training (val_holdout_from_training=false) and no fit_obs restriction is set.
    """
    has_group = bool(cfg.get("fit_obs_key"))
    val_fraction = float(cfg.get("val_fraction", 0.0))
    holdout_random = val_fraction > 0.0 and bool(cfg.get("val_holdout_from_training", True))
    if not has_group and not holdout_random:
        return None
    return np.nonzero(~held_out_mask(rna_obs, cfg, model_seed))[0]
