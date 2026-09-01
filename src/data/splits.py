"""Validation split for hyperparameter selection.

Tuning on the full set is tuning on the test FOSCTTM. This holds a barcode-hash
slice out of both losses during search; full-data runs put those cells back
(`val_holdout_from_training=false`).

`val_split_per_seed` mixes the model seed into the hash (four seeds = four splits);
off, one split serves every run. Config is not an input, so paired config
comparisons stay on the same cells. `val_stratify_by` gives each stratum its own
fraction (rare types otherwise can get zero val cells) but then a cell's side
depends on who shares its stratum. `split_digest` is written so agreement is
checkable.
"""

from __future__ import annotations

import hashlib

import numpy as np

UINT64_SPAN = float(2 ** 64)  # digest_size=8 → uniform [0, 1) via / span


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
    split. With a threshold, every cell's side is decided by its barcode alone
    and the realized count is binomial around `fraction * n`.

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


def random_slice_mask(rna_obs, cfg: dict, model_seed: int | None) -> np.ndarray:
    """The random / stratified val_fraction slice as DRAWN, whatever is done with it.

    This is the reporting slice: with `val_holdout_from_training` off the cells go back
    into the loss but their FOSCTTM is still measured and marked in-sample.
    `random_holdout_mask` is the other question -- whether those cells are
    removed from the loss.
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
    return mask


def random_holdout_mask(rna_obs, cfg: dict, model_seed: int | None) -> np.ndarray:
    """The part of the random slice that is actually removed from the loss.

    All-False when `val_holdout_from_training` is off. Deciding that here rather than at
    each call site is what keeps `held_out_mask` and `fitted_row_indices` describing the
    same cells -- they disagreed before, so a run with fit_obs_key silently dropped its
    val slice from training after the config had asked for it back.
    """
    if not bool(cfg.get("val_holdout_from_training", True)):
        return np.zeros(len(rna_obs.index), dtype=bool)
    return random_slice_mask(rna_obs, cfg, model_seed)


def group_holdout_mask(rna_obs, cfg: dict) -> np.ndarray:
    """True for cells whose ``fit_obs_key`` value is outside ``fit_obs_values``.

    Never inverted by fit_tuning_subset: a CRISPR fit restriction must not
    silently put knockout transcriptomes back into the loss.
    """
    fit_key = cfg.get("fit_obs_key") or None
    if fit_key is None:
        return np.zeros(len(rna_obs.index), dtype=bool)
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
    return ~allowed_mask


def held_out_mask(rna_obs, cfg: dict, model_seed: int | None) -> np.ndarray:
    """True for cells held out of the loss: the union of the two restrictions.

    They are independent. `val_holdout_from_training=false` returns the random slice to
    training whether or not a group restriction is also set, and a group restriction
    removes its cells whether or not a random slice exists.
    """
    return (random_holdout_mask(rna_obs, cfg, model_seed)
            | group_holdout_mask(rna_obs, cfg))


def reported_slice_mask(rna_obs, cfg: dict, model_seed: int | None) -> np.ndarray:
    """True for cells a run reports val metrics on, held out of the loss or not.

    A superset of `held_out_mask`: it keeps the drawn random slice even when the config
    returned those cells to training, so in-sample val columns are still measured.
    """
    return (random_slice_mask(rna_obs, cfg, model_seed)
            | group_holdout_mask(rna_obs, cfg))


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
    mask = held_out_mask(rna_obs, cfg, model_seed)
    return None if not mask.any() else np.nonzero(~mask)[0]
