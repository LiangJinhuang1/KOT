"""Setup every chromatin analysis script repeats, in one place.

Fourteen of the chromatin tools open the same door before they do anything interesting:
load the dataset, load the split and assert it still matches, derive the ATAC/RNA train
sides and the measured-target mask, pick the output genes, and build G. Written out each
time that is ~25 lines per script whose only job is to be identical to the others -- and
when it is NOT identical (a forgotten `has_splicing` mask, a different gene subset) two
scripts silently answer questions about different cells.

`context()` returns that state once. It lives in tools/ rather than src/ because it needs
`run_kot_chromatin`, which itself imports src -- putting it under src/ would be a cycle.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_kot_chromatin as runner
from src.data.chromatin import chromatin_features
from src.data.chromatin_r2 import shared_splicing_targets


@dataclass
class Context:
    """One dataset's cells, genes, split and chromatin input, agreed across scripts."""

    dataset: str
    split_seed: int
    transform: str
    adata: object
    split: np.ndarray
    side: np.ndarray
    measured: np.ndarray
    columns: np.ndarray
    projection: np.ndarray
    kinetic_mask: np.ndarray

    def rows(self, which: str, *, side: str | None = None,
             dynamic: np.ndarray | None = None) -> np.ndarray:
        """Cell indices for a split, optionally restricted to a train side or the field.

        Always intersected with `measured`: a cell with no spliced/unspliced counts has no
        target, and including it silently changes what a metric is averaged over.
        """
        keep = (self.split == which) & self.measured
        if side is not None:
            keep &= self.side == side
        if dynamic is not None:
            keep &= np.asarray(dynamic, dtype=bool)
        return np.flatnonzero(keep)

    def chromatin(self) -> np.ndarray:
        return chromatin_features(self.adata, self.transform)

    def targets(self) -> np.ndarray:
        """The R2 [u, s] target in shared RNA units, on the output genes."""
        return shared_splicing_targets(self.adata, self.columns)

    def velocity(self) -> dict:
        """The chromatin field built in THIS context's transform, never another's."""
        return runner.load_velocity(self.dataset, self.split_seed, self.transform)

    @property
    def n_genes(self) -> int:
        return len(self.columns)

    def blocks(self):
        """(name, column slice) for the two halves of a relay prediction."""
        return [("u", slice(0, self.n_genes)), ("s", slice(self.n_genes, None))]


def context(dataset: str, split_seed: int = 0, transform: str = "as_is") -> Context:
    adata = runner.load_dataset(dataset)
    splits = pd.read_csv(runner.split_path(dataset, split_seed), index_col=0)
    if not splits.index.equals(adata.obs_names):
        raise ValueError(f"the {dataset} split file does not match the current dataset build")
    split = splits["split"].to_numpy()
    side = splits["train_side"].to_numpy()
    measured = runner.target_cell_mask(adata, runner.SHARED_SPLICED)
    gene_mask = runner.output_gene_mask(
        adata, runner.RELAY,
        rows=np.flatnonzero((split == "train") & (side == "rna") & measured),
        target_layer=runner.SHARED_SPLICED)
    columns = np.flatnonzero(gene_mask)
    mapping, _, kinetic = runner.gene_map(adata, runner.gene_map_path(dataset))
    return Context(dataset=dataset, split_seed=split_seed, transform=transform, adata=adata,
                   split=split, side=side, measured=measured, columns=columns,
                   projection=mapping[gene_mask].toarray().astype(np.float32),
                   kinetic_mask=kinetic[gene_mask])


def load_run(name: str, checkpoint: str = "best_align"):
    """A trained model plus the settings it was trained under, by run-directory name."""
    return runner.load_checkpoint(Path("cache/chromatin/runs") / name, checkpoint,
                                  torch.device("cpu"))


def run_transform(name: str) -> str:
    """Which chromatin transform a run was trained on, for scoring it in its own space."""
    import json
    config = json.loads((Path("cache/chromatin/runs") / name / "run_config.json").read_text())
    return config.get("chromatin_transform", "as_is")


def gate_verdict(name: str) -> str:
    return "PASS" if (Path("cache/chromatin/runs") / name / "preflight_passed.json").exists() \
        else "fail"


def preflight(name: str) -> dict:
    """The run's gate checks, whichever file they landed in."""
    import json
    run = Path("cache/chromatin/runs") / name
    passed = run / "preflight_passed.json"
    return json.loads((passed if passed.exists() else run / "preflight.json").read_text())


def quantiles(values) -> dict:
    """min/p05/median/p95/max/mean, the summary these scripts kept re-deriving."""
    array = np.asarray(torch.as_tensor(values).detach().cpu(), dtype=float).ravel()
    return dict(zip(["min", "p05", "median", "p95", "max"],
                    np.quantile(array, [0, .05, .5, .95, 1]).tolist())) | {
        "mean": float(array.mean())}


def centred_norm(values: torch.Tensor) -> torch.Tensor:
    return (values - values.mean(0)).norm()
