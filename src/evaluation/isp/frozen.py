"""Reading the frozen NT-only KOT checkpoint the ISP runners predict into.

Separate from common.py because this reaches the whole KOT stack, which the GEARS and
scGPT virtualenvs cannot import. Those two models fit in the checkpoint's own
coordinates and never touch a checkpoint, so they only import common.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from run_kot_crispr import (
    build_model_and_tensors,
    check_checkpoint_units,
    discover_seed_checkpoints,
    load_dataset_for_run,
)

NT_ONLY_DATASET = "papalexi_nt_only"


@dataclass
class FrozenRun:
    """A frozen NT-only run: its data, its config, and the model name it was trained as."""

    rna: object
    protein: object
    cfg: dict
    model_name: str


@dataclass
class FrozenCheckpoint:
    """One seed's checkpoint, with the model and input tensors rebuilt around it."""

    path: Path
    model: object
    features: torch.Tensor
    run: FrozenRun


def load_nt_only_run(run_dir: Path, dataset: str = NT_ONLY_DATASET,
                     model_name: str = "") -> FrozenRun:
    """Load a run's data and assert it really is the NT-only configuration.

    Task B's whole claim is that KOT never saw a knockout cell, so a run fitted on
    anything but NT is refused here rather than quietly scored.
    """
    rna, protein, cfg, _ = load_dataset_for_run(run_dir, dataset)
    if cfg.get("fit_obs_key") != "gene" or cfg.get("fit_obs_values") != ["NT"]:
        raise ValueError("Task B requires an NT-only KOT checkpoint")
    return FrozenRun(rna, protein, cfg, model_name)


def load_frozen_checkpoint(run_dir: Path, checkpoint: str, seed: int,
                           device: torch.device) -> FrozenCheckpoint:
    """The single checkpoint for this seed, with model and input tensors rebuilt.

    Real velocity and mapping whatever the arm trained with: they only initialise the
    tensors, since prediction reads the frozen phi and the input features alone.
    """
    checkpoints, dataset, model_name = discover_seed_checkpoints(
        run_dir, checkpoint, dataset=NT_ONLY_DATASET)
    selected = [(path, value) for path, value in checkpoints if value == seed]
    if len(selected) != 1:
        raise SystemExit(
            f"Expected one checkpoint for seed {seed}, got {len(selected)}")
    run = load_nt_only_run(run_dir, dataset, model_name)
    check_checkpoint_units(selected, run.cfg, False)
    model, features, *_ = build_model_and_tensors(
        run.cfg, run.rna, run.protein, model_name, device,
        velocity_mode="real", mapping_mode="real")
    return FrozenCheckpoint(selected[0][0], model, features, run)
