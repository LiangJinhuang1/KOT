"""Public API for the KOT package."""

from __future__ import annotations

from src import __version__

__all__ = ["__version__", "run_training"]


def run_training(*args, **kwargs):
    """Run the training pipeline without importing heavy backends at package import."""
    from src.training.runner import run_training as _run_training

    return _run_training(*args, **kwargs)
