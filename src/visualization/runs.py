"""Reading the training run cache.

Every figure module walks ``cache/training`` for the same two things: the
diagnostics a run wrote, and whether that run collapsed. One copy of each lives
here so the collapse threshold cannot drift between the benchmark table and the
panels drawn from it.
"""

from __future__ import annotations

import json
from pathlib import Path

CACHE_DIR = Path("cache/training")

# The kinetics residual is scale-degenerate: a run can lower its loss by
# shrinking phi rather than by satisfying the ODE. Below this variance ratio the
# run is a diagnosed training failure, not a result.
PHI_COLLAPSE_THRESHOLD = 0.5


def read_diagnostics(path: str | Path) -> dict:
    """The ``diagnostics.json`` a finished run wrote."""
    return json.loads(Path(path).read_text())


def is_collapsed(diagnostics: dict) -> bool:
    """Whether phi collapsed in this run — see :data:`PHI_COLLAPSE_THRESHOLD`."""
    ratio = diagnostics.get("phi_variance_ratio")
    return ratio is not None and ratio < PHI_COLLAPSE_THRESHOLD


def curated_runs(cache_dir: str | Path = CACHE_DIR) -> dict[str, str]:
    """Run dirs the MANIFEST marks as kept, mapped to why each was kept.

    ``cache/training`` was curated by hand on 2026-08-23: 35 runs renamed to
    semantic names (``syn_branch_ablation``, ``bmmc_scvelo_kot``, ...) and 184
    archived to ``cache/training_archive_20260823``. Those names record which run
    belongs in which panel, so selection should follow the manifest rather than
    infer intent from timestamps — picking by timestamp chose a lambda=1 sweep
    where the branch ablation shows no effect (0.364 vs 0.374) over the curated
    matched run where it separates 19-fold (0.020 vs 0.378).

    Returns an empty mapping when no manifest exists, so callers fall back to
    their own ordering.
    """
    manifest = Path(cache_dir) / "MANIFEST.json"
    if not manifest.exists():
        return {}
    payload = json.loads(manifest.read_text())
    return {e["new"]: e.get("why", "")
            for e in payload.get("kept", []) if e.get("new")}


def run_rank(run_name: str, prefer: tuple[str, ...] = (),
             curated: dict[str, str] | None = None) -> int:
    """Selection priority: 0 = named explicitly, 1 = curated, 2 = everything else."""
    if run_name in prefer:
        return 0
    if curated is None:
        curated = curated_runs()
    return 1 if run_name in curated else 2
