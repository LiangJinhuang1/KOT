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


# Below this JVP-RHS cosine the kinetics term is being carried but not fit. The
# 20260824 tier-A sweep sets the value: healthy cells sat at 0.75-0.96 and dead ones at
# 0.04-0.18 with nothing between, and BOTH scored the same FOSCTTM -- so a run can rank
# first on alignment while its dynamics were never fitted at all.
DYN_COS_MIN = 0.5


def read_diagnostics(path: str | Path) -> dict:
    """The ``diagnostics.json`` a finished run wrote."""
    return json.loads(Path(path).read_text())


def is_collapsed(diagnostics: dict) -> bool:
    """Whether phi collapsed in this run — see :data:`PHI_COLLAPSE_THRESHOLD`."""
    ratio = diagnostics.get("phi_variance_ratio")
    return ratio is not None and ratio < PHI_COLLAPSE_THRESHOLD


def is_dyn_dead(diagnostics: dict) -> bool:
    """Whether the kinetics term was carried but never fitted — see :data:`DYN_COS_MIN`."""
    cos = diagnostics.get("jvp_rhs_cos_median")
    return cos is not None and cos < DYN_COS_MIN


def is_degenerate(diagnostics: dict, *, tol: float = 1e-12) -> bool:
    """FOSCTTM of exactly zero — a metric that measured nothing, not a perfect score.

    Same rule as :func:`src.visualization.benchmark.find_degenerate`: typically a
    shared-latent model scored against itself. It sorts FIRST in any FOSCTTM ranking,
    which is precisely why it has to be flagged rather than read off the table.
    """
    score = diagnostics.get("mean_foscttm")
    return score is not None and abs(score) < tol


def run_flags(diagnostics: dict) -> str:
    """Comma-joined reasons this run is not a usable result, or "" when it is.

    Every mode here is INVISIBLE in — or actively flattered by — the FOSCTTM ranking,
    which is why the verdict travels with the run instead of being left to whoever
    reads the table later.
    """
    flags = []
    if is_degenerate(diagnostics):
        flags.append("degenerate")
    if is_collapsed(diagnostics):
        flags.append("collapsed")
    if is_dyn_dead(diagnostics):
        flags.append("dyn-dead")
    return ",".join(flags)


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
