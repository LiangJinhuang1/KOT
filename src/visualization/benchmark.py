"""Cross-method benchmark figure — the head-to-head comparison the repo was missing.

Collects every run's ``summary.txt`` into one tidy table, then draws per-dataset
panels of FOSCTTM by method with the individual seeds visible.

Two data-fidelity rules are enforced here rather than left to the caller:

* **Degenerate arms are not plotted as peers.** A method whose FOSCTTM is exactly
  0 on every seed is not a competitor that won; it is a metric that was not
  measuring anything (typically a shared-latent upper bound scored against
  itself). Those arms are excluded and named in the returned notes.
* **Cross-run pooling is reported.** When a dataset's methods do not all come
  from one run, the panel says so, because arms trained under different runs are
  not automatically comparable (§1.2).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from src.visualization import (
    FOSCTTM_CHANCE,
    dataset_label,
    method_color,
    method_label,
    method_is_focal,
)
from src.visualization.style import (
    apply_style, chance_line, figsize, ink, panel_letter, save_figure,
)


def collect_benchmark(cache_dir: str | Path = "cache/training") -> pd.DataFrame:
    """Tidy table of every completed run: one row per (run, model, dataset, seed)."""
    rows = []
    for summary in Path(cache_dir).rglob("*/summary.txt"):
        record: dict[str, str] = {}
        for line in summary.read_text().splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                record[key.strip()] = value.strip()
        if "mean_foscttm" not in record or "model" not in record:
            continue
        parts = summary.relative_to(cache_dir).parts
        record["run"] = parts[0] if parts[0].startswith("run_") else "(legacy)"
        record["mean_foscttm"] = float(record["mean_foscttm"])
        record["seed"] = record.get("seed", "-")
        # phi_variance_ratio lives in diagnostics.json, not summary.txt, but a
        # collapsed run is a diagnosed training failure rather than a result, so
        # the flag has to travel with the score.
        diag = summary.parent / "diagnostics.json"
        record["phi_variance_ratio"] = None
        if diag.exists():
            try:
                record["phi_variance_ratio"] = json.load(open(diag)).get("phi_variance_ratio")
            except Exception:
                pass
        rows.append(record)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    cols = ["run", "dataset", "model", "seed", "mean_foscttm", "phi_variance_ratio"]
    return df[cols].drop_duplicates(subset=cols[:5])


def find_degenerate(df: pd.DataFrame, *, tol: float = 1e-12) -> list[str]:
    """Methods whose FOSCTTM is identically zero on every run — not real results."""
    degenerate = []
    for model, group in df.groupby("model"):
        if len(group) >= 2 and (group["mean_foscttm"].abs() < tol).all():
            degenerate.append(str(model))
    return sorted(degenerate)


PHI_COLLAPSE_THRESHOLD = 0.5


def find_collapsed(df: pd.DataFrame, threshold: float = PHI_COLLAPSE_THRESHOLD) -> pd.Series:
    """Runs where phi collapsed: its output variance is a fraction of the target's.

    The kinetics residual is scale-degenerate — shrinking phi (and kappa, alpha
    with it) lowers the loss without the ODE becoming any more satisfied — so a
    collapsed run is a training failure, not a measurement. Median FOSCTTM for
    collapsed runs is at chance (0.46) against 0.07 for healthy ones, and the
    ODE-consistency diagnostic *rises* under collapse, so these cannot be spotted
    from the physics metrics alone.
    """
    ratio = pd.to_numeric(df.get("phi_variance_ratio"), errors="coerce")
    return ratio.notna() & (ratio < threshold)


def find_duplicate_arms(df: pd.DataFrame, *, tol: float = 1e-9) -> list[tuple[str, str, int]]:
    """Pairs of methods that produce identical numbers wherever both were run.

    Two arms that never differ are one arm with two names; reporting them as
    separate rows in an ablation table is the failure this guards against.
    """
    findings = []
    wide = df.pivot_table(index=["run", "dataset", "seed"], columns="model",
                          values="mean_foscttm", aggfunc="first")
    models = list(wide.columns)
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            both = wide[[a, b]].dropna()
            if len(both) >= 3 and (both[a] - both[b]).abs().lt(tol).all():
                findings.append((str(a), str(b), len(both)))
    return findings


def plot_benchmark(
    df: pd.DataFrame,
    datasets: list[str],
    save_path: str | Path,
    *,
    methods: list[str] | None = None,
    exclude_degenerate: bool = True,
    exclude_collapsed: bool = True,
    runs: dict[str, str | list[str]] | None = None,
    jitter: float = 0.14,
    random_state: int = 0,
):
    """One panel per dataset: FOSCTTM by method, every seed shown.

    Small n per method, so the distribution is drawn as a jittered strip with a
    median tick rather than a bar with an error bar (§6.1).
    """
    apply_style()
    rng = np.random.default_rng(random_state)

    notes: list[str] = []
    data = df[df["dataset"].isin(datasets)].copy()
    if runs:
        keep = [(data["dataset"] == ds) & (data["run"].isin(np.atleast_1d(run)))
                for ds, run in runs.items()]
        data = data[np.logical_or.reduce(keep)] if keep else data.iloc[:0]

    if exclude_collapsed:
        collapsed = find_collapsed(data)
        if collapsed.any():
            by_model = data.loc[collapsed, "model"].value_counts()
            notes.append(
                "excluded (phi collapsed, training failure): "
                + ", ".join(f"{method_label(m)} {int(n)} run(s)"
                            for m, n in by_model.items())
            )
            data = data[~collapsed]

    dropped = find_degenerate(data) if exclude_degenerate else []
    if dropped:
        data = data[~data["model"].isin(dropped)]
        notes.append(
            f"excluded (FOSCTTM identically 0, not a measured result): "
            f"{', '.join(method_label(m) for m in dropped)}"
        )

    if methods is None:
        order = (data.groupby("model")["mean_foscttm"].median()
                 .sort_values(ascending=False).index.tolist())
    else:
        order = [m for m in methods if m in set(data["model"])]

    n_panels = len(datasets)
    height = 0.55 + 0.22 * len(order)
    fig, axes = plt.subplots(1, n_panels, figsize=figsize("full", height),
                             sharey=True, squeeze=False, layout="constrained")
    axes = axes.ravel()

    for panel, (ax, dataset) in enumerate(zip(axes, datasets)):
        sub = data[data["dataset"] == dataset]
        runs_used = sub.groupby("model")["run"].nunique()
        multi_run = sub["run"].nunique() > 1

        for row, model in enumerate(order):
            vals = sub.loc[sub["model"] == model, "mean_foscttm"].to_numpy()
            if len(vals) == 0:
                # An absent category is marked, never left as an empty slot (§6.1).
                ax.text(0.02, row, "n.d.", transform=ax.get_yaxis_transform(),
                        fontsize=6, color="0.6", ha="left", va="center")
                continue
            focal = method_is_focal(model)
            color = method_color(model)
            ax.scatter(vals, row + rng.uniform(-jitter, jitter, len(vals)),
                       s=7 if focal else 5,
                       color=color, alpha=0.85 if focal else 0.55,
                       linewidths=0.4 if focal else 0,
                       edgecolors="white" if focal else "none",
                       zorder=7)
            # With n < 3 the observations are the summary (§6.2); a median tick
            # there adds no information and its halo would hide the only points.
            if len(vals) >= 3:
                median = float(np.median(vals))
                ax.plot([median, median], [row - 0.32, row + 0.32],
                        color="white", lw=(2.8 if focal else 2.2),
                        solid_capstyle="butt", zorder=5)
                ax.plot([median, median], [row - 0.30, row + 0.30],
                        color="#1A1A1A", lw=(1.4 if focal else 1.0),
                        solid_capstyle="butt", zorder=6)

        chance_line(ax, FOSCTTM_CHANCE, axis="x", label="chance")
        ax.set_ylim(-0.7, len(order) - 0.3)
        ax.set_xlim(0, max(0.62, float(data["mean_foscttm"].max()) * 1.05))
        ax.invert_yaxis()
        ax.set_yticks(range(len(order)))
        if panel == 0:
            ax.set_yticklabels([method_label(m) for m in order])
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
        ax.set_xlabel("FOSCTTM")
        ax.set_title(dataset_label(dataset))
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        panel_letter(ax, chr(ord("a") + panel))

        counts = sub.groupby("model")["mean_foscttm"].size()
        n_text = ", ".join(f"{method_label(m)} n={int(counts[m])}"
                           for m in order if m in counts.index)
        notes.append(f"{dataset_label(dataset)}: {n_text}")
        if multi_run:
            notes.append(f"{dataset_label(dataset)}: methods pooled across "
                         f"{sub['run'].nunique()} runs — arms are not guaranteed "
                         f"comparable; pass runs= to restrict")

    # Direction of goodness, once per row of panels, upright (§3.6).
    axes[0].text(0.0, -0.34, "lower = better", transform=axes[0].transAxes,
                 fontsize=6, color="0.35", ha="left", va="top")

    save_figure(fig, Path(save_path).with_suffix(""))
    return ["Each point is one seed; the bar is the median."] + notes
