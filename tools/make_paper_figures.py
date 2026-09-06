#!/usr/bin/env python
"""Pick which runs feed which panel. Panel code lives in src/visualization/."""
from __future__ import annotations

import argparse
import shutil
from itertools import permutations
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA

from src.visualization import (
    FOSCTTM_CHANCE, LINEAGE_COLORS, LINEAGE_ORDER, METHOD_COLORS, MODALITY_COLORS,
    MODALITY_MARKERS, dataset_label, lineage, method_label, state_color_map,
    state_label,
)
from src.visualization.style import (
    apply_style, chance_line, embedding_axes, figsize, ink, median_tick,
    panel_letter, save_figure, set_verify,
)
from src.visualization.ablation import ladder_panel, read_arms, real_panel
from src.evaluation.knn_alignment import alignment_curve
from src.evaluation.protocol import resolve_oos_mode
from src.training.registry import MODELS as REGISTERED_MODELS
from src.visualization.alignment import confusion_panel, modality_mixing
from src.visualization.benchmark import collect_benchmark, find_collapsed
from src.visualization.kinetics import collect_beta, plot_beta_recovery
from src.visualization.prediction import (
    canonical_rows, coverage_strip_panel, pooled_calibration_panel,
    ranked_protein_panel,
)
from src.visualization.diagnostics import (
    anchor_panel, checkpoint_panel, collect_flags, flag_panel,
    gradient_panel, read_by_config, read_checkpoints,
)
from src.visualization.linkage import (
    dataset_colors, funnel_panel, read_coverage, terms_panel,
)
from src.visualization.methods import (
    collect_runtimes, curated_method_runs, knn_curve_panel, runtime_panel,
)
from src.visualization.physics import (
    group_violin_panel, load_per_cell, paint_panel, pair_panel,
)
from src.visualization.runs import (
    CACHE_DIR, curated_runs, is_collapsed, read_diagnostics, run_rank,
)

POINT_STYLE = dict(s=0.9, alpha=0.45, linewidths=0, rasterized=True)
PREPROCESSED_DIR = Path("cache/preprocessed")


def seed_dirs(dataset: str, model: str, require: tuple[str, ...]):
    """Every non-collapsed seed directory for a model that has the files we need."""
    out = []
    for dj in CACHE_DIR.rglob(f"*/{model}/{dataset}/seed_*/diagnostics.json"):
        d = read_diagnostics(dj)
        if "mean_foscttm" not in d:
            continue
        if is_collapsed(d):
            continue                      # collapsed run, not a result
        if any(not (dj.parent / f).exists() for f in require):
            continue
        out.append((float(d["mean_foscttm"]), dj.parent))
    return out


def pick_run(dataset: str, model: str = "kot", target: str = "median",
             require: tuple[str, ...] = ("aligned_rna.npy",),
             prefer: tuple[str, ...] = ()) -> Path | None:
    """Representative run for an illustrative panel — median, not best.

    Restricted to the best-ranked tier first (see :func:`run_rank`), so a curated
    run always wins over an uncurated sweep before "median" is applied.
    """
    cands = seed_dirs(dataset, model, require)
    if not cands:
        return None
    curated = curated_runs()
    ranked = [(run_rank(d.relative_to(CACHE_DIR).parts[0], prefer, curated), score, d)
              for score, d in cands]
    best = min(rank for rank, _, _ in ranked)
    tier = sorted((score, d) for rank, score, d in ranked if rank == best)
    return tier[len(tier) // 2][1] if target == "median" else tier[0][1]


def pick_matched_runs(dataset: str, models: list[str],
                      require: tuple[str, ...] = ("aligned_rna.npy",),
                      prefer: tuple[str, ...] = ()) -> dict[str, Path] | None:
    """Seed directories for several arms **from one run and one seed**.

    An ablation panel whose arms come from different runs is comparing things
    trained under different conditions (§1.2), which is exactly the failure the
    duplicate-arm guard exists to prevent. If no single run carries every arm,
    return None rather than silently mixing runs.
    """
    by_key: dict[tuple[str, str], dict[str, Path]] = {}
    for model in models:
        for _, d in seed_dirs(dataset, model, require):
            parts = d.relative_to(CACHE_DIR).parts
            key = (parts[0], parts[-1])          # (run, seed)
            by_key.setdefault(key, {})[model] = d
    complete = [(k, v) for k, v in by_key.items() if len(v) == len(models)]
    if not complete:
        return None
    curated = curated_runs()
    complete.sort(key=lambda kv: (run_rank(kv[0][0], prefer, curated), kv[0]))
    return complete[0][1]


def load_aligned(run: Path) -> tuple[np.ndarray, np.ndarray]:
    """The aligned RNA and protein coordinates a run wrote."""
    return np.load(run / "aligned_rna.npy"), np.load(run / "aligned_protein.npy")


def newest_protein_cache(dataset_stem: str) -> Path | None:
    """The most recently written preprocessing cache for a dataset, or None.

    The hash in a cache filename is a preprocessing-cache key, so naming one here
    pins the figure to whatever the cache happened to be the day the line was
    written: regenerating the velocity mints a new hash and the old file keeps
    answering, silently labelling cells from superseded data. Resolving by mtime
    instead means the figure follows the current cache with nothing to update.
    `load_lineages` still checks the cell count, so a cache that does not match the
    plotted run is refused rather than mislabelled.
    """
    caches = sorted(PREPROCESSED_DIR.glob(f"{dataset_stem}_*.protein.h5ad"),
                    key=lambda f: f.stat().st_mtime)
    return caches[-1] if caches else None


def load_lineages(protein_h5ad: Path | None, n: int) -> pd.Series | None:
    """Coarse lineage per cell, or None with the reason printed.

    Silently returning None once shipped a figure with two blank-but-lettered
    panels and no warning, so every refusal says why.
    """
    if protein_h5ad is None:
        print("[fig3] no lineage labels: no protein cache in cache/preprocessed/ for "
              "this dataset — run preprocessing first")
        return None
    a = ad.read_h5ad(protein_h5ad, backed="r")
    if "cell_type" not in a.obs.columns:
        print(f"[fig3] no lineage labels: {protein_h5ad.name} has no 'cell_type' column")
        return None
    if len(a.obs) != n:
        print(f"[fig3] no lineage labels: {protein_h5ad.name} has {len(a.obs)} cells "
              f"but the run has {n}")
        return None
    return a.obs["cell_type"].astype(str).map(lineage).reset_index(drop=True)


def joint_embedding(xr: np.ndarray, xp: np.ndarray, idx: np.ndarray, seed: int = 0):
    stacked = np.vstack([xr[idx], xp[idx]])
    xy = PCA(n_components=2, random_state=seed).fit_transform(stacked)
    k = len(idx)
    return xy[:k], xy[k:]


def co_embedding_scatter(ax, xy_r: np.ndarray, xy_p: np.ndarray, rng):
    """Both modalities in one panel, plotted in random order.

    Drawing one modality after the other would put whichever went last on top
    everywhere, which reads as separation that the embedding does not have.
    """
    pts = np.vstack([xy_r, xy_p])
    cols = np.array([MODALITY_COLORS["RNA"]] * len(xy_r) +
                    [MODALITY_COLORS["Protein"]] * len(xy_p))
    order = rng.permutation(len(pts))
    ax.scatter(pts[order, 0], pts[order, 1], color=cols[order], **POINT_STYLE)


def bare_axes(ax):
    """No ticks, no spines, no arrows — for grid panels where one shared arrow pair
    on the corner panel names the axes for the whole figure."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def modality_legend(ax, loc: str = "best"):
    """The RNA / Protein key, once per figure rather than once per panel."""
    return ax.legend(handles=[
        Line2D([0], [0], marker="o", color="none", markersize=3,
               markerfacecolor=MODALITY_COLORS[name], label=name)
        for name in ("RNA", "Protein")], loc=loc)


def figure3(out: Path, max_cells: int = 12000, seed: int = 0):
    """Real CITE-seq: benchmark, co-embedding, and per-lineage alignment quality."""
    apply_style()
    rng = np.random.default_rng(seed)

    bench = collect_benchmark()
    bench = bench[~find_collapsed(bench)]
    bench = bench[~bench.model.isin(["kot_anchor", "totalvi"])]
    # Curated runs only, same rule as branch_identity_pairs and collect_beta: the
    # uncurated sweep arms differ in lambda_dyn and kappa bounds, so pooling them
    # into one row per method plots a spread over configurations that were never
    # meant to be compared against each other.
    curated = curated_runs()
    if curated:
        dropped = sorted(set(bench["run"]) - set(curated))
        bench = bench[bench["run"].isin(curated)]
        print(f"[fig3] benchmark from {bench['run'].nunique()} curated runs "
              f"({len(dropped)} uncurated dropped)")

    run = pick_run("bmmc_cite_retained", prefer=("bmmc_scvelo_ablation",))
    if run is None:
        print("[fig3] no usable BMMC run with aligned arrays; skipping")
        return
    print(f"[fig3] co-embedding panels from {run}")

    xr, xp = load_aligned(run)
    fos = pd.read_csv(run / "foscttm.csv")["foscttm"].to_numpy()
    lin = load_lineages(newest_protein_cache("bmmc_cite_scvelo_results_retained"), len(xr))

    idx = rng.choice(len(xr), min(max_cells, len(xr)), replace=False)
    xy_r, xy_p = joint_embedding(xr, xp, idx, seed=seed)

    fig = plt.figure(figsize=figsize("full", 4.6), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.09, w_pad=0.07, hspace=0.07, wspace=0.06)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15],
                          width_ratios=[1.35, 1.0])

    # --- a: benchmark across both real datasets -----------------------------
    ax = fig.add_subplot(gs[0, 0])
    order = ["kot", "kot_nodyn", "maxfuse", "moscot", "scot", "glue", "uniport",
             "linear_ode"]
    datasets = ["bmmc_cite_retained", "pbmc_retained"]
    offsets = {datasets[0]: -0.17, datasets[1]: +0.17}
    marks = {datasets[0]: "o", datasets[1]: "s"}
    fills = {datasets[0]: "none", datasets[1]: "#3A3A3A"}
    ax.axhspan(-0.45, 0.45, color="#0072B2", alpha=0.07, lw=0, zorder=1)
    for ds in datasets:
        sub = bench[bench.dataset == ds]
        for row, model in enumerate(order):
            vals = sub.loc[sub.model == model, "mean_foscttm"].to_numpy()
            if not len(vals):
                continue
            y = row + offsets[ds]
            focal = model == "kot"
            ax.scatter(vals, np.full(len(vals), y) + rng.uniform(-.05, .05, len(vals)),
                       s=7 if focal else 5, marker=marks[ds],
                       facecolors=fills[ds], edgecolors="#3A3A3A",
                       linewidths=.5, alpha=.8 if focal else .5, zorder=6)
            if len(vals) >= 3:
                median_tick(ax, float(np.median(vals)), y, half=0.12, lw=1.1, zorder=7)
    chance_line(ax, FOSCTTM_CHANCE, axis="x", label="chance")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([method_label(m) for m in order])
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlim(0, 0.62)
    ax.invert_yaxis()
    ax.set_xlabel("FOSCTTM")
    ax.set_title("Alignment accuracy")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
               markeredgecolor="#3A3A3A", markersize=3, label="BMMC"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#3A3A3A",
               markeredgecolor="#3A3A3A", markersize=3, label="PBMC")], loc="lower right")
    ax.text(0.0, -0.20, "lower = better", transform=ax.transAxes,
            fontsize=6, color="0.35", ha="left", va="top")
    panel_letter(ax, "a")

    # --- b: per-lineage alignment quality -----------------------------------
    ax = fig.add_subplot(gs[0, 1])
    groups = []
    if lin is not None:
        groups = [(g, fos[(lin == g).to_numpy()]) for g in LINEAGE_ORDER
                  if (lin == g).sum() >= 30]
        if not groups:
            print("[fig3] no lineage reaches 30 cells; panel b left empty")
    if groups:
        grid = np.linspace(0, 0.75, 200)
        peak = max(float(gaussian_kde(v)(grid).max()) for _, v in groups)
        for i, (name, vals) in enumerate(reversed(groups)):
            d = gaussian_kde(vals)(grid) / peak * 0.92
            c = LINEAGE_COLORS[name]
            ax.fill_between(grid, i, i + d, color=c, alpha=.78, lw=0, zorder=3 + i)
            ax.plot(grid, i + d, color=ink(c), lw=.7, zorder=3 + i)
            ax.text(-0.012, i + .2, name, transform=ax.get_yaxis_transform(),
                    ha="right", va="center", fontsize=5.5, color=ink(c))
        chance_line(ax, FOSCTTM_CHANCE, axis="x", label="chance")
        ax.set_yticks([]); ax.spines["left"].set_visible(False)
        ax.set_xlim(0, 0.75); ax.set_ylim(-0.08, len(groups))
        ax.set_xlabel("FOSCTTM")
        ax.set_title("By lineage")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
    panel_letter(ax, "b")

    # --- c/d: co-embedding, by modality then by lineage ---------------------
    ax = fig.add_subplot(gs[1, 0])
    co_embedding_scatter(ax, xy_r, xy_p, rng)
    mix = modality_mixing(xy_r, xy_p, random_state=seed)
    ax.set_title("Co-embedding, by modality" +
                 (f"  (mixing {mix:.2f})" if mix is not None else ""))
    modality_legend(ax)
    embedding_axes(ax, "PC 1", "PC 2")
    panel_letter(ax, "c")

    ax = fig.add_subplot(gs[1, 1])
    if lin is not None:
        ls = lin.to_numpy()[idx]
        for name in LINEAGE_ORDER:
            m = ls == name
            if m.sum() == 0:
                continue
            ax.scatter(xy_r[m, 0], xy_r[m, 1], color=LINEAGE_COLORS[name], **POINT_STYLE)
            ax.scatter(xy_p[m, 0], xy_p[m, 1], color=LINEAGE_COLORS[name],
                       marker=MODALITY_MARKERS["Protein"], **POINT_STYLE)
    ax.set_title("Co-embedding, by lineage")
    ax.text(0.99, 0.02, "colours as in b", transform=ax.transAxes,
            fontsize=6, color="0.45", ha="right", va="bottom")
    embedding_axes(ax, "PC 1", "PC 2")
    panel_letter(ax, "d")

    save_figure(fig, out / "fig3_realdata")


def branch_scores(confusion_matrix) -> tuple[float, float] | None:
    """Identity and separation accuracy from one branch confusion matrix.

    identity   = accuracy as labelled — did the model pick the right branch.
    separation = best accuracy over label permutations — did the branches end up
                 apart at all, regardless of which name they got.
    """
    M = np.asarray(confusion_matrix, float)
    tot = M.sum()
    if tot <= 0:
        return None
    raw = float(np.trace(M) / tot)
    best = max(sum(M[i, p[i]] for i in range(M.shape[0]))
               for p in permutations(range(M.shape[0])))
    return raw, float(best / tot)


def branch_identity_pairs() -> pd.DataFrame | None:
    """Branch separation and identity for kot vs kot_nodyn, matched on (run, seed).

    The gap between the two is the whole point of the branch stage.

    Restricted to curated runs when a MANIFEST exists: pooling every branch sweep
    mixes configurations that are not comparable, and the uncurated ones include
    settings where the mechanism does not fire at all.
    """
    curated = curated_runs()
    rows = []
    for dj in CACHE_DIR.rglob("*/synthetic_linked_ode/seed_*/diagnostics.json"):
        parts = dj.relative_to(CACHE_DIR).parts
        run, model = parts[0], parts[1]
        if "branch" not in run or model not in ("kot", "kot_nodyn"):
            continue
        if curated and run not in curated:
            continue
        d = read_diagnostics(dj)
        cm = d.get("branch_confusion_matrix")
        if not cm or len(cm) < 2:
            continue
        if is_collapsed(d):
            continue
        sc = branch_scores(cm)
        if sc is None:
            continue
        rows.append({"run": run, "seed": parts[3], "model": model,
                     "raw": sc[0], "sep": sc[1]})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    wide = df.pivot_table(index=["run", "seed"], columns="model",
                          values=["raw", "sep"], aggfunc="first")
    wide.columns = [f"{m}_{k}" for k, m in wide.columns]
    need = ["kot_raw", "kot_nodyn_raw", "kot_sep", "kot_nodyn_sep"]
    if any(c not in wide.columns for c in need):
        return None
    return wide.dropna(subset=need).reset_index()


def branch_confusion(model: str) -> tuple[list, list] | None:
    """The median-identity curated branch run's confusion matrix for one model.

    Median rather than best, so the panel shows a typical run of that arm and not the
    seed where the degeneracy happened to resolve.
    """
    curated = curated_runs()
    hits = []
    for dj in CACHE_DIR.rglob(f"*/{model}/synthetic_linked_ode/seed_*/diagnostics.json"):
        run = dj.relative_to(CACHE_DIR).parts[0]
        if "branch" not in run or (curated and run not in curated):
            continue
        d = read_diagnostics(dj)
        cm, labels = d.get("branch_confusion_matrix"), d.get("branch_confusion_labels")
        if not cm or not labels or is_collapsed(d):
            continue
        scores = branch_scores(cm)
        if scores is not None:
            hits.append((scores[0], cm, labels))
    if not hits:
        return None
    hits.sort(key=lambda h: h[0])
    _, cm, labels = hits[len(hits) // 2]
    return cm, labels


def figure1b(out: Path, max_cells: int = 8000, seed: int = 0):
    """The ablation as a picture: what the kinetics term does to the co-embedding.

    This is the only plotted panel of Figure 1 — panel a is a schematic that has
    to be drawn by hand in a vector editor and dropped in alongside this file.
    """
    apply_style()
    rng = np.random.default_rng(seed)

    arms = [("kot", "With kinetics"), ("kot_nodyn", "Alignment only")]
    matched = pick_matched_runs("synthetic_linked_ode", [m for m, _ in arms],
                                prefer=("syn_branch_ablation",))
    if matched is None:
        print("[fig1b] no single run carries both kot and kot_nodyn with aligned "
              "arrays — arms from different runs are not comparable; skipping")
        return
    found = [(m, lbl, matched[m]) for m, lbl in arms]

    fig, axes = plt.subplots(1, 2, figsize=figsize("full", 2.3), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.08, wspace=0.05)

    for ax, (model, label, run), letter in zip(axes, found, "ab"):
        xr, xp = load_aligned(run)
        idx = rng.choice(len(xr), min(max_cells, len(xr)), replace=False)
        xy_r, xy_p = joint_embedding(xr, xp, idx, seed=seed)
        co_embedding_scatter(ax, xy_r, xy_p, rng)

        mix = modality_mixing(xy_r, xy_p, random_state=seed)
        fos = float(np.mean(pd.read_csv(run / "foscttm.csv")["foscttm"]))
        ax.set_title(label)
        ax.text(0.5, -0.04,
                f"FOSCTTM {fos:.3f}" + (f" · mixing {mix:.2f}" if mix is not None else ""),
                transform=ax.transAxes, fontsize=6.5, color="0.3",
                ha="center", va="top")
        embedding_axes(ax, "PC 1", "PC 2")
        panel_letter(ax, letter)
        print(f"[fig1b] {model}: FOSCTTM {fos:.3f} from {run}")

    modality_legend(axes[0])

    save_figure(fig, out / "fig1b_ablation")


def protein_path_of(run: Path) -> Path | None:
    """The protein h5ad this run's own config names.

    The staged synthetic dataset has three stages with different ground truth, so
    reading a fixed path would silently pair a run with another stage's truth.
    """
    cfg = yaml.safe_load((run / "run_config.yaml").read_text()) or {}
    # run_config.yaml nests the resolved paths under `dataset_paths`.
    paths = cfg.get("dataset_paths") or {}
    pp = paths.get("protein_path") or cfg.get("protein_path")
    return Path(pp) if pp else None


def figure2(out: Path, stage: str = "branch", max_cells: int = 8000, seed: int = 0):
    """Controlled recovery on the synthetic linked-ODE system.

    Defaults to the `branch` stage: it carries the branching structure the
    method is actually meant to handle, and it is the harder case. True kappa
    is constant by construction there, so panel d shows the fitted spread
    against that constant rather than a scatter.
    """
    apply_style()
    rng = np.random.default_rng(seed)

    req = ("aligned_rna.npy", "per_cell_diagnostics.npz", "run_config.yaml")
    cands = seed_dirs("synthetic_linked_ode", "kot", req)
    if not cands:
        print("[fig2] no synthetic run with per-cell arrays and a config; skipping")
        return

    resolved = [(f, d, protein_path_of(d)) for f, d in cands]
    resolved = [(f, d, pp) for f, d, pp in resolved if pp is not None and pp.exists()]
    if not resolved:
        print("[fig2] no run whose config names an existing protein h5ad; skipping")
        return
    # Prefer the requested stage's restored / ablation runs. Wrong-weight sweeps
    # (λ_dyn=1 under mean-aggregated kinetics, or gauge-normalized λ_dyn=10/100)
    # land near chance and must not become the figure just because their FOSCTTM
    # sits at the median of a mixed cache.
    on_stage = [r for r in resolved if f"/{stage}/" in str(r[2])] or resolved
    restored = [r for r in on_stage
                if f"syn_{stage}_kot_restored" in str(r[1])]
    ablation = [r for r in on_stage
                if f"syn_{stage}_ablation" in str(r[1])]
    preferred = restored or ablation or on_stage
    preferred.sort()
    _, run, prot = preferred[len(preferred) // 2]
    print(f"[fig2] panels from {run}")
    print(f"[fig2] ground truth from {prot}")
    a = ad.read_h5ad(prot)

    xr, xp = load_aligned(run)
    n = len(xr)
    obs = a.obs.iloc[:n].reset_index(drop=True)
    z = np.load(run / "per_cell_diagnostics.npz", allow_pickle=True)

    fig, axes = plt.subplots(1, 5, figsize=figsize("full", 1.85), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.07, wspace=0.06)

    # a: the generative system, coloured by state
    ax = axes[0]
    xy = PCA(n_components=2, random_state=seed).fit_transform(
        np.asarray(a.layers["protein_mean"])[:n])
    states = obs["state"].to_numpy()
    for st, c in state_color_map(states).items():
        m = states == st
        ax.scatter(xy[m, 0], xy[m, 1], color=c, label=state_label(st), **POINT_STYLE)
    ax.set_title("Ground truth")
    ax.legend(markerscale=6, loc="best")
    embedding_axes(ax, "PC 1", "PC 2")
    panel_letter(ax, "a")

    # b: co-embedding after alignment
    ax = axes[1]
    idx = rng.choice(n, min(max_cells, n), replace=False)
    xy_r, xy_p = joint_embedding(xr, xp, idx, seed=seed)
    co_embedding_scatter(ax, xy_r, xy_p, rng)
    mix = modality_mixing(xy_r, xy_p, random_state=seed)
    ax.set_title("After alignment")
    if mix is not None:
        # Inside the panel, top-left: below the axes it lands on the "PC 1" arrow
        # label that embedding_axes draws in that corner.
        ax.text(0.02, 0.98, f"mixing {mix:.2f}", transform=ax.transAxes,
                fontsize=6.5, color="0.3", ha="left", va="top")
    embedding_axes(ax, "PC 1", "PC 2")
    panel_letter(ax, "b")

    # c: where along the trajectory the pseudotime estimate breaks down
    ax = axes[2]
    t_true = obs["true_time"].to_numpy()
    err = np.asarray(z["time_err"], float)
    k = min(len(t_true), len(err))
    ax.scatter(t_true[:k], err[:k], s=0.8, alpha=0.25, linewidths=0,
               color=METHOD_COLORS["kot"], rasterized=True)
    bins = np.linspace(np.nanmin(t_true[:k]), np.nanmax(t_true[:k]), 13)
    # digitize puts a value equal to bins[-1] in an extra bin above the last one,
    # which then has no median line; clip it back into the top bin.
    which = np.clip(np.digitize(t_true[:k], bins), 1, len(bins) - 1)
    med = [np.nanmedian(err[:k][which == i]) if (which == i).sum() > 5 else np.nan
           for i in range(1, len(bins))]
    ax.plot(0.5 * (bins[:-1] + bins[1:]), med, color="#1A1A1A", lw=1.1, zorder=6)
    ax.set_xlabel("True pseudotime")
    ax.set_ylabel("Pseudotime error")
    ax.set_title("Timing error")
    ax.margins(x=0.04, y=0.08)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
    panel_letter(ax, "c")

    # d: the proof-of-mechanism panel. The branch stage is built mirror-symmetric
    # on purpose (see src/data/synthetic_linked_ode.py): the A<->B protein swap is
    # exact, so optimal transport alone cannot tell the branches apart. Both arms
    # still SEPARATE the branches; the question is whether the kinetics term
    # resolves their IDENTITY. Arms come from the same run and seed.
    ax = axes[3]
    pairs = branch_identity_pairs()
    if pairs is not None and len(pairs):
        groups = [("Separation", "sep"), ("Identity", "raw")]
        arms = [("kot", "with kinetics", METHOD_COLORS["kot"]),
                ("kot_nodyn", "alignment only", "#8C8C8C")]
        for gi, (_, key) in enumerate(groups):
            for ai, (model, _, color) in enumerate(arms):
                vals = pairs[f"{model}_{key}"].to_numpy()
                x = gi + (ai - 0.5) * 0.34
                ax.scatter(np.full(len(vals), x) + rng.uniform(-.045, .045, len(vals)),
                           vals, s=5, color=color, alpha=.65, linewidths=0, zorder=5)
                median_tick(ax, float(np.median(vals)), x, half=0.12,
                            color=ink(color), lw=1.3, orient="horizontal", zorder=6)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([g for g, _ in groups], rotation=30, ha="right")
        ax.set_xlim(-0.55, len(groups) - 0.45)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Fraction of cells")
        ax.set_title("Branch resolution")
        ax.legend(handles=[
            Line2D([0], [0], marker="o", color="none", markersize=3,
                   markerfacecolor=c, label=lbl) for _, lbl, c in arms],
            loc="lower left")
        ax.text(0.99, 0.5, f"n = {len(pairs)} seeds", transform=ax.transAxes,
                fontsize=6, color="0.45", ha="right", va="center")
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
    else:
        ax.text(0.5, 0.5, "no matched kot / kot_nodyn\nbranch pairs",
                transform=ax.transAxes, fontsize=6, color="0.45",
                ha="center", va="center")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("Branch resolution")
    panel_letter(ax, "d")

    # e: the same degeneracy as a confusion matrix. Panel d gives the numbers; this
    # says WHERE the alignment-only arm sends the cells, which is the A<->B swap the
    # mirror-symmetric construction makes unresolvable without dynamics. Only the
    # failing arm is shown — KOT's matrix is diagonal, and panel d already reports it.
    ax = axes[4]
    found = branch_confusion("kot_nodyn")
    if found is None:
        ax.text(0.5, 0.5, "no curated branch run\nwith a confusion matrix",
                transform=ax.transAxes, color="0.45", ha="center", va="center")
        ax.set_xticks([]); ax.set_yticks([])
    else:
        confusion_panel(ax, *found)
    ax.set_title("Alignment only")
    panel_letter(ax, "e")

    print(f"[fig2] stage={stage}  mixing={mix if mix is None else round(mix, 3)}")
    save_figure(fig, out / "fig2_synthetic")


# The configuration the synthetic and real ablations were both run under, and the one
# config/training.yaml declares. The sweep also ran (0.01, 500); pooling the two would
# average a training difference into the corruption effect.
ABLATION_LR_BETA = 0.001
ABLATION_WARMUP = 300


def figure5(out: Path):
    """Velocity corruption: a dose-response on synthetic, and what it does on real data."""
    apply_style()
    fig, axd = plt.subplot_mosaic([["a", "c"], ["b", "c"]], figsize=figsize("full", 3.4))

    syn = read_arms("synthetic_branch_summary.csv", ABLATION_LR_BETA, ABLATION_WARMUP)
    # kot_fixedalpha is left out: it sits at FOSCTTM ~0.47 on every rung, so it has no
    # dose-response to show and its non-monotone cosine only obscures the two arms that
    # do. It belongs in the ablation table, not in the panel making this claim.
    arms = ["kot_oracle", "kot_fixedkappa"]

    ax = axd["a"]
    ladder_panel(ax, syn, "mean_foscttm", arms, ylabel="FOSCTTM",
                 chance=FOSCTTM_CHANCE)
    ax.tick_params(labelbottom=False)
    ax.set_title("Synthetic: alignment")
    ax.legend(loc="upper left", frameon=False)
    panel_letter(ax, "a")

    ax = axd["b"]
    ladder_panel(ax, syn, "jvp_rhs_cos_median", arms, ylabel="JVP\u00b7RHS cosine")
    ax.set_title("Synthetic: ODE agreement")
    ax.set_ylim(-0.95, 1.1)
    ax.set_xlabel("Velocity corrupted")
    panel_letter(ax, "b")

    ax = axd["c"]
    real = read_arms("kinetics_ablation_real_summary.csv", ABLATION_LR_BETA, ABLATION_WARMUP)
    real_panel(ax, real, "mean_foscttm", ["bmmc_cite_retained", "pbmc_retained"],
               xlabel="FOSCTTM", chance=FOSCTTM_CHANCE)
    ax.set_title("Real CITE-seq")
    ax.legend(handles=[
        Line2D([0], [0], marker=m, color=METHOD_COLORS["kot"], lw=0, ms=3.4,
               fillstyle=f, markeredgewidth=0.8, label=dataset_label(d))
        for d, m, f in (("bmmc_cite_retained", "o", "none"), ("pbmc_retained", "s", "full"))],
        loc="lower right", frameon=False)
    panel_letter(ax, "c")

    fig.tight_layout()
    save_figure(fig, out / "fig5_velocity")


def figure6(out: Path, dataset: str = "bmmc_cite_retained", max_cells: int = 14000,
            seed: int = 0):
    """Where the ODE holds: the per-cell cosine on the embedding, by lineage, and
    against per-cell pairing quality."""
    apply_style()
    rng = np.random.default_rng(seed)

    run = pick_run(dataset, "kot",
                   require=("aligned_rna.npy", "per_cell_diagnostics.npz"))
    if run is None:
        print(f"[fig6] no {dataset} run carries per_cell_diagnostics.npz")
        return
    per_cell = load_per_cell(run)
    xr, xp = load_aligned(run)
    n = len(xr)
    idx = rng.choice(n, min(n, max_cells), replace=False)
    xy_r, _ = joint_embedding(xr, xp, idx, seed)
    cos = per_cell["jvp_rhs_cos"]

    fig, axes = plt.subplots(1, 3, figsize=figsize("full", 2.4))

    ax = axes[0]
    mappable = paint_panel(ax, xy_r, cos[idx])
    embedding_axes(ax)
    ax.set_title("ODE agreement per cell")
    bar = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.02)
    bar.set_label("JVP·RHS cosine")
    bar.outline.set_visible(False)
    panel_letter(ax, "a")

    ax = axes[1]
    lineages = load_lineages(newest_protein_cache(dataset.replace("_retained", "")), n)
    if lineages is None:
        ax.set_visible(False)
    else:
        group_violin_panel(ax, cos, lineages.to_numpy(), LINEAGE_ORDER, LINEAGE_COLORS)
        ax.set_ylabel("JVP·RHS cosine")
        ax.set_title("By lineage")
    panel_letter(ax, "b")

    ax = axes[2]
    pair_panel(ax, cos, per_cell["foscttm"])
    ax.set_xlabel("JVP·RHS cosine")
    ax.set_ylabel("FOSCTTM")
    ax.set_title("Fit versus pairing")
    chance_line(ax, FOSCTTM_CHANCE)
    panel_letter(ax, "c")

    fig.tight_layout()
    save_figure(fig, out / "fig6_percell")


def figure7(out: Path):
    """Predicted protein against measured protein: coverage, per marker, and pooled."""
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=figsize("full", 2.5),
                             gridspec_kw=dict(width_ratios=[1, 1.5, 1.2]))

    # Short names: panel a fits four group ticks in a third-width panel, and the full
    # dataset names collide over them. They are spelled out in panel b's title.
    frames = {short: canonical_rows(d) for short, d in
              (("BMMC", "bmmc_cite_retained"), ("PBMC", "pbmc_retained"))}

    ax = axes[0]
    coverage_strip_panel(ax, frames)
    ax.set_ylabel("Spearman ρ, per protein")
    panel_letter(ax, "a")

    ax = axes[1]
    ranked_protein_panel(ax, frames["PBMC"])
    ax.set_xlabel("Protein, ranked")
    ax.set_ylabel("Spearman ρ")
    ax.set_title(f'{dataset_label("pbmc_retained")}, 12 seeds')
    ax.legend(loc="lower left", frameon=False)
    panel_letter(ax, "b")

    ax = axes[2]
    # PBMC only: it is the one real dataset whose runs saved the full ADT panel.
    run = pick_run("pbmc_retained", "kot",
                   require=("phi_full_panel.npy", "protein_full_panel.npy"))
    if run is None:
        print("[fig7] no PBMC run saved phi_full_panel.npy")
        ax.set_visible(False)
    else:
        pooled_calibration_panel(ax, np.load(run / "phi_full_panel.npy"),
                                 np.load(run / "protein_full_panel.npy"))
        ax.set_xlabel("Measured ADT (z)")
        ax.set_ylabel("Predicted φ(r) (z)")
        ax.set_title("All markers pooled")
    panel_letter(ax, "c")

    fig.tight_layout()
    save_figure(fig, out / "fig7_prediction")


# Methods carried through the cross-method panels, in the Fig. 3a row order. moscot,
# SCOT and Linear ODE are absent because their curated runs did not save aligned
# arrays, not because they were dropped — the caption has to say which eight became six.
GRID_MODELS = ["kot", "kot_nodyn", "maxfuse", "glue", "uniport"]


def figure8(out: Path, dataset: str = "bmmc_cite_retained", max_cells: int = 9000,
            seed: int = 0):
    """Every method's co-embedding, coloured by modality and by lineage."""
    apply_style()
    rng = np.random.default_rng(seed)

    runs = curated_method_runs(dataset, GRID_MODELS)
    models = [m for m in GRID_MODELS if m in runs]
    if not models:
        print(f"[fig8] no curated {dataset} runs with aligned arrays")
        return
    print(f"[fig8] {len(models)} methods: {', '.join(models)}")

    fig, axes = plt.subplots(2, len(models), figsize=figsize("full", 2.6),
                             layout="constrained")
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.01, hspace=0.02)
    cache = newest_protein_cache(dataset.replace("_retained", ""))

    for col, model in enumerate(models):
        xr, xp = load_aligned(runs[model])
        n = len(xr)
        idx = rng.choice(n, min(n, max_cells), replace=False)
        xy_r, xy_p = joint_embedding(xr, xp, idx, seed)

        ax = axes[0, col]
        co_embedding_scatter(ax, xy_r, xy_p, rng)
        ax.set_title(method_label(model))
        # Mixing inside the panel, not appended to the title: a two-line title
        # collides with the panel letter, and the number belongs to the picture.
        ax.text(0.02, 0.98, f"mixing {modality_mixing(xy_r, xy_p):.2f}",
                transform=ax.transAxes, color="0.35", ha="left", va="top")
        bare_axes(ax)

        ax = axes[1, col]
        lineages = load_lineages(cache, n)
        if lineages is None:
            # MaxFuse's batching returns fewer cells than the run started with, so its
            # rows cannot be mapped back to cell types. Said on the panel rather than
            # left as a blank the reader has to interpret.
            ax.text(0.5, 0.5, f"labels unavailable\n(n = {n:,})", transform=ax.transAxes,
                    ha="center", va="center", color="0.45")
            bare_axes(ax)
            continue
        labels = lineages.to_numpy()[idx]
        for name in LINEAGE_ORDER:
            mask = labels == name
            if mask.any():
                ax.scatter(xy_r[mask, 0], xy_r[mask, 1], color=LINEAGE_COLORS[name],
                           **POINT_STYLE)
        bare_axes(ax)

    embedding_axes(axes[1, 0], "Dim 1", "Dim 2")
    modality_legend(axes[0, -1], loc="lower right")
    fig.text(0.005, 0.72, "By modality", rotation=90, va="center", ha="left")
    fig.text(0.005, 0.28, "By lineage", rotation=90, va="center", ha="left")
    panel_letter(axes[0, 0], "a")
    panel_letter(axes[1, 0], "b")
    save_figure(fig, out / "fig8_coembedding")


def figure9(out: Path, seed: int = 0):
    """Beyond FOSCTTM: where the true partner ranks, and what each method costs."""
    apply_style()
    datasets = ["bmmc_cite_retained", "pbmc_retained"]
    fractions = np.geomspace(1e-4, 0.1, 24)

    fig, axes = plt.subplots(1, 2, figsize=figsize("full", 2.3), layout="constrained")

    ax = axes[0]
    runs = curated_method_runs(datasets[0], GRID_MODELS)
    curves = {}
    for model in GRID_MODELS:
        if model not in runs:
            continue
        xr, xp = load_aligned(runs[model])
        curves[model] = alignment_curve(xr, xp, fractions, seed=seed)
        print(f"[fig9] {model}: top-1% recovery {curves[model][fractions >= 0.01][0]:.3f}")
    knn_curve_panel(ax, curves, fractions)
    ax.set_title(f"{dataset_label(datasets[0])}, 6k-cell subsample")
    ax.legend(loc="upper left", frameon=False)
    panel_letter(ax, "a")

    ax = axes[1]
    runtime_panel(ax, collect_runtimes(datasets), GRID_MODELS)
    ax.set_title("Cost")
    ax.text(0.03, 0.97, "mixed hardware", transform=ax.transAxes, color="0.45",
            ha="left", va="top")
    panel_letter(ax, "b")

    save_figure(fig, out / "fig9_beyond_foscttm")


# The two real CITE-seq panels. Papalexi has four antibodies, so its funnel is a flat
# line that says nothing about attrition; it belongs in the perturbation figure instead.
COVERAGE_DATASETS = ["bmmc_cite_retained", "pbmc_retained"]


def figure1c(out: Path):
    """How much RNA-protein linkage survives, and which term each survivor feeds."""
    apply_style()
    coverage = read_coverage(COVERAGE_DATASETS)
    # Modality colours would be wrong here (nothing on this figure is RNA-vs-protein) and
    # method colours would be wrong too, so the two datasets take neutral tints of the
    # focal blue, distinguished by lightness rather than hue.
    colors = dataset_colors(COVERAGE_DATASETS, ["#0072B2", "#9ECAE1"])

    fig, axes = plt.subplots(1, 2, figsize=figsize("full", 2.2), layout="constrained",
                             gridspec_kw=dict(width_ratios=[1.25, 1]))

    funnel_panel(axes[0], coverage, colors)
    axes[0].set_title("Attrition of the antibody panel")
    panel_letter(axes[0], "a")

    terms_panel(axes[1], coverage, colors)
    axes[1].set_title("What each link feeds")
    axes[1].legend(loc="upper right", frameon=False)
    panel_letter(axes[1], "b")

    save_figure(fig, out / "fig1c_linkage")


def supplement(out: Path):
    """Two supplementary figures: how checkpoints and gradients behaved, and how often
    a run failed or a knob mattered."""
    apply_style()
    dest = out / "appendix"
    colors = dataset_colors(COVERAGE_DATASETS, ["#0072B2", "#9ECAE1"])

    # --- S1: selection and optimisation ------------------------------------
    checkpoints = read_checkpoints()
    # KOT only. The question is what the checkpoint choice costs, and adding the
    # ablation arm doubles the points without bearing on it.
    models = ["kot"]
    fig, axes = plt.subplots(1, 3, figsize=figsize("full", 2.4), layout="constrained")

    ax = axes[0]
    checkpoint_panel(ax, checkpoints, "mean_foscttm", models)
    ax.set_ylabel("FOSCTTM")
    ax.set_title("Cost of the checkpoint choice")
    panel_letter(ax, "a")

    ax = axes[1]
    checkpoint_panel(ax, checkpoints, "jvp_rhs_cos", models)
    ax.set_ylabel("JVP·RHS cosine")
    ax.set_title("…on the other metric")
    panel_letter(ax, "b")

    ax = axes[2]
    gradient_panel(ax, read_by_config("summary_warmup_lambda_cfgBC_by_config.csv"))
    ax.set_title("Do the two terms fight?")
    ax.legend(loc="best", frameon=False)
    panel_letter(ax, "c")

    save_figure(fig, dest / "figS1_optimization")

    # --- S2: failure rate and what the knobs buy ---------------------------
    fig, axes = plt.subplots(1, 3, figsize=figsize("full", 2.4), layout="constrained")

    ax = axes[0]
    flags = collect_flags(COVERAGE_DATASETS)
    flag_panel(ax, flags, colors)
    ax.set_title("Run outcomes")
    ax.legend(loc="upper right", frameon=False)
    panel_letter(ax, "a")

    anchors = read_by_config("anchor_ablation_cfgB_by_config.csv")
    ax = axes[1]
    anchor_panel(ax, anchors, "mean_foscttm", colors)
    ax.set_ylabel("FOSCTTM")
    ax.set_title("Anchors and alignment")
    panel_letter(ax, "b")

    ax = axes[2]
    anchor_panel(ax, anchors, "beta_anchor_mean_abs_err", colors)
    ax.set_ylabel(r"$\beta$ anchor error")
    ax.set_title("Anchors and β")
    ax.legend(loc="best", frameon=False)
    panel_letter(ax, "c")

    save_figure(fig, dest / "figS2_stability")


def tables(out: Path):
    """Supplementary tables: what each method was allowed to see, and which links exist.

    Written as CSV rather than drawn. Both are reference material a reader looks a row up
    in, and a rendered image of a table cannot be searched, sorted or copied.
    """
    dest = out / "tables"
    dest.mkdir(parents=True, exist_ok=True)

    # Read from the registry rather than a list kept here: a model added to the runner
    # and forgotten in this table is exactly how a method ends up in a benchmark with no
    # declared protocol.
    protocol = pd.DataFrame(
        [{"model": method_label(m), "out_of_sample_mode": resolve_oos_mode(m)}
         for m in sorted(set(REGISTERED_MODELS) | {"kot"})])
    protocol.to_csv(dest / "protocol.csv", index=False)
    print(f"[tables] protocol.csv: {len(protocol)} models")

    links = []
    for dataset, stem in (("BMMC CITE-seq", "bmmc_cite_retained"),
                          ("PBMC CITE-seq", "pbmc_retained"),
                          ("Papalexi ECCITE-seq", "papalexi")):
        path = Path("cache/results/mapping") / f"adt_mapping_{stem}.csv"
        if not path.exists():
            print(f"[tables] no mapping CSV for {dataset}")
            continue
        table = pd.read_csv(path)
        table.insert(0, "dataset", dataset)
        links.append(table)
    if links:
        mapping = pd.concat(links, ignore_index=True)
        mapping.to_csv(dest / "adt_gene_links.csv", index=False)
        used = mapping["use_for_alignment"] & mapping["present_in_rna"]
        print(f"[tables] adt_gene_links.csv: {len(mapping)} rows, {int(used.sum())} "
              "usable alignment links")


def appendix(out: Path):
    """Copy the per-run diagnostic figures the appendix cites into one place."""
    dest = out / "appendix"
    dest.mkdir(parents=True, exist_ok=True)
    wanted = {
        "training_loss":       ("synthetic_linked_ode", "kot"),
        "grad_interaction":    ("pbmc_retained", "kot"),
        "foscttm_diagnostics": ("synthetic_linked_ode", "kot"),
    }
    for name, (dataset, model) in wanted.items():
        run = pick_run(dataset, model=model)
        if run is not None and (run / f"{name}.pdf").exists():
            src = run / f"{name}.pdf"
        else:
            hits = sorted(CACHE_DIR.rglob(f"*/{model}/{dataset}/seed_*/{name}.pdf"),
                          key=lambda q: q.stat().st_mtime, reverse=True)
            src = hits[0] if hits else None
        if src is None:
            print(f"[appendix] {name}: no PDF yet — rerun training to emit it")
            continue
        for ext in ("pdf", "png"):
            s_ = src.with_suffix("." + ext)
            if s_.exists():
                shutil.copy2(s_, dest / f"{name}.{ext}")
        print(f"[appendix] {name} <- {src}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("figures"))
    ap.add_argument("--no-verify", dest="verify", action="store_false",
                    help="Skip the bbox-collision check on every saved figure. It is "
                         "on by default; this turns it off for a quick redraw.")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    set_verify(args.verify)

    print("\n=== Figure 1b: ablation co-embedding ===")
    figure1b(args.out)

    print("\n=== Figure 1c: linkage coverage ===")
    figure1c(args.out)

    print("\n=== Figure 2: synthetic recovery ===")
    figure2(args.out)

    print("\n=== Figure 3: real CITE-seq ===")
    figure3(args.out)

    print("\n=== Figure 4: kinetic recovery ===")
    for note in plot_beta_recovery(collect_beta(), args.out / "fig4_kinetics"):
        print("   ", note)

    print("\n=== Figure 5: velocity corruption ===")
    figure5(args.out)

    print("\n=== Figure 6: per-cell physics ===")
    figure6(args.out)

    print("\n=== Figure 7: protein prediction ===")
    figure7(args.out)

    print("\n=== Figure 8: co-embedding by method ===")
    figure8(args.out)

    print("\n=== Figure 9: rank shape and cost ===")
    figure9(args.out)

    print("\n=== Appendix ===")
    appendix(args.out)
    supplement(args.out)
    tables(args.out)


if __name__ == "__main__":
    main()
