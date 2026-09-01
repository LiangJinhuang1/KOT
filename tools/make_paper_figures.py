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
    MODALITY_MARKERS, lineage, method_label, state_color_map, state_label,
)
from src.visualization.style import (
    apply_style, chance_line, embedding_axes, figsize, ink, median_tick,
    panel_letter, save_figure, set_verify,
)
from src.visualization.alignment import modality_mixing
from src.visualization.benchmark import collect_benchmark, find_collapsed
from src.visualization.kinetics import collect_beta, plot_beta_recovery
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
    order = ["kot", "kot_nodyn", "moscot", "scot", "glue", "uniport", "linear_ode"]
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

    fig, axes = plt.subplots(1, 4, figsize=figsize("full", 1.85), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.09, wspace=0.07)

    # a: the generative system, coloured by state
    ax = axes[0]
    xy = PCA(n_components=2, random_state=seed).fit_transform(
        np.asarray(a.layers["protein_mean"])[:n])
    states = obs["state"].to_numpy()
    for st, c in state_color_map(states).items():
        m = states == st
        ax.scatter(xy[m, 0], xy[m, 1], color=c, label=state_label(st), **POINT_STYLE)
    ax.set_title("Ground-truth system")
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
        ax.text(0.5, -0.04, f"mixing {mix:.2f}", transform=ax.transAxes,
                fontsize=6.5, color="0.3", ha="center", va="top")
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
        for gi, (glabel, key) in enumerate(groups):
            for ai, (model, alabel, color) in enumerate(arms):
                vals = pairs[f"{model}_{key}"].to_numpy()
                x = gi + (ai - 0.5) * 0.34
                ax.scatter(np.full(len(vals), x) + rng.uniform(-.045, .045, len(vals)),
                           vals, s=5, color=color, alpha=.65, linewidths=0, zorder=5)
                median_tick(ax, float(np.median(vals)), x, half=0.12,
                            color=ink(color), lw=1.3, orient="horizontal", zorder=6)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([g for g, _ in groups])
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

    print(f"[fig2] stage={stage}  mixing={mix if mix is None else round(mix, 3)}")
    save_figure(fig, out / "fig2_synthetic")


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

    print("\n=== Figure 2: synthetic recovery ===")
    figure2(args.out)

    print("\n=== Figure 3: real CITE-seq ===")
    figure3(args.out)

    print("\n=== Figure 4: kinetic recovery ===")
    for note in plot_beta_recovery(collect_beta(), args.out / "fig4_kinetics"):
        print("   ", note)

    print("\n=== Appendix ===")
    appendix(args.out)


if __name__ == "__main__":
    main()
