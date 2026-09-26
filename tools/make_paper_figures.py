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
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator, MaxNLocator, NullLocator
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA

from src.visualization import (
    FOSCTTM_CHANCE, LINEAGE_COLORS, LINEAGE_ORDER, METHOD_COLORS, MODALITY_COLORS,
    DATASET_COLORS, DATASET_SHORT, MODALITY_MARKERS, dataset_label, dataset_style, lineage,
    method_label, state_color_map,
    state_label,
)
from src.visualization.style import (
    PANEL_LETTER_SIZE, STYLE_STATE, apply_style, chance_line, embedding_axes, figsize,
    ink, median_tick, panel_letter, save_figure, set_verify,
)
from src.visualization.ablation import ladder_panel, read_arms, real_panel
from src.evaluation.knn_alignment import alignment_curve
from src.evaluation.protocol import resolve_oos_mode
from src.training.registry import MODELS as REGISTERED_MODELS
from src.visualization.alignment import confusion_panel
from src.visualization.benchmark import collect_benchmark, find_collapsed
from src.visualization.kinetics import collect_beta, plot_beta_recovery
from src.visualization.prediction import (
    ARM_LABELS, coverage_handles,
    CONTROL_ARMS, canonical_rows, control_rows, coverage_strip_panel,
    paired_delta_panel, paired_protein_delta, preset_proteins,
    ranked_protein_panel, marker_arm_panel,
)
from src.visualization.diagnostics import (
    TUNE_DATASETS, anchor_alignment, checkpoint_panel, collect_checkpoints, collect_flags,
    fixed_panel_beta, flag_panel, gradient_panel, heatmap_grid, read_by_config,
    read_results, share_point_panel, tune_lambda_grid,
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
from src.visualization.lineage_map import lineages_for_run

POINT_STYLE = dict(s=0.9, alpha=0.45, linewidths=0, rasterized=True)
# Fig. 8 packs ~18k points into a 0.9 x 0.8 in panel, five columns across. At the
# shared size the cloud reads as a solid blob; smaller and fainter lets the shape and
# the overlap between the modalities come through.
GRID_POINT_STYLE = dict(s=0.45, alpha=0.35, linewidths=0, rasterized=True)


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
    """Seed directories for several arms from one run and one seed.

    Mixing runs would compare arms trained under different conditions; return None instead.
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


def joint_embedding(xr: np.ndarray, xp: np.ndarray, idx: np.ndarray, seed: int = 0,
                    reduction: str = "pca", umap_neighbors: int = 15,
                    umap_min_dist: float = 0.1):
    """Two-dimensional view of the shared protein space.

    `pca` fits one linear basis on both modalities stacked. Plane distances are then a
    projection of the 134-dim distances FOSCTTM ranks, so the panel and the metric
    describe the same geometry.

    `umap` fits on PROTEIN ONLY and transforms phi(RNA) through that fitted model. That
    mirrors training, where protein is the fixed target and phi(RNA) is placed into it:
    the RNA points have no say in where the landmarks land, so the layout cannot
    manufacture overlap. Fitting on the stack instead would let UMAP pull mutual
    neighbours together, which is the very thing the panel is meant to test. UMAP
    distances are not metric, so the result shows neighbourhood agreement rather than
    the quantity FOSCTTM reports.
    """
    rna, protein = xr[idx], xp[idx]
    if reduction == "pca":
        xy = PCA(n_components=2, random_state=seed).fit_transform(np.vstack([rna, protein]))
        return xy[:len(idx)], xy[len(idx):]
    if reduction == "umap":
        import umap  # optional dependency; only the UMAP variant of Fig. 3 needs it
        reducer = umap.UMAP(n_components=2, n_neighbors=umap_neighbors,
                            min_dist=umap_min_dist, random_state=seed).fit(protein)
        return reducer.transform(rna), reducer.embedding_
    raise ValueError(f"unknown reduction {reduction!r}; use 'pca' or 'umap'")


def co_embedding_scatter(ax, xy_r: np.ndarray, xy_p: np.ndarray, rng, style=None):
    """Both modalities in one panel, plotted in random order.

    Drawing one modality last would read as separation the embedding does not have.
    """
    pts = np.vstack([xy_r, xy_p])
    cols = np.array([MODALITY_COLORS["RNA"]] * len(xy_r) +
                    [MODALITY_COLORS["Protein"]] * len(xy_p))
    order = rng.permutation(len(pts))
    ax.scatter(pts[order, 0], pts[order, 1], color=cols[order],
               **(style or POINT_STYLE))


def framed_embedding(ax):
    """Ticks and spines off without a box.

    Limits stay on matplotlib autoscale so PCA aspect is not rewritten. A
    percentile window plus a square crop had stretched MaxFuse into a disk.

    Locking the aspect instead (``set_aspect("equal", adjustable="datalim")``) keeps the
    boxes identical and the clouds undistorted, but then a cloud whose own aspect is far
    from the box's shrinks into a corner with the rest of the panel empty. Autoscale is
    the deliberate choice: shapes are not comparable across columns anyway, because each
    column is an independent PCA of a different latent space.
    """
    ax.xaxis.set_major_locator(NullLocator())
    ax.yaxis.set_major_locator(NullLocator())
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    for spine in ax.spines.values():
        spine.set_visible(False)


def co_embedding_lineage_scatter(ax, xy_r, xy_p, labels, rng, colors, style=None):
    """Both modalities, lineage colours, shuffled so no lineage sits on top."""
    labels = np.asarray(labels)
    pts = np.vstack([xy_r, xy_p])
    labs = np.concatenate([labels, labels])
    fallback = colors.get("Other", "#BBBBBB")
    cols = np.array([colors.get(str(name), fallback) for name in labs])
    order = rng.permutation(len(pts))
    ax.scatter(pts[order, 0], pts[order, 1], color=cols[order],
               **(style or POINT_STYLE))
    present = {str(name) for name in np.unique(labels)}
    return [name for name in LINEAGE_ORDER if name in present]


def co_embedding_state_scatter(ax, xy_r: np.ndarray, xy_p: np.ndarray,
                               states_r, states_p, rng):
    """Co-embedding with the same state colours on both modalities.

    Modality colours hide branch-identity errors: a swapped map still looks like
    mixed RNA/protein clouds. Identity is the state colour. Draw order is shuffled
    so neither modality sits on top.
    """
    states_r = np.asarray(states_r)
    states_p = np.asarray(states_p)
    colors = state_color_map(np.concatenate([states_r, states_p]))
    cols = np.array([colors[str(s)] for s in states_r] +
                    [colors[str(s)] for s in states_p])
    pts = np.vstack([xy_r, xy_p])
    order = rng.permutation(len(pts))
    ax.scatter(pts[order, 0], pts[order, 1], color=cols[order], **POINT_STYLE)


def bare_axes(ax):
    """No ticks, no spines, no arrows — for grid panels where one shared arrow pair
    on the corner panel names the axes for the whole figure."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def colour_key(target, names, colors, *, loc: str = "best", **kw):
    """Legend of flat colour swatches: no marker edge, so the key matches the points.

    `target` is an Axes or a Figure, so the same key can sit inside a panel or outside
    the whole grid.
    """
    opts = dict(loc=loc, frameon=False, handlelength=1.1, handleheight=1.1,
                handletextpad=0.5)
    opts.update(kw)
    return target.legend(handles=[Patch(facecolor=colors[name], edgecolor="none",
                                        label=name) for name in names],
                         **opts)


def modality_legend(target, loc: str = "best", **kw):
    """The RNA / Protein key, once per figure rather than once per panel."""
    return colour_key(target, ("RNA", "Protein"), MODALITY_COLORS, loc=loc, **kw)


# Baselines were re-run on 2026-09-08 under a declared out-of-sample protocol
# (`protocol_oos_mode: holdout_second`, `protocol_n_fit_cells` in summary.txt). The older
# `*_scvelo_*` baseline dirs carry no protocol block at all, so panel a takes these models
# from the re-run only rather than pooling two protocols in one row. KOT and MaxFuse have
# no re-run and keep their own dirs, so they stay on `protocol_oos_mode: native`.
TRAINSPLIT_PREFIX = "cite_trainsplit_"
TRAINSPLIT_BASELINES = ("glue", "uniport", "moscot", "scot", "linear_ode")

# The run behind every ranked row of Table 1 (`csv/01_alignment_real.csv`, column `run`).
# The figures read these so a number on a panel and the same number in the table come from
# the same training run -- they did not before: KOT was drawn from
# `run_20260828_101018_full_*_S2cfgB_*`, a different cfgB sweep that scores 0.119 where the
# table reports 0.129, and Fig. 6's JVP median was 0.967 against the table's 0.952.
# Superseded and NOT used: the `*_scvelo_kot` / `*_scvelo_ablation` dirs, whose summary.txt
# is eight lines with no hyperparameters at all.
CANONICAL_RUNS = {
    ("kot", "bmmc_cite_retained"):
        "run_20260829_022744_shuf_bmmc_scv_cfgB_warm300_beta1p0e-3_realVel",
    ("kot", "pbmc_retained"):
        "run_20260829_022744_shuf_pbmc_scv_cfgB_warm300_beta1p0e-3_realVel",
    ("kot_nodyn", "bmmc_cite_retained"):
        "run_20260829_042713_nodyn_bmmc_scv_cfgB_warm300_beta1p0e-3",
    ("kot_nodyn", "pbmc_retained"):
        "run_20260829_042713_nodyn_pbmc_scv_cfgB_warm300_beta1p0e-3",
    ("maxfuse", "bmmc_cite_retained"): "run_20260901_220923_maxfuse_bmmc_seed101",
    ("maxfuse", "pbmc_retained"):      "run_20260901_213640_maxfuse_pbmc",
    ("glue", "bmmc_cite_retained"):    "cite_trainsplit_glue_bmmc",
    ("glue", "pbmc_retained"):         "cite_trainsplit_glue_pbmc",
    ("uniport", "bmmc_cite_retained"): "cite_trainsplit_uniport_bmmc",
    ("uniport", "pbmc_retained"):      "cite_trainsplit_uniport_pbmc",
    ("moscot", "bmmc_cite_retained"):  "cite_trainsplit_moscot_bmmc",
    ("moscot", "pbmc_retained"):       "cite_trainsplit_moscot_pbmc",
    ("scot", "bmmc_cite_retained"):    "cite_trainsplit_scot_bmmc",
    ("scot", "pbmc_retained"):         "cite_trainsplit_scot_pbmc",
    ("linear_ode", "bmmc_cite_retained"): "cite_trainsplit_linear_ode_bmmc",
    ("linear_ode", "pbmc_retained"):      "cite_trainsplit_linear_ode_pbmc",
}

# MaxFuse on BMMC keeps its 12 seeds in 12 SEPARATE run dirs, so the table's `run` column
# names only one of them. Pinning the benchmark to that one would report n=1 where the
# table reports 12; the pin is therefore used only to choose which dir Fig. 8 and Fig. 9
# draw, and the benchmark keeps every curated sibling.
PANEL_ONLY_PINS = {"maxfuse"}


def canonical_seed_dir(model: str, dataset: str) -> Path | None:
    """Median-FOSCTTM seed inside the run CANONICAL_RUNS names, or None if not named.

    Median rather than best, matching `pick_run`: an illustrative panel showing a
    method's luckiest seed is not a comparison.
    """
    run = CANONICAL_RUNS.get((model, dataset))
    if run is None:
        return None
    hits = []
    for path in (CACHE_DIR / run / model / dataset).glob("seed_*/aligned_rna.npy"):
        diagnostics = read_diagnostics(path.parent / "diagnostics.json")
        if "mean_foscttm" not in diagnostics or is_collapsed(diagnostics):
            continue
        hits.append((float(diagnostics["mean_foscttm"]), path.parent))
    if not hits:
        raise FileNotFoundError(f"{run}: no usable {model}/{dataset} seed with aligned arrays")
    return sorted(hits)[len(hits) // 2][1]


def benchmark_panel_rows(bench: pd.DataFrame) -> pd.DataFrame:
    """One point per (dataset, model, seed), with baselines taken from the re-run.

    KOT is the reason the de-duplication exists: seeds 42, 123 and 2026 each appear in
    both `*_scvelo_kot` and `*_scvelo_ablation`, so plotting every row gave KOT eight
    points from five seeds while every other method had one point per seed. Ties go to
    the run dir covering more seeds, which keeps the fuller sweep rather than a subset.
    """
    key = pd.Series(list(zip(bench.model, bench.dataset)), index=bench.index)
    pinned = key.isin(CANONICAL_RUNS) & ~bench.model.isin(PANEL_ONLY_PINS)
    on_canonical = pinned & (bench["run"] == key.map(CANONICAL_RUNS.get))
    baseline = bench.model.isin(TRAINSPLIT_BASELINES)
    rows = pd.concat([
        bench[~baseline & ~pinned],
        bench[baseline & bench["run"].str.startswith(TRAINSPLIT_PREFIX)],
        bench[on_canonical],
    ], ignore_index=True)
    coverage = (rows.groupby(["dataset", "model", "run"])["seed"].nunique()
                    .rename("n_seeds").reset_index())
    rows = rows.merge(coverage, on=["dataset", "model", "run"])
    return (rows.sort_values(["dataset", "model", "seed", "n_seeds", "run"],
                             ascending=[True, True, True, False, True])
                .drop_duplicates(subset=["dataset", "model", "seed"], keep="first")
                .drop(columns="n_seeds"))


def pooled_lineage_foscttm(run: Path, *, max_per_lineage: int = 60000, seed: int = 0
                           ) -> tuple[dict[str, np.ndarray], int]:
    """Per-lineage FOSCTTM pooled over every seed of the sweep `run` belongs to.

    Panel a reports twelve seeds, so drawing panel b from one of them understated the
    spread and let a single seed set the lineage ordering: Dendritic is the smallest
    lineage (3,477 cells) and its median swings 0.027-0.207 across seeds, so one seed put
    it first while the pool puts it fourth. Cells repeat across seeds by construction --
    the ridgeline is the distribution over cells AND seeds, which is the point.

    KDE cost is linear in the pooled count, so each lineage is capped at
    `max_per_lineage` by a seeded subsample; the cap only bites on the largest lineages.
    """
    rng = np.random.default_rng(seed)
    pooled: dict[str, list[np.ndarray]] = {}
    seeds = sorted(d for d in run.parent.glob("seed_*") if (d / "foscttm.csv").exists())
    for seed_dir in seeds:
        values = pd.read_csv(seed_dir / "foscttm.csv")["foscttm"].to_numpy()
        lineages = lineages_for_run(seed_dir, len(values))
        if lineages is None:
            continue
        for name in LINEAGE_ORDER:
            mask = (lineages == name).to_numpy()
            if mask.sum() >= 30:
                pooled.setdefault(name, []).append(values[mask])
    out = {}
    for name, chunks in pooled.items():
        joined = np.concatenate(chunks)
        if len(joined) > max_per_lineage:
            joined = joined[rng.choice(len(joined), max_per_lineage, replace=False)]
        out[name] = joined
    return out, len(seeds)


def figure3(out: Path, max_cells: int = 12000, seed: int = 0,
            reduction: str = "pca", stem: str = "fig3_realdata"):
    """Real CITE-seq: benchmark, co-embedding, and per-lineage alignment quality.

    `reduction` only affects panels c and d. Panels a and b are identical either way,
    so a PCA and a UMAP render differ in exactly the thing being compared.
    """
    apply_style()
    rng = np.random.default_rng(seed)

    bench = collect_benchmark()
    bench = bench[~find_collapsed(bench)]
    bench = bench[~bench.model.isin(["kot_anchor", "totalvi"])]
    # Curated runs only: do not pool uncurated sweep arms with curated.
    curated = curated_runs()
    if curated:
        # The re-run baselines postdate the MANIFEST, so they are kept explicitly rather
        # than being dropped as uncurated.
        keep = (bench["run"].isin(curated)
                | bench["run"].str.startswith(TRAINSPLIT_PREFIX)
                | bench["run"].isin(set(CANONICAL_RUNS.values())))
        dropped = sorted(set(bench.loc[~keep, "run"]))
        bench = bench[keep]
        print(f"[fig3] benchmark from {bench['run'].nunique()} runs "
              f"({len(dropped)} uncurated dropped)")
    bench = benchmark_panel_rows(bench)
    for ds, g in bench.groupby("dataset"):
        counts = g.groupby("model")["seed"].nunique().to_dict()
        print(f"[fig3] {ds}: " + ", ".join(f"{m} n={counts[m]}" for m in sorted(counts)))

    run = canonical_seed_dir("kot", "bmmc_cite_retained")
    if run is None:
        run = pick_run("bmmc_cite_retained", prefer=("bmmc_scvelo_ablation",))
    if run is None:
        print("[fig3] no usable BMMC run with aligned arrays; skipping")
        return
    print(f"[fig3] co-embedding panels from {run}")

    xr, xp = load_aligned(run)
    fos = pd.read_csv(run / "foscttm.csv")["foscttm"].to_numpy()
    lin = lineages_for_run(run, len(xr))

    idx = rng.choice(len(xr), min(max_cells, len(xr)), replace=False)
    xy_r, xy_p = joint_embedding(xr, xp, idx, seed=seed, reduction=reduction)
    axis_names = ("PC 1", "PC 2") if reduction == "pca" else ("UMAP 1", "UMAP 2")

    fig = plt.figure(figsize=figsize("full", 4.6), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.09, w_pad=0.07, hspace=0.07, wspace=0.06)
    # Four equal panels. Unequal ratios drew c and d -- the SAME embedding in two
    # colourings -- at different scales, which stops a reader overlaying them, and gave
    # a and b different heights for no reason the data supports.
    gs = fig.add_gridspec(2, 2)

    ax = fig.add_subplot(gs[0, 0])
    order = ["kot", "kot_nodyn", "maxfuse", "moscot", "scot", "glue", "uniport",
             "linear_ode"]
    datasets = ["bmmc_cite_retained", "pbmc_retained"]
    offsets = {datasets[0]: -0.17, datasets[1]: +0.17}
    marks = {ds: dataset_style(ds)[0] for ds in datasets}
    fills = {datasets[0]: "none", datasets[1]: DATASET_COLORS[datasets[1]]}
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
                       facecolors=fills[ds], edgecolors=DATASET_COLORS[ds],
                       linewidths=.6, alpha=.85 if focal else .55, zorder=6)
            if len(vals) >= 3:
                median_tick(ax, float(np.median(vals)), y, half=0.12, lw=1.1, zorder=7)
    chance_line(ax, FOSCTTM_CHANCE, axis="x", label="chance")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([method_label(m) for m in order])
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlim(0, 0.76)
    ax.invert_yaxis()
    ax.set_xlabel("FOSCTTM")
    ax.set_title("Alignment error")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
               markeredgecolor=DATASET_COLORS[datasets[0]], markersize=3, label="BMMC"),
        Line2D([0], [0], marker="s", color="none",
               markerfacecolor=DATASET_COLORS[datasets[1]],
               markeredgecolor=DATASET_COLORS[datasets[1]], markersize=3, label="PBMC")],
        # One entry per line, lower right: stacked the key is narrow enough to clear the
        # Linear ODE BMMC point at 0.442, which a two-column version sat on top of.
        loc="lower right", ncol=1, frameon=False)
    ax.text(0.0, -0.20, "lower = better", transform=ax.transAxes,
            fontsize=6, color="0.35", ha="left", va="top")
    panel_letter(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    groups = []
    n_lineage_seeds = 0
    if lin is not None:
        pooled, n_lineage_seeds = pooled_lineage_foscttm(run, seed=seed)
        groups = [(g, pooled[g]) for g in LINEAGE_ORDER if g in pooled]
        print(f"[fig3] panel b pools {n_lineage_seeds} seeds")
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
                    ha="right", va="center", fontsize=6, color=ink(c))
        chance_line(ax, FOSCTTM_CHANCE, axis="x", label="chance")
        ax.set_yticks([]); ax.spines["left"].set_visible(False)
        ax.set_xlim(0, 0.75); ax.set_ylim(-0.08, len(groups))
        ax.set_xlabel("FOSCTTM")
        ax.set_title(f"By lineage ({n_lineage_seeds} seeds)"
                     if n_lineage_seeds else "By lineage")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
    panel_letter(ax, "b")

    ax = fig.add_subplot(gs[1, 0])
    co_embedding_scatter(ax, xy_r, xy_p, rng)
    ax.set_title("Co-embedding, by modality")
    # Pinned, not "best": the corner each key takes is chosen so the two panels do not
    # put their keys in the same place, and so neither lands on the denser side of its
    # own cloud.
    modality_legend(ax, loc="upper right")
    embedding_axes(ax, *axis_names)
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
    drawn = [n for n in LINEAGE_ORDER if lin is not None and (lin.to_numpy()[idx] == n).any()]
    if drawn:
        colour_key(ax, drawn, LINEAGE_COLORS, loc="upper left",
                   fontsize=STYLE_STATE["ladder"][2], labelspacing=0.3)
    embedding_axes(ax, *axis_names)
    panel_letter(ax, "d")

    save_figure(fig, out / stem)


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


FIGURE2_CONFUSION_ARMS = (
    ("kot", "Assignments:\nKOT"),
    ("kot_nodyn", "Assignments:\nNo dynamics"),
)


def confusion_from_run(run: Path) -> tuple[list, list] | None:
    """Branch confusion stored with one seed directory, or None if it was not written."""
    diagnostics = run / "diagnostics.json"
    if not diagnostics.exists():
        return None
    payload = read_diagnostics(diagnostics)
    matrix, labels = payload.get("branch_confusion_matrix"), payload.get("branch_confusion_labels")
    if not isinstance(matrix, list) or not isinstance(labels, list):
        return None
    n_labels = len(labels)
    if n_labels < 2 or len(matrix) != n_labels:
        return None
    if any(not isinstance(row, list) or len(row) != n_labels for row in matrix):
        return None
    return matrix, labels


# Fig. 1b and Fig. 2 must draw the SAME synthetic run, so the name lives in one place.
# `syn_rerun_branch_kot` (2026-09-08) replaces `syn_branch_ablation` (2026-07-01): the older
# run wrote NO branch_confusion_matrix, branch_accuracy or jvp_rhs_cos_median at all, so
# every kinetics claim on those panels had to come from somewhere else. On the rerun KOT
# scores branch accuracy 0.997 against the old run's 0.979, and the no-kinetics arm reports
# a JVP-RHS cosine of -0.791, which the old run simply could not.
SYNTHETIC_BRANCH_RUNS = ("syn_rerun_branch_kot", "syn_branch_kot_restored",
                         "syn_branch_ablation")


def figure2_nodyn_sibling(run: Path) -> Path | None:
    """Same run/seed no-kinetics directory when it is usable as a paired control."""
    parts = run.relative_to(CACHE_DIR).parts
    nodyn = CACHE_DIR.joinpath(parts[0], "kot_nodyn", *parts[2:])
    if not (nodyn / "diagnostics.json").exists():
        return None
    if is_collapsed(read_diagnostics(nodyn / "diagnostics.json")):
        return None
    if confusion_from_run(nodyn) is None:
        return None
    return nodyn


def figure2_confusion_sources(matched: dict[str, Path] | None, fallback=None
                              ) -> list[tuple[str, str, list | None, list | None]]:
    """Both kinetic arms, even when one matrix is missing.

    Plotting only the failing arm makes recovery invisible. A missing matrix stays
    an empty panel. Do not fill a hole from another run or seed: that would look
    like a paired control.
    """
    rows = []
    for model, title in FIGURE2_CONFUSION_ARMS:
        found = confusion_from_run(matched[model]) if matched and model in matched else None
        if found is None and fallback is not None:
            found = fallback(model)
        matrix, labels = found if found is not None else (None, None)
        rows.append((model, title, matrix, labels))
    return rows


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

    This is the plotted half of Figure 1; panel a is `tools/make_schematic.py`.
    """
    apply_style()
    rng = np.random.default_rng(seed)

    arms = [("kot", "KOT"), ("kot_nodyn", "No dynamics")]
    matched = pick_matched_runs("synthetic_linked_ode", [m for m, _ in arms],
                                prefer=SYNTHETIC_BRANCH_RUNS)
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

        fos = float(np.mean(pd.read_csv(run / "foscttm.csv")["foscttm"]))
        ax.set_title(label)
        ax.text(0.5, -0.04, f"FOSCTTM {fos:.3f}",
                transform=ax.transAxes, fontsize=6, color="0.3",
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
    config_path = run / "run_config.yaml"
    if config_path.exists():
        cfg = yaml.safe_load(config_path.read_text()) or {}
        paths = cfg.get("dataset_paths") or {}
        pp = paths.get("protein_path") or cfg.get("protein_path")
        if pp:
            return Path(pp)
    posix = run.as_posix()
    if "/synthetic_linked_ode/" not in posix:
        return None
    # syn_branch_ablation seeds predate run_config.yaml; the run name still names the stage.
    for stage_name in ("branch", "clean", "oracle"):
        if f"syn_{stage_name}_" in posix or f"syn_{stage_name}/" in posix:
            fallback = Path(f"cache/synthetic_linked_ode/{stage_name}/protein.h5ad")
            if fallback.exists():
                return fallback
    return None


def figure2(out: Path, stage: str = "branch", max_cells: int = 8000, seed: int = 0):
    """Controlled recovery on the synthetic linked-ODE system.

    Defaults to the `branch` stage: it carries the branching structure the
    method is actually meant to handle, and it is the harder case. The six
    panels are ground truth, state-coloured co-embeddings for KOT and the
    no-kinetics arm, branch resolution across seeds, and matched confusion
    matrices for both arms.
    """
    apply_style("iclr")
    rng = np.random.default_rng(seed)

    req = ("aligned_rna.npy", "aligned_protein.npy", "diagnostics.json")
    cands = seed_dirs("synthetic_linked_ode", "kot", req)
    if not cands:
        print("[fig2] no synthetic run with aligned arrays; skipping")
        return

    resolved = [(f, d, protein_path_of(d)) for f, d in cands]
    resolved = [(f, d, pp) for f, d, pp in resolved if pp is not None and pp.exists()]
    if not resolved:
        print("[fig2] no run whose config names an existing protein h5ad; skipping")
        return
    # Keep to the requested stage; a mixed-cache median must not pick another sweep.
    on_stage = [r for r in resolved if f"/{stage}/" in str(r[2])] or resolved
    # One ordered preference list, shared with Fig. 1b, so the two cannot drift onto
    # different synthetic runs. Name templates used to do this and could not express a
    # run whose name does not embed the stage, which is why the rerun was never reached.
    grouped = [[r for r in on_stage if r[1].relative_to(CACHE_DIR).parts[0] == name]
               for name in SYNTHETIC_BRANCH_RUNS]
    search = next((group for group in grouped if group), on_stage)
    preferred = sorted(search)
    paired = [row for row in search
              if confusion_from_run(row[1]) is not None
              and figure2_nodyn_sibling(row[1]) is not None]
    chosen = sorted(paired or preferred)
    _, run, prot = chosen[len(chosen) // 2]
    matched = {"kot": run}
    sibling = figure2_nodyn_sibling(run)
    if sibling is not None:
        matched["kot_nodyn"] = sibling
    print(f"[fig2] panels from {run}")
    print(f"[fig2] ground truth from {prot}")
    if paired:
        print(f"[fig2] chose among {len(paired)} kot/nodyn pairs")
    else:
        print("[fig2] no paired no-kinetics sibling with a confusion matrix")
    if "kot_nodyn" in matched:
        print(f"[fig2] paired no-kinetics run {matched['kot_nodyn']}")
    a = ad.read_h5ad(prot)

    xr, xp = load_aligned(run)
    n = len(xr)
    if len(xp) != n or len(a) != n:
        raise ValueError(
            f"{run}: aligned RNA ({n}), protein ({len(xp)}), and protein h5ad "
            f"({len(a)}) must share one cell order; refusing to colour by state"
        )
    obs = a.obs.reset_index(drop=True)
    if "state" not in obs.columns:
        raise ValueError(f"{prot}: missing obs['state'] needed for identity colours")
    states = obs["state"].to_numpy()

    fig, axes = plt.subplots(2, 3, figsize=figsize("full", 3.7), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.10, h_pad=0.16, wspace=0.12, hspace=0.16)

    ax = axes[0, 0]
    xy = PCA(n_components=2, random_state=seed).fit_transform(
        np.asarray(a.layers["protein_mean"]))
    for st, c in state_color_map(states).items():
        m = states == st
        ax.scatter(xy[m, 0], xy[m, 1], color=c, label=state_label(st), **POINT_STYLE)
    ax.set_title("Ground truth")
    # Pinned, not "best": the corner arrows own the lower left, and "best" drops the
    # legend straight onto the "PC 1" label.
    ax.legend(markerscale=6, loc="upper right")
    embedding_axes(ax, "PC 1", "PC 2")
    panel_letter(ax, "a")

    ax = axes[0, 1]
    idx = rng.choice(n, min(max_cells, n), replace=False)
    xy_r, xy_p = joint_embedding(xr, xp, idx, seed=seed)
    # Synthetic evaluation arrays keep cell order, so RNA i and protein i share a state label.
    co_embedding_state_scatter(ax, xy_r, xy_p, states[idx], states[idx], rng)
    ax.set_title("KOT")
    ax.text(0.98, 0.02, "colours as in a", transform=ax.transAxes,
            fontsize=6, color="0.45", ha="right", va="bottom")
    embedding_axes(ax, "PC 1", "PC 2")
    panel_letter(ax, "b")

    ax = axes[0, 2]
    ax.set_title("No dynamics")
    panel_letter(ax, "c")
    if "kot_nodyn" not in matched:
        ax.text(0.5, 0.5, "no paired no-kinetics run",
                transform=ax.transAxes, fontsize=6, color="0.45",
                ha="center", va="center")
        ax.set_xticks([]); ax.set_yticks([])
    else:
        xr_n, xp_n = load_aligned(matched["kot_nodyn"])
        if len(xr_n) != n or len(xp_n) != n:
            raise ValueError(
                f"{matched['kot_nodyn']}: aligned shapes {(len(xr_n), len(xp_n))} "
                f"do not match the kinetics run ({n})"
            )
        xy_n_r, xy_n_p = joint_embedding(xr_n, xp_n, idx, seed=seed)
        co_embedding_state_scatter(ax, xy_n_r, xy_n_p, states[idx], states[idx], rng)
        ax.text(0.98, 0.02, "colours as in a", transform=ax.transAxes,
                fontsize=6, color="0.45", ha="right", va="bottom")
        embedding_axes(ax, "PC 1", "PC 2")

    # Branch stage is mirror-symmetric on purpose: OT alone cannot tell the branches apart.
    ax = axes[1, 0]
    pairs = branch_identity_pairs()
    if pairs is not None and len(pairs):
        groups = [("Branches apart", "sep"), ("Correct branch", "raw")]
        arms = [("kot", "KOT", METHOD_COLORS["kot"]),
                ("kot_nodyn", "No dynamics", "#8C8C8C")]
        for gi, (_, key) in enumerate(groups):
            for ai, (model, _, color) in enumerate(arms):
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
        ax.set_title(f"Branch resolution ({len(pairs)} seeds)")
        ax.legend(handles=[
            Line2D([0], [0], marker="o", color="none", markersize=3,
                   markerfacecolor=c, label=lbl) for _, lbl, c in arms],
            loc="lower left")
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
    else:
        ax.text(0.5, 0.5, "no matched kot / kot_nodyn\nbranch pairs",
                transform=ax.transAxes, fontsize=6, color="0.45",
                ha="center", va="center")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("Branch resolution")
    # Matches the offset e/f need for their two-line titles, so the row's letters align.
    panel_letter(ax, "d", dy_points=26)

    for column, (model, title, matrix, labels) in enumerate(
            figure2_confusion_sources(matched), start=1):
        ax = axes[1, column]
        if matrix is None or labels is None:
            ax.text(0.5, 0.5, "no curated branch run\nwith a confusion matrix",
                    transform=ax.transAxes, color="0.45", ha="center", va="center")
            ax.set_xticks([]); ax.set_yticks([])
        else:
            confusion_panel(ax, matrix, labels)
        ax.set_title(title)
        panel_letter(ax, "ef"[column - 1], dy_points=26)
        paired = bool(matched and model in matched and confusion_from_run(matched[model]))
        if matrix is None:
            source = "missing"
        elif paired:
            source = "paired run"
        else:
            source = "unpaired source"
        print(f"[fig2] confusion {model}: {source}")

    print(f"[fig2] stage={stage}")
    save_figure(fig, out / "fig2_synthetic")


# The configuration both ablations were run under; pooling another schedule would mix training into the corruption effect.
ABLATION_LR_BETA = 0.001
ABLATION_WARMUP = 300


def figure5(out: Path):
    """Velocity corruption: a dose-response on synthetic, and what it does on real data."""
    apply_style()
    fig, axd = plt.subplot_mosaic(
        [["a", "c"], ["b", "c"]],
        figsize=figsize("full", 3.15),
        layout="constrained",
        width_ratios=[1.45, 1],
        height_ratios=[1, 1],
    )
    fig.get_layout_engine().set(w_pad=0.04, h_pad=0.06, wspace=0.10, hspace=0.08)

    syn = read_arms("synthetic_branch_summary.csv", ABLATION_LR_BETA, ABLATION_WARMUP)
    # No dose-response to show; belongs in the ablation table, not this panel.
    # Ordered by how well each arm does on the synthetic ladder.
    arms = ["kot_oracle", "kot_fixedkappa", "kot_fixedalpha"]

    ax = axd["a"]
    ladder_panel(ax, syn, "mean_foscttm", arms, ylabel="FOSCTTM",
                 chance=FOSCTTM_CHANCE)
    # Stacked panels: an edge tick on one meets the neighbour's edge tick (section 3.4).
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
    ax.set_title("Synthetic: alignment")
    # Between the two series: upper-left sat on the κ-fixed curve, and the
    # off-ladder bay collides with the reverse points.
    ax.legend(loc="center left", bbox_to_anchor=(0.16, 0.42), frameon=False,
              borderaxespad=0)
    panel_letter(ax, "a")

    ax = axd["b"]
    ladder_panel(ax, syn, "jvp_rhs_cos_median", arms, ylabel="JVP\u00b7RHS cosine")
    ax.set_title("Synthetic: ODE agreement")
    ax.set_ylim(-0.95, 1.1)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
    panel_letter(ax, "b")

    ax = axd["c"]
    real = read_arms("kinetics_ablation_real_summary.csv", ABLATION_LR_BETA, ABLATION_WARMUP)
    real_panel(ax, real, "mean_foscttm", ["bmmc_cite_retained", "pbmc_retained"],
               xlabel="FOSCTTM", chance=FOSCTTM_CHANCE)
    ax.set_title("Real CITE-seq")
    ax.legend(handles=[
        Line2D([0], [0], marker=dataset_style(d)[0], color=dataset_style(d)[1], lw=0,
               ms=3.4, fillstyle=f, markeredgewidth=0.8, label=dataset_style(d)[2])
        for d, f in (("bmmc_cite_retained", "none"), ("pbmc_retained", "full"))],
        loc="lower right", bbox_to_anchor=(1.0, 1.04), ncol=1, frameon=False,
        borderaxespad=0)
    panel_letter(ax, "c")

    fig.align_ylabels([axd["a"], axd["b"]])
    axd["a"].tick_params(labelbottom=False, bottom=False)
    axd["a"].spines["bottom"].set_visible(False)
    fig.canvas.draw()
    fig.set_layout_engine(None)
    pa, pb = axd["a"].get_position(), axd["b"].get_position()
    height = min(pa.height, pb.height)
    axd["a"].set_position([pa.x0, pa.y1 - height, pa.width, height])
    axd["b"].set_position([pa.x0, pb.y0, pa.width, height])
    save_figure(fig, out / "fig5_velocity")


def figure6(out: Path, dataset: str = "bmmc_cite_retained", max_cells: int = 14000,
            seed: int = 0):
    """Where the ODE holds: the per-cell cosine on the embedding, by lineage, and
    against per-cell pairing quality."""
    apply_style()
    rng = np.random.default_rng(seed)

    # Same KOT run as Fig. 3, 8 and 9. On the old curated run this panel showed a JVP
    # cosine median of 0.853 where Table 4 reports 0.952, because that run is the
    # superseded one; all 12 canonical seeds carry per_cell_diagnostics.npz.
    run = canonical_seed_dir("kot", dataset)
    if run is None or not (run / "per_cell_diagnostics.npz").exists():
        run = pick_run(dataset, "kot",
                       require=("aligned_rna.npy", "per_cell_diagnostics.npz"))
    if run is None:
        print(f"[fig6] no {dataset} run carries per_cell_diagnostics.npz")
        return
    print(f"[fig6] per-cell panels from {run}")
    per_cell = load_per_cell(run)
    xr, xp = load_aligned(run)
    n = len(xr)
    idx = rng.choice(n, min(n, max_cells), replace=False)
    xy_r, _ = joint_embedding(xr, xp, idx, seed)
    cos = per_cell["jvp_rhs_cos"]

    # Panel a loses width to its colourbar and panel b to its rotated lineage ticks, so
    # equal columns left c visibly smaller than the other two.
    fig, axes = plt.subplots(1, 3, figsize=figsize("full", 2.5), layout="constrained",
                             gridspec_kw=dict(width_ratios=[1.22, 1.0, 1.0]))
    fig.get_layout_engine().set(w_pad=0.06, wspace=0.08)

    ax = axes[0]
    mappable = paint_panel(ax, xy_r, cos[idx])
    # Lower-left of this cloud is the erythroid tail; the L sits in the margin.
    embedding_axes(ax, "PC 1", "PC 2", pad=-0.14, frac=0.18)
    ax.set_title("ODE agreement per cell")
    bar = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.02)
    bar.set_label("JVP·RHS cosine")
    bar.outline.set_visible(False)
    bar.locator = MaxNLocator(nbins=4, prune="both")
    bar.update_ticks()
    panel_letter(ax, "a")

    ax = axes[1]
    lineages = lineages_for_run(run, n)
    if lineages is None:
        ax.set_visible(False)
    else:
        group_violin_panel(ax, cos, lineages.to_numpy(), LINEAGE_ORDER, LINEAGE_COLORS)
        ax.set_ylabel("JVP·RHS cosine")
        ax.set_title("By lineage")
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
    panel_letter(ax, "b")

    ax = axes[2]
    pair_panel(ax, cos, per_cell["foscttm"])
    ax.set_xlabel("JVP·RHS cosine")
    ax.set_ylabel("FOSCTTM")
    ax.set_title("Fit versus pairing")
    # Right-edge label sat on the hexbin mode; the left of this panel is empty.
    chance_line(ax, FOSCTTM_CHANCE, text_x=0.02, ha="left")
    panel_letter(ax, "c")

    save_figure(fig, out / "fig6_percell")


def figure7(out: Path):
    """Predicted protein against measured protein, with matched kinetic controls."""
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=figsize("full", 4.8),
                             gridspec_kw=dict(width_ratios=[1, 1.35],
                                              height_ratios=[1, 1.05]))

    frames = {short: canonical_rows(d) for short, d in
              (("BMMC", "bmmc_cite_retained"), ("PBMC", "pbmc_retained"))}
    kot = frames["PBMC"]
    dataset = "pbmc_retained"

    ax = axes[0, 0]
    coverage_strip_panel(ax, frames)
    ax.set_ylabel("Spearman ρ, per protein")
    panel_letter(ax, "a")

    ax = axes[0, 1]
    ranked_protein_panel(ax, {"PBMC": kot})
    ax.set_xlabel("Protein, ranked")
    ax.set_ylabel("Spearman ρ")
    ax.set_title(f"{dataset_label(dataset)}, {kot['seed'].nunique()} seeds")
    ax.legend(handles=coverage_handles(), loc="upper right", frameon=False)
    panel_letter(ax, "b")

    ax = axes[1, 0]
    frames_arms = {"none": kot}
    delta_source = ARM_LABELS["nodyn"]
    try:
        from tools.score_heldout_phi import nodyn_rows
        nodyn = nodyn_rows(dataset, kot)
        deltas = paired_protein_delta(kot, nodyn)
        frames_arms["nodyn"] = nodyn
    except FileNotFoundError as exc:
        print(f"[fig7] nodyn scoring unavailable ({exc}); pairing against shuffled velocity")
        delta_source = ARM_LABELS["shuffle"]
        shuffle = control_rows(dataset, "shuffle")
        deltas = paired_protein_delta(kot, shuffle)
        frames_arms["shuffle"] = shuffle
    paired_delta_panel(ax, {"PBMC": deltas})
    ax.set_xlabel("Protein, ranked by Δ")
    ax.set_ylabel("Δ Spearman (KOT − control)")
    ax.set_title(f"KOT minus {delta_source}, {dataset_style(dataset)[2]}")
    ax.legend(handles=coverage_handles(), loc="upper right", frameon=False)
    panel_letter(ax, "c")

    for arm in CONTROL_ARMS:
        arm_rows = control_rows(dataset, arm)
        if arm_rows.empty:
            print(f"[fig7] control {arm} has no canonical rows")
            continue
        frames_arms[arm] = arm_rows
    proteins = preset_proteins(kot["protein"].unique())
    ax = axes[1, 1]
    marker_arm_panel(ax, frames_arms, proteins,
                     [arm for arm in ("none", "nodyn", *CONTROL_ARMS)
                      if arm in frames_arms])
    ax.set_ylabel("Spearman ρ")
    ax.set_title(f"Preselected markers, {dataset_style(dataset)[2]}")
    panel_letter(ax, "d")

    fig.tight_layout()
    save_figure(fig, out / "fig7_prediction")
    return None


# Methods with aligned arrays, in the Fig. 3a row order. Missing methods did not save those arrays.
GRID_MODELS = ["kot", "kot_nodyn", "maxfuse", "glue", "uniport"]

# Column headers must share one baseline. The paper-wide nodyn label carries
# λ_dyn and wraps in a fifth of the text block.
# Fig. 8 compares KOT against other methods, so the no-dynamics arm is named by the
# hyperparameter that defines it, like Figs. 3 and 9. Velocity-ablation figures use
# "No dynamics" instead; the two vocabularies answer two different questions.
FIG8_METHOD_LABELS: dict[str, str] = {}
# Same local-override idea for the lineage key: the column that holds it is narrow
# so the five panels stay wide, and "HSC / progenitor" is what sets its width.
FIG8_LINEAGE_LABELS = {"HSC / progenitor": "HSC / prog.", "NK / ILC": "NK / ILC"}

# Table 1's backing data. Fig. 8 labels its panels from here rather than from each run's
# own summary, so the figure and the table cannot quote different FOSCTTM for the same
# method. The table scores every ranked method on the same fitted-cell split (BMMC
# 80,737, PBMC 4,406), which a per-run summary does not.
TABLE1_CSV = Path("Litterature/ICLR_Tables/csv/01_alignment_real.csv")


def table1_foscttm(dataset: str) -> dict[str, float]:
    """Mean FOSCTTM per model as Table 1 reports it, or {} if the table is absent."""
    if not TABLE1_CSV.exists():
        print(f"[fig8] {TABLE1_CSV} missing; falling back to per-run FOSCTTM")
        return {}
    table = pd.read_csv(TABLE1_CSV)
    rows = table[table.dataset == dataset]
    out = {}
    for model, group in rows.groupby("model"):
        if len(group) > 1:
            raise ValueError(
                f"{TABLE1_CSV}: {model}/{dataset} has {len(group)} rows "
                f"({sorted(group.run_id)}); the panel label would be ambiguous"
            )
        out[str(model)] = float(group["mean_foscttm"].iloc[0])
    return out


def grid_runs(dataset: str) -> dict[str, Path]:
    """Curated run per grid model, with KOT and its control pinned to CANONICAL_RUNS.

    Without the override these panels would show a different KOT run from the one panel a
    reports, because `curated_method_runs` only sees what the MANIFEST lists.
    """
    runs = curated_method_runs(dataset, GRID_MODELS)
    for model in GRID_MODELS:
        pinned = canonical_seed_dir(model, dataset)
        if pinned is not None:
            runs[model] = pinned
    return runs


def figure8(out: Path, dataset: str = "bmmc_cite_retained", max_cells: int = 9000,
            seed: int = 0):
    """Every method's co-embedding, coloured by modality and by lineage."""
    apply_style()
    rng = np.random.default_rng(seed)
    small = STYLE_STATE["ladder"][1]

    runs = grid_runs(dataset)
    table_foscttm = table1_foscttm(dataset)
    models = [m for m in GRID_MODELS if m in runs]
    if not models:
        print(f"[fig8] no curated {dataset} runs with aligned arrays")
        return
    print(f"[fig8] {len(models)} methods: {', '.join(models)}")

    columns = []
    for model in models:
        xr, xp = load_aligned(runs[model])
        n = len(xr)
        idx = rng.choice(n, min(n, max_cells), replace=False)
        xy_r, xy_p = joint_embedding(xr, xp, idx, seed)
        fos = table_foscttm.get(model)
        if fos is None:
            fos = float(read_diagnostics(runs[model] / "diagnostics.json")["mean_foscttm"])
            print(f"[fig8] {model}: not in Table 1, labelled from its own run")
        lineages = lineages_for_run(runs[model], n)
        labels = None if lineages is None else lineages.to_numpy()[idx]
        columns.append((model, fos, xy_r, xy_p, labels, n))

    best = min(fos for _, fos, *_ in columns)
    colors = LINEAGE_COLORS
    drawn_lineages: set[str] = set()

    n_col = len(columns)
    # Height tuned so the panel BOX aspect (1.11) matches the data's own (1.13), i.e. the
    # clouds are drawn essentially undistorted while autoscale still fills the panel. At
    # the old 2.70 the boxes were 0.76 and every cloud was squeezed to 0.68x its true
    # shape, which is why KOT here did not look like the same cloud as in Fig. 3c.
    fig = plt.figure(figsize=figsize("full", 2.25), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.04, hspace=0.02)
    # Legend column lives inside the venue width so tight bbox cannot inflate
    # the figure and then shrink every font.
    gs = fig.add_gridspec(2, n_col + 1, width_ratios=[1] * n_col + [0.52])
    axes = np.empty((2, n_col), dtype=object)
    for row in range(2):
        for col in range(n_col):
            axes[row, col] = fig.add_subplot(gs[row, col])
    leg_mod = fig.add_subplot(gs[0, n_col])
    leg_lin = fig.add_subplot(gs[1, n_col])
    for ax in (leg_mod, leg_lin):
        ax.set_axis_off()
        ax.xaxis.set_major_locator(NullLocator())
        ax.yaxis.set_major_locator(NullLocator())
        for spine in ax.spines.values():
            spine.set_visible(False)

    tick = STYLE_STATE["ladder"][2]
    legend_kw = dict(
        fontsize=tick, title_fontsize=tick, handlelength=0.65, handleheight=0.5,
        handletextpad=0.35, labelspacing=0.12, borderaxespad=0.0,
    )

    for col, (model, fos, xy_r, xy_p, labels, n) in enumerate(columns):
        name = FIG8_METHOD_LABELS.get(model, method_label(model))
        value = f"FOSCTTM {fos:.3f}" if col == 0 else f"{fos:.3f}"
        weight = "bold" if fos == best else "normal"

        ax = axes[0, col]
        co_embedding_scatter(ax, xy_r, xy_p, rng, style=GRID_POINT_STYLE)
        ax.set_title(f"{name}\n{value}", fontsize=small, fontweight=weight, loc="center")
        framed_embedding(ax)

        ax = axes[1, col]
        if labels is None:
            ax.text(0.5, 0.5, f"labels unavailable\n(n = {n:,})", transform=ax.transAxes,
                    ha="center", va="center", color="0.45")
        else:
            drawn_lineages.update(
                co_embedding_lineage_scatter(ax, xy_r, xy_p, labels, rng, colors,
                                             style=GRID_POINT_STYLE))
        framed_embedding(ax)
        ax.set_xlim(axes[0, col].get_xlim())
        ax.set_ylim(axes[0, col].get_ylim())

    # Inside the KOT cloud the lower-left corner is occupied; park the cue
    # in the margin so the arrows do not cover cells.
    embedding_axes(axes[1, 0], "PC 1", "PC 2", pad=-0.09, frac=0.16)
    modality_legend(leg_mod, loc="center left", title="Modality", **legend_kw)
    key = [name for name in LINEAGE_ORDER if name in drawn_lineages]
    if key:
        short = {name: FIG8_LINEAGE_LABELS.get(name, name) for name in key}
        colour_key(leg_lin, key, colors, loc="center left", title="Lineage",
                   labels=[short[name] for name in key], **legend_kw)
    axes[0, 0].annotate(
        "A", xy=(0.0, 1.0), xycoords="axes fraction",
        xytext=(-2, 15), textcoords="offset points",
        fontsize=PANEL_LETTER_SIZE, fontweight="bold",
        ha="left", va="center", annotation_clip=False)
    axes[1, 0].annotate(
        "B", xy=(0.0, 1.0), xycoords="axes fraction",
        xytext=(-2, 2), textcoords="offset points",
        fontsize=PANEL_LETTER_SIZE, fontweight="bold",
        ha="left", va="bottom", annotation_clip=False)
    save_figure(fig, out / "fig8_coembedding")


def figure9(out: Path, seed: int = 0):
    """Beyond FOSCTTM: where the true partner ranks, and what each method costs."""
    apply_style()
    datasets = ["bmmc_cite_retained", "pbmc_retained"]
    fractions = np.geomspace(1e-4, 0.1, 24)

    fig, axes = plt.subplots(1, 2, figsize=figsize("full", 2.3), layout="constrained")
    # Both panels end on a log decade; without the extra gutter the last tick of one
    # meets the first tick of the other (section 3.4).
    fig.get_layout_engine().set(wspace=0.12)

    ax = axes[0]
    runs = grid_runs(datasets[0])
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
    # Cost the same runs panel a plots, not whatever the manifest still lists.
    runtime_panel(ax, collect_runtimes(datasets, set(CANONICAL_RUNS.values())),
                  GRID_MODELS)
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=4))
    ax.set_title("Cost")
    # No marker edge: the data markers are drawn edgeless, so a ringed key would not
    # match what is on the panel.
    ax.legend(handles=[
        Line2D([0], [0], marker=dataset_style(d)[0], color="none", markersize=3,
               markerfacecolor=dataset_style(d)[1], markeredgecolor="none",
               markeredgewidth=0, label=dataset_style(d)[2])
        for d in datasets], loc="upper right", frameon=False)
    panel_letter(ax, "b")

    save_figure(fig, out / "fig9_beyond_foscttm")


# The two real CITE-seq panels. Too few antibodies for a funnel.
COVERAGE_DATASETS = ["bmmc_cite_retained", "pbmc_retained"]


def figure1c(out: Path):
    """How much RNA-protein linkage survives, and which term each survivor feeds."""
    apply_style()
    coverage = read_coverage(COVERAGE_DATASETS)
    # Not RNA-vs-protein or method colours; lightness distinguishes the two datasets.
    colors = dataset_colors(COVERAGE_DATASETS, [DATASET_COLORS[d] for d in COVERAGE_DATASETS])

    # Taller and wider on the left than the panels need on their own: panel b's wrapped
    # 45-degree tick labels eat height, which compresses panel a until its count labels
    # collide. 2.4 x [1.4, 1] is the smallest pair that passes the bbox check.
    fig, axes = plt.subplots(1, 2, figsize=figsize("full", 2.4), layout="constrained",
                             gridspec_kw=dict(width_ratios=[1.4, 1]))

    funnel_panel(axes[0], coverage, colors)
    axes[0].set_title("How much of the panel survives the RNA link")
    panel_letter(axes[0], "a")

    terms_panel(axes[1], coverage, colors)
    axes[1].set_title("What training actually used")
    axes[1].legend(loc="upper right", frameon=False)
    panel_letter(axes[1], "b")

    save_figure(fig, out / "fig1c_linkage")


def supplement(out: Path):
    """Two supplementary figures: how checkpoints and gradients behaved, and how often
    a run failed or a knob mattered."""
    apply_style()
    dest = out / "appendix"
    colors = dataset_colors(COVERAGE_DATASETS, [DATASET_COLORS[d] for d in COVERAGE_DATASETS])

    # Read from the canonical KOT run directories, not the retired
    # `checkpoint_eval_summary.csv`: that table's 22 source runs are gone from the repo
    # and it carried no dataset column, so its 0.406 median belonged to no stated
    # population. The ablation arm does not bear on the checkpoint choice.
    checkpoints = collect_checkpoints(
        {key: run for key, run in CANONICAL_RUNS.items() if key[0] == "kot"})
    fig, axes = plt.subplots(1, 3, figsize=figsize("full", 2.4), layout="constrained")
    fig.get_layout_engine().set(wspace=0.10)

    ax = axes[0]
    checkpoint_panel(ax, checkpoints, "mean_foscttm", list(COVERAGE_DATASETS))
    ax.set_ylabel("FOSCTTM")
    ax.set_title("Cost of the checkpoint choice")
    # Headroom before the key: the strips reach the top of the panel, so `best` had
    # nowhere to go that was not over a PBMC point.
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom, top + .34 * (top - bottom))
    # One column: side by side the two labels are wider than a third-width panel, and
    # `set_verify` caught the left one crossing the spine.
    ax.legend(loc="upper right", frameon=False, fontsize=6, handletextpad=.3,
              labelspacing=.3, borderaxespad=.2)
    panel_letter(ax, "a")

    ax = axes[1]
    # `jvp_rhs_cos_median`, the statistic Table 4 and Figs. 10 and S2 report. The bare
    # `jvp_rhs_cos` this panel used to plot is aliased to the per-cell MEAN at
    # `src/training/kot.py:1043`, so two different statistics shared one axis label.
    checkpoint_panel(ax, checkpoints, "jvp_rhs_cos_median", list(COVERAGE_DATASETS))
    ax.set_ylabel("JVP·RHS cosine")
    ax.set_title("…on the other metric")
    panel_letter(ax, "b")

    ax = axes[2]
    gradient_panel(ax, read_by_config("summary_warmup_lambda_cfgBC_by_config.csv"))
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=4))
    ax.set_title("Do the two terms fight?")
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom, top + .30 * (top - bottom))
    ax.legend(loc="upper right", frameon=False, fontsize=6, handletextpad=.3,
              labelspacing=.3, borderaxespad=.2)
    panel_letter(ax, "c")

    save_figure(fig, dest / "figS1_optimization")

    fig, axes = plt.subplots(1, 3, figsize=figsize("full", 2.4), layout="constrained")
    fig.get_layout_engine().set(wspace=0.12)

    ax = axes[0]
    flags = collect_flags(COVERAGE_DATASETS)
    flag_panel(ax, flags, colors, headroom=1.45)
    ax.set_title("Run outcomes")
    ax.legend(loc="upper right", frameon=False)
    panel_letter(ax, "a")

    ax = axes[1]
    # Bootstrap interval on the mean, not the pooled table's raw seed SD: that SD is 2-3x
    # any effect here, so the whiskers spanned the panel and said nothing about anchors.
    share_point_panel(
        ax, anchor_alignment(read_results("anchor_ablation_cfgB.csv")), colors)
    # An edge tick here meets the neighbouring panel's edge tick (section 3.4).
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
    ax.set_ylabel("FOSCTTM")
    ax.set_title("Anchors and alignment")
    panel_letter(ax, "b")

    ax = axes[2]
    # Fixed whole-panel average, not the run-level `beta_anchor_mean_abs_err`: that key
    # averages only the proteins a run anchored, and that set grows 5 -> 53 along the
    # axis, so the old rising line was composition. Fig. 11 has the paired version.
    share_point_panel(
        ax, fixed_panel_beta(read_results("beta_anchor_fit_per_protein.csv")), colors)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
    ax.set_ylabel(r"Mean $|\beta-\beta_{\mathrm{target}}|$")
    ax.set_title("Anchors and β")
    # Headroom, then an explicit corner. `best` put the key mid-panel, where its swatches
    # sat in line with the BMMC markers and read as two more data points; upper right
    # without headroom landed on the BMMC markers at the 25% rung.
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom, top + .34 * (top - bottom))
    ax.legend(loc="upper right", frameon=False)
    panel_letter(ax, "c")

    save_figure(fig, dest / "figS2_stability")

    figure_robustness(out)


def figure_robustness(out: Path):
    """Historical tune slice: lambda_dyn × lr_beta. Not a lambda × anchor product."""
    apply_style()
    dest = out / "appendix"
    dest.mkdir(parents=True, exist_ok=True)
    summary_path = Path("cache/results/summary.csv")
    if not summary_path.exists():
        print("[figS5] cache/results/summary.csv missing; skipping")
        return
    summary = read_by_config("summary.csv")
    # FOSCTTM tops out at 0.52, not 0.50: PBMC at lambda_dyn = 1, lr_beta = 1e-3 scores
    # 0.5049, which a 0.50 ceiling clipped to the same colour as a 0.500 cell. Chance is
    # 0.5, so that one cell is the only one in the grid that is WORSE than random, and the
    # clipping hid exactly that.
    metrics = (
        ("foscttm_fitted_mean", "FOSCTTM ↓", "viridis_r", 0.10, 0.52, "FOSCTTM"),
        ("jvp_cos_med_mean", "JVP·RHS cosine ↑", "viridis", 0.0, 1.0,
         "JVP·RHS cosine"),
    )
    header_size = PANEL_LETTER_SIZE + 2.0
    letter_size = PANEL_LETTER_SIZE + 1.5
    title_size = float(plt.rcParams["axes.titlesize"]) + 1.0
    tick_size = float(plt.rcParams["xtick.labelsize"]) + 1.0
    # Constrained layout only for x (tick padding, colour bars). Vertical
    # positions are packed below, so the 3×6 boxes keep the height that a
    # title-sized gap actually needs instead of the engine's row spacing.
    fig = plt.figure(figsize=figsize("full", 3.45), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.05, hspace=0.12,
                                rect=(0.03, 0.05, 0.995, 0.94))
    grid = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 0.065])
    axes = np.empty((2, 2), dtype=object)
    images = [None, None]
    for col, dataset in enumerate(TUNE_DATASETS):
        for row, (value, _, cmap, vmin, vmax, _) in enumerate(metrics):
            table = tune_lambda_grid(summary, dataset, value)
            ax = fig.add_subplot(grid[row, col])
            axes[row, col] = ax
            images[row] = heatmap_grid(
                ax, table, value, cmap=cmap, vmin=vmin, vmax=vmax,
                xlabel="", ylabel="", cellsize=7.0, ticksize=tick_size)
            ax.tick_params(labelbottom=(row == 1), labelleft=(col == 0),
                           labelsize=tick_size)
            missing = int(table["seeds"].isna().sum())
            print(f"[figS5] {dataset} {value}: {missing} of {len(table)} cells not run")
            panel_letter(ax, "abcd"[2 * row + col], dy_points=1, size=letter_size)
    cbar0 = fig.colorbar(images[0], cax=fig.add_subplot(grid[0, 2]))
    cbar1 = fig.colorbar(images[1], cax=fig.add_subplot(grid[1, 2]))
    # Both bars: yellow at the top = better (low FOSCTTM, high cosine).
    # Direction lives in the row titles (↓ / ↑); the bars only name the metric.
    cbar0.ax.invert_yaxis()
    cbar0.set_label(metrics[0][-1])
    cbar1.set_label(metrics[1][-1])
    cbar0.ax.tick_params(labelsize=tick_size)
    cbar1.ax.tick_params(labelsize=tick_size)
    fig.supxlabel(r"$\lambda_{\mathrm{dyn}}$")
    fig.supylabel(r"$\lambda_\beta$")
    fig.canvas.draw()
    fig.set_layout_engine(None)
    # Equal-height rows, gap only for JVP·RHS cosine ↑ plus C/D letters.
    top_y1, bot_y0, gap = 0.820, 0.145, 0.108
    height = (top_y1 - bot_y0 - gap) / 2.0
    top_y0 = bot_y0 + height + gap
    for col in range(2):
        box = axes[0, col].get_position()
        axes[0, col].set_position((box.x0, top_y0, box.width, height))
        axes[1, col].set_position((box.x0, bot_y0, box.width, height))
    for cax, y0 in ((cbar0.ax, top_y0), (cbar1.ax, bot_y0)):
        box = cax.get_position()
        cax.set_position((box.x0, y0, box.width, height))
    for col, dataset in enumerate(TUNE_DATASETS):
        box = axes[0, col].get_position()
        fig.text((box.x0 + box.x1) / 2, box.y1 + 0.088, DATASET_SHORT[dataset],
                 ha="center", va="bottom", fontsize=header_size, fontweight="bold",
                 transform=fig.transFigure)
    for row in range(2):
        left, right = axes[row, 0].get_position(), axes[row, 1].get_position()
        fig.text((left.x0 + right.x1) / 2, left.y1 + 0.054, metrics[row][1],
                 ha="center", va="bottom", fontsize=title_size,
                 transform=fig.transFigure)
    # Caption, not an in-figure footnote: gray = not evaluated. A drawn note
    # would enter `fit_to_venue` and squeeze the cell labels.
    print("[figS5] caption: Gray cells were not evaluated.")
    saved = save_figure(fig, out / "figS5_lambda_beta")
    for path in saved:
        shutil.copy2(path, dest / path.name)


def tables(out: Path):
    """Supplementary tables as CSV: a rendered image cannot be searched or copied.
    """
    dest = out / "tables"
    dest.mkdir(parents=True, exist_ok=True)

    # Read methods from the registry, not a list kept here.
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
    # Pinned, not discovered. `pick_run` plus an mtime fallback landed this copier on
    # whatever happened to be newest: grad_interaction came from `pbmc_scvelo_kot`, a
    # superseded dir that records no hyperparameters and that no other figure reads, and
    # foscttm_diagnostics from a third run again. A figure the appendix cites has to name
    # its run, so these resolve through CANONICAL_RUNS / SYNTHETIC_BRANCH_RUNS.
    wanted = {
        "grad_interaction": canonical_seed_dir("kot", "pbmc_retained"),
        # training_loss was dropped on 2026-09-20: two curves settling well inside the
        # epoch budget is a statement for one sentence of text, not a figure.
        # foscttm_diagnostics is NOT copied: it now needs both the KOT and the
        # no-kinetics per-cell blocks (`plot_foscttm_diagnostics(..., compare=)`), so a
        # PDF written during a single run's training cannot reproduce the shipped panel.
    }
    for name, run in wanted.items():
        src = run / f"{name}.pdf" if run is not None and (run / f"{name}.pdf").exists() else None
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

    print("\n=== Figure 4: agreement with kinetic anchor targets ===")
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
