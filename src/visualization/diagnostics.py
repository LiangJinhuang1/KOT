"""Supplementary panels: how the runs behaved, not how the method scored.

Four questions a referee asks that no main-text panel answers.

  Which checkpoint?   A run writes four (final, best_align, best_dyn, best_total). If
                      the reported number comes from the checkpoint that minimised the
                      reported metric, the metric was selected on. The panel shows what
                      each choice costs on the OTHER metric, which is the honest way to
                      state the trade rather than defending one choice.
  Do the terms fight? `grad_cos_late` is the angle between the alignment and dynamics
                      gradients late in training and `grad_mag_ratio_late` their size
                      ratio. Together they say whether lambda_dyn bought a compromise or
                      simply let one term dominate.
  How often failed?   Collapse, degeneracy and a dead dynamics term are all invisible in
                      a FOSCTTM ranking, and one of them sorts first. Counting them over
                      every run is the only way the reader learns the failure rate.
  Does the anchor do  The beta prior is a stabiliser, not ground truth. Sweeping the
  anything?           number of anchored proteins from 0 upward separates "it steadies
                      training" from "it is doing the fitting".
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from src.visualization import (METHOD_COLORS, dataset_label, dataset_style,
                               method_label)
from src.visualization.control_effects import ANCHOR_SHARES, mean_interval, share_of
from src.visualization.runs import CACHE_DIR, read_diagnostics, run_flags

RESULTS_DIR = Path("cache/results")

# Written by every KOT run, in the order a reader should compare them: the checkpoint
# with no selection first, then the selected ones. `best_val_align` selects on a held-out
# split rather than on the reported metric, so it is the one that answers "did you select
# on what you report"; the retired summary table did not carry it at all.
CHECKPOINTS = ["final", "best_total", "best_align", "best_val_align", "best_dyn"]
# "best" is implied by the axis label, and repeating it four times is what makes the
# ticks collide in a third-width panel.
CHECKPOINT_LABELS = {"final": "final", "best_total": "total", "best_align": "align",
                     "best_val_align": "val align", "best_dyn": "dyn"}

FLAG_LABELS = {"degenerate": "degenerate", "collapsed": "collapsed", "dyn-dead": "dyn-dead"}


def collect_checkpoints(runs: dict) -> pd.DataFrame:
    """Every saved checkpoint's diagnostics, for the runs the thesis actually reports.

    Replaces `cache/results/checkpoint_eval_summary.csv`. That table pooled 257 run-seeds
    from 22 `run_20260616_*` / `run_20260617_*` directories which are absent from
    `cache/training` and from every manifest in the repo, so nothing it showed could be
    re-derived. It also had no `dataset` column: its median FOSCTTM of 0.406 is neither
    BMMC's 0.129 nor PBMC's 0.135, and there was no way to say what population it was.

    Reading the run directories instead gives a named dataset per row, the seeds the
    tables use, and `jvp_rhs_cos_median` -- the statistic Table 4 and Figs. 10 and S2
    report -- rather than the bare `jvp_rhs_cos`, which `src/training/kot.py` aliases to
    the per-cell MEAN under the same name.
    """
    rows = []
    for (model, dataset), run in runs.items():
        for seed_dir in sorted((CACHE_DIR / run / model / dataset).glob("seed_*")):
            for checkpoint in CHECKPOINTS:
                path = seed_dir / f"diagnostics_{checkpoint}.json"
                if not path.exists():
                    continue
                values = json.loads(path.read_text())
                rows.append({"run": run, "model": model, "dataset": dataset,
                             "seed": int(seed_dir.name.split("_")[1]),
                             "checkpoint": checkpoint,
                             "checkpoint_epoch": values.get("checkpoint_epoch"),
                             "mean_foscttm": values.get("mean_foscttm"),
                             "jvp_rhs_cos_median": values.get("jvp_rhs_cos_median")})
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No checkpoint diagnostics found for the requested runs")
    return frame


def checkpoint_panel(ax, table: pd.DataFrame, metric: str, datasets: list[str]):
    """One strip per checkpoint, per dataset: what the selection choice costs.

    Split by dataset, not pooled. The retired version pooled an unidentifiable mixture of
    datasets into one strip, so its median answered no question anyone had asked.

    Points are run-seeds and the tick is their median. A bar of means would hide how often
    two checkpoints land on the same value, which is the whole question.
    """
    for i, checkpoint in enumerate(CHECKPOINTS):
        for j, dataset in enumerate(datasets):
            values = table.loc[(table["checkpoint"] == checkpoint)
                               & (table["dataset"] == dataset), metric].dropna()
            if values.empty:
                continue
            marker, colour, _ = dataset_style(dataset)
            x = i + (j - (len(datasets) - 1) / 2) * 0.26
            jitter = np.random.default_rng(i * 10 + j).uniform(-0.05, 0.05, len(values))
            ax.scatter(x + jitter, values, s=5, alpha=0.5, linewidths=0, marker=marker,
                       color=colour, zorder=3,
                       label=dataset_label(dataset) if i == 0 else None)
            ax.plot([x - 0.11, x + 0.11], [values.median()] * 2, color="#1A1A1A",
                    lw=1.0, zorder=4)
    ax.set_xticks(range(len(CHECKPOINTS)))
    ax.set_xticklabels([CHECKPOINT_LABELS[c] for c in CHECKPOINTS], rotation=45, ha="right")
    ax.set_xlabel("Checkpoint selected on")
    ax.set_xlim(-0.6, len(CHECKPOINTS) - 0.4)


def gradient_panel(ax, table: pd.DataFrame):
    """Gradient angle against gradient size ratio, one point per swept configuration.

    Both axes are mechanism, not score: a configuration where the two terms are almost
    orthogonal (cosine near zero) and the ratio is large is one where lambda_dyn changed
    the loss without changing the direction training moved in.

    Restricted to `fit_tuning_subset == False`, the fitting population the thesis reports.
    The table holds the tuning-subset half of the same sweep as well, and mixing the two
    put thirty-two points in a panel that claims to describe one protocol. `lr_beta` and
    `lr_warmup_epochs` are not free either -- 1e-3 always comes with 300 and 1e-2 with
    500 -- so what remains varying is two configs (B and C) crossed with
    `dyn_warmup_epochs`, which is what the panel is for.

    The only thing a marker encodes is which dataset the configuration ran on. An earlier
    version also drew the `collapsed,dyn-dead` configurations as a third grey series, but
    a status key in a third-width panel costs more than it explains. The caveat travels in
    the caption instead, and it is worth carrying: every flagged configuration is
    `dyn_warmup_epochs == 0`, they score FOSCTTM 0.42 (BMMC) and 0.33 (PBMC) against
    0.12-0.14 for the rest, and they are the ENTIRE right-hand tail -- the ratio only
    reaches 1-12 and the cosine only reaches -0.77 among runs that never trained.

    Each point is ONE run (`n_runs` is 1 for every configuration), so there is no seed
    replication here and no interval to draw.
    """
    table = table[~table["fit_tuning_subset"].astype(bool)]
    for dataset, marker in (("bmmc_cite_retained", "o"), ("pbmc_retained", "s")):
        rows = table[table["dataset"] == dataset]
        if rows.empty:
            continue
        ax.plot(rows["grad_mag_ratio_late_mean"], rows["grad_cos_late_mean"], marker,
                ms=3.6, lw=0, markeredgewidth=0, color=dataset_style(dataset)[1],
                label=dataset_label(dataset), zorder=3)
    ax.set_xscale("log")
    ax.axhline(0, color="0.45", lw=0.7, ls=(0, (4, 2)), zorder=1)
    ax.set_xlabel("Align / dyn gradient ratio")
    ax.set_ylabel("Gradient cosine")


def collect_flags(datasets: list[str]) -> pd.DataFrame:
    """Every run's failure flags, counted per dataset. Unflagged runs count as clean."""
    rows = []
    for path in CACHE_DIR.rglob("*/diagnostics.json"):
        diagnostics = read_diagnostics(path)
        dataset = diagnostics.get("dataset")
        if dataset not in datasets or "mean_foscttm" not in diagnostics:
            continue
        flags = run_flags(diagnostics)
        rows.append({"dataset": dataset, "model": diagnostics.get("model", "?"),
                     "flags": flags.split(",") if flags else ["clean"]})
    return pd.DataFrame(rows)


def flag_panel(ax, flags: pd.DataFrame, colors: dict[str, str], headroom: float = 1.18):
    """Share of runs carrying each failure flag, per dataset, with the count annotated.

    `headroom` buys room above the tallest bar. The default clears the count label; a
    caller that also draws a legend needs more, because the key is wider than the empty
    middle columns and lands on the `clean` counts otherwise.
    """
    categories = ["clean"] + list(FLAG_LABELS)
    width = 0.8 / flags["dataset"].nunique()
    tallest = 0.0
    for i, dataset in enumerate(sorted(flags["dataset"].unique())):
        sub = flags[flags["dataset"] == dataset]
        counts = [int(sub["flags"].apply(lambda f, c=c: c in f).sum()) for c in categories]
        x = [j + i * width for j in range(len(categories))]
        # Plus n, so the key says how many runs each bar pools.
        short = dataset_label(dataset)
        ax.bar(x, [c / len(sub) for c in counts], width=width, color=colors[dataset],
               label=f"{short} (n = {len(sub)})", zorder=3)
        for xx, count in zip(x, counts):
            # Offset in points, not in data: a zero-height bar would otherwise print
            # its count directly onto the bottom spine.
            ax.annotate(str(count), (xx, count / len(sub)), textcoords="offset points",
                        xytext=(0, 2), ha="center", va="bottom", color="0.35")
        tallest = max(tallest, max(counts) / len(sub))
    ax.set_xticks([j + width / 2 for j in range(len(categories))])
    ax.set_xticklabels([FLAG_LABELS.get(c, c) for c in categories], rotation=45,
                       ha="right")
    # Headroom for the count above the tallest bar, which otherwise reaches the title.
    ax.set_ylim(0, tallest * headroom)
    # A share cannot exceed 1, so the extra room carries no ticks of its own.
    ax.set_yticks(np.arange(0, 1.01, .2))
    ax.set_ylabel("Share of runs")


def anchor_panel(ax, table: pd.DataFrame, metric: str, colors: dict[str, str]):
    """A metric against the SHARE of the panel anchored, one line per dataset.

    Shares, not counts. The sweep's rungs are 0/10/25/50/75/100% of each antibody panel,
    so BMMC's 5 anchors and PBMC's 1 are one condition; on a count axis PBMC's whole
    sweep crushed into the left tenth of the panel and the two datasets could not be read
    against each other. Fixed rather than derived, because 27/53 rounds to 51% and 3/10
    to 30% and neither is a rung.
    """
    for dataset in sorted(table["dataset"].unique()):
        sub = table[table["dataset"] == dataset].sort_values("beta_anchor_subset_n")
        share = share_of(sub["beta_anchor_subset_n"].unique())
        ax.errorbar(sub["beta_anchor_subset_n"].map(share), sub[f"{metric}_mean"],
                    yerr=sub[f"{metric}_sd"], marker="o", ms=3, lw=1.0, elinewidth=0.7,
                    capsize=1.5, color=colors[dataset], label=dataset_label(dataset),
                    zorder=3)
    ax.set_xlabel("Share anchored (%)")
    ax.set_xticks(ANCHOR_SHARES)
    ax.set_xlim(-6, 106)


def anchor_alignment(frame: pd.DataFrame) -> pd.DataFrame:
    """One FOSCTTM per seed at each rung, long form for :func:`share_point_panel`."""
    out = frame[["dataset", "seed", "beta_anchor_subset_n", "mean_foscttm"]].copy()
    share = {d: share_of(g.beta_anchor_subset_n.unique()) for d, g in out.groupby("dataset")}
    out["share"] = [share[d][n] for d, n in zip(out.dataset, out.beta_anchor_subset_n)]
    return out.rename(columns={"mean_foscttm": "value"})[["dataset", "share", "seed", "value"]]


def share_point_panel(ax, table: pd.DataFrame, colors: dict[str, str], dodge: float = 1.8):
    """Seed points plus the mean and its interval, against the share anchored.

    The per-seed cloud is the point of the panel, not decoration: the pooled table these
    panels used to read carries only a mean and a raw seed SD, and that SD is 2-3x any
    effect across the anchor sweep, so the whiskers spanned the axis and said nothing.
    Showing the seeds and a bootstrap interval on the mean separates "the sweep does
    nothing" from "the seeds are noisy", which are different claims.

    Same marker language as Fig. 11, so the supplement reads like the main text.
    """
    for i, dataset in enumerate(sorted(table["dataset"].unique())):
        sub = table[table["dataset"] == dataset]
        colour = colors[dataset]
        # A small x offset: the two datasets' clouds otherwise interleave at every rung.
        offset = (i - .5) * 2 * dodge
        for share, group in sub.groupby("share"):
            values = group["value"].to_numpy()
            mean, low, high = mean_interval(values)
            ax.scatter(np.full(len(values), share + offset), values, s=7, alpha=.3,
                       color=colour, edgecolors="none", zorder=2)
            ax.errorbar(share + offset, mean, yerr=[[mean - low], [high - mean]],
                        fmt="D", ms=3, color=colour, capsize=2, elinewidth=1, zorder=4,
                        label=dataset_label(dataset) if share == 0 else None)
    ax.set_xlabel("Share anchored (%)")
    ax.set_xticks(ANCHOR_SHARES)
    ax.set_xlim(-6, 106)


def fixed_panel_beta(frame: pd.DataFrame) -> pd.DataFrame:
    """Mean |beta - target| over a protein set that does NOT change along the axis.

    The run-level `beta_anchor_mean_abs_err` this panel used to read averages only the
    proteins that run anchored, and that set grows 5 -> 53 along the axis while being
    drawn easy-first (at zero anchors the proteins that will be anchored already score
    0.364 against 0.457 for the rest). The published rise 0.227 -> 0.298 was therefore
    composition, not fit -- see Fig. 11, where the paired version of this comparison runs
    the other way. Averaging the whole in-mask panel at every rung holds the set fixed.
    """
    frame = frame[(frame.lr_beta == .001) & (frame.lr_warmup_epochs == 300)
                  & frame.in_kinetics]
    rows = []
    for dataset, group in frame.groupby("dataset"):
        share = share_of(group.beta_anchor_subset_n.unique())
        panel = group.groupby(["beta_anchor_subset_n", "seed"]).protein.apply(frozenset)
        if panel.nunique() != 1:
            raise ValueError(f"{dataset}: the in-mask panel changes along the axis")
        for (count, seed), sub in group.groupby(["beta_anchor_subset_n", "seed"]):
            rows.append({"dataset": dataset, "share": share[count], "seed": seed,
                         "value": float(sub.abs_err.mean())})
    return pd.DataFrame(rows)



def read_results(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / name)


def read_by_config(name: str) -> pd.DataFrame:
    return read_results(name)


def read_checkpoints() -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / "checkpoint_eval_summary.csv")


# Historical tune table, locked except lambda_dyn and lr_beta. This is not the
# 12-seed canonical protocol, and it is not a lambda × anchor grid.
TUNE_LOCK = {
    "fitted_side": "tune",
    "lr_phi": 0.001,
    "lr_alpha_kappa": 0.0001,
    "lr_warmup_epochs": 300,
    "sinkhorn_reg": 0.1,
}
TUNE_DATASETS = ("bmmc_cite_retained", "pbmc_retained")
TUNE_LAMBDAS = (1, 100, 300, 500, 1000, 2000)
TUNE_BETAS = (0.001, 0.003, 0.01)


def tune_lambda_grid(summary: pd.DataFrame, dataset: str, value: str) -> pd.DataFrame:
    """lambda_dyn × lr_beta grid; missing experiments stay NaN."""
    keep = summary["dataset"] == dataset
    for column, expected in TUNE_LOCK.items():
        series = summary[column]
        if isinstance(expected, float):
            keep &= np.isclose(pd.to_numeric(series, errors="coerce"), expected)
        elif isinstance(expected, int):
            keep &= pd.to_numeric(series, errors="coerce") == expected
        else:
            keep &= series == expected
    sub = summary.loc[keep, ["lambda_dyn", "lr_beta", value, "seeds"]].copy()
    sub["lambda_dyn"] = pd.to_numeric(sub["lambda_dyn"], errors="coerce").astype(float)
    sub["lr_beta"] = pd.to_numeric(sub["lr_beta"], errors="coerce").astype(float)
    sub[value] = pd.to_numeric(sub[value], errors="coerce")
    sub["seeds"] = pd.to_numeric(sub["seeds"], errors="coerce")
    if sub.duplicated(["lambda_dyn", "lr_beta"]).any():
        raise ValueError(f"{dataset}: duplicate tune-grid cells")
    index = pd.MultiIndex.from_product(
        [np.asarray(TUNE_LAMBDAS, dtype=float), np.asarray(TUNE_BETAS, dtype=float)],
        names=["lambda_dyn", "lr_beta"])
    sub = sub.set_index(["lambda_dyn", "lr_beta"]).reindex(index).reset_index()
    sub["dataset"] = dataset
    return sub


def heatmap_grid(ax, table: pd.DataFrame, value: str, *, cmap: str, vmin, vmax,
                 fmt: str = "{:.3f}"):
    """Draw the lambda × lr_beta grid; hatched cells were not run.

    Every run cell is also printed. The differences this grid exists to show sit in the
    third decimal -- BMMC's best cell is 0.124 against 0.133 at the chosen setting -- and
    no reader can take that off a viridis ramp. The colour carries the gist, the number
    carries the comparison.
    """
    lambdas = np.asarray(TUNE_LAMBDAS, dtype=float)
    betas = np.asarray(TUNE_BETAS, dtype=float)
    matrix = table.pivot(index="lr_beta", columns="lambda_dyn", values=value)
    matrix = matrix.reindex(index=betas, columns=lambdas)
    masked = np.ma.masked_invalid(matrix.to_numpy(dtype=float))
    image = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                      interpolation="nearest")
    if "seeds" in table.columns:
        not_run = table.pivot(index="lr_beta", columns="lambda_dyn",
                              values="seeds").reindex(index=betas, columns=lambdas)
        missing = not_run.isna().to_numpy()
    else:
        missing = np.isnan(matrix.to_numpy(dtype=float))
    for row, col in np.argwhere(missing):
        ax.add_patch(Rectangle((col - 0.5, row - 0.5), 1, 1, fill=False,
                               hatch="////", edgecolor="0.6", linewidth=0.4))
    values = matrix.to_numpy(dtype=float)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            if missing[row, col] or not np.isfinite(values[row, col]):
                continue
            red, green, blue, _ = image.cmap(image.norm(values[row, col]))
            # Perceived luminance, so the label stays legible at both ends of the ramp.
            luminance = .299 * red + .587 * green + .114 * blue
            # No leading zero: both metrics live in [0, 1], the digit carries nothing,
            # and a cell here is only ~25 pt wide.
            ax.text(col, row, fmt.format(values[row, col]).replace("0.", "."),
                    ha="center", va="center", fontsize=5,
                    color="#FFFFFF" if luminance < .55 else "#1A1A1A")
    # Six lambda labels across a 150 pt panel collide flat; the project already rotates
    # crowded categorical ticks (Figs. S1, S2).
    ax.set_xticks(range(len(TUNE_LAMBDAS)), [str(v) for v in TUNE_LAMBDAS],
                  rotation=45, ha="right")
    ax.set_yticks(range(len(TUNE_BETAS)), [str(v) for v in TUNE_BETAS])
    ax.set_xlabel(r"$\lambda_{\mathrm{dyn}}$")
    ax.set_ylabel(r"$\mathrm{lr}_\beta$")
    return image
