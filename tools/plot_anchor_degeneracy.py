"""Where the β prior binds, κ rescales inversely; κ·β (what the ODE sees) does not."""
import json, glob, collections
import sys
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# The external figure-style skill this script imported no longer exists anywhere, so
# the script had stopped running at all. src/visualization/style.py is the same 8/7/6
# ladder and open frame, and it also enforces the venue width on save.
from src.visualization.style import apply_style, figsize, panel_letter, save_figure

STAMP = "run_20260830_225641_anch"
# Short dataset names, matching every other figure. This script hardcodes its titles
# rather than calling `dataset_label`, so it did not pick up the project-wide change.
PANELS = [("bmmc", "bmmc_cite_retained", "BMMC (53 anchors)"),
          ("pbmc", "pbmc_retained", "PBMC (10 anchors)")]
# The learning rate is the whole difference between the rows, so it is the whole label.
# "prior binds / never binds" was an interpretation printed as if it were an axis: at
# 1e-2 the anchor loss reaches beta (anchor |err| 0.012 on BMMC, and dropping the anchors
# halves beta), at 1e-3 it does not (|err| 0.298, beta moves 2%). That belongs in the
# caption, where it can carry those numbers.
CFGS = [("cfgC", r"lr$_\beta$ = 10$^{-2}$"),
        ("cfgB", r"lr$_\beta$ = 10$^{-3}$")]
PCT = {"bmmc": {0: 0, 5: 10, 13: 25, 27: 50, 40: 75, 53: 100},
       "pbmc": {0: 0, 1: 10, 3: 25, 5: 50, 8: 75, 10: 100}}

# One statistic throughout: every series is a MEDIAN. `beta` used to read `beta_mean`
# while kappa and alpha read their medians, so the product line was a median times a
# mean -- neither the mean nor the median of anything the ODE evaluates.
# The RHS is kappa*(alpha*s - beta*phi), so kappa*alpha and kappa*beta are the two
# products that act on the state; both are drawn.
# Five distinct hues, not two families of light/dark. Tying each product to its own
# parameter by shade put alpha next to kappa*alpha and beta next to kappa*beta, and at
# 1.5 pt against 2.4 pt those pairs were not separable. Line WIDTH still says parameter
# (thin) versus product (thick); colour now only has to say which series.
# Okabe-Ito where possible, with a saturated purple for the fifth.
SERIES = [("kappa", r"$\kappa$ (time scale)", "#0072B2", 1.5, "-"),
          ("alpha", r"$\alpha$ (translation)", "#E69F00", 1.5, "-"),
          ("beta",  r"$\beta$ (degradation)", "#CC79A7", 1.5, "-"),
          ("prod_a", r"$\kappa\cdot\alpha$ (production)", "#009E73", 2.4, "-"),
          ("prod",  r"$\kappa\cdot\beta$ (degradation)", "#762A83", 2.4, "-")]


def load(panel, ds, cfg):
    """Per-seed trajectories, each normalised to that seed's OWN full-anchor run, so the
    ratio is paired within a seed and carries no between-seed offset."""
    by_seed = collections.defaultdict(dict)
    for p in glob.glob(f"cache/training/{STAMP}_{panel}_scv_{cfg}*/kot/{ds}/seed_*/diagnostics.json"):
        d = json.load(open(p))
        seed = int(p.split("seed_")[1].split("/")[0])
        by_seed[seed][d["beta_anchor_subset_n"]] = {
            "kappa": d["kappa_median"], "alpha": d["alpha_median"],
            "beta": d["beta_median"],
            "prod": d["kappa_median"] * d["beta_median"],
            "prod_a": d["kappa_median"] * d["alpha_median"]}
    ks = sorted(PCT[panel])
    full = ks[-1]
    out = {name: [] for name, _, _, _, _ in SERIES}
    for seed, rungs in sorted(by_seed.items()):
        for name in out:
            out[name].append([rungs[k][name] / rungs[full][name] for k in ks])
    return [PCT[panel][k] for k in ks], {n: np.array(v) for n, v in out.items()}


apply_style(sizes=(8, 7, 6))
# 5.5in is the ICLR text block. At 7.1in LaTeX scales the figure down and every
# font lands below the 8/7/6 ladder this script asks for.
fig, axes = plt.subplots(2, 2, figsize=figsize("full", 3.9), sharex=True, sharey=True)
letters = iter("abcd")

for row, (cfg, _) in enumerate(CFGS):
    for col, (panel, ds, ds_label) in enumerate(PANELS):
        ax = axes[row, col]
        pcts, series = load(panel, ds, cfg)
        ax.axhline(1.0, color="0.82", lw=0.8, zorder=0)
        for name, _, color, lw, ls in SERIES:
            m = series[name].mean(axis=0)
            sd = series[name].std(axis=0, ddof=1)
            ax.fill_between(pcts, m - sd, m + sd, color=color, alpha=0.12, lw=0, zorder=1)
            ax.plot(pcts, m, ls, marker="o", color=color, lw=lw, ms=3.2, zorder=3,
                    mec="white", mew=0.6)
        ax.margins(0.06)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4, prune="both"))
        if row == 0:
            ax.set_title(ds_label, pad=5)
        ax.text(0.97, 0.05, "n = 12 seeds", transform=ax.transAxes, ha="right",
                fontsize=mpl.rcParams["legend.fontsize"], color="0.35")
        panel_letter(ax, next(letters))

for row, (_, cfg_label) in enumerate(CFGS):
    box = axes[row, 1].get_position()
    fig.text(box.x1 + 0.018, (box.y0 + box.y1) / 2, cfg_label, rotation=270,
             va="center", ha="left", fontsize=mpl.rcParams["axes.titlesize"])

handles = [mpl.lines.Line2D([], [], color=c, lw=lw, ls=ls, marker="o", ms=3.2,
                            mec="white", mew=0.6, label=lab)
           for _, lab, c, lw, ls in SERIES]
axes[0, 0].legend(handles=handles, loc="upper right", frameon=False, ncol=1,
                  handlelength=1.5, borderaxespad=0.2, labelspacing=0.25)

# No suptitle: it collided with the panel letter, and in the paper the caption says
# this. Panel titles still name the two datasets.
fig.supxlabel("Share anchored (%)", y=0.035, fontsize=mpl.rcParams["axes.labelsize"])
fig.supylabel("value / full-anchor value", x=0.016, fontsize=mpl.rcParams["axes.labelsize"])
# hspace 0.30, not 0.15: the bottom row's panel letter sits above its axes and landed
# on the spines of the row above.
fig.subplots_adjust(left=0.105, right=0.855, top=0.94, bottom=0.115, wspace=0.09, hspace=0.30)

# save_figure writes PDF+PNG, holds the figure to the venue text block, and runs the
# same bbox check every other figure in the paper is held to.
save_figure(fig, Path("figures/anchor_ablation/kappa_beta_degeneracy"), verify=True)
