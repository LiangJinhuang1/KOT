"""Where the β prior binds, κ rescales inversely; κ·β (what the ODE sees) does not."""
import json, glob, collections
import sys
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/jinliang/.claude/skills/figure-style")
from kernel import apply_figure_style, set_frame, panel_letter, panel_crops

STAMP = "run_20260830_225641_anch"
PANELS = [("bmmc", "bmmc_cite_retained", "BMMC CITE-seq (53 anchors)"),
          ("pbmc", "pbmc_retained", "PBMC CITE-seq (10 anchors)")]
CFGS = [("cfgC", "prior binds\n" + r"(lr$_\beta$ = 10$^{-2}$)"),
        ("cfgB", "prior never binds\n" + r"(lr$_\beta$ = 10$^{-3}$)")]
PCT = {"bmmc": {0: 0, 5: 10, 13: 25, 27: 50, 40: 75, 53: 100},
       "pbmc": {0: 0, 1: 10, 3: 25, 5: 50, 8: 75, 10: 100}}

SERIES = [("kappa", r"$\kappa$ (time scale)", "#0B6FA4", 1.5),
          ("alpha", r"$\alpha$ (translation)", "#E08214", 1.5),
          ("beta",  r"$\beta$ (degradation)", "#8073AC", 1.5),
          ("prod",  r"$\kappa\cdot\beta$ (what the ODE sees)", "#111111", 2.4)]


def load(panel, ds, cfg):
    """Per-seed trajectories, each normalised to that seed's OWN full-anchor run, so the
    ratio is paired within a seed and carries no between-seed offset."""
    by_seed = collections.defaultdict(dict)
    for p in glob.glob(f"cache/training/{STAMP}_{panel}_scv_{cfg}*/kot/{ds}/seed_*/diagnostics.json"):
        d = json.load(open(p))
        seed = int(p.split("seed_")[1].split("/")[0])
        by_seed[seed][d["beta_anchor_subset_n"]] = {
            "kappa": d["kappa_median"], "alpha": d["alpha_median"], "beta": d["beta_mean"],
            "prod": d["kappa_median"] * d["beta_mean"]}
    ks = sorted(PCT[panel])
    full = ks[-1]
    out = {name: [] for name, _, _, _ in SERIES}
    for seed, rungs in sorted(by_seed.items()):
        for name in out:
            out[name].append([rungs[k][name] / rungs[full][name] for k in ks])
    return [PCT[panel][k] for k in ks], {n: np.array(v) for n, v in out.items()}


apply_figure_style(frame="open", sizes=(8, 7, 6))
fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.0), sharex=True, sharey=True)
letters = iter("abcd")

for row, (cfg, _) in enumerate(CFGS):
    for col, (panel, ds, ds_label) in enumerate(PANELS):
        ax = axes[row, col]
        pcts, series = load(panel, ds, cfg)
        ax.axhline(1.0, color="0.82", lw=0.8, zorder=0)
        for name, _, color, lw in SERIES:
            m = series[name].mean(axis=0)
            s = series[name].std(axis=0, ddof=1)
            ax.fill_between(pcts, m - s, m + s, color=color, alpha=0.12, lw=0, zorder=1)
            ax.plot(pcts, m, "-o", color=color, lw=lw, ms=3.2, zorder=3, mec="white", mew=0.6)
        set_frame(ax, "open")
        ax.margins(0.06)
        ax.set_xticks([0, 25, 50, 75, 100])
        if row == 0:
            ax.set_title(ds_label, pad=5)
        ax.text(0.97, 0.05, "n = 12 seeds", transform=ax.transAxes, ha="right",
                fontsize=mpl.rcParams["legend.fontsize"], color="0.35")
        panel_letter(ax, next(letters), dx=-0.085, dy=1.03)

# Row-band headers (§7.4) on the right, where nothing else lives.
for row, (_, cfg_label) in enumerate(CFGS):
    box = axes[row, 1].get_position()
    fig.text(box.x1 + 0.018, (box.y0 + box.y1) / 2, cfg_label, rotation=270,
             va="center", ha="left", fontsize=mpl.rcParams["axes.titlesize"])

# Series identity: one frameless legend in panel (a)'s upper whitespace (§7.3, §4.6).
handles = [mpl.lines.Line2D([], [], color=c, lw=lw, marker="o", ms=3.2,
                            mec="white", mew=0.6, label=lab)
           for _, lab, c, lw in SERIES]
axes[0, 0].legend(handles=handles, loc="upper right", frameon=False, ncol=1,
                  handlelength=1.5, borderaxespad=0.2, labelspacing=0.25)

fig.suptitle(r"Anchoring $\beta$ rescales $\kappa$ inversely, and only where the prior binds",
             y=0.985, fontsize=mpl.rcParams["axes.titlesize"])
fig.supxlabel("anchors retained (%)", y=0.035, fontsize=mpl.rcParams["axes.labelsize"])
fig.supylabel("value / full-anchor value", x=0.016, fontsize=mpl.rcParams["axes.labelsize"])
fig.subplots_adjust(left=0.105, right=0.855, top=0.885, bottom=0.115, wspace=0.09, hspace=0.15)

out = Path("figures/anchor_ablation/kappa_beta_degeneracy.png")
fig.savefig(out, dpi=300)
fig.savefig(out.with_suffix(".pdf"))
print("saved", out)

fig.canvas.draw()
r = fig.canvas.get_renderer()
offview = set()
for a in fig.axes:
    for labels, lim, i in ((a.get_yticklabels(), a.get_ylim(), 1),
                           (a.get_xticklabels(), a.get_xlim(), 0)):
        for t in labels:
            v = t.get_position()[i]
            if not (min(lim) <= v <= max(lim)):
                offview.add(t)


def onscreen(t):
    b = t.get_window_extent(r)
    return (fig.bbox.x0 <= b.x1 and b.x0 <= fig.bbox.x1
            and fig.bbox.y0 <= b.y1 and b.y0 <= fig.bbox.y1)
texts = [(t, t.get_window_extent(r)) for t in fig.findobj(mpl.text.Text)
         if t.get_text().strip() and t.get_visible() and onscreen(t) and t not in offview]
spines = [(s, s.get_window_extent(r)) for a in fig.axes for s in a.spines.values() if s.get_visible()]
tl = {a: set(a.get_xticklabels(which="both") + a.get_yticklabels(which="both")) for a in fig.axes}
ov = [(a.get_text()[:22], b.get_text()[:22]) for i, (a, ba) in enumerate(texts)
      for b, bb in texts[i+1:] if ba.overlaps(bb)]
ov += [(t.get_text()[:22], "SPINE") for t, bt in texts for s, bs in spines
       if bt.overlaps(bs) and t not in tl[s.axes]]
outside = [t.get_text()[:22] for t, b in texts
           if not (fig.bbox.x0 <= b.x0 and b.x1 <= fig.bbox.x1
                   and fig.bbox.y0 <= b.y0 and b.y1 <= fig.bbox.y1)]
print("BBOX overlaps:", ov if ov else "none")
print("outside figure:", outside if outside else "none")
for letter, box in panel_crops(fig).items():
    print("panel", letter, box)
