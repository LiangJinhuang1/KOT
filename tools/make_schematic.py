#!/usr/bin/env python
"""Emit the Figure 1a method schematic as an editable SVG.

Written as SVG rather than plotted: box-and-arrow diagrams fight matplotlib, and
the output needs to stay editable in Inkscape or Illustrator through revisions.
Text is real text (not paths), so labels can be retyped downstream.

Geometry, palette, and type sizes match the plotted panels so the schematic and
the data figures read as one system.

    PYTHONPATH=. python tools/make_schematic.py --out figures/fig1a_schematic.svg
"""
from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import cairosvg
import numpy as np
from sklearn.decomposition import PCA

from src.data.synthetic_linked_ode import stage_output_paths

# 100 user units = 1 inch, so the file is authored at final printed size.
W, H = 550, 232
RNA, PROT, KOT = "#5D3A9B", "#9C6500", "#0072B2"
GREY, RULE, INK = "#767676", "#CFCFCF", "#1A1A1A"
FONT = "Liberation Sans, Arial, Helvetica, sans-serif"
BASE, SMALL, TINY = 11, 9.5, 8.5     # 8pt / 7pt / 6pt at 100 units per inch


# Asymmetric on purpose (RNA under cache/velocity/, protein under cache/) — taken
# from the runner rather than re-typed, so a change to the stage layout cannot
# leave the schematic reading a path that no longer exists.
BRANCH_RNA, BRANCH_PROTEIN = (Path(p) for p in stage_output_paths("branch"))


def load_branch(which: str, n: int = 90, seed: int = 0):
    """Real branch-stage cells, PCA'd to 2-D and returned per branch label.

    Drawn from the actual dataset rather than a formula, so the schematic shows
    the same geometry the data panels do. Falls back to `None` if the cache is
    absent, and the caller then uses the synthetic curve.
    """
    path = BRANCH_RNA if which == "rna" else BRANCH_PROTEIN
    if not path.exists():
        print(f"[schematic] {path} not generated yet; using the drawn curve")
        return None
    a = ad.read_h5ad(path)
    layer = "spliced_mean" if which == "rna" else "protein_mean"
    X = np.asarray(a.layers[layer] if layer in a.layers else a.X, dtype=float)
    xy = PCA(n_components=2, random_state=seed).fit_transform(X)
    states = a.obs["state"].astype(str).to_numpy()

    rng = np.random.default_rng(seed)
    groups = {}
    for st in ("Progenitor", "Branch_A", "Branch_B"):
        m = states == st
        if m.sum() == 0:
            continue
        take = rng.choice(np.flatnonzero(m), min(n, int(m.sum())), replace=False)
        groups[st] = xy[take]
    return groups or None


def fit_into(xy: np.ndarray, bounds: np.ndarray, box) -> np.ndarray:
    """Scale one shared bounding box into a panel rectangle, y flipped for SVG."""
    x0, y0, w, h = box
    (lo_x, lo_y), (hi_x, hi_y) = bounds
    sx = w / max(hi_x - lo_x, 1e-9)
    sy = h / max(hi_y - lo_y, 1e-9)
    k = min(sx, sy)
    cx = x0 + w / 2 - k * (lo_x + hi_x) / 2
    cy = y0 + h / 2 + k * (lo_y + hi_y) / 2
    return np.column_stack([cx + k * xy[:, 0], cy - k * xy[:, 1]])


def bounds_of(groups) -> np.ndarray:
    allxy = np.vstack(list(groups.values()))
    return np.array([allxy.min(axis=0), allxy.max(axis=0)])


def trajectory(t, *, branch=0.0, x0=0.0, y0=0.0, sx=1.0, sy=1.0):
    """Fallback curve, used only when the branch cache is missing."""
    x = t
    # clip before the fractional power: (t - 0.5) ** 1.5 is NaN for t < 0.5.
    post = np.clip(t - 0.5, 0.0, None)
    y = branch * post ** 1.5 * 3.2
    return x0 + sx * x, y0 + sy * (y + 0.06 * np.sin(3.6 * t))


def dots(xs, ys, color, r=1.5, opacity=0.75):
    return "\n".join(
        f'    <circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{color}" '
        f'fill-opacity="{opacity}"/>' for x, y in zip(xs, ys))


def arrow(x1, y1, x2, y2, color, width=1.1, head="head"):
    return (f'    <line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{color}" stroke-width="{width}" marker-end="url(#{head})"/>')


def text(x, y, s, *, size=BASE, color=INK, anchor="start", weight="normal",
         style="normal"):
    return (f'    <text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" '
            f'font-size="{size}" fill="{color}" text-anchor="{anchor}" '
            f'font-weight="{weight}" font-style="{style}">{s}</text>')


def build() -> str:
    rng = np.random.default_rng(3)
    out: list[str] = []

    out.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="5.5in" '
               f'height="{H/100:.2f}in" viewBox="0 0 {W} {H}">')
    out.append('  <defs>')
    for name, col in (("head", INK), ("headRNA", RNA), ("headKOT", KOT),
                      ("headGrey", GREY)):
        out.append(
            f'    <marker id="{name}" viewBox="0 0 10 10" refX="9" refY="5" '
            f'markerWidth="5" markerHeight="5" orient="auto-start-reverse">'
            f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{col}"/></marker>')
    out.append('  </defs>')
    out.append(f'  <rect width="{W}" height="{H}" fill="white"/>')

    # ---------------------------------------------------------------- inputs
    out.append('  <g id="inputs">')
    out.append(text(14, 24, "RNA cells", size=BASE, color=RNA, weight="600"))
    out.append(text(14, 36, "state r  +  velocity v", size=SMALL, color=GREY))
    rna_groups = load_branch("rna", n=150)
    if rna_groups:
        bnd = bounds_of(rna_groups)
        placed = {k: fit_into(v, bnd, (18, 48, 100, 78)) for k, v in rna_groups.items()}
        for xy in placed.values():
            out.append(dots(xy[:, 0], xy[:, 1], RNA))
        # A velocity arrow on the tip of each branch — the input the ODE needs.
        for st in ("Branch_A", "Branch_B"):
            if st not in placed:
                continue
            xy = placed[st]
            tip = xy[np.argmax(xy[:, 0])]
            mid = xy[np.argsort(xy[:, 0])[len(xy) // 2]]
            d = tip - mid
            n = np.hypot(*d) or 1
            out.append(arrow(tip[0], tip[1],
                             tip[0] + 15 * d[0] / n, tip[1] + 15 * d[1] / n,
                             RNA, 1.3, "headRNA"))
    else:
        t = np.linspace(0, 1, 44)
        for br in (+1, -1):
            bx, by = trajectory(t, branch=br, x0=16, y0=78, sx=100, sy=40)
            out.append(dots(bx, by, RNA))

    out.append(text(14, 152, "Protein cells", size=BASE, color=PROT, weight="600"))
    out.append(text(14, 164, "different cells, no pairing", size=SMALL, color=GREY))
    prot_groups = load_branch("protein", n=120, seed=1)
    if prot_groups:
        bnd = bounds_of(prot_groups)
        for xy in (fit_into(v, bnd, (18, 170, 100, 46)) for v in prot_groups.values()):
            out.append(dots(xy[:, 0], xy[:, 1], PROT))
    else:
        for br in (+1, -1):
            px, py = trajectory(np.linspace(0, 1, 36), branch=br, x0=16, y0=197,
                                sx=100, sy=15)
            out.append(dots(px, py, PROT))
    out.append('  </g>')

    # ------------------------------------------------------------------- phi
    out.append('  <g id="phi">')
    out.append(f'    <rect x="150" y="96" width="54" height="30" rx="3" '
               f'fill="none" stroke="{KOT}" stroke-width="1.3"/>')
    out.append(text(177, 116, "&#966;", size=17, color=KOT, anchor="middle", style="italic"))
    out.append(text(177, 90, "learned map", size=TINY, color=GREY, anchor="middle"))
    out.append(arrow(124, 111, 147, 111, INK, 1.1))
    out.append(arrow(206, 111, 244, 111, INK, 1.1))
    out.append(text(225, 105, "&#966;(r)", size=SMALL, color=KOT, anchor="middle",
                    style="italic"))
    out.append('  </g>')

    # ------------------------------------------------- shared protein space
    out.append('  <g id="shared">')
    out.append(f'    <rect x="248" y="42" width="152" height="166" rx="3" '
               f'fill="none" stroke="{RULE}" stroke-width="0.8"/>')
    out.append(text(324, 34, "Shared protein space", size=SMALL, color=GREY,
                    anchor="middle"))
    # Both clouds interleave on the same geometry: the claim is that they match.
    shared = load_branch("protein", n=130, seed=2)
    if shared:
        bnd = bounds_of(shared)
        for xy in (fit_into(v, bnd, (260, 62, 128, 118)) for v in shared.values()):
            out.append(dots(xy[:, 0], xy[:, 1], PROT, r=1.5, opacity=0.5))
            out.append(dots(xy[:, 0] + rng.normal(0, 1.7, len(xy)),
                            xy[:, 1] + rng.normal(0, 1.7, len(xy)), RNA,
                            r=1.4, opacity=0.5))
    else:
        for br in (+1, -1):
            base_x, base_y = trajectory(np.linspace(0, 1, 38), branch=br,
                                        x0=262, y0=118, sx=124, sy=44)
            out.append(dots(base_x, base_y, PROT, r=1.5, opacity=0.5))
            out.append(dots(base_x, base_y, RNA, r=1.4, opacity=0.5))

    # Constraint 1 — Sinkhorn acts on the whole cloud.
    out.append(f'    <ellipse cx="326" cy="128" rx="70" ry="48" fill="none" '
               f'stroke="{GREY}" stroke-width="0.9" stroke-dasharray="4 2.5"/>')
    out.append(text(324, 200, "Sinkhorn matches the cloud", size=TINY, color=GREY,
                    anchor="middle"))

    # Constraint 2 — the kinetics term acts on the direction of one cell's arrow.
    cx, cy = 340, 74
    out.append(f'    <circle cx="{cx}" cy="{cy}" r="16" fill="white" '
               f'fill-opacity="0.9" stroke="{KOT}" stroke-width="1" '
               f'stroke-dasharray="2.5 2"/>')
    out.append(arrow(cx - 5, cy + 6, cx + 15, cy - 9, KOT, 1.4, "headKOT"))
    out.append(arrow(cx - 5, cy + 6, cx + 11, cy + 14, GREY, 0.9, "headGrey"))
    out.append(text(cx + 19, cy - 12, "J&#966;&#183;v", size=SMALL, color=KOT,
                    style="italic"))
    out.append(f'    <line x1="{cx + 8}" y1="{cy + 17}" x2="{cx - 14}" y2="{cy + 30}" '
               f'stroke="{GREY}" stroke-width="0.5"/>')
    out.append(text(cx - 16, cy + 33, "ruled out", size=TINY, color=GREY,
                    anchor="end"))
    out.append('  </g>')

    # ---------------------------------------------------- right-hand column
    out.append('  <g id="constraint">')
    out.append(arrow(432, 72, 406, 72, INK, 1.0))
    out.append(text(436, 62, "the kinetics term", size=SMALL, color=INK, weight="600"))
    out.append(text(436, 74, "fixes the direction,", size=TINY, color=GREY))
    out.append(text(436, 85, "not just the cloud", size=TINY, color=GREY))

    out.append(f'    <line x1="436" y1="108" x2="540" y2="108" stroke="{RULE}" '
               f'stroke-width="0.8"/>')
    # Compact form: the full expression does not fit the column at a legible size.
    out.append(text(436, 126, "J&#966;&#183;v = &#954;(&#945;&#183;Sr &#8722; &#946;&#183;&#966;)",
                    size=BASE, color=INK))
    out.append(text(436, 144, "&#954;  time-scale", size=TINY, color=GREY))
    out.append(text(436, 155, "&#945;  translation rate", size=TINY, color=GREY))
    out.append(text(436, 166, "&#946;  degradation rate", size=TINY, color=GREY))
    out.append(text(436, 177, "S  gene &#8594; protein map", size=TINY, color=GREY))
    out.append(text(436, 194, "&#946; is checked against", size=TINY, color=PROT))
    out.append(text(436, 205, "measured half-lives", size=TINY, color=PROT))
    out.append('  </g>')

    out.append('</svg>')
    return "\n".join(out)


def rasterise(svg_path: Path, scale: float = 3.0) -> Path:
    """PNG alongside the SVG, so the schematic gets the same render-then-verify
    pass as the plotted panels rather than shipping unlooked-at."""
    png = svg_path.with_suffix(".png")
    cairosvg.svg2png(url=str(svg_path), write_to=str(png), scale=scale)
    return png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("figures/fig1a_schematic.svg"))
    ap.add_argument("--scale", type=float, default=3.0,
                    help="PNG preview scale; 3 gives ~300 dpi at the authored size.")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build())
    print(f"Saved: {args.out}")
    print(f"Saved: {rasterise(args.out, args.scale)}")
    print("Edit in Inkscape (free) or Illustrator; text stays editable.")


if __name__ == "__main__":
    main()
