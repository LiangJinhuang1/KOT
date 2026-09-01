"""Liberation Sans (Arial-metric, in the container). Type-42 PDF (venues reject Type-3).
Three font sizes only: a fourth means the layout is wrong.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# --- Physical widths -------------------------------------------------------
# Text-block widths in inches, per venue style file. A figure wider than its
# venue width gets scaled down by LaTeX, which shrinks every font with it — so
# figures are authored at final size and never scaled.
VENUE_WIDTHS: dict[str, dict[str, float]] = {
    # Single-column body text (\textwidth = 5.5in).
    "neurips": {"full": 5.50, "half": 2.70, "third": 1.75},
    "iclr":    {"full": 5.50, "half": 2.70, "third": 1.75},
    # Two-column body text (\columnwidth = 3.25in, \textwidth = 6.75in).
    "icml":    {"full": 6.75, "column": 3.25, "half": 3.25, "third": 2.15},
    "aaai":    {"full": 7.00, "column": 3.30, "half": 3.30, "third": 2.20},
}

DEFAULT_VENUE = "neurips"

# Role-mapped font ladder (base, small, tick) in points.
#   base  — panel titles, axis labels, direct series labels
#   small — legend entries, annotations
#   tick  — tick labels
FONT_LADDER = (8.0, 7.0, 6.0)

# Panel letters are the documented exception to the three-size ladder.
PANEL_LETTER_SIZE = 9.0

# Scatter defaults tuned for a 5.5in-wide figure at 300 dpi.
SCATTER_STYLE = dict(s=1.6, alpha=0.55, linewidths=0, rasterized=True)

STYLE_STATE: dict[str, object] = {"venue": DEFAULT_VENUE, "ladder": FONT_LADDER,
                             "verify": False}


def set_verify(enabled: bool = True) -> None:
    """Run the §9.1 bbox check inside every save_figure call.

    Off by default because it forces an extra draw; a QA pass turns it on
    once and then every figure the run produces is checked.
    """
    STYLE_STATE["verify"] = bool(enabled)


def apply_style(venue: str = DEFAULT_VENUE, sizes: tuple[float, float, float] = FONT_LADDER) -> None:
    """Install the publication rcParams. Call once, before any plotting."""
    venue = venue.lower()
    if venue not in VENUE_WIDTHS:
        raise ValueError(f"Unknown venue '{venue}'. Use one of {sorted(VENUE_WIDTHS)}.")
    base, small, tick = sizes
    STYLE_STATE["venue"] = venue
    STYLE_STATE["ladder"] = sizes

    mpl.rcParams.update({
        # Typography — Liberation Sans is the Arial-metric substitute present in
        # the container; the rest of the list is a graceful fallback chain.
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica",
                            "Nimbus Sans", "DejaVu Sans"],
        "font.size": base,
        "axes.titlesize": base,
        "axes.labelsize": base,
        "legend.fontsize": small,
        "xtick.labelsize": tick,
        "ytick.labelsize": tick,
        "figure.titlesize": base,

        # Vector output that venues accept: TrueType, not Type-3.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",

        # Frame — drop the box, keep the two spines that carry scale.
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "axes.titlelocation": "left",
        "axes.titlepad": 4.0,
        "axes.labelpad": 2.5,
        "axes.grid": False,

        # Ticks point outward so they never sit on top of the data.
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,

        # Legends live in whitespace, never in a box.
        "legend.frameon": False,
        "legend.handlelength": 1.2,
        "legend.handletextpad": 0.5,
        "legend.labelspacing": 0.3,
        "legend.borderaxespad": 0.3,
        "legend.columnspacing": 1.0,

        "lines.linewidth": 1.1,
        "lines.markersize": 3.0,
        "patch.linewidth": 0.6,

        # Raster fallback resolution; vector PDF is the primary artifact.
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "savefig.transparent": False,
    })


def figsize(width: str | float = "full", height: float = 2.0,
            venue: str | None = None) -> tuple[float, float]:
    """Return ``(w, h)`` inches for a named venue width or an explicit width."""
    if isinstance(width, (int, float)):
        return (float(width), float(height))
    venue = (venue or STYLE_STATE["venue"]).lower()  # type: ignore[union-attr]
    widths = VENUE_WIDTHS[venue]
    if width not in widths:
        raise ValueError(f"Unknown width '{width}' for {venue}. Use one of {sorted(widths)}.")
    return (widths[width], float(height))


def panel_letter(ax, letter: str, *, dy_points: float | None = None,
                 case: str = "lower"):
    """Bold panel letter, above the panel title and aligned with the axes left edge.

    Placed in offset *points* above the axes, not in axes fractions to its left:
    a letter to the left of the axes lands on top of the y tick labels, and one
    at the same height as a left-aligned title collides with it. Offset points
    also survive ``tight_layout``, which axes-fraction placement does not.
    """
    text = letter.lower() if case == "lower" else letter.upper()
    if dy_points is None:
        base = float(mpl.rcParams["axes.titlesize"]) if isinstance(
            mpl.rcParams["axes.titlesize"], (int, float)) else STYLE_STATE["ladder"][0]  # type: ignore[index]
        dy_points = float(mpl.rcParams["axes.titlepad"]) + float(base) + 2.0
    return ax.annotate(
        text, xy=(0.0, 1.0), xycoords="axes fraction",
        xytext=(0.0, dy_points), textcoords="offset points",
        fontsize=PANEL_LETTER_SIZE, fontweight="bold",
        ha="left", va="bottom", annotation_clip=False,
    )


def direction_cue(ax, text: str = "lower = better", *, loc: str = "upper right"):
    """Upright direction-of-goodness cue. Use once per row of panels, never per panel."""
    xy = {
        "upper right": (0.98, 0.98, "right", "top"),
        "upper left":  (0.02, 0.98, "left",  "top"),
        "lower right": (0.98, 0.02, "right", "bottom"),
        "lower left":  (0.02, 0.02, "left",  "bottom"),
    }[loc]
    x, y, ha, va = xy
    return ax.text(x, y, text, transform=ax.transAxes,
                   fontsize=STYLE_STATE["ladder"][1], color="0.35",  # type: ignore[index]
                   ha=ha, va=va, rotation=0)


def median_tick(ax, value: float, position: float, *, half: float = 0.30,
                color: str = "#1A1A1A", lw: float = 1.0,
                orient: str = "vertical", zorder: int = 5):
    """A median marker that stays readable on top of its own scatter.

    Drawn twice — a white halo, then the darker tick just inside it — so the
    median reads against overlapping points without hiding them (§6.1).
    """
    for extra, stroke, width in ((0.01, "white", 2 * lw), (0.0, color, lw)):
        lo, hi = position - half - extra, position + half + extra
        span, const = ([value, value], [lo, hi]) if orient == "vertical" else ([lo, hi], [value, value])
        ax.plot(span, const, color=stroke, lw=width, solid_capstyle="butt",
                zorder=zorder + (0 if extra else 1))


def save_figure(fig, path, *, formats: tuple[str, ...] = ("pdf", "png"),
                dpi: int = 300, close: bool = True, quiet: bool = False,
                verify: bool | None = None) -> list[Path]:
    """Save one figure to every requested format, PDF first.

    The PDF is what goes into LaTeX; the PNG exists for quick viewing and for the
    §9 render-then-verify pass, which needs rasterised pixels to inspect.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if verify if verify is not None else STYLE_STATE["verify"]:
        findings = check_overlaps(fig, verbose=False)
        if findings:
            print(f"[style] {path.name}: {len(findings)} bbox overlap(s)")
            for a, b in findings[:8]:
                print(f"    {a!r} <-> {b!r}")
        else:
            print(f"[style] {path.name}: bbox clean")
    written: list[Path] = []
    for fmt in formats:
        out = path.with_suffix(f".{fmt}")
        fig.savefig(out, format=fmt, dpi=dpi)
        written.append(out)
        if not quiet:
            print(f"Saved: {out}")
    if close:
        plt.close(fig)
    return written


def check_overlaps(fig, *, verbose: bool = True) -> list[tuple[str, str]]:
    """Geometric bbox check (figure-style §9.1).

    Returns the list of overlapping visible text pairs, and of text boxes that
    collide with a spine they do not belong to. An empty list means the figure
    passes the geometric half of the render-then-verify loop; it says nothing
    about contrast or leader-line correctness, which need the perceptual pass.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    texts = [
        (t, t.get_window_extent(renderer))
        for t in fig.findobj(mpl.text.Text)
        if t.get_text().strip() and t.get_visible()
    ]
    spines = [
        (s, s.get_window_extent(renderer))
        for ax in fig.axes for s in ax.spines.values() if s.get_visible()
    ]
    own_ticklabels = {
        ax: set(ax.get_xticklabels(which="both") + ax.get_yticklabels(which="both"))
        for ax in fig.axes
    }

    findings: list[tuple[str, str]] = []
    for i, (ta, ba) in enumerate(texts):
        for tb, bb in texts[i + 1:]:
            if ba.overlaps(bb):
                findings.append((ta.get_text()[:40], tb.get_text()[:40]))
    for t, bt in texts:
        for s, bs in spines:
            if bt.overlaps(bs) and t not in own_ticklabels.get(s.axes, set()):
                findings.append((t.get_text()[:40], f"spine:{s.spine_type}"))

    if verbose:
        if findings:
            print(f"[style] {len(findings)} bbox overlap(s):")
            for a, b in findings[:12]:
                print(f"    {a!r} <-> {b!r}")
        else:
            print("[style] bbox check clean.")
    return findings


def embedding_axes(ax, x_label: str = "Dim 1", y_label: str = "Dim 2",
                   *, frac: float = 0.18, pad: float = 0.02):
    """Strip a dimensionality-reduction scatter to a corner arrow pair (§6.6).

    UMAP/t-SNE/PCA coordinates have no interpretable units, so tick values are
    noise. The arrows name the axes and show orientation without spending any
    of the panel's label budget.
    """
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    small = STYLE_STATE["ladder"][1]  # type: ignore[index]
    arrow = dict(arrowstyle="-|>", lw=0.7, color="0.35", mutation_scale=5)
    ax.annotate("", xy=(pad + frac, pad), xytext=(pad, pad),
                xycoords="axes fraction", arrowprops=arrow)
    ax.annotate("", xy=(pad, pad + frac), xytext=(pad, pad),
                xycoords="axes fraction", arrowprops=arrow)
    ax.text(pad + frac / 2, pad - 0.015, x_label, transform=ax.transAxes,
            fontsize=small, color="0.35", ha="center", va="top")
    ax.text(pad - 0.015, pad + frac / 2, y_label, transform=ax.transAxes,
            fontsize=small, color="0.35", ha="right", va="center", rotation=90)


def chance_line(ax, value: float = 0.5, *, axis: str = "y",
                label: str = "chance", text_x: float = 0.99):
    """Draw the random-baseline reference for a bounded metric.

    A FOSCTTM axis without its 0.5 reference is unreadable: 0.42 looks like a
    number rather than what it is, which is a failure. Always drawn dashed and
    neutral so it cannot be mistaken for a plotted series (§8).
    """
    small = STYLE_STATE["ladder"][1]  # type: ignore[index]
    if axis == "y":
        ax.axhline(value, color="0.45", lw=0.7, ls=(0, (4, 2)), zorder=1)
        ax.text(text_x, value, f" {label} ", transform=ax.get_yaxis_transform(),
                fontsize=small, color="0.45", ha="right", va="bottom")
    else:
        ax.axvline(value, color="0.45", lw=0.7, ls=(0, (4, 2)), zorder=1)
        ax.text(value, text_x, f" {label}", transform=ax.get_xaxis_transform(),
                fontsize=small, color="0.45", ha="left", va="top")


def relative_luminance(hex_color: str) -> float:
    h = hex_color.lstrip("#")
    channels = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    channels = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
                for c in channels]
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


def contrast_ratio(hex_color: str, background: str = "#FFFFFF") -> float:
    la, lb = relative_luminance(hex_color), relative_luminance(background)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def ink(hex_color: str, *, target: float = 4.5, background: str = "#FFFFFF") -> str:
    """Darken a mark colour until it is legible as small text or a thin line.

    The Okabe-Ito palette is designed for filled marks, where its hues are
    colourblind-safe but several sit near 3:1 against white — too light for 6pt
    labels or 1pt strokes (§5.5). Darkening preserves the hue, so colour still
    threads the same entity through the figure; only the duty changes.
    """
    if contrast_ratio(hex_color, background) >= target:
        return hex_color
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    for _ in range(24):
        r, g, b = (int(c * 0.92) for c in (r, g, b))
        candidate = f"#{r:02X}{g:02X}{b:02X}"
        if contrast_ratio(candidate, background) >= target:
            return candidate
    return "#333333"
