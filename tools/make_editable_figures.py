"""Editable copies of the thesis figures.

Fig. 1a is a native draw.io schematic (boxes, arrows, labels). Each plotted
panel gets a .drawio next to its PNG so titles and callouts can be added in
diagrams.net without rerunning matplotlib. Regenerating a panel still writes
SVG beside the PDF (see save_figure); the PDF remains the LaTeX cite.
"""
from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path
from xml.sax.saxutils import quoteattr

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"

RNA, PROT, KOT = "#5D3A9B", "#9C6500", "#0072B2"
GREY, RULE, INK = "#767676", "#CFCFCF", "#1A1A1A"

# Cite stems from figures/THESIS_FIGURES.md. Fig. 1a is native draw.io, not a wrap.
PLOTTED = (
    "fig1b_ablation",
    "fig1c_linkage",
    "fig2_synthetic",
    "fig3_realdata",
    "fig4_kinetics",
    "fig5_velocity",
    "fig6_percell",
    "fig7_prediction",
    "fig8_coembedding",
    "fig9_beyond_foscttm",
    "fig10_paired_controls",
    "fig11_anchor_transfer",
    "fig12_crispr_measured_rna",
    "fig13_crispr_predicted_rna",
    "figS1_optimization",
    "figS2_stability",
    "figS3_velocity_detail",
    "figS4_crispr_effect_arrows",
    "figS5_lambda_beta",
    "kappa_beta_degeneracy",
    "training_loss",
    "foscttm_diagnostics",
    "grad_interaction",
)

PNG_SUBDIRS = ("", "appendix", "anchor_ablation")

TEXT = ("text;html=1;align=left;verticalAlign=middle;fontFamily=Helvetica;"
        "fontColor={c};fontSize={s};fontStyle={f};")
BOX = ("rounded=1;whiteSpace=wrap;html=1;fontFamily=Helvetica;fontColor={fc};"
       "fontSize={s};fillColor={fill};strokeColor={stroke};strokeWidth={sw};")
STRAIGHT = ("endArrow=block;endSize=8;html=1;rounded=0;strokeColor=#1A1A1A;"
            "strokeWidth=1.6;edgeStyle=none;")
VEL = ("endArrow=classic;endSize=7;html=1;rounded=0;strokeColor=" + RNA +
       ";strokeWidth=2;edgeStyle=none;")
KIN_OK = ("endArrow=classic;endSize=8;html=1;rounded=0;strokeColor=" + KOT +
          ";strokeWidth=2.2;edgeStyle=none;")
KIN_NO = ("endArrow=classic;endSize=7;html=1;rounded=0;strokeColor=" + GREY +
          ";strokeWidth=1.6;edgeStyle=none;")


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path} is not a PNG")
    length, chunk = struct.unpack(">I4s", header[8:16])
    if chunk != b"IHDR" or length != 13:
        raise ValueError(f"{path} is missing an IHDR chunk")
    width, height = struct.unpack(">II", header[16:24])
    return int(width), int(height)


def locate_pngs(dest: Path, stem: str) -> list[Path]:
    found = []
    for sub in PNG_SUBDIRS:
        path = dest / sub / f"{stem}.png" if sub else dest / f"{stem}.png"
        if path.exists():
            found.append(path)
    return found


def y_points(x: float, y: float, w: float, h: float, n_stem: int = 10, n_arm: int = 8):
    """A wiggle-Y like the synthetic branch, stem on the left."""
    stem, up, down = [], [], []
    for i in range(n_stem):
        t = i / (n_stem - 1)
        stem.append((
            x + 0.06 * w + t * 0.34 * w,
            y + 0.52 * h + 5 * math.sin(2.8 * t),
        ))
    fork = stem[-1]
    for i in range(1, n_arm + 1):
        t = i / n_arm
        up.append((
            fork[0] + t * 0.54 * w,
            fork[1] - t * 0.36 * h - 4 * math.sin(2.2 * t),
        ))
        down.append((
            fork[0] + t * 0.54 * w,
            fork[1] + t * 0.38 * h + 4 * math.sin(2.2 * t),
        ))
    return stem, up, down


def ensure_html(style: str) -> str:
    if "html=1" in style:
        return style
    return f"{style};html=1" if style else "html=1"


class Drawio:
    def __init__(self, name: str, width: int, height: int):
        self.name = name
        self.width = width
        self.height = height
        self.cells: list[str] = []
        self.ids = {"0", "1"}

    def claim_id(self, cell_id: str) -> None:
        if cell_id in self.ids:
            raise ValueError(f"draw.io id {cell_id!r} is reserved or already used")
        self.ids.add(cell_id)

    def add(self, cell_id: str, value: str, style: str, x: float, y: float,
            w: float, h: float, *, parent: str = "1"):
        self.claim_id(cell_id)
        style = ensure_html(style)
        self.cells.append(
            f"        <mxCell id={quoteattr(cell_id)} value={quoteattr(value)} "
            f"style={quoteattr(style)} vertex=\"1\" parent={quoteattr(parent)}>"
            f"<mxGeometry x=\"{x:.1f}\" y=\"{y:.1f}\" width=\"{w:.1f}\" "
            f"height=\"{h:.1f}\" as=\"geometry\"/></mxCell>"
        )

    def edge(self, cell_id: str, source: str, target: str, style: str, *,
             value: str = "", parent: str = "1"):
        self.claim_id(cell_id)
        style = ensure_html(style)
        self.cells.append(
            f"        <mxCell id={quoteattr(cell_id)} value={quoteattr(value)} "
            f"style={quoteattr(style)} edge=\"1\" parent={quoteattr(parent)} "
            f"source={quoteattr(source)} target={quoteattr(target)}>"
            f"<mxGeometry relative=\"1\" as=\"geometry\"/></mxCell>"
        )

    def edge_points(self, cell_id: str, x1: float, y1: float, x2: float, y2: float,
                    style: str, *, value: str = "", parent: str = "1"):
        self.claim_id(cell_id)
        style = ensure_html(style)
        self.cells.append(
            f"        <mxCell id={quoteattr(cell_id)} value={quoteattr(value)} "
            f"style={quoteattr(style)} edge=\"1\" parent={quoteattr(parent)}>"
            f"<mxGeometry relative=\"1\" as=\"geometry\">"
            f"<mxPoint x=\"{x1:.1f}\" y=\"{y1:.1f}\" as=\"sourcePoint\"/>"
            f"<mxPoint x=\"{x2:.1f}\" y=\"{y2:.1f}\" as=\"targetPoint\"/>"
            f"</mxGeometry></mxCell>"
        )

    def xml(self) -> str:
        body = "\n".join(self.cells)
        return (
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
            "<mxfile host=\"app.diagrams.net\" version=\"22.1.0\">\n"
            f"  <diagram id={quoteattr(self.name)} name={quoteattr(self.name)}>\n"
            f"    <mxGraphModel dx=\"1200\" dy=\"700\" grid=\"1\" gridSize=\"10\" "
            f"guides=\"1\" tooltips=\"1\" connect=\"1\" arrows=\"1\" fold=\"1\" "
            f"page=\"1\" pageScale=\"1\" pageWidth=\"{self.width}\" "
            f"pageHeight=\"{self.height}\" math=\"0\" shadow=\"0\">\n"
            "      <root>\n"
            "        <mxCell id=\"0\"/>\n"
            "        <mxCell id=\"1\" parent=\"0\"/>\n"
            f"{body}\n"
            "      </root>\n"
            "    </mxGraphModel>\n"
            "  </diagram>\n"
            "</mxfile>\n"
        )


def add_cloud(d: Drawio, prefix: str, stem, up, down, color: str, *, r: float = 7.0,
              opacity: int = 70, parent: str = "1"):
    style = (f"ellipse;html=1;aspect=fixed;fillColor={color};strokeColor=none;"
             f"fillOpacity={opacity};connectable=0;")
    for i, (x, y) in enumerate(stem + up + down):
        d.add(f"{prefix}-{i}", "", style, x - r / 2, y - r / 2, r, r, parent=parent)


def build_schematic() -> str:
    """Native draw.io Fig. 1a: same story as figures/fig1a_schematic.svg."""
    d = Drawio("fig1a-schematic", 1100, 500)

    d.add("rna-title", "RNA cells", TEXT.format(c=RNA, s=16, f=1), 20, 20, 200, 20)
    d.add("rna-sub", "state r + velocity v", TEXT.format(c=GREY, s=12, f=0),
          20, 40, 220, 20)
    d.add("rna-cloud", "", "group;pointerEvents=0;html=1;", 20, 60, 220, 180)
    stem, up, down = y_points(10, 10, 200, 150)
    add_cloud(d, "rna", stem, up, down, RNA, r=7.5, opacity=80, parent="rna-cloud")
    d.edge_points("rna-vel-a", up[-2][0], up[-2][1],
                  up[-1][0] + 18, up[-1][1] - 10, VEL, parent="rna-cloud")
    d.edge_points("rna-vel-b", down[-2][0], down[-2][1],
                  down[-1][0] + 16, down[-1][1] + 10, VEL, parent="rna-cloud")
    d.add("rna-port", "", "ellipse;html=1;fillColor=none;strokeColor=none;",
          230, 140, 10, 10)

    d.add("prot-title", "Protein cells", TEXT.format(c=PROT, s=16, f=1),
          20, 270, 200, 20)
    d.add("prot-sub", "different cells, no pairing", TEXT.format(c=GREY, s=12, f=0),
          20, 290, 240, 20)
    d.add("prot-cloud", "", "group;pointerEvents=0;html=1;", 20, 310, 200, 110)
    pstem, pup, pdown = y_points(10, 10, 180, 90)
    add_cloud(d, "prot", pstem, pup, pdown, PROT, r=7, opacity=85, parent="prot-cloud")
    # Protein enters shared space below φ: prot-port y=370, shared entryY=0.72
    # (~y=315). Do not source or target this edge on phi.
    d.add("prot-port", "", "ellipse;html=1;fillColor=none;strokeColor=none;",
          230, 370, 10, 10)

    d.add("phi-label", "learned map",
          TEXT.format(c=GREY, s=11, f=0) + "align=center;", 250, 160, 110, 20)
    d.add("phi", "φ",
          BOX.format(fc=KOT, s=22, fill="#ffffff", stroke=KOT, sw=2) + "fontStyle=2;",
          260, 180, 100, 50)
    d.add("phi-out", "φ(r)",
          TEXT.format(c=KOT, s=12, f=2) + "align=center;", 370, 190, 50, 20)

    d.add("shared", "",
          BOX.format(fc=INK, s=12, fill="#ffffff", stroke=RULE, sw=1.2) +
          "container=1;pointerEvents=0;",
          430, 70, 320, 340)
    d.add("shared-title", "Shared protein space",
          TEXT.format(c=GREY, s=13, f=0) + "align=center;", 430, 50, 320, 20)
    sstem, sup, sdown = y_points(40, 50, 260, 230)
    add_cloud(d, "sh-p", sstem, sup, sdown, PROT, r=6.5, opacity=55, parent="shared")
    jittered = ([(x + 4, y - 3) for x, y in pts] for pts in (sstem, sup, sdown))
    jstem, jup, jdown = jittered
    add_cloud(d, "sh-r", jstem, jup, jdown, RNA, r=6, opacity=55, parent="shared")
    d.add("sinkhorn", "",
          f"ellipse;html=1;fillColor=none;strokeColor={GREY};strokeWidth=1.2;"
          "dashed=1;dashPattern=8 5;pointerEvents=0;",
          40, 70, 240, 180, parent="shared")
    d.add("sinkhorn-label", "Sinkhorn matches the cloud",
          TEXT.format(c=GREY, s=11, f=0) + "align=center;", 430, 400, 320, 20)

    d.add("kin-title", "the kinetics term", TEXT.format(c=INK, s=15, f=1),
          780, 30, 300, 20)
    d.add("kin-sub", "fixes the direction,<br>not just the cloud",
          TEXT.format(c=GREY, s=12, f=0), 780, 50, 300, 40)
    d.add("kin-cloud", "", "group;pointerEvents=0;html=1;", 780, 100, 220, 130)
    kstem, kup, kdown = y_points(10, 10, 200, 110)
    add_cloud(d, "kin", kstem, kup, kdown, RNA, r=6, opacity=70, parent="kin-cloud")
    kx, ky = kup[3]
    d.add("kin-cell", "",
          f"ellipse;html=1;fillColor=#ffffff;strokeColor={RNA};strokeWidth=1.4;",
          kx - 6, ky - 6, 12, 12, parent="kin-cloud")
    d.edge_points("kin-jvp", kx, ky, kup[-1][0] + 8, kup[-1][1] - 6, KIN_OK,
                  parent="kin-cloud")
    d.add("kin-jvp-l", "Jφ·v", TEXT.format(c=KOT, s=13, f=2),
          kup[-1][0] + 12, kup[-1][1] - 18, 70, 20, parent="kin-cloud")
    d.edge_points("kin-bad", kx, ky, kdown[-1][0] - 10, kdown[-1][1], KIN_NO,
                  parent="kin-cloud")
    d.add("kin-bad-l", "ruled out", TEXT.format(c=GREY, s=11, f=0),
          200, 110, 90, 20, parent="kin-cloud")
    d.add("kin-eq", "Jφ·v = κ(α·Sr − β·φ)", TEXT.format(c=INK, s=15, f=0),
          780, 300, 300, 24)
    d.add("kin-k", "κ  time-scale", TEXT.format(c=GREY, s=11, f=0), 780, 330, 300, 18)
    d.add("kin-a", "α  translation rate", TEXT.format(c=GREY, s=11, f=0),
          780, 350, 300, 18)
    d.add("kin-b", "β  degradation rate", TEXT.format(c=GREY, s=11, f=0),
          780, 370, 300, 18)

    pin = ("exitX=1;exitY=0.5;exitDx=0;exitDy=0;"
           "entryX=0;entryY={ey};entryDx=0;entryDy=0;")
    d.edge("e-rna-phi", "rna-port", "phi", STRAIGHT + pin.format(ey=0.5))
    d.edge("e-phi-shared", "phi", "shared", STRAIGHT + pin.format(ey=0.35))
    d.edge("e-prot-shared", "prot-port", "shared", STRAIGHT + pin.format(ey=0.72))
    d.edge("e-shared-kin", "shared", "kin-cell",
           STRAIGHT + "exitX=1;exitY=0.45;exitDx=0;exitDy=0;"
           "entryX=0;entryY=0.5;entryDx=0;entryDy=0;")

    d.add("legend", "",
          "rounded=0;whiteSpace=wrap;html=1;fillColor=none;strokeColor=none;",
          20, 450, 520, 40)
    d.add("leg-rna", "", f"ellipse;html=1;fillColor={RNA};strokeColor=none;",
          0, 12, 12, 12, parent="legend")
    d.add("leg-rna-t", "RNA", TEXT.format(c=RNA, s=11, f=0),
          18, 8, 50, 20, parent="legend")
    d.add("leg-prot", "", f"ellipse;html=1;fillColor={PROT};strokeColor=none;",
          80, 12, 12, 12, parent="legend")
    d.add("leg-prot-t", "Protein", TEXT.format(c=PROT, s=11, f=0),
          98, 8, 70, 20, parent="legend")
    d.add("leg-kot", "", f"rounded=1;html=1;fillColor=#ffffff;strokeColor={KOT};",
          180, 10, 20, 16, parent="legend")
    d.add("leg-kot-t", "learned map φ", TEXT.format(c=KOT, s=11, f=0),
          206, 8, 140, 20, parent="legend")
    return d.xml()


def build_image_page(stem: str, png: Path) -> str:
    """One plotted figure as a draw.io page: PNG on the canvas for overlays."""
    pw, ph = png_size(png)
    scale = min(1.0, 1100 / pw)
    w, h = int(pw * scale), int(ph * scale)
    page_w = max(w + 40, 400)
    page_h = max(h + 80, 300)
    d = Drawio(stem, page_w, page_h)
    d.add("note",
          "Background is the rendered panel. Add labels on top; keep the PDF for LaTeX.",
          "text;html=1;align=left;fontSize=11;fontColor=#767676;fontFamily=Helvetica;",
          20, 10, w, 24)
    d.add("panel", "",
          "shape=image;html=1;imageAspect=0;aspect=fixed;verticalAlign=top;"
          f"connectable=0;image={png.name}",
          20, 40, w, h)
    return d.xml()


def write(path: Path, xml: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(xml, encoding="utf-8")
    print(f"Saved: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=FIGURES)
    args = ap.parse_args()
    dest = args.out_dir
    write(dest / "fig1a_schematic.drawio", build_schematic())
    for stem in PLOTTED:
        pngs = locate_pngs(dest, stem)
        if not pngs:
            print(f"[editable] skip {stem}: no PNG")
            continue
        for png in pngs:
            write(png.with_suffix(".drawio"), build_image_page(stem, png))


if __name__ == "__main__":
    main()
