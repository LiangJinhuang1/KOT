"""Turn a PowerPoint-exported SVG back into a submission-ready figure PDF.

Completes the loop opened by ``svg_for_powerpoint.py``. PowerPoint's own
"Save as Picture -> SVG" keeps vectors and live text, but it knows nothing about
the venue text block or the paper's typeface, so two things drift silently:

* the page comes out at whatever size the shapes happen to occupy, not the
  column width the figure was authored for -- and LaTeX rescaling a figure
  shrinks every font below the ladder;
* any text retyped inside PowerPoint picks up PowerPoint's default face
  (usually Calibri), which then sits beside the untouched Liberation Sans.

Both are reported, and the width is corrected on request.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cairosvg
import pypdf

# Faces the paper is authored in. Anything else in an exported figure means
# PowerPoint substituted, which is a visible inconsistency rather than an error.
EXPECTED_FONT_STEMS = ("liberationsans", "arial", "helvetica", "nimbussans")

# Single-column text block, in inches, per venue style file.
VENUE_WIDTH = {"neurips": 5.50, "iclr": 5.50, "icml": 6.75, "aaai": 7.00}


def page_size_inches(pdf_path: Path) -> tuple[float, float]:
    box = pypdf.PdfReader(str(pdf_path)).pages[0].mediabox
    return float(box.width) / 72.0, float(box.height) / 72.0


def font_report(pdf_path: Path) -> tuple[list[str], list[str], list[str]]:
    """Return (all faces, Type3 faces, non-embedded faces) across every page."""
    faces: set[str] = set()
    type3: set[str] = set()
    loose: set[str] = set()
    for page in pypdf.PdfReader(str(pdf_path)).pages:
        table = page.get("/Resources", {}).get("/Font", {})
        if not hasattr(table, "items"):
            continue
        for _, ref in table.items():
            font = ref.get_object()
            name = str(font.get("/BaseFont", "?"))
            faces.add(name)
            if font.get("/Subtype") == "/Type3":
                type3.add(name)
            descriptor = font.get("/FontDescriptor")
            if descriptor is not None:
                descriptor = descriptor.get_object()
                if not any(k in descriptor for k in ("/FontFile", "/FontFile2", "/FontFile3")):
                    loose.add(name)
    return sorted(faces), sorted(type3), sorted(loose)


def substituted_fonts(faces: list[str]) -> list[str]:
    """Faces that are not part of the paper's typeface set."""
    unexpected = []
    for face in faces:
        # Strip the six-letter subset tag PDF writers prepend, e.g. "WDPUKE+".
        stem = face.split("+")[-1].lstrip("/").replace("-", "").replace(" ", "").lower()
        if not any(stem.startswith(known) for known in EXPECTED_FONT_STEMS):
            unexpected.append(face)
    return unexpected


def convert(source: Path, destination: Path, target_width: float | None,
            also_png: bool) -> bool:
    """Write the PDF (and optionally a PNG), then report. True if it passes."""
    cairosvg.svg2pdf(url=str(source), write_to=str(destination))
    width, height = page_size_inches(destination)

    if target_width is not None and abs(width - target_width) > 0.01:
        # cairo scales the whole drawing, so the aspect ratio and every font
        # size in the figure scale together -- which is what we want, since the
        # figure was authored to be placed at exactly this width.
        cairosvg.svg2pdf(url=str(source), write_to=str(destination),
                         scale=target_width / width)
        width, height = page_size_inches(destination)

    faces, type3, loose = font_report(destination)
    swapped = substituted_fonts(faces)
    if also_png:
        cairosvg.svg2png(url=str(source), write_to=str(destination.with_suffix(".png")),
                         scale=300.0 / 96.0)

    ok = not (type3 or loose or swapped)
    print(f"{source.name} -> {destination.name}")
    print(f"  page            {width:.2f} x {height:.2f} in")
    print(f"  fonts           {len(faces)}  " + ", ".join(f.split('+')[-1] for f in faces))
    print(f"  Type3           {len(type3)}" + ("  <-- journals reject these" if type3 else ""))
    print(f"  non-embedded    {len(loose)}" + ("  <-- must be embedded" if loose else ""))
    if swapped:
        print(f"  SUBSTITUTED     {', '.join(swapped)}")
        print("                  PowerPoint replaced the paper's typeface. Select the "
              "text there, set it back to Liberation Sans (or Arial), re-export.")
    print(f"  verdict         {'PASS' if ok else 'FIX NEEDED'}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("svg", type=Path, nargs="+", help="SVG exported from PowerPoint")
    parser.add_argument("--out-dir", type=Path, default=Path("figures/from_powerpoint"))
    parser.add_argument("--venue", choices=sorted(VENUE_WIDTH), default="iclr",
                        help="scale the page to this venue's text-block width")
    parser.add_argument("--no-resize", action="store_true",
                        help="keep whatever width PowerPoint exported")
    parser.add_argument("--png", action="store_true", help="also write a 300 dpi PNG")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    target = None if args.no_resize else VENUE_WIDTH[args.venue]
    failures = 0
    for source in args.svg:
        if not source.exists():
            raise SystemExit(f"no such file: {source}")
        stem = source.stem[:-4] if source.stem.endswith("_ppt") else source.stem
        if not convert(source, args.out_dir / f"{stem}.pdf", target, args.png):
            failures += 1
    if failures:
        raise SystemExit(f"{failures} figure(s) need fixing before submission")


if __name__ == "__main__":
    main()
