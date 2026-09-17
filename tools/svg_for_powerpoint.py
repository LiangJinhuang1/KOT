"""Rewrite a figure SVG so PowerPoint ungroups it into editable blocks.

PowerPoint imports an SVG as a single picture. "Convert to Shape" turns it into
a shape group, but two things in a normal matplotlib export make the result
unusable afterwards:

* every axes is wrapped in a ``clip-path`` group, which PowerPoint either drops
  or turns into a stray rectangle;
* a scatter layer is thousands of sibling primitives, so ungrouping buries the
  handful of blocks you actually want to move (boxes, arrows, labels) among
  them, and PowerPoint slows to a crawl.

This rewrites both: clips are removed, pure-nesting groups are flattened, and
each dense point cloud is collapsed into ONE object -- either a named group or,
with ``--raster-clouds``, a single embedded bitmap. Text is never touched, so it
stays editable as text.
"""
from __future__ import annotations

import argparse
import base64
from pathlib import Path
from xml.etree import ElementTree as ET

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"

# Primitives that make up a scatter layer. A run of these longer than the
# threshold is a point cloud, not a set of individually meaningful shapes.
CLOUD_TAGS = {"circle", "path", "use", "rect"}


def tag_of(element: ET.Element) -> str:
    """Local tag name, without the ``{namespace}`` prefix ElementTree keeps."""
    return element.tag.split("}")[-1]


def strip_clip_paths(root: ET.Element) -> int:
    """Drop clip-path references and their defs. Returns how many were removed."""
    removed = 0
    for element in root.iter():
        if element.attrib.pop("clip-path", None) is not None:
            removed += 1
    for parent in root.iter():
        for child in [c for c in parent if tag_of(c) == "clipPath"]:
            parent.remove(child)
            removed += 1
    return removed


def flatten_transparent_groups(root: ET.Element) -> int:
    """Splice out ``<g>`` wrappers that carry no styling, transform or id.

    matplotlib nests several of these per axes. They add depth to the PowerPoint
    shape tree without ever being something a reader wants to select.
    """
    flattened = 0
    changed = True
    while changed:
        changed = False
        for parent in list(root.iter()):
            for child in list(parent):
                if tag_of(child) != "g" or child.attrib:
                    continue
                index = list(parent).index(child)
                for offset, grandchild in enumerate(list(child)):
                    parent.insert(index + offset, grandchild)
                parent.remove(child)
                flattened += 1
                changed = True
    return flattened


def circle_bounds(circles: list[ET.Element]) -> tuple[float, float, float, float]:
    """Axis-aligned bounds of a circle run, in user units."""
    xs_lo, xs_hi, ys_lo, ys_hi = [], [], [], []
    for circle in circles:
        cx = float(circle.get("cx", 0.0))
        cy = float(circle.get("cy", 0.0))
        r = float(circle.get("r", 0.0))
        xs_lo.append(cx - r)
        xs_hi.append(cx + r)
        ys_lo.append(cy - r)
        ys_hi.append(cy + r)
    return min(xs_lo), min(ys_lo), max(xs_hi), max(ys_hi)


def find_point_clouds(root: ET.Element, min_points: int) -> list[tuple[ET.Element, int, int]]:
    """Locate runs of >= min_points consecutive primitive siblings.

    Returns ``(parent, start_index, stop_index)`` half-open spans, latest first
    so callers can splice without invalidating earlier indices.
    """
    spans: list[tuple[ET.Element, int, int]] = []
    for parent in root.iter():
        children = list(parent)
        start = None
        for index, child in enumerate(children + [None]):
            is_primitive = child is not None and tag_of(child) in CLOUD_TAGS
            if is_primitive and start is None:
                start = index
            elif not is_primitive and start is not None:
                if index - start >= min_points:
                    spans.append((parent, start, index))
                start = None
    spans.sort(key=lambda span: (id(span[0]), span[1]), reverse=True)
    return spans


def group_clouds(root: ET.Element, min_points: int) -> int:
    """Wrap each dense run in one ``<g>`` so PowerPoint yields a single object."""
    spans = find_point_clouds(root, min_points)
    for number, (parent, start, stop) in enumerate(spans, 1):
        members = list(parent)[start:stop]
        cloud = ET.Element(f"{{{SVG_NS}}}g", {"id": f"points-{number}"})
        for member in members:
            parent.remove(member)
            cloud.append(member)
        parent.insert(start, cloud)
    return len(spans)


def raster_clouds(root: ET.Element, min_points: int, scale: int) -> int:
    """Replace each all-circle cloud with one embedded bitmap.

    Grouping alone still leaves PowerPoint thousands of shapes to lay out. A
    point cloud carries no per-point meaning, so a bitmap loses nothing an
    editor would want -- and it is the only way a 2,000-point panel stays
    responsive. Runs that are not purely circles are grouped instead, because
    their bounds cannot be derived without a full path parser.
    """
    import cairosvg  # optional; only needed for this mode

    replaced = 0
    for number, (parent, start, stop) in enumerate(find_point_clouds(root, min_points), 1):
        members = list(parent)[start:stop]
        if not all(tag_of(m) == "circle" for m in members):
            continue
        x0, y0, x1, y1 = circle_bounds(members)
        if x1 <= x0 or y1 <= y0:
            continue

        # Render the run alone, on the original canvas, then crop by viewBox so
        # the bitmap lands back at exactly the coordinates the circles occupied.
        isolated = ET.Element(f"{{{SVG_NS}}}svg", {
            "width": str(x1 - x0),
            "height": str(y1 - y0),
            "viewBox": f"{x0} {y0} {x1 - x0} {y1 - y0}",
        })
        for member in members:
            isolated.append(member)
        png = cairosvg.svg2png(
            bytestring=ET.tostring(isolated),
            output_width=int((x1 - x0) * scale),
            output_height=int((y1 - y0) * scale),
        )

        image = ET.Element(f"{{{SVG_NS}}}image", {
            "id": f"points-{number}",
            "x": str(x0), "y": str(y0),
            "width": str(x1 - x0), "height": str(y1 - y0),
            f"{{{XLINK_NS}}}href": "data:image/png;base64," + base64.b64encode(png).decode(),
        })
        for member in members:
            parent.remove(member)
        parent.insert(start, image)
        replaced += 1
    return replaced


def promote_outer_wrapper(root: ET.Element) -> int:
    """Lift the children of a lone top-level ``<g>`` up to the root.

    matplotlib wraps an entire figure in one ``<g id="figure_1">``. Left alone,
    the first Ungroup in PowerPoint yields a single object and the user has to
    ungroup again to reach anything. Promoting its children means one Ungroup
    hands over the panels themselves.
    """
    promoted = 0
    while True:
        wrappers = [c for c in root if tag_of(c) == "g"]
        others = [c for c in root
                  if tag_of(c) not in {"g", "defs", "metadata", "title", "desc"}]
        if len(wrappers) != 1 or others:
            return promoted
        wrapper = wrappers[0]
        # A wrapper carrying a transform cannot be dissolved without rewriting
        # every child's geometry, so stop rather than move the drawing.
        if "transform" in wrapper.attrib:
            return promoted
        index = list(root).index(wrapper)
        for offset, child in enumerate(list(wrapper)):
            root.insert(index + offset, child)
        root.remove(wrapper)
        promoted += 1


def top_level_shapes(root: ET.Element) -> int:
    """How many objects PowerPoint shows after one Ungroup."""
    return sum(1 for child in root if tag_of(child) not in {"defs", "metadata", "title", "desc"})


def total_shapes(root: ET.Element) -> int:
    """Every drawable element, i.e. the count after ungrouping all the way down."""
    skip = {"svg", "defs", "metadata", "title", "desc", "clipPath", "marker", "style"}
    return sum(1 for element in root.iter() if tag_of(element) not in skip)


def convert(source: Path, destination: Path, *, min_points: int, rasterize: bool,
            scale: int) -> None:
    ET.register_namespace("", SVG_NS)
    ET.register_namespace("xlink", XLINK_NS)
    tree = ET.parse(source)
    root = tree.getroot()

    before_total = total_shapes(root)
    embedded = sum(1 for e in root.iter() if tag_of(e) == "image")

    clips = strip_clip_paths(root)
    flattened = flatten_transparent_groups(root)
    promoted = promote_outer_wrapper(root)
    if rasterize:
        collapsed = raster_clouds(root, min_points, scale)
        mode = "rasterised"
        # Anything the raster pass skipped (non-circle runs) still needs grouping.
        collapsed += group_clouds(root, min_points)
    else:
        collapsed = group_clouds(root, min_points)
        mode = "grouped"

    tree.write(destination, encoding="utf-8", xml_declaration=True)

    print(f"{source.name} -> {destination.name}")
    print(f"  clip-paths removed      {clips}")
    print(f"  nesting groups removed  {flattened}")
    print(f"  outer wrappers opened   {promoted}")
    print(f"  point clouds {mode:<10} {collapsed}")
    if embedded:
        print(f"  pre-existing bitmaps    {embedded}  (stay bitmaps; rasterized=True at source)")
    print(f"  shapes: {before_total} -> {total_shapes(root)}"
          f"   |  after one Ungroup in PowerPoint: {top_level_shapes(root)} objects")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("svg", type=Path, nargs="+", help="SVG file(s) to convert")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="write <name>_ppt.svg here (default: alongside the source)")
    parser.add_argument("--min-points", type=int, default=40,
                        help="runs of at least this many primitives count as a point cloud")
    parser.add_argument("--raster-clouds", action="store_true",
                        help="collapse each circle cloud to one bitmap instead of one group")
    parser.add_argument("--scale", type=int, default=8,
                        help="bitmap oversampling for --raster-clouds (8 ~= 600 dpi)")
    args = parser.parse_args()

    for source in args.svg:
        if not source.exists():
            raise SystemExit(f"no such file: {source}")
        out_dir = args.out_dir or source.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        convert(source, out_dir / f"{source.stem}_ppt.svg",
                min_points=args.min_points, rasterize=args.raster_clouds, scale=args.scale)


if __name__ == "__main__":
    main()
