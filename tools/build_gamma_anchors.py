#!/usr/bin/env python3
"""Build a gamma-anchor CSV of measured mRNA decay rates for the regulatory R2 law.

gamma is spliced-mRNA decay; protein half-lives that anchor beta measure a different molecule.
K562 is the default because gene symbols join the RNA panel without orthologs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import urllib.request
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd

SUPPLEMENTARY_URL = (
    "https://media.springernature.com/original/springer-static/esm/"
    "art%3A10.1038%2Fnmeth.4582/MediaObjects/41592_2018_BFnmeth4582_MOESM5_ESM.xlsx"
)
SOURCE = "Schofield2018_NatMethods_TimeLapseSeq_SuppTable2"
K562_SHEET = "Table S2_K562"
SHEET_ML = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
SHEET_REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
PACKAGE_REL = "{http://schemas.openxmlformats.org/package/2006/relationships}"
LN2 = math.log(2.0)


def read_xlsx_sheet(path: Path, sheet: str) -> pd.DataFrame:
    """Read one named worksheet using only the standard library.

    The container has no openpyxl; resolve through workbook relationships so the wrong species sheet cannot be read silently.
    """
    with zipfile.ZipFile(path) as archive:
        relations = {
            node.get("Id"): node.get("Target").lstrip("/")
            for node in ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
            .iter(f"{PACKAGE_REL}Relationship")
        }
        targets = {
            node.get("name"): relations[node.get(f"{SHEET_REL}id")]
            for node in ET.fromstring(archive.read("xl/workbook.xml")).iter(f"{SHEET_ML}sheet")
        }
        if sheet not in targets:
            raise ValueError(f"worksheet {sheet!r} absent; workbook has {sorted(targets)}")
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            strings = [
                "".join(text.text or "" for text in item.iter(f"{SHEET_ML}t"))
                for item in ET.fromstring(archive.read("xl/sharedStrings.xml")).iter(f"{SHEET_ML}si")
            ]
        target = targets[sheet]
        member = target if target.startswith("xl/") else f"xl/{target}"
        rows = [parse_row(row, strings)
                for row in ET.fromstring(archive.read(member)).iter(f"{SHEET_ML}row")]
    if not rows:
        raise ValueError(f"worksheet {sheet!r} is empty")
    header, *body = rows
    columns = {column: name for column, name in header.items()}
    return pd.DataFrame([{columns[c]: v for c, v in row.items() if c in columns}
                         for row in body])


def parse_row(row: ET.Element, strings: list[str]) -> dict[str, str]:
    """Cell values of one row, keyed by column letter, with shared strings resolved."""
    values = {}
    for cell in row.iter(f"{SHEET_ML}c"):
        column = re.match(r"[A-Z]+", cell.get("r") or "A1").group(0)
        value = cell.find(f"{SHEET_ML}v")
        if cell.get("t") == "s" and value is not None:
            values[column] = strings[int(value.text)]
        elif cell.get("t") == "inlineStr":
            values[column] = "".join(text.text or "" for text in cell.iter(f"{SHEET_ML}t"))
        elif value is not None:
            values[column] = value.text
    return values


def fetch_supplementary(path: Path, url: str) -> Path:
    """Download the supplementary workbook once and keep it for provenance."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[anchors] downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as response:
        payload = response.read()
    if not payload.startswith(b"PK"):
        raise ValueError(f"{url} did not return an xlsx (got {payload[:16]!r})")
    path.write_bytes(payload)
    return path


def build_anchors(frame: pd.DataFrame, sheet: str,
                  max_replicate_log_ratio: float) -> pd.DataFrame:
    """One row per gene: measured decay rate plus a replicate-agreement weight.

    Weight is exp(-|log(t1/t2)|), the same log-rate metric the loss scores residuals in.
    """
    required = {"transcript", "half_life_rep1", "half_life_rep2", "mean_half_life"}
    if not required.issubset(frame.columns):
        raise ValueError(f"sheet {sheet!r} must carry {sorted(required)}; has {list(frame.columns)}")
    if not np.isfinite(max_replicate_log_ratio) or max_replicate_log_ratio < 0:
        raise ValueError("Replicate log-ratio threshold must be finite and nonnegative")
    frame = frame.dropna(subset=sorted(required))
    symbols = frame["transcript"].astype(str).str.strip()
    # Date-formatted gene names stored as serials.
    ambiguous = symbols.str.fullmatch(r"\d+(?:\.\d+)?") | symbols.eq("")
    if ambiguous.any():
        print(f"[anchors] excluding ambiguous gene symbols: {symbols[ambiguous].tolist()}")
    frame = frame.loc[~ambiguous].assign(transcript=symbols[~ambiguous])
    half_lives = {name: pd.to_numeric(frame[name], errors="raise").to_numpy(float)
                  for name in ("half_life_rep1", "half_life_rep2", "mean_half_life")}
    finite = np.ones(len(frame), dtype=bool)
    for name, values in half_lives.items():
        if not np.isfinite(values).all():
            raise ValueError(f"{name} in {sheet!r} holds non-finite half-lives")
        finite &= values > 0
    if not finite.all():
        print(f"[anchors] dropping {int((~finite).sum())} transcripts with a non-positive half-life")
    disagreement = np.full(len(frame), np.inf)
    disagreement[finite] = np.abs(np.log(
        half_lives["half_life_rep1"][finite] / half_lives["half_life_rep2"][finite]))
    keep = finite & (disagreement <= max_replicate_log_ratio)
    dropped = int((finite & ~keep).sum())
    if dropped:
        print(f"[anchors] dropping {dropped} transcripts whose replicates disagree by more "
              f"than {math.exp(max_replicate_log_ratio):.2g}x")
    symbols = frame["transcript"].astype(str).str.strip().to_numpy()
    anchors = pd.DataFrame({
        "gene_symbol": symbols[keep],
        "molecule": "RNA",
        "cell_line": sheet.split("_")[-1],
        "half_life_hours": half_lives["mean_half_life"][keep],
        "gamma_per_hour": LN2 / half_lives["mean_half_life"][keep],
        "half_life_rep1_hours": half_lives["half_life_rep1"][keep],
        "half_life_rep2_hours": half_lives["half_life_rep2"][keep],
        "replicate_log_ratio": disagreement[keep],
        "anchor_weight": np.exp(-disagreement[keep]),
        "source": SOURCE,
    })
    # Collapse repeats here: the loader refuses repeated symbols rather than silently averaging them.
    duplicated = anchors["gene_symbol"].duplicated(keep=False)
    if duplicated.any():
        print(f"[anchors] averaging {int(duplicated.sum())} rows over "
              f"{anchors.loc[duplicated, 'gene_symbol'].nunique()} repeated symbols")
        numeric = [c for c in anchors.columns if anchors[c].dtype.kind == "f"]
        constant = {c: "first" for c in anchors.columns
                    if c not in numeric and c != "gene_symbol"}
        anchors = (anchors.groupby("gene_symbol", as_index=False)
                   .agg({**{c: "mean" for c in numeric}, **constant}))
        anchors["gamma_per_hour"] = LN2 / anchors["half_life_hours"]
    if anchors.empty:
        raise ValueError("No usable RNA-decay anchors remain after filtering")
    return anchors.sort_values("gene_symbol", ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="Output gamma-anchor CSV")
    parser.add_argument("--xlsx", default="Datasets/RNA_halflife/nmeth4582_supp_table2.xlsx",
                        help="Local copy of Supplementary Table 2; downloaded if absent")
    parser.add_argument("--url", default=SUPPLEMENTARY_URL)
    parser.add_argument("--sheet", default=K562_SHEET,
                        help="Worksheet to read; the MEF sheet is mouse and needs orthologs")
    parser.add_argument("--max-replicate-log-ratio", type=float, default=LN2,
                        help="Drop genes whose two replicate half-lives disagree by more "
                             "than exp(this); default ln2, i.e. 2x")
    args = parser.parse_args()

    workbook = fetch_supplementary(Path(args.xlsx), args.url)
    frame = read_xlsx_sheet(workbook, args.sheet)
    anchors = build_anchors(frame, args.sheet, args.max_replicate_log_ratio)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    anchors.to_csv(out, index=False)
    out.with_suffix(".json").write_text(json.dumps({
        "source": SOURCE, "doi": "10.1038/nmeth.4582", "url": args.url,
        "sheet": args.sheet, "workbook_sha256": hashlib.sha256(workbook.read_bytes()).hexdigest(),
        "csv_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "source_rows": len(frame), "anchor_genes": len(anchors),
        "max_replicate_log_ratio": args.max_replicate_log_ratio,
        "ambiguous_symbols": frame.loc[
            frame["transcript"].astype(str).str.strip().str.fullmatch(r"\d+(?:\.\d+)?"),
            "transcript"].tolist(),
        "duplicate_policy": "arithmetic mean half-life, then gamma = ln(2) / half-life",
    }, indent=2) + "\n")
    rates = anchors["gamma_per_hour"].to_numpy()
    print(f"[anchors] wrote {len(anchors)} RNA-decay anchors -> {out}")
    print(f"[anchors] half-life hours: min {anchors['half_life_hours'].min():.3g} "
          f"median {anchors['half_life_hours'].median():.3g} "
          f"max {anchors['half_life_hours'].max():.3g}")
    print(f"[anchors] gamma/hour geometric mean {math.exp(np.log(rates).mean()):.4g}, "
          f"weight median {anchors['anchor_weight'].median():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
