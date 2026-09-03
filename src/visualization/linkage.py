"""How little linkage there actually is, and which term each surviving link feeds.

The method's premise is weak linkage between RNA and protein. That premise is a number,
not an adjective: on PBMC, 32 antibodies become 23 usable links and 13 velocity-backed
kinetic targets. Stating it before any score is what makes the rest of the paper legible.

TWO SETS THAT ARE NOT NESTED, and must not be drawn as one funnel:

  the funnel        total antibodies → mapped to a gene → curated as kinetics-eligible →
                    the gene present in RNA → surviving QC → the gene's velocity usable.
                    Each stage is a subset of the one before it.
  anchored          proteins with a literature half-life. On BMMC that is 62 while
                    velocity-backed is 29, so anchoring is a different axis, not a later
                    stage. Drawn beside the funnel, never inside it.

`final_runtime_kinetic_mask` is the mask a run actually used and is LARGER than the
velocity-backed set, because `kot_kinetics_require_velocity_gene` is false: proteins whose
gene has no usable velocity still enter the term. Both numbers are shown, since reporting
only the strict one would overstate how much of the kinetics term is velocity-backed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.visualization import DATASET_LABELS, dataset_label

MAPPING_DIR = Path("cache/results/mapping")

# Nested stages, in order. Labels are what the stage means, not the column name.
FUNNEL = [
    ("total_adts", "antibodies"),
    ("mapped_to_hgnc", "mapped to a gene"),
    ("curated_kinetic_eligible", "kinetics-eligible"),
    ("present_in_rna", "gene in RNA"),
    ("retained_after_qc", "survives QC"),
    ("velocity_usable", "velocity usable"),
]

# What each surviving link actually feeds. Not nested, so plotted as its own group.
TERMS = [
    ("present_in_rna", "alignment"),
    ("final_runtime_kinetic_mask", "kinetics (run)"),
    ("velocity_usable", "kinetics (vel.)"),
    ("anchored", "β anchored"),
]


def read_coverage(datasets: list[str]) -> pd.DataFrame:
    """One row per dataset from the per-dataset coverage summaries."""
    frames = [pd.read_csv(MAPPING_DIR / f"kinetics_coverage_{d}_summary.csv")
              for d in datasets]
    return pd.concat(frames, ignore_index=True)


def funnel_panel(ax, coverage: pd.DataFrame, colors: dict[str, str]):
    """Stages as a share of each dataset's own panel, absolute counts annotated.

    Shares rather than counts: BMMC starts from 134 antibodies and PBMC from 32, so on a
    shared count axis the PBMC funnel is a stub and the shape — the point of the panel —
    is unreadable.
    """
    height = 0.8 / len(coverage)
    for i, (_, row) in enumerate(coverage.iterrows()):
        total = float(row[FUNNEL[0][0]])
        counts = [float(row[c]) for c, _ in FUNNEL]
        y = [len(FUNNEL) - 1 - j - i * height for j in range(len(FUNNEL))]
        ax.barh(y, [c / total for c in counts], height=height,
                color=colors[row["dataset"]], zorder=3)
        for yy, count in zip(y, counts):
            ax.text(count / total + 0.015, yy, f"{count:.0f}", va="center",
                    color="0.35")
    ax.set_yticks([len(FUNNEL) - 1 - j - height / 2 for j in range(len(FUNNEL))])
    ax.set_yticklabels([label for _, label in FUNNEL])
    ax.set_xlim(0, 1.18)
    ax.set_xlabel("Share of the antibody panel")


def terms_panel(ax, coverage: pd.DataFrame, colors: dict[str, str]):
    """Absolute counts feeding each term, one bar group per dataset."""
    width = 0.8 / len(coverage)
    for i, (_, row) in enumerate(coverage.iterrows()):
        x = [j + i * width for j in range(len(TERMS))]
        ax.bar(x, [float(row[c]) for c, _ in TERMS], width=width,
               color=colors[row["dataset"]], label=dataset_label(row["dataset"]),
               zorder=3)
        for xx, (column, _) in zip(x, TERMS):
            ax.text(xx, float(row[column]), f"{row[column]:.0f}", ha="center",
                    va="bottom", color="0.35")
    ax.set_xticks([j + width / 2 for j in range(len(TERMS))])
    ax.set_xticklabels([label for _, label in TERMS], rotation=45, ha="right")
    ax.set_ylabel("Proteins")


def dataset_colors(datasets: list[str], palette: list[str]) -> dict[str, str]:
    """A colour per dataset, checked against the dataset label table."""
    missing = [d for d in datasets if d not in DATASET_LABELS]
    if missing:
        raise KeyError(f"no display name for {missing}; add it to DATASET_LABELS")
    return dict(zip(datasets, palette))
