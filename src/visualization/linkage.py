"""How little linkage there actually is, and which term each surviving link feeds.

The method's premise is weak linkage between RNA and protein. That premise is a number,
not an adjective: on PBMC, 32 antibodies become 23 usable links and 13 velocity-backed
kinetic targets. Stating it before any score is what makes the rest of the paper legible.

TWO SETS THAT ARE NOT NESTED, and must not be drawn as one funnel:

  the funnel        total antibodies → mapped to a gene → curated as kinetics-eligible →
                    the gene present in RNA → surviving QC → the gene's velocity usable.
                    Each stage is a subset of the one before it.
  anchored          proteins with a literature half-life that the run actually anchors.
                    On BMMC that is 52 while velocity-backed is 28, so anchoring is a
                    different axis, not a later stage. Drawn beside the funnel, never
                    inside it.

Panel b reports what training did, not what the data could support, so every bar is a
runtime set:

  alignment_active              every protein feeding the OT term. Needs no gene link, so
                                on BMMC it is the whole 134-antibody panel.
  final_runtime_kinetic_mask    the kinetic mask a run used. LARGER than the velocity-
                                backed set, because `kot_kinetics_require_velocity_gene`
                                is false: proteins whose gene has no usable velocity still
                                enter the term (BMMC 101, of which only 28 are backed).
  strict_velocity_kinetic_mask  that mask intersected with usable velocity. Not the same
                                as `velocity_usable` (29), which counts good velocity
                                whether or not the run used the protein -- IgD has usable
                                velocity but never enters the mask.
  anchor_in_kinetic_mask        derived here, since the summary only carries the candidate
                                pool. Matches `beta_anchor_n` in every run's diagnostics.
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
    # "(any)" because panel b's velocity bar is this set intersected with the kinetic
    # mask, which is one protein smaller on BMMC (IgD). Without the qualifier the 29
    # here and the 28 there read as a contradiction.
    ("velocity_usable", "velocity usable (any)"),
]

# What each surviving link actually feeds. Not nested, so plotted as its own group.
# The repeated "Kinetic mask" prefix is deliberate: the velocity-backed bar is a subset of
# the used bar, and the counts alone (28 vs 101) do not say so.
# Wrapped, not abbreviated: at 45 degrees a one-line "Kinetic mask: velocity-backed"
# reaches far enough left to squeeze panel a until its count labels collide.
TERMS = [
    ("alignment_active", "Alignment:\nall proteins"),
    ("final_runtime_kinetic_mask", "Kinetic mask:\nused"),
    ("strict_velocity_kinetic_mask", "Kinetic mask:\nvelocity-backed"),
    ("anchor_in_kinetic_mask", "β anchor\ntargets"),
]


def read_coverage(datasets: list[str]) -> pd.DataFrame:
    """One row per dataset, plus the two runtime counts only the per-protein table holds.

    `alignment_active` the summary never carries, and `anchored` it carries as the
    candidate pool: every protein with a literature half-life, including ones outside the
    kinetic mask that are therefore never anchored (BMMC 62 candidates, 52 anchored). Both
    are derived from the per-protein table so panel b reports what training did.
    """
    frames = []
    for dataset in datasets:
        summary = pd.read_csv(MAPPING_DIR / f"kinetics_coverage_{dataset}_summary.csv")
        detail = pd.read_csv(MAPPING_DIR / f"kinetics_coverage_{dataset}.csv")
        if len(detail) != int(summary.loc[0, "total_adts"]):
            raise ValueError(
                f"{dataset}: per-protein table has {len(detail)} rows but the summary "
                f"counts {int(summary.loc[0, 'total_adts'])} antibodies; they are out of sync"
            )
        anchored_in_mask = int(
            (detail["anchor_available"] & detail["final_runtime_kinetic_mask"]).sum()
        )
        summary["alignment_active"] = int(detail["alignment_active"].sum())
        summary["anchor_in_kinetic_mask"] = anchored_in_mask
        frames.append(summary)
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
                    color="0.35", fontsize=6)
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
                    va="bottom", color="0.35", fontsize=6)
    ax.set_xticks([j + width / 2 for j in range(len(TERMS))])
    ax.set_xticklabels([label for _, label in TERMS], rotation=45, ha="right")
    # Headroom for the count above the tallest bar, which otherwise reaches the title.
    tallest = max(float(row[c]) for _, row in coverage.iterrows() for c, _ in TERMS)
    ax.set_ylim(0, tallest * 1.18)
    ax.set_ylabel("Proteins")


def dataset_colors(datasets: list[str], palette: list[str]) -> dict[str, str]:
    """A colour per dataset, checked against the dataset label table."""
    missing = [d for d in datasets if d not in DATASET_LABELS]
    if missing:
        raise KeyError(f"no display name for {missing}; add it to DATASET_LABELS")
    return dict(zip(datasets, palette))
