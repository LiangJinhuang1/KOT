"""Shared colour vocabulary for every figure in the project.

Three independent palettes, each bound to one kind of entity. A colour never
means two things: cell state, modality, and method are separate systems, so a
reader who learns one mapping in Fig. 1 can carry it through the whole paper.

All three are drawn from the Okabe-Ito colourblind-safe set or from published
CVD-safe pairs. There is deliberately no red/green opposition anywhere.
"""

import numpy as np

# --- Cell state (Okabe-Ito) ------------------------------------------------
# Blue vs vermillion is the canonical CVD-safe opposition; the previous
# Branch_B red / Trunk green pair was a deuteranopia failure.
STATE_COLORS = {
    "Progenitor": "#767676",   # neutral grey — the root, not a branch
    "Branch_A":   "#0072B2",   # blue
    "Branch_B":   "#D55E00",   # vermillion
    "Trunk":      "#009E73",   # bluish green
}

# --- Modality --------------------------------------------------------------
# A distinct hue family from STATE_COLORS so a co-embedding panel coloured by
# modality can never be misread against a panel coloured by state. Reinforced by
# marker shape (circle = RNA, triangle = second modality) wherever both appear.
# Both clear 4.5:1 on white, so they double as line and label colours.
MODALITY_COLORS = {
    "RNA":     "#5D3A9B",   # deep purple
    "Protein": "#9C6500",   # dark amber
    "ATAC":    "#9C6500",
}

MODALITY_MARKERS = {"RNA": "o", "Protein": "^", "ATAC": "^"}

# --- Method (hierarchical: family picks hue, member samples within) ---------
# KOT is the focal series and carries the only saturated blue; comparators sit
# at lower visual weight so the contrast reads without a legend lookup.
METHOD_COLORS = {
    # KOT family — blues, focal
    "kot":                          "#0072B2",
    "kot_anchor":                   "#3E92C8",
    "kot_noanchor":                 "#7BB6DC",
    "kot_nodyn":                    "#9ECAE1",
    # Fig. 5a/b only. Hue, not a blue ramp, carries identity: navy / vermillion /
    # bluish-green stays CVD-safe and clear of panel c's BMMC blue and PBMC amber.
    "kot_fixedkappa":               "#D55E00",
    "kot_fixedalpha":               "#009E73",
    "kot_oracle":                   "#08519C",
    "kot_oracle_learnalpha":        "#2171B5",
    "kot_oracle_learnalpha_kappa":  "#4292C6",
    # Optimal-transport baselines — vermillion family
    "scot":       "#D55E00",
    "moscot":     "#E8873A",
    # Deep-generative baselines — green family
    "glue":       "#009E73",
    "uniport":    "#4CB79A",
    "totalvi":    "#8CCFBC",
    # Linkage-driven matching — its own hue. MaxFuse starts from the SAME ADT→gene
    # links KOT's Sinkhorn term sees, so it is the one baseline that shares KOT's
    # prior; giving it a family of its own keeps that distinction visible.
    "maxfuse":    "#CC79A7",
    # Convex ODE baseline — neutral
    "linear_ode": "#767676",
}

# Human-readable method names. Internal codenames never reach an axis (§5.6).
METHOD_LABELS = {
    "kot":                         "KOT",
    "kot_anchor":                  "KOT + β-anchor",
    "kot_noanchor":                "KOT, no β-anchor",
    "kot_nodyn":                   r"KOT ($\lambda_{\mathrm{dyn}}\!=\!0$)",
    "kot_fixedkappa":              "KOT, κ fixed",
    "kot_fixedalpha":              "KOT, α fixed",
    "kot_oracle":                  "KOT, oracle kinetics",
    "kot_oracle_learnalpha":       "KOT, oracle κ",
    "kot_oracle_learnalpha_kappa": "KOT, oracle β",
    "scot":       "SCOT",
    "moscot":     "moscot",
    "glue":       "GLUE",
    "uniport":    "uniPort",
    "maxfuse":    "MaxFuse",
    "totalvi":    "totalVI (paired)",
    "linear_ode": "Linear ODE",
}

# Dataset display names. Internal keys never reach a panel title (§5.6).
# Datasets carry one shape and one colour in every figure, so a reader who learns the
# key in Fig. 1c can reuse it in Fig. 3, 4, 5 and 9. Blue is KOT's colour; amber stays
# separable from it down to 2-point markers, where a light-blue tint does not. RegVelo
# variants inherit their dataset's style -- the backend is an axis, not a dataset.
DATASET_COLORS = {
    "bmmc_cite_retained": "#0072B2",
    "bmmc_cite_regvelo":  "#0072B2",
    "pbmc_retained":      "#E69F00",
    "pbmc_regvelo":       "#E69F00",
}

DATASET_MARKERS = {
    "bmmc_cite_retained": "o",
    "bmmc_cite_regvelo":  "o",
    "pbmc_retained":      "s",
    "pbmc_regvelo":       "s",
}

DATASET_SHORT = {
    "bmmc_cite_retained": "BMMC",
    "bmmc_cite_regvelo":  "BMMC",
    "pbmc_retained":      "PBMC",
    "pbmc_regvelo":       "PBMC",
}

# The two reported panels are named by the short form everywhere, because figures used
# three spellings for the same dataset -- "BMMC", "BMMC CITE-seq" and "BMMC (n=5)" -- and
# Fig. S2 carried two of them in one figure. The RegVelo rows keep the long form: they
# share a short name with their retained counterpart by design (the backend is an axis,
# not a dataset), so shortening them would make the two indistinguishable wherever a
# figure draws both, which Fig. 4's backend panel does.
DATASET_LABELS = {
    "pbmc_retained":      "PBMC",
    "pbmc_regvelo":       "PBMC CITE-seq (RegVelo)",
    "bmmc_cite_retained": "BMMC",
    "bmmc_cite_regvelo":  "BMMC CITE-seq (RegVelo)",
    "papalexi_retained":  "Papalexi ECCITE-seq",
    "papalexi_regvelo":   "Papalexi ECCITE-seq (RegVelo)",
    "synthetic_linked_ode": "Synthetic linked ODE",
    "synthetic":            "Synthetic",
}


def dataset_label(name: str) -> str:
    return DATASET_LABELS.get(str(name), str(name).replace("_", " "))


FOCAL_METHOD = "kot"

# FOSCTTM of a random alignment. Every FOSCTTM axis is drawn against this.
FOSCTTM_CHANCE = 0.5

# Neutral fallback ramp for categories with no assigned colour. Okabe-Ito order,
# so an unplanned category is still CVD-safe.
FALLBACK_COLORS = [
    "#0072B2", "#D55E00", "#009E73", "#CC79A7",
    "#56B4E9", "#E69F00", "#767676", "#F0E442",
]

# Kept for backwards compatibility with older scripts.
TAB10_FALLBACK_COLORS = FALLBACK_COLORS


def state_color_map(states) -> dict[str, str]:
    unique = sorted({str(s) for s in np.unique(states)})
    colors: dict[str, str] = {}
    fallback_i = 0
    for state in unique:
        if state in STATE_COLORS:
            colors[state] = STATE_COLORS[state]
        else:
            colors[state] = FALLBACK_COLORS[fallback_i % len(FALLBACK_COLORS)]
            fallback_i += 1
    return colors


# Coarse haematopoietic lineages. The BMMC annotation has 45 cell types, far past
# what a legible panel can carry, and the fine distinctions (CD57+ vs TIGIT+ CD8 T)
# are not what an alignment figure is about. Order runs progenitor -> differentiated.
LINEAGE_ORDER = ["HSC / progenitor", "Erythroid", "Monocyte", "Dendritic",
                 "B lineage", "CD4 T", "CD8 T", "NK / ILC", "Other"]

LINEAGE_COLORS = {
    "HSC / progenitor": "#767676",
    "Erythroid":        "#D55E00",
    "Monocyte":         "#E69F00",
    # One Dendritic everywhere: Figs. 3, 6 and 8 used to disagree, because Fig. 8 darkened
    # this to ink(target=3.0) so the marker would not vanish as a 2-point dot on white.
    # Kept as Wong yellow for consistency; if a scatter needs more contrast, raise the
    # marker size rather than forking the palette.
    "Dendritic":        "#F0E442",
    "B lineage":        "#0072B2",
    "CD4 T":            "#56B4E9",
    "CD8 T":            "#009E73",
    "NK / ILC":         "#CC79A7",
    "Other":            "#BBBBBB",
}


def lineage(cell_type: str) -> str:
    """Map a fine-grained BMMC/PBMC cell-type label to a coarse lineage."""
    t = str(cell_type)
    tl = t.lower()
    if "prog" in tl or tl.startswith("hsc"):
        return "HSC / progenitor"
    if any(k in tl for k in ("erythro", "reticulocyte", "normoblast", "proerythro")):
        return "Erythroid"
    if "mono" in tl:
        return "Monocyte"
    if "dc" in tl.split() or tl.startswith(("pdc", "cdc")) or "dendritic" in tl:
        return "Dendritic"
    # After the dendritic test on purpose, so "Plasmacytoid DC" is a DC, not a
    # plasma cell. "plasma" also covers "plasmablast"; " b " (padded) covers the
    # bare-B labels like "B1 B" and "Naive CD20+ B".
    if (" b " in f" {tl} " or "cd20+ b" in tl
            or tl.startswith("transitional b") or "plasma" in tl):
        return "B lineage"
    if tl.startswith("cd4") or tl.startswith("t reg") or tl == "dnt":
        return "CD4 T"
    if tl.startswith("cd8") or tl.startswith("mait") or tl.startswith("gdt"):
        return "CD8 T"
    if tl.startswith("nk") or tl.startswith("ilc"):
        return "NK / ILC"
    return "Other"


def state_label(name: str) -> str:
    """Plain-language cell-state name. Internal codes never reach an axis (§5.6)."""
    return str(name).replace("_", " ")


def method_color(name: str) -> str:
    return METHOD_COLORS.get(str(name), "#767676")


def dataset_style(name: str) -> tuple[str, str, str]:
    """(marker, colour, short name) for a dataset, shared by every figure."""
    key = str(name)
    return (DATASET_MARKERS.get(key, "o"),
            DATASET_COLORS.get(key, METHOD_COLORS["kot"]),
            DATASET_SHORT.get(key, dataset_label(key)))


def method_label(name: str) -> str:
    return METHOD_LABELS.get(str(name), str(name))


def method_is_focal(name: str) -> bool:
    return str(name) == FOCAL_METHOD
