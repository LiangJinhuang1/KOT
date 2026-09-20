"""Predicted protein against measured protein — the comparison only phi(r) supports.

With ``kot_use_feature_space`` the RNA-side map lands in the ADT panel itself, so a
prediction can be read off directly instead of being imputed from neighbours in a latent
space. Every other method here would need a kNN step first, and the step, not the model,
would then be carrying part of the result.

WHICH ROWS COUNT. ``protein_eval_per_protein.csv`` pools an evaluation sweep: velocity
ablations, S permutations and the anchor-subset arms all wrote into it. Reading it
without a filter averages corrupted arms into the headline number, so `canonical_rows`
keeps only the configuration config/training.yaml declares and the runs that were not
part of the anchor-subset sweep (`beta_anchor_subset_n` blank), which used the config's
own anchors.

WHAT `in_kinetics` DOES NOT MEAN. Proteins the kinetics term covers score higher than
those it does not, but coverage is not random: a protein enters the kinetic mask only
when its gene has usable velocity, which already selects well-measured genes. The gap is
worth plotting and is not by itself evidence that the ODE caused it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy import stats

from src.visualization import METHOD_COLORS, dataset_style
from src.visualization.style import STYLE_STATE

RESULTS_DIR = Path("cache/results")

# The configuration config/training.yaml declares for the real datasets.
CANONICAL = {"lr_beta": 0.001, "lr_warmup_epochs": 300,
             "kot_velocity_ablation": "none", "kot_s_permute": False}

# Covered by the kinetics term, or reached by alignment alone.
COVERAGE_COLORS = {True: METHOD_COLORS["kot"], False: "#767676"}
COVERAGE_LABELS = {True: "In kinetics", False: "Not in kinetics"}

# Name the uncovered proteins only when the column is short enough to read.
NAME_UNCOVERED_MAX = 8

# Short panel name -> the dataset key the shared marker/colour registry uses.
DATASET_KEYS = {"BMMC": "bmmc_cite_retained", "PBMC": "pbmc_retained"}


def fig_dpi(ax) -> float:
    """Points-per-inch conversion needs the figure's own dpi, not the default."""
    return float(ax.figure.dpi)


def spread(disp: list, run: list, pitch: float, floor: float | None = None) -> list:
    """Target display-y for one cluster: evenly pitched, centred, lifted off the floor.

    A cluster sitting on the panel's floor -- the isotype controls do -- would otherwise
    have its lowest label centred below the bottom spine.
    """
    n = len(run)
    centre = sum(disp[a] for a in run) / n
    top = centre + pitch * (n - 1) / 2.0
    targets = [top - pitch * a for a in range(n)]
    if floor is not None and targets[-1] < floor:
        lift = floor - targets[-1]
        targets = [t + lift for t in targets]
    return targets


def swatch_handles(labels: dict, colours: dict, keys) -> list[Patch]:
    """Legend patches, so every key in this figure matches Fig. 8's swatches."""
    return [Patch(facecolor=colours[k], edgecolor="none", label=labels[k]) for k in keys]


def coverage_handles() -> list[Patch]:
    """The In / Not in kinetics key used by panels b and c."""
    return swatch_handles(COVERAGE_LABELS, COVERAGE_COLORS, (True, False))


def series_flag(series: pd.Series) -> pd.Series:
    """True/False from a CSV column that may already be bool or still a string."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    return series.astype(str).str.strip().str.lower().isin(("true", "1", "yes"))


def blank_subset(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip().str.lower()
    return series.isna() | text.isin(("", "nan", "none"))


def canonical_rows(dataset: str) -> pd.DataFrame:
    """Per-protein evaluation rows for one dataset, sweep arms excluded."""
    df = pd.read_csv(RESULTS_DIR / "protein_eval_per_protein.csv")
    keep = df["dataset"] == dataset
    for column, value in CANONICAL.items():
        if isinstance(value, bool):
            keep &= series_flag(df[column]) == value
        else:
            keep &= df[column].astype(type(value)) == value
    return df[keep & blank_subset(df["beta_anchor_subset_n"])].copy()


def per_protein_median(df: pd.DataFrame, metric: str = "spearman") -> pd.DataFrame:
    """One row per protein: median across seeds, the seed range, and its coverage."""
    grouped = df.groupby(["protein", "in_kinetics"])[metric]
    out = grouped.agg(["median", "min", "max", "count"]).reset_index()
    return out.sort_values("median", ascending=False).reset_index(drop=True)


def coverage_strip_panel(ax, frames: dict[str, pd.DataFrame], metric: str = "spearman"):
    """Per-protein score split by kinetics coverage, one column pair per dataset.

    Points are per-protein medians over seeds, not per-seed values: a protein measured
    twelve times is one protein, and plotting every seed would let a well-covered panel
    outvote a sparse one twelve to one.
    """
    positions, labels = [], []
    for i, (name, df) in enumerate(frames.items()):
        per_protein = per_protein_median(df, metric)
        for j, covered in enumerate((True, False)):
            rows = per_protein[per_protein["in_kinetics"] == covered]
            values = rows["median"]
            x = 2 * i + j
            jitter = np.random.default_rng(0).normal(0, 0.06, len(values))
            ax.scatter(x + jitter, values, s=3.5, alpha=0.55, linewidths=0,
                       color=COVERAGE_COLORS[covered], zorder=3)
            ax.plot([x - 0.28, x + 0.28], [values.median()] * 2, color="#1A1A1A",
                    lw=1.0, zorder=4)
            # Name the uncovered proteins when there are few enough to name. On PBMC
            # that group is six: three real markers at the top and three IgG isotype
            # controls at the bottom, so the median bar between them is an artefact of
            # mixing negative controls into a comparison group. Naming them is the only
            # way the panel shows why its own bar cannot be read.
            if not covered and len(rows) <= NAME_UNCOVERED_MAX:
                # Space the names evenly in DISPLAY units, not data units: the three
                # isotype controls sit within 0.04 of each other, so any offset derived
                # from their own heights still overlaps. Points close enough to collide
                # form a cluster whose labels are dealt out at a fixed pitch, each with a
                # leader line back to its own point. Naming them is the only way the
                # panel shows why its own median bar cannot be read -- half this group
                # is negative controls.
                order = np.argsort(-values.to_numpy())
                ys = [float(values.iloc[k]) for k in order]
                pitch = 8.0
                to_pt = lambda v: ax.transData.transform((0.0, v))[1] * 72.0 / fig_dpi(ax)
                disp = [to_pt(v) for v in ys]
                floor = to_pt(ax.get_ylim()[0]) + pitch
                targets, run = [], [0]
                for a in range(1, len(disp)):
                    if disp[run[-1]] - disp[a] < pitch:
                        run.append(a)
                    else:
                        targets.extend(spread(disp, run, pitch, floor))
                        run = [a]
                targets.extend(spread(disp, run, pitch, floor))
                for a, k in enumerate(order):
                    dy = targets[a] - disp[a]
                    ax.annotate(clean_protein(rows["protein"].iloc[k]),
                                xy=(x + jitter[k], ys[a]),
                                xytext=(4, dy), textcoords="offset points",
                                fontsize=STYLE_STATE["ladder"][2], color="0.35",
                                ha="left", va="center", annotation_clip=False,
                                arrowprops=None if abs(dy) < 1 else dict(
                                    arrowstyle="-", lw=0.4, color="0.75",
                                    shrinkA=0, shrinkB=1.5))
            positions.append(x)
            labels.append(COVERAGE_LABELS[covered])
        ax.text(2 * i + 0.5, 1.0, name, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlim(-0.6, positions[-1] + 0.6)


def rank_fraction(n: int) -> np.ndarray:
    """Rank as a fraction of its own panel, so panels of different size share an axis."""
    return np.arange(n) / max(n - 1, 1)


def ranked_protein_panel(ax, frames: dict, metric: str = "spearman"):
    """Every protein in rank order per dataset, seed range as a whisker.

    Both datasets share the panel. They hold very different protein counts (BMMC 134,
    PBMC 32), so x is the rank as a FRACTION of each panel rather than an index; colour
    still carries kinetics coverage and shape carries the dataset, as in Fig. 3a.

    Uncovered proteins are named when there are few enough to name -- on PBMC that is
    six, half of them isotype controls, which is the panel's whole point. BMMC's 26 are
    left unnamed rather than crowding the panel.
    """
    out = {}
    for name, df in frames.items():
        per_protein = per_protein_median(df, metric)
        x = rank_fraction(len(per_protein))
        marker, _, _ = dataset_style(DATASET_KEYS[name])
        for covered in (True, False):
            mask = (per_protein["in_kinetics"] == covered).to_numpy()
            if not mask.any():
                continue
            ax.vlines(x[mask], per_protein["min"][mask], per_protein["max"][mask],
                      color=COVERAGE_COLORS[covered], lw=0.6, alpha=0.4, zorder=2)
            ax.scatter(x[mask], per_protein["median"][mask], s=5, marker=marker,
                       color=COVERAGE_COLORS[covered], linewidths=0, zorder=3)
        uncovered = [i for i in range(len(x)) if not per_protein["in_kinetics"][i]]
        if 0 < len(uncovered) <= NAME_UNCOVERED_MAX:
            half = len(x) / 2
            high = [i for i in uncovered if i < half]
            low = [i for i in uncovered if i >= half]
            named = ([(i, "left", 0.30, 0.96 - 0.10 * k) for k, i in enumerate(high)]
                     + [(i, "right", 0.98, 0.22 + 0.11 * k) for k, i in enumerate(low)])
            for i, ha, fx, fy in named:
                ax.annotate(clean_protein(per_protein["protein"][i]),
                            xy=(x[i], per_protein["median"][i]), xycoords="data",
                            xytext=(fx, fy), textcoords="axes fraction",
                            color="0.35", ha=ha, va="center", fontsize=6,
                            arrowprops=dict(arrowstyle="-", lw=0.5, color="0.7",
                                            shrinkA=0, shrinkB=1))
        out[name] = per_protein
    ax.set_xlim(-0.04, 1.04)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xticklabels(["best", "", "worst"])
    return out


def clean_protein(name: str) -> str:
    """The marker name without its assay suffix — 'CD4_TotalSeqB' is not a marker."""
    return str(name).split("_TotalSeq")[0].replace("_control", "")


def pooled_calibration_panel(ax, predicted: np.ndarray, observed: np.ndarray, *,
                             gridsize: int = 45, cmap: str = "Greys"):
    """Every cell and every protein at once, each protein standardised first.

    Pooling raw values would let the widest-ranging markers set the axes and smear the
    rest into the origin; standardising per protein asks the question the panel is for,
    which is whether the shape is right, not whether two panels share a scale.
    """
    pred = stats.zscore(predicted, axis=0)
    obs = stats.zscore(observed, axis=0)
    ax.hexbin(obs.ravel(), pred.ravel(), gridsize=gridsize, cmap=cmap, bins="log",
              mincnt=1, linewidths=0, rasterized=True)
    lim = float(np.percentile(np.abs(np.concatenate([obs.ravel(), pred.ravel()])), 99.5))
    ax.plot([-lim, lim], [-lim, lim], color="0.45", lw=0.7, ls=(0, (4, 2)), zorder=4)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    rho = stats.spearmanr(obs.ravel(), pred.ravel()).statistic
    ax.text(0.03, 0.97, f"ρ = {rho:.2f}", transform=ax.transAxes, ha="left", va="top")
    return rho


# Lineage markers named before looking at test Spearman. A missing name is skipped,
# not replaced by whichever protein scored best.
PRESET_MARKERS = ("CD4", "CD14", "CD19")

CONTROL_ARMS = ("shuffle", "reverse", "zero", "permS")
# Named as Tables 3 and 4 name the same arms, so the figure and the tables share one
# vocabulary. `nodyn` is "No dynamics" here and in Fig. 5; Figs. 3 and 8 label that same
# arm "KOT (lambda_dyn=0)" because there it is a METHOD in a method list, not a velocity
# input in an ablation ladder.
ARM_LABELS = {
    "none": "Real velocity",
    "nodyn": "No dynamics",
    "shuffle": "Shuffled velocity",
    "reverse": r"Reverse $v\to-v$",
    "zero": r"Zero $v\to 0$",
    "permS": r"Permuted mapping $S$",
}
NODYN_RUNS = {
    "bmmc_cite_retained": Path(
        "cache/training/run_20260829_042713_nodyn_bmmc_scv_cfgB_warm300_beta1p0e-3"),
    "pbmc_retained": Path(
        "cache/training/run_20260829_042713_nodyn_pbmc_scv_cfgB_warm300_beta1p0e-3"),
}
NODYN_EVAL_CSV = RESULTS_DIR / "protein_eval_nodyn_per_protein.csv"


def control_rows(dataset: str, arm: str) -> pd.DataFrame:
    """Canonical hyper-parameters, one velocity/link control arm."""
    df = pd.read_csv(RESULTS_DIR / "protein_eval_per_protein.csv")
    keep = (df["dataset"] == dataset) & (df["lr_beta"].astype(float) == CANONICAL["lr_beta"])
    keep &= df["lr_warmup_epochs"].astype(int) == CANONICAL["lr_warmup_epochs"]
    keep &= blank_subset(df["beta_anchor_subset_n"])
    permuted = series_flag(df["kot_s_permute"])
    if arm == "permS":
        keep &= permuted
        keep &= df["kot_velocity_ablation"].astype(str) == "none"
    else:
        keep &= ~permuted
        keep &= df["kot_velocity_ablation"].astype(str) == arm
    return df[keep].copy()


def paired_protein_delta(reference: pd.DataFrame, other: pd.DataFrame,
                         metric: str = "spearman") -> pd.DataFrame:
    """Per-protein, per-seed difference; incomplete pairs are an error."""
    keys = ["dataset", "seed", "protein"]
    left = reference[keys + [metric, "in_kinetics"]].copy()
    right = other[keys + [metric]].copy()
    if left.duplicated(keys).any() or right.duplicated(keys).any():
        raise ValueError("Duplicate dataset/seed/protein rows would bias paired deltas")
    merged = left.merge(right, on=keys, how="inner", suffixes=("_ref", "_other"),
                        validate="one_to_one")
    ref_keys = set(map(tuple, left[keys].to_numpy()))
    other_keys = set(map(tuple, right[keys].to_numpy()))
    if ref_keys != other_keys:
        raise ValueError("Prediction arms do not share the same proteins and seeds")
    merged["delta"] = merged[f"{metric}_ref"] - merged[f"{metric}_other"]
    return merged


def marker_name_matches(protein: str, marker: str) -> bool:
    return clean_protein(protein).split("-")[0] == marker


def preset_proteins(names) -> list[str]:
    """The preselected markers that actually exist in this panel, in declared order."""
    available = list(names)
    found = []
    for marker in PRESET_MARKERS:
        hits = [name for name in available if marker_name_matches(name, marker)]
        if len(hits) == 1:
            found.append(hits[0])
        elif len(hits) > 1:
            exact = [name for name in hits if clean_protein(name) == marker]
            found.append(exact[0] if exact else hits[0])
    return found


def paired_delta_panel(ax, frames: dict):
    """One point per protein: median KOT − control Spearman over seeds, both datasets.

    Fractional rank for the same reason as `ranked_protein_panel`: the two panels hold
    134 and 32 proteins and still have to share one axis.
    """
    out = {}
    for name, deltas in frames.items():
        per_protein = (deltas.groupby(["protein", "in_kinetics"])["delta"]
                       .median().reset_index()
                       .sort_values("delta", ascending=False).reset_index(drop=True))
        x = rank_fraction(len(per_protein))
        marker, _, _ = dataset_style(DATASET_KEYS[name])
        for covered in (True, False):
            mask = (per_protein["in_kinetics"] == covered).to_numpy()
            if mask.any():
                ax.scatter(x[mask], per_protein["delta"][mask], s=5, marker=marker,
                           color=COVERAGE_COLORS[covered], linewidths=0, zorder=3)
        out[name] = per_protein
    ax.axhline(0, color="0.45", lw=0.7, ls=(0, (4, 2)), zorder=1)
    ax.set_xlim(-0.04, 1.04)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xticklabels(["best", "", "worst"])
    return out


def marker_arm_panel(ax, frames: dict[str, pd.DataFrame], proteins: list[str],
                     arms: list[str]):
    """Preselected markers, one cluster per marker, arms as coloured ticks."""
    colours = {
        "none": METHOD_COLORS["kot"],
        "nodyn": METHOD_COLORS["kot_nodyn"],
        "shuffle": "#D55E00",
        "reverse": "#E69F00",
        "zero": "#767676",
        "permS": "#CC79A7",
    }
    width = 0.14
    lowest = 0.0
    for i, protein in enumerate(proteins):
        for j, arm in enumerate(arms):
            df = frames.get(arm)
            if df is None or df.empty:
                continue
            values = df.loc[df["protein"] == protein, "spearman"]
            if values.empty:
                continue
            lowest = min(lowest, float(values.min()), float(values.median()))
            x = i + (j - (len(arms) - 1) / 2) * width
            ax.scatter(np.full(len(values), x), values, s=6, color=colours[arm],
                       alpha=0.55, linewidths=0, zorder=3)
            ax.plot([x - 0.05, x + 0.05], [values.median()] * 2, color="#1A1A1A",
                    lw=0.9, zorder=4)
    ax.set_xticks(range(len(proteins)))
    ax.set_xticklabels([clean_protein(name) for name in proteins])
    # Lower limit from the data, not a fixed -0.05: at that floor the two arms whose
    # CD4 median is NEGATIVE -- no dynamics at -0.086 and permuted mapping at -0.058 --
    # had their median bars clipped off the axis, hiding the panel's sharpest result,
    # along with thirty points reaching -0.54.
    ax.axhline(0, color="0.45", lw=0.7, ls=(0, (4, 2)), zorder=1)
    ax.set_ylim(min(-0.05, lowest - 0.06), 1.05)
    handles = swatch_handles(ARM_LABELS, colours,
                             [arm for arm in arms if arm in frames])
    if handles:
        # One column, upper right: six arms across two columns read as three pairs.
        ax.legend(handles=handles, loc="upper right", frameon=False, ncols=1,
                  fontsize=6, labelspacing=0.25)
    return proteins

