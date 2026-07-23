#!/usr/bin/env python3
"""
KOT training instability analysis, grouped by experiment type.

Output structure (under --output, default ./analysis/):
  clean/           — oracle/clean/branch synthetic stages (one folder per stage)
  oracle/
  branch/
  linked_ode/      — synthetic_linked_ode + mean-scale layers
  meanscale/       — mean-scale coordinates (non-linked-ode)
  log_velocity/    — log-velocity layer + PCA representation
  phased/          — alternating-phase optimisation (phase1_epochs > 0)
  all/             — every run combined

  Per group:
    foscttm_distribution.png
    per_seed_reliability.png
    model_comparison.png
    loss_trajectories.png
    time_vs_alignment.png
    time_model_comparison.png
    summary_per_run.csv
    long_form_results.csv
    stuck_signatures.csv
    metrics_per_seed.csv          — every seed / single-run baseline (+ stage, JVP)
    foscttm_aggregate_by_stage.csv
    jvp_aggregate_by_stage.csv
    foscttm_aggregate_wide.csv    — model × clean/oracle/branch (mean ± std)
    jvp_aggregate_wide.csv

Usage:
    python analyze_instability.py
    python analyze_instability.py --output analysis/
    python analyze_instability.py --only linked_ode all
    python analyze_instability.py --runs run_20260622_234003
    python analyze_instability.py --runs run_20260622_234003 --per-run
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml


# ---------------------------------------------------------------------------
# Defaults / style
# ---------------------------------------------------------------------------

DEFAULT_BASE   = Path("cache/training")
DEFAULT_OUTPUT = Path("analysis")
THRESHOLD_GOOD = 0.1   # FOSCTTM < this counts as a "good" seed

MODEL_COLOR = {
    "kot":            "#1f77b4",
    "kot_nodyn":      "#ff7f0e",
    "kot_fixedkappa": "#2ca02c",
    "kot_fixedalpha": "#bcbd22",
    "kot_oracle":     "#7f7f7f",
    "kot_unbounded":    "#1f77b4",
    "kot_unbounded_sn": "#17becf",
    "kot_srinit":     "#9467bd",
    "kot_srinit_delayed": "#8c564b",
    "kot_anchor":     "#d62728",
}

MODEL_ORDER = [
    "kot",
    "kot_unbounded",
    "kot_unbounded_sn",
    "kot_nodyn",
    "kot_fixedkappa",
    "kot_fixedalpha",
    "kot_oracle",
    "kot_srinit",
    "kot_srinit_delayed",
    "kot_anchor",
]

STUCK_FOSCTTM_THRESHOLD = 0.3

SKIP_RUN_CHILDREN = {"cache", "results", "moscot_sweep"}

DIAGNOSTIC_KEYS = [
    "time_mae", "time_spearman", "phi_variance_ratio",
    "jvp_rhs_cos", "jvp_norm", "rhs_norm",
    "kappa_mean", "kappa_std",
    "beta_mean", "beta_std",
    "best_align_epoch", "best_align",
    "best_dyn_epoch", "best_dyn",
    "best_total_epoch", "best_total",
    "stop_epoch",
]

STAGE_ORDER = ["clean", "oracle", "branch"]

# Fallback when a run folder has no stage= line in its logs (common for older jobs).
KNOWN_RUN_STAGES: dict[str, str] = {
    "run_20260701_141322": "clean",
    "run_20260701_171218": "clean",
    "run_20260701_183237": "clean",
    "run_20260701_185533": "oracle",
    "run_20260701_200635_moscot_sweep": "clean",
    "run_oracle_kotoracles_mean_20260701_205724": "oracle",
    "run_oracle_mean_moscot_sweep": "oracle",
    "run_oracle_baselines_mean": "oracle",
    "run_branch_mean_20260701_190000": "branch",
    "run_branch_mean_20260701_205803": "branch",
}

plt.rcParams.update({
    "figure.figsize":      (10, 6),
    "figure.dpi":          110,
    "savefig.bbox":        "tight",
    "savefig.dpi":         150,
    "font.size":           10,
    "axes.titlesize":      11,
    "axes.labelsize":      10,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "legend.frameon":      False,
})


def kot_loss_models(df: pd.DataFrame) -> list[str]:
    present = {str(model) for model in df.model.dropna().unique() if str(model).startswith("kot")}
    ordered = [model for model in MODEL_ORDER if model in present]
    extras = sorted(model for model in present if model not in MODEL_ORDER)
    return ordered + extras


def is_kot_model(model: str) -> bool:
    return str(model).startswith("kot")


def models_for_plot(df: pd.DataFrame) -> list[str]:
    present = [str(model) for model in df.model.dropna().unique()]
    ordered = [model for model in MODEL_ORDER if model in present]
    extras = sorted(model for model in present if model not in MODEL_ORDER)
    return ordered + extras


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def active_dataset_block(datasets: dict) -> dict:
    """Return the first active dataset block from training.yaml."""
    if not datasets:
        return {}
    syn = datasets.get("synthetic")
    if isinstance(syn, dict) and syn:
        return syn
    for block in datasets.values():
        if isinstance(block, dict):
            return block
    return {}


def iter_model_outputs(model_dir: Path):
    """
    Yield (dataset, output_dir, summary_path) for each finished model output.

    output_dir is the seed_* folder when present, else the dataset folder for
    single-run baselines (scot, moscot, linear_ode).
    """
    for ds_dir in sorted(p for p in model_dir.iterdir() if p.is_dir() and p.name != "cache"):
        seed_dirs = sorted(
            p for p in ds_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")
        )
        if seed_dirs:
            for seed_dir in seed_dirs:
                summary = seed_dir / "summary.txt"
                if summary.exists():
                    yield ds_dir.name, seed_dir, summary
            continue
        summary = ds_dir / "summary.txt"
        if summary.exists():
            yield ds_dir.name, None, summary


def resolve_seed_dir(base_dir: Path, run: str, model: str, seed: int) -> Path | None:
    """Locate seed output dir regardless of dataset folder name."""
    model_dir = base_dir / run / model
    if not model_dir.is_dir():
        return None
    target = f"seed_{int(seed)}"
    for _, output_dir, _ in iter_model_outputs(model_dir):
        if output_dir is not None and output_dir.name == target:
            return output_dir
    legacy = model_dir / "synthetic" / target
    return legacy if legacy.is_dir() else None


def parse_summary_txt(path: Path) -> dict:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        out[k.strip()] = v.strip()
    if "mean_foscttm" in out:
        out["mean_foscttm"] = to_float(out["mean_foscttm"])
    return out


def load_run_hyperparams(run_folder: Path) -> dict:
    yaml_path = run_folder / "training.yaml"
    if not yaml_path.exists():
        resume_paths = sorted(run_folder.glob("training_resume_*.yaml"))
        yaml_path = resume_paths[-1] if resume_paths else None
    if yaml_path is None or not yaml_path.exists():
        return {}
    cfg = yaml.safe_load(yaml_path.read_text())
    defaults = cfg.get("defaults", {}) or {}
    datasets = cfg.get("datasets", {}) or {}
    syn = active_dataset_block(datasets)
    return {
        "lr":                 to_float(defaults.get("lr")),
        "sinkhorn_reg":       to_float(defaults.get("sinkhorn_reg")),
        "lambda_dyn":         to_float(defaults.get("lambda_dyn")),
        "phi_init_gain":      to_float(defaults.get("phi_init_gain")),
        "dyn_warmup_epochs":  to_int(defaults.get("dyn_warmup_epochs")),
        "use_anchor":         syn.get("use_anchor", defaults.get("use_anchor")),
        "lambda_prior":       to_float(defaults.get("lambda_prior")),
        "kot_fixed_kappa":    to_float(defaults.get("kot_fixed_kappa")),
        "phase1_epochs":      to_int(defaults.get("phase1_epochs", 0)),
        "phase2_epochs":      to_int(defaults.get("phase2_epochs", 0)),
        "phase3_epochs":      to_int(defaults.get("phase3_epochs", 0)),
        "phase3_lr_scale":    to_float(defaults.get("phase3_lr_scale")),
        "kot_rna_layer":      syn.get("kot_rna_layer", defaults.get("kot_rna_layer")),
        "kot_protein_layer":  syn.get("kot_protein_layer", defaults.get("kot_protein_layer")),
        "velocity_layer":     syn.get("velocity_layer"),
        "preprocessing_cache_version": syn.get("preprocessing_cache_version"),
    }


def load_stage_map(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {str(run): str(stage) for run, stage in data.items()}


def infer_stage(
    run_folder: Path,
    hp: dict | None = None,
    stage_map: dict[str, str] | None = None,
) -> str | None:
    """Read oracle/clean/branch from explicit map, runner logs, or run name."""
    run_name = run_folder.name
    merged = {**KNOWN_RUN_STAGES, **(stage_map or {})}
    if run_name in merged:
        return merged[run_name]

    log_paths = sorted(run_folder.glob("run_resume_*.log")) + sorted(run_folder.glob("run_*.log"))
    for log_path in reversed(log_paths):
        text = log_path.read_text(errors="ignore")
        match = re.search(r"\[runner\] stage=(\w+)", text)
        if match:
            return match.group(1)
        match = re.search(
            r"synthetic_linked_ode/(oracle|clean|branch)/rna\.h5ad",
            text,
        )
        if match:
            return match.group(1)

    hp = hp or load_run_hyperparams(run_folder)
    cache = str(hp.get("preprocessing_cache_version") or "").lower()
    match = re.search(r"synthetic_linked_ode_(oracle|clean|branch)(?:_|$)", cache)
    if match:
        return match.group(1)

    name = run_name.lower()
    if "branch" in name:
        return "branch"
    if "oracle" in name:
        return "oracle"
    return "clean"


def parse_seed_from_output(output_dir: Path | None, summary: dict) -> int | float:
    if "seed" in summary:
        seed = str(summary["seed"]).strip()
        if seed.lstrip("+-").isdigit():
            return int(seed)
    if output_dir is not None and output_dir.name.startswith("seed_"):
        tail = output_dir.name.split("_", 1)[1]
        if tail.isdigit():
            return int(tail)
    return np.nan


def load_diagnostics(output_dir: Path | None, summary_path: Path) -> dict:
    if output_dir is not None:
        diag_path = output_dir / "diagnostics.json"
    else:
        diag_path = summary_path.parent / "diagnostics.json"
    if not diag_path.exists():
        return {}
    try:
        return json.loads(diag_path.read_text())
    except json.JSONDecodeError:
        return {}


def build_result_row(
    *,
    run_name: str,
    run_group: str,
    stage: str | None,
    model: str,
    dataset: str,
    output_dir: Path | None,
    summary_path: Path,
    hp: dict,
    sweep_label: str | None = None,
) -> dict:
    summary = parse_summary_txt(summary_path)
    seed = parse_seed_from_output(output_dir, summary)
    diag = load_diagnostics(output_dir, summary_path)
    row = {
        "run": run_name,
        "run_group": run_group,
        "stage": stage,
        "model": model,
        "dataset": dataset,
        "seed": seed,
        "sweep_label": sweep_label,
        "mean_foscttm": summary.get("mean_foscttm", np.nan),
        **hp,
    }
    for key in DIAGNOSTIC_KEYS:
        row[key] = diag.get(key, np.nan)
    return row


def collect_moscot_sweep_rows(
    run_folder: Path,
    run_name: str,
    run_group: str,
    stage: str | None,
    hp: dict,
) -> list[dict]:
    rows: list[dict] = []
    sweep_root = run_folder / "moscot_sweep"
    if not sweep_root.is_dir():
        return rows
    for ds_dir in sorted(p for p in sweep_root.iterdir() if p.is_dir()):
        for variant_dir in sorted(p for p in ds_dir.iterdir() if p.is_dir()):
            summary = variant_dir / "summary.txt"
            if not summary.exists():
                continue
            rows.append(build_result_row(
                run_name=run_name,
                run_group=run_group,
                stage=stage,
                model="moscot_sweep",
                dataset=ds_dir.name,
                output_dir=variant_dir,
                summary_path=summary,
                hp=hp,
                sweep_label=variant_dir.name,
            ))
    return rows


def dedupe_metric_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the newest row per (stage, model, seed, sweep_label, dataset)."""
    if df.empty:
        return df
    out = df.copy()
    out["sweep_label"] = out["sweep_label"].fillna("")
    out["_dedupe_seed"] = out["seed"].apply(
        lambda s: f"seed_{int(s)}" if pd.notna(s) else "single"
    )
    out["_mtime"] = pd.to_datetime(out.get("output_mtime", np.nan), errors="coerce")
    sort_cols = ["stage", "model", "dataset", "_dedupe_seed", "sweep_label", "_mtime", "run"]
    present = [col for col in sort_cols if col in out.columns]
    out = out.sort_values(present)
    out = out.drop_duplicates(
        subset=["stage", "model", "dataset", "_dedupe_seed", "sweep_label"],
        keep="last",
    )
    out["sweep_label"] = out["sweep_label"].replace("", np.nan)
    return out.drop(columns=["_mtime", "_dedupe_seed"], errors="ignore")


def aggregate_metric_by_stage(
    df: pd.DataFrame,
    metric_col: str,
    group_cols: tuple[str, ...] = ("stage", "model", "sweep_label"),
) -> pd.DataFrame:
    sub = df.copy()
    sub[metric_col] = pd.to_numeric(sub[metric_col], errors="coerce")
    present = [col for col in group_cols if col in sub.columns]
    grouped = (
        sub.groupby(present, dropna=False)[metric_col]
        .agg(
            mean="mean",
            std=lambda s: float(s.std(ddof=0)) if s.notna().sum() > 1 else np.nan,
            n="count",
            n_valid=lambda s: int(s.notna().sum()),
        )
        .reset_index()
    )
    return grouped.sort_values(present)


def pivot_stage_wide(agg: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    """Wide table: one row per model, columns clean/oracle/branch mean±std."""
    if agg.empty or "stage" not in agg.columns:
        return pd.DataFrame()

    label_cols = ["model"]
    if "sweep_label" in agg.columns:
        agg = agg.copy()
        agg["sweep_label"] = agg["sweep_label"].fillna("")
        has_sweep = agg["sweep_label"].astype(str).str.len().gt(0).any()
        if has_sweep:
            label_cols.append("sweep_label")

    rows = []
    for key, group in agg.groupby(label_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(label_cols, key))
        for stage in STAGE_ORDER:
            stage_vals = group[group["stage"] == stage]
            if stage_vals.empty:
                row[f"{stage}_{metric_name}_mean"] = np.nan
                row[f"{stage}_{metric_name}_std"] = np.nan
                row[f"{stage}_{metric_name}_n"] = 0
                continue
            pick = stage_vals.iloc[0]
            row[f"{stage}_{metric_name}_mean"] = pick["mean"]
            row[f"{stage}_{metric_name}_std"] = pick["std"]
            row[f"{stage}_{metric_name}_n"] = int(pick["n_valid"])
        rows.append(row)
    wide = pd.DataFrame(rows)
    sort_cols = [col for col in label_cols if col in wide.columns]
    return wide.sort_values(sort_cols) if sort_cols else wide


def write_metric_exports(df: pd.DataFrame, output_dir: Path) -> None:
    """Per-seed long form + FOSCTTM/JVP aggregate tables (long and wide)."""
    if df.empty:
        return

    per_seed_cols = [
        "run", "stage", "run_group", "model", "sweep_label", "dataset", "seed",
        "mean_foscttm", "jvp_rhs_cos",
        "time_mae", "time_spearman",
        "kappa_mean", "beta_mean",
        "lr", "sinkhorn_reg", "lambda_dyn",
        "kot_rna_layer", "kot_protein_layer", "velocity_layer",
    ]
    present_seed = [col for col in per_seed_cols if col in df.columns]
    per_seed = df[present_seed].sort_values(
        [col for col in ["stage", "model", "sweep_label", "seed", "run"] if col in present_seed]
    )
    per_seed.to_csv(output_dir / "metrics_per_seed.csv", index=False)

    foscttm_agg = aggregate_metric_by_stage(df, "mean_foscttm")
    foscttm_agg.to_csv(output_dir / "foscttm_aggregate_by_stage.csv", index=False)
    pivot_stage_wide(foscttm_agg, "foscttm").to_csv(
        output_dir / "foscttm_aggregate_wide.csv", index=False,
    )

    jvp_agg = aggregate_metric_by_stage(df, "jvp_rhs_cos")
    jvp_agg.to_csv(output_dir / "jvp_aggregate_by_stage.csv", index=False)
    pivot_stage_wide(jvp_agg, "jvp").to_csv(
        output_dir / "jvp_aggregate_wide.csv", index=False,
    )


NUMERIC_PATTERN = re.compile(r"[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?$")


def to_float(x):
    if x is None:
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    return float(s) if NUMERIC_PATTERN.match(s) else np.nan


def to_int(x):
    if x is None:
        return 0
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, (int, float)):
        return int(x)
    s = str(x).strip()
    return int(float(s)) if NUMERIC_PATTERN.match(s) else 0


def is_meanscale_run(hp: dict) -> bool:
    """True when KOT used mean-scale RNA/protein/velocity layers."""
    rna = hp.get("kot_rna_layer")
    protein = hp.get("kot_protein_layer")
    cache = str(hp.get("preprocessing_cache_version") or "")
    velocity = str(hp.get("velocity_layer") or "")
    if rna in ("spliced_mean",) or protein in ("protein_mean",):
        return True
    if "meanscale" in cache.lower():
        return True
    if velocity == "true_velocity_mean":
        return True
    return False


def classify_run_group(hp: dict) -> str:
    """Tag a run by coordinate system / data stage (not historical v1–v3 sweeps)."""
    cache = str(hp.get("preprocessing_cache_version") or "").lower()
    if is_meanscale_run(hp):
        if "linked_ode" in cache:
            return "linked_ode"
        return "meanscale"
    if hp.get("phase1_epochs", 0) > 0:
        return "phased"
    velocity = str(hp.get("velocity_layer") or "")
    if velocity == "true_velocity_log" or "log_velocity" in cache:
        return "log_velocity"
    return "legacy"


def normalize_run_ids(run_ids: list[str] | None) -> set[str] | None:
    if not run_ids:
        return None
    out: set[str] = set()
    for rid in run_ids:
        name = rid.strip().rstrip("/")
        if not name.startswith("run_"):
            name = f"run_{name}"
        out.add(name)
    return out


def collect_all_results(
    base_dir: Path,
    run_filter: set[str] | None = None,
    stage_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Walk every run folder and assemble one long-form dataframe."""
    rows: list[dict] = []
    for run_folder in sorted(base_dir.glob("run_*")):
        if not run_folder.is_dir():
            continue
        if run_filter is not None and run_folder.name not in run_filter:
            continue

        hp = load_run_hyperparams(run_folder)
        run_group = classify_run_group(hp)
        run_name = run_folder.name
        stage = infer_stage(run_folder, hp, stage_map=stage_map)

        for model_dir in sorted(p for p in run_folder.iterdir() if p.is_dir()):
            if model_dir.name in SKIP_RUN_CHILDREN:
                continue
            model = model_dir.name
            for dataset, output_dir, summary_path in iter_model_outputs(model_dir):
                row = build_result_row(
                    run_name=run_name,
                    run_group=run_group,
                    stage=stage,
                    model=model,
                    dataset=dataset,
                    output_dir=output_dir,
                    summary_path=summary_path,
                    hp=hp,
                )
                row["output_mtime"] = summary_path.stat().st_mtime
                rows.append(row)

        for row in collect_moscot_sweep_rows(run_folder, run_name, run_group, stage, hp):
            summary_path = run_folder / "moscot_sweep" / row["dataset"] / row["sweep_label"] / "summary.txt"
            if summary_path.exists():
                row["output_mtime"] = summary_path.stat().st_mtime
            rows.append(row)

    df = pd.DataFrame(rows)
    return dedupe_metric_rows(df)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_per_seed_reliability(df: pd.DataFrame, out: Path):
    kot_models = kot_loss_models(df)
    seed_model = kot_models[0] if kot_models else (models_for_plot(df)[0] if models_for_plot(df) else None)
    if seed_model is None:
        return
    sub = df[df.model == seed_model].dropna(subset=["mean_foscttm"]).copy()
    if sub.empty:
        return
    sub["short_run"] = sub["run"].str.replace("run_2026", "").str[:8]
    pivot = sub.pivot_table(
        index="seed", columns="short_run", values="mean_foscttm", aggfunc="mean",
    )
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(max(8, 0.4 * pivot.shape[1]), 0.4 * pivot.shape[0] + 1))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=0.65)
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels([f"seed {s}" for s in pivot.index])
    ax.set_title(f"Per-seed reliability ({seed_model})\nGreen = good FOSCTTM, Red = bad")
    plt.colorbar(im, ax=ax, label="FOSCTTM (mean over cells)")
    fig.savefig(out)
    plt.close(fig)


def plot_model_comparison(df: pd.DataFrame, out: Path):
    """Side-by-side FOSCTTM distribution for the 3 models, both overall and per-seed."""
    sub = df.dropna(subset=["mean_foscttm"])
    if sub.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: overall violin per model
    models = models_for_plot(sub)
    data = [sub[sub.model == m]["mean_foscttm"].dropna().values for m in models]
    nonempty = [(model, values) for model, values in zip(models, data) if len(values) > 0]
    if not nonempty:
        plt.close(fig)
        return
    models = [model for model, values in nonempty]
    data = [values for model, values in nonempty]
    parts = axes[0].violinplot(data, showmeans=True, showmedians=True)
    for body, m in zip(parts["bodies"], models):
        body.set_facecolor(MODEL_COLOR.get(m, "gray"))
        body.set_alpha(0.6)
    axes[0].set_xticks(range(1, len(models) + 1))
    axes[0].set_xticklabels(models)
    axes[0].set_ylabel("FOSCTTM")
    axes[0].axhline(THRESHOLD_GOOD, color="green", linestyle="--", linewidth=1)
    axes[0].set_title("Overall FOSCTTM distribution by model\n(all runs × all seeds combined)")

    # Right: per-seed mean FOSCTTM for each model
    per_seed = sub.groupby(["model", "seed"])["mean_foscttm"].mean().reset_index()
    seeds = sorted(per_seed.seed.unique())
    width = 0.25
    for i, m in enumerate(models):
        vals = []
        for s in seeds:
            sel = per_seed[(per_seed.model == m) & (per_seed.seed == s)]
            vals.append(sel["mean_foscttm"].values[0] if not sel.empty else np.nan)
        x = np.arange(len(seeds)) + i * width
        axes[1].bar(x, vals, width=width, color=MODEL_COLOR.get(m, "gray"),
                    label=m, alpha=0.85)
    axes[1].set_xticks(np.arange(len(seeds)) + width * (len(models) - 1) / 2)
    axes[1].set_xticklabels(seeds, rotation=45)
    axes[1].set_xlabel("seed")
    axes[1].set_ylabel("mean FOSCTTM across runs")
    axes[1].axhline(THRESHOLD_GOOD, color="green", linestyle="--", linewidth=1)
    axes[1].set_title("Per-seed avg FOSCTTM (averaged across runs)\n"
                      "Shows always-good vs always-bad seeds")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_time_alignment(df: pd.DataFrame, out: Path):
    """FOSCTTM vs time_mae (top) and FOSCTTM vs time_spearman (bottom).

    Distinguishes failure modes:
      - high FOSCTTM + low time_spearman  → fully scrambled
      - high FOSCTTM + high time_spearman → wrong neighborhood but right time order
                                            (likely branch swap)
      - low FOSCTTM + high time_spearman  → normal convergence
    """
    sub = df.dropna(subset=["mean_foscttm", "time_mae", "time_spearman"])
    if sub.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "no time metrics", ha="center", va="center",
                transform=ax.transAxes); ax.axis("off")
        fig.savefig(out); plt.close(fig); return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for m in sorted(sub.model.unique()):
        d = sub[sub.model == m]
        c = MODEL_COLOR.get(m, "gray")
        axes[0].scatter(d["mean_foscttm"], d["time_mae"],
                        c=c, alpha=0.6, s=22, label=m)
        axes[1].scatter(d["mean_foscttm"], d["time_spearman"],
                        c=c, alpha=0.6, s=22, label=m)

    axes[0].axvline(THRESHOLD_GOOD, color="green", linestyle="--", linewidth=1)
    axes[0].set_xlabel("FOSCTTM (alignment quality, lower better)")
    axes[0].set_ylabel("time_mae (lower better)")
    axes[0].set_title("Temporal absolute error vs alignment\n"
                       "Top-right corner = fully scrambled failures")
    axes[0].legend(fontsize=8)

    axes[1].axvline(THRESHOLD_GOOD, color="green", linestyle="--", linewidth=1)
    axes[1].axhline(0.0, color="gray", linestyle=":", linewidth=1)
    axes[1].axhline(0.9, color="green", linestyle="--", linewidth=1,
                    label="spearman > 0.9 = order preserved")
    axes[1].set_xlabel("FOSCTTM (alignment quality, lower better)")
    axes[1].set_ylabel("time_spearman (higher better)")
    axes[1].set_ylim(-0.2, 1.05)
    axes[1].set_title("Temporal order preservation vs alignment\n"
                       "High FOSCTTM + low spearman = scrambled minimum")
    axes[1].legend(fontsize=8)

    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def plot_time_model_comparison(df: pd.DataFrame, out: Path):
    """Violin plots of time_mae and time_spearman per model."""
    sub = df.copy()
    models = models_for_plot(sub)
    if not models:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, col, ylab, title, refline in [
        (axes[0], "time_mae", "time_mae (lower better)",
         "Absolute pseudotime error\nNN protein match → donor cell time",
         None),
        (axes[1], "time_spearman", "time_spearman (higher better)",
         "Pseudotime rank correlation\n1 = correct global ordering",
         0.9),
    ]:
        data = [sub.loc[sub.model == m, col].dropna().values for m in models]
        nonempty = [(model, values) for model, values in zip(models, data) if len(values) > 0]
        if not nonempty:
            ax.text(0.5, 0.5, f"no {col}", ha="center", va="center",
                    transform=ax.transAxes); ax.axis("off"); continue
        plot_models = [model for model, values in nonempty]
        plot_data = [values for model, values in nonempty]
        parts = ax.violinplot(plot_data, showmeans=True, showmedians=True)
        for body, m in zip(parts["bodies"], plot_models):
            body.set_facecolor(MODEL_COLOR.get(m, "gray"))
            body.set_alpha(0.6)
        ax.set_xticks(range(1, len(plot_models) + 1))
        ax.set_xticklabels(plot_models)
        ax.set_ylabel(ylab)
        ax.set_title(title)
        if refline is not None:
            ax.axhline(refline, color="green", linestyle="--", linewidth=1,
                       label=f">{refline} = order preserved")
            ax.set_ylim(-0.2, 1.05)
            ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_time_per_seed(df: pd.DataFrame, out: Path, metric: str = "time_spearman"):
    kot_models = kot_loss_models(df)
    seed_model = kot_models[0] if kot_models else None
    if seed_model is None:
        return
    sub = df[(df.model == seed_model) & df[metric].notna()].copy()
    if sub.empty:
        return
    sub["short_run"] = sub["run"].str.replace("run_2026", "").str[:8]
    pivot = sub.pivot_table(
        index="seed", columns="short_run", values=metric, aggfunc="mean",
    )
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(max(8, 0.4 * pivot.shape[1]), 0.4 * pivot.shape[0] + 1))
    vmin, vmax = (-0.2, 1.0) if metric == "time_spearman" else (0, pivot.values.max())
    cmap = "RdYlGn" if metric == "time_spearman" else "RdYlGn_r"
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels([f"seed {s}" for s in pivot.index])
    label = "time_spearman" if metric == "time_spearman" else "time_mae"
    ax.set_title(f"Per-seed {label} (kot)\n"
                 f"{'Green = time order preserved' if metric == 'time_spearman' else 'Green = low time error'}")
    plt.colorbar(im, ax=ax, label=label)
    fig.savefig(out)
    plt.close(fig)


def plot_foscttm_distribution(df: pd.DataFrame, out: Path):
    """Histogram + ECDF of FOSCTTM showing bimodal pattern."""
    sub = df.dropna(subset=["mean_foscttm"])
    if sub.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    bins = np.linspace(0, 0.7, 36)
    for m in sorted(sub.model.unique()):
        vals = sub[sub.model == m]["mean_foscttm"].values
        axes[0].hist(vals, bins=bins, alpha=0.5, label=f"{m} (n={len(vals)})",
                     color=MODEL_COLOR.get(m, "gray"))
    axes[0].axvline(THRESHOLD_GOOD, color="green", linestyle="--", linewidth=1)
    axes[0].set_xlabel("FOSCTTM")
    axes[0].set_ylabel("count")
    axes[0].set_title("FOSCTTM distribution (bimodal pattern)")
    axes[0].legend(fontsize=8)

    for m in sorted(sub.model.unique()):
        vals = np.sort(sub[sub.model == m]["mean_foscttm"].values)
        if len(vals) == 0:
            continue
        cdf = np.arange(1, len(vals) + 1) / len(vals)
        axes[1].plot(vals, cdf, label=m, color=MODEL_COLOR.get(m, "gray"), lw=2)
    axes[1].axvline(THRESHOLD_GOOD, color="green", linestyle="--", linewidth=1)
    axes[1].set_xlabel("FOSCTTM")
    axes[1].set_ylabel("ECDF (fraction of runs ≤ x)")
    axes[1].set_title("FOSCTTM ECDF — higher curves = more runs converge")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Local-minimum diagnostics (loaded from per-epoch training_loss.csv)
# ---------------------------------------------------------------------------

def compute_stuck_signature(loss_df: pd.DataFrame, tail_window: int = 100) -> dict:
    """Quantify whether a single training trajectory is stuck.

    Returns a dict of metrics:
      plateau_value    — mean align loss over the last `tail_window` epochs
      plateau_cv       — std / mean over the same window (low CV = flat)
      escape_range     — max - min of align in the second half (low = no escape)
      descent_epoch    — epoch at which 95% of total descent was achieved
      flat_share       — fraction of epochs (in the second half) where
                         |align[t] - align[t-1]| < 1% of plateau value
      align_plateau    — best (min) align value reached
      stop_epoch       — last epoch trained
    """
    align = loss_df["align"].values
    n = len(align)
    if n < tail_window:
        tail_window = n

    tail = align[-tail_window:]
    plateau_value = float(tail.mean())
    plateau_cv    = float(tail.std() / max(plateau_value, 1e-12))

    second = align[n // 2 :]
    escape_range = float(second.max() - second.min())

    initial = float(align[:5].mean())
    final_min = float(align.min())
    total_descent = initial - final_min
    if total_descent > 1e-8:
        target = initial - 0.95 * total_descent
        below = np.where(align <= target)[0]
        descent_epoch = int(below[0]) + 1 if len(below) else n
    else:
        descent_epoch = 1

    if len(second) > 1:
        deltas = np.abs(np.diff(second))
        flat_share = float((deltas < 0.01 * max(plateau_value, 1e-12)).mean())
    else:
        flat_share = 1.0

    return {
        "plateau_value": plateau_value,
        "plateau_cv":    plateau_cv,
        "escape_range":  escape_range,
        "descent_epoch": descent_epoch,
        "flat_share":    flat_share,
        "align_min":     final_min,
        "stop_epoch":    n,
    }


def load_loss_trajectories(df: pd.DataFrame, base_dir: Path) -> dict:
    """Load training_loss.csv for every (run, model, seed) present in df."""
    trajectories = {}
    for (run, model, seed), _ in df.groupby(["run", "model", "seed"]):
        if not is_kot_model(model):
            continue
        seed_dir = resolve_seed_dir(base_dir, run, model, seed)
        if seed_dir is None:
            continue
        loss_path = seed_dir / "training_loss.csv"
        if not loss_path.exists():
            continue
        loss_df = pd.read_csv(loss_path)
        if "align" in loss_df.columns and len(loss_df) > 0:
            trajectories[(run, model, int(seed))] = loss_df
    return trajectories


def build_stuck_signatures(df: pd.DataFrame, trajectories: dict) -> pd.DataFrame:
    """Compute per-(run, model, seed) stuck-signature merged with FOSCTTM."""
    rows = []
    for (run, model, seed), loss_df in trajectories.items():
        sig = compute_stuck_signature(loss_df)
        match = df[(df["run"] == run) & (df["model"] == model) & (df["seed"] == seed)]
        foscttm = float(match["mean_foscttm"].iloc[0]) if not match.empty else np.nan
        run_group = match["run_group"].iloc[0] if not match.empty else "unknown"
        time_mae = float(match["time_mae"].iloc[0]) if not match.empty and "time_mae" in match else np.nan
        time_spearman = float(match["time_spearman"].iloc[0]) if not match.empty and "time_spearman" in match else np.nan
        rows.append({
            "run":     run,
            "run_group": run_group,
            "model":   model,
            "seed":    seed,
            "foscttm": foscttm,
            "time_mae": time_mae,
            "time_spearman": time_spearman,
            **sig,
        })
    return pd.DataFrame(rows)


def classify_stuck(sig: pd.Series, foscttm_threshold: float = 0.3) -> str:
    """Categorize the trajectory:
       'converged_good' — finished low (FOSCTTM < threshold) and stable
       'stuck_bad'      — finished high (FOSCTTM > threshold), flat tail, no escape
       'transient'      — high FOSCTTM but tail still moving
    """
    foscttm = sig.get("foscttm", np.nan)
    plateau_cv = sig.get("plateau_cv", np.nan)
    if pd.isna(foscttm):
        return "unknown"
    if foscttm < THRESHOLD_GOOD:
        return "converged_good"
    if foscttm > foscttm_threshold and plateau_cv < 0.02:
        return "stuck_bad"
    if foscttm > foscttm_threshold:
        return "transient"
    return "intermediate"


# ---------------------------------------------------------------------------
# Local-minimum plots
# ---------------------------------------------------------------------------

def plot_align_trajectory(ax, loss_df: pd.DataFrame, color: str, alpha: float = 0.35):
    align = np.clip(loss_df["align"].values.astype(float), 1e-12, None)
    ax.plot(loss_df["epoch"], align, color=color, alpha=alpha, linewidth=0.8)


def sample_trajectory_rows(df: pd.DataFrame, max_lines: int) -> pd.DataFrame:
    if len(df) <= max_lines:
        return df
    return df.sample(max_lines, random_state=0)


def plot_loss_trajectories(
    trajectories: dict,
    sig_df: pd.DataFrame,
    out: Path,
    good_threshold: float = THRESHOLD_GOOD,
    stuck_threshold: float = STUCK_FOSCTTM_THRESHOLD,
    max_lines_per_group: int = 30,
):
    kot_sig = sig_df[sig_df["model"].apply(is_kot_model)].copy()
    models = kot_loss_models(kot_sig)

    if kot_sig.empty or not models:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "no KOT loss trajectories", ha="center", va="center",
                transform=ax.transAxes)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)
        return

    fig, axes = plt.subplots(len(models), 1, figsize=(10, 4.2 * len(models)), squeeze=False)

    for ax, model in zip(axes[:, 0], models):
        model_df = kot_sig[kot_sig["model"] == model]
        good = model_df[model_df["foscttm"] < good_threshold]
        bad = model_df[model_df["foscttm"] > stuck_threshold]
        mid = model_df[
            (model_df["foscttm"] >= good_threshold) & (model_df["foscttm"] <= stuck_threshold)
        ]

        counts = {"good": 0, "bad": 0, "mid": 0}
        for rows, color, key in [
            (bad, "#d62728", "bad"),
            (good, "#1f77b4", "good"),
            (mid, "#7f7f7f", "mid"),
        ]:
            for _, row in sample_trajectory_rows(rows, max_lines_per_group).iterrows():
                traj_key = (row["run"], row["model"], int(row["seed"]))
                if traj_key not in trajectories:
                    continue
                plot_align_trajectory(ax, trajectories[traj_key], color=color)
                counts[key] += 1

        if counts["good"] + counts["bad"] + counts["mid"] == 0:
            for _, row in sample_trajectory_rows(model_df, max_lines_per_group * 3).iterrows():
                traj_key = (row["run"], row["model"], int(row["seed"]))
                if traj_key not in trajectories:
                    continue
                plot_align_trajectory(ax, trajectories[traj_key], color=MODEL_COLOR.get(model, "#1f77b4"))
                counts["mid"] += 1

        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_ylabel("Sinkhorn align loss (log scale)")
        ax.set_title(model)
        ax.plot([], [], color="#1f77b4", linewidth=2,
                label=f"converged (FOSCTTM<{good_threshold}, n={counts['good']})")
        ax.plot([], [], color="#7f7f7f", linewidth=2,
                label=f"intermediate (n={counts['mid']})")
        ax.plot([], [], color="#d62728", linewidth=2,
                label=f"stuck (FOSCTTM>{stuck_threshold}, n={counts['bad']})")
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        "KOT align loss trajectories — log scale\n"
        "One panel per KOT variant; flat high plateaus = stuck local minima",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def analyze_local_minima(
    df: pd.DataFrame,
    base_dir: Path,
    output_dir: Path,
    good_threshold: float = THRESHOLD_GOOD,
) -> pd.DataFrame:
    trajectories = load_loss_trajectories(df, base_dir)
    sig_df = build_stuck_signatures(df, trajectories) if trajectories else pd.DataFrame()
    if not sig_df.empty:
        sig_df["category"] = sig_df.apply(
            classify_stuck, axis=1, foscttm_threshold=STUCK_FOSCTTM_THRESHOLD,
        )
        sig_df.to_csv(output_dir / "stuck_signatures.csv", index=False)

    plot_loss_trajectories(
        trajectories,
        sig_df,
        output_dir / "loss_trajectories.png",
        good_threshold=good_threshold,
    )
    return sig_df


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def per_run_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per (run, model): n, mean, std, n_good, plus key diagnostics."""
    agg = df.dropna(subset=["mean_foscttm"]).groupby(["run_group", "run", "model"]).agg(
        n_seeds=("seed", "count"),
        mean_foscttm=("mean_foscttm", "mean"),
        std_foscttm=("mean_foscttm", "std"),
        min_foscttm=("mean_foscttm", "min"),
        max_foscttm=("mean_foscttm", "max"),
        n_good=("mean_foscttm", lambda v: int((v < THRESHOLD_GOOD).sum())),
        mean_time_mae=("time_mae", "mean"),
        mean_time_spearman=("time_spearman", "mean"),
        mean_jvp_rhs_cos=("jvp_rhs_cos", "mean"),
        mean_kappa=("kappa_mean", "mean"),
        mean_beta=("beta_mean", "mean"),
    ).reset_index()
    # Attach hyperparameters from the original df
    hp_cols = ["lr", "sinkhorn_reg", "lambda_dyn", "phi_init_gain",
               "dyn_warmup_epochs", "use_anchor", "phase1_epochs", "phase3_lr_scale"]
    hp_lookup = df.drop_duplicates("run")[["run"] + hp_cols].set_index("run")
    agg = agg.join(hp_lookup, on="run")
    return agg.sort_values("mean_foscttm")


# ---------------------------------------------------------------------------
# Per-folder analysis
# ---------------------------------------------------------------------------

STAGE_GROUPS = [
    ("clean",  "clean stage",  lambda s: s == "clean"),
    ("oracle", "oracle stage", lambda s: s == "oracle"),
    ("branch", "branch stage", lambda s: s == "branch"),
]

ANALYSIS_GROUPS = [
    ("linked_ode",   "synthetic linked ODE · mean-scale", lambda g: g == "linked_ode"),
    ("meanscale",    "mean-scale coordinates",            lambda g: g == "meanscale"),
    ("log_velocity", "log-velocity + PCA",                lambda g: g == "log_velocity"),
    ("phased",       "alternating-phase training",        lambda g: g == "phased"),
    ("legacy",       "older / unclassified configs",      lambda g: g == "legacy"),
    ("all",          "all runs combined",                   lambda g: True),
]

OUTPUT_FOLDER_CHOICES = [name for name, _, _ in STAGE_GROUPS + ANALYSIS_GROUPS]


def run_analysis_subset(df: pd.DataFrame, output_dir: Path, label: str,
                         base_dir: Path) -> None:
    """Generate all plots + tables + report for one filtered DataFrame.

    If the subset is empty, write a small README explaining that and return.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if df.empty:
        (output_dir / "README.txt").write_text(
            f"No runs were classified as '{label}'.\n"
            "Check classify_run_group() if this is unexpected.\n"
        )
        print(f"  [{label}] no rows — wrote placeholder README")
        return

    n_runs   = df["run"].nunique()
    n_models = df["model"].nunique()
    n_seeds  = df["seed"].nunique()
    print(f"  [{label}] {len(df)} rows across {n_runs} run(s), "
          f"{n_models} model(s), {n_seeds} seed(s)")

    summary = per_run_summary(df)
    summary.to_csv(output_dir / "summary_per_run.csv", index=False)

    long_cols = [
        "run", "stage", "run_group", "model", "sweep_label", "seed", "mean_foscttm",
        "time_mae", "time_spearman",
        "jvp_rhs_cos", "jvp_norm", "rhs_norm",
        "kappa_mean", "beta_mean",
        "lr", "sinkhorn_reg", "lambda_dyn", "phi_init_gain",
        "kot_rna_layer", "kot_protein_layer", "velocity_layer",
    ]
    present = [c for c in long_cols if c in df.columns]
    df[present].sort_values(
        [c for c in ["stage", "run", "model", "sweep_label", "seed"] if c in present]
    ).to_csv(
        output_dir / "long_form_results.csv", index=False,
    )

    write_metric_exports(df, output_dir)

    plot_foscttm_distribution(df, output_dir / "foscttm_distribution.png")
    plot_per_seed_reliability(df, output_dir / "per_seed_reliability.png")
    plot_model_comparison    (df, output_dir / "model_comparison.png")
    plot_time_alignment      (df, output_dir / "time_vs_alignment.png")
    plot_time_model_comparison(df, output_dir / "time_model_comparison.png")
    if df["model"].apply(is_kot_model).any():
        plot_time_per_seed(df, output_dir / "time_spearman_per_seed.png", "time_spearman")
        plot_time_per_seed(df, output_dir / "time_mae_per_seed.png", "time_mae")

    analyze_local_minima(df, base_dir, output_dir, good_threshold=THRESHOLD_GOOD)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global THRESHOLD_GOOD

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE,
                        help="Base directory containing run_* folders")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Top-level analysis directory. Sub-folders per run_group "
                             "(linked_ode/, log_velocity/, phased/, all/, …).")
    parser.add_argument("--foscttm-threshold", type=float, default=THRESHOLD_GOOD,
                        help="FOSCTTM below this counts as a 'good' seed")
    parser.add_argument("--only", nargs="+", default=None,
                        choices=OUTPUT_FOLDER_CHOICES,
                        help="Only generate these folders (default: all stage + run_group folders).")
    parser.add_argument("--per-run", action="store_true",
                        help="Also write one analysis folder per run_* under --output.")
    parser.add_argument("--runs", nargs="+", default=None,
                        help="Only include these run folder names, e.g. "
                             "run_20260617_203924 (with or without run_ prefix).")
    parser.add_argument("--stage-map", type=Path, default=None,
                        help="Optional JSON file mapping run folder names to stages "
                             "(clean/oracle/branch), merged over built-in defaults.")
    args = parser.parse_args()

    THRESHOLD_GOOD = args.foscttm_threshold
    args.output.mkdir(parents=True, exist_ok=True)
    run_filter = normalize_run_ids(args.runs)
    if run_filter:
        print(f"Run filter: {sorted(run_filter)}")

    stage_map = load_stage_map(args.stage_map)
    print(f"Scanning {args.base} ...")
    df = collect_all_results(args.base, run_filter=run_filter, stage_map=stage_map)
    if df.empty:
        print("No data found. Check --base path.")
        return

    print(f"Loaded {len(df)} rows across {df['run'].nunique()} runs, "
          f"{df['model'].nunique()} models, {df['seed'].nunique()} seeds.")
    if "stage" in df.columns:
        stage_counts = df["stage"].value_counts(dropna=False).to_dict()
        print("Stages:")
        for name, count in stage_counts.items():
            print(f"  {name}: {count} rows")
    group_counts = df["run_group"].value_counts().to_dict()
    print("Run groups:")
    for name, count in group_counts.items():
        print(f"  {name}: {count} rows")
    print()

    selected = args.only or OUTPUT_FOLDER_CHOICES

    for folder_name, label, predicate in STAGE_GROUPS:
        if folder_name not in selected:
            continue
        sub = df[df["stage"].apply(predicate)].copy()
        sub_dir = args.output / folder_name
        print(f"--- {folder_name}/ ({label}) ---")
        run_analysis_subset(sub, sub_dir, label, base_dir=args.base)
        print()

    for folder_name, label, predicate in ANALYSIS_GROUPS:
        if folder_name not in selected:
            continue
        if folder_name == "all":
            sub = df.copy()
        else:
            sub = df[df["run_group"].apply(predicate)].copy()
        sub_dir = args.output / folder_name
        print(f"--- {folder_name}/ ({label}) ---")
        run_analysis_subset(sub, sub_dir, label, base_dir=args.base)
        print()

    if args.per_run:
        for run_name in sorted(df["run"].unique()):
            sub = df[df["run"] == run_name].copy()
            print(f"--- {run_name}/ (per-run) ---")
            run_analysis_subset(sub, args.output / run_name, run_name, base_dir=args.base)
            print()

    print(f"✓ Analyses written under {args.output}/")
    for folder_name, _, _ in STAGE_GROUPS + ANALYSIS_GROUPS:
        if folder_name in selected:
            print(f"    {args.output / folder_name}/")
    if args.per_run:
        for run_name in sorted(df["run"].unique()):
            print(f"    {args.output / run_name}/")


if __name__ == "__main__":
    main()
