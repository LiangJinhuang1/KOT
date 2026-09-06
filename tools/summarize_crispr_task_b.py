#!/usr/bin/env python
"""Consolidate every Task B ISP run into three comparable tables.

Each ISP writes its own per-seed summaries, but they are not comparable as written: the
linear runner scores at REPLICATE level (n=282) while `run_kot_crispr.py task_b` pools over
replicates (n~90), which alone is worth about 0.07 of Spearman. Everything here is pooled,
so the ISPs sit on the same footing.

Also reports the between-perturbation ratio, which the correlation cannot substitute for:
an ISP that returns the same profile for every knockout scores near zero for a reason
completely different from one that predicts a wrong response, and only this separates them.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.crispr_metrics import score_effect_subsets

RESULTS = Path("cache/results/crispr")
# label -> (per-seed summary glob, effects glob or None, predicted-profile glob)
SOURCES = {
    "linear": (
        "task_b_linear_20260906_{arm}/seed_*/crispr_task_b_summary.csv",
        "task_b_linear_20260906_{arm}/seed_*/crispr_task_b_effects_predicted.csv",
        "task_b_linear_20260906_full/seed_42/predicted_rna_profiles.csv",
    ),
    "multipert": (
        "task_b_multipert_20260906_{arm}/seed_*/crispr_task_b_summary.csv",
        None,
        "task_b_multipert_20260906/predicted_rna.csv",
    ),
    "gears": (
        "task_b_gears_20260906/seed_*/crispr_task_b_summary.csv",
        None,
        "task_b_gears_20260906/seed_42/predicted_rna.csv",
    ),
    "scgpt": (
        "task_b_scgpt_20260906/seed_*/crispr_task_b_summary.csv",
        None,
        "task_b_scgpt_20260906/seed_42/predicted_rna.csv",
    ),
    "regvelo": (
        "task_b_regvelo_full_20260906/seed_*/crispr_task_b_summary.csv",
        None,
        "task_b_regvelo_full_20260906/seed_42/predicted_rna.csv",
    ),
}


def between_perturbation_ratio(path: Path) -> tuple[float, float, int]:
    """How much a predicted profile varies BETWEEN knockouts, against its own spread.

    A collapsed ISP returns one profile for every knockout; its ratio goes to zero while
    its correlation merely goes to zero as well, which looks like ordinary poor accuracy.
    """
    if not path.exists():
        return float("nan"), float("nan"), 0
    frame = pd.read_csv(path)
    features = [c for c in frame.columns if c.startswith("feature_")]
    if "model" in frame.columns:
        frame = frame[frame["model"] == "linear"]
    frame = frame.groupby("perturbation", as_index=False)[features].mean()
    values = frame[features].to_numpy(dtype=float)
    correlations = np.corrcoef(values)
    off_diagonal = correlations[~np.eye(len(correlations), dtype=bool)]
    ratio = float((values - values.mean(0)).std(0).mean() / values.std(1).mean())
    return ratio, float(off_diagonal.mean()), len(values)


def pooled_linear(arm: str, n_boot: int, seed: int) -> pd.Series | None:
    """Re-score the linear ISP pooled over replicates, matching every other ISP."""
    parts = []
    for path in glob.glob(str(RESULTS / f"task_b_linear_20260906_{arm}/seed_*/"
                                        "crispr_task_b_effects_predicted.csv")):
        block = pd.read_csv(path, dtype={"replicate": str})
        parts.append(block[block["model"] == "linear"])
    if not parts:
        return None
    flags = {f"in_{s}": "first" for s in
             ("primary", "significant", "post_transcriptional", "self")}
    pooled = pd.concat(parts).groupby(["perturbation", "protein"], as_index=False).agg(
        {"delta_p_phi": "mean", "delta_protein": "mean", **flags})
    scored = score_effect_subsets(pooled, "delta_p_phi", "delta_protein", n_boot, seed)
    primary = scored[scored["effect_subset"] == "primary"]
    return primary.iloc[0] if not primary.empty else None


def flat_summary(n_boot: int, seed: int) -> pd.DataFrame:
    """Every ISP x KOT arm x effect subset in one table, item 32's column names.

    Task B has no Jacobian arm: an ISP hands over predicted RNA, which goes through phi,
    so there is a `_phi` block and nothing to put in a `_jacobian` one. The columns are
    named to match `crispr_summary.csv` so the two tasks can be read side by side.
    """
    rows = []
    for label, (summary_glob, _, profile_glob) in SOURCES.items():
        ratio, pairwise, _ = between_perturbation_ratio(RESULTS / profile_glob)
        arms = ("full", "noDyn", "shuffleVel") if "{arm}" in summary_glob else ("full",)
        for arm in arms:
            files = sorted(glob.glob(str(RESULTS / summary_glob.format(arm=arm))))
            if not files:
                continue
            frame = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
            models = frame["model"].astype(str).unique() if "model" in frame else [label]
            for model in models:
                block = frame[frame["model"] == model] if "model" in frame else frame
                for subset, part in block.groupby("effect_subset"):
                    space = part["space"].iloc[0]
                    rows.append({
                        "model": label if model == label else f"{label}:{model}",
                        "condition": arm,
                        "space": space,
                        "effect_subset": subset,
                        "n_effects": int(part["n"].max()),
                        "n_seeds": int(part["seed"].nunique()),
                        "spearman_phi": float(part["spearman"].mean()),
                        "pearson_phi": float(part["pearson"].mean()),
                        "mae_phi": float(part["mae"].mean()),
                        "nrmse_phi": float(part["nrmse"].mean()),
                        "sign_phi": float(part["sign_acc"].mean()),
                        "spearman_phi_sd": float(part["spearman"].std(ddof=0)),
                        "response_cosine": float(part["response_cosine_mean"].mean())
                        if "response_cosine_mean" in part else np.nan,
                        "spearman_ci_lo": float(part["spearman_perturbation_lo"].mean())
                        if "spearman_perturbation_lo" in part else np.nan,
                        "spearman_ci_hi": float(part["spearman_perturbation_hi"].mean())
                        if "spearman_perturbation_hi" in part else np.nan,
                        "between_perturbation_ratio": ratio,
                        "mean_pairwise_profile_r": pairwise,
                        "scoring": "replicate_level" if label == "linear" else "pooled",
                    })
    order = ["primary", "significant", "post_transcriptional", "self", "all_eligible"]
    table = pd.DataFrame(rows)
    table["_rank"] = table["effect_subset"].apply(
        lambda s: order.index(s) if s in order else len(order))
    return (table.sort_values(["model", "condition", "_rank"])
            .drop(columns="_rank").reset_index(drop=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=RESULTS / "task_b_consolidated")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    comparison, per_arm = [], []
    for label, (summary_glob, _, profile_glob) in SOURCES.items():
        ratio, pairwise, n_profiles = between_perturbation_ratio(RESULTS / profile_glob)
        for arm in ("full", "noDyn", "shuffleVel"):
            files = sorted(glob.glob(str(RESULTS / summary_glob.format(arm=arm))))
            if not files or (arm != "full" and "{arm}" not in summary_glob):
                if arm != "full":
                    continue
            if not files:
                continue
            frame = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
            if "model" in frame.columns and label == "linear":
                frame = frame[frame["model"] == "linear"]
            protein = frame[(frame["space"] == "protein")
                            & (frame["effect_subset"] == "primary")]
            rna = frame[frame["space"] == "rna"]
            if protein.empty:
                continue
            row = {"isp": label, "kot_arm": arm, "seeds": int(protein["seed"].nunique()),
                   "n_effects": int(protein["n"].max()),
                   "protein_spearman": float(protein["spearman"].mean()),
                   "protein_spearman_sd": float(protein["spearman"].std(ddof=0)),
                   "protein_pearson": float(protein["pearson"].mean()),
                   "protein_nrmse": float(protein["nrmse"].mean()),
                   "sign_accuracy": float(protein["sign_acc"].mean()),
                   "response_cosine": float(protein["response_cosine_mean"].mean()),
                   "rna_spearman": float(rna["spearman"].mean()) if not rna.empty else np.nan,
                   "rna_nrmse": float(rna["nrmse"].mean()) if not rna.empty else np.nan,
                   "between_perturbation_ratio": ratio,
                   "mean_pairwise_profile_r": pairwise,
                   "n_predicted_perturbations": n_profiles,
                   "scoring": "replicate_level" if label == "linear" else "pooled"}
            per_arm.append(row)
            if arm == "full":
                comparison.append(row)

    # The linear rows above are replicate-level; add the pooled re-score so the headline
    # table compares like with like.
    for arm in ("full", "noDyn", "shuffleVel"):
        primary = pooled_linear(arm, args.n_boot, args.seed)
        if primary is None:
            continue
        ratio, pairwise, n_profiles = between_perturbation_ratio(
            RESULTS / SOURCES["linear"][2])
        row = {"isp": "linear", "kot_arm": arm, "seeds": 12,
               "n_effects": int(primary["n"]),
               "protein_spearman": float(primary["spearman"]),
               "protein_spearman_sd": np.nan,
               "protein_pearson": float(primary["pearson"]),
               "protein_nrmse": float(primary["nrmse"]),
               "sign_accuracy": float(primary["sign_acc"]),
               "response_cosine": float(primary["response_cosine_mean"]),
               "spearman_ci_lo": float(primary["spearman_perturbation_lo"]),
               "spearman_ci_hi": float(primary["spearman_perturbation_hi"]),
               "rna_spearman": np.nan, "rna_nrmse": np.nan,
               "between_perturbation_ratio": ratio,
               "mean_pairwise_profile_r": pairwise,
               "n_predicted_perturbations": n_profiles,
               "scoring": "pooled"}
        per_arm.append(row)
        if arm == "full":
            comparison = [r for r in comparison if r["isp"] != "linear"] + [row]

    headline = pd.DataFrame(comparison).sort_values("protein_spearman", ascending=False)
    everything = pd.DataFrame(per_arm).sort_values(["isp", "kot_arm"])
    flat = flat_summary(args.n_boot, args.seed)
    headline.to_csv(args.out_dir / "task_b_isp_comparison.csv", index=False)
    everything.to_csv(args.out_dir / "task_b_isp_by_kot_arm.csv", index=False)
    flat.to_csv(RESULTS / "crispr_task_b_summary.csv", index=False)
    print(headline[["isp", "n_effects", "seeds", "protein_spearman", "protein_nrmse",
                    "rna_spearman", "between_perturbation_ratio",
                    "mean_pairwise_profile_r", "scoring"]].round(4).to_string(index=False))
    print()
    for path in ("task_b_isp_comparison.csv", "task_b_isp_by_kot_arm.csv"):
        print(f"[out] {args.out_dir / path}")
    print(f"[out] {RESULTS / 'crispr_task_b_summary.csv'}  ({len(flat)} rows)")


if __name__ == "__main__":
    main()
