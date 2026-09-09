#!/usr/bin/env python3
"""Every chromatin→RNA metric on disk, in one long table plus a per-run and a per-seed view.

The console summary is a hand-picked view; this flattens the evaluation JSON generically so a new metric appears without editing this file.

The launch gate is recomputed from the evaluation JSON rather than from `preflight_passed.json`, so verdicts stay comparable across training-script versions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

RUNS_ROOT = Path("cache/chromatin/runs")
CACHE_ROOT = Path("cache/chromatin")
OUTPUT_ROOT = Path("cache/results/chromatin")

# full/noDyn must clear the launch gate before corruption arms mean anything; a corruption arm only needs to be finite.
SPREAD_FLOOR = 0.05

# Identity key so lam1/lam1000 noDyn (same run, lambda forced to 0) is not counted twice.
CHECKPOINT_PROBE_BYTES = 4_000_000

IDENTITY = ["run", "dataset", "method", "law", "condition", "lambda_dyn", "seed",
            "checkpoint", "n_test_cells", "checkpoint_sha"]

# Longest suffix first so a key is not parsed as a shorter reference name.
TASK_D_BLOCKS = ["joint_us", "unspliced"]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def checkpoint_fingerprint(run: Path, checkpoint: str) -> str:
    """Identifies the trained model itself, so two names for one run can be spotted."""
    path = run / f"checkpoint_{checkpoint}.pt"
    if not path.exists():
        return "-"
    with path.open("rb") as handle:
        return hashlib.sha256(handle.read(CHECKPOINT_PROBE_BYTES)).hexdigest()[:12]


def parse_task_d_key(key: str) -> tuple[str, str, str, bool] | None:
    """Split `<track>_vs_<reference>[_<block>][_by_group]` into its four parts.

    Returns None for the scalar bookkeeping keys (`internal_n_cells` and friends), which
    carry no track/reference pair and are emitted as plain Task D metrics instead.
    """
    if "_vs_" not in key:
        return None
    track, rest = key.split("_vs_", 1)
    by_group = rest.endswith("_by_group")
    if by_group:
        rest = rest[: -len("_by_group")]
    block = "spliced"
    for candidate in TASK_D_BLOCKS:
        if rest.endswith(f"_{candidate}"):
            block, rest = candidate, rest[: -len(candidate) - 1]
            break
    return track, rest, block, by_group


def emit(records: list[dict], identity: dict, task: str, metrics: dict,
         track: str = "-", reference: str = "-", block: str = "-",
         group: str = "all") -> None:
    """Append one record per scalar in `metrics`, skipping nested blocks."""
    for name, value in metrics.items():
        if isinstance(value, dict):
            continue
        records.append(identity | {"task": task, "track": track, "reference": reference,
                                   "block": block, "group": group, "metric": name,
                                   "value": value})


def task_d_records(records: list[dict], identity: dict, task_d: dict) -> None:
    """Keep tracks apart and score each against its references."""
    for key, value in task_d.items():
        # `_n_genes` is coverage of that reference, not a reference of its own.
        scalar_metric = "n_genes" if key.endswith("_n_genes") else None
        parsed = parse_task_d_key(key.removesuffix("_n_genes") if scalar_metric else key)
        if parsed is None:
            emit(records, identity, "d", {key: value})
            continue
        track, reference, block, by_group = parsed
        if scalar_metric:
            emit(records, identity, "d", {scalar_metric: value}, track, reference, block)
        elif not isinstance(value, dict):
            emit(records, identity, "d", {key: value}, track, reference, block)
        elif by_group:
            for group, scores in value.items():
                emit(records, identity, "d", scores, track, reference, block, str(group))
        else:
            emit(records, identity, "d", value, track, reference, block)


def kot_identity(evaluation: dict, config: dict, run: Path) -> dict:
    return {"run": run.name, "dataset": evaluation["dataset"], "method": "KOT",
            "law": evaluation["law"], "condition": evaluation["condition"],
            "lambda_dyn": config.get("lambda_dyn"), "seed": evaluation["seed"],
            "checkpoint": evaluation["checkpoint"],
            "n_test_cells": evaluation["n_test_cells"],
            "checkpoint_sha": checkpoint_fingerprint(run, evaluation["checkpoint"])}


def baseline_identity(evaluation: dict, run: Path) -> dict:
    return {"run": run.name, "dataset": evaluation["dataset"],
            "method": evaluation["method"], "law": "-", "condition": "-",
            "lambda_dyn": None, "seed": evaluation["seed"], "checkpoint": "baseline",
            "n_test_cells": evaluation["n_test_scored"], "checkpoint_sha": "-"}


def run_records(run: Path) -> list[dict]:
    """Flatten one run directory: the gate, then Tasks A-D, then the training curve's end."""
    baseline = read_json(run / "evaluation_baseline.json")
    evaluation = baseline or read_json(run / "evaluation_best_align.json")
    if not evaluation:
        return []
    records: list[dict] = []
    if baseline:
        identity = baseline_identity(evaluation, run)
        emit(records, identity, "meta", {"runtime_seconds": evaluation["runtime_seconds"],
                                         "prediction_kind": evaluation["prediction_kind"]})
    else:
        identity = kot_identity(evaluation, read_json(run / "run_config.json"), run)
        emit(records, identity, "gate", read_json(run / "preflight.json"))

    emit(records, identity, "a", evaluation.get("task_a", {}))
    for group, scores in evaluation.get("task_a_by_group", {}).items():
        emit(records, identity, "a", scores, group=str(group))
    emit(records, identity, "b", evaluation.get("task_b", {}))
    for group, score in evaluation.get("task_b_within_group", {}).items():
        emit(records, identity, "b", {"foscttm_within": score}, group=str(group))
    emit(records, identity, "c", evaluation.get("task_c", {}))
    task_d_records(records, identity, evaluation.get("task_d", {}))
    return records


def pick(frame: pd.DataFrame, task: str, metric: str, **filters) -> pd.Series:
    """One metric per run, keyed by run name, for assembling the wide per-run view."""
    rows = frame[(frame["task"] == task) & (frame["metric"] == metric)
                 & (frame["group"] == filters.pop("group", "all"))]
    for column, value in filters.items():
        rows = rows[rows[column] == value]
    return rows.set_index("run")["value"]


def gate_verdict(summary: pd.DataFrame) -> pd.DataFrame:
    """Launch criteria recomputed per run from the evaluation's own numbers.

    Corruption arms are scored too, but reported as `not_gated` rather than pass or fail.
    """
    spread_ok = summary["gate_prediction_spread_ratio"] >= SPREAD_FLOOR
    state_ok = summary["A_gene_pearson_median"] > 0
    pairing_ok = summary["B_knn_foscttm"] < summary["B_knn_foscttm_permuted_floor"]
    kinetics_ok = summary["D_bio_scvelo_cell_cosine_median"] > 0
    summary = summary.assign(
        gate_spread_pass=spread_ok, gate_state_pass=state_ok,
        gate_pairing_pass=pairing_ok, gate_kinetics_pass=kinetics_ok,
    )
    checks = ["gate_spread_pass", "gate_state_pass", "gate_pairing_pass",
              "gate_kinetics_pass"]
    gated = summary["condition"].isin(["full", "noDyn"])
    summary["gate_verdict"] = np.where(
        ~gated, "not_gated", np.where(summary[checks].all(axis=1), "pass", "fail"))
    summary["gate_failed_checks"] = [
        ", ".join(name.replace("gate_", "").replace("_pass", "")
                  for name in checks if not row[name])
        for _, row in summary.iterrows()]
    return summary


def run_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Gate quantities beside what each task scored; FOSCTTM lives next to the per-run pairing floor."""
    summary = frame.drop_duplicates("run").set_index("run")[IDENTITY[1:]].copy()
    columns = {
        "gate_prediction_spread_ratio": ("gate", "prediction_spread_ratio", {}),
        "gate_jvp_norm_median": ("gate", "jvp_norm_median", {}),
        "A_gene_pearson_median": ("a", "gene_pearson_median", {}),
        "A_gene_spearman_median": ("a", "gene_spearman_median", {}),
        "A_gene_nrmse_median": ("a", "gene_nrmse_median", {}),
        "A_cell_cosine_median": ("a", "cell_cosine_median", {}),
        "A_cell_pearson_median": ("a", "cell_pearson_median", {}),
        "B_knn_foscttm": ("b", "knn_foscttm", {}),
        "B_knn_foscttm_permuted_floor": ("b", "knn_foscttm_permuted_floor", {}),
        "B_knn_foscttm_constant_floor": ("b", "knn_foscttm_constant_floor", {}),
        "B_knn_partner_diversity": ("b", "knn_partner_diversity", {}),
        "B_knn_top_partner_share": ("b", "knn_top_partner_share", {}),
        "B_knn_top1": ("b", "knn_top1", {}),
        "B_knn_cell_type_accuracy": ("b", "knn_cell_type_accuracy", {}),
        "B_cell_type_majority_baseline": ("b", "cell_type_majority_baseline", {}),
        "B_sinkhorn_foscttm": ("b", "sinkhorn_foscttm", {}),
        "B_foscttm_within_mean": ("b", "foscttm_within", {"group": "mean"}),
        "C_label_transfer_accuracy": ("c", "label_transfer_accuracy", {}),
        "C_ari": ("c", "ari", {}),
        "C_ilisi": ("c", "ilisi", {}),
        "D_internal_law_cell_cosine_median": (
            "d", "cell_cosine_median", {"track": "internal", "reference": "law"}),
        "D_bio_law_cell_cosine_median": (
            "d", "cell_cosine_median", {"track": "biological", "reference": "law"}),
        "D_bio_scvelo_cell_cosine_median": (
            "d", "cell_cosine_median",
            {"track": "biological", "reference": "velocity_scvelo"}),
        "D_bio_scvelo_cell_cosine_centred_median": (
            "d", "cell_cosine_centred_median",
            {"track": "biological", "reference": "velocity_scvelo"}),
        "D_bio_scvelo_gene_pearson_median": (
            "d", "gene_pearson_median",
            {"track": "biological", "reference": "velocity_scvelo"}),
        "D_bio_scvelo_norm_ratio_median": (
            "d", "norm_ratio_median",
            {"track": "biological", "reference": "velocity_scvelo"}),
        "D_bio_scvelo_sign_agreement": (
            "d", "sign_agreement", {"track": "biological", "reference": "velocity_scvelo"}),
        "D_bio_scvelo_sign_agreement_chance": (
            "d", "sign_agreement_chance",
            {"track": "biological", "reference": "velocity_scvelo"}),
        "D_bio_regvelo_cell_cosine_median": (
            "d", "cell_cosine_median",
            {"track": "biological", "reference": "velocity_regvelo"}),
    }
    for name, (task, metric, filters) in columns.items():
        summary[name] = pick(frame, task, metric, **filters)
    return mark_duplicates(gate_verdict(summary.reset_index()))


def mark_duplicates(summary: pd.DataFrame) -> pd.DataFrame:
    """Name the earlier run whenever two directories hold the identical trained model.

    Kept as a column rather than dropped: the duplicate rows are real directories that
    other scripts and job files refer to by name, and silently deleting half of them from
    the export would make this table disagree with the run tree.
    """
    models = summary[summary["checkpoint_sha"] != "-"]
    first = models.groupby("checkpoint_sha")["run"].transform("first")
    summary["duplicate_of"] = ""
    is_copy = models["run"] != first
    summary.loc[is_copy[is_copy].index, "duplicate_of"] = first[is_copy]
    return summary


def seed_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Mean, sd and n across seeds, over distinct models only.

    A single seed cannot separate a real gap from fit variation, and a model counted twice
    under two directory names reports a spread of zero it did not earn.
    """
    keys = ["dataset", "method", "law", "condition", "lambda_dyn"]
    values = [c for c in summary.columns
              if c.startswith(("gate_prediction", "gate_jvp", "A_", "B_", "C_", "D_"))]
    distinct = summary[summary["duplicate_of"] == ""]
    grouped = distinct.groupby(keys, dropna=False)[values].agg(["mean", "std", "count"])
    grouped.columns = [f"{metric}_{statistic}" for metric, statistic in grouped.columns]
    return grouped.reset_index()


def ceilings_frame() -> pd.DataFrame:
    """Task A reference points, which the KOT numbers are meaningless without."""
    rows = []
    for path in sorted(CACHE_ROOT.glob("*_baselines.json")):
        dataset = path.name.removesuffix("_baselines.json")
        for name, scores in json.loads(path.read_text()).items():
            row = {"dataset": dataset, "reference": name,
                   "category": scores.get("reference_category"),
                   "cell_cosine_median": scores.get("cell_cosine_median")}
            for subset in ["all", "in_G", "not_in_G"]:
                if subset in scores:
                    row[f"gene_pearson_{subset}"] = scores[subset]["gene_pearson_median"]
                    row[f"gene_spearman_{subset}"] = scores[subset]["gene_spearman_median"]
                    row[f"n_genes_{subset}"] = scores[subset]["n_genes"]
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pattern", default="final_*",
                        help="glob over cache/chromatin/runs")
    parser.add_argument("--output-dir", default=str(OUTPUT_ROOT))
    args = parser.parse_args()

    records = [row for run in sorted(RUNS_ROOT.glob(args.pattern)) if run.is_dir()
               for row in run_records(run)]
    if not records:
        print(f"no evaluated runs under {RUNS_ROOT}/{args.pattern}")
        return 1
    long_frame = pd.DataFrame(records)
    summary = run_summary(long_frame)
    per_seed = seed_summary(summary)
    ceilings = ceilings_frame()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    written = {
        "chromatin_metrics_long.csv": long_frame,
        "chromatin_run_summary.csv": summary,
        "chromatin_seed_summary.csv": per_seed,
        "chromatin_task_a_ceilings.csv": ceilings,
    }
    for name, frame in written.items():
        frame.to_csv(output / name, index=False)
        print(f"wrote {output / name}  ({len(frame)} rows, {len(frame.columns)} columns)")

    runs = summary["run"].nunique()
    metrics = long_frame["metric"].nunique()
    print(f"\n{runs} runs, {metrics} distinct metrics, {len(long_frame)} measurements")
    gated = summary[summary["gate_verdict"] != "not_gated"]
    counts = gated["gate_verdict"].value_counts().to_dict()
    print(f"§19 gate over the {len(gated)} full/noDyn runs: {counts}")
    if "fail" in counts:
        reasons = gated.loc[gated["gate_verdict"] == "fail", "gate_failed_checks"]
        print("  failing checks: " + "; ".join(
            f"{reason} x{count}" for reason, count in reasons.value_counts().items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
