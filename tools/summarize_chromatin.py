#!/usr/bin/env python3
"""Every chromatin run's metrics in one table, arranged so the ablations can be read.

The point of §12's six conditions is the GAP between them, not any one number. This puts
them side by side per dataset and, where a corruption is supposed to hurt, prints the
delta against `full` — so "shuffle scores the same as full" shows up as a zero rather than
as two numbers a reader has to subtract.

§13's two tracks are kept apart: `bio_*` columns score the checkpoint against the TRUE
velocity and the true G whatever it was trained on, `int_*` against what it was trained
with. A corrupted run should look self-consistent on the internal track and should NOT
recover on the biological one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

RUNS_ROOT = Path("cache/chromatin/runs")
CACHE_ROOT = Path("cache/chromatin")

# Corruptions whose whole purpose is to destroy something; `full` is the reference they
# are read against, and noDyn is a control rather than a corruption of the velocity.
CORRUPTIONS = ["shuffle", "reverse", "zero", "permG"]


def flatten_baseline(run: Path) -> dict | None:
    """One row for a competing method. Same Task A-C columns as a KOT row.

    `n_test_scored` also identifies provenance: this experiment's runs all score 4,000
    test cells, so a row with a different count came from a different submission and must
    not be mixed into the table.
    """
    evaluation = run / "evaluation_baseline.json"
    if not evaluation.exists():
        return None
    ev = json.loads(evaluation.read_text())
    row = {"run": run.name, "method": ev["method"], "dataset": ev["dataset"],
           "law": "-", "cond": "-", "lam": None, "seed": ev["seed"],
           "n_test": ev["n_test_scored"], "kind": ev["prediction_kind"],
           "runtime_s": round(ev["runtime_seconds"], 1)}
    a = ev["task_a"]
    row.update({"A_gene_r": a["gene_pearson_median"], "A_gene_rho": a["gene_spearman_median"],
                "A_nrmse": a["gene_nrmse_median"], "A_cell_cos": a["cell_cosine_median"]})
    b = ev.get("task_b", {})
    row.update({"B_foscttm": b.get("knn_foscttm"),
                "B_floor": b.get("knn_foscttm_constant_floor"),
                "B_diversity": b.get("knn_partner_diversity"),
                "B_knn_type": b.get("knn_cell_type_accuracy"),
                "B_type_baseline": b.get("cell_type_majority_baseline"),
                "B_sink_top1": b.get("sinkhorn_top1"),
                "B_sink_foscttm": b.get("sinkhorn_foscttm"),
                "B_fosknn_1pct": b.get("knn_fosknn_frac0.01"),
                "B_fosknn_5pct": b.get("knn_fosknn_frac0.05")})
    c = ev.get("task_c", {})
    row.update({"C_transfer": c.get("label_transfer_accuracy"), "C_ari": c.get("ari"),
                "C_nmi": c.get("nmi"), "C_asw": c.get("cell_type_asw"),
                "C_ilisi": c.get("ilisi")})
    return row


def flatten(run: Path) -> dict | None:
    """One row per evaluated checkpoint: the gate, then Tasks A-D."""
    evaluation = run / "evaluation_best_align.json"
    gate, config = run / "preflight.json", run / "run_config.json"
    if not (evaluation.exists() and config.exists()):
        return None
    ev = json.loads(evaluation.read_text())
    cfg = json.loads(config.read_text())
    row = {
        "run": run.name, "method": "KOT", "dataset": ev["dataset"], "law": ev["law"],
        "cond": ev["condition"], "lam": cfg["lambda_dyn"], "seed": ev["seed"],
        "n_test": ev["n_test_cells"], "kind": "direct",
    }
    if gate.exists():
        g = json.loads(gate.read_text())
        row["spread"] = g.get("prediction_spread_ratio")
        row["jvp_norm"] = g.get("jvp_norm_median")

    a = ev["task_a"]
    row.update({"A_gene_r": a["gene_pearson_median"], "A_gene_rho": a["gene_spearman_median"],
                "A_nrmse": a["gene_nrmse_median"], "A_cell_cos": a["cell_cosine_median"]})
    b = ev["task_b"]
    row.update({"B_foscttm": b["knn_foscttm"], "B_floor": b.get("knn_foscttm_constant_floor"),
                "B_diversity": b.get("knn_partner_diversity"),
                "B_hub_share": b.get("knn_top_partner_share"),
                "B_knn_type": b.get("knn_cell_type_accuracy"),
                "B_sink_top1": b["sinkhorn_top1"],
                "B_sink_foscttm": b.get("sinkhorn_foscttm"),
                "B_sink_type": b.get("sinkhorn_cell_type_accuracy"),
                "B_fosknn_1pct": b.get("knn_fosknn_frac0.01"),
                "B_fosknn_5pct": b.get("knn_fosknn_frac0.05"),
                "B_fosknn_10pct": b.get("knn_fosknn_frac0.1")})
    row["B_within"] = ev.get("task_b_within_group", {}).get("mean")
    c = ev.get("task_c", {})
    row.update({"C_transfer": c.get("label_transfer_accuracy"), "C_ari": c.get("ari"),
                "C_nmi": c.get("nmi"), "C_asw": c.get("cell_type_asw"),
                "C_ilisi": c.get("ilisi"), "C_conn": c.get("graph_connectivity")})
    law = ev["task_d"].get("internal_vs_law", {})
    row.update({"int_D_cos": law.get("cell_cosine_median"),
                "int_D_cos_c": law.get("cell_cosine_centred_median")})
    bio_law = ev["task_d"].get("biological_vs_law", {})
    row.update({"bio_law_D_cos": bio_law.get("cell_cosine_median"),
                "bio_law_D_cos_c": bio_law.get("cell_cosine_centred_median")})
    d = ev["task_d"].get("biological_vs_velocity_scvelo", {})
    row.update({"bio_D_cos": d.get("cell_cosine_median"),
                "bio_D_cos_c": d.get("cell_cosine_centred_median"),
                "bio_D_pos": d.get("cell_cosine_positive_fraction"),
                "bio_D_gene_r": d.get("gene_pearson_median"),
                "bio_D_sign": d.get("sign_agreement"),
                "bio_D_sign_chance": d.get("sign_agreement_chance"),
                "bio_D_resid_sf": d.get("scale_free_residual_median")})
    reg = ev["task_d"].get("biological_vs_velocity_regvelo", {})
    row.update({"bio_regvelo_D_cos": reg.get("cell_cosine_median")})
    return row


def baseline_row(dataset: str) -> dict | None:
    """The ceilings Task A has to be read against."""
    path = CACHE_ROOT / f"{dataset}_baselines.json"
    if not path.exists():
        return None
    b = json.loads(path.read_text())
    names = [name for name in ["mean_only", "identity", "ridge", "paired_mlp"]
             if name in b]
    return {name: {"gene_r_in_G": b[name]["in_G"]["gene_pearson_median"],
                   "gene_r_all": b[name]["all"]["gene_pearson_median"],
                   "cell_cos": b[name]["cell_cosine_median"]}
            for name in names}


def print_block(frame: pd.DataFrame, title: str, columns: list[str]) -> None:
    present = [c for c in columns if c in frame.columns and frame[c].notna().any()]
    if not present:
        return
    print(f"\n--- {title}")
    keys = [c for c in ["dataset", "method", "cond", "lam", "seed"] if c in frame.columns]
    print(frame[keys + present].to_string(index=False,
                                          float_format=lambda v: f"{v:+.4f}"))


def print_ablation_gaps(frame: pd.DataFrame, columns: list[str]) -> None:
    """Each corruption minus `full`, which is what the ablation actually claims."""
    print("\n--- ablation gaps (condition - full); a corruption that changes nothing is a bug "
          "in the control, not a result")
    for dataset, block in frame.groupby("dataset"):
        reference = block[block["cond"] == "full"]
        if reference.empty:
            continue
        base = reference.iloc[0]
        rows = []
        for _, row in block[block["cond"] != "full"].iterrows():
            entry = {"dataset": dataset, "cond": row["cond"]}
            entry.update({c: row[c] - base[c] for c in columns
                          if c in block.columns and pd.notna(row[c]) and pd.notna(base[c])})
            rows.append(entry)
        if rows:
            print(pd.DataFrame(rows).to_string(index=False,
                                               float_format=lambda v: f"{v:+.4f}"))


def print_seed_summary(frame: pd.DataFrame) -> None:
    """Mean and spread across seeds, which is the only form a comparison can be read in.

    A single seed cannot separate a real difference from fit variation, so the per-run
    tables above are for diagnosis and this is the one to compare methods on.
    """
    columns = [c for c in ["A_gene_r", "A_cell_cos", "B_foscttm", "B_floor", "B_diversity",
                           "B_knn_type", "C_ari", "C_ilisi"] if c in frame.columns]
    grouped = frame.groupby(["dataset", "method", "cond", "lam"], dropna=False)
    summary = grouped[columns].agg(["mean", "std", "count"])
    print("\n--- across seeds (mean / sd / n). FOSCTTM: compare against B_floor, the score "
          "a CONSTANT map gets on the same data — 0.5 is a destroyed pairing, not chance")
    with pd.option_context("display.width", 250, "display.max_columns", 60):
        print(summary.round(4).to_string())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pattern", default="*", help="glob over cache/chromatin/runs")
    parser.add_argument("--csv", default=None, help="also write the full table here")
    parser.add_argument("--expect-n-test", type=int, default=None,
                        help="drop rows scored on a different number of cells — they came "
                             "from another submission and are not comparable")
    args = parser.parse_args()

    rows = []
    for run in sorted(RUNS_ROOT.glob(args.pattern)):
        if not run.is_dir():
            continue
        row = flatten(run) or flatten_baseline(run)
        if row is not None:
            rows.append(row)
    if not rows:
        print(f"no evaluated runs under {RUNS_ROOT}/{args.pattern}")
        return 1
    frame = pd.DataFrame(rows)
    if args.expect_n_test is not None:
        foreign = frame[frame["n_test"] != args.expect_n_test]
        if not foreign.empty:
            print(f"dropping {len(foreign)} row(s) scored on a different cell count "
                  f"(not this submission): {sorted(foreign['run'])}")
            frame = frame[frame["n_test"] == args.expect_n_test]
    frame = frame.sort_values(["dataset", "method", "law", "cond", "lam", "seed"])
    pd.set_option("display.width", 250, "display.max_columns", 60)

    for dataset in frame["dataset"].unique():
        ceilings = baseline_row(dataset)
        if ceilings:
            print(f"\n=== {dataset} ceilings (Task A is meaningless without these)")
            print(pd.DataFrame(ceilings).T.to_string(float_format=lambda v: f"{v:+.4f}"))

    print_block(frame, "health: did phi survive training at all?", ["lam", "spread", "jvp_norm"])
    print_block(frame, "Task A — RNA state from held-out ATAC",
                ["A_gene_r", "A_gene_rho", "A_nrmse", "A_cell_cos"])
    # 0.5 is a DESTROYED pairing; a constant, zero-information map scores 0.25. The
    # informative band is [0, floor], and `B_floor` is that floor measured per run.
    print_block(frame, "Task B — cell pairing (informative band is [0, B_floor]; "
                       "B_floor is what a CONSTANT map scores, ~0.25. 0.5 = destroyed pairing)",
                ["B_foscttm", "B_floor", "B_within", "B_diversity", "B_hub_share",
                 "B_knn_type", "B_sink_top1", "B_sink_foscttm", "B_sink_type",
                 "B_fosknn_1pct", "B_fosknn_5pct"])
    print("    top-k is deliberately omitted: chance is 1/n, so it is not comparable "
          "between methods scored on different numbers of cells.")
    print_block(frame, "Task C — joint integration",
                ["C_transfer", "C_ari", "C_nmi", "C_asw", "C_ilisi", "C_conn"])
    print_block(frame, "Task D — J_phi v_true vs scVelo (biological)",
                ["bio_D_cos", "bio_D_cos_c", "bio_D_pos", "bio_D_gene_r", "bio_D_sign",
                 "bio_D_sign_chance", "bio_D_resid_sf", "bio_regvelo_D_cos"])
    print_block(frame, "Task D — J_phi v vs kinetic-law RHS (internal = train v/G, "
                       "bio_law = true v + G_true)",
                ["int_D_cos", "int_D_cos_c", "bio_law_D_cos", "bio_law_D_cos_c"])
    print_ablation_gaps(frame, ["A_gene_r", "B_foscttm", "C_ari", "bio_D_cos",
                                "bio_D_cos_c", "int_D_cos", "spread"])
    print_seed_summary(frame)

    if args.csv:
        frame.to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
