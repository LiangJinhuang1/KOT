#!/usr/bin/env python3
"""One row per anchor-count rung. Alignment and scale lived in three files;
unanchored β error lived in none — that group decides whether the prior
calibrates the model or only the proteins it names. Split on lr_beta: the
two rates are not the same experiment.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from src.data.beta_anchor import resolve_beta_anchors
from tools.evaluate_checkpoints import (
    build_model_and_tensors,
    collect_run_folders,
    load_dataset_for_run,
)
from tools.evaluate_protein_prediction import (
    anchored_protein_mask,
    format_hparam,
    kinetic_mask,
)

def measured_targets(run_cfg: dict, protein_names: list, d_protein: int):
    """Physical beta targets per protein index, in the model's beta units.

    Resolved through the same call run_kot uses, including its scale, so a target here is
    the number the prior actually pulled toward -- not a re-derived one that would differ
    by whatever beta_anchor_fixed_scale froze.
    """
    indices, betas, _weights, _sigmas, _scale = resolve_beta_anchors(
        Path(run_cfg["beta_anchor_csv"]), protein_names,
        target_mean_beta=float(run_cfg.get("beta_anchor_target", 0.5)),
        aggregate=str(run_cfg.get("beta_anchor_aggregate", "median")),
        stable_cv_max=run_cfg.get("beta_anchor_stable_cv_max"),
        fixed_scale=run_cfg.get("beta_anchor_fixed_scale"),
        min_quality=run_cfg.get("beta_anchor_min_r2"),
    )
    keep = [(j, b) for j, b in zip(indices, betas) if 0 <= j < d_protein]
    return dict(keep)


# Read straight out of the run's diagnostics: alignment on the left of the claim, the
# kinetic scale on the right, and the anchored-side error anchor_diagnostics already scored.
RUN_METRICS = ["mean_foscttm", "val_foscttm", "kappa_median", "beta_median", "alpha_median",
               "jvp_rhs_cos_median", "beta_anchor_mean_abs_err"]


def run_diagnostics(seed_dir: Path, diagnostics_name: str) -> dict | None:
    """The run's finished diagnostics, or None when it wrote no per-protein beta."""
    path = seed_dir / diagnostics_name
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    return None if payload.get("beta_values") is None else payload


def rows_for_run(run_folder: Path, device, diagnostics_name: str):
    """Per-protein rows and per-seed run metrics for every model/dataset under this run."""
    rows, seed_rows = [], []
    for model_dir in sorted(p for p in run_folder.iterdir() if p.is_dir()):
        if model_dir.name in {"cache", "results"} or model_dir.name.startswith("."):
            continue
        for dataset_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            rna_adata, protein_adata, run_cfg, _name = load_dataset_for_run(
                run_folder, dataset_dir.name)
            if not run_cfg.get("beta_anchor_csv"):
                continue
            # Only for the kinetic mask: beta enters the ODE there, so that is where the
            # prior can act at all, and anchored_protein_mask needs it to reproduce the
            # run's own anchor set.
            built = build_model_and_tensors(
                run_cfg, rna_adata, protein_adata, model_dir.name, device,
                velocity_mode="real", mapping_mode="real", with_velocity_backend=False)
            d_protein = built[3].shape[1]
            kin_mask = kinetic_mask(built[6], d_protein)
            protein_names = list(protein_adata.var_names)
            targets = measured_targets(run_cfg, protein_names, d_protein)

            for seed_dir in sorted(p for p in dataset_dir.iterdir()
                                   if p.is_dir() and p.name.startswith("seed_")):
                diagnostics = run_diagnostics(seed_dir, diagnostics_name)
                if diagnostics is None:
                    continue
                beta = np.asarray(diagnostics["beta_values"], dtype=float)
                seed = int(seed_dir.name.split("_")[1])
                seed_rows.append({
                    "dataset": dataset_dir.name,
                    "lr_beta": format_hparam(run_cfg.get("lr_beta")),
                    "beta_anchor_subset_n": format_hparam(run_cfg.get("beta_anchor_subset_n")),
                    "seed": seed,
                    **{key: diagnostics.get(key) for key in RUN_METRICS},
                })
                anchored = anchored_protein_mask(run_cfg, model_dir.name, protein_names,
                                                 d_protein, kin_mask, seed)
                for index, target in targets.items():
                    rows.append({
                        "run": run_folder.name,
                        "dataset": dataset_dir.name,
                        "lr_beta": format_hparam(run_cfg.get("lr_beta")),
                        "lambda_prior": format_hparam(run_cfg.get("lambda_prior")),
                        "lr_warmup_epochs": format_hparam(run_cfg.get("lr_warmup_epochs")),
                        "beta_anchor_subset_n": format_hparam(
                            run_cfg.get("beta_anchor_subset_n")),
                        "seed": seed,
                        "protein": protein_names[index],
                        "in_kinetics": bool(kin_mask[index]),
                        "anchored": bool(anchored[index]),
                        "beta": float(beta[index]),
                        "beta_target": float(target),
                        "abs_err": float(abs(beta[index] - target)),
                    })
            print(f"  {run_folder.name} {dataset_dir.name}: {len(targets)} measured "
                  f"proteins x {len(rows)} rows so far")
    return rows, seed_rows


def claim_table(proteins: pd.DataFrame, seeds: pd.DataFrame) -> pd.DataFrame:
    """One row per (dataset, lr_beta, rung): the alignment side and the scale side together.

    Columns are ordered as the claim reads. `foscttm` and `spearman`-like accuracy belong to
    ALIGNMENT and should not move down the ladder; `kappa_median`, `beta_median` and
    `beta_err_anchored` belong to the KINETIC SCALE and should. The deltas against rung 0 are
    the comparison the claim actually rests on, so they are materialised rather than left for
    the reader to subtract: a scale that moves while alignment does not is a direction the
    data does not constrain.

    kappa_beta is reported because it is what the ODE right-hand side depends on: kappa and
    beta moving in opposite directions with a near-constant product is the degeneracy itself,
    not two independent drifts.
    """
    per_seed = seeds.copy()
    per_seed["kappa_beta"] = per_seed["kappa_median"] * per_seed["beta_median"]

    err = (proteins.groupby(["dataset", "lr_beta", "beta_anchor_subset_n", "anchored"])
           ["abs_err"].mean().unstack("anchored"))
    err.columns = [{False: "beta_err_unanchored", True: "beta_err_anchored_recomputed"}[c]
                   for c in err.columns]

    keys = ["dataset", "lr_beta", "beta_anchor_subset_n"]
    table = (per_seed.groupby(keys)
             .agg(n_seeds=("seed", "nunique"),
                  foscttm=("mean_foscttm", "mean"), foscttm_sd=("mean_foscttm", "std"),
                  jvp_rhs_cos=("jvp_rhs_cos_median", "mean"),
                  kappa_median=("kappa_median", "mean"),
                  beta_median=("beta_median", "mean"),
                  kappa_beta=("kappa_beta", "mean"),
                  beta_err_anchored=("beta_anchor_mean_abs_err", "mean"))
             .join(err).reset_index())

    # Everything is read against the prior-off rung, which is the run the claim compares to.
    table["rung"] = pd.to_numeric(table["beta_anchor_subset_n"], errors="coerce")
    base = table[table["rung"] == 0].set_index(["dataset", "lr_beta"])
    for column in ("foscttm", "kappa_median", "beta_median", "kappa_beta"):
        reference = table.set_index(["dataset", "lr_beta"]).index.map(base[column])
        table[f"{column}_pct_vs_k0"] = 100 * (table[column].to_numpy() / reference - 1)
    return table.sort_values(["dataset", "lr_beta", "rung"]).drop(columns="rung")


def paired_by_protein(frame: pd.DataFrame) -> pd.DataFrame:
    """Each protein compared against ITSELF at rungs where it was not anchored.

    The ladder is nested, so a protein is anchored at the high rungs and not at the low
    ones. Differencing within (dataset, lr_beta, seed, protein) removes the confound that
    proteins with measured half-lives are not a random sample -- without it the anchored
    group looks better simply because it is a different, better-characterised set.
    """
    keys = ["dataset", "lr_beta", "seed", "protein"]
    wide = frame.groupby(keys + ["anchored"])["abs_err"].mean().unstack("anchored")
    wide = wide.dropna()
    wide.columns = ["unanchored", "anchored"]
    wide["delta"] = wide["anchored"] - wide["unanchored"]
    return (wide.groupby(["dataset", "lr_beta"])
            .agg(anchored=("anchored", "mean"), unanchored=("unanchored", "mean"),
                 delta=("delta", "mean"), n=("delta", "size"))
            .reset_index())


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", action="append", default=None,
                        help="Run-folder glob (repeatable).")
    parser.add_argument("--diagnostics", default="diagnostics.json",
                        help="Which diagnostics file to read beta_values from. Default is "
                             "the run's own end-of-training file, matching what "
                             "plot_anchor_degeneracy.py reads, so the two agree.")
    parser.add_argument("--out-prefix", default="cache/results/beta_anchor_fit")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, seed_rows = [], []
    for run_folder in collect_run_folders(args):
        protein_rows, run_rows = rows_for_run(run_folder, device, args.diagnostics)
        rows.extend(protein_rows)
        seed_rows.extend(run_rows)
    if not rows:
        print("No runs with beta_values and a beta_anchor_csv were found.")
        return

    frame = pd.DataFrame(rows)
    claim = claim_table(frame, pd.DataFrame(seed_rows))
    Path(args.out_prefix).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(f"{args.out_prefix}_per_protein.csv", index=False)
    claim.to_csv(f"{args.out_prefix}_summary.csv", index=False)

    show = ["dataset", "lr_beta", "beta_anchor_subset_n", "n_seeds", "foscttm",
            "foscttm_pct_vs_k0", "kappa_median", "kappa_median_pct_vs_k0", "beta_median",
            "kappa_beta_pct_vs_k0", "beta_err_anchored", "beta_err_unanchored"]
    print("\n=== alignment (should not move) vs kinetic scale (should), per rung")
    print(claim[show].round(4).to_string(index=False))
    print(f"\n=== same protein, anchored vs not (removes the protein-identity confound)")
    print(paired_by_protein(frame).round(4).to_string(index=False))
    print(f"\nWrote {args.out_prefix}_per_protein.csv ({len(frame)} rows)")
    print(f"Wrote {args.out_prefix}_summary.csv ({len(claim)} rows)")


if __name__ == "__main__":
    main()
