#!/usr/bin/env python3
"""Score φ(r) on held-out cells. FOSCTTM is rank-only and can win while every
value is the wrong scale. Subsets differ by one ingredient (alignment vs
kinetics vs anchor) so a median over the whole panel cannot hide the gap.

  python tools/evaluate_protein_prediction.py --runs 'cache/training/run_*'
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

from src.data.beta_anchor import resolve_beta_anchors
from src.training.kot import (
    anchor_subset_indices,
    jvp_and_rhs,
    per_cell_cosine,
    resolve_velocity_ablation,
)
from tools.evaluate_checkpoints import (
    build_model_and_tensors,
    cfg_with_model_overrides,
    collect_run_folders,
    load_dataset_for_run,
)
from tools.summarize import KEY_COLUMNS, fitted_side

# summary.csv's config keys, then the ablation axes that would otherwise collapse
# two arms into one row (see tools/summarize.py runs --extra).
IDENTITY_COLUMNS = KEY_COLUMNS + ["kot_velocity_ablation", "kot_s_permute",
                                  "beta_anchor_subset_n"]
EVAL_COLUMNS = ["seed", "checkpoint", "dyn_velocity", "dyn_mapping", "n_eval_cells",
                "held_out"]

EPS = 1e-12
QUANTILES = (0.25, 0.50, 0.75)


def format_hparam(value):
    """Compact spelling used by cache/results/summary.csv (0.001, not 0.0010)."""
    if value is None or value == "":
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return format(value, "g")
    return str(value)


def config_identity(run_cfg: dict, dataset: str) -> dict:
    """summary.csv's identifying columns, plus the ablation axes that separate arms."""
    return {
        "dataset": dataset,
        "fitted_side": fitted_side(run_cfg),
        "lr_phi": format_hparam(run_cfg.get("lr_phi")),
        "lr_alpha_kappa": format_hparam(run_cfg.get("lr_alpha_kappa")),
        "lr_beta": format_hparam(run_cfg.get("lr_beta")),
        "lambda_dyn": format_hparam(run_cfg.get("lambda_dyn")),
        "lr_warmup_epochs": format_hparam(run_cfg.get("lr_warmup_epochs")),
        "sinkhorn_reg": format_hparam(run_cfg.get("sinkhorn_reg")),
        "kot_velocity_ablation": resolve_velocity_ablation(run_cfg),
        "kot_s_permute": format_hparam(bool(run_cfg.get("kot_s_permute", False))),
        # The anchor-count ladder differs ONLY in this key, so without it every rung
        # collapses into one group and the ablation compares itself.
        "beta_anchor_subset_n": format_hparam(run_cfg.get("beta_anchor_subset_n")),
    }


def ordered_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Identity block first (same order as summary.csv), then the evaluation columns."""
    front = IDENTITY_COLUMNS + EVAL_COLUMNS
    return frame[[c for c in front if c in frame.columns]
                 + [c for c in frame.columns if c not in front]]


def column_pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson r per COLUMN of two equally shaped matrices."""
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    denom = np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0)
    return (a * b).sum(axis=0) / (denom + EPS)


def column_spearman(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Spearman rho per column: Pearson on ranks, vectorised over columns.

    rankdata(axis=0) ranks each column independently, which is what makes this one call
    instead of a Python loop over the panel.
    """
    return column_pearson(rankdata(a, axis=0).astype(np.float64),
                          rankdata(b, axis=0).astype(np.float64))


def column_nrmse(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """RMSE per column, normalised by that column's observed sd.

    Normalising by the TRUE sd makes the number comparable across proteins of different
    dynamic range: 1.0 means the error is as large as the biological spread, i.e. no
    better than predicting the protein's mean for every cell.
    """
    rmse = np.sqrt(((pred - true) ** 2).mean(axis=0))
    return rmse / (true.std(axis=0) + EPS)


def row_pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson r per row -- cosine after centring each cell's own profile."""
    return per_cell_cosine(a - a.mean(axis=1, keepdims=True),
                           b - b.mean(axis=1, keepdims=True))


def quantile_summary(values: np.ndarray, prefix: str) -> dict:
    """median + IQR + n, the reporting shape used across this project's diagnostics."""
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return {f"{prefix}_{k}": float("nan") for k in ("median", "q25", "q75", "mean")} | {
            f"{prefix}_n": 0}
    q25, median, q75 = np.quantile(clean, QUANTILES)
    return {f"{prefix}_median": float(median), f"{prefix}_q25": float(q25),
            f"{prefix}_q75": float(q75), f"{prefix}_mean": float(clean.mean()),
            f"{prefix}_n": int(clean.size)}


def kinetic_mask(mask_t, d_protein: int) -> np.ndarray:
    """Proteins in the ODE residual. No mask means every protein participates."""
    if mask_t is None:
        return np.ones(d_protein, dtype=bool)
    return mask_t.cpu().numpy() != 0


def anchored_protein_mask(run_cfg: dict, model_name: str, protein_names: list,
                          d_protein: int, kin_mask: np.ndarray, seed: int) -> np.ndarray:
    """Proteins this run, at this seed, actually put a beta prior on.

    Every filter run_kot applies is applied here in the same order, because an anchored
    subset that is not the run's own anchor set is not a control:
      * the model's overrides first -- kot_noanchor and the oracle arms set
        use_anchor=False, and reading the CSV regardless would report an anchored /
        unanchored split for a run that anchored nothing;
      * the CSV set when one resolves, else the manual index list (run_kot keeps the
        manual list only when the CSV yields nothing);
      * intersected with the kinetic mask -- beta enters the ODE only there, so an
        anchor outside it constrains nothing;
      * then thinned by beta_anchor_subset_n, the anchor-count ablation, which draws its
        subset PER SEED. Without this every rung of that ladder (k00..k53) would report
        the full active set, k00 included, and the ablation would compare identical
        subsets.
    """
    run_cfg = cfg_with_model_overrides(run_cfg, model_name)
    mask = np.zeros(d_protein, dtype=bool)
    if not bool(run_cfg.get("use_anchor", False)):
        return mask

    indices = []
    csv_path = run_cfg.get("beta_anchor_csv")
    if csv_path and Path(csv_path).exists():
        indices, _betas, _w, _s, _scale = resolve_beta_anchors(
            Path(csv_path), protein_names,
            target_mean_beta=float(run_cfg.get("beta_anchor_target", 0.5)),
            aggregate=str(run_cfg.get("beta_anchor_aggregate", "median")),
            stable_cv_max=run_cfg.get("beta_anchor_stable_cv_max"),
            fixed_scale=run_cfg.get("beta_anchor_fixed_scale"),
            min_quality=run_cfg.get("beta_anchor_min_r2"),
        )
    if not indices:
        indices = list(run_cfg.get("kot_anchor_indices", []) or [])

    active = [j for j in indices if 0 <= j < d_protein and kin_mask[j]]
    subset_n = run_cfg.get("beta_anchor_subset_n")
    if subset_n is not None:
        active = [active[k] for k in anchor_subset_indices(len(active), int(subset_n), seed)]
    mask[active] = True
    return mask


def subset_masks(kin_mask: np.ndarray, align_mask: np.ndarray,
                 anchor_mask: np.ndarray) -> dict:
    """The protein subsets each metric is reported over.

    Each contrast is a split of the term above it, never two arbitrary halves of the
    panel. `kinetics` vs `alignment_only` splits the ALIGNED panel, so the two differ by
    the ODE residual alone -- taking `kinetics` over the whole array instead would make
    the comparison hold only for the proteins the two masks happen to share. `anchored`
    vs `kinetic_unanchored` splits the KINETIC panel for the same reason: beta appears
    only in the residual, so an anchor on a protein outside it constrains nothing and
    would otherwise pad both anchor subsets with proteins the prior never touched.
    """
    return {
        "measured_panel": np.ones_like(kin_mask),
        "alignment": align_mask,
        "kinetics": align_mask & kin_mask,
        "alignment_only": align_mask & ~kin_mask,
        "anchored": kin_mask & anchor_mask,
        "kinetic_unanchored": kin_mask & ~anchor_mask,
    }


def evaluate_one_seed(model, ckpt_path: Path, R_t, V_eff_t, P_t, S_t, mask_t,
                      align_cols, val_idx, protein_names: list,
                      anchor_mask: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """Per-protein rows and the per-cell summary for one checkpoint."""
    state = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    model.load_state_dict({k: v.to(R_t.device) for k, v in state.items()})
    model.eval()

    rows = val_idx if val_idx is not None else torch.arange(R_t.shape[0], device=R_t.device)
    R_eval, P_eval = R_t[rows], P_t[rows]

    # phi under no_grad; the JVP below needs autograd and is taken separately.
    with torch.no_grad():
        pred = model.phi(R_eval).cpu().numpy().astype(np.float64)
    true = P_eval.cpu().numpy().astype(np.float64)

    d_protein = true.shape[1]
    kin_mask = kinetic_mask(mask_t, d_protein)
    align_mask = np.zeros(d_protein, bool)
    if align_cols is None:
        align_mask[:] = True
    else:
        align_mask[align_cols.cpu().numpy()] = True

    frame = pd.DataFrame({
        "protein": protein_names[:d_protein],
        "in_kinetics": kin_mask,
        "in_alignment": align_mask,
        "is_anchored": anchor_mask,
        "spearman": column_spearman(pred, true),
        "pearson": column_pearson(pred, true),
        "nrmse": column_nrmse(pred, true),
        "true_sd": true.std(axis=0),
    })

    # Per-protein ODE agreement, on the same held-out cells. Only the kinetic panel: a
    # masked protein has both sides forced to zero, so its correlation is meaningless.
    _phi, dphi_dv, rhs, _k, _a = jvp_and_rhs(model, R_eval, V_eff_t[rows], S_t, mask_t)
    jvp_np = dphi_dv.detach().cpu().numpy().astype(np.float64)
    rhs_np = rhs.detach().cpu().numpy().astype(np.float64)
    frame["dyn_pearson"] = column_pearson(jvp_np, rhs_np)
    frame["dyn_spearman"] = column_spearman(jvp_np, rhs_np)
    frame["dyn_nrmse"] = column_nrmse(jvp_np, rhs_np)
    frame["dyn_sign_agreement"] = (np.sign(jvp_np) == np.sign(rhs_np)).mean(axis=0)
    frame.loc[~frame["in_kinetics"], ["dyn_pearson", "dyn_spearman", "dyn_nrmse",
                                      "dyn_sign_agreement"]] = np.nan

    # Per-cell profile accuracy, computed within each protein subset: a cell's profile
    # over the kinetic panel is a different question from its profile over the whole panel.
    per_cell = {}
    for name, mask in subset_masks(kin_mask, align_mask, anchor_mask).items():
        if mask.sum() < 2:      # a correlation across one protein is not defined
            continue
        per_cell[name] = {
            **quantile_summary(per_cell_cosine(pred[:, mask], true[:, mask]), "cell_cosine"),
            **quantile_summary(row_pearson(pred[:, mask], true[:, mask]), "cell_pearson"),
        }
    return frame, per_cell


def summarize(frame: pd.DataFrame, per_cell: dict, meta: dict) -> list[dict]:
    """One row per protein subset: the per-protein metric quantiles plus the per-cell pair."""
    kin = frame["in_kinetics"].to_numpy()
    align = frame["in_alignment"].to_numpy()
    anchor = frame["is_anchored"].to_numpy()
    out = []
    for name, mask in subset_masks(kin, align, anchor).items():
        if mask.sum() == 0:
            continue
        sub = frame[mask]
        row = {**meta, "subset": name, "n_proteins": int(mask.sum())}
        for metric in ("spearman", "pearson", "nrmse", "dyn_pearson", "dyn_spearman",
                       "dyn_nrmse", "dyn_sign_agreement"):
            row.update(quantile_summary(sub[metric].to_numpy(), metric))
        row.update(per_cell.get(name, {}))
        out.append(row)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", action="append", default=None,
                        help="Run-folder glob (repeatable).")
    parser.add_argument("--checkpoint", default="best_align")
    parser.add_argument(
        "--velocity", choices=["real", "as-trained"], default="real",
        help="Velocity field the dyn_* columns are measured against. The per-protein and "
             "per-cell protein metrics do not use velocity at all (phi is a function of "
             "RNA), so this never affects them. 'real' scores every arm's JVP against the "
             "TRUE field, which is what makes dyn_* comparable ACROSS arms -- and is the "
             "only option for the shuffle arm, whose field is drawn per seed and cannot "
             "be rebuilt outside the seed loop.")
    parser.add_argument(
        "--mapping", choices=["real", "as-trained"], default="real",
        help="RNA->protein map S the dyn_* RHS is built from -- the same reference "
             "question as --velocity, on the other input. 'real' scores an S-permuted arm "
             "against the TRUE map, which is the only option for that control (its S is "
             "drawn per seed) and is what makes dyn_* comparable across arms.")
    parser.add_argument("--out-prefix", default="cache/results/protein_eval")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    protein_rows, summary_rows = [], []

    for run_folder in collect_run_folders(args):
        for model_dir in sorted(p for p in run_folder.iterdir() if p.is_dir()):
            if model_dir.name in {"cache", "results"} or model_dir.name.startswith("."):
                continue
            for dataset_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
                rna_adata, protein_adata, run_cfg, _name = load_dataset_for_run(
                    run_folder, dataset_dir.name)
                built = build_model_and_tensors(
                    run_cfg, rna_adata, protein_adata, model_dir.name, device,
                    velocity_mode=args.velocity, mapping_mode=args.mapping,
                    with_velocity_backend=False)
                model, R_t, V_eff_t, P_t, S_t, _obs, mask_t, align_cols, _vb, val_idx = built
                protein_names = list(protein_adata.var_names)
                kin_mask = kinetic_mask(mask_t, P_t.shape[1])
                if val_idx is None:
                    print(f"[warn] {run_folder.name}: no validation split; scoring ALL "
                          f"cells (in-sample)")
                for seed_dir in sorted(p for p in dataset_dir.iterdir()
                                       if p.is_dir() and p.name.startswith("seed_")):
                    ckpt = seed_dir / f"checkpoint_{args.checkpoint}.pt"
                    if not ckpt.exists():
                        continue
                    seed = int(seed_dir.name.split("_")[1])
                    # Per seed: beta_anchor_subset_n draws a different subset for each.
                    anchor_mask = anchored_protein_mask(
                        run_cfg, model_dir.name, protein_names, P_t.shape[1],
                        kin_mask, seed)
                    meta = {
                        **config_identity(run_cfg, dataset_dir.name),
                        "seed": seed,
                        "checkpoint": args.checkpoint,
                        "dyn_velocity": args.velocity,
                        "dyn_mapping": args.mapping,
                        "n_eval_cells": int(len(val_idx) if val_idx is not None
                                            else R_t.shape[0]),
                        "held_out": val_idx is not None,
                    }
                    frame, per_cell = evaluate_one_seed(
                        model, ckpt, R_t, V_eff_t, P_t, S_t, mask_t, align_cols,
                        val_idx, protein_names, anchor_mask)
                    for key, value in meta.items():
                        frame[key] = value
                    protein_rows.append(frame)
                    summary_rows.extend(summarize(frame, per_cell, meta))
                    print(f"  {run_folder.name} {model_dir.name} seed "
                          f"{meta['seed']}: {meta['n_eval_cells']} cells, "
                          f"{len(frame)} proteins")

    if not protein_rows:
        print("No checkpoints found.")
        return
    per_protein = ordered_columns(pd.concat(protein_rows, ignore_index=True))
    summary = ordered_columns(pd.DataFrame(summary_rows))
    Path(args.out_prefix).parent.mkdir(parents=True, exist_ok=True)
    per_protein.to_csv(f"{args.out_prefix}_per_protein.csv", index=False)
    summary.to_csv(f"{args.out_prefix}_summary.csv", index=False)
    print(f"\nWrote {args.out_prefix}_per_protein.csv ({len(per_protein)} rows)")
    print(f"Wrote {args.out_prefix}_summary.csv ({len(summary)} rows)")


if __name__ == "__main__":
    main()
