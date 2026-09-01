from __future__ import annotations

import argparse
import csv
import importlib
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from src.utils.io import load_yaml, save_yaml
from src.utils.arrays import matrix_from_adata
from src.data.preprocessing import load_and_preprocess_cached
from src.data.synthetic_linked_ode import stage_output_paths
from src.data.splits import fitted_row_indices
from src.evaluation.foscttm import (
    calc_domainAveraged_FOSCTTM,
    evaluate_and_save,
    save_prediction_h5ad,
)
from src.evaluation.protocol import (
    HOLDOUT_SECOND,
    plan_fit_restriction,
    protocol_stamp,
    restrict_second_modality,
    scatter_rows,
)
from src.training.registry import (
    BATCH_SPLIT_MODELS,
    COUNT_MODELS,
    MODELS,
    MODEL_OVERRIDES,
    RETIRED_MODELS,
    extra_for,
)
from src.visualization.alignment import (
    coembedding_plot_kwargs,
    plot_coembedding,
    plot_coupling,
    plot_foscttm_distribution,
    plot_foscttm_sorted,
    plot_training_loss,
)
from src.visualization.results import save_comparison_table
from src.visualization.runs import run_flags

DATASETS_CONFIG = Path("config/datasets.yaml")
TRAINING_CONFIG = Path("config/training.yaml")


class Tee:
    """Duplicate writes to multiple streams (e.g., stdout + file)."""

    def __init__(self, *streams, close_streams=()):
        self.streams = streams
        self.close_streams = tuple(close_streams)

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

    def isatty(self):
        return False

    def close(self):
        for stream in self.close_streams:
            stream.close()


def load_model_runner(model_name: str):
    """Import a selected model backend without importing every optional stack."""
    spec = MODELS[model_name]
    try:
        module = importlib.import_module(spec["module"])
    except ModuleNotFoundError as exc:
        extra = extra_for(model_name)
        install_hint = (
            f" Install the optional dependencies with `pip install -e '.[{extra}]'`."
            if extra
            else " Check that the repository's vendor dependencies are available."
        )
        raise ModuleNotFoundError(
            f"Cannot load model '{model_name}' because dependency "
            f"'{exc.name}' is missing.{install_hint}"
        ) from exc
    return getattr(module, spec["function"])


def assert_override_is_effective(model_name: str, resolved: dict, overrides: dict) -> None:
    """A no-op override is a duplicate arm; catch it at launch.
    """
    if not overrides:
        return
    changed = {k: v for k, v in overrides.items() if resolved.get(k) != v}
    if not changed:
        raise ValueError(
            f"Model '{model_name}' declares overrides {sorted(overrides)} but none of "
            f"them differ from the resolved config, so this arm is a duplicate of the "
            f"base model. Either give it a real override or remove it from --models."
        )

STAGED_DATASET = "synthetic_linked_ode"

# Only `mean`: log1p adds a 1/(1+p) factor the kinetics loss does not model,
# so the residual cannot reach zero by being correct and every log run collapsed.
SCALE_LAYERS = {
    "mean": {
        "kot_rna_layer":     "spliced_mean",
        "kot_protein_layer": "protein_mean",
        "velocity_layer":    "true_velocity_mean",
    },
}

def get_repr(adata, obsm_key: str, label: str) -> np.ndarray:
    if adata is None:
        raise ValueError(f"{label} AnnData is None — check dataset config.")
    if obsm_key not in adata.obsm:
        available = list(adata.obsm.keys())
        raise ValueError(f"Key '{obsm_key}' not found in {label} obsm. Available: {available}")
    return np.array(adata.obsm[obsm_key])


def build_context(x, y, rna_adata, second_adata, second_label, modality_pair):
    return {
        "x": x,
        "y": y,
        "rna_adata": rna_adata,
        "second_adata": second_adata,
        "second_label": second_label,
        "modality_pair": modality_pair,
    }


def json_ready_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


# Sweep-readable columns. Everything else stays in diagnostics.json.
SUMMARY_CONFIG_KEYS = [
    "lambda_dyn", "lambda_prior", "lambda_kappa_prior", "kot_kappa_prior",
    "lr", "lr_phi", "lr_alpha_kappa", "lr_beta",
    "lr_warmup_epochs", "lr_warmup_start_factor", "lr_min_factor",
    "n_epochs", "early_stopping_patience", "early_stopping_monitor",
    "batch_size", "sinkhorn_reg",
    "dyn_warmup_epochs", "g_freeze_epochs",
    "kot_velocity_gauge_normalize", "kot_velocity_shuffle",
    "kot_velocity_ablation", "kot_velocity_corrupt", "kot_s_permute", "use_anchor",
    "val_fraction", "val_holdout_from_training",
    "val_split_per_seed", "val_stratify_by", "fit_tuning_subset",
    "fit_obs_key", "save_prediction_cells",
]
SUMMARY_METRIC_KEYS = [
    # Rank on val_foscttm (held-out). mean_foscttm is in-sample — both always written.
    "mean_foscttm", "val_foscttm", "val_foscttm_in_full_pool", "val_holdout",
    "train_foscttm", "train_n_cells", "val_n_cells",
    "runtime_seconds",
    "jvp_rhs_cos_median", "rel_residual_median", "time_spearman", "time_mae",
    "branch_accuracy", "branch_accuracy_branched", "loss_dyn", "loss_align",
    "phi_variance_ratio", "kappa_median", "alpha_median", "beta_mean",
    "grad_norm_median",
    "best_align", "best_align_epoch", "best_val_align", "best_val_align_epoch",
    "best_total", "stop_epoch",
    "n_train_cells", "n_held_out_cells", "n_reported_cells",
    "protocol_oos_mode", "protocol_out_of_distribution", "protocol_paired_oracle",
    "protocol_predictor", "protocol_n_fit_cells",
]


def merged_diagnostics(output_dir: Path, diagnostics: dict) -> dict:
    """Read the file back: run_kot's dict never returns to the runner, so the
    in-memory copy lacks cosine/flags and a dyn-dead run was labelled "ok".
    """
    on_disk = json.loads((output_dir / "diagnostics.json").read_text())
    return {**on_disk, **diagnostics}


def summary_record(name: str, model_name: str, run_seed, run_cfg: dict,
                   diagnostics: dict, output_dir: Path) -> dict:
    """`flag` has to travel with the row: a dyn-dead run still produces a
    respectable FOSCTTM, and a sweep table will not re-derive the verdict.
    """
    record = {
        "dataset": name,
        "model": model_name,
        "seed": "-" if run_seed is None else int(run_seed),
        "run_dir": str(output_dir),
        # "ok" not "": an empty flag is indistinguishable from a pre-flag summary.
        "flag": run_flags(diagnostics) or "ok",
    }
    record.update({key: diagnostics.get(key) for key in SUMMARY_METRIC_KEYS})
    record.update({key: run_cfg.get(key) for key in SUMMARY_CONFIG_KEYS})
    return record


def save_summary(output_dir: Path, record: dict, provenance: dict) -> None:
    """Write run_row.csv first, summary.txt last: the latter is the resume
    marker, so a crash between them retries rather than looking finished.
    """
    with open(output_dir / "run_row.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(record))
        writer.writeheader()
        writer.writerow(record)

    lines = {**record, **provenance}
    width = max(len(key) for key in lines) + 2
    with open(output_dir / "summary.txt", "w") as handle:
        for key, value in lines.items():
            if value is None or value == "":
                continue
            # %g so `lambda_dyn: 1000` not `1000.000000`.
            text = f"{value:g}" if isinstance(value, float) else str(value)
            handle.write(f"{key + ':':{width}}{text}\n")


def save_diagnostics(output_dir: Path, diagnostics: dict) -> None:
    if not diagnostics:
        return
    path = output_dir / "diagnostics.json"
    clean = {str(key): json_ready_value(value) for key, value in diagnostics.items()}
    if path.exists():
        with open(path) as f:
            existing = json.load(f)
        if isinstance(existing, dict):
            clean = {**existing, **clean}
    with open(path, "w") as f:
        json.dump(clean, f, indent=2, sort_keys=True)


def split_model_result(result):
    aligned = result[0]
    coupling = result[1]
    loss_df = None
    diagnostics = {}

    for extra in result[2:]:
        if extra is None:
            continue
        if isinstance(extra, dict):
            diagnostics.update(extra)
        else:
            loss_df = extra

    return aligned, coupling, loss_df, diagnostics


def aggregate_batch_diagnostics(batch_diags: list[dict]) -> dict:
    """Mean of the numeric diagnostics across per-batch runs."""
    out = {}
    keys = set().union(*[d.keys() for d in batch_diags]) if batch_diags else set()
    for k in keys:
        vals = [d[k] for d in batch_diags if isinstance(d.get(k), (int, float, bool))]
        if vals:
            out[k] = float(np.mean(vals))
    out["n_batches"] = len(batch_diags)
    return out


def run_alignment_per_batch(align_fn, x, y, rna_adata, second_adata,
                            second_label, modality_pair, run_cfg, batch_labels, output_dir,
                            fit_rows=None):
    """
    Run an alignment adapter independently within each batch and reassemble.

    FOSCTTM is computed per batch (a global score across independently-aligned
    batches would be meaningless) and returned as one per-cell array in the
    original cell order. Coupling / loss traces are not combined across batches
    (adapter trace files reflect the last batch).

    `fit_rows` restricts the second modality WITHIN each batch, so the holdout survives
    the split: a batch's knockout cells keep their RNA and lose their protein, exactly as
    in the single-shot path. FOSCTTM is then only defined on a batch's fitted cells.
    """
    n = x.shape[0]
    fit_mask = np.zeros(n, dtype=bool)
    fit_mask[np.arange(n) if fit_rows is None else fit_rows] = True
    aligned_rna = None
    aligned_second = None
    foscttm = np.full(n, np.nan, dtype=np.float64)
    batch_diags = []

    for b in np.unique(batch_labels):
        idx = np.where(batch_labels == b)[0]
        fit_idx = idx[fit_mask[idx]]
        if fit_idx.size == 0:
            raise ValueError(
                f"batch {b} has {len(idx)} cells and none of them are fitted; a batch with "
                "no second-modality cells cannot be aligned. Drop the batch or widen "
                "fit_obs_values."
            )
        sub_context = build_context(
            x[idx], y[fit_idx], rna_adata[idx].copy(), second_adata[fit_idx].copy(),
            second_label, modality_pair,
        )
        sub_context["output_dir"] = output_dir
        # Positions of the fitted cells WITHIN this batch, which is what a model needs to
        # pair x against y (the context's x is the batch, not the whole dataset).
        sub_context["fit_rows"] = np.nonzero(fit_mask[idx])[0]
        aligned_b, _, _, diag_b = split_model_result(align_fn(sub_context, run_cfg))
        rb = np.asarray(aligned_b[0], dtype=np.float32)
        pb = np.asarray(aligned_b[1], dtype=np.float32)
        if aligned_rna is None:
            aligned_rna = np.zeros((n, rb.shape[1]), dtype=np.float32)
            aligned_second = np.full((n, pb.shape[1]), np.nan, dtype=np.float32)
        aligned_rna[idx] = rb
        aligned_second[fit_idx] = pb
        batch_diags.append(diag_b)
        if fit_idx.size < 2:
            print(f"    [batch {b}] {fit_idx.size} fitted cell | FOSCTTM undefined; NaN")
        else:
            foscttm[fit_idx] = np.asarray(
                calc_domainAveraged_FOSCTTM(rb[fit_mask[idx]], pb))
            print(f"    [batch {b}] {len(idx)} cells ({fit_idx.size} fitted) | "
                  f"FOSCTTM={np.nanmean(foscttm[fit_idx]):.4f}")

    return [aligned_rna, aligned_second], foscttm.tolist(), aggregate_batch_diagnostics(batch_diags)


def model_names_from_cfg(run_cfg: dict) -> list[str]:
    models = run_cfg.get("models")
    if models is None:
        return [str(run_cfg.get("model", "scot"))]
    if isinstance(models, str):
        return [models]
    return [str(model) for model in models]


def resolve_models(spec, model_groups: dict) -> list[str]:
    """Resolve a --models spec into a list of model names.

    spec may be a group name (looked up in model_groups), a comma-separated
    string of model names, or an already-resolved list.
    """
    if isinstance(spec, list):
        names = spec
    elif spec in model_groups:
        names = model_groups[spec]
    else:
        names = [part.strip() for part in str(spec).split(",") if part.strip()]
    return [str(name) for name in names]


def parse_seed_spec(spec) -> list[int]:
    """Seeds from an int, a list, or a comma-separated string ("42,123,2026")."""
    if isinstance(spec, int):
        return [spec]
    if isinstance(spec, str):
        return [int(part) for part in spec.split(",") if part.strip()]
    return [int(seed) for seed in spec]


def seed_list_from_cfg(run_cfg: dict) -> tuple[list[int], bool]:
    """Return (seeds, is_multi_seed). is_multi_seed True if 'seeds' is explicitly set."""
    seeds = run_cfg.get("seeds")
    if seeds is None:
        return [int(run_cfg.get("seed", 42))], False
    return parse_seed_spec(seeds), True


def seeded_models_from_cfg(run_cfg: dict, all_models: list[str]) -> set[str]:
    """Set of models that should run across all configured seeds."""
    seeded = run_cfg.get("seeded_models")
    if seeded is None:
        return set(all_models) if "seeds" in run_cfg else set()
    if isinstance(seeded, str):
        return {seeded}
    return {str(model) for model in seeded}


def coembedding_input_matrices(x, y, rna_adata, second_adata, cfg):
    mode = str(cfg.get("viz_input", "auto")).lower()
    if mode == "repr":
        return x, y
    if mode == "feature" or (mode == "auto" and cfg.get("kot_use_feature_space")):
        return (
            matrix_from_adata(rna_adata, cfg.get("kot_rna_layer"), "RNA"),
            matrix_from_adata(second_adata, cfg.get("kot_protein_layer"), "Protein"),
        )
    return x, y


def run_one(name: str, dataset: dict, run_cfg: dict) -> None:
    model_name = str(run_cfg.get("model", "scot"))
    if model_name in RETIRED_MODELS:
        raise ValueError(f"Model '{model_name}' was retired. {RETIRED_MODELS[model_name]}")
    if model_name not in MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(MODELS)}")

    run_seed = run_cfg.get("run_seed")
    run_label = f"seed_{run_seed}" if run_seed is not None else None
    output_dir = Path(run_cfg["output_root"]) / model_name / name
    if run_label is not None:
        output_dir = output_dir / run_label

    seed_tag = f" | seed={run_seed}" if run_seed is not None else ""
    # Resume support: a finished run leaves summary.txt → skip it (and its preprocessing).
    if (output_dir / "summary.txt").exists():
        print(f"[{model_name}] already complete on [{name}]{seed_tag} — skipping")
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    # Record the fully-resolved config for THIS individual run (defaults + dataset/--set/
    # model overrides + seed) in the run's own directory. The run-root snapshot only holds
    # the un-resolved training config, so without this a single run dir cannot be traced
    # back to the exact hyperparameters it used.
    save_yaml({"dataset": name, "dataset_paths": dict(dataset), "run_cfg": dict(run_cfg)},
              output_dir / "run_config.yaml")

    print(f"\n{'=' * 50}")
    if run_label is None:
        print(f"[{model_name}] Training on [{name}]")
    else:
        print(f"[{model_name}] Training on [{name}] | seed={run_seed}")
    print(f"{'=' * 50}")

    rna_adata, protein_adata, atac_adata = load_and_preprocess_cached(
        rna_path=dataset["rna_path"],
        protein_path=dataset.get("protein_path"),
        protein_label=dataset.get("protein_label", "ADT"),
        protein_obsm_key=dataset.get("protein_obsm_key"),
        atac_path=dataset.get("atac_path"),
        atac_label=dataset.get("atac_label", "ATAC"),
        rna_raw_layer=dataset.get("rna_raw_layer"),
        rna_umap_path=dataset.get("rna_umap_path"),
        force_recompute=bool(run_cfg.get("force_preprocess", False)),
        cache_version=run_cfg.get("preprocessing_cache_version"),
        rna_min_cells=int(run_cfg.get("rna_min_cells", 3)),
        rna_n_top_genes=int(run_cfg.get("rna_n_top_genes", 2000)),
        rna_n_pcs=int(run_cfg.get("rna_n_pcs", 30)),
        rna_n_neighbors=int(run_cfg.get("rna_n_neighbors", 30)),
        add_log_velocity_layer=bool(run_cfg.get("add_log_velocity_layer", False)),
        log_velocity_scale=float(run_cfg.get("log_velocity_scale", 1.0)),
        protein_min_cells=int(run_cfg.get("protein_min_cells", 1)),
        protein_n_pcs=int(run_cfg.get("protein_n_pcs", 10)),
        atac_min_cells=int(run_cfg.get("atac_min_cells", 3)),
        atac_n_components=int(run_cfg.get("atac_n_components", 30)),
        atac_n_neighbors=int(run_cfg.get("atac_n_neighbors", 30)),
    )

    modality_pair = str(run_cfg.get("modality_pair", "rna_protein"))
    rna_repr_key = str(run_cfg.get("rna_repr", "X_pca"))
    second_repr_key = str(run_cfg.get("second_repr", "X_pca"))

    x = get_repr(rna_adata, rna_repr_key, "RNA")

    if modality_pair == "rna_protein":
        if protein_adata is None:
            raise ValueError(f"Dataset '{name}' is configured for rna_protein but has no protein data.")
        y = get_repr(protein_adata, second_repr_key, "Protein")
        second_adata = protein_adata
        second_label = "protein"
    elif modality_pair == "rna_atac":
        if atac_adata is None:
            raise ValueError(f"Dataset '{name}' is configured for rna_atac but has no ATAC data.")
        y = get_repr(atac_adata, second_repr_key, "ATAC")
        second_adata = atac_adata
        second_label = "atac"
    else:
        raise ValueError(f"Unknown modality_pair: '{modality_pair}'. Use 'rna_protein' or 'rna_atac'.")

    print(f"[runner] RNA repr ({rna_repr_key}): {x.shape}")
    print(f"[runner] {second_label.upper()} repr ({second_repr_key}): {y.shape}")

    context = build_context(x, y, rna_adata, second_adata, second_label, modality_pair)
    context["output_dir"] = output_dir

    # Enforced here for every model. Restricting only KOT used to put OOD KOT next to
    # in-distribution baselines in one table. Modes: src/evaluation/protocol.py.
    requested_rows = fitted_row_indices(rna_adata.obs, run_cfg, run_seed)
    fit_rows, oos_mode = plan_fit_restriction(run_cfg, model_name, requested_rows,
                                              second_label)
    if requested_rows is not None and fit_rows is None:
        print(f"[{model_name}] PAIRED ORACLE: ignores the fit restriction and reads "
              f"{second_label} for every cell. Stamped in-distribution.")
    if fit_rows is not None and oos_mode == HOLDOUT_SECOND:
        context = restrict_second_modality(context, fit_rows)
        print(f"[{model_name}] {oos_mode}: {second_label} restricted to the "
              f"{len(fit_rows)} fitted cells; RNA side keeps all {x.shape[0]}")

    align_fn = load_model_runner(model_name)
    batch_key = run_cfg.get("align_batch_key")
    per_batch = (
        batch_key is not None
        and model_name in BATCH_SPLIT_MODELS
        and batch_key in rna_adata.obs.columns
    )
    start_time = time.perf_counter()
    if per_batch:
        batch_labels = np.asarray(rna_adata.obs[batch_key].values)
        print(f"[{model_name}] per-batch alignment over '{batch_key}' "
              f"({len(np.unique(batch_labels))} batches)")
        aligned, batch_foscttm, diagnostics = run_alignment_per_batch(
            align_fn, x, y, rna_adata, second_adata, second_label, modality_pair,
            run_cfg, batch_labels, output_dir,
            fit_rows=fit_rows if oos_mode == HOLDOUT_SECOND else None,
        )
        coupling, loss_df = None, None
    else:
        aligned, coupling, loss_df, diagnostics = split_model_result(align_fn(context, run_cfg))
        batch_foscttm = None
        # holdout_second returns fitted cells only; scatter back to full cell index.
        if fit_rows is not None and oos_mode == HOLDOUT_SECOND:
            aligned = [aligned[0], scatter_rows(np.asarray(aligned[1]), fit_rows,
                                                x.shape[0])]
    runtime_seconds = time.perf_counter() - start_time
    diagnostics = {
        "dataset": name,
        "model": model_name,
        "runtime_seconds": float(runtime_seconds),
        **diagnostics,
    }
    if run_seed is not None:
        diagnostics["seed"] = int(run_seed)

    result_name = f"{name}_{run_label}" if run_label is not None else name

    # FOSCTTM on fitted cells only, so held-out proteins cannot act as distractors.
    # save_prediction_cells=all still writes φ on knockouts for the perturbation benchmark.
    aligned_full = [np.asarray(part) for part in aligned]
    eval_rows = fit_rows  # same rows as the restriction; a paired oracle has fit_rows=None
    eval_obs_names = rna_adata.obs_names
    aligned_eval = aligned_full
    if eval_rows is not None:
        aligned_eval = [part[eval_rows] for part in aligned_full]
        eval_obs_names = rna_adata.obs_names[eval_rows]
        if batch_foscttm is not None:
            batch_foscttm = np.asarray(batch_foscttm)[eval_rows]
        print(f"[{result_name}] benchmark FOSCTTM on the {len(eval_rows)} fitted cells")

    stamp = protocol_stamp(run_cfg, model_name, aligned_full[0].shape[0], fit_rows)
    diagnostics.update({f"protocol_{key}": value for key, value in stamp.items()
                        if not isinstance(value, list)})
    save_prediction = bool(run_cfg.get("save_prediction_h5ad", True))
    save_all_cells = str(run_cfg.get("save_prediction_cells", "fitted")) == "all"
    mean_foscttm = evaluate_and_save(
        aligned_eval, result_name, output_dir, second_label,
        model_name=model_name,
        obs_names=eval_obs_names,
        foscttm_scores=batch_foscttm,
        save_prediction=save_prediction and not save_all_cells,
        protocol=stamp,
    )
    if save_prediction and save_all_cells:
        n_all = aligned_full[0].shape[0]
        save_prediction_h5ad(
            aligned_full,
            [float("nan")] * n_all,
            mean_foscttm,
            result_name, model_name, second_label, rna_adata.obs_names,
            protocol=stamp,
        )
        print(f"[{result_name}] prediction h5ad keeps all {n_all} cells "
              f"({0 if eval_rows is None else n_all - len(eval_rows)} held out of the loss)")
    diagnostics["mean_foscttm"] = float(mean_foscttm)
    save_diagnostics(output_dir, diagnostics)

    if coupling is not None:
        np.save(output_dir / "coupling.npy", coupling)
    if loss_df is not None:
        loss_csv = output_dir / "training_loss.csv"
        loss_df.to_csv(loss_csv, index=False)
        plot_training_loss(loss_csv, output_dir, model_name)

    provenance = {
        "modality_pair": modality_pair,
        "rna_repr":      f"{rna_repr_key} {x.shape}",
        "second_repr":   f"{second_repr_key} {y.shape}",
    }
    provenance.update({
        key: diagnostics[key] for key in (
            "moscot_converged", "moscot_cost",
            "coupling_entropy", "coupling_effective_sparsity",
        ) if key in diagnostics
    })
    save_summary(
        output_dir,
        summary_record(name, model_name, run_seed, run_cfg,
                       merged_diagnostics(output_dir, diagnostics), output_dir),
        provenance,
    )

    # FOSCTTM plots are modality-agnostic (distribution degrades to a plain
    # histogram without ground-truth branches), so they run on real data too.
    plot_foscttm_distribution(
        output_dir / "foscttm.csv", rna_adata.obs, output_dir, model_name
    )
    plot_foscttm_sorted(output_dir / "foscttm.csv", output_dir, model_name)

    # Coembedding (colours by state/time) and coupling (needs time) require the
    # synthetic ground-truth columns.
    has_ground_truth = {"state", "time"}.issubset(rna_adata.obs.columns)
    if has_ground_truth:
        viz_x, viz_y = coembedding_input_matrices(x, y, rna_adata, second_adata, run_cfg)
        plot_coembedding(
            viz_x, viz_y, aligned[0], aligned[1], rna_adata.obs, output_dir, model_name,
            rna_adata=rna_adata,
            second_adata=second_adata,
            **coembedding_plot_kwargs(run_cfg),
        )
        if coupling is not None:
            plot_coupling(coupling, rna_adata.obs["time"].values, output_dir, model_name)

    print(f"[runner] results saved to {output_dir}/")


def train_all_datasets(
    config_path: Path = TRAINING_CONFIG,
    datasets_config_path: Path | None = None,
    stage: str | None = None,
    scale: str | None = None,
    run_dir: Path | None = None,
    models: str | None = None,
    datasets_filter: str | None = None,
    set_overrides: dict | None = None,
) -> None:
    train_cfg = load_yaml(config_path)
    if datasets_config_path is None:
        sibling_config = Path(config_path).with_name("datasets.yaml")
        datasets_config_path = sibling_config if sibling_config.exists() else DATASETS_CONFIG
    datasets = load_yaml(datasets_config_path).get("datasets", {})
    defaults = train_cfg.get("defaults", {})
    stage_overrides_cfg = train_cfg.get("stage_overrides", {})
    cli_overrides = dict(set_overrides or {})

    # Reject typo'd --set keys: a plain dict merge would silently inject dead keys
    # (e.g. `beta_anchor_prio=gaussian`), so validate against every key that appears
    # in the config's defaults, per-dataset overrides, stage overrides, and the
    # runtime keys the runner sets itself.
    known_keys = set(defaults)
    for ov in train_cfg.get("datasets", {}).values():
        known_keys.update((ov or {}).keys())
    for ov in stage_overrides_cfg.values():
        known_keys.update((ov or {}).keys())
    for layers in SCALE_LAYERS.values():
        known_keys.update(layers.keys())
    for ov in MODEL_OVERRIDES.values():
        known_keys.update(ov.keys())
    known_keys.update({"model", "seed", "run_seed", "models", "preprocessing_cache_version"})
    unknown = set(cli_overrides) - known_keys
    if unknown:
        raise ValueError(
            f"--set has unknown config key(s): {sorted(unknown)}. "
            f"Check for typos; known keys include: {sorted(known_keys)}"
        )

    selected_datasets = (
        set(name.strip() for name in datasets_filter.split(",")) if datasets_filter else None
    )

    model_groups = train_cfg.get("model_groups", {})
    models_spec = models if models is not None else train_cfg.get("default_models")
    selected_models = (
        resolve_models(models_spec, model_groups) if models_spec is not None else None
    )

    # Resume reuses the dir; run_one skips arms that already have summary.txt.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_root = Path(defaults.get("output_root", "cache/training"))
    run_root = Path(run_dir) if run_dir is not None else base_root / f"run_{timestamp}"
    run_root.mkdir(parents=True, exist_ok=True)
    defaults["output_root"] = str(run_root)

    snapshot_name = f"training_resume_{timestamp}.yaml" if run_dir is not None else "training.yaml"
    log_name = f"run_resume_{timestamp}.log" if run_dir is not None else "run.log"
    # Resolved config, not a copy of the file: shutil.copy would drop --set.
    config_snapshot = run_root / snapshot_name
    snapshot_cfg = dict(train_cfg)
    if cli_overrides:
        snapshot_cfg["_cli_overrides"] = dict(cli_overrides)
    save_yaml(snapshot_cfg, config_snapshot)

    log_path = run_root / log_name
    log_file = open(log_path, "w", buffering=1)
    sys.stdout = Tee(sys.stdout, log_file, close_streams=(log_file,))
    sys.stderr = Tee(sys.stderr, log_file)

    print(f"[runner] Run timestamp: {timestamp}")
    print(f"[runner] Output root:   {run_root}{'  (resume)' if run_dir is not None else ''}")
    print(f"[runner] Config saved:  {config_snapshot}")
    print(f"[runner] Log file:      {log_path}")
    if cli_overrides:
        print(f"[runner] --set overrides: {cli_overrides}")

    trained_models = []
    result_seeded_models = set()
    trained_datasets = []

    for name, overrides in train_cfg.get("datasets", {}).items():
        if selected_datasets is not None and name not in selected_datasets:
            continue
        if name not in datasets:
            raise ValueError(f"Dataset '{name}' is not defined in {datasets_config_path}.")
        if name not in trained_datasets:
            trained_datasets.append(name)
        dataset = datasets[name]
        overrides = dict(overrides or {})

        # --stage / --scale override the staged synthetic: data path, KOT feature
        # layers, per-stage models/anchors (stage_overrides), and the cache version.
        if name == STAGED_DATASET and stage is not None:
            eff_scale = scale or "mean"
            rna_path, protein_path = stage_output_paths(stage)
            dataset = {**dataset, "rna_path": rna_path, "protein_path": protein_path}
            overrides.update(stage_overrides_cfg.get(stage, {}))
            overrides.update(SCALE_LAYERS[eff_scale])
            overrides["preprocessing_cache_version"] = f"synthetic_linked_ode_{stage}_{eff_scale}"
            print(f"[runner] stage={stage} scale={eff_scale} | rna={rna_path}")

        # --set / --seeds are folded in HERE, not later at the per-model merge: the
        # seed list and the seeded-model set are read off base_cfg below, so an
        # override applied after this point would be accepted, written into
        # run_config.yaml, and still have no effect on which seeds actually run.
        base_cfg = {**defaults, **overrides, **cli_overrides}

        # Fail fast on a missing anchor CSV: a wrong path (e.g. a typo'd filename)
        # would otherwise fall through to a silent no-anchor run.
        effective_csv = base_cfg.get("beta_anchor_csv")
        if effective_csv and not Path(effective_csv).exists():
            raise FileNotFoundError(
                f"beta_anchor_csv for dataset '{name}' does not exist: {effective_csv}"
            )

        if selected_models is not None:
            base_cfg["models"] = selected_models
        all_models = model_names_from_cfg(base_cfg)

        # The oracle stage has no integer counts → NB-likelihood baselines can't run there.
        if stage == "oracle":
            all_models = [m for m in all_models if m not in COUNT_MODELS]

        seeded = seeded_models_from_cfg(base_cfg, all_models)
        result_seeded_models.update(seeded)

        multi_seeds, is_multi_seed = seed_list_from_cfg(base_cfg)
        default_seeds = [int(base_cfg.get("seed", 42))]

        for model_name in all_models:
            use_run_seed = is_multi_seed and model_name in seeded
            seeds = multi_seeds if model_name in seeded else default_seeds
            for seed in seeds:
                # Order: dataset defaults → --set → per-model overrides (kot_nodyn
                # still forces lambda_dyn=0 even if --set lambda_dyn=N). base_cfg
                # already carries --set.
                pre_override = {**base_cfg, "model": model_name, "seed": seed}
                model_overrides = MODEL_OVERRIDES.get(model_name, {})
                assert_override_is_effective(model_name, pre_override, model_overrides)
                run_cfg = {**pre_override, **model_overrides}
                if use_run_seed:
                    run_cfg["run_seed"] = seed
                run_one(name, dataset, run_cfg)
            if model_name not in trained_models:
                trained_models.append(model_name)

    for name in trained_datasets:
        save_comparison_table(
            cache_dir=run_root,
            dataset=name,
            save_dir=run_root / "results",
            model_names=trained_models or None,
            seeded_models=sorted(result_seeded_models) or None,
        )


def run_training(
    config_path: Path = TRAINING_CONFIG,
    datasets_config_path: Path | None = None,
    stage: str | None = None,
    scale: str | None = None,
    run_dir: Path | None = None,
    models: str | None = None,
    datasets_filter: str | None = None,
    set_overrides: dict | None = None,
) -> None:
    """Run training while keeping process-global output streams well scoped."""
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    try:
        train_all_datasets(
            config_path=config_path,
            datasets_config_path=datasets_config_path,
            stage=stage,
            scale=scale,
            run_dir=run_dir,
            models=models,
            datasets_filter=datasets_filter,
            set_overrides=set_overrides,
        )
    finally:
        active_streams = (sys.stdout, sys.stderr)
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        for stream in active_streams:
            if stream not in (original_stdout, original_stderr) and isinstance(stream, Tee):
                stream.close()


INT_PATTERN = re.compile(r"[+-]?\d+$")
FLOAT_PATTERN = re.compile(r"[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?$")


def parse_set_value(raw: str):
    """Parse a --set value: bool / int / float / else string."""
    key = raw.strip()
    low = key.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none"):
        return None
    if INT_PATTERN.match(key):
        return int(key)
    if FLOAT_PATTERN.match(key):
        return float(key)
    return key


def parse_set_overrides(items: list[str] | None) -> dict:
    """Parse repeated --set key=value into a dict (last wins on duplicate keys)."""
    if not items:
        return {}
    out = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got: {item!r}")
        key, raw = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"--set missing key in: {item!r}")
        out[key] = parse_set_value(raw)
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run KOT / baseline training from config.")
    parser.add_argument("--config", type=Path, default=TRAINING_CONFIG,
                        help="Training config YAML.")
    parser.add_argument(
        "--datasets-config",
        type=Path,
        default=None,
        help="Dataset config YAML. Defaults to datasets.yaml beside --config, then "
             "config/datasets.yaml.",
    )
    parser.add_argument("--stage", choices=["oracle", "clean", "branch"], default=None,
                        help="Staged synthetic: overrides data path, feature layers, "
                             "anchor indices, and the preprocessing cache version.")
    parser.add_argument("--scale", choices=sorted(SCALE_LAYERS), default=None,
                        help="KOT feature scale for the staged synthetic dataset. Only "
                             "'mean' is supported: it is the space the linked ODE is "
                             "generated in, so the kinetics term is correctly specified.")
    parser.add_argument("--models", type=str, default=None,
                        help="Models to run: a group name from the config's model_groups "
                             "(e.g. baselines, kot, all) or a comma-separated list of model "
                             "names. Defaults to the config's default_models.")
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="Resume into an existing run_* dir: only run (model, seed) "
                             "outputs that are missing, instead of a new timestamped dir.")
    parser.add_argument("--datasets", type=str, default=None,
                        help="Comma-separated subset of the config's datasets to run "
                             "(e.g. bmmc_cite). Defaults to every dataset in the config.")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated seeds for the stochastic models, e.g. "
                             "--seeds 42,123,2026. Overrides the dataset's `seeds:` list "
                             "for this run only, so a sweep can change seed count without "
                             "editing config/training.yaml (which would race with jobs "
                             "still launching).")
    parser.add_argument("--no-predictions", action="store_true",
                        help="Skip the global data/predictions/*.h5ad write. Use this for "
                             "every parallel sweep: that path is keyed only by model + "
                             "dataset + seed, so runs differing only in hyperparameters "
                             "collide on the HDF5 file lock and die.")
    parser.add_argument(
        "--set",
        dest="set_overrides",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Override a config key (repeatable), e.g. --set lambda_dyn=10. "
             "Applied after dataset defaults; per-model overrides (e.g. kot_nodyn "
             "forces lambda_dyn=0) still win.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Console entry point for installed and module-based execution."""
    args = parse_args(argv)
    overrides = parse_set_overrides(args.set_overrides)
    if args.seeds is not None:
        overrides["seeds"] = parse_seed_spec(args.seeds)
    if args.no_predictions:
        overrides["save_prediction_h5ad"] = False
    run_training(
        config_path=args.config,
        datasets_config_path=args.datasets_config,
        stage=args.stage,
        scale=args.scale,
        run_dir=args.run_dir,
        models=args.models,
        datasets_filter=args.datasets,
        set_overrides=overrides,
    )


if __name__ == "__main__":
    main()
