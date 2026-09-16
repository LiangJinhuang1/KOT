#!/usr/bin/env python3
"""KOT on chromatin→RNA: the same Jacobian push-forward, a different modality pair.

The frozen RNA→protein experiment learns phi: RNA→protein and reads its Jacobian along
an RNA velocity. This asks whether the principle survives changing BOTH the source
modality and the target kinetic law:

    chromatin --phi--> RNA,   J_phi(c) v_c  ~  the transcription law

Everything the two experiments share is imported from `src` (the phi/kappa/alpha
networks, the Sinkhorn divergence, the JVP, the optimiser and the LR schedule). Nothing
in `src/training/kot.py` is touched — that script's results are frozen.

Stages, in the order they have to be run:

  prepare   freeze a split, then fit gene selection/LSI on its allowed training cells
  audit     §3 — print and save what the data actually contains; refuse relay without u/s
  split     §4 — verify the disjoint split frozen before preprocessing by prepare
  velocity  §7/§8 — chromatin velocity: temporal OT for HSPC, a directional field for BMMC
  reference §17 RNA-only scVelo on the prepared panel (never a training input)
  train     regulatory R2 only (chromatin→unspliced→spliced)
  preflight recompute the launch gate on a checkpoint after reference exists
  evaluate  §13-§17 — Tasks A-D on --eval-split val|test; development uses paired val,
            and test is scored after the configuration is frozen

The active law (linear RNA abundances in a common normalization):

  R2  Phi(c) = [u, s], with
      du = kappa(c) [alpha(Gc) - beta (*) u]
      ds = kappa(c) [beta (*) u        - gamma (*) s]
  Align on s; constrain auxiliary u by its unpaired marginal; anchor gamma to RNA decay.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.data.chromatin import (
    CHROMATIN_TRANSFORMS,
    PREPROCESSING_PROTOCOL,
    audit_report,
    build_bmmc,
    build_hspc,
    chromatin_features,
    gene_map,
    hspc_features_tsv,
    hspc_peak_annotation,
    make_splits,
    require_relay_inputs,
    usable_splicing_genes,
    validate_preprocessing_split,
)
from src.data.chromatin_map import (GENE_MAP_MODES, assert_no_orphan_g_rows,
                                    gene_loci_from_features,
                                    permute_chromatin_projection,
                                    widen_projection)
from src.data.chromatin_r2 import (
    SHARED_SPLICED, global_linear_scale, load_gamma_anchors, gamma_anchor_loss,
    panel_library_totals, splicing_targets,
)
from src.data.chromatin_velocity import (
    VELOCITY_ESTIMATORS,
    bmmc_velocity,
    direction_diversity,
    euler_lag,
    gauge_normalize,
    hspc_velocity,
)
from src.data.regvelo_backend import run_regvelo
from src.data.velocity import preprocess_for_velocity
from src.evaluation.chromatin_eval import (
    by_group,
    foscttm_within,
    task_a_state,
    task_b_pairing,
    task_c_integration,
    task_d_kinetics,
)
from src.evaluation.foscttm import calc_domainAveraged_FOSCTTM, permuted_pairing_floor
from src.losses.chromatin_laws import (
    GLOBAL_LINEAR,
    LOG1P_MAX,
    LOG1P_MIN,
    LAWS,
    PANEL_LOG1P_COMPOSITIONAL,
    RNA_KINETIC_COORDS,
    REDUCED,
    RELAY_CONDITIONS,
    FORMULATION,
    RELAY,
    SHARED_LOG1P,
    alignment_gauge,
    block_scales,
    conditions_for_law,
    kinetics_loss,
    kinetics_losses,
    kinetics_residual,
    law_mask,
    law_rhs,
    model_kinetic_coords,
    resolve_kinetic_coords,
    state_rate_to_linear,
)
from src.losses.sinkhorn import sinkhorn_divergence
from src.models.chromatin_kot import ALPHA_INPUTS, PHI_GATES, ChromatinKOT
from src.training.chromatin_baselines import (
    BASELINES,
    BaselineInputs,
    impute_from_latent,
)
from src.training.kot import (
    FixedKappa,
    choose_torch_device,
    dyn_weight_factor,
    kappa_prior_loss,
    lr_schedule_factor,
    shuffled_batches,
    subsample_rows,
)
from src.utils.arrays import to_dense
from src.utils.io import load_yaml

CACHE_ROOT = PROJECT_ROOT / "cache" / "chromatin"
HSPC_SOURCE = PROJECT_ROOT / "Datasets" / "HSPC_GSE209878"
BMMC_PROCESSED = (PROJECT_ROOT / "Datasets" / "BMMC"
                  / "GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad")
# Separate cache name so the older incomplete BMMC velocity file is not overwritten.
BMMC_VELOCITY = (PROJECT_ROOT / "cache" / "velocity" / "bmmc_multiome_full"
                 / "bmmc_multiome_full_scvelo_results.h5ad")

DATASETS = ["hspc", "bmmc"]
VELOCITY_PROTOCOL_VERSION = 4
STABILIZATION_MODES = ("legacy", "kot_parity")
# RNA→protein kot.py: λ_κ=0.01 toward log(2), λ_reg=1e-4, clip 1. Chromatin's
# historical trainer had none of those and clipped at 5. The mode is the test.
KOT_KAPPA_PRIOR_TARGET = math.log(2)
# Alpha's G c need not live in phi's coordinates. `same` is the historical coupling.
REGULATORY_TRANSFORMS = ("same", "cp10k_linear")


def regulatory_features(adata, chromatin_transform: str, regulatory_transform: str,
                        map_features: np.ndarray | None = None) -> np.ndarray:
    """Gene-activity coordinates the transcription head reads through G.

    phi and v_c stay in `chromatin_transform`. `same` reuses those coordinates
    (including a lagged map, if training applied one). `cp10k_linear` is the
    interpretable gene-activity scale and is never lagged with phi.
    """
    name = regulatory_transform or "same"
    if name not in REGULATORY_TRANSFORMS:
        raise ValueError(
            f"regulatory-transform must be one of {REGULATORY_TRANSFORMS}, got {name!r}")
    if name == "same":
        if map_features is None:
            return chromatin_features(adata, chromatin_transform)
        return map_features
    return chromatin_features(adata, name)


def rna_protein_seeds(config_path: Path | None = None) -> list[int]:
    """The seed list the RNA→protein experiment runs, read from ITS config.

    The chromatin experiment asks whether the same method generalises to a new modality
    pair, so it has to be the same method under the same settings — including which seeds.
    Read from `config/training.yaml` rather than copied here, because a copy is a second
    source of truth that drifts silently and turns a generalisation claim into a
    comparison of two different configurations.
    """
    datasets = load_yaml(config_path or PROJECT_ROOT / "config" / "training.yaml")["datasets"]
    lists = {tuple(meta["seeds"]) for meta in datasets.values() if "seeds" in meta}
    assert len(lists) == 1, (
        f"config/training.yaml has {len(lists)} different seed lists; the chromatin runs "
        "cannot inherit 'the' RNA→protein seeds until they agree")
    return list(next(iter(lists)))


def dataset_path(dataset: str, split_seed: int | None = None) -> Path:
    if split_seed is None:
        return CACHE_ROOT / f"{dataset}.h5ad"
    return CACHE_ROOT / PREPROCESSING_PROTOCOL / f"{dataset}_split_seed{split_seed}.h5ad"


def split_path(dataset: str, seed: int) -> Path:
    return CACHE_ROOT / f"{dataset}_split_seed{seed}.csv"


def gene_map_path(dataset: str, split_seed: int | None = None) -> Path:
    if split_seed is not None:
        return dataset_path(dataset, split_seed).with_suffix(".gene_map.csv")
    return PROJECT_ROOT / "cache" / "results" / "mapping" / f"chromatin_gene_map_{dataset}.csv"


def velocity_path(dataset: str, split_seed: int = 0, transform: str = "as_is",
                  tag: str = "", *, protocol: int = 3) -> Path:
    """One cache per (dataset, split, chromatin transform, velocity protocol).

    The helper default retains historical callers; new runner stages request protocol 4.

    The transform is part of the identity because v_c is a displacement in phi's input
    space: a field built on `tfidf_lsi` is a different field, not a different encoding of
    the same one, and overwriting the `as_is` cache with it would silently destroy the
    comparison. `as_is` keeps the original filename so existing caches still resolve.
    """
    suffix = "" if split_seed == 0 else f"_split_seed{split_seed}"
    if transform != "as_is":
        suffix = f"{suffix}_{transform}"
    # A tag names a field built the same way but under different settings -- an un-gauged
    # scale, a different neighbourhood -- so the variants sit side by side instead of one
    # silently replacing another.
    if tag:
        suffix = f"{suffix}_{tag}"
    if protocol >= 4:
        return CACHE_ROOT / PREPROCESSING_PROTOCOL / f"{dataset}_velocity{suffix}_v{protocol}.npz"
    return CACHE_ROOT / f"{dataset}_velocity{suffix}.npz"


def load_velocity(dataset: str, split_seed: int = 0, transform: str = "as_is",
                  tag: str = "", *, protocol: int = 3) -> dict:
    path = velocity_path(dataset, split_seed, transform, tag, protocol=protocol)
    assert path.exists(), (
        f"{path} not built yet — run `velocity --dataset {dataset} "
        f"--chromatin-transform {transform}` first")
    with np.load(path, allow_pickle=True) as handle:
        result = {key: handle[key] for key in handle.files}
    version = int(result.get("protocol_version", 0))
    assert version == protocol, (
        f"{path} uses velocity protocol {version}; rebuild it with the source-only protocol")
    cached_seed = int(result.get("split_seed", 0))
    assert cached_seed == split_seed, (
        f"{path} was built for split seed {cached_seed}, requested {split_seed}")
    # The field is a displacement in phi's input space; a mismatched transform would still load.
    cached_transform = str(result.get("chromatin_transform", "as_is"))
    if cached_transform != transform:
        raise ValueError(
            f"{path} was built on chromatin transform {cached_transform!r}, requested "
            f"{transform!r}; rebuild it with `velocity --dataset {dataset} "
            f"--chromatin-transform {transform}`")
    return result


def load_dataset(dataset: str, split_seed: int | None = None) -> sc.AnnData:
    path = dataset_path(dataset, split_seed)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not built; run prepare --dataset {dataset} --split-seed {split_seed or 0}")
    adata = sc.read_h5ad(path)
    if split_seed is not None:
        splits = pd.read_csv(split_path(dataset, split_seed), index_col=0)
        validate_preprocessing_split(adata, splits, split_seed)
    return adata


def validate_velocity_inputs(fields, adata) -> None:
    if int(fields.get("protocol_version", 0)) < 4:
        if adata.uns.get("chromatin_preprocessing", {}).get("protocol") == PREPROCESSING_PROTOCOL:
            raise ValueError("Legacy velocity cannot be used with training-only prepared inputs")
        return  # Historical evaluations keep their original preprocessing protocol.
    metadata = adata.uns.get("chromatin_preprocessing", {})
    if str(fields.get("preprocessing_id", "")) != metadata.get("id"):
        raise ValueError("Velocity cache belongs to a different prepared dataset; rebuild velocity")
    if not np.array_equal(fields["cell_names"], adata.obs_names.to_numpy()):
        raise ValueError("Velocity cache cell order does not match the prepared dataset")
    if not np.array_equal(fields["gene_names"], adata.var_names.to_numpy()):
        raise ValueError("Velocity cache gene order does not match the prepared dataset")


def prepare_main(args: argparse.Namespace) -> int:
    output = dataset_path(args.dataset, args.split_seed)
    mapping = gene_map_path(args.dataset, args.split_seed)
    if output.exists() or mapping.exists():
        raise FileExistsError(f"Refusing to overwrite prepared inputs: {output} / {mapping}")
    frozen_path = split_path(args.dataset, args.split_seed)
    frozen = pd.read_csv(frozen_path, index_col=0) if frozen_path.exists() else None
    if frozen is not None:
        print(f"[prepare] reusing frozen populations from {frozen_path}")
    split_options = dict(split_seed=args.split_seed, val_fraction=args.val_fraction,
                         test_fraction=args.test_fraction, splits=frozen)
    if args.dataset == "hspc":
        adata = build_hspc(HSPC_SOURCE, output, args.n_top_genes, args.seed,
                          args.n_lsi, args.min_peak_cells, mapping, **split_options)
    else:
        adata = build_bmmc(BMMC_PROCESSED, BMMC_VELOCITY, output, args.n_top_genes,
                          mapping, n_lsi=args.n_lsi, min_peak_cells=args.min_peak_cells,
                          seed=args.seed, **split_options)
    if frozen is None:
        pd.DataFrame({"split": adata.obs["chromatin_split"],
                      "train_side": adata.obs["chromatin_train_side"]}).to_csv(frozen_path)
    save_audit(adata)
    return 0


def print_audit(report: dict) -> None:
    print(f"\n=== dataset audit: {report['dataset']} ===")
    for key, value in report.items():
        if isinstance(value, dict):
            items = ", ".join(f"{name}={count}" for name, count in value.items())
            print(f"  {key:26s} {items}")
        else:
            print(f"  {key:26s} {value}")


def save_audit(adata: sc.AnnData, law: str | None = None) -> dict:
    report = audit_report(adata)
    print_audit(report)
    out = CACHE_ROOT / f"{report['dataset']}_audit.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {out}")
    if law == RELAY:
        require_relay_inputs(adata)
        print("  relay inputs: OK")
    return report


def audit_main(args: argparse.Namespace) -> int:
    adata = load_dataset(args.dataset, args.split_seed)
    save_audit(adata, args.law)
    return 0


def split_main(args: argparse.Namespace) -> int:
    """Preparation freezes this split before fitting any population statistics."""
    adata = load_dataset(args.dataset, args.seed)
    splits = pd.read_csv(split_path(args.dataset, args.seed), index_col=0)
    validate_preprocessing_split(adata, splits, args.seed)
    if args.force:
        raise ValueError("Cannot replace a split after preprocessing; prepare a new split seed")
    print(f"[split] verified training-only preparation and frozen split {split_path(args.dataset, args.seed)}")
    return 0


def velocity_main(args: argparse.Namespace) -> int:
    adata = load_dataset(args.dataset, args.split_seed)
    save_audit(adata)
    # Apply the transform in memory so the cached h5ad stays untransformed for other gauges.
    adata.obsm["gene_activity"] = chromatin_features(adata, args.chromatin_transform)
    print(f"[velocity] chromatin transform: {args.chromatin_transform}")
    print("[velocity] LSI neighbour / OT graph is unchanged; only x_j-x_i is in "
          "this transform, so v_c lives in phi's coordinates")
    warn_as_is_chromatin("velocity", args.chromatin_transform)
    splits = pd.read_csv(split_path(args.dataset, args.split_seed), index_col=0)
    assert splits.index.equals(adata.obs_names), (
        "the split file does not match this dataset — run `split` on the current build")
    split = splits["split"].to_numpy()
    side = splits["train_side"].to_numpy()
    train_source = (split == "train") & (side == "atac")
    atac_available = (split != "train") | train_source
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.dataset == "hspc":
        result = hspc_velocity(adata, args.epsilon, args.ot_iterations,
                               args.confidence_quantile, device, split=split,
                               source_mask=atac_available)
    else:
        result = bmmc_velocity(adata, args.n_neighbors, args.confidence_quantile,
                               args.min_forward, train_mask=train_source,
                               estimator=args.velocity_estimator)
    assert result["velocity"].shape == adata.obsm["gene_activity"].shape, (
        f"velocity {result['velocity'].shape} does not match the ATAC input "
        f"{adata.obsm['gene_activity'].shape}")
    assert np.isfinite(result["velocity"]).all(), "velocity contains non-finite entries"
    gauge = 1.0
    if args.gauge_normalize:
        result["velocity"], gauge = gauge_normalize(
            result["velocity"], train_source, return_gauge=True)
    result["gauge"] = np.float64(gauge)
    result["preprocessing_id"] = np.str_(adata.uns["chromatin_preprocessing"]["id"])
    result["cell_names"] = adata.obs_names.to_numpy(dtype=str)
    result["gene_names"] = adata.var_names.to_numpy(dtype=str)
    # After the gauge, so the saved norms describe the saved velocity.
    result["norm"] = np.linalg.norm(result["velocity"], axis=1).astype(np.float32)

    norms = result["norm"][result["dynamic_mask"]]
    if len(norms) == 0:
        raise ValueError("chromatin field has no confident dynamic cells")
    print(f"[velocity] |v| over the dynamic set: median {np.median(norms):.4g}, "
          f"IQR [{np.quantile(norms, 0.25):.4g}, {np.quantile(norms, 0.75):.4g}]")
    diversity = direction_diversity(result["velocity"], result["dynamic_mask"])
    print(f"[velocity] direction diversity: pairwise cosine median "
          f"{diversity['pairwise_cosine_median']:+.4f}, "
          f"{diversity['energy_in_mean_direction']:.1%} of the energy is one global arrow")
    result.update({key: np.float32(value) for key, value in diversity.items()})
    result["split_seed"] = np.int64(args.split_seed)
    result["protocol_version"] = np.int64(VELOCITY_PROTOCOL_VERSION)
    result["chromatin_transform"] = np.str_(args.chromatin_transform)
    result["velocity_tag"] = np.str_(args.velocity_tag)
    result["velocity_estimator"] = np.str_(args.velocity_estimator)
    result["gauge_normalized"] = np.bool_(args.gauge_normalize)
    result["fit_source_mask"] = train_source
    out = velocity_path(args.dataset, args.split_seed, args.chromatin_transform,
                        args.velocity_tag, protocol=VELOCITY_PROTOCOL_VERSION)
    np.savez_compressed(out, **result)
    print(f"[velocity] wrote {out}")
    return 0



CONDITIONS = RELAY_CONDITIONS

# Inherit RNA–protein hyperparameters so a difference is the modality pair, not a re-tune.
TRAINING_YAML_KEYS = {
    "n_epochs": "n_epochs",
    "batch_size": "batch_size",
    "lambda_dyn": "lambda_dyn",
    "dyn_residual_weight": "dyn_residual_weight",
    "dyn_direction_weight": "dyn_direction_weight",
    "checkpoint_monitor": "checkpoint_monitor",
    "dyn_warmup_epochs": "dyn_warmup_epochs",
    "lr": "lr",
    "lr_phi": "lr_phi",
    "lr_alpha_kappa": "lr_alpha_kappa",
    "lr_rates": "lr_beta",
    "lr_warmup_epochs": "lr_warmup_epochs",
    "lr_warmup_start_factor": "lr_warmup_start_factor",
    "lr_min_factor": "lr_min_factor",
    "sinkhorn_blur": "sinkhorn_reg",
    "sinkhorn_backend": "sinkhorn_backend",
    "phi_dims": "phi_dims",
    "kappa_dims": "kappa_dims",
    "g_dims": "g_dims",
    "phi_init_gain": "phi_init_gain",
    "phi_spectral_norm": "phi_spectral_norm",
    "activation": "kot_activation",
    "init_method": "kot_init",
    "kappa_min": "kot_kappa_min",
    "kappa_max": "kot_kappa_max",
    "alpha_min": "kot_alpha_min",
    "alpha_max": "kot_alpha_max",
    "eval_every": "val_every",
    "seed": "seed",
}


def stabilization_settings(mode: str) -> dict:
    """The three regularisers RNA→protein uses, or the chromatin trainer as it stood.

    Does not change the kappa box: chromatin_r2 keeps the r2lsi clock
    [0.001, 1.5] so residual L2 can train kappa. kot_parity's κ prior is then a
    pull toward log(2), not a constant on a pinned head.
    """
    if mode == "legacy":
        return {
            "lambda_kappa_prior": 0.0,
            "lambda_reg": 0.0,
            "grad_clip": 5.0,
            "kappa_prior_target": None,
        }
    if mode == "kot_parity":
        return {
            "lambda_kappa_prior": 0.01,
            "lambda_reg": 1.0e-4,
            "grad_clip": 1.0,
            "kappa_prior_target": KOT_KAPPA_PRIOR_TARGET,
        }
    raise ValueError(
        f"stabilization must be one of {STABILIZATION_MODES}, got {mode!r}")


def resolve_stabilization(args: argparse.Namespace) -> None:
    """Fill regulariser knobs from --stabilization unless the CLI already set them."""
    mode = getattr(args, "stabilization", None) or "legacy"
    args.stabilization = mode
    for name, value in stabilization_settings(mode).items():
        if getattr(args, name, None) is None:
            setattr(args, name, value)
    if args.lambda_kappa_prior < 0.0 or args.lambda_reg < 0.0:
        raise ValueError("lambda_kappa_prior and lambda_reg must be >= 0")
    if args.grad_clip <= 0.0:
        raise ValueError("grad_clip must be > 0")
    if args.lambda_kappa_prior > 0.0 and args.kappa_prior_target is None:
        args.kappa_prior_target = KOT_KAPPA_PRIOR_TARGET
    if getattr(args, "fixed_kappa", None) is not None:
        if args.fixed_kappa <= 0.0:
            raise ValueError("fixed_kappa must be positive")
        # κ(c) is a constant; a log-space prior toward the same constant is a no-op.
        args.lambda_kappa_prior = 0.0
        args.kappa_prior_target = None


def chromatin_network_params(model) -> list[torch.nn.Parameter]:
    """phi, kappa, and the transcription head. Rates are not weight-decayed.

    Matches src/training/kot.py's network_params: the ODE rates have their own
    scale, and L2 on them would fight the gamma TimeLapse anchor.
    """
    return (
        list(model.phi.parameters())
        + list(model.kappa.parameters())
        + list(model.g.parameters())
    )


def apply_training_defaults(args: argparse.Namespace, config_path: Path) -> dict:
    """Fill every unset training flag from training.yaml's `defaults`, and say which.

    `blur` is `sinkhorn_reg` and `lr_rates` is `lr_beta` there — the chromatin model's
    rates are gamma and beta, where the RNA→protein model has only beta. Everything else
    keeps its name.
    """
    config = load_yaml(config_path)
    defaults = config.get("defaults", {}) | config.get("chromatin_r2", {})
    taken = {}
    for name, key in TRAINING_YAML_KEYS.items():
        if getattr(args, name, None) is None and key in defaults:
            setattr(args, name, defaults[key])
            taken[name] = defaults[key]
    print(f"[train] hyperparameters from {config_path}: "
          + ", ".join(f"{k}={v}" for k, v in sorted(taken.items())))
    overridden = sorted(set(TRAINING_YAML_KEYS) - set(taken))
    if overridden:
        print(f"[train] set on the command line instead: "
              + ", ".join(f"{k}={getattr(args, k)}" for k in overridden))
    return taken


def warn_as_is_chromatin(stage: str, transform: str) -> None:
    """as_is leaves depth as PC1; those runs never tested the intended chromatin map."""
    if transform == "as_is":
        print(f"[{stage}] WARNING: chromatin-transform=as_is leaves sequencing depth as "
              "PC1 (r~0.99). Pairing and JVP on this input are uninformative; pass "
              "--chromatin-transform tfidf_lsi.")


def resolve_r2_training_flags(args: argparse.Namespace) -> None:
    """Fill kinetics/checkpoint knobs argparse leaves None so yaml can override them."""
    if args.dyn_residual_weight is None:
        args.dyn_residual_weight = 1.0
    if args.dyn_direction_weight is None:
        args.dyn_direction_weight = 0.0
    if args.checkpoint_monitor is None:
        args.checkpoint_monitor = "val_align"
    if args.checkpoint_monitor not in ("val_align", "foscttm"):
        raise ValueError(
            f"checkpoint_monitor must be val_align or foscttm, got {args.checkpoint_monitor!r}")
    if args.dyn_residual_weight < 0 or args.dyn_direction_weight < 0:
        raise ValueError("dyn residual/direction weights must be >= 0")
    if args.dyn_residual_weight == 0.0 and args.dyn_direction_weight == 0.0:
        raise ValueError("need a positive --dyn-residual-weight or --dyn-direction-weight")
    if getattr(args, "phi_lag_tau", None) is None:
        args.phi_lag_tau = 0.0
    if args.phi_lag_tau != 0.0 and getattr(args, "condition", "full") not in ("full", "noDyn"):
        raise ValueError(
            "phi lag constructs c - tau v_c from the training velocity; shuffle/"
            "reverse would mix a corrupted field into the state. Use full or noDyn.")
    resolve_stabilization(args)


def run_directory(args: argparse.Namespace) -> Path:
    """Timestamp AND settings, so a run dir says what it is without opening it."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{stamp}_{args.dataset}_{args.law}_{args.condition}_seed{args.seed}"
    if args.subsample is not None:
        name = f"{name}_sub{args.subsample}"
    path = Path(args.run_dir) if args.run_dir else CACHE_ROOT / "runs" / name
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite an existing run: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def apply_velocity_condition(velocity: np.ndarray, confidence: np.ndarray,
                             dynamic: np.ndarray, condition: str,
                             seed: int, eligible: np.ndarray | None = None
                             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """§12. The training corruption, applied to the velocity the model will be fitted on.

    `shuffle` permutes WHOLE cell velocity vectors together with confidence and the dynamic
    flag. `eligible` confines that permutation to ATAC-source training rows, so held-out or
    RNA-side vectors can never enter the training loss. Coordinates are never shuffled.
    """
    if condition == "shuffle":
        rows = np.arange(len(velocity)) if eligible is None else np.flatnonzero(eligible)
        permutation = np.random.default_rng(seed).permutation(rows)
        shuffled_velocity = velocity.copy()
        shuffled_confidence = confidence.copy()
        shuffled_dynamic = dynamic.copy()
        shuffled_velocity[rows] = velocity[permutation]
        shuffled_confidence[rows] = confidence[permutation]
        shuffled_dynamic[rows] = dynamic[permutation]
        return shuffled_velocity, shuffled_confidence, shuffled_dynamic
    if condition == "reverse":
        return -velocity, confidence, dynamic
    if condition == "zero":
        return np.zeros_like(velocity), confidence, dynamic
    return velocity, confidence, dynamic


def rna_target_layer(adata: sc.AnnData, requested: str, law: str = RELAY,
                     units: str = "shared") -> str:
    """Which RNA quantity phi is aligned against, and why that one.

    R1 predicts mature RNA. `auto` prefers `spliced_lognorm` when it exists so the
    training target matches the scVelo reference used by Task D / preflight; cells and
    genes without usable splicing are dropped by `target_cell_mask` / `output_gene_mask`
    rather than zero-filled. Pass `--rna-target rna` to force the complete total-RNA
    matrix when splicing coverage is too thin for a given analysis.

    The relay law is not allowed the total-counts fallback. It predicts [u, s] and fits
    ds = beta*u - gamma*s; if s were TOTAL RNA it would contain u, the relay would be
    fitting s against a quantity that includes its own source, and beta and gamma would
    stop being splicing and degradation rates. R2 trains and scores only on cells that
    carry splicing anyway, so spliced counts are always available to it.
    """
    if law == RELAY:
        if requested not in ("auto", "spliced"):
            raise ValueError("relay law requires spliced RNA as its s target")
        if not {"unspliced", "spliced"}.issubset(adata.layers):
            raise ValueError("relay law requires measured unspliced and spliced RNA")
        if units == "spliced":
            # Per-block own-library normalisation: this is the layer the competing methods
            # predict, so scoring against them stops being a units comparison. It puts u
            # and s on DIFFERENT library factors, which breaks the beta*u -> s mass
            # balance, so the caller must have disabled the ODE.
            return "spliced_lognorm"
        return SHARED_SPLICED
    if requested != "auto":
        layer = f"{requested}_lognorm"
        if law == RELAY and layer != "spliced_lognorm":
            raise ValueError("relay law requires spliced RNA as its s target")
        if layer not in adata.layers:
            raise ValueError(f"requested RNA target layer {layer!r} is absent")
        return layer
    if law == RELAY or "spliced_lognorm" in adata.layers:
        if "spliced_lognorm" not in adata.layers:
            raise ValueError("relay law requires measured spliced RNA")
        return "spliced_lognorm"
    return "rna_lognorm"


def build_targets(adata: sc.AnnData, law: str, target_layer: str,
                  gene_mask: np.ndarray | None = None,
                  kinetic_coords: str = SHARED_LOG1P,
                  global_scale: float | None = None) -> np.ndarray:
    """R2 [u, s] in the run's kinetic coordinates; legacy single-block views stay as stored."""
    if law == RELAY and target_layer == SHARED_SPLICED:
        return splicing_targets(adata, gene_mask, kinetic_coords, global_scale)
    if law == RELAY and target_layer != "spliced_lognorm":
        raise ValueError("Regulatory R2 requires shared RNA coordinates or spliced_lognorm")
    columns = slice(None) if gene_mask is None else gene_mask
    spliced = to_dense(adata.layers[target_layer], np.float32)[:, columns]
    if law == REDUCED:
        return spliced
    unspliced = to_dense(adata.layers["unspliced_lognorm"], np.float32)[:, columns]
    return np.concatenate([unspliced, spliced], axis=1)

def output_gene_mask(adata: sc.AnnData, law: str,
                     rows: np.ndarray | None = None,
                     target_layer: str | None = None) -> np.ndarray:
    """Genes the law has measured targets for; never zero-fill missing u/s."""
    needs_splicing = law == RELAY or target_layer == "spliced_lognorm"
    if not needs_splicing:
        return np.ones(adata.n_vars, dtype=bool)
    mask = usable_splicing_genes(adata, rows=rows)
    if not mask.any():
        raise ValueError("no gene with usable measured u/s counts for this RNA target")
    return mask


def target_cell_mask(adata: sc.AnnData, target_layer: str) -> np.ndarray:
    """Cells carrying the requested target, distinct from ATAC-only source eligibility."""
    if target_layer in (SHARED_SPLICED, "spliced_lognorm", "unspliced_lognorm") and "has_splicing" in adata.obs:
        return adata.obs["has_splicing"].to_numpy(dtype=bool)
    return np.ones(adata.n_obs, dtype=bool)



def chromatin_param_groups(model, lr: float, lr_phi, lr_alpha_kappa, lr_rates) -> list[dict]:
    """One Adam group per head, mirroring the RNA→protein split.

    Not `kot_param_groups`: that one reads `model.beta_raw` by name, and this model's
    rates are gamma (always) plus beta (relay only), which have to move together — they
    are the two ends of the same relay and separate rates would decide which one absorbs
    the other's scale.
    """
    rates = [model.gamma_raw] + ([model.beta_raw] if model.law == RELAY else [])
    heads = [
        ("phi", lr_phi, list(model.phi.parameters())),
        ("alpha_kappa", lr_alpha_kappa,
         list(model.g.parameters()) + list(model.kappa.parameters())),
        ("rates", lr_rates, rates),
    ]
    return [{"name": name, "params": params, "lr": float(lr if rate is None else rate),
             "base_lr": float(lr if rate is None else rate)}
            for name, rate, params in heads if params]


def alignment_columns(law: str, align_block: str, n_output_genes: int,
                      device: torch.device) -> torch.Tensor | None:
    """Which output columns the transport plan is measured on; None means all of them.

    Under the relay law "spliced" drops the u half. The OT itself always runs in
    align_dims dimensions, so this does not shrink the divergence's space — what it
    changes is which directions those are. Measured on BMMC (2026-09-08): fitted to the
    scaled [u, s] cloud, all 16 leading directions are majority-u, mean share 0.66, and
    they carry 3.0% of s's variance where a basis fitted to s alone carries 8.6%. Every
    metric the run is judged on is computed on s.
    """
    if law == REDUCED or align_block == "joint":
        return None
    if align_block == "unspliced":
        return torch.arange(0, n_output_genes, device=device)
    return torch.arange(n_output_genes, 2 * n_output_genes, device=device)


def held_block_columns(law: str, align_block: str, n_output_genes: int,
                       device: torch.device) -> torch.Tensor | None:
    """The block the transport plan does NOT see, which the moment term therefore holds."""
    if law == REDUCED or align_block == "joint":
        return None
    other = "unspliced" if align_block == "spliced" else "spliced"
    return alignment_columns(law, other, n_output_genes, device)


def align_view(values: torch.Tensor, columns: torch.Tensor | None, scale: torch.Tensor,
               basis: torch.Tensor | None) -> torch.Tensor:
    """The coordinates the alignment cost is measured in: chosen columns, scaled, projected."""
    selected = values if columns is None else values[:, columns]
    scaled = selected / scale
    return scaled if basis is None else scaled @ basis


def alignment_basis(target: torch.Tensor, scale: torch.Tensor, rows: torch.Tensor,
                    n_dims: int, columns: torch.Tensor | None) -> torch.Tensor | None:
    """An orthonormal basis of the RNA cells' own variation, or None for no projection.

    An entropic OT between two 2048-point clouds in 1281 dimensions is close to
    uninformative: at that dimensionality every pair of sample points sits at almost the
    same distance, so the plan carries little signal and phi's gradient is weak. Measuring
    the SAME divergence in the target's leading directions restores it. phi still predicts
    every gene — only the metric the two clouds are compared in changes — and the basis is
    fitted on the RNA training half alone, which phi never sees the cells of.
    """
    if n_dims <= 0:
        return None
    observed = align_view(target[rows], columns, scale, None)
    centred = observed - observed.mean(dim=0, keepdim=True)
    basis = torch.linalg.svd(centred, full_matrices=False)[2][:n_dims].T
    explained = float((centred @ basis).var(dim=0).sum() / centred.var(dim=0).sum())
    print(f"[train] alignment measured in {n_dims} target directions "
          f"({explained:.1%} of the RNA variance)")
    return basis


def sinkhorn_step(model, chromatin: torch.Tensor, target: torch.Tensor,
                  atac_rows: torch.Tensor, rna_rows: torch.Tensor, max_points: int,
                  scale: torch.Tensor, blur: float, backend: str,
                  basis: torch.Tensor | None,
                  columns: torch.Tensor | None) -> torch.Tensor:
    """Sinkhorn divergence between phi(ATAC cells) and the RNA cells' observed profiles.

    The two sides are drawn INDEPENDENTLY from disjoint cell sets, so no cell can appear
    on both sides and there is no pairing to leak. Dividing by `scale` is what gives the
    relay law's u and s blocks equal weight in the cost.
    """
    device = atac_rows.device
    source = subsample_rows(len(atac_rows), max_points, device, atac_rows)
    sink = subsample_rows(len(rna_rows), max_points, device, rna_rows)
    predicted = align_view(model.phi(chromatin[source]), columns, scale, basis)
    observed = align_view(target[sink], columns, scale, basis)
    return sinkhorn_divergence(predicted, observed, blur=blur, backend=backend)


def alignment_scale(target: torch.Tensor, law: str) -> torch.Tensor:
    """Per-gene scale times the cloud's own diameter, so `blur` is a fraction of it."""
    scale = block_scales(target, law)
    gauge = alignment_gauge(target, scale)
    print(f"[train] alignment gauge: median scaled cell-cell distance {gauge:.4g}")
    return scale * gauge


def gene_affine_calibration(production: torch.Tensor, target: torch.Tensor,
                            source_rows: torch.Tensor, target_rows: torch.Tensor,
                            law: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Calibrate the explicit G path from disjoint ATAC and RNA train marginals.

    This is deliberately per-gene and uses no paired cell. It preserves the named
    chromatin→RNA correspondence that an unrestricted bottleneck otherwise erases, while
    matching each source coordinate to the location and scale of its RNA target.
    """
    source = production[source_rows]
    if law == RELAY:
        source = torch.cat([source, source], dim=1)
    observed = target[target_rows]
    source_mean = source.mean(dim=0)
    target_mean = observed.mean(dim=0)
    source_std = source.std(dim=0, correction=0)
    target_std = observed.std(dim=0, correction=0)
    usable = (source_std >= 1e-3) & (target_std >= 1e-3)
    scale = torch.zeros_like(target_std)
    scale[usable] = target_std[usable] / source_std[usable]
    bias = target_mean - scale * source_mean
    print(f"[train] gene-aware phi path: {int(usable.sum())}/{len(scale)} outputs "
          "have an unpaired location/scale calibration")
    return scale, bias


def block_moment_loss(model, chromatin: torch.Tensor, target: torch.Tensor,
                      source: torch.Tensor, sink: torch.Tensor, columns: torch.Tensor,
                      scale: torch.Tensor) -> torch.Tensor:
    """Unpaired per-gene mean and std match on the block the transport plan cannot see.

    Once the plan is restricted to one block, the relay law is the only thing shaping the
    other — and the law carries a free per-cell alpha, so that block can drift anywhere
    while ds = beta*u - gamma*s still balances. This holds it on the observed marginals as
    a side task, without letting it back into the geometry the pairing is decided in. The
    two cell sets are disjoint, as everywhere else, so no pairing leaks.
    """
    predicted = model.phi(chromatin[source])[:, columns]
    observed = target[sink][:, columns]
    return (((predicted.mean(dim=0) - observed.mean(dim=0)) / scale).pow(2)
            + ((predicted.std(dim=0) - observed.std(dim=0)) / scale).pow(2)).mean()


def phi_gradient_norm(model, loss: torch.Tensor) -> tuple[float, torch.Tensor]:
    """Norm of this loss's gradient on phi alone, and the flat gradient itself."""
    grads = torch.autograd.grad(loss, [p for p in model.phi.parameters()],
                                retain_graph=False, allow_unused=True)
    flat = torch.cat([g.reshape(-1) for g in grads if g is not None])
    return float(flat.norm()), flat


def pairing_foscttm(model, chromatin: torch.Tensor, target: torch.Tensor,
                    rows: torch.Tensor, law: str, n_cells: int = 2000) -> float:
    """Paired FOSCTTM on spliced RNA. val_align is unpaired Sinkhorn and can fall
    while this stays at the permuted floor."""
    prediction = spliced_block(model.phi(chromatin[rows]), law)
    observed = spliced_block(target[rows], law)
    sample = torch.randperm(len(rows), device=rows.device)[:min(n_cells, len(rows))]
    return float(np.mean(calc_domainAveraged_FOSCTTM(
        prediction[sample].detach().cpu().numpy(), observed[sample].cpu().numpy())))


def grad_interaction(model, law: str, chromatin: torch.Tensor, target: torch.Tensor,
                     velocity: torch.Tensor, production: torch.Tensor,
                     confidence: torch.Tensor, probe: torch.Tensor, sink: torch.Tensor,
                     scales: tuple, blur: float, backend: str, weight: float,
                     basis: torch.Tensor | None, residual_mask: torch.Tensor,
                     columns: torch.Tensor | None,
                     residual_weight: float = 1.0, direction_weight: float = 0.0) -> dict:
    """How much of phi's gradient the kinetics term actually supplies, and whether it fights.

    This project already learned that the kinetics term must be judged by its GRADIENT and
    not by its loss value: a residual that has been driven to ~0 can mean the law is
    satisfied, or that lambda_dyn is so small the term never moved phi at all, and the loss
    curve looks identical either way. `mag_ratio` separates those two, and a negative
    cosine says the two terms are pulling phi in opposite directions.
    """
    align_scale, dyn_scale = scales
    was_training = model.training
    model.eval()
    predicted = align_view(model.phi(chromatin[probe]), columns, align_scale, basis)
    observed = align_view(target[sink], columns, align_scale, basis)
    align = sinkhorn_divergence(predicted, observed, blur=blur, backend=backend)
    align_norm, align_grad = phi_gradient_norm(model, align)
    dyn = kinetics_loss(model, law, chromatin[probe], velocity[probe], production[probe],
                        confidence[probe], dyn_scale, residual_mask,
                        residual_weight=residual_weight, direction_weight=direction_weight)
    dyn_norm, dyn_grad = phi_gradient_norm(model, dyn)
    if was_training:
        model.train()
    return {
        "grad_align": align_norm,
        "grad_dyn": dyn_norm,
        "grad_dyn_weighted": weight * dyn_norm,
        "grad_mag_ratio": weight * dyn_norm / max(align_norm, 1e-12),
        "grad_cosine": float(torch.nn.functional.cosine_similarity(
            align_grad, dyn_grad, dim=0)),
    }


def check_r2_main(args: argparse.Namespace) -> int:
    """Read-only server check of the frozen data and actual RNA training panel."""
    adata = load_dataset(args.dataset, args.split_seed)
    require_relay_inputs(adata)
    splits = pd.read_csv(split_path(args.dataset, args.split_seed), index_col=0)
    if not splits.index.equals(adata.obs_names):
        raise ValueError("Split cell order does not match the dataset")
    fields = load_velocity(args.dataset, args.split_seed, args.chromatin_transform,
                           args.velocity_tag, protocol=VELOCITY_PROTOCOL_VERSION)
    validate_velocity_inputs(fields, adata)
    if fields["velocity"].shape != adata.obsm["gene_activity"].shape:
        raise ValueError("Cached velocity shape does not match chromatin input")
    if not np.isfinite(fields["velocity"]).all():
        raise ValueError("Cached chromatin velocity contains non-finite values")
    covered = target_cell_mask(adata, SHARED_SPLICED)
    training = splits["split"].eq("train").to_numpy()
    rna_rows = np.flatnonzero(training & splits["train_side"].eq("rna").to_numpy() & covered)
    atac_rows = np.flatnonzero(training & splits["train_side"].eq("atac").to_numpy())
    if len(rna_rows) == 0 or len(atac_rows) == 0:
        raise ValueError("Measured RNA and ATAC training populations must be nonempty")
    mask = output_gene_mask(adata, RELAY, rows=rna_rows, target_layer=SHARED_SPLICED)
    genes = adata.var_names[mask]
    anchors = load_gamma_anchors(args.gamma_anchor_csv, genes)
    _, _, kinetic_mask = gene_map(adata, gene_map_path(args.dataset))
    dynamic = np.asarray(fields["dynamic_mask"])[atac_rows]
    summary = {
        "dataset": args.dataset, "formulation": FORMULATION,
        "target_layer": SHARED_SPLICED, "aligned_output": "spliced",
        "n_atac_train": len(atac_rows), "n_measured_rna_train": len(rna_rows),
        "n_dynamic_atac_train": int(dynamic.sum()),
        "n_output_genes": len(genes), "n_kinetic_genes": int(kinetic_mask[mask].sum()),
        "n_gamma_anchors": len(anchors["indices"]),
        "gamma_anchor_fraction": len(anchors["indices"]) / len(genes),
        "gamma_anchor_sha256": anchors["sha256"],
        "time_convention": anchors["time_convention"],
        "reference_layers": [name for name in adata.layers if name.startswith("velocity_")],
    }
    print(json.dumps(summary, indent=2))
    if not dynamic.any():
        raise ValueError("No dynamic ATAC training cells; full R2 would have no kinetic loss")
    return 0


def train_main(args: argparse.Namespace) -> int:
    if args.law != RELAY or args.align_block != "spliced":
        raise ValueError("Regulatory R2 aligns on spliced RNA only")
    if args.rna_target_units == "spliced" and args.condition != "noDyn":
        raise ValueError(
            "--rna-target-units spliced normalises u and s by different library factors, "
            "so beta*u -> s no longer conserves mass; it is only valid with --condition "
            "noDyn, where the ODE term is switched off")
    if not args.gamma_anchor_csv or args.lambda_gamma_anchor <= 0:
        raise ValueError("R2 requires --gamma-anchor-csv with RNA-decay measurements and a positive prior weight")
    adata = load_dataset(args.dataset, args.split_seed)
    save_audit(adata, args.law)
    from_yaml = apply_training_defaults(args, Path(args.training_config))
    resolve_r2_training_flags(args)
    warn_as_is_chromatin("train", args.chromatin_transform)
    allowed = conditions_for_law(args.law)
    assert args.condition in allowed, (
        f"law {args.law} only runs {allowed}; got condition={args.condition!r}")
    # The protein default for lambda_dyn is too strong here and can collapse phi if inherited silently.
    if args.lambda_dyn >= 10 and "lambda_dyn" in from_yaml:
        print(f"[train] WARNING: lambda_dyn={args.lambda_dyn} inherited from "
              f"{args.training_config}. On chromatin the measured optimum is 1 and >=10 "
              "collapses phi. Pass --lambda-dyn explicitly if this is deliberate.")
    assert args.lambda_held_block == 0.0 or args.law == RELAY, (
        "--lambda-held-block supervises the relay block outside the plan; R1 has one block")
    assert args.align_block == "joint" or args.law == RELAY, (
        "--align-block splits the relay's [u, s]; the reduced law has a single block")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output_dir = run_directory(args)
    splits = pd.read_csv(split_path(args.dataset, args.split_seed), index_col=0)
    assert splits.index.equals(adata.obs_names), "split file does not match the dataset"
    fields = load_velocity(args.dataset, args.split_seed, args.chromatin_transform,
                           args.velocity_tag, protocol=VELOCITY_PROTOCOL_VERSION)
    validate_velocity_inputs(fields, adata)

    target_layer = rna_target_layer(adata, args.rna_target, args.law, args.rna_target_units)
    split_column = splits["split"].to_numpy()
    side_column = splits["train_side"].to_numpy()
    target_covered = target_cell_mask(adata, target_layer)
    if not target_covered.all():
        print(f"[train] target '{target_layer}': {int(target_covered.sum())}/{adata.n_obs} "
              "cells measured; absent targets are excluded, never zero-filled")
    gene_fit_rows = np.flatnonzero(
        target_covered & (split_column == "train") & (side_column == "rna"))
    gene_mask = output_gene_mask(
        adata, args.law, rows=gene_fit_rows, target_layer=target_layer)
    output_genes = adata.var_names[gene_mask]
    anchors = load_gamma_anchors(
        args.gamma_anchor_csv, output_genes, target=args.gamma_anchor_target,
        hours_per_model_time=args.gamma_hours_per_model_time)
    (output_dir / "gamma_anchors.json").write_text(json.dumps(anchors, indent=2))
    chromatin_np = chromatin_features(adata, args.chromatin_transform)
    kinetic_coords = resolve_kinetic_coords(getattr(args, "rna_kinetic_coords", None))
    args.rna_kinetic_coords = kinetic_coords
    args.rna_global_scale = None
    global_scale = None
    if kinetic_coords == GLOBAL_LINEAR:
        totals = panel_library_totals(adata, gene_mask)
        global_scale = global_linear_scale(totals[gene_fit_rows])
        args.rna_global_scale = global_scale
        print(f"[train] RNA kinetic coords={kinetic_coords}: linear u,s scaled by "
              f"CP10K / training-RNA median panel library {global_scale:.4g}")
    elif kinetic_coords == PANEL_LOG1P_COMPOSITIONAL:
        print(f"[train] RNA kinetic coords={kinetic_coords}: panel CP10K + log1p, "
              "with compositional dT/T in the relay")
    else:
        print(f"[train] RNA kinetic coords={kinetic_coords}: transcriptome CP10K + log1p, "
              "log chain rule only")
    target_np = build_targets(
        adata, args.law, target_layer, gene_mask,
        kinetic_coords=kinetic_coords, global_scale=global_scale)
    full_mapping, full_alignment_mask, full_kinetic_mask = gene_map(
        adata, gene_map_path(args.dataset))
    true_mapping = full_mapping[gene_mask].tocsr()
    if args.gene_map_mode != "curated":
        if args.gene_map_mode == "peak-genomic" and args.dataset != "hspc":
            raise ValueError(
                "peak-genomic G needs cellranger peak annotation; only HSPC ships it")
        before = true_mapping.nnz
        # Feature-feature correlation needs no RNA, but it must still be fitted on TRAINING
        # ATAC cells only: fitting it on every cell would let held-out geometry shape G.
        coaccess_rows = np.flatnonzero(
            (split_column == "train") & splits["train_side"].eq("atac").to_numpy())
        true_mapping = widen_projection(
            true_mapping, chromatin_np[coaccess_rows], args.gene_map_mode,
            args.g_neighbors, args.seed, output_genes=output_genes,
            feature_names=pd.Index(np.asarray(adata.uns["gene_activity_names"]).astype(str)),
            loci=(gene_loci_from_features(hspc_features_tsv(HSPC_SOURCE))
                  if args.gene_map_mode in ("genomic", "peak-genomic") else None),
            annotation=(hspc_peak_annotation(HSPC_SOURCE)
                        if args.gene_map_mode == "peak-genomic" else None),
            genomic_bp=getattr(args, "g_genomic_bp", 100_000),
            genomic_decay_bp=getattr(args, "g_genomic_decay_bp", 50_000))
        print(f"[train] G mode '{args.gene_map_mode}': {before} -> {true_mapping.nnz} links "
              f"over {true_mapping.shape[0]} genes")
    mapping = true_mapping
    alignment_mask = full_alignment_mask[gene_mask]
    kinetic_mask = full_kinetic_mask[gene_mask]
    if args.gene_map_mode == "diagonal_full":
        kinetic_mask = np.asarray(mapping.getnnz(axis=1) > 0)
    assert_no_orphan_g_rows(mapping, kinetic_mask)

    train_source = (split_column == "train") & (side_column == "atac")
    train_target = (split_column == "train") & (side_column == "rna")
    assert len(set(adata.obs_names[train_source]) & set(adata.obs_names[train_target])) == 0
    shuffle_eligible = train_source & (np.asarray(fields["norm"]) > 0)
    velocity_np, confidence_np, dynamic_np_all = apply_velocity_condition(
        fields["velocity"], fields["confidence"], fields["dynamic_mask"],
        args.condition, args.seed, eligible=shuffle_eligible)
    if getattr(args, "phi_lag_tau", 0.0) != 0.0:
        chromatin_np = euler_lag(chromatin_np, velocity_np, args.phi_lag_tau)
        print(f"[train] phi lag tau={args.phi_lag_tau:g}: Euler c - tau v_c "
              "in the cached velocity's units (gauge-normalised; τ=1 is one "
              "velocity-length step, not one physical day)")
    regulatory_np = regulatory_features(
        adata, args.chromatin_transform, getattr(args, "regulatory_transform", "same"),
        map_features=chromatin_np)

    rows = np.arange(adata.n_obs)
    if args.subsample is not None:
        rows = np.random.default_rng(args.seed).choice(rows, args.subsample, replace=False)
        rows.sort()
    keep = np.zeros(adata.n_obs, dtype=bool)
    keep[rows] = True

    device = choose_torch_device({"device": args.device})
    chromatin = torch.as_tensor(chromatin_np, device=device)
    target = torch.as_tensor(target_np, device=device)
    velocity = torch.as_tensor(velocity_np.astype(np.float32), device=device)
    confidence = torch.as_tensor(confidence_np.astype(np.float32), device=device)
    mapping_dense = mapping.toarray().astype(np.float32)
    production = torch.as_tensor(
        (chromatin_np @ mapping_dense.T).astype(np.float32), device=device)
    # phi's affine calibration always needs G c in MAP coordinates. Alpha may
    # read a different gene-activity representation; that tensor is built here.
    if getattr(args, "regulatory_transform", "same") in (None, "same"):
        regulatory_production = production
        regulatory = chromatin
    else:
        regulatory = torch.as_tensor(regulatory_np, device=device)
        regulatory_production = torch.as_tensor(
            (regulatory_np @ mapping_dense.T).astype(np.float32), device=device)
    alpha_input = regulatory if args.alpha_input == "full" else regulatory_production
    atac_rows = torch.as_tensor(
        np.flatnonzero(keep & (split_column == "train") & (side_column == "atac")), device=device)
    rna_rows = torch.as_tensor(
        np.flatnonzero(keep & target_covered & (split_column == "train")
                       & (side_column == "rna")), device=device)
    val_rows = torch.as_tensor(
        np.flatnonzero(keep & target_covered & (split_column == "val")), device=device)
    dynamic_np = keep & (split_column == "train") & (side_column == "atac") & dynamic_np_all
    dynamic_rows = torch.as_tensor(np.flatnonzero(dynamic_np), device=device)
    if min(len(atac_rows), len(rna_rows), len(val_rows)) == 0:
        raise ValueError("ATAC train, measured RNA train, and measured validation sets must be non-empty")
    if args.condition != "noDyn" and args.lambda_dyn > 0 and len(dynamic_rows) == 0:
        raise ValueError("No dynamic ATAC training cells; R2 would have no kinetic loss")
    print(f"[train] {args.dataset} {args.law}/{args.condition} seed {args.seed} -> {output_dir}")
    print(f"[train] chromatin transform: {args.chromatin_transform}  "
          f"regulatory transform: {getattr(args, 'regulatory_transform', 'same')}")
    if getattr(args, "regulatory_transform", "same") not in (None, "same"):
        print("[train] alpha reads G c under "
              f"{args.regulatory_transform}; phi and v_c stay under "
              f"{args.chromatin_transform}")
    if getattr(args, "fixed_kappa", None) is not None:
        print(f"[train] kinetics: residual_weight={args.dyn_residual_weight} "
              f"direction_weight={args.dyn_direction_weight}  "
              f"kappa(c) ≡ {float(args.fixed_kappa):g}")
    else:
        print(f"[train] kinetics: residual_weight={args.dyn_residual_weight} "
              f"direction_weight={args.dyn_direction_weight}  "
              f"kappa bounds [{args.kappa_min}, {args.kappa_max}]")
    print(f"[train] stabilization={args.stabilization}  "
          f"λ_κ={args.lambda_kappa_prior} target={args.kappa_prior_target}  "
          f"λ_reg={args.lambda_reg}  grad_clip={args.grad_clip}")
    if (args.lambda_kappa_prior > 0.0 and args.kappa_min is not None
            and args.kappa_max is not None and args.kappa_min == args.kappa_max):
        print("[train] κ prior is on but kappa is pinned; the prior is a constant "
              "and kot_parity vs legacy differs by λ_reg and grad_clip")
    print(f"[train] checkpoint monitor: {args.checkpoint_monitor} (lower is better)")
    if (args.dyn_direction_weight > 0 and args.dyn_residual_weight == 0
            and args.kappa_min is not None and args.kappa_max is not None
            and args.kappa_min < args.kappa_max):
        print("[train] WARNING: direction-only kinetics does not train kappa; pin "
              "--kappa-min and --kappa-max to the same value or residual_norm will "
              "reflect an untrained clock")
    print(f"[train] rows: atac {len(atac_rows)}, rna {len(rna_rows)}, val {len(val_rows)}, "
          f"kinetic {len(dynamic_rows)}; target '{target_layer}' ({target.shape[1]} columns); "
          f"genes: {int(alignment_mask.sum())} aligned, {int(kinetic_mask.sum())} in G")

    # Training-half scales only. Per-gene so no gene dominates the ODE; cloud diameter so blur is a cell-distance fraction.
    fit_target = target[rna_rows]
    target_scale = block_scales(fit_target, args.law)
    phi_scale, phi_bias = gene_affine_calibration(
        production, target, atac_rows, rna_rows, args.law)
    phi_projection = torch.as_tensor(
        true_mapping.toarray().astype(np.float32), device=device)
    if args.law == RELAY:
        phi_projection = torch.cat([phi_projection, phi_projection], dim=0)
    if args.phi_affine == "none":
        # Every state metric of a trained run matches its untrained affine calibration to
        # within 0.01, and the residual gate ends NEGATIVE -- consistent with the warm
        # start being a minimum the optimiser never leaves. Dropping it makes phi a plain
        # MLP with no gene-aware path, which is the control that tells the two apart.
        print("[train] phi has NO gene-affine path: plain MLP, no unpaired warm start")
        phi_projection = phi_scale = phi_bias = None
    # u and s share a library factor; no source-cell RNA enters the kinetic RHS.
    align_columns = alignment_columns(args.law, args.align_block, len(output_genes), device)
    held_columns = held_block_columns(args.law, args.align_block, len(output_genes), device)
    aligned_fit = fit_target if align_columns is None else fit_target[:, align_columns]
    align_scale = alignment_scale(aligned_fit, args.law if align_columns is None else REDUCED)
    # A transcription-head bias does not establish support for an unmapped gene.
    residual_mask = torch.as_tensor(law_mask(kinetic_mask, args.law), device=device)

    basis = alignment_basis(target, align_scale, rna_rows, args.align_dims, align_columns)
    model = ChromatinKOT(
        len(output_genes), args.law, n_input_features=adata.n_vars,
        phi_dims=args.phi_dims, kappa_dims=args.kappa_dims,
        g_dims=args.g_dims, phi_init_gain=args.phi_init_gain,
        phi_spectral_norm=args.phi_spectral_norm, activation=args.activation,
        init_method=args.init_method, kappa_min=args.kappa_min, kappa_max=args.kappa_max,
        phi_projection=phi_projection, phi_scale=phi_scale, phi_bias=phi_bias,
        phi_residual_weight=args.phi_residual_weight, phi_gate=args.phi_gate,
        phi_trainable_projection=args.trainable_g, alpha_input=args.alpha_input,
        alpha_min=args.alpha_min, alpha_max=args.alpha_max,
    ).to(device)
    model.kinetic_coords = kinetic_coords
    if getattr(args, "fixed_kappa", None) is not None:
        model.kappa = FixedKappa(float(args.fixed_kappa)).to(device)
        print(f"[train] kappa(c) ≡ {float(args.fixed_kappa):g} "
              "(state-dependent scale head removed)")
    with torch.no_grad():
        gamma_target = model.gamma_raw.new_tensor(anchors["gamma"])
        if bool((gamma_target <= 1e-6).any()):
            raise ValueError("Gamma targets fall below the model's positive rate floor")
        positive = gamma_target - 1e-6
        model.gamma_raw[anchors["indices"]] = positive + torch.log(-torch.expm1(-positive))
    optimiser = torch.optim.Adam(chromatin_param_groups(
        model, args.lr, args.lr_phi, args.lr_alpha_kappa, args.lr_rates))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimiser, lr_lambda=lambda epoch: lr_schedule_factor(
            epoch, args.lr_warmup_epochs, args.n_epochs, args.lr_warmup_start_factor,
            args.lr_min_factor))
    lambda_dyn = 0.0 if args.condition == "noDyn" else args.lambda_dyn
    residual_weight = float(args.dyn_residual_weight)
    direction_weight = float(args.dyn_direction_weight)
    monitor = args.checkpoint_monitor

    model.eval()
    with torch.no_grad():
        initial_held = sinkhorn_step(
            model, chromatin, target, val_rows, val_rows, args.sinkhorn_max_points,
            align_scale, args.sinkhorn_blur, args.sinkhorn_backend, basis, align_columns)
        initial_foscttm = pairing_foscttm(model, chromatin, target, val_rows, args.law)
    history = []
    monitor_ready_at = 0 if lambda_dyn == 0.0 else args.dyn_warmup_epochs
    if monitor_ready_at > args.n_epochs:
        raise ValueError(
            f"dyn_warmup_epochs ({args.dyn_warmup_epochs}) exceeds n_epochs "
            f"({args.n_epochs}); no dynamics-mature checkpoint can be selected")
    initial_score = initial_foscttm if monitor == "foscttm" else float(initial_held)
    best = {
        "val_align": float(initial_held),
        "foscttm": initial_foscttm,
        "score": initial_score if monitor_ready_at == 0 else float("inf"),
        "state": ({k: v.detach().clone() for k, v in model.state_dict().items()}
                  if monitor_ready_at == 0 else None),
        "epoch": 0 if monitor_ready_at == 0 else None,
    }
    print(f"[train] epoch    0  val_align {float(initial_held):.5f}  "
          f"foscttm {initial_foscttm:.4f} (gene-aware initial map)")
    if monitor_ready_at:
        print(f"[train] checkpoint selection ({monitor}) starts at epoch "
              f"{monitor_ready_at}, when lambda_dyn has reached full strength")
    for epoch in range(1, args.n_epochs + 1):
        model.train()
        optimiser.zero_grad(set_to_none=True)
        align = sinkhorn_step(model, chromatin, target, atac_rows, rna_rows,
                              args.sinkhorn_max_points, align_scale, args.sinkhorn_blur,
                              args.sinkhorn_backend, basis, align_columns)
        align.backward()
        held_value = 0.0
        if args.lambda_held_block > 0.0 and held_columns is not None:
            held = block_moment_loss(
                model, chromatin, target,
                subsample_rows(len(atac_rows), args.sinkhorn_max_points, device, atac_rows),
                subsample_rows(len(rna_rows), args.sinkhorn_max_points, device, rna_rows),
                held_columns, target_scale[held_columns])
            (args.lambda_held_block * held).backward()
            held_value = float(held.detach())

        weight = lambda_dyn * dyn_weight_factor(epoch, args.dyn_warmup_epochs)
        dyn_value = dyn_residual = dyn_direction = 0.0
        if weight > 0.0 and len(dynamic_rows) > 0:
            for batch in shuffled_batches(len(dynamic_rows), args.batch_size, device,
                                          dynamic_rows):
                share = len(batch) / len(dynamic_rows)
                parts = kinetics_losses(
                    model, args.law, chromatin[batch], velocity[batch],
                    alpha_input[batch], confidence[batch], target_scale, residual_mask,
                    residual_weight=residual_weight, direction_weight=direction_weight)
                ((weight * share) * parts["total"]).backward()
                dyn_value += share * float(parts["total"])
                dyn_residual += share * float(parts["residual"])
                dyn_direction += share * float(parts["direction"])
        kappa_prior_value = 0.0
        if (args.lambda_kappa_prior > 0.0 and args.kappa_prior_target is not None
                and len(dynamic_rows) > 0):
            kappa_target = chromatin.new_tensor(float(args.kappa_prior_target))
            for batch in shuffled_batches(len(dynamic_rows), args.batch_size, device,
                                          dynamic_rows):
                share = len(batch) / len(dynamic_rows)
                prior = kappa_prior_loss(
                    model.kappa(chromatin[batch]), kappa_target, args.lambda_kappa_prior)
                (share * prior).backward()
                kappa_prior_value += share * float(prior.detach())
        anchor = gamma_anchor_loss(model.gamma, anchors)
        (args.lambda_gamma_anchor * anchor).backward()
        reg_value = 0.0
        if args.lambda_reg > 0.0:
            loss_reg = args.lambda_reg * sum(
                param.pow(2).sum() for param in chromatin_network_params(model))
            loss_reg.backward()
            reg_value = float(loss_reg.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimiser.step()
        scheduler.step()

        if epoch % args.eval_every == 0 or epoch == args.n_epochs:
            model.eval()
            with torch.no_grad():
                held = sinkhorn_step(model, chromatin, target, val_rows, val_rows,
                                     args.sinkhorn_max_points, align_scale,
                                     args.sinkhorn_blur, args.sinkhorn_backend, basis,
                                     align_columns)
                foscttm = pairing_foscttm(model, chromatin, target, val_rows, args.law)
            row = {"epoch": epoch, "loss_align": float(align), "loss_dyn": dyn_value,
                   "loss_dyn_residual": dyn_residual, "loss_dyn_direction": dyn_direction,
                   "loss_held_block": held_value, "loss_gamma_anchor": float(anchor),
                   "loss_kappa_prior": kappa_prior_value, "loss_reg": reg_value,
                   "val_align": float(held), "foscttm": foscttm,
                   "lambda_dyn_effective": weight}
            if len(dynamic_rows) > 0:
                row.update(grad_interaction(
                    model, args.law, chromatin, target, velocity, alpha_input, confidence,
                    subsample_rows(len(dynamic_rows), args.batch_size, device, dynamic_rows),
                    subsample_rows(len(rna_rows), args.batch_size, device, rna_rows),
                    (align_scale, target_scale), args.sinkhorn_blur, args.sinkhorn_backend,
                    weight, basis, residual_mask, align_columns,
                    residual_weight=residual_weight, direction_weight=direction_weight))
            history.append(row)
            print(f"[train] epoch {epoch:4d}  align {float(align):.5f}  "
                  f"dyn {dyn_value:.5f}  dir {dyn_direction:.5f}  "
                  f"val_align {float(held):.5f}  foscttm {foscttm:.4f}  "
                  f"grad_ratio {row.get('grad_mag_ratio', float('nan')):.3g}  "
                  f"grad_cos {row.get('grad_cosine', float('nan')):+.3f}")
            score = foscttm if monitor == "foscttm" else float(held)
            if epoch >= monitor_ready_at and score < best["score"]:
                best = {"val_align": float(held), "foscttm": foscttm, "score": score,
                        "state": {k: v.detach().clone() for k, v in model.state_dict().items()},
                        "epoch": epoch}

    frame = pd.DataFrame(history)
    frame.to_csv(output_dir / "training_loss.csv", index=False)
    settings = vars(args) | {"target_layer": target_layer, "formulation": FORMULATION,
                             "gamma_anchors": anchors,
                             "preprocessing_protocol": PREPROCESSING_PROTOCOL,
                             "preprocessing_id": adata.uns["chromatin_preprocessing"]["id"],
                             "velocity_protocol_version": VELOCITY_PROTOCOL_VERSION,
                             "n_input_features": int(adata.n_vars),
                             "n_output_genes": int(len(output_genes)),
                             "hyperparameters_from_training_yaml": from_yaml,
                             "n_atac_rows": len(atac_rows), "n_rna_rows": len(rna_rows),
                             "n_kinetic_rows": len(dynamic_rows)}
    (output_dir / "run_config.json").write_text(json.dumps(settings, indent=2, default=str))
    assert best["state"] is not None, "no dynamics-mature checkpoint was recorded"
    for name, state, epoch in [("best_align", best["state"], best["epoch"]),
                               ("final", model.state_dict(), args.n_epochs)]:
        torch.save({"state_dict": state, "epoch": epoch, "law": args.law,
                    "formulation": FORMULATION, "gamma_anchors": anchors,
                    "condition": args.condition, "seed": args.seed,
                    "target_layer": target_layer, "dataset": args.dataset,
                    "alpha_projection": torch.from_numpy(mapping_dense.copy()),
                    "gene_names": list(output_genes),
                    "input_gene_names": list(adata.var_names)},
                   output_dir / f"checkpoint_{name}.pt")
    model.load_state_dict(best["state"])
    print(f"[train] best {monitor} {best['score']:.5f} at epoch {best['epoch']} "
          f"(val_align {best['val_align']:.5f}, foscttm {best['foscttm']:.4f})")
    # Score JVP only on cells that have a velocity; a median over mostly-zero rows is uninformative.
    velocity_val = np.flatnonzero(
        np.isin(np.arange(adata.n_obs), val_rows.cpu().numpy())
        & fields["dynamic_mask"] & target_covered)
    checks = preflight_checks(model, args.law, chromatin, target, alpha_input,
                              torch.as_tensor(fields["velocity"].astype(np.float32),
                                              device=device),
                              val_rows, torch.as_tensor(velocity_val, device=device),
                              target_scale, adata, args.reference_layer,
                              reference_columns=np.flatnonzero(gene_mask),
                              target_layer=target_layer)
    return persist_preflight(output_dir, checks, args.condition, "train")


def preflight_verdict(checks: dict, condition: str) -> tuple[bool, list[str]]:
    """The launch gate in §19: full/noDyn must look non-random before anything else runs.

    Corruption arms are not required to recover biology — that is the point of the
    ablation — so only finite-ness is demanded of them here.
    """
    failures = []
    if not checks["state_prediction_finite"]:
        failures.append("state prediction has non-finite values")
    if not checks["jvp_finite"]:
        failures.append("JVP has non-finite values")
    if condition not in ("full", "noDyn"):
        return len(failures) == 0, failures
    spread = checks["prediction_spread_ratio"]
    if not np.isfinite(spread) or spread < 0.05:
        failures.append(f"phi collapsed (prediction spread ratio {spread:.4g}, need >= 0.05)")
    pearson = checks["state_pearson_median"]
    if not (np.isfinite(pearson) and pearson > 0):
        failures.append(f"state Pearson {pearson} is not positive")
    foscttm = checks["foscttm"]
    floor = checks.get("foscttm_permuted_floor")
    if floor is None or not np.isfinite(floor):
        failures.append("permuted-pairing FOSCTTM floor is missing")
    elif not (np.isfinite(foscttm) and foscttm < floor):
        failures.append(
            f"FOSCTTM {foscttm:.3f} is not below the permuted-pairing floor {floor:.3f}")
    # Raw cosine is reported, not gated: every field shares the reference mean, so only the centred statistic is per-cell.
    centred = checks.get("jvp_vs_reference_cosine_centred_median")
    null = checks.get("jvp_vs_reference_cosine_centred_null_p95")
    if checks.get("jvp_vs_reference_skipped"):
        pass
    elif centred is None or null is None:
        failures.append("centred JVP vs RNA velocity cosine is missing — run `reference` first")
    elif not (np.isfinite(centred) and np.isfinite(null) and centred > null):
        failures.append(
            f"centred JVP vs RNA velocity cosine {centred:.4g} does not clear its "
            f"permutation null {null:.4g}")
    return len(failures) == 0, failures


def write_preflight_result(output_dir: Path, checks: dict, condition: str,
                           checkpoint: str = "best_align") -> tuple[bool, list[str]]:
    """Write a checkpoint-tagged gate file. Only `best_align` updates the launch marker.

    Training and the historical refresh path both score `best_align` and keep writing
    `preflight.json` / `preflight_passed.json`. Scoring `final` must not clobber that
    launch-gate pair: the two checkpoints answer different questions.
    """
    payload = json.dumps(checks, indent=2)
    (output_dir / f"preflight_{checkpoint}.json").write_text(payload)
    passed, failures = preflight_verdict(checks, condition)
    if checkpoint != "best_align":
        return passed, failures
    (output_dir / "preflight.json").write_text(payload)
    marker = output_dir / "preflight_passed.json"
    if passed:
        marker.write_text(payload)
    elif marker.exists():
        marker.unlink()
    return passed, failures


def persist_preflight(output_dir: Path, checks: dict, condition: str, stage: str,
                      refuse_failed_gates: bool = True,
                      checkpoint: str = "best_align") -> int:
    """Print the gate, write its files, and choose the process exit code.

    Training refuses a full/noDyn run that fails the biological gate. A later refresh
    still records that failure, but only treats a missing RNA-velocity reference as a
    process error — otherwise a batch re-score would stop at the first degenerate arm.
    """
    print(json.dumps(checks, indent=2))
    passed, failures = write_preflight_result(output_dir, checks, condition, checkpoint)
    if passed:
        return 0
    print(f"[{stage}] PREFLIGHT FAILED:")
    for line in failures:
        print(f"  - {line}")
    missing_reference = any("run `reference` first" in line for line in failures)
    if condition in ("full", "noDyn") and (refuse_failed_gates or missing_reference):
        if refuse_failed_gates:
            print(f"[{stage}] refusing to treat this as a launchable checkpoint "
                  "(shuffle waits until full/noDyn pass)")
        return 1
    if condition not in ("full", "noDyn"):
        print(f"[{stage}] corruption arms are allowed to fail the biological gate; "
              "the checkpoint is still written")
    else:
        print(f"[{stage}] recorded the failed gate; the checkpoint is unchanged")
    return 0


def reference_agreement(pushed: np.ndarray, prediction: np.ndarray, reference: np.ndarray,
                        velocity_rows: torch.Tensor, reference_columns: np.ndarray | None,
                        prefix: str, coords: str | None = None) -> dict:
    """Task D against one RNA-only reference velocity, on the cells and genes it covers.

    The two sides live in different coordinates and one of them has to move. Under
    log1p CP10K the JVP is a log1p rate and scVelo's velocity is LINEAR, so the
    model side is chain-ruled: dx/dt = (dy/dt) exp(y). A per-cell library scalar
    cancels in a per-cell cosine. Global-linear coordinates already predict a
    linear rate, so that conversion is skipped.

    Genes as well as cells: scVelo fits a subset of the panel and the rest is zero-filled,
    and leaving those columns in dilutes the cosine by roughly the square root of the
    covered fraction.
    """
    pushed = state_rate_to_linear(pushed, prediction, coords)
    selected = reference[velocity_rows.cpu().numpy()]
    if reference_columns is not None:
        selected = selected[:, reference_columns]
    scored = np.flatnonzero(np.abs(selected).sum(axis=1) > 0)
    genes = np.flatnonzero(np.abs(selected).sum(axis=0) > 0)
    if len(scored) == 0 or len(genes) == 0:
        return {}
    metrics = task_d_kinetics(pushed[np.ix_(scored, genes)], selected[np.ix_(scored, genes)])
    return {
        f"{prefix}_cosine_median": metrics["cell_cosine_median"],
        f"{prefix}_cosine_centred_median": metrics["cell_cosine_centred_median"],
        f"{prefix}_cosine_centred_null_p95": metrics["cell_cosine_centred_null_p95"],
        f"{prefix}_gene_pearson_median": metrics["gene_pearson_median"],
        f"{prefix}_n_cells": int(len(scored)),
        f"{prefix}_n_genes": int(len(genes)),
    }


def state_metrics(prediction: torch.Tensor, observed: torch.Tensor,
                  sample: torch.Tensor, n_probe: int = 200) -> dict:
    """Task A and the pairing floor on ONE output block.

    A phi that ignores its input still scores a respectable Sinkhorn loss by sitting on
    the target's mean. Then every per-gene correlation is ~0 and the spread across cells
    is ~0, and that ratio is what tells the two apart from the loss curve alone. FOSCTTM
    is measured here rather than left to the evaluate stage because a run whose alignment
    never happened has to be caught before the other 17 conditions are launched.
    """
    per_gene = torch.stack([
        torch.corrcoef(torch.stack([prediction[:, col], observed[:, col]]))[0, 1]
        for col in range(0, prediction.shape[1], max(1, prediction.shape[1] // n_probe))
    ])
    sampled_prediction = prediction[sample].detach().cpu().numpy()
    sampled_observed = observed[sample].cpu().numpy()
    constant = np.broadcast_to(sampled_observed.mean(axis=0, keepdims=True),
                               sampled_prediction.shape).copy()
    # Same hub check as retrieval_metrics, on the FOSCTTM sample, without recomputing
    # the pairing floor. A collapsed cloud that always retrieves one observed cell
    # still posts a plausible FOSCTTM.
    partners = torch.cdist(prediction[sample], observed[sample]).argmin(dim=1)
    return {
        "foscttm": float(np.mean(calc_domainAveraged_FOSCTTM(
            sampled_prediction, sampled_observed))),
        # Gate on the permuted floor. The constant floor is kept so older preflight files remain readable.
        "foscttm_permuted_floor": permuted_pairing_floor(
            sampled_prediction, sampled_observed),
        "foscttm_constant_floor": float(np.mean(calc_domainAveraged_FOSCTTM(
            constant, sampled_observed))),
        "foscttm_n_cells": int(len(sample)),
        "state_pearson_median": float(torch.nanmedian(per_gene)),
        "state_pearson_positive_fraction": float((per_gene > 0).float().mean()),
        "prediction_spread_ratio": float((prediction.std(dim=0).median()
                                          / observed.std(dim=0).median()).clamp(max=1e6)),
        "partner_diversity": float(len(torch.unique(partners)) / max(len(sample), 1)),
    }


def preflight_checks(model, law: str, chromatin: torch.Tensor, target: torch.Tensor,
                     production: torch.Tensor, true_velocity: torch.Tensor,
                     rows: torch.Tensor, velocity_rows: torch.Tensor, scale: torch.Tensor,
                     adata: sc.AnnData, reference_layer: str,
                     reference_columns: np.ndarray | None = None,
                     target_layer: str | None = None) -> dict:
    """The gate before launching anything else.

    Reads the true velocity even when the run trained on a corrupted field: a checkpoint
    can look self-consistent against its own corrupted law while having learned nothing.
    """
    model.eval()
    residual, moving = kinetics_residual(model, law, chromatin[velocity_rows],
                                         true_velocity[velocity_rows],
                                         production[velocity_rows])
    pushforward = residual + law_rhs(model, law, moving, chromatin[velocity_rows],
                                     production[velocity_rows])
    with torch.no_grad():
        prediction = model.phi(chromatin[rows])
        probe = velocity_rows if len(velocity_rows) > 0 else rows
        kappa = model.kappa(chromatin[probe])
        alpha = model.g(alpha_features(model, chromatin, production)[probe])
    observed = target[rows]
    # Every evaluation cell, not a draw. An unseeded randperm here made FOSCTTM a
    # property of whichever process scored the checkpoint: eight rescorings of ONE
    # byte-identical checkpoint spanned 0.2198-0.2277 (SD 0.0025), wider than the
    # differences between the arms this metric exists to compare. calc_frac_idx walks
    # rows in GPU batches, so the whole val set costs batch-sized strips, not n^2.
    sample = torch.arange(len(rows), device=rows.device)
    checks = {
        **state_metrics(spliced_block(prediction, law), spliced_block(observed, law), sample),
        "state_prediction_finite": bool(torch.isfinite(prediction).all()),
        "jvp_norm_median": float(pushforward.norm(dim=1).median()),
        "jvp_finite": bool(torch.isfinite(pushforward).all()),
        "residual_norm_median": float((residual / scale).norm(dim=1).median()),
        "jvp_rhs_cosine_median": float(torch.nn.functional.cosine_similarity(
            pushforward, pushforward - residual, dim=1).median()),
        "jvp_n_cells": int(len(velocity_rows)),
        "target_layer": target_layer,
        "jvp_vs_reference_skipped": False,
        "kappa_median": float(kappa.median()),
        "kappa_at_floor": float((kappa < 1.1e-3).float().mean()),
        "alpha_at_floor": float((alpha < 1.1e-5).float().mean()),
    }
    # Gate stays on spliced RNA; report unspliced beside it because a relay plan can steer that block.
    if law == RELAY:
        checks.update({
            f"unspliced_{key}": value for key, value in state_metrics(
                unspliced_block(prediction, law), unspliced_block(observed, law),
                sample).items()})
    # Missing reference is null, not a skip. Total-RNA targets are a different quantity from the spliced reference.
    reference = reference_velocity(adata, reference_layer)
    checks["jvp_vs_reference_cosine_median"] = None
    if target_layer is not None and target_layer not in ("spliced_lognorm", SHARED_SPLICED):
        checks["jvp_vs_reference_skipped"] = True
        checks["jvp_vs_reference_skip_reason"] = (
            f"target {target_layer!r} is not spliced; scVelo reference comparison disabled")
    elif reference is not None:
        pushed = spliced_block(pushforward.detach().cpu().numpy(), law)
        checks.update(reference_agreement(
            pushed, spliced_block(moving.detach().cpu().numpy(), law), reference,
            velocity_rows, reference_columns, "jvp_vs_reference",
            coords=model_kinetic_coords(model)))
    # Reported, never gated, so both laws stay on one spliced standard.
    if law == RELAY:
        reference_u = reference_velocity(adata, f"{reference_layer}_u")
        if reference_u is not None:
            pushed_u = unspliced_block(pushforward.detach().cpu().numpy(), law)
            checks.update(reference_agreement(
                pushed_u, unspliced_block(moving.detach().cpu().numpy(), law), reference_u,
                velocity_rows, reference_columns, "jvp_vs_reference_u",
                coords=model_kinetic_coords(model)))
    return checks


def unspliced_block(prediction, law: str):
    """The u half of a relay prediction; the reduced law has no u output."""
    assert law == RELAY, "the reduced law predicts a single block"
    return prediction[:, :prediction.shape[1] // 2]


def spliced_block(prediction: np.ndarray, law: str) -> np.ndarray:
    """The s half of a relay prediction; the whole thing under the reduced law.

    §11: every RNA-state and cell-pairing comparison reads s-hat, so R1 and R2 are
    compared on the same quantity rather than on outputs of different shapes.
    """
    if law == REDUCED:
        return prediction
    return prediction[:, prediction.shape[1] // 2:]


def load_checkpoint(run_dir: Path, name: str, device: torch.device):
    """The model as trained, plus the settings the run was trained under."""
    payload = torch.load(run_dir / f"checkpoint_{name}.pt", map_location=device,
                         weights_only=False)
    config = json.loads((run_dir / "run_config.json").read_text())
    if payload.get("formulation") != FORMULATION or config.get("formulation") != FORMULATION:
        raise ValueError("Legacy chromatin checkpoint: its law/units differ from regulatory R2; do not reinterpret it")
    # Rebuild every architecture flag: spectral norm changes parameter names and a default model cannot load.
    input_names = payload.get("input_gene_names", payload["gene_names"])
    phi_state = payload["state_dict"]
    phi_kwargs = {}
    if "phi.gene_projection" in phi_state:
        phi_kwargs = {
            "phi_projection": phi_state["phi.gene_projection"],
            "phi_scale": phi_state["phi.gene_scale"],
            "phi_bias": phi_state["phi.gene_bias"],
            "phi_residual_weight": config.get("phi_residual_weight", 0.1),
            # Older checkpoints omit phi_gate; the wrong mode changes the parameter shape.
            "phi_gate": config.get("phi_gate", "scalar-zero"),
        }
    model = ChromatinKOT(
        len(payload["gene_names"]), payload["law"], n_input_features=len(input_names),
        phi_dims=config["phi_dims"], kappa_dims=config["kappa_dims"],
        g_dims=config["g_dims"], phi_init_gain=config["phi_init_gain"],
        phi_spectral_norm=config["phi_spectral_norm"], activation=config["activation"],
        init_method=config["init_method"], kappa_min=config["kappa_min"],
        kappa_max=config["kappa_max"], alpha_min=config["alpha_min"],
        alpha_max=config["alpha_max"],
        phi_trainable_projection=bool(config.get("trainable_g", False)),
        alpha_input=config.get("alpha_input", "gc"),
        **phi_kwargs,
    ).to(device)
    model.kinetic_coords = resolve_kinetic_coords(config.get("rna_kinetic_coords"))
    if config.get("fixed_kappa") is not None:
        model.kappa = FixedKappa(float(config["fixed_kappa"])).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload, config


def reference_velocity(adata: sc.AnnData, layer: str) -> np.ndarray | None:
    """An RNA velocity computed from RNA alone, on the panel's genes and cell order.

    Written by the `reference` stage, never by training: Task D only means something
    because this quantity was never available to the model.
    """
    if layer not in adata.layers:
        print(f"[evaluate] reference layer '{layer}' absent — run `reference` first")
        return None
    return to_dense(adata.layers[layer], np.float32)


def reference_main(args: argparse.Namespace) -> int:
    """§17. scVelo on the panel's own spliced/unspliced, kept out of every training run."""
    adata = load_dataset(args.dataset, args.split_seed)
    save_audit(adata, RELAY)
    rows = np.flatnonzero(adata.obs["has_splicing"].to_numpy()) if "has_splicing" in adata.obs \
        else np.arange(adata.n_obs)
    usable = usable_splicing_genes(adata)
    print(f"[reference] scVelo on {len(rows)} cells x {int(usable.sum())} genes "
          f"with usable spliced/unspliced counts")

    subset = adata[rows][:, usable].copy()
    if args.backend == "regvelo":
        # Second backend on the same input, so agreement that holds for only one velocity model is visible.
        subset, _ = run_regvelo(subset, n_top_genes=int(usable.sum()), hvg_flavor="seurat_v3",
                                min_shared_counts=args.min_shared_counts, n_pcs=args.n_pcs,
                                n_neighbors=args.n_neighbors, grn_prior_csv=args.grn_prior_csv)
    else:
        # Same velocity pipeline as the RNA–protein runs, so the two experiments' references stay comparable.
        subset, _ = preprocess_for_velocity(
            subset, n_top_genes=subset.n_vars, hvg_flavor="seurat_v3",
            min_shared_counts=args.min_shared_counts, n_pcs=args.n_pcs,
            n_neighbors=args.n_neighbors)
        scv.tl.recover_dynamics(subset, n_jobs=args.n_jobs)
        scv.tl.velocity(subset, mode="dynamical")

    columns = adata.var_names.get_indexer(subset.var_names)
    # Relay predicts both blocks, so the unspliced reference has to exist too.
    output_layer = args.layer or f"velocity_{args.backend}"
    for source, layer in [("velocity", output_layer), ("velocity_u", f"{output_layer}_u")]:
        if source not in subset.layers:
            print(f"[reference] {args.backend} wrote no '{source}' layer — skipping {layer}")
            continue
        field = np.zeros((adata.n_obs, adata.n_vars), dtype=np.float32)
        field[np.ix_(rows, columns)] = np.nan_to_num(to_dense(subset.layers[source], np.float32))
        adata.layers[layer] = sparse.csr_matrix(field)
        covered = int(np.abs(field).sum(axis=1).astype(bool).sum())
        print(f"[reference] wrote layer '{layer}': {covered} cells carry a velocity")
    adata.obs["has_reference_velocity"] = False
    adata.obs.iloc[rows, adata.obs.columns.get_loc("has_reference_velocity")] = True
    adata.write_h5ad(dataset_path(args.dataset, args.split_seed))
    return 0


def pushforward_and_rhs(model, law: str, chromatin: torch.Tensor, velocity: torch.Tensor,
                        production: torch.Tensor, batch_size: int
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """J_phi(c) v_c, the kinetic-law RHS, and phi's state, in batches.

    The state comes back because any comparison against an RNA-only velocity has to
    convert the log1p rate to a linear one, and that conversion needs it.
    """
    push, rhs_blocks, states = [], [], []
    for start in range(0, len(chromatin), batch_size):
        span = slice(start, start + batch_size)
        residual, prediction = kinetics_residual(model, law, chromatin[span],
                                                 velocity[span], production[span])
        rhs = law_rhs(model, law, prediction, chromatin[span], production[span])
        push.append((residual + rhs).detach().cpu().numpy())
        rhs_blocks.append(rhs.detach().cpu().numpy())
        states.append(prediction.detach().cpu().numpy())
    return np.concatenate(push), np.concatenate(rhs_blocks), np.concatenate(states)


def production_rows(chromatin_np: np.ndarray, mapping, rows: np.ndarray,
                    device: torch.device) -> torch.Tensor:
    """G c for a subset of cells. G is (genes x activity), so the product is C @ G.T."""
    return torch.as_tensor(
        (chromatin_np[rows] @ mapping.toarray().T).astype(np.float32), device=device)


def production_from_phi(model, chromatin_np: np.ndarray, rows: np.ndarray,
                        device: torch.device, n_genes: int, fallback_mapping
                        ) -> torch.Tensor:
    """Project through phi's current affine path for state-map diagnostics.

    This is not alpha's training input when phi's projection is trainable. Evaluation
    of the kinetic law uses checkpoint_alpha_projection instead.
    """
    phi = model.phi
    if not hasattr(phi, "gene_projection"):
        return production_rows(chromatin_np, fallback_mapping, rows, device)
    projection = phi.gene_projection.detach()
    if projection.shape[0] == 2 * n_genes:
        projection = projection[:n_genes]
    mapping = projection.cpu().numpy()
    return torch.as_tensor(
        (chromatin_np[rows] @ mapping.T).astype(np.float32), device=device)


def checkpoint_alpha_projection(payload, config, model, curated_mapping):
    """Recover the projection used to precompute alpha inputs during training."""
    if "alpha_projection" in payload:
        mapping = sparse.csr_matrix(payload["alpha_projection"].detach().cpu().numpy())
    elif config.get("gene_map_mode", "curated") == "curated":
        mapping = curated_mapping
    elif not config.get("trainable_g", False) and hasattr(model.phi, "gene_projection"):
        mapping = sparse.csr_matrix(
            model.phi.gene_projection[:len(payload["gene_names"])].detach().cpu().numpy())
    else:
        raise ValueError(
            "Legacy checkpoint did not save alpha's training G; it cannot be recovered "
            "from a trainable phi projection")
    expected = (len(payload["gene_names"]),
                len(payload.get("input_gene_names", payload["gene_names"])))
    if mapping.shape != expected:
        raise ValueError(f"Saved alpha projection has shape {mapping.shape}, expected {expected}")
    return mapping


def alpha_features(model, chromatin: torch.Tensor, production: torch.Tensor) -> torch.Tensor:
    """Alpha's input. G is a 0/1 diagonal, so Gc is per-gene; `full` keeps the mixing the paired ceiling uses."""
    return chromatin if getattr(model, "alpha_input", "gc") == "full" else production


EVAL_SPLITS = ("val", "test")


def evaluation_split_rows(split_column: np.ndarray, target_covered: np.ndarray,
                          eval_split: str) -> np.ndarray:
    """Paired val or test cells that have a measured RNA target.

    Training cells are unpaired by construction and are not an evaluation split.
    """
    if eval_split not in EVAL_SPLITS:
        raise ValueError(f"eval_split must be val or test, got {eval_split!r}")
    rows = np.flatnonzero((split_column == eval_split) & target_covered)
    if len(rows) == 0:
        raise ValueError(f"No measured {eval_split} cells to score")
    return rows


def write_evaluation_outputs(run_dir: Path, checkpoint: str, eval_split: str,
                             results: dict, per_gene: pd.DataFrame) -> Path:
    """Split-tagged files so a val sweep cannot overwrite a frozen test score."""
    payload = json.dumps(results, indent=2, default=str)
    tagged = run_dir / f"evaluation_{checkpoint}_{eval_split}.json"
    tagged.write_text(payload)
    per_gene.to_csv(run_dir / f"task_a_per_gene_{checkpoint}_{eval_split}.csv", index=False)
    if eval_split == "test":
        (run_dir / f"evaluation_{checkpoint}.json").write_text(payload)
        per_gene.to_csv(run_dir / f"task_a_per_gene_{checkpoint}.csv", index=False)
    return tagged


def evaluate_main(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    device = choose_torch_device({"device": args.device})
    model, payload, config = load_checkpoint(run_dir, args.checkpoint, device)
    dataset, law = payload["dataset"], payload["law"]
    prepared_seed = (config["split_seed"]
                     if config.get("preprocessing_protocol") == PREPROCESSING_PROTOCOL else None)
    adata = load_dataset(dataset, prepared_seed)
    if (prepared_seed is not None
            and config.get("preprocessing_id") != adata.uns["chromatin_preprocessing"]["id"]):
        raise ValueError("Checkpoint belongs to a different prepared dataset")
    save_audit(adata, law)
    splits = pd.read_csv(split_path(dataset, config["split_seed"]), index_col=0)
    # reference rewrites the h5ad in place; an older split would misalign every row without raising.
    assert splits.index.equals(adata.obs_names), (
        "the split file does not match this dataset — rebuild the split")
    fields = load_velocity(dataset, config["split_seed"],
                           config.get("chromatin_transform", "as_is"),
                           config.get("velocity_tag", ""),
                           protocol=config.get("velocity_protocol_version", 3))
    validate_velocity_inputs(fields, adata)

    chromatin_np = chromatin_features(adata, config.get("chromatin_transform", "as_is"))
    tau = float(config.get("phi_lag_tau", 0.0) or 0.0)
    if tau != 0.0:
        chromatin_np = euler_lag(chromatin_np, fields["velocity"], tau)
        print(f"[evaluate] phi lag tau={tau:g}: scoring φ(c - tau v_c)")
    regulatory_np = regulatory_features(
        adata, config.get("chromatin_transform", "as_is"),
        config.get("regulatory_transform", "same"), map_features=chromatin_np)
    chromatin = torch.as_tensor(chromatin_np, device=device)
    input_names = list(payload.get("input_gene_names", payload["gene_names"]))
    assert input_names == list(adata.var_names), "checkpoint ATAC features do not match dataset"
    output_columns = adata.var_names.get_indexer(payload["gene_names"])
    assert (output_columns >= 0).all(), "checkpoint RNA genes do not match dataset"
    full_map, _, _ = gene_map(adata, gene_map_path(dataset))
    true_map = full_map[output_columns].tocsr()
    train_map = checkpoint_alpha_projection(payload, config, model, true_map)
    true_velocity_np = fields["velocity"]
    split_column = splits["split"].to_numpy()
    side_column = splits["train_side"].to_numpy()
    train_source = (split_column == "train") & (side_column == "atac")
    shuffle_eligible = train_source & (np.asarray(fields["norm"]) > 0)
    train_velocity_np, _, train_dynamic = apply_velocity_condition(
        fields["velocity"], fields["confidence"], fields["dynamic_mask"],
        config["condition"], config["seed"], eligible=shuffle_eligible)

    target_covered = target_cell_mask(adata, config["target_layer"])
    eval_split = args.eval_split
    eval_rows = evaluation_split_rows(split_column, target_covered, eval_split)
    print(f"[evaluate] scoring {len(eval_rows)} paired {eval_split} cells with measured "
          f"{config['target_layer']}"
          + ("" if target_covered.all() else "; absent targets are not treated as zero"))
    observed_full = build_targets(
        adata[eval_rows], law, config["target_layer"], output_columns,
        kinetic_coords=config.get("rna_kinetic_coords", SHARED_LOG1P),
        global_scale=config.get("rna_global_scale"))
    observed = spliced_block(observed_full, law)
    with torch.no_grad():
        predicted_full = model.phi(chromatin[eval_rows]).cpu().numpy()
    predicted = spliced_block(predicted_full, law)

    results = {"run_dir": str(run_dir), "checkpoint": args.checkpoint, "dataset": dataset,
               "law": law, "formulation": FORMULATION,
               "condition": config["condition"], "seed": config["seed"],
               "eval_split": eval_split, "n_eval_cells": int(len(eval_rows))}
    if eval_split == "test":
        results["n_test_cells"] = int(len(eval_rows))
    gene_names = list(payload["gene_names"])
    summary, per_gene = task_a_state(predicted, observed, gene_names)
    results["task_a"] = summary
    groups = evaluation_groups(adata, fields, eval_rows)
    if groups is not None:
        results["task_a_by_group"] = by_group(
            lambda a, b: task_a_state(a, b, gene_names)[0], predicted, observed,
            groups)

    labels = adata.obs["cell_type"].astype(str).to_numpy()[eval_rows] \
        if "cell_type" in adata.obs else None
    results["task_b"] = task_b_pairing(predicted, observed, labels,
                                       args.max_sinkhorn_cells, config["seed"])
    group_column = "day" if dataset == "hspc" else "DonorID"
    results["task_b_within_group"] = foscttm_within(
        predicted, observed, adata.obs[group_column].to_numpy()[eval_rows])

    if labels is None:
        print("[evaluate] no cell_type annotation — Task C and the cell-type "
              "retrieval metrics are not defined for this dataset")
    elif args.tasks_c:
        rows = subsample_rows_for_graph(len(eval_rows), args.max_graph_cells, config["seed"])
        results["task_c"] = task_c_integration(
            predicted[rows], observed[rows], labels[rows],
            lambda joint: leiden_clusters(joint, args.n_neighbors), args.n_neighbors)

    internal_rows = np.flatnonzero(
        (split_column == "train") & (side_column == "atac") & train_dynamic)

    results["task_d"] = {}
    # Cells, velocity, and G that entered the loss: whether optimisation satisfied its own law.
    results["task_d"]["internal_n_cells"] = int(len(internal_rows))
    if len(internal_rows) > 0:
        jvp_train, rhs_train, _ = pushforward_and_rhs(
            model, law, chromatin[internal_rows],
            torch.as_tensor(train_velocity_np[internal_rows].astype(np.float32), device=device),
            alpha_features(
                model,
                torch.as_tensor(regulatory_np[internal_rows], device=device),
                production_rows(regulatory_np, train_map, internal_rows, device)),
            args.batch_size)
        results["task_d"]["internal_pushforward_norm_median"] = float(
            np.median(np.linalg.norm(jvp_train, axis=1)))
        results["task_d"]["internal_vs_law"] = task_d_kinetics(jvp_train, rhs_train)

    # Held-out cells use uncorrupted velocity and the saved alpha-input map.
    moving = np.flatnonzero(np.abs(true_velocity_np[eval_rows]).sum(axis=1) > 0)
    results["task_d"]["biological_n_moving_cells"] = int(len(moving))
    if len(moving) > 0:
        rows_d = eval_rows[moving]
        jvp_true, rhs_true, state_true = pushforward_and_rhs(
            model, law, chromatin[rows_d],
            torch.as_tensor(true_velocity_np[rows_d].astype(np.float32), device=device),
            alpha_features(
                model,
                torch.as_tensor(regulatory_np[rows_d], device=device),
                production_rows(regulatory_np, train_map, rows_d, device)),
            args.batch_size)
        results["task_d"]["biological_pushforward_norm_median"] = float(
            np.median(np.linalg.norm(jvp_true, axis=1)))
        results["task_d"]["biological_vs_law"] = task_d_kinetics(jvp_true, rhs_true)
        for name in args.reference_layers:
            reference = reference_velocity(adata, name)
            if reference is None:
                continue
            block = reference[rows_d][:, output_columns]
            scored = np.flatnonzero(np.abs(block).sum(axis=1) > 0)
            genes = np.flatnonzero(np.abs(block).sum(axis=0) > 0)
            results["task_d"][f"biological_vs_{name}_n_genes"] = int(len(genes))
            if len(scored) == 0 or len(genes) == 0:
                continue
            # Same conversion the preflight uses; comparing a log1p rate with scVelo's
            # linear one is a units error that reads as "no agreement".
            predicted_s = state_rate_to_linear(
                spliced_block(jvp_true, law), spliced_block(state_true, law),
                model_kinetic_coords(model))
            pushed = predicted_s[np.ix_(scored, genes)]
            observed_velocity = block[np.ix_(scored, genes)]
            results["task_d"][f"biological_vs_{name}"] = task_d_kinetics(
                pushed, observed_velocity)
            if groups is not None:
                results["task_d"][f"biological_vs_{name}_by_group"] = by_group(
                    task_d_kinetics, pushed, observed_velocity, groups[moving][scored])
            unspliced_reference = reference_velocity(adata, f"{name}_u")
            if law == RELAY and unspliced_reference is not None:
                u_block = unspliced_reference[rows_d][:, output_columns]
                u_scored = np.flatnonzero(np.abs(u_block).sum(axis=1) > 0)
                u_genes = np.flatnonzero(np.abs(u_block).sum(axis=0) > 0)
                predicted_u = state_rate_to_linear(
                    unspliced_block(jvp_true, law), unspliced_block(state_true, law),
                    model_kinetic_coords(model))
                results["task_d"][f"biological_vs_{name}_unspliced_n_genes"] = int(
                    len(u_genes))
                if len(u_scored) > 0 and len(u_genes) > 0:
                    results["task_d"][f"biological_vs_{name}_unspliced"] = task_d_kinetics(
                        predicted_u[np.ix_(u_scored, u_genes)],
                        u_block[np.ix_(u_scored, u_genes)])

                common_cells = np.intersect1d(scored, u_scored)
                common_genes = np.intersect1d(genes, u_genes)
                if len(common_cells) > 0 and len(common_genes) > 0:
                    predicted_joint = np.concatenate([
                        predicted_u[np.ix_(common_cells, common_genes)],
                        predicted_s[np.ix_(common_cells, common_genes)],
                    ], axis=1)
                    reference_joint = np.concatenate([
                        u_block[np.ix_(common_cells, common_genes)],
                        block[np.ix_(common_cells, common_genes)],
                    ], axis=1)
                    results["task_d"][f"biological_vs_{name}_joint_us"] = task_d_kinetics(
                        predicted_joint, reference_joint)
    out = write_evaluation_outputs(run_dir, args.checkpoint, eval_split, results, per_gene)
    print(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {out}")
    return 0


def preflight_main(args: argparse.Namespace) -> int:
    """Recompute the launch gate after `reference` writes scVelo; does not retrain."""
    run_dir = Path(args.run_dir)
    device = choose_torch_device({"device": args.device})
    model, payload, config = load_checkpoint(run_dir, args.checkpoint, device)
    dataset, law = payload["dataset"], payload["law"]
    prepared_seed = (config["split_seed"]
                     if config.get("preprocessing_protocol") == PREPROCESSING_PROTOCOL else None)
    adata = load_dataset(dataset, prepared_seed)
    if (prepared_seed is not None
            and config.get("preprocessing_id") != adata.uns["chromatin_preprocessing"]["id"]):
        raise ValueError("Checkpoint belongs to a different prepared dataset")
    splits = pd.read_csv(split_path(dataset, config["split_seed"]), index_col=0)
    assert splits.index.equals(adata.obs_names), (
        "the split file does not match this dataset — rebuild the split")
    fields = load_velocity(dataset, config["split_seed"],
                           config.get("chromatin_transform", "as_is"),
                           config.get("velocity_tag", ""),
                           protocol=config.get("velocity_protocol_version", 3))
    validate_velocity_inputs(fields, adata)

    chromatin_np = chromatin_features(adata, config.get("chromatin_transform", "as_is"))
    tau = float(config.get("phi_lag_tau", 0.0) or 0.0)
    if tau != 0.0:
        chromatin_np = euler_lag(chromatin_np, fields["velocity"], tau)
        print(f"[preflight] phi lag tau={tau:g}: scoring φ(c - tau v_c)")
    regulatory_np = regulatory_features(
        adata, config.get("chromatin_transform", "as_is"),
        config.get("regulatory_transform", "same"), map_features=chromatin_np)
    output_columns = adata.var_names.get_indexer(payload["gene_names"])
    assert (output_columns >= 0).all(), "checkpoint RNA genes do not match dataset"
    gene_mask = np.zeros(adata.n_vars, dtype=bool)
    gene_mask[output_columns] = True
    target_layer = config["target_layer"]
    target_np = build_targets(
        adata, law, target_layer, gene_mask,
        kinetic_coords=resolve_kinetic_coords(config.get("rna_kinetic_coords")),
        global_scale=config.get("rna_global_scale"))
    full_map, _, _ = gene_map(adata, gene_map_path(dataset, prepared_seed))
    true_map = full_map[gene_mask].tocsr()
    train_map = checkpoint_alpha_projection(payload, config, model, true_map)
    mapping_dense = train_map.toarray().astype(np.float32)
    split_column = splits["split"].to_numpy()
    side_column = splits["train_side"].to_numpy()
    target_covered = target_cell_mask(adata, target_layer)
    chromatin = torch.as_tensor(chromatin_np, device=device)
    target = torch.as_tensor(target_np, device=device)
    if config.get("regulatory_transform", "same") in (None, "same"):
        regulatory = chromatin
        regulatory_production = torch.as_tensor(
            (chromatin_np @ mapping_dense.T).astype(np.float32), device=device)
    else:
        regulatory = torch.as_tensor(regulatory_np, device=device)
        regulatory_production = torch.as_tensor(
            (regulatory_np @ mapping_dense.T).astype(np.float32), device=device)
    alpha_input = regulatory if config.get("alpha_input", "gc") == "full" else regulatory_production
    rna_rows = np.flatnonzero(
        target_covered & (split_column == "train") & (side_column == "rna"))
    if len(rna_rows) == 0:
        raise ValueError("No measured RNA training cells to reconstruct the target scale")
    target_scale = block_scales(target[torch.as_tensor(rna_rows, device=device)], law)
    val_rows = evaluation_split_rows(split_column, target_covered, "val")
    velocity_val = np.flatnonzero(
        np.isin(np.arange(adata.n_obs), val_rows)
        & np.asarray(fields["dynamic_mask"])
        & target_covered)
    print(f"[preflight] {run_dir} checkpoint={args.checkpoint}  "
          f"val {len(val_rows)} cells, {len(velocity_val)} with chromatin velocity")
    checks = preflight_checks(
        model, law, chromatin, target, alpha_input,
        torch.as_tensor(fields["velocity"].astype(np.float32), device=device),
        torch.as_tensor(val_rows, device=device),
        torch.as_tensor(velocity_val, device=device),
        target_scale, adata, args.reference_layer,
        reference_columns=np.flatnonzero(gene_mask),
        target_layer=target_layer)
    return persist_preflight(
        run_dir, checks, config["condition"], "preflight", refuse_failed_gates=False,
        checkpoint=args.checkpoint)


def evaluation_groups(adata: sc.AnnData, fields: dict, rows: np.ndarray) -> np.ndarray | None:
    """The strata results are broken down by: lineage for BMMC, day for HSPC.

    §8 asks for the kinetic read per lineage where possible. The labels come from the
    velocity file, which recorded them from cell types — used only to GROUP results, never
    to build the field they group.
    """
    if "lineage" in fields:
        return np.asarray(fields["lineage"])[rows]
    if "day" in adata.obs:
        return adata.obs["day"].astype(str).to_numpy()[rows]
    return None


def subsample_rows_for_graph(n_rows: int, limit: int, seed: int) -> np.ndarray:
    """Task C builds several kNN graphs on a 2n-row stack, which is quadratic in memory."""
    if n_rows <= limit:
        return np.arange(n_rows)
    return np.sort(np.random.default_rng(seed).choice(n_rows, limit, replace=False))


def leiden_clusters(matrix: np.ndarray, n_neighbors: int) -> np.ndarray:
    """Unsupervised clusters of the observed RNA, the partition ARI and NMI score against."""
    handle = sc.AnnData(X=matrix)
    sc.pp.neighbors(handle, n_neighbors=n_neighbors, use_rep="X")
    sc.tl.leiden(handle, flavor="igraph", n_iterations=2, directed=False)
    return handle.obs["leiden"].astype(str).to_numpy()

def baseline_main(args: argparse.Namespace) -> int:
    """§14's reference points, on the same split KOT is scored on.

    A per-gene correlation has no meaning without knowing what is achievable. On this
    modality pair that ceiling is LOW: chromatin accessibility is a weak and lagged
    predictor of expression, which is the premise of every multi-omic velocity method. So
    Task A is reported against four references, in increasing order of what they are
    allowed to see:

      mean-only   predict the training mean for every cell. Per-gene correlation 0 by
                  construction; its CELL cosine is the floor a profile metric must beat.
      identity    the gene's own chromatin activity, untransformed. What G asserts, with
                  no model on top.
      ridge       supervised least squares WITH the pairing: a paired upper bound.
      paired_mlp  a nonlinear supervised translator with the same paired advantage.

    The two supervised methods are paired upper bounds, never fair unpaired competitors.
    """
    adata = load_dataset(args.dataset, args.split_seed)
    save_audit(adata)
    splits = pd.read_csv(split_path(args.dataset, args.split_seed), index_col=0)
    assert splits.index.equals(adata.obs_names), "split file does not match the dataset"
    target_layer = rna_target_layer(adata, args.rna_target)
    chromatin = np.asarray(adata.obsm["gene_activity"], dtype=np.float32)
    observed = to_dense(adata.layers[target_layer], np.float32)

    measured = target_cell_mask(adata, target_layer)
    train = np.flatnonzero((splits["split"].to_numpy() == "train") & measured)
    test = np.flatnonzero((splits["split"].to_numpy() == "test") & measured)
    train = np.random.default_rng(args.split_seed).choice(
        train, min(args.max_fit_cells, len(train)), replace=False)
    print(f"[baseline] {args.dataset}: fit on {len(train)} cells, score {len(test)}, "
          f"target '{target_layer}'")

    paired_mlp = MLPRegressor(
        hidden_layer_sizes=tuple(args.mlp_hidden), random_state=args.split_seed,
        max_iter=args.mlp_max_iter, batch_size=min(args.mlp_batch_size, len(train)),
        early_stopping=True, validation_fraction=0.1,
    ).fit(chromatin[train], observed[train])
    predictions = {
        "mean_only": np.tile(observed[train].mean(axis=0), (len(test), 1)),
        "identity": chromatin[test],
        "ridge": Ridge(alpha=args.ridge_alpha).fit(
            chromatin[train], observed[train]).predict(chromatin[test]).astype(np.float32),
        "paired_mlp": paired_mlp.predict(chromatin[test]).astype(np.float32),
    }
    # Unmapped genes cannot be predicted from chromatin; pooling them understates the covered-gene ceiling.
    _, _, kinetic_mask = gene_map(adata, gene_map_path(args.dataset))
    subsets = {"all": np.ones(adata.n_vars, dtype=bool), "in_G": kinetic_mask,
               "not_in_G": ~kinetic_mask}
    results = {}
    for name, predicted in predictions.items():
        summary, per_gene = task_a_state(predicted, observed[test], list(adata.var_names))
        per_gene["in_G"] = kinetic_mask
        # Target in the filename so parallel targets do not overwrite each other.
        per_gene.to_csv(
            CACHE_ROOT / f"{args.dataset}_{target_layer}_baseline_{name}_per_gene.csv",
            index=False)
        results[name] = {
            subset: {
                "gene_pearson_median": float(np.nanmedian(per_gene.loc[mask, "pearson"])),
                "gene_spearman_median": float(np.nanmedian(per_gene.loc[mask, "spearman"])),
                "n_genes": int(mask.sum()),
            } for subset, mask in subsets.items() if mask.any()
        }
        results[name]["cell_cosine_median"] = summary["cell_cosine_median"]
        results[name]["reference_category"] = (
            "paired_upper_bound" if name in {"ridge", "paired_mlp"} else "simple_reference")
        print(f"  {name:<10} gene pearson  all {results[name]['all']['gene_pearson_median']:+.4f}"
              f"  in_G {results[name]['in_G']['gene_pearson_median']:+.4f}"
              f"  not_in_G {results[name]['not_in_G']['gene_pearson_median']:+.4f}"
              f"   cell cosine {summary['cell_cosine_median']:.4f}")
    out = CACHE_ROOT / f"{args.dataset}_{target_layer}_baselines.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"[baseline] wrote {out}")
    return 0


def baseline_model_main(args: argparse.Namespace) -> int:
    """Run one competing method on this experiment's split and score it like KOT.

    The baselines answer Tasks A-C. Task D needs a differentiable ATAC->RNA map and only
    scDART has one, so it is left to a separate pass rather than faked for the rest.
    """
    adata = load_dataset(args.dataset, args.split_seed)
    save_audit(adata)
    splits = pd.read_csv(split_path(args.dataset, args.split_seed), index_col=0)
    assert splits.index.equals(adata.obs_names), "split file does not match the dataset"
    target_layer = rna_target_layer(adata, args.rna_target)
    chromatin = np.asarray(adata.obsm["gene_activity"], dtype=np.float32)
    rna = to_dense(adata.layers[target_layer], np.float32)

    split_column = splits["split"].to_numpy()
    side_column = splits["train_side"].to_numpy()
    measured = target_cell_mask(adata, target_layer)
    atac_train = np.flatnonzero((split_column == "train") & (side_column == "atac"))
    rna_train = np.flatnonzero((split_column == "train") & (side_column == "rna") & measured)
    test = np.flatnonzero((split_column == "test") & measured)
    assert len(set(atac_train) & set(rna_train)) == 0, "ATAC and RNA training sets overlap"

    days = adata.obs["day"].to_numpy() if "day" in adata.obs else None
    inputs = BaselineInputs(
        atac_train=chromatin[atac_train], rna_train=rna[rna_train],
        atac_test=chromatin[test], rna_test=rna[test],
        genes=list(adata.var_names), seed=args.seed,
        day_train_atac=None if days is None else days[atac_train],
        day_train_rna=None if days is None else days[rna_train],
        day_test=None if days is None else days[test],
    )
    print(f"[baseline] {args.method} on {args.dataset}: atac_train {len(atac_train)}, "
          f"rna_train {len(rna_train)}, test {len(test)}, genes {adata.n_vars}")

    started = datetime.now()
    result = BASELINES[args.method](inputs, **baseline_kwargs(args))
    runtime = (datetime.now() - started).total_seconds()

    # Score both modalities on the subsample the method actually saw, so each test cell is compared to its partner.
    scored = np.asarray(result.notes.get("test_rows", np.arange(len(test))))
    predicted = result.predicted
    if predicted is None:
        predicted = impute_from_latent(result, args.n_neighbors)
    observed = rna[test][scored]

    output_dir = Path(args.run_dir or CACHE_ROOT / "runs"
                      / f"baseline_{args.method}_{args.dataset}_seed{args.seed}")
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {"method": args.method, "dataset": args.dataset, "seed": args.seed,
               "runtime_seconds": runtime, "prediction_kind": result.prediction_kind,
               "n_test_scored": int(len(scored)),
               "notes": {k: v for k, v in result.notes.items() if not k.endswith("_rows")}}

    summary, per_gene = task_a_state(predicted, observed, list(adata.var_names))
    per_gene.to_csv(output_dir / "task_a_per_gene.csv", index=False)
    results["task_a"] = summary

    # Score B and C in the method's own space; imputing to genes would measure the imputer.
    labels = adata.obs["cell_type"].astype(str).to_numpy()[test][scored] \
        if "cell_type" in adata.obs else None
    if result.atac_latent is not None:
        assert len(result.atac_latent) == len(result.rna_latent) == len(scored), (
            "Task B needs the two modalities on the SAME cells; the method returned "
            f"{len(result.atac_latent)} ATAC and {len(result.rna_latent)} RNA rows for "
            f"{len(scored)} scored cells")
        results["task_b"] = task_b_pairing(result.atac_latent, result.rna_latent, labels,
                                           args.max_sinkhorn_cells, args.seed)
        if labels is not None:
            rows = subsample_rows_for_graph(len(scored), args.max_graph_cells, args.seed)
            results["task_c"] = task_c_integration(
                result.atac_latent[rows], result.rna_latent[rows], labels[rows],
                lambda joint: leiden_clusters(joint, args.n_neighbors), args.n_neighbors)

    out = output_dir / "evaluation_baseline.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(json.dumps(results, indent=2, default=str))
    print(f"\n[baseline] wrote {out}")
    return 0


def baseline_kwargs(args: argparse.Namespace) -> dict:
    """The per-method knobs, kept out of the driver so it reads as one flow."""
    return {
        "moscot": {"max_cells": args.max_cells, "epsilon": args.epsilon,
                   "n_comps": args.n_comps},
        "scdart": {"n_epochs": args.n_epochs, "latent_dim": args.latent_dim,
                   "max_cells": args.max_cells,
                   "device": choose_torch_device({"device": args.device})},
        "scglue": {"n_comps": args.n_comps, "max_epochs": args.n_epochs},
        "maxfuse": {"max_cells": args.max_cells, "n_comps": args.n_comps},
        "scmultinode": {"latent_dim": args.latent_dim, "iters": args.n_epochs},
    }[args.method]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="build the paired Multiome object")
    prepare.add_argument("--dataset", choices=DATASETS, required=True)
    prepare.add_argument("--n-top-genes", type=int, default=2000)
    prepare.add_argument("--n-lsi", type=int, default=51,
                         help="SVD components before the depth component is dropped")
    prepare.add_argument("--min-peak-cells", type=int, default=50)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--split-seed", type=int, default=0)
    prepare.add_argument("--val-fraction", type=float, default=0.1)
    prepare.add_argument("--test-fraction", type=float, default=0.2)
    prepare.set_defaults(func=prepare_main)

    audit = sub.add_parser("audit", help="§3 dataset audit")
    audit.add_argument("--dataset", choices=DATASETS, required=True)
    audit.add_argument("--law", choices=LAWS, default=RELAY)
    audit.add_argument("--split-seed", type=int, default=0)
    audit.set_defaults(func=audit_main)

    check_r2 = sub.add_parser("check-r2", help="read-only R2 data, velocity and anchor coverage check")
    check_r2.add_argument("--dataset", choices=DATASETS, required=True)
    check_r2.add_argument("--split-seed", type=int, default=0)
    check_r2.add_argument("--gamma-anchor-csv", required=True)
    check_r2.add_argument("--velocity-tag", default="")
    check_r2.add_argument("--chromatin-transform", choices=CHROMATIN_TRANSFORMS,
                          default="as_is",
                          help="check the velocity cache and coverage for this transform")
    check_r2.set_defaults(func=check_r2_main)

    split = sub.add_parser("split", help="verify the disjoint split frozen by prepare")
    split.add_argument("--dataset", choices=DATASETS, required=True)
    split.add_argument("--seed", type=int, default=0)
    split.add_argument("--val-fraction", type=float, default=0.1,
                       help="legacy option; choose fractions on prepare before fitting")
    split.add_argument("--test-fraction", type=float, default=0.2,
                       help="legacy option; choose fractions on prepare before fitting")
    split.set_defaults(func=split_main)

    split.add_argument("--force", action="store_true",
                       help="rejected for prepared inputs; prepare a new split seed instead")
    velocity = sub.add_parser("velocity", help="§7/§8 chromatin velocity, from ATAC alone")
    velocity.add_argument("--dataset", choices=DATASETS, required=True)
    velocity.add_argument("--velocity-estimator", choices=VELOCITY_ESTIMATORS,
                          default="quotient",
                          help="'regression' uses a weighted least-squares slope instead of "
                               "a mean of per-neighbour difference quotients, so a "
                               "near-tied pseudotime gap cannot dominate a cell")
    velocity.add_argument("--velocity-tag", default="",
                          help="name this field variant so it does not overwrite another")
    velocity.add_argument("--chromatin-transform", choices=CHROMATIN_TRANSFORMS,
                          default="as_is",
                          help="normalisation of phi's chromatin input; the velocity is a "
                               "displacement in this space, so train must match it")
    velocity.add_argument("--split-seed", type=int, default=0,
                          help="must match the split every method reuses; OT/DPT are "
                               "fitted inside that split so test cells cannot write v_train")
    velocity.add_argument("--confidence-quantile", type=float, default=0.1,
                          help="drop this lowest-confidence share from the dynamic loss")
    velocity.add_argument("--no-gauge-normalize", dest="gauge_normalize",
                          action="store_false", default=True,
                          help="keep the raw velocity scale (the RNA→protein runs "
                               "gauge-normalise, so this is off by default)")
    velocity.add_argument("--epsilon", type=float, default=0.05,
                          help="hspc: entropic OT regularisation, on median-scaled cost")
    velocity.add_argument("--ot-iterations", type=int, default=200)
    velocity.add_argument("--n-neighbors", type=int, default=30,
                          help="bmmc: ATAC LSI graph size for pseudotime and the field")
    velocity.add_argument("--min-forward", type=int, default=3,
                          help="bmmc: cells with fewer forward neighbours get no velocity")
    velocity.set_defaults(func=velocity_main)

    train = sub.add_parser("train", help="§9-§12 train one law under one condition")
    train.add_argument("--dataset", choices=DATASETS, required=True)
    train.add_argument("--law", choices=LAWS, default=RELAY)
    train.add_argument("--condition", choices=CONDITIONS, default="full",
                       help="R2 (relay) only allows full/shuffle/noDyn")
    train.add_argument("--seed", type=int, default=None,
                       help=f"one of the RNA→protein seeds {rna_protein_seeds()[:3]} ...; "
                            "unset takes training.yaml's own `seed`")
    train.add_argument("--split-seed", type=int, default=0)
    train.add_argument("--run-dir", default=None)
    train.add_argument("--subsample", type=int, default=None,
                       help="cells to keep — the small-subset check before a large run")
    train.add_argument("--phi-affine", choices=["gene", "none"], default="gene",
                       help="'none' drops phi's gene-aware affine path and its unpaired "
                            "warm start, leaving a plain MLP")
    train.add_argument("--velocity-tag", default="",
                       help="which velocity variant to train against; must match the cache")
    train.add_argument("--alpha-input", choices=ALPHA_INPUTS, default="gc",
                       help="'full' lets the transcription head read all of c instead of "
                            "the gene's own activity through the diagonal G, so the law's "
                            "right-hand side can mix across genes")
    train.add_argument("--gene-map-mode", choices=GENE_MAP_MODES, default="curated",
                       help="curated G is a 0/1 DIAGONAL selection matrix, so phi's affine "
                            "Jacobian cannot mix genes; 'coaccess' adds ATAC-correlation "
                            "links, 'genomic' gene-body midpoint proximity, 'peak-genomic' "
                            "cellranger peak→gene links (HSPC), 'diagonal_full' drops the "
                            "chromatin-support mask")
    train.add_argument("--g-neighbors", type=int, default=10,
                       help="links per gene for --gene-map-mode coaccess")
    train.add_argument("--g-genomic-bp", type=float, default=100_000,
                       help="gene-body midpoint window for --gene-map-mode genomic")
    train.add_argument("--g-genomic-decay-bp", type=float, default=50_000,
                       help="exp(-d / this) weights for genomic and peak-genomic links")
    train.add_argument("--phi-lag-tau", type=float, default=0.0,
                       help="unpaired Euler lag: phi reads c - tau v_c. tau is in the "
                            "cached velocity's units AFTER gauge-normalisation, so 1 is "
                            "one typical ||v|| step (not one HSPC day). 0 is instantaneous; "
                            "negative is a future-chromatin control")
    train.add_argument("--trainable-g", action="store_true",
                       help="let the peak->gene projection G train instead of staying frozen")
    train.add_argument("--rna-target-units", choices=["shared", "spliced"], default="shared",
                       help="'spliced' predicts spliced_lognorm, the layer the competing "
                            "methods predict; requires --condition noDyn because it puts "
                            "u and s on different library factors")
    train.add_argument(
        "--rna-kinetic-coords", choices=list(RNA_KINETIC_COORDS), default=SHARED_LOG1P,
        help="coordinates the relay ODE is written in. shared_log1p is transcriptome "
             "CP10K then log1p with the log chain rule only. "
             "panel_log1p_compositional uses the modeled-gene library so "
             "d(x/T) includes -x dT/T. global_linear is one training-RNA scale "
             "and no per-cell normalisation derivative")
    train.add_argument("--chromatin-transform", choices=CHROMATIN_TRANSFORMS, default="as_is",
                       help="must match the transform the cached velocity was built with")
    train.add_argument(
        "--regulatory-transform", choices=REGULATORY_TRANSFORMS, default="same",
        help="coordinates alpha reads through G. same keeps G c in phi's map; "
             "cp10k_linear gives the transcription head interpretable gene activity")
    train.add_argument("--rna-target", choices=["auto", "spliced"], default="auto",
                       help="spliced RNA under the shared u+s library normalization")
    train.add_argument("--gamma-anchor-csv", required=True,
                       help="external RNA decay: gene_symbol,molecule,source,gamma_per_hour or half_life_hours")
    train.add_argument("--lambda-gamma-anchor", type=float, default=1.0)
    train.add_argument("--gamma-anchor-target", type=float, default=0.5,
                       help="whole-table geometric-mean gamma in relative model time")
    train.add_argument("--gamma-hours-per-model-time", type=float, default=None,
                       help="explicit physical time conversion; otherwise relative-rate anchors")
    train.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    train.add_argument("--training-config", default="config/training.yaml",
                       help="the RNA→protein hyperparameters every unset flag is taken from")
    train.add_argument("--n-epochs", type=int, default=None)
    train.add_argument("--batch-size", type=int, default=None)
    train.add_argument("--eval-every", type=int, default=None)
    train.add_argument("--lambda-dyn", type=float, default=None)
    train.add_argument("--dyn-residual-weight", type=float, default=None,
                       help="weight on the scaled L2 residual J_phi v - kappa * flux; "
                            "chromatin_r2 yaml sets this to 1 (the r2lsi recipe)")
    train.add_argument("--dyn-direction-weight", type=float, default=None,
                       help="weight on 1-cosine(J_phi v, kappa-free flux); fallback if "
                            "the residual still collapses by shrinking kappa")
    train.add_argument("--checkpoint-monitor", choices=["val_align", "foscttm"], default=None,
                       help="which validation metric writes checkpoint_best_align.pt; "
                            "foscttm is pairing, val_align is unpaired Sinkhorn")
    train.add_argument("--dyn-warmup-epochs", type=int, default=None)
    train.add_argument("--lr", type=float, default=None)
    train.add_argument("--lr-phi", type=float, default=None)
    train.add_argument("--lr-alpha-kappa", type=float, default=None)
    train.add_argument("--lr-rates", type=float, default=None)
    train.add_argument("--lr-warmup-epochs", type=int, default=None)
    train.add_argument("--lr-warmup-start-factor", type=float, default=None)
    train.add_argument("--lr-min-factor", type=float, default=None)
    train.add_argument(
        "--stabilization", choices=STABILIZATION_MODES, default="legacy",
        help="legacy: no κ prior, no weight decay, clip 5 (chromatin trainer as it stood). "
             "kot_parity: RNA→protein regularisers (κ prior log(2) at 0.01, λ_reg=1e-4, clip 1)")
    train.add_argument("--lambda-kappa-prior", type=float, default=None)
    train.add_argument("--kappa-prior-target", type=float, default=None)
    train.add_argument("--lambda-reg", type=float, default=None)
    train.add_argument("--grad-clip", type=float, default=None)
    train.add_argument("--sinkhorn-blur", type=float, default=None)
    train.add_argument("--sinkhorn-backend", default=None,
                       help="KeOps has no CUDA in this container; tensorized stays on GPU")
    train.add_argument("--sinkhorn-max-points", type=int, default=2048)
    train.add_argument("--align-dims", type=int, default=32,
                       help="measure the Sinkhorn cost in this many target directions "
                            "(0 = the full gene space)")
    train.add_argument("--phi-dims", type=int, nargs="+", default=None)
    train.add_argument("--phi-init-gain", type=float, default=None)
    train.add_argument("--align-block", choices=["spliced"], default="spliced",
                       help="s is the primary aligned output; u is auxiliary")
    train.add_argument("--lambda-held-block", type=float, default=1.0,
                       help="relay only: weight of the unpaired per-gene mean/std match "
                            "on the block the transport plan cannot see")
    train.add_argument("--phi-gate", choices=PHI_GATES, default="scalar-zero",
                       help="how much of the nonlinear path reaches phi's output: "
                            "scalar-zero starts it at exactly 0 and throttles its "
                            "gradient, none removes the switch, per-gene starts it open")
    train.add_argument("--phi-residual-weight", type=float, default=1.0,
                       help="maximum nonlinear residual weight around the trainable, "
                            "marginally calibrated G initialisation")
    train.add_argument("--phi-spectral-norm", type=lambda v: v.lower() == "true", default=None)
    train.add_argument("--activation", default=None)
    train.add_argument("--init-method", default=None)
    train.add_argument("--kappa-min", type=float, default=None)
    train.add_argument("--kappa-max", type=float, default=None)
    train.add_argument(
        "--fixed-kappa", type=float, default=None,
        help="replace the kappa network with κ(c)≡this constant. log(2) is the "
             "RNA→protein diagnostic: if the Jacobian only works when the clock "
             "cannot move, the scale head was absorbing the residual")
    train.add_argument("--alpha-min", type=float, default=None)
    train.add_argument("--alpha-max", type=float, default=None)
    train.add_argument("--kappa-dims", type=int, nargs="+", default=None)
    train.add_argument("--g-dims", type=int, nargs="+", default=None)
    train.add_argument("--reference-layer", default="velocity_scvelo",
                       help="RNA-only velocity the gate checks the JVP against")
    train.set_defaults(func=train_main)

    evaluate = sub.add_parser("evaluate", help="§13-§17 Tasks A-D on one checkpoint")
    evaluate.add_argument("--run-dir", required=True)
    evaluate.add_argument("--checkpoint", default="best_align")
    evaluate.add_argument(
        "--eval-split", choices=list(EVAL_SPLITS), default="val",
        help="paired cells for Tasks A-D. val is the development split; score test "
             "once after the configuration is frozen")
    evaluate.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    evaluate.add_argument("--batch-size", type=int, default=512)
    evaluate.add_argument("--n-neighbors", type=int, default=15)
    evaluate.add_argument("--max-graph-cells", type=int, default=6000)
    evaluate.add_argument("--max-sinkhorn-cells", type=int, default=4000,
                          help="Task B's OT matcher builds a dense n x n cost")
    evaluate.add_argument("--no-tasks-c", dest="tasks_c", action="store_false", default=True,
                          help="skip the joint-integration task (it builds several kNN "
                               "graphs on a 2n-row stack)")
    evaluate.add_argument("--reference-layers", nargs="+",
                          default=["velocity_scvelo", "velocity_regvelo"],
                          help="RNA-only velocity layers to score Task D against")
    evaluate.set_defaults(func=evaluate_main)

    preflight = sub.add_parser(
        "preflight", help="recompute the launch gate on a checkpoint after `reference`")
    preflight.add_argument("--run-dir", required=True)
    preflight.add_argument("--checkpoint", default="best_align")
    preflight.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    preflight.add_argument("--reference-layer", default="velocity_scvelo",
                           help="RNA-only velocity the gate checks the JVP against")
    preflight.set_defaults(func=preflight_main)

    reference = sub.add_parser("reference", help="§17 RNA-only velocity for Task D")
    reference.add_argument("--dataset", choices=DATASETS, required=True)
    reference.add_argument("--layer", default=None,
                           help="output layer (default: velocity_<backend>)")
    reference.add_argument("--backend", choices=["scvelo", "regvelo"], default="scvelo")
    reference.add_argument("--grn-prior-csv", default=None,
                           help="regvelo only: tools/build_inputs.py grn-prior output")
    reference.add_argument("--min-shared-counts", type=int, default=20)
    reference.add_argument("--n-pcs", type=int, default=30)
    reference.add_argument("--n-neighbors", type=int, default=30)
    reference.add_argument("--n-jobs", type=int, default=8)
    reference.add_argument("--split-seed", type=int, default=0)
    reference.set_defaults(func=reference_main)

    baseline = sub.add_parser("baseline", help="§14 simple and paired ridge/MLP references")
    baseline.add_argument("--dataset", choices=DATASETS, required=True)
    baseline.add_argument("--split-seed", type=int, default=0)
    baseline.add_argument("--rna-target", choices=["auto", "spliced", "rna", "unspliced"],
                          default="auto",
                          help="unspliced is the target chromatin drives DIRECTLY: "
                               "transcription makes u, and s is only made from u")
    baseline.add_argument("--ridge-alpha", type=float, default=100.0)
    baseline.add_argument("--max-fit-cells", type=int, default=20000)
    baseline.add_argument("--mlp-hidden", type=int, nargs="+", default=[256, 128])
    baseline.add_argument("--mlp-max-iter", type=int, default=200)
    baseline.add_argument("--mlp-batch-size", type=int, default=256)
    baseline.set_defaults(func=baseline_main)

    model = sub.add_parser("baseline-model", help="run a competing integration method")
    model.add_argument("--method", choices=sorted(BASELINES), required=True)
    model.add_argument("--dataset", choices=DATASETS, required=True)
    model.add_argument("--seed", type=int, default=rna_protein_seeds()[0])
    model.add_argument("--split-seed", type=int, default=0)
    model.add_argument("--run-dir", default=None)
    model.add_argument("--rna-target", choices=["auto", "spliced", "rna"], default="auto")
    model.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    model.add_argument("--max-cells", type=int, default=4000,
                       help="cap for the methods that build a dense n x n coupling")
    model.add_argument("--n-comps", type=int, default=50)
    model.add_argument("--n-epochs", type=int, default=500)
    model.add_argument("--latent-dim", type=int, default=8)
    model.add_argument("--epsilon", type=float, default=0.01)
    model.add_argument("--n-neighbors", type=int, default=15)
    model.add_argument("--max-graph-cells", type=int, default=6000)
    model.add_argument("--max-sinkhorn-cells", type=int, default=4000)
    model.set_defaults(func=baseline_model_main)

    return parser


def main() -> int:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
