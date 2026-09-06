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

  prepare   build the paired Multiome object: gene activity, LSI, spliced/unspliced
  audit     §3 — print and save what the data actually contains; refuse relay without u/s
  split     §4 — one 70/10/20 split, then destroy the training pairing
  velocity  §7/§8 — chromatin velocity: temporal OT for HSPC, a directional field for BMMC
  train     §9-§12 — law R1 (chromatin→mature RNA) or R2 (chromatin→unspliced→spliced)
  evaluate  §13-§17 — Tasks A-D, each scored against the TRUE velocity and the true G

The two laws:

  R1  J_phi(c) v_c ~ kappa(c) [alpha(c) (*) Gc - gamma (*) phi(c)]
  R2  Phi(c) = [u, s], with
      du = kappa(c) [alpha(c) (*) Gc - beta (*) u]
      ds = kappa(c) [beta (*) u        - gamma (*) s]
"""

from __future__ import annotations

import argparse
import json
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
    audit_report,
    build_bmmc,
    build_hspc,
    gene_map,
    make_splits,
    require_relay_inputs,
    usable_splicing_genes,
)
from src.data.chromatin_map import permute_chromatin_projection
from src.data.chromatin_velocity import (
    bmmc_velocity,
    direction_diversity,
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
from src.evaluation.foscttm import calc_domainAveraged_FOSCTTM
from src.losses.chromatin_laws import (
    LAWS,
    REDUCED,
    REDUCED_CONDITIONS,
    RELAY,
    alignment_gauge,
    block_scales,
    conditions_for_law,
    kinetics_loss,
    kinetics_residual,
    law_mask,
    law_rhs,
)
from src.losses.sinkhorn import sinkhorn_divergence
from src.models.chromatin_kot import ChromatinKOT
from src.training.chromatin_baselines import (
    BASELINES,
    BaselineInputs,
    impute_from_latent,
)
from src.training.kot import (
    choose_torch_device,
    dyn_weight_factor,
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
# The 13-batch rebuild (55,131 cells), not the two-batch cache the barcode-suffix bug
# left behind (10,780). Separate label, so the old file is still there to compare against.
BMMC_VELOCITY = (PROJECT_ROOT / "cache" / "velocity" / "bmmc_multiome_full"
                 / "bmmc_multiome_full_scvelo_results.h5ad")

DATASETS = ["hspc", "bmmc"]
VELOCITY_PROTOCOL_VERSION = 3


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


def dataset_path(dataset: str) -> Path:
    return CACHE_ROOT / f"{dataset}.h5ad"


def split_path(dataset: str, seed: int) -> Path:
    return CACHE_ROOT / f"{dataset}_split_seed{seed}.csv"


def gene_map_path(dataset: str) -> Path:
    return PROJECT_ROOT / "cache" / "results" / "mapping" / f"chromatin_gene_map_{dataset}.csv"


def velocity_path(dataset: str, split_seed: int = 0) -> Path:
    suffix = "" if split_seed == 0 else f"_split_seed{split_seed}"
    return CACHE_ROOT / f"{dataset}_velocity{suffix}.npz"


def load_velocity(dataset: str, split_seed: int = 0) -> dict:
    path = velocity_path(dataset, split_seed)
    assert path.exists(), f"{path} not built yet — run `velocity --dataset {dataset}` first"
    with np.load(path, allow_pickle=True) as handle:
        result = {key: handle[key] for key in handle.files}
    version = int(result.get("protocol_version", 0))
    assert version == VELOCITY_PROTOCOL_VERSION, (
        f"{path} uses velocity protocol {version}; rebuild it with the source-only protocol")
    cached_seed = int(result.get("split_seed", 0))
    assert cached_seed == split_seed, (
        f"{path} was built for split seed {cached_seed}, requested {split_seed}")
    return result


def load_dataset(dataset: str) -> sc.AnnData:
    path = dataset_path(dataset)
    assert path.exists(), f"{path} not built yet — run `prepare --dataset {dataset}` first"
    return sc.read_h5ad(path)


def prepare_main(args: argparse.Namespace) -> int:
    if args.dataset == "hspc":
        adata = build_hspc(HSPC_SOURCE, dataset_path("hspc"), args.n_top_genes, args.seed,
                           args.n_lsi, args.min_peak_cells, gene_map_path("hspc"))
    else:
        adata = build_bmmc(BMMC_PROCESSED, BMMC_VELOCITY, dataset_path("bmmc"),
                           args.n_top_genes, gene_map_path("bmmc"))
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
    adata = load_dataset(args.dataset)
    save_audit(adata, args.law)
    return 0


def split_main(args: argparse.Namespace) -> int:
    adata = load_dataset(args.dataset)
    save_audit(adata)
    out = split_path(args.dataset, args.seed)
    if out.exists() and not args.force:
        frame = pd.read_csv(out, index_col=0)
        assert frame.index.equals(adata.obs_names), (
            f"saved split {out} does not match the current dataset; inspect it and rerun "
            "with --force only if replacing the frozen split is intentional")
        print(f"[split] reusing frozen split {out} (pass --force to replace it)")
        return 0
    frame = make_splits(adata, args.seed, args.val_fraction, args.test_fraction)
    frame.to_csv(out)
    counts = frame["split"].value_counts()
    sides = frame.loc[frame["split"] == "train", "train_side"].value_counts()
    print(f"[split] {dict(counts)}  train sides: {dict(sides)}")
    print(f"[split] wrote {out}")
    return 0


def velocity_main(args: argparse.Namespace) -> int:
    adata = load_dataset(args.dataset)
    save_audit(adata)
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
                               args.min_forward, train_mask=train_source)
    assert result["velocity"].shape == adata.obsm["gene_activity"].shape, (
        f"velocity {result['velocity'].shape} does not match the ATAC input "
        f"{adata.obsm['gene_activity'].shape}")
    assert np.isfinite(result["velocity"]).all(), "velocity contains non-finite entries"
    if args.gauge_normalize:
        result["velocity"] = gauge_normalize(result["velocity"])
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
    result["fit_source_mask"] = train_source
    out = velocity_path(args.dataset, args.split_seed)
    np.savez_compressed(out, **result)
    print(f"[velocity] wrote {out}")
    return 0



CONDITIONS = REDUCED_CONDITIONS

# The RNA→protein runs' hyperparameters are the starting point, not a fresh set: a
# difference between the two experiments should come from the modality pair and the law,
# not from a learning rate somebody re-picked. Left column is this script's name, right
# column the key in training.yaml's `defaults`. A CLI flag overrides whatever is read.
TRAINING_YAML_KEYS = {
    "n_epochs": "n_epochs",
    "batch_size": "batch_size",
    "lambda_dyn": "lambda_dyn",
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


def apply_training_defaults(args: argparse.Namespace, config_path: Path) -> dict:
    """Fill every unset training flag from training.yaml's `defaults`, and say which.

    `blur` is `sinkhorn_reg` and `lr_rates` is `lr_beta` there — the chromatin model's
    rates are gamma and beta, where the RNA→protein model has only beta. Everything else
    keeps its name.
    """
    defaults = load_yaml(config_path).get("defaults", {})
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


def run_directory(args: argparse.Namespace) -> Path:
    """Timestamp AND settings, so a run dir says what it is without opening it."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{stamp}_{args.dataset}_{args.law}_{args.condition}_seed{args.seed}"
    if args.subsample is not None:
        name = f"{name}_sub{args.subsample}"
    path = Path(args.run_dir) if args.run_dir else CACHE_ROOT / "runs" / name
    path.mkdir(parents=True, exist_ok=True)
    (path / "preflight_passed.json").unlink(missing_ok=True)
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


def rna_target_layer(adata: sc.AnnData, requested: str, law: str = REDUCED) -> str:
    """Which RNA quantity phi is aligned against, and why that one.

    R1 predicts the mature gene-expression state, so `auto` uses the complete RNA matrix.
    The optional velocyto cache is not a safe R1 default: in BMMC it covers only a subset
    of cells and genes, which turns hundreds of valid mature-RNA targets into artificial
    all-zero columns. A caller can still request `--rna-target spliced` explicitly.

    The relay law is not allowed the total-counts fallback. It predicts [u, s] and fits
    ds = beta*u - gamma*s; if s were TOTAL RNA it would contain u, the relay would be
    fitting s against a quantity that includes its own source, and beta and gamma would
    stop being splicing and degradation rates. R2 trains and scores only on cells that
    carry splicing anyway, so spliced counts are always available to it.
    """
    if requested != "auto":
        layer = f"{requested}_lognorm"
        if law == RELAY and layer != "spliced_lognorm":
            raise ValueError("relay law requires spliced RNA as its s target")
        if layer not in adata.layers:
            raise ValueError(f"requested RNA target layer {layer!r} is absent")
        return layer
    if law == RELAY:
        if "spliced_lognorm" not in adata.layers:
            raise ValueError("relay law requires measured spliced RNA")
        return "spliced_lognorm"
    return "rna_lognorm"


def build_targets(adata: sc.AnnData, law: str, target_layer: str,
                  gene_mask: np.ndarray | None = None) -> np.ndarray:
    """phi's alignment target: mature RNA for R1, [u, s] for R2."""
    columns = slice(None) if gene_mask is None else gene_mask
    spliced = to_dense(adata.layers[target_layer], np.float32)[:, columns]
    if law == REDUCED:
        return spliced
    unspliced = to_dense(adata.layers["unspliced_lognorm"], np.float32)[:, columns]
    return np.concatenate([unspliced, spliced], axis=1)

def output_gene_mask(adata: sc.AnnData, law: str,
                     rows: np.ndarray | None = None) -> np.ndarray:
    """Genes the law has measured targets for; R2 never treats missing u/s as zero."""
    if law == REDUCED:
        return np.ones(adata.n_vars, dtype=bool)
    mask = usable_splicing_genes(adata, rows=rows)
    if not mask.any():
        raise ValueError("relay law has no gene with usable measured u/s counts")
    return mask


def target_cell_mask(adata: sc.AnnData, target_layer: str) -> np.ndarray:
    """Cells carrying the requested target, distinct from ATAC-only source eligibility."""
    if target_layer == "spliced_lognorm" and "has_splicing" in adata.obs:
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


def alignment_basis(target: torch.Tensor, scale: torch.Tensor, rows: torch.Tensor,
                    n_dims: int) -> torch.Tensor | None:
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
    observed = target[rows] / scale
    centred = observed - observed.mean(dim=0, keepdim=True)
    basis = torch.linalg.svd(centred, full_matrices=False)[2][:n_dims].T
    explained = float((centred @ basis).var(dim=0).sum() / centred.var(dim=0).sum())
    print(f"[train] alignment measured in {n_dims} target directions "
          f"({explained:.1%} of the RNA variance)")
    return basis


def sinkhorn_step(model, chromatin: torch.Tensor, target: torch.Tensor,
                  atac_rows: torch.Tensor, rna_rows: torch.Tensor, max_points: int,
                  scale: torch.Tensor, blur: float, backend: str,
                  basis: torch.Tensor | None) -> torch.Tensor:
    """Sinkhorn divergence between phi(ATAC cells) and the RNA cells' observed profiles.

    The two sides are drawn INDEPENDENTLY from disjoint cell sets, so no cell can appear
    on both sides and there is no pairing to leak. Dividing by `scale` is what gives the
    relay law's u and s blocks equal weight in the cost.
    """
    device = atac_rows.device
    source = subsample_rows(len(atac_rows), max_points, device, atac_rows)
    sink = subsample_rows(len(rna_rows), max_points, device, rna_rows)
    predicted = model.phi(chromatin[source]) / scale
    observed = target[sink] / scale
    if basis is not None:
        predicted, observed = predicted @ basis, observed @ basis
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


def phi_gradient_norm(model, loss: torch.Tensor) -> tuple[float, torch.Tensor]:
    """Norm of this loss's gradient on phi alone, and the flat gradient itself."""
    grads = torch.autograd.grad(loss, [p for p in model.phi.parameters()],
                                retain_graph=False, allow_unused=True)
    flat = torch.cat([g.reshape(-1) for g in grads if g is not None])
    return float(flat.norm()), flat


def grad_interaction(model, law: str, chromatin: torch.Tensor, target: torch.Tensor,
                     velocity: torch.Tensor, production: torch.Tensor,
                     confidence: torch.Tensor, probe: torch.Tensor, sink: torch.Tensor,
                     scales: tuple, blur: float, backend: str, weight: float,
                     basis: torch.Tensor | None, residual_mask: torch.Tensor) -> dict:
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
    predicted = model.phi(chromatin[probe]) / align_scale
    observed = target[sink] / align_scale
    if basis is not None:
        predicted, observed = predicted @ basis, observed @ basis
    align = sinkhorn_divergence(predicted, observed, blur=blur, backend=backend)
    align_norm, align_grad = phi_gradient_norm(model, align)
    dyn = kinetics_loss(model, law, chromatin[probe], velocity[probe], production[probe],
                        confidence[probe], dyn_scale, residual_mask)
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


def train_main(args: argparse.Namespace) -> int:
    adata = load_dataset(args.dataset)
    save_audit(adata, args.law)
    from_yaml = apply_training_defaults(args, Path(args.training_config))
    allowed = conditions_for_law(args.law)
    assert args.condition in allowed, (
        f"law {args.law} only runs {allowed}; got condition={args.condition!r}")
    # training.yaml carries lambda_dyn: 1000, tuned for a ~130-protein target. On this
    # 2000-gene one it drives phi to a near-constant: measured FOSCTTM 0.474 against a
    # constant-map floor of 0.25, prediction spread 0.003. Inheriting it by omitting the
    # flag is silent, so say so rather than let a whole sweep arm collapse unremarked.
    if args.lambda_dyn >= 10 and "lambda_dyn" in from_yaml:
        print(f"[train] WARNING: lambda_dyn={args.lambda_dyn} inherited from "
              f"{args.training_config}. On chromatin the measured optimum is 1 and >=10 "
              "collapses phi. Pass --lambda-dyn explicitly if this is deliberate.")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output_dir = run_directory(args)
    splits = pd.read_csv(split_path(args.dataset, args.split_seed), index_col=0)
    assert splits.index.equals(adata.obs_names), "split file does not match the dataset"
    fields = load_velocity(args.dataset, args.split_seed)

    target_layer = rna_target_layer(adata, args.rna_target, args.law)
    split_column = splits["split"].to_numpy()
    side_column = splits["train_side"].to_numpy()
    target_covered = target_cell_mask(adata, target_layer)
    if not target_covered.all():
        print(f"[train] target '{target_layer}': {int(target_covered.sum())}/{adata.n_obs} "
              "cells measured; absent targets are excluded, never zero-filled")
    gene_fit_rows = np.flatnonzero(
        target_covered & (split_column == "train") & (side_column == "rna"))
    gene_mask = output_gene_mask(adata, args.law, rows=gene_fit_rows)
    output_genes = adata.var_names[gene_mask]
    chromatin_np = np.asarray(adata.obsm["gene_activity"], dtype=np.float32)
    target_np = build_targets(adata, args.law, target_layer, gene_mask)
    full_mapping, full_alignment_mask, full_kinetic_mask = gene_map(
        adata, gene_map_path(args.dataset))
    true_mapping = full_mapping[gene_mask].tocsr()
    mapping = true_mapping
    alignment_mask = full_alignment_mask[gene_mask]
    kinetic_mask = full_kinetic_mask[gene_mask]

    train_source = (split_column == "train") & (side_column == "atac")
    train_target = (split_column == "train") & (side_column == "rna")
    assert len(set(adata.obs_names[train_source]) & set(adata.obs_names[train_target])) == 0
    shuffle_eligible = train_source & (np.asarray(fields["norm"]) > 0)
    velocity_np, confidence_np, dynamic_np_all = apply_velocity_condition(
        fields["velocity"], fields["confidence"], fields["dynamic_mask"],
        args.condition, args.seed, eligible=shuffle_eligible)
    if args.condition == "permG":
        mapping = permute_chromatin_projection(mapping, args.seed)

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
    print(f"[train] {args.dataset} {args.law}/{args.condition} seed {args.seed} -> {output_dir}")
    print(f"[train] rows: atac {len(atac_rows)}, rna {len(rna_rows)}, val {len(val_rows)}, "
          f"kinetic {len(dynamic_rows)}; target '{target_layer}' ({target.shape[1]} columns); "
          f"genes: {int(alignment_mask.sum())} aligned, {int(kinetic_mask.sum())} in G")

    # Fitted on the RNA TRAINING half only. §4 puts the split before everything, and a
    # per-gene scale taken over all cells is a statistic of the test set leaking into the
    # loss — small, but it is exactly the kind of thing the split rule exists to prevent.
    # Two different scales on purpose: the kinetics residual is divided per gene so no
    # gene dominates the ODE, the alignment cost is divided by that AND by the cloud's
    # diameter so `blur` is a fraction of the distance between cells.
    fit_target = target[rna_rows]
    target_scale = block_scales(fit_target, args.law)
    phi_scale, phi_bias = gene_affine_calibration(
        production, target, atac_rows, rna_rows, args.law)
    # permG corrupts G_train in the kinetic RHS and nothing else (§12). phi always sees
    # the same named gene-activity coordinates; feeding the permuted G into its shortcut
    # would turn one ablation into two simultaneous corruptions.
    phi_projection = torch.as_tensor(
        true_mapping.toarray().astype(np.float32), device=device)
    if args.law == RELAY:
        phi_projection = torch.cat([phi_projection, phi_projection], dim=0)
    align_scale = alignment_scale(fit_target, args.law)
    # The residual is only defined where chromatin can drive the gene; a zero G row would
    # otherwise contribute a pure -gamma*phi decay term with no production to balance it.
    residual_mask = torch.as_tensor(law_mask(kinetic_mask, args.law), device=device)

    basis = alignment_basis(target, align_scale, rna_rows, args.align_dims)
    model = ChromatinKOT(
        len(output_genes), args.law, n_input_features=adata.n_vars,
        phi_dims=args.phi_dims, kappa_dims=args.kappa_dims,
        g_dims=args.g_dims, phi_init_gain=args.phi_init_gain,
        phi_spectral_norm=args.phi_spectral_norm, activation=args.activation,
        init_method=args.init_method, kappa_min=args.kappa_min, kappa_max=args.kappa_max,
        phi_projection=phi_projection, phi_scale=phi_scale, phi_bias=phi_bias,
        phi_residual_weight=args.phi_residual_weight,
        alpha_min=args.alpha_min, alpha_max=args.alpha_max,
    ).to(device)
    optimiser = torch.optim.Adam(chromatin_param_groups(
        model, args.lr, args.lr_phi, args.lr_alpha_kappa, args.lr_rates))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimiser, lr_lambda=lambda epoch: lr_schedule_factor(
            epoch, args.lr_warmup_epochs, args.n_epochs, args.lr_warmup_start_factor,
            args.lr_min_factor))
    lambda_dyn = 0.0 if args.condition == "noDyn" else args.lambda_dyn

    model.eval()
    with torch.no_grad():
        initial_held = sinkhorn_step(
            model, chromatin, target, val_rows, val_rows, args.sinkhorn_max_points,
            align_scale, args.sinkhorn_blur, args.sinkhorn_backend, basis)
    history = []
    monitor_ready_at = 0 if lambda_dyn == 0.0 else args.dyn_warmup_epochs
    if monitor_ready_at > args.n_epochs:
        raise ValueError(
            f"dyn_warmup_epochs ({args.dyn_warmup_epochs}) exceeds n_epochs "
            f"({args.n_epochs}); no dynamics-mature checkpoint can be selected")
    best = {
        "val_align": float(initial_held) if monitor_ready_at == 0 else float("inf"),
        "state": ({k: v.detach().clone() for k, v in model.state_dict().items()}
                  if monitor_ready_at == 0 else None),
        "epoch": 0 if monitor_ready_at == 0 else None,
    }
    print(f"[train] epoch    0  val_align {float(initial_held):.5f} (gene-aware initial map)")
    if monitor_ready_at:
        print(f"[train] best-align checkpoint selection starts at epoch "
              f"{monitor_ready_at}, when lambda_dyn has reached full strength")
    for epoch in range(1, args.n_epochs + 1):
        model.train()
        optimiser.zero_grad(set_to_none=True)
        align = sinkhorn_step(model, chromatin, target, atac_rows, rna_rows,
                              args.sinkhorn_max_points, align_scale, args.sinkhorn_blur,
                              args.sinkhorn_backend, basis)
        align.backward()

        weight = lambda_dyn * dyn_weight_factor(epoch, args.dyn_warmup_epochs)
        dyn_value = 0.0
        if weight > 0.0 and len(dynamic_rows) > 0:
            for batch in shuffled_batches(len(dynamic_rows), args.batch_size, device,
                                          dynamic_rows):
                share = len(batch) / len(dynamic_rows)
                loss = kinetics_loss(model, args.law, chromatin[batch], velocity[batch],
                                     production[batch], confidence[batch], target_scale,
                                     residual_mask)
                ((weight * share) * loss).backward()
                dyn_value += share * float(loss)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimiser.step()
        scheduler.step()

        if epoch % args.eval_every == 0 or epoch == args.n_epochs:
            model.eval()
            with torch.no_grad():
                held = sinkhorn_step(model, chromatin, target, val_rows, val_rows,
                                     args.sinkhorn_max_points, align_scale,
                                     args.sinkhorn_blur, args.sinkhorn_backend, basis)
            row = {"epoch": epoch, "loss_align": float(align), "loss_dyn": dyn_value,
                   "val_align": float(held), "lambda_dyn_effective": weight}
            if len(dynamic_rows) > 0:
                row.update(grad_interaction(
                    model, args.law, chromatin, target, velocity, production, confidence,
                    subsample_rows(len(dynamic_rows), args.batch_size, device, dynamic_rows),
                    subsample_rows(len(rna_rows), args.batch_size, device, rna_rows),
                    (align_scale, target_scale), args.sinkhorn_blur, args.sinkhorn_backend,
                    weight, basis, residual_mask))
            history.append(row)
            print(f"[train] epoch {epoch:4d}  align {float(align):.5f}  "
                  f"dyn {dyn_value:.5f}  val_align {float(held):.5f}  "
                  f"grad_ratio {row.get('grad_mag_ratio', float('nan')):.3g}  "
                  f"grad_cos {row.get('grad_cosine', float('nan')):+.3f}")
            if epoch >= monitor_ready_at and float(held) < best["val_align"]:
                best = {"val_align": float(held),
                        "state": {k: v.detach().clone() for k, v in model.state_dict().items()},
                        "epoch": epoch}

    frame = pd.DataFrame(history)
    frame.to_csv(output_dir / "training_loss.csv", index=False)
    settings = vars(args) | {"target_layer": target_layer,
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
                    "condition": args.condition, "seed": args.seed,
                    "target_layer": target_layer, "dataset": args.dataset,
                    "gene_names": list(output_genes),
                    "input_gene_names": list(adata.var_names)},
                   output_dir / f"checkpoint_{name}.pt")
    model.load_state_dict(best["state"])
    print(f"[train] best val_align {best['val_align']:.5f} at epoch {best['epoch']}")
    # The JVP checks run on validation cells that HAVE a velocity. On HSPC only day-0
    # cells do (§7), and they are 40% of the data — so a median over all validation rows
    # is the median of a mostly-zero vector and reports 0.0 whatever the model does.
    velocity_val = np.flatnonzero(
        np.isin(np.arange(adata.n_obs), val_rows.cpu().numpy())
        & fields["dynamic_mask"] & target_covered)
    checks = preflight_checks(model, args.law, chromatin, target, production,
                              torch.as_tensor(fields["velocity"].astype(np.float32),
                                              device=device),
                              val_rows, torch.as_tensor(velocity_val, device=device),
                              target_scale, adata, args.reference_layer,
                              reference_columns=np.flatnonzero(gene_mask))
    (output_dir / "preflight.json").write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))
    passed, failures = preflight_verdict(checks, args.condition)
    if not passed:
        print("[train] PREFLIGHT FAILED:")
        for line in failures:
            print(f"  - {line}")
        if args.condition in ("full", "noDyn"):
            print("[train] refusing to treat this as a launchable checkpoint "
                  "(shuffle/reverse/zero/permG wait until full/noDyn pass)")
            return 1
        print("[train] corruption arms are allowed to fail the biological gate; "
              "the checkpoint is still written")
    else:
        (output_dir / "preflight_passed.json").write_text(json.dumps(checks, indent=2))
    return 0


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
    floor = checks.get("foscttm_constant_floor")
    if floor is None or not np.isfinite(floor):
        failures.append("constant-map FOSCTTM floor is missing")
    elif not (np.isfinite(foscttm) and foscttm < floor):
        failures.append(
            f"FOSCTTM {foscttm:.3f} is not below the constant-map floor {floor:.3f}")
    cosine = checks.get("jvp_vs_reference_cosine_median")
    if cosine is None:
        failures.append("JVP vs RNA velocity is missing — run `reference` first")
    elif not (np.isfinite(cosine) and cosine > 0):
        failures.append(f"JVP vs RNA velocity cosine {cosine:.4g} is not positive")
    centred = checks.get("jvp_vs_reference_cosine_centred_median")
    if centred is None:
        failures.append("centred JVP vs RNA velocity cosine is missing")
    elif not (np.isfinite(centred) and centred > 0):
        failures.append(
            f"centred JVP vs RNA velocity cosine {centred:.4g} is not positive")
    return len(failures) == 0, failures


def preflight_checks(model, law: str, chromatin: torch.Tensor, target: torch.Tensor,
                     production: torch.Tensor, true_velocity: torch.Tensor,
                     rows: torch.Tensor, velocity_rows: torch.Tensor, scale: torch.Tensor,
                     adata: sc.AnnData, reference_layer: str,
                     reference_columns: np.ndarray | None = None,
                     foscttm_cells: int = 2000) -> dict:
    """The gate the run order puts before launching anything else.

    Deliberately reads the TRUE velocity even when the run was trained on a corrupted one
    (§13): a checkpoint scored only against its own corrupted law can look perfectly
    self-consistent while having learned nothing about chromatin.
    """
    model.eval()
    residual, moving = kinetics_residual(model, law, chromatin[velocity_rows],
                                         true_velocity[velocity_rows],
                                         production[velocity_rows])
    pushforward = residual + law_rhs(model, law, moving, chromatin[velocity_rows],
                                     production[velocity_rows])
    with torch.no_grad():
        prediction = model.phi(chromatin[rows])
    observed = target[rows]
    state_prediction = spliced_block(prediction, law)
    state_observed = spliced_block(observed, law)
    per_gene = torch.stack([
        torch.corrcoef(torch.stack([state_prediction[:, col], state_observed[:, col]]))[0, 1]
        for col in range(0, state_prediction.shape[1],
                         max(1, state_prediction.shape[1] // 200))
    ])
    # A phi that ignores its input still scores a respectable Sinkhorn loss by sitting on
    # the target's mean. Then every per-gene correlation is ~0 and the spread across cells
    # is ~0, and this ratio is what tells the two apart from the loss curve alone.
    spread = float((state_prediction.std(dim=0).median()
                    / state_observed.std(dim=0).median()).clamp(max=1e6))
    # "FOSCTTM is sensible" is on the gate list, so it is measured here and not left to
    # the evaluate stage: a run whose alignment never happened should be caught before the
    # other 17 conditions are launched.
    sample = torch.randperm(len(rows))[:foscttm_cells]
    sampled_prediction = state_prediction[sample].detach().cpu().numpy()
    sampled_observed = state_observed[sample].cpu().numpy()
    constant_prediction = np.broadcast_to(
        sampled_observed.mean(axis=0, keepdims=True), sampled_prediction.shape).copy()
    checks = {
        "foscttm": float(np.mean(calc_domainAveraged_FOSCTTM(
            sampled_prediction, sampled_observed))),
        "foscttm_constant_floor": float(np.mean(calc_domainAveraged_FOSCTTM(
            constant_prediction, sampled_observed))),
        "foscttm_n_cells": int(len(sample)),
        "state_pearson_median": float(torch.nanmedian(per_gene)),
        "state_pearson_positive_fraction": float((per_gene > 0).float().mean()),
        "prediction_spread_ratio": spread,
        "state_prediction_finite": bool(torch.isfinite(prediction).all()),
        "jvp_norm_median": float(pushforward.norm(dim=1).median()),
        "jvp_finite": bool(torch.isfinite(pushforward).all()),
        "residual_norm_median": float((residual / scale).norm(dim=1).median()),
        "jvp_rhs_cosine_median": float(torch.nn.functional.cosine_similarity(
            pushforward, pushforward - residual, dim=1).median()),
        "jvp_n_cells": int(len(velocity_rows)),
    }
    # The gate's "JVP vs RNA velocity is positive". Absent before `reference` has run,
    # which is why it is reported as null rather than silently skipped.
    reference = reference_velocity(adata, reference_layer)
    checks["jvp_vs_reference_cosine_median"] = None
    if reference is not None:
        selected = reference[velocity_rows.cpu().numpy()]
        if reference_columns is not None:
            selected = selected[:, reference_columns]
        scored = np.flatnonzero(np.abs(selected).sum(axis=1) > 0)
        # Genes as well as cells, matching Task D. scVelo fits a subset of the panel and
        # the rest is zero-filled; leaving those columns in dilutes the cosine by roughly
        # the square root of the covered fraction, on a gate item that reads
        # "JVP vs RNA velocity is positive".
        genes = np.flatnonzero(np.abs(selected).sum(axis=0) > 0)
        if len(scored) > 0 and len(genes) > 0:
            pushed = spliced_block(pushforward.detach().cpu().numpy(), law)
            reference_metrics = task_d_kinetics(
                pushed[np.ix_(scored, genes)], selected[np.ix_(scored, genes)])
            checks["jvp_vs_reference_cosine_median"] = reference_metrics[
                "cell_cosine_median"]
            checks["jvp_vs_reference_cosine_centred_median"] = reference_metrics[
                "cell_cosine_centred_median"]
            checks["jvp_vs_reference_gene_pearson_median"] = reference_metrics[
                "gene_pearson_median"]
            checks["jvp_vs_reference_n_cells"] = int(len(scored))
            checks["jvp_vs_reference_n_genes"] = int(len(genes))
    return checks


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
    # Every architecture setting, not just the layer widths. phi_spectral_norm in
    # particular changes the parameter NAMES (weight -> weight_orig/_u/_v), so a model
    # rebuilt on the defaults cannot load a checkpoint trained with it at all.
    input_names = payload.get("input_gene_names", payload["gene_names"])
    phi_state = payload["state_dict"]
    phi_kwargs = {}
    if "phi.gene_projection" in phi_state:
        phi_kwargs = {
            "phi_projection": phi_state["phi.gene_projection"],
            "phi_scale": phi_state["phi.gene_scale"],
            "phi_bias": phi_state["phi.gene_bias"],
            "phi_residual_weight": config.get("phi_residual_weight", 0.1),
        }
    model = ChromatinKOT(
        len(payload["gene_names"]), payload["law"], n_input_features=len(input_names),
        phi_dims=config["phi_dims"], kappa_dims=config["kappa_dims"],
        g_dims=config["g_dims"], phi_init_gain=config["phi_init_gain"],
        phi_spectral_norm=config["phi_spectral_norm"], activation=config["activation"],
        init_method=config["init_method"], kappa_min=config["kappa_min"],
        kappa_max=config["kappa_max"], alpha_min=config["alpha_min"],
        alpha_max=config["alpha_max"],
        **phi_kwargs,
    ).to(device)
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
    adata = load_dataset(args.dataset)
    save_audit(adata, RELAY)
    rows = np.flatnonzero(adata.obs["has_splicing"].to_numpy()) if "has_splicing" in adata.obs \
        else np.arange(adata.n_obs)
    usable = usable_splicing_genes(adata)
    print(f"[reference] scVelo on {len(rows)} cells x {int(usable.sum())} genes "
          f"with usable spliced/unspliced counts")

    subset = adata[rows][:, usable].copy()
    if args.backend == "regvelo":
        # Second reference (§17). Same spliced/unspliced input, a GRN-informed velocity
        # model — so a Task D agreement that only holds for one backend is visible.
        subset, _ = run_regvelo(subset, n_top_genes=int(usable.sum()), hvg_flavor="seurat_v3",
                                min_shared_counts=args.min_shared_counts, n_pcs=args.n_pcs,
                                n_neighbors=args.n_neighbors, grn_prior_csv=args.grn_prior_csv)
    else:
        # The same HVG → filter/normalise → PCA → neighbours → moments the RNA→protein
        # velocities were built with, so the two experiments' references are comparable and
        # this stage cannot drift from that pipeline. n_top_genes is the whole subset: the
        # gene choice was already made by the gene map.
        subset, _ = preprocess_for_velocity(
            subset, n_top_genes=subset.n_vars, hvg_flavor="seurat_v3",
            min_shared_counts=args.min_shared_counts, n_pcs=args.n_pcs,
            n_neighbors=args.n_neighbors)
        scv.tl.recover_dynamics(subset, n_jobs=args.n_jobs)
        scv.tl.velocity(subset, mode="dynamical")

    columns = adata.var_names.get_indexer(subset.var_names)
    # Both blocks: the relay law predicts [u, s], so §17's joint u/s comparison needs the
    # reference's unspliced velocity as well as its spliced one.
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
    adata.write_h5ad(dataset_path(args.dataset))
    return 0


def pushforward_and_rhs(model, law: str, chromatin: torch.Tensor, velocity: torch.Tensor,
                        production: torch.Tensor, batch_size: int
                        ) -> tuple[np.ndarray, np.ndarray]:
    """J_phi(c) v_c and the kinetic-law RHS, in batches. Needs grad for forward-mode AD."""
    push, rhs_blocks = [], []
    for start in range(0, len(chromatin), batch_size):
        span = slice(start, start + batch_size)
        residual, prediction = kinetics_residual(model, law, chromatin[span],
                                                 velocity[span], production[span])
        rhs = law_rhs(model, law, prediction, chromatin[span], production[span])
        push.append((residual + rhs).detach().cpu().numpy())
        rhs_blocks.append(rhs.detach().cpu().numpy())
    return np.concatenate(push), np.concatenate(rhs_blocks)


def production_rows(chromatin_np: np.ndarray, mapping, rows: np.ndarray,
                    device: torch.device) -> torch.Tensor:
    """G c for a subset of cells. G is (genes x activity), so the product is C @ G.T."""
    return torch.as_tensor(
        (chromatin_np[rows] @ mapping.toarray().T).astype(np.float32), device=device)


def evaluate_main(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    device = choose_torch_device({"device": args.device})
    model, payload, config = load_checkpoint(run_dir, args.checkpoint, device)
    dataset, law = payload["dataset"], payload["law"]
    adata = load_dataset(dataset)
    save_audit(adata, law)
    splits = pd.read_csv(split_path(dataset, config["split_seed"]), index_col=0)
    # `reference` rewrites the dataset h5ad in place, so a split built against an older
    # build would misalign every row without raising.
    assert splits.index.equals(adata.obs_names), (
        "the split file does not match this dataset — rebuild the split")
    fields = load_velocity(dataset, config["split_seed"])

    chromatin_np = np.asarray(adata.obsm["gene_activity"], dtype=np.float32)
    chromatin = torch.as_tensor(chromatin_np, device=device)
    input_names = list(payload.get("input_gene_names", payload["gene_names"]))
    assert input_names == list(adata.var_names), "checkpoint ATAC features do not match dataset"
    output_columns = adata.var_names.get_indexer(payload["gene_names"])
    assert (output_columns >= 0).all(), "checkpoint RNA genes do not match dataset"
    full_map, _, _ = gene_map(adata, gene_map_path(dataset))
    true_map = full_map[output_columns].tocsr()
    train_map = permute_chromatin_projection(true_map, config["seed"]) \
        if config["condition"] == "permG" else true_map
    true_velocity_np = fields["velocity"]
    split_column = splits["split"].to_numpy()
    side_column = splits["train_side"].to_numpy()
    train_source = (split_column == "train") & (side_column == "atac")
    shuffle_eligible = train_source & (np.asarray(fields["norm"]) > 0)
    train_velocity_np, _, train_dynamic = apply_velocity_condition(
        fields["velocity"], fields["confidence"], fields["dynamic_mask"],
        config["condition"], config["seed"], eligible=shuffle_eligible)

    target_covered = target_cell_mask(adata, config["target_layer"])
    test_rows = np.flatnonzero((split_column == "test") & target_covered)
    if not target_covered.all():
        print(f"[evaluate] scoring {len(test_rows)} test cells with measured "
              f"{config['target_layer']}; absent targets are not treated as zero")
    observed = to_dense(adata.layers[config["target_layer"]], np.float32)[
        np.ix_(test_rows, output_columns)]
    with torch.no_grad():
        predicted_full = model.phi(chromatin[test_rows]).cpu().numpy()
    predicted = spliced_block(predicted_full, law)

    results = {"run_dir": str(run_dir), "checkpoint": args.checkpoint, "dataset": dataset,
               "law": law, "condition": config["condition"], "seed": config["seed"],
               "n_test_cells": len(test_rows)}
    gene_names = list(payload["gene_names"])
    summary, per_gene = task_a_state(predicted, observed, gene_names)
    per_gene.to_csv(run_dir / f"task_a_per_gene_{args.checkpoint}.csv", index=False)
    results["task_a"] = summary
    groups = evaluation_groups(adata, fields, test_rows)
    if groups is not None:
        results["task_a_by_group"] = by_group(
            lambda a, b: task_a_state(a, b, gene_names)[0], predicted, observed,
            groups)

    labels = adata.obs["cell_type"].astype(str).to_numpy()[test_rows] \
        if "cell_type" in adata.obs else None
    results["task_b"] = task_b_pairing(predicted, observed, labels,
                                       args.max_sinkhorn_cells, config["seed"])
    group_column = "day" if dataset == "hspc" else "DonorID"
    results["task_b_within_group"] = foscttm_within(
        predicted, observed, adata.obs[group_column].to_numpy()[test_rows])

    if labels is None:
        print("[evaluate] no cell_type annotation — Task C and the cell-type "
              "retrieval metrics are not defined for this dataset")
    elif args.tasks_c:
        rows = subsample_rows_for_graph(len(test_rows), args.max_graph_cells, config["seed"])
        results["task_c"] = task_c_integration(
            predicted[rows], observed[rows], labels[rows],
            lambda joint: leiden_clusters(joint, args.n_neighbors), args.n_neighbors)

    internal_rows = np.flatnonzero(
        (split_column == "train") & (side_column == "atac") & train_dynamic)

    results["task_d"] = {}
    # INTERNAL: the cells that entered L_dyn, the velocity they were trained with, the G
    # they were trained with. This asks whether optimisation satisfied its own law.
    results["task_d"]["internal_n_cells"] = int(len(internal_rows))
    if len(internal_rows) > 0:
        jvp_train, rhs_train = pushforward_and_rhs(
            model, law, chromatin[internal_rows],
            torch.as_tensor(train_velocity_np[internal_rows].astype(np.float32), device=device),
            production_rows(chromatin_np, train_map, internal_rows, device),
            args.batch_size)
        results["task_d"]["internal_pushforward_norm_median"] = float(
            np.median(np.linalg.norm(jvp_train, axis=1)))
        results["task_d"]["internal_vs_law"] = task_d_kinetics(jvp_train, rhs_train)

    # BIOLOGICAL: held-out cells, TRUE velocity, TRUE G. RNA velocity is a later
    # comparison, not a substitute for the law residual.
    moving = np.flatnonzero(np.abs(true_velocity_np[test_rows]).sum(axis=1) > 0)
    results["task_d"]["biological_n_moving_cells"] = int(len(moving))
    if len(moving) > 0:
        rows_d = test_rows[moving]
        jvp_true, rhs_true = pushforward_and_rhs(
            model, law, chromatin[rows_d],
            torch.as_tensor(true_velocity_np[rows_d].astype(np.float32), device=device),
            production_rows(chromatin_np, true_map, rows_d, device),
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
            pushed = spliced_block(jvp_true, law)[np.ix_(scored, genes)]
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
                predicted_u = jvp_true[:, :jvp_true.shape[1] // 2]
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
                        spliced_block(jvp_true, law)[np.ix_(common_cells, common_genes)],
                    ], axis=1)
                    reference_joint = np.concatenate([
                        u_block[np.ix_(common_cells, common_genes)],
                        block[np.ix_(common_cells, common_genes)],
                    ], axis=1)
                    results["task_d"][f"biological_vs_{name}_joint_us"] = task_d_kinetics(

                        predicted_joint, reference_joint)
    out = run_dir / f"evaluation_{args.checkpoint}.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {out}")
    return 0


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
    adata = load_dataset(args.dataset)
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
    # Split by whether chromatin can reach the gene at all. The panel deliberately keeps
    # genes with no chromatin support, and those can only score ~0 — pooling them into one
    # median understates the ceiling for the genes the law actually covers.
    _, _, kinetic_mask = gene_map(adata, gene_map_path(args.dataset))
    subsets = {"all": np.ones(adata.n_vars, dtype=bool), "in_G": kinetic_mask,
               "not_in_G": ~kinetic_mask}
    results = {}
    for name, predicted in predictions.items():
        summary, per_gene = task_a_state(predicted, observed[test], list(adata.var_names))
        per_gene["in_G"] = kinetic_mask
        per_gene.to_csv(CACHE_ROOT / f"{args.dataset}_baseline_{name}_per_gene.csv", index=False)
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
    out = CACHE_ROOT / f"{args.dataset}_baselines.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"[baseline] wrote {out}")
    return 0


def baseline_model_main(args: argparse.Namespace) -> int:
    """Run one competing method on this experiment's split and score it like KOT.

    The baselines answer Tasks A-C. Task D needs a differentiable ATAC->RNA map and only
    scDART has one, so it is left to a separate pass rather than faked for the rest.
    """
    adata = load_dataset(args.dataset)
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

    # Subsampled methods scored only the rows they saw, and BOTH modalities use that one
    # sample so each test cell is compared against its own partner.
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

    # Tasks B and C in the method's OWN space: a latent method aligns there, and scoring
    # it after kNN imputation into gene space would measure the imputation, not the method.
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
    prepare.set_defaults(func=prepare_main)

    audit = sub.add_parser("audit", help="§3 dataset audit")
    audit.add_argument("--dataset", choices=DATASETS, required=True)
    audit.add_argument("--law", choices=["reduced", "relay"], default="reduced")
    audit.set_defaults(func=audit_main)

    split = sub.add_parser("split", help="§4 fixed 70/10/20 split with disjoint train sides")
    split.add_argument("--dataset", choices=DATASETS, required=True)
    split.add_argument("--seed", type=int, default=0)
    split.add_argument("--val-fraction", type=float, default=0.1)
    split.add_argument("--test-fraction", type=float, default=0.2)
    split.set_defaults(func=split_main)

    split.add_argument("--force", action="store_true",
                       help="replace an existing frozen split (off by default)")
    velocity = sub.add_parser("velocity", help="§7/§8 chromatin velocity, from ATAC alone")
    velocity.add_argument("--dataset", choices=DATASETS, required=True)
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
    train.add_argument("--law", choices=LAWS, default=REDUCED)
    train.add_argument("--condition", choices=CONDITIONS, default="full",
                       help="R2 (relay) only allows full/shuffle/noDyn")
    train.add_argument("--seed", type=int, default=None,
                       help=f"one of the RNA→protein seeds {rna_protein_seeds()[:3]} ...; "
                            "unset takes training.yaml's own `seed`")
    train.add_argument("--split-seed", type=int, default=0)
    train.add_argument("--run-dir", default=None)
    train.add_argument("--subsample", type=int, default=None,
                       help="cells to keep — the small-subset check before a large run")
    train.add_argument("--rna-target", choices=["auto", "spliced", "rna"], default="auto")
    train.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    train.add_argument("--training-config", default="config/training.yaml",
                       help="the RNA→protein hyperparameters every unset flag is taken from")
    train.add_argument("--n-epochs", type=int, default=None)
    train.add_argument("--batch-size", type=int, default=None)
    train.add_argument("--eval-every", type=int, default=None)
    train.add_argument("--lambda-dyn", type=float, default=None)
    train.add_argument("--dyn-warmup-epochs", type=int, default=None)
    train.add_argument("--lr", type=float, default=None)
    train.add_argument("--lr-phi", type=float, default=None)
    train.add_argument("--lr-alpha-kappa", type=float, default=None)
    train.add_argument("--lr-rates", type=float, default=None)
    train.add_argument("--lr-warmup-epochs", type=int, default=None)
    train.add_argument("--lr-warmup-start-factor", type=float, default=None)
    train.add_argument("--lr-min-factor", type=float, default=None)
    train.add_argument("--grad-clip", type=float, default=5.0)
    train.add_argument("--sinkhorn-blur", type=float, default=None)
    train.add_argument("--sinkhorn-backend", default=None,
                       help="KeOps has no CUDA in this container; tensorized stays on GPU")
    train.add_argument("--sinkhorn-max-points", type=int, default=2048)
    train.add_argument("--align-dims", type=int, default=32,
                       help="measure the Sinkhorn cost in this many target directions "
                            "(0 = the full gene space)")
    train.add_argument("--phi-dims", type=int, nargs="+", default=None)
    train.add_argument("--phi-init-gain", type=float, default=None)
    train.add_argument("--phi-residual-weight", type=float, default=1.0,
                       help="maximum nonlinear residual weight around the trainable, "
                            "marginally calibrated G initialisation")
    train.add_argument("--phi-spectral-norm", type=lambda v: v.lower() == "true", default=None)
    train.add_argument("--activation", default=None)
    train.add_argument("--init-method", default=None)
    train.add_argument("--kappa-min", type=float, default=None)
    train.add_argument("--kappa-max", type=float, default=None)
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
    reference.set_defaults(func=reference_main)

    baseline = sub.add_parser("baseline", help="§14 simple and paired ridge/MLP references")
    baseline.add_argument("--dataset", choices=DATASETS, required=True)
    baseline.add_argument("--split-seed", type=int, default=0)
    baseline.add_argument("--rna-target", choices=["auto", "spliced", "rna"], default="auto")
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
