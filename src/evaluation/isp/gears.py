"""GEARS in-silico perturbation, predicted into frozen NT-only KOT coordinates.

GEARS is the graph-based perturbation predictor item 31 asks for. It consumes a control
cell and a knockout identity and returns the predicted perturbed expression profile, so it
plugs into Task B directly. This builds a GEARS PertData from the same NT-only Papalexi
matrix the checkpoint was fitted on, trains GEARS, and writes predicted KO RNA in
frozen-KOT feature order.

Run it with the isolated interpreter, which reuses the container's torch and numpy:
    cache/.gears_venv/bin/python run_crispr_isp.py gears ...

SUPERVISION: GEARS trains on observed KO cells for the perturbations it is fitted on, so
like MultiPert it receives more than KOT does. The held-out split below keeps a
perturbation's own cells out of its own prediction; state that regime with any number.
"""
from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
from gears import GEARS, PertData
from scipy import sparse

from src.evaluation.isp.common import profile_row, usable_perturbations, write_isp_outputs
from src.utils.arrays import to_dense

CONTROL_LABEL = "NT"


def add_arguments(parser) -> None:
    parser.add_argument("--rna", type=Path,
                        default=Path("cache/velocity/papalexi_nt_only/"
                                     "papalexi_scvelo_results_nt_only.h5ad"))
    parser.add_argument("--layer", default="Ms",
                        help="layer the frozen checkpoint consumes")
    parser.add_argument("--data-dir", type=Path, default=Path("cache/gears"))
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="frozen KOT run, accepted so one sbatch contract covers "
                             "every ISP; GEARS fits in the checkpoint's own coordinates "
                             "and does not read it")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--min-ko-cells", type=int, default=20)
    parser.add_argument("--min-replicates", type=int, default=2)
    parser.add_argument("--device", default="cuda")


def gears_frame(rna: ad.AnnData, usable: set[str]) -> ad.AnnData:
    """Recast the checkpoint's own RNA matrix into the layout GEARS expects.

    GEARS keys perturbations off obs['condition'] written as 'GENE+ctrl', reads gene
    symbols from var['gene_name'], and needs a cell_type column even for a single line.
    Nothing is re-normalised here: the matrix is the one the frozen checkpoint consumes,
    so a predicted profile lands in KOT coordinates without a second conversion.
    """
    genes = rna.obs["gene"].astype(str).to_numpy()
    keep = np.isin(genes, list(usable)) | (genes == CONTROL_LABEL)
    subset = rna[keep].copy()
    labels = subset.obs["gene"].astype(str)
    subset.obs["condition"] = np.where(labels == CONTROL_LABEL, "ctrl", labels + "+ctrl")
    subset.obs["cell_type"] = "THP1"
    subset.var["gene_name"] = subset.var_names.astype(str)
    return subset


def run(args) -> None:
    rna = ad.read_h5ad(args.rna)
    if args.layer not in rna.layers:
        raise ValueError(f"{args.rna} has no layer '{args.layer}'")
    # GEARS calls adata.X.toarray() unconditionally in get_dropout_non_zero_genes, so a
    # dense matrix crashes it. Ms is a smoothed moment and barely sparse, but the CSR
    # wrapper is what upstream expects and costs nothing correctness-wise.
    rna.X = sparse.csr_matrix(to_dense(rna.layers[args.layer], dtype=np.float32))
    groups = rna.obs["gene"].astype(str).to_numpy()
    replicates = rna.obs["replicate"].astype(str).to_numpy()
    usable = set(usable_perturbations(groups, replicates, args.min_ko_cells,
                                      args.min_replicates, CONTROL_LABEL))
    frame = gears_frame(rna, usable)

    args.data_dir.mkdir(parents=True, exist_ok=True)
    pert_data = PertData(str(args.data_dir))
    pert_data.new_data_process(dataset_name="papalexi_nt_only", adata=frame)
    pert_data.prepare_split(split="simulation", seed=args.seed)
    pert_data.get_dataloader(batch_size=args.batch_size, test_batch_size=args.batch_size)

    model = GEARS(pert_data, device=args.device)
    model.model_initialize(hidden_size=64)
    model.train(epochs=args.epochs)
    model.save_model(str(args.out_dir / "model"))

    # GEARS returns the predicted perturbed profile per perturbation, already in the
    # matrix's own coordinates, so no unit conversion is needed on the way out.
    # GEARS can only score a knockout that is a node in its GO graph; the rest raise
    # rather than returning nothing, so they are filtered out and recorded as uncovered,
    # the same way RegVelo reports knockouts with no learned outgoing edges.
    observed = sorted(usable & set(pert_data.adata.obs["condition"]
                                   .astype(str).str.replace(r"\+ctrl$", "", regex=True)))
    in_graph = set(map(str, model.pert_list))
    targets = [target for target in observed if target in in_graph]
    coverage = [{"perturbation": target, "supported": False,
                 "reason": "absent from the GEARS perturbation graph"}
                for target in observed if target not in in_graph]
    if not targets:
        raise ValueError("No sufficiently sampled knockout is in the GEARS graph")
    predictions = model.predict([[target] for target in targets])
    control = np.asarray(rna[groups == CONTROL_LABEL].X.mean(axis=0)).ravel()
    profiles = []
    for target in targets:
        if target not in predictions:
            coverage.append({"perturbation": target, "supported": False,
                             "reason": "GEARS returned no prediction"})
            continue
        profile = np.asarray(predictions[target], dtype=np.float64)
        if profile.shape[0] != rna.n_vars:
            raise ValueError(
                f"GEARS returned {profile.shape[0]} features for {target}, "
                f"but the checkpoint consumes {rna.n_vars}")
        profiles.append(profile_row(target, profile))
        coverage.append({"perturbation": target, "supported": True, "reason": "ok",
                         "delta_norm": float(np.linalg.norm(profile - control))})

    manifest = {
        "model": "GEARS graph perturbation prediction", "seed": args.seed,
        "epochs": args.epochs, "layer": args.layer,
        "input_coordinates": "frozen KOT Ms (GEARS fitted directly in them)",
        "supervision": "trains on observed KO cells; simulation split",
        "n_features": int(rna.n_vars),
        "supported_perturbations": len(profiles),
        "requested_perturbations": len(observed),
    }
    write_isp_outputs(args.out_dir, profiles, coverage, manifest, "GEARS")
    print(f"[gears-isp] {len(profiles)}/{len(observed)} perturbations in the GO graph")
