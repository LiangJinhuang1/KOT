#!/usr/bin/env python3
"""MultiPert on this project's POOLED Papalexi, in MultiPert's own supervision regime.

MultiPert (Zhao et al., PLOS Comput Biol 21, e1014054, 2025) predicts a perturbed
RNA+protein profile from (control cell, GEARS perturbation embedding). It therefore
CANNOT run under this benchmark's NT-only restriction: with controls alone it has no
perturbation to learn. Its published split shuffles each perturbation's CELLS 0.6/0.2/0.2,
so it trains on cells of every knockout it is scored on.

It is run here in that regime, unchanged, and enters the comparison as a labelled
`paired_all_cells` reference beside totalVI, never in the ranking. What this buys is that
its numbers sit on OUR data instead of the ARRAYED Papalexi experiment its paper used.

  inputs  write our pooled Papalexi in MultiPert's file convention
  run     train and test MultiPert on those inputs, one seed per output directory
  deltas  turn a finished MultiPert run into per-(knockout, protein) predicted deltas
"""
from __future__ import annotations

from pathlib import Path
from scipy import sparse
import anndata as ad
import argparse
import importlib.util
import numpy as np
import pandas as pd
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.perturbation import affine_calibration
from tools.papalexi import normalize_adt


# MultiPert's own control label and obs column; data_loader.py keys on both by name.
CONTROL_LABEL = "control"
PERTURBATION_COLUMN = "perturbation"
OUR_CONTROL_LABEL = "NT"
BLOCK = 4000


def write_rna_in_blocks(source: Path, destination: Path, cells: pd.Index,
                        perturbation: pd.Series) -> None:
    """Stream the 1.4GB all-gene RNA matrix into MultiPert's convention.

    All genes, not the 691-gene retained set: MultiPert selects its own 5,000 HVGs, and
    handing it a pre-narrowed panel would be a handicap this comparison did not intend.
    """
    backed = ad.read_h5ad(source, backed="r")
    obs_names = pd.Index(backed.obs_names.astype(str))
    rows = obs_names.get_indexer(cells)
    if (rows < 0).any():
        missing = cells[rows < 0]
        raise SystemExit(f"{len(missing)} cells are absent from {source}: {list(missing[:5])}")

    blocks = []
    for start in range(0, len(rows), BLOCK):
        chunk = backed.X[rows[start:start + BLOCK]]
        blocks.append(sparse.csr_matrix(chunk))
    var = backed.var.copy()
    backed.file.close()

    rna = ad.AnnData(X=sparse.vstack(blocks).tocsr(), var=var)
    rna.obs_names = cells
    rna.obs[PERTURBATION_COLUMN] = perturbation.to_numpy()
    rna.write_h5ad(destination)
    print(f"[inputs] RNA {rna.shape} -> {destination}")


def inputs_main() -> None:
    parser = argparse.ArgumentParser(description="Write MultiPert-convention inputs.")
    parser.add_argument("--rna", type=Path,
                        default=Path("Datasets/Papalexi_GSE153056/papalexi_rna_input.h5ad"))
    parser.add_argument("--protein", type=Path,
                        default=Path("Datasets/Papalexi_GSE153056/papalexi_protein.h5ad"))
    parser.add_argument("--metadata", type=Path, default=Path("data/papalexi/ECCITE_metadata.tsv"))
    parser.add_argument("--out-dir", type=Path, default=Path("cache/multipert/data"))
    parser.add_argument("--name", default="papalexi_pooled",
                        help="MultiPert reads {name}_RNA.h5ad and {name}_protein.h5ad")
    parser.add_argument("--pert-embeddings", type=Path,
                        default=Path("vendor/multipert/pert_embeddings.npz"))
    args = parser.parse_args()

    protein = ad.read_h5ad(args.protein)
    meta = pd.read_csv(args.metadata, sep="\t", index_col=0)
    meta.index = meta.index.astype(str)
    protein.obs_names = protein.obs_names.astype(str)

    cells = protein.obs_names.intersection(meta.index)
    protein = protein[cells].copy()
    meta = meta.loc[cells]
    # MultiPert keys the control arm on the literal string 'control'.
    perturbation = meta["gene"].astype(str).replace({OUR_CONTROL_LABEL: CONTROL_LABEL})
    print(f"[inputs] {len(cells)} cells, {perturbation.nunique() - 1} knockouts, "
          f"{int((perturbation == CONTROL_LABEL).sum())} control cells")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    protein.obs[PERTURBATION_COLUMN] = perturbation.to_numpy()
    protein.X = sparse.csr_matrix(protein.X)
    protein.write_h5ad(args.out_dir / f"{args.name}_protein.h5ad")
    print(f"[inputs] protein {protein.shape} -> {args.out_dir / (args.name + '_protein.h5ad')}")

    write_rna_in_blocks(args.rna, args.out_dir / f"{args.name}_RNA.h5ad", cells, perturbation)
    report_embedding_coverage(args.pert_embeddings, perturbation, args.out_dir)


def report_embedding_coverage(path: Path, perturbation: pd.Series, out_dir: Path) -> None:
    """Which knockouts get a real GEARS embedding and which fall back to one-hot.

    MultiPert's loader prints the fallback per CELL and moves on, which scrolls past. The
    distinction matters: a knockout on the one-hot fallback is given no gene-ontology
    information at all, so its prediction is not the method's intended behaviour. CUL3 is
    the case to watch, being one of the two post-transcriptional pairs the CRISPR
    benchmark exists to test.
    """
    embeddings = np.load(path, allow_pickle=True)
    known = {str(name) for name in embeddings["pert_list"]}
    knockouts = sorted(set(perturbation) - {CONTROL_LABEL})
    covered = [k for k in knockouts if k in known]
    fallback = [k for k in knockouts if k not in known]
    print(f"[inputs] GEARS embeddings cover {len(covered)}/{len(knockouts)} knockouts")
    if fallback:
        print(f"[inputs] ONE-HOT FALLBACK (no gene-ontology embedding): {fallback}")
    pd.DataFrame({"knockout": knockouts,
                  "gears_embedding": [k in known for k in knockouts]}
                 ).to_csv(out_dir / "pert_embedding_coverage.csv", index=False)


def load_multipert_output(out_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predicted ADT, matched control ADT, and the perturbation label per test row.

    MultiPert writes `predict.h5ad` and `test_control.h5ad` row-aligned: row i of the
    prediction is the response it predicts for the control cell in row i of the control
    file, under that row's perturbation. The delta is therefore taken row-wise before any
    grouping, which is what makes it a prediction of the SHIFT and not of the level.
    """
    predicted = ad.read_h5ad(out_dir / "predict.h5ad")
    control = ad.read_h5ad(out_dir / "test_control.h5ad")
    if predicted.n_obs != control.n_obs:
        raise SystemExit(
            f"predict.h5ad has {predicted.n_obs} rows and test_control.h5ad {control.n_obs}; "
            "they must be row-aligned for a per-cell delta to mean anything.")
    return (np.asarray(predicted.obsm["adt"], dtype=np.float64),
            np.asarray(control.obsm["adt"], dtype=np.float64),
            np.asarray(predicted.obs["perturb"].astype(str)),
            pd.Index(control.obs_names.astype(str)))


def observed_adt(protein_path: Path, metadata_path: Path, normalization: str) -> pd.DataFrame:
    """The benchmark's own ADT units, so MultiPert's shift can be put on that scale."""
    protein = ad.read_h5ad(protein_path)
    protein.obs_names = protein.obs_names.astype(str)
    meta = pd.read_csv(metadata_path, sep="\t", index_col=0)
    meta.index = meta.index.astype(str)
    cells = protein.obs_names.intersection(meta.index)
    protein, meta = protein[cells].copy(), meta.loc[cells]
    counts = protein.X.todense() if sparse.issparse(protein.X) else protein.X
    variants = normalize_adt(np.asarray(counts, dtype=np.float64),
                             np.asarray(meta["nCount_RNA"], dtype=np.float64))
    return pd.DataFrame(variants[normalization], index=cells,
                        columns=[str(v) for v in protein.var_names])


def deltas_main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-(knockout, protein) predicted deltas from a MultiPert run.")
    parser.add_argument("--run-dir", type=Path, default=Path("cache/multipert/output"))
    parser.add_argument("--protein", type=Path,
                        default=Path("Datasets/Papalexi_GSE153056/papalexi_protein.h5ad"))
    parser.add_argument("--out", type=Path,
                        default=Path("cache/results/multipert_pair_deltas.csv"))
    parser.add_argument("--metadata", type=Path, default=Path("data/papalexi/ECCITE_metadata.tsv"))
    parser.add_argument("--normalization", default="rna_size",
                        choices=["rna_size", "raw_log1p", "clr"],
                        help="must match the benchmark's --normalization")
    parser.add_argument("--model-name", default="multipert")
    args = parser.parse_args()

    predicted, control, perturbation, control_cells = load_multipert_output(args.run_dir)
    observed = observed_adt(args.protein, args.metadata, args.normalization)
    adt_names = list(observed.columns)
    if predicted.shape[1] != len(adt_names):
        raise SystemExit(
            f"MultiPert predicted {predicted.shape[1]} ADT columns against {len(adt_names)} "
            f"in {args.protein}; the panels must match for the columns to be named.")

    # MultiPert works in normalize_total(1e4)+log1p ADT units, the benchmark in its own.
    # Fit a per-protein affine map from ITS control values onto the OBSERVED control
    # values of the same cells, then apply the slope to the shift; the intercept cancels
    # in a difference. Control cells only, so the calibration never sees a knockout --
    # the same rule calibrate_direct follows for the direct predictors.
    known = control_cells.isin(observed.index)
    if not known.all():
        raise SystemExit(
            f"{int((~known).sum())} of {len(control_cells)} MultiPert control cells are "
            "not in the observed ADT table; the calibration would be fitted on a "
            "different cell set than it claims.")
    scale, _ = affine_calibration(control, observed.loc[control_cells].to_numpy(),
                                  np.arange(len(control)))
    print(f"[deltas] calibration slope per ADT: "
          f"{dict(zip(adt_names, np.round(scale, 4).tolist()))}")
    rows = []
    for knockout in sorted(set(perturbation)):
        mask = perturbation == knockout
        shift = (predicted[mask] - control[mask]).mean(axis=0) * scale
        for j, adt in enumerate(adt_names):
            rows.append({"model": args.model_name, "knockout": knockout, "adt": adt,
                         "pred_delta": float(shift[j])})
    table = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(f"[deltas] {table['knockout'].nunique()} knockouts x {len(adt_names)} ADTs "
          f"-> {args.out}")


VENDOR_CODE = PROJECT_ROOT / "vendor" / "multipert" / "code"
# MultiPert's entry point is a bare `main.py` that does `from data_loader import *`, so its
# own directory has to be importable. A plain `import main` would then bind the name
# `main` process-wide and shadow this repo's own main.py, so it is loaded under a distinct
# module name instead.
sys.path.insert(0, str(VENDOR_CODE))
MULTIPERT_SPEC = importlib.util.spec_from_file_location("multipert_entry",
                                                        VENDOR_CODE / "main.py")


def run_main() -> None:
    """Drive the vendored MultiPert. Its own procedure, only the seed is ours.

    Each seed needs its own --out-dir: main() reloads `model_best_*.pth` and
    `preprocessed_*.h5ad` when it finds them, so a shared directory would make every
    later seed a re-report of the first one's checkpoint.
    """
    parser = argparse.ArgumentParser(description="Train and test MultiPert.")
    parser.add_argument("--data-dir", type=Path, default=Path("cache/multipert/data"))
    parser.add_argument("--name", default="papalexi_pooled")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="default: cache/multipert/output/seed_{seed}")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    embeddings = args.data_dir / "pert_embeddings.npz"
    if not embeddings.exists():
        raise SystemExit(
            f"MultiPert reads its GEARS embeddings from {embeddings}. Copy or link "
            "vendor/multipert/pert_embeddings.npz there first.")

    out_dir = args.out_dir or Path(f"cache/multipert/output/seed_{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    entry = importlib.util.module_from_spec(MULTIPERT_SPEC)
    MULTIPERT_SPEC.loader.exec_module(entry)

    print(f"[run] seed={args.seed} epochs={args.epochs} -> {out_dir}")
    entry.main(str(args.data_dir), args.name, str(out_dir), args.epochs, args.seed)


COMMANDS = {"inputs": inputs_main, "run": run_main, "deltas": deltas_main}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        print(f"commands: {', '.join(COMMANDS)}")
        return 2
    COMMANDS[sys.argv.pop(1)]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
