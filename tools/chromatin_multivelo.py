#!/usr/bin/env python3
"""MultiVelo on this project's multiome objects: knn chromatin smoothing, then the v_c-free fit.

KOT never sees the RNA neighbours of an ATAC cell. MultiVelo's first step does:
`knn_smooth_chrom` imputes chromatin along a paired RNA or WNN graph. That is why a
hand-rolled version of the same step moved chromatin→spliced identity correlation from
~0.03 to ~0.76 — it is a paired operation, not an unpaired competitor.

`recover_dynamics_chrom` then fits the joint chromatin–RNA ODE without our chromatin
velocity field. HSPC is the MultiVelo paper's dataset; BMMC dynamics are not launched
from here (24k–55k cells × ~1k genes is a separate decision).

Usage:
  PYTHONPATH=. python tools/chromatin_multivelo.py smooth --dataset hspc
  PYTHONPATH=. python tools/chromatin_multivelo.py dynamics --dataset hspc
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.linear_model import Ridge

from tools.chromatin_common import Context, context

import run_kot_chromatin as runner
from src.data.chromatin import normalize_log
from src.data.chromatin_r2 import shared_splicing_targets
from src.evaluation.chromatin_eval import column_pearson, task_d_kinetics
from src.utils.arrays import to_dense

RIDGE_ALPHA = 100.0
GRAPHS = ("rna", "atac", "wnn")
RESULTS = Path("cache/results/chromatin")
SITE = Path.home() / (
    f".local/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages")


def gene_columns(columns: np.ndarray, kinetic: np.ndarray, panel: str) -> np.ndarray:
    """Output-panel indices, optionally restricted to the kinetic subset."""
    if panel == "kinetic":
        return np.asarray(columns)[np.asarray(kinetic, dtype=bool)]
    if panel == "output":
        return np.asarray(columns)
    raise ValueError(f"unknown panel {panel!r}")


def median_pearson(predicted: np.ndarray, observed: np.ndarray) -> float:
    return float(np.nanmedian(column_pearson(predicted, observed)))


def require_multivelo():
    """The default container does not ship MultiVelo; the working install lives in the user site."""
    import matplotlib
    matplotlib.use("Agg")
    if SITE.is_dir() and str(SITE) not in sys.path:
        sys.path.append(str(SITE))
    try:
        import multivelo as mv
    except ImportError as exc:
        raise ImportError(
            "multivelo is not importable in this Python. It is not in codedev_v1.0.5.sif; "
            f"the expected location is {SITE}."
        ) from exc
    return mv


def measured_rows(ctx: Context) -> np.ndarray:
    """Cells with splicing AND a finite LSI row — neighbours cannot be built on NaNs."""
    finite = np.isfinite(np.asarray(ctx.adata.obsm["lsi"], dtype=np.float32)).all(axis=1)
    return np.flatnonzero(ctx.measured & finite)


def views(ctx: Context, panel: str, max_cells: int | None, seed: int):
    """RNA and gene-level ATAC AnnData on the same cells and genes, MultiVelo's expected shape."""
    columns = gene_columns(ctx.columns, ctx.kinetic_mask, panel)
    rows = measured_rows(ctx)
    if max_cells is not None and len(rows) > max_cells:
        rows = np.sort(np.random.default_rng(seed).choice(rows, max_cells, replace=False))
    names = ctx.adata.var_names[columns].astype(str)
    cells = ctx.adata.obs_names[rows].astype(str)
    counts = to_dense(ctx.adata.obsm["gene_activity_counts"][rows][:, columns], np.float32)
    activity = np.asarray(ctx.adata.obsm["gene_activity"], dtype=np.float32)[rows][:, columns]
    target = shared_splicing_targets(ctx.adata[rows], columns)
    n_genes = len(columns)
    spliced_log = target[:, n_genes:]
    unspliced_log = target[:, :n_genes]
    # PCA on shared log-spliced, not raw counts: a depth graph would inflate Mc vs s.
    rna = ad.AnnData(X=spliced_log, obs=pd.DataFrame(index=cells), var=pd.DataFrame(index=names))
    rna.layers["spliced"] = np.expm1(spliced_log).astype(np.float32)
    rna.layers["unspliced"] = np.expm1(unspliced_log).astype(np.float32)
    rna.var["highly_variable"] = True
    atac = ad.AnnData(X=sparse.csr_matrix(np.clip(counts, 0, None)),
                      obs=pd.DataFrame(index=cells), var=pd.DataFrame(index=names))
    atac.obsm["X_lsi"] = np.asarray(ctx.adata.obsm["lsi"], dtype=np.float32)[rows]
    atac.obsm["gene_activity"] = activity
    return {
        "rna": rna, "atac": atac, "rows": rows, "columns": columns,
        "activity": activity, "counts": counts, "spliced_log": spliced_log,
        "train": np.isin(rows, ctx.rows("train")),
        "test": np.isin(rows, ctx.rows("test")),
    }


def rna_graph(rna: ad.AnnData, n_neighbors: int, n_pcs: int) -> None:
    n_pcs = min(n_pcs, rna.n_vars - 1, rna.n_obs - 1)
    sc.pp.pca(rna, n_comps=n_pcs)
    sc.pp.neighbors(rna, n_neighbors=n_neighbors, n_pcs=n_pcs)
    import scvelo as scv
    scv.pp.moments(rna, n_pcs=None, n_neighbors=None)


def impute(mv, values: np.ndarray, *, conn=None, nn_idx=None, nn_dist=None,
           n_neighbors: int) -> np.ndarray:
    handle = ad.AnnData(X=sparse.csr_matrix(np.clip(values, 0, None)))
    if conn is not None:
        mv.knn_smooth_chrom(handle, conn=conn, n_neighbors=n_neighbors)
    elif nn_idx is not None:
        mv.knn_smooth_chrom(handle, nn_idx=nn_idx, nn_dist=nn_dist)
    else:
        raise ValueError("impute needs a connectivities matrix or WNN indices")
    return np.clip(to_dense(handle.layers["Mc"], np.float32), 0, None)


def graph_neighbours(mv, atac: ad.AnnData, graph: str, rna: ad.AnnData,
                     n_neighbors: int, n_pcs: int) -> dict:
    """RNA-graph smoothing is the paired leak; ATAC-graph is the unpaired control."""
    if graph == "rna":
        return {"conn": rna.obsp["connectivities"]}
    if graph == "atac":
        handle = atac.copy()
        sc.pp.neighbors(handle, use_rep="X_lsi", n_neighbors=n_neighbors)
        return {"conn": handle.obsp["connectivities"]}
    if graph == "wnn":
        n_rna = min(n_pcs, rna.n_vars - 1, rna.n_obs - 1)
        n_atac = min(n_pcs, atac.n_vars - 1, atac.n_obs - 1)
        nn_idx, nn_dist = mv.gen_wnn(rna, atac, dims=[n_rna, n_atac], nn=n_neighbors)
        return {"nn_idx": nn_idx, "nn_dist": nn_dist}
    raise ValueError(graph)


def pairing_of(graph: str) -> str:
    """RNA/WNN graphs see paired neighbours; ATAC-graph does not; identity is cell-matched."""
    base = graph.split("_vs_")[0]
    if base in {"rna", "wnn"}:
        return "paired_multiome"
    if base == "atac":
        return "unpaired_atac_graph"
    return "cell_matched"


def score_row(method: str, graph: str, predicted: np.ndarray, observed: np.ndarray,
              train: np.ndarray, test: np.ndarray) -> list[dict]:
    """Identity on all / test cells, and a paired ridge on the KOT split."""
    pairing = pairing_of(graph)
    n_genes = int(predicted.shape[1])
    rows = [
        {"method": method, "graph": graph, "score": "identity_all", "pairing": pairing,
         "gene_pearson_median": median_pearson(predicted, observed),
         "n_cells": int(len(predicted)), "n_genes": n_genes},
    ]
    if test.any():
        rows.append({"method": method, "graph": graph, "score": "identity_test",
                     "pairing": pairing,
                     "gene_pearson_median": median_pearson(predicted[test], observed[test]),
                     "n_cells": int(test.sum()), "n_genes": n_genes})
    if train.sum() > 0 and test.sum() > 0:
        fitted = Ridge(alpha=RIDGE_ALPHA).fit(predicted[train], observed[train])
        rows.append({
            "method": method, "graph": graph, "score": "ridge_test", "pairing": pairing,
            "gene_pearson_median": median_pearson(
                fitted.predict(predicted[test]).astype(np.float32), observed[test]),
            "n_cells": int(test.sum()), "n_genes": n_genes,
        })
    return rows


def write_scores(frame: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(f"\n[multivelo] wrote {out}")
    print(frame.round(4).to_string(index=False))


def smooth_main(args: argparse.Namespace) -> int:
    mv = require_multivelo()
    ctx = context(args.dataset, args.split_seed)
    packed = views(ctx, args.panel, None, args.seed)
    print(f"[multivelo] {args.dataset}: {packed['rna'].n_obs} cells × "
          f"{packed['rna'].n_vars} genes ({args.panel})", flush=True)
    rna_graph(packed["rna"], args.n_neighbors, args.n_pcs)
    observed = packed["spliced_log"]
    rows = score_row("unsmoothed_activity", "none", packed["activity"], observed,
                     packed["train"], packed["test"])
    rows.extend(score_row("unsmoothed_counts_log", "none",
                          normalize_log(packed["counts"]), observed,
                          packed["train"], packed["test"]))
    if "Ms" in packed["rna"].layers:
        ms = to_dense(packed["rna"].layers["Ms"], np.float32)
        rows.extend(score_row("unsmoothed_activity", "none_vs_Ms", packed["activity"], ms,
                              packed["train"], packed["test"]))
    # Score smoothing of gene_activity (KOT's chromatin map). Logging the count-level
    # Mc would mix a transform into the 0.03-vs-0.76 comparison.
    for graph in args.graphs:
        print(f"[multivelo] smoothing on the {graph} graph", flush=True)
        if graph == "wnn":
            try:
                neighbours = graph_neighbours(mv, packed["atac"], graph, packed["rna"],
                                              args.n_neighbors, args.n_pcs)
            except (ValueError, np.linalg.LinAlgError) as exc:
                print(f"[multivelo] SKIP wnn: {exc}", flush=True)
                continue
        else:
            neighbours = graph_neighbours(mv, packed["atac"], graph, packed["rna"],
                                          args.n_neighbors, args.n_pcs)
        mc_activity = impute(mv, packed["activity"], n_neighbors=args.n_neighbors,
                             **neighbours)
        mc_counts = impute(mv, packed["counts"], n_neighbors=args.n_neighbors,
                           **neighbours)
        rows.extend(score_row("Mc_activity", graph, mc_activity, observed,
                              packed["train"], packed["test"]))
        rows.extend(score_row("Mc_counts_log", graph, normalize_log(mc_counts), observed,
                              packed["train"], packed["test"]))
        if "Ms" in packed["rna"].layers:
            ms = to_dense(packed["rna"].layers["Ms"], np.float32)
            rows.extend(score_row("Mc_activity", f"{graph}_vs_Ms", mc_activity, ms,
                                  packed["train"], packed["test"]))
        packed["atac"].layers[f"Mc_{graph}"] = mc_counts
    packed["atac"].write_h5ad(args.atac_out or (RESULTS / f"multivelo_{args.dataset}_atac.h5ad"))
    write_scores(pd.DataFrame(rows).assign(dataset=args.dataset, panel=args.panel),
                 args.out or (RESULTS / f"multivelo_{args.dataset}_smooth.csv"))
    return 0


def dynamics_main(args: argparse.Namespace) -> int:
    mv = require_multivelo()
    ctx = context(args.dataset, args.split_seed)
    packed = views(ctx, args.panel, args.max_cells, args.seed)
    print(f"[multivelo] dynamics {args.dataset}: {packed['rna'].n_obs} cells × "
          f"{packed['rna'].n_vars} genes", flush=True)
    rna_graph(packed["rna"], args.n_neighbors, args.n_pcs)
    neighbours = graph_neighbours(mv, packed["atac"], "rna", packed["rna"],
                                  args.n_neighbors, args.n_pcs)
    packed["atac"].layers["Mc"] = impute(
        mv, packed["counts"], n_neighbors=args.n_neighbors, **neighbours)
    n_jobs = args.n_jobs or int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    fitted = mv.recover_dynamics_chrom(
        packed["rna"], packed["atac"], gene_list=list(packed["rna"].var_names),
        max_iter=args.max_iter, plot=False, save_plot=False, parallel=n_jobs > 1,
        n_jobs=n_jobs, n_pcs=args.n_pcs, n_neighbors=args.n_neighbors,
        embedding="X_pca")
    out_h5ad = args.out or (RESULTS / f"multivelo_{args.dataset}_dynamics.h5ad")
    out_h5ad.parent.mkdir(parents=True, exist_ok=True)
    fitted.write_h5ad(out_h5ad)
    print(f"[multivelo] wrote {out_h5ad}", flush=True)
    genes = pd.DataFrame({"gene": fitted.var_names.astype(str)})
    for column in ("fit_likelihood", "fit_alpha", "fit_beta", "fit_gamma", "velo_s_genes"):
        if column in fitted.var:
            genes[column] = fitted.var[column].to_numpy()
    genes.to_csv(out_h5ad.with_suffix(".genes.csv"), index=False)

    payload = {"dataset": args.dataset, "n_cells": int(fitted.n_obs),
               "n_genes": int(fitted.n_vars), "h5ad": str(out_h5ad),
               "pairing": "paired_multiome", "graph": "rna",
               "units": "multivelo_Ms_from_shared_cp10k_vs_project_scvelo",
               "n_test_cells": int(packed["test"].sum()),
               "n_velocity_genes": int(fitted.var["velo_s_genes"].sum())
               if "velo_s_genes" in fitted.var else None}
    reference = runner.reference_velocity(ctx.adata, "velocity_scvelo")
    if reference is not None and "velo_s" in fitted.layers:
        gene_index = ctx.adata.var_names.get_indexer(fitted.var_names.astype(str))
        if (gene_index < 0).any():
            print("[multivelo] scVelo comparison skipped: fitted genes missing from the panel")
        else:
            predicted = to_dense(fitted.layers["velo_s"], np.float32)
            observed = reference[packed["rows"]][:, gene_index]
            covered = np.abs(observed).sum(axis=0) > 0
            test = packed["test"]
            comparisons = [("vs_scvelo_s_all_fitted", np.ones(len(predicted), dtype=bool)),
                           ("vs_scvelo_s_test", test)]
            keep = covered & np.asarray(fitted.var["velo_s_genes"], dtype=bool) \
                if "velo_s_genes" in fitted.var else covered
            for name, cells in comparisons:
                if not cells.any() or not covered.any():
                    continue
                payload[name] = task_d_kinetics(
                    predicted[cells][:, covered], observed[cells][:, covered])
                print(f"[multivelo] {name} centred cosine "
                      f"{payload[name]['cell_cosine_centred_median']:+.4f}  "
                      f"null p95 {payload[name]['cell_cosine_centred_null_p95']:+.4f}",
                      flush=True)
                if keep.any():
                    payload[f"{name}_velocity_genes"] = task_d_kinetics(
                        predicted[cells][:, keep], observed[cells][:, keep])
    out_h5ad.with_suffix(".json").write_text(json.dumps(payload, indent=2, default=float))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("smooth", "dynamics"):
        child = sub.add_parser(name)
        child.add_argument("--dataset", choices=runner.DATASETS, default="hspc")
        child.add_argument("--split-seed", type=int, default=0)
        child.add_argument("--seed", type=int, default=0)
        child.add_argument("--panel", choices=("kinetic", "output"), default="kinetic")
        child.add_argument("--n-neighbors", type=int, default=30)
        child.add_argument("--n-pcs", type=int, default=30)
        child.add_argument("--out", type=Path, default=None)
        if name == "smooth":
            child.add_argument("--graphs", nargs="+", default=list(GRAPHS), choices=GRAPHS)
            child.add_argument("--atac-out", type=Path, default=None)
        else:
            child.add_argument("--max-cells", type=int, default=8000,
                               help="compute cap on recover_dynamics_chrom, not a new split")
            child.add_argument("--max-iter", type=int, default=5)
            child.add_argument("--n-jobs", type=int, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    return smooth_main(args) if args.command == "smooth" else dynamics_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
