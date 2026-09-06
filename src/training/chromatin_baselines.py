"""Competing unpaired integration methods, on the chromatin experiment's own split.

Every method is handed the SAME two matrices KOT gets — gene activity on the ATAC training
half, RNA on the disjoint RNA half — and is scored by the same Task A-D code. Feeding one
method raw peaks and another gene activity would make the comparison about the input
representation rather than the method, so input parity is the rule here and the cost is
stated: scGLUE and scDART are designed around raw peaks plus a peak-to-gene prior, and on a
gene-indexed panel their guidance/regularisation term is close to an identity.

Two asymmetries that favour the BASELINES, recorded so they are not read as KOT losing:

  TRANSDUCTIVE  most of these methods embed the cells they were fitted on and have no
                out-of-sample path, so they see the test cells' ATAC (never the pairing)
                during fitting. KOT is inductive: phi is applied to held-out ATAC it never
                saw. Where a method does support out-of-sample use, this module uses it.

  SUBSAMPLED    MaxFuse and moscot build dense n x n couplings, so they run on a capped
                sample. `n_cells` in the result says what each one actually saw.

What each can answer differs, and that is a property of the method, not of the harness:

  scDART     learns a nonlinear ATAC->RNA network with a gene-activity prior — the same
             object as KOT's phi, without the kinetic law. Native RNA prediction, and its
             Jacobian can be read the same way, so it reaches all four tasks.
  moscot     an OT plan; barycentric projection gives an RNA prediction.
  scGLUE     a shared latent; RNA prediction needs kNN imputation from the RNA half.
  MaxFuse    a matching plus a shared latent; same kNN route.
  scMultiNODE a shared latent with neural-ODE dynamics; needs TIMEPOINTS, so HSPC only.

Each adapter imports its own library inside the function, the same way the runner's
registry defers model imports: five vendored research codebases is five ways for this
module to become unimportable, and one broken baseline should not take the other four
down with it.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.neighbors import NearestNeighbors

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENDOR = PROJECT_ROOT / "vendor"
for name in ["scDART", "scMultiNODE", "maxfuse"]:
    path = str(VENDOR / name)
    if path not in sys.path:
        sys.path.insert(0, path)

# scMultiNODE is a research repo, not a package: its modules import each other by
# top-level name (`from model.layer import ...`), so its root has to be on the path too.
SCMULTINODE_ROOT = str(VENDOR / "scMultiNODE")


@dataclass
class BaselineInputs:
    """The same split KOT trains on, with the pairing hidden."""
    atac_train: np.ndarray
    rna_train: np.ndarray
    atac_test: np.ndarray
    rna_test: np.ndarray
    genes: list[str]
    seed: int
    day_train_atac: np.ndarray | None = None
    day_train_rna: np.ndarray | None = None
    day_test: np.ndarray | None = None


@dataclass
class BaselineResult:
    """What a method produced, and what it is allowed to be scored on."""
    predicted: np.ndarray | None = None      # test ATAC cells, in RNA gene space
    atac_latent: np.ndarray | None = None    # test ATAC cells, in the method's own space
    rna_latent: np.ndarray | None = None     # test RNA cells, same space
    train_rna_latent: np.ndarray | None = None   # for kNN imputation, never the test RNA
    train_rna_values: np.ndarray | None = None   # the RNA rows train_rna_latent embeds
    prediction_kind: str = "latent"          # "direct" when the method predicts RNA itself
    notes: dict = field(default_factory=dict)


def impute_from_latent(result: BaselineResult, n_neighbors: int) -> np.ndarray:
    """RNA for each test ATAC cell, averaged over its nearest TRAINING RNA cells.

    Training RNA, never the test RNA: a latent method has no RNA prediction of its own, and
    imputing from the held-out cells would hand it the answer it is being scored against.

    `train_rna_values` is carried on the result beside `train_rna_latent` rather than
    re-derived from row indices by the caller. A subsampled method embeds a SUBSET of the
    training cells, and an earlier version looked the values up by position in the full
    training matrix instead — 62 of 1200 rows happened to line up, the shapes matched, no
    error was raised, and every latent baseline's Task A read as chance.
    """
    assert result.train_rna_latent is not None and result.train_rna_values is not None, (
        "a latent method must return train_rna_latent AND the train_rna_values it embeds")
    assert len(result.train_rna_latent) == len(result.train_rna_values), (
        f"bank latent has {len(result.train_rna_latent)} rows but its RNA values have "
        f"{len(result.train_rna_values)} — they must be the same cells in the same order")
    neighbors = NearestNeighbors(n_neighbors=n_neighbors).fit(result.train_rna_latent)
    indices = neighbors.kneighbors(result.atac_latent, return_distance=False)
    return result.train_rna_values[indices].mean(axis=1).astype(np.float32)


def subsample(n_rows: int, limit: int, seed: int) -> np.ndarray:
    if n_rows <= limit:
        return np.arange(n_rows)
    return np.sort(np.random.default_rng(seed).choice(n_rows, limit, replace=False))


def make_adata(matrix: np.ndarray, genes: list[str], n_comps: int, seed: int) -> ad.AnnData:
    """An AnnData with a PCA, which is the representation the OT methods align on."""
    handle = ad.AnnData(X=np.asarray(matrix, dtype=np.float32),
                        var=pd.DataFrame(index=genes))
    sc.tl.pca(handle, n_comps=min(n_comps, min(handle.shape) - 1), random_state=seed)
    return handle


def run_moscot(inputs: BaselineInputs, max_cells: int, epsilon: float,
               n_comps: int) -> BaselineResult:
    """Gromov-Wasserstein between the two modalities, then barycentric projection.

    Pure GW on the two PCA geometries: it needs no shared feature space, which is what
    makes it a fair unpaired baseline even though our panel happens to share gene names.
    """
    # moscot solves through JAX/XLA, and the GPU nodes ship no ptxas — XLA then aborts the
    # whole process (SIGABRT, not an exception). CPU is also the right place for it: this
    # is a dense GW solve, not a torch workload.
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    from moscot.problems.cross_modality import TranslationProblem

    # ONE sample of test cells, used for both modalities. Sampling the two sides
    # independently would make FOSCTTM compare a cell against strangers rather than
    # against its own partner, and it would look like a near-chance alignment.
    test_rows = subsample(len(inputs.atac_test), max_cells, inputs.seed)
    bank_rows = subsample(len(inputs.rna_train), max_cells, inputs.seed + 2)
    rna_side = np.vstack([inputs.rna_train[bank_rows], inputs.rna_test[test_rows]])
    source = make_adata(inputs.atac_test[test_rows], inputs.genes, n_comps, inputs.seed)
    target = make_adata(rna_side, inputs.genes, n_comps, inputs.seed)

    problem = TranslationProblem(source, target)
    problem = problem.prepare(src_attr="X_pca", tgt_attr="X_pca", joint_attr=None)
    problem = problem.solve(alpha=1.0, epsilon=epsilon)
    translated = np.asarray(problem.translate("src", "tgt", forward=True), dtype=np.float32)
    embedding = np.asarray(target.obsm["X_pca"], dtype=np.float32)
    return BaselineResult(
        atac_latent=translated,
        rna_latent=embedding[len(bank_rows):],
        train_rna_latent=embedding[:len(bank_rows)],
        train_rna_values=inputs.rna_train[bank_rows],
        prediction_kind="latent",
        notes={"n_cells": int(len(test_rows)), "transductive": True,
               "test_rows": test_rows.tolist()},
    )


def run_scdart(inputs: BaselineInputs, n_epochs: int, latent_dim: int, max_cells: int,
               device) -> BaselineResult:
    """scDART: a nonlinear ATAC->RNA network plus a shared latent.

    `reg` is the gene-activity prior. Both modalities are on one gene panel here, so the
    prior is the identity — the same statement KOT's G makes, which is what puts the two
    methods on the same footing.
    """
    import torch
    from scDART import scDART as ScDARTModel

    # scDART computes a full diffusion distance per modality, which is O(n^2): the
    # uncapped 38k-cell run was still going after an hour.
    test_rows = subsample(len(inputs.atac_test), max_cells, inputs.seed)
    train_rna_rows = subsample(len(inputs.rna_train), max_cells, inputs.seed + 1)
    train_atac_rows = subsample(len(inputs.atac_train), max_cells, inputs.seed + 2)
    rna = np.vstack([inputs.rna_train[train_rna_rows],
                     inputs.rna_test[test_rows]]).astype(np.float32)
    atac = np.vstack([inputs.atac_train[train_atac_rows],
                      inputs.atac_test[test_rows]]).astype(np.float32)
    prior = np.eye(atac.shape[1], dtype=np.float32)

    model = ScDARTModel(n_epochs=n_epochs, latent_dim=latent_dim, seed=inputs.seed,
                        device=device, use_anchor=False)
    model.fit(rna, atac, prior)
    z_rna, z_atac = model.transform(rna, atac)

    # The gene-activity module IS the ATAC->RNA map, so the prediction is native.
    with torch.no_grad():
        predicted = model.model_dict["gene_act"](
            torch.as_tensor(inputs.atac_test[test_rows], dtype=torch.float32, device=device)
        ).cpu().numpy().astype(np.float32)

    n_train_rna, n_train_atac = len(train_rna_rows), len(train_atac_rows)
    return BaselineResult(
        predicted=predicted,
        atac_latent=z_atac[n_train_atac:],
        rna_latent=z_rna[n_train_rna:],
        train_rna_latent=z_rna[:n_train_rna],
        train_rna_values=inputs.rna_train[train_rna_rows],
        prediction_kind="direct",
        notes={"n_cells": int(len(atac)), "transductive": True, "has_jacobian": True,
               "n_epochs": n_epochs, "test_rows": test_rows.tolist(),
               "train_rna_rows": train_rna_rows.tolist()},
    )


def run_scglue(inputs: BaselineInputs, n_comps: int, max_epochs: int) -> BaselineResult:
    """scGLUE: a graph-guided VAE over the two feature spaces.

    The guidance graph pairs each ATAC feature with its gene. On a gene-activity panel that
    is one-to-one, which is scGLUE outside its intended regime (it expects peaks and a
    genomic-distance graph) — the price of input parity, and it is stated in the report.
    """
    import networkx as nx
    from scglue.models import configure_dataset, fit_SCGLUE

    atac = make_adata(np.vstack([inputs.atac_train, inputs.atac_test]),
                      [f"{g}_atac" for g in inputs.genes], n_comps, inputs.seed)
    rna = make_adata(np.vstack([inputs.rna_train, inputs.rna_test]),
                     inputs.genes, n_comps, inputs.seed)
    atac.layers["counts"] = np.clip(atac.X, 0, None)
    rna.layers["counts"] = np.clip(rna.X, 0, None)

    graph = nx.Graph()
    graph.add_nodes_from(list(rna.var_names) + list(atac.var_names))
    for node in list(rna.var_names) + list(atac.var_names):
        graph.add_edge(node, node, weight=1.0, sign=1)
    for gene in inputs.genes:
        graph.add_edge(gene, f"{gene}_atac", weight=1.0, sign=1)

    configure_dataset(rna, "Normal", use_highly_variable=False, use_rep="X_pca")
    configure_dataset(atac, "Normal", use_highly_variable=False, use_rep="X_pca")
    model = fit_SCGLUE({"rna": rna, "atac": atac}, graph,
                       fit_kws={"max_epochs": max_epochs})
    rna_embedding = model.encode_data("rna", rna)
    atac_embedding = model.encode_data("atac", atac)

    n_train_rna, n_train_atac = len(inputs.rna_train), len(inputs.atac_train)
    return BaselineResult(
        atac_latent=atac_embedding[n_train_atac:],
        rna_latent=rna_embedding[n_train_rna:],
        train_rna_latent=rna_embedding[:n_train_rna],
        train_rna_values=inputs.rna_train,
        prediction_kind="latent",
        notes={"n_cells": int(atac.n_obs), "transductive": True,
               "guidance_edges": int(graph.number_of_edges()), "max_epochs": max_epochs},
    )


def run_maxfuse(inputs: BaselineInputs, max_cells: int, n_comps: int) -> BaselineResult:
    """MaxFuse: iterative matching over shared features, refined into a shared latent.

    The gene panel is literally shared between the two modalities here, which is MaxFuse's
    best case rather than its worst — it is built for panels with few shared features.
    """
    import maxfuse as mf

    test_rows = subsample(len(inputs.atac_test), max_cells, inputs.seed)
    bank_rows = subsample(len(inputs.rna_train), max_cells, inputs.seed + 2)
    atac = inputs.atac_test[test_rows].astype(np.float32)
    rna = np.vstack([inputs.rna_train[bank_rows],
                     inputs.rna_test[test_rows]]).astype(np.float32)

    fusor = mf.model.Fusor(shared_arr1=rna, shared_arr2=atac, active_arr1=rna,
                           active_arr2=atac, method="centroid_shrinkage")
    fusor.split_into_batches(max_outward_size=max_cells, matching_ratio=3,
                             metacell_size=2, verbose=False)
    fusor.construct_graphs(n_neighbors1=15, n_neighbors2=15, svd_components1=n_comps,
                           svd_components2=n_comps, verbose=False)
    fusor.find_initial_pivots(wt1=0.3, wt2=0.3, svd_components1=n_comps,
                              svd_components2=n_comps)
    # cca_components defaults to None and match_utils asserts `1 <= cca_components`,
    # so it has to be given explicitly.
    fusor.refine_pivots(wt1=0.3, wt2=0.3, svd_components1=n_comps,
                        svd_components2=n_comps, cca_components=min(20, n_comps),
                        n_iters=1, verbose=False)
    fusor.filter_bad_matches(target="pivot", filter_prop=0.3, verbose=False)
    rna_embedding, atac_embedding = fusor.get_embedding(active_arr1=rna, active_arr2=atac)
    rna_embedding = np.asarray(rna_embedding, dtype=np.float32)
    return BaselineResult(
        atac_latent=np.asarray(atac_embedding, dtype=np.float32),
        rna_latent=rna_embedding[len(bank_rows):],
        train_rna_latent=rna_embedding[:len(bank_rows)],
        train_rna_values=inputs.rna_train[bank_rows],
        prediction_kind="latent",
        notes={"n_cells": int(len(test_rows)), "transductive": True,
               "test_rows": test_rows.tolist()},
    )


def run_scmultinode(inputs: BaselineInputs, latent_dim: int, iters: int) -> BaselineResult:
    """scMultiNODE: a shared latent whose dynamics is a neural ODE over timepoints.

    Needs real timepoints on BOTH modalities, so it runs on HSPC (day 0 / day 7) and not
    on BMMC, which has none. This is the only baseline that models dynamics at all, which
    is why it is worth running despite that restriction.
    """
    assert inputs.day_train_atac is not None, (
        "scMultiNODE needs timepoints on both modalities; BMMC has none, so this baseline "
        "is defined for HSPC only")
    import torch

    if SCMULTINODE_ROOT not in sys.path:
        sys.path.insert(0, SCMULTINODE_ROOT)
    from modal_integration.run_scMultiNODE import scMultiNODEAlign

    def per_timepoint(train, train_days, test, test_days):
        """Blocks grouped by timepoint, plus the row order needed to undo the grouping.

        The model wants one block per timepoint, so the returned rows are in timepoint
        order rather than [train..., test...]. `order` maps each returned row back to its
        position in the stacked input, which is the only way to recover which rows are the
        test cells and to put them back in a canonical order the other modality shares.
        """
        full = np.vstack([train, test])
        labels = np.concatenate([train_days, test_days])
        levels = np.unique(labels)
        # torch tensors: scMultiNODETrain concatenates the blocks with torch.cat, and the
        # timepoints reach torchdiffeq, which asserts on a plain list.
        blocks = [torch.as_tensor(full[labels == level], dtype=torch.float32)
                  for level in levels]
        order = np.concatenate([np.flatnonzero(labels == level) for level in levels])
        return blocks, torch.arange(len(levels), dtype=torch.float32), order, len(train)

    rna_blocks, rna_tps, rna_order, n_rna_train = per_timepoint(
        inputs.rna_train, inputs.day_train_rna, inputs.rna_test, inputs.day_test)
    atac_blocks, atac_tps, atac_order, n_atac_train = per_timepoint(
        inputs.atac_train, inputs.day_train_atac, inputs.atac_test, inputs.day_test)
    rna_latent, atac_latent = scMultiNODEAlign(
        rna_blocks, atac_blocks, rna_tps, atac_tps, latent_dim=latent_dim, iters=iters)
    rna_latent = np.vstack([np.asarray(block) for block in rna_latent]).astype(np.float32)
    atac_latent = np.vstack([np.asarray(block) for block in atac_latent]).astype(np.float32)

    def split(latent, order, n_train):
        """Test rows in canonical test order, and the training rows as the kNN bank."""
        test_mask = order >= n_train
        test_rows = latent[test_mask][np.argsort(order[test_mask])]
        return test_rows, latent[~test_mask]

    atac_test_latent, _ = split(atac_latent, atac_order, n_atac_train)
    rna_test_latent, rna_bank = split(rna_latent, rna_order, n_rna_train)
    # The bank rows come back in timepoint order, so the VALUES must be reordered to match
    # rather than assumed to be the original training order.
    bank_order = rna_order[rna_order < n_rna_train]
    return BaselineResult(
        atac_latent=atac_test_latent,
        rna_latent=rna_test_latent,
        train_rna_latent=rna_bank,
        train_rna_values=inputs.rna_train[bank_order],
        prediction_kind="latent",
        notes={"transductive": True, "timepoints": int(len(rna_tps)), "iters": iters},
    )


BASELINES = {
    "moscot": run_moscot,
    "scdart": run_scdart,
    "scglue": run_scglue,
    "maxfuse": run_maxfuse,
    "scmultinode": run_scmultinode,
}
