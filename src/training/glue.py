from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import networkx as nx
VENDOR_GLUE_PATH = Path(__file__).resolve().parents[2] / "vendor" / "glue"
if VENDOR_GLUE_PATH.exists():
    sys.path.insert(0, str(VENDOR_GLUE_PATH))

from scglue.models import configure_dataset, fit_SCGLUE

from src.data.projection import normalize_name


def feature_key(name: str) -> str:
    key = normalize_name(str(name))
    if key.startswith("SYNTHETIC_RNA_"):
        return key.replace("SYNTHETIC_RNA_", "SYNTHETIC_FEATURE_", 1)
    if key.startswith("SYNTHETIC_PROTEIN_"):
        return key.replace("SYNTHETIC_PROTEIN_", "SYNTHETIC_FEATURE_", 1)
    return key


def first_present(mapping, keys):
    for key in keys:
        if key in mapping:
            value = mapping[key]
            if value is not None:
                return value
    return None


def links_from_uns(rna_adata, second_adata) -> list[tuple[str, str, float, int]]:
    links = None
    for uns in (rna_adata.uns, second_adata.uns):
        for key in ("rna_protein_feature_links", "guidance_edges"):
            if key in uns:
                links = uns[key]
                break
        if links is not None:
            break

    if links is None:
        return []

    if isinstance(links, dict):
        rna_names = first_present(links, ("rna", "gene", "source"))
        second_names = first_present(links, ("second", "protein", "target"))
        if rna_names is None or second_names is None:
            raise ValueError("Guidance links must define RNA and second-modality feature names.")
        if len(rna_names) != len(second_names):
            raise ValueError("Guidance RNA and second-modality feature lists must have the same length.")
        weights = links.get("weight", [1.0] * len(rna_names))
        signs = links.get("sign", [1] * len(rna_names))
        if len(weights) != len(rna_names) or len(signs) != len(rna_names):
            raise ValueError("Guidance weights and signs must match the number of guidance links.")
        return [
            (str(gene), str(feat), float(weight), int(sign))
            for gene, feat, weight, sign in zip(rna_names, second_names, weights, signs)
        ]

    parsed = []
    for row in links:
        if isinstance(row, dict):
            gene = first_present(row, ("rna", "gene", "source"))
            feat = first_present(row, ("second", "protein", "target"))
            if gene is None or feat is None:
                raise ValueError("Each guidance link must include RNA and second-modality feature names.")
            weight = row.get("weight", 1.0)
            sign = row.get("sign", 1)
        else:
            if len(row) < 2:
                raise ValueError("Each guidance link row must contain at least two feature names.")
            gene, feat, *rest = row
            weight = rest[0] if len(rest) >= 1 else 1.0
            sign = rest[1] if len(rest) >= 2 else 1
        parsed.append((str(gene), str(feat), float(weight), int(sign)))
    return parsed


def build_guidance_graph(
    rna_adata,
    second_adata,
    *,
    uniform_fallback: bool = False,
) -> nx.Graph:
    """Bipartite RNA–second graph. GLUE requires a self-loop on every var_name."""
    rna_features    = rna_adata.var_names.tolist()
    second_features = second_adata.var_names.tolist()
    rna_set         = set(rna_features)

    vertices = rna_features + [f for f in second_features if f not in rna_set]

    graph = nx.Graph()
    graph.add_nodes_from(vertices)

    for v in vertices:
        graph.add_edge(v, v, weight=1.0, sign=1)

    second_set = set(second_features)
    added = 0
    explicit_links = links_from_uns(rna_adata, second_adata)
    for gene, feat, weight, sign in explicit_links:
        if gene in rna_set and feat in second_set:
            graph.add_edge(gene, feat, weight=weight, sign=sign)
            added += 1

    if explicit_links and added == 0:
        raise ValueError("Guidance links were provided, but none matched RNA/protein var_names.")

    if added == 0:
        rna_by_key = {}
        for gene in rna_features:
            rna_by_key.setdefault(feature_key(gene), gene)
        for feat in second_features:
            gene = rna_by_key.get(feature_key(feat))
            if gene is not None:
                graph.add_edge(gene, feat, weight=1.0, sign=1)
                added += 1

    if added == 0 and not uniform_fallback:
        raise ValueError("No GLUE cross-feature guidance edges found; enable glue_uniform_fallback to force a uniform graph.")

    if added == 0 and uniform_fallback:
        w = 1.0 / max(len(rna_features), 1)
        for feat in [f for f in second_features if f not in rna_set]:
            for gene in rna_features:
                graph.add_edge(gene, feat, weight=w, sign=1)
                added += 1

    print(
        "[glue] Guidance graph: "
        f"{len(rna_features)} RNA features, {len(second_features)} second-modality "
        f"features, {added} cross-feature edges."
    )

    return graph


def run_glue(context: dict, cfg: dict) -> tuple[list, np.ndarray | None]:
    """GLUE shared latent. No coupling. Under a fit restriction, RNA stays full-n."""
    rna_adata    = context["rna_adata"]
    second_adata = context["second_adata"]
    second_label = context["second_label"]

    latent_dim  = int(cfg.get("latent_dim", 50))
    max_epochs  = int(cfg.get("max_epochs",  200))
    lr          = float(cfg.get("lr",        2e-3))
    lam_data    = float(cfg.get("lam_data",  1.0))
    lam_kl      = float(cfg.get("lam_kl",   1.0))
    lam_graph   = float(cfg.get("lam_graph", 0.02))
    lam_align   = float(cfg.get("lam_align", 0.05))
    uniform_fallback = bool(cfg.get("glue_uniform_fallback", False))
    seed        = int(cfg.get("seed",        42))

    # RNA: NB on raw counts (paper default); protein: Normal on CLR-normalized adata.X
    configure_dataset(rna_adata,    "NB",     use_highly_variable=False, use_layer="counts")
    configure_dataset(second_adata, "Normal", use_highly_variable=False)

    graph = build_guidance_graph(
        rna_adata,
        second_adata,
        uniform_fallback=uniform_fallback,
    )

    adatas = {"rna": rna_adata, second_label: second_adata}

    model = fit_SCGLUE(
        adatas, graph,
        init_kws=dict(latent_dim=latent_dim, random_seed=seed),
        compile_kws=dict(lam_data=lam_data, lam_kl=lam_kl,
                         lam_graph=lam_graph, lam_align=lam_align, lr=lr),
        fit_kws=dict(max_epochs=max_epochs),
    )

    rna_emb    = model.encode_data("rna",        rna_adata)
    second_emb = model.encode_data(second_label, second_adata)

    return [rna_emb, second_emb], None
