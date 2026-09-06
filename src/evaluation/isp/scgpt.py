"""scGPT in-silico perturbation, transferred into frozen NT-only KOT coordinates.

scGPT is the foundation-model representative item 31 asks for. Its perturbation head,
TransformerGenerator, consumes the same GEARS PertData batches the GEARS runner builds, so
this reuses that data path and differs only in the model: a 12-layer transformer
initialised from the scGPT_human checkpoint (33M cells) rather than trained from scratch.
Starting from random weights would make it a generic transformer baseline and would not
answer what item 31 asks, so the pretrained weights are required, not optional.

Run it with the isolated interpreter:
    cache/.scgpt_venv/bin/python run_crispr_isp.py scgpt ...

SUPERVISION: like GEARS, scGPT fine-tunes on observed KO cells. That is more than KOT
receives. Report the regime with any number.
"""
from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from gears import PertData
from gears.utils import create_cell_graph_dataset_for_prediction
from scgpt.model import TransformerGenerator
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scipy import sparse
from torch import nn
from torch_geometric.loader import DataLoader

from src.evaluation.isp.common import (
    apply_mean_shift,
    profile_row,
    usable_perturbations,
    write_isp_outputs,
)
from src.evaluation.regvelo_isp import positive_feature_scales
from src.utils.arrays import to_dense

CONTROL_LABEL = "NT"
PAD_TOKEN = "<pad>"
PAD_VALUE = 0
PERT_PAD_ID = 2


def add_arguments(parser) -> None:
    parser.add_argument("--rna", type=Path,
                        default=Path("cache/velocity/papalexi_nt_only/"
                                     "papalexi_scvelo_results_nt_only.h5ad"))
    parser.add_argument("--layer", default="Ms")
    parser.add_argument("--wide-rna", type=Path,
                        default=Path("cache/multipert/data/papalexi_pooled_RNA.h5ad"),
                        help="full gene matrix, needed because most knockout genes are "
                             "not among the checkpoint's HVGs")
    parser.add_argument("--data-dir", type=Path, default=Path("cache/scgpt_data"))
    parser.add_argument("--model-dir", type=Path, default=Path("cache/scgpt"))
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="accepted so one sbatch contract covers every ISP; scGPT "
                             "fits in the checkpoint's own coordinates and does not read it")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-ko-cells", type=int, default=20)
    parser.add_argument("--min-replicates", type=int, default=2)
    parser.add_argument("--pool-size", type=int, default=300,
                        help="control cells sampled per perturbation")
    parser.add_argument("--device", default="cuda")


def load_pretrained(model: nn.Module, checkpoint: Path) -> tuple[int, int]:
    """Copy every pretrained tensor whose name and shape still match.

    The perturbation head adds and resizes layers relative to the released foundation
    checkpoint, so a strict load fails. Shape-matched partial loading is what upstream's
    own perturbation tutorial does; the counts are returned so a silent near-empty load
    cannot pass for a successful one.
    """
    pretrained = torch.load(checkpoint, map_location="cpu")
    if isinstance(pretrained, dict) and "model_state_dict" in pretrained:
        pretrained = pretrained["model_state_dict"]
    current = model.state_dict()
    matched = {key: value for key, value in pretrained.items()
               if key in current and current[key].shape == value.shape}
    current.update(matched)
    model.load_state_dict(current)
    return len(matched), len(current)


def build_generator(model_dir: Path, vocab: GeneVocab, device: torch.device):
    """The scGPT perturbation head, sized from the released checkpoint's own args.json."""
    config = json.loads((model_dir / "args.json").read_text())
    return TransformerGenerator(
        ntoken=len(vocab),
        d_model=config.get("embsize", 512),
        nhead=config.get("nheads", config.get("nhead", 8)),
        d_hid=config.get("d_hid", 512),
        nlayers=config.get("nlayers", 12),
        nlayers_cls=3, n_cls=1, vocab=vocab, dropout=0.2,
        pad_token=PAD_TOKEN, pad_value=PAD_VALUE, pert_pad_id=PERT_PAD_ID,
        do_mvc=False, cell_emb_style="cls", mvc_decoder_style="inner product, detach",
        use_fast_transformer=False,
    ).to(device)


def finetune(model, loader, gene_ids_tensor, epochs: int, lr: float) -> None:
    """Fine-tune the perturbation head on the observed KO cells in the training split."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in loader:
            batch.to(gene_ids_tensor.device)
            size = len(batch.pert)
            values = batch.x[:, 0].view(size, -1)
            flags = batch.x[:, 1].long().view(size, -1)
            target = batch.y.view(size, -1)
            optimizer.zero_grad()
            output = model(
                gene_ids_tensor.repeat(size, 1), values, flags,
                src_key_padding_mask=torch.zeros_like(values, dtype=torch.bool),
            )
            loss = criterion(output["mlm_output"], target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss)
        print(f"[scgpt] epoch {epoch + 1}/{epochs} "
              f"loss {total_loss / max(len(loader), 1):.5f}")


def run(args) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    rna = ad.read_h5ad(args.rna)
    if args.layer not in rna.layers:
        raise ValueError(f"{args.rna} has no layer '{args.layer}'")
    kot = to_dense(rna.layers[args.layer], dtype=np.float64)
    kot_genes = [str(gene) for gene in rna.var_names]
    groups = rna.obs["gene"].astype(str).to_numpy()
    replicates = rna.obs["replicate"].astype(str).to_numpy()
    usable = set(usable_perturbations(groups, replicates, args.min_ko_cells,
                                      args.min_replicates, CONTROL_LABEL))

    # scGPT's perturbation head flags the knocked-out gene at its own position INSIDE the
    # expression vector, so a knockout that is not measured cannot be represented at all.
    # Only 5 of 24 CRISPR targets are among the 1,276 HVGs the checkpoint consumes, so the
    # model is fitted on the union of those genes and the knockout genes themselves, taken
    # from the full matrix, and the predicted shift is converted back afterwards.
    wide = ad.read_h5ad(args.wide_rna)
    if list(wide.obs_names) != list(rna.obs_names):
        raise ValueError("The wide matrix does not carry the checkpoint's cells in order")
    wide_index = pd.Index([str(gene) for gene in wide.var_names])
    panel = [gene for gene in kot_genes if gene in wide_index]
    panel += sorted(t for t in usable if t in wide_index and t not in set(panel))
    columns = wide_index.get_indexer(pd.Index(panel))
    values = to_dense(wide[:, columns].X, dtype=np.float32)
    frame = ad.AnnData(sparse.csr_matrix(values),
                       obs=rna.obs.copy(), var=pd.DataFrame(index=pd.Index(panel)))
    labels = frame.obs["gene"].astype(str)
    frame.obs["condition"] = np.where(labels == CONTROL_LABEL, "ctrl", labels + "+ctrl")
    frame.obs["cell_type"] = "THP1"
    frame.var["gene_name"] = frame.var_names.astype(str)
    # A knockout whose gene is not in the panel cannot be flagged, and PertData builds a
    # graph for EVERY condition it is handed, so such cells must be dropped here rather
    # than filtered at prediction time -- otherwise dataset creation itself raises.
    representable = {t for t in usable if t in set(panel)}
    unrepresentable = sorted(usable - representable)
    keep = np.isin(groups, list(representable)) | (groups == CONTROL_LABEL)
    frame = frame[keep].copy()
    print(f"[scgpt] panel {len(panel)} genes; knockouts represented "
          f"{len(representable)}/{len(usable)}"
          + (f"; dropped {unrepresentable}" if unrepresentable else ""))

    args.data_dir.mkdir(parents=True, exist_ok=True)
    pert_data = PertData(str(args.data_dir))
    pert_data.new_data_process(dataset_name="papalexi_nt_only", adata=frame)
    pert_data.prepare_split(split="simulation", seed=args.seed)
    pert_data.get_dataloader(batch_size=args.batch_size, test_batch_size=args.batch_size)

    vocab = GeneVocab.from_file(args.model_dir / "vocab.json")
    for token in (PAD_TOKEN, "<cls>", "<eoc>"):
        if token not in vocab:
            vocab.append_token(token)
    genes = [str(gene) for gene in pert_data.adata.var["gene_name"]]
    in_vocab = np.array([gene in vocab for gene in genes])
    # Position of each KOT feature inside the scGPT panel; the panel is the KOT genes
    # present in the wide matrix plus the knockout genes, so most map directly.
    kot_to_panel = pd.Index(genes).get_indexer(pd.Index(kot_genes))
    kot_covered = kot_to_panel >= 0
    kot_safe = np.where(kot_covered, kot_to_panel, 0)
    vocab.set_default_index(vocab[PAD_TOKEN])
    # pred_perturb calls torch.from_numpy on gene_ids[raw_ids], so it needs the numpy
    # array; the training forward pass needs the same ids as a device tensor. Keep both
    # rather than converting inside either loop.
    gene_ids = np.array([vocab[gene] if gene in vocab else vocab[PAD_TOKEN]
                         for gene in genes], dtype=np.int64)
    gene_ids_tensor = torch.as_tensor(gene_ids, dtype=torch.long, device=device)

    model = build_generator(args.model_dir, vocab, device)
    matched, total = load_pretrained(model, args.model_dir / "best_model.pt")
    if matched < 0.5 * total:
        raise ValueError(
            f"Only {matched}/{total} tensors loaded from the scGPT checkpoint; this would "
            "be a randomly initialised transformer, not the foundation model item 31 asks for")
    print(f"[scgpt] loaded {matched}/{total} pretrained tensors")

    finetune(model, pert_data.dataloader["train_loader"], gene_ids_tensor,
             args.epochs, args.lr)
    torch.save(model.state_dict(), args.out_dir / "scgpt_perturb.pt")

    observed = sorted(usable & set(pert_data.adata.obs["condition"].astype(str)
                                   .str.replace(r"\+ctrl$", "", regex=True)))
    control_adata = pert_data.adata[pert_data.adata.obs["condition"] == "ctrl"]
    control = kot[groups == CONTROL_LABEL].mean(0)
    profiles = []
    coverage = [{"perturbation": target, "supported": False,
                 "reason": "knockout gene absent from the wide matrix"}
                for target in unrepresentable]
    model.eval()
    # Scale conversion from the scGPT panel's units to the checkpoint's Ms units, fitted
    # on NT cells only, exactly as the RegVelo and MultiPert runners do.
    panel_values = to_dense(pert_data.adata.X, dtype=np.float64)
    panel_conditions = pert_data.adata.obs["condition"].astype(str).to_numpy()
    panel_nt = panel_values[panel_conditions == "ctrl"][:, kot_safe]
    nt_mask = groups == CONTROL_LABEL
    rows = min(len(panel_nt), int(nt_mask.sum()))
    scales = positive_feature_scales(panel_nt[:rows], kot[nt_mask][:rows])
    panel_control = panel_values[panel_conditions == "ctrl"].mean(0)

    for target in observed:
        # create_cell_graph_dataset_for_prediction locates the knockout inside the name
        # list it is handed. Passing the measured panel would cover only 5 of 24
        # knockouts, since most CRISPR targets are not among the 1,276 HVGs. GEARS itself
        # passes its GO-graph node list here for exactly that reason, and this follows it.
        try:
            graphs = create_cell_graph_dataset_for_prediction(
                [target], control_adata, genes, device, num_samples=args.pool_size)
        except Exception as error:  # raised for a knockout the panel cannot represent
            coverage.append({"perturbation": target, "supported": False,
                             "reason": f"{type(error).__name__}: {str(error)[:80]}"})
            continue
        loader = DataLoader(graphs, batch_size=args.batch_size, shuffle=False)
        with torch.no_grad():
            batches = [model.pred_perturb(batch, include_zero_gene="all",
                                          gene_ids=gene_ids).detach().cpu().numpy()
                       for batch in loader]
        predicted = np.concatenate(batches, axis=0).astype(np.float64).mean(0)
        # No shift is claimed for a gene scGPT has no token for, nor for a KOT feature
        # absent from the panel.
        shift_panel = np.where(in_vocab, predicted - panel_control, 0.0)
        shift = np.where(kot_covered, shift_panel[kot_safe], 0.0)
        query, clipped = apply_mean_shift(kot[nt_mask], shift, scales)
        profile = query.mean(0)
        profiles.append(profile_row(target, profile))
        coverage.append({"perturbation": target, "supported": True, "reason": "ok",
                         "clipped_fraction": float(clipped.mean()),
                         "delta_norm": float(np.linalg.norm(profile - control))})

    manifest = {
        "model": "scGPT perturbation generation", "seed": args.seed,
        "epochs": args.epochs, "layer": args.layer,
        "pretrained": str((args.model_dir / "best_model.pt").resolve()),
        "pretrained_tensors_loaded": f"{matched}/{total}",
        "genes_in_vocab": int(in_vocab.sum()), "panel_genes": int(len(in_vocab)),
        "kot_features_covered": int(kot_covered.sum()),
        "kot_features_total": int(len(kot_covered)),
        "input_coordinates": "frozen KOT Ms",
        "supervision": "fine-tunes on observed KO cells; simulation split",
        "supported_perturbations": len(profiles), "requested_perturbations": len(observed),
    }
    write_isp_outputs(args.out_dir, profiles, coverage, manifest, "scGPT")
    print(f"[scgpt-isp] {len(profiles)}/{len(observed)} perturbations, "
          f"{in_vocab.sum()}/{len(in_vocab)} genes in vocab")
