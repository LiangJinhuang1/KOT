"""Ridge and MLP RNA→protein regression: the supervised bar for the CRISPR benchmark.

Both read exactly the matrices KOT's φ reads (kot_rna_layer → kot_protein_layer), so a
gap against them is a statement about the map, not about the features. Ridge because a
691-gene design matrix against ~2.3k control cells is short and wide; the MLP because a
linear map is not by itself evidence that the relationship is linear.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from torch import nn

from src.training.protein_supervised import adt_targets, fit_rows_of, rna_features


RIDGE_ALPHAS = np.logspace(-2, 6, 33)


def standardize(features: np.ndarray, fit_rows: np.ndarray) -> np.ndarray:
    """Z-score every gene on the FITTED cells only.

    Using all cells would let the knockouts set the scale of the features their own
    prediction is read off, which is a held-out statistic even though it is RNA-side.

    A gene that is CONSTANT across the fitted cells is divided by 1, not by a small
    floor. On Papalexi 208 of 1276 genes have exactly zero variance across the 2326 NT
    cells while being nonzero on knockouts, so a 1e-8 floor sent those KO features to
    ~3e7. Ridge is immune (a column that is zero on every fitted row gets a zero
    coefficient), but the MLP's first-layer weights on those inputs never receive a
    gradient, so they keep their initialisation and multiply it -- which is what made the
    MLP baseline predict ADT values in the thousands and score below zero-change. The
    columns carry no fitted information either way; scaling them by 1 keeps them
    bounded instead of letting them dominate.
    """
    mean = features[fit_rows].mean(axis=0, keepdims=True)
    sd = features[fit_rows].std(axis=0, keepdims=True)
    return (features - mean) / np.where(sd > 0, sd, 1.0)


def torch_device(cfg: dict) -> torch.device:
    requested = str(cfg.get("device", "auto"))
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def validation_split(fit_rows: np.ndarray, fraction: float,
                     seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split positions WITHIN fit_rows, so the caller can index the paired targets too."""
    order = np.random.default_rng(seed).permutation(len(fit_rows))
    n_val = max(1, int(round(fraction * len(fit_rows))))
    return order[n_val:], order[:n_val]


def build_mlp(n_features: int, n_adt: int, hidden_dims: list[int], dropout: float) -> nn.Sequential:
    layers = []
    width = n_features
    for hidden in hidden_dims:
        layers += [nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(dropout)]
        width = hidden
    layers.append(nn.Linear(width, n_adt))
    return nn.Sequential(*layers)


def train_mlp(model: nn.Module, x_train: torch.Tensor, y_train: torch.Tensor,
              x_val: torch.Tensor, y_val: torch.Tensor, cfg: dict) -> tuple[int, float]:
    """Adam + MSE with early stopping on the held-in validation split.

    Returns the epoch of the best validation loss and that loss, so a run that stopped
    at epoch 1 is visible in the diagnostics rather than looking like a trained model.
    """
    n_epochs = int(cfg.get("mlp_epochs", 300))
    batch_size = int(cfg.get("mlp_batch_size", 256))
    patience = int(cfg.get("mlp_patience", 30))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg.get("mlp_lr", 1e-3)),
        weight_decay=float(cfg.get("mlp_weight_decay", 1e-4)),
    )
    loss_fn = nn.MSELoss()
    generator = torch.Generator(device="cpu").manual_seed(int(cfg.get("seed", 42)))

    best_loss, best_epoch = float("inf"), 0
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    for epoch in range(1, n_epochs + 1):
        model.train()
        order = torch.randperm(x_train.shape[0], generator=generator).to(x_train.device)
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            optimizer.zero_grad()
            loss_fn(model(x_train[batch]), y_train[batch]).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(x_val), y_val))
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            break
    model.load_state_dict(best_state)
    return best_epoch, best_loss


def prediction_diagnostics(model_name: str, predicted: np.ndarray, observed: np.ndarray,
                           fit_rows: np.ndarray) -> dict:
    """Per-protein Pearson on the fitted cells: a floor check before the benchmark runs."""
    fitted = predicted[fit_rows]
    correlations = [
        float(np.corrcoef(fitted[:, j], observed[:, j])[0, 1])
        for j in range(observed.shape[1])
    ]
    print(f"[{model_name}] in-fit Pearson per ADT: {np.round(correlations, 3).tolist()}")
    return {
        f"{model_name}_fit_pearson_mean": float(np.mean(correlations)),
        f"{model_name}_fit_pearson_min": float(np.min(correlations)),
        "n_fit_cells": int(len(fit_rows)),
    }


def run_ridge(context: dict, cfg: dict) -> tuple[list, None, dict]:
    """RidgeCV on the paired fitted cells, applied to every cell's RNA."""
    features = rna_features(context, cfg)
    targets = adt_targets(context, cfg)
    fit_rows = fit_rows_of(context, features.shape[0])
    scaled = standardize(features, fit_rows)

    model = RidgeCV(alphas=RIDGE_ALPHAS)
    model.fit(scaled[fit_rows], targets)
    predicted = model.predict(scaled).astype(np.float32)
    print(f"[ridge] {len(fit_rows)} paired cells x {features.shape[1]} genes → "
          f"{targets.shape[1]} ADTs, alpha={model.alpha_:.4g}")

    diagnostics = {"ridge_alpha": float(model.alpha_),
                   **prediction_diagnostics("ridge", predicted, targets, fit_rows)}
    return [predicted, targets.astype(np.float32)], None, diagnostics


def run_mlp(context: dict, cfg: dict) -> tuple[list, None, dict]:
    """Two-layer MLP on the paired fitted cells, applied to every cell's RNA."""
    features = rna_features(context, cfg)
    targets = adt_targets(context, cfg)
    fit_rows = fit_rows_of(context, features.shape[0])
    scaled = standardize(features, fit_rows)

    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    device = torch_device(cfg)
    hidden_dims = [int(dim) for dim in cfg.get("mlp_hidden_dims", [256, 128])]

    train_pos, val_pos = validation_split(fit_rows, float(cfg.get("mlp_val_fraction", 0.1)), seed)
    x_fit = torch.as_tensor(scaled[fit_rows], dtype=torch.float32, device=device)
    y_fit = torch.as_tensor(targets, dtype=torch.float32, device=device)

    model = build_mlp(scaled.shape[1], targets.shape[1], hidden_dims,
                      float(cfg.get("mlp_dropout", 0.1))).to(device)
    best_epoch, best_loss = train_mlp(model, x_fit[train_pos], y_fit[train_pos],
                                      x_fit[val_pos], y_fit[val_pos], cfg)
    print(f"[mlp] hidden={hidden_dims}, best val MSE {best_loss:.4f} at epoch {best_epoch}")

    model.eval()
    with torch.no_grad():
        predicted = model(
            torch.as_tensor(scaled, dtype=torch.float32, device=device)
        ).cpu().numpy().astype(np.float32)

    diagnostics = {"mlp_best_epoch": int(best_epoch), "mlp_best_val_mse": float(best_loss),
                   **prediction_diagnostics("mlp", predicted, targets, fit_rows)}
    return [predicted, targets.astype(np.float32)], None, diagnostics
