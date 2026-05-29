import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path

from scr.metrics.FOSCTTM import calc_domainAveraged_FOSCTTM

PREDICTIONS_DIR = Path("data/predictions")


def save_prediction_h5ad(
    aligned: list,
    foscttm_scores: list,
    mean_foscttm: float,
    dataset_name: str,
    model_name: str,
    second_label: str,
    obs_names,
) -> None:
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    adata = ad.AnnData(X=aligned[0])
    adata.obs_names = list(obs_names)
    adata.obsm["X_aligned_rna"] = aligned[0].astype("float32")
    adata.obsm[f"X_aligned_{second_label}"] = aligned[1].astype("float32")
    adata.obs["foscttm"] = foscttm_scores
    adata.uns["mean_foscttm"] = mean_foscttm
    adata.uns["model"] = model_name
    adata.uns["dataset"] = dataset_name
    adata.uns["second_label"] = second_label
    path = PREDICTIONS_DIR / f"{model_name}_{dataset_name}.h5ad"
    adata.write_h5ad(path)
    print(f"Saved prediction: {path}")


def evaluate_and_save(
    aligned: list,
    name: str,
    output_dir: Path,
    second_label: str,
    model_name: str = None,
    obs_names=None,
) -> float:
    foscttm_scores = calc_domainAveraged_FOSCTTM(aligned[0], aligned[1])
    mean_foscttm = float(np.mean(foscttm_scores))
    print(f"[{name}] Mean FOSCTTM: {mean_foscttm:.4f}  (lower = better alignment)")

    pd.DataFrame({"foscttm": foscttm_scores}).to_csv(output_dir / "foscttm.csv", index=False)
    np.save(output_dir / "aligned_rna.npy", aligned[0])
    np.save(output_dir / f"aligned_{second_label}.npy", aligned[1])

    if model_name is not None and obs_names is not None:
        save_prediction_h5ad(
            aligned, foscttm_scores, mean_foscttm,
            name, model_name, second_label, obs_names,
        )

    return mean_foscttm
