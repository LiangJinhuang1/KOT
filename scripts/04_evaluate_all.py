"""
Stage 4 — Evaluation
Pure reader: loads all prediction h5ads from data/predictions/ and
prints a comparison table of mean FOSCTTM per model × dataset.
Run after all 03* scripts have completed.
"""
import pandas as pd
import scanpy as sc
from pathlib import Path

PREDICTIONS_DIR = Path("data/predictions")


def load_results() -> pd.DataFrame:
    rows = []
    for path in sorted(PREDICTIONS_DIR.glob("*.h5ad")):
        adata = sc.read_h5ad(path)
        rows.append({
            "model":        adata.uns.get("model", path.stem),
            "dataset":      adata.uns.get("dataset", "?"),
            "mean_foscttm": adata.uns.get("mean_foscttm", float("nan")),
            "n_cells":      adata.n_obs,
        })
    return pd.DataFrame(rows)


def print_table(df: pd.DataFrame) -> None:
    if df.empty:
        print("No predictions found in data/predictions/")
        return
    table = df.pivot(index="model", columns="dataset", values="mean_foscttm")
    table = table.round(4)
    print("\nFOSCTTM comparison (lower = better)\n")
    print(table.to_string())
    table.to_csv(PREDICTIONS_DIR / "foscttm_summary.csv")
    print(f"\nSaved: {PREDICTIONS_DIR}/foscttm_summary.csv")


if __name__ == "__main__":
    results = load_results()
    print_table(results)
