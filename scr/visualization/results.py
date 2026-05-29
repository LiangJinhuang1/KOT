import pandas as pd
from pathlib import Path


def collect_results(cache_dir="cache/training", dataset="synthetic", model_names=None):
    cache_dir = Path(cache_dir)
    allowed_models = set(model_names) if model_names is not None else None
    rows = []
    for summary_path in sorted(cache_dir.glob(f"*/{dataset}/summary.txt")):
        model = summary_path.parts[-3]
        if allowed_models is not None and model not in allowed_models:
            continue
        row = {"model": model}
        for line in summary_path.read_text().splitlines():
            if ":" in line:
                key, val = line.split(":", 1)
                row[key.strip()] = val.strip()
        rows.append(row)
    return pd.DataFrame(rows)


def save_comparison_table(cache_dir="cache/training", dataset="synthetic",
                          save_dir="cache/results", model_names=None):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    df = collect_results(cache_dir, dataset, model_names=model_names)
    if df.empty:
        print("[results] No summary files found yet.")
        return df
    print("\n" + df[["model", "mean_foscttm"]].to_string(index=False))
    path = save_dir / f"comparison_{dataset}.csv"
    df.to_csv(path, index=False)
    print(f"Saved: {path}")
    return df
