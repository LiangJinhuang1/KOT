import pandas as pd
from pathlib import Path


def model_set(value):
    if value is None:
        return None
    if isinstance(value, str):
        return {value}
    return {str(item) for item in value}


def collect_results(cache_dir="cache/training", dataset="synthetic", model_names=None, seeded_models=None):
    cache_dir = Path(cache_dir)
    allowed_models = model_set(model_names)
    seeded_model_set = model_set(seeded_models)
    rows = []
    if not cache_dir.exists():
        return pd.DataFrame(rows)
    summary_paths = []
    for model_dir in sorted(path for path in cache_dir.iterdir() if path.is_dir()):
        dataset_dir = model_dir / dataset
        if not dataset_dir.exists():
            continue
        seed_paths = sorted(dataset_dir.glob("seed_*/summary.txt"))
        use_seed_paths = bool(seed_paths) and (
            seeded_model_set is None or model_dir.name in seeded_model_set
        )
        if use_seed_paths:
            summary_paths.extend(seed_paths)
            continue
        base_summary = dataset_dir / "summary.txt"
        if base_summary.exists():
            summary_paths.append(base_summary)
    for summary_path in sorted(set(summary_paths)):
        rel_parts = summary_path.relative_to(cache_dir).parts
        model = rel_parts[0]
        if allowed_models is not None and model not in allowed_models:
            continue
        row = {"model": model}
        for line in summary_path.read_text().splitlines():
            if ":" in line:
                key, val = line.split(":", 1)
                row[key.strip()] = val.strip()
        if "seed" not in row and len(rel_parts) >= 4 and rel_parts[2].startswith("seed_"):
            row["seed"] = rel_parts[2].replace("seed_", "", 1)
        rows.append(row)
    return pd.DataFrame(rows)


def save_comparison_table(cache_dir="cache/training", dataset="synthetic",
                          save_dir="cache/results", model_names=None, seeded_models=None):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    df = collect_results(
        cache_dir,
        dataset,
        model_names=model_names,
        seeded_models=seeded_models,
    )
    if df.empty:
        print("[results] No summary files found yet.")
        return df
    print("\n" + df[["model", "mean_foscttm"]].to_string(index=False))
    path = save_dir / f"comparison_{dataset}.csv"
    df.to_csv(path, index=False)
    print(f"Saved: {path}")
    if "seed" in df.columns:
        aggregate = df.copy()
        aggregate["mean_foscttm"] = pd.to_numeric(aggregate["mean_foscttm"], errors="coerce")
        aggregate = (
            aggregate
            .groupby("model")["mean_foscttm"]
            .agg(["mean", "std", "min", "max", "count"])
            .reset_index()
        )
        aggregate_path = save_dir / f"comparison_{dataset}_seed_summary.csv"
        aggregate.to_csv(aggregate_path, index=False)
        print("\nSeed summary:")
        print(aggregate.to_string(index=False))
        print(f"Saved: {aggregate_path}")
    return df
