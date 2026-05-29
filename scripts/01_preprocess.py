"""
Stage 1 — Preprocessing
Runs load_and_preprocess_cached for every dataset in config/datasets.yaml.
Results are written to cache/preprocessed/ and reused by all later stages.
"""
from scr.ultils.io import load_yaml
from scr.data.preprocessing import load_and_preprocess_cached

datasets = load_yaml("config/datasets.yaml").get("datasets", {})

for name, cfg in datasets.items():
    print(f"\n[01_preprocess] {name}")
    load_and_preprocess_cached(
        rna_path=cfg["rna_path"],
        protein_path=cfg.get("protein_path"),
        protein_label=cfg.get("protein_label", "ADT"),
        atac_path=cfg.get("atac_path"),
        atac_label=cfg.get("atac_label", "ATAC"),
    )
