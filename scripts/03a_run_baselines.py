"""
Stage 3a — OT Baselines: SCOT and MOSCOT
Saves predictions to data/predictions/{scot,moscot}_{dataset}.h5ad
"""
from scr.training.runner import run_training
from scr.ultils.io import load_yaml
from pathlib import Path
import yaml

config_path = Path("config/training.yaml")
cfg = load_yaml(config_path)

for model in ["scot", "moscot"]:
    print(f"\n{'=' * 60}")
    print(f"Running baseline: {model}")
    print(f"{'=' * 60}")
    cfg["defaults"]["model"] = model
    tmp_path = Path(f"config/_tmp_{model}.yaml")
    with open(tmp_path, "w") as f:
        yaml.dump(cfg, f)
    run_training(tmp_path)
    tmp_path.unlink()
