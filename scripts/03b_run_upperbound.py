"""
Stage 3b — Upper Bound: totalVI (trained on true paired data)
Establishes the theoretical performance ceiling.
Saves predictions to data/predictions/totalvi_{dataset}.h5ad
"""
from scr.training.runner import run_training
from scr.ultils.io import load_yaml
from pathlib import Path
import yaml

config_path = Path("config/training.yaml")
cfg = load_yaml(config_path)
cfg["defaults"]["model"] = "totalvi"

tmp_path = Path("config/_tmp_totalvi.yaml")
with open(tmp_path, "w") as f:
    yaml.dump(cfg, f)

run_training(tmp_path)
tmp_path.unlink()
