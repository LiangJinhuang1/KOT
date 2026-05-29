"""
Stage 3c — Convex Baseline: Linear ODE (ours)
Solves ṗ = Ar - B·p via alternating least squares.
Justifies the need for non-linear deep networks.
Saves predictions to data/predictions/linear_ode_{dataset}.h5ad
"""
from scr.training.runner import run_training
from scr.ultils.io import load_yaml
from pathlib import Path
import yaml

config_path = Path("config/training.yaml")
cfg = load_yaml(config_path)
cfg["defaults"]["model"] = "linear_ode"

tmp_path = Path("config/_tmp_linear_ode.yaml")
with open(tmp_path, "w") as f:
    yaml.dump(cfg, f)

run_training(tmp_path)
tmp_path.unlink()
