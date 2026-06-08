"""
Stage 3d — KOT variants:
  kot          : full model, learnable β (no anchor)
  kot_anchor   : full model, β pinned to literature values via Gaussian prior
  kot_nodyn    : ablation, λ_dyn=0 (Sinkhorn-only baseline)
"""
from pathlib import Path
import yaml

from scr.training.runner import run_training
from scr.ultils.io import load_yaml

config_path = Path("config/training.yaml")

cfg = load_yaml(config_path)
cfg["defaults"]["models"] = ["kot", "kot_anchor", "kot_nodyn"]

tmp_path = Path("config/_tmp_kot.yaml")
with open(tmp_path, "w") as f:
    yaml.dump(cfg, f)
run_training(tmp_path)
tmp_path.unlink()
