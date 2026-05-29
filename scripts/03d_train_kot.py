"""
Stage 3d — KOT: full model and λ_dyn=0 ablation
Full KOT enforces the biophysical ODE constraint.
Ablation (λ_dyn=0) isolates the contribution of the kinetic loss.
Saves predictions to data/predictions/{kot,kot_nodyn}_{dataset}.h5ad
"""
from scr.training.runner import run_training
from scr.ultils.io import load_yaml
from pathlib import Path
import yaml

config_path = Path("config/training.yaml")

# Full KOT
cfg = load_yaml(config_path)
cfg["defaults"]["model"] = "kot"
cfg["defaults"]["lambda_dyn"] = 1.0

tmp_path = Path("config/_tmp_kot.yaml")
with open(tmp_path, "w") as f:
    yaml.dump(cfg, f)
run_training(tmp_path)
tmp_path.unlink()

# Ablation: kinetic loss disabled (λ_dyn=0)
cfg["defaults"]["model"] = "kot_nodyn"
cfg["defaults"]["lambda_dyn"] = 0.0

tmp_path = Path("config/_tmp_kot_nodyn.yaml")
with open(tmp_path, "w") as f:
    yaml.dump(cfg, f)
run_training(tmp_path)
tmp_path.unlink()
