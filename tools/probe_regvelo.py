#!/usr/bin/env python3
"""
Probe the RegVelo / scvi-tools / decoupler install and API surface.

Run it through the SAME sbatch harness training uses, so the CUDA/CUPTI
environment (GPU node + --nv) matches what actually works:

  sbatch --export=ALL,RUN_CMD='PYTHONPATH=. python tools/probe_regvelo.py' slurm/train_slurm.sh

Then read slurm-<jobid>.out. The printed method lists tell us the exact names to
wire into src/data/regvelo_backend.py (set_prior_grn / preprocess / velocity output).
"""

import decoupler as dc
import regvelo
import scvi
import torch
from regvelo import REGVELOVI

print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
print("scvi", scvi.__version__, "| regvelo", regvelo.__version__, "| decoupler", dc.__version__)
print("REGVELOVI public:", sorted(m for m in dir(REGVELOVI) if not m.startswith("_")))
print("regvelo public:", sorted(f for f in dir(regvelo) if not f.startswith("_")))
print("regvelo.pp:", sorted(f for f in dir(regvelo.pp) if not f.startswith("_")))
print("regvelo.tl:", sorted(f for f in dir(regvelo.tl) if not f.startswith("_")))
