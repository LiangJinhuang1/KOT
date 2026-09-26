# KOT — Kinetic induced Optimal Transport

Thesis research code. KOT learns a map `phi` between two single-cell modalities and
adds a kinetic constraint: `phi` must carry source-modality velocity onto the target
modality's rate law.

```
RNA  --phi-->  protein    (main result)
ATAC --phi-->  RNA        (exploratory; not in the paper)
```

## Install

Python ≥ 3.11.

```bash
pip install -e '.[kot,velocity]'   # training
pip install -e '.[baselines]'      # moscot, scGLUE, scvi-tools, MaxFuse
pip install -e '.[dev]'            # pytest, ruff
```

## Run

```bash
python -m src.data.velocity bmmc_cite                                  # 1. velocity
python -m src.training.runner --models kot --stage clean --scale mean  # 2. train
```

Give every run its own `--run-dir`; runs sharing one overwrite each other.

## Layout

```
src/       library: data, models, losses, training, evaluation, visualization
config/    training, dataset and velocity settings
tools/     analysis and figure scripts
figures/   generated panels
```

Datasets, caches, checkpoints and cluster scripts are not in this repository.

See `config/README.md` for training settings and the `--models` groups.
