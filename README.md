# kot

Research code for kinetic optimal transport (KOT) and multimodal single-cell
alignment. The repository can be used directly on the cluster or installed as
an editable Python distribution.

## Environment

The reproducible environment targets Python 3.11 and a CUDA 12.8 PyTorch build:

```bash
conda env create -f environment.yml
conda activate kot
python -m pip install --no-deps -e .
```

`--no-deps` is intentional after creating the Conda environment. It prevents
pip from replacing the cluster-compatible CUDA build of PyTorch.

Run training through the installed command:

```bash
kot --datasets pbmc_retained --models kot_main
```

The package also exposes the programmatic runner:

```python
import kot

kot.run_training(models="kot_main", datasets_filter="pbmc_retained")
```

The existing module invocation remains supported:

```bash
python -m src.training.runner --datasets pbmc_retained --models kot_main
```

Configuration files are project inputs rather than package data. Run commands
from the repository root, or pass `--config` and `--datasets-config` explicitly.

`config/training.yaml` is the single training config. Model groups, seeds,
staged synthetic runs, and the validation-split knobs are documented in
`config/README.md`.

## Cluster

SLURM wrappers stay on the HPC checkout under `slurm/` and are not part of this
GitHub tree. Training is the same module command from the repository root,
inside the project container; caches stay under the project directory:

```bash
python -m src.training.runner --datasets pbmc_retained --models kot_main
python -m src.training.runner --help
```

## Optional Backends

Model backends are loaded only when selected. For a non-Conda installation,
install the relevant extra, for example `pip install -e '.[kot]'` or
`pip install -e '.[velocity]'`. GLUE and uniPort still use the implementations
under `vendor/` when running from this research repository.
