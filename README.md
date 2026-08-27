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

The SLURM scripts run inside the existing Singularity image and preserve the
repository-root module command, so an editable install is optional on the
cluster:

```bash
sbatch --export=ALL,STAGE=clean,SCALE=mean,MODELS=kot slurm/train_stage.sh
```

Parallel sweeps use one runner argument string per line:

```bash
bash slurm/make_jobs.sh --out jobs.txt
sbatch --export=ALL,JOBS_FILE=jobs.txt,MAX_PARALLEL=16 slurm/parallel_train.sh
```

Submit from the repository root. The scripts use `SLURM_SUBMIT_DIR`, mount that
directory into the container, and keep caches under the project directory.

Before a long run, a quick container check is useful:

```bash
sbatch --export=ALL,RUN_CMD='python -m src.training.runner --help' slurm/train_slurm.sh
```

## Optional Backends

Model backends are loaded only when selected. For a non-Conda installation,
install the relevant extra, for example `pip install -e '.[kot]'` or
`pip install -e '.[velocity]'`. GLUE and uniPort still use the implementations
under `vendor/` when running from this research repository.
