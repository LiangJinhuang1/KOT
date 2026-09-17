# KOT — kinetic optimal transport for single-cell multimodal alignment

Research code for the thesis. KOT learns a map `phi` between two single-cell
modalities and constrains it with a *kinetic* law: the Jacobian of `phi` pushed along a
velocity in the source modality has to reproduce the target modality's own rate law.
The claim under test is that this kinetic term buys alignment that a purely
distributional (optimal-transport) objective does not.

```
    RNA  --phi-->  protein        J_phi(x) v_x  ~  the protein turnover law
    ATAC --phi-->  RNA            J_phi(c) v_c  ~  the transcription law
```

Package name is `kot` (`pyproject.toml`); the implementation lives in `src/`.

## Three experiment tracks

They share `src/` — the `phi`/`kappa`/`alpha` networks, the Sinkhorn divergence, the
JVP, the optimiser and the LR schedule — and differ in the modality pair and the
evaluation.

| track | entry point | what it answers |
|---|---|---|
| **RNA → protein** | `python -m src.training.runner` | the main result: staged synthetic recovery, then real CITE-seq (BMMC, PBMC) scored by FOSCTTM, lineage and held-out protein prediction |
| **CRISPR / Papalexi** | `run_kot_crispr.py`, `run_crispr_isp.py` | protein *prediction* under knockout, against protein predictors rather than alignment baselines |
| **chromatin → RNA** | `run_kot_chromatin.py` | whether the principle survives changing both the source modality and the target kinetic law |

The chromatin track is **excluded from the paper** (`Litterature/ICLR_Tables/REVIEW_NOTES.md:8`).
Its tooling is kept for provenance in `archive/chromatin_tools/`, its job files in
`jobs/archive/chromatin/`.

`run_kot_crispr.py` is the frozen CRISPR experiment — evaluation is written before any
model is scored. `tools/papalexi.py` is the exploratory counterpart; do not mix them.

## Install

Python ≥ 3.11. The base install is analysis-only; every heavy stack is an extra.

```bash
pip install -e '.[kot,velocity]'          # training + scVelo
pip install -e '.[baselines]'             # moscot, scGLUE, scvi-tools, MaxFuse
pip install -e '.[protein_baselines]'     # sciPENN, scButterfly (CRISPR competitors)
pip install -e '.[regvelo]'               # RegVelo velocity backend
pip install -e '.[dev]'                   # pytest, ruff, build
```

`environment.yml` pins the conda side. The login shell often lacks these; on the
cluster use the container that `slurm/train_slurm.sh` names.

## Running

```bash
# 1. velocity for a dataset key in config/velocity.yaml (--list shows them)
python -m src.data.velocity bmmc_cite

# 2. staged synthetic training (see config/README.md for --models groups)
python -m src.training.runner --models kot --stage clean --scale mean

# 3. on the cluster — one job, one command
sbatch --export=ALL,RUN_CMD='python -u -m src.training.runner --models kot --stage clean --scale mean' \
       slurm/train_slurm.sh
```

Substantial training and memory-heavy evaluation belong on SLURM, not the login node.
Every independent run needs its own `--run-dir`; parallel jobs sharing one silently
overwrite each other.

## Layout

```
src/            the library — everything the three tracks share
  data/         loading, preprocessing, splits, velocity, synthetic generators
  models/       phi / kappa / alpha networks, LinearODE
  losses/       Sinkhorn divergence, JVP physics term
  training/     runner + one module per method (kot, glue, moscot, uniport, totalvi, …)
  evaluation/   FOSCTTM, protocol gates, trajectory DTW, in-silico perturbation
  visualization/ panel code — one module per figure, shared palette in style.py
  adapters/     external-tool shims
config/         training.yaml, datasets.yaml, velocity.yaml, preprocessing.yaml, β/γ anchors
slurm/          the three submit scripts; everything else is a RUN_CMD
jobs/           one runner arg-string per line, headers record experimental intent
tools/          analysis and figure generation (panel code stays in src/visualization/)
tests/          pytest; `pytest` runs them all
figures/        generated panels — cite via figures/THESIS_FIGURES.md, not by filename
figure_data/    the numbers behind revised panels, with their own READMEs
archive/        superseded scripts kept for provenance, not for reuse
```

Untracked and local-only (see `.gitignore`): `Datasets/` (raw, 1.6 TB), `cache/`
(preprocessed, velocity, training runs, 279 GB), `data/` (predictions), `logs/`,
`vendor/` (competitor source), `Litterature/`.

`main.py` is a scratch driver with most of its dataset list commented out — prefer the
module entry points above.

## Cache

| path | holds |
|---|---|
| `cache/preprocessed/` | hashed preprocessing outputs; the hash is the cache key, so a config change makes a new file rather than overwriting |
| `cache/velocity/` | per-dataset velocity results, one directory per backend |
| `cache/training/` | one directory per run: checkpoints, diagnostics, per-cell FOSCTTM, aligned arrays |
| `cache/results/` | distilled tables |

## Documentation

Read the one that matches the task; none of them repeat another.

- **`config/README.md`** — `training.yaml`, the `--models` groups, seeds, staged synthetic runs
- **`slurm/README.md`** — which submit script, GPU vs CPU, and the recipes
- **`jobs/README.md`** — what each job file was for and what is archived
- **`figures/THESIS_FIGURES.md`** — the citation map. **Read this before quoting any
  figure**: it records, per panel, what the figure does *not* support, including several
  published numbers that have since been withdrawn
- **`AGENTS.md`** — the standing rules for changes to this repo

## Conventions

Cell and gene ordering, state/velocity units, and train/validation/test separation are
load-bearing — preserve them. Fit preprocessing on the permitted training population
only, and never train an unpaired method using held-out pairing. Numerical success is
not biological validity: compare controls on compatible cells, genes and metrics, and
do not weaken a gate to make a run pass.

`ruff` at line-length 100, `vendor/` and `archive/` excluded.
