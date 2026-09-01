# Submitting jobs

Three scripts, not one per tool. Everything else is a `RUN_CMD`.

| script | when |
|---|---|
| `train_slurm.sh` | one job, one command. GPU when the allocation has one, CPU when it does not. |
| `parallel_train.sh` | many independent runs at once, packed onto the allocation. |
| `submit_adaptive.sh` | a `jobs/*.txt` file submitted as a queue that adapts to free capacity. |

Job files live in `jobs/` (spent sweeps in `jobs/archive/`), not the repo root.

The rest of `slurm/` is only what cannot be a `RUN_CMD`: job arrays over external tools
(`starsolo_bmmc_array.sh`, `velocyto_slurm.sh`), the shared CUDA check
(`cuda_preflight.py`), the job-file generators (`make_jobs.sh`, `make_sweep_jobs.sh`,
`run_sweep_chain.sh`), and `probe_shuffle.sh`, which resolves the newest preprocessing
cache before it runs (a pinned path silently went stale for weeks).

## Ask for a GPU only when the tool can use one

`train_slurm.sh` detects the allocation: with a GPU it passes `--nv` and runs the CUDA
preflight, without one it skips both. So the partition flags decide, not the script.

Use `--cpus-per-task=N`, never `--cpus-per-gpu=N`: the two are mutually exclusive at submit
time and the script's own header already sets `--cpus-per-task`, so the per-gpu form is a
fatal submit error on both CPU and GPU jobs. (`parallel_train.sh` is the exception — its
header sets `--cpus-per-gpu`, so pass that one there.)

- **GPU** (`torch` does the work): KOT training, `evaluate_checkpoints.py`,
  `evaluate_protein_prediction.py` (φ forward + JVP), `papalexi_perturbation_benchmark.py`
  (the O(n²) kNN baselines).
- **CPU** (`--partition=jobs-cpu --gres=none`): scVelo, the numpy velocity-cosine, MaxFuse
  (numpy / scikit-learn / leiden throughout — it has no GPU path).

Do not use `jobs-cpu-long`: its 512G cap is shared group-wide and jobs sit in
`QOSGrpMemLimit`. Use `jobs-cpu` with `--time=08:00:00`; if a sweep will not fit, make it
resumable (the runner skips any (model, seed) that already wrote `summary.txt`, so
resubmitting with the same `--run-dir` continues it).

## Recipes

Each replaces a wrapper script that used to live here.

**KOT / baseline training** (GPU)

```bash
sbatch --job-name=kot --export=ALL,RUN_CMD='PYTHONPATH=. python -u -m src.training.runner \
  --datasets pbmc_retained --models kot --seeds 42' slurm/train_slurm.sh
```

**Staged synthetic training** (was `train_stage.sh`; GPU)

```bash
sbatch --job-name=kot_stage --export=ALL,RUN_CMD='PYTHONPATH=. python -u -m src.training.runner \
  --stage clean --scale mean --config config/training.yaml' slurm/train_slurm.sh
```

**Generate the staged synthetic data** (was `generate_all.sh`; CPU)

```bash
sbatch --partition=jobs-cpu --gres=none --mem=64GB --time=02:00:00 --job-name=gen_synth \
  --export=ALL,RUN_CMD='for s in oracle clean branch; do PYTHONPATH=. python -u -m src.data.synthetic_linked_ode --stage $s; done' \
  slurm/train_slurm.sh
```

**MaxFuse baseline** (was `train_maxfuse.sh`; CPU — no GPU path)

```bash
STAMP=$(date +%Y%m%d_%H%M%S)
sbatch --partition=jobs-cpu --gres=none --cpus-per-task=16 --mem=96GB --time=08:00:00 \
  --job-name=maxfuse_pbmc --export=ALL,RUN_CMD="PYTHONPATH=. MPLBACKEND=Agg python -u -m src.training.runner \
  --datasets pbmc_retained --models maxfuse --seeds 42,123,2026,6,9,11,17,21,33,77,88,101 \
  --run-dir cache/training/run_${STAMP}_maxfuse_pbmc" slurm/train_slurm.sh
```

**Checkpoint re-evaluation against the true inputs** (was `eval_realvel.sh`; GPU)

```bash
sbatch --job-name=eval_realvel --export=ALL,RUN_CMD='PYTHONPATH=. python -u tools/evaluate_checkpoints.py \
  --runs "cache/training/run_*_shuf*_*shufVel*" --velocity real \
  --output cache/results/realvel_summary.csv' slurm/train_slurm.sh
```

The velocity and the RNA→protein map are separate axes. The S-permutation control is the
same check on the other input: `--velocity as-trained --mapping real`, to
`cache/results/realmap_summary.csv`.

**Held-out protein-prediction quality** (was `protein_eval.sh`; GPU optional, memory-bound)

```bash
sbatch --mem=192GB --time=08:00:00 --job-name=protein_eval \
  --export=ALL,RUN_CMD='PYTHONPATH=. MPLBACKEND=Agg python -u tools/evaluate_protein_prediction.py \
  --runs "cache/training/run_20260829_022744_shuf_*_realVel" \
  --runs "cache/training/run_20260829_192024_shuf2_*_shufVel" \
  --runs "cache/training/run_20260831_222311_kinabl_*" \
  --runs "cache/training/run_20260830_225641_anch_*" \
  --velocity real --mapping real --out-prefix cache/results/protein_eval' slurm/train_slurm.sh
```

`--velocity/--mapping real`: the shuffle and S-permutation arms draw their input per seed,
so the true field and the true map are the only references all arms share. The anchor
ladder belongs in the same table because `anchored` is resolved per seed from each run's
own `beta_anchor_subset_n`.

**CRISPR perturbation benchmark** (was `papalexi_benchmark.sh`; GPU)

```bash
sbatch --job-name=papalexi_bench --mem=64GB --time=03:00:00 \
  --export=ALL,RUN_CMD='for cfg in cfgC cfgB; do PYTHONPATH=. MPLBACKEND=Agg python -u tools/papalexi_perturbation_benchmark.py \
  --predictions data/predictions_papalexi_$cfg --out-dir cache/results/papalexi_perturbation_$cfg; done' \
  slurm/train_slurm.sh
```

**scVelo velocity for BMMC** (was `velocity_bmmc_scvelo.sh`; CPU)

```bash
sbatch --partition=jobs-cpu --gres=none --cpus-per-task=16 --mem=128GB --time=08:00:00 \
  --job-name=vel_bmmc --export=ALL,RUN_CMD='PYTHONPATH=. python -u -m src.data.velocity \
  bmmc_cite_retained --dynamics-n-jobs 16' slurm/train_slurm.sh
```

**scVelo-vs-RegVelo cosine** (was `velocity_backend_cosine.sh`; CPU, memory only)

```bash
sbatch --partition=jobs-cpu --gres=none --mem=128GB --time=02:00:00 --job-name=vel_backend_cos \
  --export=ALL,RUN_CMD='PYTHONPATH=. python -u tools/velocity_checks.py backend-cosine \
  --pairs bmmc_cite_retained:bmmc_cite_regvelo --output cache/results/velocity_backend_cosine_bmmc.csv' \
  slurm/train_slurm.sh
```

## Running many at once

Independent runs go in a jobs file, one runner arg-string per line, and
`parallel_train.sh` packs them onto the allocation (`MAX_PARALLEL` is the total, not per
GPU). Give every line its own `--run-dir`, or parallel jobs silently overwrite each other.

```bash
sbatch --export=ALL,JOBS_FILE=jobs/jobs_kinetic_ablations.txt,MAX_PARALLEL=16 slurm/parallel_train.sh
```

It honours `SKIP_CUDA_PREFLIGHT=1`, so it also packs CPU-only work — e.g. MaxFuse seeds,
which are independent and would otherwise run one after another:

```bash
sbatch --partition=jobs-cpu --gres=none --cpus-per-task=48 --mem=192GB --time=08:00:00 \
  --export=ALL,SKIP_CUDA_PREFLIGHT=1,JOBS_FILE=jobs/jobs_maxfuse.txt,MAX_PARALLEL=6 \
  slurm/parallel_train.sh
```
