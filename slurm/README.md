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

**CRISPR competitor set** (supervised RNA→protein translators; GPU)

The CRISPR benchmark's competitors are not the diagonal-integration baselines used for
BMMC/PBMC. They are supervised translators fitted on the NT controls' PAIRED
(RNA, protein), plus totalVI as the paired ceiling. `--models crispr` runs the set.

```bash
sbatch --job-name=papalexi_crispr --mem=64GB --time=04:00:00 \
  --export=ALL,RUN_CMD='PYTHONPATH=. MPLBACKEND=Agg python -u main.py \
  --datasets papalexi_retained,papalexi_regvelo --models crispr \
  --set save_prediction_h5ad=true' slurm/train_slurm.sh
```

sciPENN and scButterfly need `pip install -e '.[protein_baselines]'`. They OOM on the
login node; run them through slurm.

**MultiPert on our pooled Papalexi** (GPU)

MultiPert maps (control cell, GEARS perturbation embedding) → perturbed profile, so it
CANNOT run under the NT-only restriction: with controls alone it has no perturbation to
learn. It runs in its own published regime (per-perturbation 0.6/0.2/0.2 CELL split, so
it trains on cells of every knockout it is scored on) and is reported as a NOT-RANKED
`paired_all_cells` reference beside totalVI. Its paper used the ARRAYED Papalexi
experiment; this runs it on our POOLED screen so the numbers sit on our data.

```bash
# once: write our data in MultiPert's convention (CPU, streams the 1.4GB RNA matrix)
sbatch --partition=jobs-cpu --gres=none --mem=96GB --time=01:30:00 \
  --job-name=multipert_inputs --export=ALL,RUN_CMD='PYTHONPATH=. python -u \
  tools/multipert.py inputs' slurm/train_slurm.sh
cp vendor/multipert/pert_embeddings.npz cache/multipert/data/

# per seed — a SEPARATE --out-dir each time, or main() reloads the previous checkpoint
sbatch --job-name=multipert_run --mem=96GB --time=03:00:00 \
  --export=ALL,RUN_CMD='PYTHONPATH=. python -u tools/multipert.py run --seed 1' \
  slurm/train_slurm.sh

# per-(knockout, protein) deltas, calibrated onto benchmark ADT units on control cells
sbatch --partition=jobs-cpu --gres=none --mem=48GB --time=00:30:00 \
  --job-name=multipert_deltas --export=ALL,RUN_CMD='PYTHONPATH=. python -u \
  tools/multipert.py deltas --run-dir cache/multipert/output/seed_1' slurm/train_slurm.sh
```

4 of our 25 knockouts have no GEARS embedding and fall back to one-hot (upstream's own
behaviour); `cache/multipert/data/pert_embedding_coverage.csv` lists them. CUL3 is among
them and is one of the two post-transcriptional pairs the benchmark exists to test.

**CRISPR perturbation benchmark** (was `papalexi_benchmark.sh`; GPU)

`--scoring-set sufficiency` (the default) scores every non-self pair whose knockout
clears the frozen design thresholds. The older significance-filtered set is a SECONDARY
analysis: selecting pairs on their own q-values defines the benchmark by its answers and
drops exactly the small-effect pairs a cross-modal model has to prove itself on.

`--perturbation-subset` adds a second ranking over one competitor's published knockout
set, re-scored from the same predictions, so the two tables differ only in which pairs
are scored. It does NOT make the training conditions comparable; every table carries a
`supervision` column for that.

```bash
sbatch --job-name=papalexi_bench --mem=64GB --time=03:00:00 \
  --export=ALL,RUN_CMD='for cfg in cfgC cfgB; do PYTHONPATH=. MPLBACKEND=Agg python -u tools/papalexi.py benchmark \
  --predictions data/predictions_papalexi_$cfg --out-dir cache/results/papalexi_perturbation_$cfg \
  --perturbation-subset config/papalexi_multipert_subset.csv \
  --external-predictions cache/results/multipert_pair_deltas.csv; done' \
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

**Chromatin→RNA (`run_kot_chromatin.py`)** — the second modality pair. Stages run in this
order; each writes into `cache/chromatin/` and the next one refuses to start without it.

```bash
# 1. build the paired object (CPU). hspc streams two 10x ARC matrices and does an LSI
#    over ~197k union peaks, so give it memory; bmmc reuses the shipped gene activity.
sbatch --partition=jobs-cpu --gres=none --mem=192GB --time=06:00:00 \
  --export=ALL,RUN_CMD='PYTHONPATH=. python -u run_kot_chromatin.py prepare --dataset hspc' \
  slurm/train_slurm.sh

# 2. audit, split, chromatin velocity (CPU, minutes)
PYTHONPATH=. python -u run_kot_chromatin.py audit    --dataset bmmc --law reduced
PYTHONPATH=. python -u run_kot_chromatin.py split    --dataset bmmc --seed 0
PYTHONPATH=. python -u run_kot_chromatin.py velocity --dataset bmmc

# 3. the Task D reference: scVelo on the panel's own spliced/unspliced (CPU, hours)
sbatch --partition=jobs-cpu --gres=none --mem=96GB --time=08:00:00 \
  --export=ALL,RUN_CMD='PYTHONPATH=. python -u run_kot_chromatin.py reference --dataset hspc' \
  slurm/train_slurm.sh

# 4. train one condition (GPU)
sbatch --job-name=chrom_r1 --export=ALL,RUN_CMD='PYTHONPATH=. python -u run_kot_chromatin.py \
  train --dataset bmmc --law reduced --condition full --seed 0' slurm/train_slurm.sh

# 5. score it: Tasks A-D, biological AND internal track (GPU)
sbatch --job-name=chrom_eval --export=ALL,RUN_CMD='PYTHONPATH=. python -u run_kot_chromatin.py \
  evaluate --run-dir cache/chromatin/runs/<run>' slurm/train_slurm.sh
```

Do not put chromatin lines in `parallel_train.sh`: that launcher hardcodes
`python -m src.training.runner`. The phase submitter uses one `train_slurm.sh` job per
condition, evaluates it, and refuses every later phase until all four pilot directories
contain `preflight_passed.json`:

```bash
# exactly four jobs: HSPC/BMMC × R1 full/noDyn, one seed
bash slurm/submit_chromatin_phase.sh pilot

# only after all four pass: eight remaining R1 jobs, then six R2 jobs
bash slurm/submit_chromatin_phase.sh r1-ablation
bash slurm/submit_chromatin_phase.sh relay
```

Together those phases are the exact 18 one-seed conditions: per dataset, R1 has
full/shuffle/reverse/zero/noDyn/permG and R2 has full/shuffle/noDyn. Development and final
runs are separate explicit phases:

```bash
# 18 conditions × 3 frozen development seeds
bash slurm/submit_chromatin_phase.sh development

# 12 seeds: six central conditions per dataset (R1/R2 full, shuffle, noDyn)
bash slurm/submit_chromatin_phase.sh final-central

# optional frozen run of all 18 conditions × 12 seeds
bash slurm/submit_chromatin_phase.sh final-all
```

The submitter passes `--lambda-dyn 1` explicitly; it never inherits the RNA→protein
default of 1000, which collapses this 2,000-gene chromatin map.

**BMMC spliced/unspliced over every batch** (CPU, long). The cached
`bmmc_multiome_scvelo_results.h5ad` covers s1d1 and s2d4 only — 10,780 of 69,249 cells,
451 of the 1,281 panel genes usable — because it was built from a two-row STARsolo report.
The velocyto output for all 13 batches is on disk and matches 55,131 cells. This writes a
SEPARATE label, so the existing cache and everything read off it stay untouched:

```bash
sbatch --partition=jobs-cpu --gres=none --mem=256GB --cpus-per-task=16 --time=08:00:00 \
  --export=ALL,RUN_CMD='PYTHONPATH=. python -u -m src.data.velocity bmmc_multiome_full --dynamics-n-jobs 16' \
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
