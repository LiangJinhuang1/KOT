# Configuration Files

## `training.yaml` — the single training config

One file drives all staged-synthetic training. Choose which models to run with the
runner's `--models` flag — either a **group name** from `model_groups` or a
**comma-separated list** of model names:

```bash
python -m src.training.runner --models baselines  --stage clean  --scale mean
python -m src.training.runner --models kot         --stage clean  --scale mean
python -m src.training.runner --models upperbound  --stage clean  --scale mean
python -m src.training.runner --models all         --stage branch --scale mean
python -m src.training.runner --models scot,moscot --stage clean  --scale mean
```

With no `--models`, the config's `default_models` group is used.

### Model groups

Defined under `model_groups:` in `training.yaml`:

| group          | models                                                        |
|----------------|---------------------------------------------------------------|
| `baselines`    | scot, moscot, glue, uniport, maxfuse, linear_ode              |
| `upperbound`   | totalvi (paired-latent ceiling)                              |
| `convex`       | linear_ode                                                    |
| `kot`          | kot, kot_noanchor, kot_nodyn                                    |
| `kot_ablation` | kot, kot_nodyn, kot_fixedkappa, kot_fixedalpha, kot_oracle    |
| `all`          | kot, kot_nodyn, moscot, scot, linear_ode, uniport, maxfuse, glue, totalvi |
| `nt_supervised`| ridge, mlp, scipenn, scbutterfly (fitted on the controls' PAIRED RNA+protein) |
| `crispr`       | nt_supervised + totalvi — the CRISPR competitor set              |
| `all_crispr`   | kot, kot_nodyn + the CRISPR competitor set                       |

The CRISPR groups exist because that task is protein PREDICTION, not unpaired alignment,
so its competitors are protein predictors. They honour the same fit restriction as KOT
(no knockout cell's ADT is read) but are fitted on the control cells' RNA↔protein
pairing, which KOT is never given. Every prediction is stamped `uses_fit_pairing` and
`tools/papalexi.py` prints the two regimes in separate blocks. `scipenn` and
`scbutterfly` need `pip install -e '.[protein_baselines]'`.

### Seeds

Models listed in `defaults.seeded_models` run across every seed in the dataset's
`seeds` list; deterministic baselines (scot, moscot, linear_ode) run once. Only the
intersection of the selected models and `seeded_models` is multi-seeded, so seeding is
correct regardless of which group you pick.

## Staged synthetic runs

`runner.py --stage <oracle|clean|branch> --scale <mean|log>` injects the staged
RNA/protein paths, the preprocessing cache version, and the KOT feature layers.
`stage_overrides` holds only genuine per-stage differences (the anchor indices) —
model selection is global via `--models`, not per-stage.

For uniPort, keep `uniport_permute_second: true` when using `mode='d'` as the unpaired
baseline: it removes paired row order before training and restores the original order
only for FOSCTTM evaluation.

## Datasets — `datasets.yaml`

Each key names one RNA file (with velocity) and one paired protein or ATAC file. Raw files
live in `Datasets/`, derived files in `cache/`; neither is in git.

| key | modalities | notes |
|---|---|---|
| `bmmc_cite` | RNA → protein (ADT) | GEO GSE194122, CITE-seq BMMC |
| `pbmc` | RNA → protein (ADT) | 10x *5k PBMC protein v3 Next GEM* |
| `papalexi_nt_only` | RNA → protein (ADT) | GEO GSE153056, pooled ECCITE-seq; preprocessing fitted on non-targeting (NT) control cells only |
| `synthetic_linked_ode` | RNA → protein | generated; `runner.py --stage` picks the stage |
| `bmmc_multiome`, `shareseq_{skin,brain,lung}` | ATAC → RNA | chromatin track, not in the paper |

Suffixes:

- `_retained` — velocity recomputed while forcing ADT target genes into the gene set, so
  the kinetic term can use them.
- `_regvelo` — same cells, but velocity from RegVelo instead of scVelo (velocity ablation).

`velocity.yaml` holds the velocity settings per key (`python -m src.data.velocity --list`).
The ADT → gene link tables are built with `python tools/build_inputs.py adt-mapping` into
`cache/results/mapping/`.

## Anchors

Anchors are measured half-lives, converted to rates with `rate = ln 2 / half-life`. The
ODE runs in pseudotime, so they fix relative rates only: `src/data/beta_anchor.py`
rescales them to a mean of `beta_anchor_target` and uses them as a soft log-normal prior.

**Protein degradation β** (RNA → protein runs):

| file | contents |
|---|---|
| `beta_anchors_<ds>_mathieson.csv` | B cells, NK, monocytes — Mathieson et al. 2018, *Nat Commun*, Supp. Data 2 |
| `beta_anchors_<ds>_tcell.csv` | T cells — Savitski et al. 2018, *Cell* |
| `beta_anchors_<ds>_matched.csv` | both sources, each ADT matched to its lineage; **the default** |

`<ds>` is `bmmc_cite` or `pbmc`. Columns: `protein_name, gene_symbol, cell_type,
half_life_hours, beta_per_hour, quality_score_or_R2, source, anchor_weight`. Switch file
with `--set beta_anchor_csv=config/beta_anchors_pbmc_mathieson.csv`. Rebuild with
`python tools/build_inputs.py halflife-anchors`. Papalexi runs use no anchors.

**RNA degradation γ** (chromatin track only): `gamma_anchors_k562_timelapse.csv`, from
K562 TimeLapse-seq (Schofield et al. 2018, *Nat Methods*, Supp. Table 2). The matching
`.json` records the source URL, SHA-256 checksums and duplicate handling.

`papalexi_multipert_subset.csv` lists the knockouts shared with MultiPert (Zhao et al.
2025) for a like-for-like comparison. Its header explains why the two experiments differ.

## SLURM

`slurm/train_slurm.sh` runs the runner on the cluster (recipes in slurm/README.md).
Select models with the
`MODELS` export (a group name or comma-list; empty = `default_models`):

```bash
sbatch --export=ALL,RUN_CMD='python -u -m src.training.runner --stage clean --scale mean --models baselines' slurm/train_slurm.sh
sbatch --export=ALL,RUN_CMD='python -u -m src.training.runner --stage branch --scale mean --models kot' slurm/train_slurm.sh
```
