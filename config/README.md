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
| `baselines`    | scot, moscot, glue, uniport, linear_ode                       |
| `upperbound`   | totalvi (paired-latent ceiling)                              |
| `convex`       | linear_ode                                                    |
| `kot`          | kot, kot_noanchor, kot_nodyn                                    |
| `kot_ablation` | kot, kot_nodyn, kot_fixedkappa, kot_fixedalpha, kot_oracle    |
| `all`          | kot, kot_nodyn, moscot, scot, linear_ode, uniport, glue, totalvi |

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

## SLURM

`slurm/train_slurm.sh` runs the runner on the cluster (recipes in slurm/README.md).
Select models with the
`MODELS` export (a group name or comma-list; empty = `default_models`):

```bash
sbatch --export=ALL,RUN_CMD='python -u -m src.training.runner --stage clean --scale mean --models baselines' slurm/train_slurm.sh
sbatch --export=ALL,RUN_CMD='python -u -m src.training.runner --stage branch --scale mean --models kot' slurm/train_slurm.sh
```
