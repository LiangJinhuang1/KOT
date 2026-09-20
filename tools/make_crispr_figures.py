#!/usr/bin/env python3
"""Build CRISPR response figures from saved effects; no new model evaluation or training."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.visualization.crispr_responses import (
    digest, plot_effect_arrows, plot_task_a, plot_task_b, pooled_effects,
    read_task_a, read_task_b, summarize_effects,
)

CAPTIONS = """# CRISPR thesis figures

## Figure 12 — Protein responses from measured knockout RNA (Task A)

[PDF](fig12_crispr_measured_rna.pdf) · [PNG](fig12_crispr_measured_rna.png)

**Question:** does a model trained on non-targeting (NT) cells translate measured knockout
RNA changes into protein responses? This is not prediction from a knockout identity alone.

Top panels: measured, KOT and no-kinetics protein changes from NT controls. Knockouts are
ordered alphabetically, with a shared symmetric color scale in the saved normalized protein
units. Gray entries are outside the primary effect set, not zero responses. Every sufficiently
sampled non-self effect is retained regardless of significance: 94 knockout–protein pairs,
24 knockouts, four proteins and three replicate rows per eligible pair. Predictions are averaged
equally over the three replicates within each model seed, then equally over the 12 seeds.
Measurements are averaged over replicates once; repeated copies across model seeds are not
independent observations. Models and source tables are from the NT-only Task A evaluation.

Bottom left: the same 94 measured/predicted effects, colored by protein; the dashed line is
identity. Bottom right: mean absolute error on exactly those pairs. Faint dots describe model
seed variation; diamonds score the seed-ensemble predictions. Bars are 95% percentile bootstrap
intervals obtained by resampling whole knockouts, carrying their protein effects together.
These intervals quantify variation across tested knockouts conditional on these replicates;
they do not establish generalization to new donors. Cognate RNA and zero-change controls are
included on the same effect set. A strong correlation does not imply lower absolute error.

## Figure 13 — Protein responses from predicted knockout RNA (Task B)

[PDF](fig13_crispr_predicted_rna.pdf) · [PNG](fig13_crispr_predicted_rna.png)

**Question:** does frozen KOT preserve predicted perturbation responses when its RNA input is
itself predicted? The upstream model is the existing linear in-silico perturbation (ISP) model.
Its protocol is **leave-one-replicate-out for seen perturbations**, with shared NT controls,
not leave-one-perturbation-out. Full/no-kinetics/shuffled-velocity KOT arms receive byte-identical
RNA profile files within each seed and use identical saved replicate-split metadata. The
generator checks training/test cell disjointness in each fold. Frozen checkpoint identities
and their recorded hashes are retained in provenance; checkpoints were not rerun here.

Left: Spearman correlation. Right: RMSE divided by the standard deviation of observed effects,
following the project metric definition. Each panel uses the common_primary set that
csv/08_crispr_task_b_common.csv scores: 76 pairs over the 19 perturbations every in-silico
perturbation baseline could produce, dropping CAV1, CD86, MARCH8, PDCD1LG2 and TNFRSF14. That
is a narrower population than Task A's 94 pairs, so the two figures' numbers are not directly
comparable; on the wider set KOT scores rho 0.426 and MAE 0.106. Faint dots are per-seed
metrics, diamonds are the mean of those per-seed metrics, and bars bootstrap whole knockouts
around that same mean. The zero-change predictor has a valid error but undefined rank correlation.
This comparison isolates the downstream KOT arm for this upstream ISP; it is not a ranking of
in-silico perturbation methods with different supervision or knockout coverage.

## Figure S4 — Perturbation-effect arrows on a shared basis

[PDF](figS4_crispr_effect_arrows.pdf) · [PNG](figS4_crispr_effect_arrows.png)

**These arrows are effect vectors, not RNA velocity, time trajectories, or individual cell
transitions.** Every arrow starts at zero change and ends at a measured or predicted pooled
four-protein response. All panels use the same PCA basis and axis scale. PCA is fitted to
measured effects solely for descriptive visualization, after prediction and scoring; it is
not training preprocessing or an evaluation metric. Displacements are multiplied directly by
the PCA loadings so zero change stays at the origin. Knockouts without all four eligible
protein effects are excluded from this visualization only, with names recorded in provenance;
all 94 primary pairs remain in the quantitative figures. All complete responses are shown,
without selecting visually successful examples. PCA loadings, explained variance and projected
coordinates are exported. Overlapping arrows can reveal similar predicted responses across
knockouts and should not be interpreted as a smooth biological flow.

## Reproduce and inspect

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 singularity exec /data/common/images/codedev_v1.0.5.sif python tools/make_crispr_figures.py --out figures/crispr_revision_20260916
```

`data/` contains the exact matched rows, pooled effects, seed metrics, ensemble metrics and
projection coordinates. `provenance.json` records source hashes, Task B split fingerprints,
checkpoint identities and projection details. Bootstrap seed is 20260916; default resamples
are 2000. Original data, predictions and earlier figure exports are preserved.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('figures/crispr_revision_20260916'))
    parser.add_argument('--n-boot', type=int, default=2000)
    args = parser.parse_args()
    if args.n_boot < 100:
        raise ValueError('Use at least 100 bootstrap resamples for figure intervals')
    args.out.mkdir(parents=True, exist_ok=True)
    data = args.out / 'data'; data.mkdir(exist_ok=True)
    task_a, inputs = read_task_a(ROOT)
    print(f'Task A: validated {len(task_a)} rows on the full predefined effect population', flush=True)
    observed_path = ROOT / 'cache/results/crispr/crispr_effects_observed.csv'
    task_b, task_b_inputs, protocols = read_task_b(ROOT, pd.read_csv(observed_path))
    inputs.update(task_b_inputs)
    print(f'Task B: validated {len(task_b)} rows and {len(protocols)} split/input records', flush=True)
    for name, frame in [('task_a', task_a), ('task_b', task_b)]:
        frame.to_csv(data / f'{name}_matched_effects.csv', index=False)
        per_seed, ensemble = pooled_effects(frame)
        seed_metrics, summary = summarize_effects(per_seed, ensemble, args.n_boot)
        per_seed.to_csv(data / f'{name}_pooled_per_seed.csv', index=False)
        ensemble.to_csv(data / f'{name}_ensemble_effects.csv', index=False)
        seed_metrics.to_csv(data / f'{name}_seed_metrics.csv', index=False)
        summary.to_csv(data / f'{name}_ensemble_metrics.csv', index=False)
        print(name, summary[['arm', 'n', 'spearman', 'mae', 'nrmse']].round(4).to_string(index=False), flush=True)
        if name == 'task_a':
            plot_task_a(ensemble, seed_metrics, summary, args.out)
            coordinates, loadings, projection = plot_effect_arrows(ensemble, args.out)
            coordinates.to_csv(data / 'effect_arrow_coordinates.csv', index=False)
            loadings.to_csv(data / 'effect_pca_loadings.csv')
        else:
            plot_task_b(seed_metrics, summary, args.out)
    for source in [Path(__file__).resolve(), ROOT / 'src/visualization/crispr_responses.py',
                   ROOT / 'src/visualization/style.py', ROOT / 'src/evaluation/crispr_metrics.py']:
        inputs[str(source.relative_to(ROOT))] = digest(source)
    provenance = {'inputs': inputs, 'task_b_protocols': protocols, 'projection': projection,
                  'bootstrap': {'unit': 'perturbation', 'resamples': args.n_boot, 'seed': 20260916},
                  'selection': 'all predefined primary effects, no significance selection',
                  'replicate_aggregation': 'equal weights', 'seed_aggregation': 'equal weights'}
    (args.out / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    (args.out / 'README.md').write_text(CAPTIONS)
    print(f'Complete: {args.out}', flush=True)


if __name__ == '__main__':
    main()
