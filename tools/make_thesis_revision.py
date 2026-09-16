#!/usr/bin/env python3
"""Render the first thesis revision from saved tables; no training or full benchmark evaluation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.visualization.control_effects import (
    anchor_comparisons, anchor_run_provenance, file_digest, plot_anchor_transfer, plot_control_effects, read_controls,
)
from src.visualization.kinetics import collect_beta, plot_beta_recovery
from src.visualization.style import set_verify
from tools.make_paper_figures import figure7

CAPTIONS = """# Thesis figure revision

## Figure 10 — Effects of velocity and gene-link controls

[PDF](fig10_paired_controls.pdf) · [PNG](fig10_paired_controls.png)

Each control is compared with the original-velocity run for the same dataset and model seed.
Blue dots show all 12 seed differences. Black diamonds show their mean; whiskers are 95%
percentile bootstrap intervals for the mean (10,000 seed resamples; fixed bootstrap seed).
The original-velocity row is identically zero by definition. The separator distinguishes
velocity interventions from gene–protein link permutation. Left: change in mean FOSCTTM
(negative is lower pairing error). Right: change in the median cosine between mapped velocity
(JVP) and the fitted ODE vector (RHS); negative means less agreement. The ODE diagnostic uses
each arm's own inputs and is not validation against an independent observed velocity field.
Cosine for zero velocity is undefined; the saved numerical zero is intentionally not plotted.

**Population:** saved checkpoint diagnostic population, including fitted cells; these are
not held-out performance estimates. Checkpoint: best_align. Settings: lr_beta=0.001,
300-epoch learning-rate warmup, lambda_dyn=1000. All 120 rows match their saved checkpoint
metrics. Within each dataset/seed, source configurations agree except for the intended
intervention and output directory; saved split digests also agree. This verifies recorded
protocols, not immutable historical data contents. Source hashes and split digests are in
`provenance.json`. Seed intervals describe training randomness, not donor uncertainty.

## Figure S3 — Velocity-only alignment detail

[PDF](figS3_velocity_detail.pdf) · [PNG](figS3_velocity_detail.png)

A magnified view of the three velocity interventions from Figure 10, on the same linear
scale for BMMC and PBMC. Gene-link permutation is omitted here because its larger effect
compresses the velocity comparisons. Data, seed pairing, interval definition, checkpoint,
and population are identical to Figure 10; this is not an additional experiment.

## Figure 11 — Anchor fit and transfer to unanchored proteins

[PDF](fig11_anchor_transfer.pdf) · [PNG](fig11_anchor_transfer.png)

Only proteins inside the kinetic mask are included. Top: absolute rate error against the
literature-derived reference, separating proteins used as anchors from those not used.
Points are means of within-seed protein averages; whiskers are seed-bootstrap 95% intervals.
Protein membership changes with anchor count, so top-row trends alone are not a transfer test.
Bottom: for each count and seed, compare each currently unanchored protein with that same
protein in the same seed's no-anchor run, then average within seed. Dots show seed averages;
diamonds and whiskers show the mean and its bootstrap interval. Negative values indicate
improved errors on the unanchored proteins. At full anchor coverage no unanchored proteins
remain inside the kinetic mask, so no unanchored estimate is plotted. Counts per seed are
included in the summary CSV.

**Scope:** this is transfer of an anchor constraint under a shared reference-derived scale,
not an independent recovery of physical half-lives. The original rate scale was resolved
using the reference panel; unanchored targets are not a strictly isolated external test.
Targets must be identical across counts for each seed/protein; the generator checks this.
Source configurations and split digests must also match, allowing only anchor count and output
directory to change. Each count must contain the same seeds and reference protein panel.
All table rows and the paired no-anchor references are exported for inspection. Runs were
filtered to lr_beta=0.001 and a 300-epoch learning-rate warmup. No configuration was selected
based on the new figure's results.

## Corrected existing figures

[Figure 4](fig4_kinetics.pdf): axes now identify literature-derived **anchor targets in model
units**, and the backend comparison reports **anchor rank agreement**. This plot shows prior
agreement, not independent kinetic recovery. Existing curated-run aggregation is preserved;
it should not be presented as a single matched configuration. Input rows are exported.

[Figure 7](fig7_prediction.pdf): **Not in kinetics** denotes proteins outside the kinetic
mask, not a no-kinetics model. The number of seeds is derived from the data. Per-marker
whiskers retain their original min–max definition (not confidence intervals). Protein-table
points are held-out-cell evaluations; the pooled-array panel follows the original run selector
and is illustrative, with its selected source and array hashes recorded in `provenance.json`. Coverage is not random
and does not establish a causal effect of the kinetics term.

## Reproduce

From the project root, in the existing container:

```bash
singularity exec /data/common/images/codedev_v1.0.5.sif python tools/make_thesis_revision.py --out figures/thesis_revision_20260915
```

This command only reads saved results and writes this revision directory. Existing figures
outside the revision directory are preserved. Machine-readable data and provenance accompany
the figures. New baseline evaluations and training sweeps are outside this first revision.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('figures/thesis_revision_20260915'))
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    data = out / 'data'
    data.mkdir(exist_ok=True)
    set_verify(True)

    source = ROOT / 'cache/results/kinetics_ablation_real_per_seed.csv'
    paired, provenance = read_controls(source, ROOT)
    paired.to_csv(data / 'control_effects_per_seed.csv', index=False)
    plot_control_effects(paired, out).to_csv(data / 'control_effects_summary.csv', index=False)

    source = ROOT / 'cache/results/beta_anchor_fit_per_protein.csv'
    anchors = pd.read_csv(source)
    anchors = anchors[(anchors.lr_beta == .001) & (anchors.lr_warmup_epochs == 300)]
    anchor_audit = anchor_run_provenance(anchors, ROOT)
    provenance['anchor_runs'] = anchor_audit['runs']
    provenance['inputs'].update(anchor_audit['inputs'])
    anchor_pairs = anchor_comparisons(anchors)
    anchor_pairs.to_csv(data / 'anchor_pairs_per_protein.csv', index=False)
    plot_anchor_transfer(anchor_pairs, out).to_csv(data / 'anchor_transfer_summary.csv', index=False)
    provenance['inputs'][str(source.relative_to(ROOT))] = file_digest(source)

    beta = collect_beta()
    beta.to_csv(data / 'fig4_anchor_observations.csv', index=False)
    for note in plot_beta_recovery(beta, out / 'fig4_kinetics'):
        print(note)
    prediction_run = figure7(out)
    provenance['panels'] = {'fig4': 'data/fig4_anchor_observations.csv',
                            'fig7_pooled_run': str(prediction_run)}
    if prediction_run is not None:
        for name in ('phi_full_panel.npy', 'protein_full_panel.npy', 'run_config.yaml', 'diagnostics.json'):
            source = ROOT / prediction_run / name
            provenance['inputs'][str(source.relative_to(ROOT))] = file_digest(source)
    for row in beta[['run', 'dataset', 'seed']].drop_duplicates().itertuples(index=False):
        folder = ROOT / 'cache/training' / row.run / 'kot' / row.dataset
        if row.seed != '-':
            folder = folder / f'seed_{row.seed}'
        source = folder / 'diagnostics.json'
        provenance['inputs'][str(source.relative_to(ROOT))] = file_digest(source)
    for source in [ROOT / 'cache/results/protein_eval_per_protein.csv',
                   ROOT / 'cache/training/MANIFEST.json',
                   ROOT / 'src/visualization/control_effects.py',
                   ROOT / 'src/visualization/kinetics.py',
                   ROOT / 'src/visualization/prediction.py',
                   ROOT / 'src/visualization/style.py',
                   ROOT / 'tools/make_paper_figures.py', Path(__file__).resolve()]:
        provenance['inputs'][str(source.relative_to(ROOT))] = file_digest(source)
    provenance['bootstrap'] = {'seed': 20260915, 'resamples': 10000, 'unit': 'model seed',
                               'interval': '95% percentile of mean paired differences'}
    provenance['control_population'] = 'saved diagnostics including fitted cells; not held-out'
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    (out / 'README.md').write_text(CAPTIONS)
    print(f'Figures, source tables, captions and provenance written to {out}')


if __name__ == '__main__':
    main()
