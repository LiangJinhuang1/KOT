"""Seed-paired control effects and anchor diagnostics from existing result tables."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.visualization.style import apply_style, figsize, panel_letter, save_figure

DATASETS = {'bmmc_cite_retained': 'BMMC', 'pbmc_retained': 'PBMC'}
ARMS = ['real', 'shuffle', 'reverse', 'zero', 'permS']
LABELS = ['Original velocity', 'Shuffled velocity', 'Reversed velocity',
          'Zero velocity', 'Permuted gene links']
METRICS = ['mean_foscttm', 'jvp_rhs_cos_median']
# The intervention and output destination are the only allowed within-seed changes.
INTERVENTION_KEYS = {'output_root', 'kot_velocity_shuffle', 'kot_velocity_ablation',
                     'kot_s_permute'}


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def control_arm(config: dict) -> str:
    if config.get('kot_s_permute', False):
        return 'permS'
    mode = config.get('kot_velocity_ablation') or 'none'
    if mode == 'none':
        return 'shuffle' if config.get('kot_velocity_shuffle', False) else 'real'
    if mode not in ARMS:
        raise ValueError(f'Unsupported velocity intervention: {mode}')
    return mode


def paired_effects(frame: pd.DataFrame) -> pd.DataFrame:
    """Require complete one-to-one seed pairs; never silently discard a control."""
    keys = ['dataset', 'seed', 'arm']
    if frame.duplicated(keys).any():
        raise ValueError('Duplicate dataset/seed/arm rows would bias paired effects')
    for dataset, group in frame.groupby('dataset'):
        if set(group.arm) != set(ARMS):
            raise ValueError(f'{dataset}: missing or unexpected control arms')
        seed_sets = [set(group.loc[group.arm == arm, 'seed']) for arm in ARMS]
        if any(seeds != seed_sets[0] for seeds in seed_sets[1:]):
            raise ValueError(f'{dataset}: control arms have different seed sets')
    reference = frame[frame.arm == 'real'][['dataset', 'seed'] + METRICS]
    paired = frame.merge(reference, on=['dataset', 'seed'], validate='many_to_one',
                         suffixes=('', '_reference'))
    for metric in METRICS:
        paired[f'{metric}_delta'] = paired[metric] - paired[f'{metric}_reference']
    # A zero vector has no direction; the stored implementation convention is not a cosine.
    paired.loc[paired.arm == 'zero', 'jvp_rhs_cos_median_delta'] = np.nan
    return paired


def read_controls(source: Path, root: Path) -> tuple[pd.DataFrame, dict]:
    frame = pd.read_csv(source)
    frame = frame[(frame.lr_beta == .001) & (frame.lr_warmup_epochs == 300)
                  & (frame.model == 'kot') & frame.dataset.isin(DATASETS)].copy()
    if frame.empty or not frame.status.eq('ok').all():
        raise ValueError('Expected completed setting-B controls; inspect missing/failed runs')
    records, provenance = [], {'inputs': {str(source): file_digest(source)}, 'runs': []}
    for row in frame.to_dict('records'):
        folder = root / row['run_dir'] / row['model'] / row['dataset'] / f"seed_{row['seed']}"
        config_path = folder / 'run_config.yaml'
        original_path = folder / 'diagnostics.json'
        checkpoint_path = folder / f"diagnostics_{row['checkpoint']}.json"
        payload = yaml.safe_load(config_path.read_text())
        config = payload['run_cfg']
        # Older runs omit this optional key; None means use the full anchor set.
        config.setdefault('beta_anchor_subset_n', None)
        diagnostics = json.loads(original_path.read_text())
        checkpoint = json.loads(checkpoint_path.read_text())
        digest = diagnostics.get('val_split_digest')
        if not digest:
            raise ValueError(f'{folder}: missing saved split digest')
        for metric in METRICS:
            if not np.isfinite(row[metric]) or not np.isclose(
                    row[metric], checkpoint[metric], rtol=1e-7, atol=1e-9):
                raise ValueError(f'{folder}: table/checkpoint mismatch for {metric}')
        protocol = {'dataset_paths': payload['dataset_paths'], 'checkpoint': row['checkpoint'],
                    'config': {key: value for key, value in config.items()
                               if key not in INTERVENTION_KEYS}, 'split_digest': digest}
        fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
        records.append({**row, 'arm': control_arm(config), 'split_digest': digest,
                        'protocol_fingerprint': fingerprint})
        provenance['runs'].append({'run': row['run'], 'dataset': row['dataset'],
                                   'seed': row['seed'], 'split_digest': digest,
                                   'checkpoint': row['checkpoint'], 'protocol_fingerprint': fingerprint})
        for path in (config_path, original_path, checkpoint_path):
            provenance['inputs'][str(path.relative_to(root))] = file_digest(path)
    records = pd.DataFrame(records)
    if (records.groupby(['dataset', 'seed']).protocol_fingerprint.nunique() != 1).any():
        raise ValueError('Control configurations or split digests differ beyond the intervention')
    return paired_effects(records), provenance


def mean_interval(values: np.ndarray) -> tuple[float, float, float]:
    """Seed-bootstrap 95% interval for a mean; seeds are not biological replicates."""
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Intervals require finite seed effects')
    rng = np.random.default_rng(20260915)
    boot = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(boot, [.025, .975])
    return float(values.mean()), float(low), float(high)


def plot_control_effects(paired: pd.DataFrame, out: Path) -> pd.DataFrame:
    apply_style('iclr')
    fig, axes = plt.subplots(2, 2, figsize=figsize('full', 4.9), sharey=True,
                             layout='constrained')
    summary = []
    for row, (dataset, label) in enumerate(DATASETS.items()):
        group = paired[paired.dataset == dataset]
        for col, metric in enumerate(METRICS):
            ax = axes[row, col]
            ax.axvline(0, color='.55', lw=.7, ls='--')
            ax.axhline(3.5, color='.85', lw=.6)
            for position, arm in enumerate(ARMS):
                values = group.loc[group.arm == arm, f'{metric}_delta'].to_numpy()
                if metric == 'jvp_rhs_cos_median' and arm == 'zero':
                    ax.text(.04, position, 'Undefined: zero vector', color='.4', fontsize=7,
                            transform=ax.get_yaxis_transform(), va='center')
                    continue
                mean, low, high = mean_interval(values)
                jitter = np.linspace(-.13, .13, len(values))
                ax.scatter(values, position + jitter, s=9, color='#0072B2', alpha=.45,
                           edgecolors='none', zorder=3)
                ax.errorbar(mean, position, xerr=[[mean-low], [high-mean]], fmt='D',
                            color='#1A1A1A', ms=3, capsize=2, elinewidth=1, zorder=4)
                summary.append({'dataset': dataset, 'arm': arm, 'metric': metric,
                                'n_seeds': len(values), 'mean_delta': mean,
                                'ci95_low': low, 'ci95_high': high})
            ax.set_yticks(range(len(ARMS)), LABELS)
            ax.set_ylim(4.55, -.55)
            ax.set_title(f'{label}: ' + ('alignment' if col == 0 else 'ODE agreement'))
            ax.set_xlabel('Δ FOSCTTM\nnegative = lower pairing error' if col == 0 else
                          'Δ JVP–RHS cosine\nnegative = less agreement')
            panel_letter(ax, 'abcd'[row*2+col])
            ax.spines[['top', 'right']].set_visible(False)
    fig.get_layout_engine().set(hspace=.10, wspace=.10)
    save_figure(fig, out / 'fig10_paired_controls', verify=True)
    plot_velocity_detail(paired, out)
    return pd.DataFrame(summary)



def plot_velocity_detail(paired: pd.DataFrame, out: Path) -> None:
    """Show the small velocity-only alignment effects on a common linear scale."""
    fig, axes = plt.subplots(1, 2, figsize=figsize('full', 2.4), sharex=True, sharey=True,
                             layout='constrained')
    arms = ['shuffle', 'reverse', 'zero']
    selected = paired[paired.arm.isin(arms)]
    extent = max(abs(selected.mean_foscttm_delta.min()), abs(selected.mean_foscttm_delta.max())) * 1.2
    for col, (dataset, label) in enumerate(DATASETS.items()):
        ax = axes[col]
        ax.axvline(0, color='.55', lw=.7, ls='--')
        for position, arm in enumerate(arms):
            values = selected.loc[(selected.dataset == dataset) & (selected.arm == arm),
                                  'mean_foscttm_delta'].to_numpy()
            mean, low, high = mean_interval(values)
            ax.scatter(values, position + np.linspace(-.12, .12, len(values)),
                       s=10, alpha=.45, color='#0072B2', edgecolors='none')
            ax.errorbar(mean, position, xerr=[[mean-low], [high-mean]], fmt='D',
                        ms=3, color='#1A1A1A', capsize=2)
        ax.set_yticks(range(3), ['Shuffled velocity', 'Reversed velocity', 'Zero velocity'])
        ax.set_ylim(2.5, -.5)
        ax.set_xlim(-extent, extent)
        ax.set_xlabel('Δ FOSCTTM versus original\nnegative = lower pairing error')
        ax.set_title(f'{label}: velocity effects (zoom)')
        ax.spines[['top', 'right']].set_visible(False)
        panel_letter(ax, 'ab'[col])
    save_figure(fig, out / 'figS3_velocity_detail', verify=True)



def anchor_run_provenance(frame: pd.DataFrame, root: Path) -> dict:
    """Check the actual anchor-run protocols as well as the exported rate units."""
    records, inputs = [], {}
    for row in frame[['run', 'dataset', 'seed']].drop_duplicates().itertuples(index=False):
        folder = root / 'cache/training' / row.run / 'kot' / row.dataset / f'seed_{row.seed}'
        config_path, diagnostic_path = folder / 'run_config.yaml', folder / 'diagnostics.json'
        payload = yaml.safe_load(config_path.read_text())
        diagnostic = json.loads(diagnostic_path.read_text())
        digest = diagnostic.get('val_split_digest')
        if not digest:
            raise ValueError(f'{folder}: missing anchor-run split digest')
        protocol = {'dataset_paths': payload['dataset_paths'], 'split_digest': digest,
                    'config': {k: v for k, v in payload['run_cfg'].items()
                               if k not in {'output_root', 'beta_anchor_subset_n'}}}
        fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
        records.append({'run': row.run, 'dataset': row.dataset, 'seed': row.seed,
                        'split_digest': digest, 'protocol_fingerprint': fingerprint})
        for path in (config_path, diagnostic_path):
            inputs[str(path.relative_to(root))] = file_digest(path)
    audit = pd.DataFrame(records)
    if (audit.groupby(['dataset', 'seed']).protocol_fingerprint.nunique() != 1).any():
        raise ValueError('Anchor-count runs differ beyond anchor count and output directory')
    return {'runs': records, 'inputs': inputs}


def anchor_comparisons(frame: pd.DataFrame) -> pd.DataFrame:
    """Compare each unanchored protein with itself in the same seed's no-anchor run."""
    keys = ['dataset', 'seed', 'protein']
    if frame.duplicated(keys + ['beta_anchor_subset_n']).any():
        raise ValueError('Duplicate anchor-count observations')
    for dataset, group in frame.groupby('dataset'):
        seed_sets = group.groupby('beta_anchor_subset_n').seed.apply(set)
        if any(seeds != seed_sets.iloc[0] for seeds in seed_sets):
            raise ValueError(f'{dataset}: anchor counts have different seed sets')
    for (dataset, seed), group in frame.groupby(['dataset', 'seed']):
        protein_sets = group.groupby('beta_anchor_subset_n').protein.apply(set)
        if any(proteins != protein_sets.iloc[0] for proteins in protein_sets):
            raise ValueError(f'{dataset}/{seed}: anchor counts have different reference panels')
    baseline = frame[frame.beta_anchor_subset_n == 0][keys + ['abs_err', 'beta_target']]
    paired = frame.merge(baseline, on=keys, validate='many_to_one', how='left',
                         suffixes=('', '_no_anchor'))
    if paired.abs_err_no_anchor.isna().any():
        raise ValueError('Missing no-anchor seed/protein baseline')
    if not np.allclose(paired.beta_target, paired.beta_target_no_anchor):
        raise ValueError('Anchor counts changed rate units or reference targets')
    if not np.isfinite(paired[['beta', 'beta_target', 'abs_err']]).all().all():
        raise ValueError('Non-finite anchor values')
    if not np.allclose(paired.abs_err, abs(paired.beta - paired.beta_target)):
        raise ValueError('Saved anchor error disagrees with beta/reference values')
    paired['error_delta'] = paired.abs_err - paired.abs_err_no_anchor
    return paired


def plot_anchor_transfer(paired: pd.DataFrame, out: Path) -> pd.DataFrame:
    apply_style('iclr')
    fig, axes = plt.subplots(2, 2, figsize=figsize('full', 4.6), layout='constrained')
    summaries = []
    for col, (dataset, name) in enumerate(DATASETS.items()):
        # Outside-mask proteins never receive the kinetic prior and dilute transfer tests.
        group = paired[(paired.dataset == dataset) & paired.in_kinetics].copy()
        ax = axes[0, col]
        for anchored, label, color in [(True, 'Anchored', '#0072B2'),
                                        (False, 'Unanchored', '#D55E00')]:
            subset = group[group.anchored == anchored]
            seed_means = subset.groupby(['beta_anchor_subset_n', 'seed']).abs_err.mean()
            means = seed_means.groupby(level=0).mean()
            ax.plot(means.index, means.values, color=color, lw=1, label=label)
            for count, values in seed_means.groupby(level=0):
                mean, low, high = mean_interval(values.to_numpy())
                ax.errorbar(count, mean, yerr=[[mean-low], [high-mean]], fmt='o',
                            color=color, ms=3, capsize=2, elinewidth=.8)
                summaries.append({'dataset': dataset, 'anchors': count, 'anchored': anchored,
                                  'metric': 'absolute_error', 'mean': mean,
                                  'ci95_low': low, 'ci95_high': high, 'n_seeds': len(values)})
        ax.set_title(f'{name}: fit to rate references')
        ax.set_ylabel('Mean absolute β error\n(model units)')
        ax.legend(loc='best', fontsize=7)
        panel_letter(ax, 'ab'[col])
        ax = axes[1, col]
        ax.axhline(0, color='.5', ls='--', lw=.7)
        unanchored = group[(~group.anchored) & (group.beta_anchor_subset_n > 0)]
        for count, subset in unanchored.groupby('beta_anchor_subset_n'):
            seed_means = subset.groupby('seed').error_delta.mean()
            mean, low, high = mean_interval(seed_means.to_numpy())
            ax.scatter(np.full(len(seed_means), count), seed_means, color='#D55E00',
                       s=8, alpha=.35, edgecolors='none')
            ax.errorbar(count, mean, yerr=[[mean-low], [high-mean]], fmt='D',
                        color='#1A1A1A', ms=3, capsize=2, elinewidth=1)
            summaries.append({'dataset': dataset, 'anchors': count, 'anchored': False,
                              'metric': 'paired_error_delta', 'mean': mean,
                              'ci95_low': low, 'ci95_high': high, 'n_seeds': len(seed_means),
                              'n_proteins_per_seed_min': subset.groupby('seed').size().min(),
                              'n_proteins_per_seed_max': subset.groupby('seed').size().max()})
        ax.set_title(f'{name}: change on unanchored proteins')
        ax.set_ylabel('Δ error versus no anchors\nnegative = improvement')
        panel_letter(ax, 'cd'[col])
        for ax in axes[:, col]:
            ax.set_xlabel('Anchors used in fitting')
            ax.set_xticks(sorted(group.beta_anchor_subset_n.unique()))
            ax.spines[['top', 'right']].set_visible(False)
    save_figure(fig, out / 'fig11_anchor_transfer', verify=True)
    return pd.DataFrame(summaries)
