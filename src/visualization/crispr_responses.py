"""CRISPR figures with explicit effect units, matched controls, and perturbation CIs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.evaluation.crispr_metrics import bootstrap_over_units, metric_suite
from src.visualization.style import apply_style, figsize, panel_letter, save_figure

KEYS = ['perturbation', 'replicate', 'protein']
PAIR_KEYS = ['perturbation', 'protein']
MODEL_ARMS = ['full', 'noDyn', 'shuffleVel']
TASK_A_ARMS = MODEL_ARMS + ['cognate_mrna', 'zero_change']
TASK_B_PROTOCOL = 'Leave-one-replicate-out, seen perturbations'
TASK_B_PANEL_TITLES = {
    'spearman': f'{TASK_B_PROTOCOL}: ranking',
    'nrmse': f'{TASK_B_PROTOCOL}: error',
}
LABELS = {'full': 'KOT', 'noDyn': 'No kinetics', 'shuffleVel': 'Shuffled velocity',
          'cognate_mrna': 'Cognate RNA', 'zero_change': 'Zero change'}
COLORS = {'full': '#0072B2', 'noDyn': '#777777', 'shuffleVel': '#D55E00',
          'cognate_mrna': '#009E73', 'zero_change': '#333333'}
PROTEINS = ['CD366', 'CD86', 'PDL1', 'PDL2']
PROTEIN_COLORS = dict(zip(PROTEINS, ['#0072B2', '#E69F00', '#009E73', '#CC79A7']))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_effects(frame: pd.DataFrame, observed: pd.DataFrame,
                     arms: list[str]) -> pd.DataFrame:
    """Reject unequal effect populations; observed values must agree with source data."""
    frame = frame.copy()
    if frame.empty or set(frame.arm) != set(arms):
        raise ValueError('Missing or unexpected CRISPR arms')
    if frame.duplicated(['arm', 'seed'] + KEYS).any():
        raise ValueError('Duplicate arm/seed/replicate/protein effect')
    if observed.duplicated(KEYS).any():
        raise ValueError('Duplicate observed effect')
    if not frame.in_primary.eq(True).all():
        raise ValueError('Figure contains effects outside the predefined primary set')
    frame = frame.merge(observed[KEYS + ['delta_protein']], on=KEYS,
                        how='left', validate='many_to_one', suffixes=('', '_source'))
    if not np.isfinite(frame[['delta_p_phi', 'delta_protein', 'delta_protein_source']]).all().all():
        raise ValueError('Missing or non-finite observed/predicted effects')
    if not np.allclose(frame.delta_protein, frame.delta_protein_source, rtol=1e-7, atol=1e-9):
        raise ValueError('Observed effects differ from the canonical observation table')
    primary = observed[observed.passes_threshold.eq(True) & observed.is_self_effect.eq(False)]
    reference = set(map(tuple, primary[KEYS].to_numpy()))
    for (arm, seed), group in frame.groupby(['arm', 'seed']):
        keys = set(map(tuple, group[KEYS].to_numpy()))
        if keys != reference:
            raise ValueError(f'{arm}/{seed}: effect population differs from the comparison')
    seed_sets = [set(frame.loc[frame.arm == arm, 'seed']) for arm in MODEL_ARMS if arm in arms]
    if any(seeds != seed_sets[0] for seeds in seed_sets[1:]):
        raise ValueError('KOT control arms have different model seeds')
    return frame.drop(columns='delta_protein_source')


def read_task_a(root: Path) -> tuple[pd.DataFrame, dict]:
    source = root / 'cache/results/crispr/evaluation_taskA_ntonly/crispr_effects_scored.csv'
    observed = root / 'cache/results/crispr/crispr_effects_observed.csv'
    frame = pd.read_csv(source)
    frame = frame[(frame.effect_set == 'primary') & frame.arm.isin(TASK_A_ARMS)]
    return validate_effects(frame, pd.read_csv(observed), TASK_A_ARMS), {
        str(source.relative_to(root)): digest(source), str(observed.relative_to(root)): digest(observed)}


def read_task_b(root: Path, observed: pd.DataFrame) -> tuple[pd.DataFrame, dict, list]:
    """Linear ISP arms have the same RNA predictions and leave-replicate-out splits."""
    blocks, hashes, protocols = [], {}, []
    for arm in MODEL_ARMS:
        paths = sorted((root / 'cache/results/crispr' /
                        f'task_b_linear_20260906_{arm}').glob('seed_*/crispr_task_b_effects_predicted.csv'))
        if not paths:
            raise ValueError(f'Missing Task B arm: {arm}')
        for path in paths:
            block = pd.read_csv(path)
            block = block[(block.model == 'linear') & block.in_primary.eq(True)].copy()
            block['arm'] = arm
            manifest_path = path.parent / 'manifest.json'
            profiles_path = path.parent / 'predicted_rna_profiles.csv'
            manifest = json.loads(manifest_path.read_text())
            if manifest['protocol'] != 'leave_one_replicate_out_seen_perturbations':
                raise ValueError('Unexpected Task B evaluation regime')
            for split in manifest['splits']:
                if set(split['training_cell_ids']) & set(split['test_cell_ids']):
                    raise ValueError('Task B has overlapping training and test cells')
            protocol = {k: manifest[k] for k in ['protocol', 'alpha', 'feature_space',
                                               'nt_preprocessing', 'splits']}
            protocol_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
            protocols.append({'arm': arm, 'seed': int(manifest['seed']),
                              'protocol_hash': protocol_hash, 'rna_hash': digest(profiles_path),
                              'checkpoint': manifest['checkpoint'],
                              'checkpoint_sha256_recorded': manifest['checkpoint_sha256']})
            blocks.append(block)
            for source in (path, manifest_path, profiles_path):
                hashes[str(source.relative_to(root))] = digest(source)
    protocol_frame = pd.DataFrame(protocols)
    if (protocol_frame.groupby('seed')[['protocol_hash', 'rna_hash']].nunique() != 1).any().any():
        raise ValueError('Task B arms have different RNA inputs or replicate splits')
    frame = pd.concat(blocks, ignore_index=True)
    zero = frame[frame.arm == 'full'].copy()
    zero['arm'], zero['delta_p_phi'] = 'zero_change', 0.
    frame = pd.concat([frame, zero], ignore_index=True)
    return validate_effects(frame, observed, MODEL_ARMS + ['zero_change']), hashes, protocols


def pooled_effects(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Equal weights for replicates, then equal weights for model seeds."""
    per_seed = frame.groupby(['arm', 'seed'] + PAIR_KEYS, as_index=False).agg(
        predicted=('delta_p_phi', 'mean'), observed=('delta_protein', 'mean'))
    ensemble = per_seed.groupby(['arm'] + PAIR_KEYS, as_index=False).agg(
        predicted=('predicted', 'mean'), observed=('observed', 'mean'), n_seeds=('seed', 'nunique'))
    return per_seed, ensemble


def summarize_effects(per_seed: pd.DataFrame, ensemble: pd.DataFrame,
                      n_boot: int = 2000) -> tuple[pd.DataFrame, pd.DataFrame]:
    seeds = []
    for (arm, seed), block in per_seed.groupby(['arm', 'seed']):
        seeds.append({'arm': arm, 'seed': seed,
                      **metric_suite(block.predicted.to_numpy(), block.observed.to_numpy())})
    summary = []
    for arm, block in ensemble.groupby('arm'):
        values = metric_suite(block.predicted.to_numpy(), block.observed.to_numpy())
        row = {'arm': arm, 'n_perturbations': block.perturbation.nunique(), **values}
        for metric in ['spearman', 'mae', 'nrmse']:
            if arm == 'zero_change' and metric == 'spearman':
                low, high = np.nan, np.nan
            else:
                low, high = bootstrap_over_units(block, 'perturbation', 'predicted',
                                                 'observed', metric, n_boot, 20260916)
            row[f'{metric}_ci_low'], row[f'{metric}_ci_high'] = low, high
        summary.append(row)
    return pd.DataFrame(seeds), pd.DataFrame(summary)


def performance_panel(ax, seed_metrics: pd.DataFrame, summary: pd.DataFrame,
                      arms: list[str], metric: str):
    """Seeds as dots; perturbation-bootstrap intervals of the ensemble as bars."""
    for position, arm in enumerate(arms):
        row = summary.set_index('arm').loc[arm]
        if not np.isfinite(row[metric]):
            ax.text(.05, position, 'Undefined: constant prediction', fontsize=6,
                    transform=ax.get_yaxis_transform(), va='center', color='.4')
            continue
        values = seed_metrics.loc[seed_metrics.arm == arm, metric].to_numpy()
        ax.scatter(values, position + np.linspace(-.10, .10, len(values)), s=8,
                   color=COLORS[arm], alpha=.4, edgecolors='none', zorder=3)
        # Percentile intervals need not contain the observed statistic, so draw bounds directly.
        low, high = row[f'{metric}_ci_low'], row[f'{metric}_ci_high']
        ax.plot([low, high], [position, position], color=COLORS[arm], lw=1.5, zorder=4)
        ax.scatter(row[metric], position, marker='D', s=16, color=COLORS[arm], zorder=5)
    ax.set_yticks(range(len(arms)), [LABELS[arm] for arm in arms])
    ax.set_ylim(len(arms)-.5, -.5)
    if metric == 'spearman':
        ax.axvline(0, color='.65', ls='--', lw=.7)
    ax.set_xlabel({'mae': 'Mean absolute error (lower is better)',
                  'spearman': 'Spearman ρ (higher is better)',
                  'nrmse': 'Normalized RMSE (lower is better)'}[metric])
    ax.spines[['top', 'right']].set_visible(False)


def response_matrices(ensemble: pd.DataFrame, arms: list[str]) -> tuple[list, list, list]:
    subset = ensemble[ensemble.arm == 'full']
    perturbations = sorted(subset.perturbation.unique())
    proteins = sorted(subset.protein.unique())
    measured = subset.pivot(index='perturbation', columns='protein', values='observed').reindex(
        index=perturbations, columns=proteins)
    matrices = [measured]
    for arm in arms:
        matrices.append(ensemble[ensemble.arm == arm].pivot(index='perturbation', columns='protein',
                         values='predicted').reindex(index=perturbations, columns=proteins))
    return perturbations, proteins, matrices


def plot_task_a(ensemble: pd.DataFrame, seed_metrics: pd.DataFrame,
                summary: pd.DataFrame, out: Path):
    apply_style('iclr')
    fig = plt.figure(figsize=figsize('full', 6.9), layout='constrained')
    grid = fig.add_gridspec(2, 1, height_ratios=[1.75, 1])
    top = grid[0].subgridspec(1, 3)
    bottom = grid[1].subgridspec(1, 2, width_ratios=[1, 1.2])
    perturbations, proteins, matrices = response_matrices(ensemble, ['full', 'noDyn'])
    limit = max(float(np.nanmax(abs(matrix.to_numpy()))) for matrix in matrices)
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad('#EEEEEE')
    axes = []
    for col, (matrix, title) in enumerate(zip(matrices, ['Measured', 'KOT', 'No kinetics'])):
        ax = fig.add_subplot(top[0, col]); axes.append(ax)
        im = ax.imshow(matrix, aspect='auto', cmap=cmap, vmin=-limit, vmax=limit,
                       interpolation='nearest')
        ax.set_xticks(range(len(proteins)), proteins, rotation=45, ha='right')
        ax.set_yticks(range(len(perturbations)), perturbations)
        ax.tick_params(labelleft=(col == 0))
        ax.set_title(title)
        if col == 0:
            ax.set_ylabel('Knockout (alphabetical)')
        ax.tick_params(length=0)
        panel_letter(ax, 'abc'[col])
    fig.colorbar(im, ax=axes, orientation='horizontal', fraction=.05, pad=.02,
                 label='Protein change from control (saved normalized units)')
    ax = fig.add_subplot(bottom[0, 0])
    block = ensemble[ensemble.arm == 'full']
    for protein, group in block.groupby('protein'):
        ax.scatter(group.observed, group.predicted, s=10, color=PROTEIN_COLORS[protein],
                   label=protein, alpha=.75, edgecolors='none')
    lim = max(abs(block.observed).max(), abs(block.predicted).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], color='.5', ls='--', lw=.7)
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel='Measured protein change',
           ylabel='Predicted protein change', title=f'Measured vs KOT ({len(block)} effects)')
    ax.legend(loc='upper left', fontsize=6, handletextpad=.3)
    ax.set_aspect('equal', adjustable='box')
    panel_letter(ax, 'd')
    ax = fig.add_subplot(bottom[0, 1])
    performance_panel(ax, seed_metrics, summary, TASK_A_ARMS, 'mae')
    ax.set_title('MAE on the same effects (not Spearman)')
    panel_letter(ax, 'e')
    save_figure(fig, out / 'fig12_crispr_measured_rna', verify=True)


def plot_task_b(seed_metrics: pd.DataFrame, summary: pd.DataFrame, out: Path):
    apply_style('iclr')
    fig, axes = plt.subplots(1, 2, figsize=figsize('full', 3.0), layout='constrained')
    fig.suptitle(TASK_B_PROTOCOL)
    arms = MODEL_ARMS + ['zero_change']
    for col, metric in enumerate(['spearman', 'nrmse']):
        performance_panel(axes[col], seed_metrics, summary, arms, metric)
        axes[col].set_title('Ranking' if metric == 'spearman' else 'Error')
        panel_letter(axes[col], 'ab'[col])
    axes[1].tick_params(labelleft=False)
    save_figure(fig, out / 'fig13_crispr_predicted_rna', verify=True)


def plot_effect_arrows(ensemble: pd.DataFrame, out: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Descriptive PCA of response vectors, not a velocity field or state trajectory."""
    apply_style('iclr')
    perturbations, proteins, matrices = response_matrices(ensemble, ['full', 'noDyn'])
    complete = np.logical_and.reduce([np.isfinite(m.to_numpy()).all(axis=1) for m in matrices])
    if complete.sum() < 3:
        raise ValueError('Too few complete protein response vectors for PCA')
    values = [matrix.to_numpy()[complete] for matrix in matrices]
    pca = PCA(n_components=2).fit(values[0])
    # Project DISPLACEMENTS directly: subtracting the PCA training mean would move zero change.
    projected = [value @ pca.components_.T for value in values]
    lim = max(np.max(abs(value)) for value in projected) * 1.12
    fig, axes = plt.subplots(1, 3, figsize=figsize('full', 2.6), sharex=True, sharey=True,
                             layout='constrained')
    rows = []
    for col, (xy, label, color) in enumerate(zip(projected, ['Measured', 'KOT', 'No kinetics'],
                                               ['#333333', '#0072B2', '#777777'])):
        ax = axes[col]
        ax.axhline(0, color='.88', lw=.6); ax.axvline(0, color='.88', lw=.6)
        ax.quiver(np.zeros(len(xy)), np.zeros(len(xy)), xy[:, 0], xy[:, 1],
                  angles='xy', scale_units='xy', scale=1, color=color, alpha=.4, width=.006)
        ax.scatter(xy[:, 0], xy[:, 1], s=8, color=color, edgecolors='none')
        ax.scatter(0, 0, s=15, marker='+', color='black', zorder=5)
        ax.set(title=label, xlim=(-lim, lim), ylim=(-lim, lim), xlabel='Effect PC 1')
        ax.set_aspect('equal', adjustable='box')
        ax.spines[['top', 'right']].set_visible(False)
        panel_letter(ax, 'abc'[col])
        for perturbation, point in zip(np.array(perturbations)[complete], xy):
            rows.append({'series': label, 'perturbation': perturbation,
                         'effect_pc1': point[0], 'effect_pc2': point[1]})
    axes[0].set_ylabel('Effect PC 2')
    save_figure(fig, out / 'figS4_crispr_effect_arrows', verify=True)
    loadings = pd.DataFrame(pca.components_.T, index=proteins, columns=['PC1', 'PC2'])
    loadings.index.name = 'protein'
    return pd.DataFrame(rows), loadings, {'n_complete_perturbations': int(complete.sum()),
        'excluded_incomplete_perturbations': list(np.array(perturbations)[~complete]),
        'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
        'basis': 'PCA fit to measured pooled effect vectors for descriptive visualization only',
        'projection': 'delta @ components.T; zero effect stays at the origin'}
