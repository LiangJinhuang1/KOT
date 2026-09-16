"""Protect pairing and rate units in thesis control figures."""
import json

import yaml
import numpy as np
import pandas as pd
import pytest

from src.visualization.control_effects import anchor_comparisons, paired_effects, read_controls


def controls():
    return pd.DataFrame([
        {'dataset': 'fixture', 'seed': seed, 'arm': arm,
         'mean_foscttm': seed / 10 + offset, 'jvp_rhs_cos_median': .8 - offset}
        for seed in [1, 2]
        for arm, offset in [('real', 0), ('shuffle', .1), ('reverse', .2),
                            ('zero', .3), ('permS', .4)]
    ])


def test_pairing_uses_seed_and_marks_zero_direction_undefined():
    result = paired_effects(controls().sample(frac=1, random_state=9))
    assert np.allclose(result[result.arm == 'shuffle'].mean_foscttm_delta, .1)
    assert np.allclose(result[result.arm == 'real'].mean_foscttm_delta, 0)
    assert result[result.arm == 'zero'].jvp_rhs_cos_median_delta.isna().all()


@pytest.mark.parametrize('bad', ['duplicate', 'missing_seed'])
def test_ambiguous_or_incomplete_control_pairs_fail(bad):
    frame = controls()
    frame = pd.concat([frame, frame.iloc[:1]]) if bad == 'duplicate' else frame.iloc[1:]
    with pytest.raises(ValueError):
        paired_effects(frame)


def anchors():
    return pd.DataFrame([
        {'dataset': 'fixture', 'seed': 1, 'protein': 'A', 'beta_anchor_subset_n': count,
         'beta': beta, 'beta_target': .5, 'abs_err': abs(beta-.5)}
        for count, beta in [(0, .8), (1, .6)]
    ])


def test_anchor_difference_matches_same_protein_and_seed():
    result = anchor_comparisons(anchors())
    assert result.loc[result.beta_anchor_subset_n == 1, 'error_delta'].iloc[0] == pytest.approx(-.2)


@pytest.mark.parametrize('bad', ['changed_units', 'missing_baseline', 'duplicate'])
def test_invalid_anchor_references_fail(bad):
    frame = anchors()
    if bad == 'changed_units':
        frame.loc[1, 'beta_target'] = .7
    elif bad == 'missing_baseline':
        frame = frame.iloc[1:]
    else:
        frame = pd.concat([frame, frame.iloc[:1]])
    with pytest.raises(ValueError):
        anchor_comparisons(frame)


def test_source_protocol_changes_rejected(tmp_path):
    frame = controls().rename(columns={'dataset': 'unused'})
    frame['dataset'] = 'bmmc_cite_retained'
    frame['model'] = 'kot'
    frame['lr_beta'] = .001
    frame['lr_warmup_epochs'] = 300
    frame['status'] = 'ok'
    frame['checkpoint'] = 'best_align'
    frame['run'] = frame.arm
    frame['run_dir'] = frame.arm
    for row in frame.itertuples():
        folder = tmp_path / row.run_dir / row.model / row.dataset / f'seed_{row.seed}'
        folder.mkdir(parents=True)
        config = {'seed': row.seed, 'lambda_dyn': 1000,
                  'kot_velocity_ablation': row.arm if row.arm in ('reverse', 'zero') else 'none',
                  'kot_velocity_shuffle': row.arm == 'shuffle', 'kot_s_permute': row.arm == 'permS'}
        # Explicit null and the historical omission must be equivalent.
        if row.arm != 'real':
            config['beta_anchor_subset_n'] = None
        (folder / 'run_config.yaml').write_text(yaml.safe_dump({'run_cfg': config, 'dataset_paths': {}}))
        (folder / 'diagnostics.json').write_text(json.dumps({'val_split_digest': 'fixture'}))
        (folder / 'diagnostics_best_align.json').write_text(json.dumps({
            'mean_foscttm': row.mean_foscttm, 'jvp_rhs_cos_median': row.jvp_rhs_cos_median}))
    source = tmp_path / 'controls.csv'
    frame.to_csv(source, index=False)
    result, _ = read_controls(source, tmp_path)
    assert len(result) == len(frame)
    config_path = tmp_path / 'shuffle/kot/bmmc_cite_retained/seed_1/run_config.yaml'
    payload = yaml.safe_load(config_path.read_text())
    payload['run_cfg']['lambda_dyn'] = 10
    config_path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError, match='configurations or split digests differ'):
        read_controls(source, tmp_path)
