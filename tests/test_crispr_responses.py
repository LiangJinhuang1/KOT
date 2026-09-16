"""Guard effect populations, replicate/seed weights and perturbation-input matching."""
import json

import numpy as np
import pandas as pd
import pytest

from src.visualization.crispr_responses import (
    MODEL_ARMS, pooled_effects, read_task_b, validate_effects,
)


def fixture_effects():
    observed = pd.DataFrame([
        {'perturbation': perturbation, 'replicate': replicate, 'protein': protein,
         'delta_protein': value, 'passes_threshold': True, 'is_self_effect': False}
        for perturbation in ['A', 'B'] for protein in ['P', 'Q']
        for replicate, value in [('r1', .1), ('r2', .3)]])
    blocks = []
    for arm in MODEL_ARMS:
        for seed in [1, 2]:
            block = observed.copy()
            block['arm'], block['seed'], block['in_primary'] = arm, seed, True
            block['delta_p_phi'] = block.delta_protein + .1 * seed
            blocks.append(block)
    return pd.concat(blocks, ignore_index=True), observed


def test_replicates_and_seeds_have_equal_weights():
    frame, observed = fixture_effects()
    per_seed, ensemble = pooled_effects(validate_effects(frame, observed, MODEL_ARMS))
    assert np.allclose(per_seed.loc[per_seed.seed == 1, 'predicted'], .3)
    assert np.allclose(ensemble.predicted, .35)
    assert np.allclose(ensemble.observed, .2)
    assert ensemble.n_seeds.eq(2).all()


@pytest.mark.parametrize('bad', ['duplicate', 'missing_for_every_arm', 'changed_observation'])
def test_invalid_primary_effect_populations_fail(bad):
    frame, observed = fixture_effects()
    if bad == 'duplicate':
        frame = pd.concat([frame, frame.iloc[:1]])
    elif bad == 'missing_for_every_arm':
        frame = frame[frame.perturbation != 'A']
    else:
        frame.loc[0, 'delta_protein'] += .1
    with pytest.raises(ValueError):
        validate_effects(frame, observed, MODEL_ARMS)


def write_task_b_fixture(root):
    frame, observed = fixture_effects()
    for arm in MODEL_ARMS:
        for seed in [1, 2]:
            folder = root / 'cache/results/crispr' / f'task_b_linear_20260906_{arm}' / f'seed_{seed}'
            folder.mkdir(parents=True)
            block = frame[(frame.arm == arm) & (frame.seed == seed)].copy()
            block['model'] = 'linear'
            block.to_csv(folder / 'crispr_task_b_effects_predicted.csv', index=False)
            (folder / 'predicted_rna_profiles.csv').write_text('same RNA inputs\n')
            manifest = {'protocol': 'leave_one_replicate_out_seen_perturbations', 'alpha': 1.,
                        'feature_space': 'fixture', 'nt_preprocessing': 'fixture', 'seed': seed,
                        'checkpoint': 'fixture.pt', 'checkpoint_sha256': 'recorded fixture hash',
                        'splits': [{'replicate': 'r1', 'training_cell_ids': ['train'],
                                    'test_cell_ids': ['test'], 'trained_targets': ['A', 'B']}]}
            (folder / 'manifest.json').write_text(json.dumps(manifest))
    return observed


def test_task_b_records_complete_matching_protocols(tmp_path):
    observed = write_task_b_fixture(tmp_path)
    frame, _, protocols = read_task_b(tmp_path, observed)
    assert len(protocols) == 6
    assert set(frame.arm) == set(MODEL_ARMS + ['zero_change'])
    assert frame[frame.arm == 'zero_change'].delta_p_phi.eq(0).all()


@pytest.mark.parametrize('bad', ['overlapping_cells', 'different_rna'])
def test_task_b_rejects_noncomparable_inputs(tmp_path, bad):
    observed = write_task_b_fixture(tmp_path)
    folder = tmp_path / 'cache/results/crispr/task_b_linear_20260906_noDyn/seed_1'
    if bad == 'different_rna':
        (folder / 'predicted_rna_profiles.csv').write_text('different RNA inputs\n')
    else:
        path = folder / 'manifest.json'
        manifest = json.loads(path.read_text())
        manifest['splits'][0]['test_cell_ids'] = ['train']
        path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        read_task_b(tmp_path, observed)
