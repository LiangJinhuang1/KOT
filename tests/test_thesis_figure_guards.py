"""Guards for Fig 7/8/S5: cell IDs, paired prediction, and an incomplete tune grid."""
import numpy as np
import pandas as pd
import pytest

from src.visualization.crispr_responses import TASK_B_PANEL_TITLES
from src.visualization.diagnostics import TUNE_BETAS, TUNE_LAMBDAS, heatmap_grid, tune_lambda_grid
from src.visualization.h5ad_obs import labels_from_codes
from src.visualization.lineage_map import aligned_cell_positions
from src.visualization.prediction import paired_protein_delta, series_flag
from tools.score_heldout_phi import dense_key


def test_save_figure_writes_svg_for_editing():
    from src.visualization.style import save_figure

    assert "svg" in save_figure.__kwdefaults__["formats"]


def test_aligned_positions_keep_full_order():
    positions = aligned_cell_positions(4, 4, np.array([0, 2]))
    assert list(positions) == [0, 1, 2, 3]


def test_aligned_positions_use_fitted_rows_not_first_n():
    fit = np.array([2, 0, 5, 7])
    positions = aligned_cell_positions(4, 10, fit)
    assert list(positions) == [2, 0, 5, 7]


def test_aligned_positions_refuse_unrecoverable_length():
    assert aligned_cell_positions(5, 10, np.array([0, 1, 2, 3])) is None
    assert aligned_cell_positions(8, 10, None) is None


def test_unlabelled_categorical_codes_are_not_the_last_category():
    labels = labels_from_codes(["T", "B", "Mono"], [0, -1, 2])
    assert list(labels) == ["T", "", "Mono"]


def test_dense_key_reads_configured_layers():
    assert dense_key("Ms", default="X") == "layers/Ms"
    assert dense_key(None, default="X") == "X"
    assert dense_key("", default="Ms") == "layers/Ms"


def _arm(seeds, proteins, values, dataset="pbmc_retained"):
    rows = []
    for seed in seeds:
        for protein, value in zip(proteins, values):
            rows.append({"dataset": dataset, "seed": seed, "protein": protein,
                         "spearman": value, "in_kinetics": protein == "CD4"})
    return pd.DataFrame(rows)


def test_paired_protein_delta_is_one_to_one():
    proteins = ["CD4", "CD14"]
    left = _arm([1, 2], proteins, [0.8, 0.4])
    right = _arm([1, 2], proteins, [0.5, 0.5])
    delta = paired_protein_delta(left, right)
    by_protein = delta.groupby("protein")["delta"].mean()
    assert by_protein["CD4"] == pytest.approx(0.3)
    assert by_protein["CD14"] == pytest.approx(-0.1)


def test_paired_protein_delta_rejects_incomplete_or_duplicate_pairs():
    proteins = ["CD4", "CD14"]
    left = _arm([1, 2], proteins, [0.8, 0.4])
    short = _arm([1], proteins, [0.5, 0.5])
    with pytest.raises(ValueError, match="same proteins"):
        paired_protein_delta(left, short)
    dup = pd.concat([left, left.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate"):
        paired_protein_delta(dup, left)


def test_series_flag_does_not_treat_false_strings_as_true():
    flags = series_flag(pd.Series(["False", "True", "false", True, False]))
    assert list(flags) == [False, True, False, True, False]


def _tune_summary(cells):
    rows = []
    for dataset, lambda_dyn, lr_beta, fos, jvp, seeds in cells:
        rows.append({
            "dataset": dataset, "fitted_side": "tune", "lr_phi": 0.001,
            "lr_alpha_kappa": 0.0001, "lr_beta": lr_beta, "lambda_dyn": lambda_dyn,
            "lr_warmup_epochs": 300, "sinkhorn_reg": 0.1, "seeds": seeds,
            "foscttm_fitted_mean": fos, "jvp_cos_med_mean": jvp,
        })
    return pd.DataFrame(rows)


def test_tune_lambda_grid_keeps_missing_cells_nan():
    summary = _tune_summary([
        ("bmmc_cite_retained", 100, 0.001, 0.13, 0.8, 4),
        ("bmmc_cite_retained", 1000, 0.01, 0.14, 0.9, 8),
    ])
    grid = tune_lambda_grid(summary, "bmmc_cite_retained", "foscttm_fitted_mean")
    assert len(grid) == len(TUNE_LAMBDAS) * len(TUNE_BETAS)
    present = grid.dropna(subset=["foscttm_fitted_mean"])
    assert len(present) == 2
    missing = grid[grid["lambda_dyn"] == 300]
    assert missing["foscttm_fitted_mean"].isna().all()
    assert missing["seeds"].isna().all()


def test_tune_lambda_grid_rejects_duplicate_cells():
    summary = _tune_summary([
        ("bmmc_cite_retained", 100, 0.001, 0.13, 0.8, 4),
        ("bmmc_cite_retained", 100.0, 0.001, 0.12, 0.7, 4),
    ])
    with pytest.raises(ValueError, match="duplicate"):
        tune_lambda_grid(summary, "bmmc_cite_retained", "foscttm_fitted_mean")


def test_heatmap_hatches_unrun_cells_not_metric_gaps():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    table = tune_lambda_grid(_tune_summary([
        ("bmmc_cite_retained", 100, 0.001, np.nan, 0.8, 4),
        ("bmmc_cite_retained", 1000, 0.01, 0.14, 0.9, 8),
    ]), "bmmc_cite_retained", "foscttm_fitted_mean")
    fig, ax = plt.subplots()
    heatmap_grid(ax, table, "foscttm_fitted_mean", cmap="viridis", vmin=0, vmax=1)
    hatches = [p for p in ax.patches if isinstance(p, Rectangle) and p.get_hatch()]
    expected_missing = len(TUNE_LAMBDAS) * len(TUNE_BETAS) - 2
    assert len(hatches) == expected_missing
    plt.close(fig)


def test_schematic_is_a_cartoon_not_a_half_life_panel():
    from tools.make_schematic import build

    svg = build()
    assert "state r" in svg
    assert "velocity v" in svg
    assert "the kinetics term" in svg
    assert "ruled out" in svg
    assert "half-lives" not in svg


def test_task_b_titles_name_the_seen_perturbation_protocol():
    from src.visualization.crispr_responses import TASK_B_PROTOCOL
    lowered = TASK_B_PROTOCOL.lower()
    assert "leave-one-replicate-out" in lowered
    assert "seen perturbation" in lowered
    for title in TASK_B_PANEL_TITLES.values():
        assert TASK_B_PROTOCOL in title
