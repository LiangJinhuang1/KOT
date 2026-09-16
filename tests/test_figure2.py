"""Figure 2 must show both kinetic arms' confusion matrices."""
import json
from pathlib import Path

from tools.make_paper_figures import (
    FIGURE2_CONFUSION_ARMS, confusion_from_run, figure2_confusion_sources,
)


def _run_with_matrix(folder: Path, matrix) -> Path:
    folder.mkdir(parents=True)
    (folder / "diagnostics.json").write_text(json.dumps({
        "branch_confusion_matrix": matrix,
        "branch_confusion_labels": ["Branch_A", "Branch_B", "Progenitor"],
    }))
    return folder


def test_figure2_confusion_arms_are_kot_and_no_kinetics():
    assert [model for model, _ in FIGURE2_CONFUSION_ARMS] == ["kot", "kot_nodyn"]


def test_confusion_from_run_requires_the_saved_matrix(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert confusion_from_run(empty) is None
    run = _run_with_matrix(tmp_path / "kot", [[9, 1, 0], [1, 8, 1], [0, 0, 10]])
    matrix, labels = confusion_from_run(run)
    assert labels[0] == "Branch_A"
    assert matrix[0][0] == 9


def test_figure2_keeps_both_slots_when_one_matrix_is_missing(tmp_path):
    kot = _run_with_matrix(tmp_path / "kot", [[10, 0, 0], [0, 10, 0], [0, 0, 10]])
    rows = figure2_confusion_sources({"kot": kot})
    assert [row[0] for row in rows] == ["kot", "kot_nodyn"]
    assert rows[0][2] is not None
    assert rows[1][2] is None
    assert rows[1][1] == "Alignment only"


def test_paired_matrices_do_not_call_fallback(tmp_path):
    kot = _run_with_matrix(tmp_path / "kot", [[10, 0, 0], [0, 10, 0], [0, 0, 10]])
    nodyn = _run_with_matrix(tmp_path / "nodyn", [[0, 10, 0], [10, 0, 0], [0, 0, 10]])

    def boom(model):
        raise AssertionError(f"fallback should not fill {model}")

    rows = figure2_confusion_sources(
        {"kot": kot, "kot_nodyn": nodyn}, fallback=boom)
    assert rows[1][2][0][1] == 10


def test_confusion_from_run_rejects_ragged_matrix(tmp_path):
    run = _run_with_matrix(tmp_path / "bad", [[1, 0], [0, 1, 0], [0, 0, 1]])
    assert confusion_from_run(run) is None
