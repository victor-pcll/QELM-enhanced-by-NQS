"""The notebook helpers find runs under outputs/."""

import json
from pathlib import Path

import numpy as np
import pytest

from QRC.NQS.paths import ExperimentPaths
from QRC.NQS.utils.outputs import (
    find_repo_root,
    latest_run,
    list_experiments,
    list_runs,
    load_checkpoint,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
STARTER = REPO_ROOT / "notebooks" / "00_outputs_starter.ipynb"


def _fake_run(outputs: Path, experiment: str, stamp: str) -> ExperimentPaths:
    paths = ExperimentPaths.at(outputs / experiment / stamp).create()
    np.savez(
        paths.exp / "checkpoint.npz",
        weights=np.zeros(0),
        metadata=np.asarray(json.dumps({"seed": 42, "stamp": stamp})),
    )
    return paths


def test_find_repo_root_from_notebooks_folder() -> None:
    assert find_repo_root(REPO_ROOT / "notebooks") == REPO_ROOT


def test_find_repo_root_fails_outside_a_project(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        find_repo_root(tmp_path)


def test_runs_are_listed_in_time_order(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    _fake_run(outputs, "a", "2026-01-02_00-00-00")
    _fake_run(outputs, "a", "2026-01-01_00-00-00")
    _fake_run(outputs, "b", "2026-01-01_00-00-00")
    (outputs / "a" / "not_a_run").mkdir()

    assert list_experiments(outputs) == ["a", "b"]
    assert [r.root.name for r in list_runs("a", outputs)] == [
        "2026-01-01_00-00-00",
        "2026-01-02_00-00-00",
    ]
    latest = latest_run("a", outputs)
    _, metadata = load_checkpoint(latest.exp / "checkpoint.npz")
    assert metadata["stamp"] == "2026-01-02_00-00-00"


def test_missing_outputs_are_empty_or_raise(tmp_path: Path) -> None:
    assert list_experiments(tmp_path / "nothing") == []
    assert list_runs("x", tmp_path) == []
    with pytest.raises(FileNotFoundError):
        latest_run("x", tmp_path)


def test_starter_notebook_reads_root_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "pyproject.toml").write_text("")
    (tmp_path / "notebooks").mkdir()
    _fake_run(tmp_path / "outputs", "default", "2026-01-01_00-00-00")
    monkeypatch.chdir(tmp_path / "notebooks")

    notebook = json.loads(STARTER.read_text())
    namespace: dict[str, object] = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            exec("".join(cell["source"]), namespace)

    assert namespace["OUTPUTS"] == tmp_path / "outputs"
    assert namespace["metadata"] == {"seed": 42, "stamp": "2026-01-01_00-00-00"}
