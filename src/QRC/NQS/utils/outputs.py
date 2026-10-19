"""Find and load runs stored under ``outputs/`` (used by the notebooks)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from QRC.NQS.paths import OUTPUTS_ROOT, ExperimentPaths


def find_repo_root(start: Path | str | None = None) -> Path:
    """Return the first parent of ``start`` that holds ``pyproject.toml``.

    Notebooks run from ``notebooks/``; this lets them reach the root
    ``outputs/`` folder without hard-coded paths.

    Args:
        start: Folder to start from. Defaults to the current directory.

    Returns:
        The repository root.

    Raises:
        FileNotFoundError: If no parent holds ``pyproject.toml``.
    """
    here = Path.cwd() if start is None else Path(start)
    here = here.resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError(f"No pyproject.toml above {here}.")


def list_experiments(outputs_root: Path | str = OUTPUTS_ROOT) -> list[str]:
    """Return experiment names (subfolders of ``outputs_root``), sorted.

    Args:
        outputs_root: The ``outputs/`` folder.

    Returns:
        Experiment names; empty when the folder does not exist.
    """
    root = Path(outputs_root)
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def list_runs(
    experiment_name: str, outputs_root: Path | str = OUTPUTS_ROOT
) -> list[ExperimentPaths]:
    """Return the runs of one experiment, oldest first.

    A run is a subfolder holding at least one of ``exp/``, ``logs/`` or
    ``plots/``. Timestamped names sort chronologically.

    Args:
        experiment_name: Folder under ``outputs_root``.
        outputs_root: The ``outputs/`` folder.

    Returns:
        Paths of each run.
    """
    root = Path(outputs_root) / experiment_name
    if not root.is_dir():
        return []
    runs = []
    for candidate in sorted(root.iterdir()):
        paths = ExperimentPaths.at(candidate)
        if any(d.is_dir() for d in (paths.exp, paths.logs, paths.plots)):
            runs.append(paths)
    return runs


def latest_run(
    experiment_name: str, outputs_root: Path | str = OUTPUTS_ROOT
) -> ExperimentPaths:
    """Return the most recent run of an experiment.

    Args:
        experiment_name: Folder under ``outputs_root``.
        outputs_root: The ``outputs/`` folder.

    Returns:
        Paths of the last run in :func:`list_runs` order.

    Raises:
        FileNotFoundError: If the experiment has no run.
    """
    runs = list_runs(experiment_name, outputs_root)
    if not runs:
        raise FileNotFoundError(
            f"No run of {experiment_name!r} under {Path(outputs_root)}."
        )
    return runs[-1]


def load_checkpoint(path: Path | str) -> tuple[np.ndarray, dict[str, Any]]:
    """Read a checkpoint written by :func:`QRC.NQS.checkpoint.save_checkpoint`.

    Args:
        path: ``.npz`` file.

    Returns:
        The ``weights`` array and the decoded ``metadata`` dictionary.
    """
    with np.load(path) as archive:
        weights = np.array(archive["weights"])
        metadata = json.loads(str(archive["metadata"].item()))
    return weights, metadata


__all__ = [
    "find_repo_root",
    "latest_run",
    "list_experiments",
    "list_runs",
    "load_checkpoint",
]
