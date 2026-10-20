"""Directory layout of one experiment run under ``outputs/``.

A run launched through ``main.py`` lives in Hydra's run directory,
``outputs/<experiment_name>/<YYYY-MM-DD_HH-MM-SS>/`` (see
``conf/config.yaml``). Inside it:

- ``exp/``: raw experiment data (checkpoints, variational parameters).
- ``logs/``: execution logs (the Hydra job log is written here).
- ``plots/``: figures written by :mod:`QELM.NQS.utils.makeplots`, next to
  ``logs/`` of the same run.

Hydra keeps its own resolved config in ``.hydra/`` at the run root.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from hydra.core.hydra_config import HydraConfig

OUTPUTS_ROOT = Path("outputs")
_UNSET = frozenset({"", "null", "none", "???"})


@dataclass(frozen=True)
class ExperimentPaths:
    """Folders of one run.

    Attributes:
        root: Run directory, ``outputs/<experiment_name>/<timestamp>/``
            for a Hydra job.
        exp: Raw experiment data.
        logs: Execution logs.
        plots: Figures, next to ``logs``.
    """

    root: Path
    exp: Path
    logs: Path
    plots: Path

    @classmethod
    def at(cls, root: Path | str) -> ExperimentPaths:
        """Describe the layout under ``root`` without touching the disk.

        Args:
            root: Run directory.

        Returns:
            Paths of the ``exp``, ``logs`` and ``plots`` subfolders.
        """
        base = Path(root)
        return cls(
            root=base, exp=base / "exp", logs=base / "logs", plots=base / "plots"
        )

    def create(self) -> ExperimentPaths:
        """Create the run directory and its three subfolders.

        Returns:
            ``self``, so calls can be chained.
        """
        for directory in (self.exp, self.logs, self.plots):
            directory.mkdir(parents=True, exist_ok=True)
        return self


def hydra_run_dir() -> Path | None:
    """Return Hydra's run directory when a job is active, else ``None``."""
    if not HydraConfig.initialized():
        return None
    output_dir = str(HydraConfig.get().runtime.output_dir).strip()
    if output_dir.lower() in _UNSET:
        return None
    return Path(output_dir)


def experiment_paths(
    root: Path | str | None = None,
    *,
    experiment_name: str = "default",
    outputs_root: Path | str = OUTPUTS_ROOT,
) -> ExperimentPaths:
    """Resolve and create the folders of the current run.

    Resolution order: an explicit ``root``; else the active Hydra run
    directory; else a new ``<outputs_root>/<experiment_name>/<timestamp>_<id>/``
    (see :func:`new_run_name`), so that code run outside Hydra (tests,
    notebooks) never overwrites a run and runs still sort by time.

    Args:
        root: Explicit run directory.
        experiment_name: Experiment folder used by the fallback.
        outputs_root: Root folder used by the fallback.

    Returns:
        The created :class:`ExperimentPaths`.
    """
    if root is None:
        root = hydra_run_dir()
    if root is None:
        root = Path(outputs_root) / experiment_name / new_run_name()
    return ExperimentPaths.at(root).create()


def new_run_name() -> str:
    """Return ``YYYY-MM-DD_HH-MM-SS_<8 hex>``: sortable by time and unique."""
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{stamp}_{uuid.uuid4().hex[:8]}"


__all__ = [
    "OUTPUTS_ROOT",
    "ExperimentPaths",
    "experiment_paths",
    "hydra_run_dir",
    "new_run_name",
]
