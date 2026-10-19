"""Logging handlers used by the Hydra job logging config."""

from __future__ import annotations

import logging
from pathlib import Path


def file_handler(filename: str, mode: str = "a") -> logging.FileHandler:
    """Build a file handler, creating the parent folder first.

    Hydra configures logging before the task runs, so the ``logs/``
    folder of a fresh run does not exist yet. ``conf/hydra/job_logging``
    points its file handler at this factory.

    Args:
        filename: Log file path, such as ``<run>/logs/main.log``.
        mode: File open mode.

    Returns:
        A :class:`logging.FileHandler` writing to ``filename``.
    """
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    return logging.FileHandler(path, mode=mode)


__all__ = ["file_handler"]
