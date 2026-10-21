"""Seed NumPy from the Hydra config."""

from __future__ import annotations

import numpy as np
from omegaconf import DictConfig


def set_seed(cfg: DictConfig) -> int:
    """Seed NumPy from ``cfg.seed``.

    Call this once at startup, before any stochastic work. The default is
    ``seed: 42`` in ``conf/config.yaml``. JAX has no global random
    state: NetKet parameter initialization and sampling receive
    ``cfg.seed`` explicitly (see :func:`QELM.NQS.variational.build_nqs`).

    Args:
        cfg: Composed Hydra config. ``cfg.seed`` is a scalar integer.

    Returns:
        The seed applied to ``numpy.random``.
    """
    seed = int(cfg.seed)
    np.random.seed(seed)
    return seed
