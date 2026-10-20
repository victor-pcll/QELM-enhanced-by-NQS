"""Fidelity susceptibility of a variational state along a parameter direction.

Moving the parameters by ``epsilon * v`` changes the normalized state by

    1 - |<psi(theta) | psi(theta + epsilon v)>|^2 = epsilon^2 chi_F(v) + O(epsilon^3)

with ``chi_F(v) = v^† S v`` and ``S`` the quantum geometric tensor. Writing
``S = g + (i/2) Omega`` and ``v = x + i y``,

    v^† S v = x^T g x + y^T g y + y^T Omega x

so for a real direction ``chi_F = v^T g v`` with ``g = Re S`` the
Fubini-Study metric, while a complex direction also picks up the Berry
curvature term ``y^T Omega x``. ``chi_F`` peaks at quantum phase
transitions when ``v`` follows a physical parameter.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from jax.flatten_util import ravel_pytree

from QRC.NQS.evaluation.QGT import fubini_study_metric, qgt_matrix, qgt_of_state


def flatten_direction(v: Any) -> np.ndarray:
    """Return a direction as a flat complex vector.

    Args:
        v: A 1D array, or a PyTree with the structure of the parameters
            (for instance a VMC gradient or a TDVP update returned by
            NetKet), flattened in NetKet's order.

    Returns:
        Complex vector of shape ``(P,)``.
    """
    if isinstance(v, dict) or hasattr(v, "keys"):
        flat, _ = ravel_pytree(v)
        return np.asarray(flat, dtype=complex)
    return np.asarray(v, dtype=complex).ravel()


def compute_fidelity_susceptibility(
    S_matrix: Any,
    v: Any,
    *,
    metric_only: bool = False,
) -> float:
    """Return the fidelity susceptibility ``chi_F`` along ``v``.

    Args:
        S_matrix: Quantum geometric tensor, in any format accepted by
            :func:`QRC.NQS.evaluation.QGT.qgt_matrix` (dense, sparse or a
            NetKet QGT operator).
        v: Direction in parameter space: a vector of length ``P`` or a
            parameter PyTree (VMC gradient, TDVP update).
        metric_only: When ``True``, return ``v^† g v`` with ``g = Re S``
            only. It equals the default for real ``v``; for complex ``v`` it
            drops the Berry-curvature term and no longer matches the
            infidelity.

    Returns:
        ``chi_F >= 0`` (exactly ``0.0`` for ``v = 0``). It scales as
        ``|alpha|^2`` under ``v -> alpha v``.

    Raises:
        ValueError: If ``v`` and ``S`` have incompatible sizes.
    """
    direction = flatten_direction(v)
    S = fubini_study_metric(S_matrix) if metric_only else qgt_matrix(S_matrix)
    if S.shape != (direction.size, direction.size):
        raise ValueError(
            f"Direction has {direction.size} entries, the QGT is {S.shape}."
        )
    if not np.any(direction):
        return 0.0
    return float(np.real(np.vdot(direction, S @ direction)))


def fidelity_susceptibility_of_state(
    state: Any,
    v: Any,
    *,
    holomorphic: bool | None = True,
    metric_only: bool = False,
) -> float:
    """Compute the QGT of ``state`` and return ``chi_F`` along ``v``.

    Args:
        state: NetKet ``FullSumState`` (exact) or ``MCState`` (sampled).
        v: Direction, see :func:`compute_fidelity_susceptibility`.
        holomorphic: Passed to the QGT (``True`` for the complex RBM).
        metric_only: See :func:`compute_fidelity_susceptibility`.

    Returns:
        The fidelity susceptibility.
    """
    S = qgt_of_state(state, holomorphic=holomorphic)
    return compute_fidelity_susceptibility(S, v, metric_only=metric_only)


__all__ = [
    "compute_fidelity_susceptibility",
    "fidelity_susceptibility_of_state",
    "flatten_direction",
]
