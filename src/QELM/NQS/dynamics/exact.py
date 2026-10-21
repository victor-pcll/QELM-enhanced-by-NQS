"""Exact real-time evolution of a state vector (the ED reference)."""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse.linalg as spla
from numpy.typing import ArrayLike


def time_grid(T: float, dt: float) -> np.ndarray:
    """Return the uniform grid ``0, dt, ..., T`` (both ends included).

    The same grid is used for the exact evolution and as ``tstops`` of the
    TDVP run, so both are sampled at identical times.

    Args:
        T: Final time, ``T > 0``.
        dt: Step, such that ``T / dt`` is an integer (to 1e-9).

    Returns:
        Times of shape ``(round(T / dt) + 1,)``.

    Raises:
        ValueError: If ``T`` or ``dt`` is not positive, or ``T / dt`` is
            not an integer.
    """
    if T <= 0 or dt <= 0:
        raise ValueError(f"Need T > 0 and dt > 0, got T={T}, dt={dt}.")
    n_steps = round(T / dt)
    if abs(n_steps * dt - T) > 1e-9 * max(1.0, T):
        raise ValueError(f"T={T} is not a multiple of dt={dt}.")
    return np.linspace(0.0, n_steps * dt, n_steps + 1)


def exact_time_evolution(
    hamiltonian: Any,
    psi0: ArrayLike,
    times: ArrayLike,
) -> np.ndarray:
    """Return ``|psi(t)> = exp(-i H t) |psi(0)>`` at each time.

    Uses ``scipy.sparse.linalg.expm_multiply``, which never forms the dense
    propagator. A uniform grid starting at 0 is evaluated in one call.

    Args:
        hamiltonian: Hermitian ``(dim, dim)`` matrix (sparse or dense), or
            a NetKet operator (converted with ``to_sparse``).
        psi0: Initial state of shape ``(dim,)``; it is not renormalized.
        times: Times of shape ``(n_times,)``.

    Returns:
        States of shape ``(n_times, dim)``, complex.

    Raises:
        ValueError: If shapes do not match.
    """
    H = hamiltonian.to_sparse() if hasattr(hamiltonian, "to_sparse") else hamiltonian
    psi = np.asarray(psi0, dtype=complex)
    t = np.asarray(times, dtype=float)
    if H.shape != (psi.size, psi.size) or psi.ndim != 1:
        raise ValueError(f"H has shape {H.shape}, psi0 has shape {psi.shape}.")
    if t.ndim != 1 or t.size == 0:
        raise ValueError("times must be a non-empty 1D array.")
    A = -1j * H
    uniform = t.size > 1 and t[0] == 0.0 and np.allclose(np.diff(t), t[1] - t[0])
    if uniform:
        return np.asarray(
            spla.expm_multiply(A, psi, start=0.0, stop=t[-1], num=t.size, endpoint=True)
        )
    return np.stack([spla.expm_multiply(A * ti, psi) for ti in t])


def expectation_values(states: ArrayLike, operator: Any) -> np.ndarray:
    """Return ``<psi|O|psi> / <psi|psi>`` for each state.

    Args:
        states: Array of shape ``(n_times, dim)`` or ``(dim,)``.
        operator: Hermitian ``(dim, dim)`` matrix or a NetKet operator.

    Returns:
        Real expectation values of shape ``(n_times,)`` (or a 0-d array).
        The imaginary part, zero for a Hermitian ``O``, is dropped.
    """
    op = operator.to_sparse() if hasattr(operator, "to_sparse") else operator
    psi = np.asarray(states, dtype=complex)
    O_psi = (op @ psi.T).T
    numerator = np.sum(psi.conj() * O_psi, axis=-1)
    norm = np.sum(np.abs(psi) ** 2, axis=-1)
    return np.asarray(numerator.real / norm)


__all__ = ["exact_time_evolution", "expectation_values", "time_grid"]
