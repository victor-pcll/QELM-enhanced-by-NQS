"""Energy variance: a convergence diagnostic for (approximate) eigenstates."""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_energy_variance(variational_state: Any, hamiltonian: Any) -> float:
    """Return the energy variance ``sigma_E^2 = <H^2> - <H>^2`` of a state.

    The variance is read from NetKet's statistics of the local energy
    ``E_loc(s) = <s|H|psi> / <s|psi>``:

        sigma_E^2 = sum_s |psi(s)|^2 |E_loc(s) - <H>|^2

    which equals ``<H^2> - <H>^2`` for a Hermitian ``H`` without building the
    ``H^2`` operator (its number of non-zero elements grows quickly with N).
    For a ``FullSumState`` the sum is exact; for an ``MCState`` it is the
    sample estimate over ``n_samples`` configurations. It vanishes exactly
    for any eigenstate of ``H`` (ground or excited) and is non-negative.

    Args:
        variational_state: NetKet ``FullSumState`` or ``MCState``.
        hamiltonian: Hermitian NetKet operator on the same Hilbert space.

    Returns:
        The variance as a non-negative float. Round-off below zero (at most
        ``1e-10 * max(1, <H>^2)``) is clipped to 0.

    Raises:
        ValueError: If the estimate is clearly negative, which no physical
            state can give.
    """
    stats = variational_state.expect(hamiltonian)
    variance = float(np.real(np.asarray(stats.variance)))
    mean = complex(np.asarray(stats.mean))
    return _non_negative(variance, scale=abs(mean) ** 2)


def exact_energy_variance(psi: Any, hamiltonian: Any) -> float:
    """Return ``||H psi||^2 / ||psi||^2 - (<psi|H|psi> / ||psi||^2)^2``.

    Dense reference for small systems, independent of NetKet's statistics.

    Args:
        psi: State vector of shape ``(dim,)``.
        hamiltonian: Hermitian ``(dim, dim)`` matrix or a NetKet operator.

    Returns:
        The variance as a non-negative float.
    """
    H = hamiltonian.to_sparse() if hasattr(hamiltonian, "to_sparse") else hamiltonian
    vec = np.asarray(psi, dtype=complex)
    norm = float(np.vdot(vec, vec).real)
    h_psi = H @ vec
    mean = float(np.vdot(vec, h_psi).real) / norm
    second = float(np.vdot(h_psi, h_psi).real) / norm
    return _non_negative(second - mean**2, scale=second)


def _non_negative(variance: float, *, scale: float) -> float:
    """Clip round-off below zero; a clearly negative variance is a bug."""
    tolerance = 1e-10 * max(1.0, scale)
    if variance < -tolerance:
        raise ValueError(f"Negative energy variance {variance:.3e}.")
    return max(variance, 0.0)


__all__ = ["compute_energy_variance", "exact_energy_variance"]
