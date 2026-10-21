"""Reservoir Hamiltonians and observables used by the dynamics experiments."""

from __future__ import annotations

import netket as nk
import numpy as np
from numpy.typing import ArrayLike

from QELM.NQS.variational.systems import mean_magnetization_z


def coupling_matrix(n_spins: int, J: float | ArrayLike) -> np.ndarray:
    """Return a symmetric ``(N, N)`` coupling matrix with a zero diagonal.

    Args:
        n_spins: Number of spins ``N``.
        J: A scalar (uniform all-to-all coupling) or an ``(N, N)`` real
            matrix. Only the strict upper triangle ``i < j`` is used; the
            result is its symmetric completion.

    Returns:
        The real symmetric coupling matrix.

    Raises:
        ValueError: If a matrix has the wrong shape, is complex, or is not
            symmetric on its off-diagonal part.
    """
    matrix = np.asarray(J)
    if matrix.ndim == 0:
        if np.iscomplexobj(matrix):
            raise ValueError("Couplings must be real.")
        full = np.full((n_spins, n_spins), float(matrix))
    else:
        if matrix.shape != (n_spins, n_spins):
            raise ValueError(
                f"J has shape {matrix.shape}, expected {(n_spins, n_spins)}."
            )
        if np.iscomplexobj(matrix):
            raise ValueError("Couplings must be real (H must be Hermitian).")
        full = matrix.astype(float)
        off = ~np.eye(n_spins, dtype=bool)
        if not np.allclose(full[off], full.T[off]):
            raise ValueError("J must be symmetric.")
    np.fill_diagonal(full, 0.0)
    return full


def all_to_all_xx_hamiltonian(
    hilbert: nk.hilbert.Spin,
    *,
    h: float,
    J: float | ArrayLike = 1.0,
) -> nk.operator.DiscreteOperator:
    """Return the all-to-all reservoir Hamiltonian.

        H = h * sum_i Z_i + sum_{i<j} J_ij X_i X_j

    with Pauli matrices. ``h`` is the transverse field along Z (in the basis
    where the RBM is written) and ``J_ij`` the XX couplings. The operator is
    real symmetric (Hermitian) and commutes with the parity ``prod_i Z_i``.

    Args:
        hilbert: Spin-1/2 Hilbert space of ``N`` sites.
        h: Field strength.
        J: Scalar or ``(N, N)`` symmetric couplings, see
            :func:`coupling_matrix`.

    Returns:
        The Hamiltonian as a NetKet operator.
    """
    n_spins = hilbert.size
    couplings = coupling_matrix(n_spins, J)
    sz = nk.operator.spin.sigmaz
    sx = nk.operator.spin.sigmax
    H = float(h) * sz(hilbert, 0)
    for i in range(1, n_spins):
        H = H + float(h) * sz(hilbert, i)
    for i in range(n_spins):
        for j in range(i + 1, n_spins):
            if couplings[i, j] != 0.0:
                H = H + couplings[i, j] * (sx(hilbert, i) @ sx(hilbert, j))
    return H


__all__ = ["all_to_all_xx_hamiltonian", "coupling_matrix", "mean_magnetization_z"]
