"""Reservoir Hamiltonians and observables used by the dynamics experiments."""

from __future__ import annotations

from typing import Literal

import netket as nk
import numpy as np
from numpy.typing import ArrayLike

from QELM.NQS.variational.systems import mean_magnetization_z

Axis = Literal["z", "x"]
Frame = Literal["lab", "hadamard"]


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
    frame: Frame = "lab",
) -> nk.operator.DiscreteOperator:
    """Return the all-to-all reservoir Hamiltonian.

        H = h * sum_i Z_i + sum_{i<j} J_ij X_i X_j

    with Pauli matrices. ``frame="hadamard"`` returns ``U H U`` with
    ``U = prod_i (X_i + Z_i) / sqrt 2`` instead, which swaps ``X`` and
    ``Z``: ``h * sum_i X_i + sum_{i<j} J_ij Z_i Z_j``. Same spectrum and
    dynamics; it is used to run TDVP from ``|0>^N``, which an RBM cannot
    write exactly, as ``U |0>^N = |+>^N``, which it can.

    ``h`` is the transverse field along Z (in the basis where the RBM is
    written) and ``J_ij`` the XX couplings. The operator is
    real symmetric (Hermitian) and commutes with the parity ``prod_i Z_i``.

    Args:
        hilbert: Spin-1/2 Hilbert space of ``N`` sites.
        h: Field strength.
        J: Scalar or ``(N, N)`` symmetric couplings, see
            :func:`coupling_matrix`.
        frame: ``"lab"`` or ``"hadamard"``.

    Returns:
        The Hamiltonian as a NetKet operator.

    Raises:
        ValueError: If ``frame`` is unknown.
    """
    if frame not in ("lab", "hadamard"):
        raise ValueError(f"frame must be 'lab' or 'hadamard', got {frame!r}.")
    n_spins = hilbert.size
    couplings = coupling_matrix(n_spins, J)
    sz = nk.operator.spin.sigmaz
    sx = nk.operator.spin.sigmax
    if frame == "hadamard":
        sz, sx = sx, sz
    H = float(h) * sz(hilbert, 0)
    for i in range(1, n_spins):
        H = H + float(h) * sz(hilbert, i)
    for i in range(n_spins):
        for j in range(i + 1, n_spins):
            if couplings[i, j] != 0.0:
                H = H + couplings[i, j] * (sx(hilbert, i) @ sx(hilbert, j))
    return H


def pure_field_hamiltonian(
    hilbert: nk.hilbert.Spin, *, h: float = 1.0, axis: Axis = "z"
) -> nk.operator.DiscreteOperator:
    """Return the quench Hamiltonian ``H_0 = -h * sum_i A_i`` with ``A = Z`` or ``X``.

    For ``h > 0`` its unique ground state is the product state with every
    spin along ``+A``: ``|0>^N`` for ``axis="z"`` and ``|+>^N`` for
    ``axis="x"``. The ground energy is ``-h N`` (``E_0 / N = -h``) and the
    gap is ``2 h``: flipping one spin costs ``2 h``. It is the initial
    Hamiltonian of the quench protocol of issue #26.

    Args:
        hilbert: Spin-1/2 Hilbert space of ``N`` sites.
        h: Field strength, ``h > 0``.
        axis: ``"z"`` or ``"x"``.

    Returns:
        ``H_0`` as a NetKet operator (real symmetric).

    Raises:
        ValueError: If ``h <= 0`` or ``axis`` is unknown.
    """
    if h <= 0:
        raise ValueError(f"h must be > 0 for a polarized ground state, got {h}.")
    pauli = {"z": nk.operator.spin.sigmaz, "x": nk.operator.spin.sigmax}
    if axis not in pauli:
        raise ValueError(f"axis must be 'z' or 'x', got {axis!r}.")
    a = pauli[axis]
    H = -float(h) * a(hilbert, 0)
    for i in range(1, hilbert.size):
        H = H - float(h) * a(hilbert, i)
    return H


__all__ = [
    "Axis",
    "Frame",
    "all_to_all_xx_hamiltonian",
    "coupling_matrix",
    "mean_magnetization_z",
    "pure_field_hamiltonian",
]
