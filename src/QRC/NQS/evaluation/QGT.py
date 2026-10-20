"""Geometry of the variational manifold from the quantum geometric tensor.

For a state ``psi_theta`` with log-derivatives
``O_k(s) = d log psi_theta(s) / d theta_k``, the quantum geometric tensor is

    S_kl = <O_k^* O_l> - <O_k^*> <O_l>        (averages over |psi(s)|^2)

It is Hermitian and positive semi-definite. It splits into

- the Fubini-Study metric (quantum Fisher information up to a factor 4)
  ``g = Re S``, real symmetric and positive semi-definite;
- the Berry curvature (symplectic form) ``Omega = 2 Im S``, real
  antisymmetric.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import netket as nk
import numpy as np


def qgt_matrix(S: Any) -> np.ndarray:
    """Return the QGT as a dense complex ``(P, P)`` NumPy array.

    Args:
        S: The QGT as a dense array (NumPy or JAX), a ``scipy.sparse``
            matrix, or a NetKet QGT linear operator (anything with
            ``to_dense()``, such as ``QGTJacobianDense`` or ``QGTOnTheFly``).

    Returns:
        A complex copy of the matrix.

    Raises:
        ValueError: If the matrix is not square.
    """
    if hasattr(S, "toarray"):  # scipy.sparse matrices and arrays
        dense = S.toarray()
    elif hasattr(S, "to_dense"):
        dense = S.to_dense()
    else:
        dense = S
    matrix = np.array(dense, dtype=complex)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"The QGT must be square, got shape {matrix.shape}.")
    return matrix


def fubini_study_metric(S: Any) -> np.ndarray:
    """Return ``g = Re S``, real symmetric, shape ``(P, P)``."""
    return qgt_matrix(S).real.copy()


def berry_curvature(S: Any) -> np.ndarray:
    """Return ``Omega = 2 Im S``, real antisymmetric, shape ``(P, P)``."""
    return 2.0 * qgt_matrix(S).imag


@dataclass(frozen=True)
class QGTAnalysis:
    """Geometric diagnostics of one QGT.

    Attributes:
        metric: Fubini-Study metric ``g = Re S``.
        curvature: Berry curvature ``Omega = 2 Im S``.
        eigenvalues: Eigenvalues of ``g`` in ascending order.
        rank: Number of eigenvalues of ``g`` above ``rtol * max eigenvalue``:
            the number of independent directions the parameters move the
            state in. ``P - rank`` counts redundant directions.
        condition_number: ``max / min`` eigenvalue of ``g``; ``inf`` when
            ``g`` is singular (rank < P).
        effective_condition_number: ``max / smallest kept`` eigenvalue,
            restricted to the ``rank`` non-singular directions.
        trace: ``Tr g``, the total Fisher information (up to a factor 4).
        curvature_norm: Frobenius norm of ``Omega``.
    """

    metric: np.ndarray
    curvature: np.ndarray
    eigenvalues: np.ndarray
    rank: int
    condition_number: float
    effective_condition_number: float
    trace: float
    curvature_norm: float

    @property
    def n_parameters(self) -> int:
        """Number of (complex) parameters ``P``."""
        return int(self.metric.shape[0])


def analyze_qgt(S: Any, *, rtol: float = 1e-10) -> QGTAnalysis:
    """Split the QGT into metric and curvature and compute diagnostics.

    Args:
        S: The QGT in any format accepted by :func:`qgt_matrix`.
        rtol: Relative cutoff for the rank, as a fraction of the largest
            eigenvalue of ``g``.

    Returns:
        The :class:`QGTAnalysis`.

    Raises:
        ValueError: If ``S`` is not square or not Hermitian.
    """
    matrix = qgt_matrix(S)
    scale = max(float(np.max(np.abs(matrix))), 1.0)
    if np.max(np.abs(matrix - matrix.conj().T)) > 1e-8 * scale:
        raise ValueError("The QGT must be Hermitian.")
    metric = matrix.real.copy()
    curvature = 2.0 * matrix.imag
    # eigvalsh reads one triangle; S Hermitian (checked above) makes g symmetric.
    eigenvalues = np.linalg.eigvalsh(metric)
    largest = float(eigenvalues[-1]) if eigenvalues.size else 0.0
    kept = eigenvalues[eigenvalues > rtol * largest] if largest > 0 else eigenvalues[:0]
    rank = int(kept.size)
    smallest = float(eigenvalues[0]) if eigenvalues.size else 0.0
    if rank == metric.shape[0] and smallest > 0:
        condition = largest / smallest
    else:
        condition = float("inf")
    effective = largest / float(kept[0]) if rank else float("inf")
    return QGTAnalysis(
        metric=metric,
        curvature=curvature,
        eigenvalues=eigenvalues,
        rank=rank,
        condition_number=float(condition),
        effective_condition_number=float(effective),
        trace=float(np.trace(metric)),
        curvature_norm=float(np.linalg.norm(curvature)),
    )


def qgt_of_state(
    state: nk.vqs.VariationalState,
    *,
    holomorphic: bool | None = None,
) -> np.ndarray:
    """Return the dense QGT of a NetKet variational state (no diagonal shift).

    Args:
        state: ``FullSumState`` (exact) or ``MCState`` (sampled estimate).
        holomorphic: Pass ``True`` for a holomorphic ansatz with complex
            parameters (complex RBM) to get the complex ``P x P`` QGT;
            ``None`` lets NetKet decide.

    Returns:
        The dense QGT.
    """
    qgt = state.quantum_geometric_tensor(
        nk.optimizer.qgt.QGTJacobianDense(holomorphic=holomorphic)
    )
    return qgt_matrix(qgt)


__all__ = [
    "QGTAnalysis",
    "analyze_qgt",
    "berry_curvature",
    "fubini_study_metric",
    "qgt_matrix",
    "qgt_of_state",
]
