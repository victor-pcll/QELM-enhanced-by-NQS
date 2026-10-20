"""Hidden representations of an RBM: SVD of its weight matrix.

For ``psi(s) = exp(sum_i a_i s_i) prod_j 2 cosh(b_j + sum_i W_ij s_i)`` the
complex weights ``W`` (shape ``(N, M)``: visible spins by hidden units, the
NetKet convention) carry the correlations between spins. Writing
``W = U diag(sigma) V^†``, the columns of ``U`` are spin patterns read by
the hidden layer and ``sigma`` ranks how strongly each pattern is used.

The spectrum is invariant under relabelling the hidden units (a gauge
symmetry of the RBM: ``psi`` does not change) and under permuting the spins
together with the state.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


def extract_rbm_weights(variational_state: Any) -> np.ndarray:
    """Return the weight matrix ``W`` of a NetKet RBM state, shape ``(N, M)``.

    Args:
        variational_state: NetKet variational state whose model is an RBM
            (``nk.models.RBM``), or directly its parameter PyTree.

    Returns:
        A NumPy copy of ``W`` with ``W[i, j]`` coupling spin ``i`` to hidden
        unit ``j``; complex when the RBM has complex parameters.

    Raises:
        KeyError: If the parameters have no RBM kernel.
    """
    params = getattr(variational_state, "parameters", variational_state)
    if not isinstance(params, Mapping):
        raise KeyError("Expected an RBM parameter mapping.")
    try:
        kernel = params["Dense"]["kernel"]
    except (KeyError, TypeError) as exc:
        raise KeyError(
            "No RBM weights found: expected parameters['Dense']['kernel']."
        ) from exc
    return np.array(kernel)


@dataclass(frozen=True)
class WeightSVD:
    """Thin SVD ``W = U diag(singular_values) Vh``.

    Attributes:
        U: Left singular vectors, shape ``(N, K)``, ``K = min(N, M)``;
            orthonormal columns (spin patterns).
        singular_values: Real, non-negative, sorted in descending order,
            shape ``(K,)``.
        Vh: Right singular vectors (conjugate-transposed), shape ``(K, M)``;
            orthonormal rows (hidden-unit mixtures).
    """

    U: np.ndarray
    singular_values: np.ndarray
    Vh: np.ndarray

    def reconstruct(self) -> np.ndarray:
        """Return ``U diag(singular_values) Vh``."""
        return (self.U * self.singular_values) @ self.Vh

    @property
    def weight_fractions(self) -> np.ndarray:
        """Return ``sigma_k^2 / sum sigma^2``, the share of ``||W||_F^2`` per mode."""
        power = self.singular_values**2
        total = power.sum()
        return power / total if total > 0 else np.zeros_like(power)

    def n_dominant_modes(self, threshold: float = 0.9) -> int:
        """Return the fewest modes that carry ``threshold`` of ``||W||_F^2``.

        Args:
            threshold: Fraction in ``(0, 1]``.

        Returns:
            A count between 1 and ``K`` (0 for ``W = 0``).

        Raises:
            ValueError: If ``threshold`` is outside ``(0, 1]``.
        """
        if not 0 < threshold <= 1:
            raise ValueError(f"threshold must be in (0, 1], got {threshold}.")
        fractions = self.weight_fractions
        if not np.any(fractions):
            return 0
        cumulative = np.cumsum(fractions)
        return int(np.searchsorted(cumulative, threshold - 1e-12) + 1)


def analyze_weights_svd(W: Any) -> WeightSVD:
    """Return the thin SVD of a (complex) weight matrix.

    Args:
        W: Matrix of shape ``(N, M)``, real or complex.

    Returns:
        The :class:`WeightSVD`; singular values are real, non-negative and
        in descending order (dominant modes first).

    Raises:
        ValueError: If ``W`` is not 2D.
    """
    matrix = np.asarray(W)
    if matrix.ndim != 2:
        raise ValueError(f"W must be a matrix, got shape {matrix.shape}.")
    U, singular_values, Vh = np.linalg.svd(matrix, full_matrices=False)
    return WeightSVD(U=U, singular_values=singular_values, Vh=Vh)


__all__ = ["WeightSVD", "analyze_weights_svd", "extract_rbm_weights"]
