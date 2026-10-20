"""Linear readout of a QELM: ridge regression on the reservoir features.

The only trained part of a QELM. With features ``F`` (shape
``(n_samples, n_features)``) and targets ``y``, the weights minimize

    ||[F, 1] w - y||^2 + lam ||w_F||^2

(the bias is not penalized). The closed form is solved on centered data:
``w_F = (Fc^T Fc + lam I)^{-1} Fc^T yc`` and ``b = mean(y) - mean(F) w_F``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike


@dataclass
class RidgeReadout:
    """Ridge regression with an unpenalized bias.

    Attributes:
        lam: Ridge penalty ``lam >= 0``. ``0`` gives the least-squares
            solution of minimal norm.
        weights: Fitted weights, shape ``(n_features,)`` or
            ``(n_features, n_outputs)``; ``None`` before :meth:`fit`.
        bias: Fitted bias, scalar or shape ``(n_outputs,)``.
    """

    lam: float = 1e-6
    weights: np.ndarray | None = field(default=None, init=False)
    bias: np.ndarray | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.lam < 0:
            raise ValueError(f"lam must be >= 0, got {self.lam}.")

    def fit(self, features: ArrayLike, targets: ArrayLike) -> RidgeReadout:
        """Fit the weights and the bias.

        Args:
            features: Real array of shape ``(n_samples, n_features)``.
            targets: Real array of shape ``(n_samples,)`` or
                ``(n_samples, n_outputs)``.

        Returns:
            ``self``.

        Raises:
            ValueError: If shapes do not match.
        """
        F = np.asarray(features, dtype=float)
        y = np.asarray(targets, dtype=float)
        if F.ndim != 2 or y.shape[:1] != F.shape[:1]:
            raise ValueError(f"features {F.shape} and targets {y.shape} do not match.")
        F_mean = F.mean(axis=0)
        y_mean = y.mean(axis=0)
        Fc = F - F_mean
        yc = y - y_mean
        if self.lam > 0:
            gram = Fc.T @ Fc + self.lam * np.eye(F.shape[1])
            self.weights = np.linalg.solve(gram, Fc.T @ yc)
        else:
            self.weights = np.linalg.lstsq(Fc, yc, rcond=None)[0]
        self.bias = np.asarray(y_mean - F_mean @ self.weights)
        return self

    def predict(self, features: ArrayLike) -> np.ndarray:
        """Return ``F w + b``.

        Args:
            features: Array of shape ``(n_samples, n_features)``.

        Returns:
            Predictions of shape ``(n_samples,)`` or ``(n_samples, n_outputs)``.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if self.weights is None or self.bias is None:
            raise RuntimeError("Call fit before predict.")
        return np.asarray(features, dtype=float) @ self.weights + self.bias


__all__ = ["RidgeReadout"]
