"""Training algorithms. Fit and predict are stubs."""

from __future__ import annotations

from typing import Self

import numpy as np
from omegaconf import DictConfig


class Trainer:
    """Read parameters from config and expose fit/predict hooks.

    Attributes:
        name: Config key for this trainer.
        learning_rate: Scalar step size from the simulation block, or
            ``None`` when that block does not define one.
    """

    name: str

    def __init__(self, learning_rate: float | None = None) -> None:
        """Store the scalar learning rate.

        Args:
            learning_rate: Optimizer step size, or ``None`` if unset.
        """
        self.learning_rate = learning_rate

    @classmethod
    def from_config(cls, cfg: DictConfig) -> Self:
        """Read ``learning_rate`` from the selected simulation block.

        Args:
            cfg: Composed Hydra config. ``cfg[cfg.sim_type]`` may contain a
                scalar ``learning_rate``.

        Returns:
            An instance of this trainer class.
        """
        sim = cfg[str(cfg.sim_type)]
        raw_learning_rate = sim.get("learning_rate")
        learning_rate = None if raw_learning_rate is None else float(raw_learning_rate)
        return cls(learning_rate=learning_rate)

    def fit(
        self,
        features: np.ndarray | None,
        targets: np.ndarray | None,
    ) -> str:
        """Record a fit request without estimating weights.

        Args:
            features: Design matrix of shape ``(n_samples, n_features)``,
                or ``None`` when no dataset has been built.
            targets: Labels of shape ``(n_samples,)`` or
                ``(n_samples, n_outputs)``, or ``None``.

        Returns:
            The status string ``"scaffold"``.
        """
        return "scaffold"

    def predict(self, features: np.ndarray | None) -> str:
        """Record a predict request without emitting a tensor.

        Args:
            features: Design matrix of shape ``(n_samples, n_features)``,
                or ``None``.

        Returns:
            The status string ``"scaffold"``. A later implementation would
            return predictions of shape ``(n_samples, n_outputs)``.
        """
        return "scaffold"

    def state_dict(self) -> dict[str, np.ndarray]:
        """Return arrays to store in a checkpoint.

        The scaffold has not estimated a weight vector. ``weights`` is a
        length-0 ``float64`` array. A later fit would store shape
        ``(n_features,)`` or ``(n_features, n_outputs)``.

        Returns:
            Mapping whose ``weights`` entry is the trainable state.
        """
        return {"weights": np.zeros((0,), dtype=np.float64)}
