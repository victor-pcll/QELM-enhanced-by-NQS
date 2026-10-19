"""Hamiltonian models. Physics is not implemented; these types are routing stubs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self, TypedDict

import numpy as np
from omegaconf import DictConfig, OmegaConf


class EvolutionState(TypedDict):
    """Record of a requested evolution. No trajectory tensor is stored.

    Attributes:
        model: Simulation name, such as ``"ising"``.
        dt: Scalar integration step.
        n_steps: Scalar number of requested steps.
        status: ``"scaffold"`` until integration is implemented.
    """

    model: str
    dt: float
    n_steps: int
    status: str


class HamiltonianModel:
    """Time-evolution model selected by ``sim_type``.

    Attributes:
        name: Config key for this model.
        params: Scalar parameters from the matching YAML block, such as
            ``N`` and ``J``.
    """

    name: str

    def __init__(self, params: Mapping[str, Any]) -> None:
        """Store scalar Hamiltonian parameters.

        Args:
            params: Parameter names mapped to scalar values from YAML.
        """
        self.params: dict[str, Any] = dict(params)

    @classmethod
    def from_config(cls, cfg: DictConfig) -> Self:
        """Build the model from the Hydra block named ``cls.name``.

        Args:
            cfg: Composed Hydra config. ``cfg[cls.name]`` is a mapping of
                scalar parameters.

        Returns:
            An instance of this model class.

        Raises:
            TypeError: If the config block is not a mapping.
        """
        raw = OmegaConf.to_container(cfg[cls.name], resolve=True)
        if not isinstance(raw, dict):
            raise TypeError(f"{cls.name} config must be a mapping.")
        params = {str(key): value for key, value in raw.items()}
        return cls(params)

    def hamiltonian(self) -> np.ndarray:
        """Return the Hamiltonian matrix for ``self.params``.

        Returns:
            Not built by this scaffold. A later implementation would return
            an operator of shape ``(dim, dim)``.

        Raises:
            NotImplementedError: Always, until the operator is defined.
        """
        raise NotImplementedError(f"{self.name} Hamiltonian is not implemented.")

    def evolve(self, *, dt: float, n_steps: int) -> EvolutionState:
        """Record a requested evolution without integrating it.

        Args:
            dt: Scalar integration step.
            n_steps: Scalar number of steps. A later implementation would
                return states of shape ``(n_steps, dim)``. This scaffold
                does not allocate that tensor.

        Returns:
            An :class:`EvolutionState` recording ``dt`` and ``n_steps``.
        """
        return {
            "model": self.name,
            "dt": dt,
            "n_steps": n_steps,
            "status": "scaffold",
        }
