"""Ising model scaffold."""

from __future__ import annotations

from qrc_env_template.models.base import HamiltonianModel


class TransverseFieldIsingModel(HamiltonianModel):
    """Transverse Field Ising Hamiltonian and time evolution.

    ``N``, ``J``, and ``learning_rate`` are scalars from
    ``conf/sim_type/ising.yaml``. The operator of shape ``(dim, dim)`` is
    not implemented.
    """

    name = "ising"
