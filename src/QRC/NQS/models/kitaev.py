"""Kitaev model scaffold."""

from __future__ import annotations

from QRC.NQS.models.base import HamiltonianModel


class KitaevModel(HamiltonianModel):
    """Kitaev Hamiltonian and time evolution.

    Parameters belong in ``conf/sim_type/kitaev.yaml`` and are not
    implemented. The operator of shape ``(dim, dim)`` is not built.
    """

    name = "kitaev"
