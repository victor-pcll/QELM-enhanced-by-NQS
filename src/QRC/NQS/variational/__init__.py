"""NetKet 3 building blocks: systems, ansätze and the config-driven setup."""

from __future__ import annotations

from QRC.NQS.variational.ansatz import StateKind, rbm, variational_state
from QRC.NQS.variational.setup import NQSSetup, build_nqs, ising_system
from QRC.NQS.variational.systems import (
    mean_magnetization_z,
    spin_chain,
    spin_half_hilbert,
    transverse_field_ising,
)

__all__ = [
    "NQSSetup",
    "StateKind",
    "build_nqs",
    "ising_system",
    "mean_magnetization_z",
    "rbm",
    "spin_chain",
    "spin_half_hilbert",
    "transverse_field_ising",
    "variational_state",
]
