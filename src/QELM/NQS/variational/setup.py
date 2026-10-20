"""Build the NetKet 3 objects of one run from the Hydra config."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import flax.linen as nn
import netket as nk
from omegaconf import DictConfig

from QELM.NQS.variational.ansatz import StateKind, rbm, variational_state
from QELM.NQS.variational.systems import (
    spin_chain,
    spin_half_hilbert,
    transverse_field_ising,
)


@dataclass(frozen=True)
class NQSSetup:
    """NetKet objects for one physical system and one ansatz.

    Attributes:
        hilbert: Spin-1/2 Hilbert space of ``N`` sites.
        graph: Lattice of the couplings.
        hamiltonian: Hermitian NetKet operator.
        model: Flax module of the ansatz.
        state: Variational state with seeded parameters.
    """

    hilbert: nk.hilbert.Spin
    graph: nk.graph.Graph
    hamiltonian: nk.operator.DiscreteOperator
    model: nn.Module
    state: nk.vqs.VariationalState


def ising_system(
    sim: Mapping[str, Any],
) -> tuple[nk.hilbert.Spin, nk.graph.Graph, nk.operator.DiscreteOperator]:
    """Build the transverse-field Ising chain from an ``ising`` config block.

    Args:
        sim: Config block or dict with scalar ``N``, ``J``, ``h`` and
            optional ``pbc`` (default ``False``).

    Returns:
        The Hilbert space, the chain and the Hamiltonian.
    """
    n_spins = int(sim["N"])
    graph = spin_chain(n_spins, pbc=bool(sim.get("pbc", False)))
    hilbert = spin_half_hilbert(n_spins)
    hamiltonian = transverse_field_ising(
        hilbert, graph, h=float(sim["h"]), J=float(sim["J"])
    )
    return hilbert, graph, hamiltonian


_SYSTEMS = {"ising": ising_system}


def build_nqs(cfg: DictConfig) -> NQSSetup:
    """Build the system named by ``cfg.sim_type`` and the ansatz in ``cfg.nqs``.

    Args:
        cfg: Composed Hydra config. ``cfg[cfg.sim_type]`` holds the system
            parameters, ``cfg.nqs`` the ansatz (``model``, ``alpha``,
            ``param_dtype``, ``state``, ``n_samples``) and ``cfg.seed`` the
            parameter-initialization seed.

    Returns:
        The NetKet objects of the run.

    Raises:
        NotImplementedError: If the simulation type has no NetKet system.
        ValueError: If ``cfg.nqs.model`` is unknown.
    """
    sim_type = str(cfg.sim_type)
    try:
        system = _SYSTEMS[sim_type]
    except KeyError as exc:
        raise NotImplementedError(
            f"No NetKet system for sim_type {sim_type!r}."
        ) from exc
    hilbert, graph, hamiltonian = system(cfg[sim_type])

    nqs = cfg.nqs
    if str(nqs.model) != "rbm":
        raise ValueError(f"Unknown NQS model {nqs.model!r}. Known: rbm.")
    model = rbm(alpha=int(nqs.alpha), param_dtype=str(nqs.param_dtype))
    state = variational_state(
        hilbert,
        model,
        seed=int(cfg.seed),
        kind=cast(StateKind, str(nqs.state)),
        n_samples=int(nqs.n_samples),
    )
    return NQSSetup(
        hilbert=hilbert,
        graph=graph,
        hamiltonian=hamiltonian,
        model=model,
        state=state,
    )


__all__ = ["NQSSetup", "build_nqs", "ising_system"]
