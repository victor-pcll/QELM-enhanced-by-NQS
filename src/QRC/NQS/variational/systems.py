"""Hilbert spaces, lattices and Hamiltonians as NetKet 3 objects."""

from __future__ import annotations

import netket as nk


def spin_chain(n_spins: int, *, pbc: bool = False) -> nk.graph.Graph:
    """Return a 1D chain of ``n_spins`` sites.

    Args:
        n_spins: Number of sites, at least 2.
        pbc: Periodic boundary conditions when true.

    Returns:
        A NetKet chain graph. With ``pbc=False`` it has ``n_spins - 1``
        edges, with ``pbc=True`` it has ``n_spins`` (for ``n_spins > 2``).

    Raises:
        ValueError: If ``n_spins < 2``.
    """
    if n_spins < 2:
        raise ValueError(f"A chain needs at least 2 sites, got {n_spins}.")
    return nk.graph.Chain(length=n_spins, pbc=pbc)


def spin_half_hilbert(n_spins: int) -> nk.hilbert.Spin:
    """Return the Hilbert space of ``n_spins`` spin-1/2, dimension ``2**n_spins``.

    Args:
        n_spins: Number of spins, at least 1.

    Returns:
        A NetKet ``Spin(s=1/2)`` Hilbert space.

    Raises:
        ValueError: If ``n_spins < 1``.
    """
    if n_spins < 1:
        raise ValueError(f"Need at least one spin, got {n_spins}.")
    return nk.hilbert.Spin(s=0.5, N=n_spins)


def transverse_field_ising(
    hilbert: nk.hilbert.Spin,
    graph: nk.graph.Graph,
    *,
    h: float,
    J: float,
) -> nk.operator.DiscreteOperator:
    """Return the transverse-field Ising Hamiltonian on ``graph``.

    NetKet convention, with Pauli matrices X and Z:

        H = -h * sum_i X_i + J * sum_<ij> Z_i Z_j

    ``J > 0`` is antiferromagnetic. The operator is Hermitian and real.

    Args:
        hilbert: Spin-1/2 Hilbert space with ``graph.n_nodes`` sites.
        graph: Lattice whose edges carry the ``Z_i Z_j`` couplings.
        h: Transverse field.
        J: Nearest-neighbour coupling.

    Returns:
        The Hamiltonian as a NetKet operator.

    Raises:
        ValueError: If the Hilbert space and graph sizes differ.
    """
    if hilbert.size != graph.n_nodes:
        raise ValueError(
            f"Hilbert space has {hilbert.size} sites, graph has {graph.n_nodes}."
        )
    return nk.operator.Ising(hilbert=hilbert, graph=graph, h=float(h), J=float(J))


def mean_magnetization_z(hilbert: nk.hilbert.Spin) -> nk.operator.DiscreteOperator:
    """Return ``(1/N) * sum_i Z_i``, with eigenvalues in ``[-1, 1]``.

    Args:
        hilbert: Spin-1/2 Hilbert space of ``N`` sites.

    Returns:
        The mean longitudinal magnetization operator.
    """
    n_spins = hilbert.size
    total = nk.operator.spin.sigmaz(hilbert, 0)
    for i in range(1, n_spins):
        total = total + nk.operator.spin.sigmaz(hilbert, i)
    return total / n_spins


__all__ = [
    "mean_magnetization_z",
    "spin_chain",
    "spin_half_hilbert",
    "transverse_field_ising",
]
