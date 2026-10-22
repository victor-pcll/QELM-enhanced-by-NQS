"""Quench initial states: the ground state of ``H_0 = -h sum_i A_i``, ``A = Z, X``.

Issue #26 starts the reservoir from the ground state of a pure field: a
product state (no entanglement), so every correlation measured after the
quench is built by the reservoir dynamics alone.

- ``axis="z"`` (the default of the reservoir): ``H_0 = -h sum_i Z_i`` and
  ``|0>^N``, every spin along +Z.
- ``axis="x"``: ``H_0 = -h sum_i X_i`` and ``|+>^N``, every spin along +X.
  It is invariant under the global flip ``P = prod_i X_i``.

**An RBM cannot write ``|0>^N`` exactly.** A single configuration needs
``|a_i| -> infinity``; with a finite visible bias the Born distribution
sits on one configuration, the quantum geometric tensor (a covariance of
log-derivatives over that distribution) collapses, and TDVP freezes (with
``a_i = 5``, infidelity 1e-8, the TDVP features do not move at all). The
Z quench is therefore simulated in the Hadamard frame
``U = prod_i (X_i + Z_i) / sqrt 2``: ``U |0>^N = |+>^N``, ``U Z_i U = X_i``
and ``U X_i U = Z_i``, so the features ``<Z_i>(t)`` from ``|0>^N`` under
``H`` equal ``<X_i>(t)`` from ``|+>^N`` under ``U H U``. This is an exact
change of basis, not an approximation; ED in both frames is compared in
the tests.

``|+>^N`` is prepared in two independent ways that the tests compare:

- as a vector, by applying a Hadamard gate ``(X + Z) / sqrt 2`` to every
  spin of ``|0>^N`` (``|0>`` is the ``Z = +1`` eigenstate);
- as an exact RBM, whose amplitude
  ``psi(s) = exp(sum_i a_i s_i) prod_j 2 cosh(b_j + sum_i W_ij s_i)``
  is the same for every configuration when ``a = 0`` and ``W = 0``.

**The RBM must sit at a regular point of its manifold.** The naive choice
``a = b = W = 0`` is exactly ``|+>^N`` but its log-derivatives are
``s_i`` (from ``a_i``), constants (from ``b_j``) and ``s_i tanh(0) = 0``
(from ``W_ij``): the tangent space only holds single-spin directions.
Pair terms such as ``J_ij Z_i Z_j |+>^N`` (which give ``s_i s_j psi``) are
projected to zero and TDVP stays frozen. Here hidden unit ``j`` is
connected only to visible spin ``j`` with weight ``w`` and zero bias:
``cosh(w s_j) = cosh(w)`` for ``s_j = +-1``, so the state is still exactly
``|+>^N``, while ``d log psi / d W_kj = s_k s_j tanh(w)`` spans all pair
directions. With ``w = 1`` TDVP follows ED to about 1e-4 on the N = 4
reservoir of the tests, for both quench axes; with ``w = 0`` it misses by
order 1 for the Z quench.
"""

from __future__ import annotations

import jax.numpy as jnp
import netket as nk
import numpy as np
from numpy.typing import ArrayLike

from QELM.NQS.measurements.hamiltonians import Axis
from QELM.NQS.variational import rbm, variational_state

PLUS_PAIR_WEIGHT = 1.0


def hadamard_plus_state(hilbert: nk.hilbert.Spin) -> np.ndarray:
    """Return ``H^{(x)N} |0>^N = |+>^N`` as a normalized vector.

    ``|0>`` is the eigenstate ``Z = +1`` of NetKet's ``sigmaz`` and the
    Hadamard gate on spin ``i`` is ``(X_i + Z_i) / sqrt 2``, built from
    NetKet's own Pauli operators so that no basis ordering is assumed.

    Args:
        hilbert: Spin-1/2 Hilbert space of ``N`` sites.

    Returns:
        Real vector of shape ``(2**N,)`` and unit norm, in NetKet's basis.
    """
    psi = zero_state(hilbert)
    for i in range(hilbert.size):
        gate = (
            nk.operator.spin.sigmax(hilbert, i) + nk.operator.spin.sigmaz(hilbert, i)
        ).to_sparse() / np.sqrt(2.0)
        psi = gate @ psi
    return np.asarray(psi.real)


def zero_state(hilbert: nk.hilbert.Spin) -> np.ndarray:
    """Return ``|0>^N`` (every spin in the ``Z = +1`` eigenstate) as a vector.

    Args:
        hilbert: Spin-1/2 Hilbert space of ``N`` sites.

    Returns:
        Real basis vector of shape ``(2**N,)``.
    """
    up = np.ones((1, hilbert.size))
    psi = np.zeros(hilbert.n_states)
    psi[int(np.asarray(hilbert.states_to_numbers(up)).ravel()[0])] = 1.0
    return psi


def quench_ground_state(hilbert: nk.hilbert.Spin, axis: Axis = "z") -> np.ndarray:
    """Return the ground state of ``H_0 = -h sum_i A_i`` (any ``h > 0``).

    Args:
        hilbert: Spin-1/2 Hilbert space of ``N`` sites.
        axis: ``"z"`` gives ``|0>^N``, ``"x"`` gives ``|+>^N``.

    Returns:
        Normalized real vector of shape ``(2**N,)``.

    Raises:
        ValueError: If ``axis`` is unknown.
    """
    if axis == "z":
        return zero_state(hilbert)
    if axis == "x":
        return hadamard_plus_state(hilbert)
    raise ValueError(f"axis must be 'z' or 'x', got {axis!r}.")


def plus_state_rbm(
    hilbert: nk.hilbert.Spin,
    *,
    alpha: int = 1,
    seed: int = 45,
    pair_weight: float = PLUS_PAIR_WEIGHT,
) -> nk.vqs.FullSumState:
    """Return a complex RBM ``FullSumState`` that is exactly ``|+>^N``.

    Visible and hidden biases are zero, and hidden unit ``j`` is connected
    to the single visible spin ``j mod N`` with weight ``pair_weight``.
    Since ``s = +-1`` and ``cosh`` is even, ``cosh(w s) = cosh(w)``: every
    configuration has the same amplitude. See the module docstring for why
    ``w != 0`` is needed by TDVP.

    Args:
        hilbert: Spin-1/2 Hilbert space of ``N`` sites.
        alpha: Hidden-unit density.
        seed: Seed of NetKet's initialization; every parameter is then
            overwritten, so the state does not depend on it.
        pair_weight: ``w``. ``0`` is the singular point of the manifold and
            is allowed only to expose it.

    Returns:
        The variational state, summed exactly over the ``2**N``
        configurations.
    """
    state = variational_state(hilbert, rbm(alpha=alpha), seed=seed)
    assert isinstance(state, nk.vqs.FullSumState)
    params = state.parameters
    n_hidden = int(np.asarray(params["Dense"]["bias"]).size)
    kernel = np.zeros((hilbert.size, n_hidden))
    kernel[np.arange(n_hidden) % hilbert.size, np.arange(n_hidden)] = pair_weight
    state.parameters = {
        "Dense": {
            "bias": jnp.zeros_like(params["Dense"]["bias"]),
            "kernel": jnp.asarray(kernel, dtype=params["Dense"]["kernel"].dtype),
        },
        "visible_bias": jnp.zeros_like(params["visible_bias"]),
    }
    return state


def entanglement_entropy(psi: ArrayLike, n_spins: int, subsystem: ArrayLike) -> float:
    """Return the von Neumann entropy ``S_A = -Tr rho_A ln rho_A`` (in nats).

    The vector is read in NetKet's basis order, where site 0 is the most
    significant digit (``hilbert.all_states()``), so it reshapes to
    ``(2,) * N`` with axis ``i`` for site ``i``.

    Args:
        psi: State vector of shape ``(2**N,)``; it is normalized here.
        n_spins: Number of spins ``N``.
        subsystem: Sites of ``A``.

    Returns:
        ``S_A`` from the Schmidt coefficients; ``0`` for a product state
        across the cut, ``ln 2`` per maximally entangled pair it splits.

    Raises:
        ValueError: If shapes or sites are inconsistent.
    """
    vec = np.asarray(psi, dtype=complex)
    if vec.shape != (2**n_spins,):
        raise ValueError(f"psi has shape {vec.shape}, expected {(2**n_spins,)}.")
    sites = sorted({int(i) for i in np.ravel(np.asarray(subsystem))})
    if any(i < 0 or i >= n_spins for i in sites):
        raise ValueError(f"Sites {sites} are not all in [0, {n_spins}).")
    rest = [i for i in range(n_spins) if i not in sites]
    tensor = (vec / np.linalg.norm(vec)).reshape((2,) * n_spins)
    matrix = np.transpose(tensor, sites + rest).reshape(2 ** len(sites), -1)
    p = np.linalg.svd(matrix, compute_uv=False) ** 2
    p = p[p > 1e-15]
    return float(max(0.0, -np.sum(p * np.log(p))))


__all__ = [
    "PLUS_PAIR_WEIGHT",
    "entanglement_entropy",
    "hadamard_plus_state",
    "plus_state_rbm",
    "quench_ground_state",
    "zero_state",
]
