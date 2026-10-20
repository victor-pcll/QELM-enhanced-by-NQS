"""Fidelity susceptibility chi_F = v^† S v (issue #5)."""

import jax
import netket as nk
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

from QRC.NQS.evaluation.fidelity import (
    compute_fidelity_susceptibility,
    fidelity_susceptibility_of_state,
    flatten_direction,
)
from QRC.NQS.evaluation.QGT import fubini_study_metric, qgt_of_state
from QRC.NQS.variational import (
    rbm,
    spin_chain,
    spin_half_hilbert,
    transverse_field_ising,
    variational_state,
)

N = 4


def _state(seed: int = 45):
    """Complex RBM moved to a generic (regular) point of its manifold."""
    hilbert = spin_half_hilbert(N)
    state = variational_state(hilbert, rbm(), seed=seed)
    flat, unravel = ravel_pytree(state.parameters)
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    noise = jax.random.normal(k1, flat.shape) + 1j * jax.random.normal(k2, flat.shape)
    state.parameters = unravel(flat + 0.1 * noise)
    return state


def _infidelity_per_eps2(state, v: np.ndarray, eps: float) -> float:
    """(1 - |<psi(theta)|psi(theta + eps v)>|^2) / eps^2 with exact vectors."""
    flat, unravel = ravel_pytree(state.parameters)
    moved = nk.vqs.FullSumState(
        state.hilbert, state.model, variables={"params": unravel(flat + eps * v)}
    )
    overlap = np.vdot(np.asarray(state.to_array()), np.asarray(moved.to_array()))
    return float((1.0 - abs(overlap) ** 2) / eps**2)


@pytest.fixture(scope="module")
def setup():
    state = _state()
    S = qgt_of_state(state, holomorphic=True)
    return state, S


def _random_directions(P: int, seed: int):
    rng = np.random.default_rng(seed)
    real = rng.normal(size=P)
    return real, real + 1j * rng.normal(size=P)


@pytest.mark.parametrize("metric_only", [False, True])
def test_positivity(setup, metric_only: bool) -> None:
    _, S = setup
    for seed in range(5):
        for v in _random_directions(S.shape[0], seed):
            chi = compute_fidelity_susceptibility(S, v, metric_only=metric_only)
            assert chi >= 0.0
            assert chi > 1e-6


@pytest.mark.parametrize("metric_only", [False, True])
@pytest.mark.parametrize("alpha", [3.0, -0.5, 2.0 - 1.5j, 1j])
def test_quadratic_scaling(setup, metric_only: bool, alpha: complex) -> None:
    _, S = setup
    _, v = _random_directions(S.shape[0], 7)
    base = compute_fidelity_susceptibility(S, v, metric_only=metric_only)
    scaled = compute_fidelity_susceptibility(S, alpha * v, metric_only=metric_only)
    assert scaled == pytest.approx(abs(alpha) ** 2 * base, rel=1e-12)


@pytest.mark.parametrize("metric_only", [False, True])
def test_basis_vector_gives_metric_diagonal(setup, metric_only: bool) -> None:
    _, S = setup
    g = fubini_study_metric(S)
    for k in range(S.shape[0]):
        e_k = np.zeros(S.shape[0])
        e_k[k] = 1.0
        chi = compute_fidelity_susceptibility(S, e_k, metric_only=metric_only)
        assert chi == pytest.approx(g[k, k], rel=1e-12, abs=1e-15)


@pytest.mark.parametrize("metric_only", [False, True])
def test_zero_vector_returns_exactly_zero(setup, metric_only: bool) -> None:
    _, S = setup
    chi = compute_fidelity_susceptibility(
        S, np.zeros(S.shape[0], dtype=complex), metric_only=metric_only
    )
    assert chi == 0.0
    assert isinstance(chi, float)


def test_real_direction_matches_infidelity(setup) -> None:
    state, S = setup
    v, _ = _random_directions(S.shape[0], 11)
    chi = compute_fidelity_susceptibility(S, v)
    assert compute_fidelity_susceptibility(S, v, metric_only=True) == pytest.approx(
        chi, rel=1e-12
    )
    assert _infidelity_per_eps2(state, v, 1e-4) == pytest.approx(chi, rel=1e-3)


def test_complex_direction_needs_the_berry_term(setup) -> None:
    """For complex v, v^† S v (not v^† Re[S] v) is the infidelity."""
    state, S = setup
    _, v = _random_directions(S.shape[0], 12)
    measured = _infidelity_per_eps2(state, v, 1e-4)
    full = compute_fidelity_susceptibility(S, v)
    metric_only = compute_fidelity_susceptibility(S, v, metric_only=True)
    assert measured == pytest.approx(full, rel=1e-3)
    assert abs(metric_only - measured) > 0.05 * measured


def test_tdvp_velocity_gives_energy_variance(setup) -> None:
    """Along the exact TDVP velocity, chi_F = Var(H) on a complete manifold.

    theta_dot = -i S^+ F, with F the energy forces, so chi_F(theta_dot) =
    F^† S^+ F: the energy variance captured by the manifold. For N = 4 the
    RBM (24 parameters) spans the 16-dimensional Hilbert space, so it is
    the full variance <H^2> - <H>^2.
    """
    state, S = setup
    hilbert = state.hilbert
    H = transverse_field_ising(hilbert, spin_chain(N, pbc=True), h=1.0, J=1.0)
    stats, forces = state.expect_and_forces(H)
    F = flatten_direction(forces)
    theta_dot = -1j * np.linalg.pinv(S, rcond=1e-12, hermitian=True) @ F
    chi = compute_fidelity_susceptibility(S, theta_dot)

    psi = np.asarray(state.to_array())
    H_dense = np.asarray(H.to_dense())
    variance = np.vdot(H_dense @ psi, H_dense @ psi).real - (
        np.vdot(psi, H_dense @ psi).real ** 2
    )
    assert chi == pytest.approx(variance, rel=1e-8)
    assert float(np.real(stats.variance)) == pytest.approx(variance, rel=1e-10)


def test_pytree_directions_from_vmc_gradient(setup) -> None:
    state, S = setup
    H = transverse_field_ising(state.hilbert, spin_chain(N, pbc=True), h=1.0, J=1.0)
    _, gradient = state.expect_and_grad(H)
    flat = flatten_direction(gradient)
    assert flat.shape == (S.shape[0],)
    from_tree = compute_fidelity_susceptibility(S, gradient)
    assert from_tree == pytest.approx(compute_fidelity_susceptibility(S, flat))
    assert fidelity_susceptibility_of_state(state, gradient) == pytest.approx(
        from_tree, rel=1e-12
    )
    assert from_tree > 0


def test_size_mismatch_raises(setup) -> None:
    _, S = setup
    with pytest.raises(ValueError):
        compute_fidelity_susceptibility(S, np.ones(S.shape[0] + 1))
