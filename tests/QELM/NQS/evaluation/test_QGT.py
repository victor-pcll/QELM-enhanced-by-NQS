"""Quantum geometric tensor: format handling, structure and known values (#4)."""

import jax
import jax.numpy as jnp
import netket as nk
import numpy as np
import pytest
import scipy.sparse as sp
from jax.flatten_util import ravel_pytree

from QELM.NQS.evaluation.QGT import (
    analyze_qgt,
    berry_curvature,
    fubini_study_metric,
    qgt_matrix,
    qgt_of_state,
)
from QELM.NQS.variational import rbm, spin_half_hilbert, variational_state

N = 4


def _state(seed: int = 0, init_std: float | None = 0.3, dtype: str = "complex"):
    state = variational_state(spin_half_hilbert(N), rbm(param_dtype=dtype), seed=seed)
    if init_std is not None:
        # Spread the parameters away from W = 0 (a regular manifold point).
        key = jax.random.PRNGKey(seed)
        flat, unravel = ravel_pytree(state.parameters)
        noise = init_std * jax.random.normal(key, flat.shape, dtype=jnp.float64)
        noise = noise.astype(flat.dtype)
        if jnp.iscomplexobj(flat):
            noise = noise + 1j * init_std * jax.random.normal(
                jax.random.PRNGKey(seed + 1000), flat.shape
            )
        state.parameters = unravel(flat + noise)
    return state


def _explicit_qgt(state) -> np.ndarray:
    """S = <O^* O> - <O^*><O> from the exact Jacobian of log psi."""
    configs = state.hilbert.all_states()
    flat, unravel = ravel_pytree(state.parameters)

    def log_psi(theta):
        return state.model.apply({"params": unravel(theta)}, configs)

    jac = np.asarray(jax.jacfwd(log_psi, holomorphic=True)(flat))
    logp = 2 * np.asarray(log_psi(flat)).real
    p = np.exp(logp - logp.max())
    p /= p.sum()
    centered = jac - p @ jac
    return centered.conj().T @ (p[:, None] * centered)


@pytest.fixture(scope="module")
def S() -> np.ndarray:
    return qgt_of_state(_state(seed=3), holomorphic=True)


def test_netket_qgt_matches_explicit_definition() -> None:
    state = _state(seed=1)
    np.testing.assert_allclose(
        qgt_of_state(state, holomorphic=True), _explicit_qgt(state), atol=1e-12
    )


def test_format_compatibility(S: np.ndarray) -> None:
    state = _state(seed=3)
    netket_operator = state.quantum_geometric_tensor(
        nk.optimizer.qgt.QGTJacobianDense(holomorphic=True)
    )
    on_the_fly = state.quantum_geometric_tensor(
        nk.optimizer.qgt.QGTOnTheFly(holomorphic=True)
    )
    inputs = [S, jnp.asarray(S), sp.csr_matrix(S), sp.coo_array(S), netket_operator]
    reference = analyze_qgt(S)
    for item in inputs:
        result = analyze_qgt(item)
        np.testing.assert_array_equal(result.metric, reference.metric)
        np.testing.assert_array_equal(result.curvature, reference.curvature)
        assert result.rank == reference.rank
        assert result.trace == reference.trace
    # The matrix-free operator agrees up to round-off.
    other = analyze_qgt(on_the_fly)
    np.testing.assert_allclose(other.metric, reference.metric, atol=1e-12)
    np.testing.assert_allclose(other.curvature, reference.curvature, atol=1e-12)


def test_metric_is_symmetric(S: np.ndarray) -> None:
    g = fubini_study_metric(S)
    assert g.dtype == np.float64
    np.testing.assert_allclose(g, g.T, atol=1e-14, rtol=0)


def test_curvature_is_antisymmetric(S: np.ndarray) -> None:
    omega = berry_curvature(S)
    np.testing.assert_allclose(omega, -omega.T, atol=1e-14, rtol=0)
    np.testing.assert_array_equal(np.diag(omega), 0.0)
    # Complex parameters at a generic point give a non-zero curvature.
    assert np.linalg.norm(omega) > 1e-3


def test_metric_is_positive_semidefinite(S: np.ndarray) -> None:
    eigenvalues = analyze_qgt(S).eigenvalues
    assert eigenvalues[0] >= -1e-12 * eigenvalues[-1]
    for seed in range(3):
        values = analyze_qgt(qgt_of_state(_state(seed), holomorphic=True)).eigenvalues
        assert values[0] >= -1e-12 * values[-1]


def test_diagnostics_on_a_known_matrix() -> None:
    g = np.diag([4.0, 1.0, 0.25, 0.0])
    omega = np.zeros((4, 4))
    omega[0, 1], omega[1, 0] = 0.6, -0.6
    S = g + 0.5j * omega
    result = analyze_qgt(S)
    np.testing.assert_allclose(result.metric, g)
    np.testing.assert_allclose(result.curvature, omega)
    assert result.rank == 3
    assert result.condition_number == np.inf
    assert result.effective_condition_number == pytest.approx(16.0)
    assert result.trace == pytest.approx(5.25)
    assert result.curvature_norm == pytest.approx(0.6 * np.sqrt(2))
    full = analyze_qgt(np.diag([2.0, 0.5]))
    assert full.rank == 2
    assert full.condition_number == pytest.approx(4.0)


def test_rbm_at_zero_weights_has_analytic_metric() -> None:
    """W = b = 0: only the visible biases move the state.

    psi(s) ∝ exp(sum_i a_i s_i), so O_{a_i} = s_i and O_{b_j} = O_{W_ij} =
    tanh(0) = 0. The metric is diagonal in a with g_ii = Var(s_i) =
    1 - tanh(2 Re a_i)^2, and rank(g) = N.
    """
    state = variational_state(spin_half_hilbert(N), rbm(), seed=0)
    a = np.array([0.1, -0.3, 0.5, 0.0]) + 0.2j
    params = jax.tree_util.tree_map(jnp.zeros_like, state.parameters)
    params["visible_bias"] = jnp.asarray(a)
    state.parameters = params

    result = analyze_qgt(qgt_of_state(state, holomorphic=True))
    assert result.rank == N
    assert result.n_parameters == N + N + N * N
    flat_order = ravel_pytree(state.parameters)[0].shape[0]
    assert flat_order == result.n_parameters
    visible = _visible_indices(state)
    expected = 1.0 - np.tanh(2 * a.real) ** 2
    np.testing.assert_allclose(np.diag(result.metric)[visible], expected, atol=1e-12)
    assert result.trace == pytest.approx(expected.sum(), abs=1e-12)
    # Visible biases commute: no Berry curvature among them.
    np.testing.assert_allclose(result.curvature, 0.0, atol=1e-12)


def _visible_indices(state) -> np.ndarray:
    """Positions of the visible biases in NetKet's flattened parameters."""
    marked = jax.tree_util.tree_map(jnp.zeros_like, state.parameters)
    marked["visible_bias"] = jnp.ones_like(marked["visible_bias"])
    flat, _ = ravel_pytree(marked)
    return np.flatnonzero(np.asarray(flat).real)


def test_real_parameters_have_no_curvature() -> None:
    state = _state(seed=2, dtype="real")
    result = analyze_qgt(qgt_of_state(state))
    np.testing.assert_array_equal(result.curvature, 0.0)
    assert result.curvature_norm == 0.0


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        qgt_matrix(np.ones((2, 3)))
    with pytest.raises(ValueError):
        analyze_qgt(np.array([[1.0, 1.0j], [1.0j, 1.0]]))
