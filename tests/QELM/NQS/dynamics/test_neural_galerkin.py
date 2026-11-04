"""Physics checks of the time-dependent Neural Galerkin method (issue #24).

Every identity used by :mod:`QELM.NQS.dynamics.neural_galerkin` is checked
against an independent construction with dense state vectors:

- the 2D TFI Hamiltonian against a Kronecker-product build, and the
  symmetric spectrum of a bipartite lattice used to place the frequencies;
- the initial condition ``|Psi(0)> = |+>^N`` for any parameters;
- the quadratic-form loss against ``||P_perp (dPsi + i H Psi)||^2`` on
  vectors, its gauge invariance and its zero on an exact solution;
- the Monte Carlo estimator against the exact sums;
- the linear variational solution against the matrix exponential;
- a short quench against exact diagonalization (ED), with energy
  conservation.
"""

from functools import reduce

import jax
import jax.numpy as jnp
import netket as nk
import numpy as np
import pytest
import scipy.linalg as sla

from QELM.NQS.dynamics import exact_time_evolution, expectation_values
from QELM.NQS.dynamics.neural_galerkin import (
    FrozenBasis,
    NeuralGalerkinSettings,
    basis_matrices,
    coefficients,
    expectation,
    init_rbm_batch,
    initial_frequencies,
    linear_variational_coefficients,
    log_rbm_batch,
    mean_magnetization_x,
    residual_loss,
    run_neural_galerkin,
    sample_pi,
    simpson_weights,
    spectral_bounds,
    tfi_square_lattice,
)

X = np.array([[0.0, 1.0], [1.0, 0.0]])
# NetKet's local state +1 is listed first: sigma^z = diag(1, -1).
Z = np.diag([1.0, -1.0])
I2 = np.eye(2)


def _site_op(op: np.ndarray, i: int, n: int) -> np.ndarray:
    return reduce(np.kron, [op if k == i else I2 for k in range(n)])


@pytest.fixture(scope="module")
def tfi3():
    return tfi_square_lattice(3, h=3.044)


def _random_window(key, n_spins, M=3, Nb=5, std=0.3, frozen=True):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    rbm = init_rbm_batch(k1, M, n_spins, 1, std)
    gamma = 0.3 * (jax.random.normal(k2, (M, Nb)) + 1j * jax.random.normal(k3, (M, Nb)))
    omega = jnp.linspace(-4.0, 4.0, Nb)
    fb = FrozenBasis()
    if frozen:
        fb = fb.extend(
            init_rbm_batch(k4, 2, n_spins, 1, std), jnp.array([0.2 - 0.1j, 0.05j])
        )
    return fb, rbm, gamma, omega


def _psi_vectors(fb, rbm, gamma, omega, states, tau):
    """Dense Psi(tau) and dPsi/dtau from the definition of the ansatz."""
    A = 1.0 + 0j * jnp.zeros(states.shape[0])
    if fb.rbm is not None:
        A = A + jnp.exp(log_rbm_batch(fb.rbm, states)) @ fb.coefficients
    phi = jnp.exp(log_rbm_batch(rbm, states))  # (dim, M)
    c, dc = coefficients(gamma, omega, jnp.asarray(tau))
    psi = A[None, :] + c[:, 1:] @ phi.T
    dpsi = dc[:, 1:] @ phi.T
    return np.asarray(psi), np.asarray(dpsi)


def _exact_matrices(fb, rbm, states, H, obs=None):
    from QELM.NQS.dynamics.neural_galerkin import _log_pi

    logp = _log_pi(fb, rbm, states)  # lam = 1
    w = jnp.exp(logp - jax.scipy.special.logsumexp(logp))
    return basis_matrices(fb, rbm, states, w, H.to_jax_operator(), obs)


# ---------------------------------------------------------------- Hamiltonian


def test_tfi_matches_kronecker_construction(tfi3) -> None:
    """H = -J sum_<ij> Z_i Z_j - h sum_i X_i on the periodic 3x3 lattice."""
    hilbert, graph, H = tfi3
    n = hilbert.size
    assert graph.n_edges == 2 * n  # periodic square lattice: 2 bonds per site
    ref = sum(-3.044 * _site_op(X, i, n) for i in range(n))
    for i, j in graph.edges():
        ref = ref - _site_op(Z, i, n) @ _site_op(Z, j, n)
    np.testing.assert_allclose(H.to_dense(), ref, atol=1e-12)


def test_plus_state_energy_and_magnetization(tfi3) -> None:
    hilbert, graph, H = tfi3
    plus = np.full(hilbert.n_states, 2 ** (-hilbert.size / 2))
    E = expectation_values(plus, H)
    assert E == pytest.approx(-3.044 * hilbert.size)
    assert expectation_values(plus, mean_magnetization_x(hilbert)) == pytest.approx(1)


def test_bipartite_spectrum_is_symmetric() -> None:
    """On an even periodic lattice E -> -E is a symmetry (sublattice rotation)."""
    _, _, H = tfi_square_lattice(4, h=2.0)
    lo, hi = spectral_bounds(H)
    assert lo == pytest.approx(-hi, rel=1e-8)
    omega = initial_frequencies(5, (lo, hi))
    np.testing.assert_allclose(omega, -np.linspace(lo, hi, 5))


def test_rejects_small_lattices() -> None:
    with pytest.raises(ValueError):
        tfi_square_lattice(2, h=1.0)


# ------------------------------------------------------------------- Ansatz


def test_initial_state_is_exactly_plus(tfi3) -> None:
    """c(0) = 0, so Psi(0) = A_0 = |+>^N whatever the parameters."""
    hilbert, _, _ = tfi3
    states = jnp.asarray(hilbert.all_states())
    _, rbm, gamma, omega = _random_window(jax.random.PRNGKey(0), hilbert.size)
    psi, _ = _psi_vectors(FrozenBasis(), rbm, gamma, omega, states, [0.0])
    np.testing.assert_allclose(psi[0], np.ones(hilbert.n_states), atol=1e-14)


def test_coefficient_derivative_is_the_time_derivative() -> None:
    _, _, gamma, omega = _random_window(jax.random.PRNGKey(1), 4, frozen=False)
    tau = jnp.array([0.03, 0.11])
    c, dc = coefficients(gamma, omega, tau)
    eps = 1e-6
    cp, _ = coefficients(gamma, omega, tau + eps)
    cm, _ = coefficients(gamma, omega, tau - eps)
    np.testing.assert_allclose(dc, (cp - cm) / (2 * eps), atol=1e-8)
    assert np.all(np.asarray(c[:, 0]) == 1.0)


def test_simpson_weights_integrate_cubics_exactly() -> None:
    w = simpson_weights(9, 0.2)
    t = np.linspace(0.0, 0.2, 9)
    assert np.sum(w * (1 + t - 3 * t**3)) == pytest.approx(
        0.2 + 0.2**2 / 2 - 3 * 0.2**4 / 4, rel=1e-12
    )
    with pytest.raises(ValueError):
        simpson_weights(8, 0.2)


# --------------------------------------------------------------------- Loss


def test_loss_matches_projected_residual_on_vectors(tfi3) -> None:
    """Eq. 2 from the matrices equals ||P_perp(dPsi + iH Psi)||^2/<Psi|Psi>."""
    hilbert, _, H = tfi3
    states = jnp.asarray(hilbert.all_states())
    fb, rbm, gamma, omega = _random_window(jax.random.PRNGKey(2), hilbert.size)
    mats = _exact_matrices(fb, rbm, states, H)
    tau = np.array([0.0, 0.05, 0.13, 0.2])
    c, dc = coefficients(gamma, omega, jnp.asarray(tau))
    L = np.asarray(residual_loss(c, dc, mats["S"], mats["K"], mats["Q"]))

    Hd = H.to_dense()
    psi, dpsi = _psi_vectors(fb, rbm, gamma, omega, states, tau)
    for k in range(tau.size):
        v = dpsi[k] + 1j * Hd @ psi[k]
        n = np.vdot(psi[k], psi[k]).real
        perp = v - psi[k] * np.vdot(psi[k], v) / n
        assert L[k] == pytest.approx(np.vdot(perp, perp).real / n, rel=1e-9)
    assert np.all(L >= -1e-12)


def test_loss_is_gauge_invariant(tfi3) -> None:
    """H -> H + E0 changes Psi by a global phase only: L is unchanged."""
    hilbert, _, H = tfi3
    states = jnp.asarray(hilbert.all_states())
    fb, rbm, gamma, omega = _random_window(jax.random.PRNGKey(3), hilbert.size)
    m = _exact_matrices(fb, rbm, states, H)
    c, dc = coefficients(gamma, omega, jnp.linspace(0, 0.2, 5))
    E0 = 7.3
    K2 = m["K"] + E0 * m["S"]
    Q2 = m["Q"] + 2 * E0 * m["K"] + E0**2 * m["S"]
    np.testing.assert_allclose(
        residual_loss(c, dc, m["S"], K2, Q2),
        residual_loss(c, dc, m["S"], m["K"], m["Q"]),
        rtol=1e-8,
        atol=1e-10,
    )


def test_loss_vanishes_on_the_exact_solution() -> None:
    """In a complete orthonormal basis, c(t) = exp(-iHt) c0 has L(t) = 0."""
    rng = np.random.default_rng(0)
    A = rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6))
    Hm = (A + A.conj().T) / 2
    S = np.eye(6)
    c0 = rng.normal(size=6) + 1j * rng.normal(size=6)
    tau = np.linspace(0.0, 0.7, 8)
    c = np.stack([sla.expm(-1j * Hm * t) @ c0 for t in tau])
    dc = (-1j * Hm @ c.T).T
    L = residual_loss(jnp.asarray(c), jnp.asarray(dc), S, Hm, Hm @ Hm)
    np.testing.assert_allclose(L, 0.0, atol=1e-10)
    # A wrong derivative is detected.
    L_bad = residual_loss(jnp.asarray(c), jnp.asarray(0.5 * dc), S, Hm, Hm @ Hm)
    assert np.all(np.asarray(L_bad) > 1e-2)


def test_linear_variational_solution_is_the_matrix_exponential() -> None:
    rng = np.random.default_rng(1)
    B = rng.normal(size=(5, 5)) + 1j * rng.normal(size=(5, 5))
    S = B.conj().T @ B + 0.5 * np.eye(5)  # non-orthogonal basis
    A = rng.normal(size=(5, 5)) + 1j * rng.normal(size=(5, 5))
    K = (A + A.conj().T) / 2
    c0 = np.eye(5)[0].astype(complex)
    tau = np.array([0.0, 0.3, 1.1])
    c = linear_variational_coefficients(S, K, c0, tau)
    for k, t in enumerate(tau):
        ref = sla.expm(-1j * t * np.linalg.solve(S, K)) @ c0
        np.testing.assert_allclose(c[k], ref, atol=1e-9)
    # The norm c^+ S c is conserved (unitary dynamics in the subspace).
    norms = np.einsum("ti,ij,tj->t", c.conj(), S, c).real
    np.testing.assert_allclose(norms, norms[0], rtol=1e-10)


@pytest.mark.parametrize(("std", "loss_rtol"), [(0.1, 0.05), (0.3, 0.25)])
def test_monte_carlo_matrices_agree_with_exact_sums(tfi3, std, loss_rtol) -> None:
    """Sampling the balanced mixture Pi gives S, K, Q, <X> within error bars.

    ``std = 0.3`` gives a deliberately peaked window (with ``lam = 1``, Pi
    has an effective support of about one configuration), the case that
    motivates the balanced weights ``lam_i = 1 / <u_i|u_i>``. Its loss
    involves ``Q = <u|H^2|u>``, whose estimator has heavy tails there, so
    the tolerance on ``L`` is looser.
    """
    hilbert, _, H = tfi3
    states = jnp.asarray(hilbert.all_states())
    fb, rbm, gamma, omega = _random_window(jax.random.PRNGKey(4), hilbert.size, std=std)
    Xop = {"X": mean_magnetization_x(hilbert).to_jax_operator()}
    exact = _exact_matrices(fb, rbm, states, H, Xop)
    log_lam = -jnp.log(jnp.diag(exact["S"]).real)
    n_chains = 64
    chains = jnp.asarray(hilbert.random_state(jax.random.PRNGKey(5), n_chains), float)

    @jax.jit
    def draw(k, ch):
        return sample_pi(
            fb, rbm, k, ch, n_samples=n_chains * 400, n_sweeps=2, log_lam=log_lam
        )

    _, chains, _ = draw(jax.random.PRNGKey(6), chains)  # burn-in
    samples, _, acc = draw(jax.random.PRNGKey(7), chains)
    assert 0.05 < float(acc) <= 1.0
    w = jnp.full(samples.shape[0], 1.0 / samples.shape[0])
    mc = basis_matrices(fb, rbm, samples, w, H.to_jax_operator(), Xop, log_lam)
    tau = jnp.linspace(0.0, 0.2, 5)
    c, dc = coefficients(gamma, omega, tau)
    np.testing.assert_allclose(
        expectation(c, mc["K"], mc["S"]),
        expectation(c, exact["K"], exact["S"]),
        atol=0.05 * 3.044 * hilbert.size,
    )
    np.testing.assert_allclose(
        expectation(c, mc["X"], mc["S"]),
        expectation(c, exact["X"], exact["S"]),
        atol=0.03,
    )
    L_mc = residual_loss(c, dc, mc["S"], mc["K_raw"], mc["Q"])
    L_ex = residual_loss(c, dc, exact["S"], exact["K"], exact["Q"])
    np.testing.assert_allclose(L_mc, L_ex, rtol=loss_rtol)


def test_sampled_loss_is_never_negative(tfi3) -> None:
    """With K_raw the estimate is a squared norm in the sample inner product.

    Even with 5 samples, (S, K_raw, Q) are the blocks of a Gram matrix, so
    L >= 0 for any coefficients; the optimizer cannot exploit the noise.
    """
    hilbert, _, H = tfi3
    fb, rbm, gamma, omega = _random_window(jax.random.PRNGKey(8), hilbert.size)
    rng = np.random.default_rng(3)
    for _ in range(20):
        samples = jnp.asarray(rng.choice([-1.0, 1.0], size=(5, hilbert.size)))
        w = jnp.full(5, 0.2)
        m = basis_matrices(fb, rbm, samples, w, H.to_jax_operator())
        g = jnp.asarray(
            rng.normal(size=gamma.shape) + 1j * rng.normal(size=gamma.shape)
        )
        c, dc = coefficients(g, omega, jnp.linspace(0.0, 0.2, 7))
        L = residual_loss(c, dc, m["S"], m["K_raw"], m["Q"])
        assert np.all(np.asarray(L) >= -1e-9 * np.max(np.abs(np.asarray(L))))
        np.testing.assert_allclose(m["K"], (m["K_raw"] + m["K_raw"].conj().T) / 2)


# --------------------------------------------------------- Dynamics vs ED


@pytest.fixture(scope="module")
def quench3():
    """3x3 quench to h = h_c, two windows, exact sums (no sampling noise)."""
    hilbert, _, H = tfi_square_lattice(3, h=3.044)
    settings = NeuralGalerkinSettings(
        n_basis=4,
        n_frequencies=16,
        window=0.2,
        n_quadrature=33,
        n_iter=600,
        learning_rate=1e-2,
        exact=True,
        n_eval_times=5,
    )
    Xbar = mean_magnetization_x(hilbert)
    result = run_neural_galerkin(H, T=0.4, observables={"X": Xbar}, settings=settings)
    psi0 = np.full(hilbert.n_states, 2 ** (-hilbert.size / 2), dtype=complex)
    exact_states = exact_time_evolution(H, psi0, result.times)
    return hilbert, H, result, expectation_values(exact_states, Xbar)


def test_short_quench_follows_ed(quench3) -> None:
    hilbert, H, result, x_exact = quench3
    assert result.observables["X"][0] == pytest.approx(1.0, abs=1e-12)
    err = np.max(np.abs(result.observables["X"] - x_exact))
    # M = 4 basis states and 600 Adam steps: <sigma^x> to 4e-3 on a drop of 0.3.
    assert err < 6e-3, err
    assert np.ptp(x_exact) > 0.25
    loss_per_spin = result.loss / hilbert.size
    assert np.max(loss_per_spin) < 3e-2
    assert np.median(loss_per_spin) < 1e-2


def test_short_quench_conserves_energy(quench3) -> None:
    hilbert, H, result, _ = quench3
    E0 = -3.044 * hilbert.size
    np.testing.assert_allclose(result.energy, E0, atol=1e-2 * abs(E0))


def test_training_reduces_the_loss(quench3) -> None:
    _, _, result, _ = quench3
    for h in result.loss_history:
        assert h[-1] < 0.1 * h[0]


def test_settings_are_validated() -> None:
    with pytest.raises(ValueError):
        NeuralGalerkinSettings(n_quadrature=256)
    with pytest.raises(ValueError):
        NeuralGalerkinSettings(n_samples=1000, n_chains=256)
    _, _, H = tfi_square_lattice(3, h=1.0)
    with pytest.raises(ValueError):
        run_neural_galerkin(
            H, T=0.3, observables={}, settings=NeuralGalerkinSettings(exact=True)
        )


def test_netket_spin_convention() -> None:
    """NetKet lists the local state +1 first and sigma^z = diag(1, -1)."""
    hi = nk.hilbert.Spin(0.5, 1)
    np.testing.assert_allclose(nk.operator.spin.sigmaz(hi, 0).to_dense(), Z)
