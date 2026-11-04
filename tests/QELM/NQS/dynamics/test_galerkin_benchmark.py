"""Checks of the Fig. 2 benchmark helpers (issue #24)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

from QELM.NQS.dynamics import TDVPSettings
from QELM.NQS.dynamics.galerkin_benchmark import (
    Trajectory,
    exact_quench,
    load_trajectories,
    plot_errors,
    plot_figure2,
    plus_state_mc,
    plus_vector,
    save_trajectories,
    tdvp_quench,
    tdvp_residual,
)
from QELM.NQS.dynamics.neural_galerkin import mean_magnetization_x, tfi_square_lattice
from QELM.NQS.qelm.initial_states import plus_state_rbm


def test_tdvp_residual_is_the_projected_residual_on_vectors() -> None:
    """L from (S, F, Var H) equals ||P_perp(J theta_dot + i H Psi)||^2 / ||Psi||^2."""
    hilbert, _, H = tfi_square_lattice(3, h=2.0)
    state = plus_state_rbm(hilbert)
    # Move away from |+>^N so that every term is non-trivial.
    rng = np.random.default_rng(0)
    state.parameters = jax.tree_util.tree_map(
        lambda p: p + 0.1 * (rng.normal(size=p.shape) + 1j * rng.normal(size=p.shape)),
        state.parameters,
    )
    loss, theta_dot = tdvp_residual(state, H, diag_shift=1e-3)

    flat, unravel = ravel_pytree(state.parameters)
    states = jnp.asarray(hilbert.all_states())

    def psi(theta):
        return jnp.exp(state.model.apply({"params": unravel(theta)}, states))

    vec = np.asarray(psi(flat))
    J = np.asarray(jax.jacfwd(psi, holomorphic=True)(flat.astype(complex)))
    v = J @ theta_dot + 1j * (H.to_sparse() @ vec)
    n = np.vdot(vec, vec).real
    perp = v - vec * np.vdot(vec, v) / n
    assert loss == pytest.approx(np.vdot(perp, perp).real / n, rel=1e-8)

    # Without regularization theta_dot minimizes the residual (McLachlan).
    loss0, td0 = tdvp_residual(state, H, diag_shift=0.0)
    assert loss0 <= loss + 1e-10
    v_bad = J @ (1.1 * td0) + 1j * (H.to_sparse() @ vec)
    perp_bad = v_bad - vec * np.vdot(vec, v_bad) / n
    assert np.vdot(perp_bad, perp_bad).real / n > loss0


def test_plus_state_mc_is_exactly_plus() -> None:
    hilbert, _, H = tfi_square_lattice(3, h=5.0)
    state = plus_state_mc(hilbert, n_samples=256, n_chains=16)
    amp = np.exp(np.asarray(state.log_value(hilbert.all_states())))
    np.testing.assert_allclose(amp / amp[0], 1.0, atol=1e-12)
    psi = amp / np.linalg.norm(amp)
    np.testing.assert_allclose(psi, plus_vector(hilbert.size), atol=1e-12)
    # The local value of sigma^x is exactly 1 on |+>^N: no sampling noise.
    assert state.expect(mean_magnetization_x(hilbert)).mean.real == pytest.approx(1)
    # E = -h N, up to the noise of the ZZ term (whose mean is 0 on |+>^N).
    E = state.expect(H)
    assert abs(E.mean.real + 5.0 * 9) < 5 * E.error_of_mean + 1e-12


def test_tdvp_quench_follows_ed_on_3x3() -> None:
    """Repository TDVP from |+>^N, exact sums, at h_c; energy is conserved."""
    tr = tdvp_quench(
        3,
        3.044,
        0.2,
        settings=TDVPSettings(dt=0.01, integrator="RK4", diag_shift=1e-6),
        exact_sum=True,
    )
    ex = exact_quench(3, 3.044, tr.times)
    assert np.max(np.abs(tr.x - ex.x)) < 1e-3
    # Regularized TDVP conserves <H> up to the projection error (~1e-5 here).
    np.testing.assert_allclose(tr.energy, -3.044 * 9, rtol=1e-4)
    np.testing.assert_allclose(ex.energy, -3.044 * 9, rtol=1e-10)
    assert tr.loss is not None and tr.loss[0] < 1e-6  # |+> is exact at t = 0


def test_exact_quench_rejects_large_lattices() -> None:
    with pytest.raises(ValueError):
        exact_quench(5, 2.0, np.array([0.0, 0.1]))


def test_save_load_and_plot(tmp_path) -> None:
    t = np.linspace(0.0, 0.2, 5)
    trs = {
        "ED": Trajectory("ED", t, np.cos(t), -np.ones(5), np.zeros(5), 1.0),
        "tNQG": Trajectory("t-NQG", t, np.cos(t) + 1e-3, -np.ones(5), 1e-3 + t, 2.0),
    }
    save_trajectories(tmp_path / "h=2.npz", trs)
    back = load_trajectories(tmp_path / "h=2.npz")
    assert set(back) == set(trs)
    np.testing.assert_allclose(back["tNQG"].loss, trs["tNQG"].loss)
    assert back["ED"].wall_time == 1.0
    assert plot_figure2({2.0: back}, 16, tmp_path / "fig2.png").is_file()
    assert plot_errors({2.0: back}, tmp_path / "err.png", 16).is_file()
