"""Loschmidt echo: TDVP under H for T, then under -H for T, returns to psi(0).

Exact dynamics gives exp(+iHT) exp(-iHT) = 1. For TDVP, flipping H flips
the projected flow dθ/dt = -i S^{-1} F, so the echo also checks that the
integrator, the QGT solve and the state bookkeeping are reversible.

System: periodic transverse-field Ising chain, N=4, h=J=1, T=1. The RBM
(alpha=1, 24 complex parameters for a 16-dimensional Hilbert space) starts
from a regular point of its manifold (init_std=0.1, see ``rbm``) with a
visible bias shifted by +0.5, so <Z>(0) is about 0.76 and the dynamics
moves it over a range of about 0.8: the echo is not trivially satisfied.
"""

import jax
import netket as nk
import numpy as np
import pytest

from QELM.NQS.dynamics import TDVPResult, TDVPSettings, run_tdvp, time_grid
from QELM.NQS.variational import (
    mean_magnetization_z,
    rbm,
    spin_chain,
    spin_half_hilbert,
    transverse_field_ising,
    variational_state,
)

SEED = 45
N = 4
T_ECHO = 1.0
SETTINGS = TDVPSettings(dt=0.01, integrator="RK45")


def _setup():
    hilbert = spin_half_hilbert(N)
    H = transverse_field_ising(hilbert, spin_chain(N, pbc=True), h=1.0, J=1.0)
    state = variational_state(hilbert, rbm(init_std=0.1), seed=SEED)
    params = jax.tree_util.tree_map(lambda x: x, state.parameters)
    params["visible_bias"] = params["visible_bias"] + 0.5
    state.parameters = params
    return hilbert, H, state


def _vector(state: nk.vqs.FullSumState) -> np.ndarray:
    return np.asarray(state.to_array())


def _fidelity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(np.vdot(a, b)) ** 2)


@pytest.fixture(scope="module")
def echo():
    hilbert, H, state = _setup()
    obs = {"Magnetization": mean_magnetization_z(hilbert)}
    psi_initial = _vector(state)

    aller = run_tdvp(H, state, T=T_ECHO, observables=obs, settings=SETTINGS)
    params_after_aller = jax.tree_util.tree_map(np.asarray, state.parameters)
    psi_mid = _vector(state)
    state_id = id(state)

    retour = run_tdvp(-1.0 * H, state, T=T_ECHO, observables=obs, settings=SETTINGS)
    return {
        "aller": aller,
        "retour": retour,
        "psi_initial": psi_initial,
        "psi_mid": psi_mid,
        "psi_final": _vector(state),
        "params_after_aller": params_after_aller,
        "state": state,
        "state_id": state_id,
    }


def test_observable_reversibility(echo) -> None:
    aller: TDVPResult = echo["aller"]
    retour: TDVPResult = echo["retour"]
    mag_aller = aller.observables["Magnetization"]
    mag_retour = retour.observables["Magnetization"]
    np.testing.assert_allclose(mag_retour[-1], mag_aller[0], atol=1e-3)
    # The whole backward trajectory retraces the forward one.
    np.testing.assert_allclose(mag_retour[::-1], mag_aller, atol=1e-3)


def test_dynamics_is_not_trivial(echo) -> None:
    mag_aller = echo["aller"].observables["Magnetization"]
    assert mag_aller[0] > 0.5
    assert np.ptp(mag_aller) > 0.3
    # Halfway through, the state is far from where it started.
    assert _fidelity(echo["psi_initial"], echo["psi_mid"]) < 0.9


def test_state_fidelity_returns_to_one(echo) -> None:
    fidelity = _fidelity(echo["psi_initial"], echo["psi_final"])
    assert fidelity == pytest.approx(1.0, abs=1e-6)
    assert fidelity <= 1.0 + 1e-12


def test_state_is_carried_over_not_reset(echo) -> None:
    # Same object, and the backward run started from the forward result:
    # its first recorded value is the forward run's last one.
    assert id(echo["state"]) == echo["state_id"]
    mag_aller = echo["aller"].observables["Magnetization"]
    mag_retour = echo["retour"].observables["Magnetization"]
    assert mag_retour[0] == pytest.approx(mag_aller[-1], abs=1e-12)
    # The parameters moved during the backward run (no silent copy).
    moved = jax.tree_util.tree_map(
        lambda a, b: np.max(np.abs(np.asarray(a) - b)),
        echo["state"].parameters,
        echo["params_after_aller"],
    )
    assert max(jax.tree_util.tree_leaves(moved)) > 1e-3


def test_time_grids_from_loggers(echo) -> None:
    grid = time_grid(T_ECHO, SETTINGS.dt)
    np.testing.assert_allclose(echo["aller"].times, grid, atol=1e-12)
    np.testing.assert_allclose(echo["retour"].times, grid, atol=1e-12)


def test_echo_detects_a_wrong_sign() -> None:
    """Negative control: evolving twice under +H does not come back."""
    hilbert, H, state = _setup()
    obs = {"Magnetization": mean_magnetization_z(hilbert)}
    psi_initial = _vector(state)
    settings = TDVPSettings(dt=0.01, integrator="RK4")
    aller = run_tdvp(H, state, T=T_ECHO, observables=obs, settings=settings)
    retour = run_tdvp(H, state, T=T_ECHO, observables=obs, settings=settings)
    error = abs(
        retour.observables["Magnetization"][-1] - aller.observables["Magnetization"][0]
    )
    assert error > 0.1
    assert _fidelity(psi_initial, _vector(state)) < 0.9
