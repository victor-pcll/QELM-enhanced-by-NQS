"""QELM feature map: exact checks and TDVP robustness over short times (#19)."""

import numpy as np
import pytest
import scipy.linalg as sla

from QELM.NQS.dynamics import TDVPSettings
from QELM.NQS.qelm import (
    QELMReservoir,
    RidgeReadout,
    feature_matrix,
    random_couplings,
    reservoir_features,
)

N = 4
INPUTS = np.linspace(-0.8, 0.8, 7)


@pytest.fixture(scope="module")
def reservoir() -> QELMReservoir:
    return QELMReservoir(random_couplings(N, seed=7), h0=1.0, kappa=1.0, T=1.0)


@pytest.fixture(scope="module")
def features(reservoir: QELMReservoir) -> tuple[np.ndarray, np.ndarray]:
    return (
        feature_matrix(reservoir, INPUTS, method="exact"),
        feature_matrix(reservoir, INPUTS, method="tdvp"),
    )


def test_couplings_are_a_fixed_symmetric_reservoir() -> None:
    J = random_couplings(5, scale=0.5, seed=3)
    np.testing.assert_array_equal(J, J.T)
    np.testing.assert_array_equal(np.diag(J), 0.0)
    assert np.max(np.abs(J)) <= 0.5
    np.testing.assert_array_equal(J, random_couplings(5, scale=0.5, seed=3))
    assert not np.array_equal(J, random_couplings(5, scale=0.5, seed=4))


def test_shapes_and_schedule(reservoir: QELMReservoir, features) -> None:
    exact, tdvp = features
    assert reservoir.n_features == 10 * N
    assert exact.shape == tdvp.shape == (INPUTS.size, 10 * N)
    np.testing.assert_allclose(reservoir.readout_times, np.arange(1, 11) / 10)
    with pytest.raises(ValueError):
        QELMReservoir(random_couplings(N), T=1.0, n_readouts=3)  # dt does not divide
    with pytest.raises(ValueError):
        reservoir_features(reservoir, 0.0, method="bogus")  # type: ignore[arg-type]


def test_exact_features_match_dense_propagator(
    reservoir: QELMReservoir, features
) -> None:
    """Independent check: <Z_i>(t) from the dense exp(-iHt)."""
    exact, _ = features
    x = INPUTS[2]
    H = reservoir.hamiltonian(x).to_dense()
    psi0 = np.asarray(reservoir.initial_state().to_array())
    psi0 = psi0 / np.linalg.norm(psi0)
    ops = [op.to_dense() for op in reservoir.observables().values()]
    expected = []
    for t in reservoir.readout_times:
        psi = sla.expm(-1j * H * t) @ psi0
        expected.extend(np.vdot(psi, op @ psi).real for op in ops)
    np.testing.assert_allclose(exact[2], expected, atol=1e-10)


def test_features_are_physical_magnetizations(features) -> None:
    for F in features:
        assert np.isrealobj(F)
        assert np.all(np.abs(F) <= 1.0 + 1e-9)


def test_inputs_change_the_features(features) -> None:
    exact, _ = features
    # The map x -> features is injective on the inputs used here.
    distances = np.linalg.norm(exact[:, None] - exact[None], axis=-1)
    assert np.min(distances[np.triu_indices(INPUTS.size, k=1)]) > 1e-3


def test_without_couplings_magnetizations_are_conserved() -> None:
    """J = 0: H is diagonal in Z, so every <Z_i>(t) = <Z_i>(0), for any x."""
    reservoir = QELMReservoir(np.zeros((3, 3)), T=0.5, n_readouts=5)
    state = reservoir.initial_state()
    z0 = np.array(
        [float(state.expect(op).mean.real) for op in reservoir.observables().values()]
    )
    for method in ("exact", "tdvp"):
        F = feature_matrix(reservoir, [-0.5, 0.7], method=method)  # type: ignore[arg-type]
        np.testing.assert_allclose(F, np.tile(z0, (2, 5)), atol=1e-8)


def test_without_input_scale_features_ignore_the_input(
    reservoir: QELMReservoir,
) -> None:
    flat = QELMReservoir(reservoir.couplings, h0=1.0, kappa=0.0, T=0.2, n_readouts=2)
    F = feature_matrix(flat, [-0.5, 0.0, 0.5], method="exact")
    np.testing.assert_allclose(F, np.tile(F[0], (3, 1)), atol=1e-14)


def test_tdvp_follows_ed_at_short_times(features) -> None:
    """The QELM premise: over T = 1 the variational features match ED."""
    exact, tdvp = features
    assert np.max(np.abs(tdvp - exact)) < 2e-4


def test_tdvp_error_grows_with_evolution_time(reservoir: QELMReservoir) -> None:
    """Short times are where TDVP is reliable: the error accumulates with t.

    One run to T = 2 with a readout at every step. The largest error up to
    t = 0.1 is far below the largest error up to t = 2 (the error at a
    single time oscillates, so running maxima are compared).
    """
    T = 2.0
    r = QELMReservoir(reservoir.couplings, T=T, n_readouts=round(T / reservoir.tdvp.dt))
    error = np.abs(
        feature_matrix(r, [0.3], method="tdvp")
        - feature_matrix(r, [0.3], method="exact")
    ).reshape(-1, N)
    running_max = np.maximum.accumulate(error.max(axis=1))
    short = running_max[np.searchsorted(r.readout_times, 0.1)]
    assert short < 1e-6
    assert running_max[-1] > 10 * short
    assert running_max[-1] < 2e-4


def test_readout_trained_on_ed_transfers_to_tdvp(features) -> None:
    """Predictions move by at most |w|_1 * max feature error."""
    exact, tdvp = features
    target = INPUTS**2
    readout = RidgeReadout(lam=1e-6).fit(exact, target)
    assert readout.weights is not None
    gap = np.max(np.abs(readout.predict(tdvp) - readout.predict(exact)))
    bound = np.sum(np.abs(readout.weights)) * np.max(np.abs(tdvp - exact))
    assert gap <= bound + 1e-12
    np.testing.assert_allclose(readout.predict(exact), target, atol=1e-3)


def test_custom_tdvp_settings_are_used() -> None:
    settings = TDVPSettings(dt=0.05, integrator="RK4", diag_shift=1e-6)
    r = QELMReservoir(random_couplings(3, seed=1), T=0.2, n_readouts=2, tdvp=settings)
    exact = feature_matrix(r, [0.1], method="exact")
    tdvp = feature_matrix(r, [0.1], method="tdvp")
    assert exact.shape == (1, 6)
    np.testing.assert_allclose(tdvp, exact, atol=1e-3)
