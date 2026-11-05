"""Accumulated TDVP-vs-ED error, local observables and entanglement entropy.

Checks, from the bottom up:

* the time integral is exact on known errors (constant, |sin t|);
* the entropy is 0 for product states, ln 2 for a Bell pair cut in two,
  follows NetKet's site ordering, and respects the Page bound;
* the local Paulis reproduce the mean magnetization and the energy of
  known product states;
* the TDVP state vectors recorded on the grid give back NetKet's own
  logged observables, and the comparison obeys exact physical constraints
  (zero error at t=0, F <= 1, |mean Δ| <= mean |Δ|, energy conservation);
* for J=0 the dynamics is a product of single-site rotations, which an RBM
  represents exactly: ED conserves <Z_i> and S, and TDVP follows it.
"""

from pathlib import Path

import numpy as np
import pytest

from QELM.NQS.dynamics import (
    ERROR_KEYS,
    ErrorComparison,
    FieldSweep,
    TDVPSettings,
    accumulated_absolute_error,
    accumulated_error_vs_field,
    compare_with_exact,
    entanglement_entropy,
    exact_time_evolution,
    expectation_values,
    local_pauli,
    run_tdvp,
    time_grid,
    trajectory_observables,
)
from QELM.NQS.measurements import all_to_all_xx_hamiltonian, mean_magnetization_z
from QELM.NQS.utils.makeplots import plot_field_scan
from QELM.NQS.variational import rbm, spin_half_hilbert, variational_state

LN2 = np.log(2.0)
SETTINGS = TDVPSettings(dt=5e-3, integrator="RK45", diag_shift=1e-6)


def _basis_state(hilbert, config) -> np.ndarray:
    """Return the basis vector of a ±1 configuration, in NetKet's ordering."""
    psi = np.zeros(hilbert.n_states, dtype=complex)
    psi[int(hilbert.states_to_numbers(np.asarray(config, dtype=float)))] = 1.0
    return psi


# --- accumulated absolute error -------------------------------------------


def test_accumulated_error_of_identical_curves_is_zero() -> None:
    t = np.linspace(0.0, 2.0, 101)
    assert accumulated_absolute_error(t, np.cos(t), np.cos(t)) == 0.0


def test_accumulated_error_of_constant_offset_is_offset_times_T() -> None:
    t = np.linspace(0.0, 3.0, 31)
    error = accumulated_absolute_error(t, np.sin(t) + 0.25, np.sin(t))
    assert error == pytest.approx(0.75, abs=1e-12)


def test_accumulated_error_integrates_absolute_value() -> None:
    # ∫_0^{2π} |sin t| dt = 4 while ∫ sin t dt = 0: signs must not cancel.
    t = np.linspace(0.0, 2 * np.pi, 20001)
    error = accumulated_absolute_error(t, np.sin(t), np.zeros_like(t))
    assert error == pytest.approx(4.0, abs=1e-6)


def test_accumulated_error_is_symmetric_and_per_column() -> None:
    t = np.linspace(0.0, 1.0, 11)
    a = np.stack([t, 2 * t], axis=1)
    b = np.zeros_like(a)
    forward = accumulated_absolute_error(t, a, b)
    np.testing.assert_allclose(forward, [0.5, 1.0], atol=1e-12)
    np.testing.assert_array_equal(forward, accumulated_absolute_error(t, b, a))


@pytest.mark.parametrize(
    ("times", "a", "b"),
    [
        (np.linspace(0, 1, 5), np.zeros(5), np.zeros(4)),
        (np.linspace(0, 1, 5), np.zeros(4), np.zeros(4)),
        (np.array([0.0, 0.5, 0.5, 1.0]), np.zeros(4), np.zeros(4)),
        (np.array([0.0]), np.zeros(1), np.zeros(1)),
    ],
)
def test_accumulated_error_rejects_bad_input(times, a, b) -> None:
    with pytest.raises(ValueError):
        accumulated_absolute_error(times, a, b)


# --- entanglement entropy --------------------------------------------------


def test_entropy_of_product_state_is_zero() -> None:
    hilbert = spin_half_hilbert(4)
    psi = _basis_state(hilbert, [1, -1, -1, 1])
    assert entanglement_entropy(psi, 4) == pytest.approx(0.0, abs=1e-12)
    plus = np.full(16, 0.25, dtype=complex)  # |+>^4, also a product state
    for cut in (1, 2, 3):
        assert entanglement_entropy(plus, 4, cut) == pytest.approx(0.0, abs=1e-12)


def test_entropy_follows_netket_site_ordering() -> None:
    # Bell pair on sites (1, 2), sites 0 and 3 up. Cutting between 1 and 2
    # splits the pair (S = ln 2); cutting after site 0 or site 2 does not.
    hilbert = spin_half_hilbert(4)
    bell = (
        _basis_state(hilbert, [1, 1, 1, 1]) + _basis_state(hilbert, [1, -1, -1, 1])
    ) / np.sqrt(2)
    assert entanglement_entropy(bell, 4, 2) == pytest.approx(LN2, abs=1e-12)
    assert entanglement_entropy(bell, 4, 1) == pytest.approx(0.0, abs=1e-12)
    assert entanglement_entropy(bell, 4, 3) == pytest.approx(0.0, abs=1e-12)


def test_entropy_of_ghz_state_and_normalization() -> None:
    hilbert = spin_half_hilbert(6)
    ghz = _basis_state(hilbert, [1] * 6) + _basis_state(hilbert, [-1] * 6)
    # Not normalized on purpose: the entropy must not depend on the norm.
    for cut in range(1, 6):
        assert entanglement_entropy(3.0 * ghz, 6, cut) == pytest.approx(LN2)


def test_entropy_bounds_on_random_states() -> None:
    rng = np.random.default_rng(0)
    n = 6
    states = rng.normal(size=(20, 2**n)) + 1j * rng.normal(size=(20, 2**n))
    s = entanglement_entropy(states, n)
    assert s.shape == (20,)
    assert np.all(s >= 0) and np.all(s <= 3 * LN2 + 1e-12)
    # Haar states are close to maximally entangled (Page: ≈ 3 ln2 - 1/2).
    assert np.mean(s) > 3 * LN2 - 0.7


def test_entropy_rejects_bad_cut_and_dimension() -> None:
    with pytest.raises(ValueError):
        entanglement_entropy(np.ones(16), 4, 0)
    with pytest.raises(ValueError):
        entanglement_entropy(np.ones(16), 4, 4)
    with pytest.raises(ValueError):
        entanglement_entropy(np.ones(8), 4)


# --- local observables -----------------------------------------------------


def test_local_paulis_on_known_product_states() -> None:
    n, h, J = 4, 0.7, 1.3
    hilbert = spin_half_hilbert(n)
    H = all_to_all_xx_hamiltonian(hilbert, h=h, J=J)
    up = _basis_state(hilbert, [1] * n)
    plus = np.full(2**n, 2 ** (-n / 2), dtype=complex)
    traj = trajectory_observables(np.stack([up, plus]), H, hilbert)

    # |↑↑↑↑>: <Z_i> = +1, <X_i> = 0, E = h N.
    np.testing.assert_allclose(traj.sigma_z[0], 1.0, atol=1e-12)
    np.testing.assert_allclose(traj.sigma_x[0], 0.0, atol=1e-12)
    assert traj.energy[0] == pytest.approx(h * n)
    # |++++>: <X_i> = +1, <Z_i> = 0, E = J N (N-1) / 2.
    np.testing.assert_allclose(traj.sigma_x[1], 1.0, atol=1e-12)
    np.testing.assert_allclose(traj.sigma_z[1], 0.0, atol=1e-12)
    assert traj.energy[1] == pytest.approx(J * n * (n - 1) / 2)
    np.testing.assert_allclose(traj.entropy, 0.0, atol=1e-12)


def test_site_average_matches_mean_magnetization() -> None:
    hilbert = spin_half_hilbert(5)
    psi = np.asarray(variational_state(hilbert, rbm(), seed=3).to_array())
    H = all_to_all_xx_hamiltonian(hilbert, h=1.0)
    traj = trajectory_observables(psi[None, :], H, hilbert)
    reference = expectation_values(psi, mean_magnetization_z(hilbert))
    assert traj.magnetization_z[0] == pytest.approx(float(reference), abs=1e-14)
    assert np.all(np.abs(traj.sigma_x) <= 1 + 1e-12)
    assert np.all(np.abs(traj.sigma_z) <= 1 + 1e-12)


def test_local_pauli_rejects_unknown_axis() -> None:
    with pytest.raises(ValueError):
        local_pauli(spin_half_hilbert(2), "w")  # type: ignore[arg-type]


# --- recorded TDVP states ----------------------------------------------------


def test_recorded_states_reproduce_netket_log() -> None:
    hilbert = spin_half_hilbert(4)
    H = all_to_all_xx_hamiltonian(hilbert, h=1.0)
    M = mean_magnetization_z(hilbert)
    state = variational_state(hilbert, rbm(), seed=7)
    result = run_tdvp(
        H,
        state,
        T=0.1,
        observables={"M": M, "E": H},
        settings=SETTINGS,
        record_states=True,
    )
    assert result.states is not None
    assert result.states.shape == (result.times.size, 16)
    # Each recorded vector is psi at its grid time: the observables NetKet
    # logged at those times are recovered to round-off.
    np.testing.assert_allclose(
        expectation_values(result.states, M), result.observables["M"], atol=1e-12
    )
    np.testing.assert_allclose(
        expectation_values(result.states, H), result.observables["E"], atol=1e-12
    )
    np.testing.assert_allclose(np.linalg.norm(result.states, axis=1), 1.0)
    # The last recorded vector is psi(T): it matches the exact evolution,
    # and it is the state left by run_tdvp (see test_tdvp_final_time.py).
    exact = exact_time_evolution(H, result.states[0], result.times)
    fidelity = np.abs(np.vdot(exact[-1], result.states[-1])) ** 2
    assert fidelity == pytest.approx(1.0, abs=1e-6)
    np.testing.assert_allclose(result.states[-1], state.to_array(), atol=1e-14)


def test_states_are_not_recorded_by_default() -> None:
    hilbert = spin_half_hilbert(2)
    H = all_to_all_xx_hamiltonian(hilbert, h=1.0)
    state = variational_state(hilbert, rbm(), seed=1)
    result = run_tdvp(H, state, T=0.01, observables={"E": H}, settings=SETTINGS)
    assert result.states is None


# --- full comparison ---------------------------------------------------------


@pytest.fixture(scope="module")
def comparison() -> ErrorComparison:
    return compare_with_exact(n_spins=4, h=1.0, J=1.0, T=0.3, settings=SETTINGS)


def test_comparison_starts_from_the_same_state(comparison) -> None:
    c = comparison
    assert c.times.shape == c.fidelity.shape == (61,)
    assert c.fidelity[0] == pytest.approx(1.0, abs=1e-12)
    for exact, tdvp in (
        (c.exact.energy, c.tdvp.energy),
        (c.exact.sigma_x, c.tdvp.sigma_x),
        (c.exact.sigma_z, c.tdvp.sigma_z),
        (c.exact.entropy, c.tdvp.entropy),
    ):
        np.testing.assert_allclose(tdvp[0], exact[0], atol=1e-12)


def test_comparison_obeys_physical_constraints(comparison) -> None:
    c = comparison
    assert np.all(c.fidelity <= 1 + 1e-12)
    assert np.all(c.exact.entropy <= 2 * LN2 + 1e-12)
    # Unitary dynamics conserves <H>; TDVP does too, up to regularization.
    np.testing.assert_allclose(c.exact.energy, c.exact.energy[0], atol=1e-10)
    np.testing.assert_allclose(c.tdvp.energy, c.exact.energy[0], atol=1e-4)
    # The exact dynamics does move the state: the check is not trivial.
    assert np.ptp(c.exact.sigma_x) > 1e-2


def test_accumulated_errors_are_small_and_consistent(comparison) -> None:
    errors = comparison.accumulated_errors()
    assert set(errors) == set(ERROR_KEYS)
    assert all(value >= 0 for value in errors.values())
    # |(1/N) Σ Δ_i| <= (1/N) Σ |Δ_i| at every time, hence after integration.
    assert errors["magnetization_z"] <= errors["sigma_z"] + 1e-15
    T = comparison.times[-1]
    # Time-averaged errors stay at the 1e-3 level of the baseline.
    for key in ("magnetization_z", "sigma_x", "sigma_z", "entropy"):
        assert errors[key] / T < 1e-3, key
    assert errors["infidelity"] / T < 1e-4
    # Direct check of one entry against the definition.
    expected = np.mean(
        [
            np.trapezoid(
                np.abs(comparison.tdvp.sigma_x[:, i] - comparison.exact.sigma_x[:, i]),
                comparison.times,
            )
            for i in range(4)
        ]
    )
    assert errors["sigma_x"] == pytest.approx(expected, rel=1e-12)


def test_free_spins_are_followed_exactly() -> None:
    # J = 0: H = h Σ Z_i is a sum of single-site rotations. <Z_i> and the
    # entanglement are conserved exactly, <X_i> precesses at frequency 2h,
    # and psi(s, t) = exp(-i h t Σ s_i) psi(s, 0) is an RBM (visible biases
    # shifted by -i h t), so TDVP has no projection error.
    c = compare_with_exact(n_spins=4, h=2.0, J=0.0, T=0.3, settings=SETTINGS)
    initial_z = np.broadcast_to(c.exact.sigma_z[0], c.exact.sigma_z.shape)
    np.testing.assert_allclose(c.exact.sigma_z, initial_z, atol=1e-10)
    np.testing.assert_allclose(c.exact.entropy, c.exact.entropy[0], atol=1e-10)
    errors = c.accumulated_errors()
    for key in ERROR_KEYS:
        assert errors[key] < 1e-6, key


# --- field sweep and plot ----------------------------------------------------


def test_field_sweep_shapes_and_round_trip(tmp_path: Path) -> None:
    sweep = accumulated_error_vs_field(
        n_spins=4, fields=[0.5, 2.0], T=0.05, settings=SETTINGS
    )
    assert sweep.fields.tolist() == [0.5, 2.0]
    assert set(sweep.errors) == set(ERROR_KEYS)
    for value in sweep.errors.values():
        assert value.shape == (2,)
    assert np.all(sweep.final_fidelity <= 1 + 1e-12)
    assert np.all(sweep.wall_time > 0)

    loaded = FieldSweep.load(sweep.save(tmp_path / "sweep.npz"))
    assert loaded.n_spins == 4 and loaded.T == pytest.approx(0.05)
    np.testing.assert_array_equal(loaded.fields, sweep.fields)
    for key in ERROR_KEYS:
        np.testing.assert_array_equal(loaded.errors[key], sweep.errors[key])

    path = plot_field_scan(
        {"N=4": sweep.fields},
        {"N=4": sweep.errors},
        panels={"energy": "E", "sigma_x": "X", "entropy": "S", "infidelity": "1-F"},
        filename="scan.png",
        directory=tmp_path,
    )
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_field_sweep_rejects_empty_fields() -> None:
    with pytest.raises(ValueError):
        accumulated_error_vs_field(n_spins=2, fields=[], T=0.01)


def test_plot_rejects_mismatched_curves(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        plot_field_scan(
            {"N=4": [1.0, 2.0]},
            {"N=6": {"energy": [1e-3, 2e-3]}},
            panels={"energy": "E"},
            filename="bad.png",
            directory=tmp_path,
        )


def test_time_grid_is_shared_with_exact_evolution(comparison) -> None:
    np.testing.assert_allclose(comparison.times, time_grid(0.3, SETTINGS.dt))
    hilbert = spin_half_hilbert(4)
    H = all_to_all_xx_hamiltonian(hilbert, h=1.0)
    psi0 = np.asarray(variational_state(hilbert, rbm(), seed=45).to_array())
    states = exact_time_evolution(H, psi0, comparison.times)
    traj = trajectory_observables(states, H, hilbert)
    np.testing.assert_allclose(traj.sigma_z, comparison.exact.sigma_z, atol=1e-12)
