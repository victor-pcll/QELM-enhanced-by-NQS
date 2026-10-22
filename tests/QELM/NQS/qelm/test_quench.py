"""Quench from the ground state of H_0 = -h sum_i A_i, A = Z or X (issue #26).

Checks, on small N against exact diagonalization:

- the prepared states are the unique ground states of H_0 (E_0 / N = -h,
  gap 2h), product states (zero entanglement), and the RBM writes |+>^N
  exactly;
- the Hadamard frame used for the Z quench is an exact change of basis;
- conservation laws and symmetries of the quench dynamics;
- TDVP follows ED from these states, and a negative control at the singular
  point of the RBM manifold does not;
- VMC on H_0 converges to E / N = -h.
"""

import netket as nk
import numpy as np
import pytest
import scipy.linalg as sla

from QELM.NQS.dynamics import (
    exact_time_evolution,
    expectation_values,
    run_tdvp,
    time_grid,
)
from QELM.NQS.embedding import VMCGroundStatePreparation, VMCSettings
from QELM.NQS.evaluation.metrics import NMSE
from QELM.NQS.measurements import all_to_all_xx_hamiltonian, pure_field_hamiltonian
from QELM.NQS.paths import ExperimentPaths
from QELM.NQS.qelm import (
    QELMReservoir,
    RidgeReadout,
    entanglement_entropy,
    feature_matrix,
    hadamard_plus_state,
    plus_state_rbm,
    quench_ground_state,
    random_couplings,
    zero_state,
)
from QELM.NQS.variational import rbm, spin_half_hilbert, variational_state

N = 4
H_FIELD = 0.7
INPUTS = np.linspace(-0.8, 0.8, 7)
AXES = ("z", "x")


def _hadamard_unitary(hilbert: nk.hilbert.Spin) -> np.ndarray:
    U = np.eye(hilbert.n_states)
    for i in range(hilbert.size):
        gate = (
            nk.operator.spin.sigmax(hilbert, i) + nk.operator.spin.sigmaz(hilbert, i)
        ).to_dense() / np.sqrt(2.0)
        U = gate @ U
    return U


@pytest.fixture(scope="module")
def hilbert() -> nk.hilbert.Spin:
    return spin_half_hilbert(N)


@pytest.fixture(scope="module")
def couplings() -> np.ndarray:
    return random_couplings(N, seed=7)


@pytest.fixture(scope="module", params=AXES)
def quench(request, couplings: np.ndarray) -> QELMReservoir:
    return QELMReservoir(couplings, T=1.0, initial="quench", quench_axis=request.param)


@pytest.fixture(scope="module")
def quench_features(quench: QELMReservoir) -> tuple[np.ndarray, np.ndarray]:
    return (
        feature_matrix(quench, INPUTS, method="exact"),
        feature_matrix(quench, INPUTS, method="tdvp"),
    )


# --- State preparation -------------------------------------------------------


def test_hadamard_on_zero_gives_the_uniform_superposition(hilbert) -> None:
    psi = hadamard_plus_state(hilbert)
    np.testing.assert_allclose(psi, np.full(2**N, 2.0 ** (-N / 2)), atol=1e-15)
    zero = zero_state(hilbert)
    assert zero.sum() == 1.0 and np.count_nonzero(zero) == 1
    for i in range(N):
        z = nk.operator.spin.sigmaz(hilbert, i)
        assert float(expectation_values(zero, z)) == pytest.approx(1.0)


@pytest.mark.parametrize("axis", AXES)
def test_quench_state_is_the_unique_ground_state_of_h0(hilbert, axis) -> None:
    H0 = pure_field_hamiltonian(hilbert, h=H_FIELD, axis=axis).to_dense()
    np.testing.assert_allclose(H0, H0.conj().T)
    energies, vectors = np.linalg.eigh(H0)
    assert energies[0] / N == pytest.approx(-H_FIELD, abs=1e-12)
    assert energies[1] - energies[0] == pytest.approx(2 * H_FIELD, abs=1e-12)
    psi = quench_ground_state(hilbert, axis)
    assert np.linalg.norm(psi) == pytest.approx(1.0)
    assert abs(np.vdot(vectors[:, 0], psi)) == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(H0 @ psi, energies[0] * psi, atol=1e-12)
    pauli = {"z": nk.operator.spin.sigmaz, "x": nk.operator.spin.sigmax}[axis]
    other = {"z": nk.operator.spin.sigmax, "x": nk.operator.spin.sigmaz}[axis]
    for i in range(N):
        assert float(expectation_values(psi, pauli(hilbert, i))) == pytest.approx(1)
        assert abs(float(expectation_values(psi, other(hilbert, i)))) < 1e-14


def test_pure_field_hamiltonian_rejects_bad_input(hilbert) -> None:
    with pytest.raises(ValueError):
        pure_field_hamiltonian(hilbert, h=0.0)
    with pytest.raises(ValueError):
        pure_field_hamiltonian(hilbert, axis="y")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        quench_ground_state(hilbert, "y")  # type: ignore[arg-type]


@pytest.mark.parametrize("alpha", [1, 2])
def test_rbm_writes_plus_state_exactly(hilbert, alpha) -> None:
    state = plus_state_rbm(hilbert, alpha=alpha)
    psi = np.array(state.to_array())
    psi = psi / np.linalg.norm(psi)
    assert abs(np.vdot(hadamard_plus_state(hilbert), psi)) == pytest.approx(
        1.0, abs=1e-14
    )
    params = state.parameters
    np.testing.assert_array_equal(np.asarray(params["visible_bias"]), 0.0)
    np.testing.assert_array_equal(np.asarray(params["Dense"]["bias"]), 0.0)
    # Each hidden unit sees exactly one visible spin.
    kernel = np.asarray(params["Dense"]["kernel"])
    np.testing.assert_array_equal(np.count_nonzero(kernel, axis=0), 1)


def test_quench_states_are_unentangled(hilbert) -> None:
    for axis in AXES:
        psi = quench_ground_state(hilbert, axis)
        for cut in ([0], [0, 1], [1, 3], [0, 1, 2]):
            assert entanglement_entropy(psi, N, cut) == pytest.approx(0.0, abs=1e-12)


def test_entanglement_entropy_known_values(hilbert) -> None:
    """Bell pair between sites 0 and 2 (fixes the site ordering), and GHZ."""
    numbers = hilbert.states_to_numbers
    up = np.ones((1, N))
    flipped = up.copy()
    flipped[0, [0, 2]] = -1
    bell = np.zeros(2**N)
    bell[int(np.ravel(numbers(up))[0])] = bell[int(np.ravel(numbers(flipped))[0])] = 1
    ln2 = np.log(2.0)
    assert entanglement_entropy(bell, N, [0]) == pytest.approx(ln2)
    assert entanglement_entropy(bell, N, [2]) == pytest.approx(ln2)
    assert entanglement_entropy(bell, N, [1]) == pytest.approx(0.0, abs=1e-12)
    assert entanglement_entropy(bell, N, [0, 2]) == pytest.approx(0.0, abs=1e-12)
    ghz = np.zeros(2**N)
    ghz[0] = ghz[-1] = 1.0
    assert entanglement_entropy(ghz, N, [0, 1]) == pytest.approx(ln2)
    with pytest.raises(ValueError):
        entanglement_entropy(ghz, N, [N])
    with pytest.raises(ValueError):
        entanglement_entropy(ghz[:-1], N, [0])


# --- Hadamard frame ----------------------------------------------------------


def test_hadamard_frame_is_an_exact_change_of_basis(hilbert, couplings) -> None:
    U = _hadamard_unitary(hilbert)
    np.testing.assert_allclose(U @ U, np.eye(2**N), atol=1e-12)
    np.testing.assert_allclose(U @ zero_state(hilbert), hadamard_plus_state(hilbert))
    lab = all_to_all_xx_hamiltonian(hilbert, h=0.4, J=couplings).to_dense()
    rotated = all_to_all_xx_hamiltonian(
        hilbert, h=0.4, J=couplings, frame="hadamard"
    ).to_dense()
    np.testing.assert_allclose(U @ lab @ U, rotated, atol=1e-12)
    with pytest.raises(ValueError):
        all_to_all_xx_hamiltonian(hilbert, h=0.4, frame="bogus")  # type: ignore[arg-type]


def test_z_quench_features_agree_between_frames(couplings) -> None:
    """<Z_i>(t) from |0>^N under H equals <X_i>(t) from |+>^N under U H U."""
    r = QELMReservoir(couplings, T=1.0, initial="quench", quench_axis="z")
    hilbert = spin_half_hilbert(N)
    grid = time_grid(r.T, r.tdvp.dt)
    for x in (-0.5, 0.3):
        lab = exact_time_evolution(r.hamiltonian(x), zero_state(hilbert), grid)
        rot = exact_time_evolution(
            r.hamiltonian(x, frame="tdvp"), hadamard_plus_state(hilbert), grid
        )
        for (name, z), x_op in zip(
            r.observables().items(), r.observables(frame="tdvp").values(), strict=True
        ):
            assert name.startswith("Z")
            np.testing.assert_allclose(
                expectation_values(lab, z), expectation_values(rot, x_op), atol=1e-12
            )


def test_frames_and_vectors_per_initial_state(couplings) -> None:
    rbm_r = QELMReservoir(couplings)
    x_r = QELMReservoir(couplings, initial="quench", quench_axis="x")
    z_r = QELMReservoir(couplings, initial="quench")
    assert z_r.quench_axis == "z"  # the default quench
    assert z_r.hadamard_frame and not x_r.hadamard_frame and not rbm_r.hadamard_frame
    hilbert = spin_half_hilbert(N)
    np.testing.assert_allclose(z_r.initial_vector(), zero_state(hilbert))
    np.testing.assert_allclose(x_r.initial_vector(), hadamard_plus_state(hilbert))
    psi = np.array(rbm_r.initial_state().to_array())
    np.testing.assert_allclose(rbm_r.initial_vector(), psi / np.linalg.norm(psi))
    with pytest.raises(ValueError):
        QELMReservoir(couplings, initial="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        QELMReservoir(couplings, initial="quench", quench_axis="y")  # type: ignore[arg-type]


# --- Quench dynamics ---------------------------------------------------------


def test_exact_quench_matches_dense_propagator(quench: QELMReservoir) -> None:
    exact = feature_matrix(quench, [0.3], method="exact")[0]
    H = quench.hamiltonian(0.3).to_dense()
    psi0 = quench_ground_state(spin_half_hilbert(N), quench.quench_axis)
    ops = [op.to_dense() for op in quench.observables().values()]
    expected = []
    for t in quench.readout_times:
        psi = sla.expm(-1j * H * t) @ psi0
        expected.extend(np.vdot(psi, op @ psi).real for op in ops)
    np.testing.assert_allclose(exact, expected, atol=1e-10)


def test_energy_of_the_quench_is_conserved(quench: QELMReservoir, couplings) -> None:
    """<H(x)> stays at its t = 0 value: h N for |0>^N, sum_{i<j} J_ij for |+>^N."""
    x = 0.3
    expected = (
        quench.field(x) * N
        if quench.quench_axis == "z"
        else float(couplings[np.triu_indices(N, 1)].sum())
    )
    grid = time_grid(quench.T, quench.tdvp.dt)
    H = quench.hamiltonian(x)
    states = exact_time_evolution(H, quench.initial_vector(), grid)
    np.testing.assert_allclose(expectation_values(states, H), expected, atol=1e-10)
    result = run_tdvp(
        quench.hamiltonian(x, frame="tdvp"),
        quench.initial_state(),
        T=quench.T,
        observables={"E": quench.hamiltonian(x, frame="tdvp")},
        settings=quench.tdvp,
    )
    assert np.max(np.abs(result.observables["E"] - expected)) < 1e-5


def test_x_quench_features_are_odd_in_the_field(couplings) -> None:
    """P = prod X_i maps H(h) to H(-h), fixes |+>^N and flips Z_i."""
    r = QELMReservoir(couplings, h0=0.0, T=1.0, initial="quench", quench_axis="x")
    for method, tol in (("exact", 1e-12), ("tdvp", 1e-8)):
        F = feature_matrix(r, [0.5, -0.5], method=method)  # type: ignore[arg-type]
        assert np.max(np.abs(F[0])) > 0.1
        np.testing.assert_allclose(F[0], -F[1], atol=tol)


def test_quench_without_couplings(quench: QELMReservoir) -> None:
    """J = 0: H commutes with Z, so <Z_i>(t) = <Z_i>(0) (1 for |0>, 0 for |+>)."""
    r = QELMReservoir(
        np.zeros((3, 3)),
        T=0.5,
        n_readouts=5,
        initial="quench",
        quench_axis=quench.quench_axis,
    )
    z0 = 1.0 if r.quench_axis == "z" else 0.0
    for method in ("exact", "tdvp"):
        F = feature_matrix(r, [-0.5, 0.7], method=method)  # type: ignore[arg-type]
        np.testing.assert_allclose(F, z0, atol=1e-8)


def test_reservoir_builds_entanglement(quench: QELMReservoir) -> None:
    grid = time_grid(quench.T, quench.tdvp.dt)
    states = exact_time_evolution(
        quench.hamiltonian(0.3), quench.initial_vector(), grid
    )
    entropy = np.array([entanglement_entropy(s, N, [0, 1]) for s in states])
    assert entropy[0] == pytest.approx(0.0, abs=1e-12)
    assert entropy[-1] > 0.01
    assert np.all(entropy <= 2 * np.log(2.0) + 1e-12)


def test_tdvp_follows_ed_from_the_quench_state(quench_features) -> None:
    exact, tdvp = quench_features
    assert np.all(np.abs(exact) <= 1 + 1e-9)
    assert np.ptp(exact) > 0.5  # the features move
    assert np.max(np.abs(tdvp - exact)) < 3e-4


def test_singular_rbm_point_freezes_the_z_quench(couplings) -> None:
    """Negative control: at W = 0 the tangent space has no pair directions."""
    r = QELMReservoir(couplings, T=1.0, initial="quench", quench_axis="z")
    obs = r.observables(frame="tdvp")
    errors = {}
    for w in (0.0, 1.0):
        result = run_tdvp(
            r.hamiltonian(0.3, frame="tdvp"),
            plus_state_rbm(spin_half_hilbert(N), pair_weight=w),
            T=r.T,
            observables=obs,
            settings=r.tdvp,
        )
        tdvp = np.stack([result.observables[k] for k in obs], axis=1)
        grid = time_grid(r.T, r.tdvp.dt)
        states = exact_time_evolution(r.hamiltonian(0.3), r.initial_vector(), grid)
        ed = np.stack(
            [expectation_values(states, op) for op in r.observables().values()], 1
        )
        errors[w] = np.max(np.abs(tdvp - ed))
    assert errors[1.0] < 3e-4
    assert errors[0.0] > 0.1


def test_quench_readout_baseline(quench: QELMReservoir, quench_features) -> None:
    """The quench reservoir learns sin(pi x) from ED, and TDVP keeps the score.

    Generalization is measured on ED features (21 inputs, held-out half).
    TDVP is checked on the 7 shared inputs: a readout fitted on ED moves by
    at most ||w||_1 times the largest feature error, and its NMSE stays small.
    """
    xs = np.linspace(-1.0, 1.0, 21)
    target = np.sin(np.pi * xs)
    F = feature_matrix(quench, xs, method="exact")
    readout = RidgeReadout(lam=1e-6).fit(F[::2], target[::2])
    assert NMSE(target[1::2], readout.predict(F[1::2])) < 1e-4

    exact, tdvp = quench_features
    y = np.sin(np.pi * INPUTS)
    readout = RidgeReadout(lam=1e-6).fit(exact, y)
    assert readout.weights is not None
    gap = np.max(np.abs(readout.predict(tdvp) - readout.predict(exact)))
    assert gap <= np.sum(np.abs(readout.weights)) * np.max(np.abs(tdvp - exact)) + 1e-12
    assert NMSE(y, readout.predict(tdvp)) < 1e-3


# --- VMC preparation ---------------------------------------------------------


@pytest.mark.parametrize("axis", AXES)
def test_vmc_finds_e_per_spin_equal_minus_h(hilbert, axis, tmp_path) -> None:
    H0 = pure_field_hamiltonian(hilbert, h=H_FIELD, axis=axis)
    state = variational_state(hilbert, rbm(), seed=45, kind="mc", n_samples=1024)
    prep = VMCGroundStatePreparation(
        H0,
        state,
        settings=VMCSettings(n_iter=200, n_samples=1024),
        paths=ExperimentPaths.at(tmp_path).create(),
        track_exact=True,
    )
    prep.run()
    exact = nk.vqs.FullSumState(hilbert, state.model, variables=state.variables)
    e_per_spin = float(complex(exact.expect(H0).mean).real) / N
    assert e_per_spin >= -H_FIELD - 1e-10  # variational bound
    assert e_per_spin == pytest.approx(-H_FIELD, abs=1e-3)
    psi = np.array(exact.to_array())
    psi = psi / np.linalg.norm(psi)
    fidelity = abs(np.vdot(quench_ground_state(hilbert, axis), psi)) ** 2
    assert fidelity > 0.99
    # Acceptance of single-spin flips: |psi|^2 is uniform for |+>^N, so every
    # move is accepted; it concentrates on one configuration for |0>^N.
    acc = prep.acceptance
    assert acc.shape == prep.iterations.shape
    assert np.all((acc >= 0) & (acc <= 1))
    if axis == "x":
        assert acc[-1] > 0.99
    else:
        assert acc[-1] < 0.1 * acc[0]
    # The exact curves are those of the parameters of each logged MC energy.
    assert prep.exact_energy.shape == prep.exact_variance.shape == acc.shape
    assert np.all(prep.exact_energy >= N * -H_FIELD - 1e-10)
    assert prep.exact_energy[-1] / N == pytest.approx(e_per_spin, abs=1e-3)
    assert np.all(prep.exact_variance >= -1e-12)
    np.testing.assert_allclose(
        prep.exact_energy[0], prep.energy[0], atol=5 * prep.energy_error[0] + 1e-8
    )
