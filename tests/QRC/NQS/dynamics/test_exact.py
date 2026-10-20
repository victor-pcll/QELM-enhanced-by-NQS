"""The exact (ED) evolution is unitary and matches analytic results."""

import netket as nk
import numpy as np
import pytest

from QRC.NQS.dynamics import exact_time_evolution, expectation_values, time_grid
from QRC.NQS.measurements import all_to_all_xx_hamiltonian
from QRC.NQS.variational import spin_half_hilbert


def _random_state(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    psi = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    return psi / np.linalg.norm(psi)


def test_time_grid() -> None:
    t = time_grid(0.5, 1e-3)
    assert t.shape == (501,)
    assert t[0] == 0.0
    assert t[-1] == pytest.approx(0.5, abs=1e-15)
    np.testing.assert_allclose(np.diff(t), 1e-3, atol=1e-15)
    with pytest.raises(ValueError):
        time_grid(0.5, 0.3)
    with pytest.raises(ValueError):
        time_grid(-1.0, 0.1)


def test_single_spin_larmor_precession() -> None:
    """H = h Z on |+x>: <X>(t) = cos(2 h t), <Z>(t) = 0."""
    h = 1.3
    hilbert = spin_half_hilbert(1)
    H = all_to_all_xx_hamiltonian(hilbert, h=h)
    plus_x = np.array([1.0, 1.0]) / np.sqrt(2)
    t = time_grid(2.0, 0.01)
    states = exact_time_evolution(H, plus_x, t)
    sx = nk.operator.spin.sigmax(hilbert, 0)
    sz = nk.operator.spin.sigmaz(hilbert, 0)
    np.testing.assert_allclose(
        expectation_values(states, sx), np.cos(2 * h * t), atol=1e-12
    )
    np.testing.assert_allclose(expectation_values(states, sz), 0.0, atol=1e-12)


@pytest.mark.parametrize("uniform", [True, False])
def test_matches_eigendecomposition(uniform: bool) -> None:
    n = 4
    hilbert = spin_half_hilbert(n)
    H = all_to_all_xx_hamiltonian(hilbert, h=2.0, J=1.0)
    psi0 = _random_state(2**n, 0)
    t = time_grid(1.0, 0.05) if uniform else np.array([0.3, 0.0, 1.7])
    states = exact_time_evolution(H, psi0, t)

    E, V = np.linalg.eigh(np.asarray(H.to_dense()))
    coeffs = V.conj().T @ psi0
    expected = np.stack([V @ (np.exp(-1j * E * ti) * coeffs) for ti in t])
    np.testing.assert_allclose(states, expected, atol=1e-10)


def test_unitarity_and_energy_conservation() -> None:
    n = 5
    hilbert = spin_half_hilbert(n)
    H = all_to_all_xx_hamiltonian(hilbert, h=10.0, J=1.0)
    psi0 = _random_state(2**n, 1)
    t = time_grid(1.0, 0.01)
    states = exact_time_evolution(H, psi0, t)
    np.testing.assert_allclose(states[0], psi0, atol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(states, axis=1), 1.0, atol=1e-12)
    energy = expectation_values(states, H)
    np.testing.assert_allclose(energy, energy[0], atol=1e-10)


def test_shape_errors() -> None:
    H = all_to_all_xx_hamiltonian(spin_half_hilbert(2), h=1.0)
    with pytest.raises(ValueError):
        exact_time_evolution(H, np.ones(3), [0.0])
    with pytest.raises(ValueError):
        exact_time_evolution(H, np.ones(4), [])
