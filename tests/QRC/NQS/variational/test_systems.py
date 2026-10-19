"""Physical checks of the NetKet systems against independent constructions."""

from functools import reduce
from math import comb

import numpy as np
import pytest

from QRC.NQS.models import IsingModel
from QRC.NQS.variational import (
    mean_magnetization_z,
    spin_chain,
    spin_half_hilbert,
    transverse_field_ising,
)

X = np.array([[0.0, 1.0], [1.0, 0.0]])
Z = np.array([[1.0, 0.0], [0.0, -1.0]])
I2 = np.eye(2)


def _site_op(op: np.ndarray, site: int, n: int) -> np.ndarray:
    return reduce(np.kron, [op if k == site else I2 for k in range(n)])


def _kron_ising(n: int, h: float, J: float, pbc: bool) -> np.ndarray:
    """H = -h sum X_i + J sum_<ij> Z_i Z_j built from Kronecker products."""
    H = sum(-h * _site_op(X, i, n) for i in range(n))
    bonds = [(i, i + 1) for i in range(n - 1)]
    if pbc and n > 2:
        bonds.append((n - 1, 0))
    for i, j in bonds:
        H = H + J * _site_op(Z, i, n) @ _site_op(Z, j, n)
    return H


def _tfim_pbc_ground_energy(n: int, h: float, J: float) -> float:
    """Exact Jordan-Wigner ground energy of the periodic chain, even ``n``.

    The ground state lies in the even-parity sector with antiperiodic
    fermions, k = pi (2m + 1) / n, and E0 = -sum_k sqrt(J^2 + h^2 - 2 J h cos k).
    """
    k = np.pi * (2 * np.arange(n) + 1) / n
    return float(-np.sum(np.sqrt(J**2 + h**2 - 2 * J * h * np.cos(k))))


def _dense(n: int, h: float, J: float, pbc: bool) -> np.ndarray:
    graph = spin_chain(n, pbc=pbc)
    hilbert = spin_half_hilbert(n)
    return np.asarray(transverse_field_ising(hilbert, graph, h=h, J=J).to_dense())


def test_hilbert_dimension_is_two_to_the_n() -> None:
    for n in (1, 3, 6):
        assert spin_half_hilbert(n).n_states == 2**n


def test_chain_edges_follow_boundary_conditions() -> None:
    assert spin_chain(5, pbc=False).n_edges == 4
    assert spin_chain(5, pbc=True).n_edges == 5


@pytest.mark.parametrize("pbc", [False, True])
def test_ising_is_hermitian_and_real(pbc: bool) -> None:
    H = _dense(5, h=0.7, J=1.3, pbc=pbc)
    np.testing.assert_allclose(H, H.conj().T, atol=1e-14)
    np.testing.assert_allclose(H.imag, 0.0, atol=1e-14)


@pytest.mark.parametrize("pbc", [False, True])
@pytest.mark.parametrize(("h", "J"), [(1.0, 1.0), (0.3, -1.2), (2.5, 0.4)])
def test_ising_spectrum_matches_kronecker_construction(
    pbc: bool, h: float, J: float
) -> None:
    n = 5
    netket_spectrum = np.linalg.eigvalsh(_dense(n, h=h, J=J, pbc=pbc))
    kron_spectrum = np.linalg.eigvalsh(_kron_ising(n, h=h, J=J, pbc=pbc))
    np.testing.assert_allclose(netket_spectrum, kron_spectrum, atol=1e-12)


@pytest.mark.parametrize(("h", "J"), [(1.0, 1.0), (0.5, 1.0), (2.0, -0.7)])
def test_ising_ground_energy_matches_jordan_wigner(h: float, J: float) -> None:
    n = 8
    e0 = np.linalg.eigvalsh(_dense(n, h=h, J=J, pbc=True))[0]
    assert e0 == pytest.approx(_tfim_pbc_ground_energy(n, h, J), abs=1e-10)


def test_ising_limits_are_exact() -> None:
    n = 4
    # Pure field: every spin aligns with X, E0 = -N |h|.
    assert np.linalg.eigvalsh(_dense(n, h=1.5, J=0.0, pbc=False))[0] == (
        pytest.approx(-n * 1.5, abs=1e-12)
    )
    # Pure coupling on an open chain: Neel state, E0 = -(N - 1) |J|.
    assert np.linalg.eigvalsh(_dense(n, h=0.0, J=2.0, pbc=False))[0] == (
        pytest.approx(-(n - 1) * 2.0, abs=1e-12)
    )


def test_mean_magnetization_spectrum() -> None:
    n = 4
    hilbert = spin_half_hilbert(n)
    M = np.asarray(mean_magnetization_z(hilbert).to_dense())
    np.testing.assert_allclose(M, np.diag(np.diag(M)), atol=0)
    values = np.sort(np.diag(M).real)
    # (N - 2k)/N with multiplicity binom(N, k).
    expected = np.sort(
        np.concatenate([np.full(comb(n, k), (n - 2 * k) / n) for k in range(n + 1)])
    )
    np.testing.assert_allclose(values, expected, atol=1e-14)


def test_mean_magnetization_commutes_with_zz_but_not_field() -> None:
    n = 4
    hilbert = spin_half_hilbert(n)
    graph = spin_chain(n)
    M = np.asarray(mean_magnetization_z(hilbert).to_dense())
    H_zz = np.asarray(transverse_field_ising(hilbert, graph, h=0.0, J=1.0).to_dense())
    H_x = np.asarray(transverse_field_ising(hilbert, graph, h=1.0, J=0.0).to_dense())
    np.testing.assert_allclose(M @ H_zz - H_zz @ M, 0.0, atol=1e-14)
    assert np.abs(M @ H_x - H_x @ M).max() > 0.1


def test_invalid_sizes_raise() -> None:
    with pytest.raises(ValueError):
        spin_chain(1)
    with pytest.raises(ValueError):
        spin_half_hilbert(0)
    with pytest.raises(ValueError):
        transverse_field_ising(spin_half_hilbert(3), spin_chain(4), h=1.0, J=1.0)


def test_ising_model_dense_hamiltonian_matches_config() -> None:
    model = IsingModel({"N": 4, "J": 1.0, "h": 0.5, "pbc": True})
    H = model.hamiltonian()
    assert H.shape == (16, 16)
    np.testing.assert_allclose(H, H.T.conj(), atol=1e-14)
    np.testing.assert_allclose(
        np.linalg.eigvalsh(H),
        np.linalg.eigvalsh(_kron_ising(4, h=0.5, J=1.0, pbc=True)),
        atol=1e-12,
    )
