"""The all-to-all reservoir Hamiltonian against an independent construction."""

from functools import reduce

import netket as nk
import numpy as np
import pytest

from QRC.NQS.measurements import all_to_all_xx_hamiltonian, coupling_matrix
from QRC.NQS.variational import spin_half_hilbert

X = np.array([[0.0, 1.0], [1.0, 0.0]])
Z = np.array([[1.0, 0.0], [0.0, -1.0]])
I2 = np.eye(2)


def _site(op: np.ndarray, site: int, n: int) -> np.ndarray:
    return reduce(np.kron, [op if k == site else I2 for k in range(n)])


def _kron_hamiltonian(n: int, h: float, J: np.ndarray) -> np.ndarray:
    """H = h sum Z_i + sum_{i<j} J_ij X_i X_j, from Kronecker products."""
    H = sum(h * _site(Z, i, n) for i in range(n))
    for i in range(n):
        for j in range(i + 1, n):
            H = H + J[i, j] * _site(X, i, n) @ _site(X, j, n)
    return H


def _dense(n: int, h: float, J) -> np.ndarray:
    return np.asarray(
        all_to_all_xx_hamiltonian(spin_half_hilbert(n), h=h, J=J).to_dense()
    )


def _random_couplings(n: int, seed: int) -> np.ndarray:
    a = np.random.default_rng(seed).normal(size=(n, n))
    return a + a.T


def test_hamiltonian_is_hermitian_and_real() -> None:
    H = _dense(5, h=10.0, J=_random_couplings(5, 0))
    np.testing.assert_allclose(H, H.conj().T, atol=1e-13)
    np.testing.assert_allclose(H.imag, 0.0, atol=0)


@pytest.mark.parametrize(
    ("h", "J"),
    [(10.0, np.ones((5, 5))), (0.3, _random_couplings(5, 1)), (-2.0, 0.5)],
)
def test_spectrum_matches_kronecker_construction(h, J) -> None:
    n = 5
    expected = _kron_hamiltonian(n, h, coupling_matrix(n, J))
    np.testing.assert_allclose(
        np.linalg.eigvalsh(_dense(n, h, J)), np.linalg.eigvalsh(expected), atol=1e-11
    )


def test_matrix_elements_match_issue_construction() -> None:
    """Same operator as the LocalOperator/np.kron construction of issue #2."""
    n, h = 4, 10.0
    hi = nk.hilbert.Spin(s=0.5, N=n)
    H_ref = nk.operator.LocalOperator(hi, dtype=float)
    for i in range(n):
        H_ref += nk.operator.LocalOperator(hi, h * Z, [i])
    for i in range(n):
        for j in range(i + 1, n):
            H_ref += nk.operator.LocalOperator(hi, np.kron(X, X), [i, j])
    np.testing.assert_allclose(
        _dense(n, h, 1.0), np.asarray(H_ref.to_dense()), atol=1e-13
    )


def test_field_only_spectrum_and_parity_symmetry() -> None:
    n, h = 4, 1.5
    hilbert = spin_half_hilbert(n)
    # J = 0: eigenvalues h (N - 2k) with multiplicity C(N, k).
    spectrum = np.sort(np.linalg.eigvalsh(_dense(n, h, 0.0)))
    expected = np.sort([h * (n - 2 * bin(k).count("1")) for k in range(2**n)])
    np.testing.assert_allclose(spectrum, expected, atol=1e-13)
    # XX flips spins in pairs: H commutes with the parity prod_i Z_i.
    parity = reduce(
        lambda a, b: a @ b,
        [nk.operator.spin.sigmaz(hilbert, i).to_dense() for i in range(n)],
    )
    H = _dense(n, h, _random_couplings(n, 2))
    np.testing.assert_allclose(H @ parity - parity @ H, 0.0, atol=1e-12)


def test_coupling_matrix_validation() -> None:
    J = coupling_matrix(3, 2.0)
    np.testing.assert_array_equal(np.diag(J), 0.0)
    np.testing.assert_array_equal(J, J.T)
    with pytest.raises(ValueError):
        coupling_matrix(3, np.ones((2, 2)))
    with pytest.raises(ValueError):
        coupling_matrix(3, np.triu(np.ones((3, 3)), 1))
    with pytest.raises(ValueError):
        coupling_matrix(3, 1j * np.ones((3, 3)))
