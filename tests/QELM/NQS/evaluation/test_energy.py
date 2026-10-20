"""Energy variance sigma_E^2 = <H^2> - <H>^2 (issue #7)."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import netket as nk
import numpy as np
import pytest

from QELM.NQS.embedding import (
    ENERGY_FILENAME,
    VARIANCE_PLOT_FILENAME,
    VMCGroundStatePreparation,
    VMCSettings,
)
from QELM.NQS.evaluation import compute_energy_variance, exact_energy_variance
from QELM.NQS.evaluation.energy import _non_negative
from QELM.NQS.paths import ExperimentPaths
from QELM.NQS.utils import makeplots
from QELM.NQS.variational import (
    rbm,
    spin_chain,
    spin_half_hilbert,
    transverse_field_ising,
    variational_state,
)

N = 4


def _field_hamiltonian(hilbert):
    """H = -sum_i Z_i: eigenstates are the Z basis states."""
    H = -1.0 * nk.operator.spin.sigmaz(hilbert, 0)
    for i in range(1, hilbert.size):
        H = H - nk.operator.spin.sigmaz(hilbert, i)
    return H


def _polarized(state, bias: float):
    """Set W = 0, b = 0, a = bias: psi(s) ∝ exp(bias * sum_i s_i)."""
    params = jax.tree_util.tree_map(jnp.zeros_like, state.parameters)
    params["visible_bias"] = params["visible_bias"] + bias
    state.parameters = params
    return state


def _ising(h: float, J: float):
    hilbert = spin_half_hilbert(N)
    return hilbert, transverse_field_ising(hilbert, spin_chain(N, pbc=True), h=h, J=J)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("kind", ["full_sum", "mc"])
def test_variance_is_non_negative(seed: int, kind: str) -> None:
    rng = np.random.default_rng(seed)
    hilbert, H = _ising(h=float(rng.uniform(-2, 2)), J=float(rng.uniform(-2, 2)))
    state = variational_state(hilbert, rbm(), seed=seed, kind=kind, n_samples=512)
    raw = float(np.real(np.asarray(state.expect(H).variance)))
    assert raw >= -1e-12
    assert compute_energy_variance(state, H) >= 0.0


@pytest.mark.parametrize("seed", range(3))
def test_full_sum_variance_equals_dense_formula(seed: int) -> None:
    hilbert, H = _ising(h=1.0, J=1.0)
    state = variational_state(hilbert, rbm(), seed=seed)
    psi = np.asarray(state.to_array())
    H_dense = np.asarray(H.to_dense())
    by_definition = (
        np.vdot(psi, H_dense @ H_dense @ psi).real
        - np.vdot(psi, H_dense @ psi).real ** 2
    )
    assert compute_energy_variance(state, H) == pytest.approx(by_definition, rel=1e-10)
    assert exact_energy_variance(psi, H) == pytest.approx(by_definition, rel=1e-10)
    # A generic (non-eigen) state has a clearly non-zero variance.
    assert by_definition > 0.1


@pytest.mark.parametrize("kind", ["full_sum", "mc"])
@pytest.mark.parametrize(("bias", "energy"), [(10.0, -N), (-10.0, N)])
def test_exact_eigenstate_has_zero_variance(
    kind: str, bias: float, energy: int
) -> None:
    """All up (ground state, E=-N) and all down (excited, E=+N) of -sum Z_i."""
    hilbert = spin_half_hilbert(N)
    H = _field_hamiltonian(hilbert)
    state = _polarized(
        variational_state(
            hilbert, rbm(param_dtype="real"), seed=0, kind=kind, n_samples=512
        ),
        bias,
    )
    assert complex(state.expect(H).mean).real == pytest.approx(energy, abs=1e-7)
    assert compute_energy_variance(state, H) < 1e-7


@pytest.mark.parametrize("alpha", [2.0, -3.0, 0.5])
@pytest.mark.parametrize("kind", ["full_sum", "mc"])
def test_variance_scales_quadratically(alpha: float, kind: str) -> None:
    hilbert, H = _ising(h=0.7, J=1.3)
    state = variational_state(hilbert, rbm(), seed=1, kind=kind, n_samples=1024)
    base = compute_energy_variance(state, H)
    # An MCState reuses its cached samples, so both estimates share them.
    scaled = compute_energy_variance(state, alpha * H)
    assert base > 0
    assert scaled == pytest.approx(alpha**2 * base, rel=1e-10)


def test_clearly_negative_variance_is_rejected() -> None:
    assert _non_negative(-1e-14, scale=1.0) == 0.0
    with pytest.raises(ValueError):
        _non_negative(-1e-3, scale=1.0)


def test_vmc_logs_and_plots_the_variance(tmp_path: Path) -> None:
    hilbert, H = _ising(h=1.0, J=1.0)
    state = variational_state(hilbert, rbm(), seed=45, kind="mc", n_samples=512)
    paths = ExperimentPaths.at(tmp_path).create()
    prep = VMCGroundStatePreparation(
        H, state, settings=VMCSettings(n_iter=40), paths=paths
    )
    prep.run()

    assert prep.energy_variance.shape == (40,)
    assert np.all(prep.energy_variance >= 0)
    with np.load(paths.exp / ENERGY_FILENAME) as saved:
        np.testing.assert_array_equal(saved["energy_variance"], prep.energy_variance)
        np.testing.assert_array_equal(saved["iters"], np.arange(40))
    log = json.loads((paths.exp / "vmc.log").read_text())
    np.testing.assert_allclose(log["Energy"]["Variance"], prep.energy_variance)
    assert (paths.plots / VARIANCE_PLOT_FILENAME).is_file()
    # The optimization reduces the variance.
    assert prep.energy_variance[-1] < prep.energy_variance[0]


def test_plot_energy_variance(tmp_path: Path) -> None:
    path = makeplots.plot_energy_variance(
        np.arange(3), np.array([1.0, 1e-5, 0.0]), directory=tmp_path
    )
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    with pytest.raises(ValueError):
        makeplots.plot_energy_variance(
            np.arange(2), np.array([1.0, -1.0]), directory=tmp_path
        )
