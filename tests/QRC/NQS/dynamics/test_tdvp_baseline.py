"""TDVP with a complex RBM reproduces exact dynamics (issue #2 baseline).

System: N=6 spins, H = h sum Z_i + sum_{i<j} X_i X_j with h=10, exact
FullSumState, RK45, QGTJacobianDense, SVD solver, SEED=45, T=0.5.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from QRC.NQS.dynamics import (
    BASELINE_SEED,
    BaselineResult,
    TDVPSettings,
    tdvp_vs_exact,
    time_grid,
)
from QRC.NQS.paths import ExperimentPaths
from QRC.NQS.variational import rbm, spin_half_hilbert, variational_state

N = 6
H_FIELD = 10.0
T = 0.5
DT = 1e-3


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> tuple[BaselineResult, Path]:
    paths = ExperimentPaths.at(tmp_path_factory.mktemp("baseline")).create()
    result = tdvp_vs_exact(
        n_spins=N,
        h=H_FIELD,
        J=1.0,
        T=T,
        seed=BASELINE_SEED,
        settings=TDVPSettings(dt=DT, integrator="RK45", linear_solver="svd"),
        paths=paths,
    )
    return result, paths.root


def test_seed_is_fixed() -> None:
    assert BASELINE_SEED == 45
    hilbert = spin_half_hilbert(N)
    a = np.asarray(variational_state(hilbert, rbm(), seed=BASELINE_SEED).to_array())
    b = np.asarray(variational_state(hilbert, rbm(), seed=BASELINE_SEED).to_array())
    np.testing.assert_array_equal(a, b)


def test_magnetization_matches_exact_dynamics(run) -> None:
    result, _ = run
    np.testing.assert_allclose(result.tdvp, result.exact, atol=1e-3, rtol=0)


def test_time_arrays_match(run) -> None:
    result, _ = run
    # NetKet's TDVP log stores the physical time t as "iters" (not a step
    # index), and tstops forces the adaptive integrator onto the ED grid.
    assert result.tdvp_times.shape == result.times.shape == (round(T / DT) + 1,)
    np.testing.assert_allclose(result.tdvp_times, result.times, atol=1e-12, rtol=0)
    np.testing.assert_allclose(result.times, time_grid(T, DT), atol=0)


def test_initial_values_coincide(run) -> None:
    result, _ = run
    assert result.tdvp[0] == pytest.approx(result.exact[0], abs=1e-12)


def test_energy_is_conserved(run) -> None:
    result, _ = run
    # Exact unitary dynamics conserves <H>; real-time TDVP does too, up to
    # the regularization and integration error.
    np.testing.assert_allclose(result.exact_energy, result.exact_energy[0], atol=1e-9)
    np.testing.assert_allclose(result.tdvp_energy, result.exact_energy[0], atol=1e-4)


def test_final_state_fidelity(run) -> None:
    result, _ = run
    assert result.final_fidelity == pytest.approx(1.0, abs=1e-4)
    assert result.final_fidelity <= 1.0 + 1e-12


def test_outputs_are_written(run) -> None:
    result, root = run
    paths = ExperimentPaths.at(root)
    assert result.plot == paths.plots / "tdvp_vs_ed.png"
    assert result.plot.is_file()
    log = json.loads((paths.exp / "tdvp.log").read_text())
    np.testing.assert_allclose(log["Magnetization"]["iters"], result.tdvp_times)
    with np.load(paths.exp / "tdvp_vs_ed.npz") as archive:
        np.testing.assert_array_equal(archive["exact"], result.exact)
