"""run_tdvp leaves the state at psi(T), never one step past it.

NetKet's driver advances its time by summing steps. On the grid dt=5e-3,
T=0.1 the sum reaches 0.09999999999999999 at the last stopping time: the
stop is recognized (np.isclose), but the end test is a strict ``t < T``,
so the driver plans one more step. No stopping time is left to cap it, so
it is a full step to t=0.105, after the last observable was logged. The
state must still hold psi(T).
"""

import netket as nk
import numpy as np
import pytest

from QELM.NQS.dynamics import TDVPSettings, exact_time_evolution, run_tdvp, time_grid
from QELM.NQS.measurements import all_to_all_xx_hamiltonian, mean_magnetization_z
from QELM.NQS.variational import rbm, spin_half_hilbert, variational_state


def _fidelity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(np.vdot(a, b)) ** 2 / (np.vdot(a, a).real * np.vdot(b, b).real))


def test_netket_overshoots_on_this_grid() -> None:
    """Guard: the raw NetKet driver does step past T here.

    If a NetKet upgrade fixes it, this test fails and the workaround in
    run_tdvp can be reconsidered.
    """
    hilbert = spin_half_hilbert(4)
    H = all_to_all_xx_hamiltonian(hilbert, h=1.0)
    state = variational_state(hilbert, rbm(), seed=7)
    grid = time_grid(0.1, 5e-3)
    driver = nk.experimental.TDVP(
        H,
        state,
        nk.experimental.dynamics.RK45(dt=5e-3),
        qgt=nk.optimizer.qgt.QGTJacobianDense(diag_shift=1e-6, holomorphic=True),
        linear_solver=nk.optimizer.solver.svd,
    )
    driver.run(T=float(grid[-1]), out=(), obs={}, tstops=grid, show_progress=False)
    assert float(driver.t) == pytest.approx(0.105, abs=1e-12)


@pytest.mark.parametrize(("dt", "T"), [(5e-3, 0.1), (1e-3, 0.1), (1e-2, 0.3)])
def test_state_holds_psi_at_final_time(dt: float, T: float) -> None:
    hilbert = spin_half_hilbert(4)
    H = all_to_all_xx_hamiltonian(hilbert, h=1.0)
    M = mean_magnetization_z(hilbert)
    state = variational_state(hilbert, rbm(), seed=7)
    psi0 = np.asarray(state.to_array())
    result = run_tdvp(
        H,
        state,
        T=T,
        observables={"M": M},
        settings=TDVPSettings(dt=dt, diag_shift=1e-6),
        record_states=True,
    )
    assert result.states is not None
    psi_T = np.asarray(state.to_array())
    exact_T = exact_time_evolution(H, psi0, time_grid(T, dt))[-1]
    # The state left by run_tdvp is the last recorded (and logged) state...
    np.testing.assert_allclose(psi_T, result.states[-1], atol=1e-14)
    # ...and that state is psi(T): the overshoot cost 1e-4 in fidelity.
    assert _fidelity(exact_T, psi_T) == pytest.approx(1.0, abs=1e-6)
    assert result.times[-1] == pytest.approx(T, abs=1e-12)
