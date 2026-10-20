"""TDVP (RBM) versus exact dynamics on the all-to-all reservoir chain."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from QELM.NQS.dynamics.exact import (
    exact_time_evolution,
    expectation_values,
    time_grid,
)
from QELM.NQS.dynamics.tdvp import TDVPSettings, run_tdvp
from QELM.NQS.measurements import all_to_all_xx_hamiltonian, mean_magnetization_z
from QELM.NQS.paths import ExperimentPaths
from QELM.NQS.utils.makeplots import plot_exact_vs_tdvp
from QELM.NQS.variational import rbm, spin_half_hilbert, variational_state

BASELINE_SEED = 45


@dataclass(frozen=True)
class BaselineResult:
    """Exact and TDVP trajectories on a shared time grid.

    Attributes:
        times: Grid used for the exact evolution, shape ``(n_times,)``.
        tdvp_times: Times read from NetKet's TDVP log, shape ``(n_times,)``.
        exact: Exact ``<Z>`` (mean magnetization), shape ``(n_times,)``.
        tdvp: TDVP ``<Z>``, shape ``(n_times,)``.
        exact_energy: Exact ``<H>``, constant up to round-off.
        tdvp_energy: TDVP ``<H>``.
        final_fidelity: ``|<psi_exact(T)|psi_tdvp(T)>|^2``.
        plot: Path of the comparison figure, when one was written.
    """

    times: np.ndarray
    tdvp_times: np.ndarray
    exact: np.ndarray
    tdvp: np.ndarray
    exact_energy: np.ndarray
    tdvp_energy: np.ndarray
    final_fidelity: float
    plot: Path | None = None

    @property
    def max_abs_error(self) -> float:
        """Largest ``|<Z>_TDVP - <Z>_exact|`` over the grid."""
        return float(np.max(np.abs(self.tdvp - self.exact)))


def tdvp_vs_exact(
    *,
    n_spins: int = 6,
    h: float = 10.0,
    J: float | ArrayLike = 1.0,
    T: float = 5.0,
    seed: int = BASELINE_SEED,
    alpha: int = 1,
    settings: TDVPSettings | None = None,
    paths: ExperimentPaths | None = None,
) -> BaselineResult:
    """Run the exact and the TDVP evolution from the same initial RBM state.

    The initial state is the (normalized) RBM with seeded parameters, held
    as an exact ``FullSumState``. Its vector is the ED initial condition,
    so any difference comes from the TDVP projection and integration.

    Args:
        n_spins: Number of spins.
        h: Field along Z.
        J: Scalar or ``(N, N)`` XX couplings.
        T: Final time.
        seed: RBM parameter seed.
        alpha: RBM hidden-unit density.
        settings: TDVP settings (``dt``, integrator, QGT shift, solver).
        paths: Run folders. When given, the TDVP JSON log and the arrays
            go to ``exp/`` and the figure to ``plots/``.

    Returns:
        Both trajectories and their diagnostics.
    """
    settings = settings or TDVPSettings()
    hilbert = spin_half_hilbert(n_spins)
    H = all_to_all_xx_hamiltonian(hilbert, h=h, J=J)
    magnetization = mean_magnetization_z(hilbert)
    state = variational_state(hilbert, rbm(alpha=alpha), seed=seed)

    times = time_grid(T, settings.dt)
    psi0 = np.asarray(state.to_array())
    exact_states = exact_time_evolution(H, psi0, times)
    exact = expectation_values(exact_states, magnetization)
    exact_energy = expectation_values(exact_states, H)

    log_path = None if paths is None else paths.exp / "tdvp"
    result = run_tdvp(
        H,
        state,
        T=T,
        observables={"Magnetization": magnetization, "Energy": H},
        settings=settings,
        log_path=log_path,
    )
    psi_T = np.asarray(state.to_array())
    fidelity = float(np.abs(np.vdot(exact_states[-1], psi_T)) ** 2)

    plot = None
    if paths is not None:
        np.savez(
            paths.exp / "tdvp_vs_ed.npz",
            times=times,
            tdvp_times=result.times,
            exact=exact,
            tdvp=result.observables["Magnetization"],
        )
        plot = plot_exact_vs_tdvp(
            times,
            exact,
            result.times,
            result.observables["Magnetization"],
            filename="tdvp_vs_ed.png",
            directory=paths.plots,
            ylabel=r"$\langle Z \rangle$",
            title=f"TDVP vs ED (N={n_spins}, h={h})",
        )
    return BaselineResult(
        times=times,
        tdvp_times=result.times,
        exact=exact,
        tdvp=result.observables["Magnetization"],
        exact_energy=exact_energy,
        tdvp_energy=result.observables["Energy"],
        final_fidelity=fidelity,
        plot=plot,
    )


__all__ = ["BASELINE_SEED", "BaselineResult", "tdvp_vs_exact"]
