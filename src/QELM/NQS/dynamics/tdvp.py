"""Real-time TDVP evolution of a NetKet variational state."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import netket as nk
import numpy as np

from QELM.NQS.dynamics.exact import time_grid

Integrator = Literal["RK4", "RK45", "RK23", "Euler", "Heun"]
LinearSolver = Literal["svd", "pinv_smooth"]


@dataclass(frozen=True)
class TDVPSettings:
    """Numerical settings of a TDVP run.

    Attributes:
        dt: Time step (initial step for adaptive integrators). The
            observables are recorded on ``time_grid(T, dt)``.
        integrator: NetKet Runge-Kutta scheme. Adaptive ``RK45`` and
            ``RK23`` are forced to land on every grid time.
        diag_shift: Tikhonov shift added to the QGT (default ``1e-4``, as
            in the VMC). It stabilizes the solve but biases the dynamics:
            on the N=6 baseline, ``1e-4`` gives a magnetization error of
            1.1e-3 at t=0.5, ``1e-6`` gives 8e-5. Pass ``1e-6`` when the
            trajectory must match ED to better than 1e-3.
        holomorphic: ``True`` for an ansatz holomorphic in complex
            parameters, such as the complex RBM.
        linear_solver: ``"svd"`` (pseudo-inverse with cutoff ``rcond``) or
            NetKet's ``"pinv_smooth"``.
        rcond: Relative singular-value cutoff of the solver; ``None`` keeps
            the solver default.
    """

    dt: float = 1e-3
    integrator: Integrator = "RK45"
    diag_shift: float = 1e-4
    holomorphic: bool = True
    linear_solver: LinearSolver = "svd"
    rcond: float | None = None


@dataclass(frozen=True)
class TDVPResult:
    """Observables recorded during a TDVP run.

    Attributes:
        times: Times read from NetKet's log, shape ``(n_times,)``.
        observables: Name mapped to the real part of the mean value,
            shape ``(n_times,)``.
        imaginary_parts: Name mapped to the largest ``|Im <O>|`` (zero for
            a Hermitian operator up to round-off).
        states: Normalized state vectors ``psi(t)`` on the same times,
            shape ``(n_times, dim)``, when ``record_states=True``; else
            ``None``. Only for states that can be written as a vector.
    """

    times: np.ndarray
    observables: dict[str, np.ndarray]
    imaginary_parts: dict[str, float] = field(default_factory=dict)
    states: np.ndarray | None = None


def _solver(settings: TDVPSettings) -> Any:
    if settings.linear_solver == "svd":
        if settings.rcond is None:
            return nk.optimizer.solver.svd
        rcond = settings.rcond
        return lambda A, b, **kw: nk.optimizer.solver.svd(A, b, rcond=rcond, **kw)
    if settings.linear_solver == "pinv_smooth":
        if settings.rcond is None:
            return nk.optimizer.solver.pinv_smooth
        rcond = settings.rcond
        return lambda A, b, **kw: nk.optimizer.solver.pinv_smooth(
            A, b, rtol=rcond, **kw
        )
    raise ValueError(f"Unknown linear solver {settings.linear_solver!r}.")


def run_tdvp(
    hamiltonian: nk.operator.AbstractOperator,
    state: nk.vqs.VariationalState,
    *,
    T: float,
    observables: Mapping[str, nk.operator.AbstractOperator],
    settings: TDVPSettings | None = None,
    log_path: Path | str | None = None,
    record_states: bool = False,
) -> TDVPResult:
    """Evolve ``state`` in real time under ``hamiltonian`` from ``t0 = 0`` to ``T``.

    Solves the TDVP equation ``S dθ/dt = -i F`` with the dense Jacobian
    QGT. ``state`` is updated in place: after the call it holds ``ψ(T)``.

    Args:
        hamiltonian: Hermitian NetKet operator.
        state: Variational state, modified in place.
        T: Final time.
        observables: Name mapped to an operator recorded on the grid.
        settings: Numerical settings; defaults to :class:`TDVPSettings`.
        log_path: When given, NetKet also writes its JSON log to
            ``<log_path>.log`` (for instance in the run's ``exp/``).
        record_states: When ``True``, also store ``state.to_array()`` at
            every grid time (``2**N`` amplitudes per time, so small ``N``
            only). It gives the quantities that are not operators, such as
            the entanglement entropy or the fidelity with ED along the way.

    Returns:
        The recorded times and observables.
    """
    settings = settings or TDVPSettings()
    integrator_cls = getattr(nk.experimental.dynamics, settings.integrator)
    driver = nk.experimental.TDVP(
        hamiltonian,
        state,
        integrator_cls(dt=settings.dt),
        qgt=nk.optimizer.qgt.QGTJacobianDense(
            diag_shift=settings.diag_shift, holomorphic=settings.holomorphic
        ),
        linear_solver=_solver(settings),
    )
    grid = time_grid(T, settings.dt)
    memory = nk.logging.RuntimeLog()
    loggers: list[Any] = [memory]
    if log_path is not None:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        loggers.append(nk.logging.JsonLog(str(log_path), save_params=False))
    recorded: list[np.ndarray] = []

    def record(step: Any, log_data: Any, drv: Any) -> bool:
        # NetKet calls this only at the stopping times (the grid), before
        # the step leaves them, so drv.state holds psi(t) at that time.
        recorded.append(np.asarray(drv.state.to_array()))
        return True

    driver.run(
        T=float(grid[-1]),
        out=tuple(loggers),
        obs=dict(observables),
        tstops=grid,
        show_progress=False,
        callback=record if record_states else None,
    )
    for logger in loggers[1:]:
        logger.flush()

    times: np.ndarray | None = None
    values: dict[str, np.ndarray] = {}
    imaginary: dict[str, float] = {}
    for name in observables:
        history = memory.data[name]
        means = np.asarray(history.Mean)
        if times is None:
            times = np.asarray(history.iters, dtype=float)
        values[name] = means.real.astype(float)
        imaginary[name] = float(np.max(np.abs(means.imag))) if means.size else 0.0
    if times is None:
        times = grid
    states = np.stack(recorded) if record_states else None
    return TDVPResult(
        times=times, observables=values, imaginary_parts=imaginary, states=states
    )


__all__ = ["TDVPResult", "TDVPSettings", "run_tdvp"]
