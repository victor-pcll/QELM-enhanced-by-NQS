"""Accumulated error of TDVP (NQS) against exact dynamics, and its field scan.

The pointwise error of a TDVP trajectory says how far it is at one time.
The accumulated absolute error

    E_O(T) = ∫_0^T |<O>_TDVP(t) - <O>_ED(t)| dt

says how much it has drifted over the whole run, which is what a reservoir
read out along the trajectory sees. It is computed here for the energy,
the local magnetizations ``<X_i>`` and ``<Z_i>``, the half-chain
entanglement entropy and the infidelity ``1 - |<psi_ED|psi_TDVP>|^2``, on
the all-to-all chain ``H = h Σ Z_i + Σ_{i<j} J_ij X_i X_j``.

All quantities are evaluated from the state vectors with the same code for
both trajectories, so a difference can only come from the states.
"""

from __future__ import annotations

import time as _time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import netket as nk
import numpy as np
from numpy.typing import ArrayLike

from QELM.NQS.dynamics.baseline import BASELINE_SEED
from QELM.NQS.dynamics.exact import (
    exact_time_evolution,
    expectation_values,
    time_grid,
)
from QELM.NQS.dynamics.tdvp import TDVPSettings, run_tdvp
from QELM.NQS.measurements import all_to_all_xx_hamiltonian
from QELM.NQS.variational import rbm, spin_half_hilbert, variational_state

Axis = Literal["x", "y", "z"]

#: Keys of :meth:`ErrorComparison.accumulated_errors`, in plotting order.
ERROR_KEYS = (
    "energy",
    "magnetization_z",
    "sigma_x",
    "sigma_z",
    "entropy",
    "infidelity",
)


def local_pauli(hilbert: nk.hilbert.Spin, axis: Axis) -> list[Any]:
    """Return the single-site Pauli operators ``[P_0, ..., P_{N-1}]``.

    Args:
        hilbert: Spin-1/2 Hilbert space of ``N`` sites.
        axis: ``"x"``, ``"y"`` or ``"z"``.

    Returns:
        ``N`` NetKet operators, Pauli matrices (eigenvalues ``±1``).

    Raises:
        ValueError: If ``axis`` is unknown.
    """
    builders = {
        "x": nk.operator.spin.sigmax,
        "y": nk.operator.spin.sigmay,
        "z": nk.operator.spin.sigmaz,
    }
    if axis not in builders:
        raise ValueError(f"axis must be 'x', 'y' or 'z', got {axis!r}.")
    return [builders[axis](hilbert, i) for i in range(hilbert.size)]


def entanglement_entropy(
    states: ArrayLike, n_spins: int, n_left: int | None = None
) -> np.ndarray:
    """Return the von Neumann entropy of the first ``n_left`` spins.

    ``S = -Tr ρ_A ln ρ_A`` with ``ρ_A`` the reduced state of sites
    ``0, ..., n_left - 1``, from the Schmidt values (SVD of the amplitudes
    reshaped to ``(2**n_left, 2**(N - n_left))``). This uses NetKet's
    ordering of the basis, where site 0 is the most significant bit. The
    entropy is in nats: ``0`` for a product state, at most
    ``min(n_left, N - n_left) ln 2``.

    Args:
        states: Pure states of shape ``(n_times, 2**N)`` or ``(2**N,)``;
            they are normalized here.
        n_spins: Number of spins ``N``.
        n_left: Size of the left block; ``N // 2`` (half chain) by default.

    Returns:
        Entropies of shape ``(n_times,)`` (or a 0-d array).

    Raises:
        ValueError: If the dimension is not ``2**N`` or the cut is empty.
    """
    psi = np.asarray(states, dtype=complex)
    if psi.shape[-1] != 2**n_spins:
        raise ValueError(f"states have dimension {psi.shape[-1]}, not 2**{n_spins}.")
    cut = n_spins // 2 if n_left is None else n_left
    if not 0 < cut < n_spins:
        raise ValueError(f"n_left must be in 1..{n_spins - 1}, got {cut}.")
    matrices = psi.reshape(*psi.shape[:-1], 2**cut, 2 ** (n_spins - cut))
    schmidt = np.linalg.svd(matrices, compute_uv=False)
    p = schmidt**2
    p = p / p.sum(axis=-1, keepdims=True)
    p_log_p = np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0)
    return np.asarray(np.maximum(-p_log_p.sum(axis=-1), 0.0))


def accumulated_absolute_error(
    times: ArrayLike, approx: ArrayLike, exact: ArrayLike
) -> np.ndarray:
    """Return ``∫ |approx(t) - exact(t)| dt`` over the time grid.

    The integral uses the trapezoidal rule along the first axis, so it is
    exact for piecewise-linear errors and has units of ``[O] × time``.
    Divide by ``T`` for the time-averaged error.

    Args:
        times: Strictly increasing times, shape ``(n_times,)``.
        approx: Approximate values, shape ``(n_times, ...)``.
        exact: Reference values, same shape as ``approx``.

    Returns:
        One integral per trailing index, shape ``approx.shape[1:]`` (a 0-d
        array for a single observable).

    Raises:
        ValueError: On mismatched shapes or a non-increasing time grid.
    """
    t = np.asarray(times, dtype=float)
    a = np.asarray(approx, dtype=float)
    b = np.asarray(exact, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"approx has shape {a.shape}, exact {b.shape}.")
    if t.ndim != 1 or a.ndim == 0 or a.shape[0] != t.size:
        raise ValueError(f"times has shape {t.shape}, values {a.shape}.")
    if t.size < 2 or np.any(np.diff(t) <= 0):
        raise ValueError("times must be strictly increasing, with 2+ points.")
    return np.asarray(np.trapezoid(np.abs(a - b), t, axis=0))


@dataclass(frozen=True)
class Trajectory:
    """Observables along one trajectory ``psi(t)``.

    Attributes:
        energy: ``<H>``, shape ``(n_times,)``.
        sigma_x: ``<X_i>``, shape ``(n_times, N)``.
        sigma_z: ``<Z_i>``, shape ``(n_times, N)``.
        entropy: Half-chain entanglement entropy (nats), ``(n_times,)``.
    """

    energy: np.ndarray
    sigma_x: np.ndarray
    sigma_z: np.ndarray
    entropy: np.ndarray

    @property
    def magnetization_z(self) -> np.ndarray:
        """Mean magnetization ``(1/N) Σ_i <Z_i>``, shape ``(n_times,)``."""
        return np.asarray(self.sigma_z.mean(axis=1))


def trajectory_observables(
    states: ArrayLike, hamiltonian: Any, hilbert: nk.hilbert.Spin
) -> Trajectory:
    """Evaluate every tracked observable on a sequence of state vectors.

    Args:
        states: Shape ``(n_times, 2**N)``.
        hamiltonian: The Hamiltonian (NetKet operator or matrix).
        hilbert: Spin-1/2 Hilbert space of ``N`` sites.

    Returns:
        The trajectory's observables.
    """
    psi = np.asarray(states, dtype=complex)
    if psi.ndim != 2:
        raise ValueError(f"states must have shape (n_times, dim), got {psi.shape}.")
    columns = {
        axis: np.stack(
            [expectation_values(psi, op) for op in local_pauli(hilbert, axis)],
            axis=1,
        )
        for axis in ("x", "z")
    }
    return Trajectory(
        energy=expectation_values(psi, hamiltonian),
        sigma_x=columns["x"],
        sigma_z=columns["z"],
        entropy=entanglement_entropy(psi, hilbert.size),
    )


@dataclass(frozen=True)
class ErrorComparison:
    """Exact and TDVP trajectories from the same initial state.

    Attributes:
        n_spins: Number of spins.
        h: Field along Z.
        times: Shared time grid, shape ``(n_times,)``.
        exact: Observables of the exact evolution.
        tdvp: Observables of the TDVP evolution.
        fidelity: ``|<psi_ED(t)|psi_TDVP(t)>|^2``, shape ``(n_times,)``.
        wall_time: Seconds spent in the TDVP run.
    """

    n_spins: int
    h: float
    times: np.ndarray
    exact: Trajectory
    tdvp: Trajectory
    fidelity: np.ndarray
    wall_time: float = 0.0

    def accumulated_errors(self) -> dict[str, float]:
        """Return the accumulated absolute error of each observable.

        ``sigma_x`` and ``sigma_z`` are averaged over sites,
        ``(1/N) Σ_i ∫ |Δ<P_i>| dt``. ``magnetization_z`` is the error of
        the site average, so it is at most ``sigma_z`` (triangle
        inequality). ``infidelity`` is ``∫ (1 - F(t)) dt``.

        Returns:
            One float per key of :data:`ERROR_KEYS`.
        """
        t = self.times
        exact, tdvp = self.exact, self.tdvp

        def acc(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            return accumulated_absolute_error(t, a, b)

        return {
            "energy": float(acc(tdvp.energy, exact.energy)),
            "magnetization_z": float(acc(tdvp.magnetization_z, exact.magnetization_z)),
            "sigma_x": float(np.mean(acc(tdvp.sigma_x, exact.sigma_x))),
            "sigma_z": float(np.mean(acc(tdvp.sigma_z, exact.sigma_z))),
            "entropy": float(acc(tdvp.entropy, exact.entropy)),
            "infidelity": float(acc(self.fidelity, np.ones_like(self.fidelity))),
        }


def compare_with_exact(
    *,
    n_spins: int,
    h: float,
    J: float | ArrayLike = 1.0,
    T: float,
    seed: int = BASELINE_SEED,
    alpha: int = 1,
    init_std: float | None = None,
    settings: TDVPSettings | None = None,
) -> ErrorComparison:
    """Run ED and TDVP from the same seeded RBM state and compare them.

    The set-up is the one of :func:`QELM.NQS.dynamics.tdvp_vs_exact`: a
    complex RBM held exactly as a ``FullSumState``, whose vector is also the
    ED initial condition. The TDVP state vector is recorded on the grid.

    Args:
        n_spins: Number of spins (``2**N`` amplitudes are stored per time).
        h: Field along Z.
        J: Scalar or ``(N, N)`` XX couplings.
        T: Final time.
        seed: RBM parameter seed.
        alpha: RBM hidden-unit density.
        init_std: RBM initialization width, see :func:`rbm`.
        settings: TDVP settings; :class:`TDVPSettings` by default.

    Returns:
        Both trajectories and the fidelity along the run.
    """
    settings = settings or TDVPSettings()
    hilbert = spin_half_hilbert(n_spins)
    H = all_to_all_xx_hamiltonian(hilbert, h=h, J=J)
    state = variational_state(hilbert, rbm(alpha=alpha, init_std=init_std), seed=seed)
    times = time_grid(T, settings.dt)
    exact_states = exact_time_evolution(H, np.asarray(state.to_array()), times)

    start = _time.perf_counter()
    result = run_tdvp(
        H, state, T=T, observables={}, settings=settings, record_states=True
    )
    wall_time = _time.perf_counter() - start
    assert result.states is not None
    if result.states.shape != exact_states.shape:
        raise RuntimeError(
            f"TDVP recorded {result.states.shape[0]} states, the grid has {times.size}."
        )
    overlap = np.sum(exact_states.conj() * result.states, axis=1)
    norms = np.sum(np.abs(exact_states) ** 2, axis=1) * np.sum(
        np.abs(result.states) ** 2, axis=1
    )
    return ErrorComparison(
        n_spins=n_spins,
        h=float(h),
        times=times,
        exact=trajectory_observables(exact_states, H, hilbert),
        tdvp=trajectory_observables(result.states, H, hilbert),
        fidelity=np.asarray(np.abs(overlap) ** 2 / norms),
        wall_time=wall_time,
    )


@dataclass(frozen=True)
class FieldSweep:
    """Accumulated errors for one ``N`` over a list of fields.

    Attributes:
        n_spins: Number of spins.
        T: Final time of every run.
        fields: Fields ``h``, shape ``(n_fields,)``.
        errors: Key of :data:`ERROR_KEYS` mapped to shape ``(n_fields,)``.
        final_fidelity: ``F(T)`` for each field.
        wall_time: TDVP wall time (seconds) for each field.
    """

    n_spins: int
    T: float
    fields: np.ndarray
    errors: dict[str, np.ndarray]
    final_fidelity: np.ndarray
    wall_time: np.ndarray

    def save(self, path: Path | str) -> Path:
        """Write the sweep to an ``.npz`` archive and return its path."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, Any] = {
            "n_spins": self.n_spins,
            "T": self.T,
            "fields": self.fields,
            "final_fidelity": self.final_fidelity,
            "wall_time": self.wall_time,
        }
        arrays.update({f"error_{key}": value for key, value in self.errors.items()})
        np.savez(target, **arrays)
        return target

    @classmethod
    def load(cls, path: Path | str) -> FieldSweep:
        """Read a sweep written by :meth:`save`."""
        with np.load(path) as archive:
            return cls(
                n_spins=int(archive["n_spins"]),
                T=float(archive["T"]),
                fields=archive["fields"],
                errors={key: archive[f"error_{key}"] for key in ERROR_KEYS},
                final_fidelity=archive["final_fidelity"],
                wall_time=archive["wall_time"],
            )


def accumulated_error_vs_field(
    *,
    n_spins: int,
    fields: Sequence[float] | ArrayLike,
    J: float | ArrayLike = 1.0,
    T: float,
    seed: int = BASELINE_SEED,
    alpha: int = 1,
    init_std: float | None = None,
    settings: TDVPSettings | None = None,
) -> FieldSweep:
    """Run :func:`compare_with_exact` for each field and collect the errors.

    Every run starts from the same seeded RBM, so only ``h`` changes.

    Args:
        n_spins: Number of spins.
        fields: Fields ``h`` to scan.
        J: Scalar or ``(N, N)`` XX couplings.
        T: Final time of every run.
        seed: RBM parameter seed.
        alpha: RBM hidden-unit density.
        init_std: RBM initialization width.
        settings: TDVP settings.

    Returns:
        The sweep.
    """
    h_values = np.asarray(fields, dtype=float).reshape(-1)
    if h_values.size == 0:
        raise ValueError("fields must not be empty.")
    errors: dict[str, list[float]] = {key: [] for key in ERROR_KEYS}
    final_fidelity: list[float] = []
    wall_time: list[float] = []
    for h in h_values:
        comparison = compare_with_exact(
            n_spins=n_spins,
            h=float(h),
            J=J,
            T=T,
            seed=seed,
            alpha=alpha,
            init_std=init_std,
            settings=settings,
        )
        for key, value in comparison.accumulated_errors().items():
            errors[key].append(value)
        final_fidelity.append(float(comparison.fidelity[-1]))
        wall_time.append(comparison.wall_time)
    return FieldSweep(
        n_spins=n_spins,
        T=float(T),
        fields=h_values,
        errors={key: np.asarray(value) for key, value in errors.items()},
        final_fidelity=np.asarray(final_fidelity),
        wall_time=np.asarray(wall_time),
    )


__all__ = [
    "ERROR_KEYS",
    "ErrorComparison",
    "FieldSweep",
    "Trajectory",
    "accumulated_absolute_error",
    "accumulated_error_vs_field",
    "compare_with_exact",
    "entanglement_entropy",
    "local_pauli",
    "trajectory_observables",
]
