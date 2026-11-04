"""QELM feature map: short-time reservoir dynamics measured on each spin.

A quantum extreme learning machine (QELM) has no recurrence: each input
``x`` is processed independently. Here ``x`` sets the longitudinal field of
the reservoir Hamiltonian

    H(x) = h(x) sum_i Z_i + sum_{i<j} J_ij X_i X_j,   h(x) = h0 + kappa * x,

with fixed random couplings ``J_ij``. Every input starts from the same
state ``|psi_0>`` (a seeded RBM, or the ground state of a pure field for
the quench of issue #26), is evolved for a short time ``T``, and the
features are the single-spin magnetizations ``<Z_i>(t_k)`` at a few readout
times ``t_k``. Only a linear readout is trained on them (see
:mod:`QELM.NQS.qelm.readout`).

The dynamics is computed exactly (ED) or variationally (TDVP on the RBM).
Short times keep TDVP close to ED: the projection error accumulates with
time, which is why the project moved from reservoir computing (long,
recurrent evolutions) to QELM.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

import netket as nk
import numpy as np
from numpy.typing import ArrayLike

from QELM.NQS.dynamics import (
    TDVPSettings,
    exact_time_evolution,
    expectation_values,
    run_tdvp,
    time_grid,
)
from QELM.NQS.measurements.hamiltonians import (
    Axis,
    all_to_all_xx_hamiltonian,
    coupling_matrix,
)
from QELM.NQS.qelm.initial_states import plus_state_rbm, quench_ground_state
from QELM.NQS.variational import rbm, spin_half_hilbert, variational_state

Method = Literal["exact", "tdvp"]
InitialState = Literal["rbm", "quench"]


def random_couplings(n_spins: int, *, scale: float = 1.0, seed: int = 0) -> np.ndarray:
    """Return symmetric couplings ``J_ij`` drawn uniformly in ``[-scale, scale]``.

    Args:
        n_spins: Number of spins.
        scale: Half-width of the uniform distribution.
        seed: NumPy seed; the same seed gives the same reservoir.

    Returns:
        A symmetric ``(N, N)`` matrix with a zero diagonal.
    """
    rng = np.random.default_rng(seed)
    upper = np.triu(rng.uniform(-scale, scale, size=(n_spins, n_spins)), k=1)
    return coupling_matrix(n_spins, upper + upper.T)


def _tdvp_default() -> TDVPSettings:
    # A small QGT shift: the features must follow ED, not the regularized flow.
    return TDVPSettings(dt=1e-2, integrator="RK45", diag_shift=1e-6)


@dataclass(frozen=True)
class QELMReservoir:
    """A fixed spin reservoir and its readout schedule.

    Attributes:
        couplings: Symmetric ``(N, N)`` XX couplings.
        h0: Field at ``x = 0``.
        kappa: Input scale: ``h(x) = h0 + kappa * x``.
        T: Last readout time. Keep it short: the TDVP error grows with T.
        n_readouts: Number of readout times ``t_k = k T / n_readouts``,
            ``k = 1..n_readouts`` (``t = 0`` carries no information on x).
        alpha: RBM hidden-unit density.
        init_std: RBM initialization width; ``0.1`` is a regular point of
            the manifold (see :func:`QELM.NQS.variational.rbm`).
        seed: Seed of the initial RBM state, shared by all inputs.
        initial: ``"rbm"`` (seeded random RBM, the default) or
            ``"quench"``: the ground state of ``H_0 = -h sum_i A_i``
            (issue #26, see :mod:`QELM.NQS.qelm.initial_states`).
            ``init_std`` is then unused.
        quench_axis: ``A`` of the quench. ``"z"`` (default) starts from
            ``|0>^N`` and runs TDVP in the Hadamard frame, where the state
            is the exact RBM ``|+>^N``; ``"x"`` starts from ``|+>^N`` in
            the lab frame.
        tdvp: TDVP settings. ``dt`` must divide ``T / n_readouts``.
    """

    couplings: np.ndarray
    h0: float = 1.0
    kappa: float = 1.0
    T: float = 1.0
    n_readouts: int = 10
    alpha: int = 1
    init_std: float = 0.1
    seed: int = 45
    initial: InitialState = "rbm"
    quench_axis: Axis = "z"
    tdvp: TDVPSettings = field(default_factory=_tdvp_default)

    def __post_init__(self) -> None:
        J = np.asarray(self.couplings)
        if J.ndim != 2 or J.shape[0] != J.shape[1] or J.shape[0] < 2:
            raise ValueError(f"couplings must be (N, N) with N >= 2, got {J.shape}.")
        if self.initial not in ("rbm", "quench"):
            raise ValueError(
                f"initial must be 'rbm' or 'quench', got {self.initial!r}."
            )
        if self.quench_axis not in ("z", "x"):
            raise ValueError(
                f"quench_axis must be 'z' or 'x', got {self.quench_axis!r}."
            )
        if self.T <= 0 or self.n_readouts < 1:
            raise ValueError("T must be > 0 and n_readouts >= 1.")
        steps = self.T / self.n_readouts / self.tdvp.dt
        if abs(steps - round(steps)) > 1e-9 or round(steps) < 1:
            raise ValueError(
                f"dt={self.tdvp.dt} must divide "
                f"T/n_readouts={self.T / self.n_readouts}."
            )

    @property
    def n_spins(self) -> int:
        """Number of spins ``N``."""
        return int(np.asarray(self.couplings).shape[0])

    @property
    def n_features(self) -> int:
        """Features per input: ``n_readouts * N``."""
        return self.n_readouts * self.n_spins

    @property
    def readout_times(self) -> np.ndarray:
        """Readout times ``t_k``, shape ``(n_readouts,)``."""
        return self.T * np.arange(1, self.n_readouts + 1) / self.n_readouts

    def field(self, x: float) -> float:
        """Return ``h(x) = h0 + kappa * x``."""
        return self.h0 + self.kappa * float(x)

    @property
    def hadamard_frame(self) -> bool:
        """Whether TDVP runs in the Hadamard frame (the Z quench)."""
        return self.initial == "quench" and self.quench_axis == "z"

    def hamiltonian(
        self, x: float, *, frame: Literal["lab", "tdvp"] = "lab"
    ) -> nk.operator.DiscreteOperator:
        """Return ``H(x)`` on the reservoir's Hilbert space.

        Args:
            x: Scalar input.
            frame: ``"lab"`` (the physical ``H(x)``) or ``"tdvp"``, the
                frame of :meth:`initial_state`: ``U H(x) U`` for the Z
                quench, ``H(x)`` otherwise.
        """
        hadamard = frame == "tdvp" and self.hadamard_frame
        return all_to_all_xx_hamiltonian(
            spin_half_hilbert(self.n_spins),
            h=self.field(x),
            J=self.couplings,
            frame="hadamard" if hadamard else "lab",
        )

    def initial_vector(self) -> np.ndarray:
        """Return the normalized lab-frame initial state, the ED starting point."""
        if self.initial == "quench":
            return quench_ground_state(
                spin_half_hilbert(self.n_spins), self.quench_axis
            ).astype(complex)
        psi = np.array(self.initial_state().to_array(), dtype=complex)
        return psi / np.linalg.norm(psi)

    def initial_state(self) -> nk.vqs.FullSumState:
        """Return a fresh copy of the initial RBM state, in the TDVP frame.

        For ``initial="quench"`` it is the exact RBM ``|+>^N``: the lab
        state for the X quench, ``U |0>^N`` for the Z quench.
        """
        hilbert = spin_half_hilbert(self.n_spins)
        if self.initial == "quench":
            return plus_state_rbm(hilbert, alpha=self.alpha, seed=self.seed)
        state = variational_state(
            hilbert,
            rbm(alpha=self.alpha, init_std=self.init_std),
            seed=self.seed,
        )
        assert isinstance(state, nk.vqs.FullSumState)
        return state

    def observables(
        self, *, frame: Literal["lab", "tdvp"] = "lab"
    ) -> dict[str, nk.operator.DiscreteOperator]:
        """Return ``{"Z0": Z_0, ..., "Z{N-1}": Z_{N-1}}``.

        In the TDVP frame of the Z quench, ``Z_i`` becomes ``U Z_i U = X_i``
        (the keys stay ``"Z{i}"``: they name the physical observable).
        """
        hilbert = spin_half_hilbert(self.n_spins)
        hadamard = frame == "tdvp" and self.hadamard_frame
        pauli = nk.operator.spin.sigmax if hadamard else nk.operator.spin.sigmaz
        return {f"Z{i}": pauli(hilbert, i) for i in range(self.n_spins)}


def _readout_indices(grid: np.ndarray, readout_times: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(grid, readout_times - 1e-9)
    if not np.allclose(grid[idx], readout_times, atol=1e-9):
        raise ValueError("Readout times are not on the integration grid.")
    return idx


def reservoir_features(
    reservoir: QELMReservoir, x: float, *, method: Method = "tdvp"
) -> np.ndarray:
    """Return the features ``<Z_i>(t_k)`` of one input.

    Args:
        reservoir: The reservoir.
        x: Scalar input.
        method: ``"exact"`` (ED from the RBM vector) or ``"tdvp"``.

    Returns:
        Real array of shape ``(n_readouts * N,)``, time-major:
        ``[Z_0(t_1), ..., Z_{N-1}(t_1), Z_0(t_2), ...]``. Each entry is in
        ``[-1, 1]``.

    Raises:
        ValueError: If ``method`` is unknown.
    """
    grid = time_grid(reservoir.T, reservoir.tdvp.dt)
    if method == "exact":
        # Always in the lab frame, from the exact initial vector.
        observables = reservoir.observables()
        states = exact_time_evolution(
            reservoir.hamiltonian(x), reservoir.initial_vector(), grid
        )
        idx = _readout_indices(grid, reservoir.readout_times)
        values = [expectation_values(states[idx], op) for op in observables.values()]
    elif method == "tdvp":
        observables = reservoir.observables(frame="tdvp")
        result = run_tdvp(
            reservoir.hamiltonian(x, frame="tdvp"),
            reservoir.initial_state(),
            T=reservoir.T,
            observables=observables,
            settings=reservoir.tdvp,
        )
        idx = _readout_indices(result.times, reservoir.readout_times)
        values = [result.observables[name][idx] for name in observables]
    else:
        raise ValueError(f"Unknown method {method!r}; use 'exact' or 'tdvp'.")
    return np.stack(values, axis=1).reshape(-1)


def feature_matrix(
    reservoir: QELMReservoir,
    inputs: Iterable[float] | ArrayLike,
    *,
    method: Method = "tdvp",
) -> np.ndarray:
    """Return the design matrix of a batch of scalar inputs.

    Args:
        reservoir: The reservoir.
        inputs: Scalar inputs, shape ``(n_samples,)``.
        method: ``"exact"`` or ``"tdvp"``.

    Returns:
        Array of shape ``(n_samples, n_features)``.
    """
    xs = np.asarray(list(np.ravel(np.asarray(inputs, dtype=float))))
    return np.stack([reservoir_features(reservoir, x, method=method) for x in xs])


__all__ = [
    "InitialState",
    "Method",
    "QELMReservoir",
    "feature_matrix",
    "random_couplings",
    "reservoir_features",
]
