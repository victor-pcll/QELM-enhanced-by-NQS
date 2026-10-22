"""Prepare a variational state in the ground state of a Hamiltonian (VMC + SR)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import netket as nk
import numpy as np
import optax
from omegaconf import DictConfig

from QELM.NQS.paths import ExperimentPaths, experiment_paths
from QELM.NQS.utils.makeplots import plot_energy_variance
from QELM.NQS.variational import rbm, variational_state

HYPERPARAMETERS_FILENAME = "hyperparameters.json"
LOG_PREFIX = "vmc"
ENERGY_FILENAME = "vmc_energy.npz"
VARIANCE_PLOT_FILENAME = "vmc_energy_variance.png"


@dataclass(frozen=True)
class VMCSettings:
    """Hyperparameters of the ground-state preparation.

    Defaults are the validated values of ``conf/vmc/default.yaml``.

    Attributes:
        learning_rate: SGD step of the natural-gradient update.
        n_iter: Number of VMC iterations.
        n_samples: Monte Carlo samples per iteration.
        alpha: RBM hidden-unit density.
        param_dtype: ``"complex"`` or ``"real"`` RBM parameters.
        diag_shift: Shift added to the quantum geometric tensor in SR.
    """

    learning_rate: float = 1e-2
    n_iter: int = 300
    n_samples: int = 2048
    alpha: int = 1
    param_dtype: str = "complex"
    diag_shift: float = 1e-4

    @classmethod
    def from_config(cls, block: Mapping[str, Any]) -> VMCSettings:
        """Read settings from a ``vmc`` config block (missing keys keep defaults).

        Args:
            block: Mapping such as ``cfg.vmc``.

        Returns:
            The settings.

        Raises:
            KeyError: If the block has a key that is not a setting.
        """
        known = set(cls.__dataclass_fields__)
        unknown = set(block) - known
        if unknown:
            raise KeyError(f"Unknown VMC settings: {sorted(unknown)}.")
        return cls(**{str(k): v for k, v in block.items()})


class VMCGroundStatePreparation:
    """Optimize ``variational_state`` towards the ground state of ``hamiltonian``.

    Wraps NetKet's ``VMC_SR`` driver (stochastic reconfiguration, i.e.
    imaginary-time evolution projected on the variational manifold). The
    hyperparameters and NetKet's JSON log are written to the ``exp/``
    folder of the run (``outputs/<experiment>/<timestamp>/exp/``):

    - ``hyperparameters.json``: the settings and the system description;
    - ``vmc.log``: energy statistics (mean, error, variance, R̂) per
      iteration;
    - ``vmc_energy.npz``: ``iters``, ``energy``, ``energy_error`` and
      ``energy_variance``, ``acceptance`` arrays (and ``exact_energy``,
      ``exact_variance`` with ``track_exact``, empty otherwise);
    - ``vmc.mpack``: the optimized variables.

    The energy variance per iteration is also plotted on a log scale to
    ``plots/vmc_energy_variance.png``.

    Attributes:
        hamiltonian: Hermitian NetKet operator.
        variational_state: ``MCState`` optimized in place.
        settings: Hyperparameters.
        paths: Run folders the outputs go to.
        iterations: Iteration indices after :meth:`run`.
        energy: Mean energy per iteration after :meth:`run`.
        energy_error: Statistical error of the mean per iteration.
        energy_variance: Sample variance of the local energy per iteration.
        acceptance: Metropolis acceptance rate of the samples drawn at each
            iteration (NetKet resets the counters when the parameters
            change, so it is not a running average).
        exact_energy: Exact ``<H>`` per iteration (``track_exact`` only).
        exact_variance: Exact ``<H^2> - <H>^2`` per iteration
            (``track_exact`` only).
    """

    def __init__(
        self,
        hamiltonian: nk.operator.AbstractOperator,
        variational_state: nk.vqs.MCState,
        optimizer: optax.GradientTransformation | None = None,
        *,
        settings: VMCSettings | None = None,
        paths: ExperimentPaths | None = None,
        metadata: Mapping[str, Any] | None = None,
        track_exact: bool = False,
    ) -> None:
        """Store the NetKet components.

        Args:
            hamiltonian: Hermitian NetKet operator.
            variational_state: Sampled variational state (``MCState``).
            optimizer: Optax optimizer; defaults to
                ``optax.sgd(settings.learning_rate)``.
            settings: Hyperparameters; defaults to :class:`VMCSettings`.
            paths: Run folders; defaults to the active Hydra run (or a new
                fallback run under ``outputs/``).
            metadata: Extra JSON-serializable entries for
                ``hyperparameters.json``, such as the system parameters.
            track_exact: Also record, at each iteration, ``<H>`` and
                ``<H^2> - <H>^2`` summed exactly over the ``2**N``
                configurations (small ``N`` only). The Monte Carlo
                estimates can miss what the sampler never visits: when
                ``|psi|^2`` sits on one configuration they read an exact
                eigenstate (zero variance) while the state is not one.

        Raises:
            TypeError: If ``variational_state`` is not an ``MCState``.
        """
        if not isinstance(variational_state, nk.vqs.MCState):
            raise TypeError("VMC_SR needs a sampled nk.vqs.MCState.")
        self.hamiltonian = hamiltonian
        self.variational_state = variational_state
        self.settings = settings or VMCSettings()
        self.optimizer = optimizer or optax.sgd(self.settings.learning_rate)
        self.paths = paths
        self.metadata = dict(metadata or {})
        self.iterations = np.zeros(0, dtype=int)
        self.energy = np.zeros(0)
        self.energy_error = np.zeros(0)
        self.energy_variance = np.zeros(0)
        self.acceptance = np.zeros(0)
        self.track_exact = track_exact
        self.exact_energy = np.zeros(0)
        self.exact_variance = np.zeros(0)

    @classmethod
    def from_config(
        cls,
        hamiltonian: nk.operator.AbstractOperator,
        hilbert: nk.hilbert.DiscreteHilbert,
        cfg: DictConfig,
        *,
        paths: ExperimentPaths | None = None,
    ) -> VMCGroundStatePreparation:
        """Build an RBM ``MCState`` and the driver from ``cfg.vmc`` and ``cfg.seed``.

        Args:
            hamiltonian: Hermitian NetKet operator on ``hilbert``.
            hilbert: Discrete Hilbert space.
            cfg: Composed Hydra config with a ``vmc`` block and ``seed``.
            paths: Run folders, see :meth:`__init__`.

        Returns:
            The preparation object, not yet run.
        """
        settings = VMCSettings.from_config(cfg.vmc)
        state = variational_state(
            hilbert,
            rbm(alpha=settings.alpha, param_dtype=settings.param_dtype),
            seed=int(cfg.seed),
            kind="mc",
            n_samples=settings.n_samples,
        )
        return cls(
            hamiltonian,
            state,
            settings=settings,
            paths=paths,
            metadata={"seed": int(cfg.seed), "sim_type": str(cfg.sim_type)},
        )

    def hyperparameters(self) -> dict[str, Any]:
        """Return the content of ``hyperparameters.json``."""
        state = self.variational_state
        return {
            **asdict(self.settings),
            "n_parameters": int(state.n_parameters),
            "n_spins": int(state.hilbert.size),
            "hamiltonian": str(self.hamiltonian),
            "model": type(state.model).__name__,
            **self.metadata,
        }

    def run(self, *, show_progress: bool = False) -> nk.vqs.MCState:
        """Run ``settings.n_iter`` VMC_SR iterations and write the outputs.

        Args:
            show_progress: Show NetKet's progress bar.

        Returns:
            The optimized variational state (the same object, updated in
            place).
        """
        paths = self.paths or experiment_paths()
        self.paths = paths
        hyper_path = paths.exp / HYPERPARAMETERS_FILENAME
        hyper_path.write_text(json.dumps(self.hyperparameters(), indent=2))

        driver = nk.driver.VMC_SR(
            self.hamiltonian,
            self.optimizer,
            diag_shift=self.settings.diag_shift,
            variational_state=self.variational_state,
        )
        memory = nk.logging.RuntimeLog()
        json_log = nk.logging.JsonLog(str(paths.exp / LOG_PREFIX))
        acceptance: list[float] = []
        exact: list[tuple[float, float]] = []

        def record(step: int, log_data: dict, driver: Any) -> bool:
            # Called after the samples of this iteration are drawn and before
            # the parameters are updated: same parameters as the logged energy.
            state = driver.state
            acceptance.append(float(state.sampler_state.acceptance))
            if self.track_exact:
                full = nk.vqs.FullSumState(
                    state.hilbert, state.model, variables=state.variables
                )
                stats = full.expect(self.hamiltonian)
                exact.append((float(np.real(stats.mean)), float(stats.variance)))
            return True

        driver.run(
            self.settings.n_iter,
            out=(memory, json_log),
            show_progress=show_progress,
            callback=record,
        )
        json_log.flush(self.variational_state)

        history = memory.data["Energy"]
        self.iterations = np.asarray(history.iters, dtype=int)
        self.energy = np.asarray(history.Mean).real.astype(float)
        self.energy_error = np.asarray(history.Sigma, dtype=float)
        self.energy_variance = np.asarray(history.Variance, dtype=float)
        self.acceptance = np.asarray(acceptance, dtype=float)
        exact_array = np.asarray(exact, dtype=float).reshape(-1, 2)
        self.exact_energy = exact_array[:, 0]
        self.exact_variance = exact_array[:, 1]
        np.savez(
            paths.exp / ENERGY_FILENAME,
            iters=self.iterations,
            energy=self.energy,
            energy_error=self.energy_error,
            energy_variance=self.energy_variance,
            acceptance=self.acceptance,
            exact_energy=self.exact_energy,
            exact_variance=self.exact_variance,
        )
        plot_energy_variance(
            self.iterations,
            self.energy_variance,
            filename=VARIANCE_PLOT_FILENAME,
            directory=paths.plots,
            title="VMC energy variance",
        )
        return self.variational_state

    @property
    def log_path(self) -> Path | None:
        """Path of NetKet's JSON log once :meth:`run` has chosen the folders."""
        return None if self.paths is None else self.paths.exp / f"{LOG_PREFIX}.log"


__all__ = [
    "ENERGY_FILENAME",
    "HYPERPARAMETERS_FILENAME",
    "LOG_PREFIX",
    "VARIANCE_PLOT_FILENAME",
    "VMCGroundStatePreparation",
    "VMCSettings",
]
