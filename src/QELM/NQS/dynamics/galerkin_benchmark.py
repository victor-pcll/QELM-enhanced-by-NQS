"""Benchmark of issue #24: Fig. 2 of Sinibaldi et al., arXiv:2412.11778.

Global quench of the 2D transverse-field Ising model

    H = -J sum_<ij> Z_i Z_j - h sum_i X_i,    J = 1, periodic L x L,

from ``|Psi_0> = |+>^N`` to ``h = 5`` (paramagnetic), ``h = 3.044 ~ h_c``
(critical) and ``h = 2`` (ferromagnetic). Three evolutions are compared on
the same observables, ``<sigma^x>(t) = (1/N) sum_i <X_i>`` and the
normalized loss ``L(t) / N`` of Eq. 2:

- t-NQG (:mod:`QELM.NQS.dynamics.neural_galerkin`);
- the repository's TDVP (:func:`QELM.NQS.dynamics.run_tdvp`) with one
  complex RBM started exactly at ``|+>^N``
  (:func:`QELM.NQS.qelm.plus_state_rbm` parameters). Its loss is the same
  ``L(t)`` evaluated with ``d_t Psi = sum_k theta_dot_k d_k Psi`` and the
  ``theta_dot`` TDVP integrates, so both methods are scored by the same
  quantity;
- exact diagonalization for ``N <= 16`` (``2^N`` amplitudes). The 6x6
  lattice of Fig. 2 (``2^36`` amplitudes) has no exact reference here.

Run from the repository root, for instance::

    uv run python -m QELM.NQS.dynamics.galerkin_benchmark --L 4 --out data/b4
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import netket as nk
import numpy as np
from jax.flatten_util import ravel_pytree
from matplotlib.figure import Figure

from QELM.NQS.dynamics.exact import exact_time_evolution, expectation_values
from QELM.NQS.dynamics.neural_galerkin import (
    NeuralGalerkinSettings,
    mean_magnetization_x,
    run_neural_galerkin,
    spectral_bounds,
    tfi_square_lattice,
)
from QELM.NQS.dynamics.tdvp import TDVPSettings, run_tdvp
from QELM.NQS.qelm.initial_states import PLUS_PAIR_WEIGHT
from QELM.NQS.variational import rbm

FIELDS: tuple[float, ...] = (5.0, 3.044, 2.0)
PHASES = {5.0: "paramagnetic", 3.044: r"critical, $h\approx h_c$", 2.0: "ferromagnetic"}


@dataclass
class Trajectory:
    """``<sigma^x>(t)``, ``<H>(t)`` and ``L(t)`` of one method."""

    method: str
    times: np.ndarray
    x: np.ndarray
    energy: np.ndarray
    loss: np.ndarray | None = None
    wall_time: float = 0.0


# --------------------------------------------------------------------------
# Exact reference
# --------------------------------------------------------------------------


def plus_vector(n_spins: int) -> np.ndarray:
    """Return ``|+>^N`` (all amplitudes ``2^{-N/2}``)."""
    return np.full(2**n_spins, 2 ** (-n_spins / 2), dtype=complex)


def exact_quench(L: int, h: float, times: np.ndarray) -> Trajectory:
    """Exact ``<sigma^x>(t)`` and ``<H>(t)`` from ``|+>^N`` (``N <= 16``)."""
    hilbert, _, H = tfi_square_lattice(L, h=h)
    if hilbert.size > 16:
        raise ValueError(f"ED needs N <= 16, got N = {hilbert.size}.")
    start = time.perf_counter()
    states = exact_time_evolution(H, plus_vector(hilbert.size), times)
    return Trajectory(
        "ED",
        np.asarray(times),
        expectation_values(states, mean_magnetization_x(hilbert)),
        expectation_values(states, H),
        np.zeros(len(times)),
        time.perf_counter() - start,
    )


# --------------------------------------------------------------------------
# TDVP with the same loss
# --------------------------------------------------------------------------


def plus_state_mc(
    hilbert: nk.hilbert.Spin,
    *,
    alpha: int = 1,
    n_samples: int = 2048,
    n_chains: int = 128,
    seed: int = 0,
    pair_weight: float = PLUS_PAIR_WEIGHT,
) -> nk.vqs.MCState:
    """Return a complex RBM ``MCState`` that is exactly ``|+>^N``.

    Same parameters as :func:`QELM.NQS.qelm.plus_state_rbm` (zero biases,
    hidden unit ``j`` tied to spin ``j mod N`` with weight ``w``), sampled
    by single-spin-flip Metropolis.
    """
    sampler = nk.sampler.MetropolisLocal(hilbert, n_chains=n_chains)
    state = nk.vqs.MCState(
        sampler,
        rbm(alpha=alpha),
        n_samples=n_samples,
        seed=seed,
        sampler_seed=seed,
    )
    params = state.parameters
    n_hidden = int(np.asarray(params["Dense"]["bias"]).size)
    kernel = np.zeros((hilbert.size, n_hidden))
    kernel[np.arange(n_hidden) % hilbert.size, np.arange(n_hidden)] = pair_weight
    state.parameters = {
        "Dense": {
            "bias": jnp.zeros_like(params["Dense"]["bias"]),
            "kernel": jnp.asarray(kernel, dtype=params["Dense"]["kernel"].dtype),
        },
        "visible_bias": jnp.zeros_like(params["visible_bias"]),
    }
    return state


def tdvp_residual(
    state: nk.vqs.VariationalState,
    hamiltonian: nk.operator.AbstractOperator,
    *,
    diag_shift: float,
) -> tuple[float, np.ndarray]:
    """Return ``(L, theta_dot)`` of the TDVP flow at the current state.

    ``theta_dot`` solves ``(S + diag_shift) theta_dot = -i F`` (NetKet's
    TDVP with the dense QGT), with ``S`` the QGT and ``F`` the forces.
    Then, with centred log-derivatives ``O_k`` and local energies,

        L = || P_perp (d_t Psi + i H Psi) ||^2 / <Psi|Psi>
          = theta_dot^+ S theta_dot + Var(H) - 2 Im(theta_dot^+ F),

    the loss of Eq. 2 for this trajectory. ``L = Var(H) - F^+ S^-1 F`` when
    ``diag_shift = 0``.
    """
    energy, forces = state.expect_and_forces(hamiltonian)
    F = np.asarray(ravel_pytree(forces)[0])
    S = np.asarray(
        nk.optimizer.qgt.QGTJacobianDense(state, holomorphic=True).to_dense()
    )
    theta_dot = np.linalg.solve(S + diag_shift * np.eye(S.shape[0]), -1j * F)
    quad = np.vdot(theta_dot, S @ theta_dot).real
    cross = -2.0 * np.vdot(theta_dot, F).imag
    return float(quad + float(energy.variance) + cross), theta_dot


def tdvp_quench(
    L: int,
    h: float,
    T: float,
    *,
    settings: TDVPSettings | None = None,
    n_samples: int = 2048,
    alpha: int = 1,
    record_every: int = 1,
    seed: int = 0,
    exact_sum: bool = False,
) -> Trajectory:
    """Run the repository's TDVP from ``|+>^N`` and record ``L(t)`` too.

    Args:
        L: Linear lattice size.
        h: Field of the quench.
        T: Final time, a multiple of ``settings.dt``.
        settings: TDVP settings; default fixed-step RK4, ``dt = 0.005``,
            ``diag_shift = 1e-4``.
        n_samples: Monte Carlo samples per step.
        alpha: RBM hidden-unit density.
        record_every: The loss is evaluated every ``record_every`` grid
            points (it needs a dense ``P x P`` solve).
        seed: Seed of the sampler.
        exact_sum: Use a ``FullSumState`` (small lattices, tests).
    """
    from QELM.NQS.qelm.initial_states import plus_state_rbm

    settings = settings or TDVPSettings(
        dt=0.005, integrator="RK4", diag_shift=1e-4, holomorphic=True
    )
    hilbert, _, H = tfi_square_lattice(L, h=h)
    Xbar = mean_magnetization_x(hilbert)
    if exact_sum:
        state: nk.vqs.VariationalState = plus_state_rbm(hilbert, alpha=alpha)
    else:
        state = plus_state_mc(hilbert, alpha=alpha, n_samples=n_samples, seed=seed)

    start = time.perf_counter()
    times: list[float] = []
    loss: list[float] = []
    step = {"n": 0}

    def callback(_step: Any, _log: Any, driver: Any) -> bool:
        if step["n"] % record_every == 0:
            times.append(float(driver.t))
            loss.append(tdvp_residual(state, H, diag_shift=settings.diag_shift)[0])
        step["n"] += 1
        return True

    result = run_tdvp(
        H,
        state,
        T=T,
        observables={"X": Xbar, "Energy": H},
        settings=settings,
        callback=callback,
    )
    wall = time.perf_counter() - start
    loss_on_grid = np.interp(result.times, np.asarray(times), np.asarray(loss))
    return Trajectory(
        "TDVP (1 RBM)",
        result.times,
        result.observables["X"],
        result.observables["Energy"],
        loss_on_grid,
        wall,
    )


# --------------------------------------------------------------------------
# t-NQG
# --------------------------------------------------------------------------


def galerkin_quench(
    L: int,
    h: float,
    T: float,
    *,
    settings: NeuralGalerkinSettings | None = None,
    log: Any = print,
    extrapolate: float = 0.0,
) -> Trajectory:
    """Run t-NQG from ``|+>^N``; the spectrum bounds use the bipartite symmetry.

    ``extrapolate > 0`` evaluates the last window beyond ``T`` (Fig. 4).
    """
    settings = settings or NeuralGalerkinSettings()
    hilbert, _, H = tfi_square_lattice(L, h=h)
    if settings.spectrum is None:
        lo, hi = spectral_bounds(H, symmetric=True, seed=settings.seed)
        settings = NeuralGalerkinSettings(**{**asdict(settings), "spectrum": (lo, hi)})
    start = time.perf_counter()
    res = run_neural_galerkin(
        H,
        T=T,
        observables={"X": mean_magnetization_x(hilbert)},
        settings=settings,
        log=log,
        extrapolate=extrapolate,
    )
    return Trajectory(
        f"t-NQG (M={settings.n_basis})",
        res.times,
        res.observables["X"],
        res.energy,
        res.loss,
        time.perf_counter() - start,
    )


# --------------------------------------------------------------------------
# I/O and figure
# --------------------------------------------------------------------------


def save_trajectories(path: Path, trajectories: dict[str, Trajectory]) -> None:
    """Write trajectories to ``path`` (``.npz``) with a JSON index."""
    arrays: dict[str, np.ndarray] = {}
    index: dict[str, Any] = {}
    for key, tr in trajectories.items():
        index[key] = {"method": tr.method, "wall_time": tr.wall_time}
        arrays[f"{key}/times"] = tr.times
        arrays[f"{key}/x"] = tr.x
        arrays[f"{key}/energy"] = tr.energy
        if tr.loss is not None:
            arrays[f"{key}/loss"] = tr.loss
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)  # type: ignore[arg-type]
    path.with_suffix(".json").write_text(json.dumps(index, indent=2))


def load_trajectories(path: Path) -> dict[str, Trajectory]:
    """Inverse of :func:`save_trajectories`."""
    data = np.load(path)
    index = json.loads(path.with_suffix(".json").read_text())
    return {
        key: Trajectory(
            meta["method"],
            data[f"{key}/times"],
            data[f"{key}/x"],
            data[f"{key}/energy"],
            data[f"{key}/loss"] if f"{key}/loss" in data else None,
            meta["wall_time"],
        )
        for key, meta in index.items()
    }


STYLES = {
    "ED": {"color": "black", "lw": 2.2, "ls": "-"},
    "tNQG": {"color": "#D55E00", "lw": 1.8, "ls": "--"},
    "TDVP": {"color": "#0072B2", "lw": 1.6, "ls": ":"},
}


def plot_figure2(
    results: dict[float, dict[str, Trajectory]],
    n_spins: int,
    path: Path,
    *,
    title: str = "",
    train_until: float | None = None,
) -> Path:
    """Draw Fig. 2 (one lattice): ``<sigma^x>(t)`` per field, ``L(t)/N`` inset.

    ``train_until`` shades the training interval ``[0, T]`` in light green
    (Fig. 4, extrapolation beyond ``T``).
    """
    fields = [h for h in FIELDS if h in results]
    fig = Figure(figsize=(4.6 * len(fields), 4.2))
    axes = np.atleast_1d(fig.subplots(1, len(fields)))
    for k, (ax, h) in enumerate(zip(axes, fields, strict=True)):
        if train_until is not None:
            ax.axvspan(0.0, train_until, color="#c8eac8", alpha=0.6, lw=0)
        inset = ax.inset_axes((0.5, 0.52, 0.46, 0.42))
        for key, tr in results[h].items():
            style = STYLES.get(key, {})
            ax.plot(tr.times, tr.x, label=tr.method, **style)
            if tr.loss is not None and key != "ED":
                inset.semilogy(tr.times, np.maximum(tr.loss / n_spins, 1e-12), **style)
        ax.set_xlabel("t")
        if k == 0:
            ax.set_ylabel(r"$\langle \sigma^x \rangle$")
        ax.set_title(f"(a{k + 1}) h = {h:g} ({PHASES.get(h, '')})", fontsize=11)
        inset.set_title(r"$\mathcal{L}(t)/N$", fontsize=9)
        inset.tick_params(labelsize=8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].legend(loc="lower left", frameon=False, fontsize=9)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def plot_errors(
    results: dict[float, dict[str, Trajectory]], path: Path, n_spins: int
) -> Path:
    """Plot ``|<sigma^x> - ED|`` and ``|<H>(t) - <H>(0)| / N`` per field."""
    fields = [h for h in FIELDS if h in results]
    fig = Figure(figsize=(4.6 * len(fields), 6.4))
    axes = fig.subplots(2, len(fields), squeeze=False)
    for k, h in enumerate(fields):
        ref = results[h]["ED"]
        for key, tr in results[h].items():
            if key == "ED":
                continue
            style = STYLES.get(key, {})
            ex = np.interp(tr.times, ref.times, ref.x)
            axes[0, k].semilogy(
                tr.times, np.maximum(np.abs(tr.x - ex), 1e-12), label=tr.method, **style
            )
            axes[1, k].semilogy(
                tr.times,
                np.maximum(np.abs(tr.energy - tr.energy[0]) / n_spins, 1e-12),
                **style,
            )
        axes[0, k].set_title(f"h = {h:g}", fontsize=11)
        axes[1, k].set_xlabel("t")
    axes[0, 0].set_ylabel(r"$|\langle\sigma^x\rangle - \mathrm{ED}|$")
    axes[1, 0].set_ylabel(r"$|E(t) - E(0)| / N$")
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> None:
    """Run the benchmark for one lattice and write ``<out>/h=<h>.npz``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--L", type=int, default=6)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--fields", type=float, nargs="+", default=list(FIELDS))
    parser.add_argument("--methods", nargs="+", default=["tNQG", "TDVP", "ED"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-basis", type=int, default=18)
    parser.add_argument("--n-frequencies", type=int, default=128)
    parser.add_argument("--n-samples", type=int, default=1024)
    parser.add_argument("--n-chains", type=int, default=128)
    parser.add_argument("--n-iter", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--tdvp-samples", type=int, default=2048)
    parser.add_argument("--tdvp-dt", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--window", type=float, default=0.2)
    parser.add_argument("--extrapolate", type=float, default=0.0)
    args = parser.parse_args(argv)

    settings = NeuralGalerkinSettings(
        n_basis=args.n_basis,
        n_frequencies=args.n_frequencies,
        n_samples=args.n_samples,
        n_chains=args.n_chains,
        n_iter=args.n_iter,
        learning_rate=args.learning_rate,
        n_eval_samples=8 * args.n_samples,
        window=args.window,
        seed=args.seed,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "settings.json").write_text(
        json.dumps(
            {
                "cli": {k: str(v) for k, v in vars(args).items()},
                "t-NQG": asdict(settings),
            },
            indent=2,
            default=str,
        )
    )
    n_spins = args.L**2
    for h in args.fields:
        path = args.out / f"h={h:g}.npz"
        done = load_trajectories(path) if path.exists() else {}
        if "tNQG" in args.methods and "tNQG" not in done:
            print(f"[L={args.L} h={h}] t-NQG", flush=True)
            done["tNQG"] = galerkin_quench(
                args.L,
                h,
                args.T,
                settings=settings,
                log=lambda s: print("   ", s, flush=True),
                extrapolate=args.extrapolate,
            )
            save_trajectories(path, done)
        if "TDVP" in args.methods and "TDVP" not in done:
            print(f"[L={args.L} h={h}] TDVP", flush=True)
            done["TDVP"] = tdvp_quench(
                args.L,
                h,
                args.T + args.extrapolate,
                settings=TDVPSettings(
                    dt=args.tdvp_dt, integrator="RK4", diag_shift=1e-4
                ),
                n_samples=args.tdvp_samples,
                record_every=max(1, round(0.02 / args.tdvp_dt)),
                seed=args.seed,
            )
            save_trajectories(path, done)
        if "ED" in args.methods and "ED" not in done and n_spins <= 16:
            end = args.T + args.extrapolate
            grid = done["tNQG"].times if "tNQG" in done else np.linspace(0, end, 51)
            done["ED"] = exact_quench(args.L, h, np.unique(grid))
            save_trajectories(path, done)
        for key, tr in done.items():
            drift = np.max(np.abs(tr.energy - tr.energy[0])) / n_spins
            print(f"    {key:5s} wall={tr.wall_time:8.1f}s  max|dE|/N={drift:.2e}")


if __name__ == "__main__":
    main()
