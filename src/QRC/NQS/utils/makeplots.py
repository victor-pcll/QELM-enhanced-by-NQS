"""All figures of the project are drawn here.

Each function builds a Matplotlib figure with the object-oriented API (no
global ``pyplot`` state, no display needed), writes it to the ``plots/``
folder of the current run (next to ``logs/``, see :mod:`QRC.NQS.paths`)
and returns the written path.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import ArrayLike

from QRC.NQS.paths import experiment_paths


def plots_dir(directory: Path | str | None = None) -> Path:
    """Return the folder where figures are written, creating it.

    Args:
        directory: Explicit folder. When ``None``, the ``plots/`` folder of
            the active Hydra run (or of a fresh fallback run) is used.

    Returns:
        An existing directory.
    """
    if directory is None:
        return experiment_paths().plots
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(
    fig: Figure,
    filename: str,
    directory: Path | str | None = None,
    *,
    dpi: int = 150,
) -> Path:
    """Write ``fig`` to ``<plots dir>/<filename>``.

    Args:
        fig: Figure to write.
        filename: Bare file name; its suffix chooses the format (``.png``,
            ``.pdf``, ``.svg``).
        directory: Target folder, see :func:`plots_dir`.
        dpi: Resolution for raster formats.

    Returns:
        Path of the written file.

    Raises:
        ValueError: If ``filename`` is empty or contains a folder.
    """
    if filename == "" or Path(filename).name != filename:
        raise ValueError("filename must be a file name, not a path.")
    path = plots_dir(directory) / filename
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def plot_time_series(
    times: ArrayLike,
    series: Mapping[str, ArrayLike],
    *,
    filename: str,
    directory: Path | str | None = None,
    xlabel: str = "t",
    ylabel: str = "",
    title: str = "",
) -> Path:
    """Plot one or more observables against time on shared axes.

    Args:
        times: Times, shape ``(n_times,)``.
        series: Label mapped to real values of shape ``(n_times,)``.
        filename: Output file name.
        directory: Target folder, see :func:`plots_dir`.
        xlabel: Horizontal axis label.
        ylabel: Vertical axis label.
        title: Figure title.

    Returns:
        Path of the written figure.

    Raises:
        ValueError: If ``series`` is empty, a series is not real, or its
            length differs from ``times``.
    """
    t = np.asarray(times, dtype=float)
    if not series:
        raise ValueError("series must contain at least one curve.")
    fig = Figure(figsize=(7, 4.5))
    ax = fig.add_subplot()
    for label, values in series.items():
        y = np.asarray(values)
        if np.iscomplexobj(y):
            raise ValueError(f"Series {label!r} is complex; plot its real part.")
        if y.shape != t.shape:
            raise ValueError(
                f"Series {label!r} has shape {y.shape}, times have {t.shape}."
            )
        ax.plot(t, y, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    return save_figure(fig, filename, directory)


def plot_convergence(
    iterations: ArrayLike,
    values: ArrayLike,
    *,
    filename: str,
    directory: Path | str | None = None,
    ylabel: str = "",
    title: str = "",
    log_y: bool = False,
    reference: float | None = None,
) -> Path:
    """Plot a quantity against optimization iterations.

    Args:
        iterations: Iteration indices, shape ``(n_iter,)``.
        values: Real values, shape ``(n_iter,)``.
        filename: Output file name.
        directory: Target folder, see :func:`plots_dir`.
        ylabel: Vertical axis label.
        title: Figure title.
        log_y: Logarithmic vertical axis; ``values`` must then be > 0.
        reference: Exact value drawn as a horizontal line, if given.

    Returns:
        Path of the written figure.

    Raises:
        ValueError: If shapes differ, or ``log_y`` is set and a value is
            not strictly positive.
    """
    x = np.asarray(iterations, dtype=float)
    y = np.asarray(values, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"iterations have shape {x.shape}, values have {y.shape}.")
    if log_y and np.any(y <= 0):
        raise ValueError("log_y needs strictly positive values.")
    fig = Figure(figsize=(7, 4.5))
    ax = fig.add_subplot()
    ax.plot(x, y, marker=".", label="NQS")
    if reference is not None:
        ax.axhline(reference, color="k", linestyle="--", label="exact")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    return save_figure(fig, filename, directory)


def plot_exact_vs_tdvp(
    times_exact: ArrayLike,
    exact: ArrayLike,
    times_tdvp: ArrayLike,
    tdvp: ArrayLike,
    *,
    filename: str,
    directory: Path | str | None = None,
    ylabel: str = "",
    title: str = "",
) -> Path:
    """Plot an exact and a TDVP observable, with their absolute error below.

    Top panel: exact curve (black line) and TDVP (red dashes). Bottom
    panel: ``|TDVP - exact|`` on a log scale, drawn only when both share
    the same time grid.

    Args:
        times_exact: Times of the exact curve, shape ``(n,)``.
        exact: Exact real values, shape ``(n,)``.
        times_tdvp: Times of the TDVP curve, shape ``(m,)``.
        tdvp: TDVP real values, shape ``(m,)``.
        filename: Output file name.
        directory: Target folder, see :func:`plots_dir`.
        ylabel: Observable label.
        title: Figure title.

    Returns:
        Path of the written figure.

    Raises:
        ValueError: If a curve and its times differ in shape, or a curve
            is complex.
    """
    t_ex = np.asarray(times_exact, dtype=float)
    t_nqs = np.asarray(times_tdvp, dtype=float)
    y_ex = np.asarray(exact)
    y_nqs = np.asarray(tdvp)
    for name, t, y in (("exact", t_ex, y_ex), ("tdvp", t_nqs, y_nqs)):
        if np.iscomplexobj(y):
            raise ValueError(f"{name} is complex; plot its real part.")
        if t.shape != y.shape:
            raise ValueError(f"{name} has shape {y.shape}, its times {t.shape}.")
    same_grid = t_ex.shape == t_nqs.shape and np.allclose(t_ex, t_nqs)

    fig = Figure(figsize=(8, 6 if same_grid else 4.5))
    if same_grid:
        ax, ax_err = fig.subplots(2, 1, sharex=True, height_ratios=[3, 1])
    else:
        ax = fig.add_subplot()
    ax.plot(t_ex, y_ex, "k-", linewidth=2, label="Exact (ED)")
    ax.plot(t_nqs, y_nqs, "r--", label="TDVP (NQS)")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    if same_grid:
        error = np.abs(y_nqs - y_ex)
        ax_err.semilogy(t_ex, np.maximum(error, 1e-16), "b-")
        ax_err.set_ylabel("|TDVP - ED|")
        ax_err.set_xlabel("t")
        ax_err.grid(True, linestyle="--", alpha=0.6)
    else:
        ax.set_xlabel("t")
    return save_figure(fig, filename, directory)


def plot_energy_variance(
    iterations: ArrayLike,
    variance: ArrayLike,
    *,
    filename: str = "energy_variance.png",
    directory: Path | str | None = None,
    floor: float = 1e-16,
    title: str = "",
) -> Path:
    """Plot the energy variance ``sigma_E^2`` against VMC iterations, log scale.

    An exact eigenstate has ``sigma_E^2 = 0``; values are drawn as
    ``max(sigma_E^2, floor)`` so that the log scale can show them.

    Args:
        iterations: Iteration indices, shape ``(n_iter,)``.
        variance: Non-negative variances, shape ``(n_iter,)``.
        filename: Output file name.
        directory: Target folder, see :func:`plots_dir`.
        floor: Smallest value drawn.
        title: Figure title.

    Returns:
        Path of the written figure.

    Raises:
        ValueError: If a variance is negative or shapes differ.
    """
    v = np.asarray(variance, dtype=float)
    if np.any(v < 0):
        raise ValueError("An energy variance cannot be negative.")
    return plot_convergence(
        iterations,
        np.maximum(v, floor),
        filename=filename,
        directory=directory,
        ylabel=r"$\sigma_E^2 = \langle H^2 \rangle - \langle H \rangle^2$",
        title=title,
        log_y=True,
    )


def plot_singular_values(
    singular_values: ArrayLike,
    *,
    filename: str = "rbm_singular_values.png",
    directory: Path | str | None = None,
    title: str = "",
) -> Path:
    """Plot an RBM weight spectrum and its cumulative weight.

    Left: singular values ``sigma_k`` against the mode index on a log scale.
    Right: cumulative share ``sum_{l<=k} sigma_l^2 / sum sigma^2`` of
    ``||W||_F^2``, to read off how many modes dominate.

    Args:
        singular_values: Non-negative values in descending order, shape
            ``(K,)``.
        filename: Output file name.
        directory: Target folder, see :func:`plots_dir`.
        title: Figure title.

    Returns:
        Path of the written figure.

    Raises:
        ValueError: If values are complex, negative, not 1D, empty, or not
            sorted in descending order.
    """
    raw = np.asarray(singular_values)
    if np.iscomplexobj(raw):
        raise ValueError("Singular values are real.")
    sigma = raw.astype(float)
    if sigma.ndim != 1 or sigma.size == 0:
        raise ValueError("Expected a non-empty 1D array of singular values.")
    if np.any(sigma < 0) or np.any(np.diff(sigma) > 0):
        raise ValueError("Singular values must be non-negative and descending.")
    modes = np.arange(1, sigma.size + 1)
    power = sigma**2
    cumulative = np.cumsum(power) / power.sum() if power.sum() > 0 else power

    fig = Figure(figsize=(10, 4))
    ax_s, ax_c = fig.subplots(1, 2)
    ax_s.semilogy(modes, np.maximum(sigma, 1e-16), "o-")
    ax_s.set_xlabel("mode k")
    ax_s.set_ylabel(r"$\sigma_k$")
    ax_s.grid(True, linestyle="--", alpha=0.6)
    ax_c.plot(modes, cumulative, "o-")
    ax_c.axhline(0.9, color="k", linestyle="--", linewidth=1, label="90 %")
    ax_c.set_ylim(0, 1.05)
    ax_c.set_xlabel("mode k")
    ax_c.set_ylabel(r"$\sum_{l \leq k} \sigma_l^2 / \sum \sigma^2$")
    ax_c.grid(True, linestyle="--", alpha=0.6)
    ax_c.legend()
    if title:
        fig.suptitle(title)
    return save_figure(fig, filename, directory)


__all__ = [
    "plot_convergence",
    "plot_energy_variance",
    "plot_exact_vs_tdvp",
    "plot_singular_values",
    "plot_time_series",
    "plots_dir",
    "save_figure",
]
