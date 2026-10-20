"""Metrics and plot generation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from omegaconf import DictConfig

from QRC.NQS.evaluation.energy import compute_energy_variance, exact_energy_variance
from QRC.NQS.evaluation.metrics import metric_status
from QRC.NQS.evaluation.plots import render_plot

Evaluator = Callable[..., str]


def evaluate(
    *,
    evolution: Mapping[str, Any],
    training_status: str,
    cfg: DictConfig,
) -> str:
    """Summarize a scaffold evaluation without scoring predictions.

    Args:
        evolution: Evolution record. ``dt`` and ``n_steps`` are scalars.
            No trajectory of shape ``(n_steps, dim)`` is present.
        training_status: Status string returned by the trainer.
        cfg: Composed Hydra config. ``cfg.evaluation.metrics`` is a list of
            scalar names and ``cfg.evaluation.plot`` is a scalar string.

    Returns:
        One status line covering metrics, the plot request, fit status,
        and evolution status.
    """
    names = [str(name) for name in cfg.evaluation.metrics]
    metrics = metric_status(names)
    plot = render_plot(str(cfg.evaluation.plot))
    return (
        f"evaluation={cfg.evaluation.name} {metrics} {plot} "
        f"fit={training_status} evolved={evolution['status']}"
    )


_EVALUATORS: dict[str, Evaluator] = {"default": evaluate}


def get_evaluator(name: str) -> Evaluator:
    """Return the evaluator for an ``evaluation.name`` config value.

    Args:
        name: Scalar evaluator name. The built-in value is ``"default"``.

    Returns:
        A callable that summarizes one run as a status string.

    Raises:
        KeyError: If ``name`` is not registered.
    """
    try:
        return _EVALUATORS[name]
    except KeyError as exc:
        known = ", ".join(sorted(_EVALUATORS))
        raise KeyError(f"Unknown evaluation {name!r}. Known: {known}.") from exc


__all__ = [
    "compute_energy_variance",
    "evaluate",
    "exact_energy_variance",
    "get_evaluator",
    "metric_status",
    "render_plot",
]
