"""Plot generation scaffold. Does not write image files."""

from __future__ import annotations


def render_plot(kind: str) -> str:
    """Record a plot request without drawing it.

    Args:
        kind: Scalar plot name from ``cfg.evaluation.plot``, such as
            ``"timeseries"``. No image of shape ``(height, width, channels)``
            is written.

    Returns:
        A status line of the form ``plot={kind} status=scaffold``.
    """
    return f"plot={kind} status=scaffold"
