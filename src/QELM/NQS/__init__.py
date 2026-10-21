"""Neural quantum states (NetKet 3) for a quantum extreme learning machine (QELM)."""

from QELM.NQS.pipeline import resolve_pipeline, run_pipeline
from QELM.NQS.seed import set_seed


def hello() -> str:
    """Return a short package greeting.

    Returns:
        The fixed greeting string.
    """
    return "Hello from QELM.NQS!"


__all__ = ["hello", "resolve_pipeline", "run_pipeline", "set_seed"]
