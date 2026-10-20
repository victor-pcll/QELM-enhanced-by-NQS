"""Neural quantum states (NetKet 3) as a probe for QRC and QELM."""

from QRC.NQS.pipeline import resolve_pipeline, run_pipeline
from QRC.NQS.seed import set_seed


def hello() -> str:
    """Return a short package greeting.

    Returns:
        The fixed greeting string.
    """
    return "Hello from QRC.NQS!"


__all__ = ["hello", "resolve_pipeline", "run_pipeline", "set_seed"]
