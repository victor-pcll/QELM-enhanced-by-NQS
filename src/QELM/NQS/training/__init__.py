"""Training algorithms."""

from __future__ import annotations

from QELM.NQS.training.base import Trainer
from QELM.NQS.training.readout import Readout
from QELM.NQS.training.ridge import RidgeRegression

_TRAINERS: dict[str, type[Trainer]] = {
    Readout.name: Readout,
    RidgeRegression.name: RidgeRegression,
}


def get_trainer(name: str) -> type[Trainer]:
    """Return the trainer class for a ``training.name`` config value.

    Args:
        name: Scalar trainer name such as ``"ridge"`` or ``"readout"``.

    Returns:
        The matching :class:`Trainer` subclass.

    Raises:
        KeyError: If ``name`` is not registered.
    """
    try:
        return _TRAINERS[name]
    except KeyError as exc:
        known = ", ".join(sorted(_TRAINERS))
        raise KeyError(f"Unknown training {name!r}. Known: {known}.") from exc


__all__ = ["Readout", "RidgeRegression", "Trainer", "get_trainer"]
