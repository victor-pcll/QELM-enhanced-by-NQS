"""Hamiltonians and time evolution."""

from __future__ import annotations

from QELM.NQS.models.base import HamiltonianModel
from QELM.NQS.models.ising import IsingModel
from QELM.NQS.models.kitaev import KitaevModel

_MODELS: dict[str, type[HamiltonianModel]] = {
    IsingModel.name: IsingModel,
    KitaevModel.name: KitaevModel,
}


def get_model(sim_type: str) -> type[HamiltonianModel]:
    """Return the model class for a ``sim_type`` config value.

    Args:
        sim_type: Scalar name such as ``"ising"`` or ``"kitaev"``.

    Returns:
        The matching :class:`HamiltonianModel` subclass.

    Raises:
        KeyError: If ``sim_type`` is not registered.
    """
    try:
        return _MODELS[sim_type]
    except KeyError as exc:
        known = ", ".join(sorted(_MODELS))
        raise KeyError(f"Unknown sim_type {sim_type!r}. Known: {known}.") from exc


__all__ = ["HamiltonianModel", "IsingModel", "KitaevModel", "get_model"]
