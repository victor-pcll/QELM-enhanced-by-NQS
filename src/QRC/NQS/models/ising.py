"""Transverse-field Ising chain."""

from __future__ import annotations

import numpy as np

from QRC.NQS.models.base import HamiltonianModel
from QRC.NQS.variational.setup import ising_system


class IsingModel(HamiltonianModel):
    """Transverse-field Ising chain, ``H = -h sum_i X_i + J sum_<ij> Z_i Z_j``.

    ``N``, ``J``, ``h``, ``pbc`` and ``learning_rate`` are scalars from
    ``conf/sim_type/ising.yaml``. The operator is built with NetKet by
    :func:`QRC.NQS.variational.ising_system`.
    """

    name = "ising"

    def hamiltonian(self) -> np.ndarray:
        """Return the dense Hamiltonian, shape ``(2**N, 2**N)``.

        Only practical for small ``N`` (memory grows as ``4**N``).

        Returns:
            The real symmetric Hamiltonian matrix in NetKet's basis order.
        """
        _, _, operator = ising_system(self.params)
        return np.asarray(operator.to_dense())
