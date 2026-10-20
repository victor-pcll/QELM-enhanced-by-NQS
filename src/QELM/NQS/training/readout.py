"""Linear readout scaffold."""

from __future__ import annotations

from QELM.NQS.training.base import Trainer


class Readout(Trainer):
    """Reservoir readout.

    ``fit`` accepts features of shape ``(n_samples, n_features)`` and does
    not estimate a weight vector.
    """

    name = "readout"
