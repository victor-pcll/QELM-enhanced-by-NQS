"""Real-time dynamics: exact evolution, TDVP and the neural Galerkin method."""

from __future__ import annotations

from QELM.NQS.dynamics.baseline import BASELINE_SEED, BaselineResult, tdvp_vs_exact
from QELM.NQS.dynamics.exact import (
    exact_time_evolution,
    expectation_values,
    time_grid,
)
from QELM.NQS.dynamics.neural_galerkin import (
    NeuralGalerkinResult,
    NeuralGalerkinSettings,
    mean_magnetization_x,
    run_neural_galerkin,
    tfi_square_lattice,
)
from QELM.NQS.dynamics.tdvp import TDVPResult, TDVPSettings, run_tdvp

__all__ = [
    "BASELINE_SEED",
    "BaselineResult",
    "NeuralGalerkinResult",
    "NeuralGalerkinSettings",
    "TDVPResult",
    "TDVPSettings",
    "exact_time_evolution",
    "expectation_values",
    "mean_magnetization_x",
    "run_neural_galerkin",
    "run_tdvp",
    "tdvp_vs_exact",
    "tfi_square_lattice",
    "time_grid",
]
