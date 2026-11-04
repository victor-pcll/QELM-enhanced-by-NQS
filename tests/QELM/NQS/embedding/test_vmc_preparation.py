"""VMC ground-state preparation converges to the exact ground state (issue #3)."""

import json
from pathlib import Path

import jax
import netket as nk
import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from QELM.NQS.embedding import (
    ENERGY_FILENAME,
    HYPERPARAMETERS_FILENAME,
    VMCGroundStatePreparation,
    VMCSettings,
)
from QELM.NQS.paths import ExperimentPaths
from QELM.NQS.variational import (
    rbm,
    spin_chain,
    spin_half_hilbert,
    transverse_field_ising,
    variational_state,
)

CONFIG_DIR = Path(__file__).resolve().parents[4] / "conf"
SEED = 45
N = 4


def _system():
    hilbert = spin_half_hilbert(N)
    H = transverse_field_ising(hilbert, spin_chain(N, pbc=True), h=1.0, J=1.0)
    return hilbert, H


def _flat(params) -> np.ndarray:
    return np.concatenate(
        [np.ravel(np.asarray(x)) for x in jax.tree_util.tree_leaves(params)]
    )


def _exact_energy(state: nk.vqs.MCState, H) -> float:
    """<H> of the state summed exactly over the 2^N configurations."""
    exact = nk.vqs.FullSumState(state.hilbert, state.model, variables=state.variables)
    return float(complex(exact.expect(H).mean).real)


@pytest.fixture(scope="module")
def prepared(tmp_path_factory: pytest.TempPathFactory):
    hilbert, H = _system()
    state = variational_state(hilbert, rbm(), seed=SEED, kind="mc", n_samples=2048)
    initial = _flat(state.parameters)
    paths = ExperimentPaths.at(tmp_path_factory.mktemp("vmc")).create()
    prep = VMCGroundStatePreparation(H, state, settings=VMCSettings(), paths=paths)
    returned = prep.run()
    e0 = float(np.linalg.eigvalsh(np.asarray(H.to_dense()))[0])
    return {"prep": prep, "state": returned, "initial": initial, "H": H, "E0": e0}


def test_energy_converges_to_exact_diagonalization(prepared) -> None:
    prep, H, e0 = prepared["prep"], prepared["H"], prepared["E0"]
    exact_final = _exact_energy(prepared["state"], H)
    assert abs(exact_final - e0) < 1e-2
    # The Monte Carlo estimate agrees too.
    assert abs(prep.energy[-1] - e0) < 1e-2


def test_variational_principle(prepared) -> None:
    # No state lies below the ground state; the MC estimate may, within noise.
    assert _exact_energy(prepared["state"], prepared["H"]) >= prepared["E0"] - 1e-10
    prep = prepared["prep"]
    assert np.all(prep.energy >= prepared["E0"] - 5 * prep.energy_error - 1e-8)


def test_energy_variance_collapses(prepared) -> None:
    # sigma_E^2 = 0 for an eigenstate: it drops by orders of magnitude.
    variance = prepared["prep"].energy_variance
    assert np.all(variance >= 0)
    assert variance[-1] < 1e-3
    assert variance[-1] < 1e-3 * variance[0]


def test_parameters_are_updated(prepared) -> None:
    final = _flat(prepared["state"].parameters)
    assert final.shape == prepared["initial"].shape
    assert not np.allclose(final, prepared["initial"])
    assert prepared["state"] is prepared["prep"].variational_state


def test_outputs_are_routed_to_the_run_folder(prepared) -> None:
    prep = prepared["prep"]
    exp = prep.paths.exp
    hyper = json.loads((exp / HYPERPARAMETERS_FILENAME).read_text())
    assert hyper["learning_rate"] == 1e-2
    assert hyper["n_samples"] == 2048
    assert hyper["alpha"] == 1
    assert hyper["diag_shift"] == 1e-4
    assert hyper["n_parameters"] == N + N + N * N
    log = json.loads(prep.log_path.read_text())
    assert len(log["Energy"]["iters"]) == prep.settings.n_iter
    assert (exp / "vmc.mpack").is_file()
    saved = np.load(exp / ENERGY_FILENAME)
    np.testing.assert_array_equal(saved["acceptance"], prep.acceptance)
    assert prep.acceptance.shape == prep.iterations.shape
    assert np.all((prep.acceptance > 0) & (prep.acceptance <= 1))


def test_default_outputs_go_to_a_run_under_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    hilbert, H = _system()
    state = variational_state(hilbert, rbm(), seed=SEED, kind="mc", n_samples=64)
    prep = VMCGroundStatePreparation(H, state, settings=VMCSettings(n_iter=2))
    prep.run()
    assert prep.paths is not None
    exp = tmp_path / prep.paths.exp
    assert prep.paths.exp.parts[:2] == ("outputs", "default")
    assert (exp / HYPERPARAMETERS_FILENAME).is_file()
    assert (exp / "vmc.log").is_file()


def test_default_config_holds_validated_hyperparameters() -> None:
    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
            cfg = compose(config_name="config", overrides=["vmc.n_iter=3"])
    finally:
        GlobalHydra.instance().clear()
    settings = VMCSettings.from_config(cfg.vmc)
    assert settings == VMCSettings(n_iter=3)
    assert VMCSettings().learning_rate == 1e-2
    assert VMCSettings().n_samples == 2048
    assert VMCSettings().alpha == 1
    assert VMCSettings().diag_shift == 1e-4

    hilbert, H = _system()
    prep = VMCGroundStatePreparation.from_config(H, hilbert, cfg)
    assert isinstance(prep.variational_state, nk.vqs.MCState)
    assert prep.variational_state.n_samples == 2048
    assert prep.metadata["seed"] == int(cfg.seed)


def test_invalid_inputs_raise() -> None:
    hilbert, H = _system()
    exact = variational_state(hilbert, rbm(), seed=SEED)
    with pytest.raises(TypeError):
        VMCGroundStatePreparation(H, exact)
    with pytest.raises(KeyError):
        VMCSettings.from_config({"learning_rat": 0.1})
