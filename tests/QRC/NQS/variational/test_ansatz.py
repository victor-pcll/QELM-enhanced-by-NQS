"""Checks of the RBM ansatz and of the variational states."""

from pathlib import Path

import jax
import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from QRC.NQS.variational import (
    build_nqs,
    rbm,
    spin_chain,
    spin_half_hilbert,
    transverse_field_ising,
    variational_state,
)

CONFIG_DIR = Path(__file__).resolve().parents[4] / "conf"


def setup_function() -> None:
    GlobalHydra.instance().clear()


def teardown_function() -> None:
    GlobalHydra.instance().clear()


def _system(n: int = 4, h: float = 1.0, J: float = 1.0):
    graph = spin_chain(n, pbc=True)
    hilbert = spin_half_hilbert(n)
    return hilbert, graph, transverse_field_ising(hilbert, graph, h=h, J=J)


def _flat(state) -> np.ndarray:
    leaves = jax.tree_util.tree_leaves(state.parameters)
    return np.concatenate([np.ravel(np.asarray(x)) for x in leaves])


@pytest.mark.parametrize(("n", "alpha"), [(4, 1), (5, 2)])
def test_rbm_parameter_count(n: int, alpha: int) -> None:
    hilbert, _, _ = _system(n)
    state = variational_state(hilbert, rbm(alpha=alpha), seed=0)
    m = alpha * n
    # visible biases + hidden biases + weights
    assert state.n_parameters == n + m + n * m


def test_rbm_parameters_are_complex_by_default() -> None:
    hilbert, _, _ = _system()
    state = variational_state(hilbert, rbm(), seed=0)
    assert np.iscomplexobj(_flat(state))
    real = variational_state(hilbert, rbm(param_dtype="real"), seed=0)
    assert not np.iscomplexobj(_flat(real))


def test_seed_reproducibility() -> None:
    hilbert, _, _ = _system()
    a = _flat(variational_state(hilbert, rbm(), seed=45))
    b = _flat(variational_state(hilbert, rbm(), seed=45))
    c = _flat(variational_state(hilbert, rbm(), seed=46))
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)


def test_full_sum_state_is_normalized() -> None:
    hilbert, _, _ = _system(5)
    psi = np.asarray(variational_state(hilbert, rbm(), seed=3).to_array())
    assert psi.shape == (2**5,)
    assert np.vdot(psi, psi).real == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_full_sum_energy_is_exact_and_variational(seed: int) -> None:
    hilbert, _, H = _system(4, h=0.8, J=1.0)
    state = variational_state(hilbert, rbm(), seed=seed)
    stats = state.expect(H)
    psi = np.asarray(state.to_array())
    H_dense = np.asarray(H.to_dense())
    exact = np.vdot(psi, H_dense @ psi)
    spectrum = np.linalg.eigvalsh(H_dense)

    # Hermitian operator: real expectation, equal to <psi|H|psi>.
    assert abs(complex(stats.mean).imag) < 1e-12
    assert complex(stats.mean).real == pytest.approx(exact.real, abs=1e-10)
    # Variational principle: E0 <= <H> <= E_max.
    assert spectrum[0] - 1e-12 <= exact.real <= spectrum[-1] + 1e-12


def test_monte_carlo_energy_agrees_with_full_sum() -> None:
    hilbert, _, H = _system(4, h=0.8, J=1.0)
    exact = complex(variational_state(hilbert, rbm(), seed=7).expect(H).mean).real
    mc = variational_state(hilbert, rbm(), seed=7, kind="mc", n_samples=8192)
    stats = mc.expect(H)
    error = float(stats.error_of_mean)
    assert 0.0 < error < 0.1
    # Chains mix: the sampler explores every magnetization sector.
    assert float(stats.R_hat) < 1.1
    assert abs(complex(stats.mean).real - exact) < 5 * error


def test_invalid_ansatz_arguments_raise() -> None:
    hilbert, _, _ = _system()
    with pytest.raises(ValueError):
        rbm(alpha=0)
    with pytest.raises(ValueError):
        rbm(param_dtype="float16")
    with pytest.raises(ValueError):
        variational_state(hilbert, rbm(), seed=0, kind="exact")  # type: ignore[arg-type]


def test_build_nqs_from_config() -> None:
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="config", overrides=["ising.N=6", "seed=45"])
    setup = build_nqs(cfg)
    assert setup.hilbert.size == 6
    assert setup.graph.n_nodes == 6
    assert setup.state.n_parameters == 6 + 6 + 36
    ref = variational_state(setup.hilbert, rbm(), seed=45)
    np.testing.assert_array_equal(_flat(setup.state), _flat(ref))


def test_build_nqs_rejects_unknown_system_and_model() -> None:
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        kitaev = compose(config_name="config", overrides=["sim_type=kitaev"])
        bad_model = compose(config_name="config", overrides=["nqs.model=mlp"])
    with pytest.raises(NotImplementedError):
        build_nqs(kitaev)
    with pytest.raises(ValueError):
        build_nqs(bad_model)
