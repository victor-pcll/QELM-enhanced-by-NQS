"""SVD of the RBM weights (issue #6)."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from QRC.NQS.evaluation.representations import (
    analyze_weights_svd,
    extract_rbm_weights,
)
from QRC.NQS.utils import makeplots
from QRC.NQS.variational import rbm, spin_half_hilbert, variational_state


def _state(n: int = 5, alpha: int = 2, kind: str = "full_sum", seed: int = 0):
    return variational_state(
        spin_half_hilbert(n), rbm(alpha=alpha), seed=seed, kind=kind, n_samples=16
    )


@pytest.mark.parametrize(("n", "alpha"), [(4, 1), (5, 2), (3, 3)])
@pytest.mark.parametrize("kind", ["full_sum", "mc"])
def test_weight_shape_is_visible_by_hidden(n: int, alpha: int, kind: str) -> None:
    W = extract_rbm_weights(_state(n, alpha, kind))
    assert W.shape == (n, alpha * n)


def test_extracted_weights_are_the_model_kernel() -> None:
    state = _state()
    W = extract_rbm_weights(state)
    np.testing.assert_array_equal(W, np.asarray(state.parameters["Dense"]["kernel"]))
    # A copy: editing it does not touch the state.
    W[0, 0] += 1.0
    assert W[0, 0] != state.parameters["Dense"]["kernel"][0, 0]
    np.testing.assert_array_equal(
        extract_rbm_weights(state.parameters), extract_rbm_weights(state)
    )


def test_complex_parameters_give_real_positive_spectrum() -> None:
    W = extract_rbm_weights(_state())
    assert np.iscomplexobj(W)
    assert np.abs(W.imag).max() > 0
    svd = analyze_weights_svd(W)
    assert np.isrealobj(svd.singular_values)
    assert np.all(svd.singular_values > 0)


def test_matrix_reconstruction() -> None:
    W = extract_rbm_weights(_state())
    svd = analyze_weights_svd(W)
    np.testing.assert_allclose(
        W, svd.U @ np.diag(svd.singular_values) @ svd.Vh, atol=1e-7
    )
    np.testing.assert_allclose(W, svd.reconstruct(), atol=1e-14)


def test_singular_vectors_are_orthonormal() -> None:
    svd = analyze_weights_svd(extract_rbm_weights(_state(n=5, alpha=2)))
    k = svd.singular_values.size
    assert svd.U.shape == (5, k) and svd.Vh.shape == (k, 10)
    np.testing.assert_allclose(svd.U.conj().T @ svd.U, np.eye(k), atol=1e-13)
    np.testing.assert_allclose(svd.Vh @ svd.Vh.conj().T, np.eye(k), atol=1e-13)


def test_spectrum_is_sorted_descending() -> None:
    for seed in range(5):
        sigma = analyze_weights_svd(
            extract_rbm_weights(_state(seed=seed))
        ).singular_values
        assert np.all(np.diff(sigma) < 0)


def test_spectrum_is_invariant_under_hidden_unit_relabelling() -> None:
    """Permuting hidden units leaves psi (and hence the spectrum) unchanged."""
    state = _state(n=4, alpha=2)
    psi = np.asarray(state.to_array())
    sigma = analyze_weights_svd(extract_rbm_weights(state)).singular_values

    perm = np.random.default_rng(0).permutation(8)
    params = jax.tree_util.tree_map(lambda x: x, state.parameters)
    params["Dense"] = {
        "kernel": params["Dense"]["kernel"][:, perm],
        "bias": params["Dense"]["bias"][perm],
    }
    state.parameters = params
    np.testing.assert_allclose(np.asarray(state.to_array()), psi, atol=1e-13)
    np.testing.assert_allclose(
        analyze_weights_svd(extract_rbm_weights(state)).singular_values,
        sigma,
        atol=1e-13,
    )


def test_rank_one_weights_have_one_mode() -> None:
    u = np.array([1.0, -1.0, 1.0, -1.0]) / 2  # staggered (Neel) pattern
    v = np.exp(1j * np.arange(8)) / np.sqrt(8)
    W = 3.0 * np.outer(u, v.conj())
    svd = analyze_weights_svd(W)
    assert svd.singular_values[0] == pytest.approx(3.0)
    np.testing.assert_allclose(svd.singular_values[1:], 0.0, atol=1e-14)
    assert abs(np.vdot(svd.U[:, 0], u)) == pytest.approx(1.0)
    assert svd.n_dominant_modes(0.99) == 1
    np.testing.assert_allclose(svd.weight_fractions.sum(), 1.0)


def test_dominant_mode_count() -> None:
    svd = analyze_weights_svd(np.diag([3.0, 2.0, 1.0]))
    # Shares 9/14, 4/14, 1/14.
    assert svd.n_dominant_modes(0.5) == 1
    assert svd.n_dominant_modes(0.9) == 2
    assert svd.n_dominant_modes(1.0) == 3
    assert analyze_weights_svd(np.zeros((2, 2))).n_dominant_modes() == 0
    with pytest.raises(ValueError):
        svd.n_dominant_modes(0.0)


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        analyze_weights_svd(np.ones(3))
    with pytest.raises(KeyError):
        extract_rbm_weights({"visible_bias": jnp.ones(3)})


def test_plot_singular_values(tmp_path: Path) -> None:
    svd = analyze_weights_svd(extract_rbm_weights(_state()))
    path = makeplots.plot_singular_values(svd.singular_values, directory=tmp_path)
    assert path == tmp_path / "rbm_singular_values.png"
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    for bad in ([1.0, 2.0], [1.0, -0.5], [], [1.0 + 1j]):
        with pytest.raises(ValueError):
            makeplots.plot_singular_values(np.asarray(bad), directory=tmp_path)
