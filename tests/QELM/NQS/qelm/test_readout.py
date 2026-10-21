"""Ridge readout of the QELM (closed form, bias not penalized)."""

import numpy as np
import pytest

from QELM.NQS.qelm import RidgeReadout


def _data(n: int = 30, p: int = 5, outputs: int | None = None, seed: int = 0):
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(n, p))
    shape = (p,) if outputs is None else (p, outputs)
    w = rng.normal(size=shape)
    b = rng.normal(size=() if outputs is None else (outputs,))
    return F, w, b, F @ w + b


@pytest.mark.parametrize("outputs", [None, 3])
def test_recovers_an_exact_linear_map(outputs: int | None) -> None:
    F, w, b, y = _data(outputs=outputs)
    readout = RidgeReadout(lam=0.0).fit(F, y)
    np.testing.assert_allclose(readout.weights, w, atol=1e-10)
    np.testing.assert_allclose(readout.bias, b, atol=1e-10)
    np.testing.assert_allclose(readout.predict(F), y, atol=1e-10)


def test_small_penalty_tends_to_least_squares() -> None:
    F, _, _, y = _data()
    y = y + np.random.default_rng(1).normal(scale=0.1, size=y.shape)
    exact = RidgeReadout(lam=0.0).fit(F, y).weights
    assert exact is not None
    for lam in (1e-4, 1e-8):
        w = RidgeReadout(lam=lam).fit(F, y).weights
        assert np.linalg.norm(w - exact) < 10 * lam * np.linalg.norm(exact) + 1e-12


def test_penalty_shrinks_weights_but_not_the_bias() -> None:
    F, _, _, y = _data()
    norms = [
        np.linalg.norm(RidgeReadout(lam=lam).fit(F, y).weights)
        for lam in (0.0, 1.0, 10.0, 1e3)
    ]
    assert all(a > b for a, b in zip(norms, norms[1:], strict=False))
    # Infinite penalty: zero weights, the bias is the mean target.
    huge = RidgeReadout(lam=1e12).fit(F, y)
    assert np.max(np.abs(huge.weights)) < 1e-8
    assert huge.bias == pytest.approx(y.mean())


def test_constant_feature_shift_only_moves_the_bias() -> None:
    F, _, _, y = _data()
    a = RidgeReadout(lam=0.5).fit(F, y)
    b = RidgeReadout(lam=0.5).fit(F + 3.0, y)
    np.testing.assert_allclose(a.weights, b.weights, atol=1e-12)
    np.testing.assert_allclose(a.predict(F), b.predict(F + 3.0), atol=1e-12)


def test_invalid_use_raises() -> None:
    with pytest.raises(ValueError):
        RidgeReadout(lam=-1.0)
    with pytest.raises(RuntimeError):
        RidgeReadout().predict(np.ones((2, 2)))
    with pytest.raises(ValueError):
        RidgeReadout().fit(np.ones((3, 2)), np.ones(4))
