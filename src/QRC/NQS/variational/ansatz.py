"""Neural-network ansätze and variational states (NetKet 3)."""

from __future__ import annotations

from typing import Literal

import flax.linen as nn
import jax.numpy as jnp
import netket as nk

StateKind = Literal["full_sum", "mc"]

_DTYPES = {"complex": jnp.complex128, "real": jnp.float64}


def rbm(alpha: int = 1, param_dtype: str = "complex") -> nn.Module:
    """Return a Restricted Boltzmann Machine ansatz.

    ``psi(s) = exp(sum_i a_i s_i) * prod_j 2 cosh(b_j + sum_i W_ij s_i)``
    with ``M = alpha * N`` hidden units, so ``N + M + N * M`` parameters.

    Args:
        alpha: Hidden-unit density, at least 1.
        param_dtype: ``"complex"`` (needed for real-time dynamics) or
            ``"real"``.

    Returns:
        A NetKet ``RBM`` Flax module.

    Raises:
        ValueError: If ``alpha < 1`` or ``param_dtype`` is unknown.
    """
    if alpha < 1:
        raise ValueError(f"alpha must be >= 1, got {alpha}.")
    try:
        dtype = _DTYPES[param_dtype]
    except KeyError as exc:
        known = ", ".join(sorted(_DTYPES))
        raise ValueError(
            f"Unknown param_dtype {param_dtype!r}. Known: {known}."
        ) from exc
    return nk.models.RBM(alpha=alpha, param_dtype=dtype)


def variational_state(
    hilbert: nk.hilbert.DiscreteHilbert,
    model: nn.Module,
    *,
    seed: int,
    kind: StateKind = "full_sum",
    n_samples: int = 1024,
) -> nk.vqs.VariationalState:
    """Build a variational state with seeded parameter initialization.

    Args:
        hilbert: Discrete Hilbert space.
        model: Flax module mapping configurations to ``log psi``.
        seed: Seed of the parameter initialization (and of the sampler for
            ``kind="mc"``). The same seed gives the same parameters.
        kind: ``"full_sum"`` sums exactly over all ``hilbert.n_states``
            configurations. ``"mc"`` samples with single-spin-flip
            Metropolis moves, which are ergodic on the whole Hilbert space.
            Exchange moves would conserve the total ``Z`` magnetization and
            leave the sampler stuck in one sector, which is wrong for
            Hamiltonians such as the transverse-field Ising chain.
        n_samples: Number of Monte Carlo samples for ``kind="mc"``.

    Returns:
        A ``FullSumState`` or an ``MCState``.

    Raises:
        ValueError: If ``kind`` is unknown.
    """
    if kind == "full_sum":
        return nk.vqs.FullSumState(hilbert=hilbert, model=model, seed=seed)
    if kind == "mc":
        sampler = nk.sampler.MetropolisLocal(hilbert)
        return nk.vqs.MCState(
            sampler,
            model,
            n_samples=n_samples,
            seed=seed,
            sampler_seed=seed,
        )
    raise ValueError(f"Unknown variational state kind {kind!r}.")


__all__ = ["StateKind", "rbm", "variational_state"]
