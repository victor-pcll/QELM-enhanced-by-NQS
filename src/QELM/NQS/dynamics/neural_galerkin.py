"""Time-dependent Neural Galerkin method (t-NQG) for real-time quantum dynamics.

Implements the method of A. Sinibaldi, D. Hendry, F. Vicentini and
G. Carleo, *Time-dependent Neural Galerkin Method for Quantum Dynamics*,
arXiv:2412.11778 (issue #24). The state on a time window ``[t_w, t_w + dT]``
is a linear combination of fixed neural states with time-dependent
coefficients (Eq. 5),

    |Psi(t)> = |A_w> + sum_{i=1}^M c_i(t - t_w) |phi_i>,
    c_i(tau) = sum_{k=1}^{N_b} gamma_ik (exp(i omega_k tau) - 1)      (Eq. 8)

where ``|A_0> = |Psi_0> = |+>^N`` and ``|A_{w+1}> = |Psi(t_w + dT)>`` is the
(frozen) state reached at the end of the previous window, ``|phi_i>`` are
complex RBMs and ``gamma``, ``omega`` and the RBM parameters are all
variational. ``c_i(0) = 0`` makes the initial condition exact.

All parameters of a window are found at once by minimizing the
time-integrated residual of the Schrodinger equation (Eqs. 2-3),

    L(t) = || P_perp (d_t Psi + i H Psi) ||^2 / <Psi|Psi>,
    L_[t_w, t_w+dT] = (1/dT) int dt L(t)      (Simpson's 1/3 rule),

with ``P_perp = 1 - |Psi><Psi| / <Psi|Psi>``. ``P_perp`` removes any change
of norm or global phase, so ``L`` is invariant under ``H -> H + const`` and
under ``Psi(t) -> z(t) Psi(t)``: it measures only the physical error.

**Estimator.** Because the ansatz is linear in the coefficients, every term
of ``L(t)`` is a quadratic form of ``c(t)`` and ``dc/dt`` with three
time-independent ``(M+1) x (M+1)`` matrices of the basis
``u = (A, phi_1, ..., phi_M)``:

    S_ij = <u_i|u_j>,   K_ij = <u_i|H|u_j>,   Q_ij = <u_i|H^2|u_j>,

    n = c^+ S c,
    r = dc^+ S dc + c^+ Q c - 2 Im(dc^+ K c)      (= ||dPsi + i H Psi||^2)
    p = c^+ S dc + i c^+ K c                      (= <Psi|dPsi + i H Psi>)
    L = r / n - |p|^2 / n^2.

This is an exact rewriting of Eq. 2, not an approximation. The matrices
are estimated by importance sampling from the time-independent mixture
``Pi(s) = sum_i |u_i(s)|^2`` (the distribution of Appendix C of the paper):
``S_ij ~ E_Pi[u_i^* u_j / Pi]`` and so on, with ``(H u_i)(s)`` from the
connected elements of ``H``. The paper instead samples ``|Psi(s, t)|^2`` at
each of the 257 integration points; both estimate the same ``L(t)``, but
here one set of samples serves every integration point, which divides the
cost by the number of points. ``L`` is homogeneous of degree zero in
``(S, K, Q)``, so the unknown normalization of ``Pi`` cancels. Gradients
are taken through the estimator with ``Pi`` held fixed, which gives an
unbiased estimate of the gradient of each matrix element.

Observables follow the same route: ``<O>(t) = c^+ O c / c^+ S c`` with
``O_ij = <u_i|O|u_j>``.

Conventions: Pauli matrices, NetKet spins ``s_i = +-1`` (eigenvalues of
``sigma^z``). The Hamiltonian is any NetKet operator with a JAX version
(``to_jax_operator``), for instance :func:`tfi_square_lattice`.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import netket as nk
import numpy as np
import optax

Array = Any
Params = dict[str, Any]


# --------------------------------------------------------------------------
# Model: the 2D transverse-field Ising model of the paper
# --------------------------------------------------------------------------


def tfi_square_lattice(
    L: int, *, h: float, J: float = 1.0, pbc: bool = True
) -> tuple[nk.hilbert.Spin, nk.graph.Lattice, nk.operator.DiscreteOperator]:
    """Return the ``L x L`` transverse-field Ising model of the paper (Eq. 7).

        H = -J sum_<ij> Z_i Z_j - h sum_i X_i

    with Pauli matrices and nearest neighbours ``<ij>`` of the square
    lattice, periodic by default as in the paper. ``J > 0`` is
    ferromagnetic. NetKet's ``Ising`` writes ``+J_nk sum Z Z``, so it is
    built with ``J_nk = -J``.

    Args:
        L: Linear size, at least 3 (``L = 2`` with periodic boundaries
            would count each bond twice).
        h: Transverse field.
        J: Ising coupling.
        pbc: Periodic boundary conditions.

    Returns:
        ``(hilbert, graph, H)``.

    Raises:
        ValueError: If ``L < 3``.
    """
    if L < 3:
        raise ValueError(f"Need L >= 3, got {L}.")
    graph = nk.graph.Square(L, pbc=pbc)
    hilbert = nk.hilbert.Spin(s=0.5, N=graph.n_nodes)
    H = nk.operator.Ising(hilbert=hilbert, graph=graph, h=float(h), J=-float(J))
    return hilbert, graph, H


def mean_magnetization_x(hilbert: nk.hilbert.Spin) -> nk.operator.LocalOperator:
    """Return ``(1/N) sum_i X_i``, the transverse magnetization of Fig. 2."""
    total = nk.operator.spin.sigmax(hilbert, 0)
    for i in range(1, hilbert.size):
        total = total + nk.operator.spin.sigmax(hilbert, i)
    return total / hilbert.size


# --------------------------------------------------------------------------
# Settings and results
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class NeuralGalerkinSettings:
    """Hyperparameters of a t-NQG run.

    Defaults are those of the 6x6 panels of Fig. 2 where the paper gives
    them (``M``, ``N_b``, ``dT``, Simpson points). The paper does not give
    the RBM density, the initialization, the learning rate or the number
    of iterations; the values here are ours and are reported with every
    result.

    Attributes:
        n_basis: Number ``M`` of RBM basis states (18 for 6x6 in Fig. 2).
        n_frequencies: Number ``N_b`` of Fourier modes (128 in 2D).
        alpha: Hidden-unit density of each RBM.
        window: Window length ``dT`` (0.2 in 2D).
        n_quadrature: Simpson points per window, odd (256 + 1 in 2D).
        n_samples: Monte Carlo samples of ``Pi`` per optimization step,
            shared by all integration points.
        n_chains: Parallel Metropolis chains.
        n_sweeps: Sweeps (``N`` single-spin-flip proposals each) between
            two optimization steps; chains persist across steps.
        n_burn_in: Sweeps discarded when a window starts.
        n_iter: Adam steps per window.
        learning_rate: Adam step (peak of a cosine decay).
        init_std: Width of the random part of the RBM parameters.
        gamma_init_std: Width of the initial ``gamma``.
        spectrum: ``(E_min, E_max)`` used to place the initial ``omega_k``
            evenly over the spectrum; ``None`` estimates it (see
            :func:`spectral_bounds`).
        symmetric_spectrum: Take ``E_max = -E_min`` when estimating the
            spectrum (true for the TFI model on a bipartite lattice).
        n_eval_samples: Samples used to measure observables at the end of
            each window (``0``: exact sums, only for small systems).
        n_eval_times: Measurement times per window (both ends included).
        eval_chunk: Samples per chunk when measuring (bounds the memory).
        exact: Sum exactly over the Hilbert space instead of sampling
            (small systems and tests only).
        seed: Seed of all random numbers.
    """

    n_basis: int = 18
    n_frequencies: int = 128
    alpha: int = 1
    window: float = 0.2
    n_quadrature: int = 257
    n_samples: int = 4096
    n_chains: int = 256
    n_sweeps: int = 1
    n_burn_in: int = 20
    n_iter: int = 400
    learning_rate: float = 3e-3
    init_std: float = 0.05
    gamma_init_std: float = 1e-3
    spectrum: tuple[float, float] | None = None
    symmetric_spectrum: bool = True
    n_eval_samples: int = 16384
    n_eval_times: int = 21
    eval_chunk: int = 512
    exact: bool = False
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_basis < 1 or self.n_frequencies < 1 or self.alpha < 1:
            raise ValueError("n_basis, n_frequencies and alpha must be >= 1.")
        if self.window <= 0:
            raise ValueError(f"window must be > 0, got {self.window}.")
        if self.n_quadrature < 3 or self.n_quadrature % 2 == 0:
            raise ValueError("Simpson's rule needs an odd n_quadrature >= 3.")
        if not self.exact and self.n_samples % self.n_chains != 0:
            raise ValueError("n_samples must be a multiple of n_chains.")
        if not self.exact and self.n_eval_samples % self.n_chains != 0:
            raise ValueError("n_eval_samples must be a multiple of n_chains.")
        if self.n_eval_times < 2:
            raise ValueError("n_eval_times must be >= 2.")


@dataclass
class NeuralGalerkinResult:
    """Trajectory and diagnostics of a t-NQG run.

    Attributes:
        times: Measurement times, shape ``(n_t,)`` (window ends repeated
            once, so each window is a contiguous block).
        observables: Name mapped to ``<O>(t)``, shape ``(n_t,)``.
        energy: ``<H>(t)``; exact dynamics conserves it.
        loss: ``L(t)`` of Eq. 2 at each time (divide by ``N`` for Fig. 2).
        window_loss: Integrated loss ``L_[t_w, t_w+dT]`` per window, from
            the evaluation samples.
        loss_history: Training loss per Adam step, one array per window.
        wall_time: Seconds per window.
        settings: Settings of the run.
    """

    times: np.ndarray
    observables: dict[str, np.ndarray]
    energy: np.ndarray
    loss: np.ndarray
    window_loss: np.ndarray
    loss_history: list[np.ndarray] = field(default_factory=list)
    wall_time: list[float] = field(default_factory=list)
    settings: NeuralGalerkinSettings | None = None


# --------------------------------------------------------------------------
# Basis states: a batch of complex RBMs
# --------------------------------------------------------------------------


def _complex(x: Mapping[str, Array]) -> Array:
    return x["re"] + 1j * x["im"]


def init_rbm_batch(
    key: Array, n_basis: int, n_spins: int, alpha: int, std: float
) -> Params:
    """Return ``n_basis`` random complex RBMs, stored as real/imaginary parts.

    ``log phi_m(s) = sum_i a_mi s_i + sum_j log cosh(b_mj + sum_i W_mij s_i)``.
    """
    n_hidden = alpha * n_spins
    shapes = {
        "a": (n_basis, n_spins),
        "b": (n_basis, n_hidden),
        "W": (n_basis, n_spins, n_hidden),
    }
    params: Params = {}
    for name, shape in shapes.items():
        key, k_re, k_im = jax.random.split(key, 3)
        params[name] = {
            "re": std * jax.random.normal(k_re, shape),
            "im": std * jax.random.normal(k_im, shape),
        }
    return params


def log_rbm_batch(params: Params, s: Array) -> Array:
    """Return ``log phi_m(s)`` for every basis state, shape ``(..., M)``."""
    a, b, W = (_complex(params[k]) for k in ("a", "b", "W"))
    s = s.astype(jnp.float64)
    theta = jnp.einsum("...n,mnh->...mh", s, W) + b
    return jnp.einsum("...n,mn->...m", s, a) + jnp.sum(
        nk.nn.activation.log_cosh(theta), -1
    )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FrozenBasis:
    """Basis states and coefficients of the windows already solved.

    ``A(s) = 1 + sum_k C_k phi_k(s)``: ``1`` is ``|+>^N`` (unnormalized) and
    ``phi_k`` the RBMs of all previous windows with ``C_k = c_k(dT)``.
    """

    rbm: Params | None = None
    coefficients: Array | None = None

    def extend(self, rbm: Params, coefficients: Array) -> FrozenBasis:
        """Append one window's RBMs with their final coefficients."""
        if self.rbm is None or self.coefficients is None:
            return FrozenBasis(rbm, jnp.asarray(coefficients))
        merged = jax.tree_util.tree_map(
            lambda x, y: jnp.concatenate([x, y], axis=0), self.rbm, rbm
        )
        return FrozenBasis(
            merged, jnp.concatenate([self.coefficients, jnp.asarray(coefficients)])
        )

    @property
    def size(self) -> int:
        return 0 if self.coefficients is None else int(self.coefficients.shape[0])


def _basis_logs(frozen: FrozenBasis, rbm: Params, s: Array) -> tuple[Array, Array]:
    """Return ``(log A-terms, log phi)`` at ``s``.

    The first array has shape ``(..., 1 + K)``: ``0`` for ``|+>^N`` and the
    frozen RBMs shifted by ``log C_k``; ``A(s)`` is the sum of their
    exponentials.
    """
    zero = jnp.zeros(s.shape[:-1] + (1,), dtype=jnp.complex128)
    if frozen.rbm is None or frozen.coefficients is None:
        base = zero
    else:
        logC = jnp.log(frozen.coefficients.astype(jnp.complex128))
        base = jnp.concatenate([zero, log_rbm_batch(frozen.rbm, s) + logC], axis=-1)
    return base, log_rbm_batch(rbm, s)


def _log_pi(
    frozen: FrozenBasis, rbm: Params, s: Array, log_lam: Array | None = None
) -> Array:
    """Return ``log Pi(s)``, ``Pi = sum_i lam_i |u_i(s)|^2``, shape ``(...)``.

    ``u = (A, phi_1, ..., phi_M)``. ``lam_i = 1`` by default; the driver
    uses ``lam_i ~ 1 / <u_i|u_i>`` so that each basis state carries the
    same share of the samples (a balanced mixture): with ``lam = 1`` a
    basis state of large norm would leave ``A`` (hence ``Psi`` near the
    start of the window) almost unsampled. Any fixed ``lam > 0`` gives an
    unbiased estimator.
    """
    base, cur = _basis_logs(frozen, rbm, s)
    if log_lam is None:
        log_lam = jnp.zeros(cur.shape[-1] + 1)
    shift = jnp.maximum(
        jnp.max(base.real, axis=-1) + 0.5 * log_lam[0],
        jnp.max(cur.real + 0.5 * log_lam[1:], axis=-1),
    )
    A = jnp.sum(jnp.exp(base - shift[..., None]), axis=-1)
    rest = jnp.sum(jnp.exp(2.0 * (cur.real - shift[..., None]) + log_lam[1:]), axis=-1)
    return 2.0 * shift + jnp.log(jnp.exp(log_lam[0]) * jnp.abs(A) ** 2 + rest)


def _amplitudes(frozen: FrozenBasis, rbm: Params, s: Array, ref: Array) -> Array:
    """Return ``u(s) e^{-ref}`` with ``u = (A, phi_1, ..., phi_M)``, ``(..., M+1)``."""
    base, cur = _basis_logs(frozen, rbm, s)
    A = jnp.sum(jnp.exp(base - ref[..., None]), axis=-1, keepdims=True)
    return jnp.concatenate([A, jnp.exp(cur - ref[..., None])], axis=-1)


# --------------------------------------------------------------------------
# Matrix elements S, K, Q (and observables) from samples
# --------------------------------------------------------------------------


def basis_matrices(
    frozen: FrozenBasis,
    rbm: Params,
    samples: Array,
    weights: Array,
    hamiltonian: Any,
    observables: Mapping[str, Any] | None = None,
    log_lam: Array | None = None,
) -> dict[str, Array]:
    """Estimate ``S``, ``K = <u|H|u>``, ``Q = <u|H^2|u>`` and ``<u|O|u>``.

    With ``samples`` drawn from ``Pi`` and uniform ``weights`` this is the
    Monte Carlo estimator; with all basis states and ``weights = Pi /
    sum Pi`` it is the exact sum (up to the common factor ``sum Pi``,
    which cancels in every ratio). ``Pi`` is held fixed under ``jax.grad``.

    Args:
        frozen: Frozen part of the basis.
        rbm: Current RBMs.
        samples: Configurations, shape ``(n, N)``.
        weights: Non-negative weights summing to one, shape ``(n,)``.
        hamiltonian: JAX NetKet operator (``get_conn_padded``).
        observables: Name mapped to JAX NetKet operators.
        log_lam: Mixture weights of ``Pi`` (see :func:`_log_pi`); must be
            those the samples were drawn with.

    Returns:
        ``{"S", "K", "K_raw", "Q", <names>}``, each ``(M+1, M+1)``. ``K``
        and the observables are Hermitized (for expectation values).
        ``K_raw = sum_s w_s u^*(s) (H u)(s)`` is not: the loss must use it,
        because ``(S, K_raw, Q)`` are then the blocks of the Gram matrix of
        the vectors ``(u, H u)`` in the sample inner product, which is
        positive semi-definite, so the estimated ``L`` is ``>= 0`` sample by
        sample (it is the squared norm of a projected vector). With the
        Hermitized ``K`` the noisy estimate can become negative, and the
        optimizer then exploits the noise.
    """
    ref = jax.lax.stop_gradient(0.5 * _log_pi(frozen, rbm, samples, log_lam))
    u = _amplitudes(frozen, rbm, samples, ref)

    def apply(op: Any) -> Array:
        conn, mels = op.get_conn_padded(samples)
        u_conn = _amplitudes(frozen, rbm, conn, ref[:, None])
        return jnp.einsum("sc,scm->sm", mels, u_conn)

    def gram(x: Array, y: Array) -> Array:
        return jnp.einsum("s,si,sj->ij", weights, x.conj(), y)

    def herm(x: Array) -> Array:
        return 0.5 * (x + x.conj().T)

    Hu = apply(hamiltonian)
    K = gram(u, Hu)
    out = {"S": gram(u, u), "K": herm(K), "K_raw": K, "Q": gram(Hu, Hu)}
    for name, op in (observables or {}).items():
        out[name] = herm(gram(u, apply(op)))
    return out


# --------------------------------------------------------------------------
# Coefficients, loss and observables as functions of time
# --------------------------------------------------------------------------


def coefficients(gamma: Array, omega: Array, tau: Array) -> tuple[Array, Array]:
    """Return ``c(tau)`` and ``dc/dtau`` with ``c_0 = 1`` (Eq. 8).

    Args:
        gamma: Complex ``(M, N_b)``.
        omega: Real ``(N_b,)``.
        tau: Times since the start of the window, ``(n_t,)``.

    Returns:
        Two arrays of shape ``(n_t, M + 1)``.
    """
    phase = jnp.exp(1j * jnp.outer(tau, omega))
    c = (phase - 1.0) @ gamma.T
    dc = (1j * omega * phase) @ gamma.T
    one = jnp.ones((tau.shape[0], 1), dtype=c.dtype)
    return jnp.concatenate([one, c], 1), jnp.concatenate([0.0 * one, dc], 1)


def residual_loss(c: Array, dc: Array, S: Array, K: Array, Q: Array) -> Array:
    """Return ``L(t) = ||P_perp (dPsi + i H Psi)||^2 / <Psi|Psi>`` (Eq. 2).

    Args:
        c: Coefficients, ``(n_t, M+1)``.
        dc: Their time derivative, ``(n_t, M+1)``.
        S: ``<u_i|u_j>``.
        K: ``<u_i|H|u_j>`` (with samples, the non-Hermitized ``K_raw`` of
            :func:`basis_matrices`).
        Q: ``<u_i|H^2|u_j>``.

    Returns:
        ``L`` at each time, real, shape ``(n_t,)``.
    """

    def form(x: Array, A: Array, y: Array) -> Array:
        return jnp.einsum("ti,ij,tj->t", x.conj(), A, y)

    n = form(c, S, c).real
    r = form(dc, S, dc).real + form(c, Q, c).real - 2.0 * form(dc, K, c).imag
    p = form(c, S, dc) + 1j * form(c, K, c)
    return r / n - jnp.abs(p) ** 2 / n**2


def expectation(c: Array, op: Array, S: Array) -> Array:
    """Return ``<Psi|O|Psi> / <Psi|Psi>`` for each row of ``c``, ``op = <u|O|u>``."""
    num = jnp.einsum("ti,ij,tj->t", c.conj(), op, c).real
    return num / jnp.einsum("ti,ij,tj->t", c.conj(), S, c).real


def simpson_weights(n: int, length: float) -> np.ndarray:
    """Return Simpson's 1/3 weights for ``n`` (odd) points on ``[0, length]``."""
    if n < 3 or n % 2 == 0:
        raise ValueError("Simpson's rule needs an odd number of points >= 3.")
    w = np.ones(n)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    return w * (length / (n - 1)) / 3.0


def linear_variational_coefficients(
    S: Array, K: Array, c0: Array, tau: Array, *, rcond: float = 1e-10
) -> np.ndarray:
    """Return ``c(tau) = exp(-i tau S^{-1} K) c0`` (Eq. 9, Appendix C).

    This is the Galerkin solution in a fixed basis: it minimizes
    ``L(t)`` over ``dc/dt`` at each time. ``S`` is inverted on its
    eigenvectors above ``rcond * max eig``.
    """
    S = np.asarray(S)
    K = np.asarray(K)
    evals, evecs = np.linalg.eigh(S)
    keep = evals > rcond * evals.max()
    X = evecs[:, keep] / np.sqrt(evals[keep])  # X^+ S X = 1
    e, V = np.linalg.eigh(X.conj().T @ K @ X)
    # Coordinates of c0 in the orthonormal basis X: y0 = X^+ S c0.
    y0 = X.conj().T @ S @ np.asarray(c0)
    y = (V * np.exp(-1j * np.outer(np.asarray(tau), e))[:, None, :]) @ (V.conj().T @ y0)
    return np.asarray(y @ X.T)


# --------------------------------------------------------------------------
# Sampling Pi with single-spin-flip Metropolis
# --------------------------------------------------------------------------


def _metropolis(
    log_prob: Callable[[Array], Array],
    key: Array,
    chains: Array,
    n_sweeps: int,
) -> tuple[Array, Array]:
    """Run ``n_sweeps * N`` single-flip Metropolis steps on every chain.

    Returns the final chains and the acceptance rate.
    """
    n_chains, n_spins = chains.shape

    def step(carry: tuple[Array, Array, Array], key: Array) -> tuple[Any, None]:
        x, lp, acc = carry
        k_site, k_acc = jax.random.split(key)
        site = jax.random.randint(k_site, (n_chains,), 0, n_spins)
        flip = jax.nn.one_hot(site, n_spins, dtype=x.dtype)
        x_new = x * (1 - 2 * flip)
        lp_new = log_prob(x_new)
        accept = jnp.log(jax.random.uniform(k_acc, (n_chains,))) < lp_new - lp
        x = jnp.where(accept[:, None], x_new, x)
        lp = jnp.where(accept, lp_new, lp)
        return (x, lp, acc + accept.mean()), None

    n_steps = n_sweeps * n_spins
    keys = jax.random.split(key, n_steps)
    (chains, _, acc), _ = jax.lax.scan(
        step, (chains, log_prob(chains), jnp.zeros(())), keys
    )
    return chains, acc / n_steps


def sample_pi(
    frozen: FrozenBasis,
    rbm: Params,
    key: Array,
    chains: Array,
    *,
    n_samples: int,
    n_sweeps: int,
    log_lam: Array | None = None,
) -> tuple[Array, Array, Array]:
    """Draw ``n_samples`` configurations of ``Pi`` from persistent chains.

    Each chain contributes one sample every ``n_sweeps`` sweeps.

    Returns ``(samples, chains, acceptance)``.
    """
    n_chains = chains.shape[0]

    def log_prob(x: Array) -> Array:
        return _log_pi(frozen, rbm, x, log_lam)

    def body(chains: Array, k: Array) -> tuple[Array, tuple[Array, Array]]:
        chains, a = _metropolis(log_prob, k, chains, n_sweeps)
        return chains, (chains, a)

    keys = jax.random.split(key, n_samples // n_chains)
    chains, (out, acc) = jax.lax.scan(body, chains, keys)
    return out.reshape(-1, chains.shape[-1]), chains, jnp.mean(acc)


# --------------------------------------------------------------------------
# Spectral bounds (initial frequencies)
# --------------------------------------------------------------------------


def spectral_bounds(
    hamiltonian: nk.operator.DiscreteOperator,
    *,
    symmetric: bool = False,
    n_iter: int = 300,
    n_samples: int = 2048,
    alpha: int = 1,
    seed: int = 0,
    max_exact: int = 2**16,
) -> tuple[float, float]:
    """Estimate ``(E_min, E_max)`` of ``H``, as in Appendix B of the paper.

    Up to ``max_exact`` basis states both bounds are exact (Lanczos).
    Otherwise ``E_min`` is the VMC ground energy (RBM, stochastic
    reconfiguration). With ``symmetric=True``, ``E_max = -E_min``: this
    holds for the TFI model on a bipartite lattice (even ``L`` with
    periodic boundaries), where rotating every spin of one sublattice by
    ``X`` maps ``H`` to ``-J ZZ + h X`` and a global ``Z`` rotation then
    maps it to ``-H`` (checked by ED in the tests). Without it, ``E_max``
    is minus the VMC ground energy of ``-H``, which a positive RBM
    describes poorly (frustrated sign structure); give ``spectrum``
    explicitly in that case.

    The bounds only set the initial frequencies, which are variational.
    """
    hilbert = hamiltonian.hilbert
    # n_states raises for huge spaces: compare log2 sizes (spin-1/2 only).
    if hilbert.size <= np.log2(max_exact):
        import scipy.sparse.linalg as spla

        Hs = hamiltonian.to_sparse()
        lo = spla.eigsh(Hs, k=1, which="SA")[0][0]
        hi = spla.eigsh(Hs, k=1, which="LA")[0][0]
        return float(lo), float(hi)

    def ground(op: Any) -> float:
        sampler = nk.sampler.MetropolisLocal(hilbert, n_chains=256)
        vs = nk.vqs.MCState(
            sampler,
            nk.models.RBM(alpha=alpha, param_dtype=jnp.complex128),
            n_samples=n_samples,
            seed=seed,
            sampler_seed=seed,
        )
        driver = nk.driver.VMC_SR(
            op, optax.sgd(1e-2), variational_state=vs, diag_shift=1e-3
        )
        driver.run(n_iter=n_iter, show_progress=False, out=())
        return float(vs.expect(op).mean.real)

    lo = ground(hamiltonian)
    hi = -lo if symmetric else -ground(-1.0 * hamiltonian)
    return lo, hi


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


@dataclass
class _Window:
    """Trainable parameters of one window."""

    rbm: Params
    gamma: Params
    omega: Array


def _initial_window(
    key: Array,
    n_spins: int,
    settings: NeuralGalerkinSettings,
    omega: Array,
    previous: Params | None,
) -> _Window:
    k_rbm, k_g_re, k_g_im = jax.random.split(key, 3)
    rbm = (
        previous
        if previous is not None
        else init_rbm_batch(
            k_rbm, settings.n_basis, n_spins, settings.alpha, settings.init_std
        )
    )
    shape = (settings.n_basis, settings.n_frequencies)
    s = settings.gamma_init_std
    gamma = {
        "re": s * jax.random.normal(k_g_re, shape),
        "im": s * jax.random.normal(k_g_im, shape),
    }
    return _Window(rbm, gamma, jnp.asarray(omega))


def initial_frequencies(n: int, spectrum: tuple[float, float]) -> np.ndarray:
    """Return ``n`` frequencies evenly spaced over ``[E_min, E_max]``.

    ``c(t)`` uses ``exp(+i omega t)`` while an eigenstate evolves as
    ``exp(-i E t)``, so ``omega_k = -E_k``.
    """
    lo, hi = spectrum
    return -np.linspace(lo, hi, n)


def run_neural_galerkin(
    hamiltonian: nk.operator.DiscreteOperator,
    *,
    T: float,
    observables: Mapping[str, nk.operator.DiscreteOperator],
    settings: NeuralGalerkinSettings | None = None,
    log: Callable[[str], None] | None = None,
    extrapolate: float = 0.0,
) -> NeuralGalerkinResult:
    """Evolve ``|+>^N`` under ``hamiltonian`` up to ``T`` with t-NQG.

    Windows of length ``settings.window`` are solved one after the other;
    each starts from the state at the end of the previous one (Appendix
    A). The RBMs of a window are initialized from those of the previous
    window.

    Args:
        hamiltonian: Hermitian NetKet operator with a JAX version.
        T: Final time, a multiple of ``settings.window``.
        observables: Name mapped to Hermitian operators measured on
            ``settings.n_eval_times`` points per window.
        settings: Hyperparameters.
        log: Called with one progress line per window.
        extrapolate: Also evaluate the last window's ansatz for this long
            after ``T``, outside the interval it was trained on (Fig. 4
            of the paper). ``c(tau)`` is defined for every ``tau``, so the
            basis and coefficients are used as they are.

    Returns:
        The trajectory, energies and losses.
    """
    settings = settings or NeuralGalerkinSettings()
    n_windows = round(T / settings.window)
    if n_windows < 1 or abs(n_windows * settings.window - T) > 1e-9:
        raise ValueError(f"T={T} is not a multiple of window={settings.window}.")
    hilbert = hamiltonian.hilbert
    n_spins = hilbert.size
    H = hamiltonian.to_jax_operator()
    obs = {k: v.to_jax_operator() for k, v in observables.items()}
    spectrum = settings.spectrum or spectral_bounds(
        hamiltonian, symmetric=settings.symmetric_spectrum, seed=settings.seed
    )
    omega0 = initial_frequencies(settings.n_frequencies, spectrum)

    tau_q = jnp.asarray(np.linspace(0.0, settings.window, settings.n_quadrature))
    w_q = jnp.asarray(simpson_weights(settings.n_quadrature, settings.window))
    tau_e = np.linspace(0.0, settings.window, settings.n_eval_times)

    all_states = jnp.asarray(hilbert.all_states()) if settings.exact else None

    def matrices(
        frozen: FrozenBasis,
        rbm: Params,
        samples: Array | None,
        log_lam: Array,
        with_obs: bool,
    ) -> dict[str, Array]:
        if samples is None:
            assert all_states is not None
            logp = jax.lax.stop_gradient(_log_pi(frozen, rbm, all_states, log_lam))
            weights = jnp.exp(logp - jax.scipy.special.logsumexp(logp))
            x = all_states
        else:
            x = samples
            weights = jnp.full(samples.shape[0], 1.0 / samples.shape[0])
        return basis_matrices(
            frozen, rbm, x, weights, H, obs if with_obs else None, log_lam
        )

    @jax.jit
    def measure_chunk(
        frozen: FrozenBasis,
        rbm: Params,
        x: Array,
        w: Array,
        log_lam: Array,
        H: Any,
        obs: Mapping[str, Any],
    ) -> dict[str, Array]:
        # The operators are arguments (JAX pytrees), not closure constants:
        # NetKet packs their internals lazily and a closure would leak tracers.
        return basis_matrices(frozen, rbm, x, w, H, obs, log_lam)

    def measure(
        frozen: FrozenBasis, rbm: Params, samples: Array | None, log_lam: Array
    ) -> dict[str, Array]:
        # Sums over chunks of samples: the connected configurations of all
        # (frozen + current) basis states do not fit in memory at once.
        if samples is None:
            return matrices(frozen, rbm, None, log_lam, with_obs=True)
        n = samples.shape[0]
        total: dict[str, Array] = {}
        for i in range(0, n, settings.eval_chunk):
            x = samples[i : i + settings.eval_chunk]
            w = jnp.full(x.shape[0], 1.0 / n)
            m = measure_chunk(frozen, rbm, x, w, log_lam, H, obs)
            total = m if not total else {k: total[k] + m[k] for k in m}
        return total

    def window_loss(
        params: Params, frozen: FrozenBasis, samples: Array | None, log_lam: Array
    ) -> tuple[Array, Array]:
        mats = matrices(frozen, params["rbm"], samples, log_lam, with_obs=False)
        c, dc = coefficients(_complex(params["gamma"]), params["omega"], tau_q)
        L = residual_loss(c, dc, mats["S"], mats["K_raw"], mats["Q"])
        return jnp.sum(w_q * L) / settings.window, jnp.diag(mats["S"]).real

    def balanced(s_diag: Array) -> Array:
        # S_ii estimates <u_i|u_i> / Z for any lam, so lam_i = 1 / S_ii gives
        # every basis state the same weight in the next mixture.
        new = -jnp.log(jnp.maximum(s_diag, 1e-300))
        return new - new.max()

    @jax.jit
    def update(
        params: Params,
        opt_state: Any,
        frozen: FrozenBasis,
        samples: Array | None,
        log_lam: Array,
    ) -> tuple[Params, Any, Array, Array]:
        (loss, s_diag), grads = jax.value_and_grad(window_loss, has_aux=True)(
            params, frozen, samples, log_lam
        )
        updates, opt_state = opt.update(grads, opt_state, params)
        return (
            optax.apply_updates(params, updates),
            opt_state,
            loss,
            balanced(s_diag),
        )

    @partial(jax.jit, static_argnames=("n_samples",))
    def draw(
        frozen: FrozenBasis,
        rbm: Params,
        key: Array,
        chains: Array,
        log_lam: Array,
        n_samples: int,
    ) -> tuple[Array, Array, Array]:
        return sample_pi(
            frozen,
            rbm,
            key,
            chains,
            n_samples=n_samples,
            n_sweeps=settings.n_sweeps,
            log_lam=log_lam,
        )

    @jax.jit
    def burn_in(
        frozen: FrozenBasis, rbm: Params, key: Array, chains: Array, log_lam: Array
    ) -> Array:
        return _metropolis(
            lambda x: _log_pi(frozen, rbm, x, log_lam), key, chains, settings.n_burn_in
        )[0]

    schedule = optax.cosine_decay_schedule(
        settings.learning_rate, settings.n_iter, alpha=0.05
    )
    opt = optax.adam(schedule)

    key = jax.random.PRNGKey(settings.seed)
    frozen = FrozenBasis()
    previous_rbm: Params | None = None
    t_all: list[np.ndarray] = []
    o_all: dict[str, list[np.ndarray]] = {k: [] for k in observables}
    e_all: list[np.ndarray] = []
    l_all: list[np.ndarray] = []
    lw_all: list[float] = []
    history: list[np.ndarray] = []
    walls: list[float] = []
    chains = None

    for w in range(n_windows):
        start = time.perf_counter()
        key, k_init, k_chain = jax.random.split(key, 3)
        win = _initial_window(k_init, n_spins, settings, omega0, previous_rbm)
        params: Params = {"rbm": win.rbm, "gamma": win.gamma, "omega": win.omega}
        opt_state = opt.init(params)
        # Balanced mixture from an exact or sampled first estimate of S_ii.
        log_lam = jnp.zeros(settings.n_basis + 1)

        if not settings.exact:
            if chains is None:
                chains = jnp.asarray(
                    hilbert.random_state(k_chain, settings.n_chains), dtype=jnp.float64
                )
            key, k_burn = jax.random.split(key)
            chains = burn_in(frozen, params["rbm"], k_burn, chains, log_lam)

        losses = []
        for _ in range(settings.n_iter):
            samples = None
            if not settings.exact:
                key, k_s = jax.random.split(key)
                samples, chains, _ = draw(
                    frozen, params["rbm"], k_s, chains, log_lam, settings.n_samples
                )
            params, opt_state, loss, log_lam = update(
                params, opt_state, frozen, samples, log_lam
            )
            losses.append(float(loss))
        history.append(np.asarray(losses))

        # Measure with fresh samples (or exactly).
        eval_samples = None
        if not settings.exact:
            key, k_e = jax.random.split(key)
            eval_samples, chains, _ = draw(
                frozen, params["rbm"], k_e, chains, log_lam, settings.n_eval_samples
            )
        mats = measure(frozen, params["rbm"], eval_samples, log_lam)
        gamma = _complex(params["gamma"])
        tau_w = tau_e
        if w == n_windows - 1 and extrapolate > 0:
            n_extra = max(2, round(extrapolate / settings.window * len(tau_e)))
            tau_w = np.concatenate(
                [
                    tau_e,
                    settings.window + np.linspace(0.0, extrapolate, n_extra + 1)[1:],
                ]
            )
        c, dc = coefficients(gamma, params["omega"], jnp.asarray(tau_w))
        t_all.append(w * settings.window + tau_w)
        for name in observables:
            o_all[name].append(np.asarray(expectation(c, mats[name], mats["S"])))
        e_all.append(np.asarray(expectation(c, mats["K"], mats["S"])))
        l_all.append(
            np.asarray(residual_loss(c, dc, mats["S"], mats["K_raw"], mats["Q"]))
        )
        cq, dcq = coefficients(gamma, params["omega"], tau_q)
        Lq = residual_loss(cq, dcq, mats["S"], mats["K_raw"], mats["Q"])
        lw_all.append(float(jnp.sum(w_q * Lq) / settings.window))

        # Freeze: A_{w+1} = A_w + sum_m c_m(dT) phi_m.
        c_end, _ = coefficients(gamma, params["omega"], jnp.asarray([settings.window]))
        frozen = frozen.extend(params["rbm"], c_end[0, 1:])
        previous_rbm = params["rbm"]
        walls.append(time.perf_counter() - start)
        if log is not None:
            log(
                f"window {w + 1}/{n_windows}: t={t_all[-1][-1]:.3f} "
                f"loss={lw_all[-1]:.3e} train={losses[-1]:.3e} "
                f"wall={walls[-1]:.1f}s"
            )

    return NeuralGalerkinResult(
        times=np.concatenate(t_all),
        observables={k: np.concatenate(v) for k, v in o_all.items()},
        energy=np.concatenate(e_all),
        loss=np.concatenate(l_all),
        window_loss=np.asarray(lw_all),
        loss_history=history,
        wall_time=walls,
        settings=settings,
    )


def with_settings(
    settings: NeuralGalerkinSettings, **changes: Any
) -> NeuralGalerkinSettings:
    """Return a copy of ``settings`` with some fields changed."""
    return replace(settings, **changes)


__all__ = [
    "FrozenBasis",
    "NeuralGalerkinResult",
    "NeuralGalerkinSettings",
    "basis_matrices",
    "coefficients",
    "expectation",
    "init_rbm_batch",
    "initial_frequencies",
    "linear_variational_coefficients",
    "log_rbm_batch",
    "mean_magnetization_x",
    "residual_loss",
    "run_neural_galerkin",
    "sample_pi",
    "simpson_weights",
    "spectral_bounds",
    "tfi_square_lattice",
    "with_settings",
]
