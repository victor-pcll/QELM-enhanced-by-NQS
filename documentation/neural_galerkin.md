# Time-dependent Neural Galerkin method (t-NQG)

Issue #24 reproduces Fig. 2 of A. Sinibaldi, D. Hendry, F. Vicentini and G. Carleo, *Time-dependent Neural Galerkin Method for Quantum Dynamics*, [arXiv:2412.11778](https://arxiv.org/abs/2412.11778). The code is `QELM.NQS.dynamics.neural_galerkin` (the method), `QELM.NQS.dynamics.galerkin_benchmark` (the quench, TDVP and ED runs, and the figures) and `notebooks/benchmarks/benchmark_neural_galerkin_2026.ipynb`.

## Model

`tfi_square_lattice(L, h=h, J=1.0)` returns the periodic `L x L` lattice and

```text
H = -J Σ_<ij> Z_i Z_j - h Σ_i X_i      (Eq. 7, Pauli matrices)
```

NetKet's `Ising` writes `+J ZZ`, so it is built with `J_nk = -J`. A test compares it with a Kronecker-product construction on the 3x3 lattice. On an even periodic lattice the spectrum is symmetric (`E -> -E`, by a sublattice rotation followed by a global one), which the tests check by Lanczos on 4x4. The initial state is `|+>^N`, with `<σ^x> = 1` and `E = -h N`.

## Ansatz and loss

On window `w` (start `t_w`, length `ΔT`):

```text
|Ψ(t)> = |A_w> + Σ_{i=1..M} c_i(t - t_w) |φ_i>,     c_i(τ) = Σ_k γ_ik (exp(i ω_k τ) - 1)
```

`|A_0> = |+>^N` and `|A_{w+1}> = |Ψ(t_w + ΔT)>`, the frozen state reached at the end of window `w` (all its RBMs with their final coefficients). `φ_i` are complex RBMs, and `γ`, `ω` and the RBM parameters are all trained with Adam on the Simpson-integrated loss

```text
L(t) = || P_⊥ (∂_t Ψ + i H Ψ) ||² / <Ψ|Ψ>,        L_[t_w, t_w+ΔT] = (1/ΔT) ∫ L(t) dt
```

Because `c(0) = 0`, `Ψ(t_w) = A_w` exactly: the initial condition is never approximated. `ω_k` start evenly spaced over `[E_min, E_max]` (`spectral_bounds`: exact up to `2^16` states, otherwise VMC for `E_min` and the symmetry for `E_max`).

## Estimator

With `u = (A, φ_1, ..., φ_M)`, the loss is an exact function of three `(M+1) x (M+1)` matrices:

```text
S = <u|u>,  K = <u|H|u>,  Q = <u|H²|u>
n = c†Sc,  r = ċ†Sċ + c†Qc - 2 Im(ċ†Kc),  p = c†Sċ + i c†Kc,   L = r/n - |p|²/n²
```

They are estimated once per Adam step by sampling the balanced mixture `Π(s) = Σ_i λ_i |u_i(s)|²` with `λ_i = 1/<u_i|u_i>` (from the previous step), and the same samples serve all 257 integration points. The paper samples `|Ψ(s, t)|²` at each integration point; both estimate the same `L(t)`, but ours costs one sampling per step instead of 257. Two details matter for correctness:

- `K` in the loss is the non-Hermitized estimate `K_raw = mean[u* (Hu)]`. Then `(S, K_raw, Q)` are the blocks of a Gram matrix in the sample inner product and the estimated `L` is a squared norm, `≥ 0` for any samples. The Hermitized `K` can give a negative estimate, which Adam exploits (we saw a negative training loss before this fix).
- Without the balanced weights, a basis state of large norm takes all the samples and `A` (the state near the start of the window) is not sampled at all.

Observables follow the same route, `<O>(t) = c†Oc / c†Sc` with `O = <u|O|u>`, measured at the end of each window with 8x more samples. `linear_variational_coefficients` gives the fixed-basis solution `c(t) = exp(-i t S⁻¹ K) c(0)` (Eq. 9, Appendix C); it is tested but not used in the runs.

## TDVP with the same loss

`galerkin_benchmark.tdvp_quench` runs the repository's `run_tdvp` (RK4, `dt = 0.005`, `diag_shift = 1e-4`, 2048 samples) from one complex RBM that is exactly `|+>^N` (the `plus_state_rbm` parameters of issue #26). `tdvp_residual` scores it with the same loss, `L = θ̇†Sθ̇ + Var(H) - 2 Im(θ̇†F)` for the `θ̇` that TDVP integrates; a test checks it against `||P_⊥(J θ̇ + iHΨ)||²` on dense vectors.

## Settings

| Setting | Paper (6x6) | Here |
| --- | --- | --- |
| `M` | 18 | 18 |
| `N_b` | 128 | 128 |
| `ΔT` | 0.2 | 0.2 |
| Simpson points | 257 | 257 |
| Samples | 512 per integration point | 1024 per Adam step, shared by all points |
| RBM density | not given | `alpha = 1` |
| Adam | not given | lr `1e-2`, cosine decay, 300 (4x4) or 200 (6x6) steps per window |
| Final time | not given | `T = 1` (4x4), `T = 0.6` (6x6) |

The 6x6 run is limited by the 4 CPU cores of the session: about 1.5 h per field. The same command with more steps and a longer `T` reproduces the full figure on a cluster.

## Results

4x4, `T = 1`, ED reference (`figure2_4x4.png`, `errors_4x4.png` in the PR):

| h | Method | max `\|Δ<σ^x>\|` | mean `\|Δ<σ^x>\|` | median `L/N` | max `\|E(t) - E(0)\|/N` | wall |
| --- | --- | --- | --- | --- | --- | --- |
| 5 | t-NQG, M = 18 | 1.6e-2 | 8.8e-3 | 2.8e-3 | 4.3e-2 | 37 min |
| 5 | TDVP, 1 RBM | 7.9e-3 | 1.9e-3 | 3.3e-3 | 2.6e-2 | 3 min |
| 3.044 | t-NQG, M = 18 | 1.5e-2 | 5.4e-3 | 1.3e-2 | 3.9e-2 | 36 min |
| 3.044 | TDVP, 1 RBM | 3.7e-2 | 1.1e-2 | 1.1e-1 | 3.1e-2 | 3 min |
| 2 | t-NQG, M = 18 | 3.0e-2 | 1.5e-2 | 4.1e-2 | 7.7e-2 | 37 min |
| 2 | TDVP, 1 RBM | 8.1e-2 | 2.3e-2 | 2.5e-1 | 5.9e-2 | 3 min |

At the critical point and in the ferromagnetic phase, t-NQG beats one TDVP RBM by 2-3x on `<σ^x>` and by about 6x on the loss. At `h = 5` a single RBM is already accurate (its loss is as low as that of t-NQG), and t-NQG (300 Adam steps per window) is limited by its optimization: it keeps a small systematic offset. `L(t)` of t-NQG peaks at the start of each window, where `ċ(0)` alone must match `-iHA_w`.

## Tests

`tests/QELM/NQS/dynamics/test_neural_galerkin.py`: Hamiltonian against Kronecker products, symmetric spectrum, `Ψ(0) = |+>^N` for any parameters, `ċ` against finite differences, Simpson's rule exact on cubics, the matrix loss against `||P_⊥(∂_tΨ + iHΨ)||²` on vectors, gauge invariance under `H -> H + E_0`, `L = 0` on the exact solution and `L > 0` on a wrong one, Eq. 9 against `expm`, Monte Carlo against exact sums (including a peaked window), `L ≥ 0` for 5 samples, and a 3x3 quench against ED (error on `<σ^x>` below 6e-3 for a drop of 0.3, energy within 1 %). `test_galerkin_benchmark.py`: the TDVP loss on vectors and its minimality, the `|+>^N` MCState, a 3x3 TDVP quench against ED, and the figure helpers.
