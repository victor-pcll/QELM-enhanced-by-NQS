# QELM

The package moved from `QRC.NQS` to `QELM.NQS` (issue #19). A quantum extreme learning machine has no recurrence. Each input is processed on its own: it drives a fixed spin reservoir for a short time, a few observables are measured, and only a linear readout is trained. Short evolutions are where the variational (TDVP) dynamics stays close to the exact one, which is why the project left reservoir computing.

## Feature map

`QELM.NQS.qelm.QELMReservoir(couplings, h0, kappa, T, n_readouts, ...)` defines the reservoir:

- Hamiltonian `H(x) = h(x) Σ Z_i + Σ_{i<j} J_ij X_i X_j` with `h(x) = h0 + kappa x`. The input sets the field, and the couplings are fixed. `random_couplings(N, scale, seed)` draws them uniformly in `[-scale, scale]`.
- Initial state: the same for every input. `initial="rbm"` (default) is a seeded RBM with `init_std=0.1`, a regular point of the manifold (see [Dynamics](dynamics.md)). `initial="quench"` is the ground state of a pure field, see [Quench initialization](#quench-initialization-issue-26).
- Features: `<Z_i>(t_k)` at `t_k = k T / n_readouts`, `k = 1..n_readouts`, in time-major order. That gives `n_readouts * N` features in `[-1, 1]`.
- TDVP settings: RK45, `dt = 1e-2`, `diag_shift = 1e-6`. The features must follow ED, so the QGT shift is kept small (see [Dynamics](dynamics.md) for the bias of `1e-4`).

`feature_matrix(reservoir, inputs, method="tdvp" | "exact")` returns the `(n_samples, n_features)` design matrix. `"exact"` evolves the RBM vector with `expm_multiply`, and `"tdvp"` runs `run_tdvp` on the RBM.

## Quench initialization (issue #26)

`QELMReservoir(..., initial="quench", quench_axis="z" | "x")` starts every input from the ground state of `H_0 = -h Σ A_i` and quenches it to `H(x)`. Both ground states are product states (zero entanglement, `E_0/N = -h`, gap `2h`), so all correlations come from the reservoir. `pure_field_hamiltonian(hilbert, h=h, axis=...)` builds `H_0`. `notebooks/02_quench_initialization.ipynb` runs the protocol (Fujii & Nakajima, Phys. Rev. Applied 8, 024030 (2017)): VMC on `H_0` with `E/N` against the target `-h`, the quench dynamics (features, conserved energy, entanglement growth) and the NMSE of each initial state.

| `quench_axis` | `H_0` | State | How TDVP runs |
| --- | --- | --- | --- |
| `"z"` (default) | `-h Σ Z_i` | `\|0>^N` | In the Hadamard frame `U = Π (X_i + Z_i)/√2`: state `U\|0>^N = \|+>^N`, Hamiltonian `U H U = h Σ X_i + Σ J_ij Z_i Z_j` (`frame="hadamard"`), observables `X_i` |
| `"x"` | `-h Σ X_i` | `\|+>^N` | In the lab frame |

Two points are needed for TDVP to follow ED:

- **`\|0>^N` is not an RBM.** A single configuration needs infinite biases; with a large finite bias the Born distribution sits on one configuration, the QGT collapses and TDVP does not move. The Hadamard frame is an exact change of basis. ED always runs in the lab frame, so the TDVP-vs-ED tests check the frame too.
- **`\|+>^N` must be written at a regular point.** `a = b = W = 0` is `\|+>^N`, but `∂ log ψ / ∂W_ij = s_i tanh(0) = 0`: the tangent space has no pair directions and the `Z_i Z_j` terms of the rotated Hamiltonian are projected away (error of order 1 on `<Z_i>`). `plus_state_rbm` connects hidden unit `j` only to spin `j` with weight `w = 1` and zero biases. Since `cosh(w s_j) = cosh(w)`, the state is still exactly `\|+>^N`, and `∂ log ψ / ∂W_kj = s_k s_j tanh(w)` spans every pair direction.

`hadamard_plus_state` applies the Hadamard gates to `\|0>^N` with NetKet's own Pauli operators, and `entanglement_entropy(psi, N, sites)` gives the von Neumann entropy of a subsystem (site 0 is the most significant digit of NetKet's basis).

Measured on N=4 (couplings seed 7, T=1, 7 inputs): TDVP within 4e-5 (Z) and 1.2e-4 (X) of ED. Held-out NMSE on `sin(πx)`: 4e-6 (Z), 2e-7 (X), 1e-7 (random RBM), identical for ED and TDVP to two digits.

## Readout

`RidgeReadout(lam).fit(F, y).predict(F)` is ridge regression with an unpenalized bias. It is solved in closed form on centered data, and `lam = 0` gives the minimal-norm least-squares solution.

## Tests

`tests/QELM/NQS/qelm/`:

| Test | What it checks |
| --- | --- |
| `test_exact_features_match_dense_propagator` | ED features equal `<Z_i>` from the dense `exp(-iHt)` to 1e-10 |
| `test_without_couplings_magnetizations_are_conserved` | `J = 0`: `H` is diagonal in `Z`, so `<Z_i>(t)` stays at its initial value for any input (ED and TDVP) |
| `test_without_input_scale_features_ignore_the_input` | `kappa = 0`: the features do not depend on `x` |
| `test_tdvp_follows_ed_at_short_times` | N=4, T=1, 7 inputs: TDVP features within 2e-4 of ED (measured: 5.5e-5) |
| `test_tdvp_error_grows_with_evolution_time` | Largest TDVP error below 1e-6 up to t=0.1, more than 10 times larger up to t=2, still below 2e-4 |
| `test_readout_trained_on_ed_transfers_to_tdvp` | A readout trained on ED features moves by at most `‖w‖₁ × max error` on TDVP features |
| `test_quench.py` | Ground states of `H_0` (`E_0/N = -h`, gap `2h`, zero entanglement), exact RBM `\|+>^N`, Hadamard frame equal to `U H U` and same features in both frames, conserved `<H(x)>(t)` (ED and TDVP), features odd in `h` for the X quench, `J = 0` keeps `<Z_i>` fixed, entanglement grows from 0, TDVP within 3e-4 of ED, negative control `W = 0` misses by more than 0.1, readout NMSE, VMC on `H_0` reaches `E/N = -h` to 1e-3 |
| `test_readout.py` | Exact recovery of a linear map, shrinkage with `lam`, unpenalized bias |

Measured on N=4, `h(x) = 1 + x`, `x = 0.3`: the largest error on `<Z_i>` is 1.6e-7 up to t=0.1, 3e-5 up to t=0.5, and 8e-5 up to t=2 and t=4.
