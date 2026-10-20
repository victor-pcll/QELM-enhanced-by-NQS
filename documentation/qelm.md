# QELM

The package moved from `QRC.NQS` to `QELM.NQS` (issue #19). A quantum extreme learning machine has no recurrence. Each input is processed on its own: it drives a fixed spin reservoir for a short time, a few observables are measured, and only a linear readout is trained. Short evolutions are where the variational (TDVP) dynamics stays close to the exact one, which is why the project left reservoir computing.

## Feature map

`QELM.NQS.qelm.QELMReservoir(couplings, h0, kappa, T, n_readouts, ...)` defines the reservoir:

- Hamiltonian `H(x) = h(x) Σ Z_i + Σ_{i<j} J_ij X_i X_j` with `h(x) = h0 + kappa x`. The input sets the field, and the couplings are fixed. `random_couplings(N, scale, seed)` draws them uniformly in `[-scale, scale]`.
- Initial state: the same seeded RBM for every input, with `init_std=0.1`, a regular point of the manifold (see [Dynamics](dynamics.md)).
- Features: `<Z_i>(t_k)` at `t_k = k T / n_readouts`, `k = 1..n_readouts`, in time-major order. That gives `n_readouts * N` features in `[-1, 1]`.
- TDVP settings: RK45, `dt = 1e-2`, `diag_shift = 1e-6`. The features must follow ED, so the QGT shift is kept small (see [Dynamics](dynamics.md) for the bias of `1e-4`).

`feature_matrix(reservoir, inputs, method="tdvp" | "exact")` returns the `(n_samples, n_features)` design matrix. `"exact"` evolves the RBM vector with `expm_multiply`, and `"tdvp"` runs `run_tdvp` on the RBM.

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
| `test_readout.py` | Exact recovery of a linear map, shrinkage with `lam`, unpenalized bias |

Measured on N=4, `h(x) = 1 + x`, `x = 0.3`: the largest error on `<Z_i>` is 1.6e-7 up to t=0.1, 3e-5 up to t=0.5, and 8e-5 up to t=2 and t=4.
