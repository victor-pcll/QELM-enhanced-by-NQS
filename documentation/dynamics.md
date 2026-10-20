# Dynamics

## Reservoir Hamiltonian

`QELM.NQS.measurements.all_to_all_xx_hamiltonian(hilbert, h=h, J=J)` builds

```text
H = h Σ_i Z_i + Σ_{i<j} J_ij X_i X_j
```

with Pauli matrices. `J` is a scalar (uniform couplings) or a real symmetric `(N, N)` matrix whose diagonal is ignored. `H` is real symmetric and commutes with the parity `Π_i Z_i`. `mean_magnetization_z(hilbert)` is `(1/N) Σ_i Z_i`. NetKet's `sigmaz` has eigenvalue `+1` on the local state `+1`, which is the convention of the `LocalOperator`/`np.kron` construction in issue #2. A test checks that both constructions give the same matrix.

## Exact evolution (ED reference)

`exact_time_evolution(H, psi0, times)` returns `exp(-iHt)|psi0>` for each time with `scipy.sparse.linalg.expm_multiply`, without forming the propagator. A uniform grid starting at 0 is evaluated in one call. `expectation_values(states, O)` returns `<psi|O|psi>/<psi|psi>`. `time_grid(T, dt)` returns `0, dt, ..., T` with both ends included. It is shared by ED and TDVP.

## TDVP

`run_tdvp(H, state, T=..., observables=..., settings=TDVPSettings(...))` runs `nk.experimental.TDVP` with `QGTJacobianDense` and updates `state` in place (after the call it holds `ψ(T)`). The observables are recorded on `time_grid(T, dt)`: the grid is passed as `tstops`, so adaptive integrators (`RK45`, `RK23`) land on exactly those times. NetKet's log stores the physical time `t` in `iters`, not a step index, so no `iters * dt` conversion is needed. Pass `log_path` to also write NetKet's JSON log, for instance `paths.exp / "tdvp"`.

| `TDVPSettings` field | Default | Note |
| --- | --- | --- |
| `dt` | `1e-3` | Initial step and recording grid |
| `integrator` | `RK45` | Any of `RK4`, `RK45`, `RK23`, `Euler`, `Heun` |
| `diag_shift` | `1e-6` | QGT regularization, see below |
| `holomorphic` | `True` | The complex RBM is holomorphic |
| `linear_solver` | `svd` | Or `pinv_smooth` |
| `rcond` | `None` | Solver cutoff; `None` keeps NetKet's default |

**The QGT shift biases the dynamics.** On the baseline (N=6, h=10, T=0.5), the largest error on `<Z>` is about:

| Setting | Max error on `<Z>` | Fidelity at T |
| --- | --- | --- |
| `diag_shift=1e-4`, SVD | 1.1e-3 | 0.99992 |
| `diag_shift=1e-6`, SVD | 8e-5 | 0.999999 |
| `diag_shift=0`, SVD | 1.2e-3 | 0.9995 |
| `diag_shift=1e-8`, `pinv_smooth` | 3e-6 | 0.9999999 |

RK4 and RK45 give the same error, so the error comes from the projection and the regularization, not from the time step. The value `1e-4` from the issue #2 example fails the required `1e-3` tolerance, so the default is `1e-6`.

## Baseline

`tdvp_vs_exact(n_spins=6, h=10.0, J=1.0, T=5.0, seed=45, settings=..., paths=...)` starts both evolutions from the same seeded RBM state, held exactly as a `FullSumState`. It returns both trajectories, the energies and the final fidelity. With `paths`, it writes `exp/tdvp.log`, `exp/tdvp_vs_ed.npz` and `plots/tdvp_vs_ed.png`. `notebooks/01_tdvp_vs_ed_baseline.ipynb` runs it for T=5. The CI test `tests/QELM/NQS/dynamics/test_tdvp_baseline.py` runs it for T=0.5.

## Loschmidt echo (reversibility test)

`tests/QELM/NQS/dynamics/test_tdvp_reversibility.py` evolves a state under `H` for `T = 1`, then evolves the same `state` object under `-H` for `T` (`run_tdvp` updates the state in place). Flipping `H` flips the projected flow `dθ/dt = -i S⁻¹ F`, so a reversible pipeline returns to `ψ(0)`. On the periodic Ising chain (N=4, h=J=1), `<Z>` goes from 0.76 through a range of 0.84. The backward trajectory retraces the forward one to about 1e-7, the final fidelity is 1 to better than 1e-6, and a negative control (twice `+H`) misses by 0.6 in `<Z>`.

**Start from a regular point of the RBM manifold.** NetKet initializes the RBM parameters with standard deviation 0.01. Near `W = 0`, `d log ψ / d W_ij = s_i tanh(θ_j)` is nearly zero, so the QGT is nearly singular and the tangent space misses the correlated directions the dynamics needs. The **first** TDVP step then jumps off the exact trajectory, and the error stays constant afterwards. On the polarized N=4 chain, the fidelity with the exact state drops to 0.961 at `t = 0.01` (RK45) and stays there. The echo exposes this as irreversibility, because the backward run starts from a regular point. `rbm(init_std=0.1)` avoids it (forward and echo fidelity 1.0000000). The N=6 baseline of issue #2 starts from an unpolarized state, and its first-step loss is within the 1e-3 tolerance.
