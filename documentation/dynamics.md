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
| `diag_shift` | `1e-4` | QGT regularization, see below |
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

RK4 and RK45 give the same error, so the error comes from the projection and the regularization, not from the time step. The default is `1e-4` (issue #17), the same as the VMC: it keeps the QGT solve stable but costs about 1e-3 on `<Z>`. The baseline, which must match ED to `1e-3`, passes `diag_shift=1e-6` explicitly (as does the notebook). `tests/QELM/NQS/dynamics/test_tdvp_baseline.py` measures both errors, so the bias stays documented by a test.

## Baseline

`tdvp_vs_exact(n_spins=6, h=10.0, J=1.0, T=5.0, seed=45, settings=..., paths=...)` starts both evolutions from the same seeded RBM state, held exactly as a `FullSumState`. It returns both trajectories, the energies and the final fidelity. With `paths`, it writes `exp/tdvp.log`, `exp/tdvp_vs_ed.npz` and `plots/tdvp_vs_ed.png`. `notebooks/results/01_tdvp_vs_ed_baseline.ipynb` runs it for T=5. The CI test `tests/QELM/NQS/dynamics/test_tdvp_baseline.py` runs it for T=0.5.

## Accumulated error against the field (issue #23)

`QELM.NQS.dynamics.accumulated_error` measures how much a TDVP trajectory drifts from ED over a whole run, not only at one time:

```text
E_O(T) = ∫_0^T |<O>_TDVP(t) - <O>_ED(t)| dt
```

with the trapezoidal rule on the shared grid (`accumulated_absolute_error`). Its unit is `[O] × time`; divide by `T` for the time average.

`compare_with_exact(n_spins, h, J, T, seed, settings)` starts both evolutions from the seeded RBM, as the baseline does, and calls `run_tdvp(..., record_states=True)`. With that flag, a NetKet callback stores `state.to_array()` at every grid time (NetKet calls it only at the `tstops`, before the step leaves them). A test checks that these vectors give back NetKet's own logged observables to round-off. Every observable is then evaluated from the state vectors with the same code for both trajectories (`trajectory_observables`):

| Key | Quantity |
| --- | --- |
| `energy` | `<H>` |
| `magnetization_z` | `(1/N) Σ_i <Z_i>` |
| `sigma_x`, `sigma_z` | `<X_i>`, `<Z_i>`; the error is averaged over sites, `(1/N) Σ_i E_{P_i}` |
| `entropy` | Half-chain von Neumann entropy of sites `0..N/2-1`, in nats, from the Schmidt values |
| `infidelity` | `∫ (1 - F) dt` with `F = |<ψ_ED|ψ_TDVP>|^2` |

The entropy reshapes the amplitudes with site 0 as the most significant bit, which is NetKet's ordering; a test places a Bell pair on chosen sites to check it. `accumulated_error_vs_field(n_spins, fields, T, ...)` repeats the comparison for each `h` and returns a `FieldSweep` (errors, final fidelity and TDVP wall time per field), which `save`/`load` as `.npz`. `plot_field_scan` draws one panel per quantity, one curve per `N`.

The tests (`tests/QELM/NQS/dynamics/test_accumulated_error.py`) check the integral on known errors, the entropy on product, Bell, GHZ and random states (Page bound), the local Paulis on product states with known energy, exact constraints on a real comparison (zero error at `t = 0`, `F ≤ 1`, `|mean Δ| ≤ mean |Δ|`, energy conservation), and the free-spin case `J = 0`. There `H = h Σ Z_i` only rotates each spin; `<Z_i>` and `S` are conserved, and `ψ(s, t) = e^{-iht Σ s_i} ψ(s, 0)` is still an RBM (visible biases shifted by `-iht`), so TDVP has no projection error.

`notebooks/results/01_tdvp_vs_ed_baseline.ipynb` runs the scan for N = 4, 6, 8 and h from 0.1 to 10 at T = 2.

**NetKet can step past `T`; `run_tdvp` undoes it.** The driver adds up its steps, and its end test is a strict `t < T` while its stop test is `np.isclose`. With `dt = 5e-3, T = 0.1` the sum reaches `0.09999999999999999`: the last stop is taken (observables logged), but the run is not over, and with no stop left the next step is not capped, so the driver ends at `t = 0.105`. The state was then `ψ(T + dt)` (infidelity `1e-4` with the exact `ψ(T)` on N = 4). `run_tdvp` keeps the parameters at each stop in a callback and puts back those of the last grid time after the run; it raises if the last stop is not `T`. `tests/QELM/NQS/dynamics/test_tdvp_final_time.py` checks the state against ED for three grids, and a guard test fails if a NetKet upgrade stops overshooting. The project grids (`dt = 1e-3`, `T = 0.5, 2, 5`) never triggered it, so earlier results are unchanged.

## Loschmidt echo (reversibility test)

`tests/QELM/NQS/dynamics/test_tdvp_reversibility.py` evolves a state under `H` for `T = 1`, then evolves the same `state` object under `-H` for `T` (`run_tdvp` updates the state in place). Flipping `H` flips the projected flow `dθ/dt = -i S⁻¹ F`, so a reversible pipeline returns to `ψ(0)`. On the periodic Ising chain (N=4, h=J=1), `<Z>` goes from 0.76 through a range of 0.84. The backward trajectory retraces the forward one to about 1e-7, the final fidelity is 1 to better than 1e-6, and a negative control (twice `+H`) misses by 0.6 in `<Z>`.

**Start from a regular point of the RBM manifold.** NetKet initializes the RBM parameters with standard deviation 0.01. Near `W = 0`, `d log ψ / d W_ij = s_i tanh(θ_j)` is nearly zero, so the QGT is nearly singular and the tangent space misses the correlated directions the dynamics needs. The **first** TDVP step then jumps off the exact trajectory, and the error stays constant afterwards. On the polarized N=4 chain, the fidelity with the exact state drops to 0.961 at `t = 0.01` (RK45) and stays there. The echo exposes this as irreversibility, because the backward run starts from a regular point. `rbm(init_std=0.1)` avoids it (forward and echo fidelity 1.0000000). The N=6 baseline of issue #2 starts from an unpolarized state, and its first-step loss is within the 1e-3 tolerance.
