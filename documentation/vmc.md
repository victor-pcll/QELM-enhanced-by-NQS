# Ground-state preparation (VMC)

`QRC.NQS.embedding.VMCGroundStatePreparation` optimizes a sampled RBM state (`nk.vqs.MCState`) towards the ground state of a Hamiltonian with NetKet's `VMC_SR` driver. Stochastic reconfiguration is imaginary-time evolution projected on the variational manifold.

```python
from QRC.NQS.embedding import VMCGroundStatePreparation, VMCSettings

prep = VMCGroundStatePreparation(hamiltonian, mc_state, settings=VMCSettings())
state = prep.run()  # same object, optimized in place
prep.energy  # mean energy per iteration
prep.energy_variance  # sigma_E^2 per iteration
```

`VMCGroundStatePreparation.from_config(hamiltonian, hilbert, cfg)` builds the RBM `MCState` from `cfg.vmc` and `cfg.seed`. The optimizer defaults to `optax.sgd(learning_rate)`. Any Optax optimizer can be passed instead.

## Defaults (`conf/vmc/default.yaml`)

| Key | Value |
| --- | --- |
| `learning_rate` | `1e-2` |
| `n_iter` | `300` |
| `n_samples` | `2048` |
| `alpha` | `1` |
| `param_dtype` | `complex` |
| `diag_shift` | `1e-4` |

Override them from the command line, for example `vmc.n_iter=500`.

## Outputs

`run()` writes to the `exp/` folder of the active Hydra run (`outputs/<experiment_name>/<timestamp>/exp/`). Outside Hydra, it writes to a new run under `outputs/default/`, or to the `paths` you pass.

- `hyperparameters.json`: the settings, the number of parameters, the system, and any `metadata`.
- `vmc.log`: NetKet's JSON log, with energy mean, error, variance and R̂ per iteration.
- `vmc.mpack`: the optimized variables.

## Validation

On the periodic transverse-field Ising chain (N=4, h=J=1, `E0 = -5.22625`), the defaults reach `|E - E0| ≈ 2e-7` in 300 iterations (summed exactly over the final state), and the energy variance falls by more than three orders of magnitude.

A larger step can stop on an excited eigenstate. With `learning_rate=5e-2` and the same seed, VMC stalls at `E = -4.4711` with a variance of `3e-7`. That is an exact excited eigenstate: its variance is zero too. So a vanishing variance shows convergence to *an* eigenstate. To confirm the ground state, compare with ED (small N) or check the energy.
