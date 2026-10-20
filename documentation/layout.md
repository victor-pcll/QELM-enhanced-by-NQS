# Layout

| Path | Role |
| --- | --- |
| `conf/` | Hydra YAML: `config.yaml` plus groups `sim_type`, `training`, and `evaluation` |
| `main.py` | Entrypoint. Seeds NumPy, loads `.env`, prints the selection, and runs the pipeline |
| `src/QRC/NQS/models/` | `IsingModel` (dense Hamiltonian built with NetKet), `KitaevModel` scaffold, and the time-evolution stub |
| `src/QRC/NQS/variational/` | NetKet 3 building blocks: Hilbert space, chain, Ising Hamiltonian, magnetization, RBM, variational states, `build_nqs(cfg)` |
| `src/QRC/NQS/measurements/` | Reservoir Hamiltonian `h Σ Z_i + Σ_{i<j} J_ij X_i X_j` and observables |
| `src/QRC/NQS/dynamics/` | Exact evolution (`expm_multiply`), TDVP driver, TDVP-vs-ED baseline. See [Dynamics](dynamics.md) |
| `src/QRC/NQS/embedding/` | Ground-state preparation by VMC (`VMCGroundStatePreparation`). See [VMC](vmc.md) |
| `src/QRC/NQS/evaluation/QGT.py` | Quantum geometric tensor analysis. See [QGT](qgt.md) |
| `src/QRC/NQS/evaluation/representations.py` | SVD of the RBM weights. See [Hidden representations](representations.md) |
| `src/QRC/NQS/utils/makeplots.py` | Every figure of the project. Writes into the run's `plots/` folder |
| `src/QRC/NQS/utils/outputs.py` | Find and load runs under `outputs/` (used by the notebooks) |
| `src/QRC/NQS/paths.py` | Run folder layout: `exp/`, `logs/`, `plots/` |
| `src/QRC/NQS/training/` | Trainer scaffolds (`RidgeRegression`, `Readout`) |
| `src/QRC/NQS/evaluation/` | Metric and plot scaffolds |
| `src/QRC/NQS/pipeline.py` | Routes sim type to a model, a trainer, and an evaluator, then saves a checkpoint |
| `src/QRC/NQS/checkpoint.py` | Writes the `.npz` checkpoint |
| `src/QRC/NQS/seed.py` | Reads `seed` from the composed config |
| `src/QRC/NQS/selection.py` | Loads `.env` and formats the selection line |
| `data/` | Local datasets. Gitignored, and not created by the package |
| `notebooks/00_outputs_starter.ipynb` | Loads the latest run of an experiment from the root `outputs/` |
| `notebooks/01_tdvp_vs_ed_baseline.ipynb` | TDVP (RBM) vs exact dynamics, N=6, h=10, T=5 |
| `notebooks/series.ipynb` | Plots a local `data/series.npy` into `outputs/series.png` |
| `tests/` | pytest |

Import the library as `QRC.NQS`.

## What one run does

`run_pipeline` resolves names, then calls the selected pieces:

1. `cfg.sim_type` selects `IsingModel` (`ising`) or `KitaevModel` (`kitaev`). `from_config` copies the scalar keys in that simulation block.
2. `cfg.training.name` selects `RidgeRegression` (`ridge`) or `Readout` (`readout`). `from_config` reads `learning_rate` from the simulation block when that key exists, and leaves it unset when it does not (the Kitaev block has no keys).
3. `model.evolve` records `dt` and `n_steps` from `cfg.test` and sets `status` to `scaffold`. It does not allocate a trajectory. `IsingModel.hamiltonian` returns the dense matrix; the Kitaev one raises `NotImplementedError`.
4. `trainer.fit(features=None, targets=None)` returns `"scaffold"`. No weight vector is estimated. `state_dict` stores a length-0 `float64` array under `weights`. `predict` is the same kind of stub.
5. The `default` evaluator lists the metric names and plot name from `cfg.evaluation` and returns `status=scaffold`. It does not score predictions or write an image.
6. `build_nqs` builds the NetKet system named by `sim_type` and the ansatz in `cfg.nqs` (default: complex RBM, `alpha=1`, exact `FullSumState`). Systems without a NetKet Hamiltonian (Kitaev) report `nqs=unavailable`.
7. `save_checkpoint` writes the archive described in [Checkpoints and local data](checkpoints.md), and the initial NQS variables are written next to it as `nqs_variables.msgpack`.

The printed pipeline line names the model, the trainer, `dt`, `n_steps`, the evaluator report, the Hilbert dimension, the number of variational parameters, `<H>` of the untrained state, and the checkpoint path.
