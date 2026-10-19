# Running

From the repository root, after `uv sync`:

```bash
uv run python main.py sim_type=ising ising.N=12 test.dt=0.05
```

Hydra composes `conf/config.yaml` and applies those overrides:

- `sim_type=ising` selects `conf/sim_type/ising.yaml` and `IsingModel`.
- `ising.N=12` overrides `N` on the Ising block. `J`, `h`, `pbc` and `learning_rate` stay at the values in that YAML file.
- `test.dt=0.05` overrides the test step. `test.n_steps` stays at the value in `conf/config.yaml`.

The process prints the run directory, the selection (`sim_type`, each key in the simulation block, and each `test` key), then the pipeline status. Everything is written under `outputs/<experiment_name>/<timestamp>/`. See [Checkpoints and local data](checkpoints.md).

Other overrides use the same form:

```bash
uv run python main.py sim_type=kitaev training=readout
uv run python main.py training=ridge evaluation=default
uv run python main.py seed=0
uv run python main.py experiment_name=tfim_scan ising.h=0.5 nqs.alpha=2
uv run python main.py nqs.state=mc nqs.n_samples=4096
uv run python main.py checkpoint.dir=/tmp/qrc-run checkpoint.use_uuid=true
```

`training=ridge` and `evaluation=default` match the defaults in `conf/config.yaml`. `training=readout` selects `Readout`. `evaluation=default` is the only registered evaluator. An unknown `sim_type`, trainer name, or evaluator name raises `KeyError`.

## Notebooks

From the repository root, run `uv sync`. `notebooks/00_outputs_starter.ipynb` finds the repository root by itself and loads the latest run of an experiment from `outputs/` (checkpoint metadata and the figures in `plots/`).

`notebooks/series.ipynb` is the older example. When `data/series.npy` is a 1D or 2D numeric array, the notebook writes `outputs/series.png`. Both paths are gitignored. That figure is the local file. `evolve` stays a scaffold and is not a plot of the dynamics.
