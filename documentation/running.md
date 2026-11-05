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
uv run python main.py checkpoint.dir=/tmp/qelm-run checkpoint.use_uuid=true
```

`training=ridge` and `evaluation=default` match the defaults in `conf/config.yaml`. `training=readout` selects `Readout`. `evaluation=default` is the only registered evaluator. An unknown `sim_type`, trainer name, or evaluator name raises `KeyError`.

## Notebooks

From the repository root, run `uv sync`. Notebooks are split in two folders:

- `notebooks/results/` holds physical results and experiments. `00_outputs_starter.ipynb` finds the repository root by itself and loads the latest run of an experiment from `outputs/` (checkpoint metadata and the figures in `plots/`). `01_tdvp_vs_ed_baseline.ipynb` compares TDVP with exact dynamics, see [dynamics.md](dynamics.md).
- `notebooks/benchmarks/` holds performance benchmarks. `01_tdvp_wall_time.ipynb` reports the TDVP wall time against N and h.

Run `notebooks/results/01_tdvp_vs_ed_baseline.ipynb` before the benchmark: both read the field scan cached in `outputs/tdvp_vs_ed_error_scan/`.
