# QRC-enhanced-by-NQS

Neural quantum states (NetKet 3) as a probe for quantum reservoir computing (QRC) and quantum extreme learning machines (QELM). The library is `QRC.NQS`, managed with [uv](https://docs.astral.sh/uv/). Requires Python 3.12 or newer.

Longer notes are in [`documentation/`](documentation/README.md): setup, Hydra and `.env`, the project layout, checkpoints, and code quality.

## Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Other install options are in the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

## Set up

```bash
uv sync
```

## Layout

| Path | Role |
| --- | --- |
| `conf/` | Hydra YAML: `config.yaml` plus groups `sim_type`, `nqs`, `training`, `evaluation` |
| `data/` | Local time series and training datasets (gitignored) |
| `src/QRC/NQS/models/` | Hamiltonians (Ising via NetKet, Kitaev scaffold) and time evolution |
| `src/QRC/NQS/variational/` | NetKet 3 systems, RBM ansatz, variational states, `build_nqs(cfg)` |
| `src/QRC/NQS/measurements/` | All-to-all reservoir Hamiltonian and observables |
| `src/QRC/NQS/dynamics/` | Exact evolution, TDVP, TDVP-vs-ED baseline |
| `src/QRC/NQS/embedding/` | Ground-state preparation by VMC |
| `src/QRC/NQS/utils/makeplots.py` | All figures; written to the run's `plots/` folder |
| `src/QRC/NQS/training/` | Training algorithms (Readout, Ridge scaffolds) |
| `src/QRC/NQS/evaluation/` | Metrics and plot scaffolds |
| `notebooks/00_outputs_starter.ipynb` | Load the latest run of an experiment from `outputs/` |
| `notebooks/01_tdvp_vs_ed_baseline.ipynb` | TDVP (RBM) vs exact dynamics baseline |
| `notebooks/series.ipynb` | Plot a local series from `data/`; the library call stays a scaffold |
| `main.py` | Entrypoint. Routes sim type → model → training → evaluation → NetKet setup |
| `outputs/` | Hydra runs, `outputs/<experiment_name>/<timestamp>/{exp,logs,plots}` (gitignored) |

## Run the pipeline

Scientific parameters live in `conf/` and can be overridden from the command line. Secrets and machine-local paths (for example `WANDB_API_KEY`) belong in `.env`, not in YAML. Copy `.env.example` to `.env` for local secrets.

```bash
cp .env.example .env
uv run python main.py sim_type=ising ising.N=12 test.dt=0.05
```

`sim_type=ising` selects `conf/sim_type/ising.yaml` and the Ising model. `training=ridge` (default) or `training=readout` selects the trainer. `N`, `J`, `learning_rate`, and `dt` stay in YAML or CLI overrides.

## Notebook

From the repository root:

```bash
uv sync
```

Then open `notebooks/series.ipynb`. `uv sync` installs `ipykernel` and Matplotlib from the dev group, so the kernel can use `.venv`.

The notebook reads `data/series.npy` (a 1D or 2D numeric array) and, when that file is present, writes `outputs/series.png`. Both directories are gitignored. The figure is the local file, not the Ising trajectory.

## Data and checkpoints

Code is committed. Large data is not. `.gitignore` excludes `data/`, `outputs/`, and `wandb/`, plus payload formats such as `*.h5`, `*.csv`, and `*.npy`. Create `data/` locally for datasets; that folder is not committed.

Each run saves a checkpoint at the end of training. `uv run python main.py experiment_name=<name>` writes every run to `outputs/<name>/<YYYY-MM-DD_HH-MM-SS>/`, with raw data in `exp/` (`checkpoint.npz`, `nqs_variables.msgpack`), the log in `logs/` and figures in `plots/`. Without a Hydra job, the file is `<checkpoint.fallback_root>/<uuid>/checkpoint.npz` (`outputs/<uuid>/checkpoint.npz` by default). Set `checkpoint.dir` to choose the directory, and `checkpoint.use_uuid=true` to nest a UUID under it.

DVC can be added later if large HDF5 or CSV files need strict versioning.

## Run tests

```bash
uv run pytest
```

## Format, lint, and types

```bash
uv run ruff format
uv run ruff check
uv run ruff format --check
uv run mypy
```

Ruff formats and lints. Mypy typechecks the package and `main.py`. At startup, `main.py` seeds NumPy from `seed` in `conf/config.yaml` (default `42`). Override it with `seed=0`.

## Develop

Package code lives in `src/QRC/NQS`. Import it as `QRC.NQS`. Add a dependency with `uv add`.
