# Configuration

Scientific parameters and secrets live in different files.

## Hydra YAML

`main.py` loads `conf/config.yaml` with Hydra (`version_base="1.3"`). Command-line overrides are applied on top. `hydra.job.chdir` is `false`, so the process stays in the repository root.

`conf/config.yaml` selects the groups and holds the shared settings:

| Key | Role |
| --- | --- |
| `defaults` | `sim_type: ising`, `training: ridge`, `evaluation: default`, `nqs: rbm`, `vmc: default`, and the `hydra/job_logging: qrc` override |
| `experiment_name` | Folder under `outputs/`. Default `default` |
| `seed` | Integer passed to NumPy at startup. See [Code quality](code-quality.md). |
| `test.dt`, `test.n_steps` | Scalar step size and step count passed to `evolve` |
| `checkpoint` | Where the run writes its `.npz`. See [Checkpoints and local data](checkpoints.md). |
| `hydra.run.dir` | `outputs/${experiment_name}/${now:%Y-%m-%d_%H-%M-%S}` |
| `hydra.sweep.dir` | `outputs/${experiment_name}/multirun_${now:%Y-%m-%d_%H-%M-%S}`, one subfolder per job |

Group files:

| File | Contents |
| --- | --- |
| `conf/sim_type/ising.yaml` | `sim_type: ising` and the `ising` block: `N`, `J`, `h`, `pbc`, `learning_rate`. Convention `H = -h sum X_i + J sum Z_i Z_j` |
| `conf/vmc/default.yaml` | VMC ground-state preparation: `learning_rate`, `n_iter`, `n_samples`, `alpha`, `param_dtype`, `diag_shift` |
| `conf/nqs/rbm.yaml` | The ansatz: `model: rbm`, `alpha`, `param_dtype` (`complex` or `real`), `state` (`full_sum` or `mc`), `n_samples` |
| `conf/hydra/job_logging/qrc.yaml` | Hydra's default logging, with the log file moved to `<run>/logs/` |
| `conf/sim_type/kitaev.yaml` | `sim_type: kitaev` and an empty `kitaev` block. No Hamiltonian parameters are defined there |
| `conf/training/ridge.yaml` | `name: ridge` |
| `conf/training/readout.yaml` | `name: readout` |
| `conf/evaluation/default.yaml` | `name: default`, a list of metric names, and a plot name |

`N`, `J`, `learning_rate`, `dt`, and `seed` are config values. They are not constants in the Python package. Change them in YAML or with a CLI override. See [Running](running.md).

## `.env` secrets

API keys and machine-local paths belong in `.env`. That file is gitignored. Copy the example and fill it in locally:

```bash
cp .env.example .env
```

`.env.example` only lists `WANDB_API_KEY`. `main.py` calls `load_secrets()`, which runs `load_dotenv()`. A missing `.env` is ignored. The pipeline does not read the key and does not upload a run.
