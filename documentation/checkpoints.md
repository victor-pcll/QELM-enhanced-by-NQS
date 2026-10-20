# Checkpoints and local data

Code is committed. Datasets and run artifacts are not.

## Ignored paths

`.gitignore` excludes:

- `data/` — local time series and training datasets. Create it yourself if you need it. The package does not write there.
- `outputs/` — Hydra run directories and checkpoints.
- `wandb/` — local Weights & Biases files. The pipeline does not create this directory.

`.env` is ignored. `results/` is ignored. Payload suffixes are ignored too, including `*.h5`, `*.hdf5`, `*.csv`, `*.npy`, and `*.npz`.

DVC is not set up.

## Checkpoint path

`run_pipeline` saves one archive at the end of the run. `cfg.checkpoint` in `conf/config.yaml` chooses the path:

| Key | When `dir` is unset |
| --- | --- |
| `checkpoint.dir` | `null` means "do not use a fixed directory" |
| `checkpoint.fallback_root` | `outputs`. Used only when `dir` is unset and no Hydra job is active |
| `checkpoint.use_uuid` | `false`. When `true`, a UUID directory is nested under the chosen directory |
| `checkpoint.filename` | `checkpoint.npz`. This must be a file name, not a path |

`uv run python main.py` starts a Hydra job, so the file goes to the `exp/` folder of the run directory from `hydra.run.dir`:

```text
outputs/<experiment_name>/<YYYY-MM-DD_HH-MM-SS>/
├── .hydra/                      resolved config of the run
├── exp/
│   ├── checkpoint.npz
│   └── nqs_variables.msgpack    Flax variables of the variational state
├── logs/
│   └── main.log                 Hydra job log
└── plots/                       figures from QRC.NQS.utils.makeplots
```

`main.py` creates `exp/`, `logs/` and `plots/` at startup. Choose the experiment folder with `experiment_name=my_study`. Reload the NQS variables with `flax.serialization.from_bytes(state.variables, path.read_bytes())`.

If nothing has initialized Hydra and `checkpoint.dir` is unset, the file is:

```text
outputs/<uuid>/checkpoint.npz
```

`outputs` there is `checkpoint.fallback_root`. Set `checkpoint.dir` to pick the directory. Set `checkpoint.use_uuid=true` to nest a UUID under that directory (or under the run's `exp/` folder, when `dir` is still unset).

## Archive contents

The file is a NumPy `.npz` with two entries:

- `weights` — `float64` array from the trainer. The scaffold stores shape `(0,)` because `fit` does not estimate weights.
- `metadata` — JSON text. Fields are `run_id`, `sim_type`, `model`, `trainer`, `seed`, `training_status`, `learning_rate`, `evolution` (`dt`, `n_steps`, `status`), and `params` (the simulation block).

Nothing is uploaded.
