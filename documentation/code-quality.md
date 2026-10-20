# Code quality

Ruff, Mypy, and pytest are in the `dev` group installed by `uv sync`. From the repository root:

```bash
uv run ruff check
uv run ruff format --check
uv run ruff format
uv run mypy
uv run pytest
```

Ruff lints and formats. `pyproject.toml` sets the target to Python 3.12, a line length of 88, and the rule sets `E`, `F`, `I`, and `UP`. `ruff check` reports lint findings. `ruff format --check` reports format drift and does not rewrite files. `ruff format` rewrites them.

Mypy typechecks `src/QELM/NQS` and `main.py` with `disallow_untyped_defs`. It does not typecheck `tests/`.

pytest collects `tests/` with `--import-mode=importlib`, so `tests/QELM/NQS/` can mirror the package layout and reuse file names. Physics tests there compare NetKet operators against Kronecker-product constructions, the exact Jordan-Wigner ground energy of the periodic Ising chain, and the variational principle `E0 <= <H> <= E_max`.

NetKet ships no type information, so `pyproject.toml` tells Mypy to treat `netket` as untyped.

GitHub Actions runs the four checks on every push and pull request: `uv run ruff check`, `uv run ruff format --check`, `uv run mypy`, and `uv run pytest`. The workflow is `.github/workflows/code-quality.yml`. It installs uv, installs the Python version in `.python-version`, syncs from `uv.lock`, and runs those commands. It does not start extra services.

## Seed

`main.py` calls `set_seed` before the pipeline. That function reads the integer `seed` from the composed config (`seed` in `conf/config.yaml`, overridable on the command line) and passes it to `numpy.random.seed`. JAX has no global random state, so the same `seed` is passed explicitly to NetKet for the parameter initialization and the Monte Carlo sampler (`build_nqs`). The same value is copied into checkpoint metadata.

```bash
uv run python main.py seed=0
```

`seed.py` does not embed a seed of its own.
