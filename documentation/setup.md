# Setup

The project requires Python 3.12 or newer. That pin is `.python-version` and `requires-python` in `pyproject.toml`. Dependencies are managed with [uv](https://docs.astral.sh/uv/).

## Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Other options are in the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

## Sync

From the repository root:

```bash
uv sync
```

`uv sync` creates `.venv`, installs the package, and installs the `dev` group (Ruff, Mypy, pytest, ipykernel, and Matplotlib). Run project commands with `uv run` so they use that environment. The series notebook uses that kernel; see [Running](running.md).

Add a dependency with `uv add`.

## Warning: `VIRTUAL_ENV ... does not match the project environment path`

```
warning: `VIRTUAL_ENV=/.../QRC MIPT/.venv` does not match the project environment path `.venv` and will be ignored; use `--active` to target the active environment instead
```

The shell still has a virtual environment activated from another folder, here the old `QRC MIPT` checkout. `uv run` ignores it and uses this repository's `.venv`, so commands still run in the right environment: the warning is harmless. Do not add `--active`, which would run the project in the old environment.

To remove the warning, leave the old environment and resync in the repository root:

```bash
deactivate        # or: unset VIRTUAL_ENV
uv sync
```

If you work with an activated environment (for instance for a notebook kernel), activate this repository's one: `source .venv/bin/activate`. After moving or renaming the repository folder, delete `.venv` and run `uv sync` again, since its scripts store absolute paths to the old folder.
