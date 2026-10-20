# Documentation

Notes for this repository. Scientific parameters stay in Hydra YAML under `conf/`. Secrets stay in `.env`. The library is `QRC.NQS` (`src/QRC/NQS/`). NetKet 3 builds the Hilbert space, the transverse-field Ising Hamiltonian and the RBM variational state. Training and evaluation are still scaffolds: they route a config and record a status string.

- [Setup](setup.md) — install uv and sync the project
- [Configuration](configuration.md) — Hydra YAML versus `.env` secrets
- [Layout](layout.md) — `conf/`, `models/`, `variational/`, `utils/`, `training/`, `evaluation/`, and `main.py`
- [Running](running.md) — command-line overrides and the series notebook
- [Checkpoints and local data](checkpoints.md) — the `outputs/<experiment>/<timestamp>/{exp,logs,plots}` layout, `data/`, and `wandb/`
- [Dynamics](dynamics.md) — exact evolution, TDVP settings, and the TDVP-vs-ED baseline
- [VMC](vmc.md) — ground-state preparation, defaults and outputs
- [QGT](qgt.md) — Fubini-Study metric, Berry curvature, rank and conditioning
- [Hidden representations](representations.md) — SVD of the RBM weights
- [Code quality](code-quality.md) — Ruff, Mypy, pytest, and the seed from config
