"""Pipeline entrypoint.

Scientific parameters come from ``conf/`` YAML and CLI overrides.
``main`` seeds NumPy from ``cfg.seed``, creates the run folders
``outputs/<experiment_name>/<timestamp>/{exp,logs,plots}``, then routes sim
type to a model, training, evaluation and the NetKet setup, and writes a
checkpoint. Secrets (API keys, local paths) are loaded from ``.env`` when
present.

Run from the repository root:

    uv run python main.py sim_type=ising ising.N=12 test.dt=0.05
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from QRC.NQS.paths import experiment_paths
from QRC.NQS.pipeline import run_pipeline
from QRC.NQS.seed import set_seed
from QRC.NQS.selection import format_selection, load_secrets


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Seed NumPy, create the run folders, load secrets, and run the pipeline.

    Args:
        cfg: Hydra config composed from ``conf/`` and CLI overrides.
            ``cfg.seed`` is a scalar integer (default 42) applied to NumPy
            before the pipeline runs.
    """
    set_seed(cfg)
    paths = experiment_paths()
    load_secrets()
    print(f"run_dir={paths.root}")
    print(format_selection(cfg))
    print(run_pipeline(cfg))


if __name__ == "__main__":
    main()
