"""Read a composed Hydra config. Values come from YAML and CLI overrides."""

from __future__ import annotations

from dotenv import load_dotenv
from omegaconf import DictConfig


def load_secrets() -> None:
    """Load secrets from a local ``.env`` file.

    Missing files are ignored. Scientific parameters stay in ``conf/``.
    """
    load_dotenv()


def format_selection(cfg: DictConfig) -> str:
    """Format the chosen simulation and test values.

    Args:
        cfg: Composed Hydra config. The active simulation block is
            ``cfg[cfg.sim_type]``. Each exposed value is a scalar
            (``N``, ``J``, ``learning_rate``, ``dt``, ``n_steps``).

    Returns:
        One line of ``key=value`` pairs, starting with ``sim_type``.
    """
    sim_type = str(cfg.sim_type)
    sim = cfg[sim_type]
    parts = [f"sim_type={sim_type}"]
    parts.extend(f"{sim_type}.{key}={value}" for key, value in sim.items())
    parts.extend(f"test.{key}={value}" for key, value in cfg.test.items())
    return " ".join(parts)
