import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from QELM.NQS.seed import set_seed
from QELM.NQS.selection import format_selection

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "conf"
OVERRIDES = ["sim_type=ising", "ising.N=12", "test.dt=0.05"]


def setup_function() -> None:
    GlobalHydra.instance().clear()


def teardown_function() -> None:
    GlobalHydra.instance().clear()


def _compose(overrides: list[str] | None = None):
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        return compose(config_name="config", overrides=overrides)


def test_default_seed_is_42() -> None:
    assert int(_compose().seed) == 42


def test_set_seed_uses_config_value() -> None:
    assert set_seed(_compose(["seed=7"])) == 7
    sample = np.random.random()
    np.random.seed(7)
    assert sample == np.random.random()


def test_config_loads_and_cli_overrides_apply() -> None:
    default = _compose()
    chosen = _compose(OVERRIDES)

    assert set(chosen.ising.keys()) == {"N", "J", "h", "pbc", "learning_rate"}
    assert set(chosen.test.keys()) == {"dt", "n_steps"}
    assert chosen.sim_type == "ising"
    assert chosen.ising.N == 12
    assert chosen.test.dt == 0.05
    assert chosen.ising.N != default.ising.N
    assert chosen.test.dt != default.test.dt
    assert chosen.ising.J == default.ising.J
    assert chosen.ising.learning_rate == default.ising.learning_rate
    assert chosen.test.n_steps == default.test.n_steps

    text = format_selection(chosen)
    assert "sim_type=ising" in text
    assert "ising.N=12" in text
    assert "test.dt=0.05" in text


def test_main_cli_override(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["HYDRA_FULL_ERROR"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "main.py"),
            *OVERRIDES,
            f"hydra.run.dir={tmp_path}",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert "ising.N=12" in result.stdout
    assert "test.dt=0.05" in result.stdout
    assert (tmp_path / "exp" / "checkpoint.npz").is_file()
    assert (tmp_path / "exp" / "nqs_variables.msgpack").is_file()
    assert (tmp_path / "logs" / "main.log").is_file()
