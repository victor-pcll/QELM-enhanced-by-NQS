"""Checkpoint writes stay on the local filesystem."""

import json
import uuid
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig

from QELM.NQS.checkpoint import save_checkpoint
from QELM.NQS.models import IsingModel
from QELM.NQS.training import RidgeRegression

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "conf"


def setup_function() -> None:
    GlobalHydra.instance().clear()


def teardown_function() -> None:
    GlobalHydra.instance().clear()


def _compose(overrides: list[str] | None = None):
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        return compose(config_name="config", overrides=overrides)


def _save(cfg):
    model = IsingModel.from_config(cfg)
    trainer = RidgeRegression.from_config(cfg)
    evolution = model.evolve(dt=float(cfg.test.dt), n_steps=int(cfg.test.n_steps))
    return save_checkpoint(
        cfg,
        model=model,
        trainer=trainer,
        evolution=evolution,
        training_status=trainer.fit(features=None, targets=None),
    )


def test_checkpoint_writer_creates_file_in_tmp_path(tmp_path: Path) -> None:
    cfg = _compose(
        [
            f"checkpoint.dir={tmp_path}",
            "checkpoint.use_uuid=false",
            "checkpoint.filename=weights.npz",
        ]
    )
    path = _save(cfg)

    assert path == tmp_path / "weights.npz"
    assert path.is_file()
    with np.load(path) as archive:
        assert archive["weights"].dtype == np.float64
        assert archive["weights"].shape == (0,)
        metadata = json.loads(str(archive["metadata"].item()))
    assert metadata["seed"] == int(cfg.seed)
    assert metadata["sim_type"] == str(cfg.sim_type)
    assert metadata["params"] == {key: cfg.ising[key] for key in cfg.ising}
    assert metadata["evolution"]["dt"] == float(cfg.test.dt)
    assert metadata["evolution"]["n_steps"] == int(cfg.test.n_steps)
    assert "wandb" not in metadata
    uuid.UUID(metadata["run_id"])


def test_checkpoint_fallback_uses_config_uuid_directory(tmp_path: Path) -> None:
    assert not HydraConfig.initialized()
    cfg = _compose(
        [
            "checkpoint.dir=null",
            f"checkpoint.fallback_root={tmp_path}",
            "checkpoint.use_uuid=false",
        ]
    )
    path = _save(cfg)

    assert path.is_file()
    assert path.parent.parent == tmp_path
    assert path.name == str(cfg.checkpoint.filename)
    uuid.UUID(path.parent.name)


def test_checkpoint_use_uuid_nests_under_configured_dir(tmp_path: Path) -> None:
    cfg = _compose(
        [
            f"checkpoint.dir={tmp_path}",
            "checkpoint.use_uuid=true",
        ]
    )
    path = _save(cfg)

    assert path.is_file()
    assert path.parent.parent == tmp_path
    assert path.name == str(cfg.checkpoint.filename)
    uuid.UUID(path.parent.name)
