"""Route a Hydra config to model, training, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import flax.serialization
from omegaconf import DictConfig

from QRC.NQS.checkpoint import save_checkpoint
from QRC.NQS.evaluation import Evaluator, get_evaluator
from QRC.NQS.models import HamiltonianModel, get_model
from QRC.NQS.training import Trainer, get_trainer
from QRC.NQS.variational import NQSSetup, build_nqs

NQS_PARAMETERS_FILENAME = "nqs_variables.msgpack"


@dataclass(frozen=True)
class PipelinePlan:
    """Classes and callables selected for one config.

    Attributes:
        sim_type: Config key, such as ``"ising"`` or ``"kitaev"``.
        model: Hamiltonian class for that simulation.
        trainer: Training class named by ``cfg.training.name``.
        evaluator: Callable named by ``cfg.evaluation.name``.
    """

    sim_type: str
    model: type[HamiltonianModel]
    trainer: type[Trainer]
    evaluator: Evaluator


def resolve_pipeline(cfg: DictConfig) -> PipelinePlan:
    """Map config names to a model, trainer, and evaluator.

    Args:
        cfg: Composed Hydra config. ``cfg.sim_type``, ``cfg.training.name``,
            and ``cfg.evaluation.name`` are scalar strings.

    Returns:
        The selected classes and evaluator. Nothing is instantiated.
    """
    sim_type = str(cfg.sim_type)
    return PipelinePlan(
        sim_type=sim_type,
        model=get_model(sim_type),
        trainer=get_trainer(str(cfg.training.name)),
        evaluator=get_evaluator(str(cfg.evaluation.name)),
    )


def run_pipeline(cfg: DictConfig) -> str:
    """Build the selected blocks and return a scaffold status line.

    Args:
        cfg: Composed Hydra config. ``cfg.test.dt`` is a scalar step size
            and ``cfg.test.n_steps`` is a scalar step count. ``N``, ``J``,
            and ``learning_rate`` stay on the simulation block.

    Returns:
        One status line naming the model, trainer, ``dt``, ``n_steps``,
        the evaluator report, the NetKet setup and the checkpoint path. No
        trajectory tensor is allocated. The checkpoint directory comes from
        ``cfg.checkpoint``; the initial NQS variables are written next to it.
    """
    plan = resolve_pipeline(cfg)
    model = plan.model.from_config(cfg)
    trainer = plan.trainer.from_config(cfg)
    evolution = model.evolve(dt=float(cfg.test.dt), n_steps=int(cfg.test.n_steps))
    training_status = trainer.fit(features=None, targets=None)
    report = plan.evaluator(
        evolution=evolution,
        training_status=training_status,
        cfg=cfg,
    )
    setup = _try_build_nqs(cfg)
    saved = save_checkpoint(
        cfg,
        model=model,
        trainer=trainer,
        evolution=evolution,
        training_status=training_status,
    )
    nqs_report = "nqs=unavailable"
    if setup is not None:
        variables = save_nqs_variables(setup, saved.parent)
        nqs_report = f"{describe_nqs(setup)} nqs_variables={variables}"
    return (
        f"model={model.name} training={trainer.name} "
        f"dt={evolution['dt']} n_steps={evolution['n_steps']} {report} "
        f"{nqs_report} checkpoint={saved}"
    )


def _try_build_nqs(cfg: DictConfig) -> NQSSetup | None:
    try:
        return build_nqs(cfg)
    except NotImplementedError:
        return None


def describe_nqs(setup: NQSSetup) -> str:
    """Summarize a NetKet setup and the variational energy of its state.

    Args:
        setup: NetKet objects of the run.

    Returns:
        A ``key=value`` line with the Hilbert dimension, the parameter
        count and ``<H>`` of the (untrained) state. ``<H>`` is real because
        the Hamiltonian is Hermitian.
    """
    energy = setup.state.expect(setup.hamiltonian)
    return (
        f"nqs={type(setup.model).__name__} hilbert_dim={setup.hilbert.n_states} "
        f"n_params={setup.state.n_parameters} "
        f"energy={float(energy.mean.real):.6f}"
    )


def save_nqs_variables(setup: NQSSetup, directory: Path) -> Path:
    """Write the Flax variables of the variational state (msgpack).

    Reload with ``flax.serialization.from_bytes(state.variables, data)``.

    Args:
        setup: NetKet objects of the run.
        directory: Existing folder, the ``exp/`` folder of the run.

    Returns:
        Path of the written file.
    """
    path = directory / NQS_PARAMETERS_FILENAME
    path.write_bytes(flax.serialization.to_bytes(setup.state.variables))
    return path


__all__ = [
    "NQS_PARAMETERS_FILENAME",
    "PipelinePlan",
    "describe_nqs",
    "resolve_pipeline",
    "run_pipeline",
    "save_nqs_variables",
]
