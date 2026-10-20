"""Hydra routes every run to outputs/<experiment_name>/<timestamp>/."""

import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

from QELM.NQS.paths import ExperimentPaths, experiment_paths

REPO_ROOT = Path(__file__).resolve().parents[3]
TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")


def _run(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["HYDRA_FULL_ERROR"] = "1"
    return subprocess.run(
        [sys.executable, *args],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def _only_run(root: Path, experiment: str) -> Path:
    runs = list((root / "outputs" / experiment).iterdir())
    assert len(runs) == 1, runs
    return runs[0]


def test_experiment_paths_layout(tmp_path: Path) -> None:
    paths = experiment_paths(tmp_path / "run")
    assert paths == ExperimentPaths.at(tmp_path / "run")
    for directory in (paths.exp, paths.logs, paths.plots):
        assert directory.is_dir()
        assert directory.parent == paths.root


def test_experiment_paths_fallback_never_reuses_a_run(tmp_path: Path) -> None:
    a = experiment_paths(experiment_name="study", outputs_root=tmp_path)
    b = experiment_paths(experiment_name="study", outputs_root=tmp_path)
    assert a.root.parent == b.root.parent == tmp_path / "study"
    assert a.root != b.root


def test_main_creates_experiment_structure(tmp_path: Path) -> None:
    result = _run(
        [str(REPO_ROOT / "main.py"), "experiment_name=my_exp", "ising.N=4"],
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr

    run = _only_run(tmp_path, "my_exp")
    assert TIMESTAMP.match(run.name), run.name
    paths = ExperimentPaths.at(run)
    assert (paths.exp / "checkpoint.npz").is_file()
    assert (paths.exp / "nqs_variables.msgpack").is_file()
    assert (paths.logs / "main.log").is_file()
    assert paths.plots.is_dir()
    assert (run / ".hydra" / "config.yaml").is_file()
    assert "hilbert_dim=16" in result.stdout


def test_plots_inside_a_hydra_job_land_next_to_its_logs(tmp_path: Path) -> None:
    script = tmp_path / "plot_job.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import hydra
            import numpy as np

            from QELM.NQS.utils.makeplots import plot_time_series


            @hydra.main(
                version_base="1.3",
                config_path={str(REPO_ROOT / "conf")!r},
                config_name="config",
            )
            def job(cfg):
                t = np.linspace(0.0, 1.0, 5)
                print(plot_time_series(t, {{"x": t}}, filename="x.png"))


            job()
            """
        )
    )
    result = _run([str(script), "experiment_name=plots_exp"], cwd=tmp_path)
    assert result.returncode == 0, result.stderr

    run = _only_run(tmp_path, "plots_exp")
    paths = ExperimentPaths.at(run)
    assert (paths.plots / "x.png").is_file()
    assert paths.plots.parent == paths.logs.parent
    assert (paths.logs / "plot_job.log").is_file()
