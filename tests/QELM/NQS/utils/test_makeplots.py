"""Plot functions run headless and write where they are told."""

from pathlib import Path

import numpy as np
import pytest

from QELM.NQS.utils import makeplots

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _is_png(path: Path) -> bool:
    return path.is_file() and path.read_bytes()[:8] == PNG_MAGIC


def test_plot_time_series_writes_png(tmp_path: Path) -> None:
    t = np.linspace(0.0, 1.0, 11)
    path = makeplots.plot_time_series(
        t,
        {"exact": np.cos(t), "tdvp": np.cos(t) + 1e-4},
        filename="mag.png",
        directory=tmp_path / "plots",
        ylabel=r"$\langle Z \rangle$",
        title="test",
    )
    assert path == tmp_path / "plots" / "mag.png"
    assert _is_png(path)


def test_plot_convergence_writes_log_scale_figure(tmp_path: Path) -> None:
    iters = np.arange(5)
    path = makeplots.plot_convergence(
        iters,
        10.0 ** (-iters.astype(float)),
        filename="variance.pdf",
        directory=tmp_path,
        log_y=True,
        reference=1e-3,
    )
    assert path.is_file()
    assert path.read_bytes()[:4] == b"%PDF"


def test_plot_inputs_are_validated(tmp_path: Path) -> None:
    t = np.arange(4.0)
    with pytest.raises(ValueError):
        makeplots.plot_time_series(t, {}, filename="a.png", directory=tmp_path)
    with pytest.raises(ValueError):
        makeplots.plot_time_series(
            t, {"x": np.arange(3.0)}, filename="a.png", directory=tmp_path
        )
    with pytest.raises(ValueError):
        makeplots.plot_time_series(
            t, {"x": t + 1j}, filename="a.png", directory=tmp_path
        )
    with pytest.raises(ValueError):
        makeplots.plot_convergence(
            t, t, filename="a.png", directory=tmp_path, log_y=True
        )
    with pytest.raises(ValueError):
        makeplots.plot_time_series(
            t, {"x": t}, filename="sub/a.png", directory=tmp_path
        )
    assert not any(tmp_path.iterdir())


def test_default_directory_is_a_fresh_run_outside_hydra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    t = np.arange(3.0)
    path = makeplots.plot_time_series(t, {"x": t}, filename="a.png")
    # outputs/default/<timestamp>_<id>/plots/a.png, next to logs/ of the same run.
    assert path.parts[:2] == ("outputs", "default")
    assert (tmp_path / path).is_file()
    assert path.parent.name == "plots"
    assert (path.parent.parent / "logs").is_dir()
    assert _is_png(path)


def test_plot_exact_vs_tdvp(tmp_path: Path) -> None:
    t = np.linspace(0.0, 1.0, 21)
    same = makeplots.plot_exact_vs_tdvp(
        t, np.cos(t), t, np.cos(t) + 1e-5, filename="same.png", directory=tmp_path
    )
    other = makeplots.plot_exact_vs_tdvp(
        t, np.cos(t), t[::2], np.cos(t[::2]), filename="other.png", directory=tmp_path
    )
    assert _is_png(same)
    assert _is_png(other)
    with pytest.raises(ValueError):
        makeplots.plot_exact_vs_tdvp(
            t, np.cos(t), t, np.cos(t) + 0j, filename="c.png", directory=tmp_path
        )
    with pytest.raises(ValueError):
        makeplots.plot_exact_vs_tdvp(
            t, np.cos(t[:-1]), t, np.cos(t), filename="s.png", directory=tmp_path
        )
