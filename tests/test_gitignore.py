"""The scratch sandbox (issue #25) stays local: everything in it is ignored
except the .gitkeep that makes the folder exist in every clone."""

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRATCH = "notebooks/scratch"


def _is_ignored(path: str) -> bool:
    # --no-index checks the rules even for files that are not on disk.
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", "--no-index", path],
        cwd=REPO_ROOT,
        check=False,
    )
    assert result.returncode in (0, 1), result
    return result.returncode == 0


@pytest.fixture(autouse=True)
def _require_git() -> None:
    if shutil.which("git") is None or not (REPO_ROOT / ".git").exists():
        pytest.skip("needs a git checkout")


def test_scratch_folder_is_tracked_through_gitkeep() -> None:
    assert (REPO_ROOT / SCRATCH / ".gitkeep").is_file()
    assert not _is_ignored(f"{SCRATCH}/.gitkeep")


@pytest.mark.parametrize(
    "name",
    ["draft.ipynb", "try_tdvp.py", "nested/run.ipynb", "result.png"],
)
def test_scratch_contents_are_ignored(name: str) -> None:
    assert _is_ignored(f"{SCRATCH}/{name}")


def test_other_notebooks_are_not_ignored() -> None:
    assert not _is_ignored("notebooks/series.ipynb")
