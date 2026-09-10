import subprocess
from pathlib import Path

from autodev.git import GitRepository


def command(path: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=path, check=True, capture_output=True, text=True)


def test_git_repository_creates_checkpoint_and_returns_revision(tmp_path: Path) -> None:
    command(tmp_path, "init")
    command(tmp_path, "config", "user.email", "test@example.com")
    command(tmp_path, "config", "user.name", "Test User")
    (tmp_path / "README.md").write_text("initial\n", encoding="utf-8")
    command(tmp_path, "add", "README.md")
    command(tmp_path, "commit", "-m", "initial")
    (tmp_path / "README.md").write_text("changed\n", encoding="utf-8")

    revision = GitRepository(tmp_path).checkpoint("feat: change readme")

    assert len(revision) == 40
    assert GitRepository(tmp_path).status() == ""


def test_git_repository_restores_a_tracked_path(tmp_path: Path) -> None:
    command(tmp_path, "init")
    command(tmp_path, "config", "user.email", "test@example.com")
    command(tmp_path, "config", "user.name", "Test User")
    path = tmp_path / "settings.txt"
    path.write_text("stable\n", encoding="utf-8")
    command(tmp_path, "add", "settings.txt")
    command(tmp_path, "commit", "-m", "initial")
    path.write_text("broken\n", encoding="utf-8")

    GitRepository(tmp_path).restore("settings.txt")

    assert path.read_text(encoding="utf-8") == "stable\n"
