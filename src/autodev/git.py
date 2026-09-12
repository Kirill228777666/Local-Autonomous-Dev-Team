"""Small Git adapter used for verified autonomous checkpoints."""

from __future__ import annotations

import subprocess
from pathlib import Path


class GitError(RuntimeError):
    pass


class GitRepository:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.resolve()

    def _run(self, *args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=self.workspace,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if completed.returncode:
            raise GitError(completed.stderr.strip() or completed.stdout.strip())
        return completed.stdout

    def status(self) -> str:
        return self._run("status", "--short")

    def diff(self) -> str:
        return self._run("diff", "--no-ext-diff")

    def checkpoint(self, message: str) -> str:
        if not message.strip():
            raise ValueError("commit message must not be blank")
        self._run("add", "-A")
        if self.status():
            self._run("commit", "-m", message)
        return self._run("rev-parse", "HEAD").strip()

    def restore(self, relative_path: str) -> None:
        target = (self.workspace / relative_path).resolve()
        if not target.is_relative_to(self.workspace):
            raise ValueError("restore target must stay inside workspace")
        self._run("restore", "--", relative_path)
