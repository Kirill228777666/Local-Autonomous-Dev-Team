"""Policy-controlled tools executed on behalf of LLM role agents."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


class ToolPolicyError(PermissionError):
    """An agent proposed an operation outside the local tool policy."""


@dataclass(frozen=True, slots=True)
class CommandResult:
    exit_code: int
    stdout: str
    stderr: str


class WorkspaceTools:
    """Filesystem and process operations constrained to one workspace."""

    _BLOCKED_COMMANDS = frozenset(
        {
            "del",
            "erase",
            "format",
            "remove-item",
            "rmdir",
            "rd",
            "shutdown",
            "diskpart",
            "reg",
        }
    )

    def __init__(self, workspace: Path, command_timeout: float = 120.0) -> None:
        self.workspace = workspace.resolve()
        self.command_timeout = command_timeout

    def _path(self, relative_path: str) -> Path:
        path = (self.workspace / relative_path).resolve()
        if not path.is_relative_to(self.workspace):
            raise ToolPolicyError("path must stay inside the workspace")
        return path

    def list_files(self) -> list[str]:
        excluded = {".git", ".autodev", ".pytest-tmp", "__pycache__"}
        return sorted(
            str(path.relative_to(self.workspace)).replace("\\", "/")
            for path in self.workspace.rglob("*")
            if path.is_file() and not any(part in excluded for part in path.relative_to(self.workspace).parts)
        )

    def read_file(self, relative_path: str) -> str:
        return self._path(relative_path).read_text(encoding="utf-8")

    def write_file(self, relative_path: str, content: str) -> None:
        path = self._path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def edit_file(self, relative_path: str, old: str, new: str) -> None:
        path = self._path(relative_path)
        content = path.read_text(encoding="utf-8")
        if old not in content:
            raise ValueError("edit target was not found")
        path.write_text(content.replace(old, new, 1), encoding="utf-8")

    def delete_file(self, relative_path: str) -> None:
        path = self._path(relative_path)
        if path.is_dir():
            raise ToolPolicyError("directory deletion is not permitted")
        path.unlink(missing_ok=True)

    def run_command(self, command: list[str]) -> CommandResult:
        if not command or not command[0].strip():
            raise ToolPolicyError("command must not be empty")
        executable = Path(command[0]).name.lower()
        if executable in self._BLOCKED_COMMANDS:
            raise ToolPolicyError(f"command '{command[0]}' is not permitted")
        completed = subprocess.run(
            command,
            cwd=self.workspace,
            capture_output=True,
            text=True,
            shell=False,
            timeout=self.command_timeout,
            check=False,
        )
        return CommandResult(completed.returncode, completed.stdout, completed.stderr)

    def run_tests(self, command: list[str]) -> CommandResult:
        return self.run_command(command)
