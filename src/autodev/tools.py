"""Policy-controlled tools executed on behalf of LLM role agents."""

from __future__ import annotations

import os
import subprocess
import time
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
            "cmd",
            "powershell",
            "pwsh",
            "bash",
            "sh",
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

    def append_file(self, relative_path: str, content: str) -> None:
        path = self._path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="") as handle:
            handle.write(content)

    def file_snapshot(self, relative_path: str) -> tuple[str, str]:
        import hashlib
        content = self.read_file(relative_path)
        return hashlib.sha256(content.encode("utf-8")).hexdigest(), content

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
        app_server = any(
            part.endswith("app.py") and (self.workspace / part).is_file()
            and any(marker in (self.workspace / part).read_text(encoding="utf-8", errors="ignore") for marker in ("Flask(", "app.run(", "uvicorn.run("))
            for part in command
        )
        if self.is_long_running(command) or app_server:
            return CommandResult(125, "", "LONG_RUNNING_COMMAND_REQUIRES_MANAGED_PROCESS")
        try:
            flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
            process = subprocess.Popen(
                command,
                cwd=self.workspace,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=False,
                creationflags=flags,
            )
            try:
                stdout, stderr = process.communicate(timeout=self.command_timeout)
            except subprocess.TimeoutExpired:
                self._kill_tree(process.pid)
                stdout, stderr = process.communicate(timeout=10)
                return CommandResult(124, stdout or "", f"command hard timed out after {self.command_timeout}s\n{stderr or ''}")
        except FileNotFoundError as error:
            return CommandResult(127, "", f"command not found: {command[0]} ({error})")
        return CommandResult(process.returncode, stdout, stderr)

    @staticmethod
    def is_long_running(command: list[str]) -> bool:
        joined = " ".join(command).lower()
        first = Path(command[0]).name.lower() if command else ""
        return (
            "flask run" in joined or "uvicorn" in joined or "hypercorn" in joined or "gunicorn" in joined
            or "http.server" in joined or "npm run dev" in joined or "npm start" in joined
            or first in {"vite", "next"}
        )

    @staticmethod
    def _kill_tree(pid: int) -> None:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, check=False)
        else:
            try:
                os.kill(pid, 15)
            except ProcessLookupError:
                pass

    def run_tests(self, command: list[str]) -> CommandResult:
        return self.run_command(command)
