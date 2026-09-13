"""Policy-controlled tools executed on behalf of LLM role agents."""

from __future__ import annotations

import difflib
import hashlib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .runtime import kill_process_tree


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
        self.require_project_python = False

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
        if path.is_file():
            previous = path.read_text(encoding="utf-8")
            # A short overwrite of a substantial source file is almost always a
            # stale-model rewrite.  Targeted edits or a full current-file rewrite
            # remain available, but silently dropping accepted routes is not.
            if len(previous) >= 500 and len(content) < len(previous) * 0.5:
                raise ToolPolicyError("DESTRUCTIVE_WRITE_REQUIRES_TARGETED_EDIT_OR_CURRENT_FULL_FILE")
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
        content = self.read_file(relative_path)
        return hashlib.sha256(content.encode("utf-8")).hexdigest(), content

    def apply_deterministic_patch(
        self,
        relative_path: str,
        expected_hash: str,
        old: str,
        desired: str,
    ) -> bool:
        """Apply an exact, compare-and-swap textual patch for a rejected write.

        This is deliberately not a general bypass for ``write_file``.  It is used
        only by the controller after the destructive-write policy has rejected a
        proposed full replacement and therefore has both the exact pre-image and
        desired post-image.  The expected content hash closes the stale-context
        race; a caller must fall back to a refreshed edit if it no longer matches.
        """
        path = self._path(relative_path)
        if not path.is_file():
            return False
        current = path.read_text(encoding="utf-8")
        if current != old or hashlib.sha256(current.encode("utf-8")).hexdigest() != expected_hash:
            return False

        old_lines = old.splitlines(keepends=True)
        desired_lines = desired.splitlines(keepends=True)
        candidate = list(old_lines)
        matcher = difflib.SequenceMatcher(a=old_lines, b=desired_lines, autojunk=False)
        for _tag, start, end, desired_start, desired_end in reversed(matcher.get_opcodes()):
            candidate[start:end] = desired_lines[desired_start:desired_end]
        patched = "".join(candidate)
        if patched != desired:
            return False

        # Verify immediately before writing as well: a file changed after the
        # initial snapshot must never be overwritten by recovery code.
        current = path.read_text(encoding="utf-8")
        if current != old or hashlib.sha256(current.encode("utf-8")).hexdigest() != expected_hash:
            return False
        path.write_text(patched, encoding="utf-8", newline="")
        return path.read_text(encoding="utf-8") == desired

    def delete_file(self, relative_path: str) -> None:
        path = self._path(relative_path)
        if path.is_dir():
            raise ToolPolicyError("directory deletion is not permitted")
        path.unlink(missing_ok=True)

    def run_command(self, command: list[str], extra_env: dict[str, str] | None = None) -> CommandResult:
        if not command or not command[0].strip():
            raise ToolPolicyError("command must not be empty")
        if self._is_activation_command(command):
            return CommandResult(126, "", "PROJECT_ENVIRONMENT_IS_ALREADY_MANAGED; use the project interpreter explicitly")
        if self.require_project_python and self._is_project_python_command(command) and self.project_python is None:
            return CommandResult(126, "", "ENVIRONMENT_NOT_INITIALIZED: project Python interpreter is missing")
        command = self.normalize_command(command)
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
                env={**os.environ, **(extra_env or {}), **({"PYTHONNOUSERSITE": "1"} if self.project_python else {})},
            )
            try:
                stdout, stderr = process.communicate(timeout=self.command_timeout)
            except subprocess.TimeoutExpired:
                # Do not call subprocess.run here: on Windows its communicate() can
                # still wait for a Flask watchdog child which inherited the pipe.
                kill_process_tree(process.pid)
                try:
                    stdout, stderr = process.communicate(timeout=min(2.0, max(0.5, self.command_timeout)))
                except subprocess.TimeoutExpired:
                    process.kill()
                    stdout, stderr = "", "process tree did not exit after forced cleanup"
                return CommandResult(124, stdout or "", f"command hard timed out after {self.command_timeout}s\n{stderr or ''}")
        except FileNotFoundError as error:
            return CommandResult(127, "", f"command not found: {command[0]} ({error})")
        return CommandResult(process.returncode, stdout, stderr)

    @property
    def project_python(self) -> Path | None:
        candidate = self.workspace / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        return candidate if candidate.is_file() else None

    def normalize_command(self, command: list[str]) -> list[str]:
        """Use the project venv without relying on mutable shell activation."""
        if not command:
            return command
        interpreter = self.project_python
        if interpreter is None:
            return list(command)
        first = Path(command[0]).name.lower()
        if first in {"python", "python.exe", "py", "py.exe"}:
            remainder = list(command[1:])
            if first.startswith("py") and remainder[:1] in (["-3"], ["-3.14"]):
                remainder = remainder[1:]
            return [str(interpreter), *remainder]
        if first in {"pip", "pip.exe"}:
            return [str(interpreter), "-m", "pip", *command[1:]]
        if first in {"pytest", "pytest.exe", "unittest"}:
            return [str(interpreter), "-m", first.removesuffix(".exe"), *command[1:]]
        return list(command)

    @staticmethod
    def _is_project_python_command(command: list[str]) -> bool:
        if not command:
            return False
        first = Path(command[0]).name.lower()
        return first in {"python", "python.exe", "py", "py.exe", "pip", "pip.exe", "pytest", "pytest.exe", "unittest"}

    @staticmethod
    def _is_activation_command(command: list[str]) -> bool:
        text = " ".join(command).replace("/", "\\").lower()
        return any(marker in text for marker in (".venv\\scripts\\activate", ".venv\\bin\\activate", "activate.ps1", "source .venv"))

    @staticmethod
    def is_long_running(command: list[str]) -> bool:
        joined = " ".join(command).lower()
        first = Path(command[0]).name.lower() if command else ""
        return (
            "flask run" in joined or "uvicorn" in joined or "hypercorn" in joined or "gunicorn" in joined
            or "http.server" in joined or "npm run dev" in joined or "npm start" in joined
            or first in {"vite", "next"}
        )

    def run_tests(self, command: list[str], extra_env: dict[str, str] | None = None) -> CommandResult:
        return self.run_command(command, extra_env)
