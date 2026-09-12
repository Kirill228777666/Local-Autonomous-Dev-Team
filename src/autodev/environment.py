"""Deterministic local environment discovery, failure classification and repairs."""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from .tools import CommandResult


class FailureKind(StrEnum):
    MISSING_EXECUTABLE = "MISSING_EXECUTABLE"
    MISSING_PYTHON_DEPENDENCY = "MISSING_PYTHON_DEPENDENCY"
    MISSING_NODE_DEPENDENCY = "MISSING_NODE_DEPENDENCY"
    IMPORT_PATH = "IMPORT_PATH"
    LOCAL_IMPORT_PATH_ERROR = "LOCAL_IMPORT_PATH_ERROR"
    ENVIRONMENT_NOT_INITIALIZED = "ENVIRONMENT_NOT_INITIALIZED"
    ASSERTION = "ASSERTION"
    PORT_CONFLICT = "PORT_CONFLICT"
    TIMEOUT = "TIMEOUT"
    PERMISSION = "PERMISSION"
    NETWORK = "NETWORK"
    APPLICATION = "APPLICATION"
    TEST_HARNESS_FAILURE = "TEST_HARNESS_FAILURE"
    SERVICE_NOT_RUNNING = "SERVICE_NOT_RUNNING"
    PROJECT_DEPENDENCY_INCOMPATIBLE = "PROJECT_DEPENDENCY_INCOMPATIBLE"
    GLOBAL_ENVIRONMENT_LEAK = "GLOBAL_ENVIRONMENT_LEAK"
    ARCHITECTURE_CONFLICT = "ARCHITECTURE_CONFLICT"


@dataclass(frozen=True, slots=True)
class Failure:
    kind: FailureKind
    command: list[str]
    detail: str
    module: str | None = None


@dataclass(frozen=True, slots=True)
class ReadmeResult:
    passed: bool
    findings: list[str]


def classify_failure(result: CommandResult, command: list[str], workspace: Path | None = None) -> Failure:
    text = f"{result.stdout}\n{result.stderr}"
    lowered = text.lower()
    if "environment_not_initialized" in lowered:
        return Failure(FailureKind.ENVIRONMENT_NOT_INITIALIZED, command, text)
    if "syntaxerror" in lowered and "-c" in command:
        return Failure(FailureKind.TEST_HARNESS_FAILURE, command, text)
    if "unittest.loader._failedtest" in lowered or ("has no attribute" in lowered and "unittest" in lowered):
        return Failure(FailureKind.TEST_HARNESS_FAILURE, command, text)
    if ("connection refused" in lowered or "failed to connect" in lowered) and any("localhost" in item or "127.0.0.1" in item for item in command):
        return Failure(FailureKind.SERVICE_NOT_RUNNING, command, text)
    if any(token in lowered for token in ("resolutionimpossible", "conflicting dependencies", "cannot import name 'url_quote'", "typingonly")):
        return Failure(FailureKind.PROJECT_DEPENDENCY_INCOMPATIBLE, command, text)
    if "site-packages" in lowered and ("ast.str" in lowered or "pluggy" in lowered):
        return Failure(FailureKind.GLOBAL_ENVIRONMENT_LEAK, command, text)
    module = re.search(r"ModuleNotFoundError: No module named ['\"]([^'\"]+)", text)
    if module:
        name = module.group(1)
        if workspace is not None and _is_local_module(workspace, name):
            return Failure(FailureKind.LOCAL_IMPORT_PATH_ERROR, command, text, name)
        kind = FailureKind.IMPORT_PATH if name in {"app", "src", "tests"} else FailureKind.MISSING_PYTHON_DEPENDENCY
        return Failure(kind, command, text, name)
    if result.exit_code == 127 or "command not found:" in text.lower() or "[winerror 2]" in text.lower():
        return Failure(FailureKind.MISSING_EXECUTABLE, command, text)
    if "cannot find module" in text.lower() or "module not found" in text.lower() and "node" in text.lower():
        return Failure(FailureKind.MISSING_NODE_DEPENDENCY, command, text)
    if "eaddrinuse" in text.lower() or "address already in use" in text.lower():
        return Failure(FailureKind.PORT_CONFLICT, command, text)
    if result.exit_code == 124 or "timed out" in text.lower():
        return Failure(FailureKind.TIMEOUT, command, text)
    if "permission denied" in text.lower() or "access is denied" in text.lower():
        return Failure(FailureKind.PERMISSION, command, text)
    if any(token in lowered for token in ("connection refused", "temporary failure", "certificate verify")):
        return Failure(FailureKind.NETWORK, command, text)
    if "assert " in text.lower() or "assertionerror" in text.lower() or "failed" in text.lower():
        return Failure(FailureKind.ASSERTION, command, text)
    return Failure(FailureKind.APPLICATION, command, text)


def _is_local_module(workspace: Path, module: str) -> bool:
    """Only a workspace module can be an import-path repair, never a pip install."""
    root = workspace.resolve()
    parts = module.split(".")
    candidate = root.joinpath(*parts)
    return candidate.is_dir() or candidate.with_suffix(".py").is_file()


class EnvironmentManager:
    def __init__(self, workspace: Path, tools: object, allow_project_dependency_install: bool = True, allow_system_package_install: bool = False) -> None:
        self.workspace = workspace.resolve()
        self.tools = tools
        self.allow_project_dependency_install = allow_project_dependency_install
        self.allow_system_package_install = allow_system_package_install
        self.last_diagnostic = ""

    @property
    def venv_python(self) -> Path:
        return self.workspace / ".venv" / ("Scripts/python.exe" if __import__("os").name == "nt" else "bin/python")

    def discover(self) -> dict[str, object]:
        def available(command: list[str]) -> tuple[bool, str]:
            result = self.tools.run_command(command)
            return result.exit_code == 0, (result.stdout or result.stderr).strip()
        python_ok, python_version = available(["py", "-3", "--version"])
        pip_ok, pip_version = available(["py", "-3", "-m", "pip", "--version"])
        git_ok, git_version = available(["git", "--version"])
        node_ok, node_version = available(["node", "--version"])
        npm_ok, npm_version = available(["npm", "--version"])
        return {"python": {"available": python_ok, "version": python_version, "venv": ".venv"}, "pip": {"available": pip_ok, "version": pip_version}, "git": {"available": git_ok, "version": git_version}, "node": {"available": node_ok, "version": node_version}, "npm": {"available": npm_ok, "version": npm_version}, "execution_context": self.execution_context()}

    def execution_context(self) -> dict[str, str]:
        return {
            "cwd": str(self.workspace),
            "python_interpreter": str(self.venv_python) if self.venv_python.exists() else "",
            "pip": f"{self.venv_python} -m pip" if self.venv_python.exists() else "",
        }

    def ensure_python_environment(self) -> bool:
        """Materialize and select the project interpreter before project commands."""
        if not self.ensure_venv():
            return False
        if not self.venv_python.is_file():
            self.last_diagnostic = f"Project virtual environment was not materialized: {self.venv_python}"
            return False
        # WorkspaceTools consumes this invariant before it accepts python/pytest.
        setattr(self.tools, "require_project_python", True)
        probe = self.tools.run_command([str(self.venv_python), "-c", "import site,sys; assert not site.ENABLE_USER_SITE; print(sys.executable)"])
        if probe.exit_code:
            self.last_diagnostic = probe.stderr or probe.stdout
            return False
        return True

    def ensure_venv(self) -> bool:
        if self.venv_python.exists():
            return True
        result = self.tools.run_command(["py", "-3", "-m", "venv", str(self.workspace / ".venv")])
        if result.exit_code:
            self.last_diagnostic = result.stderr or result.stdout
            return False
        return True

    def repair(self, failure: Failure, state: dict[str, object] | None = None) -> bool:
        if failure.kind is FailureKind.MISSING_PYTHON_DEPENDENCY and failure.module:
            if not self.allow_project_dependency_install:
                self.last_diagnostic = "Project dependency installation is disabled."
                return False
            package = {"flask": "Flask", "flask_sqlalchemy": "Flask-SQLAlchemy", "sqlalchemy": "SQLAlchemy"}.get(failure.module, failure.module)
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", package):
                self.last_diagnostic = f"Unsafe or ambiguous Python package name: {failure.module}"
                return False
            if not self.ensure_python_environment():
                return False
            fingerprint = f"{failure.module}|{self.venv_python}"
            dependencies = state.setdefault("dependencies", {}) if state is not None else {}
            failures = dependencies.setdefault("failed", []) if isinstance(dependencies, dict) else []
            if isinstance(failures, list) and fingerprint in failures:
                self.last_diagnostic = f"Dependency repair already failed without environment change: {failure.module}"
                return False
            manifest = self.workspace / "requirements.txt"
            existing = manifest.read_text(encoding="utf-8").splitlines() if manifest.exists() else []
            if not any(line.lower().split("==")[0] == package.lower() for line in existing):
                manifest.write_text("\n".join([*existing, package]) + "\n", encoding="utf-8")
            result = self.tools.run_command([str(self.venv_python), "-m", "pip", "install", package])
            self.last_diagnostic = result.stderr or result.stdout
            if isinstance(dependencies, dict):
                installed = dependencies.setdefault("installed", [])
                if result.exit_code == 0 and isinstance(installed, list) and package not in installed:
                    installed.append(package)
                if result.exit_code and isinstance(failures, list) and fingerprint not in failures:
                    failures.append(fingerprint)
                dependencies["interpreter"] = str(self.venv_python)
            return result.exit_code == 0
        if failure.kind is FailureKind.MISSING_EXECUTABLE and failure.command and failure.command[0].lower() in {"npm", "node"}:
            if not self.allow_system_package_install:
                self.last_diagnostic = "Node.js/npm required by chosen architecture but system installation is disabled."
                return False
        return False

    def verify(self, failure: Failure) -> bool:
        """Verify the repaired condition without claiming success from an install exit code."""
        if failure.kind is FailureKind.MISSING_PYTHON_DEPENDENCY and failure.module:
            probe = self.tools.run_command([str(self.venv_python), "-c", f"import {failure.module}"])
            self.last_diagnostic = probe.stderr or probe.stdout
            return probe.exit_code == 0
        return False


def validate_readme(workspace: Path) -> ReadmeResult:
    path = workspace / "README.md"
    if not path.is_file():
        return ReadmeResult(False, ["README.md is missing"])
    text = path.read_text(encoding="utf-8", errors="replace").lower()
    findings = []
    if not any(term in text for term in ("install", "setup", "установ")):
        findings.append("missing installation instructions")
    if not any(term in text for term in ("run", "start", "запуск")):
        findings.append("missing run instructions")
    if not any(term in text for term in ("test", "pytest", "тест")):
        findings.append("missing test instructions")
    return ReadmeResult(not findings, findings)
