"""Deterministic local environment discovery, failure classification and repairs."""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from .tools import CommandResult
from .capabilities import contract_allows_dependency, contract_declares_dependency


class FailureKind(StrEnum):
    MISSING_EXECUTABLE = "MISSING_EXECUTABLE"
    MISSING_PROJECT_DEPENDENCY = "MISSING_PROJECT_DEPENDENCY"
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


@dataclass(frozen=True, slots=True)
class DependencyProvisioning:
    """Result of a deterministic project-local manifest provisioning pass."""
    attempted: bool
    succeeded: bool
    packages: list[str]


def classify_failure(result: CommandResult, command: list[str], workspace: Path | None = None) -> Failure:
    text = f"{result.stdout}\n{result.stderr}"
    lowered = text.lower()
    if "environment_not_initialized" in lowered:
        return Failure(FailureKind.ENVIRONMENT_NOT_INITIALIZED, command, text)
    if "syntaxerror" in lowered and "-c" in command:
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
        # Runner wrappers such as unittest.loader._FailedTest are not the
        # cause.  The nested missing module is authoritative.  Keep local
        # application roots on the import/path route; known or declared third
        # party imports are owned project dependencies.
        kind = FailureKind.IMPORT_PATH if name in {"app", "src", "tests"} else FailureKind.MISSING_PROJECT_DEPENDENCY
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
            "system_site_packages": str(self._system_site_packages()).lower(),
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
        # The environment remains project-owned while safely reusing packages
        # already present in the selected host Python.  New project packages
        # are still installed only through this venv interpreter.
        result = self.tools.run_command(["py", "-3", "-m", "venv", "--system-site-packages", str(self.workspace / ".venv")])
        if result.exit_code:
            self.last_diagnostic = result.stderr or result.stdout
            return False
        return True

    def ensure_declared_dependencies(self) -> DependencyProvisioning:
        """Materialize missing declared packages in the owned interpreter only.

        A system-site-enabled venv can satisfy the probe without an install;
        an absent or incompatible declaration is installed into the project
        environment.  No command here ever targets the host interpreter.
        """
        if not self.ensure_python_environment():
            return DependencyProvisioning(False, False, [])
        requirements, has_requirements_file = self._declared_requirements()
        if not requirements:
            return DependencyProvisioning(False, True, [])
        missing: list[str] = []
        for requirement in requirements:
            package = _requirement_name(requirement)
            if not package:
                continue
            probe = self.tools.run_command([
                str(self.venv_python), "-c",
                f"import importlib.metadata as m; print(m.version({package!r}))",
            ])
            if probe.exit_code or not _requirement_is_satisfied(requirement, probe.stdout.strip()):
                missing.append(package)
        if not missing:
            return DependencyProvisioning(False, True, [])
        install = [str(self.venv_python), "-m", "pip", "install"]
        install.extend(["-r", "requirements.txt"] if has_requirements_file else missing)
        result = self.tools.run_command(install)
        self.last_diagnostic = result.stderr or result.stdout
        if result.exit_code:
            return DependencyProvisioning(True, False, missing)
        for package in missing:
            probe = self.tools.run_command([
                str(self.venv_python), "-c",
                f"import importlib.metadata as m; print(m.version({package!r}))",
            ])
            if probe.exit_code:
                self.last_diagnostic = probe.stderr or probe.stdout
                return DependencyProvisioning(True, False, missing)
        return DependencyProvisioning(True, True, missing)

    def repair(
        self,
        failure: Failure,
        state: dict[str, object] | None = None,
        contract: dict[str, object] | None = None,
    ) -> bool:
        if failure.kind in {FailureKind.MISSING_PROJECT_DEPENDENCY, FailureKind.MISSING_PYTHON_DEPENDENCY} and failure.module:
            if not contract_allows_dependency(contract, failure.module):
                self.last_diagnostic = f"Frozen project contract marks dependency installation as forbidden: {failure.module}"
                return False
            if not self.allow_project_dependency_install:
                self.last_diagnostic = "Project dependency installation is disabled."
                return False
            package = {"flask": "Flask", "flask_sqlalchemy": "Flask-SQLAlchemy", "sqlalchemy": "SQLAlchemy"}.get(failure.module, failure.module)
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", package):
                self.last_diagnostic = f"Unsafe or ambiguous Python package name: {failure.module}"
                return False
            declared, _has_requirements_file = self._declared_requirements()
            declared_names = {_requirement_name(item).replace("-", "_") for item in declared}
            normalized_package = package.lower().replace("-", "_")
            if contract is not None and normalized_package not in declared_names and not contract_declares_dependency(contract, package):
                self.last_diagnostic = f"Frozen project contract does not declare missing dependency: {package}"
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
            if not any(_requirement_name(line) == package.lower() for line in existing):
                manifest.write_text("\n".join([*existing, package]) + "\n", encoding="utf-8")
            # Prefer the declared project manifest when it already owns the
            # missing package.  Otherwise persist the safe mapping before the
            # package-specific installation so the repair is reproducible.
            install_command = [str(self.venv_python), "-m", "pip", "install", package]
            if any(_requirement_name(line) == package.lower() for line in existing):
                install_command = [str(self.venv_python), "-m", "pip", "install", "-r", "requirements.txt"]
            result = self.tools.run_command(install_command)
            self.last_diagnostic = result.stderr or result.stdout
            if isinstance(dependencies, dict):
                installed = dependencies.setdefault("installed", [])
                if result.exit_code == 0 and isinstance(installed, list) and package not in installed:
                    installed.append(package)
                if result.exit_code and isinstance(failures, list) and fingerprint not in failures:
                    failures.append(fingerprint)
                dependencies["interpreter"] = str(self.venv_python)
            return result.exit_code == 0 and self.verify(failure)
        if failure.kind is FailureKind.PROJECT_DEPENDENCY_INCOMPATIBLE:
            lowered = failure.detail.lower()
            if "sqlalchemy" not in lowered and "typingonly" not in lowered:
                self.last_diagnostic = "No deterministic package repair is known for this incompatibility."
                return False
            if not self.allow_project_dependency_install:
                self.last_diagnostic = "Project dependency installation is disabled."
                return False
            if not self.ensure_python_environment():
                return False
            install = self.tools.run_command(
                [str(self.venv_python), "-m", "pip", "install", "--upgrade", "SQLAlchemy"]
            )
            if install.exit_code:
                self.last_diagnostic = install.stderr or install.stdout
                return False
            version = self.tools.run_command(
                [
                    str(self.venv_python), "-c",
                    "import importlib.metadata as m; print(m.version('SQLAlchemy'))",
                ]
            )
            resolved = version.stdout.strip()
            if version.exit_code or not re.fullmatch(r"[0-9]+(?:\.[0-9A-Za-z]+)+", resolved):
                self.last_diagnostic = version.stderr or version.stdout or "Could not determine resolved SQLAlchemy version."
                return False
            self._persist_requirement("SQLAlchemy", resolved)
            self.last_diagnostic = f"Resolved SQLAlchemy=={resolved} for the project interpreter."
            return True
        if failure.kind is FailureKind.MISSING_EXECUTABLE and failure.command and failure.command[0].lower() in {"npm", "node"}:
            if not self.allow_system_package_install:
                self.last_diagnostic = "Node.js/npm required by chosen architecture but system installation is disabled."
                return False
        return False

    def _persist_requirement(self, package: str, version: str) -> None:
        manifest = self.workspace / "requirements.txt"
        lines = manifest.read_text(encoding="utf-8").splitlines() if manifest.exists() else []
        matcher = re.compile(rf"^\s*{re.escape(package)}(?:\[[^]]+\])?\s*(?:[<>=!~].*)?$", re.IGNORECASE)
        replacement = f"{package}=={version}"
        updated: list[str] = []
        replaced = False
        for line in lines:
            if matcher.match(line):
                if not replaced:
                    updated.append(replacement)
                    replaced = True
                continue
            updated.append(line)
        if not replaced:
            updated.append(replacement)
        manifest.write_text("\n".join(updated) + "\n", encoding="utf-8")

    def verify(self, failure: Failure) -> bool:
        """Verify the repaired condition without claiming success from an install exit code."""
        if failure.kind in {FailureKind.MISSING_PROJECT_DEPENDENCY, FailureKind.MISSING_PYTHON_DEPENDENCY} and failure.module:
            probe = self.tools.run_command([str(self.venv_python), "-c", f"import {failure.module}"])
            self.last_diagnostic = probe.stderr or probe.stdout
            return probe.exit_code == 0
        return False

    def _system_site_packages(self) -> bool:
        config = self.workspace / ".venv" / "pyvenv.cfg"
        if not config.is_file():
            return False
        return "include-system-site-packages = true" in config.read_text(encoding="utf-8", errors="ignore").lower()

    def _declared_requirements(self) -> tuple[list[str], bool]:
        """Read conventional Python dependency declarations without executing them."""
        requirements_file = self.workspace / "requirements.txt"
        if requirements_file.is_file():
            lines = requirements_file.read_text(encoding="utf-8", errors="ignore").splitlines()
            return _clean_requirements(lines), True
        pyproject = self.workspace / "pyproject.toml"
        if pyproject.is_file():
            text = pyproject.read_text(encoding="utf-8", errors="ignore")
            block = re.search(r"dependencies\s*=\s*\[(.*?)\]", text, flags=re.DOTALL)
            if block:
                return _clean_requirements(re.findall(r"['\"]([^'\"]+)['\"]", block.group(1))), False
        setup_cfg = self.workspace / "setup.cfg"
        if setup_cfg.is_file():
            text = setup_cfg.read_text(encoding="utf-8", errors="ignore")
            block = re.search(r"install_requires\s*=\s*(.*?)(?:\n\[|\Z)", text, flags=re.DOTALL)
            if block:
                return _clean_requirements(block.group(1).splitlines()), False
        return [], False


def _requirement_name(line: str) -> str:
    return re.split(r"[<>=!~\[\s]", line.strip(), maxsplit=1)[0].lower()


def _clean_requirements(lines: list[str]) -> list[str]:
    return [line.strip() for line in lines if line.strip() and not line.lstrip().startswith(("#", "-"))]


def _requirement_is_satisfied(requirement: str, installed: str) -> bool:
    """Conservatively compare a declared public requirement with one version."""
    match = re.match(r"\s*[A-Za-z0-9_.-]+(?:\[[^]]+\])?\s*(.*)$", requirement)
    specifier = match.group(1).strip() if match else ""
    if not specifier:
        return bool(installed)
    try:
        from packaging.specifiers import SpecifierSet
        from packaging.version import Version
        return Version(installed) in SpecifierSet(specifier)
    except Exception:
        # A nonstandard declaration must not silently reuse an unknown host
        # version.  Let the project-local installer resolve it instead.
        return False


def validate_readme(workspace: Path, contract: dict[str, object] | None = None) -> ReadmeResult:
    path = workspace / "README.md"
    if not path.is_file():
        return ReadmeResult(False, ["README.md is missing"])
    text = path.read_text(encoding="utf-8", errors="replace").lower()
    findings = []
    if not any(term in text for term in ("install", "setup", "установ")):
        findings.append("missing installation instructions")
    if not any(term in text for term in ("run", "start", "запуск")):
        findings.append("missing run instructions")
    has_executable_tests = any(
        path.name.startswith("test_") or path.name.endswith("_test.py")
        for path in workspace.rglob("*.py")
        if not any(part in {".venv", ".git", ".autodev", "__pycache__"} for part in path.parts)
    )
    # Legacy direct callers retain the conservative test-instructions rule.
    # A contract-aware controller does not document a fictional test command
    # when the project has no executable tests to run.
    require_test_instructions = contract is None or has_executable_tests
    if require_test_instructions and not any(term in text for term in ("test", "pytest", "тест")):
        findings.append("missing test instructions")
    return ReadmeResult(not findings, findings)
