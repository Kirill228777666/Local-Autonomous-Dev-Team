"""Deterministic test discovery and validation outcome classification."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from .models import Task
from .tools import CommandResult


class ValidationOutcome(StrEnum):
    PASS = "VALIDATION_PASS"
    APPLICATION_FAILURE = "VALIDATION_APPLICATION_FAIL"
    NO_TESTS = "VALIDATION_NO_TESTS"
    COMMAND_INVALID = "VALIDATION_COMMAND_INVALID"
    TOOL_MISSING = "VALIDATION_TOOL_MISSING"
    IMPORT_OR_ENVIRONMENT_ERROR = "VALIDATION_IMPORT_OR_ENVIRONMENT_ERROR"
    APPLICATION_IMPORT_ERROR = "VALIDATION_APPLICATION_IMPORT_ERROR"
    DEPENDENCY_API_MISMATCH = "VALIDATION_DEPENDENCY_API_MISMATCH"
    TEST_IMPLEMENTATION_BUG = "VALIDATION_TEST_IMPLEMENTATION_BUG"
    TIMEOUT = "VALIDATION_TIMEOUT"


@dataclass(frozen=True, slots=True)
class ValidationResult:
    kind: ValidationOutcome
    detail: str


_IGNORED_DIRECTORIES = {".git", ".autodev", ".venv", "__pycache__", "node_modules"}
_KNOWN_TEST_ROOTS = ("tests", "test", "backend/tests")


def _is_test_file(path: Path) -> bool:
    return path.suffix == ".py" and (path.name.startswith("test_") or path.name.endswith("_test.py"))


def _is_workspace_file(path: Path, workspace: Path) -> bool:
    try:
        relative = path.relative_to(workspace)
    except ValueError:
        return False
    return not any(part in _IGNORED_DIRECTORIES for part in relative.parts)


def discover_python_tests(workspace: Path) -> list[Path]:
    """Return source test files from common layouts, never cached bytecode."""
    root = workspace.resolve()
    return sorted(
        (path for path in root.rglob("*.py") if _is_workspace_file(path, root) and _is_test_file(path)),
        key=lambda path: path.as_posix(),
    )


def _test_root(workspace: Path, tests: list[Path]) -> Path:
    root = workspace.resolve()
    for relative in _KNOWN_TEST_ROOTS:
        candidate = root / relative
        if any(candidate == path.parent or candidate in path.parents for path in tests):
            return candidate
    return tests[0].parent


def _pytest_evidence(workspace: Path, tests: list[Path]) -> bool:
    for filename in ("pytest.ini", "tox.ini", "setup.cfg", "pyproject.toml", "requirements.txt", "requirements-dev.txt"):
        path = workspace / filename
        if path.is_file() and "pytest" in path.read_text(encoding="utf-8", errors="ignore").lower():
            return True
    return any(
        "pytest" in path.read_text(encoding="utf-8", errors="ignore")[:12000].lower()
        or re.search(r"^def test_", path.read_text(encoding="utf-8", errors="ignore"), flags=re.MULTILINE) is not None
        for path in tests
    )


def _is_test_runner(command: list[str]) -> bool:
    return "unittest" in command or "pytest" in command


def _test_start_directory(command: list[str]) -> str | None:
    try:
        index = command.index("-s")
    except ValueError:
        return None
    return command[index + 1] if index + 1 < len(command) else None


def classify_validation_result(result: CommandResult, command: list[str], workspace: Path) -> ValidationResult:
    """Classify validator evidence without confusing test results with bad CLI syntax."""
    detail = f"{result.stdout}\n{result.stderr}".strip()
    lowered = detail.lower()
    if result.exit_code == 0:
        return ValidationResult(ValidationOutcome.PASS, detail)
    if result.exit_code == 124 or "timed out" in lowered:
        return ValidationResult(ValidationOutcome.TIMEOUT, detail)
    if result.exit_code == 127 or "command not found" in lowered or "[winerror 2]" in lowered:
        return ValidationResult(ValidationOutcome.TOOL_MISSING, detail)
    if _is_test_runner(command) and ("no tests ran" in lowered or re.search(r"ran\s+0\s+tests", lowered)):
        return ValidationResult(ValidationOutcome.NO_TESTS, detail)
    start = _test_start_directory(command)
    if _is_test_runner(command) and "start directory is not importable" in lowered:
        if start and not discover_python_tests(workspace):
            return ValidationResult(ValidationOutcome.NO_TESTS, detail)
        return ValidationResult(ValidationOutcome.COMMAND_INVALID, detail)
    if "syntaxerror" in lowered and "-c" in command:
        return ValidationResult(ValidationOutcome.COMMAND_INVALID, detail)
    if any(token in lowered for token in ("unrecognized arguments", "invalid choice", "usage: pytest")):
        return ValidationResult(ValidationOutcome.COMMAND_INVALID, detail)
    if "no module named pytest" in lowered or "no module named unittest" in lowered:
        return ValidationResult(ValidationOutcome.TOOL_MISSING, detail)
    if any(token in lowered for token in ("before_first_request", "detachedinstanceerror", "has no attribute")):
        return ValidationResult(ValidationOutcome.DEPENDENCY_API_MISMATCH, detail)
    if "cannot import name" in lowered:
        return ValidationResult(ValidationOutcome.APPLICATION_IMPORT_ERROR, detail)
    if "unittest.loader._failedtest" in lowered and ("tests" in lowered or "test_" in lowered):
        return ValidationResult(ValidationOutcome.TEST_IMPLEMENTATION_BUG, detail)
    if "importerror" in lowered or "modulenotfounderror" in lowered:
        return ValidationResult(ValidationOutcome.APPLICATION_IMPORT_ERROR, detail)
    if "unittest.loader._failedtest" in lowered:
        return ValidationResult(ValidationOutcome.IMPORT_OR_ENVIRONMENT_ERROR, detail)
    return ValidationResult(ValidationOutcome.APPLICATION_FAILURE, detail)


class ValidationPlanner:
    """Choose a test runner only when repository evidence identifies one."""

    @staticmethod
    def command_for(workspace: Path, task: Task, environment: dict[str, object]) -> list[str] | None:
        context = environment.get("execution_context")
        interpreter = context.get("python_interpreter") if isinstance(context, dict) else None
        if not isinstance(interpreter, str) or not interpreter:
            return None

        tests = discover_python_tests(workspace)
        if tests:
            if _pytest_evidence(workspace, tests):
                return [interpreter, "-m", "pytest", "-q"]
            relative_root = _test_root(workspace, tests).relative_to(workspace).as_posix()
            prefixes = any(path.name.startswith("test_") for path in tests)
            suffixes = any(path.name.endswith("_test.py") for path in tests)
            pattern = "*.py" if prefixes and suffixes else "*_test.py" if suffixes else "test*.py"
            return [interpreter, "-m", "unittest", "discover", "-s", relative_root, "-p", pattern]

        return ValidationPlanner.fallback_for_no_tests(workspace, task, environment)

    @staticmethod
    def fallback_for_no_tests(workspace: Path, task: Task, environment: dict[str, object]) -> list[str] | None:
        """Return a non-test acceptance probe after a runner supplied no evidence."""
        context = environment.get("execution_context")
        interpreter = context.get("python_interpreter") if isinstance(context, dict) else None
        if not isinstance(interpreter, str) or not interpreter:
            return None
        text = f"{task.title} {task.description}".lower()
        frontend = any(word in text for word in ("frontend", "ui", "html", "css", "responsive", "интерфейс", "фронтенд"))
        if frontend and not (workspace / "package.json").is_file():
            patterns = ["*.html"]
            if any(word in text for word in ("style", "css", "responsive", "стил")):
                patterns.append("*.css")
            if any(word in text for word in ("action", "javascript", "search", "filter", "crud", "js", "поиск", "фильтр")):
                patterns.append("*.js")
            checks = "; ".join(
                f"assert any(p.is_file() and p.stat().st_size for p in root.rglob('{pattern}')), '{pattern} missing'"
                for pattern in patterns
            )
            return [interpreter, "-c", f"from pathlib import Path; root=Path('.'); {checks}"]
        asks_for_tests = any(word in text for word in ("test", "pytest", "unittest", "тест"))
        root = workspace.resolve()
        sources = [
            path for path in workspace.rglob("*.py")
            if _is_workspace_file(path, root) and path.name not in {"conftest.py"} and not _is_test_file(path)
        ]
        if sources and not asks_for_tests:
            relative = sorted(path.relative_to(root).as_posix() for path in sources)
            return [interpreter, "-m", "compileall", "-q", *relative]
        return None
