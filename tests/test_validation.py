from pathlib import Path

from autodev.models import Task
from autodev.validation import ValidationPlanner


def environment(interpreter: Path) -> dict[str, object]:
    return {"execution_context": {"python_interpreter": str(interpreter)}}


def test_unittest_project_uses_deterministic_discovery_with_project_interpreter(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_app.py").write_text("import unittest\n", encoding="utf-8")
    interpreter = tmp_path / ".venv" / "Scripts" / "python.exe"

    command = ValidationPlanner().command_for(tmp_path, Task.create("Backend API", "Implement endpoint"), environment(interpreter))

    assert command == [str(interpreter), "-m", "unittest", "discover", "-s", "tests", "-p", "test*.py"]


def test_pytest_project_uses_pytest_module_not_bare_executable(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_api.py").write_text("import pytest\n", encoding="utf-8")
    interpreter = tmp_path / ".venv" / "Scripts" / "python.exe"

    command = ValidationPlanner().command_for(tmp_path, Task.create("API", "Implement API"), environment(interpreter))

    assert command == [str(interpreter), "-m", "pytest", "-q"]


def test_static_frontend_validation_is_structured_and_requires_relevant_assets(tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text("<main>App</main>", encoding="utf-8")
    (tmp_path / "styles.css").write_text("main { display: grid; }", encoding="utf-8")
    interpreter = tmp_path / ".venv" / "Scripts" / "python.exe"

    command = ValidationPlanner().command_for(
        tmp_path, Task.create("Responsive frontend styling", "Create responsive CSS"), environment(interpreter)
    )

    assert command is not None
    assert command[:2] == [str(interpreter), "-c"]
    assert "*.html" in command[2] and "*.css" in command[2]
    assert "*.js" not in command[2]


def test_unknown_task_without_project_evidence_falls_back_to_tester(tmp_path: Path) -> None:
    command = ValidationPlanner().command_for(
        tmp_path, Task.create("Custom binary protocol", "Verify wire compatibility"), {}
    )

    assert command is None
