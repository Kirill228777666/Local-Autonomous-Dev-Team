from pathlib import Path

from autodev.models import Task
from autodev.tools import CommandResult
from autodev.validation import ValidationOutcome, ValidationPlanner, classify_validation_result


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


def test_unittest_zero_discovery_is_no_tests_not_a_harness_error(tmp_path: Path) -> None:
    result = CommandResult(5, "\nRan 0 tests in 0.000s\n\nNO TESTS RAN\n", "")

    outcome = classify_validation_result(
        result,
        ["python", "-m", "unittest", "discover", "-s", "tests", "-p", "test*.py"],
        tmp_path,
    )

    assert outcome.kind is ValidationOutcome.NO_TESTS


def test_successful_unittest_validator_is_validation_pass(tmp_path: Path) -> None:
    outcome = classify_validation_result(
        CommandResult(0, "Ran 1 test in 0.001s\n\nOK\n", ""),
        ["python", "-m", "unittest", "discover", "-s", "tests"],
        tmp_path,
    )

    assert outcome.kind is ValidationOutcome.PASS


def test_pytest_zero_discovery_is_no_tests_not_a_harness_error(tmp_path: Path) -> None:
    result = CommandResult(5, "\nno tests ran in 0.01s\n", "")

    outcome = classify_validation_result(result, ["python", "-m", "pytest", "-q"], tmp_path)

    assert outcome.kind is ValidationOutcome.NO_TESTS


def test_unittest_assertion_failure_is_application_failure(tmp_path: Path) -> None:
    result = CommandResult(1, "F\nFAILED (failures=1)\n", "AssertionError: expected title")

    outcome = classify_validation_result(result, ["python", "-m", "unittest", "discover", "-s", "tests"], tmp_path)

    assert outcome.kind is ValidationOutcome.APPLICATION_FAILURE


def test_test_module_import_error_is_not_a_harness_error(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_api.py").write_text("import unittest\n", encoding="utf-8")
    result = CommandResult(
        1,
        "",
        "ImportError: cannot import name 'Note' from 'app'\nFAILED (errors=1)",
    )

    outcome = classify_validation_result(result, ["python", "-m", "unittest", "discover", "-s", "tests"], tmp_path)

    assert outcome.kind is ValidationOutcome.IMPORT_OR_ENVIRONMENT_ERROR


def test_malformed_python_validator_is_command_invalid(tmp_path: Path) -> None:
    result = CommandResult(1, "", "SyntaxError: invalid syntax")

    outcome = classify_validation_result(result, ["python", "-c", "import re; with open('x'): pass"], tmp_path)

    assert outcome.kind is ValidationOutcome.COMMAND_INVALID


def test_missing_pytest_module_is_validation_tool_missing(tmp_path: Path) -> None:
    outcome = classify_validation_result(
        CommandResult(1, "", "No module named pytest"),
        ["python", "-m", "pytest", "-q"],
        tmp_path,
    )

    assert outcome.kind is ValidationOutcome.TOOL_MISSING


def test_discovery_finds_unittest_files_under_backend_tests(tmp_path: Path) -> None:
    tests = tmp_path / "backend" / "tests"
    tests.mkdir(parents=True)
    (tests / "test_api.py").write_text("import unittest\n", encoding="utf-8")
    interpreter = tmp_path / ".venv" / "Scripts" / "python.exe"

    command = ValidationPlanner().command_for(tmp_path, Task.create("Backend API", "Implement endpoint"), environment(interpreter))

    assert command == [str(interpreter), "-m", "unittest", "discover", "-s", "backend/tests", "-p", "test*.py"]


def test_discovery_uses_suffix_pattern_for_unittest_suffix_test_files(tmp_path: Path) -> None:
    tests = tmp_path / "test"
    tests.mkdir()
    (tests / "api_test.py").write_text("import unittest\n", encoding="utf-8")
    interpreter = tmp_path / ".venv" / "Scripts" / "python.exe"

    command = ValidationPlanner().command_for(tmp_path, Task.create("Backend API", "Implement endpoint"), environment(interpreter))

    assert command == [str(interpreter), "-m", "unittest", "discover", "-s", "test", "-p", "*_test.py"]


def test_pytest_configuration_selects_pytest_for_discovered_tests(tmp_path: Path) -> None:
    (tmp_path / "backend" / "tests").mkdir(parents=True)
    (tmp_path / "backend" / "tests" / "test_api.py").write_text("def test_api():\n    assert True\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\n", encoding="utf-8")
    interpreter = tmp_path / ".venv" / "Scripts" / "python.exe"

    command = ValidationPlanner().command_for(tmp_path, Task.create("Backend API", "Implement endpoint"), environment(interpreter))

    assert command == [str(interpreter), "-m", "pytest", "-q"]


def test_python_source_without_tests_uses_compile_fallback_as_non_test_evidence(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("value = 1\n", encoding="utf-8")
    interpreter = tmp_path / ".venv" / "Scripts" / "python.exe"

    command = ValidationPlanner().command_for(tmp_path, Task.create("Backend API", "Implement endpoint"), environment(interpreter))

    assert command == [str(interpreter), "-m", "compileall", "-q", "app.py"]
