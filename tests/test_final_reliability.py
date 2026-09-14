from __future__ import annotations

import sys
from pathlib import Path

from autodev.models import ProjectState, Task
from autodev.controller import TaskController
from autodev.orchestrator import AutonomousRunner
from autodev.providers import ScriptedProvider
from autodev.state_store import StateStore
from autodev.tools import CommandResult, WorkspaceTools
from autodev.validation import ValidationOutcome, ValidationPlanner, classify_validation_result


def test_business_assertion_is_not_misclassified_as_test_state_isolation(tmp_path: Path) -> None:
    result = CommandResult(1, "", "AssertionError: 0 != 1\nRan 10 tests in 0.2s")

    classified = classify_validation_result(result, [sys.executable, "-m", "unittest"], tmp_path)

    assert classified.kind is ValidationOutcome.APPLICATION_FAILURE


def test_proven_shared_database_evidence_classifies_test_state_isolation(tmp_path: Path) -> None:
    result = CommandResult(1, "", "AssertionError: 8 != 1\nRan 10 tests in 0.2s")

    classified = classify_validation_result(
        result,
        [sys.executable, "-m", "unittest"],
        tmp_path,
        state_evidence={"shared_database": True, "test_database_consumed": False},
    )

    assert classified.kind is ValidationOutcome.TEST_STATE_ISOLATION_FAILURE


def test_compileall_of_flask_source_is_not_treated_as_long_running_server(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("from flask import Flask\napp = Flask(__name__)\n", encoding="utf-8")
    tools = WorkspaceTools(tmp_path)

    result = tools.run_command([sys.executable, "-m", "compileall", "-q", "app.py"])

    assert result.exit_code != 125
    assert "LONG_RUNNING_COMMAND_REQUIRES_MANAGED_PROCESS" not in result.stderr


def test_api_acceptance_selects_behavior_test_by_contract_terms_not_compileall(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("value = 1\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_app.py").write_text(
        "def test_favorite_filter():\n    assert True\n\n"
        "def test_search_by_title():\n    assert True\n",
        encoding="utf-8",
    )
    task = Task.create(
        "Implement Search and Filtering Logic",
        "Enable query behavior",
        capability_id="api.notes_query",
        acceptance_criteria=[
            "GET /notes?search=<term> filters results by title/content match.",
            "GET /notes?favorite=true retrieves only favorite notes.",
        ],
    )

    command = ValidationPlanner.command_for(
        tmp_path,
        task,
        {"execution_context": {"python_interpreter": sys.executable}},
        acceptance_only=True,
    )

    assert command is not None
    assert any(item in command for item in ("unittest", "pytest"))
    assert "compileall" not in command
    assert any(item.endswith("test_app.py") for item in command)


def test_behavioral_api_without_owned_test_cannot_fall_back_to_compileall(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("value = 1\n", encoding="utf-8")
    task = Task.create(
        "Implement filtering",
        "Implement API filtering behavior",
        capability_id="api.example.filters",
        acceptance_criteria=["GET /items?favorite=true returns only favorite items."],
    )

    command = ValidationPlanner.command_for(
        tmp_path,
        task,
        {"execution_context": {"python_interpreter": sys.executable}},
        acceptance_only=True,
    )

    assert command is None


def test_validator_result_is_durably_owned_by_its_capability(tmp_path: Path) -> None:
    state = ProjectState.create("Build application")
    task = Task.create("Implement API", "Create endpoint", capability_id="api.example")
    state.tasks = [task]

    state.record_validator_result(
        task=task,
        validator_id="acceptance-api",
        command=[sys.executable, "-m", "compileall", "app.py"],
        result=CommandResult(1, "", "SyntaxError: invalid syntax"),
        failure_class="VALIDATION_APPLICATION_FAIL",
        previous_validator_run_id=None,
    )

    record = state.validator_runs[-1]
    assert record["capability_id"] == "api.example"
    assert record["validator_run_id"] == task.last_validator_run_id
    assert record["exact_command"][-1] == "app.py"


def test_controller_links_validator_event_and_repair_packet_to_same_run(tmp_path: Path) -> None:
    runner = AutonomousRunner(tmp_path, StateStore(tmp_path), WorkspaceTools(tmp_path), ScriptedProvider({}))
    state = ProjectState.create("Build application")
    task = Task.create("Implement API", "Create endpoint", capability_id="api.example")
    state.tasks = [task]

    result, validation, record = runner._execute_validator(state, task, [sys.executable, "-c", "raise SystemExit(1)"])
    runner._capture_repair_evidence(state, task, record, validation.kind, result)

    run_id = str(record["validator_run_id"])
    assert any(run_id in event.message for event in state.events if event.phase == validation.kind.value)
    assert task.last_repair_packet["source_validator_run_id"] == run_id


def test_coder_server_tool_error_cannot_replace_compileall_validator_provenance(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("from flask import Flask\napp = Flask(__name__)\n", encoding="utf-8")
    runner = AutonomousRunner(tmp_path, StateStore(tmp_path), WorkspaceTools(tmp_path), ScriptedProvider({}))
    runner.controller = TaskController(max_strategy_changes=0)
    state = ProjectState.create("Build application")
    task = Task.create("Implement API", "Create endpoint", capability_id="api.example")
    state.tasks = [task]

    coder_tool = runner.tools.run_command([sys.executable, "app.py"])
    validator_result, _validation, record = runner._execute_validator(
        state, task, [sys.executable, "-m", "compileall", "-q", "app.py"],
    )
    runner._retry_or_block(
        task,
        state,
        runner._validator_failure_reason(validator_result, record),
        source_validator_run_id=str(record["validator_run_id"]),
    )

    blocked = next(event for event in state.events if event.phase == "BLOCKED_ROOT_TASK")
    assert coder_tool.stderr == "LONG_RUNNING_COMMAND_REQUIRES_MANAGED_PROCESS"
    assert "LONG_RUNNING_COMMAND_REQUIRES_MANAGED_PROCESS" not in blocked.message
    assert str(record["validator_run_id"]) in blocked.message
