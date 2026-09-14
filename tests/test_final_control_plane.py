from __future__ import annotations

import subprocess
from pathlib import Path

from autodev.agents import RoleAgents
from autodev.capabilities import build_project_contract, contract_policy_violation, find_contract_policy_violations
from autodev.environment import EnvironmentManager, Failure, FailureKind
from autodev.environment import validate_readme
from autodev.models import ProjectState, Task, TaskStatus
from autodev.metrics import metrics
from autodev.orchestrator import AutonomousRunner
from autodev.providers import AgentReply, AgentRequest, ProviderError, ScriptedProvider
from autodev.state_store import StateStore
from autodev.tools import CommandResult, WorkspaceTools
from autodev.validation import ValidationOutcome
from autodev.validation import classify_validation_result


def _repository(path: Path) -> None:
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "AutoDev Test"], cwd=path, check=True)
    (path / ".gitignore").write_text(".autodev/\n", encoding="utf-8")
    subprocess.run(["git", "add", ".gitignore"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=path, check=True, capture_output=True)


class RecordingTools:
    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    def run_command(self, command: list[str]) -> CommandResult:
        self.commands.append(command)
        return CommandResult(0, "installed", "")


class TruncatedThenValidProvider(ScriptedProvider):
    def __init__(self) -> None:
        super().__init__({"CODER": [AgentReply({"actions": [], "task_status": "ready_for_validation"})]})
        self.calls = 0

    def complete(self, request: AgentRequest) -> AgentReply:
        if request.role == "CODER" and self.calls == 0:
            self.calls += 1
            raise ProviderError("Unterminated string starting at character 46000")
        self.calls += 1
        return super().complete(request)


def test_stdlib_only_contract_rejects_generated_external_import() -> None:
    contract = build_project_contract(
        "Build a Python service using the Python standard library ONLY. No external Python dependencies.",
        {"npm": {"available": False}},
    )

    violation = contract_policy_violation(contract, "service.py", "import imaginary_framework\n")

    assert contract["technology_constraints"]["stdlib_only_python"] is True
    assert violation is not None
    assert violation.code == "CONTRACT_POLICY_VIOLATION"
    assert "imaginary_framework" in violation.message


def test_forbidden_dependency_is_not_installed_by_environment_manager(tmp_path: Path) -> None:
    tools = RecordingTools()
    manager = EnvironmentManager(tmp_path, tools)
    failure = Failure(FailureKind.MISSING_PROJECT_DEPENDENCY, ["python", "app.py"], "No module", "imaginary_framework")
    contract = {"technology_constraints": {"stdlib_only_python": True}}

    assert manager.repair(failure, contract=contract) is False
    assert tools.commands == []
    assert "forbidden" in manager.last_diagnostic.lower()


def test_existing_forbidden_import_is_detected_before_environment_repair(tmp_path: Path) -> None:
    (tmp_path / "application.py").write_text("import imaginary_framework\n", encoding="utf-8")
    contract = build_project_contract("Python standard library only; no external Python dependencies", {})

    findings = find_contract_policy_violations(contract, tmp_path)

    assert findings[0][0] == "application.py"
    assert findings[0][1].subject == "imaginary_framework"


def test_forbidden_import_routes_to_coder_not_environment_repair(tmp_path: Path) -> None:
    _repository(tmp_path)
    (tmp_path / "application.py").write_text("import imaginary_framework\n", encoding="utf-8")
    runner = AutonomousRunner(tmp_path, StateStore(tmp_path), WorkspaceTools(tmp_path), ScriptedProvider({}))
    state = ProjectState.create("Python standard library only; no external Python dependencies")
    state.project_contract = build_project_contract(state.original_spec, {})
    task = Task.create("Implementation", "Implement the service")
    state.tasks = [task]

    outcome = runner._repair_environment_failure(
        state, task, ["python", "application.py"], CommandResult(1, "", "ModuleNotFoundError: No module named 'imaginary_framework'")
    )

    assert outcome.attempted is False
    assert any(event.phase == "CONTRACT_POLICY_VIOLATION" for event in state.events)
    assert not any(event.phase == "REPAIR_ATTEMPT" for event in state.events)


def test_mixed_validator_failure_keeps_application_failure_evidence(tmp_path: Path) -> None:
    result = CommandResult(
        1,
        "",
        "ModuleNotFoundError: No module named 'forbidden_framework'\n"
        "AssertionError: 0 != 1\n"
        "AttributeError: repository has no attribute get_by_id",
    )

    classified = classify_validation_result(result, ["python", "-m", "unittest", "discover"], tmp_path)

    assert classified.kind is ValidationOutcome.IMPORT_OR_ENVIRONMENT_ERROR
    assert ValidationOutcome.APPLICATION_FAILURE in classified.secondary_kinds
    assert ValidationOutcome.DEPENDENCY_API_MISMATCH in classified.secondary_kinds


def test_validator_record_has_explicit_validation_strength(tmp_path: Path) -> None:
    _repository(tmp_path)
    runner = AutonomousRunner(tmp_path, StateStore(tmp_path), WorkspaceTools(tmp_path), ScriptedProvider({}))
    state = ProjectState.create("Build a service")
    task = Task.create("Compile source", "Compile a utility")
    _, record = runner._record_validator_result(
        state, task, ["python", "-m", "compileall", "-q", "app.py"], CommandResult(0, "", "")
    )

    assert record["validation_strength"] == "STRUCTURAL"


def test_control_plane_metrics_are_derived_from_durable_events() -> None:
    state = ProjectState.create("Build service")
    state.record_event("CODER", "CONTRACT_POLICY_VIOLATION", "forbidden import")
    state.record_event("CONTROLLER", "WAITING_ON_DEPENDENCY", "blocked parent")
    state.record_event("CODER", "BATCH_CONTINUATION", "next batch")
    state.validator_runs.append({"secondary_failure_classes": ["VALIDATION_APPLICATION_FAIL"]})

    observed = metrics(state)

    assert observed["contract_policy_violations_detected"] == 1
    assert observed["tasks_waiting_on_blocked_dependencies"] == 1
    assert observed["coder_continuation_batches"] == 1
    assert observed["mixed_validation_failure_groups"] == 1


def test_blocked_hard_dependency_never_becomes_ready() -> None:
    parent = Task.create("Repository", "Build repository")
    child = Task.create("HTTP API", "Use repository", dependencies=[parent.id])
    parent.status = TaskStatus.BLOCKED
    parent.phase = "BLOCKED"
    state = ProjectState.create("Build application")
    state.tasks = [parent, child]

    assert AutonomousRunner(Path.cwd(), StateStore(Path.cwd()), WorkspaceTools(Path.cwd()), ScriptedProvider({})).controller.next_ready(state) is None


def test_missing_read_file_is_contained_as_task_failure_not_system_crash(tmp_path: Path) -> None:
    _repository(tmp_path)
    provider = ScriptedProvider({"CODER": [AgentReply({"actions": [{"kind": "read_file", "path": "missing.py"}]})]})
    runner = AutonomousRunner(tmp_path, StateStore(tmp_path), WorkspaceTools(tmp_path), provider, max_attempts=1)
    state = ProjectState.create("Build a static application")
    task = Task.create("Read source", "Inspect an expected source file")
    state.tasks = [task]

    runner._run_task(state, task)

    assert any(event.agent == "CODER" and event.phase == "TOOL_FAILED" for event in state.events)
    assert task.status in {TaskStatus.PENDING, TaskStatus.BLOCKED}
    assert state.status != "CRASHED"


def test_documentation_validator_is_rerun_by_internal_handler(tmp_path: Path) -> None:
    _repository(tmp_path)
    (tmp_path / "README.md").write_text("Install\nRun\nTest\n", encoding="utf-8")
    runner = AutonomousRunner(tmp_path, StateStore(tmp_path), WorkspaceTools(tmp_path), ScriptedProvider({}))
    state = ProjectState.create("Document a project")
    task = Task.create("README documentation", "Describe setup, run, and tests")
    task.acceptance_validator = {
        "validator_id": "readme",
        "validator_type": "internal",
        "handler": "readme",
        "command": ["README validator"],
    }

    result = runner._run_validator(state, task, ["README validator"])
    validation, record = runner._record_validator_result(state, task, ["README validator"], result)

    assert result.exit_code == 0
    assert validation.kind is ValidationOutcome.PASS
    assert record["validator_kind"] == "internal"
    assert record["execution_backend"] == "internal:readme"


def test_contract_aware_readme_does_not_invent_test_command_without_tests(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("Install the service. Run the service.", encoding="utf-8")

    result = validate_readme(tmp_path, contract={"architecture": {"test_strategy": "unittest"}})

    assert result.passed is True


def test_planned_hard_dependency_is_persisted_before_scheduler_runs(tmp_path: Path) -> None:
    _repository(tmp_path)
    provider = ScriptedProvider({"MANAGER": [AgentReply({"tasks": [
        {"title": "Data store", "description": "Create durable storage", "capability_id": "data.store"},
        {"title": "HTTP API", "description": "Use the durable storage", "capability_id": "api.http", "depends_on": ["data.store"]},
    ]})]})
    runner = AutonomousRunner(tmp_path, StateStore(tmp_path), WorkspaceTools(tmp_path), provider)
    state = ProjectState.create("Build a service")
    state.environment = {"npm": {"available": False}}
    runner._ensure_project_contract(state)

    runner._plan(state)

    parent, child = state.tasks
    assert child.dependencies == [parent.id]
    parent.status = TaskStatus.BLOCKED
    parent.phase = "BLOCKED"
    assert runner._select_next_task(state) is None


def test_dispatch_invariant_refuses_child_with_unaccepted_dependency(tmp_path: Path) -> None:
    _repository(tmp_path)
    parent = Task.create("Repository", "Build storage")
    child = Task.create("HTTP API", "Use storage", dependencies=[parent.id])
    state = ProjectState.create("Build service")
    state.tasks = [parent, child]
    runner = AutonomousRunner(tmp_path, StateStore(tmp_path), WorkspaceTools(tmp_path), ScriptedProvider({}))

    runner._run_task(state, child)

    assert child.attempts == 0
    assert child.status is TaskStatus.PENDING
    assert any(event.phase == "WAITING_ON_DEPENDENCY" for event in state.events)


def test_coder_action_batches_are_bounded_and_continue_same_attempt() -> None:
    valid = {"actions": [{"kind": "read_file", "path": "app.py"}], "task_status": "continue"}
    RoleAgents._valid_actions(valid)
    assert valid["task_status"] == "continue"

    oversized = {"actions": [{"kind": "write_file", "path": "app.py", "content": "x" * 12_001}]}
    try:
        RoleAgents._valid_actions(oversized)
    except ValueError as error:
        assert "payload" in str(error)
    else:
        raise AssertionError("oversized action batch must be rejected before tool execution")


def test_continuation_batches_execute_within_one_task_attempt(tmp_path: Path) -> None:
    _repository(tmp_path)
    provider = ScriptedProvider({
        "CODER": [
            AgentReply({"actions": [{"kind": "write_file", "path": "page.txt", "content": "first"}], "task_status": "continue"}),
            AgentReply({"actions": [{"kind": "append_file", "path": "page.txt", "content": " second"}], "task_status": "ready_for_validation"}),
        ],
        "TESTER": [AgentReply({"command": ["py", "-3", "-c", "assert open('page.txt').read() == 'first second'"]})],
        "REVIEWER": [AgentReply({"approved": True, "reasons": []})],
    })
    runner = AutonomousRunner(tmp_path, StateStore(tmp_path), WorkspaceTools(tmp_path), provider)
    state = ProjectState.create("Build a static page")
    task = Task.create("Create page", "Create a static page")
    state.tasks = [task]

    runner._run_task(state, task)

    assert task.attempts == 1
    assert task.status is TaskStatus.DONE
    assert (tmp_path / "page.txt").read_text(encoding="utf-8") == "first second"
    assert sum(event.phase == "BATCH_CONTINUATION" for event in state.events) == 1


def test_truncated_coder_protocol_recovers_with_small_next_batch(tmp_path: Path) -> None:
    _repository(tmp_path)
    provider = TruncatedThenValidProvider()
    runner = AutonomousRunner(tmp_path, StateStore(tmp_path), WorkspaceTools(tmp_path), provider)
    state = ProjectState.create("Build a page")
    task = Task.create("Page", "Create a page")

    actions = runner._request_coder_actions(state, task, "Coder implementation request")

    assert actions == []
    assert provider.calls == 2
    assert any(event.phase == "CODER_PROTOCOL_RECOVERY" for event in state.events)
