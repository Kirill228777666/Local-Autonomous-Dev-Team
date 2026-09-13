import subprocess
import sys
from pathlib import Path

from autodev.agents import RoleAgents
from autodev.metrics import metrics
from autodev.models import ProjectState, Task, TaskStatus
from autodev.orchestrator import AutonomousRunner
from autodev.providers import AgentReply, ScriptedProvider
from autodev.state_store import StateStore
from autodev.tools import CommandResult, WorkspaceTools
from autodev.validation import ValidationOutcome, ValidationPlanner, classify_validation_result
from autodev.repair import FailureClass, route_failure
from autodev.research import WebResearch


def _repository(path: Path) -> None:
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "AutoDev Test"], cwd=path, check=True)
    (path / ".gitignore").write_text(".autodev/\n", encoding="utf-8")
    subprocess.run(["git", "add", ".gitignore"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=path, check=True, capture_output=True)


def _state_for(path: Path) -> ProjectState:
    state = ProjectState.create("Build a generic Python application")
    state.environment = {"execution_context": {"python_interpreter": sys.executable}}
    state.project_contract = {"version": 1, "architecture": {"app_factory": False}}
    return state


def test_real_controller_runs_focused_repair_and_exact_validator(tmp_path: Path) -> None:
    _repository(tmp_path)
    (tmp_path / "module.py").write_text("def broken(:\n", encoding="utf-8")
    provider = ScriptedProvider({
        "CODER": [
            AgentReply({"actions": [{"kind": "write_file", "path": "module.py", "content": "def broken(:\n"}]}),
            AgentReply({"actions": [{"kind": "write_file", "path": "module.py", "content": "def fixed():\n    return 1\n"}]}),
        ],
        "REVIEWER": [AgentReply({"approved": True, "reasons": []})],
    })
    runner = AutonomousRunner(tmp_path, StateStore(tmp_path), WorkspaceTools(tmp_path), provider)
    state = _state_for(tmp_path)
    task = Task.create("Implement module", "Implement a small Python capability", capability_id="generic.module")
    state.tasks = [task]

    runner._run_task(state, task)

    assert task.status is TaskStatus.DONE
    assert any(event.phase == "FOCUSED_REPAIR_ATTEMPT" for event in state.events)
    assert any(event.phase == "FOCUSED_REPAIR_SUCCESS" for event in state.events)
    assert state.repair_memory
    assert any(event.phase == "EXACT_VALIDATOR_RERUN" for event in state.events)
    observed = metrics(state)
    assert observed["repair_failure_packets_created"] == 1
    assert observed["repair_fingerprints_seen"] == 1
    assert observed["focused_repairs_attempted"] == 1
    assert observed["focused_repairs_succeeded"] == 1


def test_real_controller_repairs_run14_backend_create_app_action_without_new_attempt(tmp_path: Path) -> None:
    _repository(tmp_path)
    (tmp_path / "backend").mkdir()
    (tmp_path / "backend" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "backend" / "app.py").write_text("app = object()\ndef init_db(): pass\n", encoding="utf-8")
    generated = """import unittest
from backend.app import create_app, init_db

class ContractTest(unittest.TestCase):
    def test_module_app(self):
        self.assertIsNotNone(create_app())
"""
    provider = ScriptedProvider({
        "CODER": [AgentReply({"actions": [{"kind": "write_file", "path": "tests/test_generated.py", "content": generated}]})],
        "REVIEWER": [AgentReply({"approved": True, "reasons": []})],
    })
    runner = AutonomousRunner(tmp_path, StateStore(tmp_path), WorkspaceTools(tmp_path), provider)
    state = _state_for(tmp_path)
    task = Task.create("Write backend tests", "Write focused backend tests", capability_id="generic.tests")
    state.tasks = [task]

    runner._run_task(state, task)

    repaired = (tmp_path / "tests" / "test_generated.py").read_text(encoding="utf-8")
    assert task.attempts == 1
    assert task.status is TaskStatus.DONE
    assert "from backend.app import app, init_db" in repaired
    assert "create_app" not in repaired
    assert any(event.phase == "CONTRACT_CONFLICT_RESOLVED" for event in state.events)


def test_non_test_capability_does_not_run_future_unittest_discovery(tmp_path: Path) -> None:
    (tmp_path / "module.py").write_text("value = 1\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_future_crud.py").write_text("raise AssertionError('future')\n", encoding="utf-8")
    task = Task.create("Initialize project", "Create project structure", capability_id="generic.init")

    command = ValidationPlanner.command_for(tmp_path, task, {"execution_context": {"python_interpreter": sys.executable}}, acceptance_only=True)

    assert command is not None
    assert "unittest" not in command
    assert "compileall" in command


def test_acceptance_only_pytest_targets_current_capability_tests_not_future_suite(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_api.py").write_text("def test_api(): pass\n", encoding="utf-8")
    (tmp_path / "tests" / "test_future_ui.py").write_text("def test_future_ui(): pass\n", encoding="utf-8")
    task = Task.create("Implement API", "Implement API endpoint", capability_id="backend.api")

    command = ValidationPlanner.command_for(tmp_path, task, {"execution_context": {"python_interpreter": sys.executable}}, acceptance_only=True)

    assert command is not None
    assert "pytest" in command
    assert "tests/test_api.py" in command
    assert "tests/test_future_ui.py" not in command


def test_test_state_evidence_is_classified_separately_from_application_logic(tmp_path: Path) -> None:
    missing_schema = classify_validation_result(CommandResult(1, "", "sqlite3.OperationalError: no such table: notes"), [sys.executable, "-m", "unittest"], tmp_path)
    leaked_rows = classify_validation_result(CommandResult(1, "", "AssertionError: 8 != 1\nRan 11 tests"), [sys.executable, "-m", "unittest"], tmp_path)

    assert missing_schema.kind is ValidationOutcome.TEST_STATE_ISOLATION_FAILURE
    assert leaked_rows.kind is ValidationOutcome.TEST_STATE_ISOLATION_FAILURE
    assert route_failure(missing_schema.kind) is FailureClass.TEST_STATE_ISOLATION_FAILURE


def test_capability_validator_identity_is_persisted_and_reused_after_repair(tmp_path: Path) -> None:
    _repository(tmp_path)
    (tmp_path / "module.py").write_text("def broken(:\n", encoding="utf-8")
    provider = ScriptedProvider({
        "CODER": [
            AgentReply({"actions": []}),
            AgentReply({"actions": [{"kind": "write_file", "path": "module.py", "content": "value = 1\n"}]}),
        ],
        "REVIEWER": [AgentReply({"approved": True, "reasons": []})],
    })
    runner = AutonomousRunner(tmp_path, StateStore(tmp_path), WorkspaceTools(tmp_path), provider)
    state = _state_for(tmp_path)
    task = Task.create("Implement module", "Implement Python module", capability_id="generic.module")
    state.tasks = [task]

    runner._run_task(state, task)

    assert task.acceptance_validator["scope"] == "acceptance"
    assert task.acceptance_validator["capability_id"] == "generic.module"
    assert task.acceptance_validator["command"]
    assert task.acceptance_validator["validator_id"]


def test_real_controller_dependency_api_failure_researches_then_repairs(tmp_path: Path) -> None:
    _repository(tmp_path)
    command = [sys.executable, "-c", "from pathlib import Path; exec(\"raise AttributeError(\\\"Engine has no attribute has_table\\\")\") if not Path('fixed.txt').exists() else None"]
    provider = ScriptedProvider({
        "CODER": [
            AgentReply({"actions": [{"kind": "write_file", "path": "seed.txt", "content": "seed"}]}),
            AgentReply({"actions": [{"kind": "write_file", "path": "fixed.txt", "content": "fixed"}]}),
        ],
        "TESTER": [AgentReply({"command": command})],
        "REVIEWER": [AgentReply({"approved": True, "reasons": []})],
    })
    research = WebResearch(fetch=lambda _url: "<main>Use SQLAlchemy inspection API.</main>")
    runner = AutonomousRunner(tmp_path, StateStore(tmp_path), WorkspaceTools(tmp_path), provider, research=research)
    runner._dependency_versions = lambda *_args: {"SQLAlchemy": "2.0.43"}  # type: ignore[method-assign]
    state = _state_for(tmp_path)
    task = Task.create("Database compatibility", "Repair a public database API", capability_id="generic.database")
    state.tasks = [task]

    runner._run_task(state, task)

    observed = metrics(state)
    assert task.status is TaskStatus.DONE
    assert task.last_repair_packet["research_evidence"]
    assert observed["research_escalations"] == 1
    assert observed["repair_attempts_with_research"] == 1
    assert observed["research_backed_repair_successes"] == 1
