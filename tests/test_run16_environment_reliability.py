from __future__ import annotations

import sys
from pathlib import Path

from autodev.agents import RoleAgents
from autodev.capabilities import repair_generated_test_for_contract
from autodev.environment import EnvironmentManager, FailureKind, classify_failure
from autodev.models import ProjectState, Task
from autodev.metrics import metrics
from autodev.orchestrator import AutonomousRunner
from autodev.providers import ScriptedProvider
from autodev.research import WebResearch
from autodev.state_store import StateStore
from autodev.repair import FailureClass
from autodev.tools import CommandResult
from autodev.validation import ValidationOutcome, classify_validation_result


class RecordingTools:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.commands: list[list[str]] = []

    def run_command(self, command: list[str]) -> CommandResult:
        self.commands.append(command)
        if "venv" in command:
            executable = self.workspace / ".venv" / "Scripts" / "python.exe"
            executable.parent.mkdir(parents=True, exist_ok=True)
            executable.write_text("", encoding="utf-8")
        return CommandResult(0, "ok", "")


class MissingManifestDependencyTools(RecordingTools):
    def __init__(self, workspace: Path) -> None:
        super().__init__(workspace)
        self.installed = False

    def run_command(self, command: list[str]) -> CommandResult:
        self.commands.append(command)
        if "venv" in command:
            executable = self.workspace / ".venv" / "Scripts" / "python.exe"
            executable.parent.mkdir(parents=True, exist_ok=True)
            executable.write_text("", encoding="utf-8")
            return CommandResult(0, "", "")
        joined = " ".join(command)
        if "importlib.metadata" in joined:
            return CommandResult(0 if self.installed else 1, "3.1.0" if self.installed else "", "not found" if not self.installed else "")
        if command[-3:] == ["pip", "install", "-r"] or " -m pip install -r requirements.txt" in joined:
            self.installed = True
            return CommandResult(0, "installed", "")
        if "import flask" in joined:
            return CommandResult(0 if self.installed else 1, "", "not found" if not self.installed else "")
        return CommandResult(0, "ok", "")


class IncompatibleHostDependencyTools(MissingManifestDependencyTools):
    def run_command(self, command: list[str]) -> CommandResult:
        self.commands.append(command)
        if "venv" in command:
            executable = self.workspace / ".venv" / "Scripts" / "python.exe"
            executable.parent.mkdir(parents=True, exist_ok=True)
            executable.write_text("", encoding="utf-8")
            return CommandResult(0, "", "")
        joined = " ".join(command)
        if "importlib.metadata" in joined:
            return CommandResult(0, "3.1.0" if self.installed else "2.0.0", "")
        if " -m pip install -r requirements.txt" in joined:
            self.installed = True
            return CommandResult(0, "installed", "")
        return CommandResult(0, "ok", "")


_FLASK_FAILED_TEST = """E
ERROR: test_app (unittest.loader._FailedTest.test_app)
ImportError: Failed to import test module: test_app
ModuleNotFoundError: No module named 'flask'
"""


def test_failed_test_wrapping_missing_flask_is_project_dependency(tmp_path: Path) -> None:
    failure = classify_failure(CommandResult(1, "", _FLASK_FAILED_TEST), [sys.executable, "-m", "unittest"], tmp_path)

    assert failure.kind is FailureKind.MISSING_PROJECT_DEPENDENCY
    assert failure.module == "flask"


def test_failed_test_wrapping_missing_flask_is_environment_validation_not_test_bug(tmp_path: Path) -> None:
    validation = classify_validation_result(CommandResult(1, "", _FLASK_FAILED_TEST), [sys.executable, "-m", "unittest"], tmp_path)

    assert validation.kind is ValidationOutcome.IMPORT_OR_ENVIRONMENT_ERROR


def test_missing_local_module_in_generated_test_is_test_implementation_bug(tmp_path: Path) -> None:
    (tmp_path / "backend").mkdir()
    (tmp_path / "backend" / "__init__.py").write_text("", encoding="utf-8")
    detail = "unittest.loader._FailedTest.test_api\nModuleNotFoundError: No module named 'backend'"

    validation = classify_validation_result(CommandResult(1, "", detail), [sys.executable, "-m", "unittest"], tmp_path)

    assert validation.kind is ValidationOutcome.TEST_IMPLEMENTATION_BUG


def test_missing_local_module_while_launching_application_is_application_import_error(tmp_path: Path) -> None:
    (tmp_path / "backend").mkdir()
    (tmp_path / "backend" / "__init__.py").write_text("", encoding="utf-8")
    detail = "ModuleNotFoundError: No module named 'backend'"

    validation = classify_validation_result(CommandResult(1, "", detail), [sys.executable, "app.py"], tmp_path)

    assert validation.kind is ValidationOutcome.APPLICATION_IMPORT_ERROR


def test_new_project_venv_requests_system_site_packages(tmp_path: Path) -> None:
    tools = RecordingTools(tmp_path)
    manager = EnvironmentManager(tmp_path, tools)

    assert manager.ensure_python_environment() is True

    assert any(command[-3:] == ["venv", "--system-site-packages", str(tmp_path / ".venv")] for command in tools.commands)


def test_read_file_action_has_type_specific_schema_without_content() -> None:
    RoleAgents._valid_actions({"actions": [{"kind": "read_file", "path": "app.py"}]})


def test_command_action_alias_is_normalized_without_irrelevant_content() -> None:
    payload = {"actions": [{"action": "run_command", "command": ["python", "-m", "compileall"], "content": ""}]}

    RoleAgents._valid_actions(payload)

    assert payload["actions"][0]["kind"] == "run_command"


def test_write_file_without_content_stays_invalid() -> None:
    try:
        RoleAgents._valid_actions({"actions": [{"kind": "write_file", "path": "app.py"}]})
    except ValueError as error:
        assert "content" in str(error)
    else:  # pragma: no cover - makes the intended strictness explicit
        raise AssertionError("write_file without content must be rejected")


def test_contract_repair_handles_module_qualified_factory_call() -> None:
    generated = "import app\nclient = app.create_app().test_client()\n"
    contract = {"architecture": {"app_factory": False, "application_symbol": "app"}}

    assert repair_generated_test_for_contract(contract, generated) == "import app\nclient = app.app.test_client()\n"


def test_contract_repair_handles_factory_import_alias() -> None:
    generated = "from app import create_app as application\nclient = application().test_client()\n"
    contract = {"architecture": {"app_factory": False, "application_symbol": "app"}}

    assert repair_generated_test_for_contract(contract, generated) == "from app import app\nclient = app.test_client()\n"


def test_dependency_api_mismatch_uses_contract_framework_for_official_research(tmp_path: Path) -> None:
    research = WebResearch(fetch=lambda _url: "<main>official Flask API</main>")
    runner = AutonomousRunner(tmp_path, StateStore(tmp_path), RecordingTools(tmp_path), ScriptedProvider({}), research=research)
    state = ProjectState.create("Build backend")
    state.project_contract = {"architecture": {"backend_framework": "Flask"}}
    state.environment = {"execution_context": {"python_interpreter": sys.executable}}
    task = Task.create("Backend tests", "Validate application")
    result = CommandResult(1, "", "AttributeError: module 'app' has no attribute 'db'")

    runner._capture_repair_evidence(state, task, [sys.executable, "-m", "unittest"], ValidationOutcome.DEPENDENCY_API_MISMATCH, result)

    assert task.last_repair_packet["failure_class"] == FailureClass.DEPENDENCY_API_MISMATCH.value
    assert task.last_repair_packet["research_evidence"]
    assert any(event.agent == "RESEARCH" and event.phase == "REQUEST" for event in state.events)


def test_missing_project_dependency_is_repaired_before_coder_with_project_python(tmp_path: Path) -> None:
    tools = RecordingTools(tmp_path)
    runner = AutonomousRunner(tmp_path, StateStore(tmp_path), tools, ScriptedProvider({}))
    state = ProjectState.create("Build backend")
    task = Task.create("Backend tests", "Import the backend")
    original = CommandResult(1, "", "ModuleNotFoundError: No module named 'flask'")

    outcome = runner._repair_environment_failure(state, task, [sys.executable, "-m", "unittest"], original)

    assert outcome.attempted is True
    assert outcome.succeeded is True
    project_python = str(tmp_path / ".venv" / "Scripts" / "python.exe")
    assert any(command[:4] == [project_python, "-m", "pip", "install"] for command in tools.commands)
    assert any(event.agent == "ENVIRONMENT" and event.phase == "REPAIR_SUCCEEDED" for event in state.events)


def test_declared_dependency_repairs_from_manifest_in_project_environment(tmp_path: Path) -> None:
    (tmp_path / "requirements.txt").write_text("Flask>=3.0\n", encoding="utf-8")
    tools = RecordingTools(tmp_path)
    manager = EnvironmentManager(tmp_path, tools)
    failure = classify_failure(CommandResult(1, "", "ModuleNotFoundError: No module named 'flask'"), [sys.executable, "-m", "unittest"], tmp_path)

    assert manager.repair(failure) is True

    project_python = str(tmp_path / ".venv" / "Scripts" / "python.exe")
    assert [project_python, "-m", "pip", "install", "-r", "requirements.txt"] in tools.commands
    assert (tmp_path / "requirements.txt").read_text(encoding="utf-8") == "Flask>=3.0\n"


def test_declared_dependency_is_provisioned_in_project_venv_before_agent_work(tmp_path: Path) -> None:
    (tmp_path / "requirements.txt").write_text("Flask>=3.0\n", encoding="utf-8")
    tools = MissingManifestDependencyTools(tmp_path)
    manager = EnvironmentManager(tmp_path, tools)

    provision = manager.ensure_declared_dependencies()

    assert provision.attempted is True
    assert provision.succeeded is True
    assert provision.packages == ["flask"]
    project_python = str(tmp_path / ".venv" / "Scripts" / "python.exe")
    assert [project_python, "-m", "pip", "install", "-r", "requirements.txt"] in tools.commands


def test_runner_provisions_manifest_dependencies_before_task_selection(tmp_path: Path) -> None:
    (tmp_path / "requirements.txt").write_text("Flask>=3.0\n", encoding="utf-8")
    tools = MissingManifestDependencyTools(tmp_path)
    store = StateStore(tmp_path)
    state = ProjectState.create("Build a Python backend")
    state.tasks = [Task.create("Deferred", "Do not execute in this smoke")]
    store.save(state)
    runner = AutonomousRunner(tmp_path, store, tools, ScriptedProvider({}))

    runner.run(max_cycles=0)

    saved = store.load()
    assert saved is not None
    assert any(event.agent == "ENVIRONMENT" and event.phase == "DEPENDENCY_PROVISIONED" for event in saved.events)
    assert metrics(saved)["dependency_installs"] == 1


def test_incompatible_host_dependency_is_installed_instead_of_reused(tmp_path: Path) -> None:
    (tmp_path / "requirements.txt").write_text("Flask>=3.0\n", encoding="utf-8")
    tools = IncompatibleHostDependencyTools(tmp_path)
    manager = EnvironmentManager(tmp_path, tools)

    provision = manager.ensure_declared_dependencies()

    assert provision.attempted is True
    assert provision.succeeded is True
    project_python = str(tmp_path / ".venv" / "Scripts" / "python.exe")
    assert [project_python, "-m", "pip", "install", "-r", "requirements.txt"] in tools.commands
