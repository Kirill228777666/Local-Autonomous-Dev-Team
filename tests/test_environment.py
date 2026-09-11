from pathlib import Path

from autodev.environment import EnvironmentManager, FailureKind, classify_failure, validate_readme
from autodev.tools import CommandResult
from autodev.models import ProjectState
from autodev.state_store import StateStore
from autodev.models import Task, TaskStatus
from autodev.orchestrator import AutonomousRunner
from autodev.providers import ScriptedProvider
from autodev.tools import WorkspaceTools


class RecordingTools:
    def __init__(self, replies: list[CommandResult] | None = None) -> None:
        self.commands: list[list[str]] = []
        self.replies = replies or []

    def run_command(self, command: list[str]) -> CommandResult:
        self.commands.append(command)
        return self.replies.pop(0) if self.replies else CommandResult(0, "ok", "")


def test_classifier_identifies_environment_failures_before_llm_repair() -> None:
    assert classify_failure(CommandResult(1, "", "ModuleNotFoundError: No module named 'flask'"), ["py", "-3", "app.py"]).kind is FailureKind.MISSING_PYTHON_DEPENDENCY
    assert classify_failure(CommandResult(127, "", "command not found: npm"), ["npm", "install"]).kind is FailureKind.MISSING_EXECUTABLE
    assert classify_failure(CommandResult(1, "", "ModuleNotFoundError: No module named 'app'"), ["py", "-3", "-m", "pytest"]).kind is FailureKind.IMPORT_PATH


def test_missing_flask_is_persisted_installed_in_project_venv_and_retried(tmp_path: Path) -> None:
    tools = RecordingTools()
    manager = EnvironmentManager(tmp_path, tools, allow_project_dependency_install=True)
    failure = classify_failure(CommandResult(1, "", "ModuleNotFoundError: No module named 'flask'"), ["py", "-3", "app.py"])

    repaired = manager.repair(failure)

    assert repaired is True
    assert "flask" in (tmp_path / "requirements.txt").read_text(encoding="utf-8").lower()
    assert any(command[1:4] == ["-m", "pip", "install"] and command[-1].lower() == "flask" for command in tools.commands)
    assert any(command[:4] == ["py", "-3", "-m", "venv"] and ".venv" in command[-1] for command in tools.commands)


def test_missing_npm_is_policy_aware_and_never_attempts_npm_install(tmp_path: Path) -> None:
    tools = RecordingTools()
    manager = EnvironmentManager(tmp_path, tools, allow_project_dependency_install=True, allow_system_package_install=False)
    failure = classify_failure(CommandResult(127, "", "command not found: npm"), ["npm", "install"])

    assert manager.repair(failure) is False
    assert manager.last_diagnostic == "Node.js/npm required by chosen architecture but system installation is disabled."
    assert not any(command[0] == "npm" for command in tools.commands)


def test_readme_validator_checks_instructions_without_running_them(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Notes\n\n## Install\npy -3 -m pip install -r requirements.txt\n\n## Run\npy -3 app.py\n\n## Test\npy -3 -m pytest\n", encoding="utf-8")
    result = validate_readme(tmp_path)
    assert result.passed is True


def test_readme_validator_rejects_missing_test_instructions(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Notes\nInstall with pip. Run app.py.\n", encoding="utf-8")
    assert validate_readme(tmp_path).passed is False


def test_russian_state_projections_and_json_round_trip_as_utf8(tmp_path: Path) -> None:
    text = "Создай локальное приложение Notes\nНе удается найти указанный файл"
    state = ProjectState.create(text)
    state.record_event("SYSTEM", "INFO", text)
    StateStore(tmp_path).save(state)
    assert StateStore(tmp_path).load().original_spec == text  # type: ignore[union-attr]
    assert text in (tmp_path / ".autodev" / "project_spec.md").read_text(encoding="utf-8")
    assert text in (tmp_path / ".autodev" / "activity.log").read_text(encoding="utf-8")


def test_node_free_pivot_supersedes_obsolete_npm_build_task(tmp_path: Path) -> None:
    frontend = Task.create("Frontend Development", "Build React UI")
    build = Task.create("Frontend Build Process", "Run npm build")
    state = ProjectState.create("Build Notes")
    state.tasks = [frontend, build]
    runner = AutonomousRunner(tmp_path, StateStore(tmp_path), WorkspaceTools(tmp_path), ScriptedProvider({}))

    runner._pivot_to_static_frontend(state, frontend)

    assert frontend.status is TaskStatus.SUPERSEDED
    assert build.status is TaskStatus.SUPERSEDED
    assert any("Static frontend" in task.title for task in state.tasks)
