import subprocess
import sys
from collections import Counter
from pathlib import Path

from autodev.models import ProjectState, Task, TaskStatus
from autodev.orchestrator import AutonomousRunner
from autodev.providers import AgentReply, AgentRequest, ScriptedProvider
from autodev.state_store import StateStore
from autodev.tools import WorkspaceTools


def repository(path: Path) -> None:
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=path, check=True)
    (path / ".gitignore").write_text(".autodev/\n", encoding="utf-8")
    subprocess.run(["git", "add", ".gitignore"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=path, check=True, capture_output=True, text=True)


class RecordingProvider(ScriptedProvider):
    def __init__(self, replies: dict[str, list[AgentReply]]) -> None:
        super().__init__(replies)
        self.calls: Counter[str] = Counter()

    def complete(self, request: AgentRequest) -> AgentReply:
        self.calls[request.role] += 1
        return super().complete(request)


def runner_with_task(path: Path, provider: RecordingProvider, task: Task) -> tuple[AutonomousRunner, ProjectState]:
    repository(path)
    state = ProjectState.create("Build a small generic application")
    state.tasks = [task]
    store = StateStore(path)
    store.save(state)
    return AutonomousRunner(path, store, WorkspaceTools(path), provider), state


def test_normal_success_selects_task_without_manager_call(tmp_path: Path) -> None:
    provider = RecordingProvider({
        "CODER": [AgentReply({"actions": [{"kind": "write_file", "path": "result.txt", "content": "ok"}]})],
        "TESTER": [AgentReply({"command": ["python", "-c", "assert open('result.txt').read() == 'ok'"]})],
        "REVIEWER": [AgentReply({"approved": True})],
    })
    runner, _ = runner_with_task(tmp_path, provider, Task.create("Result", "Create result.txt"))

    state = runner.run(max_cycles=1)

    assert state.tasks[0].status is TaskStatus.DONE
    assert provider.calls["MANAGER"] == 0
    assert provider.calls["ARCHITECT"] == 0
    assert len(state.tasks) == 1


def test_noop_failed_acceptance_recovers_inside_same_attempt_without_manager(tmp_path: Path) -> None:
    provider = RecordingProvider({
        "CODER": [
            AgentReply({"actions": []}),
            AgentReply({"actions": [{"kind": "write_file", "path": "result.txt", "content": "fixed"}]}),
        ],
        "TESTER": [AgentReply({"command": ["python", "-c", "assert open('result.txt').read() == 'fixed'"]})],
        "REVIEWER": [AgentReply({"approved": True})],
    })
    runner, _ = runner_with_task(tmp_path, provider, Task.create("Result", "Create result.txt"))

    state = runner.run(max_cycles=1)

    assert state.tasks[0].status is TaskStatus.DONE
    assert state.tasks[0].attempts == 1
    assert provider.calls == Counter({"CODER": 2, "TESTER": 1, "REVIEWER": 1})
    assert any(event.phase == "NOOP_EVIDENCE_REPROMPT" for event in state.events)


def test_repeated_real_failure_calls_architect_once_and_never_creates_repair_task(tmp_path: Path) -> None:
    provider = RecordingProvider({
        "CODER": [
            AgentReply({"actions": [{"kind": "write_file", "path": "result.txt", "content": "bad-a"}]}),
            AgentReply({"actions": [{"kind": "write_file", "path": "result.txt", "content": "bad-b"}]}),
            AgentReply({"actions": [{"kind": "write_file", "path": "result.txt", "content": "good"}]}),
        ],
        "TESTER": [
            AgentReply({"command": ["python", "-c", "assert open('result.txt').read() == 'good'"]}),
            AgentReply({"command": ["python", "-c", "assert open('result.txt').read() == 'good'"]}),
            AgentReply({"command": ["python", "-c", "assert open('result.txt').read() == 'good'"]}),
        ],
        "ARCHITECT": [AgentReply({"diagnosis": "Write the exact accepted value."})],
        "REVIEWER": [AgentReply({"approved": True})],
    })
    runner, _ = runner_with_task(tmp_path, provider, Task.create("Result", "Create result.txt"))

    state = runner.run(max_cycles=3)

    assert state.tasks[0].status is TaskStatus.DONE
    assert len(state.tasks) == 1
    assert provider.calls["MANAGER"] == 0
    assert provider.calls["ARCHITECT"] == 1
    assert state.tasks[0].strategy_generation == 1
    assert not any(task.title.startswith("Repair:") for task in state.tasks)


def test_exhausted_changed_strategy_blocks_root_without_repair_chain(tmp_path: Path) -> None:
    provider = RecordingProvider({
        "CODER": [
            AgentReply({"actions": [{"kind": "write_file", "path": "result.txt", "content": value}]})
            for value in ("a", "b", "c", "d")
        ],
        "TESTER": [
            AgentReply({"command": ["python", "-c", "raise SystemExit(1)"]}) for _ in range(4)
        ],
        "ARCHITECT": [AgentReply({"diagnosis": "Try one alternate strategy."})],
    })
    runner, _ = runner_with_task(tmp_path, provider, Task.create("Impossible", "Meet impossible acceptance"))

    state = runner.run(max_cycles=6)

    assert len(state.tasks) == 1
    assert state.tasks[0].status is TaskStatus.BLOCKED
    assert provider.calls["MANAGER"] == 0
    assert provider.calls["ARCHITECT"] == 1
    assert any("BLOCKED_ROOT_TASK" in error for error in state.tasks[0].errors)


def test_empty_deterministic_unittest_discovery_is_validation_unavailable_without_harness_retry(tmp_path: Path) -> None:
    provider = RecordingProvider({
        "CODER": [AgentReply({"actions": [{"kind": "write_file", "path": "tests/test_empty.py", "content": "# intentionally empty\n"}]})],
    })
    runner, state = runner_with_task(tmp_path, provider, Task.create("Backend tests", "Implement backend tests"))
    state.environment = {"execution_context": {"python_interpreter": sys.executable}}
    runner.store.save(state)

    state = runner.run(max_cycles=1)

    task = state.tasks[0]
    assert state.status == "BLOCKED"
    assert task.status is TaskStatus.PENDING
    assert task.failures_in_strategy == 0
    assert any(event.phase == "VALIDATION_NO_TESTS" for event in state.events)
    assert provider.calls["TESTER"] == 0
    assert provider.calls["MANAGER"] == 0


def test_import_failure_inside_discovered_test_does_not_trigger_harness_regeneration(tmp_path: Path) -> None:
    provider = RecordingProvider({
        "CODER": [AgentReply({"actions": [
            {"kind": "write_file", "path": "app.py", "content": "value = 1\n"},
            {"kind": "write_file", "path": "tests/test_api.py", "content": "import unittest\nfrom app import missing_value\n"},
        ]})],
    })
    runner, state = runner_with_task(tmp_path, provider, Task.create("Backend API", "Implement endpoint"))
    state.environment = {"execution_context": {"python_interpreter": sys.executable}}
    runner.store.save(state)

    state = runner.run(max_cycles=1)

    task = state.tasks[0]
    assert task.status is TaskStatus.PENDING
    assert task.failures_in_strategy == 1
    assert any(event.phase == "VALIDATION_IMPORT_OR_ENVIRONMENT_ERROR" for event in state.events)
    assert not any(event.phase == "HARNESS_FAILURE" for event in state.events)
    assert provider.calls["TESTER"] == 0
    assert provider.calls["MANAGER"] == 0


def test_repeated_invalid_tester_command_is_bounded_with_validation_rollback_reason(tmp_path: Path) -> None:
    provider = RecordingProvider({
        "CODER": [AgentReply({"actions": [{"kind": "write_file", "path": "result.txt", "content": "ok"}]})],
        "TESTER": [
            AgentReply({"command": [sys.executable, "-c", "import re; with open('result.txt'): pass"]}),
            AgentReply({"command": [sys.executable, "-c", "import re; with open('result.txt'): pass"]}),
            AgentReply({"command": [sys.executable, "-c", "import re; with open('result.txt'): pass"]}),
        ],
    })
    runner, _ = runner_with_task(tmp_path, provider, Task.create("Result", "Create result.txt"))

    state = runner.run(max_cycles=1)

    task = state.tasks[0]
    assert task.status is TaskStatus.BLOCKED
    assert task.rollback_reasons == {"validation_command_invalid": 1}
    assert provider.calls["MANAGER"] == 0
