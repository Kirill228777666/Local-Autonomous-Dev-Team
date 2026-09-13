from __future__ import annotations

import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from autodev.controller import TaskController
from autodev.metrics import metrics
from autodev.models import ProjectState, Task, TaskStatus
from autodev.orchestrator import AutonomousRunner
from autodev.providers import AgentReply, AgentRequest, ScriptedProvider
from autodev.state_store import StateStore
from autodev.tools import WorkspaceTools


def repository(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
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


def make_runner(path: Path, provider: RecordingProvider, tasks: list[Task]) -> AutonomousRunner:
    repository(path)
    state = ProjectState.create("Build a generic local application")
    state.tasks = tasks
    store = StateStore(path)
    store.save(state)
    return AutonomousRunner(path, store, WorkspaceTools(path), provider)


def test_destructive_replacement_is_patched_without_second_coder_call_and_resumes_actions(tmp_path: Path) -> None:
    old = "# accepted capability\n" * 80
    desired = "# replacement retained exactly\n"
    (tmp_path / "app.py").write_text(old, encoding="utf-8")
    provider = RecordingProvider({
        "CODER": [AgentReply({"actions": [
            {"kind": "write_file", "path": "before.txt", "content": "before"},
            {"kind": "write_file", "path": "app.py", "content": desired},
            {"kind": "write_file", "path": "after.txt", "content": "after"},
            {"kind": "append_file", "path": "after.txt", "content": "+done"},
        ]})],
        "TESTER": [AgentReply({"command": [sys.executable, "-c", "assert open('before.txt').read() == 'before'; assert open('after.txt').read() == 'after+done'"]})],
        "REVIEWER": [AgentReply({"approved": True, "reasons": []})],
    })
    runner = make_runner(tmp_path, provider, [Task.create("Update source", "Apply a bounded source update")])

    state = runner.run(max_cycles=1)

    assert state.tasks[0].status is TaskStatus.DONE
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == desired
    assert provider.calls["CODER"] == 1
    assert (tmp_path / "after.txt").read_text(encoding="utf-8") == "after+done"
    assert any(event.phase == "DESTRUCTIVE_WRITE_RECOVERY_SUCCESS" for event in state.events)


def test_deterministic_patch_fails_closed_when_snapshot_is_stale(tmp_path: Path) -> None:
    tools = WorkspaceTools(tmp_path)
    tools.write_file("app.py", "old\n")
    fingerprint, old = tools.file_snapshot("app.py")
    tools.write_file("app.py", "concurrent change\n")

    assert tools.apply_deterministic_patch("app.py", fingerprint, old, "desired\n") is False
    assert tools.read_file("app.py") == "concurrent change\n"


class PatchUnavailableTools(WorkspaceTools):
    def apply_deterministic_patch(self, relative_path: str, expected_hash: str, old: str, desired: str) -> bool:
        return False


def test_patch_unavailable_uses_one_targeted_edit_fallback(tmp_path: Path) -> None:
    old = "# accepted capability\n" * 80
    desired = "# desired\n"
    (tmp_path / "app.py").write_text(old, encoding="utf-8")
    provider = RecordingProvider({
        "CODER": [
            AgentReply({"actions": [{"kind": "write_file", "path": "app.py", "content": desired}]}),
            AgentReply({"actions": [{"kind": "edit_file", "path": "app.py", "old": old, "new": desired}]}),
        ],
        "TESTER": [AgentReply({"command": [sys.executable, "-c", "assert open('app.py').read() == '# desired\\n'"]})],
        "REVIEWER": [AgentReply({"approved": True, "reasons": []})],
    })
    repository(tmp_path)
    state = ProjectState.create("Build generic app")
    state.tasks = [Task.create("Update source", "Update source")]
    StateStore(tmp_path).save(state)
    runner = AutonomousRunner(tmp_path, StateStore(tmp_path), PatchUnavailableTools(tmp_path), provider)

    state = runner.run(max_cycles=1)

    assert state.tasks[0].status is TaskStatus.DONE
    assert provider.calls["CODER"] == 2
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == desired


def test_failed_local_destructive_recovery_rolls_back_and_runs_independent_task(tmp_path: Path) -> None:
    old = "# accepted capability\n" * 80
    (tmp_path / "app.py").write_text(old, encoding="utf-8")
    first = Task.create("Unsafe update", "Update source")
    second = Task.create("Independent result", "Create result")
    provider = RecordingProvider({
        "CODER": [
            AgentReply({"actions": [
                {"kind": "write_file", "path": "app.py", "content": "short\n"},
                {"kind": "write_file", "path": "must-not-run.txt", "content": "unsafe"},
            ]}),
            AgentReply({"actions": []}),
            AgentReply({"actions": [{"kind": "write_file", "path": "result.txt", "content": "ok"}]}),
        ],
        "TESTER": [AgentReply({"command": [sys.executable, "-c", "assert open('result.txt').read() == 'ok'"]})],
        "REVIEWER": [AgentReply({"approved": True, "reasons": []})],
    })
    repository(tmp_path)
    state = ProjectState.create("Build a generic local application")
    state.tasks = [first, second]
    StateStore(tmp_path).save(state)
    runner = AutonomousRunner(tmp_path, StateStore(tmp_path), PatchUnavailableTools(tmp_path), provider)
    runner.controller = TaskController(local_failures_per_strategy=1, max_strategy_changes=0)

    state = runner.run(max_cycles=2)

    assert state.tasks[0].status is TaskStatus.BLOCKED
    assert state.tasks[1].status is TaskStatus.DONE
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == old
    assert not (tmp_path / "must-not-run.txt").exists()
    assert not any(event.agent == "SYSTEM" and event.phase == "CRASHED" for event in state.events)
    data = metrics(state)
    assert data["destructive_write_recovery_open"] == 0
    assert data["attempt_rollbacks_by_reason"]["tool_policy_recovery_failure"] == 1


def test_unexpected_runner_exception_persists_crashed_terminal_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = make_runner(tmp_path, RecordingProvider({}), [Task.create("Ready", "work")])
    monkeypatch.setattr(runner, "_select_next_task", lambda state: (_ for _ in ()).throw(RuntimeError("boom")))

    with pytest.raises(RuntimeError, match="boom"):
        runner.run(max_cycles=1)

    state = StateStore(tmp_path).load()
    assert state is not None
    assert state.status == "CRASHED"
    assert state.terminal_status == "CRASHED"
    assert sum(event.agent == "SYSTEM" and event.phase == "CRASHED" for event in state.events) == 1


def test_normal_complete_and_blocked_runs_persist_one_system_terminal_event(tmp_path: Path) -> None:
    complete_provider = RecordingProvider({
        "CODER": [AgentReply({"actions": [{"kind": "write_file", "path": "result.txt", "content": "ok"}]})],
        "TESTER": [AgentReply({"command": [sys.executable, "-c", "assert open('result.txt').read() == 'ok'"]})],
        "REVIEWER": [AgentReply({"approved": True, "reasons": []})],
        "FINAL_QA": [AgentReply({"status": "PASS", "findings": []})],
    })
    complete = make_runner(tmp_path / "complete", complete_provider, [Task.create("Result", "create result")]).run(max_cycles=2)
    assert complete.status == "COMPLETE"
    assert complete.terminal_status == "COMPLETE"
    assert sum(event.agent == "SYSTEM" and event.phase == "COMPLETE" for event in complete.events) == 1

    blocked_runner = make_runner(tmp_path / "blocked", RecordingProvider({}), [Task.create("Blocked", "cannot run")])
    blocked_state = StateStore(tmp_path / "blocked").load()
    assert blocked_state is not None
    blocked_state.tasks[0].status = TaskStatus.BLOCKED
    blocked_state.tasks[0].phase = "BLOCKED"
    blocked_runner.store.save(blocked_state)
    blocked = blocked_runner.run(max_cycles=1)
    assert blocked.status == "BLOCKED"
    assert blocked.terminal_status == "BLOCKED"
    assert sum(event.agent == "SYSTEM" and event.phase == "BLOCKED" for event in blocked.events) == 1


def test_empty_action_is_structured_error_but_explicit_noop_is_valid(tmp_path: Path) -> None:
    provider = RecordingProvider({
        "CODER": [
            AgentReply({"actions": [{"kind": "write_file", "path": "x.txt", "content": ""}]}),
            AgentReply({"actions": []}),
        ],
    })
    runner = make_runner(tmp_path, provider, [Task.create("Noop", "No change needed")])

    state = runner.run(max_cycles=1)

    assert any(event.phase == "CODER_STRUCTURED_OUTPUT_INVALID" for event in state.events)
    assert any(event.agent == "CODER" and event.phase == "NOOP" for event in state.events)
    assert not (tmp_path / "x.txt").exists()
    data = metrics(state)
    assert data["coder_structured_output_invalid"] == 1
    assert data["coder_structured_output_repairs"] == 1
    assert data["coder_structured_output_repair_successes"] == 1
