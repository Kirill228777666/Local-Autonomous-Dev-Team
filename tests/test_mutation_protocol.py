"""Regression coverage for the bounded Coder decision/mutation protocol."""

from __future__ import annotations

import subprocess
from pathlib import Path

from autodev.models import ProjectState, Task, TaskStatus
from autodev.orchestrator import AutonomousRunner
from autodev.providers import AgentReply, AgentRequest, ScriptedProvider
from autodev.state_store import StateStore
from autodev.tools import WorkspaceTools


def _repository(path: Path) -> None:
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "AutoDev Test"], cwd=path, check=True)
    (path / ".gitignore").write_text(".autodev/\n", encoding="utf-8")
    subprocess.run(["git", "add", ".gitignore"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=path, check=True, capture_output=True)


class RecordingProvider(ScriptedProvider):
    def __init__(self, replies: dict[str, list[AgentReply]]) -> None:
        super().__init__(replies)
        self.requests: list[AgentRequest] = []
        self.responses: list[dict[str, object]] = []

    def complete(self, request: AgentRequest) -> AgentReply:
        self.requests.append(request)
        reply = super().complete(request)
        self.responses.append(reply.data)
        return reply


def _patch(content: str, *, operation: str = "append", done: bool = False, old: str = "") -> AgentReply:
    data: dict[str, object] = {"operation": operation, "content": content, "done": done}
    if old:
        data["old"] = old
    return AgentReply(data)


def test_large_source_is_created_in_bounded_patch_substeps_without_large_decision_payload(tmp_path: Path) -> None:
    _repository(tmp_path)
    section = "x" * 6_000
    provider = RecordingProvider({
        "CODER": [
            AgentReply({"actions": [{"kind": "mutate_file", "path": "large.txt", "intent": "create generated source"}], "task_status": "ready_for_validation"}),
            *[_patch(section, operation="create" if index == 0 else "append", done=index == 17) for index in range(18)],
        ],
        "TESTER": [AgentReply({"command": ["py", "-3", "-c", "from pathlib import Path; assert Path('large.txt').stat().st_size == 108000"]})],
        "REVIEWER": [AgentReply({"approved": True, "reasons": []})],
    })
    runner = AutonomousRunner(tmp_path, StateStore(tmp_path), WorkspaceTools(tmp_path), provider)
    state = ProjectState.create("Build a large generated local artifact")
    task = Task.create("Generate artifact", "Generate more than one hundred kilobytes of source")
    state.tasks = [task]

    runner._run_task(state, task)

    assert task.status is TaskStatus.DONE
    assert task.attempts == 1
    assert (tmp_path / "large.txt").stat().st_size == 108_000
    decision = provider.responses[0]
    assert decision["actions"] == [{"kind": "mutate_file", "path": "large.txt", "intent": "create generated source"}]
    assert all(len(request.prompt) < 14_000 for request in provider.requests)
    assert not any(event.phase == "CODER_PROTOCOL_RECOVERY_EXHAUSTED" for event in state.events)


def test_existing_file_mutation_uses_targeted_patch_and_preserves_unrelated_content(tmp_path: Path) -> None:
    _repository(tmp_path)
    (tmp_path / "app.txt").write_text("before\nTARGET = old\nafter\n", encoding="utf-8")
    provider = RecordingProvider({
        "CODER": [_patch("TARGET = new", operation="replace", old="TARGET = old", done=True)],
    })
    runner = AutonomousRunner(tmp_path, StateStore(tmp_path), WorkspaceTools(tmp_path), provider)
    state = ProjectState.create("Edit one source behavior")
    task = Task.create("Change target", "Change only the target value")

    runner._execute_coder_actions(state, task, [{"kind": "mutate_file", "path": "app.txt", "intent": "change target"}])

    assert (tmp_path / "app.txt").read_text(encoding="utf-8") == "before\nTARGET = new\nafter\n"
    assert len(provider.requests) == 1
    assert "TARGET = old" in provider.requests[0].prompt


def test_stale_patch_refreshes_context_without_consuming_semantic_retry(tmp_path: Path) -> None:
    _repository(tmp_path)
    (tmp_path / "app.txt").write_text("value = old\n", encoding="utf-8")
    provider = RecordingProvider({
        "CODER": [
            _patch("value = new", operation="replace", old="stale value", done=False),
            _patch("value = new", operation="replace", old="value = old", done=True),
        ],
    })
    runner = AutonomousRunner(tmp_path, StateStore(tmp_path), WorkspaceTools(tmp_path), provider)
    state = ProjectState.create("Edit source")
    task = Task.create("Change value", "Update one value")
    task.attempts = 1

    runner._execute_coder_actions(state, task, [{"kind": "mutate_file", "path": "app.txt", "intent": "change value"}])

    assert task.attempts == 1
    assert (tmp_path / "app.txt").read_text(encoding="utf-8") == "value = new\n"
    assert len(provider.requests) == 2
    assert any(event.phase == "MUTATION_CONTEXT_REFRESH" for event in state.events)


def test_malformed_patch_does_not_execute_partial_mutation_and_rolls_back_attempt(tmp_path: Path) -> None:
    _repository(tmp_path)
    (tmp_path / "kept.txt").write_text("original\n", encoding="utf-8")
    provider = RecordingProvider({
        "CODER": [
            AgentReply({"actions": [
                {"kind": "mutate_file", "path": "new.txt", "intent": "create new file"},
                {"kind": "mutate_file", "path": "kept.txt", "intent": "break protected file"},
            ]}),
            _patch("created\n", operation="create", done=True),
            AgentReply({"operation": "replace", "content": "bad", "done": True}),
        ],
    })
    runner = AutonomousRunner(tmp_path, StateStore(tmp_path), WorkspaceTools(tmp_path), provider, max_attempts=1)
    state = ProjectState.create("Build source")
    task = Task.create("Mutate files", "Mutate two files")
    state.tasks = [task]

    runner._run_task(state, task)

    assert task.status is TaskStatus.BLOCKED
    assert not (tmp_path / "new.txt").exists()
    assert (tmp_path / "kept.txt").read_text(encoding="utf-8") == "original\n"
    assert task.attempts == 1
