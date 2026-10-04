from __future__ import annotations

import subprocess
from pathlib import Path

from autodev.metrics import metrics
from autodev.models import ProjectState, Task, TaskStatus
from autodev.orchestrator import AutonomousRunner
from autodev.providers import AgentReply, AgentRequest, ScriptedProvider
from autodev.state_store import StateStore
from autodev.tools import CommandResult, WorkspaceTools


def _repository(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "AutoDev Test"], cwd=path, check=True)
    (path / ".gitignore").write_text(".autodev/\n", encoding="utf-8")
    subprocess.run(["git", "add", ".gitignore"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=path, check=True, capture_output=True)


class CapturingProvider(ScriptedProvider):
    def __init__(self, coder: list[AgentReply]) -> None:
        super().__init__({
            "CODER": coder,
            "REVIEWER": [AgentReply({"approved": True, "reasons": []})],
        })
        self.requests: list[AgentRequest] = []

    def complete(self, request: AgentRequest) -> AgentReply:
        self.requests.append(request)
        return super().complete(request)


def _setup(path: Path, provider: CapturingProvider) -> tuple[AutonomousRunner, ProjectState, Task]:
    _repository(path)
    state = ProjectState.create("Build a local Task Board")
    task = Task.create(
        "Task Data Model and Persistence",
        "Inspect the existing frontend and implement task persistence.",
        capability_id="data.persistence",
        intent="Use the existing page structure to implement a task data model persisted locally.",
        acceptance_criteria=["The implementation persists task data."],
    )
    task.acceptance_validator = {"validator_id": "deterministic", "command": ["probe"]}
    state.tasks = [task]
    store = StateStore(path)
    store.save(state)
    runner = AutonomousRunner(path, store, WorkspaceTools(path), provider)
    runner._run_validator = lambda *_args: CommandResult(0, "PASS", "")
    return runner, state, task


def _read(path: str, *, task_status: str = "continue") -> AgentReply:
    return AgentReply({"actions": [{"kind": "read_file", "path": path}], "task_status": task_status})


def test_repeated_read_packet_recovers_with_cached_results_in_same_attempt(tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text("<main>Task Board</main>\n", encoding="utf-8")
    (tmp_path / "style.css").write_text(".board { display: grid; }\n", encoding="utf-8")
    (tmp_path / "app.js").write_text("const tasks = [];\n", encoding="utf-8")
    reads = AgentReply({"actions": [
        {"kind": "read_file", "path": "index.html"},
        {"kind": "read_file", "path": "style.css"},
        {"kind": "read_file", "path": "app.js"},
    ], "task_status": "continue"})
    provider = CapturingProvider([
        reads,
        AgentReply({"actions": [
            {"kind": "read_file", "path": "index.html"},
            {"kind": "read_file", "path": "style.css"},
            {"kind": "read_file", "path": "app.js"},
        ], "task_status": "continue"}),
        AgentReply({"actions": [{"kind": "write_file", "path": "persistence.js", "content": "const storageKey = 'tasks';\n"}]}),
    ])
    runner, state, task = _setup(tmp_path, provider)

    runner._run_task(state, task)

    assert task.status is TaskStatus.DONE
    assert task.attempts == 1
    assert metrics(state)["semantic_retry_count"] == 0
    assert metrics(state)["duplicate_observations_suppressed"] == 3
    assert metrics(state)["observation_cache_hits"] == 3
    assert metrics(state)["no_progress_recovery_attempts"] == 1
    assert metrics(state)["no_progress_recovery_successes"] == 1
    assert metrics(state)["read_only_no_progress_batches"] == 1
    assert sum(call.kind == "read_file" for call in state.tool_executions) == 3
    assert (tmp_path / "persistence.js").read_text(encoding="utf-8") == "const storageKey = 'tasks';\n"
    # The primary Run-A defect was lost continuation state: ensure the
    # immediately following Coder request already receives successful reads.
    for observed in ("<main>Task Board</main>", ".board { display: grid; }", "const tasks = [];"):
        assert observed in provider.requests[1].prompt
    assert "NO_PROGRESS_RECOVERY" in provider.requests[2].prompt
    assert "<main>Task Board</main>" in provider.requests[2].prompt
    assert ".board { display: grid; }" in provider.requests[2].prompt
    assert "const tasks = [];" in provider.requests[2].prompt
    assert any(event.phase == "OBSERVATION_ALREADY_AVAILABLE" for event in state.events)
    assert any(event.phase == "NO_PROGRESS_OBSERVATION_REPEAT" for event in state.events)
    assert not any(event.phase == "ATTEMPT_ROLLBACK" for event in state.events)
    assert not any(event.agent == "ARCHITECT" for event in state.events)


def test_no_progress_recovery_is_bounded_without_semantic_retry_or_rollback(tmp_path: Path) -> None:
    (tmp_path / "app.js").write_text("const tasks = [];\n", encoding="utf-8")
    provider = CapturingProvider([_read("app.js") for _ in range(4)])
    runner, state, task = _setup(tmp_path, provider)
    runner.no_progress_recovery_budget = 2

    runner._run_task(state, task)

    assert task.status is TaskStatus.BLOCKED
    assert task.attempts == 1
    assert metrics(state)["semantic_retry_count"] == 0
    assert metrics(state)["no_progress_recovery_attempts"] == 2
    assert metrics(state)["no_progress_recovery_exhaustions"] == 1
    assert task.rollback_count == 0
    assert not any(event.phase == "ATTEMPT_ROLLBACK" for event in state.events)
    assert not any(event.agent == "ARCHITECT" for event in state.events)
    assert state.terminal_status != "CRASHED"
    assert any("NO_PROGRESS_RECOVERY_EXHAUSTED" in event.message for event in state.events)


def test_changed_file_hash_makes_same_read_action_new_information(tmp_path: Path) -> None:
    target = tmp_path / "app.js"
    target.write_text("const tasks = [];\n", encoding="utf-8")

    class ChangeBeforeSecondReply(CapturingProvider):
        def complete(self, request: AgentRequest) -> AgentReply:
            if request.role == "CODER" and sum(item.role == "CODER" for item in self.requests) == 1:
                target.write_text("const tasks = ['new evidence'];\n", encoding="utf-8")
            return super().complete(request)

    replies = [
        _read("app.js"),
        _read("app.js"),
        AgentReply({"actions": [{"kind": "write_file", "path": "done.txt", "content": "acted on fresh state"}]}),
    ]
    provider = ChangeBeforeSecondReply(replies)
    runner, state, task = _setup(tmp_path, provider)

    runner._run_task(state, task)

    reads = [call for call in state.tool_executions if call.kind == "read_file"]
    assert task.status is TaskStatus.DONE
    assert len(reads) == 2
    assert metrics(state)["no_progress_recovery_attempts"] == 0
    assert "const tasks = ['new evidence'];" in provider.requests[2].prompt


def test_write_invalidates_cached_read_and_line_ranges_are_distinct(tmp_path: Path) -> None:
    target = tmp_path / "app.js"
    target.write_text("one\ntwo\nthree\n", encoding="utf-8")
    provider = CapturingProvider([
        AgentReply({"actions": [{"kind": "read_file", "path": "app.js", "start_line": 2, "end_line": 3}], "task_status": "continue"}),
        AgentReply({"actions": [{"kind": "write_file", "path": "app.js", "content": "new state\n"}], "task_status": "continue"}),
        _read("app.js", task_status="ready_for_validation"),
    ])
    runner, state, task = _setup(tmp_path, provider)

    runner._run_task(state, task)

    assert task.status is TaskStatus.DONE
    assert sum(call.kind == "read_file" for call in state.tool_executions) == 2
    assert "two\nthree" in provider.requests[1].prompt
    assert "new state" in provider.requests[2].prompt
    assert metrics(state)["observation_cache_hits"] == 0


def test_a_different_file_range_is_not_a_duplicate_observation(tmp_path: Path) -> None:
    (tmp_path / "app.js").write_text("one\ntwo\nthree\n", encoding="utf-8")
    provider = CapturingProvider([
        AgentReply({"actions": [{"kind": "read_file", "path": "app.js", "start_line": 1, "end_line": 1}], "task_status": "continue"}),
        AgentReply({"actions": [{"kind": "read_file", "path": "app.js", "start_line": 2, "end_line": 3}], "task_status": "continue"}),
        AgentReply({"actions": [{"kind": "write_file", "path": "done.txt", "content": "read two ranges"}]}),
    ])
    runner, state, task = _setup(tmp_path, provider)

    runner._run_task(state, task)

    assert task.status is TaskStatus.DONE
    assert sum(call.kind == "read_file" for call in state.tool_executions) == 2
    assert "two\nthree" in provider.requests[2].prompt
    assert metrics(state)["read_only_no_progress_batches"] == 0
