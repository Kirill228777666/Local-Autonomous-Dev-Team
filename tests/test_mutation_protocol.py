"""Regression coverage for the bounded Coder decision/mutation protocol."""

from __future__ import annotations

import subprocess
import re
from pathlib import Path

from autodev.models import ProjectState, Task, TaskStatus
from autodev.agents import RoleAgents
from autodev.metrics import metrics
from autodev.orchestrator import AutonomousRunner
from autodev.providers import AgentReply, AgentRequest, ScriptedProvider
from autodev.state_store import StateStore
from autodev.tools import CommandResult, WorkspaceTools


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
        if request.raw_response and "operation" in reply.data:
            operation = reply.data.get("operation")
            done = str(reply.data.get("done", False)).lower()
            token = re.search(r"(?:CONTENT|OLD)-BEGIN-([0-9a-f]+)", request.prompt)
            assert token is not None
            marker = token.group(1)
            if operation == "replace":
                raw = (
                    f"OPERATION=replace\nDONE={done}\nOLD-BEGIN-{marker}\n{reply.data.get('old', '')}"
                    f"\nOLD-END-{marker}\nNEW-BEGIN-{marker}\n{reply.data.get('content', '')}\nNEW-END-{marker}"
                )
            else:
                raw = (
                    f"OPERATION={operation}\nDONE={done}\nCONTENT-BEGIN-{marker}\n{reply.data.get('content', '')}"
                    f"\nCONTENT-END-{marker}"
                )
            reply = AgentReply({"raw_text": raw})
        self.responses.append(reply.data)
        return reply


def _patch(content: str, *, operation: str = "append", done: bool = False, old: str = "") -> AgentReply:
    data: dict[str, object] = {"operation": operation, "content": content, "done": done}
    if old:
        data["old"] = old
    return AgentReply(data)


def test_large_multi_file_source_is_created_in_bounded_patch_substeps_without_large_decision_payload(tmp_path: Path) -> None:
    _repository(tmp_path)
    section = "x" * 2_000
    provider = RecordingProvider({
        "CODER": [
            AgentReply({"actions": [
                {"kind": "mutate_file", "path": "large-a.txt", "intent": "create at least 54000 characters of generated source"},
                {"kind": "mutate_file", "path": "large-b.txt", "intent": "create at least 54000 characters of generated source"},
            ], "task_status": "ready_for_validation"}),
            *[_patch(section, operation="create" if index in {0, 27} else "append", done=index in {26, 53}) for index in range(54)],
        ],
        "TESTER": [AgentReply({"command": ["py", "-3", "-c", "from pathlib import Path; assert Path('large-a.txt').stat().st_size + Path('large-b.txt').stat().st_size == 108000"]})],
        "REVIEWER": [AgentReply({"approved": True, "reasons": []})],
    })
    runner = AutonomousRunner(tmp_path, StateStore(tmp_path), WorkspaceTools(tmp_path), provider)
    state = ProjectState.create("Build a large generated local artifact")
    task = Task.create("Generate artifact", "Generate more than one hundred kilobytes of source")
    state.tasks = [task]

    runner._run_task(state, task)

    assert task.status is TaskStatus.DONE
    assert task.attempts == 1
    assert (tmp_path / "large-a.txt").stat().st_size + (tmp_path / "large-b.txt").stat().st_size == 108_000
    decision = provider.responses[0]
    assert decision["actions"] == [
        {"kind": "mutate_file", "path": "large-a.txt", "intent": "create at least 54000 characters of generated source"},
        {"kind": "mutate_file", "path": "large-b.txt", "intent": "create at least 54000 characters of generated source"},
    ]
    assert all(len(request.prompt) < 14_000 for request in provider.requests)
    assert not any(event.phase == "CODER_PROTOCOL_RECOVERY_EXHAUSTED" for event in state.events)
    observed = metrics(state)
    assert observed["coder_mutation_patch_requests"] == 54
    assert observed["coder_mutation_patches_applied"] == 54
    assert observed["llm_requests_attempted"] == len(provider.requests)


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
    assert "one bounded plain-text patch" in provider.requests[0].system_prompt
    assert provider.requests[0].max_output_tokens == 1_536
    assert provider.requests[0].raw_response is True


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


def test_coder_command_file_write_is_rerouted_through_bounded_mutation_protocol(tmp_path: Path) -> None:
    _repository(tmp_path)
    provider = RecordingProvider({
        "CODER": [
            AgentReply({"actions": [{"kind": "mutate_file", "path": "generated.txt", "intent": "create safe file"}]}),
            _patch("safe output\n", operation="create", done=True),
        ],
    })
    runner = AutonomousRunner(tmp_path, StateStore(tmp_path), WorkspaceTools(tmp_path), provider)
    state = ProjectState.create("Build source")
    task = Task.create("Create file", "Create one file")
    task.attempts = 1

    runner._execute_coder_actions(state, task, [{
        "kind": "run_command",
        "command": ["py", "-3", "-c", "from pathlib import Path; Path('generated.txt').write_text('bypass')"],
    }])

    assert task.attempts == 1
    assert (tmp_path / "generated.txt").read_text(encoding="utf-8") == "safe output\n"
    assert any(event.phase == "SOURCE_MUTATION_COMMAND_REJECTED" for event in state.events)
    assert "Path('generated.txt').write_text('bypass')" in provider.requests[0].prompt
    assert provider.responses[0]["actions"][0]["kind"] == "mutate_file"
    assert runner._command_writes_workspace_files(["powershell", "-Command", "'source' | Set-Content app.py"])
    assert runner._command_writes_workspace_files(["cmd", "/c", "echo source > app.py"])
    assert runner._command_writes_workspace_files(["python", "-c", "import os; os.open('app.py', os.O_WRONLY | os.O_CREAT)"])


def test_raw_mutation_payload_is_parsed_without_json_escaping_or_closing_marker() -> None:
    raw = "OPERATION=append\nDONE=true\nCONTENT-BEGIN-token\nvalue = \"quoted\"\n"

    patch = RoleAgents._parse_raw_patch(raw, "token", 2_000)

    assert patch == {"operation": "append", "content": "value = \"quoted\"\n", "done": True}


def test_valid_oversized_hunk_is_split_before_workspace_mutations(tmp_path: Path) -> None:
    _repository(tmp_path)
    runner = AutonomousRunner(tmp_path, StateStore(tmp_path), WorkspaceTools(tmp_path), ScriptedProvider({}))
    state = ProjectState.create("Build source")
    task = Task.create("Create file", "Create one bounded file")
    content = "x" * 6_500

    runner._apply_bounded_mutation_payload(state, task, "large.txt", "create", content, "", False, 6_000)

    assert (tmp_path / "large.txt").read_text(encoding="utf-8") == content
    file_mutations = [execution for execution in state.tool_executions if execution.kind in {"write_file", "append_file"}]
    assert len(file_mutations) == 2


def test_repeated_coder_continuation_after_mutation_proceeds_to_acceptance(tmp_path: Path, monkeypatch) -> None:
    _repository(tmp_path)
    decisions = [{"kind": "mutate_file", "path": "app.py", "intent": "create a valid module"}]
    provider = RecordingProvider({
        "CODER": [
            AgentReply({"actions": decisions, "task_status": "continue"}),
            _patch("VALUE = 1\n", operation="create", done=True),
            AgentReply({"actions": decisions, "task_status": "continue"}),
        ],
        "REVIEWER": [AgentReply({"approved": True, "reasons": []})],
    })
    runner = AutonomousRunner(tmp_path, StateStore(tmp_path), WorkspaceTools(tmp_path), provider)
    task = Task.create("Implement module", "Create a small module")
    task.acceptance_validator = {"validator_id": "module", "command": ["deterministic acceptance"]}
    state = ProjectState.create("Create a module")
    state.tasks = [task]
    monkeypatch.setattr(runner, "_run_validator", lambda *_args: CommandResult(0, "PASS", ""))

    runner._run_task(state, task)

    assert task.status is TaskStatus.DONE
    assert task.attempts == 1
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    assert any(event.phase == "DUPLICATE_CONTINUATION_AFTER_PROGRESS" for event in state.events)


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
