"""Regression coverage for the bounded Coder decision/mutation protocol."""

from __future__ import annotations

import subprocess
import re
from pathlib import Path

import pytest

from autodev.models import ProjectState, Task, TaskStatus
from autodev.agents import MutationFrameError, RoleAgents
from autodev.metrics import metrics
from autodev.orchestrator import AutonomousRunner, MutationProtocolExhaustedError
from autodev.providers import AgentReply, AgentRequest, ProviderError, ScriptedProvider
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
            path = re.search(r"^Path: (.+)$", request.prompt, re.MULTILINE)
            assert path is not None
            target = path.group(1)
            if operation == "replace":
                old_end = "" if reply.data.get("omit_old_end") else f"\nOLD-END-{marker}"
                raw = (
                    f"OPERATION=replace\nDONE={done}\nPATH={target}\nOLD-BEGIN-{marker}\n{reply.data.get('old', '')}"
                    f"{old_end}\nNEW-BEGIN-{marker}\n{reply.data.get('content', '')}\nNEW-END-{marker}"
                )
            else:
                raw = (
                    f"OPERATION={operation}\nDONE={done}\nPATH={target}\nCONTENT-BEGIN-{marker}\n{reply.data.get('content', '')}"
                    f"\nCONTENT-END-{marker}"
                )
            reply = AgentReply({"raw_text": raw, "response_complete": True})
        elif not request.raw_response and reply.data.get("request_id") == "AUTO":
            token = re.search(r"Mutation request id: ([0-9a-f]+)", request.prompt)
            path = re.search(r"^Path: (.+)$", request.prompt, re.MULTILINE)
            assert token is not None and path is not None
            reply.data["request_id"] = token.group(1)
            reply.data["path"] = path.group(1)
        self.responses.append(reply.data)
        return reply


def _patch(content: str, *, operation: str = "append", done: bool = False, old: str = "", omit_old_end: bool = False) -> AgentReply:
    data: dict[str, object] = {"operation": operation, "content": content, "done": done}
    if old:
        data["old"] = old
    if omit_old_end:
        data["omit_old_end"] = True
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
    raw = "OPERATION=append\nDONE=true\nPATH=app.py\nCONTENT-BEGIN-token\nvalue = \"quoted\"\nCONTENT-END-token"

    patch = RoleAgents._parse_raw_patch(raw, "token", 2_000, expected_path="app.py")

    assert patch == {"operation": "append", "content": "value = \"quoted\"", "done": True, "frame_status": "VALID"}


def test_missing_old_end_recovers_at_unique_matching_new_begin() -> None:
    raw = (
        "OPERATION=replace\nDONE=true\nPATH=style.css\n"
        "OLD-BEGIN-token\ncolor: red;\n"
        "NEW-BEGIN-token\ncolor: blue;\nNEW-END-token"
    )

    patch = RoleAgents._parse_raw_patch(raw, "token", 2_000, expected_path="style.css")

    assert patch == {
        "operation": "replace",
        "old": "color: red;",
        "content": "color: blue;",
        "done": True,
        "frame_status": "RECOVERED_VALID",
    }


def test_duplicate_new_begin_is_ambiguous_not_recovered() -> None:
    raw = (
        "OPERATION=replace\nDONE=true\nPATH=style.css\nOLD-BEGIN-token\nold\n"
        "NEW-BEGIN-token\nfirst\nNEW-BEGIN-token\nsecond\nNEW-END-token"
    )

    with pytest.raises(ProviderError, match="MUTATION_PROTOCOL_AMBIGUOUS"):
        RoleAgents._parse_raw_patch(raw, "token", 2_000, expected_path="style.css")


@pytest.mark.parametrize("raw", [
    "OPERATION=replace\nDONE=true\nPATH=style.css\nprose\nOLD-BEGIN-token\nold\nNEW-BEGIN-token\nnew\nNEW-END-token",
    "OPERATION=replace\nDONE=true\nPATH=style.css\nOLD-BEGIN-token\nold\nOLD-END-token\nprose\nNEW-BEGIN-token\nnew\nNEW-END-token",
    "OPERATION=replace\nDONE=true\nPATH=style.css\nCONTENT-BEGIN-token\nwrong-operation\nCONTENT-END-token",
])
def test_unexpected_preamble_inter_section_prose_or_boundary_family_is_ambiguous(raw: str) -> None:
    with pytest.raises(ProviderError, match="MUTATION_PROTOCOL_AMBIGUOUS"):
        RoleAgents._parse_raw_patch(raw, "token", 2_000, expected_path="style.css")


def test_wrong_request_nonce_and_wrong_target_path_are_rejected() -> None:
    nonce_mismatch = "OPERATION=append\nDONE=true\nPATH=app.css\nCONTENT-BEGIN-other\nbody\nCONTENT-END-other"
    path_mismatch = "OPERATION=append\nDONE=true\nPATH=other.css\nCONTENT-BEGIN-token\nbody\nCONTENT-END-token"

    with pytest.raises(ProviderError, match="MUTATION_PROTOCOL_AMBIGUOUS"):
        RoleAgents._parse_raw_patch(nonce_mismatch, "token", 2_000, expected_path="app.css")
    with pytest.raises(ProviderError, match="MUTATION_PROTOCOL_AMBIGUOUS"):
        RoleAgents._parse_raw_patch(path_mismatch, "token", 2_000, expected_path="app.css")


def test_fallback_json_rejects_wrong_request_identity_or_path() -> None:
    fields = {"request_id": "other", "path": "app.css", "operation": "append", "old": "", "content": "body", "done": True}

    with pytest.raises(ProviderError, match="MUTATION_PROTOCOL_AMBIGUOUS"):
        RoleAgents._parse_json_mutation(fields, "token", "app.css", 2_000)
    fields.update(request_id="token", path="other.css")
    with pytest.raises(ProviderError, match="MUTATION_PROTOCOL_AMBIGUOUS"):
        RoleAgents._parse_json_mutation(fields, "token", "app.css", 2_000)


def test_missing_new_end_is_not_guessed_at_eof_even_for_normal_completion() -> None:
    raw = "OPERATION=replace\nDONE=true\nPATH=style.css\nOLD-BEGIN-token\nold\nOLD-END-token\nNEW-BEGIN-token\nnew"

    with pytest.raises(ProviderError, match="MUTATION_PROTOCOL_AMBIGUOUS"):
        RoleAgents._parse_raw_patch(raw, "token", 2_000, expected_path="style.css", response_complete=True)


def test_trailing_prose_after_new_end_is_rejected() -> None:
    raw = (
        "OPERATION=replace\nDONE=true\nPATH=style.css\nOLD-BEGIN-token\nold\nOLD-END-token\n"
        "NEW-BEGIN-token\nnew\nNEW-END-token\nHere is the patch."
    )

    with pytest.raises(ProviderError, match="MUTATION_PROTOCOL_AMBIGUOUS"):
        RoleAgents._parse_raw_patch(raw, "token", 2_000, expected_path="style.css")


def test_ambiguous_frame_uses_one_bounded_json_fallback_in_same_semantic_attempt(tmp_path: Path) -> None:
    _repository(tmp_path)
    (tmp_path / "style.css").write_text(".card { color: red; }\n", encoding="utf-8")
    provider = RecordingProvider({
        "CODER": [
            AgentReply({"raw_text": "OPERATION=replace\nDONE=true\nPATH=style.css\nOLD-BEGIN-token\nold\nNEW-BEGIN-token\nnew\nNEW-END-token\nNEW-BEGIN-token"}),
            AgentReply({
                "request_id": "AUTO", "path": "", "operation": "replace", "old": ".card { color: red; }",
                "content": ".card { color: blue; }", "done": True,
            }),
        ],
    })
    runner = AutonomousRunner(tmp_path, StateStore(tmp_path), WorkspaceTools(tmp_path), provider)
    state = ProjectState.create("Edit a stylesheet")
    task = Task.create("Change card color", "Change the card color")
    task.attempts = 1

    runner._execute_coder_actions(state, task, [{"kind": "mutate_file", "path": "style.css", "intent": "change card color"}])

    assert task.attempts == 1
    assert (tmp_path / "style.css").read_text(encoding="utf-8") == ".card { color: blue; }\n"
    assert len(provider.requests) == 2
    assert provider.requests[-1].raw_response is False
    request_ids = [re.search(r"Mutation request id: ([0-9a-f]+)", request.prompt).group(1) for request in provider.requests]
    assert request_ids[0] != request_ids[1]
    assert all("Path: style.css" in request.prompt for request in provider.requests)
    assert metrics(state)["mutation_frames_ambiguous"] == 1
    assert metrics(state)["mutation_frame_fallbacks"] == 1
    assert metrics(state)["mutation_frame_fallback_successes"] == 1


def test_missing_old_end_is_applied_and_counted_without_semantic_retry(tmp_path: Path, monkeypatch) -> None:
    _repository(tmp_path)
    (tmp_path / "style.css").write_text(".card { color: red; }\n", encoding="utf-8")
    provider = RecordingProvider({
        "CODER": [
            AgentReply({"actions": [{"kind": "mutate_file", "path": "style.css", "intent": "change card color"}], "task_status": "ready_for_validation"}),
            _patch(".card { color: blue; }", operation="replace", old=".card { color: red; }", done=True, omit_old_end=True),
        ],
        "TESTER": [AgentReply({"command": ["py", "-3", "-c", "print('pass')"]})],
        "REVIEWER": [AgentReply({"approved": True, "reasons": []})],
    })
    runner = AutonomousRunner(tmp_path, StateStore(tmp_path), WorkspaceTools(tmp_path), provider)
    state = ProjectState.create("Edit a stylesheet")
    task = Task.create("Change card color", "Change the card color")
    task.acceptance_validator = {"validator_id": "css", "command": ["deterministic acceptance"]}
    state.tasks = [task]
    monkeypatch.setattr(runner, "_run_validator", lambda *_args: CommandResult(0, "PASS", ""))

    runner._run_task(state, task)

    assert task.status is TaskStatus.DONE
    assert task.attempts == 1
    assert (tmp_path / "style.css").read_text(encoding="utf-8") == ".card { color: blue; }\n"
    assert metrics(state)["mutation_frames_recovered"] == 1
    assert metrics(state)["mutation_frame_fallbacks"] == 0
    request_id = re.search(r"Mutation request id: ([0-9a-f]+)", provider.requests[1].prompt).group(1)
    assert any(request_id in event.message for event in state.events if event.phase == "MUTATION_PATCH_REQUEST")


def test_recovered_old_block_must_still_match_current_file_before_any_mutation(tmp_path: Path) -> None:
    _repository(tmp_path)
    (tmp_path / "style.css").write_text(".card { color: green; }\n", encoding="utf-8")
    provider = RecordingProvider({
        "CODER": [
            _patch(".card { color: blue; }", operation="replace", old=".card { color: red; }", done=True, omit_old_end=True),
            AgentReply({"request_id": "wrong", "path": "style.css", "operation": "replace", "old": "red", "content": "blue", "done": True}),
        ],
    })
    runner = AutonomousRunner(tmp_path, StateStore(tmp_path), WorkspaceTools(tmp_path), provider)
    state = ProjectState.create("Edit a stylesheet")
    task = Task.create("Change card color", "Change the card color")
    task.attempts = 1

    with pytest.raises(MutationProtocolExhaustedError, match="MUTATION_PROTOCOL_RECOVERY_EXHAUSTED"):
        runner._execute_coder_actions(state, task, [{"kind": "mutate_file", "path": "style.css", "intent": "change color"}])

    assert (tmp_path / "style.css").read_text(encoding="utf-8") == ".card { color: green; }\n"
    assert task.attempts == 1


def test_invalid_fallback_terminates_as_task_protocol_failure_without_crash(tmp_path: Path) -> None:
    _repository(tmp_path)
    (tmp_path / "style.css").write_text("a { color: red; }\n", encoding="utf-8")
    provider = RecordingProvider({
        "CODER": [
            AgentReply({"actions": [{"kind": "mutate_file", "path": "style.css", "intent": "change color"}]}),
            AgentReply({"raw_text": "OPERATION=replace\nDONE=true\nPATH=style.css\nOLD-BEGIN-wrong\nold\nNEW-BEGIN-wrong\nnew\nNEW-END-wrong"}),
            AgentReply({"request_id": "AUTO", "path": "style.css", "operation": "replace", "old": "missing", "content": "blue", "done": True}),
        ],
    })
    runner = AutonomousRunner(tmp_path, StateStore(tmp_path), WorkspaceTools(tmp_path), provider, max_attempts=1)
    state = ProjectState.create("Edit a stylesheet")
    task = Task.create("Change color", "Change the color")
    state.tasks = [task]

    runner._run_task(state, task)

    assert task.status is TaskStatus.BLOCKED
    assert task.attempts == 1
    assert (tmp_path / "style.css").read_text(encoding="utf-8") == "a { color: red; }\n"
    assert metrics(state)["mutation_frame_fallback_failures"] == 1
    assert metrics(state)["system_crashes"] == 0


def test_replace_refuses_old_text_that_matches_more_than_one_current_region(tmp_path: Path) -> None:
    _repository(tmp_path)
    (tmp_path / "style.css").write_text("a { color: red; }\nb { color: red; }\n", encoding="utf-8")
    runner = AutonomousRunner(tmp_path, StateStore(tmp_path), WorkspaceTools(tmp_path), ScriptedProvider({}))
    state = ProjectState.create("Edit a stylesheet")
    task = Task.create("Change one color", "Change one exact color region")

    with pytest.raises(MutationFrameError, match="MUTATION_PROTOCOL_AMBIGUOUS"):
        runner._apply_bounded_mutation_payload(
            state, task, "style.css", "replace", "color: blue;", "color: red;", True, 6_000
        )

    assert (tmp_path / "style.css").read_text(encoding="utf-8") == "a { color: red; }\nb { color: red; }\n"
    assert not state.tool_executions


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
