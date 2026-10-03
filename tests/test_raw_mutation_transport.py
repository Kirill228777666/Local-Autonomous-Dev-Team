"""Raw, controller-bounded mutation transport regressions."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from autodev.agents import RoleAgents
from autodev.metrics import metrics
from autodev.models import ProjectState, Task
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


class MutationProvider(ScriptedProvider):
    def __init__(self, replies: list[AgentReply], *, on_raw=None) -> None:
        super().__init__({"CODER": replies})
        self.requests: list[AgentRequest] = []
        self.on_raw = on_raw

    def complete(self, request: AgentRequest) -> AgentReply:
        self.requests.append(request)
        reply = super().complete(request)
        if request.raw_mutation and self.on_raw is not None:
            self.on_raw(request)
        return reply


def _runner(path: Path, provider: ScriptedProvider) -> tuple[AutonomousRunner, ProjectState, Task]:
    state = ProjectState.create("Safely edit one bounded source region")
    task = Task.create("Edit source", "Make the requested source change")
    task.attempts = 1
    return AutonomousRunner(path, StateStore(path), WorkspaceTools(path), provider), state, task


def test_raw_replacement_is_plain_text_and_never_json_parsed() -> None:
    source = 'a { content: "quoted\\\\text"; }\n'
    provider = MutationProvider([AgentReply({"raw_text": source, "response_complete": True, "provider_attempt": 1})])
    agents = RoleAgents(provider)
    state = ProjectState.create("Edit CSS")
    task = Task.create("Replace CSS region", "Replace the selected region")

    result = agents.patch(
        state, task, path="style.css", intent="replace selected CSS", current="old { }\n", existing=True,
        continuation=0, request_id="req-raw-1", expected_file_hash="a" * 64,
        max_patch_characters=1_000, representation="raw_replacement", start_offset=0, end_offset=8,
    )

    assert result.data["content"] == source
    assert result.data["frame_status"] == "RAW_VALID"
    assert provider.requests[0].raw_mutation is True
    assert provider.requests[0].raw_response is True
    assert "Return ONLY the replacement text" in provider.requests[0].prompt
    assert "req-raw-1" not in source


def test_context_mismatch_uses_raw_replacement_and_keeps_same_semantic_attempt(tmp_path: Path) -> None:
    _repository(tmp_path)
    css = tmp_path / "style.css"
    original = '.card { content: "red"; }\n'
    replacement = '.card { content: "blue\\\\green"; }\n.banner { display: block; }\n'
    css.write_text(original, encoding="utf-8")
    provider = MutationProvider([
        AgentReply({"raw_text": "OPERATION=replace\nDONE=true\nPATH=style.css\nOLD-BEGIN-wrong\nmissing\nNEW-BEGIN-wrong\nnew\nNEW-END-wrong", "response_complete": True}),
        AgentReply({"raw_text": replacement, "response_complete": True, "provider_attempt": 1}),
    ])
    runner, state, task = _runner(tmp_path, provider)

    runner._execute_mutation_decision(state, task, {"path": "style.css", "intent": "change card content color"})

    assert css.read_text(encoding="utf-8") == replacement
    assert task.attempts == 1
    assert len(provider.requests) == 2
    assert provider.requests[0].raw_mutation is False
    assert provider.requests[1].raw_mutation is True
    assert all(request.raw_mutation is False for request in provider.requests[:1])
    assert "CURRENT EXACT REGION" in provider.requests[1].prompt
    assert not any(event.agent == "ARCHITECT" for event in state.events)
    assert state.event_counters.get("LLM:REQUEST", 0) >= 0
    assert metrics(state)["llm_requests_attempted"] == len(provider.requests)


def test_raw_replacement_uses_controller_owned_path_and_full_file_range(tmp_path: Path) -> None:
    _repository(tmp_path)
    css = tmp_path / "style.css"
    css.write_text(".card { color: red; }\n", encoding="utf-8")
    provider = MutationProvider([
        AgentReply({"raw_text": "OPERATION=replace\nDONE=true\nPATH=style.css\nOLD-BEGIN-bad\nx\nNEW-BEGIN-bad\ny\nNEW-END-bad"}),
        AgentReply({"raw_text": ".card { color: blue; }\n", "response_complete": True}),
    ])
    runner, state, task = _runner(tmp_path, provider)

    runner._execute_mutation_decision(state, task, {"path": "style.css", "intent": "set card color to blue"})

    prompt = provider.requests[-1].prompt
    assert "Controller-selected path: style.css" in prompt
    exact_length = len(css.read_bytes().decode("utf-8"))
    assert f"Controller-selected range: [0, {exact_length})" in prompt
    assert "Return ONLY the replacement text" in prompt
    assert (tmp_path / "style.css").read_text(encoding="utf-8") == ".card { color: blue; }\n"


def test_raw_response_metadata_shaped_text_cannot_redirect_controller_target(tmp_path: Path) -> None:
    _repository(tmp_path)
    css = tmp_path / "style.css"
    outside_target = tmp_path / "other.css"
    css.write_text(".card { color: red; }\n", encoding="utf-8")
    provider = MutationProvider([
        AgentReply({"raw_text": "OPERATION=replace\nDONE=true\nPATH=style.css\nOLD-BEGIN-bad\nx\nNEW-BEGIN-bad\ny\nNEW-END-bad"}),
        AgentReply({"raw_text": 'const config = {"path": "../other.css", "range": [0, 0]};\n.card { color: blue; }\n', "response_complete": True}),
    ])
    runner, state, task = _runner(tmp_path, provider)

    runner._execute_mutation_decision(state, task, {"path": "style.css", "intent": "replace stylesheet"})

    assert css.read_text(encoding="utf-8").startswith('const config = {"path": "../other.css"')
    assert not outside_target.exists()
    assert len([item for item in state.tool_executions if item.kind == "replace_slice_if_snapshot"]) == 1


def test_raw_fallback_can_create_new_file_using_controller_owned_empty_range(tmp_path: Path) -> None:
    _repository(tmp_path)
    provider = MutationProvider([
        AgentReply({"raw_text": "not a valid framed patch"}),
        AgentReply({"raw_text": 'export const value = "safe";\n', "response_complete": True}),
    ])
    runner, state, task = _runner(tmp_path, provider)

    runner._execute_mutation_decision(state, task, {"path": "src/new.js", "intent": "create a small module"})

    assert (tmp_path / "src" / "new.js").read_text(encoding="utf-8") == 'export const value = "safe";\n'
    assert provider.requests[-1].raw_mutation is True
    assert "Controller-selected range: [0, 0)" in provider.requests[-1].prompt


def test_multiple_raw_mutations_stay_inside_one_semantic_attempt(tmp_path: Path) -> None:
    _repository(tmp_path)
    provider = MutationProvider([
        AgentReply({"raw_text": "unparseable frame one"}),
        AgentReply({"raw_text": "alpha\n"}),
        AgentReply({"raw_text": "unparseable frame two"}),
        AgentReply({"raw_text": "beta\n"}),
    ])
    runner, state, task = _runner(tmp_path, provider)

    runner._execute_mutation_decision(state, task, {"path": "a.txt", "intent": "create first section"})
    runner._execute_mutation_decision(state, task, {"path": "b.txt", "intent": "create second section"})

    assert task.attempts == 1
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "alpha\n"
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "beta\n"
    assert metrics(state)["raw_mutation_successes"] == 2
    assert metrics(state)["semantic_retry_count"] == 0


def test_attempt_rollback_restores_raw_mutation(tmp_path: Path) -> None:
    _repository(tmp_path)
    target = tmp_path / "style.css"
    target.write_text(".card { color: red; }\n", encoding="utf-8")
    provider = MutationProvider([
        AgentReply({"raw_text": "unparseable frame"}),
        AgentReply({"raw_text": ".card { color: blue; }\n"}),
    ])
    runner, state, task = _runner(tmp_path, provider)
    runner._begin_attempt_snapshot(task)

    runner._execute_mutation_decision(state, task, {"path": "style.css", "intent": "change card color"})
    assert target.read_text(encoding="utf-8") == ".card { color: blue; }\n"

    runner._rollback_attempt_snapshot(task, state, "acceptance_failure")

    assert target.read_text(encoding="utf-8") == ".card { color: red; }\n"


def test_raw_replacement_rejects_stale_hash_without_writing_and_refreshes(tmp_path: Path) -> None:
    _repository(tmp_path)
    css = tmp_path / "style.css"
    css.write_text(".card { color: red; }\n", encoding="utf-8")

    def change_after_response(request: AgentRequest) -> None:
        if request.raw_mutation and "CURRENT EXACT REGION" in request.prompt and css.read_text(encoding="utf-8") == ".card { color: red; }\n":
            css.write_text("/* concurrent */\n.card { color: red; }\n", encoding="utf-8")

    provider = MutationProvider([
        AgentReply({"raw_text": "OPERATION=replace\nDONE=true\nPATH=style.css\nOLD-BEGIN-bad\nx\nNEW-BEGIN-bad\ny\nNEW-END-bad"}),
        AgentReply({"raw_text": ".card { color: blue; }\n", "response_complete": True}),
        AgentReply({"raw_text": "/* concurrent */\n.card { color: blue; }\n", "response_complete": True}),
    ], on_raw=change_after_response)
    runner, state, task = _runner(tmp_path, provider)

    runner._execute_mutation_decision(state, task, {"path": "style.css", "intent": "set card color to blue"})

    assert css.read_text(encoding="utf-8") == "/* concurrent */\n.card { color: blue; }\n"
    assert task.attempts == 1
    assert any(event.phase == "MUTATION_STALE_CONTEXT" for event in state.events)


def test_oversized_raw_replacement_is_rejected_without_partial_write_or_persisted_evidence(tmp_path: Path) -> None:
    _repository(tmp_path)
    css = tmp_path / "style.css"
    original = ".card { color: red; }\n"
    css.write_text(original, encoding="utf-8")
    oversized = "x" * 20_000
    provider = MutationProvider([
        AgentReply({"raw_text": "OPERATION=replace\nDONE=true\nPATH=style.css\nOLD-BEGIN-bad\nx\nNEW-BEGIN-bad\ny\nNEW-END-bad"}),
        AgentReply({"raw_text": oversized, "response_complete": True, "provider_attempt": 2}),
    ])
    runner, state, task = _runner(tmp_path, provider)

    with pytest.raises(Exception, match="MUTATION_PROTOCOL"):
        runner._execute_mutation_decision(state, task, {"path": "style.css", "intent": "replace the stylesheet"})

    assert css.read_text(encoding="utf-8") == original
    assert state.mutation_diagnostics
    diagnostic = state.mutation_diagnostics[-1]
    assert diagnostic["path"] == "style.css"
    assert diagnostic["response_bytes"] == len(oversized.encode("utf-8"))
    assert len(str(diagnostic["raw_response"]).encode("utf-8")) <= 16_000
    assert diagnostic["failure_type"] == "OVERSIZED_RAW_MUTATION"
    assert diagnostic["provider_attempt"] == 2


def test_raw_fallback_preserves_destructive_write_guard_for_substantial_file(tmp_path: Path) -> None:
    _repository(tmp_path)
    css = tmp_path / "style.css"
    original = ".card { color: red; }\n" + "/* accepted behavior */\n" * 40
    css.write_text(original, encoding="utf-8")
    provider = MutationProvider([
        AgentReply({"raw_text": "not a framed patch"}),
        AgentReply({"raw_text": ".card { color: blue; }\n", "response_complete": True}),
    ])
    runner, state, task = _runner(tmp_path, provider)

    with pytest.raises(Exception, match="MUTATION_PROTOCOL_RECOVERY_EXHAUSTED: UNSAFE_DESTRUCTIVE_REPLACEMENT"):
        runner._execute_mutation_decision(state, task, {"path": "style.css", "intent": "change card color"})

    assert css.read_text(encoding="utf-8") == original
    assert state.mutation_diagnostics[-1]["failure_type"] == "UNSAFE_DESTRUCTIVE_REPLACEMENT"


def test_raw_fallback_splits_large_existing_file_into_controller_selected_regions(tmp_path: Path) -> None:
    _repository(tmp_path)
    css = tmp_path / "style.css"
    original = "".join(f".item-{index} {{ color: red; }}\n" for index in range(500))
    css.write_text(original, encoding="utf-8")

    class RegionEchoProvider(ScriptedProvider):
        def __init__(self) -> None:
            super().__init__({})
            self.requests: list[AgentRequest] = []

        def complete(self, request: AgentRequest) -> AgentReply:
            self.requests.append(request)
            if not request.raw_mutation:
                return AgentReply({"raw_text": "not a valid framed mutation"})
            region = request.prompt.split("CURRENT EXACT REGION (replace all of this region with your response):\n", 1)[1]
            return AgentReply({"raw_text": region, "response_complete": True})

    provider = RegionEchoProvider()
    runner, state, task = _runner(tmp_path, provider)

    runner._execute_mutation_decision(state, task, {"path": "style.css", "intent": "preserve all selectors and normalize the stylesheet"})

    assert css.read_text(encoding="utf-8") == original
    assert sum(request.raw_mutation for request in provider.requests) >= 3
    assert all(len(request.prompt.split("CURRENT EXACT REGION (replace all of this region with your response):\n", 1)[1].encode("utf-8")) <= 3_000 for request in provider.requests if request.raw_mutation)
    assert task.attempts == 1
    assert metrics(state)["semantic_retry_count"] == 0


def test_workspace_raw_slice_replace_is_compare_and_swap_atomic(tmp_path: Path) -> None:
    _repository(tmp_path)
    tools = WorkspaceTools(tmp_path)
    path = tmp_path / "style.css"
    current = '.card { content: "red"; }\n'
    path.write_bytes(current.encode("utf-8"))
    expected_hash = hashlib.sha256(path.read_bytes()).hexdigest()

    assert tools.replace_slice_if_snapshot("style.css", expected_hash, 0, len(current), current, '.card { content: "blue"; }\n')
    assert path.read_text(encoding="utf-8") == '.card { content: "blue"; }\n'

    after_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    assert not tools.replace_slice_if_snapshot("style.css", expected_hash, 0, len(current), current, "corrupt")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == after_hash


def test_mutation_diagnostics_survive_state_store_roundtrip(tmp_path: Path) -> None:
    _repository(tmp_path)
    store = StateStore(tmp_path)
    state = ProjectState.create("Persist local mutation evidence")
    state.mutation_diagnostics.append({
        "mutation_request_id": "req-1", "protocol_mode": "raw_replacement", "path": "style.css",
        "expected_hash": "a" * 64, "response_bytes": 12, "failure_type": "MUTATION_CONTEXT_MISMATCH",
        "provider_attempt": 1, "raw_response": "body\n",
    })

    store.save(state)
    loaded = store.load()

    assert loaded is not None
    assert loaded.mutation_diagnostics == state.mutation_diagnostics
    assert (tmp_path / ".autodev" / "mutation_diagnostics.jsonl").exists()


def test_high_level_coder_actions_remain_structured_and_separate_from_raw_payload() -> None:
    provider = MutationProvider([AgentReply({"actions": [], "task_status": "ready_for_validation"})])
    agents = RoleAgents(provider)
    state = ProjectState.create("Return a small action decision")
    task = Task.create("Inspect", "Return no actions")

    agents.code(state, task)

    assert provider.requests[0].raw_response is False
    assert provider.requests[0].raw_mutation is False
    assert "actions" in provider.requests[0].prompt
