"""Raw, controller-bounded mutation transport regressions."""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

import pytest

from autodev.agents import MutationFrameError, RoleAgents
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
    assert provider.requests[0].max_output_tokens <= 1_000 // 3
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
    assert "WRITE SCOPE (replace only this exact region)" in provider.requests[1].prompt
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
        AgentReply({"raw_text": "alpha\n", "response_complete": True}),
        AgentReply({"raw_text": "unparseable frame two"}),
        AgentReply({"raw_text": "beta\n", "response_complete": True}),
    ])
    runner, state, task = _runner(tmp_path, provider)

    runner._execute_mutation_decision(state, task, {"path": "a.txt", "intent": "create first section"})
    runner._execute_mutation_decision(state, task, {"path": "b.txt", "intent": "create second section"})

    assert task.attempts == 1
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "alpha\n"
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "beta\n"
    assert metrics(state)["raw_mutation_successes"] == 2
    assert metrics(state)["semantic_retry_count"] == 0


def test_incomplete_raw_generation_is_size_pressure_not_valid_source() -> None:
    provider = MutationProvider([AgentReply({
        "raw_text": "partial source",
        "response_complete": False,
        "done_reason": "length",
        "provider_attempt": 1,
        "response_bytes": 14,
    })])
    agents = RoleAgents(provider)
    state = ProjectState.create("Edit CSS")
    task = Task.create("Replace CSS region", "Replace the selected region")

    with pytest.raises(MutationFrameError, match="did not finish") as error:
        agents.patch(
            state, task, path="style.css", intent="replace selected CSS", current="old { }\n", existing=True,
            continuation=0, request_id="req-incomplete", expected_file_hash="a" * 64,
            max_patch_characters=1_000, representation="raw_replacement", start_offset=0, end_offset=8,
        )

    assert error.value.classification == "OVERSIZED_RAW_MUTATION"
    assert error.value.response_bytes == 14


def test_attempt_rollback_restores_raw_mutation(tmp_path: Path) -> None:
    _repository(tmp_path)
    target = tmp_path / "style.css"
    target.write_text(".card { color: red; }\n", encoding="utf-8")
    provider = MutationProvider([
        AgentReply({"raw_text": "unparseable frame"}),
        AgentReply({"raw_text": ".card { color: blue; }\n", "response_complete": True}),
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
        if request.raw_mutation and "WRITE SCOPE (replace only this exact region)" in request.prompt and css.read_text(encoding="utf-8") == ".card { color: red; }\n":
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

    runner.mutation_decomposition_budget = 0
    with pytest.raises(Exception, match="MUTATION_DECOMPOSITION_EXHAUSTED"):
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
            region = request.prompt.split("WRITE SCOPE (replace only this exact region):\n", 1)[1].split("\nEND WRITE SCOPE", 1)[0]
            return AgentReply({"raw_text": region, "response_complete": True})

    provider = RegionEchoProvider()
    runner, state, task = _runner(tmp_path, provider)

    runner._execute_mutation_decision(state, task, {"path": "style.css", "intent": "preserve all selectors and normalize the stylesheet"})

    assert css.read_text(encoding="utf-8") == original
    assert sum(request.raw_mutation for request in provider.requests) >= 3
    assert all(len(request.prompt.split("WRITE SCOPE (replace only this exact region):\n", 1)[1].split("\nEND WRITE SCOPE", 1)[0].encode("utf-8")) <= 3_000 for request in provider.requests if request.raw_mutation)
    assert task.attempts == 1
    assert metrics(state)["semantic_retry_count"] == 0


def test_oversized_raw_response_adaptively_splits_scope_and_finishes_same_attempt(tmp_path: Path) -> None:
    _repository(tmp_path)
    source = "".join(f"const item{index:03d} = 'old';\n" for index in range(120))
    target = tmp_path / "app.js"
    target.write_text(source, encoding="utf-8")

    class AdaptiveProvider(ScriptedProvider):
        def __init__(self) -> None:
            super().__init__({})
            self.requests: list[AgentRequest] = []
            self.first_raw = True

        def complete(self, request: AgentRequest) -> AgentReply:
            self.requests.append(request)
            if not request.raw_mutation:
                return AgentReply({"raw_text": "not a valid framed mutation"})
            if self.first_raw:
                self.first_raw = False
                return AgentReply({"raw_text": "x" * 1_201, "response_complete": True})
            region = request.prompt.split("WRITE SCOPE (replace only this exact region):\n", 1)[1].split("\nEND WRITE SCOPE", 1)[0]
            return AgentReply({"raw_text": region.replace("'old'", "'new'"), "response_complete": True})

    provider = AdaptiveProvider()
    runner, state, task = _runner(tmp_path, provider)
    runner.mutation_payload_limit = 1_000

    runner._execute_mutation_decision(state, task, {"path": "app.js", "intent": "change every item value from old to new"})

    assert target.read_text(encoding="utf-8") == source.replace("'old'", "'new'")
    assert task.attempts == 1
    assert metrics(state)["semantic_retry_count"] == 0
    assert metrics(state)["raw_mutation_oversized"] == 1
    assert metrics(state)["mutation_decomposition_attempts"] >= 1
    assert metrics(state)["mutation_decomposition_successes"] >= 1
    assert metrics(state)["mutation_subregions_applied"] >= 2
    assert not any(event.agent == "ARCHITECT" for event in state.events)
    assert not any(event.phase == "ATTEMPT_ROLLBACK" for event in state.events)
    assert len(provider.requests) == metrics(state)["llm_requests_attempted"]
    scopes = [
        request.prompt.split("WRITE SCOPE (replace only this exact region):\n", 1)[1].split("\nEND WRITE SCOPE", 1)[0]
        for request in provider.requests if request.raw_mutation and "WRITE SCOPE (replace only this exact region):\n" in request.prompt
    ]
    assert len(scopes) >= 2
    assert all(len(scope.encode("utf-8")) < 1_000 for scope in scopes)
    raw_prompts = [request.prompt for request in provider.requests if request.raw_mutation]
    hashes = [re.search(r"Controller expected file SHA-256: ([0-9a-f]{64})", prompt).group(1) for prompt in raw_prompts]
    assert len(set(hashes[1:])) > 1
    read_context = raw_prompts[1].split("READ-ONLY SURROUNDING CONTEXT (for understanding only; never return it):\n", 1)[1].split("\nWRITE SCOPE", 1)[0]
    first_scope = raw_prompts[1].split("WRITE SCOPE (replace only this exact region):\n", 1)[1].split("\nEND WRITE SCOPE", 1)[0]
    assert len(read_context) > len(first_scope)
    oversize_diagnostic = next(item for item in state.mutation_diagnostics if item["failure_type"] == "OVERSIZED_RAW_MUTATION")
    assert oversize_diagnostic["write_scope_size"] > oversize_diagnostic["next_scope_size"]
    assert oversize_diagnostic["payload_limit"] == 1_000


def test_oversized_subregion_is_recursively_shrunk_with_a_fresh_hash(tmp_path: Path) -> None:
    _repository(tmp_path)
    source = "".join(f"function item{index:03d}() {{ return 'old'; }}\n" for index in range(80))
    target = tmp_path / "app.js"
    target.write_text(source, encoding="utf-8")

    class RecursiveProvider(ScriptedProvider):
        def __init__(self) -> None:
            super().__init__({})
            self.requests: list[AgentRequest] = []
            self.raw_count = 0

        def complete(self, request: AgentRequest) -> AgentReply:
            self.requests.append(request)
            if not request.raw_mutation:
                return AgentReply({"raw_text": "bad frame"})
            self.raw_count += 1
            if self.raw_count in {1, 2}:
                return AgentReply({"raw_text": "x" * 1_201, "response_complete": True})
            region = request.prompt.split("WRITE SCOPE (replace only this exact region):\n", 1)[1].split("\nEND WRITE SCOPE", 1)[0]
            return AgentReply({"raw_text": region.replace("'old'", "'new'"), "response_complete": True})

    provider = RecursiveProvider()
    runner, state, task = _runner(tmp_path, provider)
    runner.mutation_payload_limit = 1_000

    runner._execute_mutation_decision(state, task, {"path": "app.js", "intent": "change each return value to new"})

    assert target.read_text(encoding="utf-8") == source.replace("'old'", "'new'")
    assert metrics(state)["raw_mutation_oversized"] == 2
    assert metrics(state)["mutation_decomposition_attempts"] == 2
    assert metrics(state)["mutation_decomposition_successes"] >= 1
    assert task.attempts == 1
    assert metrics(state)["semantic_retry_count"] == 0


def test_adaptive_decomposition_exhaustion_is_protocol_only_and_bounded(tmp_path: Path) -> None:
    _repository(tmp_path)
    source = "".join(f"const item{index:03d} = 'old';\n" for index in range(120))
    target = tmp_path / "app.js"
    target.write_text(source, encoding="utf-8")

    class AlwaysOversizedProvider(ScriptedProvider):
        def __init__(self) -> None:
            super().__init__({})
            self.requests: list[AgentRequest] = []

        def complete(self, request: AgentRequest) -> AgentReply:
            self.requests.append(request)
            if request.raw_mutation:
                return AgentReply({"raw_text": "x" * 1_201, "response_complete": True})
            return AgentReply({"raw_text": "not a valid framed mutation"})

    provider = AlwaysOversizedProvider()
    runner, state, task = _runner(tmp_path, provider)
    runner.mutation_payload_limit = 1_000
    runner.mutation_decomposition_budget = 2

    with pytest.raises(Exception, match="MUTATION_DECOMPOSITION_EXHAUSTED"):
        runner._execute_mutation_decision(state, task, {"path": "app.js", "intent": "change every item value"})

    assert target.read_text(encoding="utf-8") == source
    assert task.attempts == 1
    assert metrics(state)["semantic_retry_count"] == 0
    assert metrics(state)["mutation_decomposition_exhaustions"] == 1
    assert not any(event.agent == "ARCHITECT" for event in state.events)
    assert not any(event.phase == "ATTEMPT_ROLLBACK" for event in state.events)


def test_attempt_rollback_restores_all_successful_adaptive_subregions(tmp_path: Path) -> None:
    _repository(tmp_path)
    original = "".join(f".item-{index} {{ color: red; }}\n" for index in range(120))
    target = tmp_path / "style.css"
    target.write_text(original, encoding="utf-8")

    class FailAfterTwoSubregions(ScriptedProvider):
        def __init__(self) -> None:
            super().__init__({})
            self.raw_count = 0

        def complete(self, request: AgentRequest) -> AgentReply:
            if not request.raw_mutation:
                return AgentReply({"raw_text": "bad frame"})
            self.raw_count += 1
            if self.raw_count <= 2:
                region = request.prompt.split("WRITE SCOPE (replace only this exact region):\n", 1)[1].split("\nEND WRITE SCOPE", 1)[0]
                return AgentReply({"raw_text": region.replace("red", "blue"), "response_complete": True})
            return AgentReply({"raw_text": "x" * 1_201, "response_complete": True})

    provider = FailAfterTwoSubregions()
    runner, state, task = _runner(tmp_path, provider)
    runner.mutation_payload_limit = 1_000
    runner.mutation_decomposition_budget = 0
    runner._begin_attempt_snapshot(task)

    with pytest.raises(Exception, match="MUTATION_DECOMPOSITION_EXHAUSTED"):
        runner._execute_mutation_decision(state, task, {"path": "style.css", "intent": "change color to blue"})
    assert target.read_text(encoding="utf-8") != original
    runner._rollback_attempt_snapshot(task, state, "protocol_failure")

    assert target.read_text(encoding="utf-8") == original
    assert metrics(state)["mutation_subregions_applied"] == 2


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
