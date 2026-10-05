"""Staged source bodies keep semantic size independent from transport chunks."""

from __future__ import annotations

import hashlib
import subprocess
import time
from pathlib import Path

import pytest

from autodev.agents import MutationFrameError, RoleAgents
from autodev.metrics import metrics
from autodev.models import ProjectState, Task
from autodev.mutation import MutationStreamError, StagedMutationBody
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


class ChunkProvider(ScriptedProvider):
    def __init__(self, chunks: list[tuple[str, str]], target: Path | None = None) -> None:
        super().__init__({})
        self.chunks = list(chunks)
        self.requests: list[AgentRequest] = []
        self.target = target
        self.target_observations: list[str | None] = []
        self.on_raw = None

    def complete(self, request: AgentRequest) -> AgentReply:
        started = time.monotonic()
        if self._outcome_observer is not None:
            self._outcome_observer("CODER", "ATTEMPT", 0.0)
        self.requests.append(request)
        if self.target is not None:
            self.target_observations.append(self.target.read_text(encoding="utf-8") if self.target.exists() else None)
        if not request.raw_mutation:
            reply = AgentReply({"raw_text": "not a valid framed mutation", "response_complete": True, "done_reason": "stop"})
            if self._outcome_observer is not None:
                self._outcome_observer("CODER", "SUCCESS", time.monotonic() - started)
            return reply
        if not self.chunks:
            raise AssertionError("unexpected extra mutation chunk request")
        text, reason = self.chunks.pop(0)
        reply = AgentReply({
            "raw_text": text,
            "response_complete": True,
            "done_reason": reason,
            "response_bytes": len(text.encode("utf-8")),
            "provider_attempt": 1,
        })
        if request.raw_mutation and self.on_raw is not None:
            self.on_raw(request)
        if self._outcome_observer is not None:
            self._outcome_observer("CODER", "SUCCESS", time.monotonic() - started)
        return reply


def _runner(path: Path, provider: ScriptedProvider) -> tuple[AutonomousRunner, ProjectState, Task]:
    state = ProjectState.create("Generate a complete source file safely")
    task = Task.create("Create source", "Create the requested file")
    task.attempts = 1
    return AutonomousRunner(path, StateStore(path), WorkspaceTools(path), provider), state, task


def _staged(*, limit: int = 100_000, chunks: int = 64) -> StagedMutationBody:
    return StagedMutationBody(
        request_id="request-1",
        path="index.html",
        expected_file_hash=hashlib.sha256(b"").hexdigest(),
        start_offset=0,
        end_offset=0,
        expected_slice="",
        max_bytes=limit,
        max_chunks=chunks,
    )


def test_twenty_kib_body_is_assembled_from_bounded_two_kib_chunks() -> None:
    body = "".join(f"section-{index:05d}: unique source payload abcdefghijklmnopqrstuvwxyz\n" for index in range(340))[:20_480]
    chunks = [body[index:index + 2_000] for index in range(0, len(body), 2_000)]
    staged = _staged(limit=25_000)

    for index, chunk in enumerate(chunks):
        complete = staged.add_chunk(
            chunk,
            response_complete=True,
            done_reason="stop" if index == len(chunks) - 1 else "length",
        )
        assert complete is (index == len(chunks) - 1)
        assert len(chunk.encode("utf-8")) <= 2_000

    assert staged.complete
    assert staged.content == body
    assert staged.total_bytes == len(body.encode("utf-8"))
    assert staged.chunk_count == len(chunks)


def test_fifty_kib_source_body_does_not_require_one_large_provider_response() -> None:
    body = "<section>\n" + "".join(f"<p data-item='{index:05d}'>semantic source body {index:05d}</p>\n" for index in range(1_000)) + "</section>"
    staged = _staged(limit=80_000, chunks=100)
    pieces = [body[index:index + 1_024] for index in range(0, len(body), 1_024)]

    for index, piece in enumerate(pieces):
        staged.add_chunk(piece, response_complete=True, done_reason="stop" if index == len(pieces) - 1 else "length")

    assert staged.content == body
    assert max(len(piece.encode("utf-8")) for piece in pieces) <= 1_024


def test_large_staged_body_joins_many_chunks_without_quadratic_overlap_scan() -> None:
    body = "".join(f"<item id='{index:06d}'>payload-{index:06d}-abcdef</item>\n" for index in range(4_000))
    pieces = [body[index:index + 1_024] for index in range(0, len(body), 1_024)]
    staged = _staged(limit=len(body.encode("utf-8")) + 1, chunks=len(pieces) + 1)

    for index, piece in enumerate(pieces):
        staged.add_chunk(piece, response_complete=True, done_reason="stop" if index == len(pieces) - 1 else "length")

    assert staged.complete
    assert staged.content == body
    assert staged.chunk_count == len(pieces)


def test_length_finish_reason_keeps_chunk_incomplete_and_stop_finishes_body() -> None:
    staged = _staged()

    assert staged.add_chunk("const title = ", response_complete=True, done_reason="length") is False
    assert staged.complete is False
    assert staged.add_chunk('"Notes";\n', response_complete=True, done_reason="stop") is True
    assert staged.content == 'const title = "Notes";\n'


def test_duplicate_chunk_is_rejected_as_no_progress() -> None:
    staged = _staged()
    staged.add_chunk("same substantial chunk body", response_complete=True, done_reason="length")

    with pytest.raises(MutationStreamError, match="NO_PROGRESS"):
        staged.add_chunk("same substantial chunk body", response_complete=True, done_reason="length")


def test_unique_exact_chunk_overlap_is_removed_without_fuzzy_joining() -> None:
    staged = _staged()
    first = "A" * 80 + "UNIQUE-BOUNDARY-" + "B" * 40
    second = "UNIQUE-BOUNDARY-" + "B" * 40 + "tail"
    staged.add_chunk(first, response_complete=True, done_reason="length")

    assert staged.add_chunk(second, response_complete=True, done_reason="stop") is True
    assert staged.content == first + "tail"
    assert staged.last_join_outcome == "EXACT_OVERLAP_REMOVED"


def test_ambiguous_repeated_overlap_is_rejected() -> None:
    staged = _staged()
    anchor = "0123456789abcdefghijklmnopqrstuv"
    staged.add_chunk(anchor + " middle " + anchor, response_complete=True, done_reason="length")
    before = staged.content

    with pytest.raises(MutationStreamError, match="AMBIGUOUS"):
        staged.add_chunk(anchor + "tail", response_complete=True, done_reason="stop")
    assert staged.content == before
    assert not staged.complete


def test_staging_enforces_total_bytes_and_chunk_count_without_partial_completion() -> None:
    too_small = _staged(limit=5)
    with pytest.raises(MutationStreamError, match="MAX_BYTES"):
        too_small.add_chunk("123456", response_complete=True, done_reason="stop")
    assert too_small.content == ""

    too_few = _staged(chunks=1)
    too_few.add_chunk("part", response_complete=True, done_reason="length")
    with pytest.raises(MutationStreamError, match="MAX_CHUNKS"):
        too_few.add_chunk("rest", response_complete=True, done_reason="stop")
    assert not too_few.complete


def test_agent_returns_output_limited_raw_response_as_transport_chunk() -> None:
    provider = ScriptedProvider({"CODER": [AgentReply({
        "raw_text": "partial CSS {\n",
        "response_complete": True,
        "done_reason": "length",
        "response_bytes": 14,
    })]})
    agents = RoleAgents(provider)

    reply = agents.patch(
        ProjectState.create("Edit CSS"), Task.create("Edit", "Replace CSS"),
        path="style.css", intent="create the replacement", current="", existing=False,
        continuation=0, request_id="chunk-1", expected_file_hash=hashlib.sha256(b"").hexdigest(),
        max_patch_characters=2_000, representation="raw_replacement", start_offset=0, end_offset=0,
    )

    assert reply.data["content"] == "partial CSS {\n"
    assert reply.data["body_complete"] is False
    assert reply.data["transport_status"] == "OUTPUT_LIMIT_REACHED"


def test_new_file_is_invisible_until_final_chunk_then_created_atomically(tmp_path: Path) -> None:
    _repository(tmp_path)
    target = tmp_path / "index.html"
    content = "<!doctype html>\n" + "".join(f"<section id='section-{index:03d}'>content {index:03d}</section>\n" for index in range(200))
    chunks = [(content[index:index + 700], "length") for index in range(0, len(content), 700)]
    chunks[-1] = (chunks[-1][0], "stop")
    provider = ChunkProvider(chunks, target)
    runner, state, task = _runner(tmp_path, provider)
    runner.mutation_payload_limit = 2_000

    runner._execute_mutation_decision(state, task, {"path": "index.html", "intent": "create the complete page"})

    assert target.read_text(encoding="utf-8") == content
    assert provider.target_observations and all(item is None for item in provider.target_observations)
    assert task.attempts == 1
    assert len([request for request in provider.requests if request.raw_mutation]) == len(chunks)
    assert runner._active_mutation_requests == {}


def test_existing_file_is_atomically_replaced_after_multichunk_staging(tmp_path: Path) -> None:
    _repository(tmp_path)
    target = tmp_path / "app.js"
    target.write_text("old source\n", encoding="utf-8")
    replacement = "const view = {\n" + "".join(f"  title{index:03d}: \"Task Board {index:03d}\",\n" for index in range(250)) + "};\n"
    chunks = [(replacement[index:index + 900], "length") for index in range(0, len(replacement), 900)]
    chunks[-1] = (chunks[-1][0], "stop")
    provider = ChunkProvider(chunks, target)
    runner, state, task = _runner(tmp_path, provider)
    runner.mutation_payload_limit = 2_000
    runner._begin_attempt_snapshot(task)

    runner._execute_mutation_decision(state, task, {"path": "app.js", "intent": "replace the complete file"})

    assert target.read_text(encoding="utf-8") == replacement
    assert provider.target_observations == ["old source\n"] + ["old source\n"] * len(chunks)
    runner._rollback_attempt_snapshot(task, state, "acceptance_failure")
    assert target.read_text(encoding="utf-8") == "old source\n"


def test_mutation_provider_uses_bounded_output_setting_and_continuation_state(tmp_path: Path) -> None:
    _repository(tmp_path)
    target = tmp_path / "index.html"
    body = "<main>\n" + "".join(f"<p id='item-{index:03d}'>one bounded chunk {index:03d}</p>\n" for index in range(80)) + "</main>\n"
    pieces = [body[index:index + 1_000] for index in range(0, len(body), 1_000)]
    chunks = [(piece, "stop" if index == len(pieces) - 1 else "length") for index, piece in enumerate(pieces)]
    provider = ChunkProvider(chunks, target)
    runner, state, task = _runner(tmp_path, provider)
    runner.mutation_payload_limit = 2_000

    runner._execute_mutation_decision(state, task, {"path": "index.html", "intent": "create page"})

    raw_requests = [request for request in provider.requests if request.raw_mutation]
    assert len(raw_requests) == len(pieces)
    assert "TRANSPORT CHUNK 2" in raw_requests[1].prompt
    assert body[:1_000][-200:] in raw_requests[1].prompt
    assert target.read_text(encoding="utf-8") == body


def test_staged_generation_does_not_mutate_until_stream_is_complete(tmp_path: Path) -> None:
    _repository(tmp_path)
    target = tmp_path / "index.html"
    chunks = [("<html>\n", "length"), ("<body>ok</body>\n</html>\n", "stop")]
    provider = ChunkProvider(chunks, target)
    runner, state, task = _runner(tmp_path, provider)

    runner._execute_mutation_decision(state, task, {"path": "index.html", "intent": "create a page"})

    assert provider.target_observations == [None, None]
    assert target.read_text(encoding="utf-8") == "<html>\n<body>ok</body>\n</html>\n"


def test_small_single_response_raw_mutation_remains_one_chunk(tmp_path: Path) -> None:
    _repository(tmp_path)
    target = tmp_path / "style.css"
    body = ".card { color: blue; }\n"
    provider = ChunkProvider([(body, "stop")], target)
    runner, state, task = _runner(tmp_path, provider)

    runner._execute_mutation_decision(state, task, {"path": "style.css", "intent": "create card style"})

    assert target.read_text(encoding="utf-8") == body
    assert len([request for request in provider.requests if request.raw_mutation]) == 1


def test_stale_file_hash_discards_unfinished_body_and_restarts_against_current_file(tmp_path: Path) -> None:
    _repository(tmp_path)
    target = tmp_path / "style.css"
    target.write_text(".card { color: red; }\n", encoding="utf-8")

    def concurrent_change(request: AgentRequest) -> None:
        if request.raw_mutation and "TRANSPORT CHUNK 1" not in request.prompt and target.read_text(encoding="utf-8") == ".card { color: red; }\n":
            target.write_text("/* external */\n.card { color: red; }\n", encoding="utf-8")

    provider = ChunkProvider([
        (".card { color:", "length"),
        ("/* external */\n.card { color: blue; }\n", "stop"),
    ], target)
    provider.on_raw = concurrent_change
    runner, state, task = _runner(tmp_path, provider)

    runner._execute_mutation_decision(state, task, {"path": "style.css", "intent": "set card color blue"})

    assert target.read_text(encoding="utf-8") == "/* external */\n.card { color: blue; }\n"
    assert task.attempts == 1
    assert any(event.phase == "MUTATION_STALE_CONTEXT" for event in state.events)
    assert any(item["failure_type"] == "STALE_MUTATION_CONTEXT" for item in state.mutation_diagnostics)
    assert metrics(state)["semantic_retry_count"] == 0


def test_chunk_budget_exhaustion_never_exposes_partial_new_file(tmp_path: Path) -> None:
    _repository(tmp_path)
    target = tmp_path / "index.html"
    provider = ChunkProvider([("<main>partial", "length"), (" continuation", "length")], target)
    runner, state, task = _runner(tmp_path, provider)
    runner.mutation_patch_budget = 2

    with pytest.raises(Exception, match="MUTATION_PROTOCOL_RECOVERY_EXHAUSTED"):
        runner._execute_mutation_decision(state, task, {"path": "index.html", "intent": "create full HTML"})

    assert not target.exists()
    assert not any(item.kind == "write_file" for item in state.tool_executions)
    assert metrics(state)["semantic_retry_count"] == 0


def test_staged_chunk_diagnostics_keep_transport_identity_and_completion_metadata(tmp_path: Path) -> None:
    _repository(tmp_path)
    target = tmp_path / "index.html"
    provider = ChunkProvider([("<main>chunk one\n", "length"), ("chunk two</main>\n", "stop")], target)
    runner, state, task = _runner(tmp_path, provider)

    runner._execute_mutation_decision(state, task, {"path": "index.html", "intent": "create HTML"})

    chunks = [item for item in state.mutation_diagnostics if item["failure_type"] == "MUTATION_CHUNK_STAGED"]
    assert [item["chunk_index"] for item in chunks] == [1, 2]
    assert [item["provider_finish_reason"] for item in chunks] == ["length", "stop"]
    assert chunks[0]["final_application_outcome"] == "STAGING"
    assert chunks[-1]["final_application_outcome"] == "PENDING"
    assert state.mutation_diagnostics[-1]["final_application_outcome"] == "APPLIED"
    assert all(item["mutation_request_id"] == chunks[0]["mutation_request_id"] for item in chunks)


def test_completed_chunks_count_as_provider_progress_but_not_semantic_retries(tmp_path: Path) -> None:
    _repository(tmp_path)
    target = tmp_path / "index.html"
    body = "".join(f"<p>{index:04d}</p>\n" for index in range(150))
    pieces = [body[index:index + 600] for index in range(0, len(body), 600)]
    provider = ChunkProvider([(piece, "stop" if index == len(pieces) - 1 else "length") for index, piece in enumerate(pieces)])
    runner, state, task = _runner(tmp_path, provider)
    runner._execute_mutation_decision(state, task, {"path": "index.html", "intent": "create list"})

    assert (tmp_path / "index.html").read_text(encoding="utf-8") == body
    assert metrics(state)["mutation_stream_chunks"] == len(pieces)
    assert metrics(state)["semantic_retry_count"] == 0
    assert not any(event.agent == "ARCHITECT" for event in state.events)
    assert metrics(state)["llm_requests_attempted"] == len(provider.requests)
