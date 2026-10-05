"""Bounded live-Ollama smoke for staged raw mutation continuation.

This deliberately uses a small per-response token/byte budget and verifies that
the real provider performs multiple generations before one atomic file create.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from autodev.metrics import metrics
from autodev.models import ProjectState, Task
from autodev.orchestrator import AutonomousRunner
from autodev.providers import OllamaProvider
from autodev.state_store import StateStore
from autodev.tools import WorkspaceTools


def main() -> int:
    model = "qwen3.6:35b-coding"
    with tempfile.TemporaryDirectory(prefix="autodev-staged-ollama-") as temporary:
        workspace = Path(temporary)
        provider = OllamaProvider(
            model=model,
            base_url="http://127.0.0.1:11434",
            timeout=600,
            retries=0,
            context_limit=4_096,
            think=False,
        )
        if not provider.health():
            raise RuntimeError("local Ollama /api/tags health check failed")
        state = ProjectState.create("Generate a complete responsive HTML task board as one staged source body")
        state.selected_primary_model = model
        task = Task.create("Generate staged HTML", "Create a complete self-contained interactive responsive HTML page")
        task.attempts = 1
        state.tasks.append(task)
        runner = AutonomousRunner(workspace, StateStore(workspace), WorkspaceTools(workspace), provider)
        runner.mutation_transport_chunk_tokens = 256
        runner.mutation_transport_chunk_bytes = 1_800
        runner.mutation_max_staged_bytes = 80_000
        runner.mutation_max_staged_chunks = 96
        runner.mutation_patch_budget = 96
        runner._active_state = state
        runner._execute_mutation_decision(state, task, {
            "path": "index.html",
            "intent": (
                "Create a complete self-contained task board webpage with semantic HTML, responsive CSS, "
                "and working JavaScript for adding, completing, deleting, and filtering tasks. Include at "
                "least 24 distinct initial task rows with unique ids and varied titles. The page must have "
                "accessible labels, desktop and narrow-screen layout, "
                "and no external dependencies. Return the entire finished file body as continued source text."
            ),
        })
        target = workspace / "index.html"
        source = target.read_text(encoding="utf-8")
        observed = metrics(state)
        if len(source.encode("utf-8")) < 2_000:
            raise AssertionError(f"generated source is not large enough to exercise streaming: {len(source.encode('utf-8'))} bytes")
        if observed["mutation_stream_chunks"] < 2 or observed["mutation_stream_continuations"] < 1:
            raise AssertionError("real Ollama did not produce multiple output-limited mutation chunks")
        if observed["coder_semantic_attempts"] != 1 or observed["semantic_retry_count"] != 0:
            raise AssertionError("mutation transport consumed a semantic attempt")
        if observed["architect_escalations"] or observed["system_crashes"] or observed["mutation_stream_failures"]:
            raise AssertionError("staged mutation smoke entered an unexpected failure path")
        if observed["provider_request_outcome_gap"] != 0:
            raise AssertionError("provider request accounting is incomplete")
        print(json.dumps({
            "model": model,
            "workspace": str(workspace),
            "file": str(target),
            "file_bytes": len(source.encode("utf-8")),
            "provider_requests": observed["llm_requests_attempted"],
            "staged_chunks": observed["mutation_stream_chunks"],
            "continuations": observed["mutation_stream_continuations"],
            "semantic_attempts": observed["coder_semantic_attempts"],
            "semantic_retries": observed["semantic_retry_count"],
            "architect_escalations": observed["architect_escalations"],
            "system_crashes": observed["system_crashes"],
            "terminal": "PASS",
        }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
