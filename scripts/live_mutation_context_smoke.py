"""Verify stale/incorrect mutation context recovers before applying a real-model patch."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path

from autodev.cli import AgentProfile, load_config
from autodev.metrics import metrics
from autodev.models import ProjectState, Task
from autodev.orchestrator import AutonomousRunner
from autodev.providers import AgentReply, AgentRequest, OllamaProvider
from autodev.state_store import StateStore
from autodev.tools import WorkspaceTools


class OneStaleMutationProvider:
    """Inject one wrong anchor, then send the raw replacement request to Ollama."""

    def __init__(self, provider: OllamaProvider) -> None:
        self.provider = provider
        self.injected = False
        self.live_requests = 0
        self.raw_mutation_response = ""

    def complete(self, request: AgentRequest) -> AgentReply:
        if not self.injected and request.raw_response and not request.raw_mutation:
            prompt = request.prompt
            request_id = re.search(r"Mutation request id: ([0-9a-f]+)", prompt)
            path = re.search(r"^Path: (.+)$", prompt, re.MULTILINE)
            if request_id is None or path is None:
                raise RuntimeError("mutation request is missing its identity/path")
            marker, target = request_id.group(1), path.group(1)
            self.injected = True
            return AgentReply({
                "raw_text": (
                    f"OPERATION=replace\nDONE=true\nPATH={target}\n"
                    f"OLD-BEGIN-{marker}\n/* intentionally stale context */\nOLD-END-{marker}\n"
                    f"NEW-BEGIN-{marker}\n.card {{ color: blue; }}\nNEW-END-{marker}"
                ),
                "response_complete": True,
            })
        if request.raw_mutation:
            self.live_requests += 1
        reply = self.provider.complete(request)
        if request.raw_mutation:
            self.raw_mutation_response = str(reply.data.get("raw_text", ""))
        return reply


def _git(workspace: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=workspace, check=True, capture_output=True)


def run(config_path: Path) -> dict[str, object]:
    config = load_config(config_path)
    profile = config.profiles.get("CODER", AgentProfile(model=config.model, timeout=config.timeout))
    base = OllamaProvider(
        model=profile.model,
        base_url=config.base_url,
        temperature=profile.temperature,
        timeout=profile.timeout,
        retries=profile.retries,
        context_limit=profile.context_limit,
        keep_alive=profile.keep_alive,
    )
    provider = OneStaleMutationProvider(base)

    with tempfile.TemporaryDirectory(prefix="autodev-live-context-") as temporary:
        workspace = Path(temporary)
        _git(workspace, "init")
        _git(workspace, "config", "user.email", "autodev-smoke@local")
        _git(workspace, "config", "user.name", "AutoDev Live Smoke")
        (workspace / ".gitignore").write_text(".autodev/\n", encoding="utf-8")
        css = workspace / "style.css"
        css.write_text(".card { color: red; }\n", encoding="utf-8")
        _git(workspace, "add", ".gitignore", "style.css")
        _git(workspace, "commit", "-m", "context recovery smoke fixture")

        state = ProjectState.create("Safely change the card color after refreshing patch context.")
        state.model = profile.model
        state.selected_primary_model = profile.model
        task = Task.create("Change card color", "Change only `.card { color: red; }` to blue.")
        task.attempts = 1
        runner = AutonomousRunner(workspace, StateStore(workspace), WorkspaceTools(workspace), provider, max_attempts=1)
        runner._execute_coder_actions(state, task, [{
            "kind": "mutate_file",
            "path": "style.css",
            "intent": "Replace exactly `.card { color: red; }` with `.card { color: blue; }`.",
        }])

        observed = metrics(state)
        result = {
            "model": profile.model,
            "injected_context_mismatch": provider.injected,
            "real_model_mutation_calls": provider.live_requests,
            "semantic_attempts": task.attempts,
            "semantic_retries": observed["semantic_retry_count"],
            "context_mismatches": observed["mutation_context_mismatches"],
            "context_recovery_attempts": observed["mutation_context_recovery_attempts"],
            "context_recovery_successes": observed["mutation_context_recovery_successes"],
            "raw_mutation_requests": observed["raw_mutation_requests"],
            "raw_mutation_successes": observed["raw_mutation_successes"],
            "raw_mutation_response_bytes": len(provider.raw_mutation_response.encode("utf-8")),
            "protocol_exhaustions": observed["coder_mutation_protocol_exhaustions"],
            "architect_escalations": observed["architect_escalations"],
            "system_crashes": observed["system_crashes"],
            "resulting_file": css.read_text(encoding="utf-8"),
        }
        if profile.model != "qwen3.6:35b-coding":
            raise RuntimeError("live context smoke requires qwen3.6:35b-coding: " + json.dumps(result))
        if not provider.injected or provider.live_requests != 1:
            raise RuntimeError("expected one injected mismatch followed by one real raw model replacement: " + json.dumps(result))
        if result["resulting_file"].strip() != ".card { color: blue; }":
            raise RuntimeError("fresh-context mutation did not produce the expected CSS: " + json.dumps(result))
        if result["semantic_attempts"] != 1 or result["semantic_retries"] != 0:
            raise RuntimeError("context recovery consumed semantic retry budget: " + json.dumps(result))
        if result["context_mismatches"] != 1 or result["context_recovery_successes"] != 1 or result["raw_mutation_successes"] != 1:
            raise RuntimeError("context mismatch was not recovered: " + json.dumps(result))
        if result["protocol_exhaustions"] or result["architect_escalations"] or result["system_crashes"]:
            raise RuntimeError("context recovery escaped its bounded local path: " + json.dumps(result))
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("example-config.toml"))
    args = parser.parse_args()
    print(json.dumps(run(args.config), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
