"""Exercise adaptive raw mutation sizing with real Ollama follow-up patches."""

from __future__ import annotations

import json
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


class OversizeOnceProvider:
    """Force one transport overflow, then use the real model for every patch."""

    def __init__(self, provider: OllamaProvider, *, limit: int) -> None:
        self.provider = provider
        self.limit = limit
        self.requests: list[AgentRequest] = []
        self.real_raw_calls = 0
        self.injected_oversize = False

    def complete(self, request: AgentRequest) -> AgentReply:
        self.requests.append(request)
        if not request.raw_mutation:
            return AgentReply({"raw_text": "not a valid framed mutation", "response_complete": True})

        self.real_raw_calls += 1
        reply = self.provider.complete(request)
        if not self.injected_oversize:
            self.injected_oversize = True
            source = str(reply.data.get("raw_text", ""))
            # Preserve a real model response while deterministically exercising
            # the overflow branch; this reply is discarded and never applied.
            amplified = source + ("\n/* discarded overflow probe */\n" * (self.limit // 28 + 2))
            return AgentReply({
                **reply.data,
                "raw_text": amplified,
                "response_bytes": len(amplified.encode("utf-8")),
                "response_complete": True,
            })
        return reply


def _git(workspace: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=workspace, check=True, capture_output=True)


def run(config_path: Path, limit: int = 1_000) -> dict[str, object]:
    config = load_config(config_path)
    profile = config.profiles.get("CODER", AgentProfile(model=config.model, timeout=config.timeout))
    ollama = OllamaProvider(
        model=profile.model,
        base_url=config.base_url,
        temperature=profile.temperature,
        timeout=profile.timeout,
        retries=profile.retries,
        context_limit=profile.context_limit,
        keep_alive=profile.keep_alive,
    )
    provider = OversizeOnceProvider(ollama, limit=limit)

    with tempfile.TemporaryDirectory(prefix="autodev-live-adaptive-mutation-") as temporary:
        workspace = Path(temporary)
        _git(workspace, "init")
        _git(workspace, "config", "user.email", "autodev-smoke@local")
        _git(workspace, "config", "user.name", "AutoDev Live Smoke")
        (workspace / ".gitignore").write_text(".autodev/\n", encoding="utf-8")
        css = workspace / "style.css"
        original = "".join(
            f".card-{index:02d} {{\n  color: red;\n  margin: {index}px;\n}}\n"
            for index in range(24)
        )
        css.write_text(original, encoding="utf-8")
        _git(workspace, "add", ".gitignore", "style.css")
        _git(workspace, "commit", "-m", "adaptive mutation smoke fixture")

        state = ProjectState.create("Change the color in every card rule from red to blue.")
        state.model = profile.model
        state.selected_primary_model = profile.model
        task = Task.create("Update card colors", "Change every `.card-N` color from red to blue and preserve layout.")
        task.attempts = 1
        runner = AutonomousRunner(workspace, StateStore(workspace), WorkspaceTools(workspace), provider, max_attempts=1)
        runner.mutation_payload_limit = limit
        runner._execute_coder_actions(state, task, [{
            "kind": "mutate_file",
            "path": "style.css",
            "intent": "Change only each card rule's `color: red` declaration to `color: blue`; preserve every selector and margin.",
        }])

        actual = css.read_text(encoding="utf-8")
        observed = metrics(state)
        result: dict[str, object] = {
            "model": profile.model,
            "raw_payload_limit": limit,
            "injected_oversize_probe": provider.injected_oversize,
            "real_raw_model_calls": provider.real_raw_calls,
            "semantic_attempts": task.attempts,
            "semantic_retries": observed["semantic_retry_count"],
            "raw_mutation_oversized": observed["raw_mutation_oversized"],
            "mutation_decomposition_attempts": observed["mutation_decomposition_attempts"],
            "mutation_decomposition_successes": observed["mutation_decomposition_successes"],
            "mutation_subregions_applied": observed["mutation_subregions_applied"],
            "architect_escalations": observed["architect_escalations"],
            "system_crashes": observed["system_crashes"],
            "llm_requests_attempted": observed["llm_requests_attempted"],
            "llm_responses_completed": observed["llm_responses_completed"],
            "all_card_colors_blue": actual.count("color: blue;") == 24 and "color: red;" not in actual,
            "result_bytes": len(actual.encode("utf-8")),
        }
        if profile.model != "qwen3.6:35b-coding":
            raise RuntimeError("adaptive mutation smoke requires qwen3.6:35b-coding: " + json.dumps(result))
        if not provider.injected_oversize or provider.real_raw_calls < 3:
            raise RuntimeError("adaptive smoke did not perform overflow plus multiple live raw calls: " + json.dumps(result))
        if not result["all_card_colors_blue"]:
            raise RuntimeError("real model did not complete the bounded CSS intent: " + json.dumps(result))
        if result["semantic_attempts"] != 1 or result["semantic_retries"] != 0:
            raise RuntimeError("adaptive mutation consumed semantic retry budget: " + json.dumps(result))
        if not result["raw_mutation_oversized"] or not result["mutation_decomposition_attempts"] or not result["mutation_decomposition_successes"]:
            raise RuntimeError("adaptive sizing metrics do not show successful decomposition: " + json.dumps(result))
        if result["architect_escalations"] or result["system_crashes"]:
            raise RuntimeError("adaptive mutation escaped the protocol layer: " + json.dumps(result))
        if result["llm_requests_attempted"] != result["llm_responses_completed"]:
            raise RuntimeError("provider accounting is inconsistent: " + json.dumps(result))
        return result


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("example-config.toml"))
    parser.add_argument("--limit", type=int, default=1_000)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.limit), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
