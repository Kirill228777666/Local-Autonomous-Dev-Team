"""Run a bounded live Ollama smoke for large, multi-file Coder mutations."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from autodev.cli import AgentProfile, AppConfig, load_config
from autodev.metrics import metrics
from autodev.models import ProjectState, Task, TaskStatus
from autodev.orchestrator import AutonomousRunner
from autodev.providers import AgentReply, AgentRequest, OllamaProvider, RoleModelProvider, ScriptedProvider
from autodev.state_store import StateStore
from autodev.tools import WorkspaceTools


class RecordingProvider:
    def __init__(self, provider: OllamaProvider) -> None:
        self.provider = provider
        self.requests: list[AgentRequest] = []
        self.replies: list[dict[str, object]] = []

    def set_outcome_observer(self, observer: object) -> None:
        self.provider.set_outcome_observer(observer)  # type: ignore[arg-type]

    def health(self) -> bool:
        return self.provider.health()

    def complete(self, request: AgentRequest) -> AgentReply:
        self.requests.append(request)
        reply = self.provider.complete(request)
        self.replies.append(reply.data)
        return reply


def _git(workspace: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=workspace, check=True, capture_output=True)


def _profile(config: AppConfig) -> AgentProfile:
    return config.profiles.get("CODER", AgentProfile(model=config.model, timeout=config.timeout))


def run(config_path: Path, minimum_file_bytes: int = 52_000) -> dict[str, object]:
    config = load_config(config_path)
    profile = _profile(config)
    base = OllamaProvider(
        model=profile.model,
        base_url=config.base_url,
        temperature=profile.temperature,
        timeout=profile.timeout,
        retries=profile.retries,
        context_limit=profile.context_limit,
        keep_alive=profile.keep_alive,
    )
    coder = RecordingProvider(base)
    task = Task.create(
        "Generate two large source artifacts",
        f"Create large-a.py and large-b.py. Each file must be valid Python consisting only of comment lines and contain at least {minimum_file_bytes} characters. Add source only through bounded mutate_file patches; never use run_command to write files. Continue with small append sections until the size target is met.",
        capability_id="generated.large_artifacts",
        intent="Generate two complete large files through multiple bounded mutations.",
        acceptance_criteria=[
            f"large-a.py contains at least {minimum_file_bytes} bytes.",
            f"large-b.py contains at least {minimum_file_bytes} bytes.",
        ],
    )
    task.acceptance_validator = {
        "validator_id": "large-artifact-size",
        "capability_id": task.capability_id,
        "scope": "acceptance",
        "contract_version": task.contract_version,
        "command": [
            "py", "-3", "-c",
            "import ast; from pathlib import Path; paths=[Path('large-a.py'),Path('large-b.py')]; "
            f"assert all(p.stat().st_size >= {minimum_file_bytes} and ast.parse(p.read_text(encoding='utf-8')) for p in paths)",
        ],
    }
    with tempfile.TemporaryDirectory(prefix="autodev-live-mutation-") as temporary:
        workspace = Path(temporary)
        _git(workspace, "init")
        _git(workspace, "config", "user.email", "autodev-smoke@local")
        _git(workspace, "config", "user.name", "AutoDev Live Smoke")
        (workspace / ".gitignore").write_text(".autodev/\n", encoding="utf-8")
        _git(workspace, "add", ".gitignore")
        _git(workspace, "commit", "-m", "smoke fixture")
        deterministic = RoleModelProvider(
            coder,
            profiles={
                "TESTER": ScriptedProvider({"TESTER": [AgentReply({"command": task.acceptance_validator["command"]})]}),
                "REVIEWER": ScriptedProvider({"REVIEWER": [AgentReply({"approved": True, "reasons": []})]}),
            },
        )
        runner = AutonomousRunner(workspace, StateStore(workspace), WorkspaceTools(workspace), deterministic, max_attempts=1)
        state = ProjectState.create(task.description)
        state.model = profile.model
        state.selected_primary_model = profile.model
        state.tasks = [task]
        runner._run_task(state, task)

        sizes = {name: (workspace / name).stat().st_size if (workspace / name).is_file() else 0 for name in ("large-a.py", "large-b.py")}
        coder_replies = coder.replies
        decision_replies = [reply for reply in coder_replies if "actions" in reply]
        source_payload_lengths = [
            len(str(action.get(field, "")))
            for reply in decision_replies
            for action in reply.get("actions", [])
            if isinstance(action, dict)
            for field in ("content", "old", "new")
        ]
        patch_requests = [request for request in coder.requests if request.raw_response]
        result = {
            "model": profile.model,
            "status": task.status.value,
            "semantic_attempts": task.attempts,
            "file_sizes": sizes,
            "total_source_bytes": sum(sizes.values()),
            "mutation_patch_calls": len(patch_requests),
            "bounded_patch_output_requests": sum(
                request.max_output_tokens == 1_536 and request.max_output_characters == 16_000
                for request in patch_requests
            ),
            "max_high_level_source_payload_chars": max(source_payload_lengths, default=0),
            "protocol_recovery_exhaustions": sum(event.phase == "CODER_PROTOCOL_RECOVERY_EXHAUSTED" for event in state.events),
            "mutation_protocol_exhaustions": sum(event.phase == "MUTATION_PROTOCOL_RECOVERY_EXHAUSTED" for event in state.events),
            "architect_escalations": sum(event.agent == "ARCHITECT" for event in state.events),
            "system_crashes": sum(event.agent == "SYSTEM" and event.phase == "CRASHED" for event in state.events),
            "provider_requests_attempted": metrics(state)["llm_requests_attempted"],
            "provider_responses_completed": metrics(state)["llm_responses_completed"],
            "coder_reply_actions": [
                [
                    {
                        "kind": action.get("kind", action.get("action")),
                        "path": action.get("path"),
                        "command_length": sum(len(str(part)) for part in action.get("command", []))
                        if isinstance(action.get("command"), list) else 0,
                        "source_payload_chars": sum(
                            len(str(action.get(field, ""))) for field in ("content", "old", "new")
                        ),
                    }
                    for action in reply.get("actions", [])
                    if isinstance(action, dict)
                ]
                for reply in decision_replies
            ],
            "tool_executions": [
                {"kind": item.kind, "status": item.status.value, "detail": item.detail[:300]}
                for item in state.tool_executions
            ],
            "recovery_events": [
                {"phase": event.phase, "message": event.message[:300]}
                for event in state.events
                if "RECOVERY" in event.phase or "REJECTED" in event.phase or "EXHAUSTED" in event.phase or "MUTATION_" in event.phase
            ],
            "task_errors": list(task.errors),
            "recent_activity": [
                {"agent": event.agent, "phase": event.phase, "message": event.message[:300]}
                for event in state.events[-25:]
            ],
        }
        if task.status is not TaskStatus.DONE:
            raise RuntimeError("live mutation smoke task did not pass its deterministic validator: " + json.dumps(result))
        if task.attempts != 1 or len(patch_requests) < 2 or sum(sizes.values()) < 2 * minimum_file_bytes:
            raise RuntimeError("live mutation smoke failed its mutation protocol assertions: " + json.dumps(result))
        if result["max_high_level_source_payload_chars"] > 6_000:
            raise RuntimeError("large source appeared in the high-level decision response: " + json.dumps(result))
        if result["protocol_recovery_exhaustions"] or result["mutation_protocol_exhaustions"] or result["architect_escalations"] or result["system_crashes"]:
            raise RuntimeError("live mutation smoke hit a protocol/controller failure: " + json.dumps(result))
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("example-config.toml"))
    parser.add_argument("--minimum-file-bytes", type=int, default=52_000)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.minimum_file_bytes), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
