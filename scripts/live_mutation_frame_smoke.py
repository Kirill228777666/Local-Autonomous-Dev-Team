"""Exercise the real Coder patch request while fault-injecting a missing OLD-END marker."""

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
from autodev.providers import AgentRequest, OllamaProvider
from autodev.state_store import StateStore
from autodev.tools import WorkspaceTools


class MissingOldEndTransport:
    """Pass through one live Ollama response after removing its redundant OLD-END frame."""

    def __init__(self) -> None:
        self.injected = False
        self.requests = 0

    def __call__(self, url: str, payload: bytes, timeout: float) -> bytes:
        from autodev.providers import _http_transport

        raw = _http_transport(url, payload, timeout)
        self.requests += 1
        envelope = json.loads(raw)
        content = envelope.get("message", {}).get("content", "")
        prompt = "\n".join(
            message.get("content", "")
            for message in json.loads(payload).get("messages", [])
            if message.get("role") == "user"
        )
        request = re.search(r"Mutation request id: ([0-9a-f]+)", prompt)
        if not self.injected and isinstance(content, str) and request and "OPERATION=replace" in content:
            end = f"\nOLD-END-{request.group(1)}"
            if content.count(end) == 1:
                envelope["message"]["content"] = content.replace(end, "", 1)
                self.injected = True
        return json.dumps(envelope, ensure_ascii=False).encode("utf-8")


def _git(workspace: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=workspace, check=True, capture_output=True)


def run(config_path: Path) -> dict[str, object]:
    config = load_config(config_path)
    profile = config.profiles.get("CODER", AgentProfile(model=config.model, timeout=config.timeout))
    transport = MissingOldEndTransport()
    provider = OllamaProvider(
        model=profile.model,
        base_url=config.base_url,
        temperature=profile.temperature,
        timeout=profile.timeout,
        retries=profile.retries,
        context_limit=profile.context_limit,
        keep_alive=profile.keep_alive,
        transport=transport,
    )

    with tempfile.TemporaryDirectory(prefix="autodev-live-frame-") as temporary:
        workspace = Path(temporary)
        _git(workspace, "init")
        _git(workspace, "config", "user.email", "autodev-smoke@local")
        _git(workspace, "config", "user.name", "AutoDev Live Smoke")
        (workspace / ".gitignore").write_text(".autodev/\n", encoding="utf-8")
        css = workspace / "style.css"
        css.write_text(".card { color: red; }\n", encoding="utf-8")
        _git(workspace, "add", ".gitignore", "style.css")
        _git(workspace, "commit", "-m", "framing smoke fixture")

        state = ProjectState.create("Safely change the existing card color.")
        state.model = profile.model
        state.selected_primary_model = profile.model
        task = Task.create(
            "Change card color",
            "Replace exactly `.card { color: red; }` with `.card { color: blue; }`. Use a targeted replacement, not an appended override.",
        )
        task.attempts = 1
        state.tasks = [task]
        runner = AutonomousRunner(workspace, StateStore(workspace), WorkspaceTools(workspace), provider, max_attempts=1)
        runner._execute_coder_actions(state, task, [{
            "kind": "mutate_file",
            "path": "style.css",
            "intent": "Replace exactly `.card { color: red; }` with `.card { color: blue; }`; preserve the final newline.",
        }])

        observed = metrics(state)
        result = {
            "model": profile.model,
            "fault_injected": transport.injected,
            "provider_requests": transport.requests,
            "semantic_attempts": task.attempts,
            "mutation_frames_recovered": observed["mutation_frames_recovered"],
            "mutation_frames_ambiguous": observed["mutation_frames_ambiguous"],
            "mutation_frame_fallbacks": observed["mutation_frame_fallbacks"],
            "mutation_protocol_exhaustions": observed["coder_mutation_protocol_exhaustions"],
            "architect_escalations": sum(event.agent == "ARCHITECT" for event in state.events),
            "system_crashes": observed["system_crashes"],
            "resulting_file": css.read_text(encoding="utf-8"),
        }
        if profile.model != "qwen3.6:35b-coding":
            raise RuntimeError("live frame smoke requires the configured qwen3.6:35b-coding model: " + json.dumps(result))
        if not transport.injected or result["provider_requests"] != 1:
            raise RuntimeError("Ollama did not produce a replace frame for the controlled missing-OLD-END injection: " + json.dumps(result))
        if result["resulting_file"] != ".card { color: blue; }\n":
            raise RuntimeError("the exact targeted CSS replacement was not applied: " + json.dumps(result))
        if result["semantic_attempts"] != 1 or result["mutation_frames_recovered"] != 1:
            raise RuntimeError("the omitted OLD-END was not recovered within the same semantic attempt: " + json.dumps(result))
        if result["mutation_frames_ambiguous"] or result["mutation_frame_fallbacks"] or result["mutation_protocol_exhaustions"] or result["architect_escalations"] or result["system_crashes"]:
            raise RuntimeError("controlled frame recovery entered a fallback or failure path: " + json.dumps(result))
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("example-config.toml"))
    args = parser.parse_args()
    print(json.dumps(run(args.config), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
