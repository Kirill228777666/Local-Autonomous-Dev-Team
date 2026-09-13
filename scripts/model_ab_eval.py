"""Sequential disposable-fixture evaluation for local Coder models."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from autodev.agents import ROLE_PROMPTS, RoleAgents
from autodev.providers import AgentRequest, OllamaProvider, ProviderError
from autodev.tools import WorkspaceTools


@dataclass(frozen=True, slots=True)
class EvalCase:
    name: str
    files: dict[str, str]
    instruction: str
    required: dict[str, tuple[str, ...]]


CASES = (
    EvalCase("targeted_python_edit", {"calculator.py": "def add(a, b):\n    return a - b\n"}, "Fix add without replacing unrelated code.", {"calculator.py": ("return a + b",)}),
    EvalCase("small_test_fix", {"slug.py": "def slug(value):\n    return value\n"}, "Make slug return lowercase text with spaces replaced by hyphens.", {"slug.py": ("lower", "replace")}),
    EvalCase("failed_noop", {"feature.py": "enabled = False\n"}, "Acceptance proves enabled must be True. A no-action answer is insufficient.", {"feature.py": ("True",)}),
    EvalCase("refreshed_stale_context", {"config.py": "HOST = '127.0.0.1'\nPORT = 4000\n"}, "Current content is authoritative. Change only PORT to 8000.", {"config.py": ("HOST = '127.0.0.1'", "PORT = 8000")}),
    EvalCase("api_crud", {"api.py": "from flask import Flask\napp = Flask(__name__)\n"}, "Add GET and POST /api/items routes while preserving app creation.", {"api.py": ("/api/items", "GET", "POST")}),
    EvalCase("frontend_behavior", {"app.js": "const button = document.querySelector('button');\n"}, "Add a click listener that fetches /api/items.", {"app.js": ("addEventListener", "fetch", "/api/items")}),
    EvalCase("dependency_incompatibility", {"requirements.txt": "Flask-SQLAlchemy==3.1.1\nSQLAlchemy==2.0.23\n"}, "Python 3.14 rejects SQLAlchemy 2.0.23. Update the SQLAlchemy constraint without changing Flask-SQLAlchemy.", {"requirements.txt": ("Flask-SQLAlchemy==3.1.1", "SQLAlchemy")}),
    EvalCase("preserve_regression", {"service.py": "def core():\n    return 'ok'\n\ndef feature():\n    raise NotImplementedError\n"}, "Implement feature returning 'ready'; preserve the accepted core behavior.", {"service.py": ("return 'ok'", "return 'ready'")}),
)


def evaluate_reply(workspace: Path, case: EvalCase, data: object) -> dict[str, object]:
    workspace.mkdir(parents=True, exist_ok=True)
    for relative, content in case.files.items():
        path = workspace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    result: dict[str, object] = {
        "case": case.name,
        "valid_structured_response": False,
        "valid_tool_actions": False,
        "noop": False,
        "tool_actions": 0,
        "full_file_rewrites": 0,
        "successful_task_fix": False,
    }
    if not isinstance(data, dict):
        return result
    result["valid_structured_response"] = True
    try:
        RoleAgents._valid_actions(data)
    except ValueError:
        return result
    actions = data["actions"]
    assert isinstance(actions, list)
    result["valid_tool_actions"] = True
    result["noop"] = not actions
    result["tool_actions"] = len(actions)
    tools = WorkspaceTools(workspace)
    for action in actions:
        assert isinstance(action, dict)
        kind = action["kind"]
        path = action.get("path")
        try:
            if kind == "write_file":
                assert isinstance(path, str)
                if (workspace / path).is_file():
                    result["full_file_rewrites"] = int(result["full_file_rewrites"]) + 1
                tools.write_file(path, str(action["content"]))
            elif kind == "edit_file":
                assert isinstance(path, str)
                tools.edit_file(path, str(action["old"]), str(action["new"]))
            elif kind == "append_file":
                assert isinstance(path, str)
                tools.append_file(path, str(action["content"]))
            elif kind == "delete_file":
                assert isinstance(path, str)
                tools.delete_file(path)
            else:
                continue
        except (OSError, ValueError):
            result["valid_tool_actions"] = False
            break
    result["successful_task_fix"] = all(
        (workspace / relative).is_file()
        and all(text in (workspace / relative).read_text(encoding="utf-8", errors="replace") for text in required)
        and not (case.name == "dependency_incompatibility" and "SQLAlchemy==2.0.23" in (workspace / relative).read_text(encoding="utf-8"))
        for relative, required in case.required.items()
    )
    return result


def _prompt(case: EvalCase) -> str:
    files = "\n\n".join(f"FILE {path}:\n{content}" for path, content in case.files.items())
    return (
        "Return JSON only: {\"actions\":[...]}. Allowed actions: write_file(path,content), "
        "edit_file(path,old,new), append_file(path,content), delete_file(path), run_command(command array). "
        "Prefer a targeted edit for an existing file and preserve unrelated behavior.\n\n"
        f"TASK: {case.instruction}\n\n{files}"
    )


def run_benchmark(models: list[str], output: Path, *, think: bool = False, limit: int | None = None) -> dict[str, object]:
    selected_cases = CASES[:limit] if limit is not None else CASES
    report: dict[str, object] = {"settings": {"temperature": 0.05, "context_limit": 16384, "think": think}, "models": {}}
    for model in models:
        provider = OllamaProvider(model=model, temperature=0.05, timeout=600, retries=0, context_limit=16384, think=think)
        case_results: list[dict[str, object]] = []
        for case in selected_cases:
            started = time.monotonic()
            try:
                reply = provider.complete(AgentRequest("CODER", _prompt(case), ROLE_PROMPTS["CODER"]))
                with tempfile.TemporaryDirectory(prefix="autodev-model-eval-") as directory:
                    result = evaluate_reply(Path(directory), case, reply.data)
            except ProviderError as error:
                result = evaluate_reply(Path(tempfile.mkdtemp(prefix="autodev-model-eval-failed-")), case, None)
                result["error"] = str(error)
            result["latency_seconds"] = round(time.monotonic() - started, 3)
            case_results.append(result)
        report["models"][model] = {
            "cases": case_results,
            "valid_structured_response_rate": sum(bool(item["valid_structured_response"]) for item in case_results) / len(case_results),
            "valid_tool_action_rate": sum(bool(item["valid_tool_actions"]) for item in case_results) / len(case_results),
            "noop_rate": sum(bool(item["noop"]) for item in case_results) / len(case_results),
            "successful_task_fix_rate": sum(bool(item["successful_task_fix"]) for item in case_results) / len(case_results),
            "average_latency_seconds": sum(float(item["latency_seconds"]) for item in case_results) / len(case_results),
            "tool_actions": sum(int(item["tool_actions"]) for item in case_results),
            "unnecessary_full_file_rewrites": sum(int(item["full_file_rewrites"]) for item in case_results),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    report = run_benchmark(args.models, args.output, think=args.think, limit=args.limit)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
