"""Sequential disposable-fixture evaluation for local Coder models."""

from __future__ import annotations

import argparse
import json
import re
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from autodev.agents import RoleAgents
from autodev.models import ProjectState, Task
from autodev.providers import OllamaProvider, ProviderError
from autodev.tools import WorkspaceTools


@dataclass(frozen=True, slots=True)
class EvalCase:
    name: str
    files: dict[str, str]
    instruction: str
    required: dict[str, tuple[str, ...]]
    required_actions: tuple[str, ...] = ()


# These fixtures are deliberately small, but mirror failures observed in the
# live endurance runs.  They exercise the same Coder schema/API route as the
# product rather than asking a general chat question.
CASES = (
    EvalCase("sqlalchemy_engine_has_table", {"db.py": "def table_exists(engine, name):\n    return engine.has_table(name)\n"}, "SQLAlchemy 2.x raises AttributeError because Engine.has_table is removed. Make the minimal compatible repair using the current public inspection API; preserve function signature.", {"db.py": ("inspect", "has_table")}),
    EvalCase("module_level_flask_contract", {"app.py": "from flask import Flask\napp = Flask(__name__)\n", "tests/test_app.py": "from app import create_app\nclient = create_app().test_client()\n"}, "The authoritative project contract says app.py exports module-level symbol app, not create_app. Repair only the lower-authority generated test; do not rewrite app architecture.", {"tests/test_app.py": ("from app import app", "app.test_client")}),
    EvalCase("category_response_requires_id", {"api.py": "def create_category(name):\n    category = {'name': name}\n    return category, 201\n"}, "Acceptance requires POST category response to include persisted category id and name. Make the minimal implementation repair; do not weaken the response contract.", {"api.py": ("'id'", "'name'")}),
    EvalCase("http_error_contract", {"api.py": "def delete_note(store, note_id):\n    note = store[note_id]\n    del store[note_id]\n    return '', 204\n"}, "Deleting a missing note must return HTTP 404 and duplicate category creation must return 409 rather than an unhandled 500. Repair application error handling without weakening requirements.", {"api.py": ("404", "409")}),
    EvalCase("managed_flask_runtime", {"app.py": "from flask import Flask\napp = Flask(__name__)\nif __name__ == '__main__':\n    app.run()\n"}, "Start this long-running Flask server for validation using the managed process runtime. Do not use finite run_command for a server.", {}, ("start_process",)),
    EvalCase("preserve_accepted_regression", {"service.py": "def core():\n    return 'ok'\n\ndef feature():\n    raise NotImplementedError\n"}, "Implement feature returning 'ready'. core() is protected accepted behavior and must remain exactly 'ok'; make a minimal regression-safe change.", {"service.py": ("return 'ok'", "return 'ready'")}),
)


def _version_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", value))


def _excludes_incompatible_sqlalchemy(content: str) -> bool:
    """Return whether the manifest cannot resolve the known-bad 2.0.23 release."""
    line = next((item.strip() for item in content.splitlines() if item.strip().lower().startswith("sqlalchemy")), "")
    if not line:
        return False
    constraints = line[len("SQLAlchemy"):].replace(" ", "").split(",")
    bad = _version_tuple("2.0.23")
    for constraint in constraints:
        if constraint == "!=2.0.23":
            return True
        match = re.fullmatch(r"(==|>=|>)([0-9][0-9.]*)", constraint)
        if not match:
            continue
        operator, version = match.groups()
        parsed = _version_tuple(version)
        if operator == "==" and parsed != bad:
            return True
        if operator == ">" and parsed >= bad:
            return True
        if operator == ">=" and parsed > bad:
            return True
    return False


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
        "raw_reply": data,
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
        and not (
            case.name == "dependency_incompatibility"
            and not _excludes_incompatible_sqlalchemy((workspace / relative).read_text(encoding="utf-8"))
        )
        for relative, required in case.required.items()
    ) and all(
        any(isinstance(action, dict) and action.get("kind") == kind for action in actions)
        for kind in case.required_actions
    )
    return result


def run_benchmark(models: list[str], output: Path, *, think: bool = False, limit: int | None = None) -> dict[str, object]:
    selected_cases = CASES[:limit] if limit is not None else CASES
    report: dict[str, object] = {"settings": {"temperature": 0.05, "context_limit": 16384, "think": think}, "models": {}}
    for model in models:
        provider = OllamaProvider(model=model, temperature=0.05, timeout=600, retries=0, context_limit=16384, think=think)
        agents = RoleAgents(provider, structured_retries=1)
        case_results: list[dict[str, object]] = []
        for case in selected_cases:
            started = time.monotonic()
            try:
                with tempfile.TemporaryDirectory(prefix="autodev-model-eval-") as directory:
                    fixture = Path(directory)
                    for relative, content in case.files.items():
                        path = fixture / relative
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(content, encoding="utf-8")
                    state = ProjectState.create("Build the requested generic software capability.")
                    task = Task.create(case.name, case.instruction)
                    relevant = [f"{path} (current full content):\n{content}" for path, content in case.files.items()]
                    reply = agents.code(state, task, relevant)
                    result = evaluate_reply(fixture, case, reply.data)
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
