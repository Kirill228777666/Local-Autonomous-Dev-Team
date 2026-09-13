"""The product-proof run: a normal TODO request advances without new prompts."""

import subprocess
from pathlib import Path

from autodev.orchestrator import AutonomousRunner
from autodev.providers import AgentReply, ScriptedProvider
from autodev.state_store import StateStore
from autodev.tools import WorkspaceTools


TODO_SPEC = """Create a local TODO web application.

Features:
- create tasks;
- mark tasks complete;
- delete tasks;
- data survives restart;
- backend and frontend run locally.
"""


def git(path: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=path, check=True, capture_output=True, text=True)


def test_todo_spec_advances_multiple_autonomous_stages_and_persists_progress(tmp_path: Path) -> None:
    git(tmp_path, "init")
    git(tmp_path, "config", "user.email", "test@example.com")
    git(tmp_path, "config", "user.name", "Test User")
    (tmp_path / ".gitignore").write_text(".autodev/\n", encoding="utf-8")
    git(tmp_path, "add", ".gitignore")
    git(tmp_path, "commit", "-m", "initial")
    provider = ScriptedProvider(
        {
            "MANAGER": [
                AgentReply({"tasks": [
                    {"title": "Create persistence", "description": "Write TODO data file"},
                    {"title": "Create local app", "description": "Write local starter"},
                ]}),
                AgentReply({}), AgentReply({}), AgentReply({}),
            ],
            "CODER": [
                AgentReply({"actions": [{"kind": "write_file", "path": "todos.json", "content": "not-json"}]}),
                AgentReply({"actions": [{"kind": "write_file", "path": "todos.json", "content": "[]"}]}),
                AgentReply({"actions": [{"kind": "write_file", "path": "app.py", "content": "print('TODO app starts')\n"}]}),
            ],
            "TESTER": [
                AgentReply({"command": ["py", "-3", "-c", "import json; json.load(open('todos.json'))"]}),
                AgentReply({"command": ["py", "-3", "-c", "import json; assert json.load(open('todos.json')) == []"]}),
                AgentReply({"command": ["py", "-3", "app.py"]}),
            ],
            "REVIEWER": [AgentReply({"approved": True}), AgentReply({"approved": True})],
            "FINAL_QA": [AgentReply({"status": "PASS", "findings": []})],
        }
    )
    runner = AutonomousRunner(tmp_path, StateStore(tmp_path), WorkspaceTools(tmp_path), provider)
    runner.initialize(TODO_SPEC)

    state = runner.run(max_cycles=8)

    assert state.status == "COMPLETE"
    assert len(state.tasks) == 2
    assert state.tasks[0].attempts == 1
    assert StateStore(tmp_path).load().original_spec == TODO_SPEC.strip()  # type: ignore[union-attr]
    assert len([entry for entry in state.run_history if "Coder completed" in entry]) == 2
    assert state.last_checkpoint is not None
