import subprocess
from pathlib import Path

from autodev.models import TaskStatus
from autodev.orchestrator import AutonomousRunner
from autodev.providers import AgentReply, ScriptedProvider
from autodev.state_store import StateStore
from autodev.tools import WorkspaceTools


def git(path: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=path, check=True, capture_output=True, text=True)


def setup_repository(path: Path) -> None:
    git(path, "init")
    git(path, "config", "user.email", "test@example.com")
    git(path, "config", "user.name", "Test User")
    (path / ".gitignore").write_text(".autodev/\n", encoding="utf-8")
    git(path, "add", ".gitignore")
    git(path, "commit", "-m", "initial")


def test_runner_completes_multiple_tasks_and_repairs_a_failed_test(tmp_path: Path) -> None:
    setup_repository(tmp_path)
    provider = ScriptedProvider(
        {
            "MANAGER": [
                AgentReply(
                    {
                        "tasks": [
                            {"title": "Write artifact", "description": "Create artifact.txt"},
                            {"title": "Write second artifact", "description": "Create second.txt"},
                        ]
                    }
                ),
                AgentReply({}),
                AgentReply({}),
            ],
            "CODER": [
                AgentReply({"actions": [{"kind": "write_file", "path": "artifact.txt", "content": "bad"}]}),
                AgentReply({"actions": [{"kind": "write_file", "path": "artifact.txt", "content": "good"}]}),
                AgentReply({"actions": [{"kind": "write_file", "path": "second.txt", "content": "done"}]}),
            ],
            "TESTER": [
                AgentReply(
                    {
                        "command": [
                            "py", "-3", "-c",
                            "from pathlib import Path; raise SystemExit(Path('artifact.txt').read_text() != 'good')",
                        ]
                    }
                ),
                AgentReply(
                    {
                        "command": [
                            "py", "-3", "-c",
                            "from pathlib import Path; raise SystemExit(Path('artifact.txt').read_text() != 'good')",
                        ]
                    }
                ),
                AgentReply({"command": ["py", "-3", "-c", "from pathlib import Path; assert Path('second.txt').read_text() == 'done'"]}),
            ],
            "REVIEWER": [AgentReply({"approved": True}), AgentReply({"approved": True})],
        }
    )
    runner = AutonomousRunner(
        workspace=tmp_path,
        store=StateStore(tmp_path),
        tools=WorkspaceTools(tmp_path),
        provider=provider,
    )
    runner.initialize("Build a small project")

    state = runner.run(max_cycles=8)

    assert state.status == "COMPLETE"
    assert [task.status for task in state.tasks] == [TaskStatus.DONE, TaskStatus.DONE]
    assert state.tasks[0].attempts == 2
    assert (tmp_path / "artifact.txt").read_text(encoding="utf-8") == "good"
    assert (tmp_path / "second.txt").read_text(encoding="utf-8") == "done"
    assert StateStore(tmp_path).load() is not None
    assert state.last_checkpoint is not None


def test_runner_blocks_repeated_identical_coder_actions(tmp_path: Path) -> None:
    provider = ScriptedProvider(
        {
            "MANAGER": [
                AgentReply({"tasks": [{"title": "Loop", "description": "Do something"}]}),
                AgentReply({}),
            ],
            "CODER": [
                AgentReply({"actions": [{"kind": "write_file", "path": "loop.txt", "content": "same"}]}),
                AgentReply({"actions": [{"kind": "write_file", "path": "loop.txt", "content": "same"}]}),
                AgentReply({"actions": [{"kind": "write_file", "path": "loop.txt", "content": "same"}]}),
            ],
            "TESTER": [
                AgentReply({"command": ["py", "-3", "-c", "raise SystemExit(1)"]}),
                AgentReply({"command": ["py", "-3", "-c", "raise SystemExit(1)"]}),
            ],
        }
    )
    runner = AutonomousRunner(tmp_path, StateStore(tmp_path), WorkspaceTools(tmp_path), provider)
    runner.initialize("Build a loop test")

    state = runner.run(max_cycles=4)

    assert state.tasks[0].status is TaskStatus.BLOCKED
    assert any("repeated" in error for error in state.tasks[0].errors)


def test_runner_keeps_an_invalid_initial_plan_blocked(tmp_path: Path) -> None:
    runner = AutonomousRunner(
        tmp_path,
        StateStore(tmp_path),
        WorkspaceTools(tmp_path),
        ScriptedProvider({"MANAGER": [AgentReply({"tasks": []})]}),
    )
    runner.initialize("Build a project")

    state = runner.run()

    assert state.status == "BLOCKED"
