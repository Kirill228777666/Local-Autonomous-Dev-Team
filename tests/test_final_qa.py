import subprocess
from pathlib import Path

from autodev.orchestrator import AutonomousRunner
from autodev.providers import AgentReply, ScriptedProvider
from autodev.state_store import StateStore
from autodev.tools import WorkspaceTools


def git(path: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=path, check=True, capture_output=True, text=True)


def repository(path: Path) -> None:
    git(path, "init")
    git(path, "config", "user.email", "test@example.com")
    git(path, "config", "user.name", "Test User")
    (path / ".gitignore").write_text(".autodev/\n", encoding="utf-8")
    git(path, "add", ".gitignore")
    git(path, "commit", "-m", "initial")


def test_final_qa_pass_is_required_before_runner_completes(tmp_path: Path) -> None:
    repository(tmp_path)
    provider = ScriptedProvider(
        {
            "MANAGER": [
                AgentReply({"tasks": [{"title": "Create output", "description": "write output"}]}),
                AgentReply({}),
            ],
            "CODER": [AgentReply({"actions": [{"kind": "write_file", "path": "output.txt", "content": "ok"}]})],
            "TESTER": [AgentReply({"command": ["py", "-3", "-c", "assert open('output.txt').read() == 'ok'"]})],
            "REVIEWER": [AgentReply({"approved": True})],
            "FINAL_QA": [AgentReply({"status": "PASS", "findings": []})],
        }
    )
    runner = AutonomousRunner(tmp_path, StateStore(tmp_path), WorkspaceTools(tmp_path), provider)
    runner.initialize("Create output")

    state = runner.run(max_cycles=2)

    assert state.status == "COMPLETE"
    assert state.final_qa_status == "PASS"
    assert state.heartbeat.phase == "COMPLETE"
    assert any("Final QA passed" in event for event in state.run_history)


def test_final_qa_fail_creates_corrective_task_instead_of_completing(tmp_path: Path) -> None:
    repository(tmp_path)
    provider = ScriptedProvider(
        {
            "MANAGER": [
                AgentReply({"tasks": [{"title": "Create output", "description": "write output"}]}),
                AgentReply({}),
            ],
            "CODER": [AgentReply({"actions": [{"kind": "write_file", "path": "output.txt", "content": "ok"}]})],
            "TESTER": [AgentReply({"command": ["py", "-3", "-c", "assert open('output.txt').read() == 'ok'"]})],
            "REVIEWER": [AgentReply({"approved": True})],
            "FINAL_QA": [AgentReply({"status": "FAIL", "findings": [{"title": "Missing README", "description": "Document startup."}]})],
        }
    )
    runner = AutonomousRunner(tmp_path, StateStore(tmp_path), WorkspaceTools(tmp_path), provider)
    runner.initialize("Create output and document startup")

    state = runner.run(max_cycles=2)

    assert state.status == "RUNNING"
    assert state.final_qa_status == "FAIL"
    assert state.tasks[-1].title == "Missing README"
    assert state.tasks[-1].status.value == "PENDING"
