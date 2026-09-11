"""Reliability contracts added for the v0.3 endurance milestone."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from autodev.models import ProjectState, Task, TaskStatus, ToolExecution, ToolExecutionStatus
from autodev.metrics import metrics, write_metrics
from autodev.agents import RoleAgents
from autodev.orchestrator import AutonomousRunner
from autodev.providers import AgentReply, ScriptedProvider
from autodev.runtime import ApplicationScreenshotPipeline, ManagedProcess, ScreenshotPipeline
from autodev.state_store import StateStore
from autodev.tools import WorkspaceTools


def repository(path: Path) -> None:
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=path, check=True)
    (path / ".gitignore").write_text(".autodev/\n", encoding="utf-8")
    subprocess.run(["git", "add", ".gitignore"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=path, check=True, capture_output=True, text=True)


def test_resume_marks_interrupted_tool_unknown_and_never_reopens_done_task(tmp_path: Path) -> None:
    repository(tmp_path)
    done = Task.create("Finished", "Already checkpointed")
    done.status = TaskStatus.DONE
    active = Task.create("Interrupted", "Run a command")
    active.status = TaskStatus.RUNNING
    state = ProjectState.create("Build resilient app")
    state.tasks = [done, active]
    state.status = "RUNNING"
    state.current_task_id = active.id
    state.tool_executions = [
        ToolExecution.create(active.id, "run_command", ["py", "-3", "-c", "print('work')"])
    ]
    StateStore(tmp_path).save(state)

    recovered = AutonomousRunner(tmp_path, StateStore(tmp_path), WorkspaceTools(tmp_path), ScriptedProvider({})).resume()

    assert recovered.status == "READY"
    assert recovered.tasks[0].status is TaskStatus.DONE
    assert recovered.tasks[1].status is TaskStatus.PENDING
    assert recovered.tool_executions[0].status is ToolExecutionStatus.UNKNOWN
    assert any("interrupted tool" in entry.lower() for entry in recovered.run_history)


def test_resume_blocks_corrupt_dependency_graph_without_starting_work(tmp_path: Path) -> None:
    repository(tmp_path)
    task = Task.create("Broken", "Depends on missing task", dependencies=["missing"])
    state = ProjectState.create("Build app")
    state.tasks = [task]
    state.status = "RUNNING"
    StateStore(tmp_path).save(state)

    recovered = AutonomousRunner(tmp_path, StateStore(tmp_path), WorkspaceTools(tmp_path), ScriptedProvider({})).resume()

    assert recovered.status == "BLOCKED"
    assert "dependency" in recovered.run_history[-1].lower()


def test_resume_converts_legacy_blocked_task_into_one_bounded_corrective_task(tmp_path: Path) -> None:
    repository(tmp_path)
    task = Task.create("Backend", "Implement backend")
    task.status = TaskStatus.BLOCKED
    task.errors = ["dependency unavailable"]
    state = ProjectState.create("Build Notes")
    state.tasks = [task]
    state.status = "RUNNING"
    StateStore(tmp_path).save(state)

    recovered = AutonomousRunner(tmp_path, StateStore(tmp_path), WorkspaceTools(tmp_path), ScriptedProvider({})).resume()

    assert recovered.tasks[0].status is TaskStatus.FAILED
    assert recovered.tasks[1].repair_of == task.id
    assert recovered.tasks[1].status is TaskStatus.PENDING


def test_new_process_recovers_state_written_before_controlled_crash(tmp_path: Path) -> None:
    repository(tmp_path)
    script = """
from pathlib import Path
from autodev.models import ProjectState, Task, TaskStatus, ToolExecution
from autodev.state_store import StateStore
workspace = Path(__import__('sys').argv[1])
state = ProjectState.create('Crash recovery project')
task = Task.create('Write app', 'Write an application file')
task.status = TaskStatus.RUNNING
state.tasks = [task]
state.status = 'RUNNING'
state.current_task_id = task.id
state.tool_executions = [ToolExecution.create(task.id, 'run_command', ['py', '-3', '-c', 'print(1)'])]
StateStore(workspace).save(state)
__import__('time').sleep(60)
"""
    child = subprocess.Popen([sys.executable, "-c", script, str(tmp_path)])
    try:
        deadline = time.monotonic() + 10
        while not (tmp_path / ".autodev" / "state.json").exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert (tmp_path / ".autodev" / "state.json").exists()
    finally:
        child.kill()
        child.wait(timeout=10)

    recovered = AutonomousRunner(tmp_path, StateStore(tmp_path), WorkspaceTools(tmp_path), ScriptedProvider({})).resume()

    assert recovered.tasks[0].status is TaskStatus.PENDING
    assert recovered.tool_executions[0].status is ToolExecutionStatus.UNKNOWN
    assert any("crash recovery" in entry.lower() for entry in recovered.run_history)


def test_managed_process_waits_for_http_readiness_and_cleans_up(tmp_path: Path) -> None:
    process = ManagedProcess(["py", "-3", "-m", "http.server", "0"], timeout=3)
    # A dynamic port must be selected before starting a real command.
    from autodev.runtime import free_port

    process = ManagedProcess(["py", "-3", "-m", "http.server", str(free_port())], timeout=5)
    port = int(process.command[-1])
    process.start()
    try:
        assert process.wait_ready("127.0.0.1", port) is True
    finally:
        process.stop()
    assert process.process is not None and process.process.poll() is not None


def test_screenshot_pipeline_writes_desktop_and_narrow_artifacts(tmp_path: Path) -> None:
    captured: list[tuple[int, int]] = []

    def capture(url: str, output: Path, width: int, height: int) -> None:
        assert url == "http://127.0.0.1:1234"
        captured.append((width, height))
        output.write_bytes(b"fake-png")

    artifacts = ScreenshotPipeline(tmp_path, capture=capture).capture("http://127.0.0.1:1234")

    assert [artifact.name for artifact in artifacts] == ["desktop.png", "narrow.png"]
    assert captured == [(1440, 1000), (390, 844)]
    assert all(artifact.is_file() for artifact in artifacts)


def test_application_screenshot_pipeline_uses_one_dynamic_port_and_stops_server(tmp_path: Path) -> None:
    calls: list[str] = []

    def capture(url: str, output: Path, _width: int, _height: int) -> None:
        calls.append(url)
        output.write_bytes(b"png")

    pipeline = ApplicationScreenshotPipeline(
        tmp_path,
        ("py", "-3", "-m", "http.server", "{port}"),
        health_url="http://127.0.0.1:{port}",
        ready_timeout=5,
        capture=capture,
    )
    artifacts = pipeline.capture("http://127.0.0.1:{port}")

    assert len(artifacts) == 2
    assert len(set(calls)) == 1
    assert "{port}" not in calls[0]


def test_visual_failure_returns_task_for_bounded_repair(tmp_path: Path) -> None:
    repository(tmp_path)
    (tmp_path / "index.html").write_text("<main>Notes</main>", encoding="utf-8")

    def capture(_url: str, output: Path, _width: int, _height: int) -> None:
        output.write_bytes(b"fake-png")

    provider = ScriptedProvider(
        {
            "MANAGER": [AgentReply({"tasks": [{"title": "Build UI", "description": "Build responsive UI"}]}), AgentReply({})],
            "CODER": [AgentReply({"actions": []})],
            "TESTER": [AgentReply({"command": ["py", "-3", "-c", "print('ok')"]})],
            "DESIGNER": [
                AgentReply({"verdict": "FAIL", "issues": [{"severity": "high", "category": "overflow", "description": "Form overflows narrow viewport."}]}),
                AgentReply({"verdict": "PASS", "issues": []}),
            ],
        }
    )
    runner = AutonomousRunner(
        tmp_path,
        StateStore(tmp_path),
        WorkspaceTools(tmp_path),
        provider,
        visual_pipeline=ScreenshotPipeline(tmp_path, capture=capture),
        visual_url="http://127.0.0.1:1234",
        max_visual_repairs=2,
    )
    runner.initialize("Build a responsive Notes UI")

    state = runner.run(max_cycles=1)

    assert state.tasks[0].status is TaskStatus.PENDING
    assert state.visual_status == "FAIL"
    assert state.visual_repair_cycles == 1
    assert "overflow" in state.tasks[0].errors[-1]


def test_metrics_and_report_are_event_based_and_include_recovery_visual_data(tmp_path: Path) -> None:
    state = ProjectState.create("Build durable Notes")
    state.record_event("CODER", "LLM_CALL", "Coder request")
    state.record_event("TESTER", "LLM_CALL", "Tester request")
    state.record_event("CODER", "TOOL_STARTED", "Started command")
    state.record_event("DESIGNER", "LLM_CALL", "Designer request")
    state.run_history.extend(["Crash recovery recorded 1 interrupted tool execution(s)", "Run resumed", "Returning task to Coder: repair", "Review rejected: missing test"])
    state.visual_status = "UNAVAILABLE"
    state.visual_issues = ["high:overflow: form overlaps"]

    data = metrics(state)
    write_metrics(tmp_path, state)

    assert data["total_llm_calls"] == 3
    assert data["llm_calls_by_role"] == {"CODER": 1, "DESIGNER": 1, "TESTER": 1}
    assert data["total_tool_calls"] == 1
    assert data["crash_events"] == 1
    assert data["resume_events"] == 1
    report = (tmp_path / "run_report.md").read_text(encoding="utf-8")
    assert "Visual QA" in report
    assert "Known limitations" in report


def test_role_agents_repair_malformed_structured_reply_before_returning_it() -> None:
    state = ProjectState.create("Build app")
    task = Task.create("Write app", "Create app.py")
    provider = ScriptedProvider(
        {"CODER": [AgentReply({"actions": [{"kind": "teleport"}]}), AgentReply({"actions": [{"kind": "write_file", "path": "app.py", "content": "ok"}]})]}
    )

    reply = RoleAgents(provider, structured_retries=1).code(state, task)

    assert reply.data["actions"][0]["kind"] == "write_file"  # type: ignore[index]


def test_exhausted_task_creates_one_architect_guided_corrective_task(tmp_path: Path) -> None:
    repository(tmp_path)
    task = Task.create("Backend", "Implement backend")
    task.attempts = 3
    state = ProjectState.create("Build Notes")
    state.tasks = [task]
    runner = AutonomousRunner(
        tmp_path,
        StateStore(tmp_path),
        WorkspaceTools(tmp_path),
        ScriptedProvider({"ARCHITECT": [AgentReply({"diagnosis": "Avoid unavailable dependencies and use stdlib SQLite."})]}),
    )

    runner._retry_or_block(task, state, "Coder error: unavailable dependency")

    assert task.status is TaskStatus.FAILED
    corrective = state.tasks[-1]
    assert corrective.repair_of == task.id
    assert corrective.status is TaskStatus.PENDING
    assert "stdlib SQLite" in corrective.description


def test_missing_tool_executable_is_retried_instead_of_crashing_orchestrator(tmp_path: Path) -> None:
    repository(tmp_path)
    provider = ScriptedProvider(
        {
            "MANAGER": [AgentReply({"tasks": [{"title": "Tool", "description": "Run tool"}]}), AgentReply({})],
            "CODER": [AgentReply({"actions": [{"kind": "run_command", "command": ["definitely-missing-tool"]}]})],
        }
    )
    runner = AutonomousRunner(tmp_path, StateStore(tmp_path), WorkspaceTools(tmp_path), provider)
    runner.initialize("Run tool")

    state = runner.run(max_cycles=1)

    assert state.status == "RUNNING"
    assert state.tasks[0].status is TaskStatus.PENDING
    assert "Coder error" in state.tasks[0].errors[-1]


def test_stale_edit_is_requeried_at_tool_level_without_consuming_task_attempt(tmp_path: Path) -> None:
    repository(tmp_path)
    (tmp_path / "app.txt").write_text("current", encoding="utf-8")
    provider = ScriptedProvider({
        "MANAGER": [AgentReply({"tasks": [{"title": "Edit", "description": "Edit file"}]}), AgentReply({})],
        "CODER": [AgentReply({"actions": [{"kind": "edit_file", "path": "app.txt", "old": "stale", "new": "x"}]}), AgentReply({"actions": [{"kind": "write_file", "path": "app.txt", "content": "fixed"}]})],
        "TESTER": [AgentReply({"command": ["py", "-3", "-c", "assert open('app.txt').read() == 'fixed'"]})],
        "REVIEWER": [AgentReply({"approved": True})], "FINAL_QA": [AgentReply({"status": "PASS", "findings": []})],
    })
    runner = AutonomousRunner(tmp_path, StateStore(tmp_path), WorkspaceTools(tmp_path), provider)
    runner.initialize("Edit")
    state = runner.run(max_cycles=2)
    assert state.tasks[0].attempts == 1
    assert any(event.phase == "TOOL_RECOVERY" for event in state.events)
