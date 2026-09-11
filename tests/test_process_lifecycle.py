from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest

from autodev.runtime import ManagedProcessManager, free_port
from autodev.agents import RoleAgents
from autodev.models import TaskStatus
from autodev.orchestrator import AutonomousRunner
from autodev.providers import AgentReply, ScriptedProvider
from autodev.state_store import StateStore
from autodev.tools import WorkspaceTools


def _pid_is_alive(pid: int) -> bool:
    if os.name == "nt":
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            check=False,
        )
        return str(pid) in result.stdout
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_hard_timeout_kills_parent_and_infinite_child_within_wall_clock_bound(tmp_path: Path) -> None:
    child_pid = tmp_path / "child.pid"
    script = (
        "import pathlib, subprocess, sys, time; "
        "child=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
        f"pathlib.Path({str(child_pid)!r}).write_text(str(child.pid)); time.sleep(30)"
    )
    started = time.monotonic()
    result = WorkspaceTools(tmp_path, command_timeout=0.35).run_command(["py", "-3", "-c", script])
    elapsed = time.monotonic() - started

    assert result.exit_code == 124
    assert elapsed < 5
    assert child_pid.is_file()
    deadline = time.monotonic() + 2
    while _pid_is_alive(int(child_pid.read_text())) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _pid_is_alive(int(child_pid.read_text()))


def test_managed_process_records_logs_readiness_and_stops_process_tree(tmp_path: Path) -> None:
    manager = ManagedProcessManager(tmp_path)
    port = free_port()
    record = manager.start(
        ["py", "-3", "-m", "http.server", str(port)],
        purpose="test-server",
        expected_port=port,
    )
    try:
        assert manager.wait_ready(record.id, "127.0.0.1", port, timeout=5)
        assert (tmp_path / ".autodev" / "processes" / record.id / "stdout.log").is_file()
        assert record.status == "READY"
    finally:
        manager.stop(record.id)
    assert record.status == "STOPPED"
    assert not _pid_is_alive(record.pid)


def test_resume_record_never_kills_unverified_pid(tmp_path: Path) -> None:
    manager = ManagedProcessManager(tmp_path)
    current = os.getpid()
    report = manager.recover([{
        "id": "foreign", "pid": current, "command": ["not-the-current-process"],
        "cwd": str(tmp_path), "started_at": "2026-01-01T00:00:00+00:00",
        "purpose": "test", "expected_port": None, "status": "RUNNING", "log_dir": "",
        "ownership_token": "different-token",
    }])
    assert report == ["Skipped unverified managed process foreign"]
    assert _pid_is_alive(current)


def test_resume_cleans_owned_managed_process_tree(tmp_path: Path) -> None:
    first = ManagedProcessManager(tmp_path)
    port = free_port()
    record = first.start(["py", "-3", "-m", "http.server", str(port)], purpose="crash", expected_port=port)
    assert first.wait_ready(record.id, "127.0.0.1", port, timeout=5)

    report = ManagedProcessManager(tmp_path).recover(first.records())

    assert report == [f"Cleaned stale managed process {record.id}"]
    deadline = time.monotonic() + 2
    while _pid_is_alive(record.pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _pid_is_alive(record.pid)


def test_orchestrator_reroutes_flask_like_command_and_cleans_it_without_manual_kill(tmp_path: Path) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=tmp_path, check=True)
    (tmp_path / ".gitignore").write_text(".autodev/\n", encoding="utf-8")
    subprocess.run(["git", "add", ".gitignore"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=tmp_path, check=True, capture_output=True, text=True)
    script = """# Flask(
from http.server import BaseHTTPRequestHandler, HTTPServer
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b'ok')
HTTPServer(('127.0.0.1', 5000), Handler).serve_forever()
"""
    provider = ScriptedProvider({
        "MANAGER": [AgentReply({"tasks": [{"title": "Server", "description": "Launch API"}]}), AgentReply({})],
        "CODER": [AgentReply({"actions": [
            {"kind": "write_file", "path": "backend/app.py", "content": script},
            {"kind": "run_command", "command": ["py", "-3", "backend/app.py"]},
        ]})],
        "TESTER": [AgentReply({"command": ["py", "-3", "-c", "from urllib.request import urlopen; assert urlopen('http://127.0.0.1:5000', timeout=2).status == 200"]})],
        "REVIEWER": [AgentReply({"approved": True})],
    })
    runner = AutonomousRunner(tmp_path, StateStore(tmp_path), WorkspaceTools(tmp_path), provider)
    runner.initialize("Build a local API")

    state = runner.run(max_cycles=1)

    assert state.tasks[0].status is TaskStatus.DONE
    assert any(event.phase == "READINESS" for event in state.events)
    assert any(record["status"] == "STOPPED" for record in state.managed_processes)
