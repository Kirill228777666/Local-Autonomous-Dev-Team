from pathlib import Path

from autodev.models import Heartbeat, ProjectState, Task, TaskStatus
from autodev.orchestrator import AutonomousRunner
from autodev.providers import ScriptedProvider
from autodev.state_store import StateStore
from autodev.tools import WorkspaceTools
from autodev.watchdog import Watchdog


def test_state_persists_heartbeat_and_human_activity_log(tmp_path: Path) -> None:
    state = ProjectState.create("Build a resilient project")
    state.heartbeat = Heartbeat(agent="CODER", phase="TOOL", last_successful_action="wrote app.py")
    state.record_event("CODER", "TOOL", "Modified app.py")

    StateStore(tmp_path).save(state)

    restored = StateStore(tmp_path).load()
    activity = (tmp_path / ".autodev" / "activity.log").read_text(encoding="utf-8")
    assert restored.heartbeat.agent == "CODER"  # type: ignore[union-attr]
    assert "Coder" in activity
    assert "Modified app.py" in activity


def test_resume_reconciles_an_interrupted_active_task(tmp_path: Path) -> None:
    store = StateStore(tmp_path)
    state = ProjectState.create("Resume a project")
    task = Task.create("Write settings", "Write settings.json")
    task.status = TaskStatus.TESTING
    state.tasks.append(task)
    state.current_task_id = task.id
    state.status = "RUNNING"
    store.save(state)
    runner = AutonomousRunner(tmp_path, store, WorkspaceTools(tmp_path), ScriptedProvider({}))

    recovered = runner.resume()

    assert recovered.status == "READY"
    assert recovered.current_task_id is None
    assert recovered.tasks[0].status is TaskStatus.PENDING
    assert any("Recovered interrupted task" in entry for entry in recovered.run_history)


def test_watchdog_flags_stale_heartbeat_but_allows_recent_work() -> None:
    watchdog = Watchdog(stale_after_seconds=60)

    assert watchdog.is_stale(Heartbeat(timestamp="2020-01-01T00:00:00+00:00")) is True
    assert watchdog.is_stale(Heartbeat()) is False
