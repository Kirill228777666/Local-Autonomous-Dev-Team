from pathlib import Path

from autodev.dashboard import DashboardController, render_dashboard
from autodev.orchestrator import AutonomousRunner
from autodev.providers import ScriptedProvider
from autodev.state_store import StateStore
from autodev.tools import WorkspaceTools


def test_dashboard_renders_state_logs_final_qa_and_controls(tmp_path: Path) -> None:
    store = StateStore(tmp_path)
    state = __import__("autodev.models", fromlist=["ProjectState"]).ProjectState.create("Build dashboard")
    state.final_qa_status = "PASS"
    state.run_history.append("Tester: 12 passed")
    store.save(state)

    html = render_dashboard(store)

    assert "Build dashboard" in html
    assert "Final QA" in html
    assert "START" in html
    assert "Tester: 12 passed" in html


def test_dashboard_controller_applies_pause_control(tmp_path: Path) -> None:
    store = StateStore(tmp_path)
    runner = AutonomousRunner(tmp_path, store, WorkspaceTools(tmp_path), ScriptedProvider({}))
    runner.initialize("Build dashboard")

    DashboardController(runner).apply("pause")

    assert store.load().status == "PAUSED"  # type: ignore[union-attr]
