from pathlib import Path

import pytest
import json

import autodev.state_store as state_store_module
from autodev.models import ProjectState, Task, TaskStatus
from autodev.state_store import StateStore


def test_state_store_round_trips_original_spec_and_running_task(tmp_path: Path) -> None:
    workspace = tmp_path / "todo-project"
    state = ProjectState.create("Build a persistent TODO application")
    task = Task.create("Implement storage", "Persist TODO items in JSON")
    task.status = TaskStatus.RUNNING
    state.tasks.append(task)
    state.model = "qwen3:14b"

    StateStore(workspace).save(state)

    restored = StateStore(workspace).load()

    assert restored.original_spec == "Build a persistent TODO application"
    assert restored.tasks[0].title == "Implement storage"
    assert restored.tasks[0].status is TaskStatus.RUNNING
    assert restored.model == "qwen3:14b"
    assert (workspace / ".autodev" / "project_spec.md").read_text(encoding="utf-8").endswith(
        "Build a persistent TODO application\n"
    )
    assert "Implement storage" in (workspace / ".autodev" / "progress.md").read_text(encoding="utf-8")


def test_state_store_returns_none_before_project_is_initialized(tmp_path: Path) -> None:
    assert StateStore(tmp_path).load() is None


def test_state_store_can_save_the_same_state_repeatedly(tmp_path: Path) -> None:
    store = StateStore(tmp_path)
    state = ProjectState.create("Build a durable run")

    store.save(state)
    state.status = "RUNNING"
    store.save(state)

    assert store.load().status == "RUNNING"  # type: ignore[union-attr]


def test_state_store_retries_a_transient_windows_replace_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = StateStore(tmp_path)
    original_replace = state_store_module.os.replace
    calls = 0

    def flaky_replace(source: str | Path, destination: str | Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise PermissionError("temporarily locked")
        original_replace(source, destination)

    monkeypatch.setattr(state_store_module.os, "replace", flaky_replace)

    store.save(ProjectState.create("Build a resilient run"))

    assert calls == 2


def test_state_store_projects_summary_and_decisions_memory(tmp_path: Path) -> None:
    state = ProjectState.create("Build a local notes application")
    state.model = "qwen3:14b"
    state.decisions.append("Use JSON persistence for MVP")
    state.tasks.append(Task.create("Storage", "Persist notes"))

    StateStore(tmp_path).save(state)

    summary = (tmp_path / ".autodev" / "project_summary.md").read_text(encoding="utf-8")
    decisions = (tmp_path / ".autodev" / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
    assert "Build a local notes application" in summary
    assert "Use JSON persistence for MVP" in summary
    assert json.loads(decisions[0])["decision"] == "Use JSON persistence for MVP"


def test_state_store_rejects_workspace_outside_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="workspace"):
        StateStore(tmp_path.parent / "outside", allowed_root=tmp_path)
