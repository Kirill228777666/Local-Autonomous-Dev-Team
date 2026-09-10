from pathlib import Path

import pytest

from autodev.models import ProjectState, Task, TaskStatus
from autodev.state_store import StateStore


def test_state_store_round_trips_original_spec_and_running_task(tmp_path: Path) -> None:
    workspace = tmp_path / "todo-project"
    state = ProjectState.create("Build a persistent TODO application")
    task = Task.create("Implement storage", "Persist TODO items in JSON")
    task.status = TaskStatus.RUNNING
    state.tasks.append(task)

    StateStore(workspace).save(state)

    restored = StateStore(workspace).load()

    assert restored.original_spec == "Build a persistent TODO application"
    assert restored.tasks[0].title == "Implement storage"
    assert restored.tasks[0].status is TaskStatus.RUNNING


def test_state_store_returns_none_before_project_is_initialized(tmp_path: Path) -> None:
    assert StateStore(tmp_path).load() is None


def test_state_store_rejects_workspace_outside_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="workspace"):
        StateStore(tmp_path.parent / "outside", allowed_root=tmp_path)
