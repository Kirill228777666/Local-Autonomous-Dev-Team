from pathlib import Path

import pytest

from autodev.controller import FailureDecision, TaskController, TaskPhase, TransitionError
from autodev.models import ProjectState, Task, TaskStatus


def test_controller_selects_first_dependency_ready_task_without_manager() -> None:
    blocked_dependency = Task.create("Blocked by dependency", "wait", dependencies=["missing"])
    ready = Task.create("Ready", "work")
    state = ProjectState.create("Build app")
    state.tasks = [blocked_dependency, ready]

    assert TaskController().next_ready(state) is ready


def test_controller_rejects_illegal_phase_transition() -> None:
    task = Task.create("Feature", "Implement it")

    with pytest.raises(TransitionError, match="READY.*REVIEW"):
        TaskController().transition(task, TaskPhase.REVIEW)


def test_controller_persists_legal_phase_and_legacy_task_defaults() -> None:
    task = Task.create("Feature", "Implement it")
    TaskController().transition(task, TaskPhase.CODING)

    restored = Task.from_dict({"id": "legacy", "title": "Old", "description": "state"})

    assert task.phase == TaskPhase.CODING.value
    assert task.status is TaskStatus.RUNNING
    assert restored.phase == TaskPhase.READY.value
    assert restored.semantic_call_keys == []


def test_loop_breaker_rejects_same_root_workspace_failure_and_strategy() -> None:
    task = Task.create("Feature", "Implement it")
    task.failure_fingerprint = "failure-a"
    controller = TaskController()
    workspace = {"app.py": "hash-a"}

    assert controller.register_semantic_call(task, workspace) is True
    assert controller.register_semantic_call(task, workspace) is False

    task.strategy_generation += 1
    assert controller.register_semantic_call(task, workspace) is True


def test_failure_policy_is_local_retry_then_one_strategy_change_then_block() -> None:
    task = Task.create("Feature", "Implement it")
    controller = TaskController(local_failures_per_strategy=2, max_strategy_changes=1)

    assert controller.decide_failure(task, "assertion 1") is FailureDecision.LOCAL_RETRY
    assert controller.decide_failure(task, "assertion 1") is FailureDecision.STRATEGY_CHANGE
    controller.apply_strategy(task, "Use a different API")
    assert task.strategy_generation == 1
    assert controller.decide_failure(task, "assertion 2") is FailureDecision.BLOCK


def test_controller_blocks_root_task_without_creating_repair_children() -> None:
    task = Task.create("Impossible", "Cannot be done")
    state = ProjectState.create("Build app")
    state.tasks = [task]

    TaskController().block(task, "No supported runtime")

    assert state.tasks == [task]
    assert task.status is TaskStatus.BLOCKED
    assert task.phase == TaskPhase.BLOCKED.value
    assert task.errors[-1] == "BLOCKED_ROOT_TASK: No supported runtime"
