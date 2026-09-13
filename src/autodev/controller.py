"""Deterministic task lifecycle decisions for the autonomous runner."""

from __future__ import annotations

import hashlib
import json
import re
from enum import StrEnum

from .models import ProjectState, Task, TaskStatus


class TaskPhase(StrEnum):
    READY = "READY"
    CODING = "CODING"
    LOCAL_RECOVERY = "LOCAL_RECOVERY"
    ENVIRONMENT_REPAIR = "ENVIRONMENT_REPAIR"
    VALIDATING = "VALIDATING"
    REGRESSION_CHECK = "REGRESSION_CHECK"
    REVIEW = "REVIEW"
    CHECKPOINT = "CHECKPOINT"
    STRATEGY_CHANGE = "STRATEGY_CHANGE"
    DONE = "DONE"
    BLOCKED = "BLOCKED"


class FailureDecision(StrEnum):
    LOCAL_RETRY = "LOCAL_RETRY"
    STRATEGY_CHANGE = "STRATEGY_CHANGE"
    BLOCK = "BLOCK"


class TransitionError(ValueError):
    """A caller attempted an illegal task lifecycle transition."""


_LEGAL_TRANSITIONS = {
    TaskPhase.READY: {TaskPhase.CODING, TaskPhase.BLOCKED},
    TaskPhase.CODING: {TaskPhase.LOCAL_RECOVERY, TaskPhase.ENVIRONMENT_REPAIR, TaskPhase.VALIDATING, TaskPhase.BLOCKED},
    TaskPhase.LOCAL_RECOVERY: {TaskPhase.CODING, TaskPhase.VALIDATING, TaskPhase.STRATEGY_CHANGE, TaskPhase.BLOCKED},
    TaskPhase.ENVIRONMENT_REPAIR: {TaskPhase.VALIDATING, TaskPhase.CODING, TaskPhase.STRATEGY_CHANGE, TaskPhase.BLOCKED},
    TaskPhase.VALIDATING: {TaskPhase.CODING, TaskPhase.ENVIRONMENT_REPAIR, TaskPhase.REGRESSION_CHECK, TaskPhase.STRATEGY_CHANGE, TaskPhase.BLOCKED},
    TaskPhase.REGRESSION_CHECK: {TaskPhase.REVIEW, TaskPhase.STRATEGY_CHANGE, TaskPhase.BLOCKED},
    TaskPhase.REVIEW: {TaskPhase.CHECKPOINT, TaskPhase.STRATEGY_CHANGE, TaskPhase.BLOCKED},
    TaskPhase.CHECKPOINT: {TaskPhase.DONE, TaskPhase.BLOCKED},
    TaskPhase.STRATEGY_CHANGE: {TaskPhase.CODING, TaskPhase.BLOCKED},
    TaskPhase.DONE: set(),
    TaskPhase.BLOCKED: set(),
}

_STATUS_BY_PHASE = {
    TaskPhase.READY: TaskStatus.PENDING,
    TaskPhase.CODING: TaskStatus.RUNNING,
    TaskPhase.LOCAL_RECOVERY: TaskStatus.RUNNING,
    TaskPhase.ENVIRONMENT_REPAIR: TaskStatus.TESTING,
    TaskPhase.VALIDATING: TaskStatus.TESTING,
    TaskPhase.REGRESSION_CHECK: TaskStatus.TESTING,
    TaskPhase.REVIEW: TaskStatus.REVIEW,
    TaskPhase.CHECKPOINT: TaskStatus.REVIEW,
    TaskPhase.DONE: TaskStatus.DONE,
    TaskPhase.BLOCKED: TaskStatus.BLOCKED,
    TaskPhase.STRATEGY_CHANGE: TaskStatus.PENDING,
}


class TaskController:
    """Own legal transitions and bounded semantic retry decisions."""

    def __init__(self, local_failures_per_strategy: int = 2, max_strategy_changes: int = 1) -> None:
        self.local_failures_per_strategy = max(1, local_failures_per_strategy)
        self.max_strategy_changes = max(0, max_strategy_changes)

    @staticmethod
    def next_ready(state: ProjectState) -> Task | None:
        done_ids = {task.id for task in state.tasks if task.status is TaskStatus.DONE}
        return next(
            (
                task for task in state.tasks
                if task.status is TaskStatus.PENDING
                and all(dependency in done_ids for dependency in task.dependencies)
            ),
            None,
        )

    @staticmethod
    def transition(task: Task, target: TaskPhase) -> None:
        current = TaskPhase(task.phase)
        if target is current:
            return
        if target not in _LEGAL_TRANSITIONS[current]:
            raise TransitionError(f"illegal task transition {current.value} -> {target.value}")
        task.phase = target.value
        task.status = _STATUS_BY_PHASE[target]

    @staticmethod
    def failure_fingerprint(error: str) -> str:
        normalized = re.sub(r"0x[0-9a-fA-F]+|\b\d{4,}\b", "#", error.strip().lower())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    @staticmethod
    def workspace_fingerprint(files: dict[str, str]) -> str:
        return hashlib.sha256(json.dumps(files, sort_keys=True).encode("utf-8")).hexdigest()

    def semantic_key(self, task: Task, files: dict[str, str]) -> str:
        value = {
            "root": task.root_task_id or task.id,
            "workspace": self.workspace_fingerprint(files),
            "failure": task.failure_fingerprint,
            "strategy": task.strategy_generation,
        }
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()

    def register_semantic_call(self, task: Task, files: dict[str, str]) -> bool:
        key = self.semantic_key(task, files)
        if key in task.semantic_call_keys:
            return False
        task.semantic_call_keys.append(key)
        return True

    def decide_failure(self, task: Task, error: str) -> FailureDecision:
        task.failure_fingerprint = self.failure_fingerprint(error)
        task.failures_in_strategy += 1
        if task.strategy_generation >= self.max_strategy_changes:
            return FailureDecision.BLOCK
        if task.failures_in_strategy < self.local_failures_per_strategy:
            return FailureDecision.LOCAL_RETRY
        return FailureDecision.STRATEGY_CHANGE

    def apply_strategy(self, task: Task, strategy: str) -> None:
        task.strategy_generation += 1
        task.failures_in_strategy = 0
        task.strategy_history.append(strategy)
        task.phase = TaskPhase.READY.value
        task.status = TaskStatus.PENDING

    @staticmethod
    def block(task: Task, reason: str) -> None:
        task.phase = TaskPhase.BLOCKED.value
        task.status = TaskStatus.BLOCKED
        task.errors.append(f"BLOCKED_ROOT_TASK: {reason}")
