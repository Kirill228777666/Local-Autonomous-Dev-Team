"""Durable, sequential Manager → Coder → Tester → Reviewer orchestration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

from .agents import RoleAgents
from .git import GitError, GitRepository
from .models import ProjectState, Task, TaskStatus
from .providers import LLMProvider, ProviderError
from .state_store import StateStore
from .tools import CommandResult, ToolPolicyError, WorkspaceTools


class AutonomousRunner:
    def __init__(
        self,
        workspace: Path,
        store: StateStore,
        tools: WorkspaceTools,
        provider: LLMProvider,
        max_attempts: int = 3,
    ) -> None:
        self.workspace = workspace.resolve()
        self.store = store
        self.tools = tools
        self.agents = RoleAgents(provider)
        self.max_attempts = max_attempts

    def initialize(self, original_spec: str) -> ProjectState:
        existing = self.store.load()
        if existing is not None:
            return existing
        state = ProjectState.create(original_spec)
        state.run_history.append("Project initialized")
        self.store.save(state)
        return state

    def run(self, max_cycles: int = 100) -> ProjectState:
        state = self.store.load()
        if state is None:
            raise ValueError("project is not initialized")
        if state.status in {"PAUSED", "STOPPED", "COMPLETE"}:
            return state
        state.status = "RUNNING"
        if not state.tasks:
            self._plan(state)
        if state.status == "BLOCKED":
            return state
        for _ in range(max_cycles):
            if state.status in {"PAUSED", "STOPPED"}:
                break
            task = self._select_next_task(state)
            if task is None:
                state.status = "COMPLETE" if all(task.status is TaskStatus.DONE for task in state.tasks) else "NEEDS_ATTENTION"
                self.store.save(state)
                break
            self._run_task(state, task)
            self.store.save(state)
        return state

    def pause(self) -> ProjectState:
        return self._set_control_status("PAUSED")

    def stop(self) -> ProjectState:
        return self._set_control_status("STOPPED")

    def resume(self) -> ProjectState:
        state = self._required_state()
        if state.status == "STOPPED":
            raise ValueError("stopped projects cannot resume; start a new run explicitly")
        state.status = "READY"
        state.run_history.append("Run resumed")
        self.store.save(state)
        return state

    def _set_control_status(self, status: str) -> ProjectState:
        state = self._required_state()
        state.status = status
        state.run_history.append(f"Run {status.lower()} by user")
        self.store.save(state)
        return state

    def _required_state(self) -> ProjectState:
        state = self.store.load()
        if state is None:
            raise ValueError("project is not initialized")
        return state

    def _plan(self, state: ProjectState) -> None:
        try:
            reply = self.agents.plan(state).data
            raw_tasks = reply.get("tasks")
            if not isinstance(raw_tasks, list) or not raw_tasks:
                raise ProviderError("Manager returned no initial tasks")
            for raw_task in raw_tasks:
                if not isinstance(raw_task, dict):
                    raise ProviderError("Manager task must be an object")
                title = raw_task.get("title")
                description = raw_task.get("description")
                if not isinstance(title, str) or not isinstance(description, str):
                    raise ProviderError("Manager task needs title and description")
                state.tasks.append(Task.create(title, description))
            state.decisions.append(f"Manager planned {len(state.tasks)} tasks")
            state.run_history.append("Manager created initial plan")
            self.store.save(state)
        except ProviderError as error:
            state.status = "BLOCKED"
            state.run_history.append(f"Planning blocked: {error}")
            self.store.save(state)

    def _select_next_task(self, state: ProjectState) -> Task | None:
        pending = [task for task in state.tasks if task.status is TaskStatus.PENDING]
        if not pending:
            return None
        try:
            requested_id = self.agents.select(state).data.get("next_task_id")
            if isinstance(requested_id, str):
                selected = next((task for task in pending if task.id == requested_id), None)
                if selected is not None:
                    return selected
        except ProviderError as error:
            state.run_history.append(f"Manager selection unavailable; using plan order: {error}")
        return pending[0]

    def _run_task(self, state: ProjectState, task: Task) -> None:
        task.status = TaskStatus.RUNNING
        state.current_task_id = task.id
        task.attempts += 1
        try:
            actions = self.agents.code(state, task).data.get("actions")
            if not isinstance(actions, list):
                raise ProviderError("Coder response needs an actions array")
            fingerprint = hashlib.sha256(json.dumps(actions, sort_keys=True).encode()).hexdigest()
            if fingerprint in task.action_fingerprints:
                self._block(task, state, "repeated identical coder action without progress")
                return
            task.action_fingerprints.append(fingerprint)
            for action in actions:
                self._execute_action(action)
            state.run_history.append(f"Coder completed attempt {task.attempts} for {task.title}")
        except (ProviderError, ToolPolicyError, ValueError, KeyError, TypeError) as error:
            self._retry_or_block(task, state, f"Coder error: {error}")
            return

        task.status = TaskStatus.TESTING
        try:
            reply = self.agents.test(state, task).data
            command = reply.get("command")
            if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
                raise ProviderError("Tester response needs a string command array")
            result = self.tools.run_tests(command)
        except (ProviderError, ToolPolicyError, ValueError) as error:
            self._retry_or_block(task, state, f"Tester error: {error}")
            return
        state.run_history.append(self._result_log(task, result))
        if result.exit_code != 0:
            self._retry_or_block(task, state, f"Test failed (exit {result.exit_code}): {result.stderr or result.stdout}")
            return

        task.status = TaskStatus.REVIEW
        try:
            evidence = self._result_log(task, result) + "\nGit diff:\n" + self._git_diff()
            review = self.agents.review(state, task, evidence).data
            if review.get("approved") is not True:
                reasons = review.get("reasons", ["Reviewer rejected implementation"])
                self._retry_or_block(task, state, f"Review rejected: {reasons}")
                return
        except ProviderError as error:
            self._retry_or_block(task, state, f"Reviewer error: {error}")
            return

        try:
            state.last_checkpoint = GitRepository(self.workspace).checkpoint(f"autodev: {task.title}")
        except GitError as error:
            self._retry_or_block(task, state, f"Git checkpoint failed: {error}")
            return
        task.status = TaskStatus.DONE
        state.current_task_id = None
        state.run_history.append(f"Task approved and checkpointed: {task.title}")

    def _execute_action(self, action: object) -> None:
        if not isinstance(action, dict):
            raise ValueError("Coder action must be an object")
        kind = action.get("kind")
        if kind == "write_file":
            self.tools.write_file(self._string(action, "path"), self._string(action, "content"))
        elif kind == "edit_file":
            self.tools.edit_file(self._string(action, "path"), self._string(action, "old"), self._string(action, "new"))
        elif kind == "delete_file":
            self.tools.delete_file(self._string(action, "path"))
        elif kind == "run_command":
            command = action.get("command")
            if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
                raise ValueError("run_command requires a string command array")
            result = self.tools.run_command(command)
            if result.exit_code:
                raise ValueError(f"command failed: {result.stderr or result.stdout}")
        else:
            raise ValueError(f"unsupported action kind: {kind}")

    @staticmethod
    def _string(action: dict[str, object], key: str) -> str:
        value = action.get(key)
        if not isinstance(value, str):
            raise ValueError(f"action needs string '{key}'")
        return value

    def _retry_or_block(self, task: Task, state: ProjectState, error: str) -> None:
        task.errors.append(error)
        if task.attempts >= self.max_attempts:
            self._block(task, state, error)
            return
        if len(task.errors) >= 2:
            self._diagnose(state, task)
        task.status = TaskStatus.PENDING
        state.run_history.append(f"Returning task to Coder: {task.title}; {error}")

    def _diagnose(self, state: ProjectState, task: Task) -> None:
        try:
            diagnosis = self.agents.diagnose(state, task).data
            state.decisions.append(f"Architect diagnosis for {task.title}: {diagnosis}")
        except ProviderError as error:
            state.decisions.append(f"Architect unavailable for {task.title}: {error}")

    @staticmethod
    def _block(task: Task, state: ProjectState, reason: str) -> None:
        task.status = TaskStatus.BLOCKED
        task.errors.append(reason)
        state.current_task_id = None
        state.run_history.append(f"Task blocked: {task.title}; {reason}")

    @staticmethod
    def _result_log(task: Task, result: CommandResult) -> str:
        return f"Tester for {task.title}: exit={result.exit_code}; stdout={result.stdout.strip()}; stderr={result.stderr.strip()}"

    def _git_diff(self) -> str:
        try:
            return GitRepository(self.workspace).diff()
        except GitError:
            return "Git diff unavailable"
