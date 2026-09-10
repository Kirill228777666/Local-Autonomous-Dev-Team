"""Durable, sequential Manager → Coder → Tester → Reviewer orchestration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

from .agents import RoleAgents
from .git import GitError, GitRepository
from .models import Heartbeat, ProjectState, Task, TaskStatus
from .providers import LLMProvider, ProviderError
from .regression import RegressionRunner
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
        regression_runner: RegressionRunner | None = None,
    ) -> None:
        self.workspace = workspace.resolve()
        self.store = store
        self.tools = tools
        self.agents = RoleAgents(provider)
        self.max_attempts = max_attempts
        self.regression_runner = regression_runner or RegressionRunner(self.workspace, tools)

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
        self._mark(state, "MANAGER", "PLANNING", "Autonomous run started")
        if not state.tasks:
            self._plan(state)
        if state.status == "BLOCKED":
            return state
        for _ in range(max_cycles):
            if state.status in {"PAUSED", "STOPPED"}:
                break
            task = self._select_next_task(state)
            if task is None:
                self._finish_or_continue(state)
                self.store.save(state)
                if state.status != "RUNNING":
                    break
                continue
            self._run_task(state, task)
            self.store.save(state)
        return state

    def _finish_or_continue(self, state: ProjectState) -> None:
        blocked = [task for task in state.tasks if task.status is TaskStatus.BLOCKED]
        if blocked:
            state.status = "BLOCKED"
            state.run_history.append("Completion blocked by unresolved tasks: " + ", ".join(task.title for task in blocked))
            self._mark(state, "SYSTEM", "BLOCKED", "Completion blocked by unresolved tasks")
            return
        self._mark(state, "FINAL_QA", "REGRESSION", "Running full regression")
        regression = self.regression_runner.run()
        state.regression_history.append(regression.summary)
        state.run_history.append(regression.summary)
        if not regression.passed:
            state.tasks.append(Task.create("Restore full regression", regression.summary))
            state.final_qa_status = "NOT_RUN"
            state.run_history.append("Regression failed; created corrective task")
            self._mark(state, "MANAGER", "CORRECTIVE", "Regression failed; corrective task created")
            return
        try:
            self._mark(state, "FINAL_QA", "REVIEW", "Final QA evaluating completed project")
            reply = self.agents.final_qa(state, regression.summary + "\nGit diff:\n" + self._git_diff()).data
        except ProviderError as error:
            state.status = "BLOCKED"
            state.run_history.append(f"Final QA unavailable: {error}")
            return
        status = reply.get("status")
        findings = reply.get("findings", [])
        if status == "PASS":
            state.final_qa_status = "PASS"
            state.final_qa_findings = []
            state.status = "COMPLETE"
            state.run_history.append("Final QA passed; project complete")
            self._mark(state, "FINAL_QA", "COMPLETE", "Final QA passed; project complete")
            return
        if status != "FAIL" or not isinstance(findings, list):
            state.status = "BLOCKED"
            state.run_history.append("Final QA returned an invalid verdict")
            return
        state.final_qa_status = "FAIL"
        state.final_qa_findings = [str(finding) for finding in findings]
        for finding in findings:
            if not isinstance(finding, dict):
                continue
            title = finding.get("title")
            description = finding.get("description")
            if isinstance(title, str) and isinstance(description, str):
                state.tasks.append(Task.create(title, description))
        if not any(task.status is TaskStatus.PENDING for task in state.tasks):
            state.status = "BLOCKED"
            state.run_history.append("Final QA failed without actionable findings")
            return
        state.run_history.append("Final QA failed; created corrective tasks")
        self._mark(state, "MANAGER", "CORRECTIVE", "Final QA failed; corrective tasks created")

    def pause(self) -> ProjectState:
        return self._set_control_status("PAUSED")

    def stop(self) -> ProjectState:
        return self._set_control_status("STOPPED")

    def resume(self) -> ProjectState:
        state = self._required_state()
        if state.status == "STOPPED":
            raise ValueError("stopped projects cannot resume; start a new run explicitly")
        interrupted = next(
            (
                task
                for task in state.tasks
                if task.id == state.current_task_id and task.status in {TaskStatus.RUNNING, TaskStatus.TESTING, TaskStatus.REVIEW}
            ),
            None,
        )
        if interrupted is not None:
            interrupted.status = TaskStatus.PENDING
            state.current_task_id = None
            state.run_history.append(f"Recovered interrupted task: {interrupted.title}")
            state.record_event("SYSTEM", "RECOVERY", f"Recovered interrupted task: {interrupted.title}", interrupted.id)
        state.status = "READY"
        state.run_history.append("Run resumed")
        self.store.save(state)
        return state

    def add_requirement(self, requirement: str) -> ProjectState:
        if not requirement.strip():
            raise ValueError("requirement must not be blank")
        state = self._required_state()
        normalized = requirement.strip()
        if normalized in state.amendments:
            return state
        state.amendments.append(normalized)
        state.tasks.append(Task.create(f"Implement amendment: {normalized}", normalized))
        state.final_qa_status = "NOT_RUN"
        state.status = "READY"
        state.run_history.append(f"Requirement amendment added: {normalized}")
        self._mark(state, "MANAGER", "AMENDMENT", f"Requirement amendment added: {normalized}")
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
        done_ids = {task.id for task in state.tasks if task.status is TaskStatus.DONE}
        pending = [
            task
            for task in state.tasks
            if task.status is TaskStatus.PENDING and all(dependency in done_ids for dependency in task.dependencies)
        ]
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
        self._mark(state, "CODER", "IMPLEMENTING", f"Coder started attempt {task.attempts}: {task.title}", task)
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
            self._mark(state, "TESTER", "TESTING", f"Tester verifying: {task.title}", task)
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
            self._mark(state, "REVIEWER", "REVIEW", f"Reviewer evaluating: {task.title}", task)
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
        self._mark(state, "GIT", "CHECKPOINT", f"Task approved and checkpointed: {task.title}", task)

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

    def _mark(self, state: ProjectState, agent: str, phase: str, message: str, task: Task | None = None) -> None:
        state.heartbeat = Heartbeat(
            agent=agent,
            phase=phase,
            task_id=task.id if task is not None else state.current_task_id,
            last_successful_action=message,
            consecutive_failures=sum(1 for item in (task.errors if task is not None else []) if item),
            attempt=task.attempts if task is not None else 0,
        )
        state.record_event(agent, phase, message, task.id if task is not None else state.current_task_id)
        self.store.save(state)
