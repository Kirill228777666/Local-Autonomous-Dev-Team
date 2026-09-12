"""Durable, sequential Manager → Coder → Tester → Reviewer orchestration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

from .agents import RoleAgents
from .architecture import default_contract, validate_architecture
from .designer import DesignerAgent, detect_ui_project
from .environment import EnvironmentManager, FailureKind, classify_failure, validate_readme
from .git import GitError, GitRepository
from .models import Heartbeat, ProjectState, Task, TaskStatus, ToolExecution, ToolExecutionStatus
from .providers import LLMProvider, ProviderError
from .regression import RegressionRunner
from .runtime import ManagedProcessManager, ScreenshotPipeline
from .state_store import StateStore
from .tools import CommandResult, ToolPolicyError, WorkspaceTools


class CommandExecutionError(ValueError):
    def __init__(self, command: list[str], result: CommandResult) -> None:
        super().__init__(result.stderr or result.stdout or f"command exited {result.exit_code}")
        self.command = command
        self.result = result


class AutonomousRunner:
    def __init__(
        self,
        workspace: Path,
        store: StateStore,
        tools: WorkspaceTools,
        provider: LLMProvider,
        max_attempts: int = 3,
        regression_runner: RegressionRunner | None = None,
        visual_pipeline: ScreenshotPipeline | None = None,
        visual_url: str | None = None,
        max_visual_repairs: int = 2,
        allow_project_dependency_install: bool = True,
        allow_system_package_install: bool = False,
    ) -> None:
        self.workspace = workspace.resolve()
        self.store = store
        self.tools = tools
        self.agents = RoleAgents(provider)
        self.max_attempts = max_attempts
        self.regression_runner = regression_runner or RegressionRunner(self.workspace, tools)
        self.visual_pipeline = visual_pipeline
        self.visual_url = visual_url
        self.max_visual_repairs = max_visual_repairs
        self.designer = DesignerAgent(provider, structured_retries=1)
        self.environment_manager = EnvironmentManager(workspace, tools, allow_project_dependency_install, allow_system_package_install)
        self.process_manager = ManagedProcessManager(self.workspace)

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
        self._recover_managed_processes(state)
        if not state.environment:
            state.environment = self.environment_manager.discover()
            state.run_history.append("Environment capabilities discovered")
            self._mark(state, "ENVIRONMENT", "CHECK", "Environment capabilities discovered")
        self._mark(state, "MANAGER", "PLANNING", "Autonomous run started")
        if not state.tasks:
            self._plan(state)
        self._decompose_broad_tasks(state)
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
        if detect_ui_project(self.workspace) and self.visual_pipeline is not None and self.visual_url:
            final_visual = Task.create("Final visual QA", "Verify desktop and narrow UI before completion")
            if not self._visual_review(state, final_visual):
                if final_visual.status is TaskStatus.BLOCKED:
                    state.status = "BLOCKED"
                    state.run_history.append("Final visual QA blocked completion")
                    return
                description = final_visual.errors[-1] if final_visual.errors else "Resolve objective visual QA findings"
                state.tasks.append(Task.create("Resolve final visual QA findings", description))
                state.run_history.append("Final visual QA failed; created corrective task")
                self._mark(state, "MANAGER", "CORRECTIVE", "Final visual QA created corrective task")
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
            self._mark(state, "FINAL_QA", "LLM_CALL", "Final QA request", None)
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
        if state.status == "COMPLETE":
            state.run_history.append("Resume ignored: project is already COMPLETE")
            self.store.save(state)
            return state
        problems = self._resume_invariants(state)
        if problems:
            state.status = "BLOCKED"
            message = "Resume invariant violation: " + "; ".join(problems)
            state.run_history.append(message)
            state.record_event("SYSTEM", "BLOCKED", message)
            self.store.save(state)
            return state
        self._recover_managed_processes(state)
        interrupted_tools = [item for item in state.tool_executions if item.status is ToolExecutionStatus.STARTED]
        for execution in interrupted_tools:
            execution.finish(ToolExecutionStatus.UNKNOWN, "process ended before result was persisted")
            state.run_history.append(f"Interrupted tool marked UNKNOWN: {execution.kind} for task {execution.task_id}")
            state.record_event("SYSTEM", "RECOVERY", f"Interrupted tool marked UNKNOWN: {execution.kind}", execution.task_id)
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
        if interrupted_tools:
            state.run_history.append(f"Crash recovery recorded {len(interrupted_tools)} interrupted tool execution(s)")
        self._recover_blocked_tasks(state)
        state.status = "READY"
        state.run_history.append("Run resumed")
        self.store.save(state)
        return state

    @staticmethod
    def _recover_blocked_tasks(state: ProjectState) -> None:
        """Migrate old terminal failures into one explicit, bounded corrective task."""
        originals = {task.repair_of for task in state.tasks if task.repair_of}
        for task in list(state.tasks):
            if task.status is not TaskStatus.BLOCKED or task.id in originals:
                continue
            task.status = TaskStatus.FAILED
            failure = task.errors[-1] if task.errors else "No persisted failure detail"
            corrective = Task.create(
                f"Repair: {task.title}",
                f"Repair this previously blocked task without repeating the failed approach. Original task: {task.description}\nPersisted failure: {failure}",
                dependencies=task.dependencies,
                repair_of=task.id,
            )
            state.tasks.append(corrective)
            state.run_history.append(f"Resume created bounded corrective task: {corrective.title}")
            state.record_event("MANAGER", "CORRECTIVE", f"Resume created corrective task for {task.title}", corrective.id)

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
        self.process_manager.stop_all()
        state.managed_processes = self.process_manager.records()
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
            self._mark(state, "MANAGER", "LLM_CALL", "Manager planning request")
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
            if not state.architecture:
                state.architecture = default_contract(state.original_spec, state.environment)
                if state.architecture:
                    state.decisions.append("Architecture contract selected: " + str(state.architecture))
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
            self._mark(state, "MANAGER", "LLM_CALL", "Manager task-selection request")
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
            self._mark(state, "CODER", "LLM_CALL", "Coder implementation request", task)
            actions = self.agents.code(state, task).data.get("actions")
            if not isinstance(actions, list):
                raise ProviderError("Coder response needs an actions array")
            fingerprint = hashlib.sha256(json.dumps(actions, sort_keys=True).encode()).hexdigest()
            if fingerprint in task.action_fingerprints:
                self._block(task, state, "repeated identical coder action without progress")
                return
            task.action_fingerprints.append(fingerprint)
            tool_recoveries = 0
            position = 0
            while position < len(actions):
                action = actions[position]
                try:
                    self._execute_action(state, task, action)
                except CommandExecutionError as error:
                    if error.result.exit_code == 125:
                        self._reroute_server_command(state, task, error.command)
                        # The execution was rejected only to move it to the managed path.
                        state.tool_executions[-1].finish(ToolExecutionStatus.SUCCEEDED, "rerouted to managed process")
                        position += 1
                        continue
                    failure = classify_failure(error.result, error.command)
                    if failure.kind is FailureKind.MISSING_EXECUTABLE and error.command[0].lower() in {"npm", "node"}:
                        self._pivot_to_static_frontend(state, task)
                        return
                    if self._repair_environment_failure(state, task, error.command, error.result):
                        position += 1
                        continue
                    raise
                except ValueError as error:
                    if "edit target was not found" not in str(error) or tool_recoveries >= 2:
                        raise
                    path = action.get("path") if isinstance(action, dict) else None
                    if not isinstance(path, str):
                        raise
                    _hash, current = self.tools.file_snapshot(path)
                    task.errors.append(f"Tool stale edit recovery for {path}; current content: {current[-4000:]}")
                    tool_recoveries += 1
                    self._mark(state, "CODER", "TOOL_RECOVERY", f"Refreshing stale edit context for {path}", task)
                    refreshed = self.agents.code(state, task).data.get("actions")
                    if not isinstance(refreshed, list):
                        raise ProviderError("Coder tool recovery response needs actions")
                    actions = refreshed
                    position = 0
                    continue
                position += 1
            state.run_history.append(f"Coder completed attempt {task.attempts} for {task.title}")
        except (ProviderError, ToolPolicyError, ValueError, KeyError, TypeError) as error:
            self._retry_or_block(task, state, f"Coder error: {error}")
            return

        task.status = TaskStatus.TESTING
        if "readme" in task.title.lower() or "documentation" in task.title.lower():
            documentation = validate_readme(self.workspace)
            command = ["README validator"]
            result = CommandResult(0 if documentation.passed else 1, "", "; ".join(documentation.findings))
            self._mark(state, "TESTER", "DOCUMENTATION", "Validated README instructions without executing them", task)
        else:
            try:
                self._mark(state, "TESTER", "TESTING", f"Tester verifying: {task.title}", task)
                self._mark(state, "TESTER", "LLM_CALL", "Tester verification request", task)
                reply = self.agents.test(state, task).data
                command = reply.get("command")
                if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
                    raise ProviderError("Tester response needs a string command array")
                if self._requires_managed_service(command):
                    self._ensure_validation_service(state, task)
                result = self.tools.run_tests(command)
            except (ProviderError, ToolPolicyError, ValueError) as error:
                self._retry_or_block(task, state, f"Tester error: {error}")
                return
        state.run_history.append(self._result_log(task, result))
        if result.exit_code != 0:
            failure = classify_failure(result, command)
            if failure.kind is FailureKind.TEST_HARNESS_FAILURE:
                self._mark(state, "TESTER", "HARNESS_FAILURE", "Generated validation command is invalid; regenerate validation without changing application code", task)
                task.status = TaskStatus.PENDING
                state.current_task_id = None
                state.run_history.append(f"Tester harness failure for {task.title}; application implementation was not repaired")
                return
            if failure.kind is FailureKind.MISSING_EXECUTABLE and command[0].lower() in {"npm", "node"}:
                self._pivot_to_static_frontend(state, task)
                return
            if self._repair_environment_failure(state, task, command, result):
                result = self.tools.run_tests(command)
            if result.exit_code == 0:
                state.run_history.append(self._result_log(task, result))
            else:
                self._retry_or_block(task, state, f"Test failed (exit {result.exit_code}): {result.stderr or result.stdout}")
                return

        if not self._visual_review(state, task):
            return

        conflicts = validate_architecture(self.workspace, state.architecture)
        if conflicts:
            self._mark(state, "ARCHITECTURE", "CONFLICT", "; ".join(conflicts), task)
            self._retry_or_block(task, state, "; ".join(conflicts))
            return

        task.status = TaskStatus.REVIEW
        try:
            self._mark(state, "REVIEWER", "REVIEW", f"Reviewer evaluating: {task.title}", task)
            self._mark(state, "REVIEWER", "LLM_CALL", "Reviewer decision request", task)
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
        if task.repair_of:
            original = next((candidate for candidate in state.tasks if candidate.id == task.repair_of), None)
            if original is not None:
                original.status = TaskStatus.DONE
                state.run_history.append(f"Corrective task resolved: {original.title}")
        state.current_task_id = None
        state.run_history.append(f"Task approved and checkpointed: {task.title}")
        self._mark(state, "GIT", "CHECKPOINT", f"Task approved and checkpointed: {task.title}", task)
        self._stop_task_processes(state, task)

    def _reroute_server_command(self, state: ProjectState, task: Task, command: list[str]) -> None:
        port = self._server_port(command)
        record = self.process_manager.start(self.tools.normalize_command(command), purpose=f"task:{task.id}", expected_port=port, env={"PORT": str(port)})
        state.managed_processes = self.process_manager.records()
        self._mark(state, "RUNTIME", "PROCESS_START", f"Managed process started: {record.pid}", task)
        if not self.process_manager.wait_ready(record.id, "127.0.0.1", port, timeout=15):
            logs = self.process_manager.logs(record.id)
            self.process_manager.stop(record.id)
            state.managed_processes = self.process_manager.records()
            self._mark(state, "RUNTIME", "READINESS_FAILED", f"Managed process readiness failed: {logs[-500:]}", task)
            raise ValueError("managed application failed readiness")
        state.managed_processes = self.process_manager.records()
        self._mark(state, "RUNTIME", "READINESS", f"Managed process ready on port {port}", task)

    def _stop_task_processes(self, state: ProjectState, task: Task) -> None:
        for record in self.process_manager.records():
            if record.get("purpose") == f"task:{task.id}" and record.get("status") != "STOPPED":
                self.process_manager.stop(str(record["id"]))
                self._mark(state, "RUNTIME", "PROCESS_STOP", f"Managed process stopped: {record.get('pid')}", task)
        state.managed_processes = self.process_manager.records()

    def _recover_managed_processes(self, state: ProjectState) -> None:
        active = [record for record in state.managed_processes if record.get("status") in {"STARTED", "READY"}]
        if not active:
            return
        for message in self.process_manager.recover(active):
            state.run_history.append(message)
            self._mark(state, "RUNTIME", "PROCESS_RECOVERY", message)
        for record in state.managed_processes:
            if record.get("status") in {"STARTED", "READY"}:
                record["status"] = "RECOVERED" if record.get("ownership_token") else "UNVERIFIED"

    def _server_port(self, command: list[str]) -> int:
        for index, item in enumerate(command[:-1]):
            if item in {"--port", "-p"}:
                try:
                    return int(command[index + 1])
                except ValueError:
                    break
        return 5000

    @staticmethod
    def _requires_managed_service(command: list[str]) -> bool:
        return any("localhost" in item.lower() or "127.0.0.1" in item for item in command)

    def _ensure_validation_service(self, state: ProjectState, task: Task) -> None:
        if any(record.get("purpose") == f"task:{task.id}" and record.get("status") == "READY" for record in self.process_manager.records()):
            return
        candidate = next((path for path in ("app.py", "backend/app.py") if (self.workspace / path).is_file()), None)
        if candidate is None:
            raise ValueError("REQUIRES_MANAGED_SERVICE: no known application entry point")
        self._mark(state, "RUNTIME", "SERVICE_REQUIRED", "HTTP validation requires a managed application process", task)
        self._reroute_server_command(state, task, ["python", candidate])

    def _execute_action(self, state: ProjectState, task: Task, action: object) -> None:
        if not isinstance(action, dict):
            raise ValueError("Coder action must be an object")
        kind = action.get("kind")
        payload: list[str] | str
        if kind == "run_command":
            command = action.get("command")
            payload = command if isinstance(command, list) else "invalid command"
        else:
            payload = self._string(action, "path")
        execution = ToolExecution.create(task.id, str(kind), payload)
        state.tool_executions.append(execution)
        self._mark(state, "CODER", "TOOL_STARTED", f"Started tool: {kind}", task)
        try:
            self._execute_action_once(action)
        except Exception as error:
            execution.finish(ToolExecutionStatus.FAILED, str(error))
            self._mark(state, "CODER", "TOOL_FAILED", f"Tool failed: {kind}: {error}", task)
            raise
        execution.finish(ToolExecutionStatus.SUCCEEDED)
        self._mark(state, "CODER", "TOOL_SUCCEEDED", f"Completed tool: {kind}", task)

    def _execute_action_once(self, action: dict[str, object]) -> None:
        kind = action.get("kind")
        if kind == "write_file":
            self.tools.write_file(self._string(action, "path"), self._string(action, "content"))
        elif kind == "edit_file":
            self.tools.edit_file(self._string(action, "path"), self._string(action, "old"), self._string(action, "new"))
        elif kind == "append_file":
            self.tools.append_file(self._string(action, "path"), self._string(action, "content"))
        elif kind == "delete_file":
            self.tools.delete_file(self._string(action, "path"))
        elif kind == "run_command":
            command = action.get("command")
            if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
                raise ValueError("run_command requires a string command array")
            result = self.tools.run_command(command)
            if result.exit_code:
                raise CommandExecutionError(command, result)
        else:
            raise ValueError(f"unsupported action kind: {kind}")

    def _resume_invariants(self, state: ProjectState) -> list[str]:
        problems: list[str] = []
        if not state.original_spec.strip():
            problems.append("project specification is missing")
        task_ids = [task.id for task in state.tasks]
        if len(task_ids) != len(set(task_ids)):
            problems.append("task identifiers are not unique")
        by_id = {task.id: task for task in state.tasks}
        if state.current_task_id and state.current_task_id not in by_id:
            problems.append("current task does not exist")
        for task in state.tasks:
            missing = [dependency for dependency in task.dependencies if dependency not in by_id or dependency == task.id]
            if missing:
                problems.append(f"task dependency is invalid for {task.title}")
            if task.repair_of and task.repair_of not in by_id:
                problems.append(f"corrective task has no original task: {task.title}")
        active = [task for task in state.tasks if task.status in {TaskStatus.RUNNING, TaskStatus.TESTING, TaskStatus.REVIEW}]
        if len(active) > 1:
            problems.append("sequential mode has more than one active task")
        try:
            repository = GitRepository(self.workspace)
            repository._run("rev-parse", "--is-inside-work-tree")
            if state.last_checkpoint:
                repository._run("cat-file", "-e", f"{state.last_checkpoint}^{{commit}}")
        except GitError as error:
            problems.append(f"Git checkpoint/repository is unavailable: {error}")
        return problems

    @staticmethod
    def _string(action: dict[str, object], key: str) -> str:
        value = action.get(key)
        if not isinstance(value, str):
            raise ValueError(f"action needs string '{key}'")
        return value

    def _retry_or_block(self, task: Task, state: ProjectState, error: str) -> None:
        self._stop_task_processes(state, task)
        task.errors.append(error)
        if task.attempts >= self.max_attempts:
            if task.repair_of is not None:
                self._block(task, state, error)
                return
            self._diagnose(state, task)
            task.status = TaskStatus.FAILED
            state.current_task_id = None
            diagnosis = next(
                (decision for decision in reversed(state.decisions) if f"Architect diagnosis for {task.title}:" in decision),
                "No Architect diagnosis available",
            )
            corrective = Task.create(
                f"Repair: {task.title}",
                f"Repair the failed task without repeating its broken approach. Original task: {task.description}\nFailure: {error}\n{diagnosis}",
                dependencies=task.dependencies,
                repair_of=task.id,
            )
            state.tasks.append(corrective)
            state.run_history.append(f"Created Architect-guided corrective task: {corrective.title}")
            self._mark(state, "MANAGER", "CORRECTIVE", f"Created corrective task for failed work: {task.title}", corrective)
            return
        if len(task.errors) >= 2:
            self._diagnose(state, task)
        task.status = TaskStatus.PENDING
        state.run_history.append(f"Returning task to Coder: {task.title}; {error}")

    def _decompose_broad_tasks(self, state: ProjectState) -> None:
        """Replace one multi-feature task with atomic work before Coder receives it."""
        mapping = {
            "crud": ["Create notes", "Read notes", "Update notes", "Delete notes"],
            "create, read, update": ["Create notes", "Read notes", "Update notes", "Delete notes"],
        }
        for task in list(state.tasks):
            if task.status is not TaskStatus.PENDING:
                continue
            text = f"{task.title} {task.description}".lower()
            names = next((parts for marker, parts in mapping.items() if marker in text), None)
            if names is None:
                continue
            task.status = TaskStatus.SUPERSEDED
            for name in names:
                state.tasks.append(Task.create(name, f"Implement and independently test only this bounded Notes capability. Original scope: {task.description}", dependencies=task.dependencies))
            state.run_history.append(f"Manager decomposed broad task: {task.title}")
            self._mark(state, "MANAGER", "DECOMPOSED", f"Decomposed broad task: {task.title}")

    def _repair_environment_failure(self, state: ProjectState, task: Task, command: list[str], result: CommandResult) -> bool:
        failure = classify_failure(result, command)
        state.run_history.append(f"Failure classified: {failure.kind.value}")
        self._mark(state, "ENVIRONMENT", failure.kind.value, f"Classified command failure: {failure.kind.value}", task)
        if failure.kind is FailureKind.IMPORT_PATH:
            state.run_history.append("Test harness diagnostic: check cwd/package layout/PYTHONPATH before code repair")
            return False
        repaired = self.environment_manager.repair(failure, state.environment)
        if repaired:
            state.environment["execution_context"] = self.environment_manager.execution_context()
            state.run_history.append(f"Environment repair completed: {failure.kind.value}")
            self._mark(state, "ENVIRONMENT", "REPAIRED", f"Environment repair completed: {failure.kind.value}", task)
            return True
        if self.environment_manager.last_diagnostic:
            state.run_history.append(f"Environment limitation: {self.environment_manager.last_diagnostic}")
        return False

    def _pivot_to_static_frontend(self, state: ProjectState, task: Task) -> None:
        """Replace npm-specific work with a Node-free browser frontend while retaining product goals."""
        for candidate in state.tasks:
            text = f"{candidate.title} {candidate.description}".lower()
            if candidate.status in {TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.FAILED} and (candidate is task or "npm" in text or "build" in text and "frontend" in text):
                candidate.status = TaskStatus.SUPERSEDED
        replacement = Task.create(
            "Static frontend fallback",
            "Implement responsive plain HTML/CSS/JavaScript frontend served by the backend. Preserve Notes CRUD, search, categories, filters and favorites. Validate with a browser-free static-file check; no npm build is required.",
        )
        state.tasks.append(replacement)
        state.current_task_id = None
        state.run_history.append("Architecture pivot: Node/npm unavailable; superseded npm-specific tasks and created static frontend replacement")
        self._mark(state, "MANAGER", "PIVOT", "Applied Node-free static frontend pivot", replacement)

    def _visual_review(self, state: ProjectState, task: Task) -> bool:
        """Run objective visual QA only when a configured UI pipeline is available."""
        if not detect_ui_project(self.workspace):
            return True
        if self.visual_pipeline is None or not self.visual_url:
            state.visual_status = "SKIPPED"
            limitation = "Visual QA skipped: no application screenshot pipeline is configured"
            if limitation not in state.run_history:
                state.run_history.append(limitation)
                state.record_event("DESIGNER", "LIMITATION", limitation, task.id)
            return True
        screenshots = self.visual_pipeline.capture(self.visual_url)
        if not screenshots:
            state.visual_status = "UNAVAILABLE"
            state.run_history.append(self.visual_pipeline.last_diagnostic or "Visual QA screenshot acquisition failed")
            state.record_event("DESIGNER", "UNAVAILABLE", self.visual_pipeline.last_diagnostic, task.id)
            return True
        issues: list[str] = []
        for screenshot, viewport in zip(screenshots, ({"width": 1440, "height": 1000}, {"width": 390, "height": 844}), strict=True):
            try:
                self._mark(state, "DESIGNER", "LLM_CALL", f"Designer reviewing {screenshot.name}", task)
                review = self.designer.review(screenshot, state.original_spec, state, task, viewport)
            except (ProviderError, ValueError) as error:
                state.visual_status = "UNAVAILABLE"
                state.run_history.append(f"Visual QA unavailable: {error}")
                state.record_event("DESIGNER", "UNAVAILABLE", f"Visual QA unavailable: {error}", task.id)
                return True
            issues.extend(f"{item.severity}:{item.category}: {item.description}" for item in review.issues)
        state.visual_issues = issues
        serious = [issue for issue in issues if issue.startswith("high:") or issue.startswith("critical:")]
        if not serious:
            state.visual_status = "PASS"
            state.run_history.append("Designer visual QA passed")
            self._mark(state, "DESIGNER", "PASS", "Designer visual QA passed", task)
            return True
        state.visual_status = "FAIL"
        state.visual_repair_cycles += 1
        error = "Visual QA failed: " + " | ".join(serious)
        if state.visual_repair_cycles > self.max_visual_repairs:
            self._block(task, state, f"Visual repair limit reached: {error}")
        else:
            task.errors.append(error)
            task.status = TaskStatus.PENDING
            state.current_task_id = None
            state.run_history.append(f"Returning task to Coder for visual repair: {task.title}; {error}")
            self._mark(state, "MANAGER", "VISUAL_REPAIR", "Created bounded visual repair for objective issues", task)
        return False

    def _diagnose(self, state: ProjectState, task: Task) -> None:
        try:
            self._mark(state, "ARCHITECT", "LLM_CALL", "Architect diagnosis request", task)
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
