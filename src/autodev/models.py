"""Durable domain models for an autonomous development run."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(slots=True)
class Heartbeat:
    timestamp: str = field(default_factory=utc_now)
    agent: str = "IDLE"
    phase: str = "IDLE"
    task_id: str | None = None
    last_successful_action: str = ""
    last_model_response: str = ""
    last_tool_execution: str = ""
    consecutive_failures: int = 0
    attempt: int = 0

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> Heartbeat:
        return cls(
            timestamp=str(value.get("timestamp", utc_now())),
            agent=str(value.get("agent", "IDLE")),
            phase=str(value.get("phase", "IDLE")),
            task_id=str(value["task_id"]) if value.get("task_id") else None,
            last_successful_action=str(value.get("last_successful_action", "")),
            last_model_response=str(value.get("last_model_response", "")),
            last_tool_execution=str(value.get("last_tool_execution", "")),
            consecutive_failures=int(value.get("consecutive_failures", 0)),
            attempt=int(value.get("attempt", 0)),
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class ActivityEvent:
    timestamp: str
    agent: str
    phase: str
    message: str
    task_id: str | None = None

    @classmethod
    def create(cls, agent: str, phase: str, message: str, task_id: str | None = None) -> ActivityEvent:
        return cls(utc_now(), agent, phase, message, task_id)

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> ActivityEvent:
        return cls(
            timestamp=str(value.get("timestamp", utc_now())),
            agent=str(value.get("agent", "SYSTEM")),
            phase=str(value.get("phase", "INFO")),
            message=str(value.get("message", "")),
            task_id=str(value["task_id"]) if value.get("task_id") else None,
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class TaskStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    TESTING = "TESTING"
    REVIEW = "REVIEW"
    DONE = "DONE"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"
    SUPERSEDED = "SUPERSEDED"


class ToolExecutionStatus(StrEnum):
    """Durable lifecycle of one tool call made for a task."""

    STARTED = "STARTED"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"


@dataclass(slots=True)
class ToolExecution:
    id: str
    task_id: str
    kind: str
    payload: list[str] | str
    status: ToolExecutionStatus = ToolExecutionStatus.STARTED
    started_at: str = field(default_factory=utc_now)
    finished_at: str | None = None
    detail: str = ""

    @classmethod
    def create(cls, task_id: str, kind: str, payload: list[str] | str) -> ToolExecution:
        return cls(id=str(uuid4()), task_id=task_id, kind=kind, payload=payload)

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> ToolExecution:
        payload = value.get("payload", "")
        if not isinstance(payload, (str, list)) or (isinstance(payload, list) and not all(isinstance(item, str) for item in payload)):
            raise ValueError("tool execution payload must be text or a string array")
        return cls(
            id=str(value["id"]), task_id=str(value["task_id"]), kind=str(value["kind"]), payload=payload,
            status=ToolExecutionStatus(str(value.get("status", ToolExecutionStatus.STARTED))),
            started_at=str(value.get("started_at", utc_now())),
            finished_at=str(value["finished_at"]) if value.get("finished_at") else None,
            detail=str(value.get("detail", "")),
        )

    def finish(self, status: ToolExecutionStatus, detail: str = "") -> None:
        self.status = status
        self.finished_at = utc_now()
        self.detail = detail

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["status"] = self.status.value
        return value


@dataclass(slots=True)
class Task:
    id: str
    title: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    attempts: int = 0
    errors: list[str] = field(default_factory=list)
    action_fingerprints: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    repair_of: str | None = None
    root_task_id: str = ""
    parent_task_id: str | None = None
    failure_fingerprint: str = ""
    strategy_generation: int = 0
    phase: str = "READY"
    semantic_call_keys: list[str] = field(default_factory=list)
    failures_in_strategy: int = 0
    strategy_history: list[str] = field(default_factory=list)
    rollback_count: int = 0
    rollback_reasons: dict[str, int] = field(default_factory=dict)
    capability_id: str = ""
    intent: str = ""
    acceptance_criteria: list[str] = field(default_factory=list)
    contract_version: int = 1
    last_repair_packet: dict[str, object] = field(default_factory=dict)
    acceptance_validator: dict[str, object] = field(default_factory=dict)
    last_validator_run_id: str = ""

    @classmethod
    def create(cls, title: str, description: str, dependencies: list[str] | None = None, repair_of: str | None = None, root_task_id: str | None = None, parent_task_id: str | None = None, failure_fingerprint: str = "", strategy_generation: int = 0, capability_id: str = "", intent: str = "", acceptance_criteria: list[str] | None = None, contract_version: int = 1) -> Task:
        task_id = str(uuid4())
        return cls(id=task_id, title=title, description=description, dependencies=dependencies or [], repair_of=repair_of, root_task_id=root_task_id or task_id, parent_task_id=parent_task_id, failure_fingerprint=failure_fingerprint, strategy_generation=strategy_generation, capability_id=capability_id, intent=intent, acceptance_criteria=acceptance_criteria or [], contract_version=contract_version)

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> Task:
        return cls(
            id=str(value["id"]),
            title=str(value["title"]),
            description=str(value["description"]),
            status=TaskStatus(str(value.get("status", TaskStatus.PENDING))),
            attempts=int(value.get("attempts", 0)),
            errors=[str(error) for error in value.get("errors", [])],  # type: ignore[arg-type]
            action_fingerprints=[
                str(fingerprint) for fingerprint in value.get("action_fingerprints", [])  # type: ignore[arg-type]
            ],
            dependencies=[str(dependency) for dependency in value.get("dependencies", [])],  # type: ignore[arg-type]
            repair_of=str(value["repair_of"]) if value.get("repair_of") else None,
            root_task_id=str(value.get("root_task_id") or value["id"]),
            parent_task_id=str(value["parent_task_id"]) if value.get("parent_task_id") else None,
            failure_fingerprint=str(value.get("failure_fingerprint", "")),
            strategy_generation=int(value.get("strategy_generation", 0)),
            phase=str(value.get("phase", "READY")),
            semantic_call_keys=[str(item) for item in value.get("semantic_call_keys", [])],  # type: ignore[arg-type]
            failures_in_strategy=int(value.get("failures_in_strategy", 0)),
            strategy_history=[str(item) for item in value.get("strategy_history", [])],  # type: ignore[arg-type]
            rollback_count=int(value.get("rollback_count", 0)),
            rollback_reasons={str(key): int(count) for key, count in dict(value.get("rollback_reasons", {})).items()},  # type: ignore[arg-type]
            capability_id=str(value.get("capability_id", "")),
            intent=str(value.get("intent", "")),
            acceptance_criteria=[str(item) for item in value.get("acceptance_criteria", [])],  # type: ignore[arg-type]
            contract_version=int(value.get("contract_version", 1)),
            last_repair_packet=dict(value.get("last_repair_packet", {})),  # type: ignore[arg-type]
            acceptance_validator=dict(value.get("acceptance_validator", {})),  # type: ignore[arg-type]
            last_validator_run_id=str(value.get("last_validator_run_id", "")),
        )

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["status"] = self.status.value
        return result


@dataclass(slots=True)
class ProjectState:
    original_spec: str
    model: str = "not selected"
    tasks: list[Task] = field(default_factory=list)
    status: str = "READY"
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    current_task_id: str | None = None
    last_checkpoint: str | None = None
    final_qa_status: str = "NOT_RUN"
    final_qa_findings: list[str] = field(default_factory=list)
    regression_history: list[str] = field(default_factory=list)
    heartbeat: Heartbeat = field(default_factory=Heartbeat)
    events: list[ActivityEvent] = field(default_factory=list)
    amendments: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    run_history: list[str] = field(default_factory=list)
    tool_executions: list[ToolExecution] = field(default_factory=list)
    visual_status: str = "NOT_RUN"
    visual_issues: list[str] = field(default_factory=list)
    visual_repair_cycles: int = 0
    environment: dict[str, object] = field(default_factory=dict)
    architecture: dict[str, object] = field(default_factory=dict)
    project_contract: dict[str, object] = field(default_factory=dict)
    capability_graph: dict[str, dict[str, object]] = field(default_factory=dict)
    repair_memory: list[dict[str, object]] = field(default_factory=list)
    research_cache: dict[str, dict[str, object]] = field(default_factory=dict)
    selected_primary_model: str = ""
    event_counters: dict[str, int] = field(default_factory=dict)
    provider_state: dict[str, object] = field(default_factory=dict)
    managed_processes: list[dict[str, object]] = field(default_factory=list)
    accepted_regressions: list[dict[str, object]] = field(default_factory=list)
    # Immutable acceptance evidence. Coder tool outcomes deliberately do not
    # live here: only a designated validator may decide task acceptance.
    validator_runs: list[dict[str, object]] = field(default_factory=list)
    provider_request_stats: dict[str, object] = field(default_factory=dict)
    terminal_status: str = ""
    terminal_reason: str = ""

    @classmethod
    def create(cls, original_spec: str) -> ProjectState:
        if not original_spec.strip():
            raise ValueError("original_spec must not be blank")
        return cls(original_spec=original_spec.strip())

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> ProjectState:
        return cls(
            original_spec=str(value["original_spec"]),
            model=str(value.get("model", "not selected")),
            tasks=[Task.from_dict(task) for task in value.get("tasks", [])],  # type: ignore[arg-type]
            status=str(value.get("status", "READY")),
            created_at=str(value.get("created_at", utc_now())),
            updated_at=str(value.get("updated_at", utc_now())),
            current_task_id=(
                str(value["current_task_id"]) if value.get("current_task_id") else None
            ),
            last_checkpoint=(
                str(value["last_checkpoint"]) if value.get("last_checkpoint") else None
            ),
            final_qa_status=str(value.get("final_qa_status", "NOT_RUN")),
            final_qa_findings=[str(finding) for finding in value.get("final_qa_findings", [])],  # type: ignore[arg-type]
            regression_history=[str(result) for result in value.get("regression_history", [])],  # type: ignore[arg-type]
            heartbeat=Heartbeat.from_dict(value.get("heartbeat", {})),  # type: ignore[arg-type]
            events=[ActivityEvent.from_dict(event) for event in value.get("events", [])],  # type: ignore[arg-type]
            amendments=[str(amendment) for amendment in value.get("amendments", [])],  # type: ignore[arg-type]
            decisions=[str(decision) for decision in value.get("decisions", [])],  # type: ignore[arg-type]
            run_history=[str(entry) for entry in value.get("run_history", [])],  # type: ignore[arg-type]
            tool_executions=[ToolExecution.from_dict(item) for item in value.get("tool_executions", [])],  # type: ignore[arg-type]
            visual_status=str(value.get("visual_status", "NOT_RUN")),
            visual_issues=[str(issue) for issue in value.get("visual_issues", [])],  # type: ignore[arg-type]
            visual_repair_cycles=int(value.get("visual_repair_cycles", 0)),
            environment=dict(value.get("environment", {})),  # type: ignore[arg-type]
            architecture=dict(value.get("architecture", {})),  # type: ignore[arg-type]
            project_contract=dict(value.get("project_contract", {})),  # type: ignore[arg-type]
            capability_graph={str(key): dict(item) for key, item in dict(value.get("capability_graph", {})).items() if isinstance(item, dict)},  # type: ignore[arg-type]
            repair_memory=[dict(item) for item in value.get("repair_memory", []) if isinstance(item, dict)],  # type: ignore[arg-type]
            research_cache={str(key): dict(item) for key, item in dict(value.get("research_cache", {})).items() if isinstance(item, dict)},  # type: ignore[arg-type]
            selected_primary_model=str(value.get("selected_primary_model", "")),
            event_counters={str(key): int(count) for key, count in dict(value.get("event_counters", {})).items()},  # type: ignore[arg-type]
            provider_state=dict(value.get("provider_state", {})),  # type: ignore[arg-type]
            managed_processes=[dict(item) for item in value.get("managed_processes", []) if isinstance(item, dict)],  # type: ignore[arg-type]
            accepted_regressions=[dict(item) for item in value.get("accepted_regressions", []) if isinstance(item, dict)],  # type: ignore[arg-type]
            validator_runs=[dict(item) for item in value.get("validator_runs", []) if isinstance(item, dict)],  # type: ignore[arg-type]
            provider_request_stats=dict(value.get("provider_request_stats", {})),  # type: ignore[arg-type]
            terminal_status=str(value.get("terminal_status", "")),
            terminal_reason=str(value.get("terminal_reason", "")),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "original_spec": self.original_spec,
            "model": self.model,
            "tasks": [task.to_dict() for task in self.tasks],
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "current_task_id": self.current_task_id,
            "last_checkpoint": self.last_checkpoint,
            "final_qa_status": self.final_qa_status,
            "final_qa_findings": self.final_qa_findings,
            "regression_history": self.regression_history,
            "heartbeat": self.heartbeat.to_dict(),
            "events": [event.to_dict() for event in self.events[-500:]],
            "amendments": self.amendments,
            "decisions": self.decisions,
            "run_history": self.run_history,
            "tool_executions": [execution.to_dict() for execution in self.tool_executions[-200:]],
            "visual_status": self.visual_status,
            "visual_issues": self.visual_issues[-50:],
            "visual_repair_cycles": self.visual_repair_cycles,
            "environment": self.environment,
            "architecture": self.architecture,
            "project_contract": self.project_contract,
            "capability_graph": self.capability_graph,
            "repair_memory": self.repair_memory[-100:],
            "research_cache": self.research_cache,
            "selected_primary_model": self.selected_primary_model,
            "event_counters": self.event_counters,
            "provider_state": self.provider_state,
            "managed_processes": self.managed_processes[-100:],
            "accepted_regressions": self.accepted_regressions[-20:],
            "validator_runs": self.validator_runs[-300:],
            "provider_request_stats": self.provider_request_stats,
            "terminal_status": self.terminal_status,
            "terminal_reason": self.terminal_reason,
        }

    def record_event(self, agent: str, phase: str, message: str, task_id: str | None = None) -> None:
        self.events.append(ActivityEvent.create(agent, phase, message, task_id))
        key = f"{agent}:{phase}"
        self.event_counters[key] = self.event_counters.get(key, 0) + 1
        self.events = self.events[-500:]

    def record_validator_result(
        self,
        *,
        task: Task,
        validator_id: str,
        command: list[str],
        result: object,
        failure_class: str,
        previous_validator_run_id: str | None,
        execution_backend: str = "workspace-tools",
        related_tool_call_id: str | None = None,
    ) -> dict[str, object]:
        """Persist one immutable, capability-owned validator outcome."""
        run_id = str(uuid4())
        stdout = str(getattr(result, "stdout", ""))
        stderr = str(getattr(result, "stderr", ""))
        exit_code = int(getattr(result, "exit_code", 1))
        fingerprint_source = "\n".join((task.capability_id or task.id, failure_class, stdout[-1200:], stderr[-1200:]))
        record: dict[str, object] = {
            "validator_run_id": run_id,
            "capability_id": task.capability_id or task.id,
            "validator_id": validator_id,
            "validator_kind": "test" if any(part in {"pytest", "unittest"} for part in command) else "command",
            "exact_command": list(command),
            "cwd": "workspace",
            "start_time": utc_now(),
            "end_time": utc_now(),
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": stderr,
            "failure_class": failure_class,
            "failure_fingerprint": hashlib.sha256(fingerprint_source.encode("utf-8")).hexdigest(),
            "execution_backend": execution_backend,
            "related_tool_call_id": related_tool_call_id,
            "previous_validator_run_id": previous_validator_run_id,
        }
        self.validator_runs.append(record)
        self.validator_runs = self.validator_runs[-300:]
        task.last_validator_run_id = run_id
        return record
