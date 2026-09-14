"""Compact, durable evidence and routing for informed task-local repair."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from enum import StrEnum

from .tools import CommandResult
from .validation import ValidationOutcome


class FailureClass(StrEnum):
    CONTRACT_CONFLICT = "CONTRACT_CONFLICT"
    DEPENDENCY_API_MISMATCH = "DEPENDENCY_API_MISMATCH"
    TEST_IMPLEMENTATION_BUG = "TEST_IMPLEMENTATION_BUG"
    TEST_STATE_ISOLATION_FAILURE = "TEST_STATE_ISOLATION_FAILURE"
    APPLICATION_LOGIC_FAILURE = "APPLICATION_LOGIC_FAILURE"
    APPLICATION_IMPORT_ERROR = "APPLICATION_IMPORT_ERROR"
    RUNTIME_PROCESS_FAILURE = "RUNTIME_PROCESS_FAILURE"
    TRUE_ENVIRONMENT_FAILURE = "TRUE_ENVIRONMENT_FAILURE"
    VALIDATION_INFRASTRUCTURE_FAILURE = "VALIDATION_INFRASTRUCTURE_FAILURE"
    REGRESSION_FAILURE = "REGRESSION_FAILURE"


def route_failure(outcome: ValidationOutcome, detail: str = "") -> FailureClass:
    if outcome is ValidationOutcome.DEPENDENCY_API_MISMATCH:
        return FailureClass.DEPENDENCY_API_MISMATCH
    if outcome is ValidationOutcome.TEST_IMPLEMENTATION_BUG:
        return FailureClass.TEST_IMPLEMENTATION_BUG
    if outcome is ValidationOutcome.TEST_STATE_ISOLATION_FAILURE:
        return FailureClass.TEST_STATE_ISOLATION_FAILURE
    if outcome is ValidationOutcome.APPLICATION_IMPORT_ERROR:
        return FailureClass.APPLICATION_IMPORT_ERROR
    if outcome in {ValidationOutcome.TOOL_MISSING, ValidationOutcome.IMPORT_OR_ENVIRONMENT_ERROR}:
        return FailureClass.TRUE_ENVIRONMENT_FAILURE
    if outcome is ValidationOutcome.COMMAND_INVALID:
        return FailureClass.VALIDATION_INFRASTRUCTURE_FAILURE
    if outcome is ValidationOutcome.TIMEOUT:
        return FailureClass.RUNTIME_PROCESS_FAILURE
    return FailureClass.APPLICATION_LOGIC_FAILURE


@dataclass(slots=True)
class RepairEvidencePacket:
    failure_id: str
    capability_id: str
    contract_version: int
    attempt_number: int
    failed_command: list[str]
    validator_type: str
    failure_class: FailureClass
    source_validator_run_id: str = ""
    exception_type: str = ""
    exception_message: str = ""
    relevant_stdout: str = ""
    relevant_stderr: str = ""
    dependency_versions: dict[str, str] = field(default_factory=dict)
    acceptance_intent: list[str] = field(default_factory=list)
    project_contract: dict[str, object] = field(default_factory=dict)
    protected_capabilities: list[str] = field(default_factory=list)
    previous_repair_strategies: list[str] = field(default_factory=list)
    research_evidence: list[dict[str, str]] = field(default_factory=list)

    @classmethod
    def from_validation(cls, *, capability_id: str, contract_version: int, attempt_number: int, command: list[str], outcome: ValidationOutcome, result: CommandResult, contract: dict[str, object], dependency_versions: dict[str, str] | None = None, acceptance_intent: list[str] | None = None, protected_capabilities: list[str] | None = None, previous_repair_strategies: list[str] | None = None, source_validator_run_id: str = "") -> "RepairEvidencePacket":
        detail = f"{result.stderr}\n{result.stdout}".strip()
        exception = re.search(r"([A-Za-z_][\w.]*Error|AttributeError|KeyError):\s*([^\n]+)", detail)
        message = exception.group(2).strip() if exception else detail[-1200:]
        exception_type = exception.group(1) if exception else ""
        fingerprint_data = f"{capability_id}|{outcome.value}|{exception_type}|{message}"
        return cls(
            hashlib.sha256(fingerprint_data.encode("utf-8")).hexdigest(), capability_id, contract_version, attempt_number,
            list(command), "test" if any(item in {"pytest", "unittest"} for item in command) else "command",
            route_failure(outcome, detail), source_validator_run_id, exception_type, message, result.stdout[-2000:], result.stderr[-2000:],
            dependency_versions or {}, acceptance_intent or [], contract, protected_capabilities or [], previous_repair_strategies or [],
        )

    def research_query(self) -> str:
        package, version = next(iter(self.dependency_versions.items()), ("", ""))
        symbol = re.search(r"['\"]([A-Za-z_][\w.]*)['\"]", self.exception_message)
        missing = re.search(r"has no attribute ['\"]([A-Za-z_][\w.]*)['\"]", self.exception_message)
        terms = [package, version, symbol.group(1) if symbol else "", missing.group(1) if missing else "", self.exception_type or "AttributeError"]
        return " ".join(part for part in terms if part)

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["failure_class"] = self.failure_class.value
        return value

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "RepairEvidencePacket":
        """Rehydrate durable evidence without trusting loosely typed state."""
        failure_class = FailureClass(str(value.get("failure_class", FailureClass.APPLICATION_LOGIC_FAILURE.value)))
        return cls(
            failure_id=str(value.get("failure_id", "")),
            capability_id=str(value.get("capability_id", "")),
            contract_version=int(value.get("contract_version", 1)),
            attempt_number=int(value.get("attempt_number", 0)),
            failed_command=[str(item) for item in value.get("failed_command", []) if isinstance(item, str)],
            validator_type=str(value.get("validator_type", "command")),
            failure_class=failure_class,
            source_validator_run_id=str(value.get("source_validator_run_id", "")),
            exception_type=str(value.get("exception_type", "")),
            exception_message=str(value.get("exception_message", "")),
            relevant_stdout=str(value.get("relevant_stdout", "")),
            relevant_stderr=str(value.get("relevant_stderr", "")),
            dependency_versions={str(k): str(v) for k, v in dict(value.get("dependency_versions", {})).items()} if isinstance(value.get("dependency_versions"), dict) else {},
            acceptance_intent=[str(item) for item in value.get("acceptance_intent", []) if isinstance(item, str)],
            project_contract=dict(value.get("project_contract", {})) if isinstance(value.get("project_contract"), dict) else {},
            protected_capabilities=[str(item) for item in value.get("protected_capabilities", []) if isinstance(item, str)],
            previous_repair_strategies=[str(item) for item in value.get("previous_repair_strategies", []) if isinstance(item, str)],
            research_evidence=[dict(item) for item in value.get("research_evidence", []) if isinstance(item, dict)],
        )


class RepairMemory:
    """Remember failed strategy/evidence combinations without banning new evidence."""

    def __init__(self, records: list[dict[str, object]] | None = None) -> None:
        self.records = records if records is not None else []

    @staticmethod
    def _evidence_key(packet: RepairEvidencePacket) -> str:
        sources = "|".join(sorted(str(item.get("url", "")) for item in packet.research_evidence))
        return hashlib.sha256(sources.encode("utf-8")).hexdigest()

    def should_suppress(self, packet: RepairEvidencePacket, strategy: str) -> bool:
        evidence_key = self._evidence_key(packet)
        return any(
            item.get("failure_id") == packet.failure_id
            and item.get("strategy") == strategy
            and item.get("evidence_key") == evidence_key
            and item.get("outcome") == "failed"
            for item in self.records
        )

    def record(self, packet: RepairEvidencePacket, strategy: str, outcome: str) -> None:
        self.records.append({"failure_id": packet.failure_id, "strategy": strategy, "outcome": outcome, "evidence_key": self._evidence_key(packet)})
        del self.records[:-100]
