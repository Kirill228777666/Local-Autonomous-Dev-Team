from pathlib import Path

from autodev.models import ProjectState, Task, TaskStatus
from autodev.orchestrator import AutonomousRunner
from autodev.providers import AgentReply, AgentRequest, ProviderUnavailableError
from autodev.state_store import StateStore
from autodev.tools import WorkspaceTools


class CoderOutageProvider:
    """Manager is reachable; the next generation fails like a stopped Ollama."""

    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        self.online = False

    def health(self) -> bool:
        return self.online

    def complete(self, request: AgentRequest) -> AgentReply:
        if request.role == "MANAGER":
            return AgentReply({"next_task_id": self.task_id})
        raise ProviderUnavailableError("Ollama endpoint unavailable: [WinError 10061] connection refused")


class RecoveringHealthProvider(CoderOutageProvider):
    def __init__(self, task_id: str) -> None:
        super().__init__(task_id)
        self.probes = 0

    def health(self) -> bool:
        self.probes += 1
        return self.probes >= 2


def test_provider_outage_preserves_task_attempt_and_does_not_create_repairs(tmp_path: Path) -> None:
    task = Task.create("Atomic capability", "Create one file")
    state = ProjectState.create("Build a project")
    state.tasks = [task]
    store = StateStore(tmp_path)
    store.save(state)
    provider = CoderOutageProvider(task.id)
    runner = AutonomousRunner(
        tmp_path,
        store,
        WorkspaceTools(tmp_path),
        provider,
        provider_wait_seconds=0,
    )

    result = runner.run(max_cycles=1)

    assert result.status == "BLOCKED_PROVIDER"
    assert result.terminal_status == "BLOCKED_PROVIDER"
    assert sum(event.agent == "SYSTEM" and event.phase == "BLOCKED_PROVIDER" for event in result.events) == 1
    assert result.tasks == [task]
    assert task.status is TaskStatus.PENDING
    assert task.attempts == 0
    assert result.provider_state["circuit"] == "OPEN"
    assert not any("Created Architect-guided corrective task" in item for item in result.run_history)


def test_provider_circuit_closes_after_health_recovers_without_mutating_task(tmp_path: Path) -> None:
    task = Task.create("Atomic capability", "Create one file")
    state = ProjectState.create("Build a project")
    state.status = "WAITING_FOR_MODEL_PROVIDER"
    state.tasks = [task]
    store = StateStore(tmp_path)
    store.save(state)
    provider = CoderOutageProvider(task.id)
    provider.online = True
    runner = AutonomousRunner(tmp_path, store, WorkspaceTools(tmp_path), provider, provider_wait_seconds=0)

    result = runner.run(max_cycles=0)

    assert result.status == "RUNNING"
    assert task.status is TaskStatus.PENDING
    assert task.attempts == 0
    assert result.provider_state["circuit"] == "CLOSED"


def test_provider_wait_uses_only_health_probes_until_recovery(tmp_path: Path) -> None:
    task = Task.create("Atomic capability", "Create one file")
    state = ProjectState.create("Build a project")
    state.status = "WAITING_FOR_MODEL_PROVIDER"
    state.tasks = [task]
    store = StateStore(tmp_path)
    store.save(state)
    provider = RecoveringHealthProvider(task.id)
    runner = AutonomousRunner(
        tmp_path, store, WorkspaceTools(tmp_path), provider,
        provider_wait_seconds=1, provider_retry_interval=0.01,
    )

    result = runner.run(max_cycles=0)

    assert result.status == "RUNNING"
    assert provider.probes == 2
    assert result.event_counters["PROVIDER:HEALTH_CHECK_FAILED"] == 1
    assert task.attempts == 0
