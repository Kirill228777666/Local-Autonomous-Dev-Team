from autodev.capabilities import repair_generated_test_for_contract, reviewer_scope_violations
from autodev.repair import FailureClass, RepairEvidencePacket, RepairMemory, route_failure
from autodev.research import ResearchResult, WebResearch
from autodev.tools import CommandResult
from autodev.validation import ValidationOutcome


def test_contract_conflict_repairs_generated_factory_test_without_mutating_contract() -> None:
    contract = {"version": 1, "architecture": {"app_factory": False, "application_entry_point": "app.py", "application_symbol": "app"}}
    original = "from app import create_app\nclient = create_app().test_client()\n"

    repaired = repair_generated_test_for_contract(contract, original)

    assert repaired == "from app import app\nclient = app.test_client()\n"
    assert contract["architecture"]["app_factory"] is False


def test_reviewer_scope_filter_rejects_future_frontend_requirement_for_database_capability() -> None:
    violations = reviewer_scope_violations(
        "Database schema", "Implement SQLite schema and migrations.",
        {"architecture": {"application_entry_point": "app.py"}},
        ["Frontend assets are missing", "README is missing", "Schema does not create required table"],
    )

    assert violations == ["Frontend assets are missing", "README is missing"]


def test_dependency_api_mismatch_creates_compact_researchable_evidence_packet() -> None:
    packet = RepairEvidencePacket.from_validation(
        capability_id="database.persistence",
        contract_version=1,
        attempt_number=1,
        command=["python", "-m", "unittest"],
        outcome=ValidationOutcome.DEPENDENCY_API_MISMATCH,
        result=CommandResult(1, "", "AttributeError: 'Engine' object has no attribute 'has_table'"),
        contract={"architecture": {"database": "SQLite"}},
        dependency_versions={"SQLAlchemy": "2.0.43"},
    )

    assert packet.failure_class is FailureClass.DEPENDENCY_API_MISMATCH
    assert "has_table" in packet.exception_message
    assert packet.research_query() == "SQLAlchemy 2.0.43 Engine has_table AttributeError"
    assert "contract" not in packet.research_query().lower()


def test_repair_memory_suppresses_identical_strategy_but_allows_new_evidence() -> None:
    memory = RepairMemory()
    packet = RepairEvidencePacket("f1", "domain.x", 1, 1, [], "validator", FailureClass.APPLICATION_LOGIC_FAILURE)
    memory.record(packet, "focused-coder", "failed")

    assert memory.should_suppress(packet, "focused-coder") is True
    packet.research_evidence.append({"url": "https://example.test/docs", "text": "new API"})
    assert memory.should_suppress(packet, "focused-coder") is False


def test_repair_memory_only_suppresses_after_an_actual_repair_attempt() -> None:
    packet = RepairEvidencePacket("f1", "domain.x", 1, 1, [], "validator", FailureClass.APPLICATION_LOGIC_FAILURE)
    memory = RepairMemory()

    # Capturing evidence is not itself a failed Coder repair. The first focused
    # retry must receive the packet instead of being suppressed.
    assert memory.should_suppress(packet, "focused-coder") is False
    memory.record(packet, "focused-coder", "failed")
    assert memory.should_suppress(packet, "focused-coder") is True


def test_web_research_caches_official_source_and_never_sends_source_text() -> None:
    calls: list[str] = []

    def fetch(url: str) -> str:
        calls.append(url)
        return "<html><title>SQLAlchemy docs</title><main>Inspector has_table was removed.</main></html>"

    research = WebResearch(fetch=fetch)
    first = research.official_docs_search("SQLAlchemy", "Engine.has_table", "2.0.43", "AttributeError")
    second = research.official_docs_search("SQLAlchemy", "Engine.has_table", "2.0.43", "AttributeError")

    assert first.official is True
    assert second.cache_hit is True
    assert len(calls) == 1
    assert "source code" not in first.query.lower()


def test_research_failure_is_contained() -> None:
    research = WebResearch(fetch=lambda _url: (_ for _ in ()).throw(TimeoutError("timeout")))

    result = research.official_docs_search("SQLAlchemy", "Engine.has_table", "2.0", "AttributeError")

    assert result.ok is False
    assert result.error == "timeout"


def test_controller_records_research_cache_hit_for_repeated_dependency_evidence(tmp_path) -> None:
    from autodev.models import ProjectState, Task
    from autodev.orchestrator import AutonomousRunner
    from autodev.providers import ScriptedProvider
    from autodev.state_store import StateStore
    from autodev.tools import CommandResult, WorkspaceTools

    fetches: list[str] = []
    research = WebResearch(fetch=lambda url: fetches.append(url) or "<main>official API</main>")
    runner = AutonomousRunner(tmp_path, StateStore(tmp_path), WorkspaceTools(tmp_path), ScriptedProvider({}), research=research)
    state = ProjectState.create("Build database")
    task = Task.create("Database", "Database", capability_id="database.persistence")
    state.environment = {"execution_context": {"python_interpreter": "missing"}}
    runner._dependency_versions = lambda *_args: {"SQLAlchemy": "2.0.43"}  # type: ignore[method-assign]
    result = CommandResult(1, "", "AttributeError: 'Engine' object has no attribute 'has_table'")

    runner._capture_repair_evidence(state, task, ["python", "-m", "pytest"], ValidationOutcome.DEPENDENCY_API_MISMATCH, result)
    runner._capture_repair_evidence(state, task, ["python", "-m", "pytest"], ValidationOutcome.DEPENDENCY_API_MISMATCH, result)

    assert len(fetches) == 1
    assert any(event.agent == "RESEARCH" and event.phase == "CACHE_HIT" for event in state.events)


def test_failure_router_keeps_application_http_500_with_expected_404_out_of_environment() -> None:
    outcome = route_failure(ValidationOutcome.APPLICATION_FAILURE, "AssertionError: 500 != 404")

    assert outcome is FailureClass.APPLICATION_LOGIC_FAILURE


def test_repair_packet_state_survives_roundtrip() -> None:
    from autodev.models import ProjectState, Task

    state = ProjectState.create("Build application")
    task = Task.create("API", "Implement API")
    task.last_repair_packet = {"failure_id": "abc", "failure_class": "APPLICATION_LOGIC_FAILURE"}
    state.repair_memory = [{"failure_id": "abc", "strategy": "focused-coder", "outcome": "failed"}]
    state.research_cache = {"query": {"url": "https://docs.example", "text": "evidence"}}
    state.selected_primary_model = "qwen3-coder:30b"
    state.tasks = [task]

    restored = ProjectState.from_dict(state.to_dict())

    assert restored.tasks[0].last_repair_packet["failure_id"] == "abc"
    assert restored.repair_memory == state.repair_memory
    assert restored.research_cache == state.research_cache
    assert restored.selected_primary_model == "qwen3-coder:30b"


def test_runner_resolves_generated_test_contract_conflict_in_same_action(tmp_path) -> None:
    from autodev.models import ProjectState, Task
    from autodev.orchestrator import AutonomousRunner
    from autodev.providers import ScriptedProvider
    from autodev.state_store import StateStore
    from autodev.tools import WorkspaceTools

    runner = AutonomousRunner(tmp_path, StateStore(tmp_path), WorkspaceTools(tmp_path), ScriptedProvider({}))
    state = ProjectState.create("Build app")
    state.project_contract = {"version": 1, "architecture": {"app_factory": False, "application_entry_point": "app.py", "application_symbol": "app"}}
    task = Task.create("Tests", "Write test")

    runner._execute_coder_actions(state, task, [{"kind": "write_file", "path": "tests/test_app.py", "content": "from app import create_app\nclient = create_app().test_client()\n"}])

    assert (tmp_path / "tests/test_app.py").read_text(encoding="utf-8") == "from app import app\nclient = app.test_client()\n"
    assert any(event.phase == "CONTRACT_CONFLICT_RESOLVED" for event in state.events)


def test_repair_evidence_roundtrip_preserves_external_knowledge() -> None:
    packet = RepairEvidencePacket(
        "f1", "database.persistence", 1, 2, ["python", "-m", "pytest"], "test",
        FailureClass.DEPENDENCY_API_MISMATCH,
        dependency_versions={"SQLAlchemy": "2.0.43"},
        research_evidence=[{"url": "https://docs.sqlalchemy.org/en/20/", "text": "inspection API"}],
    )

    restored = RepairEvidencePacket.from_dict(packet.to_dict())

    assert restored.failure_class is FailureClass.DEPENDENCY_API_MISMATCH
    assert restored.dependency_versions == {"SQLAlchemy": "2.0.43"}
    assert restored.research_evidence[0]["url"].startswith("https://docs.sqlalchemy.org")
