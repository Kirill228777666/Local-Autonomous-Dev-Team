from pathlib import Path

from autodev.capabilities import CapabilityGraph, build_project_contract, normalize_plan
from autodev.models import ProjectState, Task, TaskStatus
from autodev.orchestrator import AutonomousRunner
from autodev.providers import ScriptedProvider
from autodev.state_store import StateStore
from autodev.tools import WorkspaceTools


def test_contract_is_durable_and_static_frontend_rejects_node_review_drift() -> None:
    contract = build_project_contract(
        "Create a local Notes application with backend, frontend and SQLite.",
        {"npm": {"available": False}},
    )

    assert contract["version"] == 1
    assert contract["architecture"]["frontend_strategy"] == "static-html-css-js"
    assert "package.json" in contract["forbidden_requirements"]


def test_normalizer_merges_equivalent_tasks_with_stable_capability_identity() -> None:
    state = ProjectState.create("Build a generic application")
    first = Task.create("Implement create and update", "Implement domain CRUD behavior.")
    second = Task.create("Реализовать создание и редактирование", "Same capability in another language.")
    state.tasks = [first, second]

    graph = normalize_plan(state)

    assert graph["domain.crud"]["task_id"] == first.id
    assert second.status is TaskStatus.SUPERSEDED
    assert second.root_task_id == first.root_task_id


def test_normalizer_merges_bilingual_category_capability_without_product_name() -> None:
    state = ProjectState.create("Build a generic application")
    state.tasks = [
        Task.create("Add category filtering", "Implement category assignment and filters."),
        Task.create("Добавить категории", "Реализовать фильтрацию по категориям."),
    ]

    graph = normalize_plan(state)

    assert list(graph) == ["domain.categories"]
    assert sum(task.status is TaskStatus.SUPERSEDED for task in state.tasks) == 1


def test_normalizer_is_idempotent_and_does_not_append_new_roots_on_resume() -> None:
    state = ProjectState.create("Build a generic application")
    state.tasks = [Task.create("Persistence", "Create persistent storage.")]

    normalize_plan(state)
    once = [(task.id, task.capability_id, task.status) for task in state.tasks]
    normalize_plan(state)

    assert [(task.id, task.capability_id, task.status) for task in state.tasks] == once


def test_regression_guard_runs_prior_accepted_validator_even_when_command_matches_current(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    store = StateStore(tmp_path)
    runner = AutonomousRunner(tmp_path, store, WorkspaceTools(tmp_path), ScriptedProvider({}))
    state = ProjectState.create("Build application")
    task = Task.create("Second capability", "Change behavior")
    state.tasks = [task]
    command = ["python", "-c", "raise SystemExit(1)"]
    state.accepted_regressions.append({"key": "domain.first", "capability_id": "domain.first", "command": command, "task": "First capability"})

    assert runner._run_accepted_regressions(state, task, command) is False
    assert any(event.agent == "REGRESSION" and event.phase == "CHECK" for event in state.events)


def test_context_contains_authoritative_contract_and_protected_capabilities() -> None:
    from autodev.context import ContextBuilder

    state = ProjectState.create("Build application")
    state.project_contract = {"version": 1, "architecture": {"app_factory": False}}
    state.capability_graph = {"domain.crud": {"status": "DONE", "acceptance": ["DELETE returns 204"]}}
    task = Task.create("Categories", "Implement categories")
    task.capability_id = "domain.categories"

    packet = ContextBuilder().for_task(state, task, [])

    assert "Project contract" in packet
    assert "Protected capabilities" in packet
    assert "domain.crud" in packet


def test_static_contract_marks_npm_reviewer_objection_as_conflicting() -> None:
    from autodev.capabilities import contract_conflicting_review_reasons

    contract = build_project_contract("Build a Python backend and frontend", {"npm": {"available": False}})

    assert contract_conflicting_review_reasons(contract, ["package.json and npm build are required"]) == [
        "package.json and npm build are required"
    ]


def test_module_level_app_contract_rejects_generated_factory_test() -> None:
    from autodev.capabilities import contract_conflicting_test_action

    conflict = contract_conflicting_test_action(
        {"architecture": {"app_factory": False}},
        {"kind": "write_file", "path": "tests/test_api.py", "content": "from app import create_app\n"},
    )

    assert conflict is not None and "TEST_CONTRACT_CONFLICT" in conflict


def test_initial_plan_is_normalized_without_notes_specific_atom_expansion(tmp_path: Path) -> None:
    from autodev.providers import AgentReply, ScriptedProvider

    (tmp_path / ".git").mkdir()
    provider = ScriptedProvider({
        "MANAGER": [AgentReply({"tasks": [{"title": "Implement create, update and delete", "description": "Implement CRUD only."}]})],
    })
    runner = AutonomousRunner(tmp_path, StateStore(tmp_path), WorkspaceTools(tmp_path), provider)
    runner.initialize("Build a generic Python backend")

    state = runner.run(max_cycles=0)

    assert len(state.tasks) == 1
    assert state.tasks[0].capability_id == "domain.crud"
    assert state.project_contract["mode"] == "reliable"
