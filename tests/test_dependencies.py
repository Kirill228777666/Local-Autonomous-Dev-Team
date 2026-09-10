from autodev.models import ProjectState, Task, TaskStatus
from autodev.orchestrator import AutonomousRunner
from autodev.providers import AgentReply, ScriptedProvider
from autodev.state_store import StateStore
from autodev.tools import WorkspaceTools


def test_manager_selection_skips_pending_task_until_dependencies_are_done(tmp_path) -> None:
    first = Task.create("Foundation", "Create foundation")
    second = Task.create("Feature", "Depends on foundation", dependencies=[first.id])
    state = ProjectState.create("Build project")
    state.tasks = [first, second]
    StateStore(tmp_path).save(state)
    runner = AutonomousRunner(tmp_path, StateStore(tmp_path), WorkspaceTools(tmp_path), ScriptedProvider({"MANAGER": [AgentReply({})]}))

    assert runner._select_next_task(state) is first
    first.status = TaskStatus.DONE

    assert runner._select_next_task(state) is second
