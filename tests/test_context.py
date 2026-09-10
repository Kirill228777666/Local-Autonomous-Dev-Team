from autodev.context import ContextBuilder
from autodev.models import ProjectState, Task


def test_context_builder_keeps_spec_task_and_only_tail_of_log() -> None:
    state = ProjectState.create("Build a TODO app")
    task = Task.create("Add storage", "Persist tasks")
    state.run_history = ["old event", "useful event", "latest event"]

    context = ContextBuilder(max_log_entries=2).for_task(state, task, ["src/store.py"])

    assert "Build a TODO app" in context
    assert "Add storage" in context
    assert "src/store.py" in context
    assert "old event" not in context
    assert "useful event" in context
