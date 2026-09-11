from pathlib import Path

from autodev.endurance import NOTES_SPEC, prepare_notes_workspace
from autodev.state_store import StateStore


def test_notes_endurance_workspace_is_initialized_once_from_the_single_spec(tmp_path: Path) -> None:
    workspace = tmp_path / "notes"

    first = prepare_notes_workspace(workspace)
    second = prepare_notes_workspace(workspace)

    state = StateStore(workspace).load()
    assert first == second == workspace
    assert state is not None and state.original_spec == NOTES_SPEC.strip()
    assert (workspace / ".git").is_dir()
    assert (workspace / ".autodev" / "project_spec.md").is_file()
