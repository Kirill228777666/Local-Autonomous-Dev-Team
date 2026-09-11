from pathlib import Path

import pytest

import autodev.endurance as endurance
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


def test_controlled_crash_passes_absolute_config_to_child_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = prepare_notes_workspace(tmp_path / "notes")
    config = tmp_path / "config.toml"
    config.write_text("[ollama]\nmodel = 'qwen3:14b'\n", encoding="utf-8")
    commands: list[list[str]] = []

    class FinishedProcess:
        pid = 1
        returncode = 1
        def poll(self) -> int: return 1
        def communicate(self, timeout: float) -> tuple[str, str]: return ("failed early", "")
        def kill(self) -> None: pass
        def wait(self, timeout: float) -> int: return 1

    def fake_popen(command: list[str], **_kwargs: object) -> FinishedProcess:
        commands.append(command)
        return FinishedProcess()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(endurance.subprocess, "Popen", fake_popen)
    with pytest.raises(RuntimeError, match="did not reach"):
        endurance.controlled_crash(workspace, Path("config.toml"), max_cycles=1, timeout=0)
    assert str(config.resolve()) in commands[0]
