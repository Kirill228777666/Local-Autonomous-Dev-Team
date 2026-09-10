from pathlib import Path

import pytest

from autodev.tools import CommandResult, ToolPolicyError, WorkspaceTools


def test_workspace_tools_read_and_write_only_inside_workspace(tmp_path: Path) -> None:
    tools = WorkspaceTools(tmp_path)

    tools.write_file("src/app.py", "print('hello')\n")

    assert tools.read_file("src/app.py") == "print('hello')\n"
    assert tools.list_files() == ["src/app.py"]


def test_workspace_tools_reject_path_escape(tmp_path: Path) -> None:
    tools = WorkspaceTools(tmp_path)

    with pytest.raises(ToolPolicyError, match="workspace"):
        tools.write_file("../outside.txt", "no")


def test_workspace_tools_deletes_only_a_workspace_file(tmp_path: Path) -> None:
    tools = WorkspaceTools(tmp_path)
    tools.write_file("generated.txt", "temporary")

    tools.delete_file("generated.txt")

    assert not (tmp_path / "generated.txt").exists()


def test_workspace_tools_runs_non_shell_command_and_captures_output(tmp_path: Path) -> None:
    result = WorkspaceTools(tmp_path).run_command(["py", "-3", "-c", "print('checked')"])

    assert result == CommandResult(exit_code=0, stdout="checked\n", stderr="")


def test_workspace_tools_rejects_destructive_command(tmp_path: Path) -> None:
    with pytest.raises(ToolPolicyError, match="not permitted"):
        WorkspaceTools(tmp_path).run_command(["Remove-Item", "-Recurse", "src"])


def test_workspace_tools_rejects_shell_wrappers_that_can_bypass_policy(tmp_path: Path) -> None:
    with pytest.raises(ToolPolicyError, match="not permitted"):
        WorkspaceTools(tmp_path).run_command(["cmd", "/c", "del important.txt"])
