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


def test_workspace_tools_rejects_server_command_from_finite_tool_path(tmp_path: Path) -> None:
    result = WorkspaceTools(tmp_path).run_command(["py", "-3", "-m", "http.server"])
    assert result.exit_code == 125
    assert "MANAGED_PROCESS" in result.stderr


def test_workspace_tools_hard_timeout_returns_without_waiting_for_child(tmp_path: Path) -> None:
    tools = WorkspaceTools(tmp_path, command_timeout=0.2)
    result = tools.run_command(["py", "-3", "-c", "import time; time.sleep(10)"])
    assert result.exit_code == 124
    assert "hard timed out" in result.stderr


def test_project_venv_rewrites_python_py_and_pip_commands(tmp_path: Path) -> None:
    interpreter = tmp_path / ".venv" / "Scripts" / "python.exe"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("", encoding="utf-8")
    tools = WorkspaceTools(tmp_path)

    assert tools.normalize_command(["python", "-m", "pytest"]) == [str(interpreter), "-m", "pytest"]
    assert tools.normalize_command(["py", "-3", "app.py"]) == [str(interpreter), "app.py"]
    assert tools.normalize_command(["pip", "install", "Flask"]) == [str(interpreter), "-m", "pip", "install", "Flask"]
    assert tools.normalize_command(["pytest", "-q"]) == [str(interpreter), "-m", "pytest", "-q"]
    assert tools.normalize_command(["unittest", "discover"]) == [str(interpreter), "-m", "unittest", "discover"]


def test_workspace_tools_refuses_project_python_commands_until_environment_exists(tmp_path: Path) -> None:
    tools = WorkspaceTools(tmp_path)
    tools.require_project_python = True

    result = tools.run_command(["python", "-c", "print('no leak')"])

    assert result.exit_code == 126
    assert "ENVIRONMENT_NOT_INITIALIZED" in result.stderr


def test_workspace_tools_rejects_large_destructive_whole_file_rewrite(tmp_path: Path) -> None:
    tools = WorkspaceTools(tmp_path)
    tools.write_file("app.py", "@app.route('/api/notes')\n" + ("# preserved capability\n" * 80))

    with pytest.raises(ToolPolicyError, match="DESTRUCTIVE_WRITE"):
        tools.write_file("app.py", "print('replacement')\n")


def test_workspace_tools_rejects_venv_activation_command(tmp_path: Path) -> None:
    result = WorkspaceTools(tmp_path).run_command([".venv\\Scripts\\activate"])

    assert result.exit_code == 126
    assert "PROJECT_ENVIRONMENT_IS_ALREADY_MANAGED" in result.stderr
