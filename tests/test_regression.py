import json
from pathlib import Path

from autodev.regression import RegressionRunner
from autodev.tools import WorkspaceTools


def test_regression_detects_python_and_declared_node_checks(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "package.json").write_text(
        json.dumps({"scripts": {"test": "vitest run", "build": "vite build", "lint": "eslint ."}}),
        encoding="utf-8",
    )

    commands = RegressionRunner(tmp_path, WorkspaceTools(tmp_path)).detect_commands()

    assert ["py", "-3", "-m", "pytest", "-q"] in commands
    assert ["npm", "test", "--", "--runInBand"] not in commands
    assert ["npm", "test"] in commands
    assert ["npm", "run", "build"] in commands
    assert ["npm", "run", "lint"] in commands


def test_regression_runs_detected_python_suite_and_records_exit_code(tmp_path: Path) -> None:
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_ok.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")

    result = RegressionRunner(tmp_path, WorkspaceTools(tmp_path)).run()

    assert result.passed is True
    assert result.results[0].exit_code == 0
    assert "passed" in result.summary
