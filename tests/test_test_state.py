from pathlib import Path

from autodev.test_state import ValidationStateManager
from autodev.models import ProjectState, Task
from autodev.orchestrator import AutonomousRunner
from autodev.providers import ScriptedProvider
from autodev.state_store import StateStore
from autodev.tools import WorkspaceTools


def test_test_state_uses_unique_owned_database_and_never_deletes_development_database(tmp_path: Path) -> None:
    development = tmp_path / "backend" / "notes.db"
    development.parent.mkdir()
    development.write_text("user data", encoding="utf-8")
    manager = ValidationStateManager(tmp_path)

    first = manager.for_validator("notes.api", ["python", "-m", "unittest"])
    second = manager.for_validator("notes.api", ["python", "-m", "unittest"])
    Path(first.environment["AUTODEV_TEST_DATABASE"]).write_text("test data", encoding="utf-8")
    manager.cleanup(first)

    assert first.environment["AUTODEV_TESTING"] == "1"
    assert first.path != second.path
    assert not first.path.exists()
    assert development.read_text(encoding="utf-8") == "user data"


def test_runner_executes_each_unittest_validator_with_fresh_owned_database(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_state.py").write_text(
        "import os, sqlite3, unittest\n"
        "class StateTest(unittest.TestCase):\n"
        " def test_empty(self):\n"
        "  path=os.environ['AUTODEV_TEST_DATABASE']; db=sqlite3.connect(path); db.execute('create table if not exists rows(v int)'); self.assertEqual(db.execute('select count(*) from rows').fetchone()[0], 0); db.execute('insert into rows values(1)'); db.commit(); db.close()\n",
        encoding="utf-8",
    )
    runner = AutonomousRunner(tmp_path, StateStore(tmp_path), WorkspaceTools(tmp_path), ScriptedProvider({}))
    state = ProjectState.create("Build app")
    task = Task.create("Backend tests", "Run unit tests", capability_id="generic.tests")
    command = [__import__("sys").executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test*.py"]

    first = runner._run_validator(state, task, command)
    second = runner._run_validator(state, task, command)

    assert first.exit_code == second.exit_code == 0
    assert not list((tmp_path / ".autodev" / "test-state").rglob("*.sqlite"))
