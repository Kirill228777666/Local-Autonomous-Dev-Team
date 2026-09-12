import subprocess
from pathlib import Path

from autodev.models import ProjectState, Task, TaskStatus
from autodev.metrics import metrics
from autodev.orchestrator import AutonomousRunner
from autodev.providers import AgentReply, ScriptedProvider
from autodev.state_store import StateStore
from autodev.tools import WorkspaceTools


def git(path: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=path, check=True, capture_output=True, text=True)


def setup_repository(path: Path) -> None:
    git(path, "init")
    git(path, "config", "user.email", "test@example.com")
    git(path, "config", "user.name", "Test User")
    (path / ".gitignore").write_text(".autodev/\n", encoding="utf-8")
    git(path, "add", ".gitignore")
    git(path, "commit", "-m", "initial")


def test_regression_guard_rejects_later_change_when_accepted_capability_fails(tmp_path: Path) -> None:
    setup_repository(tmp_path)
    runner = AutonomousRunner(tmp_path, StateStore(tmp_path), WorkspaceTools(tmp_path), ScriptedProvider({}))
    state = runner.initialize("Build Notes")
    task = Task.create("Later capability", "Change a later endpoint")
    state.tasks.append(task)
    state.accepted_regressions.append({"key": "accepted", "task": "Create notes", "command": ["py", "-3", "-c", "raise SystemExit(1)"]})

    assert runner._run_accepted_regressions(state, task, ["py", "-3", "-c", "print('current')"]) is False
    assert any(event.agent == "REGRESSION" and event.phase == "FAIL" for event in state.events)


def test_python_notes_run_materializes_and_persists_project_interpreter_before_planning(tmp_path: Path) -> None:
    setup_repository(tmp_path)
    runner = AutonomousRunner(tmp_path, StateStore(tmp_path), WorkspaceTools(tmp_path), ScriptedProvider({}))
    runner.initialize("Create a local Notes backend with SQLite persistence")

    state = runner.run(max_cycles=0)
    context = state.environment["execution_context"]

    assert Path(context["python_interpreter"]).is_file()
    assert Path(context["python_interpreter"]).is_absolute()


def test_destructive_write_is_recovered_inside_same_coder_attempt(tmp_path: Path) -> None:
    setup_repository(tmp_path)
    old = "# preserved capability\n" * 80
    (tmp_path / "app.py").write_text(old, encoding="utf-8")
    provider = ScriptedProvider({
        "CODER": [
            AgentReply({"actions": [
                {"kind": "write_file", "path": "created.txt", "content": "kept"},
                {"kind": "write_file", "path": "app.py", "content": "too short\n"},
            ]}),
            AgentReply({"actions": [{"kind": "edit_file", "path": "app.py", "old": "# preserved capability\n", "new": "# updated capability\n"}]}),
        ],
        "TESTER": [AgentReply({"command": ["py", "-3", "-c", "print('ok')"]})],
        "REVIEWER": [AgentReply({"approved": True, "reasons": []})],
    })
    runner = AutonomousRunner(tmp_path, StateStore(tmp_path), WorkspaceTools(tmp_path), provider)
    state = runner.initialize("Build Notes")
    task = Task.create("Update capability", "Make one safe update")
    state.tasks.append(task)

    runner._run_task(state, task)

    assert task.attempts == 1
    assert task.status is TaskStatus.DONE
    assert (tmp_path / "created.txt").read_text(encoding="utf-8") == "kept"
    assert "# updated capability" in (tmp_path / "app.py").read_text(encoding="utf-8")
    assert any(event.phase == "DESTRUCTIVE_WRITE_REJECTED" for event in state.events)
    assert not any(event.agent == "ARCHITECT" for event in state.events)


def test_tester_executes_regenerated_harness_without_new_coder_attempt(tmp_path: Path) -> None:
    setup_repository(tmp_path)
    provider = ScriptedProvider({
        "CODER": [AgentReply({"actions": []})],
        "TESTER": [
            AgentReply({"command": ["py", "-3", "-c", "if True print('x')"]}),
            AgentReply({"command": ["py", "-3", "-c", "print('ok')"]}),
        ],
        "REVIEWER": [AgentReply({"approved": True, "reasons": []})],
    })
    runner = AutonomousRunner(tmp_path, StateStore(tmp_path), WorkspaceTools(tmp_path), provider)
    state = runner.initialize("Build Notes")
    task = Task.create("Validate backend", "Run a deterministic check")
    state.tasks.append(task)

    runner._run_task(state, task)

    assert task.attempts == 1
    assert task.status is TaskStatus.DONE
    assert any(event.phase == "HARNESS_EXECUTE" for event in state.events)
    assert any(event.phase == "HARNESS_RECOVERED" for event in state.events)


def test_llm_completed_metric_never_exceeds_dispatched_requests() -> None:
    state = ProjectState.create("Build Notes")
    state.event_counters["LLM:REQUEST"] = 1
    state.event_counters["LLM:RESPONSE"] = 2

    assert metrics(state)["llm_responses_completed"] <= metrics(state)["llm_requests_attempted"]


def test_runner_completes_multiple_tasks_and_repairs_a_failed_test(tmp_path: Path) -> None:
    setup_repository(tmp_path)
    provider = ScriptedProvider(
        {
            "MANAGER": [
                AgentReply(
                    {
                        "tasks": [
                            {"title": "Write artifact", "description": "Create artifact.txt"},
                            {"title": "Write second artifact", "description": "Create second.txt"},
                        ]
                    }
                ),
                AgentReply({}),
                AgentReply({}),
            ],
            "CODER": [
                AgentReply({"actions": [{"kind": "write_file", "path": "artifact.txt", "content": "bad"}]}),
                AgentReply({"actions": [{"kind": "write_file", "path": "artifact.txt", "content": "good"}]}),
                AgentReply({"actions": [{"kind": "write_file", "path": "second.txt", "content": "done"}]}),
            ],
            "TESTER": [
                AgentReply(
                    {
                        "command": [
                            "py", "-3", "-c",
                            "from pathlib import Path; raise SystemExit(Path('artifact.txt').read_text() != 'good')",
                        ]
                    }
                ),
                AgentReply(
                    {
                        "command": [
                            "py", "-3", "-c",
                            "from pathlib import Path; raise SystemExit(Path('artifact.txt').read_text() != 'good')",
                        ]
                    }
                ),
                AgentReply({"command": ["py", "-3", "-c", "from pathlib import Path; assert Path('second.txt').read_text() == 'done'"]}),
            ],
            "REVIEWER": [AgentReply({"approved": True}), AgentReply({"approved": True})],
            "FINAL_QA": [AgentReply({"status": "PASS", "findings": []})],
        }
    )
    runner = AutonomousRunner(
        workspace=tmp_path,
        store=StateStore(tmp_path),
        tools=WorkspaceTools(tmp_path),
        provider=provider,
    )
    runner.initialize("Build a small project")

    state = runner.run(max_cycles=8)

    assert state.status == "COMPLETE"
    assert [task.status for task in state.tasks] == [TaskStatus.DONE, TaskStatus.DONE]
    assert state.tasks[0].attempts == 2
    assert (tmp_path / "artifact.txt").read_text(encoding="utf-8") == "good"
    assert (tmp_path / "second.txt").read_text(encoding="utf-8") == "done"
    assert StateStore(tmp_path).load() is not None
    assert state.last_checkpoint is not None


def test_runner_blocks_repeated_identical_coder_actions(tmp_path: Path) -> None:
    provider = ScriptedProvider(
        {
            "MANAGER": [
                AgentReply({"tasks": [{"title": "Loop", "description": "Do something"}]}),
                AgentReply({}),
            ],
            "CODER": [
                AgentReply({"actions": [{"kind": "write_file", "path": "loop.txt", "content": "same"}]}),
                AgentReply({"actions": [{"kind": "write_file", "path": "loop.txt", "content": "same"}]}),
                AgentReply({"actions": [{"kind": "write_file", "path": "loop.txt", "content": "same"}]}),
            ],
            "TESTER": [
                AgentReply({"command": ["py", "-3", "-c", "raise SystemExit(1)"]}),
                AgentReply({"command": ["py", "-3", "-c", "raise SystemExit(1)"]}),
            ],
        }
    )
    runner = AutonomousRunner(tmp_path, StateStore(tmp_path), WorkspaceTools(tmp_path), provider)
    runner.initialize("Build a loop test")

    state = runner.run(max_cycles=4)

    assert state.tasks[0].status is TaskStatus.BLOCKED
    assert any("repeated" in error for error in state.tasks[0].errors)


def test_runner_keeps_an_invalid_initial_plan_blocked(tmp_path: Path) -> None:
    runner = AutonomousRunner(
        tmp_path,
        StateStore(tmp_path),
        WorkspaceTools(tmp_path),
        ScriptedProvider({"MANAGER": [AgentReply({"tasks": []})]}),
    )
    runner.initialize("Build a project")

    state = runner.run()

    assert state.status == "BLOCKED"


def test_tester_harness_regeneration_does_not_spend_a_second_coder_attempt(tmp_path: Path) -> None:
    setup_repository(tmp_path)
    provider = ScriptedProvider(
        {
            "MANAGER": [
                AgentReply({"tasks": [{"title": "Write artifact", "description": "Create artifact.txt"}]}),
                AgentReply({}),
            ],
            "CODER": [AgentReply({"actions": [{"kind": "write_file", "path": "artifact.txt", "content": "ok"}]})],
            "TESTER": [
                AgentReply({"command": ["py", "-3", "-c", "import re; with open('artifact.txt'): pass"]}),
                AgentReply({"command": ["py", "-3", "-c", "from pathlib import Path; assert Path('artifact.txt').read_text() == 'ok'"]}),
            ],
            "REVIEWER": [AgentReply({"approved": True})],
        }
    )
    runner = AutonomousRunner(tmp_path, StateStore(tmp_path), WorkspaceTools(tmp_path), provider)
    runner.initialize("Build a small project")

    state = runner.run(max_cycles=1)

    assert state.tasks[0].status is TaskStatus.DONE
    assert state.tasks[0].attempts == 1
    assert sum(event.phase == "HARNESS_FAILURE" for event in state.events) == 1
