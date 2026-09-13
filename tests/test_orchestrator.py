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
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == "too short\n"
    assert any(event.phase == "DESTRUCTIVE_WRITE_REJECTED" for event in state.events)
    assert any(event.phase == "DESTRUCTIVE_WRITE_RECOVERY_SUCCESS" for event in state.events)
    assert not any(event.agent == "ARCHITECT" for event in state.events)
    assert not any(event.agent == "MANAGER" for event in state.events)


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


def test_provider_request_terminal_outcomes_balance_attempts() -> None:
    state = ProjectState.create("Build Notes")
    state.event_counters.update({
        "PROVIDER_REQUEST:ATTEMPT": 3,
        "PROVIDER_REQUEST:SUCCESS": 2,
        "PROVIDER_REQUEST:TIMEOUT": 1,
    })

    data = metrics(state)

    assert data["llm_requests_attempted"] == 3
    assert data["llm_responses_completed"] == 2
    assert data["provider_request_outcomes"] == {"SUCCESS": 2, "TIMEOUT": 1}
    assert data["provider_request_outcome_gap"] == 0


def test_attempt_rollback_metrics_distinguish_acceptance_from_regression() -> None:
    state = ProjectState.create("Build Notes")
    first = Task.create("First", "work")
    first.rollback_count = 2
    first.rollback_reasons = {"acceptance_failure": 2}
    second = Task.create("Second", "work")
    second.rollback_count = 1
    second.rollback_reasons = {"regression_failure": 1}
    state.tasks = [first, second]

    data = metrics(state)

    assert data["attempt_rollbacks_total"] == 3
    assert data["attempt_rollbacks_by_reason"] == {"acceptance_failure": 2, "regression_failure": 1}
    assert data["regression_rollbacks"] == 1


def test_progress_metric_requires_confirmed_meaningful_progress_event() -> None:
    state = ProjectState.create("Build Notes")
    state.run_history.extend([
        "Coder completed attempt 1 for API",
        "Returning task to Coder: API; acceptance failed",
    ])

    assert metrics(state)["tasks_progressed_after_attempt"] == 0

    state.record_event("CONTROLLER", "MEANINGFUL_PROGRESS", "Task accepted and checkpointed")

    assert metrics(state)["tasks_progressed_after_attempt"] == 1


def test_environment_repair_metrics_use_terminal_outcomes_and_report_open_attempts() -> None:
    state = ProjectState.create("Build Notes")
    state.event_counters.update({
        "ENVIRONMENT:REPAIR_ATTEMPT": 3,
        "ENVIRONMENT:REPAIR_SUCCEEDED": 1,
        "ENVIRONMENT:REPAIR_FAILED": 1,
    })

    data = metrics(state)

    assert data["environment_repair_successes"] == 1
    assert data["environment_repair_failures"] == 1
    assert data["environment_repair_open"] == 1


def test_provider_latency_metrics_are_averaged_by_completed_request_role() -> None:
    state = ProjectState.create("Build Notes")
    state.provider_request_stats = {
        "latency_ms_total_by_role": {"CODER": 4500.0},
        "max_latency_ms_by_role": {"CODER": 3000.0},
        "terminal_requests_by_role": {"CODER": 2},
    }

    data = metrics(state)

    assert data["average_llm_latency_seconds_by_role"] == {"CODER": 2.25}
    assert data["max_llm_latency_seconds_by_role"] == {"CODER": 3.0}


def test_model_timeout_metric_counts_provider_timeouts_not_runtime_readiness_messages() -> None:
    state = ProjectState.create("Build Notes")
    state.run_history.append("Application readiness timed out. Logs: server failed")

    assert metrics(state)["model_timeouts"] == 0

    state.event_counters["PROVIDER_REQUEST:TIMEOUT"] = 2

    assert metrics(state)["model_timeouts"] == 2


def test_validation_metrics_keep_no_tests_and_harness_failures_separate() -> None:
    state = ProjectState.create("Build Notes")
    state.event_counters.update({
        "TESTER:VALIDATION_PASS": 3,
        "TESTER:VALIDATION_APPLICATION_FAIL": 2,
        "TESTER:VALIDATION_NO_TESTS": 1,
        "TESTER:VALIDATION_COMMAND_INVALID": 1,
        "TESTER:VALIDATION_IMPORT_OR_ENVIRONMENT_ERROR": 2,
        "TESTER:VALIDATION_TIMEOUT": 1,
        "TESTER:HARNESS_FAILURE": 1,
    })

    data = metrics(state)

    assert data["validation_passes"] == 3
    assert data["validation_application_failures"] == 2
    assert data["validation_no_tests"] == 1
    assert data["validation_command_invalid"] == 1
    assert data["validation_environment_failures"] == 2
    assert data["validation_timeouts"] == 1
    assert data["tester_harness_failures"] == 1


def test_regression_rejection_restores_every_project_file_to_pre_attempt_state(tmp_path: Path) -> None:
    setup_repository(tmp_path)
    (tmp_path / "app.py").write_bytes(b"good app\r\n")
    (tmp_path / "keep.txt").write_bytes(b"keep me\r\n")
    git(tmp_path, "add", "app.py", "keep.txt")
    git(tmp_path, "commit", "-m", "good baseline")
    original_app = (tmp_path / "app.py").read_bytes()
    original_keep = (tmp_path / "keep.txt").read_bytes()
    provider = ScriptedProvider({
        "CODER": [AgentReply({"actions": [
            {"kind": "write_file", "path": "app.py", "content": "rejected app\n"},
            {"kind": "write_file", "path": "created.txt", "content": "new file\n"},
            {"kind": "delete_file", "path": "keep.txt"},
        ]})],
        "TESTER": [AgentReply({"command": ["py", "-3", "-c", "print('current task passes')"]})],
    })
    runner = AutonomousRunner(tmp_path, StateStore(tmp_path), WorkspaceTools(tmp_path), provider)
    state = runner.initialize("Build Notes")
    task = Task.create("Risky change", "Change one capability")
    state.tasks.append(task)
    state.accepted_regressions.append({"key": "protected", "task": "Accepted capability", "command": ["py", "-3", "-c", "raise SystemExit(1)"]})

    runner._run_task(state, task)

    assert (tmp_path / "app.py").read_bytes() == original_app
    assert (tmp_path / "keep.txt").read_bytes() == original_keep
    assert not (tmp_path / "created.txt").exists()
    assert any(event.agent == "REGRESSION" and event.phase == "FAIL" for event in state.events)


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
    # A repairable validation failure is repaired locally with its exact
    # evidence; it no longer consumes a second root-task attempt.
    assert state.tasks[0].attempts == 1
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
    assert state.tasks[0].rollback_reasons["repeated_coder_action"] == 1
    assert "other" not in state.tasks[0].rollback_reasons


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
