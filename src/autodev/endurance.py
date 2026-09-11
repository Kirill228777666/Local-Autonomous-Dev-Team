"""Reproducible Notes endurance harness; live mode intentionally uses Ollama."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from .cli import AppConfig, ensure_workspace, make_runner
from .metrics import metrics
from .regression import RegressionRunner
from .state_store import StateStore
from .tools import WorkspaceTools


NOTES_SPEC = """Создай локальное web-приложение Notes.

Требования:

- backend API;
- frontend;
- SQLite persistence;
- создание заметок;
- редактирование заметок;
- удаление заметок;
- поиск;
- категории;
- фильтрация по категориям;
- избранные заметки;
- created_at;
- updated_at;
- validation;
- error handling;
- responsive UI;
- backend tests;
- frontend build;
- README с инструкцией запуска.
"""


def prepare_notes_workspace(workspace: Path) -> Path:
    """Initialize exactly once; all later work resumes this same workspace."""
    workspace = workspace.resolve()
    ensure_workspace(workspace)
    runner = make_runner(workspace, AppConfig(), scripted=True)
    runner.initialize(NOTES_SPEC)
    return workspace


def controlled_crash(workspace: Path, config_path: Path | None, max_cycles: int, timeout: float = 240.0) -> dict[str, object]:
    """Kill a separate live orchestrator only after it durably enters active work."""
    command = [sys.executable, "-m", "autodev", "start", str(workspace), "--max-cycles", str(max_cycles)]
    if config_path is not None:
        command.extend(["--config", str(config_path.resolve())])
    process = subprocess.Popen(command, cwd=workspace, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    deadline = time.monotonic() + timeout
    reached_active_work = False
    while time.monotonic() < deadline:
        if process.poll() is not None:
            break
        try:
            state = StateStore(workspace).load()
        except (OSError, ValueError, json.JSONDecodeError):
            state = None
        if state is not None and state.status == "RUNNING" and state.current_task_id:
            reached_active_work = True
            break
        time.sleep(0.25)
    if not reached_active_work:
        output = process.communicate(timeout=5)[0] if process.poll() is not None else ""
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
        raise RuntimeError(f"orchestrator did not reach durable active work before crash deadline: {output[-2000:]}")
    process.kill()
    process.wait(timeout=20)
    return {"pid": process.pid, "exit_code": process.returncode, "active_work_observed": reached_active_work}


def independent_notes_validation(workspace: Path) -> dict[str, object]:
    """Non-LLM post-run checks with explicit evidence/limitations."""
    workspace = workspace.resolve()
    regression = RegressionRunner(workspace, WorkspaceTools(workspace)).run()
    readme = workspace / "README.md"
    sqlite_files = [str(path.relative_to(workspace)) for path in workspace.rglob("*.db")] + [
        str(path.relative_to(workspace)) for path in workspace.rglob("*.sqlite")
    ]
    state = StateStore(workspace).load()
    checks = {
        "readme_exists": readme.is_file(),
        "readme_mentions_run": readme.is_file() and any(word in readme.read_text(encoding="utf-8", errors="replace").lower() for word in ("run", "start", "запуск")),
        "sqlite_artifact_present": bool(sqlite_files),
        "regression_passed": regression.passed,
        "final_qa_passed": state is not None and state.final_qa_status == "PASS",
        "project_complete": state is not None and state.status == "COMPLETE",
    }
    report = {"checks": checks, "sqlite_files": sqlite_files, "regression": regression.summary}
    artifact = workspace.parent / "independent_validation.json"
    artifact.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def run_notes_live(config: AppConfig, config_path: Path | None, artifact_root: Path, max_cycles: int = 100) -> tuple[Path, dict[str, object]]:
    """Run one live-Ollama project, crash it, resume the same project, preserve artifacts."""
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    root = (artifact_root / run_id).resolve()
    workspace = prepare_notes_workspace(root / "notes")
    crash = controlled_crash(workspace, config_path, max_cycles)
    runner = make_runner(workspace, config)
    runner.resume()
    state = runner.run(max_cycles=max_cycles)
    validation = independent_notes_validation(workspace)
    summary = {
        "workspace": str(workspace),
        "crash": crash,
        "status": state.status,
        "metrics": metrics(state),
        "validation": validation,
    }
    (root / "endurance_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return workspace, summary
