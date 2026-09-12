"""Reproducible Notes endurance harness; live mode intentionally uses Ollama."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen

from .cli import AppConfig, VisualConfig, ensure_workspace, make_runner
from .metrics import metrics
from .regression import RegressionRunner
from .runtime import ManagedProcessManager, free_port
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
    checks: dict[str, object] = {
        "readme_exists": readme.is_file(),
        "readme_mentions_run": readme.is_file() and any(word in readme.read_text(encoding="utf-8", errors="replace").lower() for word in ("run", "start", "запуск")),
        "sqlite_artifact_present": bool(sqlite_files),
        "regression_passed": regression.passed,
        "final_qa_passed": state is not None and state.final_qa_status == "PASS",
        "project_complete": state is not None and state.status == "COMPLETE",
    }
    entry = next((path for path in (workspace / "backend" / "app.py", workspace / "app.py") if path.is_file()), None)
    api_error = ""
    if entry is not None:
        manager = ManagedProcessManager(workspace)
        port = free_port()
        command = WorkspaceTools(workspace).normalize_command(["python", str(entry.relative_to(workspace))])
        record = manager.start(command, purpose="independent-validation", expected_port=port, env={"PORT": str(port)}, timeout=30)
        base_url = f"http://127.0.0.1:{port}"
        try:
            checks["backend_starts"] = manager.wait_ready(record.id, "127.0.0.1", port, timeout=30)
            if checks["backend_starts"]:
                def request_json(path: str, method: str = "GET", data: dict[str, object] | None = None) -> tuple[int, object]:
                    body = json.dumps(data).encode("utf-8") if data is not None else None
                    request = Request(base_url + path, data=body, method=method, headers={"Content-Type": "application/json"})
                    with urlopen(request, timeout=8) as response:
                        payload = response.read().decode("utf-8")
                        return response.status, json.loads(payload) if payload else {}

                status, created = request_json("/api/notes", "POST", {"title": "Independent validation", "content": "persisted note", "category": "Work", "favorite": True})
                checks["create_note"] = status in {200, 201} and isinstance(created, dict)
                note_id = created.get("id") if isinstance(created, dict) else None
                checks["created_at_present"] = isinstance(created, dict) and bool(created.get("created_at"))
                checks["updated_at_present"] = isinstance(created, dict) and bool(created.get("updated_at"))
                checks["favorites_work"] = isinstance(created, dict) and created.get("favorite") is True
                if isinstance(note_id, int):
                    status, updated = request_json(f"/api/notes/{note_id}", "PUT", {"title": "Independent validation updated", "content": "persisted note", "category": "Work", "favorite": True})
                    checks["update_note"] = status == 200 and isinstance(updated, dict)
                    checks["updated_at_changes"] = isinstance(updated, dict) and updated.get("updated_at") != created.get("updated_at")
                    status, listing = request_json("/api/notes?search=updated&category=Work")
                    checks["search_and_category_filter"] = status == 200 and isinstance(listing, (list, dict))
                    # Persistence is meaningful only across a real managed restart.
                    manager.stop(record.id)
                    restarted = manager.start(command, purpose="independent-validation-restart", expected_port=port, env={"PORT": str(port)}, timeout=30)
                    checks["persistence_after_restart"] = manager.wait_ready(restarted.id, "127.0.0.1", port, timeout=30)
                    if checks["persistence_after_restart"]:
                        status, persisted = request_json(f"/api/notes/{note_id}")
                        checks["persistence_after_restart"] = status == 200 and isinstance(persisted, dict)
                    status, _ = request_json(f"/api/notes/{note_id}", "DELETE")
                    checks["delete_note"] = status in {200, 204}
                else:
                    checks.update({"update_note": False, "updated_at_changes": False, "search_and_category_filter": False, "delete_note": False})
                try:
                    status, _ = request_json("/api/notes", "POST", {"title": ""})
                    checks["invalid_data_rejected"] = status >= 400
                except Exception:
                    checks["invalid_data_rejected"] = True
        except Exception as error:
            api_error = str(error)
            checks.setdefault("backend_starts", False)
        finally:
            manager.stop_all()
        checks["managed_validation_cleanup"] = all(item["status"] == "STOPPED" for item in manager.records())
    else:
        checks["backend_starts"] = False
        api_error = "No backend app.py entry point was found"
    report = {
        "checks": checks,
        "sqlite_files": sqlite_files,
        "regression": regression.summary,
        "api_validation_error": api_error,
    }
    artifact = workspace.parent / "independent_validation.json"
    artifact.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def preflight_ollama(config: AppConfig) -> dict[str, object]:
    """Verify the configured external provider before creating a live run.

    This is intentionally outside the project metrics and does not use a role
    prompt: it proves endpoint/model/context availability without pretending to
    be application work.
    """
    base_url = config.base_url.rstrip("/")
    with urlopen(f"{base_url}/api/tags", timeout=10) as response:
        tags = json.loads(response.read())
    models = {
        str(item.get("name", ""))
        for item in tags.get("models", [])
        if isinstance(item, dict)
    }
    if config.model not in models:
        raise RuntimeError(f"Configured Ollama model is not installed: {config.model}")
    payload = json.dumps({
        "model": config.model,
        "stream": False,
        "format": "json",
        "keep_alive": "10m",
        "options": {"temperature": 0.05, "num_ctx": 16384},
        "messages": [{"role": "user", "content": "Return only {\"ok\":true}."}],
    }).encode("utf-8")
    request = Request(f"{base_url}/api/chat", data=payload, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=config.timeout) as response:
        envelope = json.loads(response.read())
    if not isinstance(envelope.get("message", {}).get("content"), str):
        raise RuntimeError("Ollama preflight returned no chat content")
    return {"base_url": base_url, "model": config.model, "context_limit": 16384, "endpoint": "ok", "chat": "ok"}


def run_notes_live(config: AppConfig, config_path: Path | None, artifact_root: Path, max_cycles: int = 100) -> tuple[Path, dict[str, object]]:
    """Run one live-Ollama project, crash it, resume the same project, preserve artifacts."""
    preflight = preflight_ollama(config)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    root = (artifact_root / run_id).resolve()
    workspace = prepare_notes_workspace(root / "notes")
    if not config.visual.command:
        config = AppConfig(
            model=config.model,
            base_url=config.base_url,
            timeout=config.timeout,
            max_attempts=config.max_attempts,
            profiles=config.profiles,
            visual=VisualConfig(("python", "backend/app.py"), "http://127.0.0.1:{port}", "", 30, 2),
            permissions=config.permissions,
            provider_max_wait_seconds=config.provider_max_wait_seconds,
        )
    crash = controlled_crash(workspace, config_path, max_cycles)
    runner = make_runner(workspace, config)
    runner.resume()
    state = runner.run(max_cycles=max_cycles)
    validation = independent_notes_validation(workspace)
    summary = {
        "workspace": str(workspace),
        "provider_preflight": preflight,
        "crash": crash,
        "status": state.status,
        "metrics": metrics(state),
        "validation": validation,
    }
    (root / "endurance_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return workspace, summary
