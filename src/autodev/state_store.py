"""Atomic workspace-local persistence for autonomous runs."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from .models import ProjectState, utc_now


class StateStore:
    """Stores authoritative state under one project's `.autodev` directory."""

    def __init__(self, workspace: Path, allowed_root: Path | None = None) -> None:
        self.workspace = workspace.resolve()
        root = (allowed_root or workspace).resolve()
        if not self.workspace.is_relative_to(root):
            raise ValueError("workspace must be inside allowed_root")
        self.directory = self.workspace / ".autodev"
        self.path = self.directory / "state.json"

    def load(self) -> ProjectState | None:
        if not self.path.exists():
            return None
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("state.json must contain a JSON object")
        return ProjectState.from_dict(value)

    def save(self, state: ProjectState) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        state.updated_at = utc_now()
        self._write_projections(state)
        temporary_path = self.path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(state.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for attempt in range(3):
            try:
                os.replace(temporary_path, self.path)
                return
            except PermissionError:
                if attempt == 2:
                    raise
                time.sleep(0.05 * (attempt + 1))

    def _write_projections(self, state: ProjectState) -> None:
        (self.directory / "logs").mkdir(exist_ok=True)
        specification = f"# Original project specification\n\n{state.original_spec}\n"
        if state.amendments:
            specification += "\n# Amendments\n\n" + "\n".join(f"- {amendment}" for amendment in state.amendments) + "\n"
        (self.directory / "project_spec.md").write_text(specification, encoding="utf-8")
        task_lines = [f"- [{task.status}] {task.title}: {task.description}" for task in state.tasks]
        (self.directory / "progress.md").write_text(
            f"# Progress\n\nStatus: {state.status}\n\n" + "\n".join(task_lines) + "\n",
            encoding="utf-8",
        )
        current = next((task for task in state.tasks if task.id == state.current_task_id), None)
        current_text = "No active task" if current is None else f"# {current.title}\n\n{current.description}\n"
        (self.directory / "current_task.md").write_text(current_text, encoding="utf-8")
        (self.directory / "decisions.md").write_text(
            "# Decisions\n\n" + "\n".join(f"- {decision}" for decision in state.decisions) + "\n",
            encoding="utf-8",
        )
        (self.directory / "logs" / "events.log").write_text(
            "\n".join(state.run_history) + "\n", encoding="utf-8"
        )
        activity = "\n".join(
            f"{event.timestamp} {event.agent.title():<10} {event.phase:<10} {event.message}"
            for event in state.events
        )
        (self.directory / "activity.log").write_text(activity + "\n", encoding="utf-8")
