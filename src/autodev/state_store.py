"""Atomic workspace-local persistence for autonomous runs."""

from __future__ import annotations

import json
import os
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
        temporary_path = self.path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(state.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, self.path)
