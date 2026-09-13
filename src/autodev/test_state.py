"""Owned, reproducible mutable-state context for deterministic validators."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class TestState:
    path: Path
    environment: dict[str, str]


class ValidationStateManager:
    """Allocate only AutoDev-owned temporary test storage, never user data."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.resolve()
        self.root = self.workspace / ".autodev" / "test-state"

    def for_validator(self, capability_id: str, command: list[str]) -> TestState:
        identity = hashlib.sha256((capability_id + "\0" + "\0".join(command)).encode("utf-8")).hexdigest()[:16]
        directory = self.root / identity
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{uuid4().hex}.sqlite"
        return TestState(path, {"AUTODEV_TESTING": "1", "AUTODEV_TEST_DATABASE": str(path)})

    def cleanup(self, state: TestState) -> None:
        """Remove only the generated owned file, never a project database."""
        try:
            state.path.resolve().relative_to(self.root.resolve())
        except ValueError:
            return
        state.path.unlink(missing_ok=True)
