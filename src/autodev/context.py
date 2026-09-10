"""Small, explicit context projections for local role prompts."""

from __future__ import annotations

from dataclasses import dataclass

from .models import ProjectState, Task


@dataclass(frozen=True, slots=True)
class ContextBuilder:
    max_log_entries: int = 8

    def for_task(self, state: ProjectState, task: Task, relevant_files: list[str]) -> str:
        recent_log = state.run_history[-self.max_log_entries :]
        files = "\n".join(f"- {path}" for path in relevant_files) or "- None selected"
        events = "\n".join(f"- {entry}" for entry in recent_log) or "- No prior events"
        return (
            f"Original specification:\n{state.original_spec}\n\n"
            f"Current task ({task.id}): {task.title}\n{task.description}\n\n"
            f"Relevant files:\n{files}\n\n"
            f"Recent execution evidence:\n{events}"
        )
