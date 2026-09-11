"""Small, explicit context projections for local role prompts."""

from __future__ import annotations

from dataclasses import dataclass

from .models import ProjectState, Task


@dataclass(frozen=True, slots=True)
class ContextBuilder:
    max_log_entries: int = 8
    max_characters: int = 12000

    def for_task(self, state: ProjectState, task: Task, relevant_files: list[str]) -> str:
        recent_log = state.run_history[-self.max_log_entries :]
        files = "\n".join(f"- {path}" for path in relevant_files) or "- None selected"
        events = "\n".join(f"- {entry}" for entry in recent_log) or "- No prior events"
        sections = [
            "Environment capabilities:\n" + (str(state.environment) if state.environment else "- Not discovered"),
            f"Current task ({task.id}): {task.title}\n{task.description}",
            f"Relevant files:\n{files}",
            "Recent errors:\n" + ("\n".join(f"- {error}" for error in task.errors[-4:]) or "- None"),
            "Architectural decisions:\n" + ("\n".join(f"- {decision}" for decision in state.decisions[-8:]) or "- None"),
            "Amendments:\n" + ("\n".join(f"- {amendment}" for amendment in state.amendments[-8:]) or "- None"),
            f"Recent execution evidence:\n{events}",
        ]
        reserved = sum(len(section) + 2 for section in sections)
        spec_budget = max(128, self.max_characters - reserved)
        specification = state.original_spec[:spec_budget]
        context = f"Original specification:\n{specification}\n\n" + "\n\n".join(sections)
        return context[: self.max_characters]
