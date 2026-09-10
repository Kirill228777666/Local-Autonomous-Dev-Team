"""Durable run metrics and concise reports."""
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from .models import ProjectState

def metrics(state: ProjectState) -> dict[str, object]:
    events = state.events
    counts = {name: sum(event.agent == name for event in events) for name in {event.agent for event in events}}
    return {"run_start": state.created_at, "updated_at": state.updated_at, "llm_requests_by_role": counts, "tasks_created": len(state.tasks), "tasks_completed": sum(task.status.value == "DONE" for task in state.tasks), "repair_cycles": sum("Returning task" in entry for entry in state.run_history), "reviewer_rejections": sum("Review rejected" in entry for entry in state.run_history), "final_qa_failures": sum("Final QA failed" in entry for entry in state.run_history), "git_checkpoints": sum("checkpointed" in entry for entry in state.run_history), "recovery_events": sum("Recovered interrupted" in entry for entry in state.run_history)}

def write_metrics(directory: Path, state: ProjectState) -> None:
    data = metrics(state)
    (directory / "metrics.json").write_text(json.dumps(data, indent=2)+"\n", encoding="utf-8")
    (directory / "run_report.md").write_text(f"# Run Report\n\n## Goal\n{state.original_spec}\n\n## Status\n{state.status}\n\n## Metrics\n```json\n{json.dumps(data, indent=2)}\n```\n\n## Final QA\n{state.final_qa_status}\n", encoding="utf-8")
