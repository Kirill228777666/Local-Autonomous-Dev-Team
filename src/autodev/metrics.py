"""Durable run metrics and concise reports."""
from __future__ import annotations
import json
from datetime import UTC, datetime
from pathlib import Path
from .models import ProjectState

def metrics(state: ProjectState) -> dict[str, object]:
    events = state.events
    llm_events = [event for event in events if event.phase == "LLM_CALL"]
    calls_by_role = {role: sum(event.agent == role for event in llm_events) for role in sorted({event.agent for event in llm_events})}
    tool_events = [event for event in events if event.phase == "TOOL_STARTED"]
    try:
        duration_seconds = max(0, int((datetime.fromisoformat(state.updated_at) - datetime.fromisoformat(state.created_at)).total_seconds()))
    except ValueError:
        duration_seconds = 0
    history = state.run_history
    return {
        "run_start": state.created_at,
        "updated_at": state.updated_at,
        "run_duration_seconds": duration_seconds,
        "total_llm_calls": len(llm_events),
        "llm_calls_by_role": calls_by_role,
        "total_tool_calls": len(tool_events),
        "tasks_created": len(state.tasks),
        "tasks_completed": sum(task.status.value == "DONE" for task in state.tasks),
        "repairs": sum("Returning task" in entry for entry in history),
        "reviewer_rejects": sum("Review rejected" in entry for entry in history),
        "final_qa_failures": sum("Final QA failed" in entry for entry in history),
        "architect_diagnoses": sum(event.agent == "ARCHITECT" and event.phase == "LLM_CALL" for event in events),
        "designer_calls": sum(event.agent == "DESIGNER" and event.phase == "LLM_CALL" for event in events),
        "designer_failures": sum("Visual QA failed" in entry for entry in history),
        "visual_repairs": state.visual_repair_cycles,
        "command_failures": sum(event.phase == "TOOL_FAILED" for event in events),
        "model_timeouts": sum("timed out" in entry.lower() for entry in history),
        "context_truncations": sum("context truncated" in entry.lower() for entry in history),
        "git_checkpoints": sum("checkpointed" in entry for entry in history),
        "crash_events": sum("Crash recovery" in entry for entry in history),
        "resume_events": sum(entry == "Run resumed" for entry in history),
        "watchdog_events": sum(event.agent == "WATCHDOG" for event in events),
        "environment_checks": sum(event.agent == "ENVIRONMENT" and event.phase == "CHECK" for event in events),
        "dependency_installs": sum(event.agent == "ENVIRONMENT" and event.phase == "REPAIRED" for event in events),
        "dependency_install_failures": sum("Environment limitation:" in entry and "install" in entry.lower() for entry in history),
        "missing_executables": sum(event.agent == "ENVIRONMENT" and event.phase == "MISSING_EXECUTABLE" for event in events),
        "environment_repairs": sum(event.agent == "ENVIRONMENT" and event.phase == "REPAIRED" for event in events),
        "task_superseded_count": sum(task.status.value == "SUPERSEDED" for task in state.tasks),
        "tool_recoveries": sum(event.phase == "TOOL_RECOVERY" for event in events),
        "stale_edit_recoveries": sum(event.phase == "TOOL_RECOVERY" and "stale edit" in event.message.lower() for event in events),
        "tool_recovery_failures": sum("Tool stale edit recovery" in entry and "Coder error" in entry for entry in history),
        "environment_capability_cache_hits": sum("known unavailable" in entry.lower() for entry in history),
        "task_decompositions": sum("decomposed" in entry.lower() for entry in history),
        "screenshot_attempts": sum("Screenshot" in event.message for event in events),
        "screenshot_successes": sum("screenshots captured" in event.message.lower() for event in events),
        "managed_process_starts": sum(event.agent == "RUNTIME" and event.phase == "PROCESS_START" for event in events),
        "managed_process_stops": sum(event.agent == "RUNTIME" and event.phase == "PROCESS_STOP" for event in events),
        "managed_process_failures": sum(event.agent == "RUNTIME" and event.phase == "READINESS_FAILED" for event in events),
        "readiness_checks": sum(event.agent == "RUNTIME" and event.phase in {"READINESS", "READINESS_FAILED"} for event in events),
        "readiness_failures": sum(event.agent == "RUNTIME" and event.phase == "READINESS_FAILED" for event in events),
        "process_tree_kills": sum(event.agent == "RUNTIME" and event.phase == "PROCESS_STOP" for event in events),
        "orphan_processes_cleaned": sum("Cleaned stale managed process" in entry for entry in history),
        "long_running_commands_rerouted": sum("rerouted to managed process" in item.detail for item in state.tool_executions),
        "run_command_hard_timeouts": sum("hard timed out" in item.detail for item in state.tool_executions),
        "visual_status": state.visual_status,
    }

def write_metrics(directory: Path, state: ProjectState) -> None:
    data = metrics(state)
    (directory / "metrics.json").write_text(json.dumps(data, indent=2)+"\n", encoding="utf-8")
    completed = [task.title for task in state.tasks if task.status.value == "DONE"]
    blocked = [task.title for task in state.tasks if task.status.value == "BLOCKED"]
    limitations = [entry for entry in state.run_history if "skipped" in entry.lower() or "unavailable" in entry.lower()]
    report = (
        "# Run Report\n\n"
        f"## Project\n{directory.parent.name}\n\n## Original goal\n{state.original_spec}\n\n"
        f"## Status\n{state.status}\n\n## Models\n{state.model}\n\n"
        f"## Completed tasks\n" + ("\n".join(f"- {item}" for item in completed) or "- None") + "\n\n"
        f"## Blocked tasks\n" + ("\n".join(f"- {item}" for item in blocked) or "- None") + "\n\n"
        f"## Final QA\n{state.final_qa_status}\n" + "\n".join(f"- {item}" for item in state.final_qa_findings) + "\n\n"
        f"## Visual QA\n{state.visual_status}\n" + "\n".join(f"- {item}" for item in state.visual_issues) + "\n\n"
        f"## Last checkpoint\n{state.last_checkpoint or 'None'}\n\n"
        f"## Metrics\n```json\n{json.dumps(data, indent=2)}\n```\n\n"
        "## Recovery events\n" + ("\n".join(f"- {item}" for item in state.run_history if "recovery" in item.lower() or "resumed" in item.lower()) or "- None") + "\n\n"
        "## Known limitations\n" + ("\n".join(f"- {item}" for item in limitations) or "- None recorded.") + "\n"
    )
    (directory / "run_report.md").write_text(report, encoding="utf-8")
