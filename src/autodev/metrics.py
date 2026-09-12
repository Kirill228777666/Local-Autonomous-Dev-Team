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
    def count(agent: str, phase: str) -> int:
        return state.event_counters.get(f"{agent}:{phase}", sum(event.agent == agent and event.phase == phase for event in events))
    return {
        "run_start": state.created_at,
        "updated_at": state.updated_at,
        "run_duration_seconds": duration_seconds,
        "total_llm_calls": len(llm_events),
        "llm_requests_attempted": count("LLM", "REQUEST") or len(llm_events),
        "llm_responses_completed": min(count("LLM", "RESPONSE"), count("LLM", "REQUEST") or len(llm_events)),
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
        "environment_checks": count("ENVIRONMENT", "CHECK"),
        "dependency_installs": count("ENVIRONMENT", "REPAIRED"),
        "dependency_install_failures": sum("Environment limitation:" in entry and "install" in entry.lower() for entry in history),
        "missing_executables": count("ENVIRONMENT", "MISSING_EXECUTABLE"),
        "environment_repairs": count("ENVIRONMENT", "REPAIRED"),
        "environment_repair_attempts": count("ENVIRONMENT", "REPAIR_ATTEMPT"),
        "environment_repair_successes": count("ENVIRONMENT", "REPAIRED"),
        "environment_repair_failures": count("ENVIRONMENT", "REPAIR_FAILED"),
        "environment_repair_skipped": count("ENVIRONMENT", "REPAIR_SKIPPED"),
        "repeated_environment_failure_fingerprints": sum("already failed without environment change" in entry.lower() for entry in history),
        "task_superseded_count": sum(task.status.value == "SUPERSEDED" for task in state.tasks),
        "duplicate_corrective_tasks_suppressed": count("MANAGER", "DUPLICATE_CORRECTIVE_SUPPRESSED"),
        "root_tasks_blocked": sum(task.status.value == "BLOCKED" and task.root_task_id == task.id for task in state.tasks),
        "repair_strategy_changes": sum(task.strategy_generation > 0 for task in state.tasks),
        "tool_recoveries": sum(event.phase == "TOOL_RECOVERY" for event in events),
        "stale_edit_recoveries": sum(event.phase == "TOOL_RECOVERY" and "stale edit" in event.message.lower() for event in events),
        "tool_recovery_failures": sum("Tool stale edit recovery" in entry and "Coder error" in entry for entry in history),
        "destructive_write_rejections": count("CODER", "DESTRUCTIVE_WRITE_REJECTED"),
        "destructive_write_recoveries": count("CODER", "DESTRUCTIVE_WRITE_RECOVERY_SUCCESS"),
        "destructive_write_recovery_failures": count("CODER", "DESTRUCTIVE_WRITE_RECOVERY_FAILED"),
        "targeted_edit_reprompts": count("CODER", "DESTRUCTIVE_WRITE_REJECTED"),
        "environment_capability_cache_hits": sum("known unavailable" in entry.lower() for entry in history),
        "task_decompositions": sum("decomposed" in entry.lower() for entry in history),
        "regression_checks": count("REGRESSION", "CHECK"),
        "regression_failures": count("REGRESSION", "FAIL"),
        "regression_rollbacks": count("REGRESSION", "ROLLBACK"),
        "regressive_changes_rejected": sum("REGRESSION:" in entry for entry in history),
        "coder_noop_responses": count("CODER", "NOOP"),
        "coder_noop_already_satisfied": count("CODER", "NOOP_ALREADY_SATISFIED"),
        "coder_noop_failed_acceptance": count("CODER", "NOOP_FAILED_ACCEPTANCE"),
        "coder_noop_reprompts": count("CODER", "NOOP_FAILED_ACCEPTANCE"),
        "coder_repeated_noop_failures": sum("repeated identical coder action" in entry for entry in history),
        "coder_zero_diff_attempts": count("CODER", "ZERO_DIFF"),
        "coder_regressive_attempts": sum("REGRESSION:" in entry for entry in history),
        "tasks_progressed_after_attempt": sum("Coder completed attempt" in entry for entry in history),
        "tester_harness_failures": count("TESTER", "HARNESS_FAILURE"),
        "tester_harness_regenerations": count("TESTER", "HARNESS_EXECUTE"),
        "tester_harness_recovery_successes": count("TESTER", "HARNESS_RECOVERED"),
        "tester_harness_recovery_failures": sum("TEST_HARNESS_BLOCKED" in entry for entry in history),
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
        "provider_circuit_opens": count("PROVIDER", "CIRCUIT_OPEN"),
        "provider_health_checks": count("PROVIDER", "HEALTH_CHECK_FAILED"),
        "provider_circuit_closures": count("PROVIDER", "CIRCUIT_CLOSED"),
        "provider_circuit_closes": count("PROVIDER", "CIRCUIT_CLOSED"),
        "provider_wait_timeouts": count("PROVIDER", "CIRCUIT_TIMEOUT"),
        "provider_outage_occurrences": int(state.provider_state.get("occurrences", 0)),
        "model_provider_failures": count("PROVIDER", "FAILURE"),
        "provider_connection_refused": count("PROVIDER", "CONNECTION_REFUSED"),
        "provider_http_503": count("PROVIDER", "HTTP_503"),
        "provider_recovery_wait_seconds": state.provider_state.get("recovery_wait_seconds", 0.0),
        "provider_task_attempts_preserved": count("PROVIDER", "TASK_ATTEMPT_PRESERVED"),
        "provider_outages": count("PROVIDER", "CIRCUIT_OPEN"),
        "provider_recoveries": count("PROVIDER", "CIRCUIT_CLOSED"),
        "visual_status": state.visual_status,
    }

def write_metrics(directory: Path, state: ProjectState) -> None:
    data = metrics(state)
    (directory / "metrics.json").write_text(json.dumps(data, indent=2)+"\n", encoding="utf-8")
    completed = [task.title for task in state.tasks if task.status.value == "DONE"]
    blocked = [task.title for task in state.tasks if task.status.value == "BLOCKED"]
    limitations = [entry for entry in state.run_history if "skipped" in entry.lower() or "unavailable" in entry.lower()]
    incident = state.provider_state
    incident_summary = "- None"
    if incident:
        incident_summary = (
            f"- {incident.get('incident_type', 'UNAVAILABLE')}: occurrences={incident.get('occurrences', 0)}, "
            f"first_seen={incident.get('first_seen', 'unknown')}, last_seen={incident.get('last_seen', 'unknown')}, "
            f"recovered={incident.get('recovered', False)}"
        )
    report = (
        "# Run Report\n\n"
        f"## Project\n{directory.parent.name}\n\n## Original goal\n{state.original_spec}\n\n"
        f"## Status\n{state.status}\n\n## Models\n{state.model}\n\n"
        f"## Completed tasks\n" + ("\n".join(f"- {item}" for item in completed) or "- None") + "\n\n"
        f"## Blocked tasks\n" + ("\n".join(f"- {item}" for item in blocked) or "- None") + "\n\n"
        f"## Final QA\n{state.final_qa_status}\n" + "\n".join(f"- {item}" for item in state.final_qa_findings) + "\n\n"
        f"## Visual QA\n{state.visual_status}\n" + "\n".join(f"- {item}" for item in state.visual_issues) + "\n\n"
        f"## Last checkpoint\n{state.last_checkpoint or 'None'}\n\n"
        f"## Model provider incident\n{incident_summary}\n\n"
        f"## Metrics\n```json\n{json.dumps(data, indent=2)}\n```\n\n"
        "## Recovery events\n" + ("\n".join(f"- {item}" for item in state.run_history if "recovery" in item.lower() or "resumed" in item.lower()) or "- None") + "\n\n"
        "## Known limitations\n" + ("\n".join(f"- {item}" for item in limitations) or "- None recorded.") + "\n"
    )
    (directory / "run_report.md").write_text(report, encoding="utf-8")
