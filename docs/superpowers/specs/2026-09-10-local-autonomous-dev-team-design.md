# Local Autonomous Dev Team — MVP Design

## Scope

Build a Windows-first Python CLI that starts and resumes one autonomous local software-development run. It accepts a natural-language specification, persists all state in the target workspace, calls Ollama through one provider interface, and drives a sequential Manager → Coder → Tester → Reviewer loop. The MVP supports a safe, explicit command/file/Git tool surface, pause and stop controls, loop detection, checkpoints, and a human-readable status view.

## Decisions

- **Python 3.11+ and standard library first.** The core avoids a web framework and heavyweight orchestration dependency; `pytest` is the only development dependency.
- **CLI before UI.** Commands are `init`, `start`, `resume`, `status`, `pause`, and `stop`. This is usable on Windows today and keeps a future desktop/web frontend behind an application service boundary.
- **Deterministic orchestration around the LLM.** Agents produce JSON decisions/actions. The host validates and executes them; an LLM never receives direct shell access.
- **Sequential roles by default.** One configured Ollama model is called with role-specific system prompts, avoiding multiple large GPU-resident models.
- **Workspace-only execution.** File paths must resolve inside the project workspace. Commands execute there and pass a deny-list/permission policy.
- **Durable state in `.autodev/`.** `state.json` is authoritative; Markdown files are readable projections/logs. State records spec, task attempts, errors, checkpoints, and lifecycle control flags.
- **Review gates are evidence-based.** Test output and Git diff are supplied to Reviewer. A task is committed only after test pass and reviewer approval; otherwise it returns to Coder with bounded retries and loop diagnosis.

## MVP boundary

The first deliverable proves a real multi-task unattended run with a deterministic scripted LLM test provider and an Ollama provider for normal use. Complex UI vision, browser automation, parallel agents, cloud escalation, and a dashboard server are extension points, not initial dependencies.

## Failure handling

Each task tracks attempts, action fingerprints, and test-result fingerprints. Repetition triggers a diagnosis/architect pass; repeated non-progress changes task status to `BLOCKED` instead of spinning. Restart reconciliation restores a running task and validates the Git checkpoint.

## Testing

Unit tests cover persistence, provider parsing, policy enforcement, task-state transitions, and orchestration gates. An end-to-end test creates a temporary TODO-project request, feeds deterministic agent responses, confirms more than one task executes without further input, confirms a failed test returns work to Coder, and checks durable state plus Git checkpoints.
