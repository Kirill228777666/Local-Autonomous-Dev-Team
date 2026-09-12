# Local Autonomous Dev Team

Windows-first local MVP for an autonomous software-development loop. Give it a project description once; it stores durable state, calls Ollama through role-specific prompts, executes workspace-scoped tools, independently runs tests, reviews changes, and creates Git checkpoints.

## Requirements

- Windows 11
- Python 3.11+ (`py -3`)
- Git
- [Ollama](https://ollama.com/) running locally with `qwen3-coder:30b` (the default live profile)

## Quick start

```powershell
py -3 -m pip install -e ".[dev]"
py -3 -m autodev init C:\work\my-app --spec-file C:\work\my-spec.md
py -3 -m autodev start C:\work\my-app --config .\example-config.toml
```

Open the local dashboard in a second terminal:

```powershell
py -3 -m autodev dashboard C:\work\my-app --config .\example-config.toml
# http://127.0.0.1:8765
```

The state is kept in the target workspace's `.autodev/` directory:

- `state.json` — authoritative durable state;
- `project_spec.md`, `progress.md`, `current_task.md`, `decisions.md` — readable projections;
- `logs/events.log` — chronological activity evidence.

Useful controls:

```powershell
py -3 -m autodev status C:\work\my-app
py -3 -m autodev pause C:\work\my-app
py -3 -m autodev resume C:\work\my-app --config .\example-config.toml
py -3 -m autodev stop C:\work\my-app
py -3 -m autodev requirement add C:\work\my-app "Add JSON export"
```

`pause` preserves state for a later `resume`. `stop` deliberately ends the current run. `requirement add` keeps an amendment separate from the original specification and returns it to the autonomous task queue. The LLM has no shell access: all actions are validated and executed through the workspace-only tool layer.

The default live endpoint is `http://127.0.0.1:11434` rather than `localhost`, avoiding Windows IPv6/IPv4 resolution ambiguity. If Ollama becomes unavailable, AutoDev opens a durable global provider circuit: no role prompts, task attempts, or corrective tasks are generated while it waits through health probes. A configurable `[runner] provider_max_wait_seconds` bounds that wait; its terminal state is `BLOCKED_PROVIDER` and a later `resume` preserves the original task graph.

## Reliability and inspection

Before `COMPLETE`, AutoDev runs detected project-wide checks (Python tests and declared npm test/build/lint/typecheck scripts where present), then asks a separate Final QA role to compare the result with the original specification. A Final QA failure creates corrective tasks instead of silently completing.

`.autodev/` also contains a heartbeat, human-readable `activity.log`, bounded event history, `project_summary.md`, `decisions.jsonl`, test/regression evidence, and atomic `state.json`. After an interruption, use `resume`; interrupted implementation/testing/review work is reconciled to a safe pending task.

`[models]` and `[agents.<role>]` in the config can select a model and generation settings per role. Roles remain sequential, so the machine does not need multiple large models resident in VRAM.

## v0.3 recovery and visual QA

Every tool action is persisted as `STARTED` before execution and as `SUCCEEDED` or `FAILED` afterwards. A forced process termination leaves a tool at `UNKNOWN`; `resume` validates the project specification, task graph, active-task invariant, Git repository and checkpoint before returning that task to safe pending work. `STOPPED` projects never resume, and a completed project remains completed.

For UI projects, optionally configure one local application command. AutoDev chooses a free port, waits for HTTP readiness (or a health endpoint), captures desktop and narrow screenshots with Playwright, and terminates the complete application process tree after review. Playwright or a vision model being unavailable is recorded as a limitation; it never masquerades as a visual pass.

```toml
[visual]
command = ["npm", "run", "dev", "--", "--port", "{port}"]
url = "http://127.0.0.1:{port}"
health_url = "http://127.0.0.1:{port}/health"
ready_timeout = 45
max_repairs = 2
```

The compact status and local dashboard expose task progress, heartbeat, repair/review counts, real LLM/tool calls, recovery state, visual findings, latest screenshots, and last checkpoint. Durable session evidence includes `.autodev/metrics.json`, `.autodev/run_report.md`, screenshots, and interrupted-tool records in `state.json`.

## Notes endurance scenario

This intentionally slow, opt-in scenario creates one Notes workspace from the single specification below, starts it through real local Ollama, forcibly kills the separate orchestrator only after durable active work is observed, resumes the same workspace, and writes its report/validation artifacts below `.e2e-runs/<run-id>/`:

```powershell
py -3 -m autodev e2e notes --live --config .\example-config.toml
```

It does not run as part of the normal test suite. The command never pre-seeds a Notes implementation or sends follow-up user instructions; all requirements are supplied once as the original project specification.

## Development verification

```powershell
py -3 -m pytest -q
```

The suite includes a deterministic end-to-end TODO-project run that proves multiple autonomous tasks, one failed test/repair cycle, persistence, and Git checkpoints without requiring an installed model.
