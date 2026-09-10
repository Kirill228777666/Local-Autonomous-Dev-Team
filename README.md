# Local Autonomous Dev Team

Windows-first local MVP for an autonomous software-development loop. Give it a project description once; it stores durable state, calls Ollama through role-specific prompts, executes workspace-scoped tools, independently runs tests, reviews changes, and creates Git checkpoints.

## Requirements

- Windows 11
- Python 3.11+ (`py -3`)
- Git
- [Ollama](https://ollama.com/) running locally with a pulled model, for example `ollama pull qwen3:14b`

## Quick start

```powershell
py -3 -m pip install -e ".[dev]"
py -3 -m autodev init C:\work\my-app --spec-file C:\work\my-spec.md
py -3 -m autodev start C:\work\my-app --config .\example-config.toml
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
```

`pause` preserves state for a later `resume`. `stop` deliberately ends the current run. The LLM has no shell access: all actions are validated and executed through the workspace-only tool layer.

## Development verification

```powershell
py -3 -m pytest -q
```

The suite includes a deterministic end-to-end TODO-project run that proves multiple autonomous tasks, one failed test/repair cycle, persistence, and Git checkpoints without requiring an installed model.
