# Local Autonomous Dev Team MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a Windows-first, local autonomous software-development CLI that plans, executes, tests, reviews, checkpoints, and resumes a project run.

**Architecture:** A small application service owns a durable state store and sequential role loop. Providers return validated structured decisions, while an isolated tool layer executes only workspace-scoped filesystem, commands, tests, and Git operations.

**Tech Stack:** Python 3.11+, standard library, Ollama HTTP API, pytest.

**Spec:** `docs/superpowers/specs/2026-09-10-local-autonomous-dev-team-design.md`

## Global Constraints

- Run locally on Windows 11 with Ollama; no cloud API is required.
- Execute model-proposed operations only through workspace-scoped policy-controlled tools.
- Keep agents sequential by default and model configuration role-specific.
- Persist durable run state in `.autodev/` and use verified Git checkpoints.
- Commit and push only stable, tested milestones on the existing `main` branch.

---

### Task 1: Project and durable state foundation

**Files:** Create `pyproject.toml`, `src/autodev/models.py`, `src/autodev/state_store.py`, `tests/test_state_store.py`, `.gitignore`.

**Interfaces:** Produces `ProjectState`, `Task`, `TaskStatus`, and `StateStore` for all later work.

- [ ] Write failing persistence and restart tests.
- [ ] Run them and verify missing-module failures.
- [ ] Implement typed models and atomic JSON state persistence.
- [ ] Run the focused test and full suite.
- [ ] Commit and push the stable foundation.

### Task 2: Provider, prompts, and context formation

**Files:** Create `src/autodev/providers.py`, `src/autodev/context.py`, `tests/test_providers.py`, `tests/test_context.py`.

**Interfaces:** Produces `LLMProvider.complete(request) -> AgentReply` and role/context builders.

- [ ] Write failing JSON parsing, retry, model configuration, and bounded-context tests.
- [ ] Implement `OllamaProvider` plus deterministic test provider with low-temperature defaults.
- [ ] Verify provider and context tests.
- [ ] Commit and push.

### Task 3: Safe local tool layer and Git checkpoints

**Files:** Create `src/autodev/tools.py`, `src/autodev/git.py`, `tests/test_tools.py`, `tests/test_git.py`.

**Interfaces:** Produces workspace-only file operations, validated command/test execution, status/diff/commit/restore helpers.

- [ ] Write failing path-escape and command-policy tests.
- [ ] Implement tools with actual process exit-code/stdout/stderr results.
- [ ] Test in temporary Git repositories.
- [ ] Commit and push.

### Task 4: Autonomous sequential loop

**Files:** Create `src/autodev/orchestrator.py`, `src/autodev/agents.py`, `tests/test_orchestrator.py`.

**Interfaces:** Produces `AutonomousRunner.run(max_cycles)` with Manager/Coder/Tester/Reviewer gates, retries, loop detection, and `BLOCKED` transition.

- [ ] Write failing tests for successful multi-task progression and failed-test repair routing.
- [ ] Implement minimal state-machine transitions and agent responses.
- [ ] Verify task lifecycle tests.
- [ ] Commit and push.

### Task 5: CLI, status controls, and real end-to-end proof

**Files:** Create `src/autodev/cli.py`, `src/autodev/__main__.py`, `README.md`, `tests/test_cli.py`, `tests/test_e2e.py`, `example-config.toml`.

**Interfaces:** Produces `python -m autodev {init,start,resume,status,pause,stop}`.

- [ ] Write failing CLI and E2E tests for a TODO request that needs multiple autonomous tasks and one repair cycle.
- [ ] Implement commands and readable status/action logs.
- [ ] Run the full suite and a manual CLI smoke test.
- [ ] Commit and push the MVP.
