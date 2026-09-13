# Orchestration Core Stabilization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace retry-task/Manager chatter with a durable deterministic task state machine, closed environment repair, balanced provider accounting, and evidence-based local model selection.

**Architecture:** Add a focused `TaskController` that owns task transitions and retry decisions while `AutonomousRunner` continues to integrate existing agents and services. Move real provider-attempt outcomes into provider instrumentation, preserve application snapshots with reasoned rollback accounting, and benchmark models only in disposable fixtures.

**Tech Stack:** Python 3.11+, dataclasses/enums, pytest, Ollama HTTP API, existing StateStore/WorkspaceTools/EnvironmentManager/GitRepository.

**Spec:** `docs/superpowers/specs/2026-09-13-orchestration-core-stabilization-design.md`

## Global Constraints

- Preserve sequential local-model inference and the 16384 context limit.
- Preserve project-local `.venv`, UTF-8 persistence, provider circuit, managed-process lifecycle, tool recovery, snapshots, regression checks, and Git checkpoints.
- Do not hardcode Notes-specific behavior in the controller.
- Do not resume or patch Run-9.
- Production behavior is implemented only after its focused test fails for the expected reason.

---

### Task 1: Durable deterministic task controller

**Files:**
- Create: `src/autodev/controller.py`
- Modify: `src/autodev/models.py`
- Modify: `src/autodev/orchestrator.py`
- Test: `tests/test_controller.py`

**Interfaces:**
- Produces: `TaskPhase`, `TransitionError`, `FailureDecision`, `TaskController.next_ready()`, `TaskController.transition()`, `TaskController.semantic_key()`, `TaskController.register_semantic_call()`, and `TaskController.decide_failure()`.
- Persists: task phase, semantic call keys, semantic failure count, rollback count/reasons, strategy history.

- [ ] Write failing tests for legal/illegal transitions, deterministic selection, loop-breaker rejection, one strategy change, precise root blocking, and backward-compatible task deserialization.
- [ ] Run `pytest tests/test_controller.py -q` and verify failures identify the missing controller/state fields.
- [ ] Implement the minimal controller and task serialization defaults.
- [ ] Replace Manager-based steady-state selection with `TaskController.next_ready()` and run the focused tests.
- [ ] Commit the green controller boundary.

### Task 2: Root-task local recovery flow

**Files:**
- Modify: `src/autodev/orchestrator.py`
- Modify: `src/autodev/agents.py`
- Modify: `src/autodev/context.py`
- Test: `tests/test_controller_integration.py`
- Test: `tests/test_orchestrator.py`
- Test: `tests/test_v03_recovery_runtime.py`

**Interfaces:**
- Consumes: `TaskController` failure decisions and existing snapshot/tool/harness helpers.
- Produces: evidence-rich same-attempt NOOP recovery, one local acceptance retry, one Architect strategy transition, and terminal `BLOCKED_ROOT_TASK` without ordinary corrective children.

- [ ] Write failing scripted scenarios A, B, C, G, H, I, J, K, and L with Manager/Architect/Coder call-budget assertions.
- [ ] Run the focused scenarios and record their retry-task/Manager failures.
- [ ] Extract bounded Coder action execution, retain stale/destructive recoveries, and add same-attempt NOOP evidence reprompt.
- [ ] Route genuine failures through the controller; remove ordinary corrective-task creation and Manager reselection.
- [ ] Emit a terminal Coder outcome for every semantic Coder call and focused context containing exact failure evidence/current files.
- [ ] Run controller/orchestrator/recovery tests and commit the green local-recovery flow.

### Task 3: Closed environment-repair state

**Files:**
- Modify: `src/autodev/environment.py`
- Modify: `src/autodev/orchestrator.py`
- Test: `tests/test_environment.py`
- Test: `tests/test_controller_integration.py`

**Interfaces:**
- Produces: exact-command repair verification and exactly one of `ENVIRONMENT_REPAIR_SUCCEEDED`, `ENVIRONMENT_REPAIR_FAILED`, or `ENVIRONMENT_REPAIR_ABORTED_BY_PROVIDER` per repair attempt.
- Preserves: successful manifest/venv changes across later application rollback.

- [ ] Write failing tests for dependency success/resume, unchanged fingerprint failure, outcome closure, snapshot preservation, and Run-9 SQLAlchemy incompatibility repair.
- [ ] Reproduce the SQLAlchemy 2.0.23/Python 3.14 import failure in a disposable venv and verify the resolved current release imports.
- [ ] Implement version-aware project-venv upgrade plus resolved-version manifest persistence.
- [ ] Rerun the exact original command after repair; synchronize successful environment files into the attempt baseline.
- [ ] Run environment/controller tests and commit the closed environment state.

### Task 4: Exact provider and rollback metrics

**Files:**
- Modify: `src/autodev/providers.py`
- Modify: `src/autodev/orchestrator.py`
- Modify: `src/autodev/metrics.py`
- Modify: `src/autodev/state_store.py`
- Test: `tests/test_providers.py`
- Test: `tests/test_orchestrator.py`
- Test: `tests/test_v03_recovery_runtime.py`

**Interfaces:**
- Produces: provider attempt observer and terminal outcome counters whose sum equals attempts.
- Produces: attempt rollback total/by-reason metrics; legacy regression rollback counts only actual regression rejection.

- [ ] Write failing provider tests for success, retry-success, timeout, HTTP/connection failure, malformed response, and exact outcome balance.
- [ ] Add provider-level attempt/outcome observation, including RoleModelProvider propagation and scripted-provider parity.
- [ ] Add semantic Coder terminal events for malformed/provider/policy/noop/action results.
- [ ] Write failing rollback-reason and meaningful-progress metric tests.
- [ ] Pass explicit rollback reasons through orchestration and update report metrics with compatibility aliases.
- [ ] Run provider/metrics/controller tests and commit accounting changes.

### Task 5: Deterministic common validators

**Files:**
- Create: `src/autodev/validation.py`
- Modify: `src/autodev/orchestrator.py`
- Test: `tests/test_validation.py`
- Test: `tests/test_controller_integration.py`

**Interfaces:**
- Produces: `ValidationPlanner.command_for(workspace, task, environment)` returning a safe command only when project evidence makes the validator unambiguous.
- Falls back: existing Tester decision plus bounded harness regeneration.

- [ ] Write failing tests for Python unittest discovery, pytest, compile/import, and static frontend validation selection.
- [ ] Implement minimal evidence-based command templates using the persisted project interpreter.
- [ ] Prefer deterministic commands before Tester calls and prove malformed-harness recovery has zero Manager/Coder delta.
- [ ] Run validation and neighboring orchestration tests; commit.

### Task 6: Sequential local-model A/B evaluation

**Files:**
- Create: `scripts/model_ab_eval.py`
- Create: `tests/test_model_eval.py`
- Modify if selected: `example-config.toml`
- Modify if selected: `src/autodev/cli.py`
- Modify: `README.md`

**Interfaces:**
- Produces: JSON/Markdown benchmark report under an ignored temporary artifact directory, with per-model validity, actions, NOOP, rewrite, validation, and latency metrics.

- [ ] Write failing offline tests for fixture isolation, scoring, and report schema.
- [ ] Implement eight disposable representative cases and sequential Ollama execution.
- [ ] Verify `ollama list`, run both models with identical 16384 context/temperature settings, and record evidence.
- [ ] If qwen3.6 wins materially, update defaults/config consistently; otherwise retain qwen3-coder.
- [ ] Run config/model-eval tests and commit the measured selection.

### Task 7: Full quality gate and detached Run-10

**Files:**
- Modify: `README.md` only if command/metric documentation needs final synchronization.

- [ ] Run all focused controller state-machine scenarios and call-budget assertions.
- [ ] Run the complete pytest suite, using deliberate Windows groups if needed, and record the exact total.
- [ ] Run `python -m compileall -q src tests scripts`.
- [ ] Review `git diff`, `git status`, staged paths, and ensure `.e2e-runs`/benchmark fixtures/logs are not staged.
- [ ] Push every green commit.
- [ ] Launch one fresh detached Notes Run-10 at the selected commit/model.
- [ ] Verify only PID, commit, model, distinct workspace, and no immediate startup crash; print handoff paths/command and stop monitoring.
