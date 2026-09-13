# Orchestration Core Stabilization Design

## Objective

Make the controller, rather than Manager/Architect calls, own the task lifecycle. A stable plan advances deterministically through coding, validation, regression, review, checkpoint, and a bounded strategy change. Provider outages and recoverable tool, harness, and environment failures suspend or recover the current root task without creating ordinary `Repair:` children.

## Controller boundary

`TaskController` owns legal phase transitions, deterministic ready-task selection, semantic-call loop keys, bounded local retries, one strategy-change decision, and terminal root-task blocking. `AutonomousRunner` remains the integration layer for agents, tools, environment management, snapshots, Git, visual review, and persistence. Existing provider, venv, process, rollback, and checkpoint implementations remain reusable services.

The durable task state records phase, failure and workspace fingerprints, strategy generation/history, semantic call keys, and rollback counts. Provider outage preserves phase and semantic attempt. A rejected application attempt is rolled back with an explicit reason; environment repair is performed outside that application transaction and has a terminal outcome.

## Task lifecycle

The legal phases are `READY`, `CODING`, `LOCAL_RECOVERY`, `ENVIRONMENT_REPAIR`, `VALIDATING`, `REGRESSION_CHECK`, `REVIEW`, `CHECKPOINT`, `STRATEGY_CHANGE`, `DONE`, and `BLOCKED`.

A normal task uses deterministic queue order. The Manager is called for initial planning only. A first genuine acceptance failure returns the same root task to an evidence-rich local retry. Repeating the same semantic key cannot call Coder again: the controller requests one Architect strategy change, then blocks precisely if the changed strategy still cannot progress. Ordinary failures never create corrective child tasks; only a newly discovered prerequisite may do so in future work.

NOOP is evaluated against acceptance. Passing acceptance means `TASK_ALREADY_SATISFIED`. Failing acceptance gets one same-attempt evidence reprompt, without Manager or Architect. Tool-policy, stale-edit, and malformed-harness recoveries remain local and bounded.

## Environment transactions

Environment repair is a closed sub-state. `EnvironmentManager` repairs known incompatibilities deterministically, persists the resolved dependency constraint, and the runner reruns the exact original command. The attempt ends as succeeded, failed, or provider-aborted. A successful environment change is synchronized into the application snapshot baseline so a later application rollback cannot erase it.

For the Run-9 Python 3.14 failure, a SQLAlchemy `TypingOnly` traceback is repaired by upgrading SQLAlchemy through the project venv and persisting the actually resolved version, followed by the exact original validation command. Unsupported or unchanged failures end as `ENVIRONMENT_REPAIR_FAILED`; no guessed version is hardcoded.

## Provider and rollback accounting

Each provider transport attempt records exactly one terminal outcome: `SUCCESS`, `CONNECTION_REFUSED`, `HTTP_ERROR`, `TIMEOUT`, `EMPTY_RESPONSE`, `MALFORMED_STRUCTURED_OUTPUT`, `CANCELLED`, `CIRCUIT_INTERRUPTED`, or `OTHER_ERROR`. Semantic role calls stay separate. At terminal state, attempted requests equal the sum of terminal outcomes.

Rollback metrics distinguish total application-attempt rollbacks and reasons (`acceptance_failure`, `regression_failure`, `malformed_model_action`, `noop_failed_acceptance`, `policy_failure`, `other`). The legacy `regression_rollbacks` key remains as a compatibility alias for actual regression-triggered rollbacks only. Meaningful progress requires a pass/checkpoint, reduced acceptance failure, validated prerequisite, or resolved regression—not an LLM response or rollback.

## Validation and model selection

Common Python validation commands are built deterministically when project evidence is sufficient; the Tester model remains the fallback for unknown behavioral checks. Controller integration tests use scripted agents and assert both state and call budgets.

A temporary-fixture A/B evaluation compares `qwen3-coder:30b` and `qwen3.6:27b` sequentially on eight representative structured coding scenarios. It measures valid JSON/tool actions, NOOPs, full-file rewrites, latency, and fixture-validation success. Only measured improvement changes the default model.

## Compatibility and safety

State loading supplies defaults for existing workspaces. Resume invariants, UTF-8 persistence, project-local Python isolation, provider circuit behavior, managed processes, stale-edit recovery, destructive-write protection, accumulated regression checks, and exact project-file rollback remain intact.
