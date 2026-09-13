# Local Coder model comparison — 2026-09-13

## Decision

Use `qwen3.6:35b-coding` as the one primary model for the next live endurance run.

## Method

The benchmark ran sequentially against six disposable repair fixtures based on Run-13: SQLAlchemy's removed `Engine.has_table`, a Flask module-level-app contract conflict, a missing category response id, 404/409 error behavior, managed Flask startup, and preservation of accepted behavior. It called `RoleAgents.code` with the production Coder system prompt, action schema, structured-output repair, temperature `0.05`, context limit `16384`, and Ollama thinking disabled. Each response was executed only inside its disposable fixture and checked programmatically. Existing-file full rewrites were counted separately.

| Metric | qwen3-coder:30b | qwen3.6:35b-coding |
| --- | ---: | ---: |
| Valid structured responses | 6/6 | 6/6 |
| Valid tool actions | 6/6 | 6/6 |
| Strict task fixes | 4/6 | 5/6 |
| NOOP responses | 0 | 0 |
| Existing-file full rewrites | 0 | 0 |
| Mean latency | 4.14s | 5.48s |

Both models produced strict JSON and valid tool actions on every request; no reasoning text leaked into the structured response with Ollama `think=false`. `qwen3.6:35b-coding` repaired one additional strict fixture (the SQLAlchemy compatibility repair). Both missed the deliberately combined 404/409 fixture, so the product still keeps deterministic contract evidence and validation rather than relying on model memory alone.

This is a focused Coder action benchmark, not a claim of general model superiority or end-to-end success. The 35B model is selected because it passed one more strict repair fixture at comparable mean latency and remained stable through the exact production structured-output API path.
