# Local Coder model comparison — 2026-09-13

## Decision

Keep `qwen3-coder:30b` as the default model for the next live endurance run.

## Method

The benchmark ran sequentially against eight disposable generic coding fixtures. It called `RoleAgents.code` with the production Coder system prompt, action schema, structured-output repair, temperature `0.05`, context limit `16384`, and Ollama thinking disabled. Each response was executed only inside its disposable fixture and checked programmatically. Existing-file full rewrites were counted separately.

| Metric | qwen3-coder:30b | qwen3.6:27b |
| --- | ---: | ---: |
| Valid structured responses | 8/8 | 8/8 |
| Valid tool actions | 8/8 | 8/8 |
| Strict task fixes | 8/8 | 7/8 |
| NOOP responses | 0 | 0 |
| Existing-file full rewrites | 0 | 0 |
| Mean latency | 3.57s | 12.13s |

`qwen3.6:27b` failed the strict dependency-incompatibility fixture: it replaced `SQLAlchemy==2.0.23` with `SQLAlchemy>=2.0.23,<3.0.0`, which still permits the known-bad release. The earlier substring-only scorer incorrectly called that a success; a regression test now rejects constraints which still admit `2.0.23`.

This is a focused Coder action benchmark, not a claim of general model superiority or end-to-end success. The model stays unchanged because the established Coder was about 3.4 times faster and passed one more deterministic fixture in this workload.
