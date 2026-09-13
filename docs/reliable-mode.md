# Reliable execution mode

Reliable mode freezes a compact project contract before implementation and
normalizes the initial plan into a durable capability graph.  Task titles are
only presentation; `capability_id` is scheduling identity.

Run-12 forensics found that the old `_decompose_broad_tasks` hook ran after
the Manager plan and expanded broad Russian tasks into hard-coded English
Notes tasks.  Those replacement tasks were independent roots, so overlapping
capabilities could be implemented and tested against different assumptions.
The hook is no longer part of the execution path.  Initial plans are instead
canonicalized once, persisted in `state.capability_graph`, and remain stable
across resume.

The same run recorded nine checkpoints but zero regression checks.  Accepted
validators were keyed by command and the guard skipped the command when it
matched the current validation command.  A shared full-suite validator was
therefore never run for a later capability.  Validators are now recorded by
capability and the guard executes each unique prior command before every later
checkpoint, including a command equal to current acceptance.
