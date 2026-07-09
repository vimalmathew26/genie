# Genie — Architecture

This document is the deeper companion to the README. It covers the four-layer design, the brain loop, and the engineering decisions that produced the reliability numbers.

## The layer model

Genie is built as four strict layers. Each layer consumes the one below it as a primitive and knows nothing about the ones above.

| Layer | Module(s) | Responsibility |
|---|---|---|
| 1. Window Registry | `window_registry.py` | Application lifecycle: launch, window identity, crash detection |
| 2. Element Resolution | `element_resolver.py` + 3 mixins | Turning semantic targets into executed UI actions |
| 3. Observation Capture | `observation.py` | Recording what actually happened, structurally, at zero token cost |
| 4. Brain Loop | `orchestrator.py` + 15 extracted modules | ReAct orchestration, planning, dispatch, budgets, recovery |

## Layer 1 — Window identity

The naive approach (`xdotool search --name firefox`) breaks the moment two windows match. Layer 1 replaces it with a PID-keyed session registry: every app Genie launches gets a validated X11 window ID, a stable label, and a background monitor.

Key mechanics:

- `open_app` blocks until the window ID is validated through a `_NET_CLIENT_LIST` + geometry + xprop sequence, or raises `LaunchFailureError` after 3 retries with exponential backoff on the timeout (1.0x, 1.5x, 2.0x). Timeouts are per app tier, calibrated to measured cold-launch times: 8s for GTK apps, 20s for browsers, 25s for Electron.
- Window IDs frequently belong to *child* processes (measured on Firefox and Obsidian), so validation traverses the process tree rather than trusting the launched PID.
- Singleton apps (gedit-style: launch delegates to an existing instance and exits 0 immediately) are handled by window adoption: search by `wm_class`, validate via xprop, register the existing window. Unknown apps not in any profile get dynamic `wm_class` discovery via client-list diff against a pre-launch snapshot.
- A background poller checks the client list every 250ms and enqueues closed/crashed events; the orchestrator drains them at the top of every iteration, so the brain loop always knows when a window it depends on has died.
- Chromium and Electron apps get monotonically assigned, socket-probed debug ports with no recycling, and stale browser locks are cleaned before every CDP launch.
- Concurrency discipline: two locks (registry, port ledger), never held simultaneously, and never held across an X11 round-trip. X connections are thread-local with atoms cached at creation.

## Layer 2 — Element resolution

The design rule: **the LLM never emits coordinates.** It emits semantic targets, `(app_label, role, name)`, and Layer 2 turns them into clicks, keystrokes, and reads. Three tiers:

**Tier 1, AT-SPI** — the Linux accessibility tree, primary for native apps. Resolution is a BFS over the tree with role aliasing. The interesting contract is ambiguity handling: zero matches raise a typed error; two or more matches trigger the vision tier for disambiguation rather than guessing.

**Tier 2, CDP** — Chrome DevTools Protocol for browser targets. BFS over the accessibility tree via `Accessibility.getFullAXTree`, with JS fast paths for the cases where that times out on heavy pages: unnamed input fields resolve via a "find the best input" script, and Nth-search-result navigation goes straight to `Page.navigate`, bypassing the AX tree entirely. Stale DOM nodes surface as typed transient errors that trigger a fresh BFS on retry.

**Tier 3, Set-of-Marks vision** — the fallback, triggered on ambiguity (2+ matches) or, in the discovery variant, on zero matches where the tree should have had one. A window screenshot gets numbered markers overlaid at candidate positions; the vision model returns one integer; the click lands at that marker's coordinates. ~$0.0016 per call, paid only when deterministic resolution has already failed.

Terminal apps get a fourth path: no semantic tree exists, so typing goes directly to the window and element resolution is refused with a typed error rather than faked.

Every resolution failure is classified into a four-class exception hierarchy: `TransientError`, `EnvironmentalError`, `ResourceError`, `UnrecoverableError`. This taxonomy is load-bearing; Layer 4's entire recovery strategy keys off it.

## Layer 3 — Observation

A passive recording engine with three hard rules: it never calls an LLM, never judges task progress, and never raises (unconditional no-propagation, enforced by a last-resort catch). The orchestrator calls `observe()` from a `finally` block, so every action, successful or failed, produces a structured JSONL entry: action, args (truncated), derived result class, per-action observation payload, error, duration.

Result derivation is a fixed-order case analysis, not heuristics: shell commands get their own labels (`command_failed`, `command_timeout`) derived from exit code and timeout state; element actions map their typed exceptions to `transient_failure` / `environmental_failure` / `unrecoverable`; unknown exceptions default to `unrecoverable` rather than optimistically to retryable.

Two output streams: `genie_success.jsonl` (clean trajectories) and `genie_incomplete.jsonl` (everything). The split exists because the success stream doubles as fine-tuning trace data; observation was designed from the start to make every run a potential training example.

## Layer 4 — The brain loop

A ReAct orchestrator: call the model, parse `<think>` + `<act>` (strict JSON), validate against the action schema, dispatch, observe, repeat, under a hard iteration cap and a per-task cost budget.

The structural decisions that matter:

**Static system prompt, dynamic user turn.** The system prompt never changes (making it cacheable); goal, window registry state, last observation, compressed history, and remaining budget are injected into the user message every iteration.

**Context batching.** History is a flat deque; only the last 5 entries go to the model in full detail, older entries are compressed to one-line summaries under a character cap. Measured effect: ~97% input-token reduction on long tasks.

**Error recovery keyed to the taxonomy.** Transient errors retry the same action up to 3 times with backoff, but only for idempotent actions, and exhausted retries are *reclassified* as environmental. Environmental errors are injected as observations for the model to replan around; the orchestrator stays passive. Resource errors halt in interactive mode and are self-managed in autonomous mode, except budget-cap breaches, which halt unconditionally. Unrecoverable errors hard-stop with guaranteed cleanup.

**Loop detection.** A sliding window over hashes of recent act decisions; the same act 5 times within the last 8 declares the agent stuck and halts, rather than letting it burn budget repeating itself.

**Checkpointing.** In autonomous mode, state persists every iteration, and resume works at arbitrary iteration and across subtask boundaries (validated at iterations 5, 10, and 15).

## Planning and working memory

Above the per-iteration loop sits the goal-level machinery:

**GoalTracker** decomposes a goal into ordered subtasks, then deterministically corrects the plan (minimum-count retry, same-file merge, verification-tail merge, rejection of read-only openings; details in the README). Each subtask runs as its own brain loop. On subtask failure, the system re-enters interactive planning with the failure reason. Cap: 60 subtasks.

**TaskScratchpad** is the inter-subtask memory: structured JSON (`facts, files, decisions, subtask_outcomes, errors`) under a 400-token budget, with a recency-protected eviction order (errors are evicted first, facts last). A dedicated lightweight model (`ScratchpadWriter`, 2-second hard timeout) extracts entries from observations per iteration and promotes decisions between subtasks. This is how subtask 4 knows what subtask 2 learned without inheriting its transcript.

**ScriptPlanner** is a speed path: if a goal or subtask is deterministic enough to be a bash script, it's pre-generated and run directly, skipping the ReAct loop entirely, with fallback to ReAct when it isn't.

**Prefetch** cuts iterations before they happen: URLs and project files that the goal will obviously need are fetched ahead of the loop under a 120KB budget, and dispatch then *blocks* the model from opening a browser for content it already has.

## Model routing

Five roles, each routed to the cheapest model that passes the test suite for that role: planning, execution, summarisation (scratchpad extraction), complex reasoning (fallback), and vision (disambiguation). The primary execution model was selected by head-to-head validation across all task categories at ~$0.001/iteration and locked; candidates that couldn't complete tasks reliably were rejected regardless of price. Notably, the reasoning-heavy model initially assigned to planning was reverted to the primary after latency testing: measured latency beat presumed capability.

## Cost and budget discipline

Per-task cost accumulates from actual token usage against model pricing tables. A per-task budget and a hard monthly cap are enforced in the loop's pre-flight check; breaching the cap halts unconditionally in every mode. Total cost, wall time, and iteration count are reported per task.
