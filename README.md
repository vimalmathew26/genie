# Genie

**An autonomous agent that operates a live Linux desktop to complete multi-step software tasks unsupervised.** Genie opens applications, navigates codebases, edits files, runs tests, reads its own errors, and recovers from its own failures until the goal is verifiably done.

Built solo in two months. 29,446 lines of Python across 37 modules.

---

## Verified numbers

| | |
|---|---|
| Registered actions | 73 (file ops, shell, browser, git, full GitHub REST API, code search, PR automation, client handoff) |
| Integration tests | 40/40 passing across 14 categories |
| Repeat reliability | ≥90% clean-run rate per test (each validated over 10+ independent runs) |
| Crash recovery | Checkpoint/resume at arbitrary iteration, validated at iterations 5, 10, and 15, and across subtask boundaries |
| Cost | Per-task cost tracking, ~$0.001 per iteration on the primary model, hard monthly budget cap enforced in config |
| Output hygiene | Built-in secrets scanner: client handoff packages are scanned and `.env` files excluded automatically (validated by test T37) |

Every test has a programmatic validator. Nothing on this list is self-reported by the agent; a test passes when an external check confirms the artifact (pytest exits 0, files byte-for-byte correct, git log matches, API state verified).

## What a run looks like

Test T33, one of the 40:

> Genie is pointed at a multi-file Python project containing an `ISSUE.md` describing a bug. It reads the issue, locates the off-by-one error in the source, patches it, and reruns pytest until green. The validator then confirms three things: the fix is the *correct* fix (`[:n]` present, `[:n - 1]` gone), the test suite passes, and `ISSUE.md` is byte-for-byte unmodified.

That last check matters: an agent that "fixes" a bug by editing the bug report is a failure mode this suite is designed to catch. The same discipline runs through the suite: T31 requires fixing a failing test while leaving three unrelated files byte-for-byte untouched; T32 requires adding a method to a class with every original method verbatim intact, verified by AST parse.

## How it works

The core is a ReAct loop, but the reliability numbers come from what happens *around* the LLM, not inside it. Genie is built on one assumption: **the model will be wrong, and the system has to catch it before the wrongness compounds.**

Concretely, that means the model is never the authority on three things:

**Whether the plan is good.** When the GoalTracker asks the LLM to decompose a goal into subtasks, the returned plan is treated as a draft. Code then enforces rules the model routinely violates: if the plan has fewer subtasks than the goal's size demands, it is sent back with a hard constraint and re-decomposed. If two subtasks edit the same file, they are merged so they cannot overwrite each other. If the plan opens with "explore the codebase" instead of an action, it is rejected, because exploration-first plans burn iterations without progress. None of this is prompting. It is deterministic post-processing of the model's output.

**What happened so far.** Subtasks do not inherit each other's chat transcript. Between subtasks, a separate cheap model extracts what actually matters (facts established, files touched, decisions made, errors hit) into a structured JSON scratchpad, and the next subtask starts from that. The result: subtask 4 knows what subtask 2 learned without dragging 40 iterations of raw history, and long tasks run at ~97% token reduction versus naive accumulation.

**Whether the task is done.** The model saying "done" means nothing. A task completes when an external validator confirms it: pytest exits 0, the file is byte-for-byte correct, the git log shows the commit, the API returns the expected state. This applies to all 40 integration tests, and it is why the reliability figures are trustworthy: the agent cannot grade its own homework.

Beyond that core:

**Failure is a first-class path.** Every element-resolution and execution error is classified into a typed hierarchy (transient, environmental, resource, unrecoverable), each with its own recovery policy. A loop detector catches repeated non-progress. On subtask failure, the system re-enters interactive planning with the failure reason, rather than blind retry. Checkpoints persist across crashes, including mid-plan.

**Three-tier UI resolution.** Semantic element targets (`app, role, name`) resolve through:
1. **AT-SPI**: the Linux accessibility tree, primary tier for native apps
2. **CDP**: Chrome DevTools Protocol for browser targets, with JS fast paths for heavy pages where the accessibility tree times out
3. **Vision (Set-of-Marks)**: when the tree yields zero or ambiguous matches, a screenshot with numbered markers goes to a vision model, which returns the marker to click (~$0.0016/call)

Each tier is a fallback for the one above it, so vision-model cost is paid only when deterministic resolution fails.

**Five-tier model routing.** Planning, execution, summarisation, complex reasoning, and vision each route to the cheapest model that passes the test suite for that role, with fallbacks. The primary execution model was locked after head-to-head validation across all task categories.

**Delivery pipeline.** Completed dev tasks can push a branch, open a PR via the GitHub API, or assemble a client handoff zip (secrets-scanned, with generated handoff notes), with Telegram notification and human approve/reject at the review gate. Underspecified goals trigger structured clarifying questions before execution begins.

## The test suite

`genie_suite.py` (T1–T26), `genie_phase4.py` (T27–T35), `genie_phase5.py` (T36–T40). Categories: browser, file ops, shell, URL fetching (with a 5-tier fetch cascade), full dev cycles (including TDD: scaffold an implementation against a pre-written pytest suite and iterate to green), review pipeline, crash-resume, git operations, codebase navigation, issue-to-fix, PR automation, handoff packaging, clarifying questions, and goal decomposition.

The reliability rule during development: a test did not count as done when it passed once. It counted when it passed at a >80% clean-run rate over 10+ attempts with a schema error rate under 5%.

## Status

Built and validated over two months, then deliberately shelved. After the technical validation, a systematic market analysis across six monetization paths (freelance dev delivery, codebase sales, RPA, automation gigs, and others) found no revenue model that closed, in every case due to market saturation rather than product shortfall. Shelving a working system on market evidence was the correct engineering decision, and this repo now serves as the record of the build.

The codebase is published as a reference implementation. A recorded demo run is planned.

## Running it

Genie targets a specific desktop environment and is not a turnkey install:

- **OS**: Pop!_OS 24.04 (Ubuntu 24 base)
- **Display server**: Xorg required (Wayland is not supported; input synthesis relies on `xdotool` and window management on X11 semantics)
- **Desktop**: GNOME
- **Python**: 3.12+, dependencies in `requirements.txt` (includes `pyatspi` for accessibility-tree access)
- **LLM access**: all inference is cloud-based via OpenRouter; no local GPU needed

Copy `.env.example` to `.env` and fill in:

- `OPENROUTER_API_KEY` (required)
- `GITHUB_TOKEN` (for git/PR actions)
- `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` (optional, for notifications and remote approval)

Entry point: `python genie.py` launches the GTK4 UI shell. The test suites can be run directly (`genie_suite.py`, `genie_phase4.py`, `genie_phase5.py`).

## Architecture

A deeper architecture document (window registry, element resolver internals, observation layer, orchestrator decomposition) is planned. The module layout is self-describing: `orchestrator.py` (brain loop), `goal_tracker.py` (decomposition), `element_resolver.py` + three tier mixins, `observation.py` (passive state capture), `window_registry.py` (app/window lifecycle), `dispatch.py` + `actions.py` (action execution), `batch_engine.py` (multi-action batching and conditionals), `persistence.py` (checkpoints and cost tracking).
