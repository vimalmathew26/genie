"""
Genie — planner.py
Plan phase and sequence-execution speed path.

Extracted from orchestrator.py.  All functions take ``orch``
(a GenieOrchestrator instance) as their first argument so they can
call orch._llm_call, orch._validate_act, orch._dispatch, etc.
"""
from __future__ import annotations

import config
import json
import os
import re
import threading
import time
from typing import TYPE_CHECKING

import httpx

from config import (
    LLM_SERVICE_ERROR_CODES,
    MAX_CLARIFY_QUESTIONS,
    MAX_LLM_RETRIES,
    MAX_GOALTRACKER_SUBTASKS,
    MODEL_ROSTER,
    RETRY_BACKOFF_SECONDS,
    TASK_MODEL_MAP,
)
from exceptions import ResponseTruncatedError

from goal_tracker import (
    Subtask, GoalTracker, decompose, extract_interfaces,
    _extract_file_refs, _partition_pending, _highest_tier,
    _merge_same_file_subtasks, _single_subtask_fallback,
    _build_roster_block, _compute_min_subtasks,
    _merge_verification_tail, _validate_debugging_plan,
    _parse_decompose_json,
)
from sequence import sequence_phase, parse_sequence_json, run_sequence

if TYPE_CHECKING:
    from orchestrator import GenieOrchestrator


# =========================================================================
# Clarify Phase (Phase 5.5)
# =========================================================================

_CLARIFY_SYSTEM_PROMPT = """\
You are the pre-task clarification module for Genie, a desktop automation agent.

## What Genie already is and has — NEVER ask about these

- Genie runs on Ubuntu GNOME (Xorg). OS, desktop, display are all set up.
- Genie has a pre-configured GITHUB_TOKEN. It can create repos (public or private),
  branches, PRs, issues, releases, webhooks, gists — no extra auth needed.
- Genie knows its own GitHub identity (login, email) via get_authenticated_user.
- Genie can run shell commands, write/read files, fetch URLs, control the browser,
  and automate any GUI app on the desktop.
- File paths resolve to ~/genie_workspace/ by default.
- Python 3, Node, git, curl, and common dev tools are available.

## What you must NEVER ask

- Whether Genie has access to GitHub, a token, credentials, or permissions.
- What OS or desktop environment to use.
- Whether Genie can run commands or install packages.
- Anything whose answer is already stated in the task description.
- Hypothetical implementation details the user clearly doesn't care about
  (e.g. "which Python version?" unless version matters to the task).

## What you SHOULD ask

Only questions whose answer would materially change what gets built or how.
Focus on: feature scope, naming choices, visibility/privacy, architectural trade-offs,
output format, or constraints the user hasn't made explicit.

## Scaling rules

- Simple (single file, single command, quick lookup): 1–2 questions MAX — often 0
- Medium (multi-file project, moderate feature): 3–5 questions
- Large (full system, broad spec): 6–10 questions
- Massive (platform, product rewrite): up to 15 questions

If the task is unambiguous, return an empty questions list rather than padding.

Return ONLY a JSON object — no markdown, no explanation:
{"complexity": "simple|medium|large|massive", "questions": [{"question": "<text>", "options": ["1. <opt>", ...]}, ...]}"""


def clarify(orch: "GenieOrchestrator") -> dict:
    """Ask structured clarifying questions before a task begins.

    Reads ``config.CLARIFY_ENABLED`` at call time (not import time) so the
    test harness can set ``config.CLARIFY_ENABLED = False`` without
    reimporting this module.

    Returns a dict ``{"clarifications": [{"question": str, "answer": str}, ...]}``,
    or an empty dict ``{}`` when disabled, on LLM failure, or when no questions
    are generated.  The caller injects the result into ``orch._goal`` before
    ``plan_phase()``.
    """
    if not config.CLARIFY_ENABLED:  # read live attribute, not a cached import
        return {}

    _plan_model    = TASK_MODEL_MAP.get("planning", TASK_MODEL_MAP["default"])[0]
    _plan_fallback = TASK_MODEL_MAP.get("planning", TASK_MODEL_MAP["default"])[1]

    messages = [
        {"role": "system", "content": _CLARIFY_SYSTEM_PROMPT},
        {"role": "user",   "content": f"TASK: {orch._goal}"},
    ]

    response_text = None
    model = _plan_model
    print(f"  [clarify] Generating questions via {model}…", flush=True)
    for attempt in range(1, MAX_LLM_RETRIES + 1):
        try:
            response_text = orch._llm_call(messages, model=model, normalize=False)
            break
        except ResponseTruncatedError as rte:
            response_text = rte.partial_content
            break
        except (httpx.TimeoutException, httpx.RequestError):
            if attempt == MAX_LLM_RETRIES:
                return {}
            time.sleep(RETRY_BACKOFF_SECONDS[min(attempt - 1, len(RETRY_BACKOFF_SECONDS) - 1)])
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in LLM_SERVICE_ERROR_CODES:
                model = _plan_fallback
                try:
                    response_text = orch._llm_call(messages, model=model, normalize=False)
                except Exception:
                    return {}
                break
            return {}

    if not response_text:
        return {}

    parsed = _parse_clarify_json(response_text)
    if not parsed:
        return {}

    # Support both old (list) and new (dict with complexity) formats
    if isinstance(parsed, dict):
        complexity = parsed.get("complexity", "medium")
        questions = parsed.get("questions", [])
    else:
        complexity = "medium"
        questions = parsed

    # Adaptive cap: use complexity-tiered limit instead of flat MAX_CLARIFY_QUESTIONS
    from config import CLARIFY_QUESTION_TIERS
    tier_cap = CLARIFY_QUESTION_TIERS.get(complexity, CLARIFY_QUESTION_TIERS["medium"])
    cap = min(tier_cap, MAX_CLARIFY_QUESTIONS)
    questions = questions[:cap]
    print(f"  [clarify] complexity={complexity}, cap={cap}, keeping {len(questions)} question(s)", flush=True)
    print(f"  [clarify] {len(questions)} question(s) generated", flush=True)
    clarifications = []
    for i, q in enumerate(questions):
        # Bail out early if the task was cancelled while we were showing
        # previous questions (cancel() only unblocks the *current* event).
        if orch.cancel_event.is_set():
            break

        question_text = q.get("question", "")
        options       = q.get("options", [])
        if not question_text:
            continue

        # Reset blocking event for this question
        orch._clarify_event  = threading.Event()
        orch._clarify_answer = None

        print(f"  [clarify] Q{i+1}/{len(questions)}: {question_text[:80]}", flush=True)

        # Telegram: inline keyboard with options + Other…
        # Capture message_id so we can delete it once answered or after timeout.
        msg_id = orch._telegram_notify_keyboard(
            f"[{i + 1}/{len(questions)}] {question_text}", options
        )
        orch._clarify_keyboard_msg_id = msg_id

        # Schedule auto-delete after 5 minutes in case there is no response.
        _delete_timer: threading.Timer | None = None
        if msg_id is not None:
            _delete_timer = threading.Timer(
                300.0, orch._telegram_delete_message, args=(msg_id,)
            )
            _delete_timer.daemon = True
            _delete_timer.start()

        # UI: fire custom event dict directly (bypasses _fire_on_update
        # which only carries standard brain-loop keys)
        if orch._on_update:
            try:
                orch._on_update({
                    "task_id":                orch._task_id,
                    "clarification_question": True,
                    "question":               question_text,
                    "options":                options,
                    "index":                  i,
                    "total":                  len(questions),
                })
            except Exception:
                pass

        # Block indefinitely until answered via either surface
        orch._clarify_event.wait()
        answer = orch._clarify_answer or ""
        orch._clarify_event = None

        # Delete the keyboard message immediately (cancel the 5-min timer too).
        if _delete_timer is not None:
            _delete_timer.cancel()
        if msg_id is not None:
            orch._telegram_delete_message(msg_id)
        orch._clarify_keyboard_msg_id = None

        print(f"  [clarify] A{i+1}: {answer[:80]}", flush=True)

        clarifications.append({"question": question_text, "answer": answer})

    if not clarifications:
        return {}
    return {"clarifications": clarifications, "complexity": complexity}


def _parse_clarify_json(text: str) -> list[dict] | dict | None:
    """Extract and parse the clarifying-questions JSON array or dict.

    Supports both old format (plain JSON array) and new format
    (dict with 'complexity' and 'questions' keys).
    """
    try:
        result = json.loads(text.strip())
        if isinstance(result, list):
            return result
        if isinstance(result, dict) and "questions" in result:
            return result
    except json.JSONDecodeError:
        pass

    m = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        try:
            result = json.loads(m.group(1).strip())
            if isinstance(result, list):
                return result
            if isinstance(result, dict) and "questions" in result:
                return result
        except json.JSONDecodeError:
            pass

    # Try dict (new format) first — it's more specific
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            result = json.loads(m.group(0))
            if isinstance(result, dict) and "questions" in result:
                return result
        except json.JSONDecodeError:
            pass

    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            result = json.loads(m.group(0))
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    return None


# =========================================================================
# Plan Phase
# =========================================================================

def plan_phase(orch: GenieOrchestrator) -> dict | None:
    """Pre-loop LLM call to generate a structured plan.

    Returns plan dict on success, None on failure.
    Sets orch._outcome and orch._summary on failure.
    """
    plan_prompt = (
        f"You are a task planner. Given the following goal, produce a "
        f"structured JSON plan.\n\n"
        f"GOAL: {orch._goal}\n\n"
        f"Return ONLY a JSON object with this exact schema:\n"
        f'{{"goal": "<restate goal>", '
        f'"steps": [{{"n": 1, "description": "..."}}, ...], '
        f'"estimated_iterations": <int>, '
        f'"risks": ["..."]}}\n\n'
        f"Return ONLY the JSON object — no markdown, no explanation."
    )

    messages = [
        {"role": "system", "content": "You are a structured task planner. Return only valid JSON."},
        {"role": "user", "content": plan_prompt},
    ]

    response_text = None
    _plan_model    = TASK_MODEL_MAP.get("planning", TASK_MODEL_MAP["default"])[0]
    _plan_fallback = TASK_MODEL_MAP.get("planning", TASK_MODEL_MAP["default"])[1]
    model = _plan_model

    for attempt in range(1, MAX_LLM_RETRIES + 1):
        try:
            response_text = orch._llm_call(messages, model=model, normalize=False)
            break
        except ResponseTruncatedError as rte:
            response_text = rte.partial_content
            break
        except (httpx.TimeoutException, httpx.RequestError) as exc:
            if attempt == MAX_LLM_RETRIES:
                orch._outcome = "unrecoverable"
                orch._summary = f"Plan phase: LLM network failure after {MAX_LLM_RETRIES} retries: {exc}"
                return None
            time.sleep(RETRY_BACKOFF_SECONDS[min(attempt - 1, len(RETRY_BACKOFF_SECONDS) - 1)])
            continue
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in LLM_SERVICE_ERROR_CODES:
                model = _plan_fallback
                try:
                    response_text = orch._llm_call(messages, model=model, normalize=False)
                except Exception:
                    pass
                if response_text is None:
                    orch._outcome = "unrecoverable"
                    orch._summary = "Plan phase: primary and fallback models failed"
                    return None
                break
            else:
                orch._outcome = "unrecoverable"
                orch._summary = f"Plan phase: HTTP {exc.response.status_code}"
                return None

    if response_text is None:
        orch._outcome = "unrecoverable"
        orch._summary = "Plan phase: no LLM response"
        return None

    plan = parse_plan_json(response_text)

    if plan is None:
        try:
            response_text = orch._llm_call(messages, model=_plan_fallback)
            plan = parse_plan_json(response_text)
        except Exception:
            plan = None

    if plan is None:
        orch._outcome = "unrecoverable"
        orch._summary = "Plan phase: failed to parse valid plan JSON"
        return None

    if not validate_plan_schema(plan):
        reminder_msg = (
            "Your previous response was valid JSON but did not match the "
            "required schema. Required keys: goal (str), steps (list of "
            '{{"n": int, "description": str}}), estimated_iterations (int), '
            "risks (list of str). Try again."
        )
        messages_retry = messages + [
            {"role": "assistant", "content": response_text},
            {"role": "user", "content": reminder_msg},
        ]
        try:
            response_text = orch._llm_call(messages_retry, model=_plan_model)
            plan = parse_plan_json(response_text)
            if plan is not None and not validate_plan_schema(plan):
                plan = None
        except Exception:
            plan = None

        if plan is None:
            try:
                response_text = orch._llm_call(messages, model=_plan_fallback)
                plan = parse_plan_json(response_text)
                if plan is not None and not validate_plan_schema(plan):
                    plan = None
            except Exception:
                plan = None

        if plan is None:
            orch._outcome = "unrecoverable"
            orch._summary = "Plan phase: schema validation failed after retries"
            return None

    return plan


def parse_plan_json(text: str) -> dict | None:
    """Extract and parse JSON from LLM plan response."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    m = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass

    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    return None


def validate_plan_schema(plan: dict) -> bool:
    """Return True if plan has required structure."""
    if not isinstance(plan, dict):
        return False
    if not isinstance(plan.get("goal"), str):
        return False
    steps = plan.get("steps")
    if not isinstance(steps, list) or len(steps) == 0:
        return False
    for step in steps:
        if not isinstance(step, dict):
            return False
        if not isinstance(step.get("n"), int):
            return False
        if not isinstance(step.get("description"), str):
            return False
    if not isinstance(plan.get("estimated_iterations"), int):
        return False
    if not isinstance(plan.get("risks"), list):
        return False
    return True


def display_plan(plan: dict) -> None:
    """Print plan to terminal for interactive approval."""
    print("\n" + "=" * 60)
    print("TASK PLAN")
    print("=" * 60)
    print(f"Goal: {plan.get('goal', '')}")
    print(f"Estimated iterations: {plan.get('estimated_iterations', '?')}")
    print("\nSteps:")
    for step in plan.get("steps", []):
        print(f"  {step.get('n', '?')}. {step.get('description', '')}")
    risks = plan.get("risks", [])
    if risks:
        print("\nRisks:")
        for risk in risks:
            print(f"  - {risk}")
    print("=" * 60 + "\n")



# =========================================================================
# Interactive planning session — multi-turn refinement before execution
# =========================================================================

_PLANNING_SYSTEM_PROMPT = (
    "You are an interactive task planner for Genie, a desktop automation agent.\n"
    "The user will describe a goal, and you will decompose it into subtasks.\n"
    "The user may then send refinement messages asking you to adjust the plan.\n"
    "\n"
    "On EVERY turn you MUST output ONLY a valid JSON object — no markdown, no\n"
    "explanation, no commentary.  The schema is:\n"
    '{"subtasks": [{"n": 1, "description": "...", "model_tier": "tier_0"}, ...]}\n'
    "\n"
    "=== RULES ===\n"
    + "0. SCALE SUBTASK COUNT TO PROJECT SIZE:\n"
    + "   - Small task (1\u20133 output files, single concern): 2\u20135 subtasks\n"
    + "   - Medium project (4\u201310 files, a few distinct concerns): 6\u201312 subtasks\n"
    + "   - Large project (10+ files, multiple layers/modules): 15\u201330 subtasks\n"
    + "   Always end with a verification subtask.\n"
    + "1. SEQUENTIAL BUILD-UP: subtasks run in order, each building on the previous.\n"
    + "2. ZERO REDUNDANCY: every subtask covers DIFFERENT files/features.\n"
    + "   If two subtasks cover the same file, merge them.\n"
    + "3. DESCRIPTIONS are plain-English work orders naming specific files and what\n"
    + "   to implement.  Never paste the user's goal text as a description.\n"
    + "4. FIRST SUBTASK RULE:\n"
    + "   - FRESH BUILD (goal says 'build', 'create', 'write from scratch'): delete any prior-attempt\n"
    + "     directory, then create the entire project skeleton with full module implementations.\n"
    + "   - FIX / IMPROVE / MODIFY (goal says 'fix', 'add', 'update', 'improve', 'modify', 'debug'):\n"
    + "     DO NOT delete existing files. Read the relevant files first, then make targeted edits.\n"
    + "5. LAST SUBTASK \u2014 execute the deliverable and verify output:\n"
    + "   - CLI tool: run it with several subcommands, check stdout\n"
    + "   - Script: execute it and assert correct output\n"
    + "   - Web app: start server, hit endpoints, check responses\n"
    + "   NEVER just read a file or list a directory \u2014 that does not verify behaviour.\n"
    + "   VERIFICATION CAP: combine ALL end-to-end verification steps into ONE\n"
    + "   final subtask. Do NOT split verification across multiple subtasks.\n"
    + "   Include test fixture creation (writing test data files, sample configs,\n"
    + "   etc.) inside this same verification subtask \u2014 do NOT make a separate\n"
    + "   'prepare test environment' subtask.\n"
    + "   This applies to EVERY goal type \u2014 from-scratch builds, debugging, etc.\n"
    + "6. Assign \"model_tier\" per subtask from the specialist roster below.\n"
    + "7. DEBUGGING EXCEPTION: If the goal is to fix bugs in EXISTING code (goal\n"
    + "   says 'fix', 'patch', 'debug', or 'Do NOT delete') \u2014 override Rule 4.\n"
    + "   Do NOT create a 'read and understand' or 'skeleton' first subtask.\n"
    + "   Instead: FIRST subtask fixes Bug 1 (name the specific files to patch).\n"
    + "   ONE BUG PER SUBTASK \u2014 never group multiple independent bugs into one\n"
    + "   subtask. Each subtask must state the exact file(s) it modifies.\n"
    + "   After fixing each bug, include a quick smoke-test (e.g. python -c\n"
    + "   'import module' or python -m py_compile file.py) within the SAME\n"
    + "   subtask to confirm the fix before moving on.\n"
    + "8. SINGLE-FILE RULE: NEVER split implementing a single file across\n"
    + "   multiple subtasks (e.g. 'begin cli.py' + 'complete cli.py'). If a file\n"
    + "   needs new code, allocate ONE subtask for the COMPLETE implementation.\n"
    + "   This applies to EVERY goal type \u2014 from-scratch builds and debugging alike.\n"
    + "9. When the user asks you to modify the plan, incorporate their feedback and\n"
    + "   return the COMPLETE updated subtask list \u2014 not just the changed parts.\n"
    + "\n"
    + "SPECIALIST MODEL ROSTER:\n"
    + _build_roster_block()
)

_CONTINUATION_SYSTEM_PROMPT = (
    "You are a task replanning engine. A multi-step automation task had a subtask\n"
    "failure. Given the original goal, completed subtasks (with files they created),\n"
    "the failed subtask, and the failure reason, produce a REVISED plan for the\n"
    "REMAINING work only.\n"
    "\n"
    "=== CRITICAL RULES ===\n"
    "1. DO NOT redo work that completed subtasks already finished.\n"
    "   The files listed under completed subtasks ALREADY EXIST — build on them.\n"
    "2. Produce AS MANY subtasks as needed to complete ALL remaining work.\n"
    "   Do NOT artificially compress — if 10 modules remain, produce 10 subtasks.\n"
    "   The REMAINING_SUBTASK_ESTIMATE in the user message tells you the scale.\n"
    "3. Each subtask must name the SPECIFIC FILES it will create or modify.\n"
    "4. The revised plan must COMPLETE the original goal, not just retry the\n"
    "   failed subtask. Consider an alternative approach if the same approach\n"
    "   already failed.\n"
    "5. For each subtask, assign a \"model_tier\" from the specialist roster.\n"
    "   Consider using a HIGHER tier than the one that failed.\n"
    "6. The LAST subtask must VERIFY by RUNNING the deliverable (not just\n"
    "   reading code). E.g. 'Run python3 -m myapp --help and test commands'.\n"
    "7. INCOMPLETE WORK RULE: If FAILURE REASON is 'loop_detected', the failed\n"
    "   subtask was INTERRUPTED mid-execution — its work may be PARTIALLY done\n"
    "   or not done at all. Your FIRST new subtask MUST explicitly complete\n"
    "   whatever the failed subtask was supposed to do. Do NOT skip it assuming\n"
    "   it succeeded. Check the FAILED SUBTASK PARTIAL STATE field for clues.\n"
    "8. VENV: Never use 'source venv/bin/activate'. It does not persist between\n"
    "   shell calls. Always invoke the venv's executables directly, e.g.\n"
    "   venv/bin/python, venv/bin/pip, venv/bin/pytest.\n"
    "9. PRESERVATION RULE: NEVER include a subtask that deletes, wipes, or\n"
    "   recreates the project directory. The existing files are the starting\n"
    "   point — work on top of them. A subtask like 'delete directory and\n"
    "   rebuild from scratch' is ALWAYS wrong in a continuation plan.\n"
    "10. READ STALL RULE: If FAILURE REASON is 'read_stall_exceeded', the\n"
    "   agent spent too many iterations reading files without writing changes.\n"
    "   Your FIRST new subtask MUST immediately write or modify code — do NOT\n"
    "   start with reads. Use the TASK MEMORY files section to know what\n"
    "   already exists and where changes are needed.\n"
    "11. SUBTASK CAP RULE: If FAILURE REASON is 'subtask_cap_exceeded', the\n"
    "   failed subtask hit the iteration limit — it was TOO BROAD.\n"
    "   Check TASK MEMORY to see which files were already written.\n"
    "   Split the remaining work into FINE-GRAINED subtasks: at most 1-2 files\n"
    "   modified per subtask. A subtask fixing more than 2 files is too large.\n"
    "   Do NOT produce a continuation subtask that repeats the same broad scope\n"
    "   that just failed.\n"
    "12. VERIFICATION CAP: combine ALL end-to-end verification steps into ONE\n"
    "   final subtask. Do NOT split verification across multiple subtasks.\n"
    "   Include test fixture creation (writing test data files, sample configs,\n"
    "   etc.) inside this same verification subtask. Do NOT make a separate\n"
    "   'prepare test environment' subtask.\n"
    "13. SINGLE-FILE RULE: NEVER split implementing a single file across\n"
    "   multiple subtasks (e.g. 'begin cli.py' + 'complete cli.py'). If a file\n"
    "   needs new code, allocate ONE subtask for the COMPLETE implementation.\n"
    "\n"
    "SPECIALIST MODEL ROSTER:\n"
    + _build_roster_block()
    + "\n\n"
    "Return ONLY a JSON object — no markdown, no explanation:\n"
    '{"subtasks": [{"n": 1, "description": "...", "model_tier": "tier_0"}, ...]}'
)


def _parse_subtask_list(response_text: str) -> list[Subtask] | None:
    """Parse an LLM response into a list of Subtask objects.

    Returns None on parse failure.
    """
    parsed = _parse_decompose_json(response_text)
    if parsed is None or not isinstance(parsed.get("subtasks"), list) or not parsed["subtasks"]:
        return None

    raw_subtasks = parsed["subtasks"][:MAX_GOALTRACKER_SUBTASKS]
    subtasks: list[Subtask] = []
    for i, s in enumerate(raw_subtasks):
        if not isinstance(s, dict):
            continue
        desc = s.get("description", "")
        if not desc:
            continue
        tier = s.get("model_tier", "tier_0")
        if tier not in MODEL_ROSTER:
            tier = "tier_0"
        subtasks.append(Subtask(n=i + 1, description=desc, model_tier=tier))
    return subtasks if subtasks else None


class PlanningSession:
    """Multi-turn interactive planning conversation.

    Owns a local message history separate from the brain loop.  Each
    refinement round appends the user message + previous draft as context,
    calls the LLM, and produces a new List[Subtask] draft.
    """

    def __init__(self, orch: "GenieOrchestrator", goal: str) -> None:
        self._orch = orch
        self._goal = goal
        self._plan_model = TASK_MODEL_MAP.get("planning", TASK_MODEL_MAP["default"])[0]
        self._plan_fallback = TASK_MODEL_MAP.get("planning", TASK_MODEL_MAP["default"])[1]

        # Planning conversation history — never shared with brain loop.
        self._messages: list[dict[str, str]] = [
            {"role": "system", "content": _PLANNING_SYSTEM_PROMPT},
        ]

        # Build workspace snapshot (same as decompose)
        workspace_snapshot = ""
        try:
            _ws = os.path.expanduser("~/genie_workspace")
            _entries: list[str] = []
            for root, dirs, files in os.walk(_ws):
                dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
                rel = os.path.relpath(root, _ws)
                for f in files:
                    _entries.append(os.path.join(rel, f) if rel != "." else f)
                if len(_entries) > 60:
                    break
            if _entries:
                workspace_snapshot = (
                    "\n\nEXISTING WORKSPACE (~/genie_workspace/):\n  "
                    + "\n  ".join(_entries[:60])
                    + ("\n  ... (truncated)" if len(_entries) > 60 else "")
                    + "\n\nWARNING \u2014 existing files detected in the workspace:\n"
                    "- FRESH BUILD goal (build/create/write from scratch): your FIRST subtask MUST "
                    "'rm -rf' the old project directory before creating anything new.\n"
                    "- FIX/MODIFY/IMPROVE goal (fix/add/update/debug/improve): DO NOT delete any files. "
                    "These files ARE the project. Work on top of them."
                )
        except OSError:
            pass

        self.min_subtasks, self.max_subtasks = _compute_min_subtasks(
            goal, getattr(orch, "_ask_user_complexity", "medium")
        )
        if self.max_subtasks:
            budget_hint = (
                f"\n\nEXECUTOR BUDGET: {config.MAX_ITERATIONS_PER_SUBTASK} iterations per subtask. "
                "Split the work so each subtask comfortably fits within that budget. "
                f"This goal requires EXACTLY {self.min_subtasks} subtasks — no more, no fewer."
            )
        else:
            budget_hint = (
                f"\n\nEXECUTOR BUDGET: {config.MAX_ITERATIONS_PER_SUBTASK} iterations per subtask. "
                "Split the work so each subtask comfortably fits within that budget. "
                f"This goal requires AT LEAST {self.min_subtasks} subtasks — do not produce fewer."
            )

        self._messages.append({
            "role": "user",
            "content": f"GOAL: {goal}{workspace_snapshot}{budget_hint}",
        })

        self._current_draft: list[Subtask] | None = None

    def generate_draft(self) -> list[Subtask] | None:
        """Call the LLM and produce a draft subtask list.

        Returns the parsed subtask list, or None on failure.
        """
        response_text = self._llm_call()
        if not response_text:
            return None

        subtasks = _parse_subtask_list(response_text)
        if subtasks is None:
            return None

        # Enforce minimum 2 subtasks
        if len(subtasks) == 1:
            subtasks.append(Subtask(
                n=2,
                description=(
                    "Verify the deliverable by executing it with run_command — do NOT just read_file. "
                    "Run every command or entry point described in the original goal, inspect stdout/stderr, "
                    "and confirm correct output. If a run_command returns a non-zero exit_code, "
                    "use write_file to fix the code and re-run until all commands succeed. "
                    "Only call done when every verification run_command exits 0."
                ),
                model_tier="tier_0",
            ))

        # -- Enforce computed minimum subtask count (Fix 2 A+B) -------------------
        if len(subtasks) < self.min_subtasks:
            print(
                f"  [planning] Only {len(subtasks)} subtask(s) but min={self.min_subtasks} "
                "— retrying with hard constraint",
                flush=True,
            )
            constraint_msg = (
                f"Your previous decomposition produced only {len(subtasks)} subtask(s). "
                f"This goal requires AT LEAST {self.min_subtasks} subtasks. "
                "Re-decompose with at least that many — one module or feature boundary per subtask."
            )
            retry_messages = list(self._messages) + [
                {"role": "assistant", "content": response_text},
                {"role": "user", "content": constraint_msg},
            ]
            try:
                retry_text = self._orch._llm_call(
                    retry_messages, model=self._plan_model, normalize=False
                )
                if retry_text:
                    retry_subtasks = _parse_subtask_list(retry_text)
                    if retry_subtasks and len(retry_subtasks) > len(subtasks):
                        print(
                            f"  [planning] Retry produced {len(retry_subtasks)} subtask(s)",
                            flush=True,
                        )
                        response_text = retry_text
                        subtasks = retry_subtasks
            except Exception as _e:
                print(f"  [planning] Retry failed: {_e}", flush=True)

        # Record the assistant response for conversation continuity
        self._messages.append({"role": "assistant", "content": response_text})
        # Enforce max_subtasks cap for debug/fix tasks
        if self.max_subtasks and len(subtasks) > self.max_subtasks:
            print(f"  [planning] Trimming {len(subtasks)} subtasks to max={self.max_subtasks}", flush=True)
            subtasks = subtasks[:self.max_subtasks]
        # Merge verification-only tail
        subtasks = _merge_verification_tail(self._goal, subtasks)
        # Enforce SINGLE-FILE RULE: merge subtasks that touch the same file
        subtasks = _merge_same_file_subtasks(subtasks)
        # Validate debugging plan structure (same checks as decompose)
        subtasks = _validate_debugging_plan(
            self._goal, subtasks, self._messages, response_text,
            self._plan_model, self._orch
        )
        self._current_draft = subtasks
        return subtasks

    def refine(self, user_message: str) -> list[Subtask] | None:
        """Send a refinement message and produce a new draft.

        Appends the user message to the planning conversation history,
        calls the LLM, and returns the new draft.
        """
        self._messages.append({"role": "user", "content": user_message})
        return self.generate_draft()

    @property
    def current_draft(self) -> list[Subtask] | None:
        return self._current_draft

    def _llm_call(self) -> str | None:
        """Make an LLM call with retry/fallback logic."""
        model = self._plan_model
        for attempt in range(1, MAX_LLM_RETRIES + 1):
            try:
                return self._orch._llm_call(
                    self._messages, model=model, normalize=False,
                )
            except ResponseTruncatedError as rte:
                return rte.partial_content
            except (httpx.TimeoutException, httpx.RequestError):
                if attempt == MAX_LLM_RETRIES:
                    print("  [planning] LLM network failure", flush=True)
                    return None
                time.sleep(RETRY_BACKOFF_SECONDS[min(attempt - 1, len(RETRY_BACKOFF_SECONDS) - 1)])
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code in LLM_SERVICE_ERROR_CODES:
                    model = self._plan_fallback
                    try:
                        return self._orch._llm_call(
                            self._messages, model=model, normalize=False,
                        )
                    except Exception:
                        return None
                return None
        return None


def run_planning_session(
    orch: "GenieOrchestrator",
    goal: str,
) -> PlanningSession:
    """Create and initialise a PlanningSession, generating draft 1.

    Returns the PlanningSession object. The caller reads
    ``session.current_draft`` for the initial subtask list and can
    call ``session.refine(msg)`` for subsequent rounds.
    """
    session = PlanningSession(orch, goal)
    session.generate_draft()
    return session


def generate_continuation_draft(
    orch: "GenieOrchestrator",
    tracker: GoalTracker,
    failure_reason: str,
) -> list[Subtask] | None:
    """Generate a single continuation-plan draft after a subtask failure.

    Uses scope-limited continuation: pending subtasks unrelated to the
    failure are preserved verbatim.  The LLM only replans the failed
    subtask + any pending subtasks that share files with it.

    Returns a List[Subtask] (replanned + preserved) or None on failure.
    """
    _plan_model = TASK_MODEL_MAP.get("planning", TASK_MODEL_MAP["default"])[0]
    _plan_fallback = TASK_MODEL_MAP.get("planning", TASK_MODEL_MAP["default"])[1]

    failed_st = tracker.subtasks[tracker.current_index]

    # ---- Scope-limited continuation: partition pending subtasks ----
    pending_subtasks = [
        s for s in tracker.subtasks[tracker.current_index + 1:]
        if s.status == "pending"
    ]
    affected, unaffected = _partition_pending(failed_st, pending_subtasks)

    _affected_desc = "; ".join(
        f"ST{s.n}({','.join(_extract_file_refs(s.description)) or '?'})"
        for s in affected
    )
    _unaffected_desc = "; ".join(
        f"ST{s.n}({','.join(_extract_file_refs(s.description)) or '?'})"
        for s in unaffected
    )
    print(
        f"  [goaltracker] Continuation scope: affected=[{_affected_desc}] "
        f"preserved=[{_unaffected_desc}]",
        flush=True,
    )

    # Compute remaining work estimate for the planning model.
    _done_count = sum(1 for s in tracker.subtasks if s.status == "done")
    _original_total = len(tracker.subtasks)
    # Only count affected + failed as needing replan
    _replan_estimate = max(1, len(affected) + 1)  # +1 for the failed subtask itself

    # Include partial-state clues for loop_detected failures so the
    # replanning LLM knows what work may be incomplete.
    _partial_state = ""
    if failed_st.result_summary:
        _partial_state += f"\nFAILED SUBTASK RESULT SUMMARY: {failed_st.result_summary}"

    # ---- P1-E: Negative memory from structured handoffs ----
    _handoff_block = ""
    _do_not_retry_cmds: list[str] = []
    try:
        _scratchpad = getattr(orch, '_scratchpad', None)
        if _scratchpad is not None:
            _handoffs = getattr(_scratchpad, 'handoffs', [])
            if _handoffs:
                _hlines: list[str] = []
                for _h in _handoffs:
                    _sn = _h.get('subtask_n', '?')
                    _hs = _h.get('status', '?')
                    _hd = _h.get('description', '')
                    _hm = _h.get('handoff_message', '')
                    _hf = _h.get('commands_failed', [])
                    _hw = _h.get('files_written', [])
                    _hlines.append(
                        f"  ST {_sn} ({_hs}): {_hd}"
                        + (f"\n    Files written: {', '.join(_hw[:8])}" if _hw else "")
                        + (f"\n    Failed commands: {'; '.join(_hf[:5])}" if _hf else "")
                        + (f"\n    Handoff: {_hm[:300]}" if _hm else "")
                    )
                    _do_not_retry_cmds.extend(_hf[:5])
                _handoff_block = "\nRECENT SUBTASK HANDOFFS:\n" + "\n".join(_hlines)
    except Exception:
        pass

    _do_not_retry_block = ""
    if _do_not_retry_cmds:
        # Deduplicate while preserving order
        _seen: set[str] = set()
        _unique_cmds: list[str] = []
        for _c in _do_not_retry_cmds:
            if _c not in _seen:
                _seen.add(_c)
                _unique_cmds.append(_c)
        _do_not_retry_block = (
            "\n\nDO NOT RETRY THESE FAILED APPROACHES:\n"
            + "\n".join(f"  - {c}" for c in _unique_cmds[:10])
        )

    _continuation_depth = f"\nCONTINUATION ATTEMPT: #{tracker.continuation_count + 1} of 3 maximum."
    if tracker.continuation_count >= 1:
        _continuation_depth += " Previous continuation attempts also failed — try a SUBSTANTIALLY different approach."

    # ---- Build prompt: scope LLM to only the affected portion ----
    # Show the original pending plan so LLM can anchor on existing decisions.
    _original_pending_block = ""
    if pending_subtasks:
        _pending_lines = []
        for s in pending_subtasks:
            _label = "AFFECTED — must replan" if s in affected else "PRESERVED — do not include"
            _pending_lines.append(f"  ST{s.n} [{_label}]: {s.description}")
        _original_pending_block = (
            "\n\nORIGINAL REMAINING PLAN (from before this failure):\n"
            + "\n".join(_pending_lines)
            + "\n\nSubtasks marked PRESERVED will be automatically kept — "
            "do NOT include them in your output. Only replan the AFFECTED "
            "subtasks and the failed subtask."
        )

    _affected_block = ""
    if affected:
        _aff_lines = [f"  ST{s.n}: {s.description}" for s in affected]
        _affected_block = (
            "\n\nAFFECTED PENDING SUBTASKS (share files with the failed subtask — "
            "must be replanned):\n" + "\n".join(_aff_lines)
        )

    user_content = (
        f"ORIGINAL GOAL: {tracker.original_goal}\n\n"
        f"ORIGINAL PLAN SIZE: {_original_total} subtasks total\n"
        f"COMPLETED: {_done_count} subtasks done\n"
        f"REPLAN SCOPE: {_replan_estimate} subtasks need replanning "
        f"({len(unaffected)} subtasks will be auto-preserved)\n"
        f"{_continuation_depth}\n\n"
        f"COMPLETED SUBTASKS:\n{tracker.completed_context()}\n\n"
        f"FAILED SUBTASK {failed_st.n}: {failed_st.description}\n"
        f"FAILURE REASON: {failure_reason}{_partial_state}"
        f"{_affected_block}"
        f"{_handoff_block}"
        f"{_original_pending_block}"
        f"{_do_not_retry_block}\n\n"
        f"Produce a revised list of subtasks to replace ONLY the failed subtask"
        f" and the {len(affected)} affected subtask(s) listed above.\n"
        f"You need AT LEAST {_replan_estimate} subtasks — do not produce fewer."
    )

    messages = [
        {"role": "system", "content": _CONTINUATION_SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

    response_text = None
    model = _plan_model
    print(f"  [goaltracker] Generating continuation draft via {model}…", flush=True)

    for attempt in range(1, MAX_LLM_RETRIES + 1):
        try:
            response_text = orch._llm_call(messages, model=model, normalize=False)
            break
        except ResponseTruncatedError as rte:
            response_text = rte.partial_content
            break
        except (httpx.TimeoutException, httpx.RequestError):
            if attempt == MAX_LLM_RETRIES:
                return None
            time.sleep(RETRY_BACKOFF_SECONDS[min(attempt - 1, len(RETRY_BACKOFF_SECONDS) - 1)])
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in LLM_SERVICE_ERROR_CODES:
                model = _plan_fallback
                try:
                    response_text = orch._llm_call(messages, model=model, normalize=False)
                except Exception:
                    return None
                break
            return None

    if not response_text:
        return None

    replanned = _parse_subtask_list(response_text)
    if not replanned:
        return None

    # ---- Post-LLM deterministic enforcement ----

    # 1. Merge same-file subtasks (SINGLE-FILE RULE enforcement)
    replanned = _merge_same_file_subtasks(replanned)

    # 2. Merge verification tail
    replanned = _merge_verification_tail(tracker.original_goal, replanned)

    # 3. Validate debugging plan structure (same checks as initial decompose)
    replanned = _validate_debugging_plan(
        tracker.original_goal, replanned, messages, response_text, model, orch,
    )

    # 4. Splice: replanned first, then preserved (unaffected) in original order
    final = list(replanned) + list(unaffected)

    # 5. Renumber starting after the failed subtask
    base_n = failed_st.n + 1
    for i, s in enumerate(final):
        s.n = base_n + i

    if unaffected:
        print(
            f"  [goaltracker] Continuation: {len(replanned)} replanned + "
            f"{len(unaffected)} preserved = {len(final)} total remaining subtasks",
            flush=True,
        )
    return final
