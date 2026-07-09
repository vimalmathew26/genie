"""
Genie — orchestrator.py
Layer 4 ReAct Brain Loop / GenieOrchestrator.

Drives autonomous task execution. Calls the LLM, parses responses,
dispatches actions, captures observations, manages context history,
enforces budgets, and handles error recovery.
"""
from __future__ import annotations

import collections.abc
import dataclasses
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request, urllib.parse
import uuid
from collections import Counter, deque

from websockets.sync.client import connect as _ws_connect

import httpx
import psutil

import actions
import batch_engine as batch_mod
import context_builder
import dispatch as dispatch_mod
import persistence
import prefetch as prefetch_mod
import planner as planner_mod
import response_parser
import script_planner as script_planner_mod
from scratchpad_writer import ScratchpadWriter
from task_scratchpad import TaskScratchpad
from config import (
    ACTION_IDEMPOTENT,
    APP_PROFILES,
    ARGS_TRUNCATION_CHARS,
    CHECKPOINT_PATH,
    ERROR_CLASS_ENVIRONMENTAL,
    ERROR_CLASS_RESOURCE,
    ERROR_CLASS_TRANSIENT,
    ERROR_CLASS_UNRECOVERABLE,
    ERROR_CLASSIFICATION_RULES,
    LLM_SERVICE_ERROR_CODES,
    LOG_DIR,
    LOOP_DETECT_THRESHOLD,
    LOOP_DETECT_WINDOW,
    SCHEMA_ERROR_HALT_THRESHOLD,
    MAX_ACTION_RETRIES,
    MAX_ITERATIONS_PER_SUBTASK,
    MAX_ITERATIONS_PER_TASK,
    MAX_LLM_RETRIES,
    MODEL_ROSTER,
    MONTHLY_BUDGET_CAP,
    REACT_HISTORY_WINDOW,
    READ_STALL_THRESHOLD,
    RETRY_BACKOFF_SECONDS,
    TASK_BUDGET_DEFAULTS,
    TASK_MODEL_MAP,
    CLASSIFIER_MODEL,
    CLASSIFY_SYSTEM_PROMPT,
    CLASSIFIABLE_TASK_TYPES,
    COMPLEXITY_CLASSIFIER_SYSTEM_PROMPT,
    COMPLEXITY_CLASSIFIABLE_TIERS,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
    SCRIPT_PLANNER_ENABLED,
    TELEGRAM_POLL_TIMEOUT,
    TELEGRAM_PROGRESS_INTERVAL,
    VERIFICATION_ENABLED,
    MAX_VERIFY_FIX_ATTEMPTS,
    log,
)
from element_resolver import ElementResolver
from exceptions import (
    EnvironmentalError,
    ResourceError,
    ResponseTruncatedError,
    SchemaValidationError,
    TransientError,
    UnrecoverableError,
)
from llm_client import LLMClient
from observation import Observer
from review_queue import ReviewEntry, ReviewQueue
from verifier import SubtaskVerifier
from window_registry import WindowRegistry
from xdotool_controller import XdotoolController




# =========================================================================
# TaskResult dataclass
# =========================================================================

@dataclasses.dataclass
class TaskResult:
    task_id: str
    outcome: str       # "done"|"abort"|"budget_exceeded"|"iteration_exceeded"|"unrecoverable"|"cancelled"
    summary: str
    iterations: int
    cost_usd: float



from system_prompt import SYSTEM_PROMPT


# =========================================================================
# Script planner gate — whitelist of shell-only subtask patterns
# =========================================================================

# Verbs that indicate the subtask's primary deliverable is a shell operation,
# not authored file content.  Script planner is ONLY allowed for these.
_SHELL_ONLY_VERBS = re.compile(
    r'^\s*(?:'
    r'install[\s,]|'
    r'set\s+up\s+(?:the\s+)?(?:virtual\s+env|venv|environment|dependencies|packages)|'
    r'clone[\s,]|'
    r'run[\s,]|'
    r'execute[\s,]|'
    r'launch[\s,]|'
    r'start[\s,]|'
    r'stop[\s,]|'
    r'restart[\s,]|'
    r'kill[\s,]|'
    r'verify[\s,]|'
    r'validate[\s,]|'
    r'check[\s,]|'
    r'test\s+that[\s,]|'
    r'confirm[\s,]|'
    r'delete[\s,]|'
    r'remove[\s,]|'
    r'clean[\s,]|'
    r'download[\s,]|'
    r'fetch[\s,]|'
    r'git[\s,]|'
    r'pip[\s,]|'
    r'npm[\s,]|'
    r'apt[\s,]'
    r')',
    re.IGNORECASE,
)

# Verbs that indicate file/code authoring — script planner must NOT handle these.
_AUTHORING_VERBS = re.compile(
    r'(?:'
    r'\bimplement\b|'
    r'\bwrite\b|'
    r'\bdefine\b|'
    r'\bbuild\b|'
    r'\badd\s+(?:a\s+)?(?:class|function|method|module|endpoint|route|handler|logic|support)|'
    r'\bcreate\b.*\.py\b|'
    r'\bfinalize\b|'
    r'\bcomplete\b.*\.py\b|'
    r'\bfill\s+in\b|'
    r'\bcode\b|'
    r'\bprogram\b|'
    r'\bcompose\b|'
    r'\bwire\s+up\b|'
    r'\bintegrate\b|'
    r'\bdesign\b'
    r')',
    re.IGNORECASE,
)


def _is_shell_only_subtask(description: str) -> bool:
    """Return True only when the subtask is purely a shell operation.

    The script planner is an optimisation for tasks like 'install deps',
    'run pytest', 'clone repo'.  It must never handle subtasks that require
    authoring file content — those must go to _brain_loop.

    Gate logic:
      - If the description contains any authoring verb → False (brain loop).
      - Else if the description starts with a recognised shell verb → True.
      - Otherwise → False (brain loop, the safe default).
    """
    if _AUTHORING_VERBS.search(description):
        return False
    if _SHELL_ONLY_VERBS.match(description):
        return True
    return False


# =========================================================================
# GenieOrchestrator
# =========================================================================

class GenieOrchestrator:
    """Layer 4 ReAct brain loop orchestrator."""

    def __init__(self, controller: XdotoolController) -> None:
        self.observer = Observer(log_dir=LOG_DIR)
        self.registry = WindowRegistry()
        self.element_resolver = ElementResolver(self.registry, controller)
        actions.init(controller)

        self.cancel_event = threading.Event()
        self.pause_event = threading.Event()
        self.pause_event.set()  # not paused by default — set means "running"
        self._telegram_stop_event = threading.Event()
        self._telegram_update_offset: int = 0
        self._monthly_budget_warning_sent: bool = False
        self._review_queue = ReviewQueue()

        self._act_hashes: deque[str] = deque(maxlen=LOOP_DETECT_WINDOW)
        self._last_recorded_action: str = ""
        self._files_written_this_subtask: set[str] = set()  # paths written in current subtask — exempt from loop detection

        self._llm = LLMClient()
        self._scratchpad_writer = ScratchpadWriter()

        # Per-task state (reset each run_task)
        self._mode: str = "interactive"
        self._on_update = None
        self._task_id: str = ""
        self._goal: str = ""
        self._task_type: str = "default"
        self._task_budget: float = 0.0
        self._task_cost_usd: float = 0.0
        self._monthly_cost_usd: float = 0.0
        self._halt_reason: str | None = None
        self._outcome: str | None = None
        self._summary: str = ""
        self._iteration: int = 0
        self._model: str = ""
        self._fallback_model: str = ""
        self._history: deque = deque(maxlen=REACT_HISTORY_WINDOW)
        self._last_obs: dict | None = None
        self._telegram_thread: threading.Thread | None = None
        self._consecutive_chat_count: int = 0
        self._consecutive_env_failures: int = 0  # escalation ladder counter
        self._consecutive_schema_errors: int = 0  # halt after SCHEMA_ERROR_HALT_THRESHOLD in a row
        self._consecutive_command_failures: int = 0  # P0-B: run-command stall

        # Clarify phase (Phase 5.5) — reset each run_task
        self._clarify_event: threading.Event | None = None
        self._clarify_answer: str | None = None
        self._clarify_awaiting_freetext: bool = False
        self._clarify_keyboard_msg_id: int | None = None  # message_id of active keyboard Q
        self._clarify_freetext_prompt_msg_id: int | None = None  # message_id of "Type your answer:"
        self._clarify_freetext_delete_timer: threading.Timer | None = None
        self._plan_confirm_event: threading.Event | None = None
        self._plan_confirm_approved: bool | None = None

        # GoalTracker (Phase 5.3) — reset each run_task
        self._goaltracker: planner_mod.GoalTracker | None = None
        self._original_goal: str = ""

        # TaskScratchpad — shared working memory (Phase 6)
        self._scratchpad: TaskScratchpad = TaskScratchpad()
        self._scratchpad_miss_count: int = 0
        self._last_done_handoff: str = ""  # P1-D: brain model's handoff msg
        self._subtask_failed_commands: list[str] = []  # P1-C: deduped failed cmds
        self._warmstart_handoff: dict | None = None  # Fix 2C: carried across reset
        self._done_bounce_count: int = 0  # Case C: soft bounce for zero-write dones
        self._stall_nudge_active: bool = False  # Fix D: post-stall read warning
        self._productive_done_bounce: int = 0  # Fix B: bounce for 0-productive done

        # Per-subtask extra API params (e.g. reasoning.budget_tokens for nothink)
        self._extra_body: dict | None = None

        # Cached CDP URLs captured just before cleanup kills browser processes.
        # Keyed by registry label (e.g. "chrome", "chrome_1").
        # Validators can read these even after Chrome is terminated.
        self._last_cdp_urls: dict[str, str] = {}

        # Cached registry snapshot captured just before cleanup clears the
        # registry.  Validators can read this after run_task() returns even
        # though the live registry is empty.
        self._last_registry_snapshot: dict[str, dict] = {}

        # Workspace cache — maps resolved absolute file path → current content.
        # Persists across subtask boundaries so subtask N+1 sees files written
        # by subtask N without any read_file calls.  Injected into every LLM
        # message via context_builder; never compressed or truncated.
        self._workspace_cache: dict[str, str] = {}

        # Smart prefetch — project files loaded once, injected per-subtask
        self._project_file_cache: dict[str, str] = {}   # ALL project files (not rendered)
        self._injected_project_files: set[str] = set()   # tracks which are in workspace_cache
        self._project_dir: str = ""                       # project root path

        # Estimated iterations — populated from plan phase, sent in events
        self._estimated_iterations: int | None = None

        # Plan reentry — used when a subtask fails and the UI re-enters
        # planning state for interactive replanning.
        self._plan_reentry_event: threading.Event | None = None
        self._plan_reentry_result: list | None = None

    # =====================================================================
    # Loop Detection (copied verbatim from GenieAgent in genie.py)
    # =====================================================================

    def get_last_chrome_url(self) -> str | None:
        """Return the last cached Chrome tab URL captured before cleanup.

        Searches ``_last_cdp_urls`` for any entry whose key contains
        'chrome'.  Returns the first match, or None.
        """
        for label, url in self._last_cdp_urls.items():
            if "chrome" in label:
                return url
        return None

    @staticmethod
    def _hash_act(act: dict) -> str:
        """Compute a deterministic hash of an ACT decision."""
        canonical = json.dumps(
            {"action": act.get("action"), "args": act.get("args")},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode()).hexdigest()

    def record_act(self, act: dict) -> None:
        """Record a fresh LLM-generated ACT decision for loop detection.

        Loop detection exists to catch the LLM retrying the same FAILING
        side-effect action (run_command, write_file, etc.) repeatedly.

        Read-only / observation actions (read_file, list_dir, think) are
        NEVER recorded here — they are harmless, idempotent, and already
        governed by READ_STALL_THRESHOLD which is the correct mechanism
        for "too many observations without progress".

        Exceptions (not counted toward loop detection):
        1. write_file/append_file immediately followed by run_command:
           write→verify cycle; the command is progress, not a loop.
        2. write_file/append_file: adds the target path to
           _files_written_this_subtask for bookkeeping.
        """
        action_name = act.get("action", "")

        # --- Read-only actions: never part of loop detection ---------------
        # READ_STALL_THRESHOLD already handles "only reading, not writing".
        if action_name in ("read_file", "list_dir", "think"):
            self._last_recorded_action = action_name
            return

        # Track writes so bookkeeping stays correct.
        if action_name in ("write_file", "append_file"):
            path = act.get("args", {}).get("path") or act.get("args", {}).get("file_path", "")
            if path:
                self._files_written_this_subtask.add(path)

        # Exempt: run_command immediately after a write (write→verify cycle).
        if action_name == "run_command" and self._last_recorded_action in (
            "write_file", "append_file",
        ):
            self._last_recorded_action = action_name
            return

        self._act_hashes.append(self._hash_act(act))
        self._last_recorded_action = action_name

    def is_loop_detected(self) -> bool:
        """Return True when the agent appears stuck in a loop."""
        if len(self._act_hashes) < LOOP_DETECT_THRESHOLD:
            return False
        counts = Counter(self._act_hashes)
        return counts.most_common(1)[0][1] >= LOOP_DETECT_THRESHOLD

    def reset_loop_detector(self) -> None:
        """Clear loop-detection state."""
        self._act_hashes.clear()

    # =====================================================================
    # Public interface
    # =====================================================================

    def cancel(self) -> None:
        """Set cancel event — brain loop exits at next iteration top.

        Also unblocks any pending UI-wait events (plan reentry, plan confirm,
        clarify question) so the task thread is never left stuck indefinitely
        — the "Cancelling…" state resolves promptly regardless of what the
        task was waiting for.
        """
        self.cancel_event.set()
        # Unblock pending plan-reentry wait
        if self._plan_reentry_event is not None and not self._plan_reentry_event.is_set():
            self._plan_reentry_result = None
            self._plan_reentry_event.set()
        # Unblock pending plan-confirm wait
        if self._plan_confirm_event is not None and not self._plan_confirm_event.is_set():
            self._plan_confirm_approved = False
            self._plan_confirm_event.set()
        # Unblock pending clarify-question wait
        if self._clarify_event is not None and not self._clarify_event.is_set():
            self._clarify_answer = ""
            self._clarify_event.set()

    def answer_clarification(self, answer: str) -> None:
        """Accept an answer to the current clarifying question.

        Called from both the Telegram listener and the GTK UI button handler.
        Guards against double-answer (event already set) and stale calls
        (no clarification in progress).
        """
        if self._clarify_event is None or self._clarify_event.is_set():
            return
        self._clarify_answer = answer
        self._clarify_event.set()

    def _block_for_ask_user(self, question: str, options: list) -> str:
        """Block the brain loop mid-task to ask the user a question.

        Reuses the _clarify_event / answer_clarification() mechanism.
        Called from batch_engine when the LLM emits an ask_user action.
        Returns the user's answer string.
        """
        self._clarify_event = threading.Event()
        self._clarify_answer = None

        print(f"  [ask_user] Q: {question[:80]}", flush=True)

        # Telegram: keyboard with options + Other…
        msg_id = self._telegram_notify_keyboard(question, options)
        self._clarify_keyboard_msg_id = msg_id

        # Schedule auto-delete after 5 minutes
        _delete_timer: threading.Timer | None = None
        if msg_id is not None:
            _delete_timer = threading.Timer(
                300.0, self._telegram_delete_message, args=(msg_id,)
            )
            _delete_timer.daemon = True
            _delete_timer.start()

        # UI: fire clarification_question event so the GTK window renders it
        if self._on_update:
            try:
                self._on_update({
                    "task_id":                self._task_id,
                    "clarification_question": True,
                    "question":               question,
                    "options":                options,
                    "index":                  0,
                    "total":                  1,
                })
            except Exception:
                pass

        # Block until answered via Telegram or UI
        self._clarify_event.wait()
        answer = self._clarify_answer or ""
        self._clarify_event = None

        # Clean up Telegram message
        if _delete_timer is not None:
            _delete_timer.cancel()
        if msg_id is not None:
            self._telegram_delete_message(msg_id)
        self._clarify_keyboard_msg_id = None

        print(f"  [ask_user] A: {answer[:80]}", flush=True)
        return answer

    def answer_plan_confirm(self, approved: bool) -> None:
        """Called from the UI thread when user approves or rejects the plan.

        Unblocks run_task() which is waiting on _plan_confirm_event.
        No-op if no confirmation is pending.
        """
        if self._plan_confirm_event is None or self._plan_confirm_event.is_set():
            return
        self._plan_confirm_approved = approved
        self._plan_confirm_event.set()

    def answer_plan_reentry(self, subtasks: list | None) -> None:
        """Called from the UI thread when user approves a continuation plan.

        ``subtasks`` is the approved List[Subtask], or None to cancel.
        Unblocks _run_goaltracker_loop() which is waiting on
        _plan_reentry_event.
        """
        if self._plan_reentry_event is None or self._plan_reentry_event.is_set():
            return
        self._plan_reentry_result = subtasks
        self._plan_reentry_event.set()

    # -----------------------------------------------------------------
    # Phase 5.4: One-shot task-type classifier
    # -----------------------------------------------------------------
    def _classify_task(self, goal: str) -> str:
        """Send the goal through a cheap one-shot LLM call to pick a task_type.

        Returns one of the CLASSIFIABLE_TASK_TYPES strings.  Falls back to
        'default' on any error or unexpected output.
        """
        messages = [
            {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
            {"role": "user", "content": goal},
        ]
        try:
            raw, cost_usd = self._llm.call(
                messages, model=CLASSIFIER_MODEL, max_tokens=32,
            )
            self._task_cost_usd += cost_usd
            self._monthly_cost_usd += cost_usd
            if cost_usd > 0:
                self._write_monthly_cost()
            label = raw.strip().lower().rstrip(".")
            if label in CLASSIFIABLE_TASK_TYPES:
                print(f"[ORCHESTRATOR] auto-classified task_type = {label}")
                return label
            # LLM returned something unexpected — fall back
            print(
                f"[ORCHESTRATOR] classifier returned unknown label '{label}', "
                f"falling back to 'default'",
                file=sys.stderr,
            )
            return "default"
        except Exception as exc:
            print(
                f"[ORCHESTRATOR] classifier error: {exc}, "
                f"falling back to 'default'",
                file=sys.stderr,
            )
            return "default"

    def _classify_complexity(self) -> str:
        """One-shot LLM call to pick a MODEL_ROSTER complexity tier for the goal.

        Returns one of the COMPLEXITY_CLASSIFIABLE_TIERS strings.
        Falls back to 'tier_0' on any error or unexpected output.
        """
        messages = [
            {"role": "system", "content": COMPLEXITY_CLASSIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": self._goal},
        ]
        try:
            raw, cost_usd = self._llm.call(
                messages, model=CLASSIFIER_MODEL, max_tokens=16,
            )
            self._task_cost_usd += cost_usd
            self._monthly_cost_usd += cost_usd
            if cost_usd > 0:
                self._write_monthly_cost()
            label = raw.strip().lower()
            if label in COMPLEXITY_CLASSIFIABLE_TIERS:
                print(f"[ORCHESTRATOR] complexity tier = {label}")
                return label
            print(
                f"[ORCHESTRATOR] complexity classifier returned unknown tier '{label}', "
                f"falling back to 'tier_0'",
                file=sys.stderr,
            )
            return "tier_0"
        except Exception as exc:
            print(
                f"[ORCHESTRATOR] complexity classifier error: {exc}, "
                f"falling back to 'tier_0'",
                file=sys.stderr,
            )
            return "tier_0"

    def run_task(
        self,
        goal: str | None = None,
        mode: str = "interactive",
        task_type: str = "default",
        per_task_budget: float | None = None,
        on_update=None,
        checkpoint: dict | None = None,
        skip_plan: bool = False,
        auto_confirm: bool = False,
        review: bool = False,
        max_revisions: int = 2,
        poll_interval: float = 5.0,
        use_goaltracker: bool = False,
        pre_built_plan: list | None = None,
        plan_min_subtasks: int = 0,
        plan_max_subtasks: int | None = None,
    ) -> TaskResult:
        """Blocking call — runs plan phase + brain loop, returns TaskResult.

        skip_plan: when True, bypass the plan LLM call and confirmation gate
        entirely. Use for trivial/short goals where the overhead is not justified.

        auto_confirm: when True and skip_plan is False, skip the plan phase and
        interactive approval gate, going straight to brain loop.  Useful for
        browser/interactive tasks that need adaptive execution but should not
        require human confirmation (e.g. automated test runs).

        review: when True, after a successful task (outcome=="done"), start the
        Telegram review bot in a background thread, send a notification to your
        phone, and wait for your approve/reject tap. On reject with feedback,
        re-runs the task with feedback appended (up to max_revisions times).
        When False (default), returns immediately — no Telegram interaction.

        use_goaltracker: when True, decompose the goal into subtasks via LLM
        and run each subtask through its own brain loop iteration.  Subtask
        failures surface for interactive replanning.

        pre_built_plan: when provided (and use_goaltracker is True), skip the
        decompose LLM call and construct the GoalTracker directly from this
        pre-approved subtask list.
        """
        if review:
            return self._run_task_with_review(
                goal=goal, mode=mode, task_type=task_type,
                per_task_budget=per_task_budget, on_update=on_update,
                checkpoint=checkpoint, skip_plan=skip_plan,
                max_revisions=max_revisions, poll_interval=poll_interval,
                use_goaltracker=use_goaltracker,
            )

        # -- Validate task_type -----------------------------------------------
        if task_type == "auto":
            task_type = self._classify_task(goal or "")
        if task_type not in TASK_MODEL_MAP:
            print(
                f"[ORCHESTRATOR] unknown task_type '{task_type}', "
                f"falling back to 'default'",
                file=sys.stderr,
            )
            task_type = "default"

        # -- Resolve checkpoint vs fresh start --------------------------------
        if checkpoint is not None:
            self._task_id = checkpoint["task_id"]
            self._goal = checkpoint["goal"]
            self._task_type = checkpoint.get("task_type", "default")
            self._task_budget = checkpoint.get(
                "per_task_budget",
                TASK_BUDGET_DEFAULTS.get(self._task_type, 2.0),
            )
            self._iteration = checkpoint.get("iteration", 0)
            self._task_cost_usd = checkpoint.get("cost_usd", 0.0)
            initial_sequence = checkpoint.get("sequence", 0)
            resume = True
            _task_start_time: float = time.time()
        else:
            if goal is None:
                raise ValueError("goal is required when no checkpoint provided")
            self._task_id = str(uuid.uuid4())
            self._goal = goal
            self._task_type = task_type
            self._task_budget = (
                per_task_budget
                if per_task_budget is not None
                else TASK_BUDGET_DEFAULTS.get(task_type, 2.0)
            )
            self._iteration = 0
            self._task_cost_usd = 0.0
            initial_sequence = 0
            resume = False
            _task_start_time: float = time.time()

        self._mode = mode
        self._on_update = on_update
        self._halt_reason = None
        self._outcome = None
        self._summary = ""
        self._history.clear()
        self.reset_loop_detector()
        self._last_recorded_action = ""
        self._files_written_this_subtask = set()
        self._consecutive_chat_count = 0
        self._consecutive_env_failures = 0
        self._consecutive_schema_errors = 0
        self._consecutive_readonly = 0
        self._consecutive_command_failures = 0
        self._subtask_has_written = False  # stall detection only fires after first write_file/append_file
        self._subtask_productive_iters = 0  # only mutating actions count toward cap
        self._task_file_read_counts: dict[str, int] = {}  # per-file read counter (persists across subtasks)
        self._read_stall_warned = False  # ensures warning fires once even if counter jumps
        self._last_written_path = ""  # Fix P0-write-reread: track last written path
        self._post_write_reread_count = 0  # Fix P0-write-reread: consecutive re-reads of just-written file
        self._subtask_retry_count = 0  # Fix P2: retry-before-replan counter
        self._last_done_handoff = ""
        self._subtask_failed_commands = []
        self._stall_nudge_active = False  # Fix D: track active stall nudge
        self._productive_done_bounce = 0  # Fix B: bounce counter for 0-productive done
        self.cancel_event.clear()
        self.pause_event.set()  # not paused
        self._prefetched_content = ""  # populated by _prefetch_goal_urls
        self._fetch_url_called = False    # set True when fetch_url runs (dispatch or prefetch)
        self._fetch_domain_counts: dict[str, int] = {}  # per-domain fetch_url counter
        self._clarify_event = None
        self._clarify_answer = None
        self._clarify_awaiting_freetext = False
        self._ask_user_count: int = 0
        self._ask_user_complexity: str = "medium"
        self._plan_confirm_event = None
        self._plan_confirm_approved = None
        self._goaltracker = None
        self._original_goal = ""
        self._workspace_cache = {}  # clear between tasks — never carry over stale files
        self._project_file_cache = {}
        self._injected_project_files = set()
        self._project_dir = ""
        self._scratchpad = TaskScratchpad()
        self._scratchpad_miss_count = 0
        self._last_done_handoff = ""
        self._subtask_failed_commands = []

        # -- Model selection ---------------------------------------------------
        models = TASK_MODEL_MAP.get(self._task_type, TASK_MODEL_MAP["default"])
        self._model = models[0]
        self._fallback_model = models[1]

        # -- Monthly cost loading ---------------------------------------------
        self._load_monthly_cost()

        # -- Observer start ----------------------------------------------------
        self.observer.start_task(self._task_id, initial_sequence=initial_sequence)

        # -- Telegram listener (autonomous only) ------------------------------
        if self._mode == "autonomous":
            self._telegram_stop_event.clear()
            self._telegram_thread = threading.Thread(
                target=self._telegram_listener, daemon=True,
            )
            self._telegram_thread.start()

        try:
            # -- Resume path: synthetic last observation -----------------------
            if resume:
                # Restore scratchpad from checkpoint if present
                _sp_data = checkpoint.get("scratchpad")
                if _sp_data:
                    self._scratchpad = TaskScratchpad.from_dict(_sp_data)
                else:
                    self._scratchpad = TaskScratchpad()
                self._scratchpad_miss_count = 0

                if checkpoint.get("goaltracker"):
                    # GoalTracker resume — reconstruct tracker, skip done subtasks
                    tracker = planner_mod.GoalTracker.from_dict(checkpoint["goaltracker"])
                    self._goaltracker = tracker
                    self._original_goal = self._goal
                    self._run_goaltracker_loop(tracker)
                else:
                    self._last_obs = {
                        "result": None,
                        "observation": {
                            "resume_notice": (
                                f"Task resumed from checkpoint at iteration "
                                f"{self._iteration}. No prior observation available."
                            ),
                        },
                        "error": None,
                    }
                    self._brain_loop()
            else:
                self._last_obs = None

                # -- Pre-fetch URLs from goal text (deterministic) --------
                self._prefetch_goal_urls()

                # -- Phase 5.5 — clarifying questions (fresh starts only) ------
                # Skip when a pre_built_plan was already approved via the
                # interactive planning session — the user already reviewed and
                # approved the decomposition so extra questions are redundant
                # and would silently block the task thread.
                if pre_built_plan is None:
                    clarifications = planner_mod.clarify(self)
                    if clarifications.get("clarifications"):
                        block = "\n\nClarifications:\n" + "".join(
                            f"- Q: {c['question']}\n  A: {c['answer']}\n"
                            for c in clarifications["clarifications"]
                        )
                        self._goal += block
                    # Store complexity for mid-task ask_user budget
                    self._ask_user_complexity = clarifications.get("complexity", "medium")

                # -- Script planner goal-level fast-path (Phase 6) ----------------
                # If the entire goal can be scripted in one shot, skip GoalTracker
                # entirely and execute directly.
                if SCRIPT_PLANNER_ENABLED and use_goaltracker:
                    _goal_planner = script_planner_mod.ScriptPlanner(
                        llm_client=self._llm, model=self._model,
                    )
                    _goal_segment = _goal_planner.plan(self._goal)
                    if not _goal_segment.fallback_react:
                        _goal_handled = self._try_script_planner(
                            self._goal, subtask_n=0,
                        )
                        if _goal_handled:
                            use_goaltracker = False

                # -- GoalTracker path (Phase 5.3) -----------------------------
                if use_goaltracker:
                    if pre_built_plan is not None:
                        # Plan already built and approved via interactive
                        # planning session — skip decompose() entirely.
                        tracker = planner_mod.GoalTracker(
                            original_goal=self._goal,
                            subtasks=pre_built_plan,
                            min_subtasks=plan_min_subtasks,
                            max_subtasks=plan_max_subtasks,
                        )
                    else:
                        tracker = planner_mod.decompose(self)
                    self._goaltracker = tracker
                    self._original_goal = self._goal

                    # Confirmation gate — same mechanism as plan confirm -----
                    # Skip if pre_built_plan was provided (already approved
                    # in the interactive planning session).
                    if self._mode == "interactive" and pre_built_plan is None:
                        subtask_plan = {
                            "goal": tracker.original_goal,
                            "steps": [
                                {"n": s.n, "description": s.description}
                                for s in tracker.subtasks
                            ],
                            "estimated_iterations": len(tracker.subtasks) * MAX_ITERATIONS_PER_SUBTASK,
                            "risks": [],
                        }
                        self._plan_confirm_event = threading.Event()
                        self._plan_confirm_approved = None

                        def _stdin_confirm_gt() -> None:
                            try:
                                confirm = input("Approve subtask plan? (y/n): ").strip().lower()
                                self.answer_plan_confirm(confirm == "y")
                            except (EOFError, OSError):
                                pass

                        threading.Thread(target=_stdin_confirm_gt, daemon=True).start()

                        if self._on_update:
                            try:
                                self._on_update({
                                    "task_id": self._task_id,
                                    "plan_confirm": True,
                                    "plan": subtask_plan,
                                })
                            except Exception:
                                pass

                        self._plan_confirm_event.wait()
                        approved = self._plan_confirm_approved
                        self._plan_confirm_event = None
                        if not approved:
                            self.observer.end_task()
                            return TaskResult(
                                task_id=self._task_id,
                                outcome="cancelled",
                                summary="User rejected subtask plan",
                                iterations=0,
                                cost_usd=self._task_cost_usd,
                            )

                    self._estimated_iterations = len(tracker.subtasks) * MAX_ITERATIONS_PER_SUBTASK
                    self._run_goaltracker_loop(tracker)

                else:
                    # -- Complexity classification (non-GoalTracker path) ----
                    tier = self._classify_complexity()
                    if tier in MODEL_ROSTER:
                        self._model = MODEL_ROSTER[tier]["model_id"]
                        self._fallback_model = MODEL_ROSTER[tier]["fallback"]
                        self._extra_body = MODEL_ROSTER[tier].get("extra_body")
                        print(
                            f"[ORCHESTRATOR] complexity routing: tier={tier}, "
                            f"model={self._model}"
                        )

                    # -- Plan phase (fresh start only) ------------------------
                    if skip_plan:
                        # Skip sequence phase when URLs were already pre-fetched into
                        # history. The sequence model has no context of the fetched
                        # content (it would time out or produce a stale plan). Brain
                        # loop already has the fetched data in _history/_last_obs.
                        if self._prefetched_content:
                            self._brain_loop()
                        else:
                            # Trivial goal fast-path: attempt a single-shot sequence
                            # call that returns the full action list. If it succeeds,
                            # execute all actions without further LLM round-trips
                            # (O(1) LLM cost instead of O(N)). On failure, fall
                            # through to brain loop.
                            sequence, seq_llm_response, seq_llm_messages = self._sequence_phase()
                            if sequence is not None:
                                task_done = self._run_sequence(
                                    sequence,
                                    llm_response=seq_llm_response,
                                    llm_messages=seq_llm_messages,
                                )
                                if not task_done:
                                    # Sequence hit an environmental error mid-way.
                                    # Brain loop resumes from populated _history + _last_obs.
                                    self._brain_loop()
                            else:
                                # Sequence phase failed (LLM error / bad JSON / schema).
                                # Fall back to the full ReAct brain loop.
                                self._brain_loop()
                    elif auto_confirm:
                        # Brain-loop fast-path: skip both sequence and plan phases,
                        # go straight to adaptive brain loop.  Used for browser /
                        # interactive tasks where the sequence path is too brittle
                        # and human plan approval isn't needed.
                        self._brain_loop()
                    else:
                        plan = self._plan_phase()
                        if plan is None:
                            # Plan failed — TaskResult already set by _plan_phase
                            return TaskResult(
                                task_id=self._task_id,
                                outcome=self._outcome or "unrecoverable",
                                summary=self._summary or "Plan phase failed",
                                iterations=0,
                                cost_usd=self._task_cost_usd,
                            )

                        # Plan confirmation — interactive mode only.
                        # In "batch" mode (test harness, API callers) the plan is
                        # auto-approved so the harness never blocks on stdin.
                        if self._mode == "interactive":
                            self._display_plan(plan)  # keep — prints to terminal as log
                            self._plan_confirm_event = threading.Event()
                            self._plan_confirm_approved = None

                            # Terminal path: always spawn a daemon thread so the
                            # user can type y/n in the terminal regardless of
                            # whether the GTK UI is also showing the prompt.
                            def _stdin_confirm() -> None:
                                try:
                                    confirm = input("Approve plan? (y/n): ").strip().lower()
                                    self.answer_plan_confirm(confirm == "y")
                                except (EOFError, OSError):
                                    pass

                            threading.Thread(target=_stdin_confirm, daemon=True).start()

                            # UI path: also fire the event if a UI callback exists.
                            # Whichever source (terminal or UI) responds first wins;
                            # answer_plan_confirm() is guarded against double-set.
                            if self._on_update:
                                try:
                                    self._on_update({
                                        "task_id": self._task_id,
                                        "plan_confirm": True,
                                        "plan": plan,
                                    })
                                except Exception:
                                    pass

                            self._plan_confirm_event.wait()
                            approved = self._plan_confirm_approved
                            self._plan_confirm_event = None
                            if not approved:
                                self.observer.end_task()
                                return TaskResult(
                                    task_id=self._task_id,
                                    outcome="cancelled",
                                    summary="User rejected plan",
                                    iterations=0,
                                    cost_usd=self._task_cost_usd,
                                )

                        self._estimated_iterations = plan.get("estimated_iterations")
                        self.observer.observe_plan(plan)
                        self._brain_loop()

        finally:
            self._cleanup_opened_apps()
            self.observer.end_task()
            self._increment_task_count()
            self._write_monthly_cost()

            # Telegram listener shutdown
            if self._telegram_thread is not None:
                self._telegram_stop_event.set()
                self._telegram_thread.join(timeout=15)
                self._telegram_thread = None

            # Scratchpad disk cleanup (always — scratchpad lives in checkpoint)
            TaskScratchpad.clear_disk()

            # ── §3.3 Review Queue: enqueue completed tasks ───────────
            if self._outcome == "done":
                try:
                    self._review_queue.enqueue(ReviewEntry(
                        task_id=self._task_id,
                        goal=self._goal,
                        summary=self._summary,
                        output_files=self._collect_output_files(),
                        cost_usd=self._task_cost_usd,
                        iterations=self._iteration,
                        self_assessment=self._summary,
                    ))
                except Exception as exc:
                    print(
                        f"[ORCHESTRATOR] review queue write failed: {exc}",
                        file=sys.stderr,
                    )

        persistence.append_task_log(
            task_id=self._task_id,
            goal=self._goal,
            task_type=self._task_type,
            model=self._model,
            outcome=self._outcome or "unrecoverable",
            iterations=self._iteration,
            cost_usd=self._task_cost_usd,
            wall_time_s=time.time() - _task_start_time,
            scratchpad_misses=self._scratchpad_miss_count,
        )
        return TaskResult(
            task_id=self._task_id,
            outcome=self._outcome or "unrecoverable",
            summary=self._summary or "Unknown outcome",
            iterations=self._iteration,
            cost_usd=self._task_cost_usd,
        )

    # =====================================================================
    # §3.3 Review Helpers
    # =====================================================================

    def _collect_output_files(self) -> list[str]:
        """Scan history for write_file / done actions and collect output paths.

        Returns a deduplicated list of file paths that Genie wrote during
        the task. Best-effort — returns [] if nothing is found.
        """
        paths: list[str] = []
        seen: set[str] = set()
        for entry in self._history:
            # History entries from make_history_entry use flat keys:
            #   {"action": "write_file", "args": {"path": ...}, ...}
            action_name = entry.get("action", "")
            args = entry.get("args") or {}
            if action_name == "write_file":
                p = args.get("path", "")
                if p and p not in seen:
                    paths.append(p)
                    seen.add(p)
        return paths

    def _run_task_with_review(
        self,
        goal: str | None,
        max_revisions: int = 2,
        poll_interval: float = 5.0,
        use_goaltracker: bool = False,
        **run_kwargs,
    ) -> TaskResult:
        """Internal: run task + Telegram review loop with auto-revision."""
        import review_bot as _review_bot

        # Start Telegram bot in background thread for this task's lifetime
        _bot_stop = threading.Event()
        _bot_thread = threading.Thread(
            target=_review_bot.run_bot_in_thread,
            args=(_bot_stop,),
            daemon=True,
            name="review-bot",
        )
        _bot_thread.start()

        try:
            current_goal = goal
            total_cost = 0.0
            total_iterations = 0

            for revision in range(max_revisions + 1):
                result = self.run_task(
                    goal=current_goal, review=False,
                    use_goaltracker=use_goaltracker,
                    **run_kwargs,
                )
                total_cost += result.cost_usd
                total_iterations += result.iterations

                if result.outcome != "done":
                    return result

                task_id = result.task_id
                print(f"[REVIEW] Task {task_id} queued for review. "
                      f"Waiting for Telegram decision...")

                while True:
                    entry = self._review_queue.get_by_task_id(task_id)
                    if entry is None:
                        time.sleep(poll_interval)
                        continue
                    status = entry.get("status", "pending")
                    if status in ("approved", "delivered"):
                        print(f"[REVIEW] Task {task_id} APPROVED ✅")
                        return TaskResult(
                            task_id=task_id,
                            outcome="done",
                            summary=result.summary + " [APPROVED]",
                            iterations=total_iterations,
                            cost_usd=total_cost,
                        )
                    elif status == "rejected":
                        feedback = entry.get("feedback", "")
                        print(f"[REVIEW] Task {task_id} REJECTED ❌ — "
                              f"feedback: {feedback[:200]}")
                        if revision < max_revisions:
                            current_goal = (
                                f"{goal}\n\n"
                                f"--- REVISION REQUEST (attempt {revision + 2}) ---\n"
                                f"Previous output was rejected. Feedback:\n"
                                f"{feedback}\n"
                                f"Please revise and address all feedback."
                            )
                            self._review_queue.update_status(
                                task_id, "revision_pending",
                            )
                            print(f"[REVIEW] Starting revision {revision + 2}...")
                            break
                        else:
                            print(f"[REVIEW] Max revisions ({max_revisions}) reached.")
                            return TaskResult(
                                task_id=task_id,
                                outcome="rejected",
                                summary=result.summary + f" [REJECTED after {max_revisions + 1} attempts]",
                                iterations=total_iterations,
                                cost_usd=total_cost,
                            )
                    else:
                        time.sleep(poll_interval)

            return result  # type: ignore[possibly-undefined]
        finally:
            _bot_stop.set()
            _bot_thread.join(timeout=5)

    # =====================================================================
    # Plan / Sequence (delegated to planner module)
    # =====================================================================

    def _plan_phase(self) -> dict | None:
        """Pre-loop LLM call to generate a structured plan."""
        return planner_mod.plan_phase(self)

    @staticmethod
    def _display_plan(plan: dict) -> None:
        """Pretty-print a plan dict."""
        planner_mod.display_plan(plan)

    def _sequence_phase(self) -> tuple[list[dict] | None, str | None, list | None]:
        """Speed path: ask LLM to convert plan into an executable sequence."""
        return planner_mod.sequence_phase(self)

    def _run_sequence(self, sequence: list[dict],
                      llm_response: str | None = None,
                      llm_messages: list | None = None) -> bool:
        """Execute a validated action sequence without re-prompting the LLM."""
        return planner_mod.run_sequence(self, sequence,
                                        llm_response=llm_response,
                                        llm_messages=llm_messages)

    # =====================================================================
    # CDP utility — read active tab URL (best-effort, no exceptions)
    # =====================================================================

    @staticmethod
    def _cdp_read_active_tab_url(cdp_port: int, timeout: float = 3.0) -> str | None:
        """Return the URL of the active Chrome page tab, or None on any failure.

        Prefers real web pages (http/https) over Chrome internal pages
        (chrome://omnibox-popup, chrome://newtab, devtools://, etc.)
        which may appear as ``type=="page"`` targets.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(
                    f"http://localhost:{cdp_port}/json", timeout=2
                ) as resp:
                    tabs = json.loads(resp.read().decode())
                pages = [t for t in tabs if t.get("type") == "page"]
                # Prefer http(s) pages over chrome:// internal pages
                web_pages = [
                    p for p in pages
                    if p.get("url", "").startswith("http")
                ]
                if web_pages:
                    return web_pages[0].get("url", "")
                # Fall back to any page target
                if pages:
                    return pages[0].get("url", "")
            except Exception:
                time.sleep(0.3)
        return None

    @staticmethod
    def _cdp_navigate_nth_serp_result(cdp_port: int, n: int = 0) -> str | None:
        """Use CDP to find and navigate Chrome to the Nth (0-based) external
        search result on the current page.  Returns the target URL on success,
        or None if fewer than N+1 results found or an error occurred.

        Uses the same multi-engine JS as element_resolver._SERP_RESULTS_JS,
        but in a standalone form (no import dependency on ElementResolver).
        Supports Google, Bing, DuckDuckGo, and generic fallback.
        """
        JS = ElementResolver._SERP_RESULTS_JS

        try:
            with urllib.request.urlopen(
                f"http://localhost:{cdp_port}/json", timeout=3
            ) as resp:
                tabs = json.loads(resp.read().decode())
            pages = [t for t in tabs if t.get("type") == "page"]
            if not pages:
                return None
            # Prefer http(s) pages over chrome:// internal pages
            web_pages = [p for p in pages if p.get("url", "").startswith("http")]
            ws_url = (web_pages or pages)[0]["webSocketDebuggerUrl"]
        except Exception:
            return None

        try:
            with _ws_connect(ws_url) as ws:
                # Step 1 — evaluate JS to get all result URLs
                ws.send(json.dumps({"id": 1, "method": "Runtime.evaluate",
                                    "params": {"expression": JS,
                                               "returnByValue": True}}))
                raw = ws.recv(timeout=6)
                resp_obj = json.loads(raw)
                results_json = (
                    resp_obj.get("result", {})
                             .get("result", {})
                             .get("value", "") or ""
                )
                if not results_json:
                    return None
                urls = json.loads(results_json)
                if not isinstance(urls, list) or n >= len(urls):
                    return None
                target_url = urls[n]
                if not target_url or not target_url.startswith("http"):
                    return None

                # Step 2 — navigate to the result URL
                ws.send(json.dumps({"id": 2, "method": "Page.navigate",
                                    "params": {"url": target_url}}))
                deadline = time.time() + 8
                while time.time() < deadline:
                    raw = ws.recv(timeout=max(0.5, deadline - time.time()))
                    msg = json.loads(raw)
                    if msg.get("id") == 2:
                        break
                return target_url
        except Exception:
            return None

    def _chrome_cdp_port(self) -> int | None:
        """Return the CDP port for any open Chrome window from the registry."""
        with self.registry._registry_lock:
            for entry in self.registry._registry.values():
                port = entry.get("cdp_port")
                if port:
                    return port
        return None

    def _dismiss_chrome_restore_popup(self, wid: str) -> None:
        """Dismiss Chrome's 'Restore pages?' crash-recovery dialog.

        Chrome renders this dialog entirely within the browser window frame
        (visible in vision captures, blocks AT-SPI tree reads) on first open
        after an unclean shutdown.  Sending Escape via xdotool dismisses it;
        if no dialog is present the keystroke harmlessly clears URL-bar focus.

        We wait 1.5 s first so Chrome has time to finish rendering.  The WID
        is passed explicitly to avoid relying on ambient keyboard focus.
        """
        try:
            time.sleep(1.5)
            subprocess.run(
                ["xdotool", "key", "--window", str(wid), "Escape"],
                capture_output=True,
                timeout=3,
            )
            time.sleep(0.3)
            log(f"_dismiss_chrome_restore_popup: sent Escape to WID {wid}")
        except Exception as exc:
            log(f"_dismiss_chrome_restore_popup: ignored error — {exc}")

    # =====================================================================
    # Batch Engine (delegated to batch_engine.py)
    # =====================================================================

    @staticmethod
    def _resolve_placeholder_from_history(
        content: str, history: collections.abc.Iterable, last_result: dict | None = None,
    ) -> str:
        return batch_mod.resolve_placeholder_from_history(content, history, last_result)

    def _execute_batch(self, action_list: list[dict], iteration: int, llm_messages=None, llm_response=None) -> str:
        return batch_mod.execute_batch(self, action_list, iteration, llm_messages=llm_messages, llm_response=llm_response)


    # =====================================================================
    # Brain Loop
    # =====================================================================

    # -----------------------------------------------------------------
    # Post-task cleanup
    # -----------------------------------------------------------------

    def _cleanup_opened_apps(self) -> None:
        """Terminate every application that Genie opened during this task.

        Iterates the window registry, sends SIGTERM (then SIGKILL after a
        short grace period) to each tracked process, and clears the registry
        so the next task starts with a clean slate.  Errors are swallowed —
        cleanup must never prevent TaskResult from being returned.

        Before killing CDP-capable apps (Chrome, Electron), caches their
        last active-tab URL in ``self._last_cdp_urls`` so validators can
        read the final browser state even after the process is gone.
        """
        # -- Snapshot CDP URLs before killing anything -------------------------
        with self.registry._registry_lock:
            entries = list(self.registry._registry.items())

        # -- Snapshot full registry for validators that run after cleanup ------
        self._last_registry_snapshot = {
            label: dict(entry) for label, entry in entries
        }

        for label, entry in entries:
            cdp_port = entry.get("cdp_port")
            if cdp_port is not None:
                try:
                    url = self._cdp_read_active_tab_url(cdp_port, timeout=2.0)
                    if url:
                        self._last_cdp_urls[label] = url
                except Exception:
                    pass

        # -- Kill processes ----------------------------------------------------
        for label, entry in entries:
            proc = entry.get("process")
            wid = entry.get("wid")
            closed = False

            # -- Try killing via Popen handle first ----------------------------
            if proc is not None:
                try:
                    if proc.poll() is None:
                        # Still alive — terminate normally
                        proc.terminate()
                        try:
                            proc.wait(timeout=3)
                        except Exception:
                            proc.kill()
                            try:
                                proc.wait(timeout=2)
                            except Exception:
                                pass
                        closed = True
                except Exception:
                    pass

            # -- Fallback: forked apps (eog, gedit, etc.) exit the wrapper
            #    immediately so proc.poll() is not None.  Close via WID. ------
            if not closed and wid is not None:
                try:
                    subprocess.run(
                        ["xdotool", "windowclose", str(wid)],
                        capture_output=True, timeout=3,
                    )
                    time.sleep(0.5)
                    # If windowclose didn't work, try killing the WID's PID
                    # but ONLY if the PID belongs to the expected app (safety
                    # guard — avoid killing unrelated processes like the test
                    # runner or VS Code).
                    result = subprocess.run(
                        ["xdotool", "getwindowpid", str(wid)],
                        capture_output=True, text=True, timeout=2,
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        _pid = int(result.stdout.strip())
                        # Verify the PID is NOT our own process tree
                        if _pid != os.getpid() and _pid != os.getppid():
                            try:
                                _p = psutil.Process(_pid)
                                _pname = _p.name().lower()
                                # Only kill if the process looks like a
                                # GUI app (not python, bash, code, etc.)
                                _safe_to_kill = not any(
                                    t in _pname
                                    for t in ("python", "bash", "sh",
                                              "code", "node", "copilot")
                                )
                                if _safe_to_kill:
                                    os.kill(_pid, 9)
                            except (psutil.NoSuchProcess, ProcessLookupError):
                                pass
                    closed = True
                except Exception:
                    pass

            if closed:
                log(f"post-task cleanup: closed {label} (PID {proc.pid if proc else '?'}, WID {wid})")
            else:
                log(f"post-task cleanup: could not close {label} (PID {proc.pid if proc else '?'}, WID {wid})")


        # Clear registry so next task starts fresh
        self.registry.clear()

    # =====================================================================
    # GoalTracker Loop (Phase 5.3)
    # =====================================================================

    def _run_goaltracker_loop(self, tracker: planner_mod.GoalTracker) -> None:
        """Execute subtasks sequentially, each through its own brain loop.

        Restores ``self._goal`` to ``self._original_goal`` on all exit paths
        via try/finally.
        """
        # Save the initial model so we can restore after the loop
        _saved_model = self._model
        _saved_fallback = self._fallback_model
        _saved_extra_body = self._extra_body

        try:
            # -- Pre-load existing project files into workspace_cache ----------
            # For debugging/patching tasks on existing projects, this eliminates
            # the 20-40 iteration read_file exploration phase by injecting all
            # source files into LLM context from iteration 1.
            self._prefetch_project_files()

            while True:
                subtask = tracker.next_subtask()
                if subtask is None:
                    break

                tracker.mark_running(subtask)

                # -- Phase 5.4: per-subtask model swap -------------------------
                tier_info = MODEL_ROSTER.get(subtask.model_tier)
                if tier_info:
                    self._model = tier_info["model_id"]
                    self._fallback_model = tier_info.get("fallback", _saved_fallback)
                    self._extra_body = tier_info.get("extra_body")  # e.g. nothink for tier_1
                    print(
                        f"  [goaltracker] Subtask {subtask.n}/{len(tracker.subtasks)}"
                        f" → {subtask.model_tier} ({self._model})",
                        flush=True,
                    )
                else:
                    # Unknown tier — fall back to task-level model
                    self._model = _saved_model
                    self._fallback_model = _saved_fallback
                    self._extra_body = _saved_extra_body

                self._fire_on_update(
                    0, None, None,
                    subtask_n=subtask.n,
                    subtask_description=subtask.description,
                    subtask_total=len(tracker.subtasks),
                )
                completed_ctx = tracker.completed_context()
                self._goal = (
                    f"SUBTASK {subtask.n}/{len(tracker.subtasks)}: {subtask.description}"
                    f"\n\nORIGINAL USER GOAL (the FULL spec — every requirement listed here "
                    f"must be satisfied before the final subtask calls done):\n"
                    f"{self._original_goal}"
                    f"\n\nPRIOR SUBTASK CONTEXT (what already exists — do NOT redo):\n"
                    f"{completed_ctx}"
                )
                self._reset_for_subtask()
                self._inject_subtask_files(self._goal)  # smart prefetch: inject goal-relevant files
                self._pre_generate_greenfield_draft(subtask)  # pre-generate stubs → implementations
                self._scratchpad.set_subtask(subtask.n)
                self._prefetch_goal_urls()
                self._in_goaltracker_subtask = True
                # Track per-subtask iteration cap
                subtask_start_iter = self._iteration
                try:
                    # -- Script planner fast-path (Phase 6) --------------------
                    # Only use script planner for pure shell tasks (install,
                    # clone, run, verify-by-running, etc.).  Any subtask that
                    # requires authoring file content must go to _brain_loop.
                    if SCRIPT_PLANNER_ENABLED and _is_shell_only_subtask(subtask.description):
                        _sp_handled = self._try_script_planner(
                            subtask.description,
                        )
                        if not _sp_handled:
                            self._brain_loop()
                    else:
                        self._brain_loop()
                finally:
                    self._in_goaltracker_subtask = False

                # -- Per-subtask iteration cap check --
                subtask_iters = self._iteration - subtask_start_iter
                if (
                    MAX_ITERATIONS_PER_SUBTASK > 0
                    and subtask_iters >= MAX_ITERATIONS_PER_SUBTASK
                    and self._outcome not in ("done", None)
                ):
                    print(
                        f"  [goaltracker] Subtask {subtask.n} hit per-subtask"
                        f" cap ({subtask_iters}/{MAX_ITERATIONS_PER_SUBTASK})",
                        flush=True,
                    )

                if self._outcome == "done":
                    # Capture files written during this subtask
                    subtask.files_written = self._collect_output_files()

                    # -- Post-subtask verification with optional fix retries --
                    _is_final = (subtask.n == len(tracker.subtasks))
                    _verify_attempt = 0
                    _verify_passed = True  # assume pass (fail-open default)
                    # Derive project_dir from files_written — use the
                    # deepest common directory of written files so the
                    # verifier scans only the relevant project, not the
                    # entire WORKSPACE_DIR (which may contain unrelated
                    # projects and their venvs).
                    _fw = subtask.files_written or []
                    if _fw:
                        _project_dir = os.path.commonpath(_fw)
                        if os.path.isfile(_project_dir):
                            _project_dir = os.path.dirname(_project_dir)
                    else:
                        _project_dir = None  # fallback to WORKSPACE_DIR
                    while _verify_attempt <= MAX_VERIFY_FIX_ATTEMPTS:
                        try:
                            _verifier = SubtaskVerifier(project_dir=_project_dir)
                            _vresults = _verifier.verify(
                                files_written=subtask.files_written,
                                is_final=_is_final,
                            )
                            # Collect all failing results
                            _failed = [r for r in _vresults if not r.passed]
                            if not _failed:
                                _verify_passed = True
                                if _verify_attempt > 0:
                                    print(
                                        f"  [verifier] Subtask {subtask.n} "
                                        f"issues fixed after {_verify_attempt} attempt(s)",
                                        flush=True,
                                    )
                                break  # all checks passed

                            # Log the issues
                            for _vr in _failed:
                                print(
                                    f"  [verifier] Subtask {subtask.n} "
                                    f"{_vr.level}: {len(_vr.issues)} issue(s)"
                                    f" (attempt {_verify_attempt}/{MAX_VERIFY_FIX_ATTEMPTS})",
                                    flush=True,
                                )
                                for _vi in _vr.issues[:5]:
                                    print(
                                        f"    - {_vi.file}: {_vi.message}",
                                        flush=True,
                                    )
                            # Log non-failing results that had internal errors
                            for _vr in _vresults:
                                if _vr.passed and _vr.error:
                                    print(
                                        f"  [verifier] Subtask {subtask.n} "
                                        f"{_vr.level}: {_vr.error}",
                                        flush=True,
                                    )

                            # If retries exhausted OR verification disabled, proceed
                            if (
                                _verify_attempt >= MAX_VERIFY_FIX_ATTEMPTS
                                or not VERIFICATION_ENABLED
                            ):
                                _verify_passed = False
                                print(
                                    f"  [verifier] Subtask {subtask.n} "
                                    f"retries exhausted — proceeding (fail-open)",
                                    flush=True,
                                )
                                break

                            # -- Retry: re-enter brain_loop with fix hints --
                            _verify_attempt += 1
                            _fix_hints = "\n".join(r.fix_hint for r in _failed if r.fix_hint)
                            print(
                                f"  [verifier] Subtask {subtask.n} "
                                f"re-entering brain_loop (attempt {_verify_attempt}/"
                                f"{MAX_VERIFY_FIX_ATTEMPTS}) to fix issues",
                                flush=True,
                            )
                            # Minimal state reset — only clear outcome so
                            # brain_loop re-enters.  Do NOT call
                            # _reset_for_subtask() — keep history, iteration
                            # counter, files_written, all other state intact.
                            self._outcome = None
                            self._halt_reason = None
                            self._summary = ""
                            # Inject fix hints into goal context
                            self._goal += (
                                f"\n\n--- VERIFICATION FIX REQUEST "
                                f"(attempt {_verify_attempt}/{MAX_VERIFY_FIX_ATTEMPTS}) ---\n"
                                f"The previous done was rejected by automated verification.\n"
                                f"Fix the following issues, then call done again:\n\n"
                                f"{_fix_hints}\n"
                                f"\nIMPORTANT: Fix ONLY these specific issues. "
                                f"Do not rewrite files that are already correct."
                            )
                            # Re-enter brain_loop
                            self._in_goaltracker_subtask = True
                            try:
                                self._brain_loop()
                            finally:
                                self._in_goaltracker_subtask = False
                            # Re-collect files after fix attempt
                            subtask.files_written = self._collect_output_files()

                        except Exception as _ve:
                            print(
                                f"  [verifier] Subtask {subtask.n} crashed "
                                f"(fail-open): {_ve}",
                                flush=True,
                            )
                            _verify_passed = True  # fail-open
                            break

                    tracker.mark_done(self._summary)
                    # Fix 2C: stash handoff for warm-start injection after next reset
                    self._warmstart_handoff = {
                        "subtask_n": subtask.n,
                        "status": "done",
                        "files_written": (subtask.files_written or [])[:10],
                        "commands_failed": self._subtask_failed_commands[:5],
                        "summary": (self._last_done_handoff or self._summary or "")[:300],
                    }
                    # Structured handoff at subtask boundary (done)
                    self._scratchpad.add_handoff({
                        "subtask_n": subtask.n,
                        "description": subtask.description,
                        "status": "done",
                        "files_written": subtask.files_written or [],
                        "commands_failed": self._subtask_failed_commands[:5],
                        "handoff_message": self._last_done_handoff or self._summary or "",
                    })
                    # Inter-subtask scratchpad write
                    _isw_ok, _isw_cost = self._scratchpad_writer.write_inter_subtask(
                        self._scratchpad, subtask.n, subtask.description,
                        "done", self._summary,
                    )
                    self._task_cost_usd += _isw_cost
                    self._monthly_cost_usd += _isw_cost
                    if not _isw_ok:
                        self._scratchpad_miss_count += 1
                    self._scratchpad.save()
                    self._write_checkpoint()
                else:
                    tracker.mark_failed(
                        self._summary or self._halt_reason or "unknown"
                    )
                    # Fix 2C: stash handoff for warm-start injection after next reset
                    self._warmstart_handoff = {
                        "subtask_n": subtask.n,
                        "status": "failed",
                        "files_written": list(self._files_written_this_subtask or set())[:10],
                        "commands_failed": self._subtask_failed_commands[:5],
                        "summary": (self._summary or self._halt_reason or "unknown")[:300],
                    }
                    # Structured handoff at subtask boundary (failed)
                    self._scratchpad.add_handoff({
                        "subtask_n": subtask.n,
                        "description": subtask.description,
                        "status": "failed",
                        "files_written": list(self._files_written_this_subtask or set()),
                        "commands_failed": self._subtask_failed_commands[:5],
                        "handoff_message": self._summary or self._halt_reason or "unknown",
                    })
                    # Inter-subtask scratchpad write (failed path)
                    _isw_ok, _isw_cost = self._scratchpad_writer.write_inter_subtask(
                        self._scratchpad, subtask.n, subtask.description,
                        "failed", self._summary or self._halt_reason or "unknown",
                    )
                    self._task_cost_usd += _isw_cost
                    self._monthly_cost_usd += _isw_cost
                    if not _isw_ok:
                        self._scratchpad_miss_count += 1
                    self._scratchpad.save()

                    # P0-A: Continuation cascade cap
                    tracker.continuation_count += 1
                    if tracker.continuation_count > 3:
                        self._halt("max_continuations_exceeded")
                        break

                    # Fix P2: retry-before-replan for zero-productivity aborts.
                    # If the subtask aborted without making any productive
                    # changes (no writes, no successful commands), retry it
                    # once with a hint before invoking the full replan pipeline.
                    if (
                        self._subtask_productive_iters == 0
                        and self._subtask_retry_count == 0
                    ):
                        self._subtask_retry_count += 1
                        subtask.status = "pending"
                        print(
                            f"  [retry] Subtask {subtask.n} aborted with 0 "
                            f"productive iterations — retrying once before replan",
                            flush=True,
                        )
                        self._warmstart_handoff = {
                            "subtask_n": subtask.n,
                            "status": "retrying",
                            "files_written": [],
                            "commands_failed": self._subtask_failed_commands[:5],
                            "summary": (
                                "Previous attempt aborted without making any changes. "
                                "Read any relevant files, then create or modify the required file directly. "
                                "Do NOT use conditional if/then/else action blocks. "
                                "Use simple sequential actions: read_file → write_file → done."
                            ),
                        }
                        _preserve_retry_count = self._subtask_retry_count
                        self._reset_for_subtask()
                        self._subtask_retry_count = _preserve_retry_count
                        continue  # re-enter while-loop; next_subtask() finds pending subtask

                    # -- Interactive plan reentry (replaces autonomous replan) --
                    failure_reason = self._summary or self._halt_reason or "unknown"
                    continuation_draft = planner_mod.generate_continuation_draft(
                        self, tracker, failure_reason,
                    )

                    # Surface to UI for interactive replanning
                    self._plan_reentry_event = threading.Event()
                    self._plan_reentry_result = None

                    failed_st = tracker.subtasks[tracker.current_index]
                    completed_info = []
                    for s in tracker.subtasks:
                        if s.status == "done":
                            completed_info.append({
                                "n": s.n,
                                "description": s.description,
                                "files_written": s.files_written or [],
                            })

                    if self._on_update:
                        try:
                            self._on_update({
                                "task_id": self._task_id,
                                "plan_reentry": True,
                                "completed_subtasks": completed_info,
                                "failed_subtask": {
                                    "n": failed_st.n,
                                    "description": failed_st.description,
                                },
                                "failure_reason": failure_reason,
                                "continuation_draft": [
                                    {"n": s.n, "description": s.description, "model_tier": s.model_tier}
                                    for s in (continuation_draft or [])
                                ],
                            })
                        except Exception:
                            pass

                    # Block until user approves a continuation plan
                    self._plan_reentry_event.wait()
                    approved_subtasks = self._plan_reentry_result
                    self._plan_reentry_event = None
                    self._plan_reentry_result = None

                    if approved_subtasks is None:
                        # User cancelled — abort
                        self._halt("cancelled")
                        break

                    # Replace remaining subtasks with approved plan
                    tracker.subtasks = tracker.subtasks[: tracker.current_index + 1] + approved_subtasks
                    # P0-C: Enforce max_subtasks on continuation splices
                    _max_allowed = (tracker.max_subtasks or 50) + 3
                    if len(tracker.subtasks) > _max_allowed:
                        tracker.subtasks = tracker.subtasks[:_max_allowed]
                        print(f"  [goaltracker] Trimmed continuation to {_max_allowed} subtasks", flush=True)
                    tracker.current_index += 1
                    self._goaltracker = tracker
                    self._write_checkpoint()

                if self.cancel_event.is_set():
                    break

            # Final outcome
            self._outcome = "done" if tracker.all_done() else self._outcome
            self._summary = tracker.final_summary()
        finally:
            self._goal = self._original_goal
            self._model = _saved_model
            self._fallback_model = _saved_fallback
            self._extra_body = _saved_extra_body

    # -----------------------------------------------------------------
    # Script planner fast-path (Phase 6)
    # -----------------------------------------------------------------

    def _try_script_planner(self, subtask_goal: str, subtask_n: int = 0) -> bool:
        """Attempt to execute a subtask via ScriptPlanner.

        Returns True if the subtask was fully handled (done or failed with
        fallback context injected).  Returns False if the caller should
        proceed to _brain_loop.
        """
        planner = script_planner_mod.ScriptPlanner(
            llm_client=self._llm, model=self._model,
        )
        prior_results: str | None = None
        _sp_rendered = self._scratchpad.render()
        if _sp_rendered:
            prior_results = _sp_rendered
        segment = planner.plan(subtask_goal, prior_results=prior_results)

        if segment.fallback_react:
            return False  # caller should run _brain_loop

        MAX_REPLANS = 2
        replan_failures = 0
        last_result: str | None = None

        while True:
            # Execute the script via the standard dispatch pipeline so that
            # traces, history and cost accounting all work correctly.
            self._iteration += 1  # each script planner action = one iteration
            act_dict = {"action": "run_command", "args": {"cmd": segment.script}}
            self.record_act(act_dict)

            t_s = time.time()
            result = None
            error = None
            wid = None
            tier = None
            try:
                result, wid, tier = self._dispatch(act_dict)
            except Exception as exc:
                error = exc

            self.observer.observe(
                act_dict, result=result, error=error,
                attempt=1, t_start=t_s,
                llm_messages=None, llm_response=None,
            )
            self._last_obs = self.observer.last_entry
            if self._last_obs:
                self._history.append(self._make_history_entry(act_dict, self._last_obs))
            self._fire_on_update(self._iteration, act_dict, self._last_obs)

            _res = result if isinstance(result, dict) else {}
            exit_code = _res.get("exit_code", 1)
            stdout = _res.get("stdout", "")
            stderr = _res.get("stderr", str(error) if error is not None else "")

            if exit_code == 0 and not segment.observe_after:
                # Subtask fully completed
                self._outcome = "done"
                self._summary = f"Script planner completed subtask: {subtask_goal}"
                # Scratchpad writer extracts facts from script planner output
                if stdout:
                    _spw_ok, _spw_cost = self._scratchpad_writer.write_iteration(
                        self._scratchpad, "run_command",
                        {"cmd": segment.script}, stdout, stderr,
                    )
                    self._task_cost_usd += _spw_cost
                    self._monthly_cost_usd += _spw_cost
                    if not _spw_ok:
                        self._scratchpad_miss_count += 1
                return True

            if exit_code == 0 and segment.observe_after:
                # Need another planning round with the output
                last_result = stdout
                segment = planner.plan(subtask_goal, last_result=last_result,
                                       prior_results=prior_results)
                if segment.fallback_react:
                    return False  # fall through to _brain_loop
                replan_failures = 0  # success resets counter
                continue

            # exit_code != 0 — re-plan with failure context
            replan_failures += 1
            if replan_failures >= MAX_REPLANS:
                # Inject failure context into history and fall through
                failure_ctx = (
                    f"ScriptPlanner failed {replan_failures} times for "
                    f"subtask: {subtask_goal}\n"
                    f"Last exit_code={exit_code}\nstderr={stderr}"
                )
                self._history.append({
                    "role": "script_planner_failure",
                    "content": failure_ctx,
                })
                return False  # fall through to _brain_loop

            last_result = f"FAILED exit_code={exit_code}\nstderr={stderr}"
            segment = planner.plan(subtask_goal, last_result=last_result,
                                   prior_results=prior_results)
            if segment.fallback_react:
                return False  # fall through to _brain_loop

    def _reset_for_subtask(self) -> None:
        """Reset per-subtask state between GoalTracker subtask iterations.

        Does NOT reset: _task_cost_usd, cancel_event, _task_id,
        _task_budget, _iteration (continues incrementing across subtasks
        for trace continuity).

        Observer intentionally NOT reset — sequence continues across
        subtasks for trace continuity.
        """
        self._history.clear()
        self._last_obs = None
        self._outcome = None
        self._halt_reason = None  # prevent stale halt_reason from poisoning next subtask
        self._summary = ""
        self.reset_loop_detector()
        self._last_recorded_action = ""
        self._files_written_this_subtask = set()
        self._consecutive_chat_count = 0
        self._consecutive_env_failures = 0
        self._consecutive_schema_errors = 0
        self._consecutive_readonly = 0
        self._consecutive_command_failures = 0
        self._subtask_has_written = False  # stall detection only fires after first write_file/append_file
        self._subtask_productive_iters = 0  # only mutating actions count toward cap
        # FIX A: _task_file_read_counts intentionally NOT reset — persists across subtask boundaries
        self._read_stall_warned = False  # ensures warning fires once even if counter jumps
        self._last_written_path = ""  # Fix P0-write-reread: track last written path
        self._post_write_reread_count = 0  # Fix P0-write-reread: consecutive re-reads of just-written file
        self._subtask_retry_count = 0  # Fix P2: retry-before-replan counter
        self._last_done_handoff = ""
        self._subtask_failed_commands = []
        self._done_bounce_count = 0
        self._stub_bounce_count = 0
        self._stall_nudge_active = False  # Fix D: reset stall nudge flag per subtask
        self._productive_done_bounce = 0  # Fix B: reset per subtask
        self._fetch_url_called = False
        self._prefetched_content = ""
        self._estimated_iterations = None

        # Fix 2C: warm-start — inject previous subtask context into fresh history
        ws = self._warmstart_handoff
        if ws:
            files_str = ", ".join(ws.get("files_written", [])[:6]) or "none"
            failed_str = "; ".join(ws.get("commands_failed", [])[:3]) or "none"
            note = (
                f"[Context from subtask {ws.get('subtask_n', '?')} "
                f"({ws.get('status', '?')})]\n"
                f"Files written: {files_str}\n"
                f"Commands that failed: {failed_str}\n"
                f"Summary: {ws.get('summary', '')}"
            )
            self._history.append({
                "action": "_subtask_context",
                "args": {},
                "result": "info",
                "observation": note,
            })
            self._warmstart_handoff = None

    def _brain_loop(self) -> None:
        """Main ReAct brain loop. Modifies self._outcome, self._summary."""

        start_iteration = self._iteration + 1

        for iteration in range(start_iteration, MAX_ITERATIONS_PER_TASK + 1):
            self._iteration = iteration
            t_start = None
            error_class = None

            # -- Pause check (top, autonomous only) ---------------------------
            if self._mode == "autonomous":
                self._pause_wait()

            # -- Pre-flight: cancel → log_write_failed → budget(task) → budget(monthly) --
            if self.cancel_event.is_set():
                self._halt("cancelled")
                break

            if self.observer.log_write_failed:
                self._halt("unrecoverable")
                break

            if self._task_cost_usd >= self._task_budget:
                self._halt("budget_exceeded")
                break

            if self._monthly_cost_usd >= MONTHLY_BUDGET_CAP:
                self._halt("budget_exceeded")
                break

            # -- Per-subtask iteration cap (prevents one stuck subtask from
            # burning the entire task budget) --
            # Only MUTATING actions (write_file, append_file, run_command,
            # delete_file) count toward the cap.  Read-only iterations
            # (read_file, list_dir, think, checkpoint, ...) are free — bounded
            # by the per-task budget and stall detection instead.
            # A hard ceiling of 3x the cap applies to TOTAL iterations as an
            # absolute safety net against runaway cost even with only reads.
            _total_subtask_iters = iteration - start_iteration
            if (
                getattr(self, "_in_goaltracker_subtask", False)
                and MAX_ITERATIONS_PER_SUBTASK > 0
                and (
                    self._subtask_productive_iters >= MAX_ITERATIONS_PER_SUBTASK
                    or _total_subtask_iters >= MAX_ITERATIONS_PER_SUBTASK * 2
                )
            ):
                self._halt("subtask_cap_exceeded")
                break

            # -- LLM call with retry wrapper ----------------------------------
            user_turn = self._build_user_turn(
                self._goal, iteration, self._task_cost_usd,
                self._monthly_cost_usd, self._last_obs, self._history,
            )
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_turn},
                {"role": "assistant", "content": "<think>\n"},
            ]

            raw_response = None
            llm_failed = False
            current_model = self._model

            # Fire transient "thinking" event so UI shows thinking state
            self._fire_on_update(iteration, None, None, transient_status="thinking")

            raw_response = self._llm_call_with_retry(iteration, messages)
            if raw_response is None:
                break

            # -- Parse response ------------------------------------------------
            full_response = "<think>\n" + raw_response
            think_content, act_payload = None, None

            try:
                think_content, act_payload = self._parse_response(full_response)
            except SchemaValidationError as parse_err:
                # Extract and log think content even on parse failure
                _think = self._extract_think_content(full_response)
                if _think is not None and _think.strip():
                    self.observer.observe_think(_think.strip())

                # Schema injection path
                truncated_raw = full_response[:ARGS_TRUNCATION_CHARS]
                synth_dict = {
                    "action": "schema_validation_error",
                    "args": {"raw_response": truncated_raw},
                }
                self.observer.observe(
                    synth_dict, result=None, error=parse_err,
                    attempt=1, t_start=t_start,
                    llm_response=full_response, llm_messages=messages,
                )
                self._last_obs = self.observer.last_entry
                if self._last_obs:
                    self._history.append(self._make_history_entry(
                        synth_dict, self._last_obs,
                    ))
                self._fire_on_update(iteration, synth_dict, self._last_obs)
                self._consecutive_schema_errors += 1
                if self._consecutive_schema_errors >= SCHEMA_ERROR_HALT_THRESHOLD:
                    self._halt("schema_error_loop")
                    break
                continue

            # -- observe_think() fires after parse, before validate -----------
            self._consecutive_schema_errors = 0  # successful parse resets counter
            if think_content is not None:
                self.observer.observe_think(think_content)

            # -- Mid-iteration cancel check (after LLM, before dispatch) ------
            if self.cancel_event.is_set():
                self._halt("cancelled")
                break

            # ==============================================================
            # BATCH PATH (§1.5): act_payload is a list of action dicts
            # ==============================================================
            if isinstance(act_payload, list):
                signal = self._handle_batch_path(
                    act_payload, iteration, messages, full_response, t_start,
                )
                if signal in ("done", "abort", "halt"):
                    break
                if signal == "schema_error":
                    self._consecutive_schema_errors += 1
                    if self._consecutive_schema_errors >= SCHEMA_ERROR_HALT_THRESHOLD:
                        self._halt("schema_error_loop")
                        break
                    continue
                continue  # "ok" — batch consumed the iteration

            # SINGLE-ACTION PATH (pre-1.5 behavior, unchanged)
            # ==============================================================
            act_dict = act_payload  # guaranteed to be a dict here

            # -- Schema validation ---------------------------------------------
            try:
                self._validate_act(act_dict)
            except SchemaValidationError as val_err:
                truncated_raw = full_response[:ARGS_TRUNCATION_CHARS]
                synth_dict = {
                    "action": "schema_validation_error",
                    "args": {"raw_response": truncated_raw},
                }
                self.observer.observe(
                    synth_dict, result=None, error=val_err,
                    attempt=1, t_start=t_start,
                    llm_response=full_response, llm_messages=messages,
                )
                self._last_obs = self.observer.last_entry
                if self._last_obs:
                    self._history.append(self._make_history_entry(
                        synth_dict, self._last_obs,
                    ))
                self._fire_on_update(iteration, synth_dict, self._last_obs)
                self._consecutive_schema_errors += 1
                if self._consecutive_schema_errors >= SCHEMA_ERROR_HALT_THRESHOLD:
                    self._halt("schema_error_loop")
                    break
                continue

            action_name = act_dict["action"]
            action_args = act_dict["args"]

            # -- Terminal intercept: done / abort / chat -----------------------
            if action_name == "done":
                rejection = self._validate_done(act_dict)
                if rejection:
                    self._last_obs = rejection
                    self._history.append({
                        "action": "done",
                        "args": act_dict.get("args") or {},
                        "result": rejection["result"],
                        "observation": rejection.get("observation", {}),
                    })
                    self._fire_on_update(iteration, act_dict, self._last_obs)
                    continue
                t_start = time.time()
                self.observer.observe(
                    act_dict, result={"status": "ok"}, error=None,
                    attempt=1, t_start=t_start,
                    llm_response=full_response, llm_messages=messages,
                )
                self._last_obs = self.observer.last_entry
                self._outcome = "done"
                self._summary = action_args.get("summary", "")
                self._last_done_handoff = action_args.get("handoff", "")  # P1-D
                self._fire_on_update(
                    iteration, act_dict, self._last_obs,
                    outcome="done",
                    message=action_args.get("message", ""),
                )
                break

            if action_name == "abort":
                t_start = time.time()
                self.observer.observe(
                    act_dict, result={"status": "ok"}, error=None,
                    attempt=1, t_start=t_start,
                    llm_response=full_response, llm_messages=messages,
                )
                self._last_obs = self.observer.last_entry
                self._outcome = "abort"
                self._summary = action_args.get("reason", "")
                self._fire_on_update(
                    iteration, act_dict, self._last_obs,
                    outcome="abort",
                )
                break

            if action_name == "chat":
                t_start = time.time()
                self.observer.observe(
                    act_dict, result={"status": "ok"}, error=None,
                    attempt=1, t_start=t_start,
                    llm_response=full_response, llm_messages=messages,
                )
                self._last_obs = self.observer.last_entry
                if self._last_obs:
                    self._history.append(self._make_history_entry(
                        act_dict, self._last_obs,
                    ))
                self._fire_on_update(
                    iteration, act_dict, self._last_obs,
                    message=action_args.get("message"),
                )
                # chat continues loop — not terminal
                self.record_act(act_dict)
                self._consecutive_chat_count += 1
                if self._consecutive_chat_count >= 4:
                    self._summary = action_args.get("message", "Conversational response completed.")
                    self._halt("chat_limit")
                    break
                continue

            # -- record_act (after terminal intercept, before execution) -------
            self._consecutive_chat_count = 0
            self.record_act(act_dict)

            # -- Action retry while-loop (inner loop) -------------------------
            attempt = 1
            result = None
            error = None
            wid = None
            tier = None

            while True:
                t_start = time.time()
                result = None
                error = None
                wid = None
                tier = None

                try:
                    result, wid, tier = self._dispatch(act_dict)
                except Exception as e:
                    error = e
                finally:
                    self.observer.observe(
                        act_dict, result=result, error=error,
                        attempt=attempt, t_start=t_start,
                        tier=tier, wid=wid,
                        llm_response=full_response, llm_messages=messages,
                    )

                self._last_obs = self.observer.last_entry
                if self._last_obs:
                    self._history.append(self._make_history_entry(
                        act_dict, self._last_obs,
                    ))

                # -- Scratchpad writer (per-iteration) ----------------------------
                # Extract facts from raw dispatch output before history truncation.
                _raw_stdout = ""
                _raw_stderr = ""
                if isinstance(result, dict):
                    _raw_stdout = str(result.get("stdout", ""))
                    # read_file returns {"content": ...} not {"stdout": ...}.
                    # Treat file content as stdout so the writer extracts
                    # file-role entries ("scheduler.py": "DAG execution loop").
                    if not _raw_stdout and action_name == "read_file":
                        _raw_stdout = str(result.get("content", ""))
                    _raw_stderr = str(result.get("stderr", ""))
                elif error is not None:
                    _raw_stderr = str(error)
                if _raw_stdout or _raw_stderr:
                    _sw_ok, _sw_cost = self._scratchpad_writer.write_iteration(
                        self._scratchpad, action_name, action_args,
                        _raw_stdout, _raw_stderr,
                    )
                    self._task_cost_usd += _sw_cost
                    self._monthly_cost_usd += _sw_cost
                    if not _sw_ok:
                        self._scratchpad_miss_count += 1
                        # Fallback: inject bare path→"read" entry so the
                        # files category is never empty even on writer timeout.
                        if action_name == "read_file" and action_args.get("path"):
                            self._scratchpad.update_category(
                                "files", {action_args["path"]: "read"},
                            )

                error_class = (
                    self.classify_error(error, action_name) if error else None
                )

                if (
                    error_class == ERROR_CLASS_TRANSIENT
                    and attempt < MAX_ACTION_RETRIES
                ):
                    attempt += 1
                    time.sleep(
                        RETRY_BACKOFF_SECONDS[
                            min(attempt - 2, len(RETRY_BACKOFF_SECONDS) - 1)
                        ]
                    )
                    continue
                break

            # -- TRANSIENT exhaustion → reclassify to ENVIRONMENTAL ----------
            if error_class == ERROR_CLASS_TRANSIENT:
                error_class = ERROR_CLASS_ENVIRONMENTAL

            # -- Post-inner-loop dispatch -------------------------------------
            if error_class == ERROR_CLASS_UNRECOVERABLE:
                self._halt("unrecoverable")
                break

            if error_class == ERROR_CLASS_RESOURCE:
                if self._mode == "interactive":
                    print(
                        f"\n[GENIE] RESOURCE error: {error}\n"
                        f"Resolve the issue and restart the task.\n",
                        file=sys.stderr,
                    )
                    self._halt("resource_halt")
                    break
                # autonomous: fall through — LLM self-manages

            # ENVIRONMENTAL: fall through — LLM replans

            # -- Track consecutive environmental failures for escalation -------
            _obs_result = (self._last_obs or {}).get("result", "")
            if _obs_result == "environmental_failure":
                self._consecutive_env_failures += 1
            else:
                self._consecutive_env_failures = 0

            # -- Stall detection -----------------------------------------------
            if self._check_single_action_stalls(act_dict, iteration):
                break

            # -- Cost accumulation already done in _llm_call ------------------

            # -- Telegram progress --------------------------------------------
            if (
                self._mode == "autonomous"
                and iteration % TELEGRAM_PROGRESS_INTERVAL == 0
            ):
                self._telegram_notify(
                    f"🔄 Progress: iteration {iteration}/{MAX_ITERATIONS_PER_TASK}\n"
                    f"Last action: {action_name}\n"
                    f"Task cost: ${self._task_cost_usd:.4f}\n"
                    f"Monthly: ${self._monthly_cost_usd:.4f}"
                )

            # -- Monthly budget warning (80%) ---------------------------------
            if (
                not self._monthly_budget_warning_sent
                and self._monthly_cost_usd >= MONTHLY_BUDGET_CAP * 0.80
            ):
                self._monthly_budget_warning_sent = True
                self._telegram_notify(
                    f"⚠️ Monthly budget at {self._monthly_cost_usd / MONTHLY_BUDGET_CAP * 100:.0f}%\n"
                    f"Spent: ${self._monthly_cost_usd:.2f} / ${MONTHLY_BUDGET_CAP:.2f}"
                )

            # -- Checkpoint (every iteration) ---------------------------------
            self._write_checkpoint()

            # -- on_update callback -------------------------------------------
            self._fire_on_update(iteration, act_dict, self._last_obs)

            # -- Pause check (bottom, autonomous only) ------------------------
            if self._mode == "autonomous":
                self._pause_wait()

        # -- Post-loop ---------------------------------------------------------
        if self._halt_reason is None and self._outcome is None:
            self._halt("iteration_exceeded")

        # Map halt_reason to outcome
        if self._outcome is None:
            halt_map = {
                "unrecoverable": "unrecoverable",
                "loop_detected": "unrecoverable",
                "schema_error_loop": "unrecoverable",
                "resource_halt": "unrecoverable",
                "command_stall_exceeded": "unrecoverable",
                "max_continuations_exceeded": "unrecoverable",
                "subtask_cap_exceeded": "iteration_exceeded",
                "post_write_reread_loop": "iteration_exceeded",
                "read_stall_exceeded": "iteration_exceeded",
                "budget_exceeded": "budget_exceeded",
                "cancelled": "cancelled",
                "iteration_exceeded": "iteration_exceeded",
                "chat_limit": "done",
            }
            self._outcome = halt_map.get(self._halt_reason, "unrecoverable")
            if not self._summary:
                self._summary = f"Task halted: {self._halt_reason}"


    # =================================================================
    # Brain-loop helper: LLM call with retry
    # =================================================================

    def _llm_call_with_retry(self, iteration: int, messages: list) -> str | None:
        """LLM call with retry logic.

        Returns the raw response string on success, or None if all retries
        failed (self._halt() will have been called internally).
        """
        raw_response = None
        llm_failed = False
        current_model = self._model

        for llm_attempt in range(1, MAX_LLM_RETRIES + 1):
            try:
                raw_response = self._llm_call(messages, model=current_model)
                break
            except (httpx.TimeoutException, httpx.RequestError):
                if llm_attempt == MAX_LLM_RETRIES:
                    self._halt("unrecoverable")
                    llm_failed = True
                    break
                self._fire_on_update(
                    iteration, None, None,
                    transient_status=f"llm_retry_{llm_attempt}_of_{MAX_LLM_RETRIES}",
                )
                time.sleep(
                    RETRY_BACKOFF_SECONDS[
                        min(llm_attempt - 1, len(RETRY_BACKOFF_SECONDS) - 1)
                    ]
                )
                continue
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code in LLM_SERVICE_ERROR_CODES:
                    # Switch to fallback, one attempt
                    try:
                        raw_response = self._llm_call(
                            messages, model=self._fallback_model,
                        )
                    except Exception:
                        self._halt("unrecoverable")
                        llm_failed = True
                    break
                else:
                    self._halt("unrecoverable")
                    llm_failed = True
                    break
            except ResponseTruncatedError as rte:
                # Response cut off by max_tokens — retry once with
                # doubled budget.  If the second attempt also truncates,
                # fall through and let the parser try to salvage the
                # partial response.
                if llm_attempt == 1:
                    from config import MODEL_MAX_TOKENS, DEFAULT_MAX_TOKENS
                    _base = MODEL_MAX_TOKENS.get(current_model, DEFAULT_MAX_TOKENS)
                    _retry_tokens = min(_base * 2, 65536)
                    try:
                        raw_response = self._llm_call(
                            messages, model=current_model,
                            max_tokens=_retry_tokens,
                        )
                    except ResponseTruncatedError as rte2:
                        # Still truncated — use the partial content
                        raw_response = rte2.partial_content
                    except Exception:
                        # Non-truncation failure on retry — use
                        # the original partial content
                        raw_response = rte.partial_content
                    break
                else:
                    # Already retried — use the partial content
                    raw_response = rte.partial_content
                    break
            except Exception:
                # Catch-all for unexpected errors (JSON decode failures,
                # KeyError from malformed API responses, SSL errors, etc.).
                # Retry with backoff instead of crashing immediately.
                if llm_attempt == MAX_LLM_RETRIES:
                    self._halt("unrecoverable")
                    llm_failed = True
                    break
                time.sleep(
                    RETRY_BACKOFF_SECONDS[
                        min(llm_attempt - 1, len(RETRY_BACKOFF_SECONDS) - 1)
                    ]
                )
                continue

        if llm_failed:
            return None
        if raw_response is None:
            # API returned null content — retry once with fallback model
            # before giving up.  This handles cases where the primary
            # model returns empty / null content sporadically.
            try:
                raw_response = self._llm_call(
                    messages, model=self._fallback_model,
                )
            except Exception:
                pass
        if raw_response is None:
            self._halt("unrecoverable")
            return None

        return raw_response

    # =================================================================
    # Brain-loop helper: validate done guards
    # =================================================================

    def _validate_done(self, act_dict: dict) -> dict | None:
        """Validate whether a done action should be accepted.

        Returns a rejection observation dict if done should be blocked,
        or None if done is accepted (all guards passed).
        """
        action_args = act_dict.get('args') or {}

        _has_successful_write = any(
            e.get("action") == "write_file"
            and e.get("result") == "success"
            for e in self._history
        )

        # -- Failed-verification guard (always active): if the
        #    most recent run_command had a non-zero exit_code, or
        #    write_file was called but the script has not been
        #    re-run since, block done with a directive message.
        _last_run: dict | None = None
        _last_run_pos = -1
        _last_write_pos = -1
        for _idx, _e in enumerate(self._history):
            if _e.get("action") == "run_command":
                _last_run = _e
                _last_run_pos = _idx
            if (
                _e.get("action") == "write_file"
                and _e.get("result") == "success"
            ):
                _last_write_pos = _idx
        # Case A: wrote a fix but never re-ran the script since.
        # Skip for non-executable deliverables (docs, data, config)
        # that are never "run" — the guard is only meaningful for
        # scripts / code files that produce an exit code.
        _NON_EXEC_EXTS = (
            ".md", ".txt", ".rst", ".html", ".xml",
            ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
            ".csv", ".tsv", ".log",
        )
        _last_write_path = (
            self._history[_last_write_pos].get("args", {}).get("path", "")
            if _last_write_pos >= 0 else ""
        )
        _last_write_is_doc = any(
            _last_write_path.endswith(ext) for ext in _NON_EXEC_EXTS
        )
        _wrote_without_rerun = (
            _last_write_pos > _last_run_pos and not _last_write_is_doc
        )
        # Case B: last run_command had a non-zero exit_code.
        # Suppress when a doc-type deliverable was written after
        # the failed run — that run was an abandoned intermediate
        # step, not a broken output script.
        _last_run_failed = (
            _last_run is not None
            and (
                (_last_run.get("observation", {}).get("exit_code") or 0) != 0
            )
            and not (_last_write_is_doc and _last_write_pos > _last_run_pos)
        )
        if _wrote_without_rerun:
            _w_path = (
                self._history[_last_write_pos].get("args", {})
                .get("path", "<file>")
            )
            _verify_guard_obs = {
                "result": "environmental_failure",
                "observation": {
                    "done_blocked": (
                        f"done REJECTED: You wrote the fix to "
                        f"'{_w_path}' but have NOT re-run the "
                        f"script since then. "
                        f"REQUIRED NEXT ACTION: run_command to "
                        f"execute the fixed script and verify "
                        f"exit_code=0. Do not call done again "
                        f"until that run_command succeeds."
                    ),
                },
                "error": None,
            }
            return _verify_guard_obs
        elif _last_run_failed:
            _failed_exit_code = (
                _last_run.get("observation", {}).get("exit_code")
            )
            _failed_cmd = (
                _last_run.get("args", {}).get("cmd", "<unknown>")
            )
            _failed_stderr = str(
                _last_run.get("observation", {}).get("stderr", "")
            )[:200]
            _verify_guard_obs = {
                "result": "environmental_failure",
                "observation": {
                    "done_blocked": (
                        f"done REJECTED: The script is still "
                        f"broken. run_command('{_failed_cmd}') "
                        f"returned exit_code={_failed_exit_code}. "
                        f"stderr: {_failed_stderr!r}. "
                        f"REQUIRED NEXT ACTIONS: (1) write_file "
                        f"with the corrected content, then "
                        f"(2) run_command to verify exit_code=0. "
                        f"Do NOT call done again until the "
                        f"verification run_command succeeds."
                    ),
                },
                "error": None,
            }
            return _verify_guard_obs
        if (
            self._subtask_productive_iters == 0
            and self._productive_done_bounce == 0
        ):
            self._productive_done_bounce += 1
            _fixb_obs = {
                "result": "environmental_failure",
                "observation": {
                    "done_blocked": (
                        "done REJECTED: You have not performed any productive "
                        "actions (write_file, run_command, etc.) in this subtask. "
                        "Read the existing code and implement the required changes "
                        "before calling done."
                    ),
                },
                "error": None,
            }
            return _fixb_obs
        _EDIT_VERB_RE = re.compile(
            r'\b(fix|modify|update|implement|add|create|write|'
            r'patch|change|edit|replace)\b', re.I,
        )
        _cur_st = (
            self._goaltracker.subtasks[self._goaltracker.current_index]
            if self._goaltracker
            and self._goaltracker.current_index < len(self._goaltracker.subtasks)
            else None
        )
        if (
            _cur_st
            and _EDIT_VERB_RE.search(_cur_st.description)
            and not _has_successful_write
            and self._done_bounce_count == 0
        ):
            self._done_bounce_count += 1
            _bounce_obs = {
                "result": "environmental_failure",
                "observation": {
                    "done_blocked": (
                        "done BOUNCED: You haven't written any "
                        "files in this subtask. If the required "
                        "changes are already present, re-verify "
                        "by reading the target file(s) and emit "
                        "done again. If not, make the edits "
                        f"first. Subtask: {_cur_st.description[:100]}"
                    ),
                },
                "error": None,
            }
            return _bounce_obs
        _goal_lower = self._goal.lower()
        _goal_has_search = any(
            w in _goal_lower for w in ("search", "look up", "find")
        )
        _goal_has_click = any(
            w in _goal_lower
            for w in (
                "click", "first result", "second result",
                "third result", "show me that page", "show me the page",
            )
        )
        _goal_has_type = any(
            phrase in _goal_lower
            for phrase in (
                "type '", "type \"", "type the", "type into",
                "enter '", "enter \"", "search for",
            )
        )
        _has_type_element = any(
            e.get("action") == "type_element"
            for e in self._history
        )
        _has_type_text = any(
            e.get("action") == "type_text"
            and "wikipedia" not in (e.get("args", {}).get("text", "") or "").lower()
            and "http" not in (e.get("args", {}).get("text", "") or "").lower()
            for e in self._history
        )
        _has_click_element = any(
            e.get("action") == "click_element"
            for e in self._history
        )
        if (
            not _has_successful_write
            and _goal_has_search
            and _goal_has_click
            and not _has_type_element
            and not _has_click_element
        ):
            # Reject done — the LLM hasn't even started the search.
            _presearch_obs = {
                "result": "environmental_failure",
                "observation": {
                    "presearch_guard": (
                        "done REJECTED: The goal requires searching "
                        "and clicking a result, but no search has "
                        "been performed yet. You navigated to the "
                        "search engine but did not type a query in "
                        "the search box. Use type_element to enter "
                        "the search query into the page's search "
                        "input, then press_key enter, wait for "
                        "results, and click the requested result."
                    ),
                },
                "error": None,
            }
            return _presearch_obs
        if (
            not _has_successful_write
            and _goal_has_type
            and not _has_type_element
            and not _has_type_text
        ):
            _pretype_obs = {
                "result": "environmental_failure",
                "observation": {
                    "pretype_guard": (
                        "done REJECTED: The goal requires typing into "
                        "a field (e.g. a search box), but no "
                        "type_element or type_text for the query has "
                        "been dispatched. Wait for the page to load "
                        "(wait 3-5 seconds), then use type_element "
                        "to enter the text into the input field."
                    ),
                },
                "error": None,
            }
            return _pretype_obs
        _serp_blocked = False
        _goal_needs_nav = any(
            phrase in _goal_lower
            for phrase in (
                "click the first", "click on the first", "click first",
                "first link", "first result", "first search result",
                "open the first", "click the link", "click a link",
                "click the result", "click on the result",
                "show me that page", "show me the page",
                "click the second", "click on the second", "click second",
                "second link", "second result", "second search result",
                "click the third", "click on the third", "click third",
                "third link", "third result", "third search result",
            )
        )
        # Determine which result index the goal requests (0-based).
        # Default to 0 (first). "second" → 1, "third" → 2.
        _goal_result_index = 0
        _ORDINAL_MAP = {
            "second": 1, "2nd": 1,
            "third": 2, "3rd": 2,
            "fourth": 3, "4th": 3,
            "fifth": 4, "5th": 4,
        }
        for _ord_word, _ord_idx in _ORDINAL_MAP.items():
            if _ord_word in _goal_lower:
                _goal_result_index = _ord_idx
                break
        _SERP_PATTERNS = (
                "google.com/search",
                "bing.com/search",
                "search.yahoo.com/search",
                "duckduckgo.com/?q=",
            )
        # Broader pattern: still on a search engine domain at all
        # (homepage, SERP, or redirect page like bing.com/?FORM=...).
        # If the goal required clicking a result, being on any search
        # engine page means the click failed or hit the wrong element.
        _SEARCH_ENGINE_DOMAINS = (
                "google.com",
                "bing.com",
                "search.yahoo.com",
                "duckduckgo.com",
            )

        if _goal_needs_nav and not _has_successful_write:
            _cdp_port = self._chrome_cdp_port()
            if _cdp_port is not None:
                _current_url = self._cdp_read_active_tab_url(_cdp_port)
                _on_serp = _current_url and any(
                    p in _current_url for p in _SERP_PATTERNS
                )
                _on_search_engine = _current_url and any(
                    d in _current_url for d in _SEARCH_ENGINE_DOMAINS
                )
                if _on_serp:
                    # On a SERP — attempt autonomous navigation
                    _nav_url = self._cdp_navigate_nth_serp_result(
                        _cdp_port, _goal_result_index
                    )
                    if _nav_url:
                        # Navigation succeeded — let the loop continue
                        # so the LLM sees the new page and calls done.
                        _serp_blocked = True
                        self.reset_loop_detector()
                        _serp_obs = {
                            "result": "success",
                            "observation": {
                                "serp_guard": (
                                    "The previous click_element landed on "
                                    "a navigation element rather than a "
                                    "search result. Genie recovered by "
                                    f"navigating directly to result "
                                    f"#{_goal_result_index + 1}: "
                                    f"{_nav_url}. "
                                    "The page is now loading. "
                                    "Call done now — the task is complete."
                                ),
                                "navigated_to": _nav_url,
                            },
                            "error": None,
                        }
                    else:
                        # CDP navigation also failed — tell LLM to
                        # use Tab+Enter to focus the first result.
                        _serp_blocked = True
                        _serp_obs = {
                            "result": "environmental_failure",
                            "observation": {
                                "serp_guard": (
                                    "done REJECTED: Chrome is still on "
                                    "the search results page. "
                                    "The click_element landed on a "
                                    "navigation element, not a result. "
                                    "Press Tab once to move focus to the "
                                    "first search result, then press Enter "
                                    "to open it."
                                ),
                                "current_url": _current_url,
                            },
                            "error": None,
                        }
                    return _serp_obs
                elif _on_search_engine and not _on_serp:
                    # Still on a search engine domain but NOT on a
                    # SERP (e.g. bing.com homepage after
                    # clicking the logo). The search failed or
                    # the click navigated away from results.
                    # Tell the LLM to redo the search from the
                    # page's search box.
                    _serp_blocked = True
                    _serp_obs = {
                        "result": "environmental_failure",
                        "observation": {
                            "serp_guard": (
                                "done REJECTED: Chrome is still on "
                                f"a search engine page ({_current_url}), "
                                "not on an external result page. "
                                "The previous click landed on a "
                                "navigation element instead of a "
                                "search result. You need to: "
                                "1) type_element to enter the search "
                                "query in the page's search box, "
                                "2) press_key enter, 3) wait 2 seconds, "
                                "4) click_element with role='link' "
                                f"name='' index={_goal_result_index} "
                                "to click the correct result."
                            ),
                            "current_url": _current_url,
                        },
                        "error": None,
                    }
                    return _serp_obs
        _STUB_RE = re.compile(
            r'\b(TODO|FIXME|HACK|XXX)\b'
            r'|#\s*placeholder'
            r'|raise\s+NotImplementedError',
            re.I,
        )
        _stub_files: list[str] = []
        # Check write_file content in history — keep only LAST write per path
        # so a rewritten file doesn't trigger on stale earlier content.
        _latest_writes: dict[str, str] = {}
        for _he in self._history:
            if _he.get("action") == "write_file" and _he.get("result") == "success":
                _wf_path = (_he.get("args") or {}).get("path", "")
                _wf_content = (_he.get("args") or {}).get("content", "")
                if _wf_path:
                    _latest_writes[_wf_path] = _wf_content
        for _wf_path, _wf_content in _latest_writes.items():
            if _wf_content and _STUB_RE.search(_wf_content):
                _stub_files.append(_wf_path)
        # Also check workspace_cache for files mentioned in subtask desc
        if self._goaltracker and not _stub_files:
            _cur_st_stub = (
                self._goaltracker.subtasks[self._goaltracker.current_index]
                if self._goaltracker.current_index < len(self._goaltracker.subtasks)
                else None
            )
            if _cur_st_stub:
                for _ws_path, _ws_content in self._workspace_cache.items():
                    _fname = os.path.basename(_ws_path)
                    if (
                        _fname in _cur_st_stub.description
                        and _ws_content
                        and _STUB_RE.search(_ws_content)
                    ):
                        _stub_files.append(_ws_path)
        if _stub_files and getattr(self, '_stub_bounce_count', 0) < 2:
            self._stub_bounce_count = getattr(self, '_stub_bounce_count', 0) + 1
            _stub_list = ", ".join(os.path.basename(p) for p in _stub_files[:3])
            _stub_obs = {
                "result": "environmental_failure",
                "observation": {
                    "done_blocked": (
                        f"done REJECTED: Files still contain TODO/placeholder "
                        f"stubs: {_stub_list}. Replace ALL stub function "
                        f"bodies with complete working implementations. "
                        f"Do NOT call done until every TODO is replaced "
                        f"with real code."
                    ),
                },
                "error": None,
            }
            return _stub_obs

        return None

    # =================================================================
    # Brain-loop helper: handle batch path
    # =================================================================

    def _handle_batch_path(self, act_payload: list, iteration: int,
                           messages: list, full_response: str,
                           t_start: float | None) -> str:
        """Handle the batch action path.

        Returns a control signal string:
        - "done", "abort", "halt": caller should break
        - "schema_error": caller should handle schema error counting + continue
        - "ok": batch consumed the iteration, caller should continue
        """
        # Fix P0-trunc: Trim truncated batch tail — if the last
        # item is a non-dict (stray string/null from token-limit
        # truncation), drop it instead of rejecting the whole batch.
        if len(act_payload) >= 2 and not isinstance(act_payload[-1], dict):
            _dropped = repr(act_payload[-1])[:120]
            print(f"  [batch] Trimmed truncated tail: {_dropped}", flush=True)
            act_payload = act_payload[:-1]

        # Validate all actions in the batch up front
        batch_valid = True
        for batch_item in act_payload:
            # Non-dict items (stray strings, null) are invalid
            if not isinstance(batch_item, dict):
                _bad_item = repr(batch_item)[:120]
                truncated_raw = full_response[:ARGS_TRUNCATION_CHARS]
                synth_dict = {
                    "action": "schema_validation_error",
                    "args": {"raw_response": truncated_raw},
                }
                _bad_err = SchemaValidationError(
                    f"batch item is not a dict: {_bad_item}"
                )
                self.observer.observe(
                    synth_dict, result=None, error=_bad_err,
                    attempt=1, t_start=t_start,
                    llm_response=full_response, llm_messages=messages,
                )
                self._last_obs = self.observer.last_entry
                if self._last_obs:
                    self._history.append(self._make_history_entry(
                        synth_dict, self._last_obs,
                    ))
                self._fire_on_update(iteration, synth_dict, self._last_obs)
                batch_valid = False
                break
            # Skip conditional nodes — they're validated at eval time
            if "if" in batch_item and "action" not in batch_item:
                continue
            try:
                self._validate_act(batch_item)
            except SchemaValidationError as val_err:
                truncated_raw = full_response[:ARGS_TRUNCATION_CHARS]
                synth_dict = {
                    "action": "schema_validation_error",
                    "args": {"raw_response": truncated_raw},
                }
                self.observer.observe(
                    synth_dict, result=None, error=val_err,
                    attempt=1, t_start=t_start,
                    llm_response=full_response, llm_messages=messages,
                )
                self._last_obs = self.observer.last_entry
                if self._last_obs:
                    self._history.append(self._make_history_entry(
                        synth_dict, self._last_obs,
                    ))
                self._fire_on_update(iteration, synth_dict, self._last_obs)
                batch_valid = False
                break
        if not batch_valid:
            return "schema_error"  # re-call LLM

        # Loop detection: batch_engine.record_act() handles per-action
        # recording inside execute_batch — no pre-batch record needed.
        first_act = act_payload[0] if act_payload else None
        self._consecutive_chat_count = 0

        batch_result = self._execute_batch(act_payload, iteration, llm_messages=messages, llm_response=full_response)

        if batch_result in ("done", "abort"):
            return batch_result
        if batch_result == "halt":
            return "halt"

        # -- Batch read-stall tracking --------------------------------
        # Individual batch actions bypass the single-action stall
        # counter, so we track it here: if EVERY non-conditional
        # action in the batch was readonly, increment; otherwise reset.
        _READONLY_SET = {"read_file", "list_dir", "search_codebase", "ast_search", "checkpoint"}
        _batch_action_names = [
            a["action"]
            for a in act_payload
            if "action" in a  # skip conditional {if/then/else} nodes
        ]
        _readonly_in_batch = [
            a for a in _batch_action_names if a in _READONLY_SET
        ]
        if _batch_action_names and all(
            a in _READONLY_SET for a in _batch_action_names
        ):
            self._consecutive_readonly = (
                getattr(self, "_consecutive_readonly", 0)
                + 1  # count iterations, not individual batch items
            )
            # Track per-file read counts for same-file loop detection
            for a in act_payload:
                if isinstance(a, dict) and a.get("action") == "read_file":
                    rpath = (a.get("args") or {}).get("path", "") or (a.get("args") or {}).get("file_path", "")
                    if rpath:
                        self._task_file_read_counts[rpath] = self._task_file_read_counts.get(rpath, 0) + 1
                        # Fix P1-reread: detect re-reading a just-written file
                        if rpath == self._last_written_path:
                            self._post_write_reread_count += 1
            # Fix D: Post-stall read injection — warn if reading after stall nudge
            if self._stall_nudge_active:
                _stall_read_obs = {
                    "result": "environmental_failure",
                    "observation": {
                        "post_stall_warning": (
                            "STOP READING. You just received a command stall warning. "
                            "Do not read more files — fix the failed command first. "
                            "Review the stall hint above and try a different approach."
                        ),
                    },
                    "error": None,
                }
                self._last_obs = _stall_read_obs
        else:
            self._consecutive_readonly = 0
            self._stall_nudge_active = False  # Fix D: clear stall flag on mutating action
            # Count mutating actions toward subtask cap
            _MUTATING = {"write_file", "append_file", "run_command", "run_background", "delete_file"}
            _mutating_count = sum(
                1 for a in act_payload
                if isinstance(a, dict) and a.get("action") in _MUTATING
            )
            self._subtask_productive_iters += _mutating_count
            if any(
                a.get("action") in ("write_file", "append_file")
                for a in act_payload
                if isinstance(a, dict)
            ):
                self._subtask_has_written = True  # first write arms stall detection
                # Fix P1-reread: track the last written file path
                for _wa in act_payload:
                    if isinstance(_wa, dict) and _wa.get("action") in ("write_file", "append_file"):
                        _wp = (_wa.get("args") or {}).get("path", "")
                        if _wp:
                            self._last_written_path = _wp
                            self._post_write_reread_count = 0
                        # P3: capture API signatures from written content into scratchpad
                        _written_content = (_wa.get("args") or {}).get("content", "")
                        _api_path = _wp or (_wa.get("args") or {}).get("path", "")
                        if _written_content and _api_path and hasattr(self, "_scratchpad"):
                            _api_summary = self._extract_api_summary(_written_content, _api_path)
                            if _api_summary:
                                try:
                                    self._scratchpad.update_files({_api_path: _api_summary})
                                except Exception:
                                    pass  # scratchpad update is best-effort
            # Fix P1-reread: run_command resets reread tracking
            # (re-reading after running a command is legitimate)
            if any(
                isinstance(a, dict) and a.get("action") == "run_command"
                for a in act_payload
            ):
                self._post_write_reread_count = 0

        # -- P0-B: Batch run-command failure stall detection -----------
        _CMD_FAIL_RESULTS = {"command_failed", "environmental_failure"}
        _batch_cmd_names = [
            a for a in act_payload
            if isinstance(a, dict) and a.get("action") == "run_command"
        ]
        if _batch_cmd_names:
            # Grab the tail of history matching this batch size.
            # deque does not support slicing, so materialise to list.
            _hist_list = list(self._history)
            _recent_obs = _hist_list[-len(_batch_cmd_names):]
            _fails_in_batch = sum(
                1 for h in _recent_obs
                if h.get("action") == "run_command"
                and h.get("result") in _CMD_FAIL_RESULTS
            )
            _ok_in_batch = sum(
                1 for h in _recent_obs
                if h.get("action") == "run_command"
                and h.get("result") not in _CMD_FAIL_RESULTS
            )
            if _fails_in_batch > 0 and _ok_in_batch == 0:
                self._consecutive_command_failures += _fails_in_batch
                # Track failed commands for handoff
                for h in _recent_obs:
                    if (h.get("action") == "run_command"
                            and h.get("result") in _CMD_FAIL_RESULTS):
                        _fc = (h.get("args") or {}).get("cmd", "")
                        if _fc and _fc not in self._subtask_failed_commands:
                            self._subtask_failed_commands.append(_fc)
            elif _ok_in_batch > 0:
                self._consecutive_command_failures = 0

            if self._consecutive_command_failures >= 10:
                self._halt("command_stall_exceeded")
                return "halt"
            elif self._consecutive_command_failures >= 5:
                # Fix C: signal-specific stall hint
                _stall_hint = self._build_cmd_stall_hint()
                _cmd_nudge_obs = {
                    "result": "environmental_failure",
                    "observation": {
                        "command_stall_warning": (
                            f"You have had {self._consecutive_command_failures} "
                            f"consecutive failed commands. {_stall_hint}"
                        ),
                    },
                    "error": None,
                }
                self._last_obs = _cmd_nudge_obs
                self._stall_nudge_active = True  # Fix D: arm post-stall read warning

        # -- Same-file loop detection: halt if one file read >15 times ---
        _SAME_FILE_READ_HALT = 15
        _SAME_FILE_READ_WARN = 8
        _worst_file_reads = max(self._task_file_read_counts.values()) if self._task_file_read_counts else 0
        if _worst_file_reads >= _SAME_FILE_READ_HALT:
            self._halt("read_stall_exceeded")
            return "halt"
        elif _worst_file_reads == _SAME_FILE_READ_WARN:
            _worst_file = max(self._task_file_read_counts, key=self._task_file_read_counts.get)
            _same_file_obs = {
                "result": "environmental_failure",
                "observation": {
                    "same_file_loop_warning": (
                        f"You have read '{_worst_file}' {_worst_file_reads} times "
                        "in this subtask. The file content is already in "
                        "your WORKSPACE STATE. Stop reading and start "
                        "writing changes with write_file. Continued "
                        "re-reading will terminate this subtask."
                    ),
                },
                "error": None,
            }
            self._last_obs = _same_file_obs

        # Fix P1-reread: halt on re-reading just-written file ≥4 times
        # (2 was too aggressive — writing then reading back 1-3x is normal)
        if self._post_write_reread_count >= 4:
            print(
                f"  [stall] Post-write reread loop: {self._last_written_path!r} "
                f"read {self._post_write_reread_count}x after being written",
                flush=True,
            )
            self._halt("post_write_reread_loop")
            return "halt"

        # Post-write stall detection
        if (
            READ_STALL_THRESHOLD > 0
            and self._subtask_has_written
            and self._consecutive_readonly >= READ_STALL_THRESHOLD * 2
        ):
            # Hard cap: agent ignored the warning and kept reading.
            self._halt("read_stall_exceeded")
            return "halt"
        elif (
            READ_STALL_THRESHOLD > 0
            and self._subtask_has_written
            and self._consecutive_readonly >= READ_STALL_THRESHOLD
            and not self._read_stall_warned
        ):
            self._read_stall_warned = True
            _batch_stall_obs = {
                "result": "environmental_failure",
                "observation": {
                    "read_stall_warning": (
                        f"You have submitted {self._consecutive_readonly} "
                        "consecutive batches containing only read-only "
                        "actions (read_file / list_dir) without writing "
                        "any changes or running any commands. Check "
                        "your TASK MEMORY (files section) for what you "
                        "already know, then use write_file to create or "
                        "modify the file you need. If you continue "
                        "reading without acting, this subtask will be "
                        "terminated."
                    ),
                },
                "error": None,
            }
            self._last_obs = _batch_stall_obs
            # Do NOT reset _consecutive_readonly — escalate to halt
            # at 2× threshold if the agent keeps reading.

        # Pre-write stall detection: catch stuck read loops even
        # before the first write (uses higher thresholds)
        _PRE_WRITE_WARN = 8
        _PRE_WRITE_HALT = 15
        if (
            not self._subtask_has_written
            and self._consecutive_readonly >= _PRE_WRITE_HALT
        ):
            self._halt("read_stall_exceeded")
            return "halt"
        elif (
            not self._subtask_has_written
            and self._consecutive_readonly >= _PRE_WRITE_WARN
            and not self._read_stall_warned
        ):
            self._read_stall_warned = True
            _pre_write_obs = {
                "result": "environmental_failure",
                "observation": {
                    "pre_write_stall_warning": (
                        f"You have done {self._consecutive_readonly} "
                        "consecutive read-only iterations without "
                        "writing ANY changes. The files you need are "
                        "likely already in your WORKSPACE STATE. You "
                        "MUST use write_file to implement changes NOW. "
                        "Continued reading will terminate this subtask."
                    ),
                },
                "error": None,
            }
            self._last_obs = _pre_write_obs

        # "checkpoint", "failure", "exhausted" → loop detection, then re-call LLM
        if self.is_loop_detected():
            self._halt("loop_detected")
            return "halt"

        # Telegram / budget warning / checkpoint — same as single-action path
        if (
            self._mode == "autonomous"
            and iteration % TELEGRAM_PROGRESS_INTERVAL == 0
        ):
            action_label = first_act.get("action", "batch") if first_act else "batch"
            self._telegram_notify(
                f"🔄 Progress: iteration {iteration}/{MAX_ITERATIONS_PER_TASK}\n"
                f"Last batch: {len(act_payload)} actions, first={action_label}\n"
                f"Task cost: ${self._task_cost_usd:.4f}\n"
                f"Monthly: ${self._monthly_cost_usd:.4f}"
            )
        if (
            not self._monthly_budget_warning_sent
            and self._monthly_cost_usd >= MONTHLY_BUDGET_CAP * 0.80
        ):
            self._monthly_budget_warning_sent = True
            self._telegram_notify(
                f"⚠️ Monthly budget at {self._monthly_cost_usd / MONTHLY_BUDGET_CAP * 100:.0f}%\n"
                f"Spent: ${self._monthly_cost_usd:.2f} / ${MONTHLY_BUDGET_CAP:.2f}"
            )
        self._write_checkpoint()
        if self._mode == "autonomous":
            self._pause_wait()
        return "ok"

    # =================================================================
    # Brain-loop helper: check single-action stalls
    # =================================================================

    def _check_single_action_stalls(self, act_dict: dict, iteration: int) -> bool:
        """Check for stall/loop conditions in the single-action path.

        Returns True if brain_loop should halt (break), False to continue.
        Calls self._halt() internally when halting.
        """
        action_name = act_dict.get('action', '')
        action_args = act_dict.get('args') or {}
        _obs_result = (self._last_obs or {}).get("result", "")

        # -- P0-B: Run-command failure stall detection (single-action) ----
        if action_name == "run_command":
            if _obs_result in ("command_failed", "environmental_failure"):
                self._consecutive_command_failures += 1
                _fc = (action_args or {}).get("cmd", "")
                if _fc and _fc not in self._subtask_failed_commands:
                    self._subtask_failed_commands.append(_fc)
            else:
                self._consecutive_command_failures = 0

            if self._consecutive_command_failures >= 10:
                self._halt("command_stall_exceeded")
                return True
            elif self._consecutive_command_failures >= 5:
                # Fix C: signal-specific stall hint
                _stall_hint = self._build_cmd_stall_hint()
                _cmd_nudge_obs = {
                    "result": "environmental_failure",
                    "observation": {
                        "command_stall_warning": (
                            f"You have had {self._consecutive_command_failures} "
                            f"consecutive failed commands. {_stall_hint}"
                        ),
                    },
                    "error": None,
                }
                self._last_obs = _cmd_nudge_obs
                self._stall_nudge_active = True  # Fix D: arm post-stall read warning

        # -- Loop detection (outside inner while-loop) --------------------
        if self.is_loop_detected():
            self._halt("loop_detected")
            return True

        # -- Read-only stall detection ------------------------------------
        _READONLY_ACTIONS = {"read_file", "list_dir", "search_codebase", "ast_search", "checkpoint"}
        _MUTATING_ACTIONS = {"write_file", "append_file", "run_command", "run_background", "delete_file"}
        if action_name in _READONLY_ACTIONS:
            self._consecutive_readonly = getattr(
                self, "_consecutive_readonly", 0
            ) + 1
            # Track per-file read counts for same-file loop detection
            if action_name == "read_file":
                rpath = (act_dict.get("args") or {}).get("path", "") or (act_dict.get("args") or {}).get("file_path", "")
                if rpath:
                    self._task_file_read_counts[rpath] = self._task_file_read_counts.get(rpath, 0) + 1
                    # Fix P1-reread: detect re-reading a just-written file
                    if rpath == self._last_written_path:
                        self._post_write_reread_count += 1
            # Fix D: Post-stall read injection — warn if reading after stall nudge
            if self._stall_nudge_active:
                _stall_read_obs = {
                    "result": "environmental_failure",
                    "observation": {
                        "post_stall_warning": (
                            "STOP READING. You just received a command stall warning. "
                            "Do not read more files — fix the failed command first. "
                            "Review the stall hint above and try a different approach."
                        ),
                    },
                    "error": None,
                }
                self._last_obs = _stall_read_obs
        else:
            self._consecutive_readonly = 0
            self._stall_nudge_active = False  # Fix D: clear stall flag on mutating action
            if action_name in _MUTATING_ACTIONS:
                self._subtask_productive_iters += 1
            if action_name in ("write_file", "append_file"):
                self._subtask_has_written = True  # first write arms stall detection
                # Fix P1-reread: track last written path
                _wp = (act_dict.get("args") or {}).get("path", "")
                if _wp:
                    self._last_written_path = _wp
                    self._post_write_reread_count = 0  # reset on new write target
                # P3: capture API signatures from written content into scratchpad
                _written_content = (act_dict.get("args") or {}).get("content", "")
                _api_path = _wp or (act_dict.get("args") or {}).get("path", "")
                if _written_content and _api_path and hasattr(self, "_scratchpad"):
                    _api_summary = self._extract_api_summary(_written_content, _api_path)
                    if _api_summary:
                        try:
                            self._scratchpad.update_files({_api_path: _api_summary})
                        except Exception:
                            pass  # scratchpad update is best-effort
            if action_name in ("run_command", "run_background"):
                # Fix P1-reread: run_command resets reread tracking
                # (re-reading after a command to check output is legitimate)
                self._post_write_reread_count = 0

        # -- Same-file loop detection: halt if one file read >15 times ---
        _SAME_FILE_READ_HALT = 15
        _SAME_FILE_READ_WARN = 8
        _worst_file_reads = max(self._task_file_read_counts.values()) if self._task_file_read_counts else 0
        if _worst_file_reads >= _SAME_FILE_READ_HALT:
            self._halt("read_stall_exceeded")
            return True
        elif _worst_file_reads == _SAME_FILE_READ_WARN:
            _worst_file = max(self._task_file_read_counts, key=self._task_file_read_counts.get)
            _same_file_obs = {
                "result": "environmental_failure",
                "observation": {
                    "same_file_loop_warning": (
                        f"You have read '{_worst_file}' {_worst_file_reads} times "
                        "in this subtask. The file content is already in "
                        "your WORKSPACE STATE. Stop re-reading and use "
                        "write_file to implement changes. Continued "
                        "re-reading will terminate this subtask."
                    ),
                },
                "error": None,
            }
            self._last_obs = _same_file_obs

        # Fix P1-reread: halt on re-reading just-written file ≥4 times
        if self._post_write_reread_count >= 4:
            print(
                f"  [stall] Post-write reread loop: {self._last_written_path!r} "
                f"read {self._post_write_reread_count}x after being written",
                flush=True,
            )
            self._halt("post_write_reread_loop")
            return True

        # Post-write stall detection
        if (
            READ_STALL_THRESHOLD > 0
            and self._subtask_has_written
            and self._consecutive_readonly >= READ_STALL_THRESHOLD * 2
        ):
            # Hard cap: agent ignored the warning and kept reading.
            self._halt("read_stall_exceeded")
            return True
        elif (
            READ_STALL_THRESHOLD > 0
            and self._subtask_has_written
            and self._consecutive_readonly >= READ_STALL_THRESHOLD
            and not self._read_stall_warned
        ):
            self._read_stall_warned = True
            # Inject a nudge — fires once; counter keeps incrementing.
            nudge_obs = {
                "result": "environmental_failure",
                "observation": {
                    "read_stall_warning": (
                        f"You have performed {self._consecutive_readonly} "
                        "consecutive read-only actions without writing "
                        "any changes. Check your TASK MEMORY (files "
                        "section) for what you already know, then use "
                        "write_file to create or modify the file you "
                        "need. If you continue reading without acting, "
                        "this subtask will be terminated."
                    ),
                },
                "error": None,
            }
            self._last_obs = nudge_obs
            # Do NOT reset _consecutive_readonly — escalate to halt
            # at 2× threshold if the agent keeps reading.

        # Pre-write stall detection: catch stuck read loops even
        # before the first write (uses higher thresholds)
        _PRE_WRITE_WARN = 8
        _PRE_WRITE_HALT = 15
        if (
            not self._subtask_has_written
            and self._consecutive_readonly >= _PRE_WRITE_HALT
        ):
            self._halt("read_stall_exceeded")
            return True
        elif (
            not self._subtask_has_written
            and self._consecutive_readonly >= _PRE_WRITE_WARN
            and not self._read_stall_warned
        ):
            self._read_stall_warned = True
            _pre_write_obs = {
                "result": "environmental_failure",
                "observation": {
                    "pre_write_stall_warning": (
                        f"You have done {self._consecutive_readonly} "
                        "consecutive read-only actions without writing "
                        "ANY changes. The files you need are likely "
                        "already in your WORKSPACE STATE. You MUST use "
                        "write_file to implement changes NOW. Continued "
                        "reading will terminate this subtask."
                    ),
                },
                "error": None,
            }
            self._last_obs = _pre_write_obs


        return False

    # =====================================================================
    # Halt helper
    # =====================================================================

    def _halt(self, reason: str) -> None:
        """Set halt reason and fire Telegram notification if autonomous."""
        self._halt_reason = reason
        # Persist scratchpad so handoff data survives abrupt halts.
        try:
            self._scratchpad.save()
        except Exception:
            pass
        if self._mode == "autonomous":
            self._telegram_notify(
                f"🔴 HALT: {reason}\n"
                f"Task: {self._task_id}\n"
                f"Iterations: {self._iteration}\n"
                f"Cost: ${self._task_cost_usd:.4f}"
            )

    # =====================================================================
    # Pause wait helper
    # =====================================================================

    def _pause_wait(self) -> None:
        """Block while pause_event is cleared. Check cancel every 1s."""
        while not self.pause_event.is_set():
            if self.cancel_event.is_set():
                return
            self.pause_event.wait(timeout=1.0)

    # =====================================================================
    # LLM Call
    # =====================================================================

    def _llm_call(
        self,
        messages: list[dict],
        model: str | None = None,
        max_tokens: int | None = None,
        normalize: bool | None = None,
        extra_body: dict | None = None,
    ) -> str:
        """Single LLM API call. Returns raw response text.

        Raises httpx.TimeoutException, httpx.RequestError, httpx.HTTPStatusError.
        Raises ResponseTruncatedError when finish_reason == 'length'.
        Handles cost accumulation on success (and on truncation).

        max_tokens: when None (default), resolved from MODEL_MAX_TOKENS keyed
        by model_id, falling back to DEFAULT_MAX_TOKENS (8192).
        normalize: when True, non-Qwen responses are passed through the
        normalization layer (cheap Qwen reformat call).  When None (default),
        auto-detects based on whether the model belongs to a Qwen-family tier.
        """
        if model is None:
            model = self._model
        if max_tokens is None:
            from config import MODEL_MAX_TOKENS, DEFAULT_MAX_TOKENS
            max_tokens = MODEL_MAX_TOKENS.get(model, DEFAULT_MAX_TOKENS)
        # Merge per-subtask extra_body (e.g. nothink for tier_1) unless overridden by caller
        _extra = extra_body if extra_body is not None else self._extra_body
        try:
            content, cost_usd = self._llm.call(messages, model, max_tokens, extra_body=_extra)
        except ResponseTruncatedError as rte:
            # Still accumulate cost for the truncated call
            self._task_cost_usd += rte.cost_usd
            self._monthly_cost_usd += rte.cost_usd
            self._write_monthly_cost()
            raise
        self._task_cost_usd += cost_usd
        self._monthly_cost_usd += cost_usd
        self._write_monthly_cost()

        # -- Phase 5.4: normalization layer for models with format issues ------
        # Auto-detect: skip normalization for models that output valid action
        # JSON reliably (SKIP_NORMALIZATION_MODELS). Normalize everything else
        # (e.g. qwen3-coder-next which aborts with bare <think>\n<tool_call>\n,
        # Gemini Flash, Claude etc.).
        # Caller can force-override via the normalize parameter.
        should_normalize = normalize
        if should_normalize is None:
            from config import SKIP_NORMALIZATION_MODELS
            should_normalize = model not in SKIP_NORMALIZATION_MODELS
        if should_normalize:
            normalized, norm_cost = self._llm.normalize_response(content)
            self._task_cost_usd += norm_cost
            self._monthly_cost_usd += norm_cost
            if norm_cost > 0:
                self._write_monthly_cost()
            content = normalized

        return content

    # =====================================================================
    # Response Parsing & Validation (delegated to response_parser.py)
    # =====================================================================

    def _parse_response(self, full_response: str) -> tuple[str | None, dict | list[dict]]:
        return response_parser.parse_response(full_response)

    @staticmethod
    def _extract_think_content(full_response: str) -> str | None:
        return response_parser.extract_think_content(full_response)

    def _validate_act(self, act_dict: dict) -> None:
        response_parser.validate_act(act_dict)

    # =====================================================================
    # Dispatch
    # =====================================================================

    def _dispatch(self, act_dict: dict) -> tuple[dict, str | None, str | None]:
        return dispatch_mod.dispatch(self, act_dict)

    # =====================================================================
    # URL Pre-fetch (Layer 1: deterministic content injection)
    # =====================================================================

    def _prefetch_project_files(self) -> None:
        prefetch_mod.prefetch_project_files(self)

    def _inject_subtask_files(self, subtask_goal: str) -> None:
        prefetch_mod.inject_subtask_files(self, subtask_goal)

    def _pre_generate_greenfield_draft(self, subtask) -> None:
        prefetch_mod.pre_generate_greenfield_draft(self, subtask)

    def _prefetch_goal_urls(self) -> None:
        prefetch_mod.prefetch_goal_urls(self)

    @staticmethod
    def _extract_api_summary(content: str, path: str) -> str:
        return prefetch_mod.extract_api_summary(content, path)

    def _build_cmd_stall_hint(self) -> str:
        """Fix C: Build a signal-specific hint for command stall nudges.

        Inspects the last observation for exit_code and stderr patterns
        to provide actionable guidance instead of generic "try different".
        """
        _last_exit = None
        _last_stderr = ""
        if self._last_obs:
            _obs_data = self._last_obs.get("observation", {})
            if isinstance(_obs_data, dict):
                _last_exit = _obs_data.get("exit_code")
                _last_stderr = str(_obs_data.get("stderr", ""))[:300]
        _stderr_lower = _last_stderr.lower()

        if _last_exit in (137, -9):
            return (
                "The process was KILLED (exit code 137 — SIGKILL/OOM). "
                "This usually means excessive memory use. "
                "Try: (1) reduce data size, (2) use a lighter algorithm, "
                "or (3) split processing into smaller chunks."
            )
        if _last_exit in (-15, 143):
            return (
                "The process received SIGTERM (exit code 143). "
                "Try: (1) check if a previous process is still running "
                "on the same port with 'lsof -i:<PORT>', (2) kill stale "
                "processes with 'pkill -f <name>', or (3) use a different port."
            )
        if "address already in use" in _stderr_lower:
            return (
                "A port is already in use (Address already in use). "
                "Kill the existing process: run_command('lsof -ti:<PORT> | xargs kill -9') "
                "or use a different port number."
            )
        if "permission denied" in _stderr_lower:
            return (
                "Permission denied. Try: (1) check file ownership with 'ls -la', "
                "(2) use a path you have write access to (~/genie_workspace/), "
                "or (3) add execute permission with 'chmod +x'."
            )
        if "no such file or directory" in _stderr_lower:
            return (
                "File or directory not found. Check: (1) the path exists, "
                "(2) spelling is correct, (3) create missing dirs with 'mkdir -p'."
            )
        # Default: generic hint
        return (
            "The current approach is NOT working. Try a COMPLETELY DIFFERENT "
            "strategy or call done/abort if the subtask cannot be completed."
        )

    def classify_error(self, error: Exception, action: str) -> str:
        """Classify an exception into one of four error classes."""

        # 1. ElementResolverError fast path
        if isinstance(error, TransientError):
            result = ERROR_CLASS_TRANSIENT
        elif isinstance(error, EnvironmentalError):
            return ERROR_CLASS_ENVIRONMENTAL
        elif isinstance(error, ResourceError):
            return ERROR_CLASS_RESOURCE
        elif isinstance(error, UnrecoverableError):
            return ERROR_CLASS_UNRECOVERABLE
        else:
            # 2. ERROR_CLASSIFICATION_RULES fallback
            result = None
            exc_type_name = type(error).__name__
            exc_msg_lower = str(error).lower()

            for match_type, match_value, err_class in ERROR_CLASSIFICATION_RULES:
                if match_type == "exception_type":
                    if exc_type_name == match_value:
                        result = err_class
                        break
                elif match_type == "message_substr":
                    if match_value in exc_msg_lower:
                        result = err_class
                        break

            # 3. No match → UNRECOVERABLE
            if result is None:
                return ERROR_CLASS_UNRECOVERABLE

        # 4. Idempotency check — only when TRANSIENT
        if result == ERROR_CLASS_TRANSIENT:
            if not ACTION_IDEMPOTENT.get(action, False):
                return ERROR_CLASS_ENVIRONMENTAL

        return result

    # =====================================================================
    # Context Assembly
    # =====================================================================

    def _build_user_turn(
        self, goal: str, iteration: int, task_cost: float,
        monthly_cost: float, last_obs: dict | None,
        history_deque: deque,
    ) -> str:
        """Build the user-turn message for the LLM."""
        with self.registry._registry_lock:
            reg_snapshot = dict(self.registry._registry)
        _scratchpad_text = self._scratchpad.render() if not self._scratchpad.is_empty() else ""
        return context_builder.build_user_turn(
            goal=goal,
            iteration=iteration,
            task_cost=task_cost,
            monthly_cost=monthly_cost,
            last_obs=last_obs,
            history_deque=history_deque,
            registry_snapshot=reg_snapshot,
            cdp_url_reader=self._cdp_read_active_tab_url,
            consecutive_env_failures=self._consecutive_env_failures,
            workspace_cache=self._workspace_cache,
            scratchpad_text=_scratchpad_text,
        )

    @staticmethod
    def _make_history_entry(act_dict: dict, obs_entry: dict) -> dict:
        """Build a history deque entry from act_dict and observer.last_entry."""
        return context_builder.make_history_entry(act_dict, obs_entry)

    # =====================================================================
    # on_update callback
    # =====================================================================

    def _fire_on_update(
        self, iteration: int, act_dict: dict | None,
        obs_entry: dict | None,
        outcome: str | None = None,
        message: str | None = None,
        **kwargs,
    ) -> None:
        """Fire on_update callback with event_dict.

        Extra keyword arguments are merged into the event dict verbatim,
        allowing callers to attach event-specific fields (e.g. subtask
        boundary metadata) without changing the signature.
        """
        if not self._on_update:
            return
        event_dict = {
            "task_id": self._task_id,
            "iteration": iteration,
            "action": act_dict.get("action") if act_dict else None,
            "args": act_dict.get("args") if act_dict else None,
            "result": obs_entry.get("result") if obs_entry else None,
            "observation": obs_entry.get("observation") if obs_entry else None,
            "task_cost_usd": self._task_cost_usd,
            "monthly_cost_usd": self._monthly_cost_usd,
            "estimated_iterations": self._estimated_iterations,
            "task_budget_usd": self._task_budget,
            "outcome": outcome,
            "message": message,
        }
        event_dict.update(kwargs)
        # Tag outcomes that come from within a GoalTracker subtask so the UI
        # knows not to tear down the shared task widget between subtasks.
        if outcome is not None and getattr(self, "_in_goaltracker_subtask", False):
            event_dict["is_subtask_outcome"] = True
        try:
            self._on_update(event_dict)
        except Exception as exc:
            print(
                f"[ORCHESTRATOR] on_update callback error: {exc}",
                file=sys.stderr,
            )

    # =====================================================================
    # Cost tracking
    # =====================================================================

    def _load_monthly_cost(self) -> None:
        """Load monthly cost from cost_monthly.json. Reset on month mismatch."""
        cost, reset = persistence.load_monthly_cost()
        self._monthly_cost_usd = cost
        if reset:
            self._monthly_budget_warning_sent = False

    def _write_monthly_cost(self) -> None:
        """Atomic write to cost_monthly.json."""
        persistence.write_monthly_cost(self._monthly_cost_usd)

    def _increment_task_count(self) -> None:
        """Increment task_count in cost_monthly.json."""
        persistence.increment_task_count(self._monthly_cost_usd)

    # =====================================================================
    # Checkpoint
    # =====================================================================

    def _write_checkpoint(self) -> None:
        """Atomic checkpoint write — autonomous mode only."""
        persistence.write_checkpoint(
            task_id=self._task_id,
            goal=self._original_goal if self._goaltracker else self._goal,
            task_type=self._task_type,
            per_task_budget=self._task_budget,
            iteration=self._iteration,
            sequence=self.observer.sequence,
            cost_usd=self._task_cost_usd,
            last_observation=self._last_obs,
            goaltracker=self._goaltracker.to_dict() if self._goaltracker else None,
            scratchpad=self._scratchpad.to_dict() if self._scratchpad else None,
        )
        # Persist scratchpad to its own file as well (fast restore)
        if self._scratchpad:
            self._scratchpad.save()

    # =====================================================================
    # Telegram
    # =====================================================================

    def _telegram_notify(self, text: str) -> "int | None":
        """Send a Telegram message. Returns message_id on success, else None."""
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            return None
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = json.dumps({
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
        }).encode("utf-8")
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            resp = urllib.request.urlopen(req, timeout=10)
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("result", {}).get("message_id")
        except Exception as exc:
            print(
                f"[ORCHESTRATOR] Telegram notify failed: {exc}",
                file=sys.stderr,
            )
            return None

    def _telegram_notify_keyboard(self, text: str, options: list[str]) -> "int | None":
        """Send a Telegram message with inline keyboard buttons.

        Each option occupies its own row for readability.  An extra
        \"Other\u2026\" button is appended as the last row to allow free-text
        answers via the plain-message fallback path.

        Returns the Telegram message_id on success, or None on failure.
        """
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            return None
        rows = [[{"text": opt, "callback_data": str(i + 1)}] for i, opt in enumerate(options)]
        rows.append([{"text": "Other\u2026", "callback_data": "OTHER"}])
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = json.dumps({
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
            "reply_markup": {"inline_keyboard": rows},
        }).encode("utf-8")
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            resp = urllib.request.urlopen(req, timeout=10)
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("result", {}).get("message_id")
        except Exception as exc:
            print(
                f"[ORCHESTRATOR] Telegram keyboard notify failed: {exc}",
                file=sys.stderr,
            )
            return None

    def _telegram_delete_message(self, message_id: int) -> None:
        """Delete a Telegram message by message_id. Fire-and-forget."""
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID or not message_id:
            return
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/deleteMessage"
        payload = json.dumps({
            "chat_id": TELEGRAM_CHAT_ID,
            "message_id": message_id,
        }).encode("utf-8")
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            urllib.request.urlopen(req, timeout=10)
        except Exception:
            pass  # silenced — message may already be gone

    def _telegram_answer_callback(self, callback_query_id: str) -> None:
        """Acknowledge a Telegram callback query (clears button loading spinner).

        Fire-and-forget — exceptions are silenced so the listener thread
        never aborts due to a network blip on this non-critical call.
        """
        if not TELEGRAM_BOT_TOKEN:
            return
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/answerCallbackQuery"
        payload = json.dumps({"callback_query_id": callback_query_id}).encode("utf-8")
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            urllib.request.urlopen(req, timeout=5)
        except Exception:
            pass  # deliberately silenced

    def _telegram_listener(self) -> None:
        """Background thread: poll Telegram for inbound commands and answers.

        Handles three update types:
          - callback_query: inline keyboard button taps (clarifying Q answers)
          - plain message (numeric): numeric fallback answer while NOT in freetext mode
          - plain message (text): freetext answer when _clarify_awaiting_freetext is True
          - control messages: "stop", "pause", "resume" (only when no clarify in progress)
        """
        url_base = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
        while not self._telegram_stop_event.is_set():
            try:
                params = urllib.parse.urlencode({
                    "offset": self._telegram_update_offset,
                    "timeout": TELEGRAM_POLL_TIMEOUT,
                })
                url = f"{url_base}?{params}"
                req = urllib.request.Request(url)
                resp = urllib.request.urlopen(req, timeout=TELEGRAM_POLL_TIMEOUT + 5)
                data = json.loads(resp.read().decode("utf-8"))
                if not data.get("ok"):
                    continue
                for update in data.get("result", []):
                    update_id = update.get("update_id", 0)
                    self._telegram_update_offset = update_id + 1

                    # --- inline keyboard button press -----------------------
                    cq = update.get("callback_query")
                    if cq:
                        cq_id   = cq.get("id", "")
                        cq_data = cq.get("data", "")
                        self._telegram_answer_callback(cq_id)
                        if cq_data == "OTHER":
                            self._clarify_awaiting_freetext = True
                            prompt_msg_id = self._telegram_notify("Type your answer:")
                            self._clarify_freetext_prompt_msg_id = prompt_msg_id
                            if prompt_msg_id is not None:
                                t = threading.Timer(
                                    300.0, self._telegram_delete_message,
                                    args=(prompt_msg_id,)
                                )
                                t.daemon = True
                                t.start()
                                self._clarify_freetext_delete_timer = t
                        else:
                            # Numeric option selected — answer directly
                            self.answer_clarification(cq_data)
                        continue

                    # --- plain text message ---------------------------------
                    msg  = update.get("message", {})
                    text = msg.get("text", "").strip()
                    if not text:
                        continue
                    text_lower = text.lower()

                    if self._clarify_awaiting_freetext:
                        self.answer_clarification(text)
                        self._clarify_awaiting_freetext = False
                        # Delete the "Type your answer:" prompt immediately
                        if self._clarify_freetext_delete_timer is not None:
                            self._clarify_freetext_delete_timer.cancel()
                            self._clarify_freetext_delete_timer = None
                        if self._clarify_freetext_prompt_msg_id is not None:
                            self._telegram_delete_message(self._clarify_freetext_prompt_msg_id)
                            self._clarify_freetext_prompt_msg_id = None
                    elif text.isdigit():
                        # Numeric fallback: user typed number instead of tapping button
                        self.answer_clarification(text)
                    elif text_lower == "stop":
                        self.cancel()
                    elif text_lower == "pause":
                        self.pause_event.clear()
                    elif text_lower == "resume":
                        self.pause_event.set()
            except Exception as exc:
                print(
                    f"[ORCHESTRATOR] Telegram listener error: {exc}",
                    file=sys.stderr,
                )
                time.sleep(2)
