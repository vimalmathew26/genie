"""Sequence execution speed path — plan-to-action-list conversion and runner. Extracted from planner.py."""
from __future__ import annotations

import json
import re
import time
from typing import TYPE_CHECKING

import httpx

from config import (
    APP_PROFILES,
    ERROR_CLASS_TRANSIENT,
    ERROR_CLASS_UNRECOVERABLE,
    LLM_SERVICE_ERROR_CODES,
    MAX_ACTION_RETRIES,
    MAX_LLM_RETRIES,
    RETRY_BACKOFF_SECONDS,
    TASK_MODEL_MAP,
)
from exceptions import ResponseTruncatedError, SchemaValidationError
import context_builder

if TYPE_CHECKING:
    from orchestrator import GenieOrchestrator



# =========================================================================
# Sequence Execution Speed Path
# =========================================================================

SEQUENCE_SYSTEM_PROMPT = """\
You are Genie, a desktop automation agent on Ubuntu GNOME (Xorg).
Given a goal, return the COMPLETE action sequence as a JSON array.

## Response format
Return ONLY a JSON array — no markdown, no explanation, no comments.
Each element must be either:
  - An action: {"action": "<name>", "args": {<key>: <value>}}
  - A conditional: {"if": "<condition>", "then": [...], "else": [...]}
The array MUST end with a done or abort action (or a conditional whose branches end with one).

## Action reference (names and required args)
open_app:    {app}                            — launch app; label format: "chrome_1"
press_key:   {key}                            — key string, colon-separated: "ctrl:l"
type_text:   {text}                           — type raw text into focused window
click_element: {app, role, name}
               Optional: index (0-based, default 0). index=1 clicks the SECOND result.
type_element:  {app, role, name, text}
read_element:  {app, role, name}
               Optional: index (0-based, default 0). index=1 reads the SECOND link.
look:          {app}                          — screenshot + vision description of window state
               Optional: question (specific question about what's visible)
focus_window:  {app}
wait:          {seconds}
run_command:   {cmd}
write_file:    {path, content}                — write content to a file (creates dirs, overwrites)
append_file:   {path, content}                — append content to an existing file
read_file:     {path}                         — read content of a file from disk
checkpoint:    {}                             — stop execution, observe state, re-call LLM
list_clipboard_history: {}                    — returns GPaste history as [{index, preview}]
get_clipboard_item: {index}                   — returns full text of history item
paste_clipboard_item: {index}                 — promotes item to active clipboard (follow with press_key ctrl:v)
done:          {summary, message}
abort:         {reason}
chat:          {message}                      — reply to the user directly (no desktop action needed)

## Conditional execution
Use {"if": "<condition>", "then": [...], "else": [...]} to branch on the last action's result.
Conditions reference result dict keys: exit_code, status, stdout, stderr, timed_out.
Operators: ==, !=, >, <, >=, <=, in, not in.
Example: {"if": "exit_code != 0", "then": [...], "else": [...]}

## write_file content rule
In write_file / append_file content, ALWAYS include the LITERAL value — NEVER use placeholder
tokens like ${result.stdout}, <output_from_run_command>, or <version>.
If you need to chain a run_command result into a write_file, use a conditional or
pipe the output in the shell: run_command {cmd: "python3 -c 'print(6*7)' > /tmp/out.txt"}

## Chrome navigation rule
Chrome address bar is NOT reachable via type_element.
To navigate to a URL in Chrome: press_key ctrl:l → type_text <url> → press_key enter
WARNING: the address bar uses Chrome's DEFAULT search engine (Google).
★ For Google searches: press_key ctrl:l → type_text <search query> → press_key enter.
  Do NOT navigate to google.com first — the address bar already IS a Google search bar.
To search on Bing, DuckDuckGo, etc.: navigate to that engine's URL first, then
use type_element {role: "textfield", name: ""} to type in the page's own search box.

## Clicking elements
- name="" (empty string) means "first element of that role on the page".
  Use it when you don't know the exact element text yet.
- index (optional, 0-based): 0=first (default), 1=second, 2=third, etc.
  Example — click first search result link:
    {"action": "wait", "args": {"seconds": 2}}
    {"action": "click_element", "args": {"app": "chrome_1", "role": "link", "name": ""}}
  Example — click SECOND search result link:
    {"action": "wait", "args": {"seconds": 2}}
    {"action": "click_element", "args": {"app": "chrome_1", "role": "link", "name": "", "index": 1}}
- ALWAYS emit a wait {seconds: 2} before click_element when the previous action
  was press_key enter, type_text, open_app, or any page navigation. This lets the
  page render before the accessibility tree is queried.
- After any stochastic action (click_element, type_element, open_app), emit checkpoint
  if you need to see the result before deciding the next step.

## Rules
- NEVER use coordinate-click unless there is literally no other way.
- The done message field should be a natural, user-facing sentence.
- DO NOT open a browser, terminal, or any app for tasks that only require writing files,
  running commands, or producing content from knowledge. Use write_file / run_command directly.
  NEVER use open_app for shell tasks — there is NO terminal app available.
- For "run a command and save the output" tasks, use:
  run_command {cmd: "your_command > /path/to/output.txt"}   (shell redirection)
  or a conditional chain: run_command → write_file.
- Emit ONLY the JSON array. No other output.
- If the goal is a greeting, question, or conversational input requiring no desktop action,
  respond with EXACTLY: [{"action": "chat", "args": {"message": "<your reply>"}}, {"action": "done", "args": {"summary": "Conversational response", "message": ""}}]
  Never emit only done for a conversational input — always chat first.
"""


def sequence_phase(orch: GenieOrchestrator) -> tuple[list[dict] | None, str | None, list | None]:
    """Single LLM call that returns the full action sequence for the goal.

    Returns (sequence, llm_response, llm_messages) on success,
    or (None, None, None) on failure.
    """
    with orch.registry._registry_lock:
        reg = dict(orch.registry._registry)

    reg_lines = []
    for label, entry in sorted(reg.items()):
        status = "open" if entry.get("wid") else "closed"
        reg_lines.append(f"  {label}: {status}")

    reg_str = "\n".join(reg_lines) if reg_lines else "  (empty)"
    apps_str = ", ".join(sorted(APP_PROFILES.keys()))

    user_content = (
        f"GOAL: {orch._goal}\n\n"
        f"REGISTRY (currently open windows):\n{reg_str}\n\n"
        f"AVAILABLE APPS: {apps_str}\n\n"
        "Return the complete action sequence as a JSON array."
    )

    messages = [
        {"role": "system", "content": SEQUENCE_SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

    raw = None
    _plan_model    = TASK_MODEL_MAP.get("planning", TASK_MODEL_MAP["default"])[0]
    _plan_fallback = TASK_MODEL_MAP.get("planning", TASK_MODEL_MAP["default"])[1]
    # nothink: sequence phase wants a JSON array, not a reasoning trace.
    # 512 tokens: 5-10 actions × ~50 tokens each — fail fast if model won't fit.
    _SEQ_EXTRA_BODY = {"reasoning": {"budget_tokens": 0}}
    _SEQ_MAX_TOKENS = 512
    model = _plan_model
    for attempt in range(1, MAX_LLM_RETRIES + 1):
        try:
            raw = orch._llm_call(messages, model=model, max_tokens=_SEQ_MAX_TOKENS,
                                 extra_body=_SEQ_EXTRA_BODY)
            break
        except ResponseTruncatedError as rte:
            raw = rte.partial_content
            break
        except (httpx.TimeoutException, httpx.RequestError):
            if attempt == MAX_LLM_RETRIES:
                return None, None, None
            time.sleep(
                RETRY_BACKOFF_SECONDS[
                    min(attempt - 1, len(RETRY_BACKOFF_SECONDS) - 1)
                ]
            )
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in LLM_SERVICE_ERROR_CODES:
                model = _plan_fallback
                try:
                    raw = orch._llm_call(messages, model=model, max_tokens=_SEQ_MAX_TOKENS,
                                         extra_body=_SEQ_EXTRA_BODY)
                except Exception:
                    return None, None, None
                break
            return None, None, None

    if not raw:
        return None, None, None

    sequence = parse_sequence_json(raw)
    if sequence is None:
        return None, None, None

    for act in sequence:
        try:
            orch._validate_act(act)
        except SchemaValidationError:
            return None, None, None

    if not sequence:
        return None, None, None
    last_action = sequence[-1].get("action")
    if last_action not in ("done", "abort"):
        return None, None, None

    return sequence, raw, messages


def parse_sequence_json(text: str) -> list[dict] | None:
    """Extract and parse a JSON array from the sequence LLM response."""
    text = text.strip()

    code_block = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if code_block:
        text = code_block.group(1).strip()

    try:
        result = json.loads(text)
        if isinstance(result, list):
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


def run_sequence(orch: GenieOrchestrator, sequence: list[dict],
                 llm_response: str | None = None,
                 llm_messages: list | None = None) -> bool:
    """Execute a pre-planned action sequence without further LLM calls.

    Returns True  if the task reached done/abort/halt.
    Returns False if a non-fatal error occurred mid-sequence.
    """
    orch._iteration = 1

    for seq_idx, act_dict in enumerate(sequence):
        action_name = act_dict["action"]
        action_args = act_dict["args"]

        if action_name == "done":
            # Guard: block done when the script hasn't been verified.
            # Case A: write_file succeeded but no run_command since
            # Case B: last run_command had non-zero exit_code
            _last_run_d: dict | None = None
            _last_run_d_pos = -1
            _last_write_d_pos = -1
            for _di, _de in enumerate(orch._history):
                if _de.get("action") == "run_command":
                    _last_run_d = _de
                    _last_run_d_pos = _di
                if (
                    _de.get("action") == "write_file"
                    and _de.get("result") == "success"
                ):
                    _last_write_d_pos = _di
            # Skip for non-executable deliverables (docs, data, config)
            _NON_EXEC_EXTS_D = (
                ".md", ".txt", ".rst", ".html", ".xml",
                ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
                ".csv", ".tsv", ".log",
            )
            _last_write_d_path = (
                orch._history[_last_write_d_pos].get("args", {}).get("path", "")
                if _last_write_d_pos >= 0 else ""
            )
            _last_write_d_is_doc = any(
                _last_write_d_path.endswith(ext) for ext in _NON_EXEC_EXTS_D
            )
            _wrote_without_rerun_d = (
                _last_write_d_pos > _last_run_d_pos and not _last_write_d_is_doc
            )
            # Suppress when a doc-type deliverable was written after
            # the failed run — that run was an abandoned intermediate step.
            _last_run_d_failed = (
                _last_run_d is not None
                and ((_last_run_d.get("observation", {}).get("exit_code") or 0) != 0)
                and not (_last_write_d_is_doc and _last_write_d_pos > _last_run_d_pos)
            )
            if _wrote_without_rerun_d:
                _wd_path = (
                    orch._history[_last_write_d_pos].get("args", {})
                    .get("path", "<file>")
                )
                _block_obs = {
                    "result": "environmental_failure",
                    "observation": {
                        "done_blocked": (
                            f"done REJECTED: You wrote the fix to "
                            f"'{_wd_path}' but have NOT re-run the "
                            f"script since then. "
                            f"REQUIRED NEXT ACTION: run_command to "
                            f"execute the fixed script and verify "
                            f"exit_code=0. Do not call done again "
                            f"until that run_command succeeds."
                        ),
                    },
                    "error": None,
                }
                orch._last_obs = _block_obs
                orch._history.append({
                    "action": "done",
                    "args": action_args,
                    "result": _block_obs["result"],
                    "observation": _block_obs["observation"],
                })
                orch._fire_on_update(orch._iteration, act_dict, orch._last_obs)
                return False
            elif _last_run_d_failed:
                _d_ec = _last_run_d.get("observation", {}).get("exit_code")
                _d_cmd = _last_run_d.get("args", {}).get("cmd", "<unknown>")
                _d_stderr = str(
                    _last_run_d.get("observation", {}).get("stderr", "")
                )[:200]
                _block_obs = {
                    "result": "environmental_failure",
                    "observation": {
                        "done_blocked": (
                            f"done REJECTED: The script is still broken. "
                            f"run_command('{_d_cmd}') returned "
                            f"exit_code={_d_ec}. stderr: {_d_stderr!r}. "
                            f"REQUIRED NEXT ACTIONS: (1) write_file with "
                            f"the corrected content, then (2) run_command "
                            f"to verify exit_code=0. Do NOT call done again "
                            f"until the verification run_command succeeds."
                        ),
                    },
                    "error": None,
                }
                orch._last_obs = _block_obs
                orch._history.append({
                    "action": "done",
                    "args": action_args,
                    "result": _block_obs["result"],
                    "observation": _block_obs["observation"],
                })
                orch._fire_on_update(orch._iteration, act_dict, orch._last_obs)
                return False  # fall back to brain loop
            # ----------------------------------------------------------------
            t_s = time.time()
            orch.observer.observe(
                act_dict, result={"status": "ok"}, error=None,
                attempt=1, t_start=t_s,
                llm_response=llm_response, llm_messages=llm_messages,
            )
            orch._last_obs = orch.observer.last_entry
            orch._outcome = "done"
            orch._summary = action_args.get("summary", "")
            orch._fire_on_update(
                orch._iteration, act_dict, orch._last_obs,
                outcome="done",
                message=action_args.get("message", ""),
            )
            return True

        if action_name == "abort":
            # Guard: block abort when the most-recent run_command in history
            # returned exit_code=0 — the script ran successfully, so aborting
            # is contradictory.  Mirrors the same guard in batch_engine.py.
            _last_run_s: dict | None = next(
                (e for e in reversed(orch._history)
                 if e.get("action") == "run_command"),
                None,
            )
            _last_exit_s = (
                _last_run_s.get("observation", {}).get("exit_code")
                if _last_run_s is not None else None
            )
            if _last_exit_s is not None and _last_exit_s == 0:
                _block_obs = {
                    "result": "environmental_failure",
                    "observation": {
                        "abort_blocked": (
                            "abort REJECTED: The last run_command exited with "
                            "exit_code=0 — the script ran successfully. "
                            "Call done to report success, or re-read the output "
                            "if you need to verify the content. "
                            "Do NOT abort when the command succeeded."
                        ),
                    },
                    "error": None,
                }
                orch._last_obs = _block_obs
                orch._history.append({
                    "action": "abort",
                    "args": action_args,
                    "result": _block_obs["result"],
                    "observation": _block_obs["observation"],
                })
                orch._fire_on_update(orch._iteration, act_dict, orch._last_obs)
                # Fall back to brain loop to handle the failure observation
                return False
            t_s = time.time()
            orch.observer.observe(
                act_dict, result={"status": "ok"}, error=None,
                attempt=1, t_start=t_s,
                llm_response=llm_response, llm_messages=llm_messages,
            )
            orch._last_obs = orch.observer.last_entry
            orch._outcome = "abort"
            orch._summary = action_args.get("reason", "")
            orch._fire_on_update(
                orch._iteration, act_dict, orch._last_obs,
                outcome="abort",
            )
            return True

        if action_name == "chat":
            t_s = time.time()
            orch.observer.observe(
                act_dict, result={"status": "ok"}, error=None,
                attempt=1, t_start=t_s,
                llm_response=llm_response, llm_messages=llm_messages,
            )
            orch._last_obs = orch.observer.last_entry
            orch._fire_on_update(
                orch._iteration, act_dict, orch._last_obs,
                message=action_args.get("message"),
            )
            orch._iteration += 1
            continue

        attempt = 1
        result = None
        error = None
        wid = None
        tier = None

        while True:
            t_s = time.time()
            result = None
            error = None
            wid = None
            tier = None

            try:
                result, wid, tier = orch._dispatch(act_dict)
            except Exception as exc:
                error = exc
            finally:
                orch.observer.observe(
                    act_dict, result=result, error=error,
                    attempt=attempt, t_start=t_s,
                    tier=tier, wid=wid,
                    llm_response=llm_response, llm_messages=llm_messages,
                )

            orch._last_obs = orch.observer.last_entry
            if orch._last_obs:
                orch._history.append(
                    context_builder.make_history_entry(act_dict, orch._last_obs)
                )

            if error is not None:
                err_class = orch.classify_error(error, action_name)

                if err_class == ERROR_CLASS_TRANSIENT and attempt < MAX_ACTION_RETRIES:
                    attempt += 1
                    time.sleep(
                        RETRY_BACKOFF_SECONDS[
                            min(attempt - 2, len(RETRY_BACKOFF_SECONDS) - 1)
                        ]
                    )
                    continue

                if err_class == ERROR_CLASS_UNRECOVERABLE:
                    orch._halt("unrecoverable")
                    orch._fire_on_update(orch._iteration, act_dict, orch._last_obs)
                    return True

                orch._fire_on_update(orch._iteration, act_dict, orch._last_obs)
                return False

            break

        # If a shell command failed (non-zero exit), the pre-planned sequence
        # can no longer be trusted — return False so the brain loop takes over
        # and sees the failure output before deciding what to do next.
        if (
            action_name == "run_command"
            and isinstance(result, dict)
            and result.get("exit_code") not in (None, 0)
        ):
            orch._fire_on_update(orch._iteration, act_dict, orch._last_obs)
            return False

        # -- RC-C guard (sequence path): if run_command just succeeded and
        #    a subsequent step in the pre-planned sequence is write_file/
        #    append_file whose content does NOT use _from_run_command
        #    placeholder tokens, the content was pre-composed before seeing
        #    stdout (hallucination risk).  Stop here and return False so the
        #    brain loop re-queries the LLM with the actual command output.
        if action_name == "run_command" and seq_idx + 1 < len(sequence):
            _seq_remaining = sequence[seq_idx + 1:]
            _seq_has_blind_write = any(
                s.get("action") in ("write_file", "append_file")
                and "_from_run_command" not in str(s.get("args", {}).get("content", ""))
                for s in _seq_remaining
            )
            if _seq_has_blind_write:
                orch._fire_on_update(orch._iteration, act_dict, orch._last_obs)
                return False

        orch._fire_on_update(orch._iteration, act_dict, orch._last_obs)

        if orch.cancel_event.is_set():
            orch._halt("cancelled")
            return True

    return False
