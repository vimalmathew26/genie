"""
Genie — batch_engine.py
Batch execution engine, conditional evaluator, and template interpolation.

Extracted from orchestrator.py (§1.5 / §1.6).  Functions that previously
used ``self`` now take ``orch`` (a GenieOrchestrator instance) as their
first argument.  Pure functions (evaluate_condition, interpolate, etc.)
are plain module-level functions with no ``orch`` parameter.
"""
from __future__ import annotations

import collections.abc
import json
import os
import re
import sys
import time
from typing import TYPE_CHECKING

from config import (
    ACTION_IDEMPOTENT,
    ASK_USER_TIERS,
    ARGS_TRUNCATION_CHARS,
    ERROR_CLASS_RESOURCE,
    ERROR_CLASS_TRANSIENT,
    ERROR_CLASS_UNRECOVERABLE,
    MAX_ACTION_RETRIES,
    RETRY_BACKOFF_SECONDS,
)
from exceptions import EnvironmentalError, SchemaValidationError

if TYPE_CHECKING:
    from orchestrator import GenieOrchestrator


# =========================================================================
# Literal parser for conditional expressions (§1.5)
# =========================================================================

def _parse_literal(raw: str):
    """Parse a literal value from a condition RHS string.

    Supports: int, float, bool (true/false), None, and quoted strings.
    Returns the parsed value.
    """
    raw = raw.strip()
    low = raw.lower()
    if low == "none" or low == "null":
        return None
    if low == "true":
        return True
    if low == "false":
        return False
    # Strip quotes
    if (raw.startswith('"') and raw.endswith('"')) or \
       (raw.startswith("'") and raw.endswith("'")):
        return raw[1:-1]
    # Try int
    try:
        return int(raw)
    except ValueError:
        pass
    # Try float
    try:
        return float(raw)
    except ValueError:
        pass
    # Return as string
    return raw


# =========================================================================
# Conditional Expression Evaluator (§1.5)
# =========================================================================

def evaluate_condition(condition: str, last_result: dict) -> bool:
    """Evaluate a simple condition string against the last action's result.

    Supports operators: ==, !=, >, <, >=, <=, 'in', 'not in', 'is None', 'is not None'.
    Left-hand side must be a key from last_result (dotted access not supported).
    Right-hand side is a literal (int, float, string, None).

    Returns False on any parse/eval failure (safe default — takes 'else' branch).
    """
    if not isinstance(condition, str) or not condition.strip():
        return False
    condition = condition.strip()

    # Handle 'is None' / 'is not None'
    m = re.match(r'^(\w+)\s+is\s+not\s+None$', condition)
    if m:
        key = m.group(1)
        return last_result.get(key) is not None
    m = re.match(r'^(\w+)\s+is\s+None$', condition)
    if m:
        key = m.group(1)
        return last_result.get(key) is None

    # Handle 'not in'
    m = re.match(r'^(\w+)\s+not\s+in\s+(.+)$', condition)
    if m:
        key, rhs_raw = m.group(1), m.group(2).strip()
        lhs_val = last_result.get(key, "")
        rhs_val = _parse_literal(rhs_raw)
        if isinstance(rhs_val, str):
            return str(rhs_val) not in str(lhs_val)
        return lhs_val != rhs_val

    # Handle 'in'
    m = re.match(r'^(\w+)\s+in\s+(.+)$', condition)
    if m:
        key, rhs_raw = m.group(1), m.group(2).strip()
        lhs_val = last_result.get(key, "")
        rhs_val = _parse_literal(rhs_raw)
        if isinstance(rhs_val, str):
            return str(rhs_val) in str(lhs_val)
        return lhs_val == rhs_val

    # Comparison operators: ==, !=, >=, <=, >, <
    for op in ("!=", ">=", "<=", "==", ">", "<"):
        idx = condition.find(op)
        if idx != -1:
            key = condition[:idx].strip()
            rhs_raw = condition[idx + len(op):].strip()
            lhs_val = last_result.get(key)
            rhs_val = _parse_literal(rhs_raw)

            try:
                if op == "==":
                    return lhs_val == rhs_val
                elif op == "!=":
                    return lhs_val != rhs_val
                elif op == ">":
                    return lhs_val > rhs_val
                elif op == "<":
                    return lhs_val < rhs_val
                elif op == ">=":
                    return lhs_val >= rhs_val
                elif op == "<=":
                    return lhs_val <= rhs_val
            except (TypeError, ValueError):
                return False

    return False


# =========================================================================
# Template Interpolation for Batch Actions (§1.6)
# =========================================================================

def interpolate_result_templates(content: str, last_result: dict) -> str:
    """Replace JS-style ``${result.<key>}`` and ``${result.<key>.trim()}``
    placeholders in *content* with values from *last_result*.

    Unresolvable placeholders are left unchanged so nothing is silently
    swallowed.
    """
    def _replacer(m: re.Match) -> str:
        key = m.group(1)          # e.g. "stdout"
        do_trim = m.group(2) is not None  # .trim() present?
        val = last_result.get(key)
        if val is None:
            return m.group(0)     # leave placeholder as-is
        text = str(val)
        if do_trim:
            text = text.strip()
        return text

    # Match ${result.KEY} and ${result.KEY.trim()}
    return re.sub(
        r'\$\{result\.(\w+?)(?:\.(trim)\(\))?\}',
        _replacer,
        content,
    )


def resolve_placeholder_from_history(
    content: str, history: collections.abc.Iterable, last_result: dict | None = None,
) -> str:
    """Detect and resolve placeholder tokens in write_file/append_file content.

    Handles two families of LLM-generated placeholders:
      1. JS-style: ``${result.stdout}`` / ``${result.stdout.trim()}``
      2. Angular: ``<..._from_read_element>``, ``<..._from_run_command>``
         or generic ``<version_string>``, ``<output>``, etc.

    For (1) delegates to ``interpolate_result_templates`` using *last_result*.
    For (2) searches *history* backwards for the most recent action whose
    result contains a non-trivial text payload and substitutes it.

    Returns the (possibly unchanged) content string.
    """
    # -- Pass 1: JS-style ${result.*} ------------------------------------
    if last_result and "${" in content:
        content = interpolate_result_templates(content, last_result)

    # -- Pass 2: Angle-bracket placeholders ------------------------------
    angle_re = re.compile(r'<([a-z][a-z0-9_]*(?:_[a-z0-9_]+)*)>')
    matches = list(angle_re.finditer(content))
    if not matches:
        return content

    resolved: dict[str, str] = {}
    for m in matches:
        ph_name = m.group(1)
        if ph_name in resolved:
            continue

        _from_read = "read_element" in ph_name or "read" in ph_name
        _from_cmd = "run_command" in ph_name or "command" in ph_name or "stdout" in ph_name
        _generic = not _from_read and not _from_cmd

        for entry in reversed(list(history)):
            action = entry.get("action", "")
            obs = entry.get("observation")
            if not obs or not isinstance(obs, dict):
                continue

            val = (
                obs.get("value")
                or obs.get("content")
                or obs.get("stdout")
                or obs.get("text")
            )
            if val is None:
                continue

            val = str(val).strip()
            if not val:
                continue

            if _from_read and action == "read_element":
                resolved[ph_name] = val
                break
            elif _from_cmd and action == "run_command":
                resolved[ph_name] = val
                break
            elif _generic and action in ("read_element", "run_command", "read_file"):
                resolved[ph_name] = val
                break

    for ph_name, val in resolved.items():
        content = content.replace(f"<{ph_name}>", val)

    return content


# =========================================================================
# Batch Execution Engine (§1.5)
# =========================================================================

def execute_batch(
    orch: GenieOrchestrator, action_list: list[dict], iteration: int,
    llm_messages=None, llm_response=None,
    _seen_non_idempotent: "set[str] | None" = None,
) -> str:
    """Execute a list of actions sequentially, stopping on failure/checkpoint/terminal.

    Handles if/else conditional nodes inline.

    Returns one of:
        "done"        — terminal done action reached
        "abort"       — terminal abort action reached
        "checkpoint"  — checkpoint action reached; caller re-calls LLM
        "failure"     — an action failed mid-batch; caller re-calls LLM
        "halt"        — unrecoverable or resource error; _halt() already called
        "exhausted"   — all actions completed successfully; caller re-calls LLM
    """
    last_action_result: dict = {}  # result dict from last executed action
    # Track which non-idempotent actions have already been dispatched in this
    # batch.  A second occurrence of the same non-idempotent action is almost
    # always an LLM hallucination (retry loop or "simulate scenario" pattern).
    # Shared across recursive execute_batch calls (if/then branches) so that
    # nesting conditionals cannot bypass the guard.
    if _seen_non_idempotent is None:
        _seen_non_idempotent = set()

    idx = 0
    while idx < len(action_list):
        item = action_list[idx]
        idx += 1

        # -- Conditional node: {"if": ..., "then": [...], "else": [...]} --
        if "if" in item and "action" not in item:
            cond = item["if"]
            cond_true = evaluate_condition(cond, last_action_result)
            branch = (
                item.get("then", [])
                if cond_true
                else item.get("else", [])
            )
            # Fix P0-abort-guard: If the else branch contains an abort
            # but the most recent action actually succeeded, the condition
            # likely misfired (e.g. result=='ok' vs expected 'success').
            # Skip the abort to prevent false termination.
            if not cond_true and branch:
                _has_abort_in_else = any(
                    isinstance(b, dict) and b.get("action") == "abort"
                    for b in (branch if isinstance(branch, list) else [branch])
                )
                _last_succeeded = last_action_result.get("succeeded", False)
                if _has_abort_in_else and _last_succeeded:
                    print(
                        f"  [batch] Skipped false abort: condition '{cond}' "
                        f"evaluated False but prior action succeeded. "
                        f"Continuing batch.",
                        flush=True,
                    )
                    continue  # skip this conditional entirely
            if branch:
                sub_result = execute_batch(orch, branch, iteration, llm_messages=llm_messages, llm_response=llm_response, _seen_non_idempotent=_seen_non_idempotent)
                if sub_result != "exhausted":
                    return sub_result
            continue

        # -- Standard action node -----------------------------------------
        act_dict = item
        action_name = act_dict.get("action")

        # -- Duplicate non-idempotent action guard -------------------------
        # The same non-idempotent action appearing twice in a single batch is
        # almost always a hallucination (e.g. three create_repo calls, or a
        # retry loop baked into the batch).  Block and surface a clear error so
        # the LLM can recover cleanly instead of accumulating side-effects.
        if ACTION_IDEMPOTENT.get(action_name) is False:
            if action_name in _seen_non_idempotent:
                _dup_obs: dict = {
                    "result": "environmental_failure",
                    "observation": {
                        "action_blocked": (
                            f"'{action_name}' already executed once in this batch. "
                            f"Non-idempotent actions may only appear ONCE per batch — "
                            f"calling the same creation/deletion action multiple times "
                            f"creates duplicate objects. "
                            f"If the first call succeeded, end this batch with done. "
                            f"If the task is incomplete, use checkpoint then continue."
                        ),
                    },
                    "error": None,
                }
                orch._last_obs = _dup_obs
                orch._history.append({
                    "action":      action_name,
                    "args":        act_dict.get("args", {}),
                    "result":      "environmental_failure",
                    "observation": _dup_obs["observation"],
                })
                orch._fire_on_update(iteration, act_dict, orch._last_obs)
                return "failure"
            _seen_non_idempotent.add(action_name)

        # -- RC-D guard: block fetch_url (and other network actions) in
        #    debugging tasks — they are never relevant and just waste iterations.
        if (
            action_name in ("fetch_url", "open_browser", "open_pr")
            and getattr(orch, "_task_type", None) == "debugging"
        ):
            _offtask_obs = {
                "result": "environmental_failure",
                "observation": {
                    "action_blocked": (
                        f"{action_name} is not allowed during a debugging task. "
                        f"REQUIRED NEXT ACTIONS IN ONE BATCH: "
                        f"[write_file, run_command, done]. "
                        f"Fix the script locally — no network calls needed."
                    ),
                },
                "error": None,
            }
            orch._last_obs = _offtask_obs
            orch._history.append({
                "action": action_name,
                "args": act_dict.get("args", {}),
                "result": "environmental_failure",
                "observation": _offtask_obs["observation"],
            })
            orch._fire_on_update(iteration, act_dict, orch._last_obs)
            return "failure"

        # Validate
        try:
            orch._validate_act(act_dict)
        except SchemaValidationError as val_err:
            synth_dict = {
                "action": "schema_validation_error",
                "args": {"raw_response": json.dumps(act_dict)[:ARGS_TRUNCATION_CHARS]},
            }
            orch.observer.observe(
                synth_dict, result=None, error=val_err,
                attempt=1, t_start=time.time(),
                llm_messages=llm_messages, llm_response=llm_response,
            )
            orch._last_obs = orch.observer.last_entry
            if orch._last_obs:
                orch._history.append(orch._make_history_entry(synth_dict, orch._last_obs))
            orch._fire_on_update(iteration, synth_dict, orch._last_obs)
            return "failure"

        action_args = act_dict.get("args", {})

        # -- checkpoint: stop batch, observe, return to brain loop --------
        if action_name == "checkpoint":
            t_s = time.time()
            orch.observer.observe(
                act_dict, result={"status": "ok"}, error=None,
                attempt=1, t_start=t_s,
                llm_messages=llm_messages, llm_response=llm_response,
            )
            orch._last_obs = orch.observer.last_entry
            if orch._last_obs:
                orch._history.append(orch._make_history_entry(act_dict, orch._last_obs))

            # -- Post-success hint: if the last run_command succeeded and
            #    write_file was called before it, the task is almost certainly
            #    complete. Inject a directive hint so the next LLM call calls
            #    done immediately rather than wasting another round-trip.
            _cp_last_run: dict | None = None
            _cp_last_run_pos = -1
            _cp_last_write_pos = -1
            for _ci, _ce in enumerate(orch._history):
                if _ce.get("action") == "run_command":
                    _cp_last_run = _ce
                    _cp_last_run_pos = _ci
                if (
                    _ce.get("action") == "write_file"
                    and _ce.get("result") == "success"
                ):
                    _cp_last_write_pos = _ci
            _cp_run_succeeded = (
                _cp_last_run is not None
                and _cp_last_run.get("observation", {}).get("exit_code") == 0
            )
            _cp_wrote_before_run = (
                0 <= _cp_last_write_pos < _cp_last_run_pos
            )
            if (
                _cp_run_succeeded
                and _cp_wrote_before_run
                and getattr(orch, "_task_type", None) == "debugging"
            ):
                # Short-circuit: the fix is already verified. Instead of
                # returning "checkpoint" (which burns another LLM round-trip),
                # synthesise a done outcome right now.
                _cp_cmd = _cp_last_run.get("args", {}).get("cmd", "<script>")
                t_done = time.time()
                _done_synth = {
                    "action": "done",
                    "args": {
                        "summary": (
                            f"Fix verified: run_command('{_cp_cmd}') returned "
                            f"exit_code=0. checkpoint auto-converted to done."
                        ),
                        "message": "Task complete.",
                    },
                }
                orch.observer.observe(
                    _done_synth, result={"status": "ok"}, error=None,
                    attempt=1, t_start=t_done,
                    llm_messages=llm_messages, llm_response=llm_response,
                )
                orch._last_obs = orch.observer.last_entry
                orch._outcome = "done"
                orch._summary = _done_synth["args"]["summary"]
                orch._last_done_handoff = _done_synth["args"].get("handoff", "")  # P1-D
                orch._fire_on_update(
                    iteration, _done_synth, orch._last_obs,
                    outcome="done",
                    message=_done_synth["args"]["message"],
                )
                return "done"

            orch._fire_on_update(iteration, act_dict, orch._last_obs)
            return "checkpoint"

        # -- done: terminal -----------------------------------------------
        if action_name == "done":
            # -- Failed-verification guard (always active) -------------------
            # Case A: write_file succeeded but no run_command since
            # Case B: last run_command has non-zero exit_code
            _last_run_b: dict | None = None
            _last_run_b_pos = -1
            _last_write_b_pos = -1
            for _bi, _be in enumerate(orch._history):
                if _be.get("action") == "run_command":
                    _last_run_b = _be
                    _last_run_b_pos = _bi
                if (
                    _be.get("action") == "write_file"
                    and _be.get("result") == "success"
                ):
                    _last_write_b_pos = _bi
            _NON_EXEC_EXTS_B = (
                ".md", ".txt", ".rst", ".html", ".xml",
                ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
                ".csv", ".tsv", ".log",
            )
            _last_write_b_path = (
                orch._history[_last_write_b_pos].get("args", {}).get("path", "")
                if _last_write_b_pos >= 0 else ""
            )
            _last_write_b_is_doc = any(
                _last_write_b_path.endswith(ext) for ext in _NON_EXEC_EXTS_B
            )
            _wrote_without_rerun_b = (
                _last_write_b_pos > _last_run_b_pos and not _last_write_b_is_doc
            )
            _last_run_b_failed = (
                _last_run_b is not None
                and (
                    (_last_run_b.get("observation", {}).get("exit_code") or 0) != 0
                )
                and not (_last_write_b_is_doc and _last_write_b_pos > _last_run_b_pos)
            )
            if _wrote_without_rerun_b:
                _wb_path = (
                    orch._history[_last_write_b_pos].get("args", {})
                    .get("path", "<file>")
                )
                _block_obs = {
                    "result": "environmental_failure",
                    "observation": {
                        "done_blocked": (
                            f"done REJECTED: You wrote the fix to "
                            f"'{_wb_path}' but have NOT re-run the "
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
                orch._fire_on_update(iteration, act_dict, orch._last_obs)
                return "failure"
            elif _last_run_b_failed:
                _failed_ec = _last_run_b.get("observation", {}).get("exit_code")
                _failed_cmd = _last_run_b.get("args", {}).get("cmd", "<unknown>")
                _failed_stderr = str(
                    _last_run_b.get("observation", {}).get("stderr", "")
                )[:200]
                _block_obs = {
                    "result": "environmental_failure",
                    "observation": {
                        "done_blocked": (
                            f"done REJECTED: The script is still broken. "
                            f"run_command('{_failed_cmd}') returned "
                            f"exit_code={_failed_ec}. "
                            f"stderr: {_failed_stderr!r}. "
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
                orch._fire_on_update(iteration, act_dict, orch._last_obs)
                return "failure"
            # -- Case C: soft bounce for zero-write edit subtasks ----------
            _EDIT_VERB_RE_B = re.compile(
                r'\b(fix|modify|update|implement|add|create|write|'
                r'patch|change|edit|replace)\b', re.I,
            )
            _gt_b = getattr(orch, '_goaltracker', None)
            _cur_st_b = (
                _gt_b.subtasks[_gt_b.current_index]
                if _gt_b and _gt_b.current_index < len(_gt_b.subtasks)
                else None
            )
            _has_write_b = _last_write_b_pos >= 0
            if (
                _cur_st_b
                and _EDIT_VERB_RE_B.search(_cur_st_b.description)
                and not _has_write_b
                and getattr(orch, '_done_bounce_count', 0) == 0
            ):
                orch._done_bounce_count += 1
                _bounce_obs = {
                    "result": "environmental_failure",
                    "observation": {
                        "done_blocked": (
                            "done BOUNCED: You haven't written any "
                            "files in this subtask. If the required "
                            "changes are already present, re-verify "
                            "by reading the target file(s) and emit "
                            "done again. If not, make the edits "
                            f"first. Subtask: {_cur_st_b.description[:100]}"
                        ),
                    },
                    "error": None,
                }
                orch._last_obs = _bounce_obs
                orch._history.append({
                    "action": "done",
                    "args": action_args,
                    "result": _bounce_obs["result"],
                    "observation": _bounce_obs["observation"],
                })
                orch._fire_on_update(iteration, act_dict, orch._last_obs)
                return "failure"
            # ----------------------------------------------------------------

            # -- Stub/TODO content guard: detect placeholder content --------
            _STUB_RE_B = re.compile(
                r'\b(TODO|FIXME|HACK|XXX)\b'
                r'|#\s*placeholder'
                r'|raise\s+NotImplementedError',
                re.I,
            )
            _stub_files_b: list[str] = []
            for _he_b in orch._history:
                if _he_b.get("action") == "write_file" and _he_b.get("result") == "success":
                    _wf_content_b = (_he_b.get("args") or {}).get("content", "")
                    _wf_path_b = (_he_b.get("args") or {}).get("path", "")
                    if _wf_content_b and _STUB_RE_B.search(_wf_content_b):
                        _stub_files_b.append(_wf_path_b)
            if not _stub_files_b:
                _gt_stub = getattr(orch, '_goaltracker', None)
                _cur_st_stub_b = (
                    _gt_stub.subtasks[_gt_stub.current_index]
                    if _gt_stub and _gt_stub.current_index < len(_gt_stub.subtasks)
                    else None
                )
                if _cur_st_stub_b:
                    for _ws_p, _ws_c in getattr(orch, '_workspace_cache', {}).items():
                        _fn_b = os.path.basename(_ws_p)
                        if (
                            _fn_b in _cur_st_stub_b.description
                            and _ws_c
                            and _STUB_RE_B.search(_ws_c)
                        ):
                            _stub_files_b.append(_ws_p)
            if _stub_files_b and getattr(orch, '_stub_bounce_count', 0) < 2:
                orch._stub_bounce_count = getattr(orch, '_stub_bounce_count', 0) + 1
                _stub_list_b = ", ".join(os.path.basename(p) for p in _stub_files_b[:3])
                _stub_obs_b = {
                    "result": "environmental_failure",
                    "observation": {
                        "done_blocked": (
                            f"done REJECTED: Files still contain TODO/placeholder "
                            f"stubs: {_stub_list_b}. Replace ALL stub function "
                            f"bodies with complete working implementations. "
                            f"Do NOT call done until every TODO is replaced "
                            f"with real code."
                        ),
                    },
                    "error": None,
                }
                orch._last_obs = _stub_obs_b
                orch._history.append({
                    "action": "done",
                    "args": action_args,
                    "result": _stub_obs_b["result"],
                    "observation": _stub_obs_b["observation"],
                })
                orch._fire_on_update(iteration, act_dict, orch._last_obs)
                return "failure"
            # ---- end stub content guard ----

            t_s = time.time()
            orch.observer.observe(
                act_dict, result={"status": "ok"}, error=None,
                attempt=1, t_start=t_s,
                llm_messages=llm_messages, llm_response=llm_response,
            )
            orch._last_obs = orch.observer.last_entry
            orch._outcome = "done"
            orch._summary = action_args.get("summary", "")
            orch._last_done_handoff = action_args.get("handoff", "")  # P1-D
            orch._fire_on_update(
                iteration, act_dict, orch._last_obs,
                outcome="done",
                message=action_args.get("message", ""),
            )
            return "done"

        # -- abort: terminal ----------------------------------------------
        if action_name == "abort":
            # Guard: block abort when the most-recent run_command in history
            # succeeded (exit_code == 0).  Checking orch._history (not
            # last_action_result) makes this work even when abort is nested
            # inside a conditional branch — sub-execute_batch calls start
            # with last_action_result={}, so last_action_result.get("exit_code")
            # would be None and the guard would silently pass.
            _last_run_h: dict | None = next(
                (e for e in reversed(orch._history)
                 if e.get("action") == "run_command"),
                None,
            )
            _last_exit = (
                _last_run_h.get("observation", {}).get("exit_code")
                if _last_run_h is not None else None
            )
            if _last_exit is not None and _last_exit == 0:
                _block_obs = {
                    "result": "environmental_failure",
                    "observation": {
                        "abort_blocked": (
                            "abort REJECTED: The last run_command exited with "
                            "exit_code=0 — the script ran successfully. "
                            "You must call done (task complete) or use an "
                            "if/else conditional to branch on the actual "
                            "exit_code. Do NOT abort when the command succeeded."
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
                orch._fire_on_update(iteration, act_dict, orch._last_obs)
                return "failure"
            # ----------------------------------------------------------------
            t_s = time.time()
            orch.observer.observe(
                act_dict, result={"status": "ok"}, error=None,
                attempt=1, t_start=t_s,
                llm_messages=llm_messages, llm_response=llm_response,
            )
            orch._last_obs = orch.observer.last_entry
            orch._outcome = "abort"
            orch._summary = action_args.get("reason", "")
            orch._fire_on_update(
                iteration, act_dict, orch._last_obs,
                outcome="abort",
            )
            return "abort"

        # -- chat: non-terminal, execute and continue ---------------------
        if action_name == "chat":
            t_s = time.time()
            orch.observer.observe(
                act_dict, result={"status": "ok"}, error=None,
                attempt=1, t_start=t_s,
                llm_messages=llm_messages, llm_response=llm_response,
            )
            orch._last_obs = orch.observer.last_entry
            if orch._last_obs:
                orch._history.append(orch._make_history_entry(act_dict, orch._last_obs))
            orch._fire_on_update(
                iteration, act_dict, orch._last_obs,
                message=action_args.get("message"),
            )
            orch.record_act(act_dict)
            last_action_result = {"status": "ok"}
            continue

        # -- ask_user: block mid-task until user answers, then checkpoint --
        if action_name == "ask_user":
            question = action_args.get("question", "")
            options  = action_args.get("options", [])
            if not isinstance(options, list):
                options = []

            # Per-task budget check
            complexity = getattr(orch, "_ask_user_complexity", "medium")
            budget = ASK_USER_TIERS.get(complexity, ASK_USER_TIERS["medium"])
            if orch._ask_user_count >= budget:
                _limit_obs = {
                    "result": "environmental_failure",
                    "observation": {
                        "ask_user_blocked": (
                            f"ask_user limit reached ({budget} for \"{complexity}\" task). "
                            f"You have used all {budget} mid-task question(s) allowed. "
                            f"Make a reasonable decision and proceed without asking."
                        ),
                    },
                    "error": None,
                }
                orch._last_obs = _limit_obs
                orch._history.append(orch._make_history_entry(act_dict, _limit_obs))
                orch._fire_on_update(iteration, act_dict, _limit_obs)
                return "failure"

            # Block until answered, then resume
            answer = orch._block_for_ask_user(question, options)
            orch._ask_user_count += 1

            t_s = time.time()
            orch.observer.observe(
                act_dict,
                result={"status": "ok", "question": question, "answer": answer},
                error=None,
                attempt=1, t_start=t_s,
                llm_messages=llm_messages, llm_response=llm_response,
            )
            orch._last_obs = orch.observer.last_entry
            if orch._last_obs:
                orch._history.append(orch._make_history_entry(act_dict, orch._last_obs))
            orch._fire_on_update(iteration, act_dict, orch._last_obs)
            # Stop the batch here — LLM gets a fresh call with the answer in context
            return "checkpoint"

        # -- Regular action: dispatch with retry --------------------------

        # -- Template interpolation (§1.6) --------------------------------
        if action_name in ("write_file", "append_file"):
            content = action_args.get("content", "")
            if "${" in content or "<" in content:
                content = resolve_placeholder_from_history(
                    content, orch._history, last_action_result,
                )
                act_dict = dict(act_dict)
                act_dict["args"] = dict(action_args, content=content)
                action_args = act_dict["args"]
            # Guard: reject empty writes so the LLM retries
            final_content = action_args.get("content", "")
            if not final_content or not final_content.strip():
                error = EnvironmentalError(
                    "write_file called with empty content. "
                    "Re-read the data you need (read_element / run_command) "
                    "and provide the actual text in the content field."
                )
                orch.observer.observe(
                    act_dict, result=None, error=error,
                    attempt=1, t_start=time.time(),
                    llm_messages=llm_messages, llm_response=llm_response,
                )
                orch._last_obs = orch.observer.last_entry
                if orch._last_obs:
                    orch._history.append(orch._make_history_entry(act_dict, orch._last_obs))
                return "failure"

        orch.record_act(act_dict)
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
            except Exception as e:
                error = e
            finally:
                orch.observer.observe(
                    act_dict, result=result, error=error,
                    attempt=attempt, t_start=t_s,
                    tier=tier, wid=wid,
                    llm_messages=llm_messages, llm_response=llm_response,
                )

            orch._last_obs = orch.observer.last_entry
            if orch._last_obs:
                orch._history.append(orch._make_history_entry(act_dict, orch._last_obs))

            # -- P1-A: intra-subtask writes removed (batch path) --------------
            # Keep file tracking for scratchpad files section
            if action_name == "read_file" and action_args.get("path"):
                orch._scratchpad.update_files({action_args["path"]: "read"})
            elif action_name == "write_file" and action_args.get("path"):
                orch._scratchpad.update_files({action_args["path"]: "written"})

            # -- Track run_command failures for handoff ----------------------
            if action_name == "run_command":
                _obs_entry = orch.observer.last_entry or {}
                _obs_res = _obs_entry.get("result", "")
                if _obs_res in ("command_failed", "environmental_failure"):
                    _fc = (action_args or {}).get("cmd", "")
                    if _fc and _fc not in orch._subtask_failed_commands:
                        orch._subtask_failed_commands.append(_fc)

            if error is not None:
                error_class = orch.classify_error(error, action_name)

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

                if error_class == ERROR_CLASS_UNRECOVERABLE:
                    orch._halt("unrecoverable")
                    orch._fire_on_update(iteration, act_dict, orch._last_obs)
                    return "halt"

                if error_class == ERROR_CLASS_RESOURCE:
                    if orch._mode == "interactive":
                        print(
                            f"\n[GENIE] RESOURCE error: {error}\n"
                            f"Resolve the issue and restart the task.\n",
                            file=sys.stderr,
                        )
                        orch._halt("resource_halt")
                        return "halt"

                # TRANSIENT exhausted → ENVIRONMENTAL, or genuine ENVIRONMENTAL:
                # stop batch, let brain loop re-call LLM with failure context
                orch._fire_on_update(iteration, act_dict, orch._last_obs)
                return "failure"

            break  # success

        # -- RC-A/RC-B hint: when run_command exits non-zero, inject a
        #    directive so the next LLM call builds the correct
        #    [write_file, run_command, done] batch and does NOT call
        #    done/checkpoint prematurely or re-run without fixing.
        if action_name == "run_command":
            _hint_exit = (result or {}).get("exit_code") if isinstance(result, dict) else None
            if _hint_exit is not None and _hint_exit != 0:
                # RC-B check: was the previous history entry also run_command
                # with no write_file in between? (re-ran without fixing)
                _h_list = list(orch._history)
                _hlen = len(_h_list)
                _recent_write = any(
                    e.get("action") == "write_file" and e.get("result") == "success"
                    for e in _h_list[-6:]
                )
                _prev_was_run = (
                    _hlen >= 2
                    and _h_list[-2].get("action") == "run_command"
                    and not any(
                        e.get("action") == "write_file"
                        for e in _h_list[max(0, _hlen - 3):_hlen - 1]
                    )
                )
                if isinstance(orch._last_obs, dict):
                    orch._last_obs = dict(orch._last_obs)
                    _hint_inner = orch._last_obs.get("observation")
                    if isinstance(_hint_inner, dict):
                        _hint_inner = dict(_hint_inner)
                    elif _hint_inner is None:
                        _hint_inner = {}
                    orch._last_obs["observation"] = _hint_inner
                    if _prev_was_run and not _recent_write:
                        _hint_inner["action_hint"] = (
                            "You just re-ran the script without changing it — "
                            "it will still fail. "
                            "REQUIRED NEXT ACTIONS IN ONE BATCH: "
                            "[write_file, run_command, done]. "
                            "Write the fix first, then run, then done."
                        )
                    else:
                        _hint_inner["action_hint"] = (
                            f"Script failed (exit_code={_hint_exit}). "
                            "REQUIRED NEXT ACTIONS IN ONE BATCH: "
                            "[write_file, run_command, done] — all three together. "
                            "Do NOT use checkpoint. Do NOT call done yet. "
                            "Do NOT re-run without writing the fix first."
                        )

        orch._fire_on_update(iteration, act_dict, orch._last_obs)

        # -- RC-C guard: if run_command just executed and subsequent actions
        #    in this batch include write_file/append_file whose content does
        #    NOT use _from_run_command placeholder tokens, the model pre-wrote
        #    that content before seeing stdout (hallucination risk).  Break the
        #    batch here and return "checkpoint" so the brain loop re-queries
        #    the LLM with the actual command output in context.
        if action_name == "run_command" and idx < len(action_list):
            _rc_remaining = action_list[idx:]
            _has_blind_write = any(
                r.get("action") in ("write_file", "append_file")
                and "_from_run_command" not in str(r.get("args", {}).get("content", ""))
                for r in _rc_remaining
                if "action" in r  # skip conditional nodes
            )
            if _has_blind_write:
                return "checkpoint"

        # Store result for conditional evaluation
        last_action_result = result if isinstance(result, dict) else {"status": "ok"}
        # Fix P0-abort-guard: inject 'succeeded' boolean for reliable
        # condition evaluation — avoids 'ok' vs 'success' mismatches
        _ERROR_RESULTS = {"command_failed", "environmental_failure", "error", "failed"}
        _act_result_str = last_action_result.get("result", "ok")
        last_action_result["succeeded"] = (
            _act_result_str not in _ERROR_RESULTS
            and not isinstance(result, Exception)
        )

    return "exhausted"
