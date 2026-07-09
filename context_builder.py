"""
Genie — context_builder.py
LLM context assembly: user-turn construction, history truncation, and
run-length-encoded compression.

Extracted from orchestrator.py.  All functions are stateless — they
accept data as arguments and return strings / dicts.
"""
from __future__ import annotations

import json
import os
from collections import deque

from config import (
    APP_PROFILES,
    CONTEXT_RECENT_WINDOW,
    CONTEXT_SUMMARY_MAX_CHARS,
    HISTORY_OBSERVATION_TRUNCATION_CHARS,
    MAX_ITERATIONS_PER_TASK,
    WORKSPACE_DIR,
)


# =========================================================================
# User-turn assembly
# =========================================================================

def build_user_turn(
    goal: str,
    iteration: int,
    task_cost: float,
    monthly_cost: float,
    last_obs: dict | None,
    history_deque: deque,
    registry_snapshot: dict[str, dict],
    cdp_url_reader=None,
    consecutive_env_failures: int = 0,
    prefetched_content: str = "",
    workspace_cache: dict[str, str] | None = None,
    scratchpad_text: str = "",
) -> str:
    """Build the user-turn message for the LLM.

    Args:
        registry_snapshot: dict of {label: entry_dict} from the window registry.
        cdp_url_reader:    callable(cdp_port, timeout) -> str|None.
                           If provided, appends the active tab URL for CDP windows.
        consecutive_env_failures: current escalation-ladder counter.
        prefetched_content: pre-fetched URL content block to inject after the goal.
    """
    parts = []

    # Header block
    _home = os.path.expanduser("~")
    parts.append(f"GOAL: {goal}")
    parts.append(f"HOME: {_home}")
    parts.append(f"WORKSPACE: {WORKSPACE_DIR}  (default dir for file operations)")
    parts.append(f"  Use {_home}/Downloads, {_home}/Desktop etc. for user folders.")
    parts.append(f"  NEVER use /home/user/ — the actual home is {_home}.")
    if prefetched_content:
        parts.append("")
        parts.append(prefetched_content)
    parts.append(f"ITERATION: {iteration} of {MAX_ITERATIONS_PER_TASK}")
    parts.append(
        f"BUDGET USED: ${task_cost:.4f} (task) | ${monthly_cost:.4f} (monthly)"
    )

    # Registry snapshot
    parts.append("")
    parts.append("REGISTRY:")
    if not registry_snapshot:
        parts.append("(empty)")
    else:
        for label, entry in sorted(registry_snapshot.items()):
            status = "open" if entry.get("wid") else "closed"
            title = entry.get("wm_name", "—") or "—"
            cdp_suffix = ""
            cdp_port = entry.get("cdp_port")
            if cdp_port and status == "open" and cdp_url_reader is not None:
                try:
                    _url = cdp_url_reader(cdp_port, timeout=1.5)
                    if _url:
                        cdp_suffix = f"  | url={_url}"
                except Exception:
                    pass
            parts.append(f"{label}  | {status}  | {title}{cdp_suffix}")

    parts.append("")
    parts.append("AVAILABLE APPS: " + ", ".join(sorted(APP_PROFILES.keys())))

    # Failure-count warning
    if history_deque and last_obs and last_obs.get("result") == "environmental_failure":
        _fail_action = last_obs.get("action", "")
        if _fail_action:
            _recent_fails = sum(
                1 for e in history_deque
                if e.get("action") == _fail_action
                and e.get("result") == "environmental_failure"
            )
            if _recent_fails >= 3:
                parts.append("")
                parts.append(
                    f"⚠ WARNING: {_fail_action} has returned environmental_failure "
                    f"{_recent_fails} times in history. "
                    "Repeating the same call is unlikely to succeed. "
                    "Switch to a different strategy or skip this step."
                )

    # Workspace state — injected from persistent cache, never compressed.
    if workspace_cache:
        parts.append("")
        parts.append("WORKSPACE STATE:")
        for _ws_path, _ws_content in sorted(workspace_cache.items()):
            parts.append(f"--- {_ws_path} ---")
            parts.append(_ws_content)
            parts.append(f"--- end {_ws_path} ---")

    # Task memory (scratchpad) — best-effort, may be incomplete.
    # Brain model treats these as strong hints, not ground truth.
    # If the brain's own observation contradicts a scratchpad entry,
    # the observation wins.
    if scratchpad_text:
        parts.append("")
        parts.append("TASK MEMORY (best-effort — may be incomplete):")
        parts.append(scratchpad_text)

    # Last observation
    parts.append("")
    parts.append("LAST OBSERVATION:")
    if last_obs is not None:
        obs_view = {
            "result": last_obs.get("result"),
            "observation": last_obs.get("observation", {}),
            "error": last_obs.get("error"),
        }
        parts.append(json.dumps(obs_view, indent=2, default=str))

        # Recovery hints — escalation ladder
        if last_obs.get("result") == "environmental_failure":
            _last_action = last_obs.get("action", "")
            if _last_action in ("read_element", "click_element", "type_element", "focus_window"):
                _n = consecutive_env_failures
                if _n <= 1:
                    parts.append(
                        "RECOVERY HINT: environmental_failure usually means the page "
                        "is still loading or the element is not yet visible. "
                        "Use wait {seconds: 3} then retry the SAME action. "
                        "Do NOT re-open the browser or re-navigate."
                    )
                elif _n <= 3:
                    parts.append(
                        f"RECOVERY HINT (attempt {_n}): Retrying the same action is "
                        "not working. Try a DIFFERENT approach: "
                        "use a different role or name, try read_element with "
                        'role="frame" name="" to get page content, or navigate '
                        "to the target URL directly with press_key ctrl:l. "
                        "Do NOT keep retrying the exact same action."
                    )
                else:
                    parts.append(
                        f"⚠ RECOVERY HINT (attempt {_n}): This action has failed "
                        f"{_n} times. STOP retrying it. Switch strategy entirely: "
                        "use run_command with curl to fetch the page content, "
                        "or try a completely different source URL. "
                        "If the task output is already written to disk, call done."
                    )

        # Schema validation error hint
        if last_obs.get("action") == "schema_validation_error":
            _err = last_obs.get("error", {})
            _err_msg = _err.get("message", "unknown") if isinstance(_err, dict) else str(_err)
            parts.append(
                f"FORMAT ERROR: Your last response failed schema validation ({_err_msg}). "
                "You MUST structure your response EXACTLY as:\n"
                "  <think>\n  your reasoning\n  </think>\n"
                "  <act>\n  [{\"action\": \"...\", \"args\": {...}}]\n  </act>\n"
                "Rules: (1) </think> MUST close the think block. "
                "(2) <act> block MUST contain ONLY a valid JSON array of action objects. "
                "(3) NEVER output <tool_call> tags — they are forbidden. "
                "(4) NEVER put plain text or reasoning inside <act>."
            )

        # observation_partial warning
        obs_inner = last_obs.get("observation", {})
        if isinstance(obs_inner, dict) and obs_inner.get("observation_partial") is True:
            parts.append(
                "NOTE: observation incomplete — one or more passive "
                "sub-calls failed. Missing fields are unknown, not absent. "
                "Verify state before proceeding if action outcome is uncertain."
            )
    else:
        parts.append("(none)")

    # History — context batching: compressed summary + recent detail
    parts.append("")
    if history_deque:
        history_list = list(history_deque)
        total = len(history_list)

        if total > CONTEXT_RECENT_WINDOW:
            old_entries = history_list[:-CONTEXT_RECENT_WINDOW]
            recent_entries = history_list[-CONTEXT_RECENT_WINDOW:]
            # Approximate iteration index of oldest entry
            old_start_idx = max(1, iteration - total + 1)

            summary_block = compress_old_history(old_entries, old_start_idx)
            parts.append(f"HISTORY SUMMARY ({len(old_entries)} earlier actions):")
            parts.append(summary_block)

            parts.append("")
            parts.append(f"RECENT ACTIONS (last {len(recent_entries)}):")
            recent_detailed = [truncate_history_entry(e) for e in recent_entries]
            parts.append(json.dumps(recent_detailed, indent=2, default=str))
        else:
            parts.append("HISTORY:")
            history_detailed = [truncate_history_entry(e) for e in history_list]
            parts.append(json.dumps(history_detailed, indent=2, default=str))
    else:
        parts.append("HISTORY:")
        parts.append("[]")

    # Completion hint
    if history_deque:
        last = history_deque[-1]
        last_action = last.get("action", "")
        last_key = (last.get("args") or {}).get("key", "").lower()
        last_result = last.get("result", "")

        _goal_lower = goal.lower()
        _goal_needs_click = any(
            phrase in _goal_lower
            for phrase in (
                "click the first", "click on the first", "click first",
                "first link", "first result", "first search result",
                "open the first", "click the link", "click a link",
                "click the result", "click on the result",
                "click the second", "click on the second", "click second",
                "second link", "second result", "second search result",
                "click the third", "click on the third", "click third",
                "third link", "third result", "third search result",
                "show me that page", "show me the page",
            )
        )
        _click_already_done = any(
            entry.get("action") == "click_element"
            for entry in history_deque
        )
        _suppress_hint = _goal_needs_click and not _click_already_done

        if (
            last_action == "press_key"
            and last_result == "success"
            and ("enter" in last_key or "return" in last_key)
            and not _suppress_hint
        ):
            parts.append("")
            parts.append(
                "ACTION REQUIRED: The last action was press_key enter — "
                "a form/search/navigation was just submitted. "
                "CALL done NOW unless the goal explicitly requires reading "
                "or verifying page content. Do not navigate, click, or "
                "inspect further. A single well-formed <act> block with "
                'action=\'done\' is the only correct response.'
            )

    return "\n".join(parts)


# =========================================================================
# History truncation
# =========================================================================

def truncate_history_entry(entry: dict) -> dict:
    """Return a history entry with observation fields capped."""
    return {
        "action": entry.get("action"),
        "args": truncate_args_fields(entry.get("args", {})),
        "result": entry.get("result"),
        "observation": truncate_obs_fields(entry.get("observation", {})),
    }


def truncate_obs_fields(obs: dict) -> dict:
    """Cap each field in observation dict at HISTORY_OBSERVATION_TRUNCATION_CHARS."""
    if not isinstance(obs, dict):
        return obs
    truncated = {}
    for k, v in obs.items():
        if isinstance(v, str) and len(v) > HISTORY_OBSERVATION_TRUNCATION_CHARS:
            truncated[k] = v[:HISTORY_OBSERVATION_TRUNCATION_CHARS] + "…"
        else:
            truncated[k] = v
    return truncated


def truncate_args_fields(args: dict) -> dict:
    """Cap large string args (e.g. write_file content) in history entries."""
    if not isinstance(args, dict):
        return args
    truncated = {}
    for k, v in args.items():
        if isinstance(v, str) and len(v) > HISTORY_OBSERVATION_TRUNCATION_CHARS:
            truncated[k] = v[:HISTORY_OBSERVATION_TRUNCATION_CHARS] + "…"
        else:
            truncated[k] = v
    return truncated


# =========================================================================
# History entry construction
# =========================================================================

def make_history_entry(act_dict: dict, obs_entry: dict) -> dict:
    """Build a history deque entry from act_dict and observer.last_entry."""
    return {
        "action": act_dict.get("action"),
        "args": act_dict.get("args"),
        "result": obs_entry.get("result") if obs_entry else None,
        "observation": obs_entry.get("observation", {}) if obs_entry else {},
    }


# =========================================================================
# Context Batching — Summary Compression (RLE)
# =========================================================================

def compress_history_entry(entry: dict, index: int) -> str:
    """Compress a single history entry to a one-line summary.

    Example output: '[3] open_app chrome_1 → success'
    """
    action = entry.get("action", "?")
    args = entry.get("args", {})
    result = entry.get("result", "?")

    if action in ("open_app", "focus_window"):
        args_str = args.get("app", "")
    elif action == "press_key":
        args_str = args.get("key", "")
    elif action == "type_text":
        text = args.get("text", "")
        args_str = f'"{text[:40]}{"..." if len(text) > 40 else ""}"'
    elif action in ("click_element", "read_element"):
        args_str = f'{args.get("app", "")} {args.get("role", "")}:{args.get("name", "")[:30]}'
        if action == "read_element" and result == "success":
            obs = entry.get("observation", {})
            content = obs.get("element_content", obs.get("text", ""))
            if content:
                args_str += f' => "{str(content)[:50]}{"..." if len(str(content)) > 50 else ""}"'
    elif action == "type_element":
        text = args.get("text", "")
        args_str = (
            f'{args.get("app", "")} {args.get("role", "")}:'
            f'{args.get("name", "")[:20]} '
            f'"{text[:30]}{"..." if len(text) > 30 else ""}"'
        )
    elif action == "run_command":
        cmd = args.get("cmd", "")
        args_str = f'"{cmd[:60]}{"..." if len(cmd) > 60 else ""}"'
        obs = entry.get("observation", {})
        exit_code = obs.get("exit_code")
        if exit_code is not None:
            args_str += f" [exit={exit_code}]"
    elif action in ("write_file", "read_file", "append_file", "delete_file", "list_dir"):
        args_str = args.get("path", "")
    elif action == "wait":
        args_str = f'{args.get("seconds", "?")}s'
    elif action == "done":
        args_str = args.get("summary", "")[:40]
    elif action == "abort":
        args_str = args.get("reason", "")[:40]
    elif action == "chat":
        msg = args.get("message", "")
        args_str = f'"{msg[:40]}{"..." if len(msg) > 40 else ""}"'
    else:
        first_val = next(iter(args.values()), "") if args else ""
        args_str = str(first_val)[:40]

    obs = entry.get("observation", {})
    error_hint = ""
    if result and "failure" in str(result):
        err_msg = obs.get("error_message", obs.get("error", ""))
        if err_msg:
            error_hint = f' ({str(err_msg)[:60]})'

    return f"[{index}] {action} {args_str} -> {result}{error_hint}"


def compress_old_history(entries: list[dict], start_index: int) -> str:
    """Compress old history entries with run-length encoding.

    Consecutive entries with the same (action, result) pair are collapsed:
      [3-7] read_element chrome_1 paragraph: → environmental_failure ×5
    """
    if not entries:
        return ""

    groups: list[tuple[int, int, int, dict, tuple]] = []
    for i, entry in enumerate(entries):
        action = entry.get("action", "?")
        result = entry.get("result", "?")
        if action == "wait":
            if groups:
                groups[-1] = (
                    groups[-1][0], start_index + i,
                    groups[-1][2], groups[-1][3], groups[-1][4],
                )
            continue
        key = (action, result)
        if groups and groups[-1][4] == key:
            groups[-1] = (
                groups[-1][0], start_index + i,
                groups[-1][2] + 1, entry, key,
            )
        else:
            groups.append((start_index + i, start_index + i, 1, entry, key))

    lines = []
    for group_start, group_end, count, entry, _key in groups:
        if count == 1 and group_start == group_end:
            lines.append(compress_history_entry(entry, group_start))
        else:
            line = compress_history_entry(entry, group_start)
            bracket_end = line.index("]") if "]" in line else 0
            line = f"[{group_start}-{group_end}]{line[bracket_end + 1:]} (×{count})"
            lines.append(line)

    full_summary = "\n".join(lines)
    if len(full_summary) <= CONTEXT_SUMMARY_MAX_CHARS:
        return full_summary

    trimmed = []
    total_len = 0
    for line in reversed(lines):
        if total_len + len(line) + 1 > CONTEXT_SUMMARY_MAX_CHARS - 50:
            break
        trimmed.append(line)
        total_len += len(line) + 1

    omitted = len(lines) - len(trimmed)
    trimmed.reverse()
    if omitted > 0:
        trimmed.insert(0, f"... ({omitted} earlier actions omitted)")
    return "\n".join(trimmed)
