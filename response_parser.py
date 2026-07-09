"""
Genie — response_parser.py
LLM response parsing and schema validation.

Extracted from orchestrator.py.  All functions are stateless — they
accept data as arguments and return results or raise SchemaValidationError.
"""
from __future__ import annotations

import json
import re

from config import (
    ACTION_SCHEMA,
    BATCH_ENABLED,
    BATCH_MAX_ACTIONS,
)
from exceptions import SchemaValidationError
try:
    from json_repair import repair_json as _json_repair
except ImportError:  # optional dependency — degrades gracefully
    _json_repair = None  # type: ignore[assignment]

# =========================================================================
# Response Parsing
# =========================================================================

def _try_repair_truncated_json(candidate: str) -> str | None:
    """Attempt to close truncated JSON from a max_tokens-cut response.

    The model emitted ``<act>`` followed by a JSON object or array that was
    cut off mid-way.  We try to:
      1.  Trim any trailing partial string value (unmatched ``"``).
      2.  Close open braces / brackets in reverse order.
      3.  Validate the result with ``json.loads()``.

    Returns the repaired JSON string on success, ``None`` on failure.
    """
    if not candidate:
        return None
    # Must start with { or [ to look like JSON
    if candidate[0] not in ('{', '['):
        return None

    # Strip any trailing partial string literal (odd quote count means
    # a string was cut mid-way — truncate from the last unmatched quote).
    in_string = False
    escape = False
    last_good = 0  # index of last structurally complete position
    stack: list[str] = []
    i = 0
    while i < len(candidate):
        ch = candidate[i]
        if in_string:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_string = False
                last_good = i
        else:
            if ch == '"':
                in_string = True
            elif ch in ('{', '['):
                stack.append('}' if ch == '{' else ']')
                last_good = i
            elif ch in ('}', ']'):
                if stack and stack[-1] == ch:
                    stack.pop()
                last_good = i
            elif ch in (',', ':'):
                last_good = i
        i += 1

    # If we ended inside a string, back up to just before the opening quote
    if in_string:
        # Find the opening quote of the truncated string
        q = candidate.rfind('"', 0, i)
        if q > 0:
            # Trim back to before this token.  Walk backwards past any
            # preceding colon, comma, or whitespace.
            trim_to = q
            j = q - 1
            while j >= 0 and candidate[j] in (' ', '\t', '\n', '\r', ':', ','):
                trim_to = j
                j -= 1
            # If j landed on a closing quote AND the separator we just
            # skipped included a ':', the truncated string was a *value*
            # and the token at j is its *key*.  Remove the key too.
            skipped = candidate[trim_to:q]
            if j >= 0 and candidate[j] == '"' and ':' in skipped:
                # Walk back past the key string
                k = j - 1
                while k >= 0 and candidate[k] != '"':
                    k -= 1
                if k >= 0:
                    # Also skip comma/whitespace before the key
                    k -= 1
                    while k >= 0 and candidate[k] in (' ', '\t', '\n', '\r', ','):
                        k -= 1
                    trim_to = k + 1
            candidate = candidate[:trim_to]
            # Recompute stack after trimming
            stack = []
            for ch in candidate:
                if ch in ('{', '['):
                    stack.append('}' if ch == '{' else ']')
                elif ch in ('}', ']'):
                    if stack and stack[-1] == ch:
                        stack.pop()
    else:
        # Trim any trailing comma or colon (incomplete next field)
        candidate = candidate.rstrip()
        if candidate and candidate[-1] in (',', ':'):
            candidate = candidate[:-1]

    # Close remaining open brackets/braces
    closers = ''.join(reversed(stack))
    repaired = candidate + closers

    # Validate
    try:
        parsed = json.loads(repaired)
    except (json.JSONDecodeError, ValueError):
        return None

    # Must contain an "action" key at top level or in first array element
    if isinstance(parsed, dict) and "action" in parsed:
        return repaired
    if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict) and "action" in parsed[0]:
        return repaired
    return None


def _try_parse_xml_action_call(text: str) -> str | None:
    """Fallback 3d — XML tag-per-action format (no <function=...> wrapper).

    Handles patterns like::

        <tool_call>
        <run_command>
        <cmd>python3 -m pytest ...</cmd>
        </run_command>
        </tool_call>

    The outer tag name is the action (must be a known ACTION_SCHEMA key).
    Inner tags are arg key/value pairs; values are decoded from JSON if
    possible, otherwise kept as plain strings.  Multiple blocks are
    collected into a batch list.
    Returns a JSON string on success, ``None`` if no known action tag found.
    """
    known = set(ACTION_SCHEMA.keys())
    pat = re.compile(
        r"<(" + "|".join(re.escape(a) for a in known) + r")>\s*(.*?)\s*</\1>",
        re.DOTALL,
    )
    matches = pat.findall(text)
    if not matches:
        return None

    actions = []
    arg_pat = re.compile(r"<([^/>\s][^>\s]*)>(.*?)</\1>", re.DOTALL)
    for action_name, inner in matches:
        args: dict = {}
        for arg_name, arg_value in arg_pat.findall(inner):
            val = arg_value.strip()
            try:
                args[arg_name] = json.loads(val)
            except (json.JSONDecodeError, ValueError):
                args[arg_name] = val
        actions.append({"action": action_name, "args": args})

    if not actions:
        return None
    return json.dumps(actions[0]) if len(actions) == 1 else json.dumps(actions)


def _try_parse_hermes_function_call(text: str) -> str | None:
    """Fallback 3c — Hermes/OpenAI native function-call format.

    Handles one or more blocks like::

        <tool_call>
        <function=ACTION_NAME>
        <args>
        {"key": "value"}
        </args>
        </function>
        </tool_call>

    Also handles ``<function=NAME>{...}</function>`` without ``<args>`` tags.
    Multiple blocks are collected into a batch list.
    Converts to Genie's canonical ``{"action": ..., "args": {...}}`` format.
    Returns a JSON string on success, ``None`` if pattern not found.
    """
    pattern = re.compile(
        r"<function=([^>]+)>\s*(?:<args>\s*)?(.*?)(?:\s*</args>)?\s*</function>",
        re.DOTALL,
    )
    matches = pattern.findall(text)
    if not matches:
        return None

    actions = []
    for action_name, args_raw in matches:
        action_name = action_name.strip()
        args_raw = args_raw.strip()
        try:
            args_dict = json.loads(args_raw) if args_raw else {}
        except json.JSONDecodeError:
            # Strip any residual XML-ish noise and retry
            clean = re.sub(r"<[^>]+>", "", args_raw).strip()
            try:
                args_dict = json.loads(clean) if clean else {}
            except json.JSONDecodeError:
                return None
        if not isinstance(args_dict, dict):
            return None
        actions.append({"action": action_name, "args": args_dict})

    if not actions:
        return None
    return json.dumps(actions[0]) if len(actions) == 1 else json.dumps(actions)


def parse_response(full_response: str) -> tuple[str | None, dict | list[dict]]:
    """Parse <think>/<act> from full LLM response.

    Returns (think_content, act_payload) where act_payload is:
      - A single dict  (legacy single-action format)
      - A list of dicts (batch format, 1.5+)
    Raises SchemaValidationError on parse failure.
    """
    think_content = None
    # Strip Llama chat template tokens that leak into assistant turn completions
    full_response = full_response.replace("<|start_header_id|>", "").replace("<|end_header_id|>", "").replace("<|end_header|>", "")
    remaining = full_response

    # Find </think> delimiter
    think_end = full_response.find("</think>")
    if think_end != -1:
        # Everything between <think>\n and </think>
        think_start = 0
        if full_response.startswith("<think>\n"):
            think_start = len("<think>\n")
        elif full_response.startswith("<think>"):
            think_start = len("<think>")
        think_content = full_response[think_start:think_end].strip() or None
        remaining = full_response[think_end + len("</think>"):]
    else:
        # No </think> — valid model output (model emitted <act> only).
        think_content = None

    # Extract <act>...</act> — try strict match first, then several
    # lenient fallbacks for known LLM quirks.
    def _try_extract_act(text: str) -> str | None:
        # 1. Canonical form
        m = re.search(r"<act>\s*(.*?)\s*</act>", text, re.DOTALL)
        if m:
            return m.group(1).strip()
        # 2. Any closing </act> paired with a garbled opening
        m = re.search(r"<[^>]{0,10}act[^>]{0,5}>\s*(.*?)\s*</act>",
                      text, re.DOTALL)
        if m:
            return m.group(1).strip()
        # 2b. <act> present but no </act> — response likely truncated by
        #     max_tokens.  Extract what follows <act> and try to close the
        #     incomplete JSON so downstream json.loads() can succeed.
        m = re.search(r"<act>\s*(.*)", text, re.DOTALL)
        if m:
            candidate = m.group(1).strip()
            repaired = _try_repair_truncated_json(candidate)
            if repaired is not None:
                return repaired
            # Also try broader json_repair for unclosed strings / other truncations
            if _json_repair is not None:
                candidate2 = _json_repair(candidate)
                if candidate2 and candidate2 not in ('null', '""', '[]', '{}'):
                    return candidate2
        # 3. <tool_call> family — check most-specific patterns first.
        # 3c. Hermes/OpenAI native function-call format (most specific):
        #     <tool_call><function=ACTION_NAME><args>{...}</args></function></tool_call>
        #     Emitted by qwen3-coder-next and other coder-tuned models.
        #     Must run before step 3a because step 3a's greedy <tool_call> match
        #     would capture the inner <function=...> text as raw JSON and fail.
        result = _try_parse_hermes_function_call(text)
        if result is not None:
            return result
        # 3d. XML tag-per-action format (no <function=...> wrapper):
        #     <tool_call><run_command><cmd>...</cmd></run_command></tool_call>
        #     Also emitted by coder-tuned models as an alternative to Hermes format.
        result = _try_parse_xml_action_call(text)
        if result is not None:
            return result
        # 3b. Bare terminal action name inside <tool_call> (e.g. <tool_call>done</act>)
        if re.search(r"<tool_call>\s*done\s*(?:</tool_call>|</act>)", text, re.IGNORECASE | re.DOTALL):
            return '{"action": "done", "args": {"summary": "task completed", "message": "task completed"}}'
        if re.search(r"<tool_call>\s*abort\s*(?:</tool_call>|</act>)", text, re.IGNORECASE | re.DOTALL):
            return '{"action": "abort", "args": {"reason": "model emitted bare abort"}}'
        # 3a. <tool_call>JSON...</tool_call> or <tool_call>JSON...</act> (legacy)
        m = re.search(
            r"<tool_call>\s*(.*?)\s*(?:</tool_call>|</act>)",
            text, re.DOTALL)
        if m:
            return m.group(1).strip()
        # 4. Bare JSON object or array with an "action" key (no XML wrapper)
        m = re.search(r'(\[\s*\{.*\}\s*\])', text, re.DOTALL)
        if m:
            return m.group(1).strip()
        m = re.search(r'(\{\s*"action"\s*:.*\})', text, re.DOTALL)
        if m:
            return m.group(1).strip()
        return None

    act_json_str = _try_extract_act(remaining)
    if act_json_str is None and think_end != -1:
        act_json_str = _try_extract_act(full_response)
    # Fix 2: JSON-in-think fallback — if the model dumped a JSON action object
    # inside the <think> block (no </think> or <act> wrapper), fish it out.
    if act_json_str is None and think_content:
        m = re.search(
            r'(\[\s*\{.*"action".*\}\s*\]|\{\s*"action"\s*:.*\})',
            think_content, re.DOTALL,
        )
        if m:
            act_json_str = m.group(1).strip()
    if act_json_str is None:
        raise SchemaValidationError("no act block in response")
    try:
        act_payload = json.loads(act_json_str)
    except json.JSONDecodeError:
        # Step 1: existing structural repair (handles missing ] / truncated arrays)
        repaired = _try_repair_truncated_json(act_json_str)
        # Step 2: broader LLM-output repair (missing }, unclosed strings, etc.)
        if repaired is None and _json_repair is not None:
            candidate = _json_repair(act_json_str)
            if candidate and candidate not in ('null', '""', '[]', '{}'):
                repaired = candidate
        if repaired is not None:
            try:
                act_payload = json.loads(repaired)
            except json.JSONDecodeError:
                raise SchemaValidationError("act block not valid JSON")
        else:
            raise SchemaValidationError("act block not valid JSON")

    # Normalize: single dict stays dict, list stays list
    if isinstance(act_payload, list):
        # Flatten nested lists: qwen3-coder-next sometimes wraps the batch in an
        # extra array, producing [[action1, action2]] instead of [action1, action2].
        flattened: list = []
        for _item in act_payload:
            if isinstance(_item, list):
                flattened.extend(_item)
            else:
                flattened.append(_item)
        act_payload = flattened
        if not BATCH_ENABLED:
            # Batching disabled — take only the first action
            if not act_payload:
                raise SchemaValidationError("act block is empty list")
            act_payload = act_payload[0]
        elif len(act_payload) == 0:
            raise SchemaValidationError("act block is empty list")
        elif len(act_payload) == 1:
            act_payload = act_payload[0]  # single-element list → dict
        elif len(act_payload) > BATCH_MAX_ACTIONS:
            act_payload = act_payload[:BATCH_MAX_ACTIONS]
    elif not isinstance(act_payload, dict):
        raise SchemaValidationError("act block must be JSON object or array")

    return think_content, act_payload


def extract_think_content(full_response: str) -> str | None:
    """Extract think content from response, ignoring act block issues.

    Used on the SchemaValidationError path to recover think content
    that parse_response couldn't return due to act block failure.
    """
    think_end = full_response.find("</think>")
    if think_end != -1:
        think_start = 0
        if full_response.startswith("<think>\n"):
            think_start = len("<think>\n")
        elif full_response.startswith("<think>"):
            think_start = len("<think>")
        content = full_response[think_start:think_end].strip()
        return content if content else None
    else:
        # No </think> — extract everything after <think>
        if full_response.startswith("<think>\n"):
            content = full_response[len("<think>\n"):].strip()
        elif full_response.startswith("<think>"):
            content = full_response[len("<think>"):].strip()
        else:
            content = full_response.strip()
        return content if content else None


# =========================================================================
# Argument Name Normalization
# =========================================================================

# Common wrong argument names emitted by LLMs → canonical names.
# The normalizer (llm_client.py) fixes *action* names but not *arg* names.
# This cheap dict-remap catches the most frequent hallucinations.
_ARG_NAME_ALIASES: dict[str, dict[str, str]] = {
    # read_file: model says "filepath", "file", "file_path", "filename" → "path"
    "read_file":    {"filepath": "path", "file": "path", "file_path": "path", "filename": "path"},
    "write_file":   {"filepath": "path", "file": "path", "file_path": "path", "filename": "path"},
    "append_file":  {"filepath": "path", "file": "path", "file_path": "path", "filename": "path"},
    "delete_file":  {"filepath": "path", "file": "path", "file_path": "path", "filename": "path"},
    "list_dir":     {"filepath": "path", "directory": "path", "dir": "path", "dir_path": "path", "folder": "path"},
    # run_command: model says "command", "shell", "bash", "script" → "cmd"
    "run_command":  {"command": "cmd", "shell": "cmd", "bash": "cmd", "script": "cmd"},
    "run_background": {"command": "cmd", "shell": "cmd", "bash": "cmd", "script": "cmd"},
    # done: model says "result", "output" → "summary"; "text", "response" → "message"
    "done":         {"result": "summary", "output": "summary", "text": "message", "response": "message"},
    # abort: model says "message", "error" → "reason"
    "abort":        {"message": "reason", "error": "reason"},
    # press_key: model says "keys", "hotkey", "shortcut" → "key"
    "press_key":    {"keys": "key", "hotkey": "key", "shortcut": "key"},
    # type_text: model says "content", "input", "string" → "text"
    "type_text":    {"content": "text", "input": "text", "string": "text"},
    # fetch_url: model says "link", "href" → "url"
    "fetch_url":    {"link": "url", "href": "url"},
    # open_app: model says "name", "application" → "app"
    "open_app":     {"name": "app", "application": "app"},
    "focus_window": {"name": "app", "application": "app", "window": "app"},
    # click: model says "coord_x"/"coord_y" → "x"/"y"
    "click":        {"coord_x": "x", "coord_y": "y", "pos_x": "x", "pos_y": "y"},
}


def normalize_arg_names(act_dict: dict) -> dict:
    """Remap common wrong argument names to canonical names in-place.

    Mutates and returns act_dict.  Non-destructive: if a canonical name
    already exists, the alias is left alone (avoids overwriting good data).
    """
    action = act_dict.get("action")
    args = act_dict.get("args")
    if not isinstance(action, str) or not isinstance(args, dict):
        return act_dict

    aliases = _ARG_NAME_ALIASES.get(action)
    if not aliases:
        return act_dict

    for wrong_name, canonical in aliases.items():
        if wrong_name in args and canonical not in args:
            args[canonical] = args.pop(wrong_name)

    # Special case: "done" with completely empty args — inject defaults
    if action == "done" and not args:
        args["summary"] = "task completed"
        args["message"] = "task completed"

    return act_dict


# =========================================================================
# Schema Validation
# =========================================================================

def validate_act(act_dict: dict) -> None:
    """Validate act_dict against ACTION_SCHEMA. Raises SchemaValidationError."""
    # Guard: batch items may be strings or other non-dict values if the LLM
    # emitted a malformed array.  Catch this before .get() crashes.
    if not isinstance(act_dict, dict):
        raise SchemaValidationError(
            f"action item is not a dict (got {type(act_dict).__name__}: "
            f"{repr(act_dict)[:80]})"
        )
    # 0. Normalize argument names before validation
    normalize_arg_names(act_dict)

    # 1. action key present and is str
    action = act_dict.get("action")
    if not isinstance(action, str):
        raise SchemaValidationError("missing or invalid 'action' field")

    # 2. action in ACTION_SCHEMA
    if action not in ACTION_SCHEMA:
        raise SchemaValidationError(f"unknown action: {action}")

    # 3. args key present and is dict
    args = act_dict.get("args")
    if not isinstance(args, dict):
        raise SchemaValidationError("missing or invalid 'args' field")

    # 4. Per-action required arg checks
    schema = ACTION_SCHEMA[action]
    for key, expected_type in schema.items():
        if key not in args:
            raise SchemaValidationError(
                f"arg '{key}': missing from args"
            )
        val = args[key]
        if not isinstance(val, expected_type):
            type_name = (
                expected_type.__name__
                if isinstance(expected_type, type)
                else str(expected_type)
            )
            raise SchemaValidationError(
                f"arg '{key}': expected {type_name}, "
                f"got {type(val).__name__}"
            )
