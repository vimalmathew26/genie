"""
Genie — scratchpad_writer.py
ScratchpadWriter: extracts structured facts from observations and updates
the TaskScratchpad after every brain loop iteration.

Two prompt templates:
  1. Per-iteration: observation → fact extraction (verbatim-only)
  2. Inter-subtask: outcome → outcome entry + decision promotions (wholesale)

Model: ministral-8b-2512 (locked).  Hard 2s timeout — abandoned if exceeded,
scratchpad unchanged, brain loop continues.
"""
from __future__ import annotations

import json
import sys
import time

import httpx

from config import (
    MODEL_PRICING,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    SCRATCHPAD_WRITER_INPUT_CHARS,
    SCRATCHPAD_WRITER_MODEL,
    SCRATCHPAD_WRITER_STDERR_CHARS,
    SCRATCHPAD_WRITER_TIMEOUT_S,
)
from task_scratchpad import TaskScratchpad


# =========================================================================
# Prompt templates
# =========================================================================

_ITERATION_SYSTEM_PROMPT = """\
You are a structured fact extractor for an automated software agent.

You receive:
- ACTION: the action name and args the agent just executed
- STDOUT: raw stdout from the action (may be truncated)
- STDERR: raw stderr or error message (may be empty)
- CURRENT SCRATCHPAD: the existing working memory (JSON)

Your job: update the scratchpad with new facts discovered in this observation.

RULES:
1. VERBATIM ONLY for facts, decisions, errors.
   Every value you write MUST appear as a literal string in STDOUT or STDERR.
   Do NOT infer, guess, derive, summarize, or rephrase.
   If a value does not appear verbatim in the output, do NOT include it.
2. Keys should be short, descriptive, snake_case identifiers.
3. Only add NEW information. Do not repeat entries already in the scratchpad
   unless correcting a value that the new observation supersedes.
4. For "errors": only record failures, exceptions, or unexpected outcomes.
   Key by a short identifier (e.g. "pip_install", "git_config").
5. For "files": record path → role/purpose mappings discovered in output.
   Key = filename (e.g. "scheduler.py"), value = brief role description.
6. For "decisions": record tool/framework/approach choices made during execution.
7. If the observation contains nothing extractable, return empty updates.
8. READ_FILE ACTIONS: when ACTION is "read_file", the STDOUT is source code.
   Your MAIN job is to extract ONE "files" entry:
     key = the filename from the ACTION args path
     value = the role/purpose from the docstring, class names, or imports
   Example: {"files": {"persistence.py": "SQLite WAL persistence layer"}}
   For facts: extract at most 1-2 KEY architectural details (e.g. the DB
   library used, the web framework). Do NOT dump variable values, repr()
   output, Pydantic field values, enum member listings, or test assertions.
9. FACT DISCIPLINE: for ANY action, keep facts to at most 3 concise entries
   per observation. Prefer high-level insights (library versions, config
   paths, test pass/fail counts) over individual data values. Never record
   repr() of Python objects, individual field values from data structures,
   or raw variable assignments as facts.

Return ONLY a JSON object with this exact schema — no commentary, no markdown:
{
  "facts": {"key": "value", ...},
  "files": {"key": "value", ...},
  "decisions": {"key": "value", ...},
  "errors": {"key": "value", ...}
}

Omit categories with no updates (or use empty dict {}).
Return {} if nothing new to extract."""

_INTERSUBTASK_SYSTEM_PROMPT = """\
You are a subtask completion processor for an automated software agent.

You receive:
- SUBTASK: the subtask description that just completed
- OUTCOME: "done" or "failed"
- SUMMARY: the orchestrator's completion summary
- CURRENT ERRORS: the errors dict from the scratchpad

Your job:
1. Write a subtask_outcomes entry: {"SUBTASK_NUMBER": "done"} or
   {"SUBTASK_NUMBER": "failed — <brief reason from summary>"}.
2. Scan CURRENT ERRORS for entries that represent DURABLE WORKAROUNDS —
   patterns like "used X instead", "fallback to Y", "workaround: Z",
   "switched to", "alternative:". Promote those to decisions.
   Promoted entries should be REMOVED from errors.
3. Leave non-workaround errors unchanged.

Return ONLY a JSON object with this exact schema — no commentary, no markdown:
{
  "subtask_outcomes": {"N": "done_or_failed_reason"},
  "decisions": {"key": "value", ...},
  "errors": {"key": "value", ...}
}

"decisions" and "errors" are the COMPLETE updated dicts — not deltas.
The orchestrator will replace these categories wholesale."""


# =========================================================================
# ScratchpadWriter
# =========================================================================

class ScratchpadWriter:
    """Extracts structured facts from observations and updates the scratchpad.

    Owns a dedicated httpx client with a tight timeout separate from
    the brain loop's LLMClient (which has 120s read timeout).
    """

    def __init__(self) -> None:
        self._http = httpx.Client(
            timeout=httpx.Timeout(
                connect=SCRATCHPAD_WRITER_TIMEOUT_S,
                read=SCRATCHPAD_WRITER_TIMEOUT_S,
                write=SCRATCHPAD_WRITER_TIMEOUT_S,
                pool=SCRATCHPAD_WRITER_TIMEOUT_S,
            ),
        )
        self._model = SCRATCHPAD_WRITER_MODEL

    # ----------------------------------------------------------------
    # Per-iteration write
    # ----------------------------------------------------------------

    def write_iteration(
        self,
        scratchpad: TaskScratchpad,
        action_name: str,
        action_args: dict,
        raw_stdout: str,
        raw_stderr: str,
    ) -> tuple[bool, float]:
        """Extract facts from one brain loop iteration and update scratchpad.

        Args:
            scratchpad: the live TaskScratchpad to update in place
            action_name: name of the action that was executed
            action_args: args dict of the action
            raw_stdout: raw stdout from _dispatch(), pre-truncation
            raw_stderr: raw stderr or exception message

        Returns:
            (success, cost_usd).  success=False on timeout or parse failure.
            On failure, scratchpad is unchanged.
        """
        # Truncate inputs to configured caps.
        stdout_truncated = raw_stdout[:SCRATCHPAD_WRITER_INPUT_CHARS] if raw_stdout else ""
        stderr_truncated = raw_stderr[:SCRATCHPAD_WRITER_STDERR_CHARS] if raw_stderr else ""

        # Build compact args summary (avoid dumping huge write_file content).
        args_summary = _summarize_args(action_name, action_args)

        user_content = (
            f"ACTION: {action_name} {args_summary}\n"
            f"STDOUT:\n{stdout_truncated}\n"
        )
        if stderr_truncated:
            user_content += f"STDERR:\n{stderr_truncated}\n"

        current_json = json.dumps(scratchpad.to_dict(), default=str)
        # Cap the current scratchpad injection to avoid bloating the writer input.
        if len(current_json) > 2000:
            current_json = current_json[:2000] + "...(truncated)"
        user_content += f"CURRENT SCRATCHPAD:\n{current_json}"

        messages = [
            {"role": "system", "content": _ITERATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        response_text, cost = self._llm_call(messages)
        if response_text is None:
            return False, cost

        parsed = self._parse_json(response_text)
        if parsed is None:
            return False, cost

        # Validate and apply updates — verbatim type check.
        for category in ("facts", "files", "decisions", "errors"):
            updates = parsed.get(category)
            if not updates or not isinstance(updates, dict):
                continue
            # Type-validate: all keys and values must be strings.
            clean = {}
            for k, v in updates.items():
                if isinstance(k, str) and isinstance(v, str) and v.strip():
                    clean[k] = v
            if clean:
                scratchpad.update_category(category, clean)

        # Compress after updates.
        scratchpad.compress()
        return True, cost

    # ----------------------------------------------------------------
    # Inter-subtask write
    # ----------------------------------------------------------------

    def write_inter_subtask(
        self,
        scratchpad: TaskScratchpad,
        subtask_number: int,
        subtask_description: str,
        outcome: str,
        summary: str,
    ) -> tuple[bool, float]:
        """Process subtask completion: write outcome entry + promote workarounds.

        Args:
            scratchpad: the live TaskScratchpad to update in place
            subtask_number: the subtask index (1-based)
            subtask_description: the subtask's description string
            outcome: "done" or "failed"
            summary: orchestrator's summary string for this subtask

        Returns:
            (success, cost_usd).  On failure, only the subtask_outcomes
            entry is written (without LLM — guaranteed), but decision
            promotion is skipped.
        """
        current_errors = dict(scratchpad.errors)

        # If no errors to scan, skip the LLM call entirely —
        # just write the outcome entry directly.
        if not current_errors:
            outcome_val = "done" if outcome == "done" else f"failed — {summary[:80]}"
            scratchpad.update_category(
                "subtask_outcomes", {str(subtask_number): outcome_val},
            )
            scratchpad.compress()
            return True, 0.0

        user_content = (
            f"SUBTASK: {subtask_number}. {subtask_description}\n"
            f"OUTCOME: {outcome}\n"
            f"SUMMARY: {summary}\n"
            f"CURRENT ERRORS:\n{json.dumps(current_errors, default=str)}"
        )

        messages = [
            {"role": "system", "content": _INTERSUBTASK_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        response_text, cost = self._llm_call(messages)
        if response_text is None:
            # Fallback: write outcome entry without LLM.
            outcome_val = "done" if outcome == "done" else f"failed — {summary[:80]}"
            scratchpad.update_category(
                "subtask_outcomes", {str(subtask_number): outcome_val},
            )
            scratchpad.compress()
            return False, cost

        parsed = self._parse_json(response_text)
        if parsed is None:
            # Fallback: write outcome entry without LLM.
            outcome_val = "done" if outcome == "done" else f"failed — {summary[:80]}"
            scratchpad.update_category(
                "subtask_outcomes", {str(subtask_number): outcome_val},
            )
            scratchpad.compress()
            return False, cost

        # Wholesale replacement of subtask_outcomes, decisions, errors.
        new_outcomes = parsed.get("subtask_outcomes")
        new_decisions = parsed.get("decisions")
        new_errors = parsed.get("errors")

        # Validate types — all must be dict[str, str] if present.
        if not isinstance(new_outcomes, dict):
            new_outcomes = {str(subtask_number): "done" if outcome == "done" else f"failed — {summary[:80]}"}
        if new_decisions is not None and not isinstance(new_decisions, dict):
            new_decisions = None
        if new_errors is not None and not isinstance(new_errors, dict):
            new_errors = None

        # Merge the new outcome into existing outcomes (don't replace all outcomes).
        merged_outcomes = dict(scratchpad.subtask_outcomes)
        merged_outcomes.update(new_outcomes)

        scratchpad.replace_categories(
            subtask_outcomes=merged_outcomes,
            decisions=new_decisions,
            errors=new_errors,
        )
        scratchpad.compress()
        return True, cost

    # ----------------------------------------------------------------
    # LLM call with hard timeout
    # ----------------------------------------------------------------

    def _llm_call(self, messages: list[dict]) -> tuple[str | None, float]:
        """Fire a single LLM call with hard timeout.

        Returns (response_text, cost_usd).
        response_text is None on any failure (timeout, HTTP error, parse).
        cost_usd is 0.0 on failure (no tokens consumed if request didn't complete).
        """
        payload = {
            "model": self._model,
            "messages": messages,
            "max_tokens": 1024,
        }
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }

        t_start = time.monotonic()
        try:
            resp = self._http.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
        except (httpx.TimeoutException, httpx.RequestError, httpx.HTTPStatusError) as exc:
            elapsed = time.monotonic() - t_start
            print(
                f"[SCRATCHPAD_WRITER] LLM call failed ({elapsed:.1f}s): {exc}",
                file=sys.stderr,
            )
            return None, 0.0

        data = resp.json()
        try:
            content = data["choices"][0]["message"].get("content", "")
        except (KeyError, IndexError):
            return None, 0.0

        # Cost calculation — same pattern as LLMClient.
        usage = data.get("usage", {})
        cost = usage.get("cost")
        if cost is not None:
            cost_usd = float(cost)
        else:
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            pricing = MODEL_PRICING.get(self._model, (0.0, 0.0))
            cost_usd = (
                prompt_tokens * pricing[0] + completion_tokens * pricing[1]
            ) / 1_000_000

        return content, cost_usd

    # ----------------------------------------------------------------
    # JSON parsing
    # ----------------------------------------------------------------

    @staticmethod
    def _parse_json(text: str) -> dict | None:
        """Extract a JSON object from the LLM response.

        Handles common wrapping: markdown fences, leading/trailing whitespace.
        Returns None on parse failure.
        """
        if not text:
            return None
        # Strip markdown fences if present.
        cleaned = text.strip()
        if cleaned.startswith("```"):
            # Remove opening fence (```json or ```)
            first_newline = cleaned.find("\n")
            if first_newline != -1:
                cleaned = cleaned[first_newline + 1:]
            # Remove closing fence
            if cleaned.rstrip().endswith("```"):
                cleaned = cleaned.rstrip()[:-3].rstrip()

        # Find the outermost { ... }
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None

        try:
            obj = json.loads(cleaned[start:end + 1])
            if isinstance(obj, dict):
                return obj
            return None
        except json.JSONDecodeError:
            return None


# =========================================================================
# Helpers
# =========================================================================

def _summarize_args(action_name: str, args: dict) -> str:
    """Build a compact string summary of action args.

    For actions with large content fields (write_file, append_file,
    type_element, type_text), truncate to avoid bloating writer input.
    """
    if not args:
        return "{}"

    _LARGE_FIELDS = {"content", "text"}
    compact = {}
    for k, v in args.items():
        if k in _LARGE_FIELDS and isinstance(v, str) and len(v) > 200:
            compact[k] = v[:200] + f"...({len(v)} chars)"
        else:
            compact[k] = v

    try:
        return json.dumps(compact, default=str)
    except (TypeError, ValueError):
        return str(compact)[:500]