"""
Genie — llm_client.py
LLM API wrapper with cost accumulation.

Extracted from orchestrator.py to isolate the HTTP transport + cost logic
from the brain loop.
"""
from __future__ import annotations

import httpx

from config import (
    ACTION_SCHEMA,
    LLM_CONNECT_TIMEOUT,
    LLM_READ_TIMEOUT,
    MODEL_PRICING,
    NORMALIZER_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
)
from exceptions import ResponseTruncatedError


_VALID_ACTIONS = ", ".join(sorted(ACTION_SCHEMA.keys()))

# Models known to reject assistant-prefill (trailing assistant message in
# the messages array).  Qwen and DeepSeek accept it; everything else does not.
_PREFILL_SUPPORTED_PREFIXES = ("qwen/", "deepseek/")

_NORMALIZE_SYSTEM_PROMPT = f"""\
You are a JSON extraction and correction assistant. The text below is a raw LLM \
response that should contain a ReAct-style action decision. Your job:

1. Extract the JSON action object — strip any preamble, markdown fences, \
explanation, or think blocks.
2. Validate the "action" field against the ONLY valid action names:
   {_VALID_ACTIONS}
3. If the action name is NOT in that list, map it to the closest valid action.
   Common mappings:
   - "none", "finish", "complete", "end", "stop", "no_action", "noop" → "done"
   - "run_bash", "exec", "execute", "shell", "bash", "terminal" → "run_command"
   - "open", "launch" → "open_app"
   - "write", "save" → "write_file"
   - "read" → "read_file"
   - "ls", "dir" → "list_dir"
   - "type", "input" → "type_text"
   - "press", "hotkey", "shortcut" → "press_key"
   - "cancel" → "abort"
   For any other unknown action, use your best judgment to pick the closest match.
4. Fix argument names to match the expected schema:
   - For read_file, write_file, append_file, delete_file, list_dir: \
use "path" (NOT "filepath", "file", "file_path", "filename", "directory")
   - For run_command, run_background: use "cmd" (NOT "command", "shell", "bash")
   - For done: use "summary" and "message" (NOT "result", "output", "text")
   - For abort: use "reason" (NOT "message", "error")
   - For press_key: use "key" (NOT "keys", "hotkey")
   - For open_app, focus_window: use "app" (NOT "name", "application")
5. Preserve all arg values exactly as-is — only rename the keys if needed.

Expected output schema (single action):
{{"action": "<valid_action_name>", "args": {{<action_args>}}}}

Or a list for batched actions:
[{{"action": "...", "args": {{...}}}}, ...]

If the response contains <think>...</think> or <act>...</act> tags, extract \
the content inside <act> tags. If no <act> tags exist but JSON is present \
anywhere in the text (including inside <think> blocks), extract that JSON.
If it contains ```json fences, extract the JSON inside.

Return ONLY the corrected JSON — no wrapping, no commentary."""


class LLMClient:
    """Thin wrapper around OpenRouter-compatible chat completions API."""

    def __init__(self) -> None:
        self._http = httpx.Client(
            timeout=httpx.Timeout(
                connect=LLM_CONNECT_TIMEOUT,
                read=LLM_READ_TIMEOUT,
                write=LLM_CONNECT_TIMEOUT,
                pool=LLM_CONNECT_TIMEOUT,
            ),
        )

    def call(
        self,
        messages: list[dict],
        model: str,
        max_tokens: int = 8192,
        extra_body: dict | None = None,
    ) -> tuple[str, float]:
        """Single LLM API call.

        Returns (response_text, cost_usd).
        Raises httpx.TimeoutException, httpx.RequestError, httpx.HTTPStatusError.
        """
        # Strip trailing assistant prefill for models that don't support it.
        # DeepSeek and Qwen accept <think>\n prefill; Claude/GPT/etc. reject it
        # with a 400 "does not support assistant message prefill" error.
        supports_prefill = any(
            model.startswith(pfx) for pfx in _PREFILL_SUPPORTED_PREFIXES
        )
        if (
            not supports_prefill
            and messages
            and messages[-1].get("role") == "assistant"
        ):
            messages = messages[:-1]

        payload: dict = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if extra_body:
            payload.update(extra_body)
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }

        resp = self._http.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            json=payload,
            headers=headers,
        )
        resp.raise_for_status()

        data = resp.json()
        message = data["choices"][0]["message"]
        content = message.get("content")
        # DeepSeek R1 (and other reasoning-first models) return content=None
        # and put the actual response in the 'reasoning' field instead.
        if content is None:
            content = message.get("reasoning") or ""
        finish_reason = data["choices"][0].get("finish_reason", "")

        # -- Cost calculation --
        usage = data.get("usage", {})
        cost = usage.get("cost")
        if cost is not None:
            cost_usd = float(cost)
        else:
            # Fallback: token-based pricing
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            pricing = MODEL_PRICING.get(model, (0.0, 0.0))
            cost_usd = (
                prompt_tokens * pricing[0] + completion_tokens * pricing[1]
            ) / 1_000_000

        # Detect max_tokens truncation — the response is incomplete
        if finish_reason == "length":
            raise ResponseTruncatedError(content, cost_usd)

        return content, cost_usd

    def normalize_response(self, raw_response: str) -> tuple[str, float]:
        """Reformat a non-Qwen model response into clean action JSON.

        Sends the raw response through the NORMALIZER_MODEL (cheap Qwen call)
        to strip preamble, code fences, think blocks, and extract the action
        JSON in the expected schema.

        Returns (normalized_text, normalizer_cost_usd).
        On failure (network, parse), returns the raw_response unchanged with
        cost 0.0 — the existing parser hardening in response_parser.py will
        handle it as a best-effort fallback.
        """
        messages = [
            {"role": "system", "content": _NORMALIZE_SYSTEM_PROMPT},
            {"role": "user", "content": raw_response},
        ]
        try:
            normalized, cost = self.call(
                messages, model=NORMALIZER_MODEL, max_tokens=4096,
            )
            return normalized, cost
        except Exception:
            # Normalization is best-effort — don't block the brain loop
            return raw_response, 0.0
