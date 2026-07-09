"""
Genie — script_planner.py
Generates bash script segments for deterministic subtasks before any
action fires.  Sits between subtask decomposition and execution.
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import dataclass

from config import CMD_BLOCKLIST, MODEL
from llm_client import LLMClient

log = logging.getLogger("genie.script_planner")

# ---------------------------------------------------------------------------
# PlanSegment dataclass
# ---------------------------------------------------------------------------

@dataclass
class PlanSegment:
    script: str
    observe_after: bool
    fallback_react: bool


_FALLBACK = PlanSegment(script="", observe_after=False, fallback_react=True)

# ---------------------------------------------------------------------------
# Planner system prompt
# ---------------------------------------------------------------------------

SCRIPT_PLANNER_SYSTEM_PROMPT = """\
You are the planner for Genie, a desktop automation agent.
Your job: given a subtask goal, produce a JSON object with exactly three fields.

Output ONLY valid JSON. No explanation. No markdown. No extra fields.

{
  "script": "<bash script or empty string>",
  "observe_after": <true|false>,
  "fallback_react": <true|false>
}

RULE 1 — Set fallback_react=true when ANY of these apply:
  - Task requires clicking, typing into, or reading a UI element
  - Task requires visual inspection of a screen or window
  - Task requires opening a desktop application
  - Task cannot be completed with shell commands alone
  - Task requires WRITING or AUTHORING file content (Python code, configs,
    YAML/TOML, SQL schemas, Markdown, JSON data files, etc.). Writing a
    file with meaningful content is NOT a shell task — set fallback_react=true.
    A heredoc that contains real source code is still authoring — fallback_react=true.
  script must be "" when fallback_react=true.

RULE 2 — Set observe_after=true when:
  - A later segment will need data produced by this script
  - You are not certain this script fully completes the subtask

RULE 3 — Set observe_after=false when:
  - The script fully completes the subtask on exit 0
  - No further decisions are needed after execution

ENVIRONMENT:
  Shell: bash on Ubuntu 24 / Pop!_OS, Xorg
  Available: gh (authenticated), git, python3, pip, npm, curl, jq
  GITHUB_TOKEN: already in environment
  Working dir default: ~/genie_workspace/
  User dirs: ~/Documents/, ~/Downloads/, ~/Desktop/
  NEVER use rm -rf outside ~/genie_workspace/
  Always start script with: set -e

EXAMPLES:

GOAL: "Clone SmartStudy repo to ~/Documents"
OUTPUT:
{"script": "set -e\\nUSERNAME=$(gh api user -q .login)\\ngh repo clone \\"$USERNAME/SmartStudy\\" ~/Documents/SmartStudy", "observe_after": false, "fallback_react": false}

GOAL: "Find the latest open issue on SmartStudy and summarise it"
OUTPUT:
{"script": "set -e\\ngh issue list --repo $(gh api user -q .login)/SmartStudy --state open --limit 1 --json number,title,body", "observe_after": true, "fallback_react": false}

GOAL: "Open the SmartStudy repo page in Chrome and screenshot it"
OUTPUT:
{"script": "", "observe_after": false, "fallback_react": true}

PRIOR_RESULTS:
  When provided, PRIOR_RESULTS contains stdout from previously completed
  subtasks in this task. Use it to avoid re-doing work already done and
  to extract values needed for the current subtask (e.g. a repo URL or
  username captured in a prior step)."""

# ---------------------------------------------------------------------------
# Blocklist scan (Gate 1)
# ---------------------------------------------------------------------------

def _scan_blocklist(script: str) -> str | None:
    """Return the offending line if any line matches CMD_BLOCKLIST, else None."""
    for line in script.splitlines():
        line_stripped = line.strip()
        if not line_stripped or line_stripped.startswith("#"):
            continue
        line_lower = line_stripped.lower()
        for match_type, match_value in CMD_BLOCKLIST:
            if match_type == "substr":
                if match_value in line_lower:
                    return line_stripped
            elif match_type == "regex":
                if re.search(match_value, line_lower):
                    return line_stripped
    return None


# ---------------------------------------------------------------------------
# Bash syntax check (Gate 2)
# ---------------------------------------------------------------------------

def _bash_syntax_check(script: str) -> str | None:
    """Return stderr string if bash -n rejects the script, else None."""
    try:
        result = subprocess.run(
            ["bash", "-n"],
            input=script,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return result.stderr
    except subprocess.TimeoutExpired:
        return "bash -n timed out"
    except Exception as exc:  # noqa: BLE001
        return str(exc)
    return None


# ---------------------------------------------------------------------------
# ScriptPlanner
# ---------------------------------------------------------------------------

class ScriptPlanner:
    """Generates deterministic bash script segments for subtasks."""

    def __init__(self, llm_client: LLMClient, model: str = MODEL) -> None:
        self._llm = llm_client
        self._model = model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(self, subtask_goal: str, last_result: str | None = None,
             prior_results: str | None = None) -> PlanSegment:
        """Ask the LLM for a script segment.  Never raises."""
        try:
            return self._plan_inner(subtask_goal, last_result, prior_results)
        except Exception:  # noqa: BLE001
            log.warning("script_planner unexpected error", exc_info=True)
            return _FALLBACK

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _plan_inner(self, subtask_goal: str, last_result: str | None,
                    prior_results: str | None) -> PlanSegment:
        # Build user message
        parts = []
        if prior_results is not None:
            parts.append(f"PRIOR_RESULTS:\n{prior_results}")
        if last_result is not None:
            parts.append(f"LAST_RESULT:\n{last_result}")
        parts.append(f"GOAL: {subtask_goal}")
        user_content = "\n\n".join(parts)

        messages = [
            {"role": "system", "content": SCRIPT_PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        try:
            raw_text, _cost = self._llm.call(messages, model=self._model, max_tokens=2048)
        except Exception as exc:  # noqa: BLE001
            log.warning("script_planner LLM call failed: %s", exc)
            return _FALLBACK

        # Parse JSON
        try:
            data = json.loads(raw_text.strip())
        except (json.JSONDecodeError, ValueError) as exc:
            log.warning("script_planner malformed JSON: %s — raw: %.200s", exc, raw_text)
            return _FALLBACK

        # Validate required fields
        if not isinstance(data, dict):
            log.warning("script_planner response is not a dict: %.200s", raw_text)
            return _FALLBACK

        for field in ("script", "observe_after", "fallback_react"):
            if field not in data:
                log.warning("script_planner missing field '%s': %.200s", field, raw_text)
                return _FALLBACK

        script = str(data["script"])
        observe_after = bool(data["observe_after"])
        fallback_react = bool(data["fallback_react"])

        # If fallback_react, return immediately (no gates needed)
        if fallback_react:
            return PlanSegment(script="", observe_after=False, fallback_react=True)

        # --- Gate 1: blocklist scan ---
        bad_line = _scan_blocklist(script)
        if bad_line is not None:
            log.warning("script_planner blocklist rejection: %s", bad_line)
            return _FALLBACK

        # --- Gate 2: bash -n syntax check ---
        syntax_err = _bash_syntax_check(script)
        if syntax_err is not None:
            log.warning("script_planner bash -n failed: %s", syntax_err)
            return _FALLBACK

        return PlanSegment(
            script=script,
            observe_after=observe_after,
            fallback_react=False,
        )
