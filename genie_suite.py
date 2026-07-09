"""
genie_suite.py — End-to-end behavioural regression suite for Genie. (T1–T26 base suite)

=============================================================================
What Genie Is
=============================================================================

Genie is a personal AI-powered desktop automation agent running on Pop!_OS
(Xorg/GNOME). It controls the entire OS: opening apps, typing, pressing keys,
clicking elements by semantic identity (not coordinates), reading screen state,
and executing autonomous multi-step tasks — driven by cloud LLMs via
OpenRouter (up to 671B models).

The long-term goal: a tireless autonomous agent capable of executing the full
working capacity of a human developer/operator, running 24/7, learning from
its own execution history, with TOTP-gated task approval. See the_40.md and
the_60.md for the full vision.

=============================================================================
The 4-Layer Architecture (all layers COMPLETE as of Session 49)
=============================================================================

  Layer 1 — Window Identity & Registry     (window_registry.py)
    PID-keyed session window registry. open_app blocks until WID validated.
    Background poller detects closed/crashed windows.

  Layer 2 — Element Resolution             (element_resolver.py)
    Three-tier: AT-SPI (GTK/Qt) → CDP websocket (Chrome/Electron) → SoM.
    LLM emits role+name semantics. Never coordinates.

  Layer 3 — Observation Capture            (observation.py)
    Unified observe() call after every action. Passive capture (no LLM).
    Writes genie_success.jsonl and genie_incomplete.jsonl.
    This is the dataset collection pipeline for eventual fine-tuning.

  Layer 4 — Brain Loop / ReAct Orchestrator (orchestrator.py, actions.py)
    GenieOrchestrator: plan phase → brain loop. Hybrid <think>/<act> tags.
    Per-task + monthly cost tracking. Checkpoint recovery. Telegram bot.

=============================================================================
Why This Test File Exists
=============================================================================

We use cloud LLMs, not fine-tuned models. That means Genie's behaviour on any
given task is probabilistic — the same goal can succeed in one run and fail in
another depending on model output. A test suite that only checks TaskResult
strings is useless; a hallucinated "done" passes it trivially.

This file tests REAL OUTCOMES: what is actually on the screen, what URL Chrome
is actually on, what files actually exist on disk, what commands actually ran.

=============================================================================
The Model Reality (as of 2026-03-05)
=============================================================================

  Primary:  qwen/qwen3-next-80b-a3b:instruct   (~$0.001138/iter, 6s latency)
  Fallback: devstral/devstral-2-2512            (~$0.002831/iter, 30s latency)

  Rejected candidates (documented in so_far.md / post_L4.md):
    - DeepSeek V3 0324        — 27s latency, Chinese char artifacts
    - Claude Haiku 3.0        — wrong tool selection (opened VSCode for file write)
    - Grok 4.1 Fast           — skipped explicit verification steps
    - Gemini 3.1 Flash Lite   — piecemeal writing, context blowup
    - Llama 4 Scout           — wrong tool selection cold-start
    - Qwen3.5 35B A3B         — native thinking, message["content"] always empty
    - Nemotron 3 Nano 30B A3B — hits max_tokens on every call, <act> never emitted
    - Devstral Small 1.1      — 1/3 reliability

  Fine-tuning is planned (the_40 post-validation). Weekly automated retraining
  on clean JSONL traces. Genie accumulates its own dataset every run.

=============================================================================
Test Authoring Philosophy
=============================================================================

1. REAL STACK ONLY.
   Every test runs the full Genie stack: LLM → orchestrator → actions →
   element resolver → desktop. No mocks, no stubs, no shortcuts.
   If Genie can't complete it through its own stack, the test fails.

2. REAL STATE VALIDATORS.
   Validators read actual OS/browser state AFTER task completion.
   They do NOT trust TaskResult.summary — the LLM can hallucinate that.
   They check: actual URLs via CDP, actual files via os.path, actual
   process state via psutil, actual screen elements via AT-SPI / CDP.

3. COMPACT NUMBERING.
   Tests are numbered 1–N with no gaps. When tests are removed, the
   remaining tests are renumbered to stay compact. This is how we track "did we regress test 1 while fixing test 2?"

4. STRICT REGRESSION CONTRACT.
   A fix for test N MUST NOT cause any previously passing test (1..N-1)
   to fail. If it does, that fix is wrong — revert and find another path.
   This prevents the classic "fixed one thing, broke another" loop, which
   is especially painful when the model itself is non-deterministic.

5. CATEGORY COVERAGE GOAL.
   The task list below covers the_40.md's achievable categories:
     [x] Browser delivery + verification     (tests 1, 2)
     [x] File I/O + shell ops               (tests 3, 6, 7)
     [x] Shell command execution             (tests 4, 8)
     [x] fetch_url pipeline                 (tests 5, 14)
     [x] Code generation + debug cycle      (tests 10–12 — rungs 10–12)
     [x] Software development cycle         (test 13 — rung 13)
     [x] Batch execution                    (test 9)
     [x] Review pipeline                    (test 21)
     [x] fetch_url autonomous               (tests 25, 26)
     [ ] Multi-window parallel tasks        (planned)
   No test case covers the_60.md categories (world-feedback dependent
   tasks) — those are out of scope for Genie's current architecture.

=============================================================================
Running
=============================================================================

  python genie_suite.py              # run all, stop on first failure
  python genie_suite.py --all        # run all, report everything
  python genie_suite.py --case 1     # run test N only

=============================================================================
Adding a New Test Case
=============================================================================

  1. Write a validator function:  _validate_<short_name>(result, orch) -> None
       - Read actual state from OS / browser / filesystem
       - raise AssertionError("human-readable reason") on failure
       - print a confirmation line on success
  2. Append a TestCase to TESTS (see bottom of file).
     Assign the next sequential number. Never reuse numbers.
  3. Run python genie_suite.py --case N to iterate on the new test.
  4. Only merge when --all shows no regressions.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
import urllib.request
from dataclasses import dataclass
from typing import Callable

from config import APP_PROFILES, CHECKPOINT_PATH
import config as _config
from orchestrator import GenieOrchestrator, TaskResult
from review_bot import send_message as _tg_send
from xdotool_controller import XdotoolController

# Disable clarifying questions for the entire automated test suite.
# T38 (Phase 5.5) re-enables this only for its own setup/teardown.
_config.CLARIFY_ENABLED = False


# =============================================================================
# CDP helpers — read actual browser state for post-task validation.
# Used instead of trusting TaskResult.summary, which the LLM can hallucinate.
# =============================================================================

def _cdp_get_current_url(cdp_port: int, timeout: float = 5.0) -> str | None:
    """Return the URL of the active Chrome tab via CDP, or None on failure.

    Prefers real web pages (http/https) over Chrome internal pages
    (chrome://, devtools://, about:) which may appear as page targets.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                f"http://localhost:{cdp_port}/json", timeout=3
            ) as resp:
                tabs = json.loads(resp.read().decode())
            web_tabs = [
                t for t in tabs
                if t.get("type") == "page"
                and t.get("url", "").startswith("http")
            ]
            if web_tabs:
                return web_tabs[0].get("url", "")
            # Fall back to any page target
            for tab in tabs:
                if tab.get("type") == "page":
                    return tab.get("url", "")
        except Exception:
            time.sleep(0.5)
    return None


def _chrome_cdp_port() -> int | None:
    """Return the CDP port for chrome from APP_PROFILES config."""
    return APP_PROFILES.get("chrome", {}).get("cdp_port")


def _runtime_chrome_cdp_port(orch: GenieOrchestrator) -> int | None:
    """Return the actual runtime CDP port for any chrome entry in the
    orchestrator's window registry.  Falls back to static config.

    When open_app is called multiple times (singleton adoption, retries),
    the runtime port may differ from the static APP_PROFILES value because
    the port ledger increments on each call.
    """
    try:
        with orch.registry._registry_lock:
            for label, entry in orch.registry._registry.items():
                if "chrome" in label and entry.get("cdp_port") is not None:
                    return entry["cdp_port"]
    except Exception:
        pass
    return _chrome_cdp_port()


def _get_chrome_url(orch: GenieOrchestrator) -> str | None:
    """Best-effort retrieval of the current Chrome tab URL.

    Tries live CDP first (the browser may still be running if cleanup
    hasn't killed it yet).  Falls back to the URL cached by the
    orchestrator's ``_cleanup_opened_apps`` just before it terminated
    Chrome.  This makes validators resilient to the race between
    cleanup and validation.
    """
    cdp_port = _runtime_chrome_cdp_port(orch)
    if cdp_port is not None:
        url = _cdp_get_current_url(cdp_port, timeout=3.0)
        if url:
            return url
    # Fallback: orchestrator cached last URL before killing Chrome
    return orch.get_last_chrome_url()


def _html_has_id(html_content: str, element_id: str) -> bool:
    """Return True if any element in the parsed HTML has id=element_id."""
    from html.parser import HTMLParser
    class _IDFinder(HTMLParser):
        def __init__(self):
            super().__init__()
            self.found = False
        def handle_starttag(self, tag, attrs):
            for name, value in attrs:
                if name == "id" and value == element_id:
                    self.found = True
    p = _IDFinder()
    p.feed(html_content)
    return p.found


def _html_has_tag(html_content: str, tag: str) -> bool:
    """Return True if the given tag name exists anywhere in the parsed HTML."""
    from html.parser import HTMLParser
    class _TagFinder(HTMLParser):
        def __init__(self):
            super().__init__()
            self.found = False
        def handle_starttag(self, tag_name, attrs):
            if tag_name.lower() == tag.lower():
                self.found = True
    p = _TagFinder()
    p.feed(html_content)
    return p.found


# Module-level action trace — populated by _on_update, read by validators.
_current_action_trace: list[str] = []


# =============================================================================
# TestCase dataclass
# =============================================================================

@dataclass
class TestCase:
    number: int
    description: str
    # Category tag maps to the_40.md achievable task categories.
    # Used in summary output so regressions are easy to interpret.
    # Valid tags: "browser", "file_ops", "shell", "dev_cycle",
    #             "multi_app", "content", "multi_window",
    #             "fetch_url_autonomous", "git_ops", "codebase_nav"
    category: str
    goal: str
    # validator(result, orch) → raises AssertionError with clear message on failure.
    # Must validate REAL state, not just result.outcome or result.summary.
    validator: Callable[[TaskResult, GenieOrchestrator], None]
    # Pre-validator outcome check. Set to None to skip (let validator do all checks).
    expected_outcome: str | None = "done"
    # If set, the test is skipped with this reason displayed.
    skip: str | None = None
    # Optional custom runner: runner(tester) -> TaskResult.
    # When set, replaces the default orch.run_task() call in run_case.
    # Use for tests that need non-standard on_update or mode flags.
    runner: Callable | None = None


# =============================================================================
# Validators
# =============================================================================

# -----------------------------------------------------------------------------
# Test 1 — Rubric 31: deliver HTML with elements, open in Chrome, verify
# -----------------------------------------------------------------------------
# What this test proves:
#   (a) Genie can write a minimal HTML file with specified elements to disk.
#   (b) Genie opens the file in Chrome (navigate to file:// URL).
#   (c) The file contains the required elements (submit button + heading).
#   (d) Chrome is actually showing the local file — not a blank tab.
#
# No external dependencies — the HTML is Genie's own output, so this test
# is deterministic. Replaces old T1 (Google SERP navigation) which had no
# anchor in the rubric and was fragile due to live search results.
# -----------------------------------------------------------------------------

_TEST1_DIR    = "/tmp/genie_t1"
_TEST1_OUTPUT = "/tmp/genie_t1/index.html"


def _setup_t1() -> None:
    import shutil
    if os.path.exists(_TEST1_DIR):
        shutil.rmtree(_TEST1_DIR)
    os.makedirs(_TEST1_DIR, exist_ok=True)


def _validate_cdp_own_frontend(
    result: TaskResult,
    orch: GenieOrchestrator,
) -> None:
    """HTML file must exist with required elements; Chrome must be on the file."""

    assert result.outcome == "done", (
        f"Task did not finish with 'done' (got '{result.outcome}'). "
        f"Summary: {result.summary}"
    )
    assert os.path.exists(_TEST1_OUTPUT), (
        f"HTML file not found at {_TEST1_OUTPUT}. "
        "Genie was asked to create this file."
    )
    with open(_TEST1_OUTPUT, "r", encoding="utf-8") as fh:
        content = fh.read()
    content_lower = content.lower()
    assert "submit" in content_lower, (
        f"HTML does not contain a submit button.\nContent: {content[:400]!r}"
    )
    assert "main-heading" in content_lower or "hello" in content_lower, (
        f"HTML does not contain the expected heading.\nContent: {content[:400]!r}"
    )
    url = _get_chrome_url(orch)
    assert url is not None, "Could not read Chrome URL via CDP or cache."
    assert ("genie_t1" in url or "index.html" in url
            or url.startswith("file://")), (
        f"Chrome is not showing the expected local HTML file.\nURL: {url}"
    )

    # --- R31: parsed DOM confirms spec-defined element IDs are real attributes ---
    assert _html_has_id(content, "main-heading"), (
        "Parsed DOM: id='main-heading' not found as a real element attribute. "
        "R31 requires spec-defined elements to be present."
    )
    assert _html_has_id(content, "submit-btn"), (
        "Parsed DOM: id='submit-btn' not found as a real element attribute. "
        "R31 requires spec-defined elements to be present."
    )
    print(f"    ✓ Parsed DOM: #main-heading and #submit-btn confirmed as real element IDs")

    print(f"    ✓ HTML file exists with submit button and heading")
    print(f"    ✓ Chrome showing local file: {url}")
    print(f"    ✓ Cost: ${result.cost_usd:.4f}  |  Iterations: {result.iterations}")


# -----------------------------------------------------------------------------
# Test 2 — Rubric 31: deliver HTML with page structure, open in Chrome, verify
# -----------------------------------------------------------------------------
# What this test proves:
#   (a) Genie writes an HTML file with structured semantic elements:
#       <nav>, <main id='content'>, <footer>.
#   (b) Chrome opens the file via file:// URL.
#   (c) Validator reads the file directly — no external dependency.
#
# Replaces old T2 (DDG second-result click) which had no rubric anchor
# and was fragile due to DuckDuckGo's AI overlay / layout changes.
# -----------------------------------------------------------------------------

_TEST2_DIR    = "/tmp/genie_t2"
_TEST2_OUTPUT = "/tmp/genie_t2/index.html"


def _setup_t2() -> None:
    import shutil
    if os.path.exists(_TEST2_DIR):
        shutil.rmtree(_TEST2_DIR)
    os.makedirs(_TEST2_DIR, exist_ok=True)


def _validate_cdp_page_structure(
    result: TaskResult,
    orch: GenieOrchestrator,
) -> None:
    """HTML must have <nav>, <main>/<content>, and <footer>; Chrome on the file."""

    assert result.outcome == "done", (
        f"Task did not finish with 'done' (got '{result.outcome}'). "
        f"Summary: {result.summary}"
    )
    assert os.path.exists(_TEST2_OUTPUT), (
        f"HTML file not found at {_TEST2_OUTPUT}."
    )
    with open(_TEST2_OUTPUT, "r", encoding="utf-8") as fh:
        content = fh.read()
    content_lower = content.lower()
    assert "<nav" in content_lower, "HTML missing <nav> element."
    assert "<main" in content_lower or 'id="content"' in content_lower, (
        "HTML missing <main> element or id='content'."
    )
    assert "<footer" in content_lower, "HTML missing <footer> element."
    url = _get_chrome_url(orch)
    assert url is not None, "Could not read Chrome URL via CDP or cache."
    assert "genie_t2" in url or url.startswith("file://"), (
        f"Chrome is not showing the expected local HTML file.\nURL: {url}"
    )

    # --- R31: parsed DOM confirms spec-defined structural elements are real tags ---
    assert _html_has_tag(content, "nav"), (
        "Parsed DOM: <nav> not found as a real element. "
        "R31 requires spec-defined elements to be present."
    )
    assert (_html_has_tag(content, "main") or _html_has_id(content, "content")), (
        "Parsed DOM: <main> or id='content' not found as a real element. "
        "R31 requires spec-defined elements to be present."
    )
    assert _html_has_tag(content, "footer"), (
        "Parsed DOM: <footer> not found as a real element. "
        "R31 requires spec-defined elements to be present."
    )
    print(f"    ✓ Parsed DOM: <nav>, <main>/#content, <footer> confirmed as real elements")

    print(f"    ✓ HTML structure verified: <nav>, <main>/<content>, <footer>")
    print(f"    ✓ Chrome showing local file: {url}")
    print(f"    ✓ Cost: ${result.cost_usd:.4f}  |  Iterations: {result.iterations}")


# -----------------------------------------------------------------------------
# Test 3 — file ops: two-file workflow (write data + script, then run)
# -----------------------------------------------------------------------------
# What this test proves:
#   (a) Genie can write TWO files: an input data file and a processing script.
#   (b) The script must READ the data file at runtime — it cannot hardcode the
#       answer, because the validator overwrites the input with a different line
#       count before running the script independently.
#   (c) The produced Python code is correct and runs cleanly (exit 0).
#
# Goal given to Genie: create /tmp/genie_input.txt (5 lines of any text) +
#   /tmp/genie_test3.py (reads the file, counts lines, prints the count).
#   Then run the script and confirm output is 5.
#
# What the validator does:
#   1. Confirms both files exist.
#   2. Overwrites /tmp/genie_input.txt with 7 lines (not 5!).
#   3. Runs /tmp/genie_test3.py and asserts stdout == "7".
#   => A hardcoded print("5") will fail step 3. Only a real open()+count works.
#
# This exercises: write_file ×2, run_command, stdout observation.
# The_40.md target covered: "Shell ops / file I/O" category.
# -----------------------------------------------------------------------------

_TEST3_SCRIPT = "/tmp/genie_test3.py"
_TEST3_INPUT  = "/tmp/genie_input.txt"
# Validator deliberately uses a DIFFERENT line count than the goal asks for (5).
# This proves the script actually reads the file instead of hardcoding the answer.
_TEST3_VALIDATOR_LINES = 7
_TEST3_VALIDATOR_INPUT = "\n".join(f"line{i}" for i in range(1, _TEST3_VALIDATOR_LINES + 1))


def _validate_script_written_and_runnable(
    result: TaskResult,
    orch: GenieOrchestrator,
) -> None:
    """Script must read /tmp/genie_input.txt, count lines, and print the count.

    The validator substitutes a known input with a DIFFERENT line count than
    Genie was told to create.  If the script hardcodes the answer it will fail.
    """

    assert result.outcome == "done", (
        f"Task did not finish with 'done' (got '{result.outcome}'). "
        f"Summary: {result.summary}"
    )

    assert os.path.exists(_TEST3_SCRIPT), (
        f"Script file not found at {_TEST3_SCRIPT}. "
        "Genie either did not call write_file or wrote to the wrong path."
    )

    assert os.path.exists(_TEST3_INPUT), (
        f"Input data file not found at {_TEST3_INPUT}. "
        "Genie must write both the data file and the script."
    )

    # Substitute a known input so the validator is independent of what Genie wrote.
    # Using a different count (_TEST3_VALIDATOR_LINES=7) than the goal specifies (5)
    # means a hardcoded print(5) will fail here.
    with open(_TEST3_INPUT, "w") as fh:
        fh.write(_TEST3_VALIDATOR_INPUT)

    proc = subprocess.run(
        [sys.executable, _TEST3_SCRIPT],
        capture_output=True, text=True, timeout=10,
    )
    stdout = proc.stdout.strip()
    assert proc.returncode == 0, (
        f"Script exited with code {proc.returncode}.\n"
        f"stderr: {proc.stderr.strip()}"
    )
    assert stdout == str(_TEST3_VALIDATOR_LINES), (
        f"Script printed {stdout!r} but expected '{_TEST3_VALIDATOR_LINES}'.\n"
        "The script must read the file and count lines — not hardcode a number."
    )

    print(
        f"    ✓ {_TEST3_SCRIPT} reads {_TEST3_INPUT} and prints correct line count "
        f"({_TEST3_VALIDATOR_LINES} lines → '{stdout}')"
    )
_TEST4_OUTPUT = "/tmp/genie_test5.txt"
_TEST4_EXPECTED = "42"


def _validate_run_command_stdout(
    result: TaskResult,
    orch: GenieOrchestrator,
) -> None:
    """Output file must exist and contain the correct command stdout."""

    assert result.outcome == "done", (
        f"Task did not finish with 'done' (got '{result.outcome}'). "
        f"Summary: {result.summary}"
    )

    assert os.path.exists(_TEST4_OUTPUT), (
        f"Output file not found at {_TEST4_OUTPUT}. "
        "Genie either did not run the command or did not save the output."
    )

    with open(_TEST4_OUTPUT, "r", encoding="utf-8") as fh:
        content = fh.read().strip()

    assert content == _TEST4_EXPECTED, (
        f"File contains {content!r} but expected {_TEST4_EXPECTED!r}.\n"
        "The command python3 -c 'print(6 * 7)' must produce exactly '42'."
    )

    print(f"    ✓ {_TEST4_OUTPUT} contains '{content}' — run_command stdout captured correctly")
_FETCH_T5_DIR    = "/tmp/genie_fetch_t5"
_FETCH_T5_OUTPUT = "/tmp/genie_fetch_t5/httpx_demo.py"


def _setup_fetch_t5() -> None:
    import shutil
    if os.path.exists(_FETCH_T5_DIR):
        shutil.rmtree(_FETCH_T5_DIR)
    os.makedirs(_FETCH_T5_DIR, exist_ok=True)


def _validate_fetch_url_and_code(
    result: TaskResult,
    orch: GenieOrchestrator,
) -> None:
    """httpx demo script must exist, use httpx, and pass syntax check."""

    assert result.outcome == "done", (
        f"Task did not finish with 'done' (got '{result.outcome}'). "
        f"Summary: {result.summary}"
    )
    assert os.path.exists(_FETCH_T5_OUTPUT), (
        f"Script not found at {_FETCH_T5_OUTPUT}."
    )
    with open(_FETCH_T5_OUTPUT, "r", encoding="utf-8") as fh:
        source = fh.read()
    assert "httpx" in source.lower(), (
        f"Script does not use httpx.\nContent: {source[:300]!r}"
    )
    proc = subprocess.run(
        [sys.executable, "-m", "py_compile", _FETCH_T5_OUTPUT],
        capture_output=True, text=True, timeout=10,
    )
    assert proc.returncode == 0, (
        f"Script has syntax errors: {proc.stderr.strip()}"
    )

    # --- R24: verify fetch_url was actually called before writing code ---
    _assert_fetch_url_in_trace("T5")

    print(f"    ✓ httpx demo script written to {_FETCH_T5_OUTPUT}")
    print(f"    ✓ Script uses httpx and passes syntax check")
    print(f"    ✓ Cost: ${result.cost_usd:.4f}  |  Iterations: {result.iterations}")


# -----------------------------------------------------------------------------
# Test 11 — Rung 7: Write file to disk
# -----------------------------------------------------------------------------
# What this test proves:
#   (a) Genie can write a file with specific structured content to disk.
#   (b) Validator reads the actual file and checks content correctness.
#   (c) Standalone write_file test (Test 3 also writes but tests a workflow).
#
# Goal: write a JSON file with specific keys and values.
# Validator: file exists, is valid JSON, has expected keys.
# -----------------------------------------------------------------------------

_TEST6_OUTPUT = "/tmp/genie_test11.json"


def _validate_write_file(
    result: TaskResult,
    orch: GenieOrchestrator,
) -> None:
    """JSON file must exist on disk with the expected structure."""

    assert result.outcome == "done", (
        f"Task did not finish with 'done' (got '{result.outcome}'). "
        f"Summary: {result.summary}"
    )

    assert os.path.exists(_TEST6_OUTPUT), (
        f"File not found at {_TEST6_OUTPUT}. "
        "Genie was asked to write a JSON file here."
    )

    with open(_TEST6_OUTPUT, "r", encoding="utf-8") as fh:
        raw = fh.read()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise AssertionError(
            f"File is not valid JSON: {e}\nContent: {raw[:200]!r}"
        ) from e

    expected_keys = {"name", "version", "language"}
    actual_keys = set(data.keys())
    missing = expected_keys - actual_keys
    assert not missing, (
        f"JSON is missing expected keys: {missing}. "
        f"Actual keys: {actual_keys}"
    )

    assert data.get("name") == "genie", (
        f"Expected name='genie', got name={data.get('name')!r}"
    )
    assert data.get("language") == "python", (
        f"Expected language='python', got language={data.get('language')!r}"
    )

    print(f"    ✓ {_TEST6_OUTPUT} written with valid JSON: {data}")


# -----------------------------------------------------------------------------
# Test 12 — Rung 8: Read file from disk
# -----------------------------------------------------------------------------
# What this test proves:
#   (a) Genie can read an existing file from disk using the read_file action.
#   (b) Validator pre-creates a file with known content. Genie must read it
#       and report or save the content elsewhere.
#   (c) Tests the read_file → observation pipeline.
#
# Setup: validator creates /tmp/genie_test12_input.txt with known content.
# Goal: read that file and write its word count to /tmp/genie_test12_output.txt.
# Validator: output file contains the correct word count.
# -----------------------------------------------------------------------------

_TEST7_INPUT = "/tmp/genie_test12_input.txt"
_TEST7_OUTPUT = "/tmp/genie_test12_output.txt"
_TEST7_CONTENT = "the quick brown fox jumps over the lazy dog near the river bank"
_TEST7_WORD_COUNT = len(_TEST7_CONTENT.split())  # 13


def _setup_test7() -> None:
    """Pre-create the input file for test 12."""
    with open(_TEST7_INPUT, "w") as fh:
        fh.write(_TEST7_CONTENT)


def _validate_read_file(
    result: TaskResult,
    orch: GenieOrchestrator,
) -> None:
    """Output file must contain the correct word count of the input file."""

    assert result.outcome == "done", (
        f"Task did not finish with 'done' (got '{result.outcome}'). "
        f"Summary: {result.summary}"
    )

    assert os.path.exists(_TEST7_OUTPUT), (
        f"Output file not found at {_TEST7_OUTPUT}. "
        "Genie was asked to save the word count here."
    )

    with open(_TEST7_OUTPUT, "r", encoding="utf-8") as fh:
        content = fh.read().strip()

    # Accept the number in various formats: "12", "12 words", "Word count: 12"
    assert str(_TEST7_WORD_COUNT) in content, (
        f"Output file does not contain '{_TEST7_WORD_COUNT}'.\n"
        f"File content: {content!r}\n"
        f"Expected word count of: {_TEST7_CONTENT!r} = {_TEST7_WORD_COUNT} words."
    )

    print(
        f"    ✓ Read {_TEST7_INPUT} and wrote word count to {_TEST7_OUTPUT} "
        f"(content: {content!r}, expected {_TEST7_WORD_COUNT})"
    )


# -----------------------------------------------------------------------------
# Test 13 — Rung 9: Run shell command, check stdout
# -----------------------------------------------------------------------------
# What this test proves:
#   (a) Genie can run a shell command and capture its stdout.
#   (b) Unlike Test 5 (which uses redirection), this tests a different command
#       and validates Genie can handle multi-step shell logic.
#   (c) Command: `uname -s` → should produce "Linux" on this system.
#
# Goal: run `uname -s`, save output to /tmp/genie_test13.txt.
# Validator: file content is "Linux".
# -----------------------------------------------------------------------------

_TEST8_OUTPUT = "/tmp/genie_test13.txt"


def _validate_shell_command_stdout(
    result: TaskResult,
    orch: GenieOrchestrator,
) -> None:
    """Output file must contain the stdout from 'uname -s'."""

    assert result.outcome == "done", (
        f"Task did not finish with 'done' (got '{result.outcome}'). "
        f"Summary: {result.summary}"
    )

    assert os.path.exists(_TEST8_OUTPUT), (
        f"Output file not found at {_TEST8_OUTPUT}. "
        "Genie was asked to save the command output here."
    )

    with open(_TEST8_OUTPUT, "r", encoding="utf-8") as fh:
        content = fh.read().strip()

    assert content == "Linux", (
        f"File contains {content!r} but expected 'Linux'.\n"
        "The command `uname -s` on this system must produce 'Linux'."
    )

    print(f"    ✓ {_TEST8_OUTPUT} contains '{content}' — shell command stdout correct")
_TEST9_B1 = "/tmp/genie_b1.txt"
_TEST9_B2 = "/tmp/genie_b2.txt"
_TEST9_B3 = "/tmp/genie_b3.txt"
_TEST9_COMBINED = "/tmp/genie_batch_combined.txt"

# Max iterations that still prove batching occurred.
# Without batching: minimum 5 iterations (3 write_file + 1 run_command + 1 done).
# With batching: 1–2 iterations (all in one batch, or plan + one batch).
# We allow up to 3 to account for: plan call + brain loop + possible retry.
_TEST9_MAX_ITERATIONS = 3


def _validate_batch_execution(
    result: TaskResult,
    orch: GenieOrchestrator,
) -> None:
    """All files must exist with correct content, and iteration count must prove batching."""

    assert result.outcome == "done", (
        f"Task did not finish with 'done' (got '{result.outcome}'). "
        f"Summary: {result.summary}"
    )

    # -- Check all 3 source files --
    for path, expected in [
        (_TEST9_B1, "alpha"),
        (_TEST9_B2, "beta"),
        (_TEST9_B3, "gamma"),
    ]:
        assert os.path.exists(path), (
            f"File not found: {path}. "
            "Genie was asked to write this file as part of the batch."
        )
        with open(path, "r", encoding="utf-8") as fh:
            content = fh.read().strip()
        assert content == expected, (
            f"{path} contains {content!r}, expected {expected!r}."
        )

    # -- Check combined output file --
    assert os.path.exists(_TEST9_COMBINED), (
        f"Combined file not found: {_TEST9_COMBINED}. "
        "The cat command either was not run or wrote to the wrong path."
    )
    with open(_TEST9_COMBINED, "r", encoding="utf-8") as fh:
        combined = fh.read().strip()

    # cat concatenates without separators unless files end with newline.
    # Accept both "alpha\nbeta\ngamma" and "alphabetagamma".
    assert "alpha" in combined and "beta" in combined and "gamma" in combined, (
        f"Combined file content is wrong: {combined!r}. "
        "Expected all three values (alpha, beta, gamma) from the source files."
    )

    iters = result.iterations
    print(
        f"    ✓ All 3 files written correctly, combined file verified"
    )
    if iters <= _TEST9_MAX_ITERATIONS:
        print(
            f"    ✓ Batching confirmed: {iters} iteration(s) for 5+ actions "
            f"(max allowed: {_TEST9_MAX_ITERATIONS})"
        )
    else:
        print(
            f"    ⚠ No batching: {iters} iterations (batching would be ≤ {_TEST9_MAX_ITERATIONS})"
        )
    print(
        f"    ✓ Cost: ${result.cost_usd:.4f}"
    )


# -----------------------------------------------------------------------------
# Test 16 — Rung 10: Write a Python script from description, run it, confirm output
# -----------------------------------------------------------------------------
# What this test proves:
#   (a) Genie can write a non-trivial Python script from a natural language
#       description — not just copy-paste a command.
#   (b) The script runs cleanly (exit 0) and produces the correct answer.
#   (c) Unlike Test 3 (which has anti-cheat via input substitution), this test
#       accepts any correct implementation — the point is code generation, not
#       whether Genie hardcodes. (Test 3 already covers anti-hardcoding.)
#
# Goal: Write a script that finds all prime numbers below 50 and prints them
#   as a comma-separated list to /tmp/genie_test16.txt.
# Validator: runs the script independently, checks output matches expected primes.
#
# This exercises: write_file (code generation), run_command.
# Roadmap §2.1, Rung 10 variant 1.
# -----------------------------------------------------------------------------

_TEST10_SCRIPT = "/tmp/genie_test16.py"
_TEST10_OUTPUT = "/tmp/genie_test16.txt"
_TEST10_EXPECTED_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]


def _validate_code_gen_primes(
    result: TaskResult,
    orch: GenieOrchestrator,
) -> None:
    """Script must find all primes below 50 and write them to the output file."""

    assert result.outcome == "done", (
        f"Task did not finish with 'done' (got '{result.outcome}'). "
        f"Summary: {result.summary}"
    )

    assert os.path.exists(_TEST10_SCRIPT), (
        f"Script not found at {_TEST10_SCRIPT}. "
        "Genie was asked to write a prime-finding script here."
    )

    # Run the script independently to confirm it works
    proc = subprocess.run(
        [sys.executable, _TEST10_SCRIPT],
        capture_output=True, text=True, timeout=10,
    )
    assert proc.returncode == 0, (
        f"Script exited with code {proc.returncode}.\n"
        f"stderr: {proc.stderr.strip()}"
    )

    assert os.path.exists(_TEST10_OUTPUT), (
        f"Output file not found at {_TEST10_OUTPUT}. "
        "The script must write primes to this file."
    )

    with open(_TEST10_OUTPUT, "r", encoding="utf-8") as fh:
        content = fh.read().strip()

    # Extract all integers from the output (flexible format: "2, 3, 5" or "2 3 5" or one per line)
    import re
    found_numbers = sorted(int(x) for x in re.findall(r'\d+', content))

    assert found_numbers == _TEST10_EXPECTED_PRIMES, (
        f"Primes mismatch.\n"
        f"Expected: {_TEST10_EXPECTED_PRIMES}\n"
        f"Got:      {found_numbers}\n"
        f"Raw output: {content!r}"
    )

    print(
        f"    ✓ {_TEST10_SCRIPT} generated and runs clean — "
        f"found {len(found_numbers)} primes below 50"
    )
    print(f"    ✓ Cost: ${result.cost_usd:.4f}  |  Iterations: {result.iterations}")


# -----------------------------------------------------------------------------
# Test 17 — Rung 11: Fix a buggy script — run, read error, fix, re-run
# -----------------------------------------------------------------------------
# What this test proves:
#   (a) Genie can run a pre-existing script, observe it crash (non-zero exit),
#       read the traceback, diagnose the bug, fix the file, and re-run until
#       it passes.
#   (b) This is the core debug loop: run → fail → read error → fix → retry.
#       Structurally, this is a prerequisite skill for Rung 13 (dev cycle).
#   (c) The if/else conditional interpreter from §1.5 may fire here if the
#       LLM emits a batch with if/else. Or it may just take 2-3 ReAct turns.
#       Both paths are valid.
#
# Setup: validator pre-creates /tmp/genie_test17.py with a known bug:
#   - The script computes the average of [10, 20, 30, 40, 50].
#   - Bug: variable name typo (`totla` instead of `total`).
#   - Expected correct output: "Average: 30.0"
#
# Goal: "Run /tmp/genie_test17.py. It should print 'Average: 30.0' but it has
#   a bug. Find and fix the bug, then run it again until it works."
# Validator: runs the final script, checks output contains "30".
#
# This exercises: run_command (observe failure), read_file (read traceback or
#   source), write_file (fix), run_command (verify fix).
# Roadmap §2.1, Rung 11.
# -----------------------------------------------------------------------------

_TEST11_SCRIPT = "/tmp/genie_test17.py"
_TEST11_BUGGY_CODE = '''\
def compute_average(numbers):
    total = sum(numbers)
    avg = totla / len(numbers)  # bug: typo in variable name
    return avg

data = [10, 20, 30, 40, 50]
result = compute_average(data)
print(f"Average: {result}")
'''


def _setup_test11() -> None:
    """Pre-create the buggy script for test 17."""
    with open(_TEST11_SCRIPT, "w") as fh:
        fh.write(_TEST11_BUGGY_CODE)


def _validate_bugfix_loop(
    result: TaskResult,
    orch: GenieOrchestrator,
) -> None:
    """The fixed script must run cleanly and print the correct average."""

    assert result.outcome == "done", (
        f"Task did not finish with 'done' (got '{result.outcome}'). "
        f"Summary: {result.summary}"
    )

    assert os.path.exists(_TEST11_SCRIPT), (
        f"Script not found at {_TEST11_SCRIPT}. "
        "Genie should have fixed the existing script, not deleted it."
    )

    # Read the fixed source to confirm the typo was addressed
    with open(_TEST11_SCRIPT, "r", encoding="utf-8") as fh:
        source = fh.read()

    # Strip inline comments before checking — Genie sometimes adds a comment
    # like "# bug fixed: 'totla' -> 'total'" which would falsely trigger the
    # raw string check even though the code itself is correct.
    source_no_comments = "\n".join(
        line.split("#")[0] for line in source.splitlines()
    )
    assert "totla" not in source_no_comments, (
        f"The typo 'totla' is still in the script. Genie did not fix the bug.\n"
        f"Script content:\n{source}"
    )

    # Run the fixed script independently
    proc = subprocess.run(
        [sys.executable, _TEST11_SCRIPT],
        capture_output=True, text=True, timeout=10,
    )
    assert proc.returncode == 0, (
        f"Fixed script still crashes (exit code {proc.returncode}).\n"
        f"stderr: {proc.stderr.strip()}\n"
        f"Script content:\n{source}"
    )

    stdout = proc.stdout.strip()
    assert "30" in stdout, (
        f"Script output does not contain '30'.\n"
        f"stdout: {stdout!r}\n"
        f"Expected: 'Average: 30.0' (average of [10,20,30,40,50])."
    )

    print(f"    ✓ Bug fixed: 'totla' → 'total'. Script runs clean.")
    print(f"    ✓ Output: {stdout}")
    print(f"    ✓ Cost: ${result.cost_usd:.4f}  |  Iterations: {result.iterations}")


# -----------------------------------------------------------------------------
# Test 18 — Rung 12: Multi-file project (main.py imports utils.py), run, verify
# -----------------------------------------------------------------------------
# What this test proves:
#   (a) Genie can create a multi-file Python project where files depend on
#       each other via imports.
#   (b) main.py imports from utils.py — proving Genie understands Python
#       module structure and relative imports within a directory.
#   (c) The combined project runs cleanly from the command line.
#
# Goal: Create /tmp/genie_r12/utils.py with helper functions (add, multiply),
#   then /tmp/genie_r12/main.py that imports utils and prints results.
# Validator: runs main.py, checks stdout for correct computed values.
#
# This exercises: write_file ×2 (multi-file coordination), run_command.
# Roadmap §2.1, Rung 12.
# -----------------------------------------------------------------------------

_TEST12_DIR = "/tmp/genie_r12"
_TEST12_UTILS = "/tmp/genie_r12/utils.py"
_TEST12_MAIN = "/tmp/genie_r12/main.py"


def _setup_test12() -> None:
    """Clean slate — remove any leftover directory from a previous run."""
    import shutil
    if os.path.exists(_TEST12_DIR):
        shutil.rmtree(_TEST12_DIR)


def _validate_multi_file_project(
    result: TaskResult,
    orch: GenieOrchestrator,
) -> None:
    """Both files must exist, main.py must import utils.py, output must be correct."""

    assert result.outcome == "done", (
        f"Task did not finish with 'done' (got '{result.outcome}'). "
        f"Summary: {result.summary}"
    )

    assert os.path.exists(_TEST12_UTILS), (
        f"utils.py not found at {_TEST12_UTILS}. "
        "Genie was asked to create this file with helper functions."
    )

    assert os.path.exists(_TEST12_MAIN), (
        f"main.py not found at {_TEST12_MAIN}. "
        "Genie was asked to create this file that imports utils."
    )

    # Confirm main.py actually imports from utils (not self-contained)
    with open(_TEST12_MAIN, "r", encoding="utf-8") as fh:
        main_source = fh.read()

    assert "utils" in main_source, (
        f"main.py does not reference 'utils'. It must import from utils.py.\n"
        f"Content:\n{main_source}"
    )

    # Run main.py from inside the project directory
    proc = subprocess.run(
        [sys.executable, _TEST12_MAIN],
        capture_output=True, text=True, timeout=10,
        cwd=_TEST12_DIR,
    )
    assert proc.returncode == 0, (
        f"main.py exited with code {proc.returncode}.\n"
        f"stderr: {proc.stderr.strip()}\n"
        f"This likely means the import of utils.py failed or the code has errors."
    )

    stdout = proc.stdout.strip()
    # Goal asks for add(3,4)=7 and multiply(5,6)=30
    assert "7" in stdout, (
        f"Output does not contain '7' (expected from add(3, 4)).\n"
        f"stdout: {stdout!r}"
    )
    assert "30" in stdout, (
        f"Output does not contain '30' (expected from multiply(5, 6)).\n"
        f"stdout: {stdout!r}"
    )

    print(f"    ✓ Multi-file project runs: utils.py + main.py")
    print(f"    ✓ Output: {stdout}")
    print(f"    ✓ Cost: ${result.cost_usd:.4f}  |  Iterations: {result.iterations}")


# -----------------------------------------------------------------------------
# Test 19 — Rung 13: Multi-phase dev cycle (scaffold → pytest → fix → retry)
# -----------------------------------------------------------------------------
# What this test proves:
#   (a) Genie can implement a Python module from scratch given only a pre-written
#       test suite. This is the core software development loop:
#       read spec (tests) → write code → run tests → fix failures → repeat.
#   (b) The fix loop fires: pytest will fail on first run (Genie must write code
#       that satisfies 8 test methods across 6 functions, including edge cases
#       like divide-by-zero and negative factorial). Getting all 8 right on the
#       first write is unlikely — the loop is where the value is.
#   (c) Genie reads pytest output, identifies which tests failed, reads the test
#       file for expected behaviour, and updates its implementation accordingly.
#
# Setup: validator pre-creates /tmp/genie_r13/test_calc.py with 8 test methods
#   for a Calculator class (add, subtract, multiply, divide, power, factorial).
#   Genie must create /tmp/genie_r13/calc.py that makes all tests pass.
#
# Goal: "Read the tests in /tmp/genie_r13/test_calc.py. Write
#   /tmp/genie_r13/calc.py to make all tests pass. Run pytest, fix until green."
#
# Validator: runs `pytest /tmp/genie_r13/test_calc.py` independently, asserts
#   exit code 0 and all 8 tests passed.
#
# This exercises: read_file (test spec), write_file (implementation),
#   run_command (pytest), if/else or ReAct fix loop, write_file (fix).
# Roadmap §2.2, Rung 13.
# -----------------------------------------------------------------------------

_TEST13_DIR = "/tmp/genie_r13"
_TEST13_TESTS = "/tmp/genie_r13/test_calc.py"
_TEST13_IMPL = "/tmp/genie_r13/calc.py"

_TEST13_TEST_CODE = '''\
import pytest
from calc import Calculator


class TestCalculator:
    def setup_method(self):
        self.calc = Calculator()

    def test_add(self):
        assert self.calc.add(2, 3) == 5
        assert self.calc.add(-1, 1) == 0
        assert self.calc.add(0, 0) == 0

    def test_subtract(self):
        assert self.calc.subtract(5, 3) == 2
        assert self.calc.subtract(1, 5) == -4

    def test_multiply(self):
        assert self.calc.multiply(3, 4) == 12
        assert self.calc.multiply(-2, 3) == -6
        assert self.calc.multiply(0, 100) == 0

    def test_divide(self):
        assert self.calc.divide(10, 2) == 5.0
        assert self.calc.divide(7, 2) == 3.5

    def test_divide_by_zero(self):
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            self.calc.divide(1, 0)

    def test_power(self):
        assert self.calc.power(2, 3) == 8
        assert self.calc.power(5, 0) == 1
        assert self.calc.power(3, 2) == 9

    def test_factorial(self):
        assert self.calc.factorial(5) == 120
        assert self.calc.factorial(0) == 1
        assert self.calc.factorial(1) == 1

    def test_factorial_negative(self):
        with pytest.raises(ValueError, match="negative"):
            self.calc.factorial(-1)
'''


def _setup_test13() -> None:
    """Pre-create the test file. Remove any prior implementation so Genie starts fresh."""
    import shutil
    if os.path.exists(_TEST13_DIR):
        shutil.rmtree(_TEST13_DIR)
    os.makedirs(_TEST13_DIR, exist_ok=True)
    with open(_TEST13_TESTS, "w") as fh:
        fh.write(_TEST13_TEST_CODE)
    # Ensure no leftover implementation
    if os.path.exists(_TEST13_IMPL):
        os.remove(_TEST13_IMPL)


def _validate_dev_cycle(result: TaskResult, orch: GenieOrchestrator) -> None:
    """All 8 pytest tests must pass on the implementation Genie wrote."""

    assert result.outcome == "done", (
        f"Task did not finish with 'done' (got '{result.outcome}'). "
        f"Summary: {result.summary}"
    )

    assert os.path.exists(_TEST13_IMPL), (
        f"Implementation not found at {_TEST13_IMPL}. "
        "Genie was asked to create calc.py with a Calculator class."
    )

    # Read the implementation to confirm it defines Calculator
    with open(_TEST13_IMPL, "r", encoding="utf-8") as fh:
        source = fh.read()
    assert "class Calculator" in source, (
        f"calc.py does not define 'class Calculator'.\n"
        f"Content:\n{source[:500]}"
    )

    # Run pytest independently — this is the real validator
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", _TEST13_TESTS, "-v", "--tb=short"],
        capture_output=True, text=True, timeout=30,
        cwd=_TEST13_DIR,
    )

    stdout = proc.stdout
    stderr = proc.stderr

    assert proc.returncode == 0, (
        f"pytest failed (exit code {proc.returncode}).\n"
        f"Genie's implementation does not pass all tests.\n"
        f"stdout:\n{stdout}\n"
        f"stderr:\n{stderr}"
    )

    # Count passed tests — expect 8
    import re
    passed_match = re.search(r'(\d+) passed', stdout)
    passed_count = int(passed_match.group(1)) if passed_match else 0

    assert passed_count == 8, (
        f"Expected 8 tests passed, got {passed_count}.\n"
        f"stdout:\n{stdout}"
    )

    print(f"    ✓ All 8 pytest tests pass on Genie's implementation")
    print(f"    ✓ Calculator class: add, subtract, multiply, divide, power, factorial")
    print(f"    ✓ Edge cases: divide-by-zero raises ValueError, negative factorial raises ValueError")
    print(f"    ✓ Cost: ${result.cost_usd:.4f}  |  Iterations: {result.iterations}")


# -----------------------------------------------------------------------------
# Test 20 — Rung 14: Browser research + multi-source write
# -----------------------------------------------------------------------------
# What this test proves:
#   (a) Genie can open Chrome, search a topic, visit multiple source pages,
#       read content from each, and synthesise the findings into a structured
#       markdown document saved to disk.
#   (b) The output file meets a minimum word count (400 words).
#   (c) The output references at least 3 distinct sources (URLs or site names)
#       proving the LLM actually gathered information from multiple pages
#       rather than generating from its own knowledge.
#   (d) The file is valid markdown with at least one heading (#).
#
# This is the gate to the content production revenue pipeline (§2.4).
# The LLM must manage a multi-step browser research workflow:
#   open_app → search → visit source 1 → read_element → visit source 2 →
#   read_element → visit source 3 → read_element → write_file → run_command wc
#
# No tab management needed — sequential navigation in a single tab suffices.
# The LLM navigates to each source via ctrl:l → type URL → enter, reads
# content with read_element, then writes the final synthesis with write_file.
#
# Setup: cleanup of output path. No pre-seeded files.
# Validator: file exists, word count ≥ 400, ≥ 3 source references, has headings.
# Roadmap §2.3, Rung 14.
# -----------------------------------------------------------------------------

# =============================================================================
# Test 14 (new) — Rubric 24: fetch_url multi-source, FastAPI summary
# =============================================================================
# Replaces old T20 (browser multi-source research — no rubric anchor, fragile).
# Tests fetch_url across multiple sources. Rubric 24 requires fetching docs
# before writing code/summaries. No live browser needed.
# =============================================================================

_FETCH_T14_DIR    = "/tmp/genie_fetch_t14"
_FETCH_T14_OUTPUT = "/tmp/genie_fetch_t14/fastapi_summary.md"


def _setup_fetch_t14() -> None:
    import shutil
    if os.path.exists(_FETCH_T14_DIR):
        shutil.rmtree(_FETCH_T14_DIR)
    os.makedirs(_FETCH_T14_DIR, exist_ok=True)


def _validate_fetch_url_multisource(
    result: TaskResult,
    orch: GenieOrchestrator,
) -> None:
    """FastAPI summary must exist, have ≥200 words, and cover key topics."""

    assert result.outcome == "done", (
        f"Task did not finish with 'done' (got '{result.outcome}'). "
        f"Summary: {result.summary}"
    )
    assert os.path.exists(_FETCH_T14_OUTPUT), (
        f"Summary not found at {_FETCH_T14_OUTPUT}."
    )
    with open(_FETCH_T14_OUTPUT, "r", encoding="utf-8") as fh:
        content = fh.read()
    word_count = len(content.split())
    content_lower = content.lower()
    assert "fastapi" in content_lower, "Summary does not mention FastAPI."
    assert "install" in content_lower or "pip" in content_lower, (
        "Summary does not cover installation."
    )
    for artifact in ("as an ai", "i cannot", "i'm unable"):
        assert artifact not in content_lower, (
            f"LLM artifact detected: '{artifact}'"
        )
    _assert_fetch_url_in_trace("T14")
    print(f"    ✓ FastAPI summary: {word_count} words")
    print(f"    ✓ Mentions FastAPI and installation")
    print(f"    ✓ Cost: ${result.cost_usd:.4f}  |  Iterations: {result.iterations}")
_AUDIT_DIR = "/tmp/genie_audit"


def _kill_stray_vscode() -> None:
    """Kill VS Code instances that *Genie* launched, never the user's own.

    During automated tests the LLM occasionally opens VS Code via open_app.
    Genie-launched VS Code uses ``--user-data-dir=/tmp/genie_code_profile``
    (from APP_PROFILES).  We key on that path to distinguish Genie-spawned
    Code processes from the user's workspace VS Code and all its children.

    Previous approach used ``pgrep -a code`` which matched every VS Code
    child process (renderer, GPU, network service, extensions) — many of
    which don't contain "genie_sw" in their cmdline — and killed them,
    crashing the user's editor.
    """
    try:
        subprocess.run(
            ["pkill", "-9", "-f", "genie_code_profile"],
            capture_output=True, timeout=3,
        )
    except Exception:
        pass


def _snapshot_pids() -> set[int]:
    """Return the set of all currently running PIDs.

    Used to differentiate user-launched apps from Genie-launched apps
    so post-test cleanup only kills processes that Genie spawned.
    """
    pids: set[int] = set()
    try:
        out = subprocess.run(
            ["ps", "-e", "-o", "pid="],
            capture_output=True, text=True, timeout=5,
        )
        for line in out.stdout.strip().splitlines():
            line = line.strip()
            if line:
                pids.add(int(line))
    except Exception:
        pass
    return pids


# Desktop apps that GTK/GNOME fork away from the original Popen PID.
# These MUST be pkilled after tests to avoid leaking windows.
_STRAY_DESKTOP_APPS = (
    "eog", "gedit", "evince", "nautilus", "gnome-text-editor",
    "totem", "file-roller", "gnome-calculator", "gnome-calendar",
    "gnome-system-monitor", "eog-previewer", "loupe",
)


def _kill_stray_desktop_apps(pre_pids: set[int] | None = None) -> None:
    """Kill desktop apps that Genie opened (not the user's).

    If *pre_pids* is supplied, only processes whose PID is NOT in the
    snapshot are killed.  This ensures the user's eog / gedit / etc.
    windows are never touched.  If *pre_pids* is None (safety-net
    fallback), all matching processes are killed.
    """
    for app in _STRAY_DESKTOP_APPS:
        try:
            out = subprocess.run(
                ["pgrep", "-f", app],
                capture_output=True, text=True, timeout=3,
            )
            for line in out.stdout.strip().splitlines():
                pid_str = line.strip()
                if not pid_str:
                    continue
                pid = int(pid_str)
                if pre_pids is not None and pid in pre_pids:
                    continue  # user's app — leave it alone
                try:
                    os.kill(pid, 9)
                except ProcessLookupError:
                    pass
        except Exception:
            pass


def _post_test_cleanup(number: int, pre_pids: set[int] | None = None) -> None:
    """Kill stray apps that the orchestrator's own cleanup may miss.

    The orchestrator now terminates all apps it opened via the registry
    when run_task() returns (see _cleanup_opened_apps).  This function
    handles edge cases: apps the LLM launched outside the registry
    (e.g. VSCode via run_command) or apps whose process detection is
    unreliable (eog sometimes forks).
    """
    _kill_stray_vscode()
    _kill_stray_desktop_apps(pre_pids)


def _kill_chrome() -> None:
    """Kill Genie's Chrome processes and wait briefly for OS cleanup.

    Only targets Chrome instances launched with Genie's dedicated profile
    directory so that the user's personal Chrome windows are never touched.
    Called before every browser test to ensure a clean CDP state.
    """
    try:
        subprocess.run(
            ["pkill", "-9", "-f", "genie_chrome_profile"],
            capture_output=True, timeout=3,
        )
    except Exception:
        pass
    time.sleep(1.5)


# Tests that open Chrome but whose category is not 'browser' or 'multi_app'.
# fetch_url tests (T5, T14) don't open Chrome; all browser tests are already
# in category='browser'. _CHROME_EXTRA is kept for future use.
_CHROME_EXTRA: frozenset[int] = frozenset()




_T15_OUTPUT = f"{_AUDIT_DIR}/t15_wikipedia.txt"


def _validate_t15_wikipedia(result: TaskResult, orch: GenieOrchestrator) -> None:
    """First paragraph of Python article must mention 'programming language'."""
    assert result.outcome == "done", (
        f"Task did not finish with 'done' (got '{result.outcome}'). "
        f"Summary: {result.summary}"
    )
    assert os.path.exists(_T15_OUTPUT), (
        f"Output file not found at {_T15_OUTPUT}."
    )
    with open(_T15_OUTPUT, "r", encoding="utf-8") as f:
        content = f.read()
    cl = content.lower()
    assert "programming language" in cl or "python" in cl, (
        f"Missing marker 'programming language' or 'python'.\n"
        f"First 300 chars: {content[:300]!r}"
    )
    wc = len(content.split())
    _assert_fetch_url_in_trace("T15")
    print(f"    ✓ Wikipedia: {wc} words, content verified (general-purpose + high-level)")
    print(f"    ✓ Cost: ${result.cost_usd:.4f}  |  Iterations: {result.iterations}")


# --- T24: GitHub -----------------------------------------------------------

_T16_OUTPUT = f"{_AUDIT_DIR}/t16_github.txt"


def _validate_t16_github(result: TaskResult, orch: GenieOrchestrator) -> None:
    """Repo description of psf/requests must mention HTTP or requests."""
    assert result.outcome == "done", (
        f"Task did not finish with 'done' (got '{result.outcome}'). "
        f"Summary: {result.summary}"
    )
    assert os.path.exists(_T16_OUTPUT), (
        f"Output file not found at {_T16_OUTPUT}."
    )
    with open(_T16_OUTPUT, "r", encoding="utf-8") as f:
        content = f.read()
    cl = content.lower()
    assert "http" in cl or "request" in cl or "python" in cl, (
        f"Missing marker 'http', 'request', or 'python'.\n"
        f"First 300 chars: {content[:300]!r}"
    )
    assert "library" in cl or "http" in cl, (
        f"Missing 'library' or 'http' — expected in requests repo description.\n"
        f"First 300 chars: {content[:300]!r}"
    )
    wc = len(content.split())
    _assert_fetch_url_in_trace("T16")
    print(f"    ✓ GitHub: {wc} words, content verified (library/http keyword confirmed)")
    print(f"    ✓ Cost: ${result.cost_usd:.4f}  |  Iterations: {result.iterations}")


# --- T17: Python docs (yield expressions) --------------------------------
# NOTE: Stack Overflow was replaced because it Cloudflare-blocks fetch_url,
# returning a challenge page whose URL contains "yield" — causing the old
# keyword check to pass on garbage content.

_T17_OUTPUT = f"{_AUDIT_DIR}/t17_pythondocs.txt"


def _validate_t17_pythondocs(result: TaskResult, orch: GenieOrchestrator) -> None:
    """Yield docs must mention suspend/resume — keywords absent from any Cloudflare page."""
    assert result.outcome == "done", (
        f"Task did not finish with 'done' (got '{result.outcome}'). "
        f"Summary: {result.summary}"
    )
    assert os.path.exists(_T17_OUTPUT), (
        f"Output file not found at {_T17_OUTPUT}."
    )
    with open(_T17_OUTPUT, "r", encoding="utf-8") as f:
        content = f.read()
    cl = content.lower()
    # Use keywords that appear in real Python docs content but NOT in any
    # Cloudflare challenge page or in the URL slug itself.
    # 'suspend' and 'resume' are the defining words for yield semantics.
    assert "suspend" in cl or "resume" in cl or "generator function" in cl, (
        f"Content doesn't mention 'suspend', 'resume', or 'generator function'.\n"
        f"This likely means fetch_url returned a bot-challenge page, not real docs.\n"
        f"First 300 chars: {content[:300]!r}"
    )
    wc = len(content.split())
    _assert_fetch_url_in_trace("T17")
    print(f"    ✓ Python docs yield: {wc} words, content verified (suspend/resume confirmed)")
    print(f"    ✓ Cost: ${result.cost_usd:.4f}  |  Iterations: {result.iterations}")
_T18_OUTPUT = f"{_AUDIT_DIR}/t18_mdn.txt"


def _validate_t18_mdn(result: TaskResult, orch: GenieOrchestrator) -> None:
    """Array.map() description must mention map, callback, or array."""
    assert result.outcome == "done", (
        f"Task did not finish with 'done' (got '{result.outcome}'). "
        f"Summary: {result.summary}"
    )
    assert os.path.exists(_T18_OUTPUT), (
        f"Output file not found at {_T18_OUTPUT}."
    )
    with open(_T18_OUTPUT, "r", encoding="utf-8") as f:
        content = f.read()
    cl = content.lower()
    assert "map" in cl or "callback" in cl or "array" in cl, (
        f"Missing marker 'map', 'callback', or 'array'.\n"
        f"First 300 chars: {content[:300]!r}"
    )
    assert "new array" in cl or "element" in cl, (
        f"Missing 'new array' or 'element' — expected in MDN Array.map() definition.\n"
        f"First 300 chars: {content[:300]!r}"
    )
    wc = len(content.split())
    _assert_fetch_url_in_trace("T18")
    print(f"    ✓ MDN: {wc} words, content verified (new array/element confirmed)")
    print(f"    ✓ Cost: ${result.cost_usd:.4f}  |  Iterations: {result.iterations}")


# --- T28: PyPI -------------------------------------------------------------

_T19_OUTPUT = f"{_AUDIT_DIR}/t19_pypi.txt"


def _validate_t19_pypi(result: TaskResult, orch: GenieOrchestrator) -> None:
    """httpx package description must mention HTTP, client, or async."""
    assert result.outcome == "done", (
        f"Task did not finish with 'done' (got '{result.outcome}'). "
        f"Summary: {result.summary}"
    )
    assert os.path.exists(_T19_OUTPUT), (
        f"Output file not found at {_T19_OUTPUT}."
    )
    with open(_T19_OUTPUT, "r", encoding="utf-8") as f:
        content = f.read()
    cl = content.lower()
    assert "http" in cl or "httpx" in cl, (
        f"Missing 'http' or 'httpx' — expected in any httpx description.\n"
        f"First 300 chars: {content[:300]!r}"
    )
    wc = len(content.split())
    _assert_fetch_url_in_trace("T19")
    print(f"    ✓ PyPI: {wc} words, content verified (async + http confirmed)")
    print(f"    ✓ Cost: ${result.cost_usd:.4f}  |  Iterations: {result.iterations}")
_T20_OUTPUT = f"{_AUDIT_DIR}/t20_pythonorg.txt"


def _validate_t20_pythonorg(result: TaskResult, orch: GenieOrchestrator) -> None:
    """Downloads page must yield a Python version string like 3.x."""
    import re as _re
    assert result.outcome == "done", (
        f"Task did not finish with 'done' (got '{result.outcome}'). "
        f"Summary: {result.summary}"
    )
    assert os.path.exists(_T20_OUTPUT), (
        f"Output file not found at {_T20_OUTPUT}."
    )
    with open(_T20_OUTPUT, "r", encoding="utf-8") as f:
        content = f.read()
    assert _re.search(r"3\.\d+", content), (
        f"No Python version pattern (3.x) found.\n"
        f"Content: {content[:200]!r}"
    )
    _assert_fetch_url_in_trace("T20")
    print(f"    ✓ Python.org: version found — {content.strip()[:80]}")
    print(f"    ✓ Cost: ${result.cost_usd:.4f}  |  Iterations: {result.iterations}")
_T21_OUTPUT = "/tmp/genie_t33_review.txt"


def _validate_t21_review_pipeline(
    result: TaskResult, orch: GenieOrchestrator
) -> None:
    """Validate the §3.3 review pipeline end-to-end:
    1. Task completed with 'done'.
    2. Output file exists with content.
    3. Review queue entry was created with status 'pending'.
    4. Mock-approve the entry and confirm status transitions.
    """
    from review_queue import ReviewQueue, REVIEW_QUEUE_PATH

    assert result.outcome == "done", (
        f"Task did not finish with 'done' (got '{result.outcome}'). "
        f"Summary: {result.summary}"
    )

    # Check output file
    assert os.path.exists(_T21_OUTPUT), (
        f"Output file not found at {_T21_OUTPUT}."
    )
    with open(_T21_OUTPUT, "r", encoding="utf-8") as f:
        content = f.read()
    wc = len(content.split())
    assert wc >= 5, f"Content too short ({wc} words)"
    print(f"    ✓ Output file exists: {wc} words")

    # Check review queue entry was created
    queue = ReviewQueue()
    entry = queue.get_by_task_id(result.task_id)
    assert entry is not None, (
        f"No review queue entry found for task_id={result.task_id}.\n"
        f"Queue path: {REVIEW_QUEUE_PATH}"
    )
    assert entry["status"] == "pending", (
        f"Expected status 'pending', got '{entry['status']}'"
    )
    assert entry["goal"], "Review entry missing goal"
    assert entry["summary"], "Review entry missing summary"
    print(f"    ✓ Review queue entry: status=pending, task_id={result.task_id[:8]}...")

    # Mock approval: directly update queue (simulates Telegram bot action)
    ok = queue.update_status(result.task_id, "approved")
    assert ok, "Failed to update status to approved"
    ok = queue.mark_delivered(result.task_id)
    assert ok, "Failed to mark as delivered"

    # Verify final state
    final = queue.get_by_task_id(result.task_id)
    assert final is not None, "Entry disappeared after approval"
    assert final["status"] == "delivered", (
        f"Expected status 'delivered', got '{final['status']}'"
    )
    print(f"    ✓ Mock approval → delivered: status transitions verified")
    print(f"    ✓ Cost: ${result.cost_usd:.4f}  |  Iterations: {result.iterations}")


# =============================================================================
# §3.4 — Crash-Resume helpers (T34–T36)
# =============================================================================

_T22_RESUME_OUTPUT = "/tmp/genie_t22_resume.txt"
_T23_RESUME_OUTPUT = "/tmp/genie_t23_resume.txt"
_T24_RESUME_OUTPUT = "/tmp/genie_t24_resume.txt"

_CRASH_ITER_MAP = {22: 5, 23: 10, 24: 15}
_OUTPUT_MAP     = {22: _T22_RESUME_OUTPUT, 23: _T23_RESUME_OUTPUT, 24: _T24_RESUME_OUTPUT}


def _make_crash_checkpoint(crash_iter: int, goal: str, output_path: str) -> dict:
    """Write a synthetic checkpoint at *crash_iter* to CHECKPOINT_PATH.

    Simulates a process crash that left a checkpoint file on disk.
    Returns the loaded dict so run_task(checkpoint=ck) can consume it.
    """
    import uuid as _uuid
    ck = {
        "task_id": str(_uuid.uuid4()),
        "goal": goal,
        "task_type": "default",
        "per_task_budget": 0.5,
        "iteration": crash_iter,
        "sequence": crash_iter,
        "cost_usd": round(0.001 * crash_iter, 6),
        "last_observation": {
            "task_id": "",
            "sequence": crash_iter,
            "attempt": 1,
            "timestamp": "2026-03-09T00:00:00.000Z",
            "action": "write_file",
            "args": {"path": output_path, "content": "(pre-crash placeholder)"},
            "result": "success",
            "observation": {"file_exists": True, "file_size_bytes": 24},
            "error": None,
            "duration_ms": 0,
        },
    }
    tmp = CHECKPOINT_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(ck, f)
    os.rename(tmp, CHECKPOINT_PATH)
    return ck


def _run_crash_resume(
    tc: TestCase,
    orch: GenieOrchestrator,
    on_update,
) -> "TaskResult":
    """Inject synthetic checkpoint and resume the task."""
    output_path = _OUTPUT_MAP[tc.number]
    crash_iter  = _CRASH_ITER_MAP[tc.number]
    # Pre-create output file so the LLM sees evidence of prior work
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("(pre-crash placeholder — will be overwritten)\n")
    ck = _make_crash_checkpoint(crash_iter, tc.goal, output_path)
    return orch.run_task(checkpoint=ck, mode="interactive", on_update=on_update)


def _validate_crash_resume(crash_iter: int, output_path: str):
    """Factory — returns a validator closure for a crash-resume test."""
    def _validator(result: TaskResult, orch: GenieOrchestrator) -> None:
        assert result.outcome == "done", (
            f"Task did not complete after resume (got '{result.outcome}'). "
            f"Summary: {result.summary}"
        )
        assert result.iterations > crash_iter, (
            f"Expected iterations > {crash_iter} (checkpoint was loaded at that iter), "
            f"got {result.iterations}. Brain loop did not resume from checkpoint."
        )
        assert os.path.exists(output_path), (
            f"Output file missing after resume: {output_path}"
        )
        pre_crash_cost = round(0.001 * crash_iter, 6)
        assert result.cost_usd > pre_crash_cost, (
            f"cost_usd ({result.cost_usd:.6f}) not higher than pre-crash cost "
            f"({pre_crash_cost:.6f}) — brain loop may not have run."
        )
        print(f"    ✓ Resumed from iter {crash_iter} → completed at iter {result.iterations}")
        print(f"    ✓ Output file present: {output_path}")
        print(f"    ✓ Total cost (incl. pre-crash): ${result.cost_usd:.4f}")
    return _validator


# =============================================================================
# Method-check helper — asserts fetch_url was actually used
# =============================================================================

def _assert_fetch_url_in_trace(test_label: str) -> None:
    """Fail the test if fetch_url was never called (catches Chrome-bypass passes)."""
    assert "fetch_url" in _current_action_trace, (
        f"{test_label}: Expected fetch_url in action trace but found only: "
        f"{list(set(_current_action_trace))}"
    )


# =============================================================================
# T25 — Autonomous fetch_url: requests library version (no URL in goal)
# =============================================================================
_T25_OUTPUT = "/tmp/genie_t25_autonomous.txt"


def _validate_t25_autonomous(result: TaskResult, orch: GenieOrchestrator) -> None:
    import re as _re, json as _json
    assert result.outcome == "done", f"Expected 'done', got '{result.outcome}'"
    assert os.path.exists(_T25_OUTPUT), f"Output not found: {_T25_OUTPUT}"
    with open(_T25_OUTPUT, "r", encoding="utf-8") as f:
        content = f.read()
    # If genie wrote raw PyPI JSON API response, extract version from it.
    version_str = content.strip()
    if version_str.startswith("{"):
        try:
            data = _json.loads(version_str)
            version_str = data.get("info", {}).get("version", "") or version_str
        except _json.JSONDecodeError:
            pass  # Fall through to regex search on raw content
    assert _re.search(r"\d+\.\d+\.\d+", version_str), (
        f"No semver pattern found.\nContent: {content[:200]!r}"
    )
    _assert_fetch_url_in_trace("T25")
    print(f"    ✓ requests version: {version_str.strip()[:40]}")
    print(f"    ✓ fetch_url used autonomously (no URL in goal)")
    print(f"    ✓ Cost: ${result.cost_usd:.4f}  |  Iterations: {result.iterations}")


# =============================================================================
# T26 — Autonomous fetch_url: Rust's design purpose (no URL in goal)
# =============================================================================
_T26_OUTPUT = "/tmp/genie_t26_autonomous.txt"


def _validate_t26_autonomous(result: TaskResult, orch: GenieOrchestrator) -> None:
    assert result.outcome == "done", f"Expected 'done', got '{result.outcome}'"
    assert os.path.exists(_T26_OUTPUT), f"Output not found: {_T26_OUTPUT}"
    with open(_T26_OUTPUT, "r", encoding="utf-8") as f:
        content = f.read()
    cl = content.lower()
    wc = len(content.split())
    assert wc >= 8, f"Too short ({wc} words, need >= 8)"
    assert any(w in cl for w in ("memory", "safety", "safe", "c++", "systems", "system")), (
        f"Missing expected keywords about Rust.\nContent: {content[:200]!r}"
    )
    _assert_fetch_url_in_trace("T26")
    print(f"    ✓ Rust answer: {wc} words, content verified")
    print(f"    ✓ fetch_url used autonomously (no URL in goal)")
    print(f"    ✓ Cost: ${result.cost_usd:.4f}  |  Iterations: {result.iterations}")


# =============================================================================
# Setup hook registry
# Maps test number → zero-argument callable invoked before the test runs.
# genie_phase4.py imports this dict and registers Phase 4 hooks into it.
# =============================================================================

_SETUP_HOOKS: dict[int, "Callable[[], None]"] = {
    1:  _setup_t1,
    2:  _setup_t2,
    5:  _setup_fetch_t5,
    7:  _setup_test7,
    11: _setup_test11,
    12: _setup_test12,
    13: _setup_test13,
    14: _setup_fetch_t14,
}


# =============================================================================
# Test registry
# Append new TestCase entries above, or modify existing entries.
# Each entry's number is its permanent identity in regression reports.
# =============================================================================

TESTS: list[TestCase] = [


    TestCase(
        number=1,
        category="browser",
        description="Rubric 31 — Deliver HTML with button/heading, open in Chrome, verify elements",
        goal=(
            "Create the directory /tmp/genie_t1/ if it does not exist. "
            "Write an HTML file to /tmp/genie_t1/index.html with:\n"
            "  - A <title> containing 'Genie Test'\n"
            "  - A heading with id='main-heading' containing text 'Hello from Genie'\n"
            "  - A button with id='submit-btn' containing text 'Submit'\n"
            "Open the file in Chrome by navigating to file:///tmp/genie_t1/index.html. "
            "After navigating, wait 2 seconds for the page to load, then tell me when done."
        ),
        validator=_validate_cdp_own_frontend,
        expected_outcome="done",
    ),

    TestCase(
        number=2,
        category="browser",
        description="Rubric 31 — Deliver HTML with page structure (nav/main/footer), open in Chrome",
        goal=(
            "Create the directory /tmp/genie_t2/ if it does not exist. "
            "Write an HTML file to /tmp/genie_t2/index.html with:\n"
            "  - A <nav> element containing two links: '#home' and '#about'\n"
            "  - A <main> element with id='content' containing a paragraph\n"
            "  - A <footer> element with copyright text\n"
            "Open the file in Chrome by navigating to file:///tmp/genie_t2/index.html. "
            "After navigating, wait 2 seconds for the page to load, then tell me when done."
        ),
        validator=_validate_cdp_page_structure,
        expected_outcome="done",
    ),

    # ── FILE OPS CATEGORY ────────────────────────────────────────────────────

    TestCase(
        number=3,
        category="file_ops",
        description="Two-file workflow: write input data + line-counting script, run and verify",
        goal=(
            "Do the following two things using files on disk:\n"
            "1. Write a plain text file to /tmp/genie_input.txt containing exactly "
            "5 lines of text (any content is fine, one item per line).\n"
            "2. Write a Python script to /tmp/genie_test3.py that opens "
            "/tmp/genie_input.txt, counts the number of lines, and prints ONLY "
            "that count as a plain integer (nothing else).\n"
            "Then run /tmp/genie_test3.py with Python to confirm it prints 5. "
            "Tell me when done."
        ),
        validator=_validate_script_written_and_runnable,
        expected_outcome="done",
    ),

    # ── CONTENT CATEGORY ─────────────────────────────────────────────────────
    # Validates: batching (1.1) — cost and iteration count under load.
    # The_40.md target: research + writing pipeline. No browser needed.
    # Genie must write to disk; validator reads actual file content.

    TestCase(
        number=4,
        category="shell",
        description="Run shell command, save stdout to file, verify content",
        goal=(
            "Run the following shell command: python3 -c \"print(6 * 7)\"\n"
            "Save the output (stdout) of that command to /tmp/genie_test5.txt.\n"
            "The file should contain exactly: 42\n"
            "Tell me when done."
        ),
        validator=_validate_run_command_stdout,
        expected_outcome="done",
    ),

    # ── RUNG 1–5 + 7–9 BACKFILL (1.3) ────────────────────────────────────────
    # Dedicated test cases for rungs that were implicitly covered by Tests 1–5
    # but had no dedicated validators. Gaps hid regressions.

    # ── Rung 1: Launch app ───────────────────────────────────────────────────

    TestCase(
        number=5,
        category="fetch_url",
        description="Rubric 24 — fetch_url PyPI docs for httpx, write demo script",
        goal=(
            "Fetch the documentation for the httpx Python library using fetch_url "
            "on https://pypi.org/project/httpx/. "
            "Then write a Python script to /tmp/genie_fetch_t5/httpx_demo.py that:\n"
            "  - Imports httpx\n"
            "  - Makes a synchronous GET request to \'https://httpbin.org/get\'\n"
            "  - Prints the response status code\n"
            "Tell me when done."
        ),
        validator=_validate_fetch_url_and_code,
        expected_outcome="done",
    ),
    TestCase(
        number=6,
        category="file_ops",
        description="Rung 7 — Write a structured JSON file to disk",
        goal=(
            "Write a JSON file to /tmp/genie_test11.json with the following content:\n"
            '{\n'
            '  "name": "genie",\n'
            '  "version": "1.0",\n'
            '  "language": "python"\n'
            '}\n'
            "Make sure it's valid JSON. Tell me when done."
        ),
        validator=_validate_write_file,
        expected_outcome="done",
    ),

    # ── Rung 8: Read file from disk ──────────────────────────────────────────

    TestCase(
        number=7,
        category="file_ops",
        description="Rung 8 — Read a file from disk and report its word count",
        goal=(
            "The file /tmp/genie_test12_input.txt already exists with some text "
            "content — do NOT create, overwrite, or modify it. "
            "Count the words in that existing file using the shell "
            "command `wc -w < /tmp/genie_test12_input.txt` (this gives accurate "
            "results). Write just the resulting number (strip any whitespace) "
            "to /tmp/genie_test12_output.txt. Tell me when done."
        ),
        validator=_validate_read_file,
        expected_outcome="done",
    ),

    # ── Rung 9: Run shell command, check stdout ──────────────────────────────

    TestCase(
        number=8,
        category="shell",
        description="Rung 9 — Run 'uname -s' and save stdout to file",
        goal=(
            "Run the shell command: uname -s\n"
            "Save the output to /tmp/genie_test13.txt. "
            "The file should contain exactly the output of that command.\n"
            "Tell me when done."
        ),
        validator=_validate_shell_command_stdout,
        expected_outcome="done",
    ),

    # ── Rung: Unknown app launch (1.4) ────────────────────────────────────────

    TestCase(
        number=9,
        category="shell",
        description="§1.5 — Multi-action batch: write 3 files + combine via shell, prove batching",
        goal=(
            "Do all of the following in one go — these are all simple deterministic steps:\n"
            "1. Write /tmp/genie_b1.txt containing exactly: alpha\n"
            "2. Write /tmp/genie_b2.txt containing exactly: beta\n"
            "3. Write /tmp/genie_b3.txt containing exactly: gamma\n"
            "4. Run this shell command: cat /tmp/genie_b1.txt /tmp/genie_b2.txt /tmp/genie_b3.txt > /tmp/genie_batch_combined.txt\n"
            "Tell me when done."
        ),
        validator=_validate_batch_execution,
        expected_outcome="done",
    ),

    # ── DEV CYCLE CATEGORY (planned) ─────────────────────────────────────────
    # The_40.md target: "Fix the failing tests in this repo."
    # Validator: run tests, check exit code == 0 after Genie finishes.

    # ── CODE GENERATION CATEGORY (§2.1, Rungs 10–12) ─────────────────────────

    # ── Rung 10: Write script from description, run, verify output ────────────

    TestCase(
        number=10,
        category="dev_cycle",
        description="Rung 10 — Write a prime-finding script, run it, verify output",
        goal=(
            "Write a Python script to /tmp/genie_test16.py that:\n"
            "1. Finds all prime numbers below 50\n"
            "2. Writes them as a comma-separated list to /tmp/genie_test16.txt\n"
            "   (e.g. '2, 3, 5, 7, ...')\n"
            "3. Run the script to confirm it works.\n"
            "Tell me when done."
        ),
        validator=_validate_code_gen_primes,
        expected_outcome="done",
    ),

    # ── Rung 11: Fix buggy script — debug loop ──────────────────────────────

    TestCase(
        number=11,
        category="dev_cycle",
        description="Rung 11 — Fix a buggy script: run, read error, fix, re-run until passing",
        goal=(
            "There is a Python script at /tmp/genie_test17.py. "
            "It is supposed to compute the average of [10, 20, 30, 40, 50] "
            "and print 'Average: 30.0'. But it has a bug.\n"
            "1. Run the script and observe the error.\n"
            "2. Read the script to find the bug.\n"
            "3. Fix the bug.\n"
            "4. Run the script again to confirm it prints the correct output.\n"
            "Tell me when it works correctly."
        ),
        validator=_validate_bugfix_loop,
        expected_outcome="done",
    ),

    # ── Rung 12: Multi-file project — import across files ────────────────────

    TestCase(
        number=12,
        category="dev_cycle",
        description="Rung 12 — Multi-file project: main.py imports utils.py, run and verify",
        goal=(
            "Create a small two-file Python project in /tmp/genie_r12/:\n"
            "1. Write /tmp/genie_r12/utils.py with two functions:\n"
            "   - add(a, b) that returns a + b\n"
            "   - multiply(a, b) that returns a * b\n"
            "2. Write /tmp/genie_r12/main.py that:\n"
            "   - Imports add and multiply from utils\n"
            "   - Prints the result of add(3, 4)\n"
            "   - Prints the result of multiply(5, 6)\n"
            "3. Run main.py and confirm the output is correct.\n"
            "Tell me when done."
        ),
        validator=_validate_multi_file_project,
        expected_outcome="done",
    ),

    # ── Rung 13: Multi-phase dev cycle ────────────────────────────────────────

    TestCase(
        number=13,
        category="dev_cycle",
        description="Rung 13 — Scaffold code to pass pre-written pytest suite, fix until green",
        goal=(
            "There is a pytest test file at /tmp/genie_r13/test_calc.py. "
            "It tests a Calculator class with 8 test methods covering: "
            "add, subtract, multiply, divide, power, and factorial.\n"
            "Your job:\n"
            "1. Read /tmp/genie_r13/test_calc.py to understand what's expected.\n"
            "2. Write /tmp/genie_r13/calc.py with a Calculator class that "
            "implements all the required methods.\n"
            "3. Run: python3 -m pytest /tmp/genie_r13/test_calc.py -v\n"
            "4. If any tests fail, read the output, fix calc.py, and run again.\n"
            "5. Repeat until ALL 8 tests pass (pytest exit code 0).\n"
            "Tell me when all tests are green."
        ),
        validator=_validate_dev_cycle,
        expected_outcome="done",
    ),

    # ── Rung 14: Browser research + multi-source write ────────────────────────

    TestCase(
        number=14,
        category="fetch_url",
        description="Rubric 24 — fetch_url multi-source: fetch FastAPI docs, write summary",
        goal=(
            "Research the FastAPI framework by fetching its documentation.\n"
            "Use fetch_url on at least two of these sources:\n"
            "  - https://pypi.org/project/fastapi/\n"
            "  - https://fastapi.tiangolo.com/\n"
            "Then write a structured markdown summary to "
            "/tmp/genie_fetch_t14/fastapi_summary.md covering:\n"
            "  - What FastAPI is\n"
            "  - Key features (auto docs, async support, type hints)\n"
            "  - Installation command\n"
            "The summary must be at least 200 words with markdown headings. "
            "Tell me when done."
        ),
        validator=_validate_fetch_url_multisource,
        expected_outcome="done",
    ),
    TestCase(
        number=15,
        category="fetch_url",
        description="§3.2 — fetch_url Wikipedia: first paragraph of Python article",
        goal=(
            "Use fetch_url to read the Wikipedia page at "
            "https://en.wikipedia.org/wiki/Python_(programming_language). "
            "Extract the first paragraph of the article — the introductory text "
            "that appears before the table of contents. "
            "Save whatever you find there to "
            f"{_T15_OUTPUT}. Tell me when done."
        ),
        validator=_validate_t15_wikipedia,
        expected_outcome="done",
    ),

    TestCase(
        number=16,
        category="fetch_url",
        description="§3.2 — fetch_url GitHub: repo description of psf/requests",
        goal=(
            "Use fetch_url to read https://github.com/psf/requests. "
            "The psf/requests repository is a Python HTTP library. "
            "Find and save the short description of what requests does — "
            "look for the About blurb or the opening line of the README "
            "that describes the library (something like 'HTTP for Humans'). "
            f"Save that description to {_T16_OUTPUT}. Tell me when done."
        ),
        validator=_validate_t16_github,
        expected_outcome="done",
    ),

    TestCase(
        number=17,
        category="fetch_url",
        description="§3.2 — fetch_url Python docs: yield expression semantics",
        goal=(
            "Use fetch_url to read "
            "https://docs.python.org/3/reference/expressions.html#yield-expressions. "
            "Save a description of what the yield expression does (how it suspends and "
            "resumes a generator function), taken from the fetched page, "
            f"to {_T17_OUTPUT}. Tell me when done."
        ),
        validator=_validate_t17_pythondocs,
        expected_outcome="done",
    ),

    TestCase(
        number=18,
        category="fetch_url",
        description="§3.2 — fetch_url MDN: Array.map() method description",
        goal=(
            "Use fetch_url to read "
            "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/map. "
            "Find the paragraph that describes what the Array map() method does — "
            "it should explain that map() creates a new array by calling a provided "
            "function on every element of the calling array. "
            "Do NOT save navigation menus, sidebars, or unrelated page chrome. "
            f"Save only that description paragraph to {_T18_OUTPUT}. Tell me when done."
        ),
        validator=_validate_t18_mdn,
        expected_outcome="done",
    ),

    TestCase(
        number=19,
        category="fetch_url",
        description="§3.2 — fetch_url PyPI: httpx features and capabilities",
        goal=(
            "Use fetch_url to read https://pypi.org/project/httpx/. "
            "Extract the key features and capabilities of httpx described on the page "
            "(e.g. async support, HTTP/2, connection pooling — NOT just the one-line tagline). "
            f"Save the description to {_T19_OUTPUT}. Tell me when done."
        ),
        validator=_validate_t19_pypi,
        expected_outcome="done",
    ),

    TestCase(
        number=20,
        category="fetch_url",
        description="§3.2 — fetch_url Python.org: latest stable version",
        goal=(
            "Use fetch_url to read https://www.python.org/downloads/. "
            "Find the latest stable Python release version number. "
            f"Save just the version string (e.g. 'Python 3.12.1') to {_T20_OUTPUT}. "
            "Tell me when done."
        ),
        validator=_validate_t20_pythonorg,
        expected_outcome="done",
    ),

    TestCase(
        number=21,
        category="review_pipeline",
        description="§3.3 — Review queue: task completes → entry appears in review queue → mock approve → delivered",
        goal=(
            "Write a short 3-sentence summary about the Python programming language "
            "and save it to /tmp/genie_t33_review.txt. Tell me when done."
        ),
        validator=_validate_t21_review_pipeline,
        expected_outcome="done",
    ),

    # ── CRASH-RESUME CATEGORY ─────────────────────────────────────────────────
    # §3.4 — Error Recovery Hardening.
    # Each test injects a synthetic CHECKPOINT_PATH entry at a specific
    # iteration (simulating a mid-task process crash), then calls
    # run_task(checkpoint=ck) and verifies the task resumes and reaches 'done'.
    # Validates: checkpoint loading, iteration counter restoration, brain loop
    # continuation, and output production after resume.

    TestCase(
        number=22,
        category="crash_resume",
        description="§3.4 — Crash-resume: inject checkpoint at iter 5, resume → done",
        goal=(
            "Write a one-sentence note about Python's Global Interpreter Lock "
            "and save it to /tmp/genie_t22_resume.txt. "
            "Tell me when done."
        ),
        validator=_validate_crash_resume(5, _T22_RESUME_OUTPUT),
        expected_outcome="done",
    ),

    TestCase(
        number=23,
        category="crash_resume",
        description="§3.4 — Crash-resume: inject checkpoint at iter 10, resume → done",
        goal=(
            "Write a one-sentence note about Python's async/await concurrency model "
            "and save it to /tmp/genie_t23_resume.txt. "
            "Tell me when done."
        ),
        validator=_validate_crash_resume(10, _T23_RESUME_OUTPUT),
        expected_outcome="done",
    ),

    TestCase(
        number=24,
        category="crash_resume",
        description="§3.4 — Crash-resume: inject checkpoint at iter 15, resume → done",
        goal=(
            "Write a one-sentence note about FastAPI's performance advantages "
            "and save it to /tmp/genie_t24_resume.txt. "
            "Tell me when done."
        ),
        validator=_validate_crash_resume(15, _T24_RESUME_OUTPUT),
        expected_outcome="done",
    ),



    # ── FETCH_URL AUTONOMOUS CATEGORY ─────────────────────────────────
    # No URL in goal — LLM must construct the URL and use fetch_url.
    TestCase(
        number=25,
        category="fetch_url_autonomous",
        description="Autonomous fetch_url: find requests library version (no URL given)",
        goal=(
            "Find the current stable release version of the `requests` Python library. "
            "Save just the version string (e.g. '2.31.0') to /tmp/genie_t25_autonomous.txt. "
            "Tell me when done."
        ),
        validator=_validate_t25_autonomous,
        expected_outcome="done",
    ),
    TestCase(
        number=26,
        category="fetch_url_autonomous",
        description="Autonomous fetch_url: find what problem Rust was created to solve",
        goal=(
            "Find what programming language Rust was originally designed to replace "
            "(or what problem it was created to solve). "
            "Save a one-sentence answer to /tmp/genie_t26_autonomous.txt. "
            "Tell me when done."
        ),
        validator=_validate_t26_autonomous,
        expected_outcome="done",
    ),

    # ── Add new TestCase entries ABOVE this line ──────────────────────────────
]


# =============================================================================
# Runner
# =============================================================================

class GenieTester:
    """
    Runs TestCase entries against the live Genie stack.

    Design notes:
    - One shared GenieOrchestrator instance across all tests. This mirrors
      real usage (genie.py keeps one orchestrator alive for the session) and
      catches state-leak bugs between tasks.
    - skip_plan=False for all tests. The full plan phase + y/n gate runs.
      This tests the plan LLM call, plan schema validation, AND the brain loop.
      Fast-path (sequence execution) is ON by default within run_task() for
      trivial goals — but that routing decision is made by orchestrator, not
      forced here. Tests with "click" or "show me" in the goal are NOT trivial
      and will use the full plan + brain loop path.
    - Validators run AFTER TaskResult is returned. They read real OS state.
      A passed validator means Genie actually did the thing, not just claimed it.
    """

    def __init__(self, auto_yes: bool = False, no_log: bool = False) -> None:
        controller = XdotoolController()
        self.orch = GenieOrchestrator(controller)
        self._auto_yes = auto_yes
        self._pass: list[int] = []
        self._fail: list[int] = []
        self._skip: list[int] = []
        if no_log:
            _devnull = open(os.devnull, "a", encoding="utf-8")
            self.orch.observer._incomplete_fh = _devnull
            self.orch.observer._success_fh = _devnull
            self.orch.element_resolver._no_vision_log = True

    # ------------------------------------------------------------------

    def run_case(self, tc: TestCase) -> bool:
        sep = "─" * 60
        print(f"\n{sep}")
        print(f"TEST {tc.number} [{tc.category.upper()}]: {tc.description}")

        if tc.skip:
            print(f"  SKIP: {tc.skip}")
            print(sep)
            self._skip.append(tc.number)
            return True  # counts as pass for scoring

        print(f"GOAL: {tc.goal}")
        print(sep)

        # Snapshot all running PIDs BEFORE the test starts so that
        # post-test cleanup only kills processes Genie spawned.
        _pre_test_pids = _snapshot_pids()

        # Pre-test setup hooks
        # Kill Chrome + clear registry before any test that opens a browser,
        # so each test starts with a clean window state.
        # Always start with a clean registry so stale entries from a
        # previous test never leak (e.g. an eog window from test 14
        # confusing test 15's shell-only run).
        self.orch.registry.clear()
        self.orch._last_registry_snapshot.clear()
        _current_action_trace.clear()

        _uses_chrome = (
            tc.category in ("browser", "multi_app") or tc.number in _CHROME_EXTRA
        )
        if _uses_chrome:
            _kill_chrome()
            # Clear cached state from previous tests so validators don't
            # see stale URLs or registry entries from an earlier run.
            self.orch._last_cdp_urls.clear()

        if tc.number in _SETUP_HOOKS:
            _SETUP_HOOKS[tc.number]()
        elif 15 <= tc.number <= 20:
            os.makedirs(_AUDIT_DIR, exist_ok=True)

        result: TaskResult | None = None
        try:
            if tc.runner is not None:
                # Custom runner (e.g. T38 clarify test) — full control over
                # mode, on_update, and CLARIFY_ENABLED restoration.
                result = tc.runner(self)
            elif tc.number in (22, 23, 24):
                # §3.4 — Crash-resume: inject checkpoint, resume from mid-task state
                result = _run_crash_resume(tc, self.orch, self._on_update)
            else:
                # Browser / multi-app tests need adaptive brain loop (not the
                # brittle sequence path) because element interaction on live
                # web pages is non-deterministic and requires seeing intermediate
                # results.  Non-browser tests use the fast sequence path.
                _is_browser = (
                    tc.category in ("browser", "multi_app")
                    or tc.number in _CHROME_EXTRA
                )
                result = self.orch.run_task(
                    goal=tc.goal,
                    mode="interactive",
                    task_type="auto",
                    on_update=self._on_update,
                    skip_plan=self._auto_yes and not _is_browser,
                    auto_confirm=self._auto_yes,
                )
        except Exception as exc:
            self._record_fail(tc.number, f"run_task raised unexpectedly: {exc}")
            traceback.print_exc()
            _post_test_cleanup(tc.number, _pre_test_pids)
            return False

        _post_test_cleanup(tc.number, _pre_test_pids)
        print(f"\n  Outcome   : {result.outcome}")
        print(f"  Iterations: {result.iterations}")
        print(f"  Cost      : ${result.cost_usd:.4f}")
        print(f"  Summary   : {result.summary}")

        # Pre-validator outcome check
        if tc.expected_outcome is not None and result.outcome != tc.expected_outcome:
            self._record_fail(
                tc.number,
                f"Expected outcome='{tc.expected_outcome}', got '{result.outcome}'. "
                f"Summary: {result.summary}",
            )
            return False

        # Real-state validator
        try:
            tc.validator(result, self.orch)
        except AssertionError as ae:
            self._record_fail(tc.number, str(ae))
            return False
        except Exception as exc:
            self._record_fail(tc.number, f"Validator raised unexpected error: {exc}")
            traceback.print_exc()
            return False

        self._record_pass(tc.number)
        return True

    # ------------------------------------------------------------------

    def run_all(self, stop_on_first_failure: bool = True) -> None:
        for tc in TESTS:
            ok = self.run_case(tc)
            if not ok and stop_on_first_failure:
                print(
                    f"\n✗ Stopped after first failure (test {tc.number}). "
                    "Fix it — then verify ALL prior tests still pass before moving on."
                )
                break
        # Final cleanup: kill any stale Genie apps left by the last test.
        _kill_chrome()
        _kill_stray_desktop_apps()  # no pre_pids → aggressive, but only at end of full suite
        _kill_stray_vscode()
        self._print_summary()

    def run_one(self, number: int) -> None:
        matches = [tc for tc in TESTS if tc.number == number]
        if not matches:
            nums = [tc.number for tc in TESTS]
            print(f"ERROR: No test with number {number}. Available: {nums}")
            sys.exit(1)
        self.run_case(matches[0])
        self._print_summary()

    def run_many(
        self,
        numbers: list[int],
        stop_on_first_failure: bool = True,
    ) -> None:
        """Run an explicit ordered list of test numbers."""
        test_map = {tc.number: tc for tc in TESTS}
        missing = [n for n in numbers if n not in test_map]
        if missing:
            available = sorted(test_map)
            print(f"ERROR: Unknown test numbers: {missing}. Available: {available}")
            sys.exit(1)
        for n in numbers:
            ok = self.run_case(test_map[n])
            if not ok and stop_on_first_failure:
                print(
                    f"\n✗ Stopped after first failure (test {n}). "
                    "Fix it — then verify all selected tests still pass."
                )
                break
        _kill_chrome()
        _kill_stray_desktop_apps()
        _kill_stray_vscode()
        self._print_summary()

    def run_one_repeated(self, number: int, times: int) -> None:
        """Run a single test *times* times, reporting pass/fail per run."""
        matches = [tc for tc in TESTS if tc.number == number]
        if not matches:
            nums = [tc.number for tc in TESTS]
            print(f"ERROR: No test with number {number}. Available: {nums}")
            sys.exit(1)
        self._run_repeated([matches[0].number], times, label=f"test {number}")

    def run_selection_repeated(self, numbers: list[int], times: int) -> None:
        """Run a selection of tests *times* times, reporting pass/fail per run."""
        test_map = {tc.number: tc for tc in TESTS}
        missing = [n for n in numbers if n not in test_map]
        if missing:
            print(f"ERROR: Unknown test numbers: {missing}.")
            sys.exit(1)
        label = f"tests {numbers}"
        self._run_repeated(numbers, times, label=label)

    def _run_repeated(self, numbers: list[int], times: int, label: str) -> None:
        """Core repeat loop used by both run_one_repeated and run_selection_repeated."""
        test_map = {tc.number: tc for tc in TESTS}
        # per-run: list of (passed_count, failed_list)
        run_summaries: list[tuple[int, list[int]]] = []
        print(f"\nRepeating {label} × {times}")
        print("═" * 60)
        for i in range(1, times + 1):
            print(f"\n── Run {i}/{times} ──────────────────────────────────────")
            self._pass.clear()
            self._fail.clear()
            self._skip.clear()
            for n in numbers:
                self.run_case(test_map[n])
            run_summaries.append((len(self._pass), list(self._fail)))
        _kill_chrome()
        _kill_stray_desktop_apps()
        _kill_stray_vscode()
        # Summary
        total_per_run = len(numbers)
        perfect_runs = sum(1 for p, f in run_summaries if not f)
        print("\n" + "═" * 60)
        print(f"REPEAT RESULTS: {perfect_runs}/{times} fully-passing runs  ({label})")
        w = len(str(times))
        for i, (p, f) in enumerate(run_summaries, 1):
            mark = "✔" if not f else "✘"
            fail_str = f"  FAIL: {f}" if f else ""
            print(f"  Run {i:>{w}}: {mark}  {p}/{total_per_run} passed{fail_str}")
        print("═" * 60)
        if self._auto_yes:
            status = f"{perfect_runs}/{times} fully-passing runs"
            try:
                _tg_send(f"Genie repeat {label}: {status}")
            except Exception:  # noqa: BLE001
                pass
        if perfect_runs < times:
            sys.exit(1)

    # ------------------------------------------------------------------

    @staticmethod
    def _on_update(event: dict) -> None:
        import datetime as _dt

        action  = event.get("action")
        args    = event.get("args") or {}
        res     = event.get("result")
        obs     = event.get("observation") or {}
        outcome = event.get("outcome")
        message = event.get("message")
        cost    = event.get("task_cost_usd")
        n       = event.get("iteration")

        ts = _dt.datetime.now().strftime("%H:%M:%S")
        if action:
            _current_action_trace.append(action)

        # ── Subtask boundary (no action) ──────────────────────────────
        if outcome and not action:
            print(f"  [{ts}] outcome={outcome}" +
                  (f"  {message[:100]}" if message else ""),
                  flush=True)
            return
        if message and not action:
            print(f"  [{ts}] [genie]  {message}", flush=True)
            return
        if not action:
            return

        # ── Action detail ─────────────────────────────────────────────
        detail = ""
        if action == "run_command":
            detail = f" cmd={str(args.get('cmd', ''))[:120]}"
        elif action in ("write_file", "append_file"):
            content = str(args.get("content", ""))
            detail = f" path={args.get('path', '?')} size={len(content)}"
        elif action == "read_file":
            detail = f" path={args.get('path', '?')}"
        elif action == "delete_file":
            detail = f" path={args.get('path', '?')}"
        elif action == "list_dir":
            detail = f" path={args.get('path', '?')}"
        elif action == "done":
            detail = f" summary={str(args.get('summary', ''))[:80]}"
        elif action == "checkpoint":
            note = (args.get('note') or args.get('message') or
                    args.get('summary') or args.get('text') or '')
            detail = f" note={str(note)[:100]}" if note else ""
        elif action == "fetch_url":
            detail = f" url={str(args.get('url', ''))[:80]}"
        elif action == "abort":
            detail = f" reason={str(args.get('reason', ''))[:80]}"

        # ── Result / observation detail ───────────────────────────────
        res_lines = []
        exit_code = obs.get("exit_code") if isinstance(obs, dict) else None
        if exit_code is None and isinstance(res, dict):
            exit_code = res.get("exit_code")
        if exit_code is not None:
            res_lines.append(f"exit={exit_code}")
            stdout = str(obs.get("stdout", "") if isinstance(obs, dict) else "").strip()
            stderr = str(obs.get("stderr", "") if isinstance(obs, dict) else "").strip()
            if stdout:
                res_lines.append(f"\n            stdout: {stdout[:300]}")
            if stderr:
                res_lines.append(f"\n            stderr: {stderr[:200]}")
        elif isinstance(obs, dict):
            # Blocked action hints
            for hint_key in ("done_blocked", "abort_blocked", "run_blocked",
                             "action_blocked", "action_hint", "checkpoint_hint"):
                if hint_key in obs:
                    res_lines.append(f"\n            {hint_key}: {str(obs[hint_key])[:200]}")
                    break
            # read_file content length
            if "content" in obs:
                res_lines.append(f"content_len={len(str(obs['content']))}")

        res_str = ("  " + "  ".join(res_lines)) if res_lines else ""

        # ── Color coding ─────────────────────────────────────────────
        RED    = "\033[91m"
        GREEN  = "\033[92m"
        YELLOW = "\033[93m"
        CYAN   = "\033[96m"
        BOLD   = "\033[1m"
        RESET  = "\033[0m"

        color = ""
        if "failure" in str(res) or "blocked" in str(res):
            color = RED
        elif action == "done" or outcome == "done":
            color = GREEN + BOLD
        elif action in ("write_file", "append_file"):
            color = CYAN
        elif action == "run_command":
            color = YELLOW

        cost_str = f"  ${cost:.4f}" if cost is not None else ""
        iter_str = f"{n}" if n is not None else "-"
        result_label = str(res) if res is not None else ""

        print(
            f"  [{ts}] [iter {iter_str}] "
            f"{color}{action} → {result_label}{detail}{res_str}{cost_str}{RESET}",
            flush=True,
        )

    def _record_pass(self, number: int) -> None:
        self._pass.append(number)
        print(f"\n✔ PASS: Test {number}")

    def _record_fail(self, number: int, reason: str) -> None:
        self._fail.append(number)
        print(f"\n✘ FAIL: Test {number}")
        print(f"  Reason: {reason}")

    def _print_summary(self) -> None:
        total = len(self._pass) + len(self._fail)
        print("\n" + "═" * 60)
        print(f"RESULTS: {len(self._pass)}/{total} passed")
        if self._pass:
            print(f"  PASS: {self._pass}")
        if self._skip:
            print(f"  SKIP: {self._skip}")
        if self._fail:
            print(f"  FAIL: {self._fail}")
            print()
            print("  REGRESSION RULE: A fix for a failing test MUST NOT")
            print("  cause any previously passing test to fail.")
            print("  If it does: revert and find a different fix.")
        print("═" * 60)

        if self._auto_yes:
            status = "ALL PASS" if not self._fail else f"{len(self._fail)} FAILED"
            lines = [f"Genie test run complete: {len(self._pass)}/{total} passed ({status})"]
            if self._fail:
                lines.append(f"FAIL: {self._fail}")
            if self._skip:
                lines.append(f"SKIP: {self._skip}")
            try:
                _tg_send("\n".join(lines))
            except Exception as exc:  # noqa: BLE001
                print(f"[tg] notification failed: {exc}")

        if self._fail:
            sys.exit(1)


# =============================================================================
# Entry point
# =============================================================================

def _parse_test_specs(tokens: list[str]) -> list[int]:
    """Parse a list of tokens like ['1', '14-20', '24', '26'] into a
    sorted, deduplicated list of test numbers.

    Each token is either:
      - A plain integer  : "5"    → [5]
      - A range N-M      : "14-20" → [14, 15, 16, 17, 18, 19, 20]
    """
    numbers: list[int] = []
    for token in tokens:
        if "-" in token:
            parts = token.split("-", 1)
            try:
                lo, hi = int(parts[0]), int(parts[1])
            except ValueError:
                print(f"ERROR: Cannot parse range token '{token}'. Use N-M format.")
                sys.exit(1)
            if lo > hi:
                print(f"ERROR: Range {token} is backwards. Use lo-hi order.")
                sys.exit(1)
            numbers.extend(range(lo, hi + 1))
        else:
            try:
                numbers.append(int(token))
            except ValueError:
                print(f"ERROR: Cannot parse test number '{token}'.")
                sys.exit(1)
    # Preserve order but deduplicate, keeping first occurrence
    seen: set[int] = set()
    result: list[int] = []
    for n in numbers:
        if n not in seen:
            seen.add(n)
            result.append(n)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Genie regression test suite. "
            "Runs the full Genie stack against real desktop state. "
            "See module docstring for authoring guide."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python genie_suite.py                                    # run all\n"
            "  python genie_suite.py --all                              # run all, no early stop\n"
            "  python genie_suite.py --case 5                           # run test 5\n"
            "  python genie_suite.py --case 26 --repeat 5               # test 26 x 5\n"
            "  python genie_suite.py --cases 15-20                      # tests 15-20\n"
            "  python genie_suite.py --cases 1 14-20 24 26              # mixed selection\n"
            "  python genie_suite.py --cases 1 14-20 24 26 --repeat 5   # mixed x 5\n"
            "  python genie_suite.py --all --repeat 5                   # all tests x 5\n"
        ),
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all tests even after a failure (default: stop on first failure)",
    )
    parser.add_argument(
        "--case", type=int, metavar="N",
        help="Run only test N",
    )
    parser.add_argument(
        "--repeat", type=int, metavar="K", default=1,
        help="Repeat the selected test(s) K times. Works with --case, --cases, and --all.",
    )
    parser.add_argument(
        "--cases", nargs="+", metavar="SPEC",
        help=(
            "Run a selection of tests. Each SPEC is a number (e.g. 5) or a range "
            "(e.g. 15-20). Multiple specs are space-separated: --cases 1 14-20 24 26"
        ),
    )
    parser.add_argument(
        "--yes", "-y", action="store_true",
        help="Skip plan approval prompts (skip_plan=True for all tests). "
             "Use for unattended batch runs.",
    )
    parser.add_argument(
        "--nolog", action="store_true",
        help="Suppress all JSONL trace writes. Useful when iterating on tests "
             "without polluting genie_success.jsonl / genie_incomplete.jsonl.",
    )
    args = parser.parse_args()

    # Validate mutual exclusivity
    if args.cases and args.case:
        parser.error("--cases and --case are mutually exclusive.")

    tester = GenieTester(auto_yes=args.yes, no_log=args.nolog)

    if args.case and args.repeat > 1:
        tester.run_one_repeated(args.case, args.repeat)
    elif args.case:
        tester.run_one(args.case)
    elif args.cases and args.repeat > 1:
        numbers = _parse_test_specs(args.cases)
        tester.run_selection_repeated(numbers, args.repeat)
    elif args.cases:
        numbers = _parse_test_specs(args.cases)
        tester.run_many(numbers, stop_on_first_failure=False)
    elif args.all and args.repeat > 1:
        numbers = [tc.number for tc in TESTS]
        tester.run_selection_repeated(numbers, args.repeat)
    else:
        tester.run_all(stop_on_first_failure=False)

