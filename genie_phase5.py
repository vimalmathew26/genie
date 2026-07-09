"""
genie_phase5.py — Phase 5 test suite for Genie. (Delivery Automation: T36+)

Imports the T1–T35 registry from genie_phase4 and appends Phase 5 tests.
Running this file executes the full unified suite (T1–T37+).

Phase 5 scope (roadmap.md §5.1–§5.5):
  §5.1 — PR automation (open_pr action, verify via GitHub API, Telegram notify)
  §5.2 — Client handoff package (assemble_handoff action, secrets scan, Telegram)
  §5.3 — GoalTracker / task decomposition (multi-part tasks, replan on failure)
  §5.4 — Specialist model routing (subtask-type → model selection)
  §5.5 — Clarifying questions (structured Q&A before task start)

Authoring rules (same as genie_suite.py):
  1. REAL STACK ONLY. No mocks, no stubs.
  2. REAL STATE VALIDATORS. Check disk/API/Telegram state — never trust TaskResult.summary.
  3. Test numbers continue from 36. No gaps.
  4. Regression contract: a fix for T(N) must not break T(1)..T(N-1).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import time
import zipfile

import genie_suite
from config import WORKSPACE_DIR
from genie_phase4 import (
    TESTS as _BASE_TESTS,
    _SETUP_HOOKS,
)
from genie_suite import (
    TestCase,
    GenieTester,
    _parse_test_specs,
)
from orchestrator import GenieOrchestrator
from genie import TaskResult
from planner import GoalTracker


# =============================================================================
# Phase 5 validators and test data
# =============================================================================


# ---------------------------------------------------------------------------
# §5.1 — PR Automation
# ---------------------------------------------------------------------------

# ── T36 — push branch to GitHub, open PR via open_pr action, verify via API ─

# Requires GITHUB_TEST_REPO=owner/repo and GITHUB_TOKEN in .env.
# Test is auto-skipped when these vars are absent.
_T36_GITHUB_REPO  = os.environ.get("GITHUB_TEST_REPO", "")
_T36_GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
_T36_DIR          = "/tmp/genie_t36"
_T36_BRANCH       = "genie-pr-test"        # fixed name — setup cleans up previous runs
_T36_PR_TITLE     = "Genie Phase 5.1 test PR"


def _t36_api(method: str, path: str, data: dict | None = None) -> object:
    """Minimal GitHub REST API helper for T36 setup/cleanup.

    Returns the parsed JSON body (list or dict) on success, or {} on any
    error.  Not used in the brain loop — test-harness only.
    """
    import json as _json
    import urllib.request as _ur

    token = _T36_GITHUB_TOKEN
    url   = f"https://api.github.com{path}"
    body  = _json.dumps(data).encode("utf-8") if data is not None else None
    req   = _ur.Request(
        url,
        data=body,
        headers={
            "Authorization":        f"Bearer {token}",
            "Accept":               "application/vnd.github+json",
            "Content-Type":         "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        method=method,
    )
    try:
        with _ur.urlopen(req, timeout=15) as resp:
            raw = resp.read()
            return _json.loads(raw) if raw else {}
    except Exception:
        return {}


def _setup_t36() -> None:
    repo   = _T36_GITHUB_REPO
    token  = _T36_GITHUB_TOKEN
    branch = _T36_BRANCH

    if not repo or not token:
        raise RuntimeError(
            "T36 requires GITHUB_TEST_REPO and GITHUB_TOKEN env vars. "
            "Add them to .env and rerun."
        )

    owner = repo.split("/")[0]

    # ── Cleanup from previous run: close any open PRs for this branch ────────
    prs_raw = _t36_api("GET", f"/repos/{repo}/pulls?state=open&head={owner}:{branch}")
    prs = prs_raw if isinstance(prs_raw, list) else []
    for pr in prs:
        pr_num = pr.get("number")
        if pr_num:
            _t36_api("PATCH", f"/repos/{repo}/pulls/{pr_num}", {"state": "closed"})

    # ── Cleanup from previous run: delete remote branch if it exists ─────────
    _t36_api("DELETE", f"/repos/{repo}/git/refs/heads/{branch}")

    # ── Clone repo and push a fresh feature branch with a unique commit ───────
    if os.path.exists(_T36_DIR):
        shutil.rmtree(_T36_DIR)
    os.makedirs(_T36_DIR, exist_ok=True)

    auth_url = f"https://{token}@github.com/{repo}.git"
    subprocess.run(
        ["git", "clone", "--depth=1", auth_url, _T36_DIR],
        check=True, capture_output=True, text=True,
    )
    subprocess.run(
        ["git", "-C", _T36_DIR, "config", "user.email", "genie@local"],
        check=True, capture_output=True, text=True,
    )
    subprocess.run(
        ["git", "-C", _T36_DIR, "config", "user.name", "Genie"],
        check=True, capture_output=True, text=True,
    )
    subprocess.run(
        ["git", "-C", _T36_DIR, "checkout", "-b", branch],
        check=True, capture_output=True, text=True,
    )
    ts        = int(time.time())
    test_file = os.path.join(_T36_DIR, f"genie_test_{ts}.txt")
    with open(test_file, "w") as f:
        f.write(f"Genie Phase 5.1 test commit. ts={ts}\n")
    subprocess.run(
        ["git", "-C", _T36_DIR, "add", "."],
        check=True, capture_output=True, text=True,
    )
    subprocess.run(
        ["git", "-C", _T36_DIR, "commit", "-m", f"genie phase 5.1 test [{ts}]"],
        check=True, capture_output=True, text=True,
    )
    subprocess.run(
        ["git", "-C", _T36_DIR, "push", "origin", branch],
        check=True, capture_output=True, text=True,
    )


def _validate_t36(result: TaskResult, orch: GenieOrchestrator) -> None:
    import json as _json
    import urllib.request as _ur

    repo   = _T36_GITHUB_REPO
    branch = _T36_BRANCH
    token  = _T36_GITHUB_TOKEN
    owner  = repo.split("/")[0]

    # Verify PR exists via GitHub API
    url = (
        f"https://api.github.com/repos/{repo}/pulls"
        f"?state=open&head={owner}:{branch}"
    )
    req = _ur.Request(
        url,
        headers={
            "Authorization":        f"Bearer {token}",
            "Accept":               "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with _ur.urlopen(req, timeout=15) as resp:
        prs = _json.loads(resp.read().decode("utf-8"))

    assert prs, (
        f"No open PR found for branch {branch!r} in repo {repo!r}. "
        "Genie may not have called open_pr."
    )
    pr = prs[0]
    assert _T36_PR_TITLE.lower() in pr["title"].lower(), (
        f"PR title mismatch. Expected to contain {_T36_PR_TITLE!r}, "
        f"got: {pr['title']!r}"
    )

    pr_url    = pr["html_url"]
    pr_number = pr["number"]
    print(f"    ✓ PR #{pr_number} exists: {pr_url}")

    # Cleanup: close PR and delete remote branch (best-effort)
    _t36_api("PATCH", f"/repos/{repo}/pulls/{pr_number}", {"state": "closed"})
    print(f"    ✓ PR #{pr_number} closed (cleanup)")
    _t36_api("DELETE", f"/repos/{repo}/git/refs/heads/{branch}")
    print(f"    ✓ Remote branch {branch!r} deleted (cleanup)")

    print(f"    ✓ Cost: ${result.cost_usd:.4f}  |  Iterations: {result.iterations}")


# ---------------------------------------------------------------------------
# §5.2 — Handoff package
# ---------------------------------------------------------------------------

_T37_REPO_DIR  = os.path.join(WORKSPACE_DIR, "handoff_test_repo")
_T37_OUT_DIR   = os.path.join(WORKSPACE_DIR, "handoff_test_repo_handoff")
_T37_SUMMARY   = "Built the hello-world demo application."
_T37_ENDPOINT  = "https://api.example.com/v1"


def _setup_t37() -> None:
    """Scaffold a small repo with clean files plus intentional exclusion targets."""
    for d in (_T37_REPO_DIR, _T37_OUT_DIR):
        if os.path.exists(d):
            shutil.rmtree(d)
    os.makedirs(_T37_REPO_DIR, exist_ok=True)

    with open(os.path.join(_T37_REPO_DIR, "main.py"), "w") as fh:
        fh.write("print('hello world')\n")
    with open(os.path.join(_T37_REPO_DIR, "requirements.txt"), "w") as fh:
        fh.write("requests==2.31.0\n")
    # .env with a fake secret — must be excluded by HANDOFF_EXCLUDE_PATTERNS
    with open(os.path.join(_T37_REPO_DIR, ".env"), "w") as fh:
        fh.write("DB_PASSWORD=hunter2\n")
    # __pycache__ bytecode — must be excluded
    pycache = os.path.join(_T37_REPO_DIR, "__pycache__")
    os.makedirs(pycache, exist_ok=True)
    with open(os.path.join(pycache, "cache.pyc"), "wb") as fh:
        fh.write(b"\x00\x01\x02\x03")


def _validate_t37(result: TaskResult, orch: GenieOrchestrator) -> None:
    zip_path        = os.path.join(_T37_OUT_DIR, "handoff.zip")
    handoff_md_path = os.path.join(_T37_OUT_DIR, "HANDOFF.md")

    assert os.path.exists(zip_path), (
        f"handoff.zip not found at {zip_path}. "
        "Genie may not have called assemble_handoff."
    )
    assert os.path.getsize(zip_path) > 0, "handoff.zip is empty"
    print(f"    ✓ handoff.zip exists ({os.path.getsize(zip_path)} bytes)")

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()

        # Exclusions
        assert ".env" not in names, f".env leaked into zip: {names}"
        pycache_entries = [n for n in names if "__pycache__" in n or n.endswith(".pyc")]
        assert not pycache_entries, f"pycache leaked into zip: {pycache_entries}"
        print("    ✓ .env and __pycache__ excluded from zip")

        # Required inclusions
        assert "main.py" in names, f"main.py missing from zip: {names}"
        assert "requirements.txt" in names, f"requirements.txt missing from zip: {names}"
        print("    ✓ main.py and requirements.txt present in zip")

        # HANDOFF.md in zip
        assert "HANDOFF.md" in names, f"HANDOFF.md missing from zip: {names}"
        content = zf.read("HANDOFF.md").decode("utf-8")
        assert _T37_SUMMARY in content, (
            f"Task summary not found in HANDOFF.md.\nExpected: {_T37_SUMMARY!r}\nGot:\n{content}"
        )
        assert _T37_ENDPOINT in content, (
            f"Endpoint not found in HANDOFF.md.\nExpected: {_T37_ENDPOINT!r}\nGot:\n{content}"
        )
        print("    ✓ HANDOFF.md present in zip with task summary and endpoint")

    # Standalone HANDOFF.md
    assert os.path.exists(handoff_md_path), (
        f"Standalone HANDOFF.md not found at {handoff_md_path}"
    )
    print("    ✓ Standalone HANDOFF.md written to output_dir")

    print(f"    ✓ Cost: ${result.cost_usd:.4f}  |  Iterations: {result.iterations}")

    # Cleanup
    for d in (_T37_REPO_DIR, _T37_OUT_DIR):
        if os.path.exists(d):
            shutil.rmtree(d)
    print("    ✓ Test repo and output dir cleaned up")


# ---------------------------------------------------------------------------
# §5.5 — Clarifying questions (T38)
# ---------------------------------------------------------------------------
# Goal: prove that clarify() generates ≥2 questions, blocks until answered,
# and injects answers into orch._goal before plan_phase runs.
#
# Strategy: two sub-runs with the same underspecified goal.
#   Sub-run A — "UI-style immediate answer": on_update fires and calls
#               answer_clarification() inline (same thread, before .wait()).
#               Verifies that a pre-set answer is returned instantly.
#   Sub-run B — "Telegram-style async answer": on_update spawns a daemon
#               thread with a 50 ms delay before calling answer_clarification().
#               Verifies that _clarify_event.wait() actually blocks and unblocks.
# Both sub-runs must:
#   (a) capture ≥2 questions
#   (b) produce a orch._goal string containing "Clarifications:"
#   (c) complete with outcome == "done"

_T38_GOAL = (
    "Write a Python script that fetches a URL and prints its page title. "
    "Save the script to /tmp/genie_t38_scraper.py."
)

def _setup_t38() -> None:
    import config as _c
    _c.CLARIFY_ENABLED = True
    # Remove stale output from previous runs
    scraper = "/tmp/genie_t38_scraper.py"
    if os.path.exists(scraper):
        os.remove(scraper)
    print("    ✓ T38 setup: CLARIFY_ENABLED=True, stale output cleared")


def _teardown_t38() -> None:
    import config as _c
    _c.CLARIFY_ENABLED = False
    print("    ✓ T38 teardown: CLARIFY_ENABLED=False")


def _run_t38(tester: "GenieTester") -> "TaskResult":
    """Custom runner for T38.

    Executes two sub-runs:
      A) on_update answers immediately (same-thread, pre-wait path)
      B) on_update spawns a thread with 50 ms delay (async path)

    Both sub-runs must pass for T38 to pass.
    The captured questions/answers from the LAST sub-run are stored on the
    tester as ``tester._t38_captured`` for the validator.
    """
    import threading as _threading
    import config as _c

    # T38 needs the plan-phase path (clarify is only in the else branch):
    # mode="batch" skips the interactive input() prompt,
    # skip_plan=False + auto_confirm=False enters the else branch.

    def _make_on_update(captured: list, delayed: bool):
        """Return a custom on_update that auto-answers clarification questions."""
        def _cb(event: dict) -> None:
            if event.get("clarification_question"):
                question = event.get("question", "")
                options  = event.get("options", [])
                # Select answer by semantic matching on question content.
                # Positional indexing is unreliable — Gemini generates
                # questions in non-deterministic order, so index 0 may be
                # "which library?" one run and "which URL?" the next.
                q_lower = question.lower()
                if any(kw in q_lower for kw in ("url", "website", "address", "endpoint")):
                    answer = "http://example.com"
                elif any(kw in q_lower for kw in ("library", "http", "request", "fetch")):
                    answer = "requests"
                elif any(kw in q_lower for kw in ("parse", "html", "extract", "title")):
                    answer = "html.parser"
                elif any(kw in q_lower for kw in ("error", "fail", "handle", "missing")):
                    answer = "print error message and exit"
                elif options:
                    answer = options[0]
                else:
                    answer = "default"
                captured.append({"question": question, "answer": answer})
                if delayed:
                    def _delayed_answer():
                        import time as _t
                        _t.sleep(0.05)
                        tester.orch.answer_clarification(answer)
                    _threading.Thread(target=_delayed_answer, daemon=True).start()
                else:
                    # Inline call — safe because .wait() hasn't been entered yet
                    tester.orch.answer_clarification(answer)
            # Forward normal events to the standard logger
            tester._on_update(event)
        return _cb

    results = []
    for sub, delayed in enumerate([False, True], start=1):
        captured: list = []
        label = "Telegram-style async" if delayed else "UI-style immediate"
        print(f"\n  [T38] Sub-run {sub}/2: {label}")

        # Reset goal state between sub-runs (setup_hook already set flag)
        _c.CLARIFY_ENABLED = True
        result = tester.orch.run_task(
            goal=_T38_GOAL,
            mode="batch",          # skips interactive plan approval
            task_type="default",
            on_update=_make_on_update(captured, delayed),
            skip_plan=False,
            auto_confirm=False,
        )
        print(f"  [T38] Sub-run {sub} outcome={result.outcome} "
              f"questions={len(captured)}")

        # Per-sub-run assertions
        assert len(captured) >= 2, (
            f"T38 sub-run {sub} ({label}): clarify() generated only "
            f"{len(captured)} question(s); expected ≥2."
        )
        for i, qa in enumerate(captured):
            assert qa["answer"], (
                f"T38 sub-run {sub}: question {i} has empty answer."
            )
        assert result.outcome == "done", (
            f"T38 sub-run {sub} ({label}): outcome={result.outcome!r}, "
            f"summary={result.summary!r}"
        )
        assert "Clarifications:" in tester.orch._goal, (
            f"T38 sub-run {sub}: 'Clarifications:' not found in orch._goal."
        )
        results.append((result, captured))

    # Validator picks up the last sub-run result + all captured Q/A pairs
    all_captured = results[0][1] + results[1][1]
    tester._t38_results  = results
    tester._t38_captured = all_captured
    _teardown_t38()
    # Return the last sub-run TaskResult for the standard outcome check
    return results[-1][0]


def _validate_t38(result: "TaskResult", orch: GenieOrchestrator) -> None:
    """Validate T38: clarify() generated ≥2 questions per sub-run, answers
    injected into orch._goal, and task completed successfully in both sub-runs."""
    # We stored results on the tester; assert via the captured questions
    # (passed through orch for compatibility — real validation is in _run_t38)

    # Validate last sub-run outcome
    assert result.outcome == "done", (
        f"T38: last sub-run did not complete: outcome={result.outcome!r}, "
        f"summary={result.summary!r}"
    )

    # Validate answers were injected into orch._goal
    assert "Clarifications:" in orch._goal, (
        f"T38: clarification answers not injected into orch._goal.\n"
        f"orch._goal tail:\n{orch._goal[-500:]}"
    )

    # Validate the scraper output file exists (task actually completed)
    scraper = "/tmp/genie_t38_scraper.py"
    assert os.path.exists(scraper), (
        f"T38: expected output file {scraper!r} not found — task did not run to completion"
    )
    assert os.path.getsize(scraper) > 0, (
        f"T38: output file {scraper!r} is empty"
    )
    print(f"    ✓ Output file exists: {scraper} ({os.path.getsize(scraper)} bytes)")
    print(f"    ✓ orch._goal contains 'Clarifications:'")
    print(f"    ✓ Task outcome: {result.outcome}")


# ---------------------------------------------------------------------------
# §5.3 — GoalTracker (Task Decomposition)
# ---------------------------------------------------------------------------

# ── T39 — multi-part task completes fully via GoalTracker ──────────────────

_T39_DIR = "/tmp/genie_t39"
_T39_GOAL = (
    "Write a Python module at /tmp/genie_t39/mathlib.py with an add() and "
    "multiply() function, write pytest tests at /tmp/genie_t39/test_mathlib.py "
    "for both functions, run pytest on /tmp/genie_t39/test_mathlib.py and "
    "verify all tests pass"
)


def _setup_t39() -> None:
    """Clean up any leftovers from prior T39 runs."""
    if os.path.exists(_T39_DIR):
        shutil.rmtree(_T39_DIR)
    os.makedirs(_T39_DIR, exist_ok=True)


def _run_t39(tester: GenieTester) -> "TaskResult":
    """Custom runner for T39 — uses use_goaltracker=True."""
    import config as _cfg
    _cfg.CLARIFY_ENABLED = False
    try:
        result = tester.orch.run_task(
            goal=_T39_GOAL,
            mode="batch",
            task_type="default",
            on_update=tester._on_update,
            skip_plan=False,
            auto_confirm=True,
            use_goaltracker=True,
        )
    finally:
        _cfg.CLARIFY_ENABLED = True
    return result


def _validate_t39(result: "TaskResult", orch: GenieOrchestrator) -> None:
    """Validate T39: multi-part task completed via GoalTracker."""
    # Check GoalTracker was used and all subtasks completed
    tracker = orch._goaltracker
    assert tracker is not None, "T39: GoalTracker was not set on orchestrator"
    assert isinstance(tracker, GoalTracker), (
        f"T39: _goaltracker is {type(tracker)}, expected GoalTracker"
    )
    assert tracker.all_done(), (
        f"T39: not all subtasks done. Statuses: "
        f"{[(s.n, s.status) for s in tracker.subtasks]}"
    )
    assert len(tracker.subtasks) >= 2, (
        f"T39: expected ≥2 subtasks, got {len(tracker.subtasks)}"
    )

    # Verify the actual output files exist
    mathlib = os.path.join(_T39_DIR, "mathlib.py")
    test_file = os.path.join(_T39_DIR, "test_mathlib.py")
    assert os.path.exists(mathlib), f"T39: {mathlib} not found"
    assert os.path.exists(test_file), f"T39: {test_file} not found"
    assert os.path.getsize(mathlib) > 0, f"T39: {mathlib} is empty"
    assert os.path.getsize(test_file) > 0, f"T39: {test_file} is empty"

    # Actually run pytest to confirm tests pass
    proc = subprocess.run(
        ["python3", "-m", "pytest", test_file, "-v", "--tb=short"],
        capture_output=True, text=True, cwd=_T39_DIR, timeout=30,
    )
    assert proc.returncode == 0, (
        f"T39: pytest failed (rc={proc.returncode}).\n"
        f"stdout:\n{proc.stdout[-500:]}\n"
        f"stderr:\n{proc.stderr[-500:]}"
    )

    # Verify mathlib has add() and multiply()
    with open(mathlib) as f:
        src = f.read()
    assert "def add" in src, "T39: add() not found in mathlib.py"
    assert "def multiply" in src, "T39: multiply() not found in mathlib.py"

    # Verify original goal was restored
    assert orch._goal == orch._original_goal, (
        f"T39: _goal not restored to _original_goal after GoalTracker loop"
    )

    print(f"    ✓ GoalTracker: {len(tracker.subtasks)} subtasks, all done")
    print(f"    ✓ mathlib.py exists with add() and multiply()")
    print(f"    ✓ test_mathlib.py exists, pytest passes")
    print(f"    ✓ Task outcome: {result.outcome}")


# ── T40 — subtask failure triggers replanning and recovery ─────────────────

_T40_DIR = "/tmp/genie_t40"


def _setup_t40() -> None:
    """Plant a buggy module + test suite that will fail on first run."""
    if os.path.exists(_T40_DIR):
        shutil.rmtree(_T40_DIR)
    os.makedirs(_T40_DIR, exist_ok=True)

    # Buggy module — multiply() has an off-by-one error
    buggy_module = (
        "def add(a, b):\n"
        "    return a + b\n"
        "\n"
        "def multiply(a, b):\n"
        "    return a * b + 1  # BUG: off-by-one\n"
    )
    with open(os.path.join(_T40_DIR, "mathlib.py"), "w") as f:
        f.write(buggy_module)

    # Test suite that will catch the bug
    test_code = (
        "from mathlib import add, multiply\n"
        "\n"
        "def test_add():\n"
        "    assert add(2, 3) == 5\n"
        "    assert add(0, 0) == 0\n"
        "    assert add(-1, 1) == 0\n"
        "\n"
        "def test_multiply():\n"
        "    assert multiply(2, 3) == 6\n"
        "    assert multiply(0, 5) == 0\n"
        "    assert multiply(-2, 3) == -6\n"
    )
    with open(os.path.join(_T40_DIR, "test_mathlib.py"), "w") as f:
        f.write(test_code)


_T40_GOAL = (
    "Run the existing pytest suite in /tmp/genie_t40/ and report the results. "
    "Do not modify any files in this first step — only run tests and report. "
    "If tests fail, fix the bug in /tmp/genie_t40/mathlib.py, then run pytest "
    "again until all tests pass."
)


def _run_t40(tester: GenieTester) -> "TaskResult":
    """Custom runner for T40 — uses use_goaltracker=True."""
    import config as _cfg
    _cfg.CLARIFY_ENABLED = False
    try:
        result = tester.orch.run_task(
            goal=_T40_GOAL,
            mode="batch",
            task_type="default",
            on_update=tester._on_update,
            skip_plan=False,
            auto_confirm=True,
            use_goaltracker=True,
        )
    finally:
        _cfg.CLARIFY_ENABLED = True
    return result


def _validate_t40(result: "TaskResult", orch: GenieOrchestrator) -> None:
    """Validate T40: buggy module fixed and pytest passes via GoalTracker.

    We do not assert replan_count — a capable agent legitimately solves
    this without triggering a replan, and that is correct behaviour.
    What we validate is the real output state: bug gone, tests green.
    """
    tracker = orch._goaltracker
    assert tracker is not None, "T40: GoalTracker was not set on orchestrator"
    assert isinstance(tracker, GoalTracker), (
        f"T40: _goaltracker is {type(tracker)}, expected GoalTracker"
    )

    # Final outcome should be done — the bug was fixed and tests pass
    assert result.outcome == "done", (
        f"T40: expected outcome 'done', got {result.outcome!r}. "
        f"Summary: {result.summary!r}"
    )
    assert tracker.all_done(), (
        f"T40: not all subtasks done. Statuses: "
        f"{[(s.n, s.status) for s in tracker.subtasks]}"
    )

    # Verify the bug was actually fixed — run pytest independently
    test_file = os.path.join(_T40_DIR, "test_mathlib.py")
    proc = subprocess.run(
        ["python3", "-m", "pytest", test_file, "-v", "--tb=short"],
        capture_output=True, text=True, cwd=_T40_DIR, timeout=30,
    )
    assert proc.returncode == 0, (
        f"T40: pytest still fails after task completion (rc={proc.returncode}).\n"
        f"stdout:\n{proc.stdout[-500:]}\n"
        f"stderr:\n{proc.stderr[-500:]}"
    )

    # Verify the off-by-one bug is gone from the source
    with open(os.path.join(_T40_DIR, "mathlib.py")) as f:
        src = f.read()
    assert "a * b + 1" not in src, (
        f"T40: off-by-one bug still present in mathlib.py"
    )

    # Verify original goal was restored
    assert orch._goal == orch._original_goal, (
        f"T40: _goal not restored to _original_goal after GoalTracker loop"
    )

    print(f"    ✓ GoalTracker: {len(tracker.subtasks)} subtasks, all done")
    print(f"    ✓ replan_count = {getattr(tracker, 'replan_count', 'n/a')} (informational)")
    print(f"    ✓ pytest passes, off-by-one bug gone")
    print(f"    ✓ Task outcome: {result.outcome}")


# =============================================================================
# PHASE5_TESTS list and _SETUP_HOOKS registration
# =============================================================================

PHASE5_TESTS: list[TestCase] = [
    TestCase(
        number=36,
        category="pr_automation",
        description="§5.1 — push branch to GitHub, open PR via open_pr action, verify via API",
        goal=(
            f"Open a pull request on GitHub repository "
            f"{_T36_GITHUB_REPO or '<GITHUB_TEST_REPO>'}.\n"
            f"Head branch: {_T36_BRANCH}.\n"
            "Base branch: main.\n"
            f"Title: '{_T36_PR_TITLE}'.\n"
            "Body: 'Automated test PR opened by Genie agent (Phase 5.1).'.\n"
            "Use the open_pr action to create the pull request.\n"
            "Tell me the PR URL when done."
        ),
        validator=_validate_t36,
        expected_outcome="done",
        skip=(
            None
            if _T36_GITHUB_REPO
            else "GITHUB_TEST_REPO env var not set — add it to .env to enable T36"
        ),
    ),
    TestCase(
        number=37,
        category="handoff_package",
        description="§5.2 — assemble handoff zip for a pre-scaffolded repo, verify contents and exclusions",
        goal=(
            f"Create a client handoff package for the project at {_T37_REPO_DIR}.\n"
            f"Use the assemble_handoff action with:\n"
            f"  repo_path: {_T37_REPO_DIR}\n"
            f"  task_summary: '{_T37_SUMMARY}'\n"
            f"  output_dir: {_T37_OUT_DIR}\n"
            f"  endpoints: ['{_T37_ENDPOINT}']\n"
            "Tell me the path to the generated handoff.zip when done."
        ),
        validator=_validate_t37,
        expected_outcome="done",
    ),
    TestCase(
        number=38,
        category="clarify",
        description="§5.5 — clarifying questions: ≥2 questions generated, answers injected, task completes",
        goal=_T38_GOAL,
        validator=_validate_t38,
        expected_outcome="done",
        runner=_run_t38,
    ),
    TestCase(
        number=39,
        category="goaltracker",
        description="§5.3 — GoalTracker: multi-part task decomposes and completes fully",
        goal=_T39_GOAL,
        validator=_validate_t39,
        expected_outcome="done",
        runner=_run_t39,
    ),
    TestCase(
        number=40,
        category="goaltracker",
        description="§5.3 — GoalTracker: buggy module detected, fixed, pytest green",
        goal=_T40_GOAL,
        validator=_validate_t40,
        expected_outcome="done",
        runner=_run_t40,
    ),
]

_SETUP_HOOKS.update({
    36: _setup_t36,
    37: _setup_t37,
    38: _setup_t38,
    39: _setup_t39,
    40: _setup_t40,
})

TESTS: list[TestCase] = _BASE_TESTS + PHASE5_TESTS

# Patch genie_suite.TESTS so GenieTester's run_one / run_all / run_many
# (which reference genie_suite.TESTS directly) see all Phase 5 entries.
genie_suite.TESTS = TESTS


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Genie unified regression suite (T1–T35 base + Phase 5). "
            "Runs the full Genie stack against real desktop state."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python genie_phase5.py                                   # run all\n"
            "  python genie_phase5.py --all                             # run all, no early stop\n"
            "  python genie_phase5.py --case 36                         # run T36 only\n"
            "  python genie_phase5.py --case 36 --repeat 5              # T36 x 5\n"
            "  python genie_phase5.py --case 37                         # run T37 only\n"
            "  python genie_phase5.py --case 38                         # run T38 only\n"
            "  python genie_phase5.py --case 39                         # run T39 only\n"
            "  python genie_phase5.py --case 40                         # run T40 only\n"
            "  python genie_phase5.py --cases 36-40                     # Phase 5 §5.1-§5.5\n"
            "  python genie_phase5.py --cases 27-40                     # Phase 4 + 5 tests\n"
            "  python genie_phase5.py --cases 39-40                     # GoalTracker tests\n"
            "  python genie_phase5.py --cases 38 --repeat 5             # T38 x 5\n"
            "  python genie_phase5.py --all --repeat 5                  # all tests x 5\n"
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
            "Run a selection of tests. Each SPEC is a number (e.g. 36) or a range "
            "(e.g. 27-36). Multiple specs are space-separated: --cases 1 14-20 36"
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
        tester.run_all(stop_on_first_failure=not args.all)
