"""
genie_phase4.py — Phase 4 test suite for Genie. (SW Developer Capabilities: T27–T35)

Imports the T1–T26 base registry from genie_suite and appends Phase 4 tests.
Running this file executes the full unified suite (T1–T35).

Phase 4 scope (roadmap.md §4.1–§4.4):
  §4.1 — Git operations (clone, branch, commit, diff, status)
  §4.2 — Codebase navigation (grep, read, summarise file/function)
  §4.3 — Issue → fix cycle (read issue, locate code, patch, verify)
  §4.4 — Dependency management (pip install, requirements.txt, lockfile)

Authoring rules (same as genie_suite.py):
  1. REAL STACK ONLY. No mocks, no stubs.
  2. REAL STATE VALIDATORS. Check disk/process/git state — never trust TaskResult.summary.
  3. Test numbers continue from 27. No gaps.
  4. Regression contract: a fix for T(N) must not break T(1)..T(N-1).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import textwrap

import genie_suite
from genie_suite import (
    TESTS as _BASE_TESTS,
    TestCase,
    GenieTester,
    _parse_test_specs,
    _SETUP_HOOKS,
)
from orchestrator import GenieOrchestrator
from genie import TaskResult


# =============================================================================
# Phase 4 validators and test data
# =============================================================================


# ---------------------------------------------------------------------------
# §4.1 — Git Operations
# ---------------------------------------------------------------------------

# ── T27 — git init, write file, commit, verify log ──────────────────────────

_T27_DIR = "/tmp/genie_t27"
_T27_FILE = "/tmp/genie_t27/hello.txt"
_T27_COMMIT_MSG = "initial commit from genie"


def _setup_t27() -> None:
    if os.path.exists(_T27_DIR):
        shutil.rmtree(_T27_DIR)


def _validate_t27(result: TaskResult, orch: GenieOrchestrator) -> None:
    assert os.path.exists(_T27_FILE), (
        f"Expected file {_T27_FILE} to exist, but it does not."
    )
    proc = subprocess.run(
        ["git", "-C", _T27_DIR, "log", "--oneline"],
        check=False, capture_output=True, text=True,
    )
    assert proc.returncode == 0, (
        f"'git log --oneline' failed with rc={proc.returncode}, "
        f"stderr: {proc.stderr.strip()!r}"
    )
    lines = [l for l in proc.stdout.strip().splitlines() if l.strip()]
    assert len(lines) == 1, (
        f"Expected exactly 1 commit line, got {len(lines)}: {lines!r}"
    )
    assert _T27_COMMIT_MSG in proc.stdout.lower(), (
        f"Expected commit message containing {_T27_COMMIT_MSG!r} (case-insensitive), "
        f"got: {proc.stdout.strip()!r}"
    )
    print(f"    ✓ {_T27_FILE} exists")
    print(f"    ✓ git log shows 1 commit: {lines[0]}")
    print(f"    ✓ Cost: ${result.cost_usd:.4f}  |  Iterations: {result.iterations}")


# ── T28 — create branch, switch to it, verify active branch ─────────────────

_T28_DIR = "/tmp/genie_t28"
_T28_BRANCH = "feature/genie-test"


def _setup_t28() -> None:
    if os.path.exists(_T28_DIR):
        shutil.rmtree(_T28_DIR)
    os.makedirs(_T28_DIR, exist_ok=True)
    subprocess.run(["git", "init", _T28_DIR], check=True,
                   capture_output=True, text=True)
    readme = os.path.join(_T28_DIR, "README.md")
    with open(readme, "w") as f:
        f.write("seed\n")
    subprocess.run(["git", "-C", _T28_DIR, "config", "user.email", "genie@local"],
                   check=True, capture_output=True, text=True)
    subprocess.run(["git", "-C", _T28_DIR, "config", "user.name", "Genie"],
                   check=True, capture_output=True, text=True)
    subprocess.run(["git", "-C", _T28_DIR, "add", "README.md"],
                   check=True, capture_output=True, text=True)
    subprocess.run(["git", "-C", _T28_DIR, "commit", "-m", "seed commit"],
                   check=True, capture_output=True, text=True)


def _validate_t28(result: TaskResult, orch: GenieOrchestrator) -> None:
    proc = subprocess.run(
        ["git", "-C", _T28_DIR, "branch"],
        check=False, capture_output=True, text=True,
    )
    assert proc.returncode == 0, (
        f"'git branch' failed with rc={proc.returncode}, "
        f"stderr: {proc.stderr.strip()!r}"
    )
    active = [l.strip() for l in proc.stdout.splitlines() if l.strip().startswith("*")]
    assert active, (
        f"No active branch found in 'git branch' output: {proc.stdout!r}"
    )
    active_name = active[0].lstrip("* ").strip()
    assert active_name == _T28_BRANCH, (
        f"Expected active branch '{_T28_BRANCH}', got: {active_name!r}"
    )
    print(f"    ✓ Active branch is '{active_name}'")
    print(f"    ✓ Cost: ${result.cost_usd:.4f}  |  Iterations: {result.iterations}")


# ── T29 — read git diff --staged, write summary containing known keyword ────

_T29_DIR = "/tmp/genie_t29"
_T29_SUMMARY = "/tmp/genie_t29/diff_summary.txt"
_T29_MARKER = "GENIE_PHASE4_MARKER"


def _setup_t29() -> None:
    if os.path.exists(_T29_DIR):
        shutil.rmtree(_T29_DIR)
    os.makedirs(_T29_DIR, exist_ok=True)
    subprocess.run(["git", "init", _T29_DIR], check=True,
                   capture_output=True, text=True)
    subprocess.run(["git", "-C", _T29_DIR, "config", "user.email", "genie@local"],
                   check=True, capture_output=True, text=True)
    subprocess.run(["git", "-C", _T29_DIR, "config", "user.name", "Genie"],
                   check=True, capture_output=True, text=True)
    constants_path = os.path.join(_T29_DIR, "constants.py")
    with open(constants_path, "w") as f:
        f.write('APP_VERSION = "1.0.0"\n')
    subprocess.run(["git", "-C", _T29_DIR, "add", "constants.py"],
                   check=True, capture_output=True, text=True)
    subprocess.run(["git", "-C", _T29_DIR, "commit", "-m", "initial"],
                   check=True, capture_output=True, text=True)
    with open(constants_path, "a") as f:
        f.write('GENIE_PHASE4_MARKER = "phase4_diff_test"\n')
    subprocess.run(["git", "-C", _T29_DIR, "add", "constants.py"],
                   check=True, capture_output=True, text=True)
    if os.path.exists(_T29_SUMMARY):
        os.remove(_T29_SUMMARY)


def _validate_t29(result: TaskResult, orch: GenieOrchestrator) -> None:
    assert os.path.exists(_T29_SUMMARY), (
        f"Expected summary file {_T29_SUMMARY} to exist, but it does not."
    )
    with open(_T29_SUMMARY, "r", encoding="utf-8") as f:
        content = f.read()
    assert len(content) > 20, (
        f"Summary file is too short ({len(content)} chars, need > 20). "
        f"Content: {content!r}"
    )
    assert _T29_MARKER in content, (
        f"Expected marker {_T29_MARKER!r} in summary file (case-sensitive), "
        f"but it was not found. Content: {content[:300]!r}"
    )
    print(f"    ✓ {_T29_SUMMARY} exists with {len(content)} chars")
    print(f"    ✓ Marker '{_T29_MARKER}' found in summary")
    print(f"    ✓ Cost: ${result.cost_usd:.4f}  |  Iterations: {result.iterations}")


# ---------------------------------------------------------------------------
# §4.2 — Codebase Navigation
# ---------------------------------------------------------------------------

# ── T30 — find function in multi-file project, write location to file ────────

_T30_DIR = "/tmp/genie_t30"
_T30_RESULT = "/tmp/genie_t30/result.txt"
_T30_TARGET_FUNC = "calculate_discount"
_T30_TARGET_FILE = "pricing.py"


def _setup_t30() -> None:
    if os.path.exists(_T30_DIR):
        shutil.rmtree(_T30_DIR)
    os.makedirs(_T30_DIR, exist_ok=True)

    with open(os.path.join(_T30_DIR, "pricing.py"), "w") as f:
        f.write(textwrap.dedent("""\
            def calculate_discount(price, pct):
                \"\"\"Apply a percentage discount to a price and return the discounted value.\"\"\"
                return price * (1 - pct / 100)
        """))

    # Decoy files — none contain "calculate_discount"
    decoy_files = {
        "models.py": "def get_model():\n    return 'default'\n",
        "utils.py": "def greet(name):\n    return f'Hello, {name}'\n",
        "config.py": "def get_config():\n    return {}\n",
        "db.py": "def connect():\n    return None\n",
        "auth.py": "def login(user, pw):\n    return True\n",
        "routes.py": "def index():\n    return 'OK'\n\ndef health():\n    return 'healthy'\n",
        "tasks.py": "def run_task(name):\n    return f'ran {name}'\n",
        "cache.py": "def get(key):\n    return None\n\ndef put(key, val):\n    pass\n",
        "logger.py": "def info(msg):\n    print(msg)\n\ndef warn(msg):\n    print(f'WARN: {msg}')\n\ndef error(msg):\n    print(f'ERROR: {msg}')\n",
    }
    for fname, content in decoy_files.items():
        with open(os.path.join(_T30_DIR, fname), "w") as f:
            f.write(content)

    if os.path.exists(_T30_RESULT):
        os.remove(_T30_RESULT)


def _validate_t30(result: TaskResult, orch: GenieOrchestrator) -> None:
    assert os.path.exists(_T30_RESULT), (
        f"Expected result file {_T30_RESULT} to exist, but it does not."
    )
    with open(_T30_RESULT, "r", encoding="utf-8") as f:
        content = f.read()
    content_lower = content.lower()
    assert _T30_TARGET_FILE.lower() in content_lower, (
        f"Expected '{_T30_TARGET_FILE}' in result file (case-insensitive), "
        f"got: {content!r}"
    )
    assert _T30_TARGET_FUNC.lower() in content_lower, (
        f"Expected '{_T30_TARGET_FUNC}' in result file (case-insensitive), "
        f"got: {content!r}"
    )
    print(f"    ✓ {_T30_RESULT} contains '{_T30_TARGET_FILE}' and '{_T30_TARGET_FUNC}'")
    print(f"    ✓ Cost: ${result.cost_usd:.4f}  |  Iterations: {result.iterations}")


# ── T31 — fix failing test without touching other files ──────────────────────

_T31_DIR = "/tmp/genie_t31"
_T31_TARGET = "/tmp/genie_t31/math_utils.py"
_T31_TESTFILE = "/tmp/genie_t31/test_math_utils.py"

# These are set in _setup_t31() after writing the files.
_T31_HELPERS_CONTENT: str = ""
_T31_VALIDATORS_CONTENT: str = ""
_T31_FORMATTERS_CONTENT: str = ""


def _setup_t31() -> None:
    global _T31_HELPERS_CONTENT, _T31_VALIDATORS_CONTENT, _T31_FORMATTERS_CONTENT

    if os.path.exists(_T31_DIR):
        shutil.rmtree(_T31_DIR)
    os.makedirs(_T31_DIR, exist_ok=True)

    with open(_T31_TARGET, "w") as f:
        f.write(textwrap.dedent("""\
            def multiply(a, b):
                return a + b   # bug: should be a * b
        """))

    with open(_T31_TESTFILE, "w") as f:
        f.write(textwrap.dedent("""\
            from math_utils import multiply

            def test_multiply():
                assert multiply(3, 4) == 12
                assert multiply(0, 5) == 0
                assert multiply(-2, 3) == -6
        """))

    helpers_path = os.path.join(_T31_DIR, "helpers.py")
    with open(helpers_path, "w") as f:
        f.write("def ping():\n    return 'pong'\n")
    _T31_HELPERS_CONTENT = open(helpers_path, "r").read()

    validators_path = os.path.join(_T31_DIR, "validators.py")
    with open(validators_path, "w") as f:
        f.write("def is_positive(n):\n    return n > 0\n")
    _T31_VALIDATORS_CONTENT = open(validators_path, "r").read()

    formatters_path = os.path.join(_T31_DIR, "formatters.py")
    with open(formatters_path, "w") as f:
        f.write("def fmt_name(first, last):\n    return f'{first} {last}'\n")
    _T31_FORMATTERS_CONTENT = open(formatters_path, "r").read()


def _validate_t31(result: TaskResult, orch: GenieOrchestrator) -> None:
    proc = subprocess.run(
        ["python3", "-m", "pytest", _T31_TESTFILE, "-v", "--tb=short"],
        check=False, capture_output=True, text=True,
        cwd=_T31_DIR,
    )
    assert proc.returncode == 0, (
        f"pytest failed with rc={proc.returncode}.\n"
        f"stdout: {proc.stdout[-500:]!r}\nstderr: {proc.stderr[-300:]!r}"
    )

    # Verify untouched files
    for fname, expected in [
        ("helpers.py", _T31_HELPERS_CONTENT),
        ("validators.py", _T31_VALIDATORS_CONTENT),
        ("formatters.py", _T31_FORMATTERS_CONTENT),
    ]:
        fpath = os.path.join(_T31_DIR, fname)
        actual = open(fpath, "r").read()
        assert actual == expected, (
            f"File {fname} was modified! "
            f"Expected length {len(expected)}, got length {len(actual)}."
        )

    print(f"    ✓ pytest {_T31_TESTFILE} passes")
    print(f"    ✓ helpers.py, validators.py, formatters.py untouched")
    print(f"    ✓ Cost: ${result.cost_usd:.4f}  |  Iterations: {result.iterations}")


# ── T32 — add new method to existing class without modifying other methods ───

_T32_FILE = "/tmp/genie_t32_invoice.py"
_T32_NEW_METHOD = "compute_tax"

# Set in _setup_t32() after writing the file.
_T32_INIT_SRC: str = ""
_T32_GET_AMOUNT_SRC: str = ""
_T32_APPLY_DISCOUNT_SRC: str = ""
_T32_TO_DICT_SRC: str = ""


def _setup_t32() -> None:
    global _T32_INIT_SRC, _T32_GET_AMOUNT_SRC, _T32_APPLY_DISCOUNT_SRC, _T32_TO_DICT_SRC

    source = textwrap.dedent("""\
        class Invoice:
            def __init__(self, amount):
                self.amount = amount

            def get_amount(self):
                return self.amount

            def apply_discount(self, pct):
                return self.amount * (1 - pct / 100)

            def to_dict(self):
                return {"amount": self.amount}
    """)
    with open(_T32_FILE, "w") as f:
        f.write(source)

    # Read back and store each method's source verbatim
    raw = open(_T32_FILE, "r").read()
    # Simple line-based extraction: each method block starts with "    def "
    # and continues until the next "    def " or end of class.
    lines = raw.splitlines(keepends=True)
    methods: dict[str, str] = {}
    current_name: str | None = None
    current_lines: list[str] = []
    for line in lines:
        if line.strip().startswith("def ") and "self" in line:
            if current_name is not None:
                methods[current_name] = "".join(current_lines)
            # Extract method name
            name = line.strip().split("(")[0].replace("def ", "")
            current_name = name
            current_lines = [line]
        elif current_name is not None:
            current_lines.append(line)
    if current_name is not None:
        methods[current_name] = "".join(current_lines)

    _T32_INIT_SRC = methods["__init__"]
    _T32_GET_AMOUNT_SRC = methods["get_amount"]
    _T32_APPLY_DISCOUNT_SRC = methods["apply_discount"]
    _T32_TO_DICT_SRC = methods["to_dict"]


def _validate_t32(result: TaskResult, orch: GenieOrchestrator) -> None:
    import ast as _ast

    with open(_T32_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    try:
        _ast.parse(content)
    except SyntaxError as exc:
        raise AssertionError(
            f"{_T32_FILE} has a SyntaxError: {exc}"
        ) from exc

    assert _T32_NEW_METHOD in content, (
        f"Expected '{_T32_NEW_METHOD}' to appear in {_T32_FILE}, "
        f"but it was not found."
    )

    # Accept minor whitespace variations in the expression
    import re as _re
    assert _re.search(r"self\.amount\s*\*\s*\(\s*rate\s*/\s*100\s*\)", content), (
        f"Expected expression 'self.amount * (rate / 100)' not found "
        f"in {_T32_FILE}. Content:\n{content[:500]!r}"
    )

    for name, src in [
        ("__init__", _T32_INIT_SRC),
        ("get_amount", _T32_GET_AMOUNT_SRC),
        ("apply_discount", _T32_APPLY_DISCOUNT_SRC),
        ("to_dict", _T32_TO_DICT_SRC),
    ]:
        assert src in content, (
            f"Original method '{name}' was altered or removed. "
            f"Expected this verbatim block to be present:\n{src!r}"
        )

    print(f"    ✓ {_T32_FILE} parses without SyntaxError")
    print(f"    ✓ '{_T32_NEW_METHOD}' method present with correct expression")
    print(f"    ✓ Original methods untouched")
    print(f"    ✓ Cost: ${result.cost_usd:.4f}  |  Iterations: {result.iterations}")


# ---------------------------------------------------------------------------
# §4.3 — Issue → Fix Cycle
# ---------------------------------------------------------------------------

# ── T33 — read ISSUE.md, locate bug, fix it, pytest green ───────────────────

_T33_DIR = "/tmp/genie_t33"
_T33_ISSUE = "/tmp/genie_t33/ISSUE.md"
_T33_SOURCE = "/tmp/genie_t33/slicer.py"
_T33_TESTFILE = "/tmp/genie_t33/test_slicer.py"

# Set in _setup_t33() after writing ISSUE.md.
_T33_ISSUE_CONTENT: str = ""


def _setup_t33() -> None:
    global _T33_ISSUE_CONTENT

    if os.path.exists(_T33_DIR):
        shutil.rmtree(_T33_DIR)
    os.makedirs(_T33_DIR, exist_ok=True)

    with open(_T33_SOURCE, "w") as f:
        f.write(textwrap.dedent("""\
            def first_n(items, n):
                \"\"\"Return the first n items from a list.\"\"\"
                return items[:n - 1]   # bug: should be items[:n]
        """))

    with open(_T33_TESTFILE, "w") as f:
        f.write(textwrap.dedent("""\
            from slicer import first_n

            def test_first_n():
                assert first_n([1, 2, 3, 4, 5], 3) == [1, 2, 3]
                assert first_n([10, 20], 2) == [10, 20]
                assert first_n([], 0) == []
        """))

    with open(_T33_ISSUE, "w") as f:
        f.write(textwrap.dedent("""\
            # Bug Report

            **Function:** `first_n` in `slicer.py`

            **Symptom:** `first_n([1, 2, 3, 4, 5], 3)` returns `[1, 2]` instead
            of `[1, 2, 3]`. The function is returning one fewer item than
            requested.

            **Expected behaviour:** `first_n(items, n)` should return exactly
            the first `n` items.
        """))

    _T33_ISSUE_CONTENT = open(_T33_ISSUE, "r").read()


def _validate_t33(result: TaskResult, orch: GenieOrchestrator) -> None:
    proc = subprocess.run(
        ["python3", "-m", "pytest", _T33_TESTFILE, "-v", "--tb=short"],
        check=False, capture_output=True, text=True,
        cwd=_T33_DIR,
    )
    assert proc.returncode == 0, (
        f"pytest failed with rc={proc.returncode}.\n"
        f"stdout: {proc.stdout[-500:]!r}\nstderr: {proc.stderr[-300:]!r}"
    )

    import re as _re
    with open(_T33_SOURCE, "r", encoding="utf-8") as f:
        source = f.read()
    assert "[:n]" in source, (
        f"Expected correct slice '[:n]' in {_T33_SOURCE}, "
        f"but it was not found. Content: {source!r}"
    )
    # Strip Python single-line comments before checking for the bug pattern,
    # so a comment like "# fixed: was items[:n - 1]" does not cause a false failure.
    source_no_comments = _re.sub(r"#[^\n]*", "", source)
    assert "[:n - 1]" not in source_no_comments, (
        f"Bug '[:n - 1]' still present in non-comment code of {_T33_SOURCE}. "
        f"Content: {source!r}"
    )

    actual_issue = open(_T33_ISSUE, "r").read()
    assert actual_issue == _T33_ISSUE_CONTENT, (
        f"ISSUE.md was modified! Expected length {len(_T33_ISSUE_CONTENT)}, "
        f"got length {len(actual_issue)}."
    )

    print(f"    ✓ pytest {_T33_TESTFILE} passes")
    print(f"    ✓ Bug '[:n - 1]' removed, '[:n]' present")
    print(f"    ✓ ISSUE.md untouched")
    print(f"    ✓ Cost: ${result.cost_usd:.4f}  |  Iterations: {result.iterations}")


# ---------------------------------------------------------------------------
# §4.4 — Dependency Management
# ---------------------------------------------------------------------------

# ── T34 — create venv, install requirements.txt, run script, verify stdout ───

_T34_DIR = "/tmp/genie_t34"
_T34_REQS = "/tmp/genie_t34/requirements.txt"
_T34_SCRIPT = "/tmp/genie_t34/check_imports.py"
_T34_VENV = "/tmp/genie_t34/venv"
_T34_EXPECTED_OUTPUT = "ok"


def _setup_t34() -> None:
    if os.path.exists(_T34_DIR):
        shutil.rmtree(_T34_DIR)
    os.makedirs(_T34_DIR, exist_ok=True)

    with open(_T34_REQS, "w") as f:
        f.write("httpx\nrich\n")

    with open(_T34_SCRIPT, "w") as f:
        f.write("import httpx\nimport rich\nprint(\"ok\")\n")


def _validate_t34(result: TaskResult, orch: GenieOrchestrator) -> None:
    venv_python = os.path.join(_T34_VENV, "bin", "python")
    assert os.path.exists(venv_python), (
        f"Expected venv python at {venv_python}, but it does not exist."
    )

    proc = subprocess.run(
        [venv_python, _T34_SCRIPT],
        check=False, capture_output=True, text=True,
    )
    assert proc.returncode == 0, (
        f"Script failed with rc={proc.returncode}.\n"
        f"stdout: {proc.stdout!r}\nstderr: {proc.stderr[-300:]!r}"
    )
    assert proc.stdout.strip() == _T34_EXPECTED_OUTPUT, (
        f"Expected stdout '{_T34_EXPECTED_OUTPUT}', got: {proc.stdout.strip()!r}"
    )

    print(f"    ✓ venv created at {_T34_VENV}")
    print(f"    ✓ Script output: {proc.stdout.strip()!r}")
    print(f"    ✓ Cost: ${result.cost_usd:.4f}  |  Iterations: {result.iterations}")


# ── T35 — observe ModuleNotFoundError, install missing package, rerun ────────

_T35_DIR = "/tmp/genie_t35"
_T35_SCRIPT = "/tmp/genie_t35/run_me.py"
_T35_VENV = "/tmp/genie_t35/venv"


def _setup_t35() -> None:
    if os.path.exists(_T35_DIR):
        shutil.rmtree(_T35_DIR)
    os.makedirs(_T35_DIR, exist_ok=True)

    subprocess.run(["python3", "-m", "venv", _T35_VENV], check=True)

    with open(_T35_SCRIPT, "w") as f:
        f.write('import arrow\nnow = arrow.now()\nprint(f"current time: {now}")\n')


def _validate_t35(result: TaskResult, orch: GenieOrchestrator) -> None:
    venv_python = os.path.join(_T35_VENV, "bin", "python")

    proc_import = subprocess.run(
        [venv_python, "-c", "import arrow"],
        check=False, capture_output=True, text=True,
    )
    assert proc_import.returncode == 0, (
        f"'import arrow' failed in venv with rc={proc_import.returncode}. "
        f"stderr: {proc_import.stderr.strip()!r}"
    )

    proc_run = subprocess.run(
        [venv_python, _T35_SCRIPT],
        check=False, capture_output=True, text=True,
    )
    assert proc_run.returncode == 0, (
        f"Script failed with rc={proc_run.returncode}.\n"
        f"stdout: {proc_run.stdout!r}\nstderr: {proc_run.stderr[-300:]!r}"
    )
    assert "current time:" in proc_run.stdout, (
        f"Expected 'current time:' in stdout, got: {proc_run.stdout.strip()!r}"
    )

    print(f"    ✓ 'import arrow' succeeds in venv")
    print(f"    ✓ Script output: {proc_run.stdout.strip()[:60]!r}")
    print(f"    ✓ Cost: ${result.cost_usd:.4f}  |  Iterations: {result.iterations}")


# =============================================================================
# PHASE4_TESTS list and _SETUP_HOOKS registration
# =============================================================================

PHASE4_TESTS: list[TestCase] = [
    TestCase(
        number=27,
        category="git_ops",
        description="§4.1 — git init, write file, commit, verify log",
        goal=(
            "Create the directory /tmp/genie_t27 if it does not exist.\n"
            "Run: git init /tmp/genie_t27\n"
            "Write the text 'hello from genie' to /tmp/genie_t27/hello.txt.\n"
            "Run: git -C /tmp/genie_t27 config user.email 'genie@local'\n"
            "Run: git -C /tmp/genie_t27 config user.name 'Genie'\n"
            "Run: git -C /tmp/genie_t27 add hello.txt\n"
            "Run: git -C /tmp/genie_t27 commit -m 'initial commit from genie'\n"
            "Tell me when done."
        ),
        validator=_validate_t27,
        expected_outcome="done",
    ),
    TestCase(
        number=28,
        category="git_ops",
        description="§4.1 — create branch, switch to it, verify active branch",
        goal=(
            "In the existing git repository at /tmp/genie_t28:\n"
            "Create a new branch named 'feature/genie-test'.\n"
            "Switch to that branch.\n"
            "Tell me when done."
        ),
        validator=_validate_t28,
        expected_outcome="done",
    ),
    TestCase(
        number=29,
        category="git_ops",
        description="§4.1 — read git diff --staged, write summary with known keyword",
        goal=(
            "In the repository at /tmp/genie_t29, run: git diff --staged\n"
            "Read the output carefully.\n"
            "Write a one-paragraph plain-text summary of what changed to "
            "/tmp/genie_t29/diff_summary.txt.\n"
            "Tell me when done."
        ),
        validator=_validate_t29,
        expected_outcome="done",
    ),
    TestCase(
        number=30,
        category="codebase_nav",
        description="§4.2 — find function in multi-file project, write location to file",
        goal=(
            "In the Python project at /tmp/genie_t30, find the function "
            "responsible for applying a discount to a price.\n"
            "Write its filename (basename only, e.g. 'foo.py') and the function "
            "name on separate lines to /tmp/genie_t30/result.txt.\n"
            "Tell me when done."
        ),
        validator=_validate_t30,
        expected_outcome="done",
    ),
    TestCase(
        number=31,
        category="codebase_nav",
        description="§4.2 — fix failing test in multi-file project without touching other files",
        goal=(
            "In the Python project at /tmp/genie_t31, the test file "
            "test_math_utils.py is failing. Read the test, find the source "
            "function responsible for the failure in the project files, fix only "
            "that function so the tests pass. Do not modify any other file.\n"
            "Run pytest /tmp/genie_t31/test_math_utils.py to confirm green.\n"
            "Tell me when done."
        ),
        validator=_validate_t31,
        expected_outcome="done",
    ),
    TestCase(
        number=32,
        category="codebase_nav",
        description="§4.2 — add new method to existing class without modifying other methods",
        goal=(
            "In the file /tmp/genie_t32_invoice.py, add a new method to the "
            "Invoice class with this exact signature:\n"
            "  def compute_tax(self, rate: float) -> float:\n"
            "It must return self.amount * (rate / 100).\n"
            "Do not modify any existing methods.\n"
            "Tell me when done."
        ),
        validator=_validate_t32,
        expected_outcome="done",
    ),
    TestCase(
        number=33,
        category="codebase_nav",
        description="§4.3 — read ISSUE.md, locate bug, fix it, pytest green, ISSUE.md unmodified",
        goal=(
            "Read the issue description at /tmp/genie_t33/ISSUE.md.\n"
            "Find the bug described in the project files under /tmp/genie_t33/.\n"
            "Fix only the buggy function. The function is named 'first_n' — do not rename it.\n"
            "Do not modify ISSUE.md or test_slicer.py.\n"
            "Run pytest /tmp/genie_t33/test_slicer.py to confirm all tests pass.\n"
            "Tell me when done."
        ),
        validator=_validate_t33,
        expected_outcome="done",
    ),
    TestCase(
        number=34,
        category="shell",
        description="§4.4 — create venv, install requirements.txt, run script, verify stdout",
        goal=(
            "In /tmp/genie_t34:\n"
            "1. Create a Python virtual environment at /tmp/genie_t34/venv using:\n"
            "   python3 -m venv /tmp/genie_t34/venv\n"
            "2. Install dependencies using:\n"
            "   /tmp/genie_t34/venv/bin/pip install -r /tmp/genie_t34/requirements.txt\n"
            "3. Run the script using:\n"
            "   /tmp/genie_t34/venv/bin/python /tmp/genie_t34/check_imports.py\n"
            "Confirm the script prints 'ok' and exits cleanly.\n"
            "Tell me when done."
        ),
        validator=_validate_t34,
        expected_outcome="done",
    ),
    TestCase(
        number=35,
        category="shell",
        description="§4.4 — observe ModuleNotFoundError, install missing package into venv, rerun",
        goal=(
            "The script /tmp/genie_t35/run_me.py fails with a ModuleNotFoundError.\n"
            "A virtual environment already exists at /tmp/genie_t35/venv.\n"
            "Install the missing package (arrow) into the venv using EXACTLY this command:\n"
            "  /tmp/genie_t35/venv/bin/pip install arrow\n"
            "Do NOT use any other pip or install location.\n"
            "Then run the script using:\n"
            "  /tmp/genie_t35/venv/bin/python /tmp/genie_t35/run_me.py\n"
            "Confirm it exits cleanly and prints the current time.\n"
            "Tell me when done."
        ),
        validator=_validate_t35,
        expected_outcome="done",
    ),
]

_SETUP_HOOKS.update({
    27: _setup_t27,
    28: _setup_t28,
    29: _setup_t29,
    30: _setup_t30,
    31: _setup_t31,
    32: _setup_t32,
    33: _setup_t33,
    34: _setup_t34,
    35: _setup_t35,
})

TESTS: list[TestCase] = _BASE_TESTS + PHASE4_TESTS

# Patch the module-level TESTS in genie_suite so that GenieTester's
# run_one / run_all / run_many methods (which reference genie_suite.TESTS
# directly) can see the Phase 4 entries.
genie_suite.TESTS = TESTS


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Genie unified regression suite (T1–T26 base + Phase 4). "
            "Runs the full Genie stack against real desktop state."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python genie_phase4.py                                   # run all\n"
            "  python genie_phase4.py --all                             # run all, no early stop\n"
            "  python genie_phase4.py --case 5                          # run test 5\n"
            "  python genie_phase4.py --case 26 --repeat 5              # test 26 x 5\n"
            "  python genie_phase4.py --cases 15-20                     # tests 15-20\n"
            "  python genie_phase4.py --cases 1 14-20 24 26             # mixed selection\n"
            "  python genie_phase4.py --cases 1 14-20 24 26 --repeat 5  # mixed x 5\n"
            "  python genie_phase4.py --all --repeat 5                  # all tests x 5\n"
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
