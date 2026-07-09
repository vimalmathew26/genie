"""
Genie — actions.py
Pure action execution layer. Owns all action handler functions extracted
from genie.py.  No orchestration, no error classification, no LLM calls,
no Layer 3 / observe() knowledge.

Module-level init:
    Call ``init(controller)`` once at orchestrator startup before invoking
    any raw-input action (click, press_key, type_text).
"""

import base64
import json
import ast
import os
import re
import signal
import subprocess
import time
import urllib.parse
import urllib.request

from config import (
    CMD_BLOCKLIST,
    DEFAULT_CMD_TIMEOUT,
    MAX_READ_FILE_BYTES,
    READ_FILE_TRUNCATION_MARKER,
    WORKSPACE_DIR,
)
from xdotool_controller import XdotoolController


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_controller: XdotoolController | None = None


def init(controller: XdotoolController) -> None:
    """Store the shared XdotoolController reference.  Called once."""
    global _controller
    _controller = controller


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_blocklist(cmd: str) -> None:
    """Raise RuntimeError if *cmd* matches any CMD_BLOCKLIST entry.

    Workspace-scoped allowlist: rm -rf is permitted when the target path is
    clearly scoped to ~/genie_workspace/ (the model's designated sandbox).
    This lets cleanup subtasks delete old project directories without an
    unrecoverable blocklist error.
    """
    _workspace = os.path.expanduser("~/genie_workspace")
    _home = os.path.expanduser("~")

    cmd_stripped = cmd.strip()
    cmd_lower = cmd_stripped.lower()

    # Workspace allowlist: permit rm -rf <path> when path is under ~/genie_workspace/
    # Matches: rm -rf ~/genie_workspace/foo, rm -rf /home/user/genie_workspace/foo,
    #          rm -rf genie_workspace/foo  (relative), rm -r ~/genie_workspace/...
    _RM_RF_PATTERNS = ("rm -rf ", "rm -r ")
    for _pat in _RM_RF_PATTERNS:
        if cmd_lower.startswith(_pat):
            _target = cmd_stripped[len(_pat):].strip().rstrip("/")
            _target_expanded = os.path.expanduser(_target)
            _target_abs = os.path.abspath(_target_expanded)
            if _target_abs.startswith(_workspace) or _target_abs == _workspace:
                return  # safe — within the workspace sandbox
            # Also allow relative paths that don't escape workspace
            if not _target.startswith("/") and not _target.startswith("~"):
                _target_from_ws = os.path.abspath(os.path.join(_workspace, _target))
                if _target_from_ws.startswith(_workspace):
                    return  # safe relative path

    for match_type, match_value in CMD_BLOCKLIST:
        if match_type == "substr":
            if match_value in cmd_lower:
                if "rm -rf" in cmd_lower or "rm -r " in cmd_lower:
                    raise RuntimeError(
                        f"rm -rf/rm -r is blocked outside ~/genie_workspace "
                        f"(safety policy). Do NOT retry this command. "
                        f"If you are trying to clone a repo: choose a target "
                        f"path that does not already exist — "
                        f"'gh repo clone OWNER/REPO ~/Documents/NEWDIR' "
                        f"creates NEWDIR fresh, no cleanup needed. "
                        f"If you are running scripts: keep all working files "
                        f"inside ~/genie_workspace/ where deletion is allowed. "
                        f"Blocked: {cmd}"
                    )
                raise RuntimeError(f"command rejected by blocklist: {cmd}")
        elif match_type == "regex":
            if re.search(match_value, cmd_lower):
                raise RuntimeError(f"command rejected by blocklist: {cmd}")


def _ensure_controller() -> XdotoolController:
    """Return the controller or raise if init() was never called."""
    if _controller is None:
        raise RuntimeError("actions.py not initialized — call init() first")
    return _controller


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def _resolve_path(path: str) -> str:
    """Resolve *path* relative to WORKSPACE_DIR for user-facing file actions.

    - Normalizes hallucinated /home/user/ to the real home directory.
    - Absolute paths (start with "/") pass through (after normalization).
    - Tilde paths (start with "~") are expanded via os.path.expanduser.
    - Relative paths are joined with WORKSPACE_DIR.
    """
    # LLMs frequently hallucinate /home/user/ — rewrite to real home.
    _real_home = os.path.expanduser("~")
    if path.startswith("/home/user/"):
        path = _real_home + path[len("/home/user"):]
    elif path == "/home/user":
        path = _real_home

    if path.startswith("/"):
        return path
    if path.startswith("~"):
        return os.path.expanduser(path)
    return os.path.join(WORKSPACE_DIR, path)


# ---------------------------------------------------------------------------
# File operations
# ---------------------------------------------------------------------------

def read_file(path: str) -> str:
    """Read file content, truncated at MAX_READ_FILE_BYTES.

    Returns:
        File content as str, with truncation marker appended if needed.
    Raises:
        FileNotFoundError, OSError on failure.
    """
    path = _resolve_path(path)
    size = os.path.getsize(path)
    with open(path, "r", errors="replace") as f:
        content = f.read(MAX_READ_FILE_BYTES)
    if size > MAX_READ_FILE_BYTES:
        marker = READ_FILE_TRUNCATION_MARKER.format(
            total_bytes=size, limit=MAX_READ_FILE_BYTES,
        )
        content += f"\n{marker}"
    return content


def write_file(path: str, content: str) -> None:
    """Write *content* to *path*, creating parent directories as needed.

    Raises:
        OSError on failure.
    """
    path = _resolve_path(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


def append_file(path: str, content: str) -> None:
    """Append *content* to *path*, creating parent directories as needed.

    Raises:
        OSError on failure.
    """
    path = _resolve_path(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a") as f:
        f.write(content)


def delete_file(path: str) -> None:
    """Delete the file at *path*.

    Raises:
        FileNotFoundError if the path does not exist.
        OSError on other failures.
    """
    path = _resolve_path(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"file not found: {path}")
    os.remove(path)


def list_dir(path: str) -> list[str]:
    """Return a sorted list of entries in *path*.

    Raises:
        NotADirectoryError, OSError on failure.
    """
    path = _resolve_path(path)
    return sorted(os.listdir(path))


def index_codebase(path: str) -> dict:
    """Walk a codebase and write CODEBASE_INDEX.md with all symbols.

    Returns:
        {"index_path": str, "file_count": int, "symbol_count": int}
    Raises:
        ValueError if *path* is not a directory.
    """
    path = _resolve_path(path)
    if not os.path.isdir(path):
        raise ValueError(f"path is not a directory: {path}")

    SKIP_DIRS = {".git", "__pycache__", "node_modules", "venv", ".venv", "dist", "build"}
    MAX_FILES = 500

    py_count = 0
    js_count = 0
    all_files: list[str] = []

    for root, dirs, files in os.walk(path):
        # skip hidden dirs and known junk
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext == ".py":
                py_count += 1
            elif ext in (".js", ".ts", ".jsx", ".tsx"):
                js_count += 1
            if ext in (".py", ".js", ".ts", ".jsx", ".tsx"):
                all_files.append(os.path.join(root, f))

    total_files_found = len(all_files)
    truncated = total_files_found > MAX_FILES
    all_files = all_files[:MAX_FILES]

    # Auto-detect language
    index_py = py_count >= js_count if (py_count or js_count) else True
    index_js = js_count >= py_count if (py_count or js_count) else True

    _JS_PATTERNS = [
        re.compile(r"^\s*(?:export\s+)?(?:default\s+)?class\s+(\w+)"),
        re.compile(r"^\s*(?:export\s+)?(?:default\s+)?function\s+(\w+)\s*\("),
        re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\("),
    ]

    sections: list[str] = []
    unparseable: list[str] = []
    file_count = 0
    symbol_count = 0

    for fpath in sorted(all_files):
        rel = os.path.relpath(fpath, path)
        ext = os.path.splitext(fpath)[1].lower()

        symbols: list[tuple[int, str, str]] = []  # (lineno, kind, name)

        if ext == ".py" and index_py:
            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
                    source = fh.read()
                tree = ast.parse(source, filename=fpath)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        symbols.append((node.lineno, "class", node.name))
                    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        symbols.append((node.lineno, "def", node.name))
            except (SyntaxError, OSError):
                unparseable.append(rel)
                continue
        elif ext in (".js", ".ts", ".jsx", ".tsx") and index_js:
            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
                    lines = fh.readlines()
                for lineno_0, line in enumerate(lines):
                    for pat in _JS_PATTERNS:
                        m = pat.match(line)
                        if m:
                            name = m.group(1)
                            kind = "class" if "class" in pat.pattern else "def"
                            symbols.append((lineno_0 + 1, kind, name))
                            break
            except OSError:
                unparseable.append(rel)
                continue
        else:
            continue

        if not symbols:
            continue

        symbols.sort(key=lambda s: s[0])
        file_count += 1
        symbol_count += len(symbols)
        lines_out = [f"## {rel} ({len(symbols)} symbols)"]
        for lineno, kind, name in symbols:
            lines_out.append(f"- line {lineno}: [{kind}] {name}")
        sections.append("\n".join(lines_out))

    # Build final markdown
    header = (
        "# Codebase Index\n"
        "Generated by index_codebase. Use this to navigate before reading files.\n"
    )
    body = "\n\n".join(sections)
    if unparseable:
        body += "\n\n## _unparseable\n" + "\n".join(f"- {p}" for p in unparseable)
    if truncated:
        body += f"\n\n[INDEX TRUNCATED — {total_files_found} files found, showing first {MAX_FILES}]"

    index_path = os.path.join(path, "CODEBASE_INDEX.md")
    with open(index_path, "w", encoding="utf-8") as fh:
        fh.write(header + "\n" + body + "\n")

    return {"index_path": index_path, "file_count": file_count, "symbol_count": symbol_count}


# ---------------------------------------------------------------------------
# Codebase search
# ---------------------------------------------------------------------------

def search_codebase(path: str, pattern: str, glob: str | None = None,
                    max_results: int = 50) -> dict:
    """Grep-style regex search across files under *path*.

    Returns a structured dict with ``matches``, ``total_matches``,
    ``truncated``.  Never raises — invalid regex or I/O errors are
    returned as structured dicts or silently skipped.
    """
    import fnmatch as _fnmatch

    # Validate regex up-front.
    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        return {"error": "invalid_regex", "detail": str(exc)}

    path = _resolve_path(path)

    SKIP_DIRS = {
        ".git", "__pycache__", "node_modules", "venv", ".venv",
        "dist", "build",
    }

    matches: dict[str, list[dict]] = {}
    total_matches = 0
    truncated = False

    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        for fname in files:
            if truncated:
                break
            if glob is not None and not _fnmatch.fnmatch(fname, glob):
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "rb") as fb:
                    head = fb.read(8192)
                if b"\x00" in head:
                    continue  # binary file — skip silently
                with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
                    for lineno, line in enumerate(fh, start=1):
                        if compiled.search(line):
                            rel = os.path.relpath(fpath, path)
                            matches.setdefault(rel, []).append(
                                {"line": lineno, "content": line.rstrip("\n")}
                            )
                            total_matches += 1
                            if total_matches >= max_results:
                                truncated = True
                                break
            except (OSError, PermissionError, UnicodeDecodeError):
                continue  # unreadable file — skip silently
        if truncated:
            break

    return {
        "matches": matches,
        "total_matches": total_matches,
        "truncated": truncated,
    }


# ---------------------------------------------------------------------------
# AST-based structural search
# ---------------------------------------------------------------------------

def ast_search(path: str, query_type: str, name: str,
               glob: str = "*.py", max_results: int = 50) -> dict:
    """Search Python files under *path* for structural AST matches.

    *query_type* is one of ``"class"``, ``"function"``, ``"import"``.
    *name* is a regex matched against the symbol name.

    Returns a structured dict with ``matches``, ``total_matches``,
    ``truncated``.  Never raises — all errors are returned as structured
    dicts or silently skipped.
    """
    import fnmatch as _fnmatch

    VALID_TYPES = {"class", "function", "import"}
    if query_type not in VALID_TYPES:
        return {
            "error": "invalid_query_type",
            "detail": f"{query_type} is not one of: class, function, import",
        }

    try:
        compiled = re.compile(name)
    except re.error as exc:
        return {"error": "invalid_regex", "detail": str(exc)}

    path = _resolve_path(path)

    SKIP_DIRS = {
        ".git", "__pycache__", "node_modules", "venv", ".venv",
        "dist", "build",
    }

    matches: dict[str, list[dict]] = {}
    total_matches = 0
    truncated = False

    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        for fname in files:
            if truncated:
                break
            if not _fnmatch.fnmatch(fname, glob):
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
                    source = fh.read()
            except (OSError, PermissionError, UnicodeDecodeError):
                continue
            try:
                tree = ast.parse(source, filename=fpath)
            except (SyntaxError, ValueError):
                continue

            rel = os.path.relpath(fpath, path)

            for node in ast.walk(tree):
                if truncated:
                    break

                if query_type == "class":
                    if isinstance(node, ast.ClassDef) and compiled.search(node.name):
                        matches.setdefault(rel, []).append(
                            {"line": node.lineno, "name": node.name}
                        )
                        total_matches += 1

                elif query_type == "function":
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and compiled.search(node.name):
                        matches.setdefault(rel, []).append(
                            {"line": node.lineno, "name": node.name}
                        )
                        total_matches += 1

                elif query_type == "import":
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            if compiled.search(alias.name):
                                matches.setdefault(rel, []).append(
                                    {"line": node.lineno, "name": alias.name}
                                )
                                total_matches += 1
                                if total_matches >= max_results:
                                    truncated = True
                                    break
                    elif isinstance(node, ast.ImportFrom):
                        # Match against module name
                        if node.module and compiled.search(node.module):
                            matches.setdefault(rel, []).append(
                                {"line": node.lineno, "name": node.module}
                            )
                            total_matches += 1
                        # Match against imported names
                        if not truncated:
                            for alias in (node.names or []):
                                if compiled.search(alias.name):
                                    matches.setdefault(rel, []).append(
                                        {"line": node.lineno, "name": alias.name}
                                    )
                                    total_matches += 1
                                    if total_matches >= max_results:
                                        truncated = True
                                        break

                if total_matches >= max_results:
                    truncated = True

        if truncated:
            break

    return {
        "matches": matches,
        "total_matches": total_matches,
        "truncated": truncated,
    }


# ---------------------------------------------------------------------------
# Shell operations
# ---------------------------------------------------------------------------

def run_command(cmd: str, timeout: int | None = None) -> dict:
    """Execute *cmd* in a shell and return the result dict.

    **Never raises** (except for blocklist violations).  Subprocess
    failures (timeout, non-zero exit) are captured in the returned dict.

    Returns:
        {"exit_code": int | None, "stdout": str, "stderr": str,
         "timed_out": bool}

    Raises:
        RuntimeError if *cmd* matches CMD_BLOCKLIST.
    """
    # Normalize bare `python` → `python3` when used as a command.
    # Matches `python` not preceded by alphanum/_ and not followed by
    # digit/letter/_/./- (so python3, cpython, python-foo are untouched).
    cmd = re.sub(r'(?<![a-zA-Z0-9_])python(?![0-9a-zA-Z_./\\-])', 'python3', cmd)
    _check_blocklist(cmd)

    if timeout is None:
        timeout = DEFAULT_CMD_TIMEOUT

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=WORKSPACE_DIR,
        )
        return {
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired:
        return {
            "exit_code": None,
            "stdout": "",
            "stderr": "",
            "timed_out": True,
        }
    except (UnicodeEncodeError, UnicodeDecodeError) as exc:
        # The cmd string contains garbage bytes (e.g. LLM emitted raw binary
        # tokens in the argument). Encoding it for the OS shell fails.
        # Return a command_failed result so the brain loop can recover,
        # rather than propagating the exception to classify_error which
        # would fall through to ERROR_CLASS_UNRECOVERABLE.
        return {
            "exit_code": 1,
            "stdout": "",
            "stderr": f"UnicodeError: command contains non-encodable bytes and cannot be executed. {exc}",
            "timed_out": False,
        }
    except (ValueError, OSError) as exc:
        # ValueError: e.g. embedded null bytes in command string.
        # OSError: e.g. /bin/sh not found or other OS-level launch failures.
        return {
            "exit_code": 1,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
            "timed_out": False,
        }


def run_background(cmd: str) -> dict:
    """Launch *cmd* in a detached background process.

    Returns:
        {"pid": int}

    Raises:
        RuntimeError if *cmd* matches CMD_BLOCKLIST.
        OSError / subprocess.SubprocessError on Popen failure.
    """
    _check_blocklist(cmd)

    proc = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        cwd=WORKSPACE_DIR,
    )
    return {"pid": proc.pid}


def kill_process(pid: int) -> None:
    """Send SIGTERM to the process identified by *pid*.

    Raises:
        OSError / ProcessLookupError on failure.
    """
    os.kill(pid, signal.SIGTERM)


# ---------------------------------------------------------------------------
# Raw input (require init guard)
# ---------------------------------------------------------------------------

def click(x: int, y: int) -> None:
    """Move the mouse to (*x*, *y*) and left-click.

    Raises:
        RuntimeError if init() was not called.
    """
    ctrl = _ensure_controller()
    ctrl.click(x, y)


def press_key(key: str) -> None:
    """Simulate a key press via xdotool.

    Raises:
        RuntimeError if init() was not called or the key is unknown.
    """
    ctrl = _ensure_controller()
    err = ctrl.press_key(key)
    if err:
        raise RuntimeError(err)


def type_text(text: str) -> None:
    """Type *text* into the currently focused window.

    Raises:
        RuntimeError if init() was not called.
    """
    ctrl = _ensure_controller()
    ctrl.type_text(text)


# ---------------------------------------------------------------------------
# Timing / conversational
# ---------------------------------------------------------------------------

def wait(seconds: float) -> None:
    """Sleep for *seconds*."""
    time.sleep(seconds)


# ---------------------------------------------------------------------------
# Clipboard (GPaste)
# ---------------------------------------------------------------------------

_GPASTE_TIMEOUT = 5  # seconds — gpaste-client should respond instantly


def list_clipboard_history(max_items: int = 100) -> list[dict]:
    """Return the GPaste clipboard history as a list of {index, preview} dicts.

    Each preview is truncated to ~120 chars for LLM context efficiency.
    Uses ``gpaste-client --use-index --oneline`` under the hood.

    Returns:
        [{"index": 0, "preview": "first 120 chars..."}, ...]
    Raises:
        RuntimeError on gpaste-client failure.
    """
    result = subprocess.run(
        ["gpaste-client", "--use-index", "--oneline"],
        capture_output=True, text=True, timeout=_GPASTE_TIMEOUT,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"gpaste-client failed (exit {result.returncode}): {result.stderr.strip()}"
        )
    items: list[dict] = []
    for line in result.stdout.strip().splitlines():
        if not line.strip():
            continue
        # Format: "0: content text here..."
        colon_pos = line.find(":")
        if colon_pos == -1:
            continue
        idx_str = line[:colon_pos].strip()
        if not idx_str.isdigit():
            continue
        idx = int(idx_str)
        preview = line[colon_pos + 1:].strip()[:120]
        items.append({"index": idx, "preview": preview})
        if len(items) >= max_items:
            break
    return items


def get_clipboard_item(index: int) -> str:
    """Return the full text of GPaste history item at *index*.

    Returns:
        The clipboard item text.
    Raises:
        RuntimeError on gpaste-client failure.
        IndexError if the index is out of range.
    """
    result = subprocess.run(
        ["gpaste-client", "--use-index", "get", str(index)],
        capture_output=True, text=True, timeout=_GPASTE_TIMEOUT,
    )
    if result.returncode != 0:
        err = result.stderr.strip().lower()
        if "out of" in err or "invalid" in err or "not found" in err:
            raise IndexError(f"clipboard history index {index} out of range")
        raise RuntimeError(
            f"gpaste-client get failed (exit {result.returncode}): {result.stderr.strip()}"
        )
    return result.stdout


def paste_clipboard_item(index: int) -> None:
    """Promote GPaste history item *index* to the active clipboard (position 0).

    After this call, the next ``ctrl+v`` paste will insert this item.
    Does NOT simulate the paste keystroke — the LLM should follow with
    ``press_key ctrl:v`` if it wants to paste into the focused window.

    Raises:
        RuntimeError on gpaste-client failure.
        IndexError if the index is out of range.
    """
    result = subprocess.run(
        ["gpaste-client", "--use-index", "select", str(index)],
        capture_output=True, text=True, timeout=_GPASTE_TIMEOUT,
    )
    if result.returncode != 0:
        err = result.stderr.strip().lower()
        if "out of" in err or "invalid" in err or "not found" in err:
            raise IndexError(f"clipboard history index {index} out of range")
        raise RuntimeError(
            f"gpaste-client select failed (exit {result.returncode}): {result.stderr.strip()}"
        )


# =============================================================================
# fetch_url — multi-tier URL content fetcher (Fix 3)
# =============================================================================

_FETCH_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# Patterns for known API fast-paths
_PYPI_RE   = re.compile(r"pypi\.org/project/([^/?#]+)")
_GITHUB_RE = re.compile(r"github\.com/([^/?#]+/[^/?#]+)")
_MDN_RE    = re.compile(r"developer\.mozilla\.org/[a-z-]+/docs/(.+?)(?:\?|#|$)")

# Bot-challenge / error page markers — any of these means we got blocked
_BOT_MARKERS = (
    "challenge-platform",
    "cf-browser-verification",
    "just a moment",
    "attention required",
    "ddos-guard",
    "access denied",
    "403 forbidden",
    "too many requests",
    "rateLimitError",
    "please enable javascript",
    "enable cookies to continue",
    "checking if the site",
    "browser is checking",
    "you have been blocked",
)


def _is_valid_content(text: str) -> bool:
    """Return True if *text* looks like real page content.

    Filters empty responses and bot-challenge / Cloudflare interception pages.
    """
    if not text or len(text.strip()) < 120:
        return False
    lower = text.lower()
    return not any(b in lower for b in _BOT_MARKERS)


def _html_to_text(html: str) -> str:
    """Convert HTML to plain text using html2text (strips tags/scripts/styles)."""
    import html2text as _h2t
    h = _h2t.HTML2Text()
    h.ignore_links  = True
    h.ignore_images = True
    h.body_width    = 0
    return h.handle(html).strip()


def _find_relevant_chunk(text: str, query: str, chunk: int = 4000) -> str:
    """Return the *chunk*-char window of *text* that best matches *query*.

    Falls back to the first *chunk* chars if no query is given or no match is
    found.  Adds ellipsis markers at truncation boundaries.
    """
    if not query or len(text) <= chunk:
        return text[:chunk] if len(text) > chunk else text

    lower_text  = text.lower()
    lower_query = query.lower()
    terms       = lower_query.split()

    best_pos   = 0
    best_score = 0
    step       = 200
    for i in range(0, max(1, len(text) - chunk), step):
        window = lower_text[i: i + chunk]
        score  = sum(window.count(t) for t in terms)
        if score > best_score:
            best_score = score
            best_pos   = i

    result = text[best_pos: best_pos + chunk]
    if best_pos > 0:
        result = "\u2026" + result
    if best_pos + chunk < len(text):
        result += "\u2026"
    return result


def fetch_url(url: str, query: str = "") -> str:
    """Fetch URL content via a 5-tier cascade and return cleaned plain text.

    Tiers (tried in order, first valid result wins):
      1. llms.txt       — domain/llms.txt LLM-friendly index
      2. Known APIs     — PyPI JSON, GitHub API (GITHUB_TOKEN), MDN API
      3. curl + html2text — raw HTTP fetch, HTML stripped to text
      4. Jina AI reader — r.jina.ai/{url} Markdown mirror
      5. Playwright     — stealth headless Chromium for JS-heavy / CDN-shielded pages

    Args:
        url:   The URL to fetch.
        query: Optional search query — used to window the result to the most
               relevant 4000-char chunk, reducing LLM context cost.

    Returns:
        Cleaned plain text (≤ 4000 chars when *query* is given).

    Raises:
        exceptions.FetchError: All 5 tiers exhausted without valid content.
    """
    from exceptions import FetchError

    parsed = urllib.parse.urlparse(url)
    origin = parsed.scheme + "://" + parsed.netloc   # e.g. https://pypi.org
    errors: list[str] = []

    def _get(target: str, extra_headers: dict | None = None,
             timeout: int = 10) -> str | None:
        """Minimal urllib GET helper; returns decoded body or None on error."""
        try:
            headers = {"User-Agent": _FETCH_UA}
            if extra_headers:
                headers.update(extra_headers)
            req = urllib.request.Request(target, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read(262144).decode("utf-8", errors="replace")
        except Exception as exc:
            errors.append(f"{target}: {exc}")
            return None

    # ── Tier 1: llms.txt ─────────────────────────────────────────────────────
    llms_url = origin.rstrip("/") + "/llms.txt"
    raw = _get(llms_url, timeout=5)
    if raw and _is_valid_content(raw):
        return _find_relevant_chunk(raw, query)

    # ── Tier 2: Known APIs ────────────────────────────────────────────────────
    pypi_m = _PYPI_RE.search(url)
    if pypi_m:
        pkg = pypi_m.group(1).split("/")[0]
        raw = _get(f"https://pypi.org/pypi/{pkg}/json")
        if raw:
            try:
                data = json.loads(raw)
                info = data.get("info", {})
                text = "\n".join([
                    f"Package : {info.get('name', pkg)}",
                    f"Version : {info.get('version', '')}",
                    f"Summary : {info.get('summary', '')}",
                    f"License : {info.get('license', '')}",
                    f"Home    : {info.get('home_page') or info.get('project_url', '')}",
                    "",
                    info.get("description", "").strip(),
                ])
                if _is_valid_content(text):
                    return _find_relevant_chunk(text, query)
            except Exception as exc:
                errors.append(f"pypi json parse: {exc}")

    github_m = _GITHUB_RE.search(url)
    if github_m:
        repo    = github_m.group(1).strip("/")
        token   = os.environ.get("GITHUB_TOKEN", "")
        gh_hdrs = {"Accept": "application/vnd.github+json"}
        if token:
            gh_hdrs["Authorization"] = f"token {token}"
        raw = _get(f"https://api.github.com/repos/{repo}", gh_hdrs)
        if raw:
            try:
                data  = json.loads(raw)
                lines = [
                    f"Repository  : {data.get('full_name', repo)}",
                    f"Description : {data.get('description', '')}",
                    f"Stars       : {data.get('stargazers_count', 0)}",
                    f"Language    : {data.get('language', '')}",
                    f"Topics      : {', '.join(data.get('topics', []))}",
                    f"URL         : {data.get('html_url', '')}",
                ]
                # Fetch README
                readme_raw = _get(
                    f"https://api.github.com/repos/{repo}/readme", gh_hdrs
                )
                if readme_raw:
                    try:
                        readme_data = json.loads(readme_raw)
                        readme_text = base64.b64decode(
                            readme_data.get("content", "")
                        ).decode("utf-8", errors="replace")
                        lines += ["", "README:", readme_text[:8000]]
                    except Exception:
                        pass
                text = "\n".join(lines)
                if _is_valid_content(text):
                    return _find_relevant_chunk(text, query)
            except Exception as exc:
                errors.append(f"github api parse: {exc}")

    mdn_m = _MDN_RE.search(url)
    if mdn_m:
        path    = mdn_m.group(1).strip("/")
        api_url = f"https://developer.mozilla.org/api/v1/doc/{path}"
        raw = _get(api_url)
        if raw:
            try:
                data = json.loads(raw)
                body = data.get("body", data.get("bodySafe", ""))
                # Strip HTML from MDN body field
                body = re.sub(r"<[^>]+>", " ", body)
                if _is_valid_content(body):
                    return _find_relevant_chunk(body, query)
            except Exception as exc:
                errors.append(f"mdn api parse: {exc}")

    # ── Tier 3: curl + html2text ─────────────────────────────────────────────
    try:
        proc = subprocess.run(
            [
                "curl", "-sL", "--max-time", "15",
                "--user-agent", _FETCH_UA,
                "--header", "Accept: text/html,application/xhtml+xml",
                "--", url,
            ],
            capture_output=True,
            timeout=20,
        )
        if proc.returncode == 0 and proc.stdout:
            html = proc.stdout.decode("utf-8", errors="replace")
            text = _html_to_text(html)
            if _is_valid_content(text):
                return _find_relevant_chunk(text, query)
            else:
                errors.append(f"curl tier: bot-challenge or empty ({len(text)} chars)")
        else:
            errors.append(f"curl exited {proc.returncode}")
    except Exception as exc:
        errors.append(f"curl tier: {exc}")

    # ── Tier 4: Jina AI reader ────────────────────────────────────────────────
    jina_url = "https://r.jina.ai/" + url
    raw = _get(jina_url, timeout=20)
    if raw and _is_valid_content(raw):
        return _find_relevant_chunk(raw, query)
    elif raw:
        errors.append(f"jina tier: bot-challenge ({len(raw)} chars)")

    # ── Tier 5: Playwright stealth ────────────────────────────────────────────
    try:
        from playwright.sync_api import sync_playwright  # type: ignore[import]
        with sync_playwright() as _pw:
            browser = _pw.chromium.launch(
                headless=True,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                ],
            )
            ctx  = browser.new_context(user_agent=_FETCH_UA)
            page = ctx.new_page()
            # Spoof navigator.webdriver
            page.add_init_script(
                "Object.defineProperty(navigator,'webdriver',{get:()=>undefined})"
            )
            page.goto(url, timeout=30000, wait_until="networkidle")
            html = page.content()
            browser.close()
        text = _html_to_text(html)
        if _is_valid_content(text):
            return _find_relevant_chunk(text, query)
        else:
            errors.append(f"playwright tier: bot-challenge ({len(text)} chars)")
    except Exception as exc:
        errors.append(f"playwright tier: {exc}")

    # ── All tiers exhausted ───────────────────────────────────────────────────
    raise FetchError(
        f"fetch_url: all 5 tiers failed for {url!r}.\n"
        f"Errors: {'; '.join(errors)}\n"
        "Options: use pretrained knowledge, try a different URL, or "
        "ask the user to provide the content directly."
    )


# ---------------------------------------------------------------------------
# Facade re-exports — github_actions split (backward compat)
# ---------------------------------------------------------------------------
from github_actions import (  # noqa: F401, E402
    open_pr, create_repo, delete_repo, list_repos, fork_repo,
    list_branches, create_branch, delete_branch,
    list_prs, merge_pr, close_pr,
    create_issue, close_issue, list_issues,
    create_release, add_collaborator,
    get_file_contents, put_file, get_authenticated_user,
    update_repo, set_repo_topics, search_repos,
    list_labels, create_label, protect_branch,
    list_webhooks, create_webhook, delete_webhook,
    list_workflows, trigger_workflow, list_workflow_runs,
    create_gist, list_gists,
    star_repo, unstar_repo,
    create_org_repo, list_org_members, list_teams,
    list_packages, delete_package_version,
    list_notifications, mark_notifications_read,
    get_audit_log,
    assemble_handoff,
)
